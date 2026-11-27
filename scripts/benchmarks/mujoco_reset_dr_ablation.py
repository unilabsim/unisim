"""Reset cost of the MuJoCo adapter by randomization class and row count.

Usage:

  uv run --no-sync python scripts/benchmarks/mujoco_reset_dr_ablation.py SCENE.xml \\
      --base-name pelvis --num-envs 4096 --rows 1 32 128

For each row count, rows are reset through the host path with one randomization
class. The median wall time of ``_host_set_state`` is reported next to the
mean time of each native executor call it makes. A class whose terms no derived constant
reads (friction, solver parameters, damping, actuator gains) commits without
the ``set_const`` recompute; one with mass, center-of-mass or inertia terms
pays for it on the selected rows.

The batch is reset once with every term before anything is measured, which is
the steady state of a run whose randomization mixes the classes: every field
is expanded from then on. It matters because the executor recomputes every
group that any expanded field of the batch feeds, so after a mass or
center-of-mass term has been written once, a commit of terms no constant reads
still costs a full recompute unless the adapter skips it.

This measures the adapter's host reset path on one machine. It is not an RL
throughput benchmark, and it needs the ``mujoco`` extra and Torch.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLASSES: dict[str, tuple[str, ...]] = {
    "state only": (),
    "terms no constant reads": ("friction", "solref", "solimp", "damping", "gains"),
    "terms a constant reads": ("com", "mass"),
}
ALL_TERMS = tuple(dict.fromkeys(term for terms in CLASSES.values() for term in terms))


class TimedPool:
    """Forwards to the executor and accumulates the time of every native call."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self.total: dict[str, float] = defaultdict(float)
        self.count: dict[str, int] = defaultdict(int)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._pool, name)
        if not callable(attr):
            return attr

        def timed(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return attr(*args, **kwargs)
            finally:
                self.total[name] += time.perf_counter() - started
                self.count[name] += 1

        return timed

    def clear(self) -> None:
        self.total.clear()
        self.count.clear()


def payload(backend: Any, rows: int, terms: tuple[str, ...]) -> Any:
    """Full tables for ``rows`` environments, holding each term or leaving it out."""
    import torch

    from unisim.dr.types import TensorResetRandomizationPayload

    model = backend.model
    kp, kd = backend.get_actuator_gains()
    spread = torch.linspace(0.8, 1.2, rows)
    fields: dict[str, Any] = {}

    def table(values: Any, factors: Any = None) -> Any:
        # ``values`` is a model-shaped table; the same one goes to every row.
        base = torch.as_tensor(values, dtype=torch.float32)[None].expand(rows, *values.shape)
        if factors is None:
            return base.clone()
        shape = (rows,) + (1,) * values.ndim
        return base * factors.view(shape)

    if "friction" in terms:
        fields["geom_friction"] = table(backend.get_geom_friction(), spread)
    if "solref" in terms:
        fields["geom_solref"] = table(model.geom_solref, spread)
    if "solimp" in terms:
        fields["geom_solimp"] = table(model.geom_solimp, spread)
    if "damping" in terms:
        fields["dof_damping"] = table(model.dof_damping, spread)
    if "gains" in terms:
        fields["kp"] = table(kp, spread)
        fields["kd"] = table(kd, spread)
    if "com" in terms:
        fields["base_com_offset"] = torch.linspace(-0.02, 0.02, 3 * rows).view(rows, 3) / 3.0
    if "mass" in terms:
        fields["base_mass_delta"] = torch.linspace(-0.3, 0.8, rows) / 10.0
    return TensorResetRandomizationPayload(**fields)


def measure(backend: Any, pool: TimedPool, rows: int, terms: tuple[str, ...], calls: int) -> dict:
    import numpy as np

    qpos = np.broadcast_to(backend.get_default_qpos(), (rows, backend.nq))
    qvel = np.zeros((rows, backend.nv))
    load = payload(backend, rows, terms)
    chosen = np.arange(min(rows, backend.num_envs), dtype=np.int64)
    backend._host_set_state(chosen, qpos[: chosen.size], qvel[: chosen.size], load)
    pool.clear()
    walls = []
    for _ in range(calls):
        first = int(np.random.default_rng().integers(0, max(1, backend.num_envs - rows)))
        chosen = np.arange(first, first + rows, dtype=np.int64)
        started = time.perf_counter()
        result = backend._host_set_state(chosen, qpos, qvel, load)
        walls.append((time.perf_counter() - started) * 1000.0)
    # The median ignores the occasional call that meets another thread of the
    # machine; the native times are means because they are not kept per call.
    reset_ms = float(np.median(walls))
    native = {name: seconds / calls * 1000.0 for name, seconds in pool.total.items()}
    return {
        "rows": rows,
        "reset_ms": reset_ms,
        "native_ms": native,
        "model_update_ms": result["timing"]["set_state_reset_rand_ms"],
        "set_const_calls": pool.count.get("set_const", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_file")
    parser.add_argument("--base-name", help="root body of the reset base aliases")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import numpy as np
    import torch

    from unisim import create_backend
    from unisim.scene import SceneCfg

    backend = create_backend(
        "mujoco",
        SceneCfg(model_file=args.model_file),
        args.num_envs,
        0.002,
        base_name=args.base_name,
        np_dtype=np.float64,
    )
    backend.materialize()
    pool = backend._pool
    timed = TimedPool(pool)
    backend._pool = timed
    warm_rows = min(max(args.rows), args.num_envs)
    backend._host_set_state(
        np.arange(warm_rows, dtype=np.int64),
        np.broadcast_to(backend.get_default_qpos(), (warm_rows, backend.nq)),
        np.zeros((warm_rows, backend.nv)),
        payload(backend, warm_rows, ALL_TERMS),
    )
    report: dict[str, Any] = {
        "mjbatch_uni": importlib.metadata.version("mjbatch-uni"),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "model_file": args.model_file,
        "num_envs": args.num_envs,
        "threads": pool.num_threads,
        "cases": [],
    }
    print(
        f"mjbatch-uni {report['mjbatch_uni']} | {Path(args.model_file).name} | "
        f"{args.num_envs} envs | {pool.num_threads} threads"
    )
    for label, terms in CLASSES.items():
        for rows in args.rows:
            case = measure(backend, timed, rows, terms, args.calls)
            report["cases"].append({"class": label, **case})
            native = "  ".join(
                f"{name} {seconds:.3f}"
                for name, seconds in sorted(case["native_ms"].items(), key=lambda item: -item[1])
            )
            print(
                f"  {label:24s} {rows:5d} rows: reset {case['reset_ms']:8.3f} ms | "
                f"model update {case['model_update_ms']:7.3f} ms | "
                f"set_const calls {case['set_const_calls']:2d} | native: {native}",
                flush=True,
            )
    backend._pool = pool
    backend.close()
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
