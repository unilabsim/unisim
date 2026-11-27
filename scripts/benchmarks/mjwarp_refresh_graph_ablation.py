#!/usr/bin/env python3
"""Ablation for the MJWarp final-state and derived-constant CUDA graphs (#559).

For one scene and several world counts this measures the host launch cost
(``submit_ms``: one call enqueued onto an idle device) and the steady-state
wall time (``wall_ms``: a pipelined loop, bounded by the device when the host
runs ahead and by the host otherwise) of

* one control step: the physics substeps followed by the first tensor reads
  (tracked-body views and the raw sensor block), with and without a
  randomizing mask reset of the terminated rows in the same step, and
* one randomizing reset: a selected-row and a mask reset, each carrying either
  mass/COM rows or inertia/armature rows,

under three backend graph modes:

* ``all``: every backend-owned graph, including the refresh graphs from #559;
* ``step-only``: step/reset graphs only, with both refreshes launched kernel
  by kernel (the behaviour before #559);
* ``none``: no backend graph at all.

Tensor value validation is deferred, as on the training hot path.  This is a
maintainer ablation of launch overhead, not an RL throughput benchmark.

Example:

.. code-block:: console

   PYTHONPATH=src uv run --no-sync python \\
     scripts/benchmarks/mjwarp_refresh_graph_ablation.py --num-envs 2048 4096
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    ROOT.parent / "UniLab" / "src" / "unilab" / "assets" / "robots" / "g1" / "scene_flat.xml"
)
MODES = ("all", "step-only", "none")
RESET_PHASES = (
    "selected_reset_mass_com",
    "selected_reset_inertia_armature",
    "mask_reset_mass_com",
    "mask_reset_inertia_armature",
)


def _git(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _nvidia_smi(query: str) -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _provenance() -> dict[str, Any]:
    distributions = {}
    for name in ("torch", "warp-lang", "mujoco-warp", "mujoco-core"):
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            distributions[name] = "missing"
    return {
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "unisim_dirty": bool(_git("status", "--porcelain")),
        "gpu": _nvidia_smi("name"),
        "driver_version": _nvidia_smi("driver_version"),
        "python": platform.python_version(),
        "distributions": distributions,
    }


def _apply_mode(backend: Any, mode: str) -> None:
    """Select which backend-owned graphs the hot paths may replay."""
    if mode == "step-only":
        # The dispatchers fall back to their eager kernels one level down.
        backend._refresh_graph = None
        backend._set_const_graphs = {}
    elif mode == "none":
        backend._cuda_graph_enabled = False


def _measure(operation: Callable[[], None], *, iterations: int, warmup: int) -> dict[str, float]:
    """Return the host launch cost and the steady-state wall time of one operation.

    ``submit_ms`` times one call enqueued onto an idle device, so it is what
    the host spends launching the operation.  ``wall_ms`` is the per-iteration
    time of a pipelined loop: the device limit when the host runs ahead, the
    host limit otherwise.
    """
    import torch

    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        operation()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - started) / iterations * 1e3
    submit = 0.0
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        operation()
        submit += time.perf_counter() - started
    return {"submit_ms": submit / iterations * 1e3, "wall_ms": wall}


def _capture_cost(backend: Any) -> dict[str, float]:
    """Time both refresh dispatchers while every backend graph is re-recorded."""
    spent = {"final_state_refresh": 0.0, "derived_constants": 0.0}

    def timed(name: str, function: Callable[..., None]) -> Callable[..., None]:
        def wrapper(*arguments: Any, **keywords: Any) -> None:
            started = time.perf_counter()
            try:
                function(*arguments, **keywords)
            finally:
                spent[name] += time.perf_counter() - started

        return wrapper

    backend._launch_final_state_refresh = timed(
        "final_state_refresh", backend._launch_final_state_refresh
    )
    backend._refresh_derived_constants = timed(
        "derived_constants", backend._refresh_derived_constants
    )
    try:
        started = time.perf_counter()
        backend._initialize_cuda_graphs(backend._warp.get_device())
        total = time.perf_counter() - started
    finally:
        # Drop the instance wrappers so the class methods serve the hot paths again.
        del backend._launch_final_state_refresh
        del backend._refresh_derived_constants
    if not backend._cuda_graph_enabled:
        sys.exit(f"backend graph capture failed: {backend._cuda_graph_disable_reason}")
    return {
        "graph_capture_total_ms": total * 1e3,
        "final_state_refresh_capture_ms": spent["final_state_refresh"] * 1e3,
        "derived_constants_capture_ms": spent["derived_constants"] * 1e3,
    }


def _randomization(backend: Any, rows: int) -> dict[str, Any]:
    """Randomizing payloads for ``rows`` rows, one per refresh level."""
    import torch

    from unisim.dr.types import TensorResetRandomizationPayload

    generator = torch.Generator(device="cuda").manual_seed(0)

    def uniform(shape: tuple[int, ...], low: float, high: float) -> Any:
        noise = torch.rand(shape, generator=generator, device="cuda")
        return low + (high - low) * noise

    def defaults(name: str) -> Any:
        return torch.as_tensor(
            np.asarray(getattr(backend._cpu_model, name), dtype=np.float32), device="cuda"
        )

    nbody, nv = backend._nbody, backend._nv
    identity = torch.zeros((rows, nbody, 4), dtype=torch.float32, device="cuda")
    identity[..., 0] = 1.0
    return {
        "mass_com": TensorResetRandomizationPayload(
            body_mass=uniform((rows, nbody), 0.75, 1.25) * defaults("body_mass"),
            body_ipos=uniform((rows, nbody, 3), -0.005, 0.005),
            body_iquat=identity,
        ),
        "inertia_armature": TensorResetRandomizationPayload(
            body_inertia=uniform((rows, nbody, 3), 0.8, 1.2) * defaults("body_inertia"),
            dof_armature=uniform((rows, nv), 0.0, 0.01),
        ),
    }


def _run_case(args: argparse.Namespace, mode: str, num_envs: int, scene: Any) -> dict[str, Any]:
    import torch
    import warp

    from unisim import MjwarpBackend

    started = time.perf_counter()
    backend = MjwarpBackend(
        scene,
        num_envs=num_envs,
        sim_dt=args.sim_dt,
        add_body_sensors=True,
        tracked_body_names=tuple(args.tracked_body) or None,
        defer_tensor_validation=True,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    result: dict[str, Any] = {"num_envs": num_envs, "mode": mode}
    try:
        if not backend._cuda_graph_enabled:
            sys.exit(f"backend graphs are unavailable: {backend._cuda_graph_disable_reason}")
        result["construction_ms"] = (time.perf_counter() - started) * 1e3
        result["capture"] = _capture_cost(backend)
        _apply_mode(backend, mode)

        device = backend.get_state_views(("qpos",))["qpos"].device
        ctrl = torch.zeros((num_envs, backend._nu), dtype=torch.float32, device=device)
        mujoco = backend._mujoco
        key = mujoco.mj_name2id(backend._cpu_model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
        if key >= 0:
            ctrl[:] = torch.as_tensor(
                backend._cpu_model.key_ctrl[key], dtype=torch.float32, device=device
            )

        def control_step() -> None:
            backend.step_tensor(ctrl, nsteps=args.nsteps)
            backend.get_tracked_body_views()
            backend.get_state_views(("qpos", "qvel", "sensordata"))

        state = backend.get_state_views(("qpos", "qvel"))
        qpos, qvel = state["qpos"].clone(), state["qvel"].clone()
        rows = torch.arange(args.reset_rows, dtype=torch.int64, device=device)
        mask = torch.zeros((num_envs,), dtype=torch.bool, device=device)
        mask[rows] = True
        selected = _randomization(backend, args.reset_rows)
        full_width = _randomization(backend, num_envs)
        phases: dict[str, Callable[[], None]] = {"control_step": control_step}
        for level in ("mass_com", "inertia_armature"):
            phases[f"selected_reset_{level}"] = (
                lambda level=level: backend.set_state_tensor(
                    rows, qpos[rows], qvel[rows], selected[level]
                )
            )
            phases[f"mask_reset_{level}"] = (
                lambda level=level: backend.set_state_tensor_mask(
                    mask, qpos, qvel, full_width[level]
                )
            )

        def autoreset_control_step() -> None:
            """One step of a loop that randomizes reset rows every control step."""
            control_step()
            backend.set_state_tensor_mask(mask, qpos, qvel, full_width["mass_com"])

        phases["control_step_with_mask_dr_reset"] = autoreset_control_step
        result["phases"] = {
            name: _measure(operation, iterations=args.iterations, warmup=args.warmup)
            for name, operation in phases.items()
        }
        warp.synchronize_device()
    finally:
        backend.close()
    return result


def _summarize(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for num_envs in sorted({case["num_envs"] for case in cases}):
        for phase in ("control_step", "control_step_with_mask_dr_reset", *RESET_PHASES):
            for mode in MODES:
                samples = [
                    case["phases"][phase] for case in cases if case["num_envs"] == num_envs
                    and case["mode"] == mode
                ]
                summary.append(
                    {
                        "num_envs": num_envs,
                        "phase": phase,
                        "mode": mode,
                        "submit_ms": statistics.median(item["submit_ms"] for item in samples),
                        "wall_ms": statistics.median(item["wall_ms"] for item in samples),
                    }
                )
    return summary


def _print_report(summary: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    print(f"{'envs':>5}  {'phase':<36} {'mode':<10} {'submit_ms':>10} {'wall_ms':>9}")
    for row in summary:
        print(
            f"{row['num_envs']:>5}  {row['phase']:<36} {row['mode']:<10} "
            f"{row['submit_ms']:>10.3f} {row['wall_ms']:>9.3f}"
        )
    print()
    print("Per construction (median over repeats):")
    capture_keys = (
        ("all graphs", "graph_capture_total_ms"),
        ("final-state refresh", "final_state_refresh_capture_ms"),
        ("derived constants", "derived_constants_capture_ms"),
    )
    for num_envs in sorted({case["num_envs"] for case in cases}):
        group = [case for case in cases if case["num_envs"] == num_envs]
        construction = statistics.median(case["construction_ms"] for case in group)
        captured = "  ".join(
            f"{label} {statistics.median(case['capture'][key] for case in group):.1f} ms"
            for label, key in capture_keys
        )
        print(f"{num_envs:>5}  construction {construction:.1f} ms  graph capture: {captured}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--keyframe", default="stand")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--nsteps", type=int, default=10, help="physics substeps per control step")
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--reset-rows", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tracked-body", action="append", default=[])
    parser.add_argument("--nconmax", type=int, default=512)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.model_file.is_file():
        sys.exit(f"model file not found: {args.model_file}")

    import torch
    import warp

    from unisim.scene import SceneCfg

    warp.init()
    if not torch.cuda.is_available() or not bool(warp.get_device().is_cuda):
        sys.exit("this ablation requires an active CUDA Warp device")
    scene = SceneCfg(
        model_file=str(args.model_file), default_keyframe_name=args.keyframe or None
    )

    cases: list[dict[str, Any]] = []
    for num_envs in args.num_envs:
        for _ in range(args.repeats):
            for mode in MODES:
                cases.append(_run_case(args, mode, num_envs, scene))
                gc.collect()
    summary = _summarize(cases)
    _print_report(summary, cases)
    report = {
        "provenance": _provenance(),
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "summary": summary,
        "cases": cases,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
