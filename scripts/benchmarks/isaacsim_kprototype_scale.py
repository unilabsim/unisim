#!/usr/bin/env python3
"""Scale-bench IsaacSim K-prototype per-env variants over the full host-to-worker path.

The measured unit is one ``backend.step(ctrl, nsteps)`` call, including the
shared-memory IPC round trip to the IsaacSim worker process. ``materialize_s``
covers host-side MJCF compilation of every variant, Kit startup, MJCF-to-USD
conversion (or USD cache hits), and prototype cloning. This is an end-to-end
host<->worker throughput measurement, not a native IsaacSim physics limit.

The scene maps the unisimtoolreal assets onto three entities: a fixed-base
29-actuator robot, a fixed table, and a floating rigid ``tool`` entity whose
per-env identity is selected by a K-variant ``FixedVariantPlan``. Only tools
with exactly two geoms (``handle`` + ``head``) are eligible: the worker
requires an identical geom-name sequence across one variant plan, and the
eraser class carries a single geom. Variants are the first K entries of the
name-sorted 2-geom pool, so smaller K sets are strict subsets of larger ones
and the content-addressed USD caches are reused across a sweep.

Example smoke run:

    OMNI_KIT_ACCEPT_EULA=yes uv run --no-sync python \
      scripts/benchmarks/isaacsim_kprototype_scale.py \
      --num-envs 32 --num-variants 4 --steps 20 --warmup 5 \
      --output /tmp/kproto_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from unisim import create_backend
from unisim.backend.isaacsim.raw_usd_cache import ENV_RAW_USD_CACHE, ENV_ROLE_USD_CACHE
from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
from unisim.scene import SceneCfg

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_ROOT = Path("/home/user/ws/unilabsim2/unisimtoolreal")
DEFAULT_CACHE_DIR = Path("~/.cache/unisim/isaacsim")
SWEEP_VARIANTS = (100, 500, 1100)
SWEEP_ENVS = (2048, 4096, 8192)
EXPECTED_TOOL_GEOMS = ("handle", "head")
TOOL_MANIFEST_SCHEMA = "unitoolreal.tools.v1"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--num-envs", type=_positive_int, default=1024)
    parser.add_argument("--num-variants", type=_positive_int, default=16)
    parser.add_argument("--steps", type=_positive_int, default=200)
    parser.add_argument("--warmup", type=_nonnegative_int, default=20)
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--sim-dt", type=_positive_float, default=1.0 / 60.0)
    parser.add_argument("--nsteps", type=_positive_int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--worker-timeout-s", type=_positive_float, default=7200.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Run the built-in grid variants="
            f"{list(SWEEP_VARIANTS)} x envs={list(SWEEP_ENVS)} (K ascending, then N "
            "ascending) instead of the single --num-variants/--num-envs point."
        ),
    )
    return parser


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _gpu_memory_used_mb() -> float | None:
    """Return summed VRAM usage in MiB across visible GPUs, or None when unavailable."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(
            sum(int(line.strip()) for line in completed.stdout.splitlines() if line.strip())
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _load_two_geom_tool_sources(assets_root: Path) -> tuple[ModelSourceDescriptor, ...]:
    """Return the name-sorted catalog of two-geom tools as nested-stable sources.

    Manifest entries carry ``head_mesh`` exactly when the MJCF model has two
    geoms (verified against the asset files); eraser-class tools have one geom
    and would break the worker's cross-variant geom-name check. Sorting by
    name keeps the first-K subsets nested across a sweep so the content
    addressed USD caches are shared between measurement points.
    """
    manifest_path = assets_root / "manifests" / "tools.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TOOL_MANIFEST_SCHEMA:
        raise ValueError(f"unexpected tools manifest schema in {manifest_path}")
    entries = sorted(
        (entry for entry in manifest["tools"] if "head_mesh" in entry),
        key=lambda entry: entry["name"],
    )
    sources = []
    for entry in entries:
        model = assets_root / entry["model"]
        if not model.is_file():
            raise ValueError(f"tool model missing: {model}")
        geom_names = tuple(re.findall(r'<geom[^>]*name="([^"]+)"', model.read_text()))
        if geom_names != EXPECTED_TOOL_GEOMS:
            raise ValueError(f"{model} geoms {geom_names} != {EXPECTED_TOOL_GEOMS}")
        sources.append(ModelSourceDescriptor(str(model)))
    return tuple(sources)


def _build_scene(
    assets_root: Path, assignment: np.ndarray, variants: tuple[ModelSourceDescriptor, ...]
) -> SceneCfg:
    robot = assets_root / "assets" / "robot" / "robot.xml"
    table = assets_root / "assets" / "table" / "table.xml"
    for path in (robot, table):
        if not path.is_file():
            raise ValueError(f"scene asset missing: {path}")
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                ModelSourceDescriptor(str(robot)),
                root_mode="fixed",
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.15)),
            ),
            SceneEntitySpec(
                "table",
                ModelSourceDescriptor(str(table)),
                kind="rigid",
                root_mode="fixed",
            ),
            SceneEntitySpec(
                "tool",
                variants[0],
                kind="rigid",
                root_mode="floating",
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.3)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            "tool",
            FixedVariantPlan(assignment, variants, layout=FixedVariantLayout.SAME_LAYOUT),
        ),
    )


def _cache_report(worker_metadata: dict[str, Any], key: str) -> dict[str, Any] | None:
    report = worker_metadata.get(key)
    if not isinstance(report, dict):
        return None
    return {name: value for name, value in report.items() if name != "entries"}


def _run_case(
    args: argparse.Namespace,
    tool_pool: tuple[ModelSourceDescriptor, ...],
    num_variants: int,
    num_envs: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "num_variants": num_variants,
        "num_envs": num_envs,
        "steps": args.steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "sim_dt": args.sim_dt,
        "nsteps": args.nsteps,
        "materialize_s": None,
        "vram_before_mb": None,
        "vram_after_mb": None,
        "vram_delta_mb": None,
        "steps_per_s": None,
        "env_steps_per_s": None,
        "physics_ms_mean": None,
        "ipc_ms_mean": None,
        "control_upload_ms_mean": None,
        "raw_usd_cache": None,
        "role_usd_cache": None,
        "error": None,
    }
    backend = None
    try:
        if num_variants > len(tool_pool):
            raise ValueError(
                f"requested {num_variants} variants but only {len(tool_pool)} "
                "two-geom tools are available"
            )
        variants = tool_pool[:num_variants]
        assignment = np.arange(num_envs, dtype=np.int64) % num_variants
        scene = _build_scene(args.assets_root, assignment, variants)

        vram_before = _gpu_memory_used_mb()
        started = time.perf_counter()
        backend = create_backend(
            "isaacsim",
            scene,
            num_envs=num_envs,
            sim_dt=args.sim_dt,
            isaacsim_worker_timeout_s=args.worker_timeout_s,
        )
        # Capture the worker's handshake metadata (USD cache hit report) the
        # same way tests/contract/test_worker_scene_native.py does.
        worker_metadata: dict[str, Any] = {}
        original_bind = backend._bind_scene_metadata

        def bind_metadata(metadata: dict[str, Any]) -> None:
            worker_metadata.update(metadata)
            original_bind(metadata)

        backend._bind_scene_metadata = bind_metadata
        backend.materialize()
        record["materialize_s"] = time.perf_counter() - started
        record["vram_before_mb"] = vram_before
        record["vram_after_mb"] = _gpu_memory_used_mb()
        if vram_before is not None and record["vram_after_mb"] is not None:
            record["vram_delta_mb"] = record["vram_after_mb"] - vram_before

        backend.reset()
        ctrl = np.zeros((num_envs, backend.num_actuators), dtype=np.float32)
        for _ in range(args.warmup):
            backend.step(ctrl, args.nsteps)

        repeat_rates: list[float] = []
        physics_ms: list[float] = []
        ipc_ms: list[float] = []
        control_ms: list[float] = []
        for _ in range(args.repeats):
            timed_start = time.perf_counter()
            for _ in range(args.steps):
                result = backend.step(ctrl, args.nsteps)
                timing = (result or {}).get("timing", {})
                physics_ms.append(float(timing.get("physics_ms", 0.0)))
                ipc_ms.append(float(timing.get("worker_ipc_total_ms", 0.0)))
                control_ms.append(float(timing.get("control_upload_ms", 0.0)))
            repeat_rates.append(args.steps / (time.perf_counter() - timed_start))

        record["repeat_steps_per_s"] = repeat_rates
        record["steps_per_s"] = statistics.fmean(repeat_rates)
        record["env_steps_per_s"] = record["steps_per_s"] * num_envs * args.nsteps
        record["physics_ms_mean"] = statistics.fmean(physics_ms)
        record["ipc_ms_mean"] = statistics.fmean(ipc_ms)
        record["control_upload_ms_mean"] = statistics.fmean(control_ms)
        record["raw_usd_cache"] = _cache_report(worker_metadata, "raw_usd_cache")
        record["role_usd_cache"] = _cache_report(worker_metadata, "role_usd_cache")
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None:
            backend.close()
    return record


def main() -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    args = _parser().parse_args()
    args.assets_root = args.assets_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    os.environ[ENV_RAW_USD_CACHE] = str(cache_dir / "raw-usd")
    os.environ[ENV_ROLE_USD_CACHE] = str(cache_dir / "role-usd")

    points = (
        [(k, n) for k in sorted(SWEEP_VARIANTS) for n in sorted(SWEEP_ENVS)]
        if args.sweep
        else [(args.num_variants, args.num_envs)]
    )
    tool_pool = _load_two_geom_tool_sources(args.assets_root)
    cases = []
    for num_variants, num_envs in points:
        print(
            f"[kproto-scale] case variants={num_variants} envs={num_envs} starting",
            flush=True,
        )
        record = _run_case(args, tool_pool, num_variants, num_envs)
        cases.append(record)
        if record["error"] is not None:
            print(f"[kproto-scale] case FAILED: {record['error']}", flush=True)
        else:
            print(
                f"[kproto-scale] case done: materialize_s={record['materialize_s']:.1f} "
                f"steps/s={record['steps_per_s']:.1f} "
                f"vram_delta_mb={record['vram_delta_mb']}",
                flush=True,
            )

    try:
        baseline_revision = _git("merge-base", "HEAD", "main")
    except subprocess.SubprocessError:
        baseline_revision = _git("rev-parse", "HEAD")
    report = {
        "kind": "isaacsim mapped-entity K-prototype scale; host<->worker path with IPC",
        "baseline_revision": baseline_revision,
        "candidate_revision": _git("rev-parse", "HEAD"),
        "candidate_dirty": bool(_git("status", "--porcelain")),
        "command": "uv run --no-sync python " + " ".join(sys.argv),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sweep": bool(args.sweep),
        "cache_dir": str(cache_dir),
        "tool_pool_size": len(tool_pool),
        "cases": cases,
    }
    output = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
