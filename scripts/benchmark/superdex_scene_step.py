#!/usr/bin/env python3
"""Measure raw SuperDex scene-step throughput without a UniLab environment.

The measured unit is one independent ``Scene.step(dt)`` across one scene. It
excludes model loading, scene construction, actions, state consumption,
observations, rewards, resets, collectors, and learners.

``direct`` measures the original public API path: Python loops over independent
scenes and calls ``Scene.step`` once per scene. ``native`` measures the C++
``SceneBatchExecutor`` path. The native API necessarily writes its required
force/state/link/contact output buffers, but this script does not inspect them.
It is therefore a native physics-barrier measurement, not an RL throughput
benchmark and not a single solver-substep microbenchmark.

To compare an upstream build with the roadmap build, use distinct extension
directories. ``compare`` launches fresh child processes so that the two
``mochi_physics`` shared libraries cannot coexist in one Python interpreter:

    uv run --no-sync python scripts/benchmark/superdex_scene_step.py compare \
      --model "$SUPERDEX_ASSETS_PATH/bots/arms/fr3_v2/fr3_v2.superdex_bot" \
      --baseline-extension-dir /path/to/upstream/build/bin \
      --native-extension-dir /path/to/roadmap/build-cpu-pool/bin \
      --effort-limits 20 20 20 20 5 5 5 \
      --num-scenes 128 --workers 16 --warmup 20 --steps 100 --repeats 3

The JSON result records the selected extension file, hardware-visible CPU
affinity, every repeat, and median scene-step/s. Run each comparison on an
otherwise idle host with identical Python and SuperDex asset versions.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from unisim.backend.superdex.materialization import materialize_model
from unisim.scene import SceneCfg

_RESULT_PREFIX = "SUPERDEX_SCENE_STEP_RESULT="


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _extension_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"extension directory does not exist: {path}")
    if not any(path.glob("mochi_physics*.so")) and not any(path.glob("mochi_physics*.dylib")):
        raise argparse.ArgumentTypeError(f"no mochi_physics extension found in: {path}")
    return path


def _model_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"model file does not exist: {path}")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("direct", "native", "compare"))
    parser.add_argument("--model", required=True, type=_model_path)
    parser.add_argument(
        "--effort-limits",
        type=_positive_float,
        nargs="+",
        help="Optional finite positive effort limits required by some .superdex_bot models.",
    )
    parser.add_argument("--num-scenes", type=_positive_int, default=128)
    parser.add_argument("--workers", type=_positive_int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=_positive_int, default=100)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--dt", type=_positive_float, default=0.002)
    parser.add_argument(
        "--extension-dir",
        type=_extension_dir,
        help="Extension directory for a direct/native child process.",
    )
    parser.add_argument(
        "--baseline-extension-dir",
        type=_extension_dir,
        help="Upstream extension directory used by compare mode.",
    )
    parser.add_argument(
        "--native-extension-dir",
        type=_extension_dir,
        help="Roadmap extension directory used by compare mode.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used for isolated compare children.",
    )
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.mode == "compare":
        if args.extension_dir is not None:
            raise ValueError(
                "compare mode uses --baseline-extension-dir and --native-extension-dir"
            )
        if args.baseline_extension_dir is None or args.native_extension_dir is None:
            raise ValueError("compare mode requires both extension directory arguments")
    elif args.extension_dir is None:
        raise ValueError(f"{args.mode} mode requires --extension-dir")


def _load_runtime(extension_dir: Path) -> tuple[Any, Any, str]:
    # Import the selected extension before the facade can prepend its packaged
    # native directory. Child processes make this selection unambiguous.
    sys.path.insert(0, str(extension_dir))
    native = importlib.import_module("mochi_physics")
    physics = importlib.import_module("superdex.physics")
    robotics = importlib.import_module("superdex.robotics")
    return physics, robotics, str(Path(native.__file__).resolve())


def _visible_cpu_ids() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def _spawn_scenes(
    physics: Any,
    robotics: Any,
    model: Path,
    num_scenes: int,
    effort_limits: Sequence[float] | None,
) -> tuple[Any, list[Any], list[Any], list[list[Any]], list[Any]]:
    plan = materialize_model(physics, robotics, SceneCfg(str(model)), effort_limits=effort_limits)
    scenes: list[Any] = []
    actors: list[Any] = []
    links: list[list[Any]] = []
    closers: list[Any] = []
    try:
        for index in range(num_scenes):
            scene = physics.create_scene(f"scene-step-{index}")
            scene.set_gravity(plan.gravity)
            actor, close = plan.spawn_actor(scene)
            scenes.append(scene)
            actors.append(actor)
            links.append([scene.get_actor(handle) for handle in actor.get_nested_link_actors()])
            closers.append(close)
        if not actors or any(actor.get_num_dofs() != actors[0].get_num_dofs() for actor in actors):
            raise RuntimeError("all benchmark scenes must expose the same number of DOFs")
        if any(len(scene_links) != len(links[0]) for scene_links in links):
            raise RuntimeError("all benchmark scenes must expose the same number of links")
        return plan, scenes, actors, links, closers
    except Exception:
        _close_scenes(physics, plan, scenes, closers)
        raise


def _close_scenes(physics: Any, plan: Any, scenes: Sequence[Any], closers: Sequence[Any]) -> None:
    for close in closers:
        close()
    for scene in scenes:
        physics.destroy_scene(scene)
    plan.cleanup()


def _run_one_direct(scenes: Sequence[Any], *, dt: float, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        for scene in scenes:
            scene.step(dt)
    started = time.perf_counter()
    for _ in range(steps):
        for scene in scenes:
            scene.step(dt)
    return time.perf_counter() - started


def _run_one_native(
    physics: Any,
    scenes: Sequence[Any],
    actors: Sequence[Any],
    links: Sequence[Sequence[Any]],
    *,
    workers: int,
    dt: float,
    warmup: int,
    steps: int,
) -> float:
    executor_cls = getattr(physics, "SceneBatchExecutor", None)
    if executor_cls is None:
        raise RuntimeError(
            "native mode requires the roadmap SceneBatchExecutor extension; "
            "use direct mode for the upstream build"
        )
    dtype = np.float64 if physics.uses_double_precision() else np.float32
    num_scenes = len(scenes)
    dofs = actors[0].get_num_dofs()
    forces = np.zeros((num_scenes, dofs), dtype=dtype)
    qpos = np.empty_like(forces)
    qvel = np.empty_like(forces)
    link_state = np.empty((num_scenes, len(links[0]), 16), dtype=dtype)
    contact = np.empty((num_scenes, 0, 3), dtype=dtype)
    diverged = np.empty(num_scenes, dtype=np.uint8)
    with executor_cls(
        scenes,
        actors,
        links,
        [[] for _ in scenes],
        [[] for _ in scenes],
        [],
        [],
        num_workers=min(workers, num_scenes),
    ) as executor:
        for _ in range(warmup):
            executor.step(dt, forces, qpos, qvel, link_state, contact, diverged)
        started = time.perf_counter()
        for _ in range(steps):
            executor.step(dt, forces, qpos, qvel, link_state, contact, diverged)
        elapsed = time.perf_counter() - started
        if diverged.any():
            raise RuntimeError(
                f"SuperDex solver diverged in scene {int(np.flatnonzero(diverged)[0])}"
            )
    return elapsed


def _run_mode(args: argparse.Namespace) -> dict[str, Any]:
    assert args.extension_dir is not None
    physics, robotics, extension_file = _load_runtime(args.extension_dir)
    if physics.is_initialized():
        raise RuntimeError(
            "benchmark requires a fresh Python process with an uninitialized runtime"
        )
    physics.initialize(num_worker_threads=0)
    measurements: list[float] = []
    try:
        for _ in range(args.repeats):
            plan, scenes, actors, links, closers = _spawn_scenes(
                physics, robotics, args.model, args.num_scenes, args.effort_limits
            )
            try:
                if args.mode == "direct":
                    elapsed = _run_one_direct(
                        scenes, dt=args.dt, warmup=args.warmup, steps=args.steps
                    )
                else:
                    elapsed = _run_one_native(
                        physics,
                        scenes,
                        actors,
                        links,
                        workers=args.workers,
                        dt=args.dt,
                        warmup=args.warmup,
                        steps=args.steps,
                    )
                measurements.append(args.num_scenes * args.steps / elapsed)
            finally:
                _close_scenes(physics, plan, scenes, closers)
    finally:
        physics.shutdown()
    return {
        "mode": args.mode,
        "metric": "scene-step/s",
        "extension_file": extension_file,
        "model": str(args.model),
        "num_scenes": args.num_scenes,
        "workers": 1 if args.mode == "direct" else min(args.workers, args.num_scenes),
        "sdk_workers": 0,
        "warmup_batch_steps": args.warmup,
        "measured_batch_steps": args.steps,
        "dt_seconds": args.dt,
        "repeats": measurements,
        "median_scene_steps_per_second": statistics.median(measurements),
        "min_scene_steps_per_second": min(measurements),
        "max_scene_steps_per_second": max(measurements),
        "affinity_cpu_ids": _visible_cpu_ids(),
        "scope": {
            "excluded": [
                "model/scene construction",
                "actions",
                "state consumption",
                "observations",
                "rewards",
                "resets",
                "collectors",
                "learners",
            ],
            "direct": "Python loop of Scene.step(dt) over independent scenes",
            "native": "SceneBatchExecutor.step(dt, buffers) including mandatory buffer writes",
        },
    }


def _run_child(args: argparse.Namespace, mode: str, extension_dir: Path) -> dict[str, Any]:
    command = [
        str(args.python_executable),
        str(Path(__file__).resolve()),
        mode,
        "--model",
        str(args.model),
        "--extension-dir",
        str(extension_dir),
        "--num-scenes",
        str(args.num_scenes),
        "--workers",
        str(args.workers),
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
        "--repeats",
        str(args.repeats),
        "--dt",
        str(args.dt),
    ]
    if args.effort_limits is not None:
        command.extend(("--effort-limits", *(str(value) for value in args.effort_limits)))
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            return json.loads(line.removeprefix(_RESULT_PREFIX))
    raise RuntimeError(
        f"benchmark child did not emit {_RESULT_PREFIX!r}; stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _compare(args: argparse.Namespace) -> dict[str, Any]:
    assert args.baseline_extension_dir is not None
    assert args.native_extension_dir is not None
    baseline = _run_child(args, "direct", args.baseline_extension_dir)
    native = _run_child(args, "native", args.native_extension_dir)
    baseline_rate = baseline["median_scene_steps_per_second"]
    native_rate = native["median_scene_steps_per_second"]
    return {
        "metric": "scene-step/s",
        "baseline_direct_scene_step": baseline,
        "native_cxx_batch_scene_step": native,
        "native_over_baseline_median_ratio": native_rate / baseline_rate,
    }


def main() -> None:
    args = _parser().parse_args()
    _validate(args)
    result = _compare(args) if args.mode == "compare" else _run_mode(args)
    print(f"{_RESULT_PREFIX}{json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
