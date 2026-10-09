#!/usr/bin/env python3
"""Phase-local mapped IsaacSim CUDA IPC regression probe for unisim issue #350.

Consumes only public backend tensor APIs and records provenance/performance.
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
from pathlib import Path
from typing import Any

import torch

from unisim import IsaacSimBackend
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import EntityInitialState, SceneEntitySpec
from unisim.scene import SceneCfg

ASSET_ROOT = (
    Path(__file__).resolve().parents[3] / "UniLab" / "src" / "unilab" / "assets" / "robots" / "g1"
)
TRACKED = (
    "robot/pelvis",
    "robot/left_hip_roll_link",
    "robot/left_knee_link",
    "robot/left_ankle_roll_link",
    "robot/right_hip_roll_link",
    "robot/right_knee_link",
    "robot/right_ankle_roll_link",
    "robot/torso_link",
    "robot/left_shoulder_roll_link",
    "robot/left_elbow_link",
    "robot/left_wrist_yaw_link",
    "robot/right_shoulder_roll_link",
    "robot/right_elbow_link",
    "robot/right_wrist_yaw_link",
)
SCALARS = ("pelvis_local_linvel", "torso_gyro")


def scene() -> SceneCfg:
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                ModelSourceDescriptor(str(ASSET_ROOT / "isaacsim" / "g1_stand_entity.xml")),
                root_mode="floating",
                initial_state=EntityInitialState(
                    position=(0.0, 0.0, 0.754), quaternion=(1.0, 0.0, 0.0, 0.0)
                ),
            ),
            SceneEntitySpec(
                "floor",
                ModelSourceDescriptor(str(ASSET_ROOT / "isaacsim" / "flat_floor_entity.xml")),
                kind="rigid",
                root_mode="fixed",
            ),
        ),
        default_keyframe_name="stand",
    )


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": len(values),
    }


def nvidia_snapshot() -> dict[str, Any]:
    rows: list[str] = []
    errors: list[str] = []
    for query in (
        "--query-gpu=index,name,driver_version",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
    ):
        try:
            result = subprocess.run(
                ["nvidia-smi", query, "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
            continue
        if result.returncode:
            errors.append(result.stderr.strip())
            continue
        rows.extend(line for line in result.stdout.splitlines() if line.strip())
    return {"rows": rows, "errors": errors}


def run(num_envs: int, warmup: int, iters: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("probe requires CUDA Torch")
    backend = IsaacSimBackend(
        scene(),
        num_envs,
        1.0 / 450.0,
        device_id=0,
        worker_timeout_s=300.0,
        tensor_cuda_ipc=True,
        share_friction_materials=True,
    )
    phase_names = ("step_ms", "state_read_ms", "sensor_read_ms", "reset_ms", "iteration_ms")
    phase = {name: [] for name in phase_names}
    reset_counts: list[int] = []
    backend_timings: list[dict[str, Any] | None] = []
    reset_timings: list[dict[str, Any] | None] = []
    reset_rows_total = 0
    try:
        backend.materialize()
        capabilities = backend.get_tensor_capabilities()
        widths = backend.get_public_state_widths()
        inventory = backend.get_sensor_inventory()
        body_names = backend.get_tracked_body_views().body_names
        assert capabilities.execution.value == "device_resident"
        assert capabilities.process_topology.value == "external_worker"
        assert capabilities.data_plane.value == "cuda_ipc"
        assert capabilities.selected_reset_publication is not None
        assert capabilities.selected_reset_publication.value == "authoritative_views"
        assert capabilities.tracked_body_views
        assert tuple(name for name in body_names if name in TRACKED) == TRACKED
        assert {item.name for item in inventory} >= {
            "robot/pelvis_local_linvel",
            "robot/torso_gyro",
            *(f"track_pos_w_{name}" for name in TRACKED),
        }
        device = torch.device("cuda:0")
        action = torch.zeros((num_envs, backend.num_actuators), device=device, dtype=torch.float32)
        state = backend.get_state_views(device=device)
        row_pattern = torch.arange(num_envs, device=device) % (16 if num_envs >= 16 else num_envs)
        stride = int(row_pattern.max().item()) + 1
        # First full reset initializes/attaches and gives authoritative state.
        backend.reset()
        for iteration in range(warmup + iters):
            if iteration == warmup:
                torch.cuda.synchronize()
                phase = {name: [] for name in phase}
                reset_counts = []
                backend_timings = []
                reset_timings = []
                reset_rows_total = 0
            action.uniform_(-1, 1)
            started = time.perf_counter()
            mark = time.perf_counter()
            backend_timings.append(backend.step_tensor(action, nsteps=3))
            phase["step_ms"].append((time.perf_counter() - mark) * 1e3)

            mark = time.perf_counter()
            state = backend.get_state_views(device=device)
            phase["state_read_ms"].append((time.perf_counter() - mark) * 1e3)

            mark = time.perf_counter()
            _scalars = [backend.get_sensor_view(name, device=device) for name in SCALARS]
            _tracked = backend.get_tracked_body_views(device=device)
            phase["sensor_read_ms"].append((time.perf_counter() - mark) * 1e3)
            rows = torch.nonzero(row_pattern == (iteration % stride), as_tuple=False).reshape(-1)
            qpos = state["qpos"].index_select(0, rows).clone()
            qvel = state["qvel"].index_select(0, rows).clone()
            qpos[:, 0] += 0.001
            qvel[:, 0] += 0.002
            reset_rows_total += int(rows.numel())
            reset_counts.append(int(rows.numel()))
            mark = time.perf_counter()
            reset_timings.append(backend.set_state_tensor(rows, qpos, qvel))
            phase["reset_ms"].append((time.perf_counter() - mark) * 1e3)
            # Immediate authoritative publication check with no readiness step.
            fresh = backend.get_state_views(device=device)
            assert bool(torch.isfinite(fresh["qpos"][rows]).all())
            assert bool(torch.isfinite(backend.get_tracked_body_views(device=device).pos_w).all())
            torch.cuda.synchronize()
            phase["iteration_ms"].append((time.perf_counter() - started) * 1e3)
        return {
            "backend": "isaacsim",
            "tensor_execution": capabilities.execution.value,
            "process_topology": capabilities.process_topology.value,
            "data_plane": capabilities.data_plane.value,
            "selected_reset_publication": capabilities.selected_reset_publication.value,
            "tracked_body_views": capabilities.tracked_body_views,
            "num_envs": num_envs,
            "warmup": warmup,
            "iters": iters,
            "public_state_widths": {"nq": widths.nq, "nv": widths.nv},
            "tracked_body_names": list(body_names),
            "sensor_inventory_count": len(inventory),
            "throughput_env_control_steps_per_s": num_envs
            / (statistics.mean(phase["iteration_ms"]) / 1e3),
            "phases": {key: stats(value) for key, value in phase.items()},
            "reset_rows": stats([float(v) for v in reset_counts]),
            "reset_rows_total": reset_rows_total,
            "readiness_step_calls": 0,
            "backend_timings": {
                "step_tensor": stats_timing(backend_timings),
                "set_state_tensor": stats_timing(reset_timings),
            },
            "gpu": nvidia_snapshot(),
        }
    finally:
        state = None
        fresh = None
        _scalars = None
        _tracked = None
        action = None
        rows = None
        qpos = None
        qvel = None
        gc.collect()
        backend.close()


def stats_timing(results: list[dict[str, Any] | None]) -> dict[str, dict[str, float]]:
    samples: dict[str, list[float]] = {}
    for result in results:
        for key, value in (result or {}).get("timing", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                samples.setdefault(key, []).append(float(value))
    return {key: stats(values) for key, values in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "scope": "issue-350 phase-local mapped CUDA IPC regression probe",
        "provenance": {
            "unisim_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
            ).strip(),
            "unisim_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=Path(__file__).resolve().parents[2],
                    text=True,
                ).strip()
            ),
            "unisim_worktree": str(Path.cwd()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "argv": sys.argv,
        },
        **run(args.num_envs, args.warmup, args.iters),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
