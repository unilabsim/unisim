"""Opt-in real-CUDA canonical G1 diagnostic-parity evidence for Newton.

The default suite only exercises the artifact contract and fail-closed checks.
The numerical rollout is deliberately opt-in: it imports Newton, MuJoCo, and
MJWarp and writes a complete trajectory without declaring a parity threshold.
"""

from __future__ import annotations

import math
import os
import platform
import re
import subprocess
from copy import deepcopy
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.adapters.isaac.g1_parity_harness import (
    PROFILER_ENVIRONMENT_VARIABLES,
    SCALAR_SENSOR_FIELDS,
    TRACKED_BODIES,
    TRACKED_SENSOR_FIELDS,
    G1ControlStep,
    G1Snapshot,
    ParityThresholds,
    assert_control_step_trajectory_parity,
    compare_control_step_trajectories,
    compare_snapshots,
    deterministic_control_trajectory,
    gpu_compute_process_snapshot,
    parse_stand_fixture,
    selected_reset_state,
    serialize_capabilities,
    snapshot_to_numpy,
)

SCHEMA_VERSION = 2
SIM_DT = 0.006666666666666667
CONTROL_SUBSTEPS = 3
CONTROL_STEP_COUNT = 4
NUM_ENVS = 2
SELECTED_ROW = 1
CANONICAL_REFERENCES = ("mujoco", "mjwarp")
COMPARISON_KEYS = (
    "newton_vs_mjwarp",
    "newton_vs_mujoco",
    "mjwarp_vs_mujoco",
)
EXPECTED_CAPABILITIES = {
    "execution": "device_resident",
    "state_views": True,
    "state_fields": ["qpos", "qvel"],
    "sensor_views": True,
    "stepping": True,
    "selected_reset": True,
    "reset_randomization": False,
    "fixed_variants": False,
    "host_pre_step_control": False,
    "packed_host_bridge": False,
    "process_topology": "in_process",
    "data_plane": "direct",
    "torch_devices": ["cuda"],
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOURCE_FILES = {
    "newton_backend_sha256": Path("src/unisim/backend/newton/backend.py"),
    "g1_parity_harness_sha256": Path("tests/adapters/isaac/g1_parity_harness.py"),
    "newton_g1_cuda_parity_test_sha256": Path("tests/adapters/newton/test_g1_cuda_parity.py"),
}
_RUNTIME_STRING_FIELDS = (
    "newton",
    "mujoco",
    "mujoco_warp",
    "platform",
    "python",
    "torch",
    "torch_cuda_device",
    "warp",
)
_THRESHOLD_FIELDS = (
    "qpos",
    "qvel",
    "body_pos",
    "body_quat_rad",
    "body_velocity",
    "scalar_sensor",
)
_SCOPES = {
    "diagnostic": "Newton canonical G1 diagnostic parity; no threshold inferred or asserted",
    "acceptance": "Newton canonical G1 gross-divergence parity acceptance",
}
_THRESHOLD_ENVIRONMENT = {
    "qpos": "UNISIM_TEST_NEWTON_G1_STEP_QPOS_ATOL",
    "qvel": "UNISIM_TEST_NEWTON_G1_STEP_QVEL_ATOL",
    "body_pos": "UNISIM_TEST_NEWTON_G1_STEP_BODY_POS_ATOL",
    "body_quat_rad": "UNISIM_TEST_NEWTON_G1_STEP_QUAT_ATOL_RAD",
    "body_velocity": "UNISIM_TEST_NEWTON_G1_STEP_BODY_VEL_ATOL",
    "scalar_sensor": "UNISIM_TEST_NEWTON_G1_STEP_SCALAR_SENSOR_ATOL",
}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, str] = {}
    for name, relative_path in _SOURCE_FILES.items():
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Newton parity source provenance is missing {path}")
        result[name] = _sha256(path)
    return result


def _sensor_fields() -> tuple[str, ...]:
    return (*SCALAR_SENSOR_FIELDS, *TRACKED_SENSOR_FIELDS)


def _host_snapshot(backend: Any, source: str) -> G1Snapshot:
    states = backend.get_state(("qpos", "qvel"))
    sensors = {name: snapshot_to_numpy(backend.get_sensor_data(name)) for name in _sensor_fields()}
    return G1Snapshot(
        source=source,
        qpos=np.asarray(states["qpos"], dtype=np.float32).copy(),
        qvel=np.asarray(states["qvel"], dtype=np.float32).copy(),
        sensors=sensors,
    )


def _device_snapshot(backend: Any, source: str) -> G1Snapshot:
    states = backend.get_state_views(("qpos", "qvel"))
    sensors = {name: snapshot_to_numpy(backend.get_sensor_view(name)) for name in _sensor_fields()}
    return G1Snapshot(
        source=source,
        qpos=snapshot_to_numpy(states["qpos"]),
        qvel=snapshot_to_numpy(states["qvel"]),
        sensors=sensors,
    )


def _apply_selected_reset(backend: Any, rows: Any, qpos: Any, qvel: Any, *, tensor: bool) -> None:
    if tensor:
        backend.set_state_tensor(rows, qpos, qvel)
    else:
        backend.set_state(
            rows.detach().cpu().numpy() if hasattr(rows, "detach") else rows,
            qpos,
            qvel,
        )


def _selected_reset_metrics(before: G1Snapshot, after: G1Snapshot, row: int) -> dict[str, float]:
    metrics = {
        "qpos": float(np.max(np.abs(after.qpos[row] - before.qpos[row]))),
        "qvel": float(np.max(np.abs(after.qvel[row] - before.qvel[row]))),
    }
    for name in _sensor_fields():
        metrics[name] = float(np.max(np.abs(after.sensors[name][row] - before.sensors[name][row])))
    if not math.isfinite(max(metrics.values(), default=0.0)):
        raise ValueError("unselected-row invariance metrics must be finite")
    return metrics


def _assert_unselected_rows_unchanged(
    snapshots: tuple[tuple[G1Snapshot, G1Snapshot, str], ...],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for before, after, source in snapshots:
        metrics = _selected_reset_metrics(before, after, row=1 - SELECTED_ROW)
        assert max(metrics.values()) <= 2e-5, f"{source} selected reset changed an unselected row"
        result[source] = metrics
    return result


def _assert_canonical_selected_reset(
    snapshot: G1Snapshot, qpos: np.ndarray, qvel: np.ndarray
) -> None:
    assert snapshot.qpos.shape == (NUM_ENVS, 36)
    assert snapshot.qvel.shape == (NUM_ENVS, 35)
    assert float(np.max(np.abs(snapshot.qpos[SELECTED_ROW] - qpos[0]))) <= 2e-5
    assert float(np.max(np.abs(snapshot.qvel[SELECTED_ROW] - qvel[0]))) <= 2e-5
    assert np.isfinite(snapshot.qpos).all()
    assert np.isfinite(snapshot.qvel).all()
    for values in snapshot.sensors.values():
        assert np.isfinite(values).all()


def _visible_device_selector(logical_index: int, cuda_visible_devices: str | None) -> int | str:
    if cuda_visible_devices is None or cuda_visible_devices.strip() == "":
        return logical_index
    selectors = [value.strip() for value in cuda_visible_devices.split(",")]
    if not selectors or logical_index >= len(selectors):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r} has no selector for "
            f"logical CUDA device {logical_index}"
        )
    selector = selectors[logical_index]
    if not selector:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES contains an empty device selector: {selectors!r}")
    if selector.isdigit():
        return int(selector)
    return selector


def _physical_gpu_snapshot(torch: Any) -> dict[str, str]:
    logical_index = int(torch.cuda.current_device())
    selector = _visible_device_selector(logical_index, os.environ.get("CUDA_VISIBLE_DEVICES"))
    command = [
        "nvidia-smi",
        f"--id={selector}",
        "--query-gpu=index,name,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot identify physical CUDA device with {command[0]}") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"physical CUDA device audit failed ({result.returncode}): {result.stderr.strip()}"
        )
    index, name, uuid, driver_version = (
        part.strip() for part in result.stdout.strip().split(",", 3)
    )
    properties = torch.cuda.get_device_properties(logical_index)
    if name != str(properties.name):
        raise RuntimeError(
            f"physical GPU {name!r} does not match Torch CUDA device {properties.name!r}"
        )
    return {"index": index, "name": name, "uuid": uuid, "driver_version": driver_version}


def _profiler_environment() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PROFILER_ENVIRONMENT_VARIABLES:
        value = os.environ.get(name)
        if value:
            raise RuntimeError(f"Newton G1 parity must not enable worker profiler {name}")
        result[name] = None
    return result


def _acceptance_mode() -> bool:
    mode = os.environ.get("UNISIM_TEST_NEWTON_G1_PARITY_MODE", "diagnostic")
    if mode not in _SCOPES:
        raise RuntimeError("UNISIM_TEST_NEWTON_G1_PARITY_MODE must be diagnostic or acceptance")
    return mode == "acceptance"


def _thresholds_from_environment() -> ParityThresholds:
    values: dict[str, float] = {}
    missing: list[str] = []
    for field, name in _THRESHOLD_ENVIRONMENT.items():
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            missing.append(name)
            continue
        try:
            values[field] = float(raw)
        except ValueError as exc:
            raise RuntimeError(f"Newton acceptance threshold {name} must be finite") from exc
    if missing:
        raise RuntimeError("Newton acceptance thresholds are missing: " + ", ".join(missing))
    thresholds = ParityThresholds(**values)
    if not all(math.isfinite(value) and value >= 0.0 for value in asdict(thresholds).values()):
        raise RuntimeError("Newton acceptance thresholds must be finite and nonnegative")
    return thresholds


def _runtime_versions(torch: Any, mujoco: Any, warp: Any, mujoco_warp: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "newton": metadata.version("newton"),
        "mujoco": mujoco.__version__,
        "mujoco_warp": mujoco_warp.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device": str(properties.name),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "warp": warp.__version__,
    }


def _step_record(step: G1ControlStep) -> dict[str, Any]:
    return {
        "index": step.index,
        "control": step.control.tolist(),
        "state": step.snapshot.array_report(),
    }


def _validate_finite_json(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Newton parity artifact contains a non-finite float")
    elif isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError("Newton parity artifact object keys must be strings")
            _validate_finite_json(key)
            _validate_finite_json(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_finite_json(nested)
    elif not isinstance(value, str | int | bool | type(None)):
        raise ValueError(f"Newton parity artifact contains unsupported JSON type {type(value)!r}")


def _validate_sensor_shapes(sensors: Any) -> None:
    if not isinstance(sensors, dict) or set(sensors) != set(_sensor_fields()):
        raise ValueError("Newton parity snapshot has an invalid sensor field set")
    for name in _sensor_fields():
        expected = (NUM_ENVS, 4 if name.startswith("track_quat_w_") else 3)
        values = np.asarray(sensors[name])
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"Newton parity sensor {name!r} has an invalid array")


def _validate_state_arrays(state: Any) -> None:
    if not isinstance(state, dict) or set(state) != {"qpos", "qvel", "sensors"}:
        raise ValueError("Newton parity state record has invalid fields")
    qpos = np.asarray(state["qpos"])
    qvel = np.asarray(state["qvel"])
    if qpos.shape != (NUM_ENVS, 36) or not np.isfinite(qpos).all():
        raise ValueError("Newton parity qpos trajectory has an invalid array")
    if qvel.shape != (NUM_ENVS, 35) or not np.isfinite(qvel).all():
        raise ValueError("Newton parity qvel trajectory has an invalid array")
    _validate_sensor_shapes(state["sensors"])


def _validate_path_record(record: Any, label: str) -> None:
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "sha256"}
        or not isinstance(record["path"], str)
        or not isinstance(record["sha256"], str)
        or _SHA256_PATTERN.fullmatch(record["sha256"]) is None
    ):
        raise ValueError(f"Newton parity {label} provenance is invalid")


def _validate_report(report: Any) -> None:
    required = {
        "schema_version",
        "mode",
        "scope",
        "backend",
        "canonical_references",
        "num_envs",
        "selected_row",
        "sim_dt",
        "control_substeps",
        "control_step_count",
        "scene",
        "stand_fixture_crosscheck",
        "stand_qpos",
        "stand_ctrl",
        "scalar_sensors",
        "tracked_bodies",
        "tracked_sensor_fields",
        "body_metric_semantics",
        "source_provenance",
        "runtime",
        "gpu",
        "profiler_environment",
        "newton_tensor_capabilities",
        "assertions",
        "thresholds",
        "reset",
        "control_steps",
    }
    if not isinstance(report, dict) or set(report) != required:
        raise ValueError("Newton parity artifact top-level schema is invalid")
    if report["schema_version"] != SCHEMA_VERSION or report["backend"] != "newton":
        raise ValueError("Newton parity artifact identity is invalid")
    mode = report.get("mode")
    if mode not in _SCOPES or report["scope"] != _SCOPES[mode]:
        raise ValueError("Newton parity artifact scope is invalid")
    if tuple(report["canonical_references"]) != CANONICAL_REFERENCES:
        raise ValueError("Newton parity canonical references are invalid")
    if report["num_envs"] != NUM_ENVS or report["selected_row"] != SELECTED_ROW:
        raise ValueError("Newton parity rollout shape is invalid")
    if report["sim_dt"] != SIM_DT or report["control_substeps"] != CONTROL_SUBSTEPS:
        raise ValueError("Newton parity timestep metadata is invalid")
    if report["control_step_count"] != CONTROL_STEP_COUNT:
        raise ValueError("Newton parity trajectory length is invalid")
    _validate_path_record(report["scene"], "scene")
    _validate_path_record(report["stand_fixture_crosscheck"], "stand fixture")
    if (
        np.asarray(report["stand_qpos"]).shape != (36,)
        or not np.isfinite(report["stand_qpos"]).all()
    ):
        raise ValueError("Newton stand qpos provenance is invalid")
    if (
        np.asarray(report["stand_ctrl"]).shape != (29,)
        or not np.isfinite(report["stand_ctrl"]).all()
    ):
        raise ValueError("Newton stand control provenance is invalid")
    if tuple(report["scalar_sensors"]) != SCALAR_SENSOR_FIELDS:
        raise ValueError("Newton parity scalar sensor contract is invalid")
    if tuple(report["tracked_bodies"]) != TRACKED_BODIES:
        raise ValueError("Newton parity tracked-body contract is invalid")
    if tuple(report["tracked_sensor_fields"]) != TRACKED_SENSOR_FIELDS:
        raise ValueError("Newton parity tracked-sensor contract is invalid")
    if report["newton_tensor_capabilities"] != EXPECTED_CAPABILITIES:
        raise ValueError("Newton tensor capabilities are invalid")
    expected_assertions = {
        "canonical_initial_reset": True,
        "numerical_thresholds": mode == "acceptance",
        "parity_acceptance": mode == "acceptance",
        "runtime_device_resident": True,
        "unselected_reset_rows_unchanged": True,
        "zero_copy_session": True,
    }
    if report["assertions"] != expected_assertions:
        raise ValueError("Newton diagnostic assertions are invalid")
    if mode == "diagnostic":
        if report["thresholds"] is not None:
            raise ValueError("Newton diagnostic parity must not serialize thresholds")
    elif (
        not isinstance(report["thresholds"], dict)
        or tuple(report["thresholds"]) != _THRESHOLD_FIELDS
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            for value in report["thresholds"].values()
        )
    ):
        raise ValueError("Newton acceptance thresholds are invalid")
    if report["body_metric_semantics"] != {
        "collection": "after each control step",
        "pose": "root-relative full transform",
        "velocity": "world-frame linear/angular velocity",
    }:
        raise ValueError("Newton body metric semantics are invalid")
    if not isinstance(report["source_provenance"], dict) or set(report["source_provenance"]) != set(
        _SOURCE_FILES
    ):
        raise ValueError("Newton source provenance is invalid")
    for value in report["source_provenance"].values():
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("Newton source provenance contains an invalid SHA256")
    runtime = report["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        *_RUNTIME_STRING_FIELDS,
        "cuda_visible_devices",
        "torch_cuda_available",
        "torch_cuda_device_count",
    }:
        raise ValueError("Newton runtime provenance is invalid")
    if any(not isinstance(runtime[name], str) for name in _RUNTIME_STRING_FIELDS):
        raise ValueError("Newton runtime string provenance is invalid")
    if not isinstance(runtime["cuda_visible_devices"], str | None):
        raise ValueError("Newton CUDA_VISIBLE_DEVICES provenance is invalid")
    if (
        runtime["torch_cuda_available"] is not True
        or isinstance(runtime["torch_cuda_device_count"], bool)
        or not isinstance(runtime["torch_cuda_device_count"], int)
        or runtime["torch_cuda_device_count"] <= 0
    ):
        raise ValueError("Newton Torch CUDA provenance is invalid")
    if not isinstance(report["gpu"], dict) or set(report["gpu"]) != {
        "compute_processes_before",
        "compute_processes_after",
        "physical_device",
    }:
        raise ValueError("Newton GPU provenance is invalid")
    if (
        not isinstance(report["gpu"]["physical_device"], dict)
        or set(report["gpu"]["physical_device"]) != {"index", "name", "uuid", "driver_version"}
        or any(
            not isinstance(report["gpu"]["physical_device"][name], str)
            for name in ("index", "name", "uuid", "driver_version")
        )
    ):
        raise ValueError("Newton physical GPU provenance is invalid")
    if not isinstance(report["profiler_environment"], dict) or any(
        report["profiler_environment"].get(name) is not None
        for name in PROFILER_ENVIRONMENT_VARIABLES
    ):
        raise ValueError("Newton parity profiler provenance is invalid")

    reset = report["reset"]
    if not isinstance(reset, dict) or set(reset) != {
        "arrays",
        "comparisons",
        "snapshots",
        "unselected_row_invariance",
        "unselected_row_invariance_asserted",
    }:
        raise ValueError("Newton reset record schema is invalid")
    if tuple(reset["arrays"]) != ("newton", "mjwarp", "mujoco"):
        raise ValueError("Newton reset backends are invalid")
    for state in reset["arrays"].values():
        _validate_state_arrays(state)
    if tuple(reset["comparisons"]) != COMPARISON_KEYS:
        raise ValueError("Newton reset comparison keys are invalid")
    if reset["unselected_row_invariance_asserted"] is not True:
        raise ValueError("Newton selected-reset invariance was not asserted")

    control_steps = report["control_steps"]
    if not isinstance(control_steps, dict) or set(control_steps) != {
        "comparisons",
        "controls",
        "records",
    }:
        raise ValueError("Newton control-step schema is invalid")
    if len(control_steps["controls"]) != CONTROL_STEP_COUNT:
        raise ValueError("Newton control trajectory length is invalid")
    if tuple(control_steps["records"]) != ("newton", "mjwarp", "mujoco"):
        raise ValueError("Newton control-step backends are invalid")
    for records in control_steps["records"].values():
        if len(records) != CONTROL_STEP_COUNT:
            raise ValueError("Newton per-step trajectory length is invalid")
        for index, record in enumerate(records):
            if record["index"] != index:
                raise ValueError("Newton control-step indices are invalid")
            if np.asarray(record["control"]).shape != (NUM_ENVS, 29):
                raise ValueError("Newton control trajectory has an invalid shape")
            _validate_state_arrays(record["state"])
    if tuple(control_steps["comparisons"]) != COMPARISON_KEYS:
        raise ValueError("Newton control comparison keys are invalid")
    for comparison in control_steps["comparisons"].values():
        if comparison["step_count"] != CONTROL_STEP_COUNT:
            raise ValueError("Newton comparison trajectory length is invalid")
    _validate_finite_json(report)


def _resolve_path(environment_name: str, default: Path) -> Path:
    value = os.environ.get(environment_name)
    path = Path(value).expanduser() if value else default
    if not path.is_file():
        raise FileNotFoundError(
            f"Newton G1 parity fixture {environment_name} is not a file: {path}"
        )
    return path.resolve()


def _run_newton_g1_parity(output_path: str | Path) -> dict[str, Any]:
    torch = pytest.importorskip("torch")
    mujoco = pytest.importorskip("mujoco")
    warp = pytest.importorskip("warp")
    mujoco_warp = pytest.importorskip("mujoco_warp")
    pytest.importorskip("newton")
    acceptance = _acceptance_mode()
    thresholds = _thresholds_from_environment() if acceptance else None
    warp.init()
    if not torch.cuda.is_available() or not bool(warp.get_device().is_cuda):
        pytest.skip("Newton canonical G1 parity requires CUDA Torch, Warp, and Newton")

    import json

    from tests.adapters.isaac.g1_parity_harness import (
        DEFAULT_CANONICAL_SCENE,
        DEFAULT_ISAACSIM_ROBOT,
    )
    from unisim import MjwarpBackend, MuJoCoBackend, NewtonBackend
    from unisim.scene import SceneCfg

    canonical_scene_path = _resolve_path("UNISIM_TEST_G1_CANONICAL_SCENE", DEFAULT_CANONICAL_SCENE)
    stand_fixture_path = _resolve_path("UNISIM_TEST_G1_ISAACSIM_ROBOT", DEFAULT_ISAACSIM_ROBOT)
    stand_qpos, stand_ctrl = parse_stand_fixture(canonical_scene_path, stand_fixture_path)
    scene_cfg = SceneCfg(model_file=str(canonical_scene_path), default_keyframe_name="stand")
    physical_device = _physical_gpu_snapshot(torch)
    compute_processes_before = gpu_compute_process_snapshot(int(physical_device["index"]))
    if acceptance and compute_processes_before:
        raise RuntimeError(
            "Newton acceptance parity requires an initially idle selected GPU; "
            f"active processes: {compute_processes_before}"
        )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "acceptance" if acceptance else "diagnostic",
        "scope": _SCOPES["acceptance" if acceptance else "diagnostic"],
        "backend": "newton",
        "canonical_references": list(CANONICAL_REFERENCES),
        "num_envs": NUM_ENVS,
        "selected_row": SELECTED_ROW,
        "sim_dt": SIM_DT,
        "control_substeps": CONTROL_SUBSTEPS,
        "control_step_count": CONTROL_STEP_COUNT,
        "scene": {
            "path": str(canonical_scene_path),
            "sha256": _sha256(canonical_scene_path),
        },
        "stand_fixture_crosscheck": {
            "path": str(stand_fixture_path),
            "sha256": _sha256(stand_fixture_path),
        },
        "stand_qpos": stand_qpos.tolist(),
        "stand_ctrl": stand_ctrl.tolist(),
        "scalar_sensors": list(SCALAR_SENSOR_FIELDS),
        "tracked_bodies": list(TRACKED_BODIES),
        "tracked_sensor_fields": list(TRACKED_SENSOR_FIELDS),
        "body_metric_semantics": {
            "collection": "after each control step",
            "pose": "root-relative full transform",
            "velocity": "world-frame linear/angular velocity",
        },
        "source_provenance": _source_provenance(),
        "runtime": _runtime_versions(torch, mujoco, warp, mujoco_warp),
        "gpu": {
            "compute_processes_before": compute_processes_before,
            "physical_device": physical_device,
        },
        "profiler_environment": _profiler_environment(),
    }

    newton_backend: Any | None = None
    mjwarp_backend: Any | None = None
    mujoco_backend: Any | None = None
    try:
        mujoco_backend = MuJoCoBackend(
            scene_cfg,
            NUM_ENVS,
            SIM_DT,
            base_name="pelvis",
            add_body_sensors=True,
            tracked_body_names=TRACKED_BODIES,
        )
        mjwarp_backend = MjwarpBackend(
            scene_cfg,
            NUM_ENVS,
            SIM_DT,
            base_name="pelvis",
            add_body_sensors=True,
        )
        newton_backend = NewtonBackend(
            scene_cfg,
            NUM_ENVS,
            SIM_DT,
            base_name="pelvis",
            device=f"cuda:{torch.cuda.current_device()}",
        )
        assert mujoco_backend is not None
        assert mjwarp_backend is not None
        assert newton_backend is not None
        mujoco_backend.materialize()
        mjwarp_backend.materialize()
        newton_backend.materialize()

        rows_np = np.asarray((SELECTED_ROW,), dtype=np.int64)
        reset_qpos, reset_qvel = selected_reset_state(
            stand_qpos, np.zeros(35, dtype=np.float32), row_count=1
        )
        full_qpos = np.repeat(stand_qpos[None, :], NUM_ENVS, axis=0)
        full_qvel = np.zeros((NUM_ENVS, 35), dtype=np.float32)
        all_rows_torch = torch.arange(NUM_ENVS, dtype=torch.int64, device="cuda")
        rows_torch = torch.as_tensor(rows_np, device=all_rows_torch.device)
        full_qpos_torch = torch.as_tensor(full_qpos, device=all_rows_torch.device)
        full_qvel_torch = torch.as_tensor(full_qvel, device=all_rows_torch.device)
        reset_qpos_torch = torch.as_tensor(reset_qpos, device=all_rows_torch.device)
        reset_qvel_torch = torch.as_tensor(reset_qvel, device=all_rows_torch.device)
        control_trajectory = deterministic_control_trajectory(
            stand_ctrl, num_envs=NUM_ENVS, steps=CONTROL_STEP_COUNT
        )
        control_trajectory_torch = torch.as_tensor(control_trajectory, device=all_rows_torch.device)

        _apply_selected_reset(
            mujoco_backend,
            np.arange(NUM_ENVS, dtype=np.int64),
            full_qpos,
            full_qvel,
            tensor=False,
        )
        _apply_selected_reset(
            mjwarp_backend, all_rows_torch, full_qpos_torch, full_qvel_torch, tensor=True
        )
        _apply_selected_reset(
            newton_backend, all_rows_torch, full_qpos_torch, full_qvel_torch, tensor=True
        )
        mujoco_initial = _host_snapshot(mujoco_backend, "mujoco")
        mjwarp_initial = _device_snapshot(mjwarp_backend, "mjwarp")
        newton_initial = _device_snapshot(newton_backend, "newton")

        _apply_selected_reset(mujoco_backend, rows_np, reset_qpos, reset_qvel, tensor=False)
        _apply_selected_reset(
            mjwarp_backend, rows_torch, reset_qpos_torch, reset_qvel_torch, tensor=True
        )
        _apply_selected_reset(
            newton_backend, rows_torch, reset_qpos_torch, reset_qvel_torch, tensor=True
        )
        mujoco_reset = _host_snapshot(mujoco_backend, "mujoco")
        mjwarp_reset = _device_snapshot(mjwarp_backend, "mjwarp")
        newton_reset = _device_snapshot(newton_backend, "newton")
        for snapshot in (mujoco_reset, mjwarp_reset, newton_reset):
            _assert_canonical_selected_reset(snapshot, reset_qpos, reset_qvel)
        unselected_metrics = _assert_unselected_rows_unchanged(
            (
                (mujoco_initial, mujoco_reset, "mujoco"),
                (mjwarp_initial, mjwarp_reset, "mjwarp"),
                (newton_initial, newton_reset, "newton"),
            )
        )

        capabilities = serialize_capabilities(newton_backend.get_tensor_capabilities())
        assert capabilities == EXPECTED_CAPABILITIES
        report["newton_tensor_capabilities"] = capabilities
        report["assertions"] = {
            "canonical_initial_reset": True,
            "numerical_thresholds": thresholds is not None,
            "parity_acceptance": thresholds is not None,
            "runtime_device_resident": True,
            "unselected_reset_rows_unchanged": True,
            "zero_copy_session": capabilities["data_plane"] == "direct"
            and capabilities["execution"] == "device_resident",
        }
        report["thresholds"] = None if thresholds is None else asdict(thresholds)
        report["reset"] = {
            "arrays": {
                "newton": newton_reset.array_report(),
                "mjwarp": mjwarp_reset.array_report(),
                "mujoco": mujoco_reset.array_report(),
            },
            "snapshots": {
                "newton": newton_reset.report(),
                "mjwarp": mjwarp_reset.report(),
                "mujoco": mujoco_reset.report(),
            },
            "comparisons": {
                "newton_vs_mjwarp": compare_snapshots(
                    mjwarp_reset, newton_reset, include_step=False
                ),
                "newton_vs_mujoco": compare_snapshots(
                    mujoco_reset, newton_reset, include_step=False
                ),
                "mjwarp_vs_mujoco": compare_snapshots(
                    mujoco_reset, mjwarp_reset, include_step=False
                ),
            },
            "unselected_row_invariance": unselected_metrics,
            "unselected_row_invariance_asserted": True,
        }

        mujoco_steps: list[G1ControlStep] = []
        mjwarp_steps: list[G1ControlStep] = []
        newton_steps: list[G1ControlStep] = []
        for index, (control, control_torch) in enumerate(
            zip(control_trajectory, control_trajectory_torch, strict=True)
        ):
            mujoco_backend.step(control, nsteps=CONTROL_SUBSTEPS)
            mjwarp_backend.step_tensor(control_torch, nsteps=CONTROL_SUBSTEPS)
            newton_backend.step_tensor(control_torch, nsteps=CONTROL_SUBSTEPS)
            mujoco_steps.append(
                G1ControlStep(index, control, _host_snapshot(mujoco_backend, "mujoco"))
            )
            mjwarp_steps.append(
                G1ControlStep(index, control, _device_snapshot(mjwarp_backend, "mjwarp"))
            )
            newton_steps.append(
                G1ControlStep(index, control, _device_snapshot(newton_backend, "newton"))
            )
        report["control_steps"] = {
            "controls": control_trajectory.tolist(),
            "records": {
                "newton": [_step_record(step) for step in newton_steps],
                "mjwarp": [_step_record(step) for step in mjwarp_steps],
                "mujoco": [_step_record(step) for step in mujoco_steps],
            },
            "comparisons": {
                "newton_vs_mjwarp": compare_control_step_trajectories(mjwarp_steps, newton_steps),
                "newton_vs_mujoco": compare_control_step_trajectories(mujoco_steps, newton_steps),
                "mjwarp_vs_mujoco": compare_control_step_trajectories(mujoco_steps, mjwarp_steps),
            },
        }
        if thresholds is not None:
            assert_control_step_trajectory_parity(
                report["control_steps"]["comparisons"]["newton_vs_mujoco"], thresholds
            )
            assert_control_step_trajectory_parity(
                report["control_steps"]["comparisons"]["newton_vs_mjwarp"], thresholds
            )
    finally:
        for backend in (newton_backend, mjwarp_backend, mujoco_backend):
            if backend is not None:
                backend.close()

    report["gpu"]["compute_processes_after"] = gpu_compute_process_snapshot(
        int(report["gpu"]["physical_device"]["index"])
    )
    _validate_report(report)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_NEWTON_G1_PARITY_NATIVE") != "1",
    reason="set UNISIM_TEST_NEWTON_G1_PARITY_NATIVE=1 for real Newton CUDA G1 parity",
)
def test_newton_full_g1_cuda_diagnostic_parity(tmp_path: Path) -> None:
    output = os.environ.get(
        "UNISIM_TEST_NEWTON_G1_PARITY_OUTPUT",
        str(tmp_path / "newton-g1-cuda-diagnostic-parity.json"),
    )
    report = _run_newton_g1_parity(output)
    acceptance = _acceptance_mode()
    assert report["assertions"]["parity_acceptance"] is acceptance
    assert (report["thresholds"] is not None) is acceptance


def test_exact_public_field_contract() -> None:
    assert len(TRACKED_BODIES) == 14
    assert len(TRACKED_SENSOR_FIELDS) == 56
    assert TRACKED_SENSOR_FIELDS == (
        *(f"track_pos_w_{body}" for body in TRACKED_BODIES),
        *(f"track_quat_w_{body}" for body in TRACKED_BODIES),
        *(f"track_linvel_w_{body}" for body in TRACKED_BODIES),
        *(f"track_angvel_w_{body}" for body in TRACKED_BODIES),
    )
    assert set(_sensor_fields()) == {
        *SCALAR_SENSOR_FIELDS,
        *TRACKED_SENSOR_FIELDS,
    }


def test_visible_device_selector_handles_numeric_and_uuid_selectors() -> None:
    assert _visible_device_selector(0, "1") == 1
    assert _visible_device_selector(1, "0,GPU-test-uuid") == "GPU-test-uuid"
    with pytest.raises(RuntimeError, match="no selector"):
        _visible_device_selector(2, "0,1")
    with pytest.raises(RuntimeError, match="empty device selector"):
        _visible_device_selector(1, "0,")


def test_profiler_environment_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _profiler_environment() == {name: None for name in PROFILER_ENVIRONMENT_VARIABLES}
    monkeypatch.setenv(PROFILER_ENVIRONMENT_VARIABLES[0], "/tmp/trace.json")
    with pytest.raises(RuntimeError, match="must not enable worker profiler"):
        _profiler_environment()


def test_acceptance_mode_and_threshold_environment_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNISIM_TEST_NEWTON_G1_PARITY_MODE", "invalid")
    with pytest.raises(RuntimeError, match="diagnostic or acceptance"):
        _acceptance_mode()

    monkeypatch.setenv("UNISIM_TEST_NEWTON_G1_PARITY_MODE", "acceptance")
    for name in _THRESHOLD_ENVIRONMENT.values():
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="thresholds are missing"):
        _thresholds_from_environment()

    for name in _THRESHOLD_ENVIRONMENT.values():
        monkeypatch.setenv(name, "0.1")
    thresholds = _thresholds_from_environment()
    assert asdict(thresholds) == {name: 0.1 for name in _THRESHOLD_FIELDS}

    monkeypatch.setenv(next(iter(_THRESHOLD_ENVIRONMENT.values())), "nan")
    with pytest.raises(RuntimeError, match="finite and nonnegative"):
        _thresholds_from_environment()


def test_state_validation_rejects_wrong_fields_shapes_and_nonfinite() -> None:
    valid_sensors = {
        name: np.zeros((NUM_ENVS, 4 if name.startswith("track_quat_w_") else 3), dtype=np.float32)
        for name in _sensor_fields()
    }
    valid = {
        "qpos": np.zeros((NUM_ENVS, 36), dtype=np.float32),
        "qvel": np.zeros((NUM_ENVS, 35), dtype=np.float32),
        "sensors": valid_sensors,
    }
    _validate_state_arrays(valid)
    wrong_sensor = {
        **valid,
        "sensors": {**valid_sensors, "unexpected": np.zeros((NUM_ENVS, 3))},
    }
    with pytest.raises(ValueError, match="invalid sensor field set"):
        _validate_state_arrays(wrong_sensor)
    nonfinite = deepcopy(valid)
    nonfinite["qvel"][0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid array"):
        _validate_state_arrays(nonfinite)


def test_finite_json_validation_fails_closed() -> None:
    _validate_finite_json({"value": [1.0, True, None, "text"]})
    for invalid in (float("nan"), float("inf"), {"bytes": b"invalid"}, {1: "non-string-key"}):
        with pytest.raises(ValueError, match="non-finite|unsupported JSON|keys must be strings"):
            _validate_finite_json(invalid)


def _minimal_state() -> dict[str, Any]:
    sensors = {
        name: np.zeros(
            (NUM_ENVS, 4 if name.startswith("track_quat_w_") else 3), dtype=np.float32
        ).tolist()
        for name in _sensor_fields()
    }
    return {
        "qpos": np.zeros((NUM_ENVS, 36), dtype=np.float32).tolist(),
        "qvel": np.zeros((NUM_ENVS, 35), dtype=np.float32).tolist(),
        "sensors": sensors,
    }


def _minimal_report() -> dict[str, Any]:
    state = _minimal_state()
    step_record = {
        "index": 0,
        "control": np.zeros((NUM_ENVS, 29), dtype=np.float32).tolist(),
        "state": deepcopy(state),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "diagnostic",
        "scope": "Newton canonical G1 diagnostic parity; no threshold inferred or asserted",
        "backend": "newton",
        "canonical_references": list(CANONICAL_REFERENCES),
        "num_envs": NUM_ENVS,
        "selected_row": SELECTED_ROW,
        "sim_dt": SIM_DT,
        "control_substeps": CONTROL_SUBSTEPS,
        "control_step_count": CONTROL_STEP_COUNT,
        "scene": {"path": "/tmp/g1.xml", "sha256": "0" * 64},
        "stand_fixture_crosscheck": {"path": "/tmp/stand.xml", "sha256": "0" * 64},
        "stand_qpos": [0.0] * 36,
        "stand_ctrl": [0.0] * 29,
        "scalar_sensors": list(SCALAR_SENSOR_FIELDS),
        "tracked_bodies": list(TRACKED_BODIES),
        "tracked_sensor_fields": list(TRACKED_SENSOR_FIELDS),
        "body_metric_semantics": {
            "collection": "after each control step",
            "pose": "root-relative full transform",
            "velocity": "world-frame linear/angular velocity",
        },
        "source_provenance": {name: "0" * 64 for name in _SOURCE_FILES},
        "runtime": {
            "cuda_visible_devices": "0",
            "newton": "test",
            "mujoco": "test",
            "mujoco_warp": "test",
            "platform": "test",
            "python": "test",
            "torch": "test",
            "torch_cuda_available": True,
            "torch_cuda_device": "test",
            "torch_cuda_device_count": 1,
            "warp": "test",
        },
        "gpu": {
            "compute_processes_before": [],
            "compute_processes_after": [],
            "physical_device": {
                "index": "0",
                "name": "test",
                "uuid": "GPU-test",
                "driver_version": "test",
            },
        },
        "profiler_environment": {name: None for name in PROFILER_ENVIRONMENT_VARIABLES},
        "newton_tensor_capabilities": deepcopy(EXPECTED_CAPABILITIES),
        "assertions": {
            "canonical_initial_reset": True,
            "numerical_thresholds": False,
            "parity_acceptance": False,
            "runtime_device_resident": True,
            "unselected_reset_rows_unchanged": True,
            "zero_copy_session": True,
        },
        "thresholds": None,
        "reset": {
            "arrays": {name: deepcopy(state) for name in ("newton", "mjwarp", "mujoco")},
            "comparisons": {name: {"metric": 0.0} for name in COMPARISON_KEYS},
            "snapshots": {name: {"source": name} for name in ("newton", "mjwarp", "mujoco")},
            "unselected_row_invariance": {},
            "unselected_row_invariance_asserted": True,
        },
        "control_steps": {
            "controls": np.zeros((CONTROL_STEP_COUNT, NUM_ENVS, 29), dtype=np.float32).tolist(),
            "records": {
                name: [
                    {**deepcopy(step_record), "index": index} for index in range(CONTROL_STEP_COUNT)
                ]
                for name in ("newton", "mjwarp", "mujoco")
            },
            "comparisons": {name: {"step_count": CONTROL_STEP_COUNT} for name in COMPARISON_KEYS},
        },
    }


def test_report_validation_requires_complete_durable_schema() -> None:
    report = _minimal_report()
    _validate_report(report)

    wrong_thresholds = deepcopy(report)
    wrong_thresholds["thresholds"] = {"qpos": 0.1}
    with pytest.raises(ValueError, match="must not serialize thresholds"):
        _validate_report(wrong_thresholds)

    acceptance = deepcopy(report)
    acceptance.update(
        {
            "mode": "acceptance",
            "scope": _SCOPES["acceptance"],
            "thresholds": {name: 0.1 for name in _THRESHOLD_FIELDS},
            "assertions": {
                "canonical_initial_reset": True,
                "numerical_thresholds": True,
                "parity_acceptance": True,
                "runtime_device_resident": True,
                "unselected_reset_rows_unchanged": True,
                "zero_copy_session": True,
            },
        }
    )
    _validate_report(acceptance)

    missing_sensor = deepcopy(report)
    del missing_sensor["reset"]["arrays"]["newton"]["sensors"][SCALAR_SENSOR_FIELDS[0]]
    with pytest.raises(ValueError, match="invalid sensor field set"):
        _validate_report(missing_sensor)

    truncated_trajectory = deepcopy(report)
    truncated_trajectory["control_steps"]["records"]["newton"].pop()
    with pytest.raises(ValueError, match="trajectory length"):
        _validate_report(truncated_trajectory)

    invalid_provenance = deepcopy(report)
    invalid_provenance["source_provenance"]["newton_backend_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="invalid SHA256"):
        _validate_report(invalid_provenance)

    missing_runtime = deepcopy(report)
    del missing_runtime["runtime"]["newton"]
    with pytest.raises(ValueError, match="runtime provenance is invalid"):
        _validate_report(missing_runtime)
