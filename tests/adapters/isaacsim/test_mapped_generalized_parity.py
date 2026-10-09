"""Opt-in IsaacSim mapped multi-entity tensor parity evidence.

The fixture uses a fixed articulation, a floating rigid object, and a fixed
table with collisions disabled.  It complements full-G1 contact parity by
checking state, selected reset, body projections, and the two supported scalar
projections without making contact representation part of this contract.
"""

from __future__ import annotations

import gc
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.adapters.isaac.g1_parity_harness import (
    PROFILER_ENVIRONMENT_VARIABLES,
    gpu_compute_process_snapshot,
    gpu_device_snapshot,
    require_acceptance_gpu_idle,
    snapshot_to_numpy,
    write_json_report,
)

SIM_DT = 0.002
NUM_ENVS = 2
SELECTED_ROW = 1
SUBSTEPS = 3
STEP_COUNT = 4
RESET_ATOL = 2e-5
UNSELECTED_ATOL = 2e-5
BODIES = ("pelvis", "torso_link", "object", "table")
BODY_KINDS = ("pos", "quat", "linvel", "angvel")
SCALARS = ("pelvis_local_linvel", "torso_gyro")
BODY_FIELDS = tuple(f"track_{kind}_w_{body}" for body in BODIES for kind in BODY_KINDS)
SENSOR_FIELDS = (*SCALARS, *BODY_FIELDS)
THRESHOLD_ENV = {
    "qpos": "UNISIM_TEST_ISAACSIM_GENERALIZED_QPOS_ATOL",
    "qvel": "UNISIM_TEST_ISAACSIM_GENERALIZED_QVEL_ATOL",
    "body_pos": "UNISIM_TEST_ISAACSIM_GENERALIZED_BODY_POS_ATOL",
    "body_quat_rad": "UNISIM_TEST_ISAACSIM_GENERALIZED_BODY_QUAT_ATOL_RAD",
    "body_velocity": "UNISIM_TEST_ISAACSIM_GENERALIZED_BODY_VEL_ATOL",
    "scalar_sensor": "UNISIM_TEST_ISAACSIM_GENERALIZED_SCALAR_SENSOR_ATOL",
}


@dataclass(frozen=True)
class FixturePaths:
    canonical_scene: Path
    robot: Path
    object: Path
    table: Path

    def report(self) -> dict[str, Any]:
        items = (
            ("canonical_scene", self.canonical_scene),
            ("isaacsim_robot", self.robot),
            ("isaacsim_object", self.object),
            ("isaacsim_table", self.table),
        )
        return {name: {"path": str(path), "sha256": sha256(path)} for name, path in items}


@dataclass(frozen=True)
class Snapshot:
    source: str
    qpos: np.ndarray
    qvel: np.ndarray
    sensors: dict[str, np.ndarray]

    def report(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "qpos_shape": list(self.qpos.shape),
            "qvel_shape": list(self.qvel.shape),
            "sensor_shapes": {k: list(v.shape) for k, v in self.sensors.items()},
        }

    def arrays(self) -> dict[str, Any]:
        return {
            "qpos": self.qpos.tolist(),
            "qvel": self.qvel.tolist(),
            "sensors": {k: v.tolist() for k, v in self.sensors.items()},
        }


@dataclass(frozen=True)
class Thresholds:
    qpos: float
    qvel: float
    body_pos: float
    body_quat_rad: float
    body_velocity: float
    scalar_sensor: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


OPTION = '<option gravity="0 0 0" timestep="0.002" integrator="implicitfast"/>'
ROBOT_BODY = """<body name="pelvis" pos="-1 0 0.5">
  <inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
  <geom name="pelvis_collision" type="sphere" size=".08" mass="0" contype="0" conaffinity="0"/>
  <body name="torso_link" pos="0 0 .3">
    <joint name="drive_joint" type="hinge" axis="0 1 0"/>
    <inertial mass=".5" pos="0 0 0" diaginertia=".003 .003 .003"/>
    <geom name="torso_collision" type="sphere" size=".06" mass="0" contype="0" conaffinity="0"/>
    <site name="imu_in_torso" pos=".1 0 0"/>
  </body>
</body>"""
OBJECT_BODY = """<body name="object" pos="1 0 .8">
  <freejoint/>
  <inertial mass=".7" pos="0 0 0" diaginertia=".004 .005 .006"/>
  <geom name="object_collision" type="box" size=".08 .08 .08" mass="0" contype="0" conaffinity="0"/>
</body>"""
TABLE_BODY = """<body name="table" pos="0 0 -.2">
  <inertial mass="8" pos="0 0 0" diaginertia=".1 .1 .1"/>
  <geom name="table_collision" type="box" size="1 1 .05" mass="0" contype="0" conaffinity="0"/>
</body>"""
ROBOT_TAIL = """</worldbody>
<actuator><position name="drive" joint="drive_joint" kp="20" kv="2"/></actuator>
<sensor>
  <velocimeter site="imu_in_torso" name="pelvis_local_linvel"/>
  <gyro site="imu_in_torso" name="torso_gyro"/>
</sensor></mujoco>"""


def write_fixture(directory: Path) -> FixturePaths:
    directory.mkdir(parents=True, exist_ok=True)
    paths = FixturePaths(
        directory / "generalized-canonical.xml",
        directory / "generalized-robot.xml",
        directory / "generalized-object.xml",
        directory / "generalized-table.xml",
    )
    texts = {
        paths.canonical_scene: (
            '<mujoco model="canonical">'
            + OPTION
            + "<worldbody>"
            + ROBOT_BODY
            + OBJECT_BODY
            + TABLE_BODY
            + ROBOT_TAIL
        ),
        paths.robot: ('<mujoco model="robot">' + OPTION + "<worldbody>" + ROBOT_BODY + ROBOT_TAIL),
        paths.object: (
            '<mujoco model="object">'
            + OPTION
            + "<worldbody>"
            + OBJECT_BODY
            + "</worldbody></mujoco>"
        ),
        paths.table: (
            '<mujoco model="table">' + OPTION + "<worldbody>" + TABLE_BODY + "</worldbody></mujoco>"
        ),
    }
    for path, text in texts.items():
        path.write_text(text, encoding="utf-8")
    return paths


def mapped_scene(paths: FixturePaths) -> Any:
    from unisim.dr.types import ModelSourceDescriptor
    from unisim.entities import EntityInitialState, SceneEntitySpec
    from unisim.scene import SceneCfg

    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                ModelSourceDescriptor(str(paths.robot)),
                root_mode="fixed",
                initial_state=EntityInitialState(position=(-1.0, 0.0, 0.5)),
            ),
            SceneEntitySpec(
                "object",
                ModelSourceDescriptor(str(paths.object)),
                kind="rigid",
                root_mode="floating",
                initial_state=EntityInitialState(position=(1.0, 0.0, 0.8)),
            ),
            SceneEntitySpec(
                "table",
                ModelSourceDescriptor(str(paths.table)),
                kind="rigid",
                root_mode="fixed",
                initial_state=EntityInitialState(position=(0.0, 0.0, -0.2)),
            ),
        )
    )


def reset_values() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    full_q = np.zeros((NUM_ENVS, 8), np.float32)
    full_v = np.zeros((NUM_ENVS, 7), np.float32)
    full_q[:, 0] = 0.1
    full_q[:, 1:4] = (1.0, 0.2, 0.8)
    full_q[:, 4] = 1.0
    full_v[:, 0] = 0.1
    full_v[:, 1:4] = (0.4, -0.2, 0.1)
    full_v[:, 3:6] = (0.35, -0.22, 0.18)
    full_v[:, 6] = 0.2
    selected_q = full_q[[SELECTED_ROW]].copy()
    selected_v = full_v[[SELECTED_ROW]].copy()
    selected_q[:, 0] = 0.4
    selected_q[:, 1:4] = (1.3, -0.3, 0.9)
    selected_v[:, 0] = -0.2
    selected_v[:, 1:4] = (-0.3, 0.4, 0.2)
    selected_v[:, 3:6] = (0.2, 0.3, -0.1)
    selected_v[:, 6] = 0.3
    return full_q, full_v, selected_q, selected_v


def host_snapshot(backend: Any, source: str) -> Snapshot:
    states = backend.get_state(("qpos", "qvel"))
    sensors = {name: snapshot_to_numpy(backend.get_sensor_data(name)) for name in SENSOR_FIELDS}
    return validate_snapshot(
        source,
        np.asarray(states["qpos"], np.float32).copy(),
        np.asarray(states["qvel"], np.float32).copy(),
        sensors,
    )


def device_snapshot(backend: Any, source: str, *, include_aggregate: bool = False) -> Snapshot:
    states = backend.get_state_views(("qpos", "qvel"))
    sensors = {name: snapshot_to_numpy(backend.get_sensor_view(name)) for name in SENSOR_FIELDS}
    if source == "isaacsim" and include_aggregate:
        tracked = backend.get_tracked_body_views()
        aggregate = {
            f"track_{kind}_w_{body}": snapshot_to_numpy(getattr(tracked, field)[..., index, :])
            for index, body in enumerate(tracked.body_names)
            for kind, field in (
                ("pos", "pos_w"),
                ("quat", "quat_w"),
                ("linvel", "lin_vel_w"),
                ("angvel", "ang_vel_w"),
            )
        }
        sensors.update(aggregate)
    return validate_snapshot(
        source,
        snapshot_to_numpy(states["qpos"]),
        snapshot_to_numpy(states["qvel"]),
        sensors,
    )


def validate_snapshot(
    source: str, qpos: np.ndarray, qvel: np.ndarray, sensors: dict[str, np.ndarray]
) -> Snapshot:
    if qpos.shape != (NUM_ENVS, 8) or qvel.shape != (NUM_ENVS, 7):
        raise ValueError(f"{source} state shape mismatch")
    for name, values in sensors.items():
        expected = (NUM_ENVS, 4 if "quat" in name else 3)
        if values.shape != expected:
            raise ValueError(f"{source} sensor {name} shape {values.shape} != {expected}")
    if not all(np.isfinite(x).all() for x in (qpos, qvel, *sensors.values())):
        raise ValueError(f"{source} snapshot contains non-finite values")
    return Snapshot(source, qpos, qvel, sensors)


def vector_metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise ValueError(f"parity shape mismatch {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("parity arrays must be finite")
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"max_abs": float(np.max(diff)), "rms": float(np.sqrt(np.mean(diff**2)))}


def quaternion_metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    metric = vector_metric(actual, expected)
    if not np.allclose(np.linalg.norm(actual, axis=-1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("parity quaternions must be unit norm")
    if not np.allclose(np.linalg.norm(expected, axis=-1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("parity quaternions must be unit norm")
    dot = np.abs(np.sum(actual * expected, axis=-1))
    metric["max_dot_abs"] = float(np.max(dot))
    metric["max_angle_rad"] = float(np.max(2 * np.arccos(np.clip(dot, -1, 1))))
    return metric


def compare(reference: Snapshot, candidate: Snapshot) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "qpos": vector_metric(candidate.qpos, reference.qpos),
        "qvel": vector_metric(candidate.qvel, reference.qvel),
    }
    for name in SENSOR_FIELDS:
        values = (candidate.sensors[name], reference.sensors[name])
        metrics[name] = quaternion_metric(*values) if "quat" in name else vector_metric(*values)
    metrics["summary"] = {
        "max_all": max(metric["max_abs"] for metric in metrics.values()),
        "max_qpos": metrics["qpos"]["max_abs"],
        "max_qvel": metrics["qvel"]["max_abs"],
    }
    return metrics


def assert_unselected(before: Snapshot, after: Snapshot, source: str) -> None:
    row = 1 - SELECTED_ROW
    for field in ("qpos", "qvel"):
        error = vector_metric(getattr(after, field)[row], getattr(before, field)[row])["max_abs"]
        assert error <= UNSELECTED_ATOL, f"{source} unselected {field} changed: {error}"


def assert_reset(metrics: dict[str, Any]) -> None:
    worst = max(metrics["qpos"]["max_abs"], metrics["qvel"]["max_abs"])
    assert worst <= RESET_ATOL, f"selected reset state parity exceeded {RESET_ATOL}: {worst}"


def assert_publication(full: Snapshot, selected: Snapshot, stepped: Snapshot | None = None) -> None:
    row = SELECTED_ROW
    body_delta = vector_metric(
        selected.sensors["track_pos_w_object"][row], full.sensors["track_pos_w_object"][row]
    )["max_abs"]
    scalar_delta = max(
        vector_metric(selected.sensors[name][row], full.sensors[name][row])["max_abs"]
        for name in SCALARS
    )
    assert body_delta > 1e-5 and scalar_delta > 1e-5, "reset did not refresh publications"
    if stepped is not None:
        assert vector_metric(stepped.qpos[row], selected.qpos[row])["max_abs"] > 1e-5
        assert (
            vector_metric(
                stepped.sensors["track_pos_w_object"][row],
                selected.sensors["track_pos_w_object"][row],
            )["max_abs"]
            > 1e-6
        )


def mode() -> str:
    value = os.environ.get("UNISIM_TEST_ISAACSIM_GENERALIZED_PARITY_MODE", "diagnostic")
    if value not in {"diagnostic", "acceptance"}:
        raise RuntimeError("generalized parity mode must be diagnostic or acceptance")
    return value


def thresholds() -> Thresholds:
    missing = [name for name in THRESHOLD_ENV.values() if not os.environ.get(name)]
    if missing:
        raise RuntimeError("missing generalized thresholds: " + ", ".join(missing))
    values: dict[str, float] = {}
    for field, name in THRESHOLD_ENV.items():
        try:
            value = float(os.environ[name])
        except ValueError as error:
            raise RuntimeError(f"{name} must be finite") from error
        if not np.isfinite(value) or value < 0:
            raise RuntimeError(f"{name} must be finite and non-negative")
        values[field] = value
    return Thresholds(**values)


def assert_threshold(metrics: dict[str, Any], limits: Thresholds) -> None:
    worst = {
        "qpos": 0.0,
        "qvel": 0.0,
        "body_pos": 0.0,
        "body_quat": 0.0,
        "body_velocity": 0.0,
        "scalar_sensor": 0.0,
    }
    for name, metric in metrics.items():
        if name == "summary":
            continue
        if name == "qpos":
            group = "qpos"
        elif name == "qvel":
            group = "qvel"
        elif name.startswith("track_pos_w_"):
            group = "body_pos"
        elif name.startswith("track_quat_w_"):
            group = "body_quat"
        elif name.startswith(("track_linvel_w_", "track_angvel_w_")):
            group = "body_velocity"
        elif name in SCALARS:
            group = "scalar_sensor"
        else:
            group = None
        if group:
            metric_key = "max_angle_rad" if group == "body_quat" else "max_abs"
            worst[group] = max(worst[group], metric[metric_key])
    limit_values = {
        "qpos": limits.qpos,
        "qvel": limits.qvel,
        "body_pos": limits.body_pos,
        "body_quat": limits.body_quat_rad,
        "body_velocity": limits.body_velocity,
        "scalar_sensor": limits.scalar_sensor,
    }
    for group, value in worst.items():
        assert value <= limit_values[group], f"{group} parity {value} > {limit_values[group]}"


def materialization_report(backend: Any) -> dict[str, Any]:
    report = getattr(backend, "_worker_materialization_report", None) or {}
    raw = report.get("raw_usd_cache", {}).get("entries", ())
    role = report.get("role_usd_cache", {}).get("entries", ())
    assert raw and role, "missing USD materialization provenance"
    versions: dict[str, str] = {}
    for entry in raw:
        versions.update({str(k): str(v) for k, v in entry.get("runtime_versions", {}).items()})
    assert versions, "missing worker runtime versions"
    return {"raw_usd": list(raw), "role_usd": list(role), "runtime_versions": versions}


def source_provenance(paths: FixturePaths) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    names = {
        "backend": root / "src/unisim/backend/isaacsim/backend.py",
        "scene_worker": root / "src/unisim/backend/isaacsim/scene_worker.py",
        "scene_materialization": root
        / "src/unisim/backend/subprocess_ipc/scene_materialization.py",
        "harness": Path(__file__).resolve(),
    }
    return {
        "fixtures": paths.report(),
        "sources": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in names.items()
        },
    }


def run_parity(output_path: Path) -> dict[str, Any]:
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    torch = pytest.importorskip("torch")
    warp = pytest.importorskip("warp")
    current_mode = mode()
    limits = None if current_mode == "diagnostic" else thresholds()
    profilers = {name: os.environ.get(name, "") for name in PROFILER_ENVIRONMENT_VARIABLES}
    if current_mode == "acceptance" and any(profilers.values()):
        raise RuntimeError("generalized acceptance must not enable worker profilers")
    compute_processes_before = (
        require_acceptance_gpu_idle(0)
        if current_mode == "acceptance"
        else gpu_compute_process_snapshot(0)
    )

    warp.init()
    if not torch.cuda.is_available() or not bool(warp.get_device().is_cuda):
        pytest.skip("generalized IsaacSim parity requires CUDA Torch and Warp")

    paths = write_fixture(output_path.parent)
    from unisim import IsaacSimBackend, MjwarpBackend, MuJoCoBackend
    from unisim.scene import SceneCfg

    canonical = SceneCfg(model_file=str(paths.canonical_scene))
    mujoco_backend = MuJoCoBackend(
        canonical,
        NUM_ENVS,
        SIM_DT,
        base_name="pelvis",
        add_body_sensors=True,
        tracked_body_names=BODIES,
    )
    mjwarp_backend = MjwarpBackend(
        canonical, NUM_ENVS, SIM_DT, base_name="pelvis", add_body_sensors=True
    )
    isaacsim_backend = IsaacSimBackend(
        mapped_scene(paths),
        NUM_ENVS,
        SIM_DT,
        device_id=0,
        worker_timeout_s=120.0,
        tensor_cuda_ipc=True,
    )
    mujoco_backend.materialize()
    mjwarp_backend.materialize()
    isaacsim_backend.materialize()
    public_widths = isaacsim_backend.get_public_state_widths()
    sensor_inventory = isaacsim_backend.get_sensor_inventory()
    tracked_body_names = isaacsim_backend.get_tracked_body_views().body_names
    assert (public_widths.nq, public_widths.nv) == (8, 7)
    assert set(SCALARS) <= {descriptor.name for descriptor in sensor_inventory}
    assert {
        descriptor.name for descriptor in sensor_inventory if descriptor.name.startswith("track_")
    } == set(BODY_FIELDS)
    assert tracked_body_names == BODIES
    full_q, full_v, selected_q, selected_v = reset_values()
    control = np.full((NUM_ENVS, 1), 0.5, np.float32)
    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": current_mode,
        "backend": "isaacsim",
        "scope": {
            "scene": "mapped_multi_entity_collision_disabled",
            "primary_reference": "mjwarp",
            "secondary_reference": "mujoco",
            "reference_reason": (
                "MJWarp shares the device-resident control-boundary publication; "
                "MuJoCo is independent CPU evidence"
            ),
            "not_contact_parity": True,
        },
        "num_envs": NUM_ENVS,
        "selected_row": SELECTED_ROW,
        "sim_dt": SIM_DT,
        "control_substeps": SUBSTEPS,
        "control_step_count": STEP_COUNT,
        "sensor_contract": {
            "scalar_sensors": list(SCALARS),
            "bodies": list(BODIES),
            "body_sensor_fields": list(BODY_FIELDS),
        },
        "gpu": {
            "required_idle": current_mode == "acceptance",
            "compute_processes_before": compute_processes_before,
            "own_process_pid": os.getpid(),
        },
        "gpu_device": gpu_device_snapshot(0),
        "profiler_environment": profilers,
        "host_runtime_versions": {
            "torch": torch.__version__,
            "mujoco": pytest.importorskip("mujoco").__version__,
            "warp": warp.__version__,
            "mujoco_warp": pytest.importorskip("mujoco_warp").__version__,
        },
    }
    report.update(source_provenance(paths))
    cuda = torch.device("cuda:0")
    all_t = torch.tensor([0, 1], dtype=torch.int64, device=cuda)
    selected_t = torch.tensor([SELECTED_ROW], dtype=torch.int64, device=cuda)
    full_qt = torch.tensor(full_q, device=cuda)
    full_vt = torch.tensor(full_v, device=cuda)
    selected_qt = torch.tensor(selected_q, device=cuda)
    selected_vt = torch.tensor(selected_v, device=cuda)
    control_t = torch.tensor(control, device=cuda)
    all_np = np.array([0, 1], np.int64)
    selected_np = np.array([SELECTED_ROW], np.int64)
    trajectories = {"mujoco": [], "mjwarp": [], "isaacsim": []}
    try:
        mujoco_backend.set_state(all_np, full_q, full_v)
        mjwarp_backend.set_state_tensor(all_t, full_qt, full_vt)
        isaacsim_backend.set_state_tensor(all_t, full_qt, full_vt)
        mujoco_full = host_snapshot(mujoco_backend, "mujoco")
        mjwarp_full = device_snapshot(mjwarp_backend, "mjwarp")
        isaacsim_full = device_snapshot(isaacsim_backend, "isaacsim")

        mujoco_backend.set_state(selected_np, selected_q, selected_v)
        mjwarp_backend.set_state_tensor(selected_t, selected_qt, selected_vt)
        isaacsim_backend.set_state_tensor(selected_t, selected_qt, selected_vt)
        mujoco_selected = host_snapshot(mujoco_backend, "mujoco")
        mjwarp_selected = device_snapshot(mjwarp_backend, "mjwarp")
        isaacsim_selected = device_snapshot(isaacsim_backend, "isaacsim")
        isaacsim_selected_aggregate = device_snapshot(
            isaacsim_backend, "isaacsim", include_aggregate=True
        )
        for before, after, name in (
            (mujoco_full, mujoco_selected, "mujoco"),
            (mjwarp_full, mjwarp_selected, "mjwarp"),
            (isaacsim_full, isaacsim_selected, "isaacsim"),
        ):
            assert_unselected(before, after, name)
        assert_publication(isaacsim_full, isaacsim_selected)
        assert_publication(isaacsim_full, isaacsim_selected_aggregate)
        reset_metrics = {
            "isaacsim_vs_mujoco": compare(mujoco_selected, isaacsim_selected),
            "isaacsim_vs_mjwarp": compare(mjwarp_selected, isaacsim_selected),
            "mjwarp_vs_mujoco": compare(mujoco_selected, mjwarp_selected),
        }
        for metric in reset_metrics.values():
            assert_reset(metric)

        for _ in range(STEP_COUNT):
            mujoco_backend.step(control, nsteps=SUBSTEPS)
            mjwarp_backend.step_tensor(control_t, nsteps=SUBSTEPS)
            isaacsim_backend.step_tensor(control_t, nsteps=SUBSTEPS)
            trajectories["mujoco"].append(host_snapshot(mujoco_backend, "mujoco").arrays())
            trajectories["mjwarp"].append(device_snapshot(mjwarp_backend, "mjwarp").arrays())
            trajectories["isaacsim"].append(device_snapshot(isaacsim_backend, "isaacsim").arrays())
        final_mujoco = host_snapshot(mujoco_backend, "mujoco")
        final_mjwarp = device_snapshot(mjwarp_backend, "mjwarp")
        final_isaacsim = device_snapshot(isaacsim_backend, "isaacsim")
        assert_publication(isaacsim_full, isaacsim_selected, final_isaacsim)
        step_metrics = {
            "isaacsim_vs_mujoco": compare(final_mujoco, final_isaacsim),
            "isaacsim_vs_mjwarp": compare(final_mjwarp, final_isaacsim),
        }
        report["isaacsim_tensor_capabilities"] = {
            "execution": "device_resident",
            "process_topology": "external_worker",
            "data_plane": "cuda_ipc",
            "state_fields": ["qpos", "qvel"],
            "selected_reset": True,
            "selected_reset_publication": "authoritative_views",
            "sensor_views": True,
            "tracked_body_views": True,
            "public_state_widths": {"nq": public_widths.nq, "nv": public_widths.nv},
            "sensor_inventory": {
                descriptor.name: descriptor.width for descriptor in sensor_inventory
            },
            "tracked_body_names": list(tracked_body_names),
        }
        report["materialization"] = materialization_report(isaacsim_backend)
        report["reset"] = {
            "snapshots": {
                "mujoco": mujoco_selected.report(),
                "mjwarp": mjwarp_selected.report(),
                "isaacsim": isaacsim_selected.report(),
            },
            "arrays": {
                "mujoco": mujoco_selected.arrays(),
                "mjwarp": mjwarp_selected.arrays(),
                "isaacsim": isaacsim_selected.arrays(),
            },
            "comparisons": reset_metrics,
            "state_asserted": True,
            "state_atol": RESET_ATOL,
            "unselected_row_atol": UNSELECTED_ATOL,
        }
        report["control_steps"] = {
            "controls": control.tolist(),
            "arrays": trajectories,
            "final_comparisons": step_metrics,
            "asserted": limits is not None,
            "thresholds": None if limits is None else limits.__dict__,
        }
    finally:
        del isaacsim_selected_aggregate
        del all_t, selected_t, full_qt, full_vt, selected_qt, selected_vt, control_t
        gc.collect()
        isaacsim_backend.close()
        mjwarp_backend.close()
        mujoco_backend.close()
        gc.collect()
    if current_mode == "acceptance":
        report["gpu"]["compute_processes_after"] = require_acceptance_gpu_idle(
            0, quiesce_timeout_s=10.0, allowed_pids={os.getpid()}
        )
    write_json_report(output_path, report)
    if limits is not None:
        assert_threshold(step_metrics["isaacsim_vs_mujoco"], limits)
        assert_threshold(step_metrics["isaacsim_vs_mjwarp"], limits)
    return report


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACSIM_GENERALIZED_PARITY_NATIVE") != "1",
    reason="set UNISIM_TEST_ISAACSIM_GENERALIZED_PARITY_NATIVE=1",
)
def test_isaacsim_mapped_generalized_tensor_parity(tmp_path: Path) -> None:
    output = os.environ.get("UNISIM_TEST_ISAACSIM_GENERALIZED_PARITY_OUTPUT")
    if output is None and mode() == "acceptance":
        raise RuntimeError("generalized acceptance requires an explicit artifact output")
    output_path = Path(output or tmp_path / "isaacsim-mapped-generalized-parity.json")
    assert run_parity(output_path)["isaacsim_tensor_capabilities"]["data_plane"] == "cuda_ipc"


def test_fixture_mapped_layout_matches_canonical(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    import mujoco

    from unisim import IsaacSimBackend

    paths = write_fixture(tmp_path)
    backend = IsaacSimBackend(mapped_scene(paths), NUM_ENVS, SIM_DT, tensor_cuda_ipc=True)
    model = mujoco.MjModel.from_xml_path(str(paths.canonical_scene))
    try:
        layout = backend.get_scene_layout()
        assert (layout.nq, layout.nv, layout.nu, layout.nbody) == (8, 7, 1, 5)
        assert [entity.root_mode for entity in layout.entities] == ["fixed", "floating", "fixed"]
        assert [joint.name for e in layout.entities for joint in e.joints] == ["drive_joint"]
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)
        ]
        assert names == list(BODIES)
        assert [body for e in layout.entities for body in e.body_names] == names
    finally:
        backend.close()


def test_public_tensor_construction_contract_is_inventory_backed(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("torch")
    import mujoco

    from unisim import IsaacSimBackend

    paths = write_fixture(tmp_path)
    backend = IsaacSimBackend(mapped_scene(paths), NUM_ENVS, SIM_DT, tensor_cuda_ipc=True)
    model = mujoco.MjModel.from_xml_path(str(paths.canonical_scene))
    try:
        capabilities = backend.get_tensor_capabilities()
        assert capabilities.execution.value == "device_resident"
        assert capabilities.process_topology.value == "external_worker"
        assert capabilities.data_plane.value == "cuda_ipc"
        assert capabilities.selected_reset_publication is not None
        assert capabilities.selected_reset_publication.value == "authoritative_views"
        assert capabilities.tracked_body_views
        widths = backend.get_public_state_widths()
        assert (widths.nq, widths.nv) == (model.nq, model.nv)
        layout = backend.get_scene_layout()
        assert (widths.nq, widths.nv) == (layout.nq, layout.nv)
        inventory = backend.get_sensor_inventory()
        assert set(SCALARS) <= {descriptor.name for descriptor in inventory}
        assert {
            descriptor.name for descriptor in inventory if descriptor.name.startswith("track_")
        } == set(BODY_FIELDS)
        assert all(
            descriptor.width == (4 if "quat" in descriptor.name else 3)
            for descriptor in inventory
        )
        views = backend.get_tracked_body_views()
        assert views.body_names == BODIES
        assert views.pos_w.shape == (NUM_ENVS, len(BODIES), 3)
        assert views.quat_w.shape == (NUM_ENVS, len(BODIES), 4)
        assert views.lin_vel_w.shape == (NUM_ENVS, len(BODIES), 3)
        assert views.ang_vel_w.shape == (NUM_ENVS, len(BODIES), 3)
    finally:
        backend.close()


def test_sensor_contract_is_exact() -> None:
    assert len(SCALARS) == 2
    assert len(BODY_FIELDS) == 16
    assert len(SENSOR_FIELDS) == 18
    assert "track_pos_w_object" in SENSOR_FIELDS
    assert "track_quat_w_table" in SENSOR_FIELDS


def test_comparison_detects_state_body_scalar_and_quaternion_drift() -> None:
    sensors = {
        name: np.zeros((NUM_ENVS, 4 if "quat" in name else 3), np.float32) for name in SENSOR_FIELDS
    }
    for values in sensors.values():
        if values.shape[-1] == 4:
            values[..., 0] = 1.0
    reference = Snapshot("r", np.zeros((2, 8)), np.zeros((2, 7)), sensors)
    drifted = {
        name: values.copy() + (0.2 if name in {"track_pos_w_object", "torso_gyro"} else 0)
        for name, values in sensors.items()
    }
    drifted["track_quat_w_object"][:, 1] = np.sin(0.2)
    drifted["track_quat_w_object"][:, 0] = np.cos(0.2)
    candidate = Snapshot(
        "c", np.full((2, 8), 0.1, np.float32), np.full((2, 7), 0.2, np.float32), drifted
    )
    metrics = compare(reference, candidate)
    assert metrics["qpos"]["max_abs"] == pytest.approx(0.1)
    assert metrics["qvel"]["max_abs"] == pytest.approx(0.2)
    assert metrics["track_pos_w_object"]["max_abs"] == pytest.approx(0.2)
    assert metrics["torso_gyro"]["max_abs"] == pytest.approx(0.2)
    assert metrics["track_quat_w_object"]["max_angle_rad"] == pytest.approx(0.4)


def test_quaternion_metric_rejects_nonunit_and_normalizes_sign() -> None:
    actual = np.asarray([[1, 0, 0, 0], [0, -1, 0, 0]], np.float32)
    expected = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    assert quaternion_metric(actual, expected)["max_angle_rad"] == pytest.approx(0)
    with pytest.raises(ValueError, match="unit norm"):
        quaternion_metric(actual * 2, expected)


def test_mode_and_thresholds_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNISIM_TEST_ISAACSIM_GENERALIZED_PARITY_MODE", "bad")
    with pytest.raises(RuntimeError, match="diagnostic or acceptance"):
        mode()
    monkeypatch.setenv("UNISIM_TEST_ISAACSIM_GENERALIZED_PARITY_MODE", "acceptance")
    for name in THRESHOLD_ENV.values():
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="missing generalized thresholds"):
        thresholds()
    for name in THRESHOLD_ENV.values():
        monkeypatch.setenv(name, "nan")
    with pytest.raises(RuntimeError, match="finite and non-negative"):
        thresholds()


def test_materialization_provenance_fail_closed() -> None:
    with pytest.raises(AssertionError, match="materialization provenance"):
        materialization_report(SimpleNamespace(_worker_materialization_report={}))
    with pytest.raises(AssertionError, match="worker runtime versions"):
        materialization_report(
            SimpleNamespace(
                _worker_materialization_report={
                    "raw_usd_cache": {"entries": ({"runtime_versions": {}},)},
                    "role_usd_cache": {"entries": ({},)},
                }
            )
        )
