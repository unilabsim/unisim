"""Reusable full-G1 parity fixtures and metric helpers for Isaac tensor runs.

The module deliberately keeps SDK imports inside the opt-in runtime path.  The
SDK-free tests exercise fixture resolution, capability serialization, and metric
fail-closed behavior without starting either external Isaac worker.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from unisim.backend.base import (
    SelectedResetPublication,
    TensorDataPlane,
    TensorExecution,
    TensorProcessTopology,
)

UNILAB_ROOT = Path(__file__).resolve().parents[4] / "UniLab"
DEFAULT_CANONICAL_SCENE = (
    UNILAB_ROOT / "src" / "unilab" / "assets" / "robots" / "g1" / "scene_flat.xml"
)
DEFAULT_ISAACSIM_FIXTURE = UNILAB_ROOT / "tests" / "fixtures" / "isaacsim_g1_tensor_cuda_ipc"
DEFAULT_ISAACSIM_ROBOT = DEFAULT_ISAACSIM_FIXTURE / "g1_stand_entity.xml"
DEFAULT_ISAACSIM_FLOOR = DEFAULT_ISAACSIM_FIXTURE / "flat_floor_entity.xml"
DEFAULT_ISAACSIM_CONTACT_SENSORS = DEFAULT_ISAACSIM_FIXTURE / "g1_floor_contact_sensors.xml"
ISAACSIM_CONTACT_SENSOR_FIELDS = (
    "left_foot_floor_force",
    "right_foot_floor_force",
)
G1_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
TRACKED_SENSOR_FIELDS = (
    *(f"track_pos_w_{name}" for name in TRACKED_BODIES),
    *(f"track_quat_w_{name}" for name in TRACKED_BODIES),
    *(f"track_linvel_w_{name}" for name in TRACKED_BODIES),
    *(f"track_angvel_w_{name}" for name in TRACKED_BODIES),
)
SCALAR_SENSOR_FIELDS = ("pelvis_local_linvel", "torso_gyro")
PROFILER_ENVIRONMENT_VARIABLES = (
    "UNISIM_ISAAC_WORKER_PROFILE_TRACE",
    "UNISIM_ISAAC_WORKER_PROFILE_START_COMMAND",
    "UNISIM_ISAAC_WORKER_PROFILE_STOP_COMMAND",
)


@dataclass(frozen=True)
class G1FixturePaths:
    canonical_scene: Path
    isaacsim_robot: Path
    isaacsim_floor: Path
    isaacsim_contact_sensors: Path

    def report(self) -> dict[str, Any]:
        return {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("canonical_scene", self.canonical_scene),
                ("isaacsim_robot", self.isaacsim_robot),
                ("isaacsim_floor", self.isaacsim_floor),
                ("isaacsim_contact_sensors", self.isaacsim_contact_sensors),
            )
        }


@dataclass(frozen=True)
class G1Snapshot:
    source: str
    qpos: np.ndarray
    qvel: np.ndarray
    sensors: dict[str, np.ndarray]

    def report(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "qpos_shape": list(self.qpos.shape),
            "qvel_shape": list(self.qvel.shape),
            "sensor_shapes": {name: list(values.shape) for name, values in self.sensors.items()},
        }

    def array_report(self) -> dict[str, Any]:
        return {
            "qpos": self.qpos.tolist(),
            "qvel": self.qvel.tolist(),
            "sensors": {name: values.tolist() for name, values in self.sensors.items()},
        }


@dataclass(frozen=True)
class ArrayMetric:
    max_abs: float
    rms: float

    def report(self) -> dict[str, float]:
        return {"max_abs": self.max_abs, "rms": self.rms}


@dataclass(frozen=True)
class QuaternionMetric:
    max_dot_abs: float
    max_angle_rad: float

    def report(self) -> dict[str, float]:
        return {
            "max_dot_abs": self.max_dot_abs,
            "max_angle_rad": self.max_angle_rad,
        }


@dataclass(frozen=True)
class ParityThresholds:
    qpos: float
    qvel: float
    body_pos: float
    body_quat_rad: float
    body_velocity: float
    scalar_sensor: float


@dataclass(frozen=True)
class G1ControlStep:
    index: int
    control: np.ndarray
    snapshot: G1Snapshot


@dataclass(frozen=True)
class SensorStaleWindow:
    qpos_change: float
    qvel_change: float
    max_sensor_abs: float

    def report(self) -> dict[str, float]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_g1_fixture_paths() -> G1FixturePaths:
    """Resolve explicit fixture paths, falling back to the sibling UniLab checkout."""

    def resolve(environment_name: str, default: Path) -> Path:
        value = os.environ.get(environment_name)
        candidate = Path(value).expanduser() if value else default
        if candidate.is_file():
            return candidate.resolve()
        if value is not None:
            raise FileNotFoundError(
                f"G1 parity fixture {environment_name} is not a file: {candidate}"
            )
        return candidate

    return G1FixturePaths(
        canonical_scene=resolve("UNISIM_TEST_G1_CANONICAL_SCENE", DEFAULT_CANONICAL_SCENE),
        isaacsim_robot=resolve("UNISIM_TEST_G1_ISAACSIM_ROBOT", DEFAULT_ISAACSIM_ROBOT),
        isaacsim_floor=resolve("UNISIM_TEST_G1_ISAACSIM_FLOOR", DEFAULT_ISAACSIM_FLOOR),
        isaacsim_contact_sensors=resolve(
            "UNISIM_TEST_G1_ISAACSIM_CONTACT_SENSORS",
            DEFAULT_ISAACSIM_CONTACT_SENSORS,
        ),
    )


def parse_stand_fixture(
    canonical_scene: Path, isaacsim_robot: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Read and cross-check the canonical and mapped stand qpos/ctrl values."""

    import xml.etree.ElementTree as ET

    def values(path: Path) -> tuple[np.ndarray, np.ndarray]:
        root = ET.parse(path).getroot()
        key = root.find("./keyframe/key[@name='stand']")
        if key is None or key.get("qpos") is None or key.get("ctrl") is None:
            raise ValueError(f"{path} does not declare a complete stand keyframe")
        return (
            np.asarray([float(value) for value in key.attrib["qpos"].split()], dtype=np.float32),
            np.asarray([float(value) for value in key.attrib["ctrl"].split()], dtype=np.float32),
        )

    canonical_qpos, canonical_ctrl = values(canonical_scene)
    mapped_qpos, mapped_ctrl = values(isaacsim_robot)
    np.testing.assert_allclose(mapped_qpos, canonical_qpos, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(mapped_ctrl, canonical_ctrl, atol=0.0, rtol=0.0)
    if canonical_qpos.shape != (7 + len(G1_JOINTS),):
        raise ValueError(f"canonical G1 qpos has shape {canonical_qpos.shape}")
    if canonical_ctrl.shape != (len(G1_JOINTS),):
        raise ValueError(f"canonical G1 ctrl has shape {canonical_ctrl.shape}")
    return canonical_qpos, canonical_ctrl


def validate_mapped_robot_source(canonical_scene: Path, isaacsim_robot: Path) -> dict[str, Any]:
    """Fail closed before Kit startup when the mapped robot source drifts numerically."""

    import mujoco

    canonical = mujoco.MjModel.from_xml_path(str(canonical_scene))
    mapped = mujoco.MjModel.from_xml_path(str(isaacsim_robot))

    # The explicit loops keep this diagnostic independent from model-private helpers.
    joint_names: list[str] = []
    for index in range(canonical.njnt):
        joint_names.append(canonical.joint(index).name)
    mapped_joint_names = [mapped.joint(index).name for index in range(mapped.njnt)]
    if joint_names != mapped_joint_names:
        raise ValueError("mapped G1 joint order differs from canonical scene")

    actuator_names = [canonical.actuator(index).name for index in range(canonical.nu)]
    mapped_actuator_names = [mapped.actuator(index).name for index in range(mapped.nu)]
    if actuator_names != mapped_actuator_names:
        raise ValueError("mapped G1 actuator order differs from canonical scene")

    body_names = [canonical.body(index).name for index in range(1, canonical.nbody)]
    mapped_body_names = [mapped.body(index).name for index in range(1, mapped.nbody)]
    if body_names != mapped_body_names:
        raise ValueError("mapped G1 body order differs from canonical scene")

    # The canonical scene owns its floor geom on the world body; the mapped
    # profile carries the floor as a separate rigid entity instead.
    canonical_robot_geom_ids = [
        index for index in range(canonical.ngeom) if int(canonical.geom_bodyid[index]) != 0
    ]
    geom_names = [canonical.geom(index).name for index in canonical_robot_geom_ids]
    mapped_geom_names = [mapped.geom(index).name for index in range(mapped.ngeom)]
    if geom_names != mapped_geom_names:
        raise ValueError("mapped G1 geometry identity differs from canonical scene")
    canonical_geom_body_names = [
        canonical.body(int(canonical.geom_bodyid[index])).name for index in canonical_robot_geom_ids
    ]
    mapped_geom_body_names = [
        mapped.body(int(mapped.geom_bodyid[index])).name for index in range(mapped.ngeom)
    ]
    if canonical_geom_body_names != mapped_geom_body_names:
        raise ValueError("mapped G1 geometry ownership differs from canonical scene")

    for field in (
        "body_mass",
        "body_ipos",
        "body_inertia",
        "body_iquat",
        "body_pos",
        "body_quat",
        "dof_armature",
        "dof_damping",
        "dof_frictionloss",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "jnt_type",
        "jnt_limited",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_gear",
    ):
        expected = np.asarray(getattr(canonical, field))
        actual = np.asarray(getattr(mapped, field))
        if expected.shape != actual.shape:
            raise ValueError(f"mapped G1 field {field} changed shape")
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7, err_msg=field)

    for field in (
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
        "geom_solref",
        "geom_solimp",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
    ):
        expected = np.asarray(getattr(canonical, field))[canonical_robot_geom_ids]
        actual = np.asarray(getattr(mapped, field))
        if expected.shape != actual.shape:
            raise ValueError(f"mapped G1 field {field} changed shape")
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7, err_msg=field)

    return {
        "joint_count": len(joint_names),
        "actuator_count": len(actuator_names),
        "body_count": len(body_names),
        "geom_count": len(geom_names),
    }


def expected_isaac_cuda_ipc_capabilities() -> dict[str, Any]:
    return {
        "execution": TensorExecution.DEVICE_RESIDENT.value,
        "state_views": True,
        "state_fields": ["qpos", "qvel"],
        "sensor_views": True,
        "stepping": True,
        "selected_reset": True,
        "selected_reset_publication": SelectedResetPublication.AUTHORITATIVE_VIEWS.value,
        "tracked_body_views": True,
        "reset_randomization": False,
        "fixed_variants": False,
        "host_pre_step_control": False,
        "packed_host_bridge": False,
        "process_topology": TensorProcessTopology.EXTERNAL_WORKER.value,
        "data_plane": TensorDataPlane.CUDA_IPC.value,
        "torch_devices": ["cuda"],
    }


def serialize_capabilities(capabilities: Any) -> dict[str, Any]:
    return {
        "execution": capabilities.execution.value,
        "state_views": bool(capabilities.state_views),
        "state_fields": sorted(capabilities.state_fields),
        "sensor_views": bool(capabilities.sensor_views),
        "stepping": bool(capabilities.stepping),
        "selected_reset": bool(capabilities.selected_reset),
        "selected_reset_publication": (
            None
            if capabilities.selected_reset_publication is None
            else capabilities.selected_reset_publication.value
        ),
        "tracked_body_views": bool(capabilities.tracked_body_views),
        "reset_randomization": bool(capabilities.reset_randomization),
        "fixed_variants": bool(capabilities.fixed_variants),
        "host_pre_step_control": bool(capabilities.host_pre_step_control),
        "packed_host_bridge": bool(capabilities.packed_host_bridge),
        "process_topology": capabilities.process_topology.value,
        "data_plane": capabilities.data_plane.value,
        "torch_devices": list(capabilities.torch_devices),
    }


def assert_expected_isaac_cuda_ipc_capabilities(backend: Any) -> dict[str, Any]:
    actual = serialize_capabilities(backend.get_tensor_capabilities())
    expected = expected_isaac_cuda_ipc_capabilities()
    assert actual == expected
    return actual


def array_metric(actual: np.ndarray, expected: np.ndarray) -> ArrayMetric:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        raise ValueError(f"parity shape mismatch: {actual_array.shape} != {expected_array.shape}")
    if not np.isfinite(actual_array).all() or not np.isfinite(expected_array).all():
        raise ValueError("parity arrays must be finite")
    delta = actual_array - expected_array
    return ArrayMetric(
        max_abs=float(np.max(np.abs(delta))),
        rms=float(np.sqrt(np.mean(np.square(delta)))),
    )


def quaternion_metric(actual: np.ndarray, expected: np.ndarray) -> QuaternionMetric:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape or actual_array.shape[-1] != 4:
        raise ValueError(f"invalid quaternion parity shape: {actual_array.shape}")
    if not np.isfinite(actual_array).all() or not np.isfinite(expected_array).all():
        raise ValueError("parity quaternions must be finite")
    actual_unit = np.linalg.norm(actual_array, axis=-1)
    expected_unit = np.linalg.norm(expected_array, axis=-1)
    if not np.allclose(actual_unit, 1.0, rtol=0.0, atol=2e-3):
        raise ValueError(f"non-unit actual quaternions: {actual_unit}")
    if not np.allclose(expected_unit, 1.0, rtol=0.0, atol=2e-3):
        raise ValueError(f"non-unit expected quaternions: {expected_unit}")
    dots = np.abs(np.sum(actual_array * expected_array, axis=-1))
    angles = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    return QuaternionMetric(
        max_dot_abs=float(np.max(dots)),
        max_angle_rad=float(np.max(angles)),
    )


def compare_snapshots(
    reference: G1Snapshot,
    candidate: G1Snapshot,
    *,
    include_step: bool,
    include_sensors: bool = True,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "qpos": array_metric(candidate.qpos, reference.qpos).report(),
        "qvel": array_metric(candidate.qvel, reference.qvel).report(),
    }
    body_metrics: dict[str, dict[str, float]] = {}
    if include_sensors:
        for name in SCALAR_SENSOR_FIELDS:
            metrics[name] = array_metric(
                candidate.sensors[name], reference.sensors[name]
            ).report()
        for name in TRACKED_SENSOR_FIELDS:
            # IsaacGym places independent rows on a positive-spacing world grid,
            # while MuJoCo/MJWarp keep both logical rows at the model origin.  Pose
            # parity is therefore a full transform into the pelvis frame.  Merely
            # subtracting world positions cancels a translation but would let a
            # world-frame rotation drift masquerade as parity.
            expected = reference.sensors[name]
            actual = candidate.sensors[name]
            if name.startswith("track_pos_w_"):
                body_name = name.removeprefix("track_pos_w_")
                expected, _ = root_relative_body_pose(
                    expected,
                    reference.sensors[f"track_quat_w_{body_name}"],
                    reference.sensors["track_pos_w_pelvis"],
                    reference.sensors["track_quat_w_pelvis"],
                )
                actual, _ = root_relative_body_pose(
                    actual,
                    candidate.sensors[f"track_quat_w_{body_name}"],
                    candidate.sensors["track_pos_w_pelvis"],
                    candidate.sensors["track_quat_w_pelvis"],
                )
            if name.startswith("track_quat_w_"):
                body_name = name.removeprefix("track_quat_w_")
                _, expected_relative = root_relative_body_pose(
                    reference.sensors[f"track_pos_w_{body_name}"],
                    expected,
                    reference.sensors["track_pos_w_pelvis"],
                    reference.sensors["track_quat_w_pelvis"],
                )
                _, actual_relative = root_relative_body_pose(
                    candidate.sensors[f"track_pos_w_{body_name}"],
                    actual,
                    candidate.sensors["track_pos_w_pelvis"],
                    candidate.sensors["track_quat_w_pelvis"],
                )
                body_metrics[name] = quaternion_metric(
                    actual_relative, expected_relative
                ).report()
            else:
                body_metrics[name] = array_metric(actual, expected).report()
    else:
        metrics["sensor_comparison"] = "fail_closed_until_first_step"
    metrics["body_sensors"] = body_metrics
    if include_step:
        metrics["summary"] = _parity_summary(metrics)
    return metrics


def _parity_summary(metrics: dict[str, Any]) -> dict[str, float]:
    scalar_values = [metrics["qpos"]["max_abs"], metrics["qvel"]["max_abs"]]
    scalar_values.extend(metrics[name]["max_abs"] for name in SCALAR_SENSOR_FIELDS)
    for name, value in metrics["body_sensors"].items():
        scalar_values.append(
            value["max_angle_rad"] if name.startswith("track_quat_w_") else value["max_abs"]
        )
    return {"max_all": float(max(scalar_values))}


def assert_reset_parity(metrics: dict[str, Any], atol: float) -> None:
    # Isaac's rigid-body state tensor remains at the pre-reset pose until the
    # first SDK step (documented by acquire/refresh_rigid_body_state_tensor).
    # State tensors are the authoritative selected-reset boundary; derived
    # sensors are still retained diagnostically for the first step comparison.
    worst = max(metrics["qpos"]["max_abs"], metrics["qvel"]["max_abs"])
    if worst > atol:
        raise AssertionError(f"selected-reset parity exceeded {atol}: max error {worst}")


def required_acceptance_environment() -> dict[str, str]:
    variables = {
        "UNISIM_TEST_ISAAC_G1_STEP_QPOS_ATOL": None,
        "UNISIM_TEST_ISAAC_G1_STEP_QVEL_ATOL": None,
        "UNISIM_TEST_ISAAC_G1_STEP_BODY_POS_ATOL": None,
        "UNISIM_TEST_ISAAC_G1_STEP_QUAT_ATOL_RAD": None,
        "UNISIM_TEST_ISAAC_G1_STEP_BODY_VEL_ATOL": None,
        "UNISIM_TEST_ISAAC_G1_STEP_SCALAR_SENSOR_ATOL": None,
    }
    missing: list[str] = []
    values: dict[str, str] = {}
    for name in variables:
        value = os.environ.get(name)
        if value is None or not value.strip():
            missing.append(name)
        else:
            values[name] = value
    if missing:
        raise RuntimeError(
            "acceptance mode requires explicit step tolerances; missing " + ", ".join(missing)
        )
    active_profilers = [name for name in PROFILER_ENVIRONMENT_VARIABLES if os.environ.get(name)]
    if active_profilers:
        raise RuntimeError(
            "acceptance parity must not enable Isaac worker profilers: "
            + ", ".join(active_profilers)
        )
    return values


def thresholds_from_environment() -> ParityThresholds:
    values = required_acceptance_environment()
    try:
        thresholds = ParityThresholds(
            qpos=float(values["UNISIM_TEST_ISAAC_G1_STEP_QPOS_ATOL"]),
            qvel=float(values["UNISIM_TEST_ISAAC_G1_STEP_QVEL_ATOL"]),
            body_pos=float(values["UNISIM_TEST_ISAAC_G1_STEP_BODY_POS_ATOL"]),
            body_quat_rad=float(values["UNISIM_TEST_ISAAC_G1_STEP_QUAT_ATOL_RAD"]),
            body_velocity=float(values["UNISIM_TEST_ISAAC_G1_STEP_BODY_VEL_ATOL"]),
            scalar_sensor=float(values["UNISIM_TEST_ISAAC_G1_STEP_SCALAR_SENSOR_ATOL"]),
        )
    except ValueError as exc:
        raise RuntimeError("acceptance step tolerances must be finite numbers") from exc
    if not all(math.isfinite(value) and value >= 0.0 for value in asdict(thresholds).values()):
        raise RuntimeError("acceptance step tolerances must be finite and nonnegative")
    return thresholds


def assert_step_parity(metrics: dict[str, Any], thresholds: ParityThresholds) -> None:
    if metrics["qpos"]["max_abs"] > thresholds.qpos:
        raise AssertionError(f"control-step qpos error {metrics['qpos']['max_abs']}")
    if metrics["qvel"]["max_abs"] > thresholds.qvel:
        raise AssertionError(f"control-step qvel error {metrics['qvel']['max_abs']}")
    for name in SCALAR_SENSOR_FIELDS:
        if metrics[name]["max_abs"] > thresholds.scalar_sensor:
            raise AssertionError(f"control-step {name} error {metrics[name]['max_abs']}")
    for name, value in metrics["body_sensors"].items():
        if name.startswith("track_pos_w_") and value["max_abs"] > thresholds.body_pos:
            raise AssertionError(f"control-step {name} error {value['max_abs']}")
        if name.startswith("track_quat_w_") and value["max_angle_rad"] > thresholds.body_quat_rad:
            raise AssertionError(f"control-step {name} error {value['max_angle_rad']}")
        if (
            name.startswith(("track_linvel_w_", "track_angvel_w_"))
            and value["max_abs"] > thresholds.body_velocity
        ):
            raise AssertionError(f"control-step {name} error {value['max_abs']}")


def compare_control_step_trajectories(
    reference: Sequence[G1ControlStep], candidate: Sequence[G1ControlStep]
) -> dict[str, Any]:
    """Compare every control step instead of reducing a trajectory to its endpoint."""

    if len(reference) != len(candidate):
        raise ValueError(
            f"control-step trajectory length mismatch: {len(reference)} != {len(candidate)}"
        )
    steps: list[dict[str, Any]] = []
    for reference_step, candidate_step in zip(reference, candidate, strict=True):
        if reference_step.index != candidate_step.index:
            raise ValueError("control-step trajectory indices are misaligned")
        if not np.array_equal(reference_step.control, candidate_step.control):
            raise ValueError(f"control step {reference_step.index} inputs differ")
        steps.append(
            {
                "index": reference_step.index,
                "comparison": compare_snapshots(
                    reference_step.snapshot, candidate_step.snapshot, include_step=True
                ),
            }
        )
    scalar_names = ("qpos", "qvel", *SCALAR_SENSOR_FIELDS)
    maxima = {
        name: max(step["comparison"][name]["max_abs"] for step in steps) for name in scalar_names
    }
    for prefix in ("track_pos_w_", "track_linvel_w_", "track_angvel_w_"):
        maxima[prefix] = max(
            step["comparison"]["body_sensors"][name]["max_abs"]
            for step in steps
            for name in step["comparison"]["body_sensors"]
            if name.startswith(prefix)
        )
    maxima["track_quat_w_"] = max(
        step["comparison"]["body_sensors"][name]["max_angle_rad"]
        for step in steps
        for name in step["comparison"]["body_sensors"]
        if name.startswith("track_quat_w_")
    )
    return {"step_count": len(steps), "steps": steps, "maxima": maxima}


def assert_control_step_trajectory_parity(
    metrics: dict[str, Any], thresholds: ParityThresholds
) -> None:
    for step in metrics["steps"]:
        try:
            assert_step_parity(step["comparison"], thresholds)
        except AssertionError as exc:
            raise AssertionError(f"control step {step['index']}: {exc}") from exc


def sensor_stale_window(before: G1Snapshot, after: G1Snapshot) -> SensorStaleWindow:
    """Measure Isaac's documented reset-to-first-step rigid-body sensor delay."""

    return SensorStaleWindow(
        qpos_change=array_metric(after.qpos, before.qpos).max_abs,
        qvel_change=array_metric(after.qvel, before.qvel).max_abs,
        max_sensor_abs=max(
            array_metric(after.sensors[name], before.sensors[name]).max_abs
            for name in (*SCALAR_SENSOR_FIELDS, *TRACKED_SENSOR_FIELDS)
        ),
    )


def assert_reset_sensor_stale_window(
    metrics: SensorStaleWindow, *, min_state_change: float, sensor_atol: float
) -> None:
    if metrics.qpos_change < min_state_change or metrics.qvel_change < min_state_change:
        raise AssertionError("reset state did not change while waiting for the first SDK refresh")
    if metrics.max_sensor_abs > sensor_atol:
        raise AssertionError(
            f"reset-derived sensor changed before the first SDK step: {metrics.max_sensor_abs}"
        )


def sensor_refresh_magnitude(before: G1Snapshot, after: G1Snapshot) -> float:
    return max(
        array_metric(after.sensors[name], before.sensors[name]).max_abs
        for name in (*SCALAR_SENSOR_FIELDS, *TRACKED_SENSOR_FIELDS)
    )


def gpu_compute_process_snapshot(device_index: int = 0) -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot audit GPU contention with {command[0]}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"GPU contention audit failed ({result.returncode}): {result.stderr.strip()}"
        )
    processes: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        pid, process_name, used_memory = (part.strip() for part in line.split(",", 2))
        processes.append({"pid": pid, "process_name": process_name, "used_memory_mib": used_memory})
    return processes


def gpu_device_snapshot(device_index: int = 0) -> dict[str, str]:
    command = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=index,name,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot identify GPU with {command[0]}") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"GPU identity audit failed: {result.stderr.strip()}")
    index, name, uuid, driver_version = (
        part.strip() for part in result.stdout.strip().split(",", 3)
    )
    return {
        "index": index,
        "name": name,
        "uuid": uuid,
        "driver_version": driver_version,
    }


def require_acceptance_gpu_idle(
    device_index: int = 0,
    *,
    quiesce_timeout_s: float = 0.0,
    allowed_pids: frozenset[int] | set[int] = frozenset(),
) -> list[dict[str, str]]:
    processes = gpu_compute_process_snapshot(device_index)
    external_processes = [row for row in processes if int(row.get("pid", "-1")) not in allowed_pids]
    deadline = time.monotonic() + quiesce_timeout_s
    while external_processes and time.monotonic() < deadline:
        time.sleep(0.25)
        processes = gpu_compute_process_snapshot(device_index)
        external_processes = [
            row for row in processes if int(row.get("pid", "-1")) not in allowed_pids
        ]
    if external_processes and os.environ.get("UNISIM_TEST_ISAAC_ALLOW_CONTENDED") != "1":
        raise RuntimeError(
            "acceptance parity requires an idle GPU; "
            f"external compute processes: {external_processes}"
        )
    return processes


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def snapshot_to_numpy(value: Any) -> np.ndarray:
    """Detach a Torch tensor, or copy an SDK-free NumPy view."""

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def selected_reset_state(
    qpos: np.ndarray, qvel: np.ndarray, row_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic, nonzero selected-reset state from stand values."""

    reset_qpos = np.asarray(qpos, dtype=np.float32).copy()
    reset_qvel = np.asarray(qvel, dtype=np.float32).copy()
    reset_qpos[1] += 0.015
    reset_qpos[7:] += np.linspace(0.015, -0.025, num=reset_qpos[7:].size, dtype=np.float32)
    reset_qvel[:6] = np.asarray([0.025, -0.015, 0.01, 0.002, -0.003, 0.004], dtype=np.float32)
    reset_qvel[6:] = np.linspace(0.05, -0.08, num=reset_qvel[6:].size, dtype=np.float32)
    return (
        np.repeat(reset_qpos[None, :], row_count, axis=0),
        np.repeat(reset_qvel[None, :], row_count, axis=0),
    )


def deterministic_control_trajectory(
    base_control: np.ndarray, *, num_envs: int, steps: int
) -> np.ndarray:
    """Create a deterministic, varying multi-step control sequence.

    The factors are deliberately small and symmetric around stand control so a
    real parity run remains near the configured G1 keyframe while still probing
    that each control step advances independently.
    """

    if steps < 2:
        raise ValueError("a control trajectory must contain at least two steps")
    factors = np.linspace(0.99, 1.01, num=steps, dtype=np.float32)
    controls = np.asarray(base_control, dtype=np.float32)[None, None, :] * factors[:, None, None]
    return np.repeat(controls, num_envs, axis=1)


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    return np.concatenate((values[..., :1], -values[..., 1:]), axis=-1)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    left_real = left_array[..., :1]
    left_vec = left_array[..., 1:]
    right_real = right_array[..., :1]
    right_vec = right_array[..., 1:]
    real = left_real * right_real - np.sum(left_vec * right_vec, axis=-1, keepdims=True)
    vec = left_real * right_vec + right_real * left_vec + np.cross(left_vec, right_vec)
    return np.concatenate((real, vec), axis=-1)


def _quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat_array = np.asarray(quaternion, dtype=np.float64)
    vec_array = np.asarray(vector, dtype=np.float64)
    quat_vec = quat_array[..., 1:]
    return (
        vec_array
        - 2.0 * quat_array[..., :1] * np.cross(quat_vec, vec_array)
        + 2.0 * np.cross(quat_vec, np.cross(quat_vec, vec_array))
    )


def root_relative_body_pose(
    body_position: np.ndarray,
    body_quaternion: np.ndarray,
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a tracked body pose in the root-body frame."""

    position = _quat_rotate_inverse(
        root_quaternion, np.asarray(body_position) - np.asarray(root_position)
    )
    quaternion = _quat_multiply(_quat_conjugate(root_quaternion), np.asarray(body_quaternion))
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return position, quaternion


def pytest_skip_if_fixture_unavailable(paths: G1FixturePaths) -> None:
    try:
        missing = [
            name
            for name, path in (
                ("canonical_scene", paths.canonical_scene),
                ("isaacsim_robot", paths.isaacsim_robot),
                ("isaacsim_floor", paths.isaacsim_floor),
                ("isaacsim_contact_sensors", paths.isaacsim_contact_sensors),
            )
            if not path.is_file()
        ]
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    if missing:
        pytest.skip(f"G1 parity fixtures are unavailable: {', '.join(missing)}")
