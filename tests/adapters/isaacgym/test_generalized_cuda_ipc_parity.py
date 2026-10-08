"""Opt-in generalized multi-entity CUDA IPC parity evidence.

The regular SDK-free lifecycle test proves that the external IsaacGym worker can
execute this synthetic scene.  This module adds the missing numerical evidence:
it rolls the same scene forward on the public MuJoCo and MJWarp backends and
records complete state/body-sensor trajectories.  Diagnostic mode never turns
first-run differences into a threshold decision; acceptance mode requires a
predeclared, all-step threshold policy and an explicit artifact path.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.adapters.isaac.g1_parity_harness import (
    PROFILER_ENVIRONMENT_VARIABLES,
    array_metric,
    gpu_compute_process_snapshot,
    gpu_device_snapshot,
    quaternion_metric,
    require_acceptance_gpu_idle,
    write_json_report,
)
from tests.adapters.isaacgym.scene_client import SceneClient
from tests.adapters.isaacgym.scene_fixture import scene_payload
from unisim.backend.isaacgym.tensor import (
    IsaacGymCudaIpcArenaLayout,
    torch_from_cuda_pointer,
)
from unisim.backend.subprocess_ipc import cuda_ipc
from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import (
    EntityInitialState,
    EntityVariantBinding,
    SceneEntitySpec,
)
from unisim.scene import SceneCfg
from unisim.scene_layout import CompiledSceneLayout

SIM_DT = 0.001
CONTROL_SUBSTEPS = 3
CONTROL_STEP_COUNT = 4
SELECTED_ROW = 1
RESET_ECHO_ATOL = 2e-5
TRACKED_BODIES = (
    "robot/base",
    "robot/finger",
    "object/base",
    "object/lid",
    "table/base",
    "target/base",
)
TRACKED_SENSOR_FIELDS = (
    "track_pos_w_",
    "track_quat_w_",
    "track_linvel_w_",
    "track_angvel_w_",
)
PARITY_MODE_ENV = "UNISIM_TEST_ISAACGYM_GENERALIZED_PARITY_MODE"
THRESHOLD_ENV = {
    "qpos": "UNISIM_TEST_ISAACGYM_GENERALIZED_QPOS_ATOL",
    "qvel": "UNISIM_TEST_ISAACGYM_GENERALIZED_QVEL_ATOL",
    "body_pos": "UNISIM_TEST_ISAACGYM_GENERALIZED_BODY_POS_ATOL",
    "body_quat_rad": "UNISIM_TEST_ISAACGYM_GENERALIZED_BODY_QUAT_ATOL_RAD",
    "body_velocity": "UNISIM_TEST_ISAACGYM_GENERALIZED_BODY_VEL_ATOL",
}
ACCEPTANCE_COMPARISONS = ("isaacgym_vs_mujoco", "isaacgym_vs_mjwarp")
ALL_COMPARISONS = (*ACCEPTANCE_COMPARISONS, "mjwarp_vs_mujoco")


@dataclass(frozen=True)
class GeneralizedThresholds:
    qpos: float
    qvel: float
    body_pos: float
    body_quat_rad: float
    body_velocity: float

    def report(self) -> dict[str, float]:
        return {
            "qpos": self.qpos,
            "qvel": self.qvel,
            "body_pos": self.body_pos,
            "body_quat_rad": self.body_quat_rad,
            "body_velocity": self.body_velocity,
        }


@dataclass(frozen=True)
class ReferenceSources:
    scene: SceneCfg
    robot: ModelSourceDescriptor
    object_a: ModelSourceDescriptor
    object_b: ModelSourceDescriptor
    table: ModelSourceDescriptor
    target: ModelSourceDescriptor


@dataclass(frozen=True)
class RolloutSnapshot:
    source: str
    step_index: int
    control: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    sensors: dict[str, np.ndarray]

    def report(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "step_index": self.step_index,
            "control": self.control.tolist(),
            "qpos": self.qpos.tolist(),
            "qvel": self.qvel.tolist(),
            "sensors": {name: values.tolist() for name, values in self.sensors.items()},
        }


def _write_source(directory: Path, name: str, text: str) -> ModelSourceDescriptor:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return ModelSourceDescriptor(str(path.resolve()))


def _reference_sources(directory: Path) -> ReferenceSources:
    """Build public-backend sources for the synthetic worker fixture.

    Two source-only differences are intentional and recorded in the artifact:
    MJWarp requires unique geom names, and public MuJoCo references keep the
    position actuator which the IsaacGym worker importer removes and represents
    through its explicit drive table.  The target is represented as a physical
    kinematic rigid body: the raw worker fixture labels it as a mirror but gives
    it a different one-body source catalog, which the public mirror contract
    intentionally cannot express.  It is collision-disabled and joint-free, so
    neither difference participates in the compared dynamics.
    """

    robot = _write_source(
        directory,
        "robot.xml",
        """
        <mujoco>
          <compiler angle="radian"/>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="base">
              <inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
              <geom name="base_geom" type="sphere" size=".1"/>
              <body name="finger" pos="0 0 .3">
                <joint name="drive_joint" type="hinge" axis="0 1 0" range="-1 1"/>
                <inertial pos="0 0 0" mass=".2" diaginertia=".001 .001 .001"/>
                <geom name="finger_geom" type="sphere" size=".05"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position name="drive" joint="drive_joint" kp="20" kv="1"/>
          </actuator>
        </mujoco>
        """,
    )
    object_template = """
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <body name="base">
          <freejoint/>
          <inertial pos=".05 0 0" mass="{mass}" diaginertia=".01 .02 .03"/>
          <geom name="base_geom" type="box" size=".1 .1 .1"/>
          <body name="lid" pos="0 0 .2">
            <joint name="passive" type="hinge" axis="0 1 0"/>
            <inertial pos=".05 0 0" mass=".2" diaginertia=".001 .002 .003"/>
            <geom name="lid_geom" type="box" size=".08 .08 .02"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    object_a = _write_source(directory, "object0.xml", object_template.format(mass=1.0))
    object_b = _write_source(directory, "object1.xml", object_template.format(mass=3.0))
    rigid_template = """
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <body name="base">
          <inertial pos="0 0 0" mass="{mass}" diaginertia=".01 .01 .01"/>
          <geom name="{geom}" type="box" size="{size}"/>
        </body>
      </worldbody>
    </mujoco>
    """
    table = _write_source(
        directory,
        "table.xml",
        rigid_template.format(mass=10.0, geom="table_geom", size="2 2 .1"),
    )
    target = _write_source(
        directory,
        "target.xml",
        rigid_template.format(mass=1.0, geom="target_geom", size=".1 .1 .1"),
    )
    yaw = np.sqrt(0.5)
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                robot,
                root_mode="fixed",
                initial_state=EntityInitialState((-1.0, 0.0, 0.4)),
            ),
            SceneEntitySpec(
                "object",
                object_a,
                initial_state=EntityInitialState((0.0, 0.0, 1.0), (yaw, 0.0, 0.0, yaw)),
            ),
            SceneEntitySpec(
                "table",
                table,
                kind="rigid",
                root_mode="fixed",
                initial_state=EntityInitialState((0.0, 0.0, 0.5)),
            ),
            SceneEntitySpec(
                "target",
                target,
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                initial_state=EntityInitialState((0.0, 0.0, 0.5)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(
                np.asarray([1, 1, 0, 1, 0], dtype=np.int32),
                (object_a, object_b),
                FixedVariantLayout.SAME_LAYOUT,
            ),
        ),
    )
    return ReferenceSources(scene, robot, object_a, object_b, table, target)


def _assert_layout_equivalent(actual: CompiledSceneLayout, expected: CompiledSceneLayout) -> None:
    scalar_fields = ("nq", "nv", "nu", "nbody")
    for field in scalar_fields:
        if getattr(actual, field) != getattr(expected, field):
            raise ValueError(f"generalized reference layout {field} differs")
    if tuple(entity.name for entity in actual.entities) != tuple(
        entity.name for entity in expected.entities
    ):
        raise ValueError("generalized reference entity order differs")
    for actual_entity, expected_entity in zip(actual.entities, expected.entities, strict=True):
        for field in ("kind", "root_mode", "body_names"):
            if getattr(actual_entity, field) != getattr(expected_entity, field):
                raise ValueError(
                    f"generalized reference entity {actual_entity.name} {field} differs"
                )
        actual_joints = (
            (
                joint.name,
                joint.kind,
                tuple(int(index) for index in joint.qpos_indices),
                tuple(int(index) for index in joint.qvel_indices),
                joint.body_name,
            )
            for joint in actual_entity.joints
        )
        expected_joints = (
            (
                joint.name,
                joint.kind,
                tuple(int(index) for index in joint.qpos_indices),
                tuple(int(index) for index in joint.qvel_indices),
                joint.body_name,
            )
            for joint in expected_entity.joints
        )
        if tuple(actual_joints) != tuple(expected_joints):
            raise ValueError(f"generalized reference entity {actual_entity.name} joints differ")
        if actual_entity.actuator_names != expected_entity.actuator_names:
            raise ValueError(f"generalized reference entity {actual_entity.name} actuators differ")


def _sensor_names() -> tuple[str, ...]:
    return tuple(
        f"{prefix}{body_name}" for prefix in TRACKED_SENSOR_FIELDS for body_name in TRACKED_BODIES
    )


def _reference_snapshot(
    backend: Any,
    source: str,
    step_index: int,
    control: np.ndarray,
) -> RolloutSnapshot:
    state = backend.get_state(("qpos", "qvel"))
    sensors = {
        name: np.asarray(backend.get_sensor_data(name), dtype=np.float32).copy()
        for name in _sensor_names()
    }
    return RolloutSnapshot(
        source=source,
        step_index=step_index,
        control=np.asarray(control, dtype=np.float32).copy(),
        qpos=np.asarray(state["qpos"], dtype=np.float32).copy(),
        qvel=np.asarray(state["qvel"], dtype=np.float32).copy(),
        sensors=sensors,
    )


def _candidate_snapshot(
    source: str,
    step_index: int,
    control: np.ndarray,
    qpos: Any,
    qvel: Any,
    body_state: Any,
) -> RolloutSnapshot:
    sensors: dict[str, np.ndarray] = {}
    for name in TRACKED_BODIES:
        body_id = 1 + TRACKED_BODIES.index(name)
        sensors[f"track_pos_w_{name}"] = body_state[:, body_id, 0:3].detach().cpu().numpy().copy()
        sensors[f"track_quat_w_{name}"] = body_state[:, body_id, 3:7].detach().cpu().numpy().copy()
        sensors[f"track_linvel_w_{name}"] = (
            body_state[:, body_id, 7:10].detach().cpu().numpy().copy()
        )
        sensors[f"track_angvel_w_{name}"] = (
            body_state[:, body_id, 10:13].detach().cpu().numpy().copy()
        )
    return RolloutSnapshot(
        source=source,
        step_index=step_index,
        control=np.asarray(control, dtype=np.float32).copy(),
        qpos=qpos.detach().cpu().numpy().copy(),
        qvel=qvel.detach().cpu().numpy().copy(),
        sensors=sensors,
    )


def _assert_snapshot_contract(snapshot: RolloutSnapshot, num_envs: int) -> None:
    if snapshot.qpos.shape != (num_envs, 9):
        raise ValueError(f"{snapshot.source} qpos shape is {snapshot.qpos.shape}")
    if snapshot.qvel.shape != (num_envs, 8):
        raise ValueError(f"{snapshot.source} qvel shape is {snapshot.qvel.shape}")
    expected_sensor_names = set(_sensor_names())
    if set(snapshot.sensors) != expected_sensor_names:
        raise ValueError(f"{snapshot.source} sensor set differs")
    for name, values in snapshot.sensors.items():
        expected_last = 4 if name.startswith("track_quat_w_") else 3
        if values.shape != (num_envs, expected_last):
            raise ValueError(f"{snapshot.source} sensor {name} shape is {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{snapshot.source} sensor {name} is non-finite")
    if not np.isfinite(snapshot.qpos).all() or not np.isfinite(snapshot.qvel).all():
        raise ValueError(f"{snapshot.source} state is non-finite")


def _compare_snapshots(
    actual: RolloutSnapshot, expected: RolloutSnapshot, *, include_sensors: bool
) -> dict[str, Any]:
    if actual.step_index != expected.step_index:
        raise ValueError("control-step indices are misaligned")
    metrics: dict[str, Any] = {
        "qpos": array_metric(actual.qpos, expected.qpos).report(),
        "qvel": array_metric(actual.qvel, expected.qvel).report(),
    }
    if include_sensors:
        sensor_metrics: dict[str, Any] = {}
        for name in _sensor_names():
            actual_values = actual.sensors[name]
            expected_values = expected.sensors[name]
            if name.startswith("track_quat_w_"):
                sensor_metrics[name] = quaternion_metric(actual_values, expected_values).report()
            else:
                sensor_metrics[name] = array_metric(actual_values, expected_values).report()
        metrics["tracked_body_sensors"] = sensor_metrics
    return metrics


def _trajectory_metrics(
    actual: list[RolloutSnapshot], expected: list[RolloutSnapshot]
) -> list[dict[str, Any]]:
    if len(actual) != len(expected) or not actual:
        raise ValueError("complete trajectory length mismatch")
    return [
        _compare_snapshots(candidate, reference, include_sensors=True)
        for candidate, reference in zip(actual, expected, strict=True)
    ]


def _trajectory_maxima(steps: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {
        "qpos_max_abs": max(step["qpos"]["max_abs"] for step in steps),
        "qvel_max_abs": max(step["qvel"]["max_abs"] for step in steps),
        "body_position_max_abs": max(
            metric["max_abs"]
            for step in steps
            for name, metric in step["tracked_body_sensors"].items()
            if name.startswith("track_pos_w_")
        ),
        "body_velocity_max_abs": max(
            metric["max_abs"]
            for step in steps
            for name, metric in step["tracked_body_sensors"].items()
            if name.startswith(("track_linvel_w_", "track_angvel_w_"))
        ),
        "body_sensor_max_abs": max(
            metric["max_abs"]
            for step in steps
            for name, metric in step["tracked_body_sensors"].items()
            if not name.startswith("track_quat_w_")
        ),
        "body_quaternion_max_angle_rad": max(
            metric["max_angle_rad"]
            for step in steps
            for name, metric in step["tracked_body_sensors"].items()
            if name.startswith("track_quat_w_")
        ),
    }
    return result


def _parity_mode() -> str:
    value = os.environ.get(PARITY_MODE_ENV, "diagnostic")
    if value not in {"diagnostic", "acceptance"}:
        raise RuntimeError(f"{PARITY_MODE_ENV} must be diagnostic or acceptance")
    return value


def _acceptance_thresholds() -> GeneralizedThresholds:
    missing = [name for name in THRESHOLD_ENV.values() if os.environ.get(name) is None]
    if missing:
        raise RuntimeError("missing generalized acceptance thresholds: " + ", ".join(missing))
    values: dict[str, float] = {}
    for field, name in THRESHOLD_ENV.items():
        raw = os.environ[name]
        try:
            value = float(raw)
        except ValueError as error:
            raise RuntimeError(f"{name} must be a finite non-negative number") from error
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"{name} must be a finite non-negative number")
        values[field] = value
    return GeneralizedThresholds(**values)


def _step_threshold_groups(step: dict[str, Any]) -> dict[str, float]:
    sensors = step["tracked_body_sensors"]
    return {
        "qpos": float(step["qpos"]["max_abs"]),
        "qvel": float(step["qvel"]["max_abs"]),
        "body_pos": max(
            float(metric["max_abs"])
            for name, metric in sensors.items()
            if name.startswith("track_pos_w_")
        ),
        "body_quat_rad": max(
            float(metric["max_angle_rad"])
            for name, metric in sensors.items()
            if name.startswith("track_quat_w_")
        ),
        "body_velocity": max(
            float(metric["max_abs"])
            for name, metric in sensors.items()
            if name.startswith(("track_linvel_w_", "track_angvel_w_"))
        ),
    }


def _all_step_threshold_evidence(
    comparisons: dict[str, list[dict[str, Any]]], limits: GeneralizedThresholds
) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {}
    for comparison in ACCEPTANCE_COMPARISONS:
        rows: list[dict[str, Any]] = []
        for step_index, step in enumerate(comparisons[comparison]):
            groups = _step_threshold_groups(step)
            if any(not math.isfinite(value) or value < 0 for value in groups.values()):
                raise ValueError(f"{comparison} step {step_index} has a non-finite metric")
            rows.append(
                {
                    "step_index": step_index,
                    **groups,
                    "within_threshold": all(
                        groups[field] <= getattr(limits, field) for field in THRESHOLD_ENV
                    ),
                }
            )
        evidence[comparison] = rows
    return evidence


def _expect_exact_keys(value: Any, expected: tuple[str, ...], context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        missing = sorted(wanted - actual)
        unexpected = sorted(actual - wanted)
        raise ValueError(f"{context} keys differ; missing={missing}, unexpected={unexpected}")


def _validate_report(report: Any) -> None:
    """Strictly validate a reloaded schema-2 parity artifact."""

    top_level = (
        "schema_version",
        "mode",
        "run_id",
        "scope",
        "backend",
        "candidate_execution",
        "references",
        "num_envs",
        "selected_row",
        "sim_dt",
        "control_substeps",
        "control_step_count",
        "tracked_bodies",
        "tracked_sensor_fields",
        "reset",
        "control_steps",
        "source_only_reference_normalizations",
        "source_provenance",
        "worker_metadata",
        "process_provenance",
        "host_runtime_versions",
        "gpu",
        "gpu_device",
        "profiler_environment",
        "threshold_policy",
        "assertions",
    )
    _expect_exact_keys(report, top_level, "report")
    if report["schema_version"] != 2:
        raise ValueError("report schema_version must be 2")
    mode = report["mode"]
    if mode not in {"diagnostic", "acceptance"}:
        raise ValueError("report mode must be diagnostic or acceptance")
    if not isinstance(report["run_id"], str) or not report["run_id"]:
        raise ValueError("report run_id is missing")
    if report["backend"] != "isaacgym":
        raise ValueError("report backend must be isaacgym")
    if list(report["references"]) != ["mujoco", "mjwarp"]:
        raise ValueError("report references must be mujoco and mjwarp")
    if report["control_step_count"] != CONTROL_STEP_COUNT:
        raise ValueError("report control trajectory is incomplete")
    if tuple(report["tracked_bodies"]) != TRACKED_BODIES:
        raise ValueError("report tracked bodies differ")
    if tuple(report["tracked_sensor_fields"]) != TRACKED_SENSOR_FIELDS:
        raise ValueError("report tracked sensor fields differ")

    _expect_exact_keys(
        report["scope"],
        (
            "task",
            "mode",
            "same_control_and_reset_inputs",
            "cross_engine_numerical_thresholds",
            "complete_trajectories",
        ),
        "scope",
    )
    scope = report["scope"]
    if scope["mode"] != mode:
        raise ValueError("top-level and scope modes differ")
    if scope["cross_engine_numerical_thresholds"] != (mode == "acceptance"):
        raise ValueError("scope threshold claim differs from mode")
    if not scope["same_control_and_reset_inputs"] or not scope["complete_trajectories"]:
        raise ValueError("scope completeness claims are false")

    _expect_exact_keys(
        report["candidate_execution"],
        (
            "process_topology",
            "data_plane",
            "sdk_imported_in_parent",
            "authored_scalar_sensors",
            "tracked_body_sensor_projection",
        ),
        "candidate_execution",
    )
    candidate = report["candidate_execution"]
    if candidate["process_topology"] != "external_python38_worker":
        raise ValueError("candidate process topology differs")
    if candidate["data_plane"] != "cuda_ipc" or candidate["sdk_imported_in_parent"]:
        raise ValueError("candidate data plane is not parent-free CUDA IPC")
    if candidate["authored_scalar_sensors"] != []:
        raise ValueError("this generalized fixture must not claim authored scalar sensors")

    _expect_exact_keys(
        report["reset"],
        (
            "selected_qpos",
            "selected_qvel",
            "snapshots",
            "state_comparisons",
            "asserted",
            "reset_echo_atol",
            "sensor_comparison",
            "sensor_stale_window_reason",
        ),
        "reset",
    )
    reset = report["reset"]
    _expect_exact_keys(
        reset["snapshots"], ("mujoco", "mjwarp", "isaacgym_cuda_ipc"), "reset snapshots"
    )
    _expect_exact_keys(
        reset["state_comparisons"],
        ACCEPTANCE_COMPARISONS,
        "reset state comparisons",
    )
    if list(reset["asserted"]) != ["selected_state_echo", "unselected_row_invariance"]:
        raise ValueError("reset assertions differ")
    if reset["sensor_comparison"] != "deferred_until_first_step":
        raise ValueError("reset sensor freshness boundary differs")

    _expect_exact_keys(
        report["control_steps"],
        (
            "controls",
            "snapshots",
            "comparisons",
            "maxima",
            "asserted",
            "thresholds",
            "threshold_assertions",
        ),
        "control_steps",
    )
    control = report["control_steps"]
    _expect_exact_keys(
        control["snapshots"], ("mujoco", "mjwarp", "isaacgym_cuda_ipc"), "control snapshots"
    )
    for source, snapshots in control["snapshots"].items():
        if not isinstance(snapshots, list) or len(snapshots) != CONTROL_STEP_COUNT:
            raise ValueError(f"{source} control trajectory is incomplete")
        if [row["step_index"] for row in snapshots] != list(range(CONTROL_STEP_COUNT)):
            raise ValueError(f"{source} control indices are not contiguous")
        if any(row["source"] != source for row in snapshots):
            raise ValueError(f"{source} control source labels differ")
    _expect_exact_keys(control["comparisons"], ALL_COMPARISONS, "control comparisons")
    _expect_exact_keys(control["maxima"], ALL_COMPARISONS, "control maxima")
    expected_sensor_names = set(_sensor_names())
    for comparison, steps in control["comparisons"].items():
        if not isinstance(steps, list) or len(steps) != CONTROL_STEP_COUNT:
            raise ValueError(f"{comparison} metric trajectory is incomplete")
        for step_index, step in enumerate(steps):
            _expect_exact_keys(
                step,
                ("qpos", "qvel", "tracked_body_sensors"),
                f"{comparison}[{step_index}]",
            )
            _expect_exact_keys(step["qpos"], ("max_abs", "rms"), f"{comparison} qpos metric")
            _expect_exact_keys(step["qvel"], ("max_abs", "rms"), f"{comparison} qvel metric")
            if set(step["tracked_body_sensors"]) != expected_sensor_names:
                raise ValueError(f"{comparison} tracked sensor set differs")
            for name, metric in step["tracked_body_sensors"].items():
                if name.startswith("track_quat_w_"):
                    _expect_exact_keys(metric, ("max_dot_abs", "max_angle_rad"), f"{name} metric")
                else:
                    _expect_exact_keys(metric, ("max_abs", "rms"), f"{name} metric")
        expected_maxima = {
            "qpos_max_abs",
            "qvel_max_abs",
            "body_position_max_abs",
            "body_velocity_max_abs",
            "body_sensor_max_abs",
            "body_quaternion_max_angle_rad",
        }
        _expect_exact_keys(
            control["maxima"][comparison], tuple(expected_maxima), f"{comparison} maxima"
        )

    if mode == "diagnostic":
        if control["thresholds"] is not None or control["threshold_assertions"] is not None:
            raise ValueError("diagnostic artifacts must not contain threshold assertions")
        if "all_step_thresholds" in control["asserted"]:
            raise ValueError("diagnostic artifacts must not claim threshold assertions")
    else:
        _expect_exact_keys(control["thresholds"], tuple(THRESHOLD_ENV), "thresholds")
        for field, value in control["thresholds"].items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"threshold {field} must be finite and non-negative")
        threshold_assertions = control["threshold_assertions"]
        _expect_exact_keys(
            threshold_assertions, ACCEPTANCE_COMPARISONS, "threshold assertions"
        )
        for comparison, rows in threshold_assertions.items():
            if not isinstance(rows, list) or len(rows) != CONTROL_STEP_COUNT:
                raise ValueError(f"{comparison} threshold trajectory is incomplete")
            for step_index, row in enumerate(rows):
                _expect_exact_keys(
                    row,
                    ("step_index", *THRESHOLD_ENV, "within_threshold"),
                    f"{comparison} threshold row {step_index}",
                )
                if row["step_index"] != step_index or row["within_threshold"] is not True:
                    raise ValueError(f"{comparison} threshold row {step_index} failed")
                actual_groups = _step_threshold_groups(
                    control["comparisons"][comparison][step_index]
                )
                for field, value in actual_groups.items():
                    if row[field] != value:
                        raise ValueError(
                            f"{comparison} threshold row {step_index} {field} differs"
                        )
                    if value > control["thresholds"][field]:
                        raise ValueError(
                            f"{comparison} step {step_index} {field} exceeded threshold"
                        )

    _expect_exact_keys(
        report["host_runtime_versions"],
        ("python", "torch", "mujoco", "warp", "mujoco_warp"),
        "host_runtime_versions",
    )
    if not all(report["host_runtime_versions"].values()):
        raise ValueError("host runtime provenance is incomplete")
    _expect_exact_keys(
        report["process_provenance"],
        ("parent_pid", "worker_pid", "worker_returncode"),
        "process_provenance",
    )
    if any(
        not isinstance(report["process_provenance"][name], int)
        for name in report["process_provenance"]
    ):
        raise ValueError("process provenance is incomplete")

    _expect_exact_keys(
        report["gpu"],
        (
            "required_idle",
            "compute_processes_before",
            "compute_processes_after",
            "own_process_pid",
        ),
        "gpu",
    )
    gpu = report["gpu"]
    if gpu["required_idle"] != (mode == "acceptance"):
        raise ValueError("GPU idle requirement differs from mode")
    if mode == "acceptance":
        for phase in ("compute_processes_before", "compute_processes_after"):
            external = [
                row for row in gpu[phase] if int(row.get("pid", "-1")) != gpu["own_process_pid"]
            ]
            if external:
                raise ValueError(f"acceptance {phase} found external GPU compute processes")
    _expect_exact_keys(
        report["gpu_device"], ("index", "name", "uuid", "driver_version"), "gpu_device"
    )
    if not all(report["gpu_device"].values()):
        raise ValueError("GPU identity provenance is incomplete")
    _expect_exact_keys(
        report["profiler_environment"], PROFILER_ENVIRONMENT_VARIABLES, "profiler_environment"
    )
    if any(report["profiler_environment"].values()):
        raise ValueError("parity artifacts must not enable worker profilers")
    _expect_exact_keys(
        report["threshold_policy"],
        ("mode", "variables", "frozen_before_rollout", "asserted_every_step"),
        "threshold_policy",
    )
    policy = report["threshold_policy"]
    if policy["mode"] != mode or policy["variables"] != THRESHOLD_ENV:
        raise ValueError("threshold policy provenance differs")
    if policy["frozen_before_rollout"] is not True:
        raise ValueError("threshold policy was not frozen before rollout")
    if policy["asserted_every_step"] != (mode == "acceptance"):
        raise ValueError("threshold policy assertion scope differs")

    if not report["source_provenance"] or any(
        set(entry) != {"path", "sha256"}
        or not entry["path"]
        or len(entry["sha256"]) != 64
        for entry in report["source_provenance"].values()
    ):
        raise ValueError("source provenance is incomplete")
    _expect_exact_keys(
        report["assertions"],
        (
            "worker_layout_exact",
            "initial_state_echo",
            "selected_state_echo",
            "unselected_row_invariance",
            "all_trajectories_finite",
            "complete_trajectory_count",
            "cross_engine_parity_threshold",
            "reason",
        ),
        "assertions",
    )
    assertions = report["assertions"]
    if assertions["complete_trajectory_count"] != CONTROL_STEP_COUNT:
        raise ValueError("asserted trajectory count differs")
    if assertions["cross_engine_parity_threshold"] != (mode == "acceptance"):
        raise ValueError("asserted threshold claim differs from mode")


def _assert_reset_echo(
    snapshot: RolloutSnapshot,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    row: int | None = None,
) -> None:
    actual_qpos = snapshot.qpos if row is None else snapshot.qpos[[row]]
    actual_qvel = snapshot.qvel if row is None else snapshot.qvel[[row]]
    if array_metric(actual_qpos, qpos).max_abs > RESET_ECHO_ATOL:
        raise AssertionError(f"{snapshot.source} selected qpos reset was not echoed")
    if array_metric(actual_qvel, qvel).max_abs > RESET_ECHO_ATOL:
        raise AssertionError(f"{snapshot.source} selected qvel reset was not echoed")


def _assert_unselected_rows_unchanged(
    before: RolloutSnapshot, after: RolloutSnapshot, selected_row: int
) -> None:
    rows = tuple(index for index in range(before.qpos.shape[0]) if index != selected_row)
    unselected_rows = list(rows)
    if array_metric(after.qpos[unselected_rows], before.qpos[unselected_rows]).max_abs > (
        RESET_ECHO_ATOL
    ):
        raise ValueError(f"{after.source} unselected qpos rows changed")
    if array_metric(after.qvel[unselected_rows], before.qvel[unselected_rows]).max_abs > (
        RESET_ECHO_ATOL
    ):
        raise ValueError(f"{after.source} unselected qvel rows changed")
    for name, values in before.sensors.items():
        if (
            array_metric(after.sensors[name][unselected_rows], values[unselected_rows]).max_abs
            > RESET_ECHO_ATOL
        ):
            # Isaac body FK is intentionally allowed to remain stale until the
            # first SDK step.  Do not treat that documented publication window as
            # a state-write failure; it is compared from control step zero below.
            if after.source == "isaacgym_cuda_ipc" and name.startswith("track_"):
                continue
            raise ValueError(f"{after.source} unselected sensor {name} rows changed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cuda_ipc_wire(
    layout: IsaacGymCudaIpcArenaLayout,
    allocation: Any,
    control_event: Any,
    state_event: Any,
    reset_event: Any,
) -> dict[str, Any]:
    memory_handle = allocation.export_handle()
    control_handle = control_event.export_handle()
    state_handle = state_event.export_handle()
    reset_handle = reset_event.export_handle()

    def memory_wire(handle: Any) -> dict[str, Any]:
        return {
            "opaque_handle": handle.opaque_handle,
            "device_uuid": handle.device_uuid,
            "size_bytes": handle.size_bytes,
            "abi_version": handle.abi_version,
            "alignment_bytes": handle.alignment_bytes,
        }

    def event_wire(handle: Any) -> dict[str, Any]:
        return {
            "opaque_handle": handle.opaque_handle,
            "device_uuid": handle.device_uuid,
            "abi_version": handle.abi_version,
            "blocking_sync": handle.blocking_sync,
        }

    return {
        "arena": layout.wire(),
        "memory": memory_wire(memory_handle),
        "control_event": event_wire(control_handle),
        "state_event": event_wire(state_handle),
        "reset_event": event_wire(reset_handle),
        "sensors": [],
    }


def _selected_reset(initial_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos = initial_qpos[0].copy()
    qpos[0] = 0.21
    qpos[1:4] = (0.15, -0.1, 1.2)
    yaw = np.sin(np.pi / 8.0)
    qpos[4:8] = (np.cos(np.pi / 8.0), 0.0, 0.0, yaw)
    qpos[8] = 0.3
    qvel = np.zeros_like(qpos[:8])
    qvel[0] = -0.2
    qvel[1:4] = (0.25, -0.5, 0.75)
    return qpos[None, :], qvel[None, :]


def _control_trajectory(num_envs: int) -> np.ndarray:
    controls = (0.37, -0.2, 0.11, -0.42)
    return np.asarray(
        [[[control] for _ in range(num_envs)] for control in controls],
        dtype=np.float32,
    )


def test_reference_scene_preserves_synthetic_worker_public_layout(tmp_path: Path) -> None:
    from unisim.mjcf_compiler import compose_scene

    payload = scene_payload(tmp_path / "worker-assets")
    references = _reference_sources(tmp_path / "reference-assets")
    with compose_scene(references.scene, 5, SIM_DT) as composed:
        _assert_layout_equivalent(
            composed.layout, CompiledSceneLayout.from_dict(payload["scene_layout"])
        )


def test_robot_commands_stay_inside_matching_radian_joint_limits(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")

    payload = scene_payload(tmp_path / "worker-assets")
    references = _reference_sources(tmp_path / "reference-assets")
    robot = payload["scene_entities"][0]
    assert robot["name"] == "robot"
    np.testing.assert_allclose(robot["variants"][0]["dof_lower"], (-1.0,))
    np.testing.assert_allclose(robot["variants"][0]["dof_upper"], (1.0,))

    model = mujoco.MjModel.from_xml_path(references.robot.model_file)
    joint = model.joint("drive_joint")
    np.testing.assert_allclose(joint.range, (-1.0, 1.0), atol=1e-7, rtol=0)

    selected_qpos, _ = _selected_reset(np.asarray(payload["initial_qpos"], dtype=np.float32))
    commands = np.concatenate(
        (selected_qpos[:, 0], _control_trajectory(payload["num_envs"]).reshape(-1))
    )
    assert np.all(commands >= joint.range[0])
    assert np.all(commands <= joint.range[1])


def test_snapshot_contract_rejects_shape_and_nonfinite_errors() -> None:
    snapshot = RolloutSnapshot(
        source="synthetic",
        step_index=0,
        control=np.zeros((5, 1), dtype=np.float32),
        qpos=np.zeros((5, 9), dtype=np.float32),
        qvel=np.zeros((5, 8), dtype=np.float32),
        sensors={
            name: np.zeros((5, 4 if name.startswith("track_quat_w_") else 3), dtype=np.float32)
            for name in _sensor_names()
        },
    )
    _assert_snapshot_contract(snapshot, 5)
    snapshot.qpos[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _assert_snapshot_contract(snapshot, 5)


def test_trajectory_metrics_retain_every_step_and_do_not_assert_thresholds() -> None:
    def snapshot(step: int, offset: float) -> RolloutSnapshot:
        return RolloutSnapshot(
            source="synthetic",
            step_index=step,
            control=np.zeros((5, 1), dtype=np.float32),
            qpos=np.full((5, 9), offset, dtype=np.float32),
            qvel=np.full((5, 8), offset, dtype=np.float32),
            sensors={
                name: np.full(
                    (5, 4 if name.startswith("track_quat_w_") else 3),
                    offset,
                    dtype=np.float32,
                )
                if not name.startswith("track_quat_w_")
                else np.broadcast_to(
                    np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
                    (5, 4),
                ).copy()
                for name in _sensor_names()
            },
        )

    actual = [snapshot(0, 0.0), snapshot(1, 2.0)]
    expected = [snapshot(0, 1.0), snapshot(1, 0.5)]
    metrics = _trajectory_metrics(actual, expected)
    assert [step["qpos"]["max_abs"] for step in metrics] == [1.0, 1.5]
    assert _trajectory_maxima(metrics)["qpos_max_abs"] == 1.5
    assert "threshold" not in _trajectory_metrics(actual, actual)[0]


def test_mode_and_thresholds_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PARITY_MODE_ENV, raising=False)
    assert _parity_mode() == "diagnostic"
    monkeypatch.setenv(PARITY_MODE_ENV, "invalid")
    with pytest.raises(RuntimeError, match="diagnostic or acceptance"):
        _parity_mode()

    monkeypatch.setenv(PARITY_MODE_ENV, "acceptance")
    for name in THRESHOLD_ENV.values():
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="missing generalized acceptance thresholds"):
        _acceptance_thresholds()
    for name in THRESHOLD_ENV.values():
        monkeypatch.setenv(name, "nan")
    with pytest.raises(RuntimeError, match="finite non-negative"):
        _acceptance_thresholds()


def _minimal_schema2_report(mode: str) -> dict[str, Any]:
    def snapshot(source: str, step_index: int) -> RolloutSnapshot:
        result = RolloutSnapshot(
            source=source,
            step_index=step_index,
            control=np.zeros((5, 1), dtype=np.float32),
            qpos=np.zeros((5, 9), dtype=np.float32),
            qvel=np.zeros((5, 8), dtype=np.float32),
            sensors={
                name: np.zeros(
                    (5, 4 if name.startswith("track_quat_w_") else 3), dtype=np.float32
                )
                for name in _sensor_names()
            },
        )
        for name, values in result.sensors.items():
            if name.startswith("track_quat_w_"):
                values[:, 0] = 1.0
        return result

    trajectories = {
        source: [snapshot(source, index) for index in range(CONTROL_STEP_COUNT)]
        for source in ("mujoco", "mjwarp", "isaacgym_cuda_ipc")
    }
    comparisons = {
        "isaacgym_vs_mujoco": _trajectory_metrics(
            trajectories["isaacgym_cuda_ipc"], trajectories["mujoco"]
        ),
        "isaacgym_vs_mjwarp": _trajectory_metrics(
            trajectories["isaacgym_cuda_ipc"], trajectories["mjwarp"]
        ),
        "mjwarp_vs_mujoco": _trajectory_metrics(trajectories["mjwarp"], trajectories["mujoco"]),
    }
    limits = GeneralizedThresholds(1.0, 1.0, 1.0, 1.0, 1.0)
    assertions = (
        "finite",
        "shape",
        "complete_trajectory",
        *(() if mode == "diagnostic" else ("all_step_thresholds",)),
    )
    return {
        "schema_version": 2,
        "mode": mode,
        "run_id": "schema2-validation",
        "scope": {
            "task": "synthetic validation",
            "mode": mode,
            "same_control_and_reset_inputs": True,
            "cross_engine_numerical_thresholds": mode == "acceptance",
            "complete_trajectories": True,
        },
        "backend": "isaacgym",
        "candidate_execution": {
            "process_topology": "external_python38_worker",
            "data_plane": "cuda_ipc",
            "sdk_imported_in_parent": False,
            "authored_scalar_sensors": [],
            "tracked_body_sensor_projection": "worker_cuda_body_state",
        },
        "references": ("mujoco", "mjwarp"),
        "num_envs": 5,
        "selected_row": SELECTED_ROW,
        "sim_dt": SIM_DT,
        "control_substeps": CONTROL_SUBSTEPS,
        "control_step_count": CONTROL_STEP_COUNT,
        "tracked_bodies": TRACKED_BODIES,
        "tracked_sensor_fields": TRACKED_SENSOR_FIELDS,
        "reset": {
            "selected_qpos": [[0.0] * 9],
            "selected_qvel": [[0.0] * 8],
            "snapshots": {source: steps[0].report() for source, steps in trajectories.items()},
            "state_comparisons": {
                comparison: comparisons[comparison][0] for comparison in ACCEPTANCE_COMPARISONS
            },
            "asserted": ("selected_state_echo", "unselected_row_invariance"),
            "reset_echo_atol": RESET_ECHO_ATOL,
            "sensor_comparison": "deferred_until_first_step",
            "sensor_stale_window_reason": "validation",
        },
        "control_steps": {
            "controls": [[[0.0]] for _ in range(CONTROL_STEP_COUNT)],
            "snapshots": {
                source: [row.report() for row in steps] for source, steps in trajectories.items()
            },
            "comparisons": comparisons,
            "maxima": {
                comparison: _trajectory_maxima(steps)
                for comparison, steps in comparisons.items()
            },
            "asserted": assertions,
            "thresholds": None if mode == "diagnostic" else limits.report(),
            "threshold_assertions": (
                None
                if mode == "diagnostic"
                else _all_step_threshold_evidence(comparisons, limits)
            ),
        },
        "source_only_reference_normalizations": {},
        "source_provenance": {
            "synthetic": {
                "path": "synthetic.py",
                "sha256": "0" * 64,
            }
        },
        "worker_metadata": {"synthetic": True},
        "process_provenance": {"parent_pid": 1, "worker_pid": 2, "worker_returncode": 0},
        "host_runtime_versions": {
            "python": "3",
            "torch": "2",
            "mujoco": "3",
            "warp": "1",
            "mujoco_warp": "3",
        },
        "gpu": {
            "required_idle": mode == "acceptance",
            "compute_processes_before": [],
            "compute_processes_after": [],
            "own_process_pid": 1,
        },
        "gpu_device": {
            "index": "0",
            "name": "synthetic",
            "uuid": "synthetic",
            "driver_version": "synthetic",
        },
        "profiler_environment": {name: "" for name in PROFILER_ENVIRONMENT_VARIABLES},
        "threshold_policy": {
            "mode": mode,
            "variables": THRESHOLD_ENV,
            "frozen_before_rollout": True,
            "asserted_every_step": mode == "acceptance",
        },
        "assertions": {
            "worker_layout_exact": True,
            "initial_state_echo": True,
            "selected_state_echo": True,
            "unselected_row_invariance": True,
            "all_trajectories_finite": True,
            "complete_trajectory_count": CONTROL_STEP_COUNT,
            "cross_engine_parity_threshold": mode == "acceptance",
            "reason": "synthetic validation",
        },
    }


def test_schema2_artifact_reload_validator_is_strict(tmp_path: Path) -> None:
    report = _minimal_schema2_report("acceptance")
    output = tmp_path / "schema2.json"
    write_json_report(output, report)
    with output.open(encoding="utf-8") as stream:
        reloaded = json.load(stream)
    _validate_report(reloaded)

    reloaded["unexpected"] = True
    with pytest.raises(ValueError, match="report keys differ"):
        _validate_report(reloaded)
    del reloaded["unexpected"]
    reloaded["control_steps"]["threshold_assertions"]["isaacgym_vs_mujoco"][0][
        "within_threshold"
    ] = False
    with pytest.raises(ValueError, match="threshold row 0 failed"):
        _validate_report(reloaded)

    diagnostic = _minimal_schema2_report("diagnostic")
    _validate_report(diagnostic)
    diagnostic["control_steps"]["thresholds"] = GeneralizedThresholds(
        1.0, 1.0, 1.0, 1.0, 1.0
    ).report()
    with pytest.raises(ValueError, match="diagnostic artifacts must not"):
        _validate_report(diagnostic)


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACGYM_GENERALIZED_PARITY_NATIVE") != "1",
    reason="set UNISIM_TEST_ISAACGYM_GENERALIZED_PARITY_NATIVE=1 for real generalized parity",
)
def test_isaacgym_generalized_cuda_ipc_parity(tmp_path: Path) -> None:
    output = os.environ.get("UNISIM_TEST_ISAACGYM_GENERALIZED_PARITY_OUTPUT")
    current_mode = _parity_mode()
    if output is None and current_mode == "acceptance":
        raise RuntimeError("generalized acceptance requires an explicit artifact output")
    limits = _acceptance_thresholds() if current_mode == "acceptance" else None
    run_id = uuid.uuid4().hex
    profiler_values = {name: os.environ.get(name, "") for name in PROFILER_ENVIRONMENT_VARIABLES}
    if any(profiler_values.values()):
        raise RuntimeError("generalized parity must not run with Isaac worker profilers enabled")

    mujoco = pytest.importorskip("mujoco")
    warp = pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    torch = pytest.importorskip("torch")
    warp.init()
    if not torch.cuda.is_available() or not bool(warp.get_device().is_cuda):
        pytest.skip("generalized parity requires CUDA Torch and Warp")
    if not cuda_ipc.cuda_driver_available():
        pytest.skip("CUDA IPC driver is unavailable")

    from unisim import MjwarpBackend, MuJoCoBackend

    compute_processes_before = (
        require_acceptance_gpu_idle(0)
        if current_mode == "acceptance"
        else gpu_compute_process_snapshot(0)
    )
    payload = scene_payload(tmp_path / "worker-assets")
    references = _reference_sources(tmp_path / "reference-assets")
    initial_qpos = np.asarray(payload["initial_qpos"], dtype=np.float32)
    initial_qvel = np.asarray(payload["initial_qvel"], dtype=np.float32)
    selected_qpos, selected_qvel = _selected_reset(initial_qpos)
    controls = _control_trajectory(payload["num_envs"])
    control_torch = torch.as_tensor(controls, dtype=torch.float32, device="cuda:0")

    mujoco_backend = MuJoCoBackend(
        references.scene,
        num_envs=payload["num_envs"],
        sim_dt=SIM_DT,
        base_name=TRACKED_BODIES[0],
        add_body_sensors=True,
        tracked_body_names=TRACKED_BODIES,
    )
    mjwarp_backend = MjwarpBackend(
        references.scene,
        num_envs=payload["num_envs"],
        sim_dt=SIM_DT,
        base_name=TRACKED_BODIES[0],
        add_body_sensors=True,
    )
    mujoco_backend.materialize()

    client = SceneClient(payload, tmp_path / "isaacgym-worker.log")
    assert client.meta is not None
    _assert_layout_equivalent(
        CompiledSceneLayout.from_dict(client.meta["scene_layout"]), client.layout
    )
    layout = IsaacGymCudaIpcArenaLayout.create(
        payload["num_envs"],
        client.layout.nq,
        client.layout.nv,
        client.layout.nu,
        client.layout.nbody,
    )
    transport = cuda_ipc.CudaIpcTransport(0)
    allocation = transport.allocate(layout.size_bytes)
    control_event = transport.create_event()
    state_event = transport.create_event()
    reset_event = transport.create_event()
    attached = False
    candidate_steps: list[RolloutSnapshot] = []
    mujoco_steps: list[RolloutSnapshot] = []
    mjwarp_steps: list[RolloutSnapshot] = []

    qpos: Any = None
    qvel: Any = None
    control: Any = None
    reset_rows: Any = None
    reset_qpos: Any = None
    reset_qvel: Any = None
    body_state: Any = None
    try:
        qpos = torch_from_cuda_pointer(
            torch, allocation.pointer + layout.qpos_offset, layout.qpos_shape, 0
        )
        qvel = torch_from_cuda_pointer(
            torch, allocation.pointer + layout.qvel_offset, layout.qvel_shape, 0
        )
        control = torch_from_cuda_pointer(
            torch, allocation.pointer + layout.ctrl_offset, layout.ctrl_shape, 0
        )
        reset_rows = torch_from_cuda_pointer(
            torch,
            allocation.pointer + layout.reset_indices_offset,
            layout.reset_indices_shape,
            0,
            dtype="int64",
        )
        reset_qpos = torch_from_cuda_pointer(
            torch,
            allocation.pointer + layout.reset_qpos_offset,
            layout.reset_qpos_shape,
            0,
        )
        reset_qvel = torch_from_cuda_pointer(
            torch,
            allocation.pointer + layout.reset_qvel_offset,
            layout.reset_qvel_shape,
            0,
        )
        body_state = torch_from_cuda_pointer(
            torch,
            allocation.pointer + layout.body_state_offset,
            layout.body_state_shape,
            0,
        )

        attach = client.request(
            "ISAACGYM_CUDA_IPC_ATTACH",
            _cuda_ipc_wire(layout, allocation, control_event, state_event, reset_event),
        )
        attached = True
        assert attach is not None
        assert attach["device_uuid"] == transport.identity.uuid
        assert "isaacgym" not in sys.modules
        state_event.wait_stream(torch.cuda.current_stream().cuda_stream)

        reset_sequence = [1]

        def candidate_reset(rows: list[int], next_qpos: np.ndarray, next_qvel: np.ndarray) -> None:
            reset_rows[: len(rows)] = torch.as_tensor(
                rows, dtype=torch.int64, device=reset_rows.device
            )
            reset_qpos[: len(rows)].copy_(
                torch.as_tensor(next_qpos, dtype=torch.float32, device=reset_qpos.device)
            )
            reset_qvel[: len(rows)].copy_(
                torch.as_tensor(next_qvel, dtype=torch.float32, device=reset_qvel.device)
            )
            reset_event.record(torch.cuda.current_stream().cuda_stream)
            client.request(
                "ISAACGYM_CUDA_IPC_SET_STATE",
                {"count": len(rows), "sequence": reset_sequence[0]},
            )
            reset_sequence[0] += 1
            state_event.wait_stream(torch.cuda.current_stream().cuda_stream)

        all_rows = list(range(payload["num_envs"]))
        mujoco_backend.set_state(np.asarray(all_rows, dtype=np.int64), initial_qpos, initial_qvel)
        mjwarp_backend.set_state(np.asarray(all_rows, dtype=np.int64), initial_qpos, initial_qvel)
        candidate_reset(all_rows, initial_qpos, initial_qvel)

        mujoco_initial = _reference_snapshot(
            mujoco_backend, "mujoco", -1, np.zeros_like(controls[0])
        )
        mjwarp_initial = _reference_snapshot(
            mjwarp_backend, "mjwarp", -1, np.zeros_like(controls[0])
        )
        candidate_initial = _candidate_snapshot(
            "isaacgym_cuda_ipc",
            -1,
            np.zeros_like(controls[0]),
            qpos,
            qvel,
            body_state,
        )
        for snapshot in (mujoco_initial, mjwarp_initial, candidate_initial):
            _assert_snapshot_contract(snapshot, payload["num_envs"])
            _assert_reset_echo(snapshot, initial_qpos, initial_qvel)

        mujoco_backend.set_state(
            np.asarray([SELECTED_ROW], dtype=np.int64), selected_qpos, selected_qvel
        )
        mjwarp_backend.set_state(
            np.asarray([SELECTED_ROW], dtype=np.int64), selected_qpos, selected_qvel
        )
        candidate_reset([SELECTED_ROW], selected_qpos, selected_qvel)

        mujoco_reset = _reference_snapshot(mujoco_backend, "mujoco", -1, np.zeros_like(controls[0]))
        mjwarp_reset = _reference_snapshot(mjwarp_backend, "mjwarp", -1, np.zeros_like(controls[0]))
        candidate_reset_snapshot = _candidate_snapshot(
            "isaacgym_cuda_ipc",
            -1,
            np.zeros_like(controls[0]),
            qpos,
            qvel,
            body_state,
        )
        for snapshot in (
            mujoco_initial,
            mjwarp_initial,
            candidate_initial,
            mujoco_reset,
            mjwarp_reset,
            candidate_reset_snapshot,
        ):
            _assert_snapshot_contract(snapshot, payload["num_envs"])
        for before, after in (
            (mujoco_initial, mujoco_reset),
            (mjwarp_initial, mjwarp_reset),
            (candidate_initial, candidate_reset_snapshot),
        ):
            _assert_unselected_rows_unchanged(before, after, SELECTED_ROW)
        for snapshot in (mujoco_reset, mjwarp_reset, candidate_reset_snapshot):
            _assert_reset_echo(snapshot, selected_qpos, selected_qvel, row=SELECTED_ROW)

        for step_index in range(CONTROL_STEP_COUNT):
            step_control = controls[step_index]
            step_control_torch = control_torch[step_index]
            mujoco_backend.step(step_control, nsteps=CONTROL_SUBSTEPS)
            mjwarp_backend.step(step_control, nsteps=CONTROL_SUBSTEPS)
            control.copy_(step_control_torch)
            control_event.record(torch.cuda.current_stream().cuda_stream)
            client.request("ISAACGYM_CUDA_IPC_STEP", {"nsteps": CONTROL_SUBSTEPS})
            state_event.wait_stream(torch.cuda.current_stream().cuda_stream)

            mujoco_after = _reference_snapshot(mujoco_backend, "mujoco", step_index, step_control)
            mjwarp_after = _reference_snapshot(mjwarp_backend, "mjwarp", step_index, step_control)
            candidate_after = _candidate_snapshot(
                "isaacgym_cuda_ipc",
                step_index,
                step_control,
                qpos,
                qvel,
                body_state,
            )
            for snapshot in (mujoco_after, mjwarp_after, candidate_after):
                _assert_snapshot_contract(snapshot, payload["num_envs"])
            mujoco_steps.append(mujoco_after)
            mjwarp_steps.append(mjwarp_after)
            candidate_steps.append(candidate_after)

        assert len(candidate_steps) == len(mujoco_steps) == len(mjwarp_steps) == CONTROL_STEP_COUNT
        comparisons = {
            "isaacgym_vs_mujoco": _trajectory_metrics(candidate_steps, mujoco_steps),
            "isaacgym_vs_mjwarp": _trajectory_metrics(candidate_steps, mjwarp_steps),
            "mjwarp_vs_mujoco": _trajectory_metrics(mjwarp_steps, mujoco_steps),
        }
        maxima = {name: _trajectory_maxima(steps) for name, steps in comparisons.items()}
        threshold_evidence = (
            None if limits is None else _all_step_threshold_evidence(comparisons, limits)
        )

        detach = client.request("ISAACGYM_CUDA_IPC_DETACH")
        attached = False
        assert detach is not None
        assert detach["isaacgym_imported"] is True
        assert "isaacgym" not in sys.modules
    finally:
        if attached and client.proc.poll() is None:
            try:
                client.request("ISAACGYM_CUDA_IPC_DETACH")
            except Exception:
                pass
        client.close()
        mujoco_backend.close()
        mjwarp_backend.close()
        qpos = None
        qvel = None
        control = None
        reset_rows = None
        reset_qpos = None
        reset_qvel = None
        body_state = None
        gc.collect()
        state_event.close()
        control_event.close()
        reset_event.close()
        allocation.close()
        transport.close()

    fixture_path = Path(__file__).with_name("scene_fixture.py")
    compute_processes_after = (
        require_acceptance_gpu_idle(0, quiesce_timeout_s=10.0, allowed_pids={os.getpid()})
        if current_mode == "acceptance"
        else gpu_compute_process_snapshot(0)
    )
    source_paths = {
        "scene_fixture": fixture_path,
        "parity_test": Path(__file__),
        "backend_tensor": Path(__file__).parents[3]
        / "src"
        / "unisim"
        / "backend"
        / "isaacgym"
        / "tensor.py",
        "backend_scene_worker": Path(__file__).parents[3]
        / "src"
        / "unisim"
        / "backend"
        / "isaacgym"
        / "scene_worker.py",
        "backend_worker": Path(__file__).parents[3]
        / "src"
        / "unisim"
        / "backend"
        / "isaacgym"
        / "worker.py",
        "reference_robot": Path(references.robot.model_file),
        "reference_object_0": Path(references.object_a.model_file),
        "reference_object_1": Path(references.object_b.model_file),
        "reference_table": Path(references.table.model_file),
        "reference_target": Path(references.target.model_file),
        **{
            f"worker_{entity['name']}_source_{index}": Path(source)
            for entity in payload["scene_entities"]
            for index, source in enumerate(entity["sources"])
        },
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "mode": current_mode,
        "run_id": run_id,
        "scope": {
            "task": "existing isaacgym multi-entity synthetic scene",
            "mode": current_mode,
            "same_control_and_reset_inputs": True,
            "cross_engine_numerical_thresholds": limits is not None,
            "complete_trajectories": True,
        },
        "backend": "isaacgym",
        "candidate_execution": {
            "process_topology": "external_python38_worker",
            "data_plane": "cuda_ipc",
            "sdk_imported_in_parent": False,
            "authored_scalar_sensors": [],
            "tracked_body_sensor_projection": "worker_cuda_body_state",
        },
        "references": ("mujoco", "mjwarp"),
        "num_envs": payload["num_envs"],
        "selected_row": SELECTED_ROW,
        "sim_dt": SIM_DT,
        "control_substeps": CONTROL_SUBSTEPS,
        "control_step_count": CONTROL_STEP_COUNT,
        "tracked_bodies": TRACKED_BODIES,
        "tracked_sensor_fields": TRACKED_SENSOR_FIELDS,
        "reset": {
            "selected_qpos": selected_qpos.tolist(),
            "selected_qvel": selected_qvel.tolist(),
            "snapshots": {
                "mujoco": mujoco_reset.report(),
                "mjwarp": mjwarp_reset.report(),
                "isaacgym_cuda_ipc": candidate_reset_snapshot.report(),
            },
            "state_comparisons": {
                "isaacgym_vs_mujoco": _compare_snapshots(
                    candidate_reset_snapshot, mujoco_reset, include_sensors=False
                ),
                "isaacgym_vs_mjwarp": _compare_snapshots(
                    candidate_reset_snapshot, mjwarp_reset, include_sensors=False
                ),
            },
            "asserted": ("selected_state_echo", "unselected_row_invariance"),
            "reset_echo_atol": RESET_ECHO_ATOL,
            "sensor_comparison": "deferred_until_first_step",
            "sensor_stale_window_reason": (
                "Isaac rigid-body state remains stale until the first SDK step"
            ),
        },
        "control_steps": {
            "controls": controls.tolist(),
            "snapshots": {
                "mujoco": [step.report() for step in mujoco_steps],
                "mjwarp": [step.report() for step in mjwarp_steps],
                "isaacgym_cuda_ipc": [step.report() for step in candidate_steps],
            },
            "comparisons": comparisons,
            "maxima": maxima,
            "asserted": (
                "finite",
                "shape",
                "complete_trajectory",
                *(() if limits is None else ("all_step_thresholds",)),
            ),
            "thresholds": None if limits is None else limits.report(),
            "threshold_assertions": threshold_evidence,
        },
        "source_only_reference_normalizations": {
            "reference_geom_names": (
                "MJWarp requires unique geom names; names do not affect the compared physics"
            ),
            "restored_position_actuator": (
                "the worker removes MuJoCo general actuators and restores kp/kv in its drive table"
            ),
            "target_physical_kinematic_source": (
                "the raw worker's one-body kinematic mirror catalog is represented by an "
                "equivalent collision-disabled physical rigid source"
            ),
        },
        "source_provenance": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "worker_metadata": client.meta,
        "process_provenance": {
            "parent_pid": os.getpid(),
            "worker_pid": client.proc.pid,
            "worker_returncode": client.proc.returncode,
        },
        "host_runtime_versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "warp": warp.__version__,
            "mujoco_warp": pytest.importorskip("mujoco_warp").__version__,
        },
        "gpu": {
            "required_idle": current_mode == "acceptance",
            "compute_processes_before": compute_processes_before,
            "compute_processes_after": compute_processes_after,
            "own_process_pid": os.getpid(),
        },
        "gpu_device": gpu_device_snapshot(0),
        "profiler_environment": profiler_values,
        "threshold_policy": {
            "mode": current_mode,
            "variables": THRESHOLD_ENV,
            "frozen_before_rollout": True,
            "asserted_every_step": limits is not None,
        },
        "assertions": {
            "worker_layout_exact": True,
            "initial_state_echo": True,
            "selected_state_echo": True,
            "unselected_row_invariance": True,
            "all_trajectories_finite": True,
            "complete_trajectory_count": CONTROL_STEP_COUNT,
            "cross_engine_parity_threshold": limits is not None,
            "reason": (
                "predeclared thresholds were checked at every control step"
                if limits is not None
                else "no generalized-task threshold was asserted; this artifact records diagnostics"
            ),
        },
    }

    output_path = Path(
        output or tmp_path / "isaacgym-generalized-cuda-ipc-parity.json"
    )
    write_json_report(output_path, report)
    with output_path.open(encoding="utf-8") as stream:
        reloaded = json.load(stream)
    _validate_report(reloaded)
