"""Opt-in full-G1 CUDA IPC numerical-parity evidence tests.

The tests intentionally do not run in the SDK-free suite.  Diagnostic mode is
useful under contention and writes exact metrics, while acceptance mode requires
an idle GPU, no worker profiler, and explicit first-step tolerances.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.adapters.isaac.g1_parity_harness import (
    DEFAULT_CANONICAL_SCENE,
    DEFAULT_ISAACSIM_CONTACT_SENSORS,
    DEFAULT_ISAACSIM_FLOOR,
    DEFAULT_ISAACSIM_ROBOT,
    PROFILER_ENVIRONMENT_VARIABLES,
    SCALAR_SENSOR_FIELDS,
    TRACKED_BODIES,
    TRACKED_SENSOR_FIELDS,
    G1ControlStep,
    G1Snapshot,
    ParityThresholds,
    array_metric,
    assert_control_step_trajectory_parity,
    assert_expected_isaac_cuda_ipc_capabilities,
    assert_reset_parity,
    assert_reset_view_publication,
    compare_control_step_trajectories,
    compare_snapshots,
    deterministic_control_trajectory,
    gpu_compute_process_snapshot,
    gpu_device_snapshot,
    parse_stand_fixture,
    pytest_skip_if_fixture_unavailable,
    required_acceptance_environment,
    reset_view_publication_delta,
    resolve_g1_fixture_paths,
    root_relative_body_pose,
    selected_reset_state,
    sensor_refresh_magnitude,
    snapshot_to_numpy,
    thresholds_from_environment,
    validate_mapped_robot_source,
    write_json_report,
)

SIM_DT = 0.006666666666666667
CONTROL_SUBSTEPS = 3
CONTROL_STEP_COUNT = 4
RESET_ATOL = 2e-4


def _acceptance_mode() -> bool:
    return os.environ.get("UNISIM_TEST_ISAAC_G1_PARITY_MODE", "diagnostic") == "acceptance"


def _contention_report() -> dict[str, Any]:
    if _acceptance_mode():
        from tests.adapters.isaac.g1_parity_harness import require_acceptance_gpu_idle

        return {
            "required_idle": True,
            "compute_processes_before": require_acceptance_gpu_idle(0),
            "own_process_pid": os.getpid(),
        }
    return {
        "required_idle": False,
        "compute_processes_before": gpu_compute_process_snapshot(0),
        "note": "diagnostic output is not numerical acceptance evidence",
    }


def _host_snapshot(backend: Any, source: str) -> G1Snapshot:
    states = backend.get_state(("qpos", "qvel"))
    sensors = {name: snapshot_to_numpy(backend.get_sensor_data(name)) for name in _sensor_fields()}
    return G1Snapshot(
        source=source,
        qpos=np.asarray(states["qpos"], dtype=np.float32).copy(),
        qvel=np.asarray(states["qvel"], dtype=np.float32).copy(),
        sensors=sensors,
    )


def _device_snapshot(
    backend: Any, source: str, *, sensor_views: bool = True
) -> G1Snapshot:
    states = backend.get_state_views(("qpos", "qvel"))
    sensors = (
        {name: snapshot_to_numpy(backend.get_sensor_view(name)) for name in _sensor_fields()}
        if sensor_views
        else {
            name: np.zeros(
                (
                    states["qpos"].shape[0],
                    4 if name.startswith("track_quat_w_") else 3,
                ),
                dtype=np.float32,
            )
            for name in _sensor_fields()
        }
    )
    return G1Snapshot(
        source=source,
        qpos=snapshot_to_numpy(states["qpos"]),
        qvel=snapshot_to_numpy(states["qvel"]),
        sensors=sensors,
    )


def _sensor_fields() -> tuple[str, ...]:
    return (*SCALAR_SENSOR_FIELDS, *TRACKED_SENSOR_FIELDS)


def _apply_selected_reset(
    backend: Any, rows: Any, qpos: np.ndarray, qvel: np.ndarray, *, tensor: bool
) -> None:
    if tensor:
        backend.set_state_tensor(rows, qpos, qvel)
    else:
        backend.set_state(
            rows.detach().cpu().numpy() if hasattr(rows, "detach") else rows,
            qpos,
            qvel,
        )


def _assert_unselected_rows_unchanged(
    before: G1Snapshot, after: G1Snapshot, row: int, backend_name: str
) -> None:
    qpos_metric = array_metric(after.qpos[1 - row], before.qpos[1 - row])
    qvel_metric = array_metric(after.qvel[1 - row], before.qvel[1 - row])
    assert qpos_metric.max_abs <= 2e-5, f"{backend_name} unselected qpos changed"
    assert qvel_metric.max_abs <= 2e-5, f"{backend_name} unselected qvel changed"


def _run_full_g1_parity(backend_name: str, output_path: str | Path) -> dict[str, Any]:
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")

    torch = pytest.importorskip("torch")
    warp = pytest.importorskip("warp")
    warp.init()
    if not torch.cuda.is_available() or not bool(warp.get_device().is_cuda):
        pytest.skip("full-G1 parity requires CUDA Torch and Warp")

    fixtures = resolve_g1_fixture_paths()
    pytest_skip_if_fixture_unavailable(fixtures)
    stand_qpos, stand_ctrl = parse_stand_fixture(fixtures.canonical_scene, fixtures.isaacsim_robot)
    mapped_topology = validate_mapped_robot_source(
        fixtures.canonical_scene, fixtures.isaacsim_robot
    )
    thresholds = thresholds_from_environment() if _acceptance_mode() else None
    contention = _contention_report()
    host_runtimes = {
        "torch": torch.__version__,
        "mujoco": pytest.importorskip("mujoco").__version__,
        "warp": warp.__version__,
        "mujoco_warp": pytest.importorskip("mujoco_warp").__version__,
    }

    from unisim import IsaacGymBackend, IsaacSimBackend, MjwarpBackend, MuJoCoBackend
    from unisim.dr.types import ModelSourceDescriptor
    from unisim.entities import EntityInitialState, SceneEntitySpec
    from unisim.scene import SceneCfg

    canonical_scene = SceneCfg(
        model_file=str(fixtures.canonical_scene), default_keyframe_name="stand"
    )
    mujoco_backend = MuJoCoBackend(
        canonical_scene,
        num_envs=2,
        sim_dt=SIM_DT,
        base_name="pelvis",
        add_body_sensors=True,
        tracked_body_names=TRACKED_BODIES,
    )
    mujoco_backend.materialize()
    mjwarp_backend = MjwarpBackend(
        canonical_scene,
        num_envs=2,
        sim_dt=SIM_DT,
        base_name="pelvis",
        add_body_sensors=True,
    )
    if backend_name == "isaacgym":
        isaac_backend = IsaacGymBackend(
            canonical_scene,
            num_envs=2,
            sim_dt=SIM_DT,
            device_id=0,
            env_spacing=2.0,
        )
    elif backend_name == "isaacsim":
        mapped_scene = SceneCfg(
            entity_assets=(
                SceneEntitySpec(
                    "robot",
                    ModelSourceDescriptor(str(fixtures.isaacsim_robot)),
                    root_mode="floating",
                    initial_state=EntityInitialState(
                        position=(0.0, 0.0, 0.754),
                        quaternion=(1.0, 0.0, 0.0, 0.0),
                    ),
                ),
                SceneEntitySpec(
                    "floor",
                    ModelSourceDescriptor(str(fixtures.isaacsim_floor)),
                    kind="rigid",
                    root_mode="fixed",
                ),
            ),
            default_keyframe_name="stand",
        )
        isaac_backend = IsaacSimBackend(
            mapped_scene,
            num_envs=2,
            sim_dt=SIM_DT,
            device_id=0,
            worker_timeout_s=120.0,
            tensor_cuda_ipc=True,
        )
    else:
        raise ValueError(f"unsupported Isaac backend {backend_name!r}")

    report: dict[str, Any] = {
        "schema_version": 2,
        "mode": "acceptance" if _acceptance_mode() else "diagnostic",
        "backend": backend_name,
        "num_envs": 2,
        "selected_row": 1,
        "sim_dt": SIM_DT,
        "control_substeps": CONTROL_SUBSTEPS,
        "control_step_count": CONTROL_STEP_COUNT,
        "fixtures": fixtures.report(),
        "mapped_robot_topology": mapped_topology,
        "stand_qpos": stand_qpos.tolist(),
        "stand_ctrl": stand_ctrl.tolist(),
        "gpu": contention,
        "gpu_device": gpu_device_snapshot(0),
        "host_runtime_versions": host_runtimes,
        "profiler_environment": {
            name: os.environ.get(name, "") for name in PROFILER_ENVIRONMENT_VARIABLES
        },
    }

    rows_np = np.array([1], dtype=np.int64)
    rows_torch = torch.tensor([1], dtype=torch.int64, device="cuda:0")
    all_rows_torch = torch.tensor([0, 1], dtype=torch.int64, device=rows_torch.device)
    reset_qpos, reset_qvel = selected_reset_state(stand_qpos, np.zeros(35), row_count=1)
    reset_qpos_torch = torch.as_tensor(reset_qpos, device=rows_torch.device)
    reset_qvel_torch = torch.as_tensor(reset_qvel, device=rows_torch.device)
    full_qpos = np.repeat(stand_qpos[None, :], 2, axis=0)
    full_qvel = np.zeros((2, 35), dtype=np.float32)
    full_qpos_torch = torch.as_tensor(full_qpos, device=rows_torch.device)
    full_qvel_torch = torch.as_tensor(full_qvel, device=rows_torch.device)
    control_trajectory = deterministic_control_trajectory(
        stand_ctrl, num_envs=2, steps=CONTROL_STEP_COUNT
    )
    control_trajectory_torch = torch.as_tensor(control_trajectory, device=rows_torch.device)

    views: dict[str, Any] = {}
    try:
        _apply_selected_reset(
            mujoco_backend,
            np.array([0, 1], dtype=np.int64),
            full_qpos,
            full_qvel,
            tensor=False,
        )
        _apply_selected_reset(
            mjwarp_backend,
            all_rows_torch,
            full_qpos_torch,
            full_qvel_torch,
            tensor=True,
        )
        _apply_selected_reset(
            isaac_backend,
            all_rows_torch,
            full_qpos_torch,
            full_qvel_torch,
            tensor=True,
        )
        host_initial_reset = _host_snapshot(mujoco_backend, "mujoco")
        device_initial_reset = _device_snapshot(mjwarp_backend, "mjwarp")
        isaac_sensor_views = backend_name != "isaacgym"
        isaac_initial_reset = _device_snapshot(
            isaac_backend, backend_name, sensor_views=isaac_sensor_views
        )
        assert array_metric(host_initial_reset.qpos, full_qpos).max_abs <= 2e-5, (
            "canonical MuJoCo did not start from the stand keyframe"
        )
        _apply_selected_reset(
            mujoco_backend,
            rows_np,
            reset_qpos,
            reset_qvel,
            tensor=False,
        )
        _apply_selected_reset(
            mjwarp_backend,
            rows_torch,
            reset_qpos_torch,
            reset_qvel_torch,
            tensor=True,
        )
        _apply_selected_reset(
            isaac_backend,
            rows_torch,
            reset_qpos_torch,
            reset_qvel_torch,
            tensor=True,
        )
        capabilities = assert_expected_isaac_cuda_ipc_capabilities(isaac_backend)

        host_full_reset = _host_snapshot(mujoco_backend, "mujoco")
        device_full_reset = _device_snapshot(mjwarp_backend, "mjwarp")
        # Issue #349 requires IsaacGym public views to be authoritative
        # immediately after set_state_tensor; no warm-up step is allowed here.
        isaac_sensor_views = True
        isaac_full_reset = _device_snapshot(isaac_backend, backend_name)
        for before, after, source in (
            (host_initial_reset, host_full_reset, "mujoco"),
            (device_initial_reset, device_full_reset, "mjwarp"),
            (isaac_initial_reset, isaac_full_reset, backend_name),
        ):
            _assert_unselected_rows_unchanged(before, after, row=1, backend_name=source)
        reset_comparisons = {
            "isaac_vs_mujoco": compare_snapshots(
                host_full_reset,
                isaac_full_reset,
                include_step=False,
                include_sensors=isaac_sensor_views,
            ),
            "isaac_vs_mjwarp": compare_snapshots(
                device_full_reset,
                isaac_full_reset,
                include_step=False,
                include_sensors=isaac_sensor_views,
            ),
            "mjwarp_vs_mujoco": compare_snapshots(
                host_full_reset, device_full_reset, include_step=False
            ),
        }
        if _acceptance_mode():
            for name, comparison in reset_comparisons.items():
                if name != "isaac_vs_mujoco":
                    continue
                assert_reset_parity(comparison, RESET_ATOL)

        reset_publication_delta = reset_view_publication_delta(
            isaac_initial_reset, isaac_full_reset
        )

        host_steps: list[G1ControlStep] = []
        device_steps: list[G1ControlStep] = []
        isaac_steps: list[G1ControlStep] = []
        for step_index, (control, control_torch) in enumerate(
            zip(control_trajectory, control_trajectory_torch, strict=True)
        ):
            mujoco_backend.step(control, nsteps=CONTROL_SUBSTEPS)
            mjwarp_backend.step_tensor(control_torch, nsteps=CONTROL_SUBSTEPS)
            isaac_backend.step_tensor(control_torch, nsteps=CONTROL_SUBSTEPS)
            host_after_step = _host_snapshot(mujoco_backend, "mujoco")
            device_after_step = _device_snapshot(mjwarp_backend, "mjwarp")
            isaac_after_step = _device_snapshot(isaac_backend, backend_name)
            host_steps.append(G1ControlStep(step_index, control, host_after_step))
            device_steps.append(G1ControlStep(step_index, control, device_after_step))
            isaac_steps.append(G1ControlStep(step_index, control, isaac_after_step))

        first_step_refresh = sensor_refresh_magnitude(isaac_full_reset, isaac_steps[0].snapshot)

        step_comparisons = {
            "isaac_vs_mujoco": compare_control_step_trajectories(host_steps, isaac_steps),
            "isaac_vs_mjwarp": compare_control_step_trajectories(device_steps, isaac_steps),
            "mjwarp_vs_mujoco": compare_control_step_trajectories(host_steps, device_steps),
        }
        report.update(
            {
                "isaac_tensor_capabilities": capabilities,
                "reset": {
                    "snapshots": {
                        "mujoco": host_full_reset.report(),
                        "mjwarp": device_full_reset.report(),
                        backend_name: isaac_full_reset.report(),
                    },
                    "arrays": {
                        "mujoco": host_full_reset.array_report(),
                        "mjwarp": device_full_reset.array_report(),
                        backend_name: isaac_full_reset.array_report(),
                    },
                    "comparisons": reset_comparisons,
                    "asserted": _acceptance_mode(),
                    "asserted_fields": ("qpos", "qvel", "sensors"),
                    "unasserted_sensor_reason": (
                        "IsaacSim reset sensor publication remains diagnostic"
                        if backend_name == "isaacsim"
                        else None
                    ),
                    "reset_view_publication": {
                        "metrics": reset_publication_delta.report(),
                        "public_access": "authoritative_views",
                        "asserted": _acceptance_mode(),
                        "unasserted_backend_reason": (
                            "IsaacSim publishes reset-affected body sensors at the reset "
                            "boundary in this path; the metric remains diagnostic until a "
                            "backend-specific publication contract is reviewed"
                            if backend_name == "isaacsim"
                            else None
                        ),
                    },
                },
                "control_steps": {
                    "controls": control_trajectory.tolist(),
                    "snapshots": {
                        "mujoco": [step.snapshot.report() for step in host_steps],
                        "mjwarp": [step.snapshot.report() for step in device_steps],
                        backend_name: [step.snapshot.report() for step in isaac_steps],
                    },
                    "arrays": {
                        "mujoco": [step.snapshot.array_report() for step in host_steps],
                        "mjwarp": [step.snapshot.array_report() for step in device_steps],
                        backend_name: [step.snapshot.array_report() for step in isaac_steps],
                    },
                    "comparisons": step_comparisons,
                    "asserted": thresholds is not None,
                    "thresholds": None if thresholds is None else thresholds.__dict__,
                    "first_step_sensor_refresh": first_step_refresh,
                },
            }
        )
    finally:
        views.clear()
        gc.collect()
        isaac_backend.close()
        mjwarp_backend.close()
        mujoco_backend.close()
        gc.collect()

    if _acceptance_mode():
        from tests.adapters.isaac.g1_parity_harness import require_acceptance_gpu_idle

        report["gpu"]["compute_processes_after"] = require_acceptance_gpu_idle(
            0,
            quiesce_timeout_s=10.0,
            allowed_pids={os.getpid()},
        )

    write_json_report(output_path, report)
    # Keep the complete diagnostic trajectory even when an acceptance threshold
    # rejects it; threshold decisions need the per-step error curve, not only
    # the first failing scalar from pytest.
    if thresholds is not None:
        assert_control_step_trajectory_parity(step_comparisons["isaac_vs_mujoco"], thresholds)
        assert_control_step_trajectory_parity(step_comparisons["isaac_vs_mjwarp"], thresholds)
    return report


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACGYM_G1_PARITY_NATIVE") != "1",
    reason="set UNISIM_TEST_ISAACGYM_G1_PARITY_NATIVE=1 for real IsaacGym full-G1 parity",
)
def test_isaacgym_full_g1_cuda_ipc_parity(tmp_path: Path) -> None:
    output = os.environ.get(
        "UNISIM_TEST_ISAACGYM_G1_PARITY_OUTPUT",
        str(tmp_path / "isaacgym-g1-cuda-ipc-parity.json"),
    )
    report = _run_full_g1_parity("isaacgym", output)
    assert report["isaac_tensor_capabilities"]["data_plane"] == "cuda_ipc"


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACSIM_G1_PARITY_NATIVE") != "1",
    reason="set UNISIM_TEST_ISAACSIM_G1_PARITY_NATIVE=1 for real IsaacSim full-G1 parity",
)
def test_isaacsim_full_g1_cuda_ipc_parity(tmp_path: Path) -> None:
    output = os.environ.get(
        "UNISIM_TEST_ISAACSIM_G1_PARITY_OUTPUT",
        str(tmp_path / "isaacsim-g1-cuda-ipc-parity.json"),
    )
    report = _run_full_g1_parity("isaacsim", output)
    assert report["isaac_tensor_capabilities"]["data_plane"] == "cuda_ipc"


def test_array_metric_rejects_shape_and_nonfinite_errors() -> None:
    expected = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        array_metric(np.zeros((2, 3)), expected)
    with pytest.raises(ValueError, match="finite"):
        array_metric(np.asarray([[0.0, np.nan], [0.0, 0.0]]), expected)
    metric = array_metric(np.asarray([[1.0, -2.0], [0.0, 0.0]]), expected)
    assert metric.max_abs == pytest.approx(2.0)
    assert metric.rms == pytest.approx(np.sqrt(1.25))


def test_quaternion_metric_normalizes_opposite_sign_and_rejects_nonunit() -> None:
    actual = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]])
    expected = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    metric = quaternion_metric_for_test(actual, expected)
    assert metric.max_angle_rad == pytest.approx(0.0)
    with pytest.raises(ValueError, match="non-unit"):
        quaternion_metric_for_test(actual * 2.0, expected)


def quaternion_metric_for_test(actual: np.ndarray, expected: np.ndarray) -> Any:
    from tests.adapters.isaac.g1_parity_harness import quaternion_metric

    return quaternion_metric(actual, expected)


def test_expected_capability_contract_serializes_exact_cuda_ipc_matrix() -> None:
    from tests.adapters.isaac.g1_parity_harness import expected_isaac_cuda_ipc_capabilities

    assert expected_isaac_cuda_ipc_capabilities() == {
        "execution": "device_resident",
        "state_views": True,
        "state_fields": ["qpos", "qvel"],
        "sensor_views": True,
        "stepping": True,
        "selected_reset": True,
        "selected_reset_publication": "authoritative_views",
        "reset_randomization": False,
        "fixed_variants": False,
        "host_pre_step_control": False,
        "packed_host_bridge": False,
        "process_topology": "external_worker",
        "data_plane": "cuda_ipc",
        "torch_devices": ["cuda"],
    }


def test_acceptance_tolerances_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "UNISIM_TEST_ISAAC_G1_STEP_QPOS_ATOL",
        "UNISIM_TEST_ISAAC_G1_STEP_QVEL_ATOL",
        "UNISIM_TEST_ISAAC_G1_STEP_BODY_POS_ATOL",
        "UNISIM_TEST_ISAAC_G1_STEP_QUAT_ATOL_RAD",
        "UNISIM_TEST_ISAAC_G1_STEP_BODY_VEL_ATOL",
        "UNISIM_TEST_ISAAC_G1_STEP_SCALAR_SENSOR_ATOL",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="missing.*STEP_QPOS_ATOL"):
        required_acceptance_environment()
    for name in (
        "UNISIM_TEST_ISAAC_G1_STEP_QPOS_ATOL",
        "UNISIM_TEST_ISAAC_G1_STEP_QVEL_ATOL",
        "UNISIM_TEST_ISAAC_G1_STEP_BODY_POS_ATOL",
        "UNISIM_TEST_ISAAC_G1_STEP_QUAT_ATOL_RAD",
        "UNISIM_TEST_ISAAC_G1_STEP_BODY_VEL_ATOL",
        "UNISIM_TEST_ISAAC_G1_STEP_SCALAR_SENSOR_ATOL",
    ):
        monkeypatch.setenv(name, "nan")
    with pytest.raises(RuntimeError, match="finite"):
        thresholds_from_environment()


def test_acceptance_mode_rejects_worker_profilers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "UNISIM_TEST_ISAAC_G1_STEP_QPOS_ATOL",
        "UNISIM_TEST_ISAAC_G1_STEP_QVEL_ATOL",
        "UNISIM_TEST_ISAAC_G1_STEP_BODY_POS_ATOL",
        "UNISIM_TEST_ISAAC_G1_STEP_QUAT_ATOL_RAD",
        "UNISIM_TEST_ISAAC_G1_STEP_BODY_VEL_ATOL",
        "UNISIM_TEST_ISAAC_G1_STEP_SCALAR_SENSOR_ATOL",
    ):
        monkeypatch.setenv(name, "0.1")
    monkeypatch.setenv(PROFILER_ENVIRONMENT_VARIABLES[0], "/tmp/trace.json")
    with pytest.raises(RuntimeError, match="must not enable Isaac worker profilers"):
        thresholds_from_environment()


def test_reset_parity_failure_names_worst_metric() -> None:
    from tests.adapters.isaac.g1_parity_harness import ArrayMetric

    metrics = {
        "qpos": ArrayMetric(0.1, 0.1).report(),
        "qvel": ArrayMetric(0.0, 0.0).report(),
        "pelvis_local_linvel": ArrayMetric(0.0, 0.0).report(),
        "torso_gyro": ArrayMetric(0.0, 0.0).report(),
        "body_sensors": {
            "track_pos_w_pelvis": ArrayMetric(0.0, 0.0).report(),
        },
    }
    with pytest.raises(AssertionError, match="max error 0.1"):
        assert_reset_parity(metrics, 0.05)


class DeterministicG1Backend:
    """SDK-free model of public state/sensor views and reset refresh behavior."""

    def __init__(
        self,
        *,
        grid_spacing: float = 0.0,
        drift_step: int | None = None,
        drift_qpos: float = 0.0,
        drift_qvel: float = 0.0,
    ) -> None:
        self.qpos = np.zeros((2, 36), dtype=np.float32)
        self.qvel = np.zeros((2, 35), dtype=np.float32)
        self.grid_spacing = grid_spacing
        self.drift_step = drift_step
        self.drift_qpos = drift_qpos
        self.drift_qvel = drift_qvel
        self.step_index = 0
        self.body_offsets = np.linspace(0.1, 0.5, num=len(TRACKED_BODIES) * 3)
        self.body_offsets = self.body_offsets.reshape(len(TRACKED_BODIES), 3)
        self.sensors = {
            name: np.zeros((2, 4 if name.startswith("track_quat_w_") else 3))
            for name in _sensor_fields()
        }

    def get_state_views(self, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
        del fields
        return {"qpos": self.qpos, "qvel": self.qvel}

    def get_sensor_view(self, name: str) -> np.ndarray:
        return self.sensors[name]

    def set_state_tensor(self, rows: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> None:
        # Issue #349: selected reset publishes authoritative state and all
        # declared derived views before returning.
        self.qpos[rows] = qpos
        self.qvel[rows] = qvel
        self._publish_sensors()

    def step_tensor(self, control: np.ndarray, nsteps: int) -> None:
        del nsteps
        dt = SIM_DT * CONTROL_SUBSTEPS
        self.qpos[:, :3] += self.qvel[:, :3] * dt
        self.qpos[:, 7:] += self.qvel[:, 6:] * dt
        self.qvel[:, :29] += control * 0.01
        if self.step_index == self.drift_step:
            self.qpos[:, 7] += self.drift_qpos
            self.qvel[:, 6] += self.drift_qvel
        self.step_index += 1
        self._publish_sensors()

    def _publish_sensors(self) -> None:
        for row in range(2):
            root_position = self.qpos[row, :3].copy()
            root_position[0] += row * self.grid_spacing
            root_quaternion = self.qpos[row, 3:7]
            for body_index, body in enumerate(TRACKED_BODIES):
                position = root_position + self.body_offsets[body_index]
                self.sensors[f"track_pos_w_{body}"][row] = position
                self.sensors[f"track_quat_w_{body}"][row] = root_quaternion
                self.sensors[f"track_linvel_w_{body}"][row] = self.qvel[row, :3]
                self.sensors[f"track_angvel_w_{body}"][row] = self.qvel[row, 3:6]
        self.sensors["pelvis_local_linvel"][:] = self.qvel[:, :3]
        self.sensors["torso_gyro"][:] = self.qvel[:, 3:6]


def _run_deterministic_multi_step_backends(
    candidate: DeterministicG1Backend,
) -> tuple[list[G1ControlStep], list[G1ControlStep]]:
    reference = DeterministicG1Backend()
    all_rows = np.array([0, 1], dtype=np.int64)
    selected_row = np.array([1], dtype=np.int64)
    initial_qpos = np.zeros((2, 36), dtype=np.float32)
    initial_qvel = np.zeros((2, 35), dtype=np.float32)
    initial_qpos[:, 3] = 1.0
    reference.set_state_tensor(all_rows, initial_qpos, initial_qvel)
    candidate.set_state_tensor(all_rows, initial_qpos, initial_qvel)
    reference_before_reset = _device_snapshot(reference, "reference")
    candidate_before_reset = _device_snapshot(candidate, "candidate")

    reset_qpos, reset_qvel = selected_reset_state(initial_qpos[0], initial_qvel[0], row_count=1)
    reference.set_state_tensor(selected_row, reset_qpos, reset_qvel)
    candidate.set_state_tensor(selected_row, reset_qpos, reset_qvel)
    reference_after_reset = _device_snapshot(reference, "reference")
    candidate_after_reset = _device_snapshot(candidate, "candidate")
    for before, after in (
        (reference_before_reset, reference_after_reset),
        (candidate_before_reset, candidate_after_reset),
    ):
        assert_reset_view_publication(
            reset_view_publication_delta(before, after),
            min_state_change=1e-3,
        )

    controls = deterministic_control_trajectory(
        np.linspace(-1.0, 1.0, num=29).astype(np.float32),
        num_envs=2,
        steps=CONTROL_STEP_COUNT,
    )
    reference_steps: list[G1ControlStep] = []
    candidate_steps: list[G1ControlStep] = []
    for step_index, control in enumerate(controls):
        reference.step_tensor(control, CONTROL_SUBSTEPS)
        candidate.step_tensor(control, CONTROL_SUBSTEPS)
        reference_steps.append(
            G1ControlStep(step_index, control, _device_snapshot(reference, "reference"))
        )
        candidate_steps.append(
            G1ControlStep(step_index, control, _device_snapshot(candidate, "candidate"))
        )
    assert sensor_refresh_magnitude(reference_after_reset, reference_steps[0].snapshot) > 1e-6
    return reference_steps, candidate_steps


def test_deterministic_multi_step_trajectory_survives_reset_and_grid_offset() -> None:
    reference_steps, candidate_steps = _run_deterministic_multi_step_backends(
        DeterministicG1Backend(grid_spacing=2.0)
    )
    metrics = compare_control_step_trajectories(reference_steps, candidate_steps)
    assert metrics["step_count"] == CONTROL_STEP_COUNT
    assert (
        array_metric(reference_steps[-1].snapshot.qpos, reference_steps[0].snapshot.qpos).max_abs
        > 1e-3
    )
    assert (
        array_metric(reference_steps[-1].snapshot.qvel, reference_steps[0].snapshot.qvel).max_abs
        > 1e-3
    )
    assert metrics["maxima"]["qpos"] == 0.0
    assert metrics["maxima"]["qvel"] == 0.0
    assert metrics["maxima"]["track_pos_w_"] < 1e-12
    assert metrics["maxima"]["track_quat_w_"] == 0.0


def test_multi_step_parity_catches_intermediate_qpos_and_qvel_drift() -> None:
    reference_steps, drifted_steps = _run_deterministic_multi_step_backends(
        DeterministicG1Backend(drift_step=2, drift_qpos=0.2)
    )
    metrics = compare_control_step_trajectories(reference_steps, drifted_steps)
    assert metrics["maxima"]["qpos"] == pytest.approx(0.2)
    thresholds = ParityThresholds(
        qpos=0.05,
        qvel=0.1,
        body_pos=0.03,
        body_quat_rad=0.12,
        body_velocity=0.6,
        scalar_sensor=0.2,
    )
    with pytest.raises(AssertionError, match=r"control step 2:.*qpos"):
        assert_control_step_trajectory_parity(metrics, thresholds)


def test_multi_step_parity_catches_intermediate_qvel_drift() -> None:
    reference_steps, drifted_steps = _run_deterministic_multi_step_backends(
        DeterministicG1Backend(drift_step=3, drift_qvel=0.3)
    )
    metrics = compare_control_step_trajectories(reference_steps, drifted_steps)
    assert metrics["maxima"]["qvel"] == pytest.approx(0.3)
    thresholds = ParityThresholds(
        qpos=0.05,
        qvel=0.1,
        body_pos=0.03,
        body_quat_rad=0.12,
        body_velocity=0.6,
        scalar_sensor=0.2,
    )
    with pytest.raises(AssertionError, match=r"control step 3:.*qvel"):
        assert_control_step_trajectory_parity(metrics, thresholds)


def test_body_position_parity_ignores_isaac_env_grid_offset() -> None:
    sensors = {
        name: np.zeros((2, 4 if name.startswith("track_quat_w_") else 3), dtype=np.float32)
        for name in _sensor_fields()
    }
    for name, values in sensors.items():
        if name.startswith("track_quat_w_"):
            values[:, 0] = 1.0
    reference = G1Snapshot("reference", np.zeros((2, 1)), np.zeros((2, 1)), sensors)

    shifted = {
        name: (
            np.full((2, 3), 2.0, dtype=np.float32)
            if name.startswith("track_pos_w_")
            else values.copy()
        )
        for name, values in sensors.items()
    }
    candidate = G1Snapshot("candidate", np.zeros((2, 1)), np.zeros((2, 1)), shifted)
    metrics = compare_snapshots(reference, candidate, include_step=True)
    assert metrics["summary"]["max_all"] == 0.0
    assert metrics["body_sensors"]["track_pos_w_pelvis"]["max_abs"] == 0.0


def test_body_pose_parity_uses_complete_root_relative_transform() -> None:
    yaw = np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
    identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    root_offset = np.asarray([0.2, -0.4, 0.1], dtype=np.float32)

    def sensors(
        root_position: np.ndarray, root_quaternion: np.ndarray, body_position: np.ndarray
    ) -> dict[str, Any]:
        values = {
            name: np.zeros((2, 4 if name.startswith("track_quat_w_") else 3))
            for name in _sensor_fields()
        }
        for body in TRACKED_BODIES:
            values[f"track_pos_w_{body}"][:] = body_position
            values[f"track_quat_w_{body}"][:] = root_quaternion
        reference_relative, reference_relative_quat = root_relative_body_pose(
            root_position + np.asarray([0.4, 0.2, 0.1]),
            yaw,
            root_position,
            yaw,
        )
        candidate_relative, candidate_relative_quat = root_relative_body_pose(
            root_position + root_offset, identity, root_position, identity
        )
        np.testing.assert_allclose(reference_relative, root_offset, atol=1e-7)
        np.testing.assert_allclose(candidate_relative, root_offset, atol=1e-7)
        np.testing.assert_allclose(reference_relative_quat, identity, atol=1e-7)
        np.testing.assert_allclose(candidate_relative_quat, identity, atol=1e-7)
        return values

    reference = G1Snapshot(
        "reference",
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
        sensors(
            np.asarray([1.0, 2.0, 3.0]),
            yaw,
            np.asarray([1.0, 2.0, 3.0]) + np.asarray([0.4, 0.2, 0.1]),
        ),
    )
    candidate = G1Snapshot(
        "candidate",
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
        sensors(
            np.asarray([4.0, -1.0, 0.5]),
            identity,
            np.asarray([4.0, -1.0, 0.5]) + root_offset,
        ),
    )
    metrics = compare_snapshots(reference, candidate, include_step=True)
    assert metrics["summary"]["max_all"] == 0.0
    assert metrics["body_sensors"]["track_pos_w_torso_link"]["max_abs"] == 0.0
    assert metrics["body_sensors"]["track_quat_w_torso_link"]["max_angle_rad"] == 0.0


def test_snapshot_report_uses_public_shapes() -> None:
    snapshot = G1Snapshot(
        source="fake",
        qpos=np.zeros((2, 36), dtype=np.float32),
        qvel=np.zeros((2, 35), dtype=np.float32),
        sensors={
            name: np.zeros((2, 3), dtype=np.float32)
            for name in (SCALAR_SENSOR_FIELDS[0], "track_pos_w_pelvis")
        },
    )
    assert snapshot.report()["qpos_shape"] == [2, 36]
    assert snapshot.report()["sensor_shapes"]["track_pos_w_pelvis"] == [2, 3]
    assert len(TRACKED_BODIES) == 14
    assert SimpleNamespace(source="fake").source == "fake"


def test_host_snapshot_materializes_live_sensor_views() -> None:
    class LiveViewBackend:
        def __init__(self) -> None:
            self.qpos = np.zeros((2, 36), dtype=np.float32)
            self.qvel = np.zeros((2, 35), dtype=np.float32)
            self.sensors = {name: np.zeros((2, 3), dtype=np.float32) for name in _sensor_fields()}

        def get_state(self, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
            assert fields == ("qpos", "qvel")
            return {"qpos": self.qpos, "qvel": self.qvel}

        def get_sensor_data(self, name: str) -> np.ndarray:
            return self.sensors[name]

    backend = LiveViewBackend()
    snapshot = _host_snapshot(backend, "live-view")

    backend.qpos += 1.0
    backend.qvel += 2.0
    for values in backend.sensors.values():
        values += 3.0

    np.testing.assert_array_equal(snapshot.qpos, np.zeros_like(snapshot.qpos))
    np.testing.assert_array_equal(snapshot.qvel, np.zeros_like(snapshot.qvel))
    for name, values in snapshot.sensors.items():
        assert not np.shares_memory(values, backend.sensors[name])
        np.testing.assert_array_equal(values, np.zeros_like(values))


def test_acceptance_gpu_idle_allows_bounded_cuda_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.adapters.isaac import g1_parity_harness
    from tests.adapters.isaac.g1_parity_harness import require_acceptance_gpu_idle

    snapshots = iter(([{"pid": "4181417"}], [{"pid": "4180901"}]))
    sleeps: list[float] = []
    monkeypatch.setattr(
        g1_parity_harness, "gpu_compute_process_snapshot", lambda _: next(snapshots)
    )
    monkeypatch.setattr(g1_parity_harness.time, "sleep", lambda value: sleeps.append(value))

    assert require_acceptance_gpu_idle(0, quiesce_timeout_s=1.0, allowed_pids={4180901}) == [
        {"pid": "4180901"}
    ]
    assert sleeps == [0.25]


@pytest.mark.skipif(
    not all(
        path.is_file()
        for path in (
            DEFAULT_CANONICAL_SCENE,
            DEFAULT_ISAACSIM_ROBOT,
            DEFAULT_ISAACSIM_FLOOR,
            DEFAULT_ISAACSIM_CONTACT_SENSORS,
        )
    ),
    reason="sibling UniLab G1 fixture checkout is unavailable",
)
def test_mapped_g1_fixture_is_cold_validated_without_isaac_runtime() -> None:
    pytest.importorskip("mujoco")
    paths = resolve_g1_fixture_paths()
    qpos, ctrl = parse_stand_fixture(paths.canonical_scene, paths.isaacsim_robot)
    assert qpos.shape == (36,)
    assert ctrl.shape == (29,)
    assert validate_mapped_robot_source(paths.canonical_scene, paths.isaacsim_robot) == {
        "joint_count": 30,
        "actuator_count": 29,
        "body_count": 30,
        "geom_count": 76,
    }
