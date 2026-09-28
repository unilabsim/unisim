"""Canonical SuperDex G1 packed publication and control-path parity tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unisim import create_backend
from unisim.backend.base import TensorIOSpec
from unisim.scene import SceneCfg

if sys.version_info[:2] not in ((3, 12), (3, 13)):
    pytest.skip("SuperDex wheels require Python 3.12 or 3.13", allow_module_level=True)
pytest.importorskip("superdex.physics")
torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")


_ROOT = Path(__file__).resolve().parents[3]
_G1_SCENE = (
    _ROOT.parent / "UniLab" / "src" / "unilab" / "assets" / "robots" / "g1" / "scene_flat.xml"
)
_TRACKED_BODIES = (
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
_AUTHORED_SENSORS = (
    "pelvis_local_linvel",
    "pelvis_gyro",
    "pelvis_upvector",
    "torso_gyro",
    "torso_upvector",
    "left_foot_pos",
    "left_foot_quat",
    "left_foot_upvector",
    "right_foot_pos",
    "right_foot_quat",
    "right_foot_upvector",
    "left_foot_contact_0",
    "left_foot_contact_1",
    "left_foot_contact_2",
    "left_foot_contact_3",
    "right_foot_contact_0",
    "right_foot_contact_1",
    "right_foot_contact_2",
    "right_foot_contact_3",
)
_ACCELEROMETERS = ("pelvis_acceleration", "torso_acceleration")
_ALL_PUBLIC_SENSORS = _AUTHORED_SENSORS + tuple(
    f"{prefix}_{body_name}"
    for body_name in _TRACKED_BODIES
    for prefix in ("track_pos_w", "track_quat_w", "track_linvel_w", "track_angvel_w")
)
_RESET_ROWS = np.asarray((1, 3), dtype=np.int64)


def _named_values(
    backend: Any, names: tuple[str, ...], *, rows: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    result = {name: backend.get_sensor_data(name) for name in names}
    if rows is not None:
        result = {name: value[rows] for name, value in result.items()}
    return result


def _packet_floats_per_env(backend: Any) -> int:
    model = backend.model
    widths = {"qpos": model.nq, "qvel": model.nv}
    for name in _ALL_PUBLIC_SENSORS:
        if name.startswith("track_quat_w_"):
            widths[name] = 4
        elif name.startswith("track_"):
            widths[name] = 3
        else:
            sensor = next(sensor for sensor in model.sensors if sensor.name == name)
            widths[name] = {"framequat": 4, "contact_found": 1}.get(str(sensor.kind), 3)
    return sum(widths.values())


def _max_error(actual: Any, expected: np.ndarray) -> float:
    actual_host = actual.detach().cpu().numpy()
    return float(np.max(np.abs(actual_host.astype(np.float64) - expected.astype(np.float64))))


def _compare_publication(
    views: dict[str, Any],
    expected: dict[str, np.ndarray],
    *,
    selected_rows: bool,
) -> float:
    assert set(views) == set(expected)
    errors = {
        name: _max_error(
            views[name][_RESET_ROWS] if selected_rows else views[name],
            expected[name][_RESET_ROWS] if selected_rows else expected[name],
        )
        for name in expected
    }
    assert errors and max(errors.values()) <= 1.0e-6
    return max(errors.values())


def test_canonical_g1_full_public_packed_publication_and_control_parity() -> None:
    """Check all 75 public views and the packed CUDA control path for 12 steps."""

    if not _G1_SCENE.is_file():
        pytest.skip(f"canonical UniLab G1 scene is unavailable: {_G1_SCENE}")
    if not torch.cuda.is_available():
        pytest.skip("canonical SuperDex packed parity requires CUDA")

    reference = create_backend("superdex", SceneCfg(str(_G1_SCENE)), 4, 1.0 / 150.0)
    candidate = create_backend("superdex", SceneCfg(str(_G1_SCENE)), 4, 1.0 / 150.0)
    plan = None
    try:
        assert reference.num_envs == candidate.num_envs == 4
        assert (candidate.model.nq, candidate.model.nv, candidate.num_actuators) == (36, 35, 29)
        assert len(_AUTHORED_SENSORS) == 19
        assert len(_ALL_PUBLIC_SENSORS) == 75
        assert set(candidate._unsupported_sensors) == set(_ACCELEROMETERS)

        initial_reference_state = reference.get_state(("qpos", "qvel"))
        initial_candidate_state = candidate.get_state(("qpos", "qvel"))
        for name in initial_reference_state:
            np.testing.assert_allclose(
                initial_candidate_state[name], initial_reference_state[name], atol=1.0e-5
            )
        for name in _ALL_PUBLIC_SENSORS:
            np.testing.assert_allclose(
                candidate.get_sensor_data(name), reference.get_sensor_data(name), atol=1.0e-5
            )

        plan = candidate.compile_host_bridge_io(
            TensorIOSpec(
                state_fields=("qpos", "qvel"),
                sensor_names=_ALL_PUBLIC_SENSORS,
                device="cuda",
            )
        )
        assert plan.transfer_stats == {
            "d2h_count": 0,
            "h2d_count": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
            "synchronization_count": 0,
        }
        initial_views = plan.read_state_sensors()
        initial_expected = {
            **candidate.get_state(("qpos", "qvel")),
            **_named_values(candidate, _ALL_PUBLIC_SENSORS),
        }
        assert _compare_publication(initial_views, initial_expected, selected_rows=False) <= 1e-6

        for step_index in range(12):
            reset_state = reference.get_state(("qpos", "qvel"))
            qpos = reset_state["qpos"][_RESET_ROWS].copy()
            qvel = reset_state["qvel"][_RESET_ROWS].copy()
            qpos[:, 0] += np.asarray((0.017, -0.013)) * (step_index + 1)
            qpos[:, 7] += np.asarray((0.003, -0.002)) * (step_index + 1)
            qvel[:, 3:6] += (
                np.asarray(((0.05, -0.04, 0.03), (-0.03, 0.04, -0.05))) * (step_index + 1) * 0.1
            )
            reference.set_state(_RESET_ROWS, qpos.copy(), qvel.copy())
            plan.apply_reset(
                torch.tensor(_RESET_ROWS, dtype=torch.int64, device="cuda"),
                torch.tensor(qpos, dtype=torch.float32, device="cuda"),
                torch.tensor(qvel, dtype=torch.float32, device="cuda"),
            )

            selected_views = plan.read_selected_state_sensors()
            after_reset_expected = {
                **reference.get_state(("qpos", "qvel")),
                **_named_values(reference, _ALL_PUBLIC_SENSORS),
            }
            assert (
                _compare_publication(selected_views, after_reset_expected, selected_rows=True)
                <= 1e-6
            )
            assert (
                _compare_publication(selected_views, after_reset_expected, selected_rows=False)
                <= 1e-6
            )

            ctrl = np.tile(candidate._default_ctrl[0].astype(np.float32), (candidate.num_envs, 1))
            ctrl[:, 0] += np.float32(0.035 * (step_index % 3) - 0.035)
            ctrl[:, 7] -= np.float32(0.027 * (step_index % 4))
            reference.step(ctrl, 3)
            plan.write_control(torch.tensor(ctrl, dtype=torch.float32, device="cuda"))
            plan.step(3)
            full_views = plan.read_state_sensors()
            same_engine_expected = {
                **candidate.get_state(("qpos", "qvel")),
                **_named_values(candidate, _ALL_PUBLIC_SENSORS),
            }
            independent_control_expected = {
                **reference.get_state(("qpos", "qvel")),
                **_named_values(reference, _ALL_PUBLIC_SENSORS),
            }
            assert (
                _compare_publication(full_views, same_engine_expected, selected_rows=False) <= 1e-6
            )
            assert (
                _compare_publication(full_views, independent_control_expected, selected_rows=False)
                <= 1.0e-5
            )

        for name in _ACCELEROMETERS:
            with pytest.raises(NotImplementedError, match="no substitute is published"):
                candidate.get_sensor_data(name)

        assert _packet_floats_per_env(candidate) == 296
        assert plan.transfer_stats == {
            "d2h_count": 24,
            "h2d_count": 25,
            "d2h_bytes": 12_480,
            "h2d_bytes": 89_984,
            "synchronization_count": 49,
        }
    finally:
        if plan is not None:
            plan.close()
        candidate.close()
        reference.close()
