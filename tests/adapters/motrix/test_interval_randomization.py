"""Interval domain randomization for the MotrixSim native packed profile.

Covers the wrench and velocity-kick interval terms on the native (single-MJCF)
profile: per-row body torque through ``Link.add_external_torque`` delta
tracking, and world-frame linear/angular velocity deltas applied to the
configured free root's ``dof_vel`` block.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")
pytest.importorskip("mujoco")

import torch  # noqa: E402

from unisim import MotrixBackend  # noqa: E402
from unisim.dr.types import (  # noqa: E402
    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_TORQUE,
    IntervalRandomizationPlan,
    IntervalTermOp,
)
from unisim.scene import SceneCfg  # noqa: E402

MODEL = """<mujoco model='motrix-interval-dr'>
  <option timestep='0.002' gravity='0 0 -9.81'/>
  <worldbody>
    <body name='base' pos='0 0 1'>
      <freejoint name='root'/>
      <inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>
      <geom name='base_geom' type='sphere' size='.08'/>
    </body>
  </worldbody>
</mujoco>"""

FIXED_BASE_MODEL = """<mujoco model='motrix-interval-fixed'>
  <option timestep='0.002' gravity='0 0 -9.81'/>
  <worldbody>
    <body name='pedestal' pos='0 0 .5'>
      <joint name='slide' type='slide' axis='1 0 0'/>
      <inertial pos='0 0 0' mass='5' diaginertia='.5 .5 .5'/>
      <geom name='pedestal_geom' type='box' size='.1 .1 .1'/>
    </body>
  </worldbody>
</mujoco>"""


@pytest.fixture
def backend(tmp_path: Path) -> MotrixBackend:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL, encoding="utf-8")
    result = MotrixBackend(SceneCfg(str(model_path)), num_envs=4, sim_dt=0.002)
    try:
        yield result
    finally:
        result.close()


def _velocity_op(term: str, payload: np.ndarray, body_ids: tuple[int, ...]) -> IntervalTermOp:
    return IntervalTermOp(
        term=term,
        body_ids=torch.as_tensor(np.asarray(body_ids, dtype=np.int64)),
        payload=torch.as_tensor(np.asarray(payload)),
    )


def test_velocity_delta_terms_declared_on_free_root(backend: MotrixBackend) -> None:
    capabilities = backend.get_dr_capabilities()
    assert INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA in capabilities.supported_interval_terms
    assert INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA in capabilities.supported_interval_terms
    assert INTERVAL_TERM_BODY_TORQUE in capabilities.supported_interval_terms


def test_linear_velocity_delta_applies_per_row(backend: MotrixBackend) -> None:
    base_body = int(backend.get_body_ids(("base",))[0])
    linear = np.zeros((4, 1, 3), dtype=np.float32)
    linear[:, 0, 0] = (1.0, 0.0, 2.0, 0.0)
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(_velocity_op(INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA, linear, (base_body,)),)
        )
    )
    velocity = backend._host_get_base_lin_vel()
    np.testing.assert_allclose(velocity[:, 0], [1.0, 0.0, 2.0, 0.0], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(velocity[:, 1:], 0.0, rtol=0.0, atol=0.0)


def test_angular_velocity_delta_world_to_body_frame(backend: MotrixBackend) -> None:
    base_body = int(backend.get_body_ids(("base",))[0])
    # The construction base orientation is the identity, so a world-frame
    # kick and the stored body-frame velocity coincide axis-for-axis.
    angular = np.zeros((4, 1, 3), dtype=np.float32)
    angular[1, 0, 2] = 0.7
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(_velocity_op(INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA, angular, (base_body,)),)
        )
    )
    angular_velocity = backend._host_get_base_ang_vel()
    np.testing.assert_allclose(angular_velocity[1], [0.0, 0.0, 0.7], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(angular_velocity[[0, 2, 3]], 0.0, rtol=0.0, atol=0.0)

    # Kicks accumulate: a second identical op doubles the stored velocity.
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(_velocity_op(INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA, angular, (base_body,)),)
        )
    )
    angular_velocity = backend._host_get_base_ang_vel()
    np.testing.assert_allclose(angular_velocity[1], [0.0, 0.0, 1.4], rtol=1e-6, atol=1e-7)


def test_velocity_delta_rejects_non_base_body(backend: MotrixBackend) -> None:
    other_body = int(backend.get_body_ids(("base",))[0]) + 1
    linear = np.zeros((4, 1, 3), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="only supports the configured"):
        backend._apply_body_velocity_delta(np.asarray([other_body]), linear, None)


def test_velocity_delta_fails_closed_without_free_root(tmp_path: Path) -> None:
    model_path = tmp_path / "fixed.xml"
    model_path.write_text(FIXED_BASE_MODEL, encoding="utf-8")
    fixed = MotrixBackend(
        SceneCfg(str(model_path)), num_envs=2, sim_dt=0.002, base_name="pedestal"
    )
    try:
        capabilities = fixed.get_dr_capabilities()
        assert INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA not in capabilities.supported_interval_terms
        assert (
            INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA not in capabilities.supported_interval_terms
        )
        pedestal = int(fixed.get_body_ids(("pedestal",))[0])
        linear = np.zeros((2, 1, 3), dtype=np.float32)
        with pytest.raises(NotImplementedError, match="floating base"):
            fixed._apply_body_velocity_delta(np.asarray([pedestal]), linear, None)
    finally:
        fixed.close()


def test_velocity_delta_validates_payload(backend: MotrixBackend) -> None:
    base_body = int(backend.get_body_ids(("base",))[0])
    with pytest.raises(ValueError, match="has shape"):
        backend._apply_body_velocity_delta(
            np.asarray([base_body]), np.zeros((4, 3), dtype=np.float32), None
        )
    bad = np.zeros((4, 1, 3), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        backend._apply_body_velocity_delta(np.asarray([base_body]), bad, None)


def test_body_torque_interval_native(backend: MotrixBackend) -> None:
    base_body = int(backend.get_body_ids(("base",))[0])
    torque = np.zeros((4, 1, 3), dtype=np.float32)
    torque[:, 0, 1] = (0.4, -0.4, 0.4, -0.4)
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(
                IntervalTermOp(
                    term=INTERVAL_TERM_BODY_TORQUE,
                    body_ids=torch.as_tensor(np.asarray([base_body], dtype=np.int64)),
                    payload=torch.as_tensor(torque),
                ),
            )
        )
    )
    # The torque is staged as a pending native wrench; stepping consumes it
    # and the free base angular velocity responds about y.
    backend._host_step(np.zeros((4, 0), dtype=np.float32))
    angular_velocity = backend._host_get_base_ang_vel()
    assert angular_velocity[0, 1] > 0.0
    assert angular_velocity[1, 1] < 0.0
