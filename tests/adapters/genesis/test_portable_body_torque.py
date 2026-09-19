"""Real Genesis CPU acceptance for portable world-frame body torque."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("genesis")
pytest.importorskip("torch")

from tests.adapters.genesis.test_portable_entities import _scene
from unisim.backend.genesis.backend import GenesisBackend
from unisim.dr.interval import INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE
from unisim.dr.types import IntervalRandomizationPlan, IntervalTermOp
from unisim.entities import EntityStatePatch, SceneResetRequest


def test_portable_entities_body_torque_mapping_and_reset_cancellation(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    backend = GenesisBackend(scene, 5, 0.002)
    backend.materialize()
    layout = backend.get_scene_layout()
    object_body = layout.get_entity("object").body_ids[0]
    passive_base_body = layout.get_entity("passive").body_ids[0]
    object_root = layout.get_entity("object").root_qpos_indices

    capabilities = backend.get_dr_capabilities()
    assert capabilities.supports_interval_body_torque
    assert capabilities.supports_interval_term("body_torque")
    zero_force = np.zeros((5, 1, 3), np.float32)
    with pytest.raises(ValueError, match="body torque must have shape"):
        backend.apply_body_force(
            np.asarray((object_body,)), zero_force, torque=np.zeros((5, 3), np.float32)
        )
    with pytest.raises(ValueError, match="body torque contains NaN or Inf"):
        backend.apply_body_force(
            np.asarray((object_body,)),
            zero_force,
            torque=np.full((5, 1, 3), np.inf, np.float32),
        )

    qpos = backend._qpos_cache[1].copy()
    qvel = backend._qvel_cache[1].copy()
    qpos[:, object_root] = np.asarray(
        (2.0, 0.0, 1.0, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)), dtype=np.float32
    )
    backend.set_state(np.arange(5, dtype=np.intp), qpos, qvel)

    torque = np.zeros((5, 2, 3), dtype=np.float32)
    torque[:, :, 0] = 0.06
    backend.apply_body_force(
        np.asarray((object_body, passive_base_body)),
        zero_force.repeat(2, axis=1),
        torque=torque,
    )
    backend.apply_body_force(np.asarray((object_body,)), zero_force, torque=torque[:, :1])
    pending = backend._portable_pending_body_torques
    assert pending is not None
    np.testing.assert_allclose(pending[:, object_body, 0], 0.12, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(pending[:, passive_base_body, 0], 0.06, rtol=0.0, atol=1e-8)

    backend.reset_entities(
        SceneResetRequest(
            (2, 4),
            (EntityStatePatch("passive", joint_positions=np.zeros((2, 1), np.float32)),),
        )
    )
    np.testing.assert_allclose(pending[:, object_body, 0], 0.12, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(
        pending[[0, 1, 3], passive_base_body, 0], 0.06, rtol=0.0, atol=1e-8
    )
    np.testing.assert_allclose(pending[[2, 4], passive_base_body, :], 0.0, rtol=0.0, atol=1e-8)

    backend.step(np.zeros((5, 1), dtype=np.float32))
    object_velocity = backend.get_entity_state("object")["root_velocity"]
    expected_angular_velocity = 0.12 / np.asarray((0.03, 0.03, 0.03, 0.04, 0.04)) * 0.002
    np.testing.assert_allclose(
        object_velocity[:, 3], expected_angular_velocity, rtol=2e-3, atol=1e-8
    )
    np.testing.assert_allclose(object_velocity[:, [4, 5]], 0.0, atol=2e-6)
    passive_velocity = backend.get_entity_state("passive")["root_velocity"]
    assert np.all(passive_velocity[[0, 1, 3], 3] > 0.0)
    np.testing.assert_array_equal(passive_velocity[[2, 4], 3], 0.0)
    np.testing.assert_array_equal(pending, 0.0)

    consumed_angular_velocity = object_velocity[:, 3:].copy()
    stale_wrench = np.full((5, 1, 3), 0.07, dtype=np.float32)
    stale_wrench[:, :, 1:] = 0.0
    backend.apply_body_force(
        np.asarray((object_body,)), stale_wrench, torque=stale_wrench.copy()
    )
    interval_force = np.full((5, 1, 3), 0.06, dtype=np.float32)
    interval_force[:, :, 1:] = 0.0
    interval_torque = np.full((5, 1, 3), 0.06, dtype=np.float32)
    interval_torque[:, :, 1:] = 0.0
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(
                IntervalTermOp(INTERVAL_TERM_BODY_FORCE, interval_force, body_ids=(object_body,)),
                IntervalTermOp(INTERVAL_TERM_BODY_TORQUE, interval_torque, body_ids=(object_body,)),
                IntervalTermOp(INTERVAL_TERM_BODY_TORQUE, interval_torque, body_ids=(object_body,)),
            )
        )
    )
    np.testing.assert_allclose(
        backend._portable_pending_body_forces[:, object_body, 0],
        0.06,
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(pending[:, object_body, 0], 0.12, rtol=0.0, atol=1e-8)
    backend.step(np.zeros((5, 1), dtype=np.float32))
    interval_velocity = backend.get_entity_state("object")["root_velocity"]
    expected_linear_velocity = 0.06 / np.asarray((0.5, 0.5, 0.5, 1.5, 1.5))[:, None] * 0.002
    np.testing.assert_allclose(
        interval_velocity[:, 0], expected_linear_velocity[:, 0], rtol=2e-3, atol=1e-8
    )
    np.testing.assert_allclose(interval_velocity[:, 1:3], 0.0, atol=2e-6)
    np.testing.assert_allclose(
        interval_velocity[:, 3],
        consumed_angular_velocity[:, 0] + expected_angular_velocity,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        interval_velocity[:, 4:6], consumed_angular_velocity[:, 1:], atol=1e-7
    )
    np.testing.assert_array_equal(pending, 0.0)
