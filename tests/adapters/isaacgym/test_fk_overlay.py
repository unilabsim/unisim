"""Post-reset FK body-state overlay for the legacy IsaacGym path (#141).

PhysX keeps pre-write link poses until the first ``simulate``, so the legacy
worker overlays exact MJCF forward kinematics onto freshly reset envs and
clears the overlay with the first physics step.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.adapters.isaacgym.test_legacy_adoption import _adopt, _kinematics_payload
from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.kinematics import forward_kinematics


def _state_rows(count=2, joints=2):
    """Consistent canonical qpos/qvel rows for the synthetic adoption layout."""
    qpos = np.zeros((count, 7 + joints), dtype=np.float32)
    qpos[:, 0] = [0.3, -0.2][:count]
    qpos[:, 2] = [1.1, 0.9][:count]
    qpos[:, 3] = 1.0
    qpos[:, 7:] = [[0.4, -0.3], [0.1, 0.2]][:count]
    qvel = np.zeros((count, 6 + joints), dtype=np.float32)
    qvel[:, :3] = [[0.1, 0.2, 0.3], [-0.2, 0.1, 0.0]][:count]
    qvel[:, 3:6] = [[0.05, 0.1, -0.2], [0.3, 0.0, 0.1]][:count]
    qvel[:, 6:] = [[0.6, -0.5], [0.2, 0.4]][:count]
    return qpos, qvel


def _fk_rows(qpos, qvel):
    return forward_kinematics(
        _kinematics_payload(), qpos.astype(np.float64), qvel.astype(np.float64)
    )


def _quat_rotate(quat, vec):
    return protocol.quat_rotate(np.asarray(quat)[None, :], np.asarray(vec)[None, :])[0]


def _fill_reset_slots(ctx, envs, qpos, qvel):
    """Populate the canonical reset channels like the legacy reset codec does."""
    count = len(envs)
    roots = np.zeros((count, 1, 13), dtype=np.float32)
    roots[:, 0, :7] = qpos[:, :7]
    roots[:, 0, 7:10] = qvel[:, :3]
    roots[:, 0, 10:] = np.stack(
        [_quat_rotate(qpos[row, 3:7], qvel[row, 3:6]) for row in range(count)]
    )
    ctx.slots["reset_env_ids"][:count] = envs
    ctx.slots["reset_qpos"][:count] = qpos
    ctx.slots["reset_qvel"][:count] = qvel
    ctx.slots["reset_entity_root_state"][:count] = roots
    for name in ("reset_qpos_mask", "reset_qvel_mask", "reset_root_mask"):
        ctx.slots[name][:] = 1
    return {"count": count, "entity_names": ["legacy_model"]}


def test_adoption_requires_kinematics_payload():
    with pytest.raises(RuntimeError, match="missing mjcf_kinematics"):
        _adopt({"mjcf_kinematics": None})


def test_adoption_rejects_kinematics_layout_mismatch():
    tables = _kinematics_payload()
    tables["body_names"] = ["base", "renamed"]
    with pytest.raises(RuntimeError, match="does not match the adopted public layout"):
        _adopt({"mjcf_kinematics": tables})


def test_refresh_overlays_fk_rows_and_clears_contact_for_staged_envs():
    ctx, runtime, legacy = _adopt()
    runtime.refresh()  # publish native rows first
    qpos, qvel = _state_rows()
    runtime.stage_fk_overlay_rows(np.array([1]), qpos[1:], qvel[1:])
    runtime.refresh()

    canonical = ctx.slots["body_state"]
    expected = _fk_rows(qpos[1:], qvel[1:])[0]
    np.testing.assert_allclose(canonical[1], expected, atol=1e-6)
    # The untouched env keeps its native rows.
    native = ctx._body_state.values
    assert canonical[0, 0, 0] == np.float32(native[runtime.body_ids[0, 0], 0])
    # Stale contact forces are cleared only for the staged env.
    assert np.all(ctx.slots["contact_force"][1] == 0.0)
    assert not np.all(ctx.slots["contact_force"][0] == 0.0)
    # The legacy projection adds its historical COM correction on top of the
    # exact FK link-origin velocity.
    com = np.asarray(runtime.body_com[1])  # (nbody, 3) = [[.1,0,0],[.2,0,0]]
    correction = np.cross(
        expected[:, 10:13],
        np.stack([_quat_rotate(expected[b, 3:7], com[b]) for b in range(2)]),
    )
    np.testing.assert_allclose(
        legacy["body_state"][1, :, 7:10], expected[:, 7:10] + correction, atol=1e-6
    )
    np.testing.assert_allclose(legacy["body_state"][1, :, 0:3], expected[:, 0:3], atol=1e-6)


def test_reset_stages_overlay_from_effective_state_with_partial_masks():
    ctx, runtime, legacy = _adopt()
    runtime.refresh()
    runtime._submit_pending = lambda: None
    qpos, qvel = _state_rows()
    payload = _fill_reset_slots(ctx, np.array([0, 1]), qpos, qvel)
    # Keep the first joint column (the one the FK tables actually read) and the
    # second joint's velocity untouched: the overlay must compose the effective
    # state from the uploaded columns and the current slot values.
    ctx.slots["reset_qpos_mask"][7] = 0
    ctx.slots["reset_qvel_mask"][7] = 0
    runtime.reset(payload)

    assert set(runtime.pending_body_fk) == {0, 1}
    effective_qpos = ctx.slots["qpos"].astype(np.float64)
    effective_qvel = ctx.slots["qvel"].astype(np.float64)
    effective_qpos[:, [i for i in range(9) if i != 7]] = qpos[
        :, [i for i in range(9) if i != 7]
    ]
    effective_qvel[:, [i for i in range(8) if i != 7]] = qvel[
        :, [i for i in range(8) if i != 7]
    ]
    expected = _fk_rows(effective_qpos, effective_qvel)
    for env in (0, 1):
        np.testing.assert_allclose(runtime.pending_body_fk[env], expected[env], atol=1e-6)
        np.testing.assert_allclose(ctx.slots["body_state"][env], expected[env], atol=1e-6)
        assert np.all(ctx.slots["contact_force"][env] == 0.0)


def test_first_step_clears_pending_overlay_and_republishes_native_rows():
    ctx, runtime, legacy = _adopt()
    qpos, qvel = _state_rows()
    runtime.stage_fk_overlay_rows(np.array([0, 1]), qpos, qvel)
    runtime.refresh()
    assert runtime.pending_body_fk

    # Minimal gym/torch shims so step() runs its first-simulate clear.
    ctx.sim = object()
    ctx.device = "cpu"
    ctx.gymtorch = SimpleNamespace(unwrap_tensor=lambda tensor: tensor)
    ctx.gym.set_dof_position_target_tensor = lambda sim, tensor: True
    ctx.gym.simulate = lambda sim: None
    ctx.gym.fetch_results = lambda sim, fetch: None
    ctx.torch = SimpleNamespace(
        zeros_like=np.zeros_like,
        from_numpy=lambda array: SimpleNamespace(to=lambda device: array),
        as_tensor=lambda array, dtype=None, device=None: np.asarray(array),
    )
    runtime._submit_pending = lambda: None
    runtime.step({"nsteps": 1})

    assert runtime.pending_body_fk == {}
    native = ctx._body_state.values
    np.testing.assert_array_equal(
        ctx.slots["body_state"][:, :, 0], native[runtime.body_ids][:, :, 0]
    )
