"""Tests for the multi-root IPC slot extension (SimToolReal step 1.3c-1).

Pure-Python coverage only: slot shape derivation (legacy byte-compat lock),
host body-name → data-source mapping (robot bodies stay on ``body_state``,
rigid entity roots route to their per-entity slots), and the set_state
transaction assembly/validation.  The worker-side PhysX writes are exercised
by the backend-level Kit probe run by the main agent (step 1.3c-2).
"""

from __future__ import annotations

import numpy as np
import pytest

from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.backend import (
    MjcfSubprocessBackend,
    SubprocessModelInfo,
    SubprocessWorkerError,
)
from unisim.dr.interval import INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE
from unisim.dr.types import IntervalRandomizationPlan
from unisim.scene import SceneCfg, SceneEntitySpec


class _EntityAssetHarnessBackend(MjcfSubprocessBackend):
    """Family harness opting into composition consumption for host-side tests.

    The constructor gate rejects declared entity assets/ground planes unless
    the adapter opts in; these tests exercise the family's serialization and
    binding machinery that the IsaacSim realization stands on.
    """

    def _supports_entity_assets(self) -> bool:
        return True

    def _supports_ground_plane(self) -> bool:
        return True


ROBOT_URDF = """<?xml version="1.0"?>
<robot name="two_link">
  <link name="base_link"/>
  <link name="arm"/>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/><child link="arm"/>
    <limit lower="-1.57" upper="1.57" effort="300" velocity="10"/>
  </joint>
  <joint name="free_spin" type="continuous">
    <parent link="arm"/><child link="arm_tip"/>
    <limit effort="5" velocity="11.6"/>
  </joint>
  <link name="arm_tip"/>
</robot>
"""

OBJECT_URDF = """<?xml version="1.0"?>
<robot name="cube">
  <link name="cube_link"/>
</robot>
"""

GOALVIZ_URDF = """<?xml version="1.0"?>
<robot name="goal">
  <link name="goal_link"/>
</robot>
"""

TABLE_MJCF = """<mujoco model='table'>
  <worldbody><body name='table_body'><freejoint/>
    <geom type='box' size='0.3 0.3 0.02'/></body></worldbody>
</mujoco>"""

NUM_ENVS = 4
# Robot scan: joints shoulder/free_spin; bodies base_link/arm/arm_tip
# (arm_tip hangs off a movable continuous joint, so it is not merged).
NUM_DOF = 2
NUM_BODIES = 3


@pytest.fixture()
def asset_files(tmp_path):
    paths = {}
    for key, name, text in (
        ("robot", "two_link.urdf", ROBOT_URDF),
        ("object", "cube.urdf", OBJECT_URDF),
        ("goalviz", "goal.urdf", GOALVIZ_URDF),
        ("table", "table.xml", TABLE_MJCF),
    ):
        path = tmp_path / name
        path.write_text(text)
        paths[key] = str(path)
    return paths


def _entity_specs(files):
    return (
        SceneEntitySpec(
            name="robot", model_file=files["robot"], asset_format="urdf",
            materialization="articulation", root_mode="fixed",
        ),
        SceneEntitySpec(
            name="table", model_file=files["table"], asset_format="mjcf",
            materialization="rigid", root_mode="floating",
        ),
        SceneEntitySpec(
            name="object", model_file=files["object"], asset_format="urdf",
            materialization="rigid", root_mode="floating",
        ),
        SceneEntitySpec(
            name="goalviz", model_file=files["goalviz"], asset_format="urdf",
            materialization="rigid", root_mode="kinematic",
        ),
    )


def _worker_meta(with_entities=True):
    meta = {
        "num_dof": NUM_DOF,
        "num_bodies": NUM_BODIES,
        "dof_names": ["shoulder", "free_spin"],
        "body_names": ["base_link", "arm", "arm_tip"],
        "gravity": [0.0, 0.0, -9.81],
    }
    if with_entities:
        meta["entities"] = [
            {"name": "robot", "materialization": "articulation", "root_mode": "fixed"},
            {"name": "table", "materialization": "rigid", "root_mode": "floating"},
            {"name": "object", "materialization": "rigid", "root_mode": "floating"},
            {"name": "goalviz", "materialization": "rigid", "root_mode": "kinematic"},
        ]
    return meta


@pytest.fixture()
def multi_asset_backend(asset_files):
    backend = _EntityAssetHarnessBackend(
        SceneCfg(model_file=asset_files["robot"], entity_assets=_entity_specs(asset_files)),
        num_envs=NUM_ENVS,
        sim_dt=0.01,
    )
    backend._bind_model_metadata(_worker_meta())
    backend._allocate_slots()
    yield backend
    backend.close()


@pytest.fixture()
def legacy_backend(asset_files):
    backend = _EntityAssetHarnessBackend(
        SceneCfg(model_file=asset_files["robot"]), num_envs=NUM_ENVS, sim_dt=0.01
    )
    backend._bind_model_metadata(_worker_meta(with_entities=False))
    backend._allocate_slots()
    yield backend
    backend.close()


# ---------------------------------------------------------------------------
# protocol.slot_shapes
# ---------------------------------------------------------------------------

def test_legacy_slot_layout_byte_identical():
    shapes = protocol.slot_shapes(2, 3, 4)
    assert shapes == {
        "ctrl": (2, 3),
        "root_state": (2, 13),
        "dof_state": (2, 3, 2),
        "body_state": (2, 4, 13),
        "contact_force": (2, 4, 3),
        "reset_env_ids": (2,),
        "reset_qpos": (2, 10),
        "reset_qvel": (2, 9),
    }
    assert protocol.SLOT_NAMES == (
        "ctrl", "root_state", "dof_state", "body_state", "contact_force",
        "reset_env_ids", "reset_qpos", "reset_qvel",
    )


def test_rigid_entity_slots_appended_after_legacy():
    shapes = protocol.slot_shapes(
        8, 29, 30, rigid_root_entities=("table", "object", "goalviz")
    )
    assert list(shapes)[:8] == list(protocol.SLOT_NAMES)
    for entity in ("table", "object", "goalviz"):
        assert shapes[f"entity_root_state__{entity}"] == (8, 13)
        assert shapes[f"entity_reset_state__{entity}"] == (8, 13)
    assert protocol.slot_dtype("entity_root_state__object") == np.dtype("float32")
    assert protocol.slot_dtype("entity_reset_state__object") == np.dtype("float32")
    assert protocol.slot_nbytes("entity_root_state__object", (8, 13)) == 8 * 13 * 4
    assert shapes[protocol.WRENCH_FORCE_SLOT] == (8, 33, 3)
    assert shapes[protocol.WRENCH_TORQUE_SLOT] == (8, 33, 3)
    assert protocol.slot_dtype(protocol.WRENCH_FORCE_SLOT) == np.dtype("float32")


def test_entity_slot_names_fail_closed():
    for bad in ("", "9object", "my-object", "obj ect"):
        with pytest.raises(ValueError, match="identifier"):
            protocol.slot_shapes(1, 0, 1, rigid_root_entities=(bad,))
    with pytest.raises(ValueError, match="unique"):
        protocol.slot_shapes(1, 0, 1, rigid_root_entities=("object", "object"))
    with pytest.raises(ValueError, match="identifier"):
        protocol.slot_dtype("entity_root_state__9bad")
    with pytest.raises(ValueError, match="unknown shm slot"):
        protocol.slot_dtype("nope")


# ---------------------------------------------------------------------------
# Host slot allocation
# ---------------------------------------------------------------------------

def test_allocate_slots_multi_asset(multi_asset_backend):
    slots = multi_asset_backend._slots
    assert list(slots)[:8] == list(protocol.SLOT_NAMES)
    assert slots["ctrl"].shape == (NUM_ENVS, NUM_DOF)
    assert slots["body_state"].shape == (NUM_ENVS, NUM_BODIES, 13)
    for entity in ("table", "object", "goalviz"):
        assert slots[f"entity_root_state__{entity}"].shape == (NUM_ENVS, 13)
        assert slots[f"entity_reset_state__{entity}"].shape == (NUM_ENVS, 13)
    assert slots[protocol.WRENCH_FORCE_SLOT].shape == (NUM_ENVS, NUM_BODIES + 3, 3)
    assert slots[protocol.WRENCH_TORQUE_SLOT].shape == (NUM_ENVS, NUM_BODIES + 3, 3)
    assert multi_asset_backend._rigid_root_entities == ("table", "object", "goalviz")


def test_allocate_slots_legacy_unchanged(legacy_backend):
    assert list(legacy_backend._slots) == list(protocol.SLOT_NAMES)
    assert legacy_backend._rigid_root_entities == ()


def test_multi_asset_interval_wrench_stages_only_rigid_root_rows(multi_asset_backend):
    backend = multi_asset_backend
    backend._model_info = SubprocessModelInfo(
        num_dof=NUM_DOF,
        num_bodies=NUM_BODIES,
        dof_names=("shoulder", "free_spin"),
        body_names=("base_link", "arm", "arm_tip"),
        gravity=(0.0, 0.0, -9.81),
        use_gpu_pipeline=True,
    )
    backend._require_state = lambda operation: None
    force = np.full((NUM_ENVS, 1, 3), 2.0, dtype=np.float32)
    torque = np.full((NUM_ENVS, 1, 3), 0.5, dtype=np.float32)
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            body_ids=np.asarray([NUM_BODIES + 1], dtype=np.int32),
            body_force=force,
            body_torque=torque,
        )
    )
    np.testing.assert_array_equal(
        backend._slots[protocol.WRENCH_FORCE_SLOT][:, NUM_BODIES + 1, :], force[:, 0, :]
    )
    np.testing.assert_array_equal(
        backend._slots[protocol.WRENCH_TORQUE_SLOT][:, NUM_BODIES + 1, :], torque[:, 0, :]
    )
    assert backend.get_dr_capabilities().supports_interval_body_force
    assert backend.get_dr_capabilities().supports_interval_body_torque
    with pytest.raises(ValueError, match="select unique rigid roots"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(
                body_ids=np.asarray([1], dtype=np.int32),
                body_force=force,
            )
        )


# ---------------------------------------------------------------------------
# Interval wrench staging (handler table + public apply_body_force)
# ---------------------------------------------------------------------------

def test_interval_term_handlers_table_is_cached(multi_asset_backend):
    backend = multi_asset_backend
    table = backend._interval_term_handlers()
    assert set(table) == {INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE}
    # The table is built once on the cold path and never rebuilt per plan.
    assert backend._interval_term_handlers() is table


def test_apply_body_force_stages_dense_slots_directly(multi_asset_backend):
    backend = multi_asset_backend
    force = np.full((NUM_ENVS, 1, 3), 3.0, dtype=np.float32)
    torque = np.full((NUM_ENVS, 1, 3), -1.5, dtype=np.float32)
    backend.apply_body_force(np.asarray([NUM_BODIES], dtype=np.int32), force, torque)
    np.testing.assert_array_equal(
        backend._slots[protocol.WRENCH_FORCE_SLOT][:, NUM_BODIES, :], force[:, 0, :]
    )
    np.testing.assert_array_equal(
        backend._slots[protocol.WRENCH_TORQUE_SLOT][:, NUM_BODIES, :], torque[:, 0, :]
    )
    # The public staging entry accumulates within the control step like the
    # plan path.
    backend.apply_body_force(np.asarray([NUM_BODIES], dtype=np.int32), force)
    np.testing.assert_array_equal(
        backend._slots[protocol.WRENCH_FORCE_SLOT][:, NUM_BODIES, :], 2.0 * force[:, 0, :]
    )


def test_apply_body_force_torque_none_leaves_torque_slot_untouched(multi_asset_backend):
    backend = multi_asset_backend
    backend._slots[protocol.WRENCH_TORQUE_SLOT][:] = 7.0
    force = np.full((NUM_ENVS, 1, 3), 1.25, dtype=np.float32)
    backend.apply_body_force(np.asarray([NUM_BODIES + 2], dtype=np.int32), force)
    np.testing.assert_array_equal(
        backend._slots[protocol.WRENCH_FORCE_SLOT][:, NUM_BODIES + 2, :], force[:, 0, :]
    )
    # torque=None writes only the force channel (base docstring semantics).
    np.testing.assert_array_equal(backend._slots[protocol.WRENCH_TORQUE_SLOT], 7.0)


# ---------------------------------------------------------------------------
# Host body-name → data-source mapping
# ---------------------------------------------------------------------------

def test_body_ids_extended_for_rigid_roots(multi_asset_backend):
    backend = multi_asset_backend
    ids = backend.get_body_ids(["arm", "table_body", "cube_link", "goal_link"])
    # Robot bodies keep the body_state indices; rigid roots take extended ids
    # num_bodies + declaration order.
    assert list(ids) == [1, 3, 4, 5]
    with pytest.raises(ValueError, match="not found"):
        backend.get_body_ids(["unknown_body"])


def test_selected_body_state_routes_rigid_roots(multi_asset_backend):
    backend = multi_asset_backend
    robot_marker = np.full((NUM_ENVS, NUM_BODIES, 13), 1.0, dtype=np.float32)
    object_marker = np.full((NUM_ENVS, 13), 2.0, dtype=np.float32)
    goalviz_marker = np.full((NUM_ENVS, 13), 3.0, dtype=np.float32)
    np.copyto(backend._slots["body_state"], robot_marker)
    np.copyto(backend._slots["entity_root_state__object"], object_marker)
    np.copyto(backend._slots["entity_root_state__goalviz"], goalviz_marker)

    mixed = backend._selected_body_state(np.asarray([1, 4, 5], dtype=np.int32))
    assert mixed.shape == (NUM_ENVS, 3, 13)
    np.testing.assert_array_equal(mixed[:, 0, :], 1.0)
    np.testing.assert_array_equal(mixed[:, 1, :], 2.0)
    np.testing.assert_array_equal(mixed[:, 2, :], 3.0)

    # Public getters slice the routed state correctly.
    state = np.arange(13, dtype=np.float32)
    np.copyto(backend._slots["entity_root_state__object"], np.tile(state, (NUM_ENVS, 1)))
    ids = backend.get_body_ids(["cube_link"])
    tiled = lambda sl: np.tile(sl, (NUM_ENVS, 1))  # noqa: E731
    np.testing.assert_array_equal(backend.get_body_pos_w(ids)[:, 0, :], tiled(state[0:3]))
    np.testing.assert_array_equal(backend.get_body_quat_w(ids)[:, 0, :], tiled(state[3:7]))
    np.testing.assert_array_equal(backend.get_body_lin_vel_w(ids)[:, 0, :], tiled(state[7:10]))
    np.testing.assert_array_equal(backend.get_body_ang_vel_w(ids)[:, 0, :], tiled(state[10:13]))
    pos, quat, lin_vel, ang_vel = backend.get_body_state_w(ids)
    np.testing.assert_array_equal(pos[:, 0, :], tiled(state[0:3]))
    np.testing.assert_array_equal(quat[:, 0, :], tiled(state[3:7]))
    np.testing.assert_array_equal(lin_vel[:, 0, :], tiled(state[7:10]))
    np.testing.assert_array_equal(ang_vel[:, 0, :], tiled(state[10:13]))

    with pytest.raises(ValueError, match="body_ids"):
        backend._selected_body_state(np.asarray([6], dtype=np.int32))


def test_robot_only_selection_unchanged(multi_asset_backend):
    backend = multi_asset_backend
    marker = np.arange(NUM_ENVS * NUM_BODIES * 13, dtype=np.float32).reshape(
        NUM_ENVS, NUM_BODIES, 13
    )
    np.copyto(backend._slots["body_state"], marker)
    selected = backend._selected_body_state(np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(selected, marker[:, [0], :])


def test_rigid_root_name_collision_fails_closed(asset_files, tmp_path):
    bad_object = tmp_path / "bad_cube.urdf"
    bad_object.write_text('<robot name="cube"><link name="arm"/></robot>')
    specs = list(_entity_specs(asset_files))
    specs[2] = SceneEntitySpec(
        name="object", model_file=str(bad_object), asset_format="urdf",
        materialization="rigid", root_mode="floating",
    )
    backend = _EntityAssetHarnessBackend(
        SceneCfg(model_file=asset_files["robot"], entity_assets=tuple(specs)),
        num_envs=NUM_ENVS,
        sim_dt=0.01,
    )
    try:
        with pytest.raises(SubprocessWorkerError, match="collides"):
            backend._bind_model_metadata(_worker_meta())
    finally:
        backend.close()


def test_worker_entity_mismatch_fails_closed(asset_files):
    backend = _EntityAssetHarnessBackend(
        SceneCfg(model_file=asset_files["robot"], entity_assets=_entity_specs(asset_files)),
        num_envs=NUM_ENVS,
        sim_dt=0.01,
    )
    try:
        meta = _worker_meta()
        meta["entities"] = meta["entities"][:3]  # worker dropped goalviz
        with pytest.raises(SubprocessWorkerError, match="do not match"):
            backend._bind_model_metadata(meta)
        meta = _worker_meta()
        del meta["entities"]
        with pytest.raises(SubprocessWorkerError, match="no entities list"):
            backend._bind_model_metadata(meta)
    finally:
        backend.close()


def test_unexpected_worker_entities_fail_closed(asset_files):
    backend = _EntityAssetHarnessBackend(
        SceneCfg(model_file=asset_files["robot"]), num_envs=NUM_ENVS, sim_dt=0.01
    )
    try:
        with pytest.raises(SubprocessWorkerError, match="declares no entity_assets"):
            backend._bind_model_metadata(_worker_meta(with_entities=True))
    finally:
        backend.close()


def test_rigid_root_ids_unavailable_before_materialize(asset_files):
    backend = _EntityAssetHarnessBackend(
        SceneCfg(model_file=asset_files["robot"], entity_assets=_entity_specs(asset_files)),
        num_envs=NUM_ENVS,
        sim_dt=0.01,
    )
    try:
        with pytest.raises(ValueError, match="not found"):
            backend.get_body_ids(["cube_link"])
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# set_state transaction assembly
# ---------------------------------------------------------------------------

def _capture_request(backend):
    captured = []

    def fake_request(cmd, payload, *, expect):
        captured.append((cmd, payload))
        return {"timing": {}}

    backend._request = fake_request
    return captured


def test_set_state_combined_transaction(multi_asset_backend):
    backend = multi_asset_backend
    captured = _capture_request(backend)
    rows = np.asarray([1, 3])
    qpos = np.full((2, 7 + NUM_DOF), 0.5, dtype=np.float32)
    qvel = np.zeros((2, 6 + NUM_DOF), dtype=np.float32)
    object_state = np.full((2, 13), 2.0, dtype=np.float32)
    goalviz_state = np.full((2, 13), 3.0, dtype=np.float32)
    backend.set_state(
        rows,
        qpos,
        qvel,
        entity_root_states={"object": object_state, "goalviz": goalviz_state},
    )
    assert len(captured) == 1
    cmd, payload = captured[0]
    assert cmd == protocol.CMD_SET_STATE
    # One transaction: robot + both entity roots against the same env rows.
    assert payload == {
        "count": 2,
        "robot": True,
        "entity_roots": ["goalviz", "object"],
    }
    np.testing.assert_array_equal(backend._slots["reset_env_ids"][:2], [1, 3])
    np.testing.assert_array_equal(backend._slots["reset_qpos"][:2], qpos)
    np.testing.assert_array_equal(backend._slots["reset_qvel"][:2], qvel)
    np.testing.assert_array_equal(
        backend._slots["entity_reset_state__object"][:2], object_state
    )
    np.testing.assert_array_equal(
        backend._slots["entity_reset_state__goalviz"][:2], goalviz_state
    )


def test_set_state_goalviz_only_transaction(multi_asset_backend):
    backend = multi_asset_backend
    captured = _capture_request(backend)
    goalviz_state = np.full((1, 13), 4.0, dtype=np.float32)
    backend.set_state(np.asarray([2]), entity_root_states={"goalviz": goalviz_state})
    cmd, payload = captured[0]
    assert payload == {"count": 1, "robot": False, "entity_roots": ["goalviz"]}
    np.testing.assert_array_equal(
        backend._slots["entity_reset_state__goalviz"][:1], goalviz_state
    )


def test_set_state_legacy_payload_byte_identical(legacy_backend):
    backend = legacy_backend
    captured = _capture_request(backend)
    qpos = np.zeros((2, 7 + NUM_DOF), dtype=np.float32)
    qvel = np.zeros((2, 6 + NUM_DOF), dtype=np.float32)
    backend.set_state(np.asarray([0, 2]), qpos, qvel)
    cmd, payload = captured[0]
    assert payload == {"count": 2}


def test_set_state_fail_closed_branches(multi_asset_backend, legacy_backend):
    backend = multi_asset_backend
    _capture_request(backend)
    qpos = np.zeros((1, 7 + NUM_DOF), dtype=np.float32)
    qvel = np.zeros((1, 6 + NUM_DOF), dtype=np.float32)
    state = np.zeros((1, 13), dtype=np.float32)
    rows = np.asarray([0])

    with pytest.raises(ValueError, match="declared rigid scene entities"):
        backend.set_state(rows, qpos, qvel, entity_root_states={"robot": state})
    with pytest.raises(ValueError, match=r"\(1, 13\)"):
        backend.set_state(
            rows, qpos, qvel, entity_root_states={"object": np.zeros((1, 12), np.float32)}
        )
    bad = state.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        backend.set_state(rows, qpos, qvel, entity_root_states={"object": bad})
    with pytest.raises(ValueError, match="together"):
        backend.set_state(rows, qpos, None)
    with pytest.raises(ValueError, match="silent no-op"):
        backend.set_state(rows)
    bad_qpos = qpos.copy()
    bad_qpos[0, 3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        backend.set_state(rows, bad_qpos, qvel)
    # Entity writes on a legacy scene (no declared rigid entities) fail closed.
    with pytest.raises(ValueError, match="declared rigid scene entities"):
        legacy_backend.set_state(rows, qpos, qvel, entity_root_states={"object": state})


def test_set_state_empty_rows_is_a_noop(multi_asset_backend):
    backend = multi_asset_backend
    captured = _capture_request(backend)
    result = backend.set_state(
        np.asarray([], dtype=np.intp),
        np.zeros((0, 7 + NUM_DOF), dtype=np.float32),
        np.zeros((0, 6 + NUM_DOF), dtype=np.float32),
    )
    assert captured == []
    assert "timing" in result


def test_set_state_cancels_staged_wrench_rows(multi_asset_backend):
    """A reset cancels any wrench staged for the reset rows.

    The original clears its wrench buffers inside the task reset
    (reset_utils.py:405-406) and re-gates at the next pre-physics step, so a
    freshly reset row never receives a pre-reset impulse.  The worker applies
    the staged slots at the next CMD_STEP and clears them only afterwards, so
    the host must zero the selected rows here.
    """
    backend = multi_asset_backend
    _capture_request(backend)
    rows = np.asarray([1, 3])
    backend._slots[protocol.WRENCH_FORCE_SLOT][:] = 5.0
    backend._slots[protocol.WRENCH_TORQUE_SLOT][:] = -5.0
    qpos = np.full((2, 7 + NUM_DOF), 0.5, dtype=np.float32)
    qvel = np.zeros((2, 6 + NUM_DOF), dtype=np.float32)
    backend.set_state(rows, qpos, qvel)
    # Reset rows carry no staged wrench...
    np.testing.assert_array_equal(backend._slots[protocol.WRENCH_FORCE_SLOT][rows], 0.0)
    np.testing.assert_array_equal(backend._slots[protocol.WRENCH_TORQUE_SLOT][rows], 0.0)
    # ...untouched rows keep theirs byte-for-byte.
    untouched = [index for index in range(NUM_ENVS) if index not in (1, 3)]
    np.testing.assert_array_equal(
        backend._slots[protocol.WRENCH_FORCE_SLOT][untouched], 5.0
    )
    np.testing.assert_array_equal(
        backend._slots[protocol.WRENCH_TORQUE_SLOT][untouched], -5.0
    )


# ---------------------------------------------------------------------------
# Pre-step control registration (declared gap)
# ---------------------------------------------------------------------------


def test_set_pre_step_control_fails_closed(legacy_backend):
    """Registering a per-substep host callback fails closed on the family.

    Every physics substep is integrated inside the worker process, so a host
    callback cannot run inside one; accepting the registration would silently
    drop it (declared gap).  Clearing with ``None``
    keeps the base unregister contract because "no callback" is this family's
    real state.
    """
    with pytest.raises(NotImplementedError, match="declared gap"):
        legacy_backend.set_pre_step_control(lambda owner, ctrl: ctrl)
    # ``None`` stays the accepted clear: position-actuator envs keep the
    # direct control path and never register a callback.
    legacy_backend.set_pre_step_control(None)
    assert legacy_backend._pre_step_control_fn is None
