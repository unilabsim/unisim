"""Scene-worker reset randomization and body-wrench tests with a fake Gym."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from tests.adapters.isaacgym.scene_fixture import add_public_geoms, scene_payload
from unisim.backend.isaacgym.scene_worker import SceneWorker
from unisim.backend.subprocess_ipc import protocol
from unisim.entities import EntityStatePatch, SceneResetRequest
from unisim.entity_state import prepare_scene_reset

pytest.importorskip("mujoco")

# (num_envs, nq, nv, nu, nbody) from scene_fixture: 5 envs, 9/8/1, 7 bodies.
_DOF_FIELDS = [
    (name, float)
    for name in (
        "driveMode",
        "stiffness",
        "damping",
        "effort",
        "armature",
        "friction",
        "hasLimits",
        "lower",
        "upper",
    )
]


def _vec3(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def _body_prop(mass=1.0):
    return SimpleNamespace(
        mass=mass,
        com=_vec3(),
        inertia=SimpleNamespace(x=_vec3(0.01), y=_vec3(0, 0.01), z=_vec3(0, 0, 0.01)),
    )


def _payload(tmp_path):
    return add_public_geoms(scene_payload(tmp_path))


def _worker(tmp_path):
    payload = _payload(tmp_path)
    ctx = SimpleNamespace(protocol=protocol)
    worker = SceneWorker(ctx, payload)
    ctx.slots = {
        name: np.zeros(shape, dtype=protocol.slot_dtype(name))
        for name, shape in protocol.scene_slot_shapes(worker.num_envs, worker.layout).items()
    }
    ctx.slots["qpos"][:] = worker.qpos0
    ctx.slots["qvel"][:] = worker.qvel0
    ctx.slots["entity_root_state"][:] = worker.roots0
    worker.refresh = lambda: None
    worker._submit_pending = lambda: None

    dofs, bodies, shapes, ranges = {}, {}, {}, {}
    for env in range(worker.num_envs):
        for index, entity in enumerate(worker.layout.entities):
            actor = (env, index)
            nj = len(entity.joints)
            props = np.zeros(nj, dtype=_DOF_FIELDS)
            if nj:
                props["stiffness"] = 20.0
                props["damping"] = 1.0
            dofs[actor] = props
            bodies[actor] = [_body_prop() for _ in entity.body_names]
            shape_list = []
            range_list = []
            for body_name in entity.body_names:
                count = sum(1 for geom in entity.geoms if geom.body_name == body_name)
                range_list.append(SimpleNamespace(start=len(shape_list), count=count))
                shape_list.extend(SimpleNamespace(friction=0.5) for _ in range(count))
            shapes[actor] = shape_list
            ranges[actor] = range_list

    gym = SimpleNamespace(
        get_actor_dof_properties=lambda env, actor: dofs[actor].copy(),
        set_actor_dof_properties=lambda env, actor, props: dofs.__setitem__(
            actor, props.copy()
        ),
        get_actor_asset=lambda env, actor: actor,
        get_asset_rigid_body_names=lambda asset: tuple(
            worker.layout.entities[asset[1]].body_names
        ),
        get_actor_rigid_body_properties=lambda env, actor: copy.deepcopy(bodies[actor]),
        set_actor_rigid_body_properties=lambda env, actor, props: bodies.__setitem__(
            actor, copy.deepcopy(props)
        ),
        get_actor_rigid_shape_properties=lambda env, actor: copy.deepcopy(shapes[actor]),
        set_actor_rigid_shape_properties=lambda env, actor, props: shapes.__setitem__(
            actor, copy.deepcopy(props)
        ),
        get_actor_rigid_body_shape_indices=lambda env, actor: ranges[actor],
    )
    ctx.gym = gym
    ctx.gymapi = SimpleNamespace(ENV_SPACE=1, Vec3=lambda x, y, z: _vec3(x, y, z))
    ctx.env_handles = list(range(worker.num_envs))
    ctx.use_gpu_pipeline = False

    worker.records = []
    for env in range(worker.num_envs):
        records = []
        for index, entity in enumerate(worker.layout.entities):
            spec = worker.specs[index]
            records.append(
                {
                    "actor": (env, index),
                    "actor_id": 100 + 7 * env + index,
                    "source_id": spec["assignment"][env],
                    "dof_ids": (
                        (50 + env,)
                        if index == 0
                        else (80 + 2 * env,)
                        if index == 1
                        else ()
                    ),
                    "native_joint_names": [joint.name for joint in entity.joints],
                }
            )
        worker.records.append(records)
    worker.metadata = {
        "scene_entities_actual": [
            {"name": entity.name} for entity in worker.layout.entities
        ]
    }
    worker._bind_refresh_indices()
    return worker, SimpleNamespace(dofs=dofs, bodies=bodies, shapes=shapes)


def _stage(worker, request):
    slots = worker.ctx.slots
    prepared = prepare_scene_reset(
        worker.layout, request, slots["qpos"], slots["qvel"], slots["entity_root_state"]
    )
    count = len(prepared.env_ids)
    for name, values in (
        ("reset_env_ids", prepared.env_ids),
        ("reset_qpos", prepared.qpos),
        ("reset_qvel", prepared.qvel),
        ("reset_entity_root_state", prepared.roots),
    ):
        slots[name][:count] = values
    for name, values in (
        ("reset_qpos_mask", prepared.qpos_mask),
        ("reset_qvel_mask", prepared.qvel_mask),
        ("reset_root_mask", prepared.root_mask),
    ):
        slots[name][:] = values
    return {"count": count, "entity_names": list(prepared.entity_names)}


def _randomization(count=1):
    armature = np.zeros((count, 8), dtype=np.float32)
    armature[:, 0] = 0.1
    armature[:, 7] = 0.2
    frictionloss = np.zeros((count, 8), dtype=np.float32)
    frictionloss[:, 7] = 0.3
    return {
        "kp": np.full((count, 1), 33.0, dtype=np.float32).tolist(),
        "kd": np.full((count, 1), 3.0, dtype=np.float32).tolist(),
        "body_mass": np.full((count, 7), 2.5, dtype=np.float32).tolist(),
        "body_ipos": np.full((count, 7, 3), 0.01, dtype=np.float32).tolist(),
        "body_inertia": np.full((count, 7, 3), 0.02, dtype=np.float32).tolist(),
        "dof_armature": armature.tolist(),
        "dof_frictionloss": frictionloss.tolist(),
        "geom_friction": np.tile(
            np.array([0.7, 0.7, 0.0], dtype=np.float32), (count, 6, 1)
        ).tolist(),
    }


def test_reset_randomization_writes_and_reads_back_all_terms(tmp_path) -> None:
    worker, native = _worker(tmp_path)
    pose = np.array([[0, 0, 2, 1, 0, 0, 0.0]])
    payload = _stage(
        worker, SceneResetRequest((4,), (EntityStatePatch("object", root_pose=pose),))
    )
    payload["randomization"] = _randomization()
    reply = worker.reset(payload)

    robot = native.dofs[(4, 0)]
    assert robot["stiffness"][0] == pytest.approx(33.0)
    assert robot["damping"][0] == pytest.approx(3.0)
    assert robot["armature"][0] == pytest.approx(0.1)
    assert native.dofs[(4, 1)]["armature"][0] == pytest.approx(0.2)
    assert native.dofs[(4, 1)]["friction"][0] == pytest.approx(0.3)
    # Unselected environments keep construction values.
    assert native.dofs[(0, 0)]["stiffness"][0] == pytest.approx(20.0)
    for entity_index in range(4):
        for prop in native.bodies[(4, entity_index)]:
            assert prop.mass == pytest.approx(2.5)
            assert (prop.com.x, prop.com.y, prop.com.z) == pytest.approx((0.01, 0.01, 0.01))
        for shape in native.shapes[(4, entity_index)]:
            assert shape.friction == pytest.approx(0.7)
    assert native.bodies[(0, 0)][0].mass == pytest.approx(1.0)
    # COM caches follow the mutation so state refresh stays consistent.
    np.testing.assert_allclose(worker.body_com[4, 1:], 0.01, atol=1e-6)
    np.testing.assert_allclose(worker.root_com[4, 1], [0.01, 0.01, 0.01], atol=1e-6)
    np.testing.assert_allclose(worker.body_com[0], 0.0, atol=1e-6)

    records = {record["name"]: record for record in reply["native_entity_records"]}
    assert records["robot"]["dof_stiffness"][4] == [pytest.approx(33.0)]
    assert records["object"]["body_mass"][4] == [pytest.approx(2.5)] * 2
    assert records["table"]["geom_friction"][4] == [[pytest.approx(0.7)] * 2 + [0.0]]
    assert records["robot"]["body_mass"][0] == [1.0, 1.0]
    # The worker's own records and metadata follow the verified readback.
    assert worker.records[4][0]["dof_stiffness"] == [pytest.approx(33.0)]
    assert worker.records[4][0]["body_mass"] == [pytest.approx(2.5)] * 2
    actual = {entry["name"]: entry for entry in worker.metadata["scene_entities_actual"]}
    assert actual["object"]["body_mass"][4] == [pytest.approx(2.5)] * 2
    assert actual["object"]["dof_friction"][4] == [pytest.approx(0.3)]


def test_reset_without_randomization_returns_no_records(tmp_path) -> None:
    worker, _ = _worker(tmp_path)
    pose = np.array([[0, 0, 2, 1, 0, 0, 0.0]])
    payload = _stage(
        worker, SceneResetRequest((4,), (EntityStatePatch("object", root_pose=pose),))
    )
    assert worker.reset(payload) == {"timing": {}}


@pytest.mark.parametrize(
    ("term", "value"),
    [
        ("gravity", [[0.0, 0.0, -9.81]]),
        ("kp", [[-1.0]]),
        ("kp", [[1.0, 2.0]]),
        ("body_mass", [[0.0] * 7]),
        ("body_inertia", [[[0.0, 0.1, 0.1]] * 7]),
        ("dof_damping", [[0.0] * 8]),
        ("geom_friction", [[[0.1, 0.2, 0.0]] * 6]),
        ("geom_friction", [[[0.1, 0.1, 0.5]] * 6]),
    ],
)
def test_invalid_randomization_fails_before_native_mutation(tmp_path, term, value) -> None:
    worker, native = _worker(tmp_path)
    before = native.dofs[(4, 0)].copy()
    pose = np.array([[0, 0, 2, 1, 0, 0, 0.0]])
    payload = _stage(
        worker, SceneResetRequest((4,), (EntityStatePatch("object", root_pose=pose),))
    )
    payload["randomization"] = {term: value}
    with pytest.raises(ValueError):
        worker.reset(payload)
    np.testing.assert_array_equal(native.dofs[(4, 0)], before)
    assert native.bodies[(4, 0)][0].mass == pytest.approx(1.0)
    assert not worker.faulted


def test_randomization_root_dof_columns_fail_closed(tmp_path) -> None:
    worker, _ = _worker(tmp_path)
    pose = np.array([[0, 0, 2, 1, 0, 0, 0.0]])
    payload = _stage(
        worker, SceneResetRequest((4,), (EntityStatePatch("object", root_pose=pose),))
    )
    armature = np.zeros((1, 8), dtype=np.float32)
    armature[:, 3] = 0.5  # floating-root column of the object entity
    payload["randomization"] = {"dof_armature": armature.tolist()}
    with pytest.raises(ValueError, match="floating-root"):
        worker.reset(payload)
    assert not worker.faulted


def test_legacy_worker_rejects_randomization(tmp_path) -> None:
    worker, _ = _worker(tmp_path)
    worker.pending_body_fk = {}
    pose = np.array([[0, 0, 2, 1, 0, 0, 0.0]])
    payload = _stage(
        worker, SceneResetRequest((4,), (EntityStatePatch("object", root_pose=pose),))
    )
    payload["randomization"] = {"kp": [[30.0]]}
    with pytest.raises(ValueError, match="mapped scene"):
        worker.reset(payload)
    assert not worker.faulted


def test_readback_mismatch_faults_worker(tmp_path) -> None:
    worker, _ = _worker(tmp_path)
    # A native setter that silently drops writes must trip the readback audit.
    worker.ctx.gym.set_actor_dof_properties = lambda env, actor, props: None
    pose = np.array([[0, 0, 2, 1, 0, 0, 0.0]])
    payload = _stage(
        worker, SceneResetRequest((4,), (EntityStatePatch("object", root_pose=pose),))
    )
    payload["randomization"] = {"kp": [[33.0]]}
    with pytest.raises(RuntimeError, match="readback differs"):
        worker.reset(payload)
    assert worker.faulted


def test_gpu_pipeline_com_readback_substitution(tmp_path) -> None:
    worker, native = _worker(tmp_path)
    worker.ctx.use_gpu_pipeline = True

    def stale_set(env, actor, props):
        # Preview 4's GPU pipeline honors COM writes physically but keeps the
        # pre-write COM in the property readback; mass/inertia read back fine.
        kept = copy.deepcopy(native.bodies[actor])
        for new, old in zip(props, kept):
            old.mass = new.mass
            old.inertia = new.inertia
        native.bodies[actor] = kept
        return True

    worker.ctx.gym.set_actor_rigid_body_properties = stale_set
    pose = np.array([[0, 0, 2, 1, 0, 0, 0.0]])
    payload = _stage(
        worker, SceneResetRequest((4,), (EntityStatePatch("object", root_pose=pose),))
    )
    payload["randomization"] = _randomization()
    reply = worker.reset(payload)

    records = {record["name"]: record for record in reply["native_entity_records"]}
    assert records["object"]["body_ipos"][4] == [[pytest.approx(0.01)] * 3] * 2
    assert records["object"]["body_ipos"][0] == [[0.0] * 3] * 2
    # The committed COM caches follow the requested offsets (what PhysX
    # simulates with), not the stale GPU-pipeline readback.
    np.testing.assert_allclose(worker.body_com[4, 1:], 0.01, atol=1e-6)
    assert not worker.faulted


def _step_context(worker):
    ctx = worker.ctx
    per_env = 6  # robot 2 + object 2 + table 1 + target 1
    total = per_env * worker.num_envs
    ctx.device = "cpu"
    ctx.sim = object()
    ctx._body_state = np.zeros((total, 13), dtype=np.float32)
    worker.control_dofs[:, 0] = np.arange(worker.num_envs)
    for env in range(worker.num_envs):
        worker.body_ids[env] = -1
        worker.body_ids[env, [1, 2]] = env * per_env + np.array([0, 1])
        worker.body_ids[env, [3, 4]] = env * per_env + np.array([2, 3])
        worker.body_ids[env, 5] = env * per_env + 4
        worker.body_ids[env, 6] = env * per_env + 5

    class Tensor(np.ndarray):
        def to(self, device):
            return self

    ctx.torch = SimpleNamespace(
        from_numpy=lambda array: array.view(Tensor),
        as_tensor=lambda array, dtype=None, device=None: array,
    )
    ctx.gymtorch = SimpleNamespace(unwrap_tensor=lambda array: array)
    worker.targets = np.zeros(200, dtype=np.float32)
    calls = []
    ctx.gym = SimpleNamespace(
        set_dof_position_target_tensor=lambda sim, targets: True,
        apply_rigid_body_force_tensors=lambda sim, forces, torques, space: calls.append(
            (np.asarray(forces).copy(), np.asarray(torques).copy(), space)
        )
        or True,
        simulate=lambda sim: None,
        fetch_results=lambda sim, flag: None,
    )
    return calls


def test_step_scatters_body_wrench_into_env_space_and_reapplies(tmp_path) -> None:
    worker, _ = _worker(tmp_path)
    calls = _step_context(worker)
    wrench = np.zeros((worker.num_envs, worker.layout.nbody, 6), dtype=np.float32)
    wrench[2, 3] = [1, 2, 3, 4, 5, 6]  # object base, env 2
    wrench[0, 1] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.25]  # robot finger, env 0
    reply = worker.step({"nsteps": 2, "body_wrench": wrench.tobytes(order="C")})
    assert len(calls) == 2  # PhysX consumes forces per substep; both are armed.
    for forces, torques, space in calls:
        assert space == 1
        np.testing.assert_allclose(forces[2 * 6 + 2], [1, 2, 3])
        np.testing.assert_allclose(torques[2 * 6 + 2], [4, 5, 6])
        # Robot public bodies 1/2 are bound to env-local natives 0/1 (swapped).
        np.testing.assert_allclose(forces[0], [0.5, 0, 0])
        np.testing.assert_allclose(torques[0], [0, 0, 0.25])
        assert np.count_nonzero(forces) == 4
    assert reply["timing"]
    # The worker retains the Torch wrappers: unwrap_tensor only borrows the
    # storage pointer, so PhysX would read freed memory otherwise (#272).
    assert worker._wrench_torch is not None
    np.testing.assert_allclose(worker._wrench_torch[0][2 * 6 + 2], [1, 2, 3])


def test_step_rejects_malformed_wrench_without_stepping(tmp_path) -> None:
    worker, _ = _worker(tmp_path)
    calls = _step_context(worker)
    with pytest.raises(ValueError, match="C-order float32 bytes"):
        worker.step({"nsteps": 1, "body_wrench": b"\x00" * 8})
    assert calls == []
    assert not worker.faulted
    with pytest.raises(ValueError, match="NaN or Inf"):
        worker.step(
            {
                "nsteps": 1,
                "body_wrench": np.full(
                    (worker.num_envs, worker.layout.nbody, 6), np.nan, dtype=np.float32
                ).tobytes(order="C"),
            }
        )
    assert not worker.faulted


def test_step_wrench_native_failure_faults_worker(tmp_path) -> None:
    worker, _ = _worker(tmp_path)
    calls = _step_context(worker)
    worker.ctx.gym.apply_rigid_body_force_tensors = lambda *args: False
    wrench = np.zeros((worker.num_envs, worker.layout.nbody, 6), dtype=np.float32)
    wrench[1, 5] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="body wrench setter failed"):
        worker.step({"nsteps": 1, "body_wrench": wrench.tobytes(order="C")})
    assert worker.faulted
    assert calls == []
