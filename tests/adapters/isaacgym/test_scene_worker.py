"""Native-worker mapping tests without importing IsaacGym or Torch."""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.adapters.isaacgym.scene_fixture import scene_payload
from unisim.backend.isaacgym.scene_worker import SceneWorker
from unisim.backend.subprocess_ipc import protocol
from unisim.entities import EntityStatePatch, SceneResetRequest
from unisim.entity_state import prepare_scene_reset

pytest.importorskip("mujoco")


def _worker(tmp_path):
    payload = scene_payload(tmp_path)
    ctx = SimpleNamespace(protocol=protocol)
    worker = SceneWorker(ctx, payload)
    ctx.slots = {
        name: np.zeros(shape, dtype=protocol.slot_dtype(name))
        for name, shape in protocol.scene_slot_shapes(worker.num_envs, worker.layout).items()
    }
    ctx.slots["qpos"][:] = worker.qpos0
    ctx.slots["qvel"][:] = worker.qvel0
    ctx.slots["entity_root_state"][:] = worker.roots0
    worker.root_com[:, 1] = [0.05, 0, 0]
    # Deliberately permuted native addresses catch accidental env/actor arithmetic.
    worker.records = [
        [
            {
                "actor_id": 100 + 7 * env + index,
                "dof_ids": ((50 + env,) if index == 0 else (80 + 2 * env,) if index == 1 else ()),
            }
            for index in range(4)
        ]
        for env in range(5)
    ]
    worker.refresh = lambda: None
    calls = []
    worker._submit_pending = lambda: calls.append(
        {k: v.copy() for k, v in worker.pending_roots.items()}
    )
    return worker, calls


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


def test_worker_source_parses_with_python38_and_imports_without_sdk() -> None:
    import unisim.backend.isaacgym.scene_worker as module

    path = Path(module.__file__)
    ast.parse(path.read_text(), feature_version=(3, 8))
    spec = importlib.util.spec_from_file_location("isolated_gym_scene", path)
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    assert hasattr(isolated, "SceneWorker")


def test_com_velocity_conversion_has_independent_ninety_degree_oracle(tmp_path) -> None:
    worker, _ = _worker(tmp_path)
    # Local COM x rotated by Rz90 lies on world y; omega z cross COM = world -x.
    row = np.array([1, 2, 3, np.sqrt(0.5), 0, 0, np.sqrt(0.5), 2, 3, 4, 0, 0, 2.0])
    native = worker._native_root(row, np.array([0.1, 0, 0]))
    np.testing.assert_allclose(native[7:10], [1.8, 3, 4], atol=1e-7)
    np.testing.assert_allclose(native[3:7], [0, 0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-7)
    # Read-side oracle separately starts at native COM velocity, not write output.
    actual = worker._public_state(
        np.array([1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5), 1.8, 3, 4, 0, 0, 2.0]),
        np.array([0.1, 0, 0]),
    )
    np.testing.assert_allclose(actual[7:10], [2, 3, 4], atol=1e-7)


def test_repeated_resets_submit_union_of_actual_actor_ids(tmp_path) -> None:
    worker, calls = _worker(tmp_path)
    pose = np.array([[0, 0, 2, 1, 0, 0, 0.0]])
    first = _stage(worker, SceneResetRequest((4,), (EntityStatePatch("object", root_pose=pose),)))
    worker.reset(first)
    second = _stage(worker, SceneResetRequest((1,), (EntityStatePatch("target", root_pose=pose),)))
    worker.reset(second)
    assert set(calls[0]) == {129}
    assert set(calls[1]) == {129, 110}
    assert len(calls) == 2
    np.testing.assert_allclose(calls[1][129][:3], [0, 0, 2])


def test_joint_position_only_preserves_velocity_and_unselected_control(tmp_path) -> None:
    worker, _ = _worker(tmp_path)
    worker.ctx.slots["qvel"][4, 0] = 0.7
    worker.ctx.slots["ctrl"][:] = 0.9
    payload = _stage(
        worker,
        SceneResetRequest((4,), (EntityStatePatch("robot", joint_positions=np.array([[0.3]])),)),
    )
    worker.ctx.slots["reset_qvel"][0, 0] = 99  # An unselected field is not a write.
    worker.reset(payload)
    np.testing.assert_allclose(worker.pending_dofs[54], [0.3, 0.7])
    assert worker.pending_dof_actors == {128}
    np.testing.assert_allclose(worker.ctx.slots["ctrl"][:, 0], [0.9, 0.9, 0.9, 0.9, 0.3])


@pytest.mark.parametrize("tamper", ["unselected_entity", "root_masks", "nan", "fixed", "bad_env"])
def test_bad_reset_does_not_mutate_pending_union_or_call_native(tmp_path, tamper) -> None:
    worker, calls = _worker(tmp_path)
    worker.pending_roots[900] = np.arange(13, dtype=np.float32)
    original = worker.pending_roots[900].copy()
    payload = _stage(
        worker,
        SceneResetRequest(
            (4,), (EntityStatePatch("object", root_pose=np.array([[0, 0, 2, 1, 0, 0, 0.0]])),)
        ),
    )
    if tamper == "unselected_entity":
        worker.ctx.slots["reset_qpos_mask"][0] = 1
    elif tamper == "root_masks":
        worker.ctx.slots["reset_root_mask"][1, 0] = 0
    elif tamper == "nan":
        worker.ctx.slots["reset_qpos"][0, 2] = np.nan
    elif tamper == "fixed":
        payload["entity_names"].append("table")
        worker.ctx.slots["reset_root_mask"][2, 0] = 1
    else:
        worker.ctx.slots["reset_env_ids"][0] = 5
    with pytest.raises(ValueError):
        worker.reset(payload)
    assert calls == []
    assert set(worker.pending_roots) == {900}
    np.testing.assert_array_equal(worker.pending_roots[900], original)
    assert not worker.faulted


@pytest.mark.parametrize(
    "tamper", ["bool_assignment", "missing_variant", "mirror_identity", "general"]
)
def test_init_rejects_bad_identity_and_unsafe_importer_inputs_before_sdk(tmp_path, tamper) -> None:
    payload = scene_payload(tmp_path)
    if tamper == "bool_assignment":
        payload["scene_entities"][1]["assignment"][0] = True
    elif tamper == "missing_variant":
        payload["scene_entities"][1]["variants"].pop()
    elif tamper == "mirror_identity":
        payload["scene_entities"][3]["assignment"][0] = 0
    else:
        path = Path(payload["scene_entities"][0]["sources"][0])
        path.write_text(
            path.read_text().replace("</mujoco>", "<actuator><general/></actuator></mujoco>")
        )
    with pytest.raises(ValueError):
        SceneWorker(SimpleNamespace(protocol=protocol), payload)


def test_passive_joint_has_no_drive_while_names_are_native_reordered(tmp_path) -> None:
    worker, _ = _worker(tmp_path)
    worker.ctx.gymapi = SimpleNamespace(DOF_MODE_POS=1, DOF_MODE_NONE=0)
    props = np.zeros(
        1,
        dtype=[
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
        ],
    )
    variant = copy.deepcopy(worker.specs[1]["variants"][0])
    variant.update(dof_stiffness=[999], dof_damping=[999], dof_effort=[999])
    worker._apply_drives(props, worker.layout.entities[1], variant, ("passive",))
    assert props["driveMode"].tolist() == [0]
    assert (
        props["stiffness"].tolist() == props["damping"].tolist() == props["effort"].tolist() == [0]
    )


def test_initial_ctrl_is_distinct_from_joint_position_and_maps_by_env_and_action(tmp_path) -> None:
    payload = scene_payload(tmp_path)
    payload["initial_qpos"][4][0] = 0.91
    payload["initial_ctrl"] = [[0.11], [0.22], [0.33], [0.44], [0.55]]
    worker, _ = _worker(tmp_path)
    worker.initial_ctrl = np.asarray(payload["initial_ctrl"])
    worker.qpos0[4, 0] = 0.91
    assert worker.initial_ctrl is not None
    worker.control_dofs = np.asarray([[52], [50], [54], [51], [53]], dtype=np.int64)
    worker.targets = np.zeros(100)
    worker._stage_initial()
    np.testing.assert_allclose(worker.targets[[52, 50, 54, 51, 53]], [0.11, 0.22, 0.33, 0.44, 0.55])
    assert worker.pending_dofs[54][0] == pytest.approx(0.91)


@pytest.mark.parametrize("value", [[[0.1]], [[np.nan]] * 5, [[True]] * 5])
def test_initial_ctrl_shape_and_finite_values_are_validated(tmp_path, value) -> None:
    payload = scene_payload(tmp_path)
    payload["initial_ctrl"] = value
    with pytest.raises(ValueError, match="initial_ctrl"):
        SceneWorker(SimpleNamespace(protocol=protocol), payload)


def test_missing_initial_ctrl_falls_back_to_bound_joint_positions(tmp_path) -> None:
    worker, _ = _worker(tmp_path)
    worker.initial_ctrl = None
    worker.qpos0[:, 0] = [.11, .22, .33, .44, .55]
    worker.control_dofs = np.arange(50, 55).reshape(5, 1)
    worker.targets = np.zeros(100)
    worker._stage_initial()
    np.testing.assert_allclose(worker.initial_ctrl[:, 0], [.11, .22, .33, .44, .55])
    np.testing.assert_allclose(worker.targets[50:55], [.11, .22, .33, .44, .55])


def test_native_commit_failure_faults_worker_before_later_commands(tmp_path) -> None:
    worker, _ = _worker(tmp_path)

    class Tensor(np.ndarray):
        def to(self, device):
            return self

        def long(self):
            return self.astype(np.int64)

    ctx = worker.ctx
    ctx.device = "cpu"
    ctx._root_state = np.zeros((200, 13), dtype=np.float32)
    ctx.torch = SimpleNamespace(from_numpy=lambda array: array.view(Tensor))
    ctx.gymtorch = SimpleNamespace(unwrap_tensor=lambda array: array)
    ctx.gym = SimpleNamespace(set_actor_root_state_tensor_indexed=lambda *args: False)
    ctx.sim = object()
    worker.pending_roots[129] = np.zeros(13, dtype=np.float32)
    with pytest.raises(RuntimeError, match="native root setter failed"):
        SceneWorker._submit_pending(worker)
    assert worker.faulted
    with pytest.raises(RuntimeError, match="faulted"):
        worker.reset({})
    with pytest.raises(RuntimeError, match="faulted"):
        worker.step({"nsteps": 1})
