"""SDK-free lifecycle checks for mapped IsaacSim body wrenches."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from test_mapped_scene import _payload

from unisim.backend.base import PreStepControlOutput
from unisim.backend.isaacsim.backend import IsaacSimBackend
from unisim.backend.isaacsim.scene_worker import SceneWorkerContext
from unisim.backend.subprocess_ipc import protocol
from unisim.dr.interval import INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE
from unisim.dr.types import IntervalRandomizationPlan
from unisim.entities import EntityStatePatch, SceneResetRequest
from unisim.scene_layout import CompiledSceneLayout


def _layout() -> CompiledSceneLayout:
    return protocol.load_scene_layout(_payload()["scene_layout"])


def _host() -> IsaacSimBackend:
    layout = _layout()
    owner = IsaacSimBackend.__new__(IsaacSimBackend)
    owner._num_envs = 2
    owner._entity_scene = SimpleNamespace(layout=layout)
    owner._entity_scene.control_lower = np.empty((0,), dtype=np.float32)  # type: ignore[attr-defined]
    owner._entity_scene.control_upper = np.empty((0,), dtype=np.float32)  # type: ignore[attr-defined]
    owner._staged_body_wrench = np.zeros((2, layout.nbody, 6), dtype=np.float32)
    owner._body_wrench_pending = False
    owner._closed = False
    owner._worker_dead_error = None
    owner._model_info = object()
    owner._slots = {
        name: np.zeros(shape, dtype=protocol.slot_dtype(name))
        for name, shape in protocol.scene_slot_shapes(2, layout).items()
    }
    owner._stale_body_ids = set()
    return owner


def test_direct_submissions_accumulate_and_bad_values_do_not_partially_mutate():
    owner = _host()
    force = np.full((2, 1, 3), 2.0, dtype=np.float32)
    owner.apply_body_force(np.asarray([2]), force)
    owner.apply_body_force(np.asarray([2]), force, torque=np.full((2, 1, 3), 3.0))
    np.testing.assert_allclose(owner._staged_body_wrench[:, 2, 0:3], 4.0)
    np.testing.assert_allclose(owner._staged_body_wrench[:, 2, 3:6], 3.0)
    assert owner._body_wrench_pending

    before = owner._staged_body_wrench.copy()
    with pytest.raises(ValueError, match="body force"):
        owner.apply_body_force(np.asarray([2]), np.ones((1, 1, 3)))
    with pytest.raises(ValueError, match="NaN or Inf"):
        owner.apply_body_force(np.asarray([2]), np.full((2, 1, 3), np.nan))
    with pytest.raises(ValueError, match="body ids"):
        owner.apply_body_force(np.asarray([3]), np.zeros((2, 1, 3)))
    np.testing.assert_array_equal(owner._staged_body_wrench, before)


def test_step_payload_is_detached_and_consumed_after_success():
    owner = _host()
    owner.apply_body_force(np.asarray([1]), np.full((2, 1, 3), 5.0))
    payload = owner._step_payload(4)
    assert payload["nsteps"] == 4
    assert isinstance(payload["body_wrench"], bytes)
    owner._after_step(payload)
    assert not np.any(owner._staged_body_wrench)
    assert "body_wrench" not in owner._step_payload(1)
    assert not owner._body_wrench_pending


def test_step_sends_one_command_and_consumes_wrench():
    owner = _host()
    owner._entity_scene.control_lower = np.empty((0,), dtype=np.float32)  # type: ignore[attr-defined]
    owner._entity_scene.control_upper = np.empty((0,), dtype=np.float32)  # type: ignore[attr-defined]
    owner.apply_body_force(np.asarray([1]), np.ones((2, 1, 3)))
    commands: list = []
    owner._request = lambda cmd, payload, **kwargs: commands.append((cmd, payload))  # type: ignore[method-assign]
    owner.step(np.empty((2, 0), dtype=np.float32), nsteps=3)
    assert len(commands) == 1
    assert commands[0][0] == protocol.CMD_STEP
    assert commands[0][1]["nsteps"] == 3
    assert "body_wrench" in commands[0][1]
    assert not np.any(owner._staged_body_wrench)
    assert "body_wrench" not in owner._step_payload(1)


def test_pre_step_control_runs_at_each_refreshed_worker_substep_boundary():
    owner = _host()
    commands: list = []
    observed_states: list[np.ndarray] = []
    callback_indices: list[int] = []

    def request(cmd, payload, **kwargs):
        commands.append((cmd, payload))
        # Stand in for the worker's post-substep shared-state refresh. The
        # next callback must observe this value before its worker command.
        owner._slots["qvel"][:, 0] += 1.0
        return {"timing": {"physics_ms": 1.0}}

    owner._request = request  # type: ignore[method-assign]
    owner._staged_body_wrench[:, 1, 2] = 2.0
    owner._body_wrench_pending = True

    def callback(owner_, ctrl):
        callback_indices.append(len(callback_indices))
        observed_states.append(owner_._slots["qvel"][:, 0].copy())
        force = np.zeros((2, 1, 3), dtype=np.float32)
        force[:, 0, 0] = 10.0 + len(callback_indices)
        return PreStepControlOutput(ctrl=ctrl, body_ids=np.asarray([2]), force=force)

    owner.set_pre_step_control(callback)
    owner.step(np.empty((2, 0), dtype=np.float32), nsteps=3)

    assert len(commands) == 3
    assert all(command[0] == protocol.CMD_STEP for command in commands)
    assert all(command[1]["nsteps"] == 1 for command in commands)
    assert callback_indices == [0, 1, 2]
    np.testing.assert_allclose(
        observed_states, [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], atol=0.0
    )
    for index, (_, payload) in enumerate(commands):
        wrench = np.frombuffer(payload["body_wrench"], dtype=np.float32).reshape(2, 3, 6)
        np.testing.assert_allclose(wrench[:, 1, 2], 2.0)
        np.testing.assert_allclose(wrench[:, 2, 0], 11.0 + index)
        np.testing.assert_allclose(wrench[:, 0], 0.0)
    assert not np.any(owner._staged_body_wrench)
    assert not owner._body_wrench_pending


def test_failed_pre_step_callback_clears_staged_interval_wrench():
    owner = _host()
    owner._staged_body_wrench[:, 2, 5] = 3.0
    owner._body_wrench_pending = True

    def interrupted(owner_, ctrl):
        raise RuntimeError("callback failed")

    owner.set_pre_step_control(interrupted)
    with pytest.raises(RuntimeError, match="callback failed"):
        owner.step(np.empty((2, 0), dtype=np.float32), nsteps=2)
    assert not np.any(owner._staged_body_wrench)
    assert not owner._body_wrench_pending


def test_pre_step_callback_cannot_stage_wrench_directly():
    owner = _host()

    def stages_directly(owner_, ctrl):
        owner_.apply_body_force(np.asarray([2]), np.ones((2, 1, 3)))
        return ctrl

    owner.set_pre_step_control(stages_directly)
    with pytest.raises(RuntimeError, match="apply_body_force must not be called"):
        owner.step(np.empty((2, 0), dtype=np.float32), nsteps=1)
    assert not np.any(owner._staged_body_wrench)


def test_failed_step_retains_staged_wrench():
    owner = _host()
    owner._entity_scene.control_lower = np.empty((0,), dtype=np.float32)  # type: ignore[attr-defined]
    owner._entity_scene.control_upper = np.empty((0,), dtype=np.float32)  # type: ignore[attr-defined]
    owner.apply_body_force(np.asarray([1]), np.full((2, 1, 3), 6.0))
    before = owner._staged_body_wrench.copy()

    def fail(cmd, payload, **kwargs):
        raise RuntimeError("worker step failed")

    owner._request = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="worker step failed"):
        owner.step(np.empty((2, 0), dtype=np.float32), nsteps=1)
    np.testing.assert_array_equal(owner._staged_body_wrench, before)
    assert owner._body_wrench_pending


def test_new_interval_plan_replaces_unconsumed_direct_wrench():
    owner = _host()
    owner.apply_body_force(np.asarray([2]), np.ones((2, 1, 3)))
    from unisim.dr.interval import IntervalTermOp

    plan = IntervalRandomizationPlan(
        ops=(
            IntervalTermOp(
                INTERVAL_TERM_BODY_FORCE,
                np.full((2, 1, 3), 7.0, dtype=np.float32),
                body_ids=np.asarray([2]),
            ),
        )
    )
    owner.apply_interval_randomization(plan)
    np.testing.assert_allclose(owner._staged_body_wrench[:, 2, 0:3], 7.0)
    assert owner.get_dr_capabilities().supports_interval_term(INTERVAL_TERM_BODY_FORCE)
    assert owner.get_dr_capabilities().supports_interval_term(INTERVAL_TERM_BODY_TORQUE)


def test_legacy_execution_remains_unsupported_and_callback_is_rejected():
    owner = _host()
    owner._entity_scene = None
    owner._staged_body_wrench = None
    with pytest.raises(NotImplementedError, match="does not support interval body force"):
        owner.apply_body_force(np.asarray([0]), np.zeros((2, 1, 3)))
    with pytest.raises(NotImplementedError, match="host pre-step callbacks"):
        owner.set_pre_step_control(lambda backend, ctrl: ctrl)
    assert not owner.get_dr_capabilities().supported_interval_terms


def _reset_owner(commands: list, *, fail: bool = False) -> IsaacSimBackend:
    owner = _host()

    def request(cmd, payload, **kwargs):
        commands.append((cmd, payload))
        if fail:
            raise RuntimeError("worker reset failed")
        return None

    owner._request = request  # type: ignore[method-assign]
    return owner


def _full_reset_owner(commands: list) -> IsaacSimBackend:
    owner = _reset_owner(commands)
    layout = owner._entity_scene.layout  # type: ignore[attr-defined]
    owner._entity_scene.qpos = np.zeros((2, layout.nq), dtype=np.float32)  # type: ignore[attr-defined]
    owner._entity_scene.qpos[:, 4] = 1.0  # type: ignore[attr-defined]
    owner._entity_scene.qvel = np.zeros((2, layout.nv), dtype=np.float32)  # type: ignore[attr-defined]
    owner._entity_scene.roots = np.zeros((2, len(layout.entities), 13), dtype=np.float32)  # type: ignore[attr-defined]
    owner._entity_scene.roots[:, :, 3] = 1.0  # type: ignore[attr-defined]
    owner._entity_scene.payload = {"initial_ctrl": np.zeros((2, 0), dtype=np.float32)}  # type: ignore[attr-defined]
    return owner


def _object_reset() -> SceneResetRequest:
    return SceneResetRequest(
        (1,),
        (
            EntityStatePatch(
                "object", root_pose=np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]], np.float32)
            ),
        ),
    )


def test_entity_reset_clears_only_selected_environment_and_entity_rows():
    commands: list = []
    owner = _reset_owner(commands)
    owner._staged_body_wrench[:] = 1.0
    owner._body_wrench_pending = True
    owner.reset_entities(_object_reset())
    assert commands
    np.testing.assert_array_equal(owner._staged_body_wrench[0], 1.0)
    np.testing.assert_array_equal(owner._staged_body_wrench[1, :2], 1.0)
    np.testing.assert_array_equal(owner._staged_body_wrench[1, 2], 0.0)
    assert owner._body_wrench_pending


def test_failed_entity_reset_retains_staged_wrench():
    commands: list = []
    owner = _reset_owner(commands, fail=True)
    owner._staged_body_wrench[:] = 2.0
    owner._body_wrench_pending = True
    with pytest.raises(RuntimeError, match="worker reset failed"):
        owner.reset_entities(_object_reset())
    assert np.all(owner._staged_body_wrench == 2.0)


def test_full_reset_clears_every_body_in_selected_rows():
    commands: list = []
    owner = _full_reset_owner(commands)
    owner._staged_body_wrench[:] = 3.0
    owner._body_wrench_pending = True
    owner.reset(np.asarray([1], dtype=np.int32))
    assert commands
    np.testing.assert_array_equal(owner._staged_body_wrench[0], 3.0)
    assert not np.any(owner._staged_body_wrench[1])
    assert owner._body_wrench_pending


def _wrench_worker() -> tuple[SceneWorkerContext, list[dict]]:
    ctx = SceneWorkerContext.__new__(SceneWorkerContext)
    ctx.layout = _layout()
    ctx.num_envs = 2
    ctx.sim_dt = 0.002
    ctx.faulted = False
    ctx.device = "cpu"
    ctx.torch = SimpleNamespace(
        float32=np.float32,
        long=np.int64,
        as_tensor=lambda values, dtype=None, device=None: np.asarray(values, dtype=dtype),
        empty=lambda size, dtype=None, device=None: np.empty(size, dtype=dtype),
        zeros=lambda size, dtype=None, device=None: np.zeros(size, dtype=dtype),
    )
    ctx.slots = {"ctrl": np.zeros((2, 0), dtype=np.float32)}
    ctx._set_control_targets = lambda control: None
    ctx.refresh_state_slots = lambda: None
    operations: list[dict] = []

    def asset(name: str):
        return SimpleNamespace(
            num_bodies=2 if name == "robot" else 1,
            set_external_force_and_torque=lambda force, torque, **kwargs: operations.append(
                {"entity": name, "force": np.asarray(force), "torque": np.asarray(torque), **kwargs}
            ),
            write_data_to_sim=lambda: operations.append({"entity": name, "write": True}),
            update=lambda dt: None,
        )

    ctx.assets = [asset("robot"), asset("object")]
    ctx.maps = [
        {"public_for_native": np.asarray([1, 0]), "bodies": np.asarray([1, 0])},
        {"public_for_native": np.asarray([0, 1]), "bodies": np.asarray([0])},
    ]
    ctx.contact_sensors = []
    ctx.contact_sensor_maps = []
    ctx.contact_force_sensors = []
    ctx.sim = SimpleNamespace(step=lambda render=False: None)
    return ctx, operations


def test_worker_maps_public_rows_and_bodies_and_clears_after_all_substeps():
    ctx, operations = _wrench_worker()
    wrench = np.zeros((2, 3, 6), dtype=np.float32)
    wrench[:, :, 0:2] = np.asarray(
        [
            [[10.0, 20.0], [30.0, 40.0], [0.0, 0.0]],
            [[11.0, 21.0], [31.0, 41.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    wrench[:, 2, 5] = [30.0, 50.0]
    ctx.step({"nsteps": 2, "body_wrench": wrench.tobytes(order="C")})
    stages = [item for item in operations if "force" in item]
    assert len(stages) == 4
    assert all(stage["is_global"] is True for stage in stages[:2])
    assert all("positions" not in stage for stage in stages[:2])
    np.testing.assert_allclose(stages[0]["force"][:, :, 0], [[11.0, 31.0], [10.0, 30.0]])
    np.testing.assert_array_equal(stages[0]["body_ids"], [1, 0])
    np.testing.assert_allclose(stages[1]["torque"][:, 0, 2], [30.0, 50.0])
    assert [stage["force"].shape for stage in stages[2:]] == [(2, 2, 3), (2, 1, 3)]
    assert [stage["torque"].shape for stage in stages[2:]] == [(2, 2, 3), (2, 1, 3)]
    assert len([item for item in operations if item.get("write")]) == 4
    assert not ctx.faulted


def test_worker_fault_cleans_staged_native_wrench():
    ctx, operations = _wrench_worker()

    def fail(render=False):
        raise RuntimeError("physics failed")

    ctx.sim.step = fail  # type: ignore[method-assign]
    wrench = np.ones((2, 3, 6), dtype=np.float32)
    with pytest.raises(RuntimeError, match="physics failed"):
        ctx.step({"nsteps": 1, "body_wrench": wrench.tobytes(order="C")})
    assert ctx.faulted
    assert len([item for item in operations if "force" in item]) == 4


def test_worker_rejects_malformed_wrench_before_native_writes():
    ctx, operations = _wrench_worker()
    with pytest.raises(ValueError, match="body wrench"):
        ctx.step({"nsteps": 1, "body_wrench": np.ones((1, 3, 6), dtype=np.float32).tobytes()})
    with pytest.raises(ValueError, match="NaN or Inf"):
        malformed = np.full((2, 3, 6), np.nan, dtype=np.float32)
        ctx.step({"nsteps": 1, "body_wrench": malformed.tobytes()})
    assert operations == []
    assert not ctx.faulted
