"""One native submit path preserves whole-world and selected-entity intents."""

from pathlib import Path

import numpy as np
import pytest

from unisim import create_backend
from unisim.dr.types import ResetRandomizationPayload
from unisim.entities import EntityStatePatch, SceneResetRequest
from unisim.scene import SceneCfg

from .test_entity_runtime import _scene


def _legacy(tmp_path: Path):
    # Valid legacy topology intentionally outside SceneEntitySpec v1: a
    # world-level geom, top-level hinge, anonymous body, tendon actuator.
    source = tmp_path / "legacy.xml"
    source.write_text("""<mujoco><option gravity="0 0 0"/>
      <worldbody><geom type="plane" size="1 1 .1"/>
        <body name="rotor" pos="0 0 2"><joint name="hinge"/>
          <geom size=".1" mass="1"/><body pos="0 0 .3"><geom size=".03" mass=".2"/></body>
        </body>
        <body name="free" pos="2 0 2"><freejoint/><geom size=".1" mass="1"/></body>
        <body name="target" mocap="true" pos="3 0 2"><geom size=".03"/></body>
      </worldbody><tendon><fixed name="transmission">
        <joint joint="hinge" coef="1"/></fixed></tendon>
      <actuator><general name="drive" tendon="transmission" dyntype="filter" dynprm=".1"/>
      </actuator>
      <sensor><jointpos name="joint_position" joint="hinge"/></sensor></mujoco>""")
    backend = create_backend(
        "mujoco", SceneCfg(model_file=str(source)), num_envs=3, sim_dt=0.002, np_dtype=np.float64
    )
    backend.materialize()
    return backend


class _CountingPool:
    def __init__(self, pool, *, fail=None):
        self.pool = pool
        self.calls = []
        self.fail = fail

    def __getattr__(self, name):
        return getattr(self.pool, name)

    def reset(self, ids):
        self.calls.append(("reset", ids.copy()))
        if self.fail == "reset":
            raise RuntimeError("native reset failed")
        return self.pool.reset(ids)

    def forward(self, ids):
        self.calls.append(("forward", ids.copy()))
        if self.fail == "forward":
            raise RuntimeError("native forward failed")
        return self.pool.forward(ids)

    def expand(self, name):
        self.calls.append(("expand", name))
        return self.pool.expand(name)

    def set_const(self, ids):
        self.calls.append(("set_const", ids.copy()))
        return self.pool.set_const(ids)


def test_patch_full_set_and_default_reset_share_one_commit(tmp_path, monkeypatch):
    backend = create_backend(
        "mujoco",
        _scene(tmp_path, n=2, variants=False),
        num_envs=2,
        sim_dt=0.002,
        np_dtype=np.float64,
    )
    try:
        backend.materialize()
        native = _CountingPool(backend._pool)
        backend._pool = native
        plans = []
        commit = backend._commit_state

        def track(plan):
            plans.append(plan)
            return commit(plan)

        monkeypatch.setattr(backend, "_commit_state", track)
        backend._ctrl_view[:] = 0.8
        backend._act_view[:] = 0.2
        backend.reset_entities(
            SceneResetRequest(
                (1,), (EntityStatePatch("object", joint_positions=np.array([[0.7]])),)
            )
        )
        assert len(plans) == 1 and not plans[-1].reset_world
        assert [name for name, _ in native.calls] == ["forward"]
        np.testing.assert_array_equal(backend._ctrl_view, 0.8)
        np.testing.assert_array_equal(backend._act_view, 0.2)
        native.calls.clear()
        snapshot = backend.get_state()
        backend.set_state(np.array([1]), snapshot["qpos"][[1]], snapshot["qvel"][[1]])
        assert len(plans) == 2 and plans[-1].reset_world
        assert [name for name, _ in native.calls] == ["reset", "forward"]
        np.testing.assert_array_equal(backend._ctrl_view[:, 0], [0.8, 0])
        np.testing.assert_array_equal(backend._act_view[:, 0], [0.2, 0])
        backend.reset(np.array([0]))
        assert len(plans) == 3 and plans[-1].reset_world and plans[-1].defaults
    finally:
        backend.close()


@pytest.mark.parametrize(
    "problem",
    [
        "shape",
        "nan",
        "overflow",
        "negative_row",
        "large_row",
        "duplicate_row",
        "float_row",
        "bool_row",
        "late_dr_shape",
        "late_dr_nan",
        "dr_object",
        "late_base_offset",
    ],
)
def test_all_legacy_inputs_validate_before_any_native_write(tmp_path, problem):
    backend = _legacy(tmp_path)
    try:
        backend.step(np.ones((3, 1)))
        snapshot = backend.get_state()
        ctrl, act, time = (
            backend._ctrl_view.copy(),
            backend._act_view.copy(),
            backend._time_view.copy(),
        )
        pending = backend._pending_xfrc_applied.copy()
        mass = backend._pool.expand("body_mass").copy()
        native = _CountingPool(backend._pool)
        backend._pool = native
        ids = np.array([2, 0])
        qpos, qvel = snapshot["qpos"][ids], snapshot["qvel"][ids]
        randomization = None
        if problem == "shape":
            qpos = qpos[:, :-1]
        elif problem == "nan":
            qvel[0, 0] = np.nan
        elif problem == "overflow":
            # macOS/Windows longdouble may be float64, so exercise narrowing
            # to float32 explicitly instead of assuming an extended exponent.
            backend._np_dtype = np.float32
            qvel = qvel.astype(np.float64)
            qvel[0, 0] = np.finfo(np.float64).max
        elif problem == "negative_row":
            ids = np.array([-1, 0])
        elif problem == "large_row":
            ids = np.array([3, 0])
        elif problem == "duplicate_row":
            ids = np.array([0, 0])
        elif problem == "float_row":
            ids = np.array([1.0, 0.0])
        elif problem == "bool_row":
            ids = np.array([True, False])
        elif problem == "late_dr_shape":
            randomization = ResetRandomizationPayload(
                body_mass=mass[[2, 0]] * 2, kd=np.zeros((2, 9))
            )
        elif problem == "late_dr_nan":
            randomization = ResetRandomizationPayload(
                body_mass=mass[[2, 0]] * 2, kd=np.full((2, 1), np.nan)
            )
        elif problem == "dr_object":
            randomization = object()
        elif problem == "late_base_offset":
            randomization = ResetRandomizationPayload(
                body_mass=mass[[2, 0]] * 2, base_com_offset=np.zeros((2, 7))
            )
        with pytest.raises((ValueError, TypeError)):
            backend.set_state(ids, qpos, qvel, randomization)
        assert native.calls == []
        for name in snapshot:
            np.testing.assert_array_equal(backend.get_state()[name], snapshot[name])
        np.testing.assert_array_equal(backend._ctrl_view, ctrl)
        np.testing.assert_array_equal(backend._act_view, act)
        np.testing.assert_array_equal(backend._time_view, time)
        np.testing.assert_array_equal(backend._pending_xfrc_applied, pending)
        np.testing.assert_array_equal(native.pool.expand("body_mass"), mass)
        assert not backend._entity_faulted
    finally:
        backend.close()


def test_legacy_complex_model_whole_reset_native_semantics_and_order(tmp_path):
    backend = _legacy(tmp_path)
    try:
        assert backend._entity_layout is None
        native = _CountingPool(backend._pool)
        backend._pool = native
        backend.step(np.ones((3, 1)), nsteps=3)
        backend._xfrc_view[:] = 0.2
        backend._warm_view[:] = 0.3
        before = backend.get_state()
        ctrl, act, time = (
            backend._ctrl_view.copy(),
            backend._act_view.copy(),
            backend._time_view.copy(),
        )
        qpos, qvel = before["qpos"][[2, 0]].copy(), before["qvel"][[2, 0]].copy()
        qpos[:, 0] = [0.6, 0.9]
        qvel[:, 0] = [0.2, 0.5]
        backend.set_state(np.array([2, 0]), qpos, qvel)
        np.testing.assert_array_equal(backend.get_state()["qpos"][[2, 0]], qpos)
        np.testing.assert_array_equal(backend.get_state()["qvel"][[2, 0]], qvel)
        np.testing.assert_array_equal(backend._ctrl_view[[2, 0]], 0)
        np.testing.assert_array_equal(backend._act_view[[2, 0]], 0)
        np.testing.assert_array_equal(backend._time_view[[2, 0]], 0)
        np.testing.assert_array_equal(backend._xfrc_view[[2, 0]], 0)
        np.testing.assert_array_equal(backend._warm_view[[2, 0]], 0)
        np.testing.assert_array_equal(backend.get_state()["qpos"][1], before["qpos"][1])
        np.testing.assert_array_equal(backend._ctrl_view[1], ctrl[1])
        np.testing.assert_array_equal(backend._act_view[1], act[1])
        np.testing.assert_array_equal(backend._time_view[1], time[1])
        assert [name for name, _ in native.calls] == ["reset", "forward"]
        np.testing.assert_array_equal(native.calls[0][1], [0, 2])
    finally:
        backend.close()


@pytest.mark.parametrize("failure", ["reset", "forward"])
def test_legacy_native_failure_faults_all_state_consumption(tmp_path, failure):
    backend = _legacy(tmp_path)
    try:
        snapshot = backend.get_state()
        backend._pool = _CountingPool(backend._pool, fail=failure)
        with pytest.raises(RuntimeError, match="native"):
            backend.set_state(np.array([0]), snapshot["qpos"][[0]], snapshot["qvel"][[0]])
        for call in (
            backend.get_state,
            backend.get_dof_pos,
            backend.get_base_pos,
            lambda: backend.get_sensor_data("joint_position"),
            lambda: backend.step(np.zeros((3, 1))),
        ):
            with pytest.raises(RuntimeError, match="faulted"):
                call()
    finally:
        backend.close()


@pytest.mark.parametrize(
    "field",
    [
        "body_mass",
        "body_inertia",
        "dof_armature",
        "dof_damping",
        "dof_frictionloss",
        "geom_size",
        "geom_friction",
        "body_iquat",
        "base_mass_delta",
    ],
)
def test_invalid_model_domain_rejected_before_native_reset_or_model_write(tmp_path, field):
    backend = _legacy(tmp_path)
    try:
        backend._base_body_id = backend._model.body("free").id
        backend._base_name = "free"
        backend.step(np.ones((3, 1)))
        before = backend.get_state()
        initial_mass = backend._pool.expand("body_mass").copy()
        initial_ctrl = backend._ctrl_view.copy()
        initial_act = backend._act_view.copy()
        initial_time = backend._time_view.copy()
        if field == "base_mass_delta":
            value = -initial_mass[[2, 0], backend._base_body_id] - 0.5
        else:
            value = np.broadcast_to(
                getattr(backend._model, field), (2, *getattr(backend._model, field).shape)
            ).copy()
            if field == "body_iquat":
                value[:, backend._base_body_id] = 0
            else:
                value.reshape(2, -1)[:, -1] = -0.1
        payload = ResetRandomizationPayload(**{field: value})
        native = _CountingPool(backend._pool)
        backend._pool = native
        error = "unit wxyz" if field == "body_iquat" else "nonnegative"
        with pytest.raises(ValueError, match=error):
            backend.set_state(
                np.array([2, 0]), before["qpos"][[2, 0]], before["qvel"][[2, 0]], payload
            )
        assert native.calls == []
        for name in before:
            np.testing.assert_array_equal(backend.get_state()[name], before[name])
        np.testing.assert_array_equal(native.pool.expand("body_mass"), initial_mass)
        np.testing.assert_array_equal(backend._ctrl_view, initial_ctrl)
        np.testing.assert_array_equal(backend._act_view, initial_act)
        np.testing.assert_array_equal(backend._time_view, initial_time)
        assert not backend._entity_faulted
    finally:
        backend.close()


def test_negative_mass_delta_is_valid_when_result_stays_nonnegative(tmp_path):
    backend = _legacy(tmp_path)
    try:
        backend._base_body_id = backend._model.body("free").id
        backend._base_name = "free"
        before = backend.get_state()
        backend.set_state(
            np.array([2, 0]),
            before["qpos"][[2, 0]],
            before["qvel"][[2, 0]],
            ResetRandomizationPayload(base_mass_delta=np.array([-0.2, -0.4])),
        )
        np.testing.assert_allclose(
            backend._pool.expand("body_mass")[[2, 0], backend._base_body_id], [0.8, 0.6]
        )
        np.testing.assert_array_equal(backend._pool.expand("body_mass")[:, 0], 0)
    finally:
        backend.close()
