"""CPU kernel fixtures; these do not claim a CPU MJWarp backend profile."""

from __future__ import annotations

# ruff: noqa: E402
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")

from unisim.backend.mjwarp.backend import MjwarpBackend
from unisim.backend.mjwarp.physics_step_tensor_data import MjwarpPhysicsStepTensorDataView
from unisim.backend.mjwarp.state_commit import StateCommitPlan
from unisim.backend.model_index import CompiledModelIndex


@pytest.fixture(scope="module", autouse=True)
def _cpu_warp_scope():
    with wp.ScopedDevice("cpu"):
        yield


class _CPUBackend:
    def __init__(self) -> None:
        self._num_envs = 3
        self._sim_dt = 0.005
        self._body_ids = {"left": 1, "right": 2, "other": 3}
        self._cpu_model = SimpleNamespace(
            nbody=4, geom_bodyid=np.array([0, 1, 2, 1, 3], dtype=np.int32)
        )
        self._device_model = object()
        self._warp = wp
        self._faulted = False
        self._closed = False
        self._joint_columns = {"ankle_a": (1,), "ankle_b": (4,), "multi": (2, 3)}
        frames = np.array(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            ],
            dtype=np.float32,
        )
        self._device_data = SimpleNamespace(
            qvel=wp.zeros((3, 5), dtype=float, device="cpu"),
            nacon=wp.array([4], dtype=int, device="cpu"),
            contact=SimpleNamespace(
                geom=wp.array([[1, 2], [1, 3], [1, -1], [2, 4], [0, 1]],
                              dtype=wp.vec2i, device="cpu"),
                worldid=wp.array([0, 0, 1, 2, 2], dtype=int, device="cpu"),
                frame=wp.array(frames, dtype=wp.mat33, device="cpu"),
            ),
        )
        self._force_values = np.array(
            [[2, 101, 203, 0, 0, 0], [3, 107, 211, 0, 0, 0],
             [4, 109, 223, 0, 0, 0], [5, 113, 227, 0, 0, 0],
             [10000, 127, 229, 0, 0, 0]],
            dtype=np.float32,
        )
        self._forces = wp.array(self._force_values, dtype=wp.spatial_vector, device="cpu")
        self._mujoco_warp = SimpleNamespace(contact_force=self._contact_force)

    def _require_entity_healthy(self) -> None:
        if self._faulted:
            raise RuntimeError("fixture backend is faulted")
        if self._closed:
            raise RuntimeError("fixture backend is closed")

    def _torch_view(self, array: Any, *, require_contiguous: bool = True):
        view = wp.to_torch(array)
        if require_contiguous:
            assert view.is_contiguous()
        return view

    def get_joint_state_qvel_indices(self, names):
        try:
            return np.array(
                [column for name in names for column in self._joint_columns[name]],
                dtype=np.int32,
            )
        except KeyError as exc:
            raise ValueError("unknown fixture joint") from exc

    def get_body_ids(self, names):
        try:
            return np.array([self._body_ids[name] for name in names], dtype=np.int32)
        except KeyError as exc:
            raise ValueError("unknown fixture body") from exc

    def _contact_force(self, model, data, contacts, to_world_frame, force) -> None:
        assert model is self._device_model
        assert data is self._device_data
        assert contacts.size == 5
        assert to_world_frame is False
        wp.copy(force, self._forces)

    def scale_forces(self, scale: float) -> None:
        self._forces.assign(self._force_values * scale)


def _view(backend: _CPUBackend, *, history_length: int = 3):
    return MjwarpPhysicsStepTensorDataView(
        backend, ("ankle_b", "ankle_a"), ("right", "left"), history_length
    )


def _filled_view():
    backend = _CPUBackend()
    view = _view(backend)
    view.before_step()
    wp.to_torch(backend._device_data.qvel).add_(1)
    view.after_step()
    return backend, view


def _expected_body_forces():
    # Contact 1 joins two geoms on left and cancels. Contact 4 is inactive.
    return torch.tensor(
        [[[2, 0, 0], [-2, 0, 0]], [[0, 0, 0], [0, 0, -4]], [[0, 5, 0], [0, 0, 0]]],
        dtype=torch.float32,
    )


def _sampling_backend_shell(fake: _CPUBackend):
    backend = MjwarpBackend.__new__(MjwarpBackend)
    backend.__dict__.update(vars(fake))
    backend.get_joint_state_qvel_indices = fake.get_joint_state_qvel_indices
    backend._state_torch_views = {}
    backend._physics_step_tensor_views = ()
    return backend


def _state_commit_shell(fake: _CPUBackend, view):
    backend = _sampling_backend_shell(fake)
    backend._physics_step_tensor_views = (view,)
    backend._nq, backend._nv, backend._nu, backend._nmocap = 6, 5, 2, 0
    backend._fixed_variant_plan = None
    backend._tracked_body_names = ()
    backend._tracked_body_state_dirty = False
    backend._host_cache_stale = False
    backend._tensor_reset_defaults = None
    backend._torch_true_verdict = None
    backend._sensor_cache = np.arange(6, dtype=np.float32).reshape(3, 2)
    backend._ctrl_staging = np.zeros((3, 2), dtype=np.float32)
    backend._mocap_pos = np.zeros((3, 0, 3), dtype=np.float32)
    backend._mocap_quat = np.zeros((3, 0, 4), dtype=np.float32)
    backend._time_cache = np.arange(3, dtype=np.float32)
    backend._xfrc_staging = np.zeros((3, 4, 6), dtype=np.float32)
    backend._xfrc_pending = False
    backend._device_xfrc_pending = False
    backend._interval_root_velocity_qvel_ids = None
    backend._push_body_id = None
    shapes = {
        "qpos": (3, 6), "ctrl": (3, 2), "act": (3, 0), "qfrc_applied": (3, 5),
        "xfrc_applied": (3, 4, 6), "qacc_warmstart": (3, 5), "time": (3,),
        "mocap_pos": (3, 0, 3), "mocap_quat": (3, 0, 4),
    }
    for name, shape in shapes.items():
        setattr(backend._device_data, name, wp.zeros(shape, dtype=float, device="cpu"))
    backend._reset_mask_device = wp.zeros(3, dtype=bool, device="cpu")
    backend._upload = lambda target, values: target.assign(np.ascontiguousarray(values))
    backend._execute_device_forward = lambda: None
    backend._execute_device_reset = lambda: None
    backend._synchronize = lambda: None
    backend._refresh_host_cache = lambda: None
    backend._can_use_reset_scratch = lambda rows: False
    return backend


def test_final_substep_derivative_uses_requested_joint_order_and_stable_aliases() -> None:
    backend = _CPUBackend()
    view = _view(backend)
    acceleration = view.joint_acceleration
    history = view.contact_force_history
    velocity = wp.to_torch(backend._device_data.qvel)
    assert acceleration.shape == (3, 2)
    assert history.shape == (3, 3, 2, 3)
    assert acceleration.dtype == history.dtype == torch.float32
    assert acceleration.device == history.device == torch.device("cpu")
    assert acceleration.is_contiguous() and history.is_contiguous()
    delta = torch.tensor([[0.25, 0.5], [0.5, -0.25], [-0.5, 0.75]])

    for substep in range(1, 5):
        view.before_step()
        velocity[:, 4].add_(delta[:, 0] * substep)
        velocity[:, 1].add_(delta[:, 1] * substep)
        velocity[:, 0].add_(1000)
        view.after_step()

    torch.testing.assert_close(acceleration, delta * 4 / backend._sim_dt)
    assert view.joint_acceleration.data_ptr() == acceleration.data_ptr()
    assert view.contact_force_history.data_ptr() == history.data_ptr()


def test_contact_history_is_signed_normal_only_world_force_and_rolls_every_substep() -> None:
    backend = _CPUBackend()
    view = _view(backend)
    history = view.contact_force_history
    expected = _expected_body_forces()
    for scale in range(1, 6):
        backend.scale_forces(scale)
        view.before_step()
        view.after_step()

    torch.testing.assert_close(history, torch.stack([expected * i for i in (5, 4, 3)], dim=1))
    backend._device_data.nacon.assign(np.array([0], dtype=np.int32))
    view.before_step()
    view.after_step()
    torch.testing.assert_close(
        history, torch.stack([torch.zeros_like(expected), expected * 5, expected * 4], dim=1)
    )
    assert history.data_ptr() == view.contact_force_history.data_ptr()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_selected_reset_clears_only_requested_rows_and_keeps_aliases(dtype) -> None:
    _, view = _filled_view()
    acceleration = view.joint_acceleration
    history = view.contact_force_history
    before_acceleration = acceleration.clone()
    before_history = history.clone()

    view.reset(torch.tensor([2, 0], dtype=dtype))

    torch.testing.assert_close(acceleration[[0, 2]], torch.zeros_like(acceleration[[0, 2]]))
    torch.testing.assert_close(history[[0, 2]], torch.zeros_like(history[[0, 2]]))
    torch.testing.assert_close(acceleration[1], before_acceleration[1])
    torch.testing.assert_close(history[1], before_history[1])
    assert acceleration.data_ptr() == view.joint_acceleration.data_ptr()
    assert history.data_ptr() == view.contact_force_history.data_ptr()


def test_slice_reset_and_empty_reset_preserve_other_rows() -> None:
    _, view = _filled_view()
    before_acceleration = view.joint_acceleration.clone()
    before_history = view.contact_force_history.clone()
    view.reset(torch.empty(0, dtype=torch.int64))
    torch.testing.assert_close(view.joint_acceleration, before_acceleration)
    torch.testing.assert_close(view.contact_force_history, before_history)

    view.reset(slice(0, None, 2))
    torch.testing.assert_close(view.joint_acceleration[1], before_acceleration[1])
    torch.testing.assert_close(view.contact_force_history[1], before_history[1])
    assert torch.count_nonzero(view.joint_acceleration[[0, 2]]) == 0
    assert torch.count_nonzero(view.contact_force_history[[0, 2]]) == 0


def test_reset_accepts_noncontiguous_integer_rows() -> None:
    _, view = _filled_view()
    before_acceleration = view.joint_acceleration[1].clone()
    before_history = view.contact_force_history[1].clone()
    rows = torch.tensor([0, 99, 2, 99], dtype=torch.int64)[::2]
    assert not rows.is_contiguous()

    view.reset(rows)

    assert torch.count_nonzero(view.joint_acceleration[[0, 2]]) == 0
    assert torch.count_nonzero(view.contact_force_history[[0, 2]]) == 0
    torch.testing.assert_close(view.joint_acceleration[1], before_acceleration)
    torch.testing.assert_close(view.contact_force_history[1], before_history)


def test_reset_all_clears_samples_before_another_physics_step() -> None:
    _, view = _filled_view()
    view.reset()
    assert torch.count_nonzero(view.joint_acceleration) == 0
    assert torch.count_nonzero(view.contact_force_history) == 0


@pytest.mark.parametrize(
    "rows",
    [
        [0], np.array([0]), torch.tensor(0), torch.tensor([[0]]),
        torch.tensor([0.0]), torch.tensor([True]), torch.tensor([0], dtype=torch.int16),
        torch.tensor([0, 0]), torch.tensor([-1]), torch.tensor([3]),
        torch.empty(1, dtype=torch.int64, device="meta"), slice(None, None, 0),
        slice(False, 1),
    ],
    ids=[
        "list", "numpy", "scalar", "matrix", "float", "bool", "int16", "duplicate",
        "negative", "out-of-range", "wrong-device", "zero-stride", "bool-slice",
    ],
)
def test_invalid_reset_is_rejected_before_any_sample_mutation(rows) -> None:
    _, view = _filled_view()
    before_acceleration = view.joint_acceleration.clone()
    before_history = view.contact_force_history.clone()
    with pytest.raises((TypeError, ValueError)):
        view.reset(rows)
    torch.testing.assert_close(view.joint_acceleration, before_acceleration)
    torch.testing.assert_close(view.contact_force_history, before_history)


@pytest.mark.parametrize(
    ("joints", "bodies", "history_length"),
    [
        ((), (), 3), (("ankle_a",), (), 0), (("ankle_a",), (), True),
        (("ankle_a",), (), 1.5), (("ankle_a", "ankle_a"), (), 3),
        ((), ("left", "left"), 3), ("ankle_a", (), 3), ((), "left", 3),
        (("",), (), 3), (("missing",), (), 3), ((), ("missing",), 3),
        (("multi",), (), 3),
    ],
    ids=[
        "empty", "zero-history", "bool-history", "float-history", "duplicate-joint",
        "duplicate-body", "joint-string", "body-string", "empty-name", "unknown-joint",
        "unknown-body", "multi-dof",
    ],
)
def test_cold_binding_rejects_invalid_layouts(joints, bodies, history_length) -> None:
    with pytest.raises((TypeError, ValueError)):
        MjwarpPhysicsStepTensorDataView(_CPUBackend(), joints, bodies, history_length)


@pytest.mark.parametrize("joint_only", [True, False], ids=["joint-only", "body-only"])
def test_binding_can_request_only_one_sample_kind(joint_only: bool) -> None:
    backend = _CPUBackend()
    view = MjwarpPhysicsStepTensorDataView(
        backend, ("ankle_a",) if joint_only else (), () if joint_only else ("left",), 1
    )
    view.before_step()
    wp.to_torch(backend._device_data.qvel).add_(1)
    view.after_step()
    assert view.joint_acceleration.shape == (3, 1 if joint_only else 0)
    assert view.contact_force_history.shape == (3, 1, 0 if joint_only else 1, 3)
    view.reset(torch.tensor([1]))


@pytest.mark.parametrize("status", ["faulted", "closed"])
def test_unhealthy_backend_rejects_sample_reads_and_reset(status: str) -> None:
    backend, view = _filled_view()
    setattr(backend, "_" + status, True)
    with pytest.raises(RuntimeError, match=status):
        _ = view.joint_acceleration
    with pytest.raises(RuntimeError, match=status):
        _ = view.contact_force_history
    with pytest.raises(RuntimeError, match=status):
        view.reset()


@pytest.mark.parametrize("device", [None, "cpu"])
def test_public_binding_uses_exact_device_and_registers_sampling_view(device) -> None:
    backend = _sampling_backend_shell(_CPUBackend())
    view = backend.bind_physics_step_tensor_data(
        joint_names=("ankle_b", "ankle_a"), body_names=("right", "left"), device=device
    )
    assert backend._physics_step_tensor_views == (view,)
    assert view.joint_names == ("ankle_b", "ankle_a")
    assert view.body_names == ("right", "left")
    assert view.history_length == 3
    assert view.joint_acceleration.device == torch.device("cpu")


@pytest.mark.parametrize("device", ["meta", "cuda:1"])
def test_public_binding_rejects_other_device_before_registering_view(device) -> None:
    backend = _sampling_backend_shell(_CPUBackend())
    with pytest.raises(ValueError, match="must live on"):
        backend.bind_physics_step_tensor_data(joint_names=("ankle_a",), device=device)
    assert backend._physics_step_tensor_views == ()


def test_public_binding_uses_native_mixed_joint_layout_and_rejects_ball_and_free() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="early"><joint name="early_hinge"/><geom size=".1"/></body>
      <body name="floating"><freejoint name="free_joint"/><geom size=".1"/>
        <body name="ball_child" pos="0 0 .3">
          <joint name="ball_joint" type="ball"/><geom size=".1"/>
          <body name="ankle_body" pos="0 0 .3"><joint name="ankle"/><geom size=".1"/></body>
        </body>
      </body>
      <body name="late"><joint name="late_slide" type="slide"/><geom size=".1"/></body>
    </worldbody></mujoco>""")
    backend = MjwarpBackend.__new__(MjwarpBackend)
    backend._num_envs, backend._sim_dt = 2, 0.01
    backend._cpu_model = model
    backend._compiled_index = CompiledModelIndex.from_model(model)
    backend._joint_ids = {model.joint(i).name: i for i in range(model.njnt)}
    backend._body_ids = {model.body(i).name: i for i in range(model.nbody)}
    backend._root_qvel_dim = 0
    backend._state_torch_views = {}
    backend._physics_step_tensor_views = ()
    backend._device_data = SimpleNamespace(
        qvel=wp.zeros((2, model.nv), dtype=float, device="cpu"),
        contact=SimpleNamespace(geom=wp.zeros(0, dtype=wp.vec2i, device="cpu")),
    )
    names = ("late_slide", "early_hinge", "ankle")
    view = backend.bind_physics_step_tensor_data(joint_names=names)
    np.testing.assert_array_equal(backend.get_joint_state_qvel_indices(names), [11, 0, 10])
    view.before_step()
    velocity = wp.to_torch(backend._device_data.qvel)
    velocity[0] = torch.arange(model.nv)
    velocity[1] = torch.arange(model.nv) * 2
    view.after_step()
    torch.testing.assert_close(
        view.joint_acceleration,
        torch.tensor([[1100.0, 0.0, 1000.0], [2200.0, 0.0, 2000.0]]),
    )
    for name in ("ball_joint", "free_joint"):
        with pytest.raises(ValueError, match="single-DoF"):
            backend.bind_physics_step_tensor_data(joint_names=(name,))
    assert backend._physics_step_tensor_views == (view,)


@pytest.mark.parametrize("selected", [[], [1]], ids=["empty", "selected"])
def test_host_state_commit_clears_only_affected_sampling_rows(selected) -> None:
    fake, view = _filled_view()
    backend = _state_commit_shell(fake, view)
    before_acceleration = view.joint_acceleration.clone()
    before_history = view.contact_force_history.clone()
    before_velocity = wp.to_torch(backend._device_data.qvel).clone()
    rows = np.array(selected, dtype=np.int32)
    velocity = before_velocity.numpy().copy()
    velocity[rows] = -7
    channels = {
        name: getattr(backend._device_data, name).numpy().copy()
        for name in ("ctrl", "act", "qfrc_applied", "xfrc_applied", "qacc_warmstart")
    }
    backend._commit_state(
        StateCommitPlan(
            rows=rows, qpos=np.zeros((3, 6), dtype=np.float32), qvel=velocity,
            mocap_pos=backend._mocap_pos.copy(), mocap_quat=backend._mocap_quat.copy(),
            channels=channels, staged_wrenches=backend._xfrc_staging.copy(),
            time=backend._time_cache.copy(),
        )
    )

    expected_acceleration, expected_history = before_acceleration, before_history
    expected_acceleration[selected] = 0
    expected_history[selected] = 0
    torch.testing.assert_close(view.joint_acceleration, expected_acceleration)
    torch.testing.assert_close(view.contact_force_history, expected_history)
    torch.testing.assert_close(wp.to_torch(backend._device_data.qvel), torch.from_numpy(velocity))


def test_tensor_selected_state_reset_clears_samples_and_preserves_other_rows(monkeypatch) -> None:
    fake, view = _filled_view()
    backend = _state_commit_shell(fake, view)
    before_acceleration = view.joint_acceleration.clone()
    before_history = view.contact_force_history.clone()
    before_velocity = wp.to_torch(backend._device_data.qvel).clone()

    def cpu_completion_stream(device):
        assert torch.device(device).type == "cpu"
        return SimpleNamespace(synchronize=lambda: None)

    monkeypatch.setattr(torch.cuda, "current_stream", cpu_completion_stream)
    backend.set_state_tensor(
        torch.tensor([1], dtype=torch.int64),
        torch.full((1, 6), -2, dtype=torch.float32),
        torch.full((1, 5), -7, dtype=torch.float32),
    )

    before_acceleration[1] = 0
    before_history[1] = 0
    before_velocity[1] = -7
    torch.testing.assert_close(view.joint_acceleration, before_acceleration)
    torch.testing.assert_close(view.contact_force_history, before_history)
    torch.testing.assert_close(wp.to_torch(backend._device_data.qvel), before_velocity)


@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
def test_backend_samples_before_and_after_every_graph_or_eager_substep(graph: bool) -> None:
    backend = MjwarpBackend.__new__(MjwarpBackend)
    events: list[str] = []
    backend._cuda_graph_enabled = graph
    backend._step_graph = object()
    backend._device_model = object()
    backend._device_data = object()
    backend._physics_step_tensor_views = [
        SimpleNamespace(
            before_step=lambda: events.append("before"),
            after_step=lambda: events.append("after"),
        )
    ]

    def graph_step(captured):
        assert captured is backend._step_graph
        events.append("graph")

    def eager_step(model, data):
        assert model is backend._device_model and data is backend._device_data
        events.append("eager")

    backend._warp = SimpleNamespace(capture_launch=graph_step)
    backend._mujoco_warp = SimpleNamespace(step=eager_step)
    backend._execute_device_steps(4)

    assert events == ["before", "graph" if graph else "eager", "after"] * 4


def test_mjwarp_declares_physics_step_sampling_capability() -> None:
    backend = MjwarpBackend.__new__(MjwarpBackend)
    backend._interval_root_velocity_qvel_ids = None
    backend._push_body_id = None
    assert backend.get_tensor_capabilities().physics_step_data is True


@pytest.mark.parametrize("selected", [[False, False, False], [False, True, False], [True] * 3])
def test_mask_state_commit_clears_samples_without_changing_other_rows(selected):
    fake, view = _filled_view()
    backend = _state_commit_shell(fake, view)
    mask = torch.tensor(selected)
    acc, history = view.joint_acceleration, view.contact_force_history
    expected_acc, expected_history = acc.clone(), history.clone()
    expected_acc[mask], expected_history[mask] = 0, 0
    before_velocity = wp.to_torch(backend._device_data.qvel).clone()
    backend.set_state_tensor_mask(
        mask, torch.zeros(3, 6), torch.full((3, 5), -7, dtype=torch.float32)
    )
    torch.testing.assert_close(acc, expected_acc, rtol=0, atol=0)
    torch.testing.assert_close(history, expected_history, rtol=0, atol=0)
    torch.testing.assert_close(
        wp.to_torch(backend._device_data.qvel)[~mask], before_velocity[~mask], rtol=0, atol=0
    )
    assert acc.data_ptr() == view.joint_acceleration.data_ptr()
    assert history.data_ptr() == view.contact_force_history.data_ptr()


def test_mask_sample_reset_stays_full_width_and_off_host(monkeypatch):
    _, view = _filled_view()
    mask = torch.tensor([False, True, False])
    before = view.contact_force_history.clone()

    def reject(*args, **kwargs):
        raise AssertionError("mask reset must not compact rows or cross the host boundary")

    with monkeypatch.context() as patches:
        for method in ("nonzero", "cpu", "numpy", "item", "tolist"):
            patches.setattr(torch.Tensor, method, reject)
        view._reset_mask(mask)
    torch.testing.assert_close(view.contact_force_history[[0, 2]], before[[0, 2]])
    assert not bool(view.contact_force_history[1].any())


def test_tensor_completion_is_published_after_all_substep_samples():
    fake, view = _filled_view()
    backend = _state_commit_shell(fake, view)
    backend._pre_step_control_fn = None
    backend._cuda_graph_enabled = False
    backend._torch_execution_stream = lambda device: SimpleNamespace(device=device)
    observed = []

    def native_step(model, data):
        wp.to_torch(data.qvel).add_(1)

    backend._mujoco_warp.step = native_step

    def complete(*, stream):
        observed.append(view.joint_acceleration.clone())
        return "completion-after-sampling"

    backend._step_completion = complete
    result = backend.step_tensor(torch.zeros(3, 2), nsteps=4)
    assert result["completion"] == "completion-after-sampling"
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], torch.full((3, 2), 200.0))
