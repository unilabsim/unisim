"""SDK-free contracts for the IsaacGym G1 CUDA IPC sensor/body views."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from unisim.backend.isaacgym.tensor import IsaacGymCudaIpcPlan, IsaacGymCudaIpcWorkerRuntime


class _Tensor:
    def __init__(self, marker: tuple[Any, ...]) -> None:
        self.marker = marker

    def __getitem__(self, key: Any) -> "_Tensor":
        return _Tensor((*self.marker, key))


class _Cuda:
    def __init__(self, index: int) -> None:
        self.index = index

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None


class _WorkerCuda:
    @staticmethod
    def device(_index: int) -> _Cuda:
        return _Cuda(_index)

    @staticmethod
    def current_stream(_index: int) -> SimpleNamespace:
        return SimpleNamespace(cuda_stream=987)


class _Torch:
    cuda = SimpleNamespace(
        device=lambda index: _Cuda(index),
        current_stream=lambda index: SimpleNamespace(cuda_stream=321),
    )


def _plan() -> tuple[IsaacGymCudaIpcPlan, list[str]]:
    plan = IsaacGymCudaIpcPlan.__new__(IsaacGymCudaIpcPlan)
    plan.closed = False
    plan.device = "cuda:0"
    plan.device_index = 0
    plan.arena = SimpleNamespace(nbody=3)
    plan._torch = _Torch()
    waits: list[str] = []
    plan.state_event = SimpleNamespace(wait_stream=lambda stream: waits.append(stream))
    plan._body_ids_by_name = {"pelvis": 1, "torso_link": 2}
    plan._sensor_spec_names = frozenset({"pelvis_local_linvel", "torso_gyro"})
    plan._body_state = _Tensor(("body",))
    plan._sensor_state = _Tensor(("sensor",))
    return plan, waits


def test_g1_sensor_and_tracked_body_views_are_persistent_slices() -> None:
    plan, waits = _plan()
    assert plan.get_sensor_view("pelvis_local_linvel").marker == ("sensor", (slice(None), 0))
    assert plan.get_sensor_view("torso_gyro").marker == ("sensor", (slice(None), 1))
    assert plan.get_sensor_view("track_pos_w_pelvis").marker == (
        "body",
        (slice(None), 1, slice(0, 3)),
    )
    assert plan.get_sensor_view("track_quat_w_torso_link").marker == (
        "body",
        (slice(None), 2, slice(3, 7)),
    )
    assert plan.get_sensor_view("track_linvel_w_pelvis").marker == (
        "body",
        (slice(None), 1, slice(7, 10)),
    )
    assert plan.get_sensor_view("track_angvel_w_torso_link").marker == (
        "body",
        (slice(None), 2, slice(10, 13)),
    )
    assert waits == [321] * 6


def test_sensor_view_requests_fail_closed_without_worker_pipe() -> None:
    plan, waits = _plan()
    plan._sensor_spec_names = frozenset()
    with pytest.raises(KeyError, match="unknown IsaacGym CUDA IPC tensor sensor"):
        plan.get_sensor_view("pelvis_gyro")
    with pytest.raises(KeyError, match="unknown IsaacGym CUDA IPC tracked body"):
        plan.get_sensor_view("track_pos_w_left_foot")
    with pytest.raises(NotImplementedError, match="cannot serve tensor sensor"):
        plan.get_sensor_view("pelvis_local_linvel")
    assert waits == []


def test_sensor_views_are_available_after_reset_without_a_readiness_step() -> None:
    """Issue #349: views are readable after attach/reset without a step."""
    plan, waits = _plan()
    assert plan.get_sensor_view("track_pos_w_pelvis").marker == (
        "body",
        (slice(None), 1, slice(0, 3)),
    )
    assert plan.get_sensor_view("pelvis_local_linvel").marker == ("sensor", (slice(None), 0))
    assert waits == [321] * 2


def test_duplicate_unqualified_body_names_fail_closed() -> None:
    entity_scene = SimpleNamespace(
        layout=SimpleNamespace(
            entities=(
                SimpleNamespace(name="robot", body_names=("base",), body_ids=(1,)),
                SimpleNamespace(name="object", body_names=("base",), body_ids=(3,)),
            )
        )
    )
    mapping = IsaacGymCudaIpcPlan._map_entity_body_names(entity_scene)
    assert mapping == {
        "robot/base": 1,
        "object/base": 3,
    }


class _WorkerTensor:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)
        self.dtype = "float32"
        self.device = "cpu"

    def numel(self) -> int:
        return int(self.values.size)

    def reshape(self, *shape: int) -> "_WorkerTensor":
        return _WorkerTensor(self.values.reshape(shape))

    def index_select(self, axis: int, index: "_WorkerTensor") -> "_WorkerTensor":
        return _WorkerTensor(self.values.take(index.values.astype(np.int64), axis=axis))

    def clone(self) -> "_WorkerTensor":
        return _WorkerTensor(self.values.copy())

    def __getitem__(self, key: Any) -> "_WorkerTensor":
        if isinstance(key, tuple):
            key = tuple(item.values if isinstance(item, _WorkerTensor) else item for item in key)
        return _WorkerTensor(self.values[key])

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, tuple):
            key = tuple(item.values if isinstance(item, _WorkerTensor) else item for item in key)
        native = value.values if isinstance(value, _WorkerTensor) else value
        self.values[key] = native

    def zero_(self) -> None:
        self.values.fill(0.0)

    def __neg__(self) -> "_WorkerTensor":
        return _WorkerTensor(-self.values)

    def __add__(self, value: Any) -> "_WorkerTensor":
        native = value.values if isinstance(value, _WorkerTensor) else value
        return _WorkerTensor(self.values + native)

    def __mul__(self, value: Any) -> "_WorkerTensor":
        native = value.values if isinstance(value, _WorkerTensor) else value
        return _WorkerTensor(self.values * native)

    def __isub__(self, value: Any) -> "_WorkerTensor":
        native = value.values if isinstance(value, _WorkerTensor) else value
        self.values -= native
        return self

    def __rmul__(self, value: Any) -> "_WorkerTensor":
        native = value.values if isinstance(value, _WorkerTensor) else value
        return _WorkerTensor(native * self.values)

    def to(self, *, device: Any, dtype: Any) -> "_WorkerTensor":
        del device, dtype
        return self


class _WorkerTorch:
    float32 = "float32"
    cuda = _WorkerCuda

    @staticmethod
    def cross(left: _WorkerTensor, right: _WorkerTensor, *, dim: int) -> _WorkerTensor:
        return _WorkerTensor(np.cross(left.values, right.values, axis=dim))

    @staticmethod
    def tensor(values: Any, *, dtype: Any, device: Any) -> _WorkerTensor:
        del device
        return _WorkerTensor(np.asarray(values, dtype=np.float32))

    @staticmethod
    def from_numpy(values: Any) -> _WorkerTensor:
        return _WorkerTensor(values)


def test_worker_projects_env_local_body_state_and_g1_scalar_sensor_on_device_tensors() -> None:
    runtime = IsaacGymCudaIpcWorkerRuntime.__new__(IsaacGymCudaIpcWorkerRuntime)
    runtime.ctx = SimpleNamespace(torch=_WorkerTorch)
    runtime.arena = SimpleNamespace(num_envs=2, nbody=3)
    native_body = np.zeros((4, 13), dtype=np.float32)
    native_body[:] = (
        (10.5, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0),
        (10.5, 1.0, 2.0, 0.0, np.sqrt(0.5), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0),
        (20.5, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0),
        (20.5, 3.0, 4.0, 0.0, np.sqrt(0.5), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0),
    )
    native_body[:, 10:13] = (0.0, 1.0, 0.0)
    runtime.ctx._body_state = _WorkerTensor(native_body)
    runtime.body_state = _WorkerTensor(np.zeros((2, 3, 13)))
    runtime.sensor_state = _WorkerTensor(np.zeros((2, 2, 3)))
    runtime.body_rows = _WorkerTensor((0, 0, 1, 1))
    runtime.body_columns = _WorkerTensor((1, 2, 1, 2))
    runtime.root_body_columns = _WorkerTensor((1,))
    runtime.native_body_ids = _WorkerTensor((0, 1, 2, 3))
    runtime.body_com = _WorkerTensor(np.full((2, 2, 3), (0.5, 0.0, 0.0)))
    runtime.actor_ids = _WorkerTensor(((0,), (1,)))
    runtime.root_com = _WorkerTensor(np.zeros((2, 1, 3)))
    runtime.env_origins = _WorkerTensor(((10.0, 0.0, 0.0), (20.0, 0.0, 0.0)))
    runtime.ctx._root_state = _WorkerTensor(
        (
            (11.5, 1.5, 2.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0),
            (21.5, 3.5, 4.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0),
        )
    )
    runtime.sensor_specs = {
        "pelvis_local_linvel": {
            "kind": "local_linvel",
            "body_id": 1,
            "local_pos": _WorkerTensor((0.0, 0.0, 0.0)),
            "local_quat": _WorkerTensor((1.0, 0.0, 0.0, 0.0)),
        },
        "torso_gyro": {
            "kind": "gyro",
            "body_id": 2,
            "local_pos": _WorkerTensor((0.0, 0.0, 0.0)),
            "local_quat": _WorkerTensor((np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))),
        },
    }

    runtime._publish_body_state()
    runtime._publish_scalar_sensors()

    np.testing.assert_allclose(
        runtime.body_state.values[:, 1, 0:3],
        ((11.5, 1.5, 2.5), (21.5, 3.5, 4.5)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        runtime.body_state.values[:, 2, 0:3],
        ((10.5, 1.0, 2.0), (20.5, 3.0, 4.0)),
        atol=1e-6,
    )
    # Native xyzw quaternions become public wxyz.
    np.testing.assert_allclose(
        runtime.body_state.values[:, 1, 3:7],
        ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        runtime.body_state.values[:, 2, 3:7],
        ((0.0, 0.0, np.sqrt(0.5), 0.0), (0.0, 0.0, np.sqrt(0.5), 0.0)),
        atol=1e-6,
    )
    # Link-origin velocity subtracts omega x COM in world coordinates.
    np.testing.assert_allclose(
        runtime.sensor_state.values[:, 0],
        ((1.0, 0.0, 3.0), (1.0, 0.0, 3.0)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        runtime.sensor_state.values[:, 1],
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        atol=1e-6,
    )


def test_worker_selected_reset_publication_preserves_unselected_body_rows() -> None:
    """Issue #349: reset completes projection before replying, without stepping."""

    runtime = IsaacGymCudaIpcWorkerRuntime.__new__(IsaacGymCudaIpcWorkerRuntime)
    runtime.body_state = _WorkerTensor(np.zeros((2, 2, 13), dtype=np.float64))
    runtime.body_state.values[..., 3] = 1.0
    runtime.body_state.values[:, 1, 0] = 7.0
    runtime.sensor_state = _WorkerTensor(np.zeros((2, 2, 3)))
    runtime.sensor_specs = {}
    calls: list[str] = []
    runtime._refresh_native = lambda: calls.append("refresh")
    runtime._publish_body_state = lambda: calls.append("body")
    runtime._publish_scalar_sensors = lambda: calls.append("sensors")
    runtime._public_roots = lambda roots: _WorkerTensor(
        np.zeros((2, 1, 13), dtype=np.float64)
    )
    runtime.ctx = SimpleNamespace(
        _root_state=object(),
        _dof_state=_WorkerTensor(np.zeros((1, 2))),
        torch=_WorkerTorch,
    )
    runtime.root_projections = [
        {
            "mode": "fixed",
            "qpos": _WorkerTensor((0,)),
            "qvel": _WorkerTensor((0,)),
        }
    ]
    runtime.joint_projections = []
    runtime.qpos = _WorkerTensor(np.zeros((2, 1)))
    runtime.qvel = _WorkerTensor(np.zeros((2, 1)))
    scene = SimpleNamespace(
        faulted=False,
        pending_roots={},
        pending_dofs={},
        pending_dof_actors=set(),
    )
    runtime.ctx.scene_worker = scene
    runtime.expected_reset_sequence = 0
    runtime.closed = False
    runtime.device_index = 0
    runtime.state_event = SimpleNamespace(record=lambda stream: calls.append(("event", stream)))

    runtime.publish_state(record_event=True)

    assert calls == ["refresh", "body", "sensors", ("event", 987)]
    # Jointless/fixed publication resets only default identity rows; direct FK
    # ownership stays in the worker projection and must not touch row 0's
    # already-authoritative values in this focused unit oracle.
    assert runtime.body_state.values[0, 1, 0] == 7.0


def test_worker_sensor_descriptors_validate_kind_position_and_quaternion() -> None:
    runtime = IsaacGymCudaIpcWorkerRuntime.__new__(IsaacGymCudaIpcWorkerRuntime)
    runtime.ctx = SimpleNamespace(torch=_WorkerTorch, device="cpu")
    runtime.arena = SimpleNamespace(num_envs=2, nbody=3)
    valid = (
        {
            "name": "pelvis_local_linvel",
            "kind": "local_linvel",
            "body_id": 1,
            "local_pos": (0.04525, 0.0, -0.08339),
            "local_quat": (1.0, 0.0, 0.0, 0.0),
        },
        {
            "name": "torso_gyro",
            "kind": "gyro",
            "body_id": 2,
            "local_pos": (0.0, 0.0, 0.0),
            "local_quat": (0.5, 0.5, 0.5, 0.5),
        },
    )
    assert set(runtime._bind_sensor_specs(valid)) == {
        "pelvis_local_linvel",
        "torso_gyro",
    }

    wrong_kind = [dict(valid[0], kind="gyro")]
    with pytest.raises(ValueError, match="unsupported IsaacGym CUDA IPC sensor"):
        runtime._bind_sensor_specs(wrong_kind)
    malformed_position = [dict(valid[0], local_pos=(0.0, 0.0))]
    with pytest.raises(ValueError, match="unsupported IsaacGym CUDA IPC sensor"):
        runtime._bind_sensor_specs(malformed_position)
    nonunit_quat = [dict(valid[0], local_quat=(2.0, 0.0, 0.0, 0.0))]
    with pytest.raises(ValueError, match="unsupported IsaacGym CUDA IPC sensor"):
        runtime._bind_sensor_specs(nonunit_quat)


def test_worker_local_linear_velocity_includes_sensor_site_point_velocity() -> None:
    runtime = IsaacGymCudaIpcWorkerRuntime.__new__(IsaacGymCudaIpcWorkerRuntime)
    runtime.ctx = SimpleNamespace(torch=_WorkerTorch)
    runtime.body_state = _WorkerTensor(np.zeros((2, 3, 13)))
    runtime.sensor_state = _WorkerTensor(np.zeros((2, 2, 3)))
    runtime.body_state.values[:, 1, 3:7] = (
        (1.0, 0.0, 0.0, 0.0),
        (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
    )
    runtime.body_state.values[:, 1, 7:10] = ((1.0, 2.0, 3.0), (0.0, 1.0, 0.0))
    runtime.body_state.values[:, 1, 10:13] = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    runtime.sensor_specs = {
        "pelvis_local_linvel": {
            "kind": "local_linvel",
            "body_id": 1,
            "local_pos": _WorkerTensor((0.25, 0.0, 0.0)),
            "local_quat": _WorkerTensor((1.0, 0.0, 0.0, 0.0)),
        }
    }

    runtime._publish_scalar_sensors()

    np.testing.assert_allclose(
        runtime.sensor_state.values[:, 0],
        ((1.0, 2.0, 2.75), (1.0, 0.25, 0.0)),
        atol=1e-6,
    )
