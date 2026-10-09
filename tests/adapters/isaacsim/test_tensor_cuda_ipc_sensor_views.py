"""SDK-free contracts for IsaacSim CUDA IPC G1 sensor/body projections."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from unisim.backend.isaacsim.backend import IsaacSimBackend
from unisim.backend.isaacsim.scene_worker import SceneWorkerContext


class _Tensor:
    is_cuda = True

    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def numel(self) -> int:
        return int(self.values.size)

    def clone(self) -> "_Tensor":
        return _Tensor(self.values.copy())

    def unsqueeze(self, axis: int) -> "_Tensor":
        return _Tensor(np.expand_dims(self.values, axis))

    def expand(self, *shape: int) -> "_Tensor":
        return _Tensor(np.broadcast_to(self.values, shape))

    def index_select(self, _axis: int, index: "_Tensor") -> "_Tensor":
        return _Tensor(self.values.take(index.values.astype(np.int64), axis=_axis))

    def __getitem__(self, key: Any) -> "_Tensor":
        if isinstance(key, tuple):
            key = tuple(item.values if isinstance(item, _Tensor) else item for item in key)
        return _Tensor(self.values[key])

    def __setitem__(self, key: Any, value: Any) -> None:
        native = value.values if isinstance(value, _Tensor) else value
        self.values[key] = native

    def __neg__(self) -> "_Tensor":
        return _Tensor(-self.values)

    def __add__(self, value: Any) -> "_Tensor":
        native = value.values if isinstance(value, _Tensor) else value
        return _Tensor(self.values + native)

    def __mul__(self, value: Any) -> "_Tensor":
        native = value.values if isinstance(value, _Tensor) else value
        return _Tensor(self.values * native)

    def __rmul__(self, value: Any) -> "_Tensor":
        native = value.values if isinstance(value, _Tensor) else value
        return _Tensor(native * self.values)

    def zero_(self) -> None:
        self.values.fill(0.0)

    def sub_(self, value: Any) -> None:
        self.values -= value.values

    def index_copy_(self, axis: int, index: "_Tensor", source: "_Tensor") -> None:
        selector = tuple(
            slice(None) if item != axis else index.values for item in range(self.values.ndim)
        )
        self.values[selector] = source.values


class _Torch:
    Tensor = _Tensor

    @staticmethod
    def cross(left: _Tensor, right: _Tensor, *, dim: int) -> _Tensor:
        if left.values.ndim != right.values.ndim:
            raise ValueError("torch.cross requires matching tensor ranks")
        return _Tensor(np.cross(left.values, right.values, axis=dim))


class _View:
    def __init__(self, marker: tuple[Any, ...]) -> None:
        self.marker = marker

    def __getitem__(self, key: Any) -> "_View":
        if len(self.marker) == 2:
            rows, bodies, _columns = self.marker[1]
            selected_columns = key[2] if isinstance(key, tuple) else key
            return _View((self.marker[0], (rows, bodies, selected_columns)))
        return _View((*self.marker, key))


def test_worker_projects_body_state_and_g1_scalar_sensors_without_host_data() -> None:
    ctx = SceneWorkerContext.__new__(SceneWorkerContext)
    ctx.torch = _Torch
    ctx.device = "cuda:0"
    ctx._cuda_origins = _Tensor([[10.0, 0.0, 0.0]])
    native_body = np.zeros((1, 1, 13), dtype=np.float32)
    native_body[0, 0, 0:3] = (10.5, 1.0, 2.0)
    native_body[0, 0, 3:7] = (np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0)
    native_body[0, 0, 7:10] = (0.0, 1.0, 0.0)
    native_body[0, 0, 10:13] = (0.0, 0.0, 1.0)
    arena = SimpleNamespace(
        body_state=_Tensor(np.zeros((1, 1, 13))),
        sensor_state=_Tensor(np.zeros((1, 2, 3))),
    )
    ctx._cuda_ipc = arena
    ctx._cuda_maps = [
        {
            "asset": SimpleNamespace(data=SimpleNamespace(body_link_state_w=_Tensor(native_body))),
            "public_body_ids": _Tensor([[0]]),
            "native_rows": _Tensor([[0]]),
            "bodies": _Tensor([[0]]),
        }
    ]
    ctx._cuda_sensor_specs = {
        "pelvis_local_linvel": {
            "kind": "local_linvel",
            "body_id": 0,
            "local_pos": _Tensor((0.0, 1.0, 0.0)),
            "local_quat": _Tensor((1.0, 0.0, 0.0, 0.0)),
        }
    }

    ctx._publish_cuda_body_state()
    ctx._publish_cuda_scalar_sensors()

    np.testing.assert_allclose(arena.body_state.values[0, 0, 0:3], (0.5, 1.0, 2.0), atol=1e-6)
    np.testing.assert_allclose(
        arena.body_state.values[0, 0, 3:7],
        (np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0),
        atol=1e-6,
    )
    # The site velocity is the link velocity plus omega x r at the world-frame
    # site offset. Inverse rotation expresses that world vector in site frame.
    np.testing.assert_allclose(arena.sensor_state.values[0, 0], (0.0, 1.0, -1.0), atol=1e-6)


def _backend_and_log() -> tuple[IsaacSimBackend, list[str]]:
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._cuda_ipc_arena = SimpleNamespace(
        body_state=_View(("body",)),
        sensor_state=_View(("sensor",)),
        qpos=SimpleNamespace(device="cuda:0"),
        device_index=0,
        layout=SimpleNamespace(nbody=4),
    )
    backend._ensure_cuda_ipc_arena = lambda: backend._cuda_ipc_arena  # type: ignore[method-assign]
    waits: list[str] = []
    backend._cuda_ipc_arena.wait_state = lambda: waits.append("state")  # type: ignore[attr-defined]
    backend._cuda_body_ids_by_name = {"robot/pelvis": 0, "robot/torso_link": 1}
    backend._cuda_sensor_spec_names = frozenset({"pelvis_local_linvel", "torso_gyro"})
    backend._cuda_sensor_aliases = {
        "pelvis_local_linvel": {"slot": 0, "descriptor": {}, "ambiguous": False},
        "torso_gyro": {"slot": 1, "descriptor": {}, "ambiguous": False},
    }
    backend._cuda_tracked_body_inventory = ("robot/pelvis", "robot/torso_link")
    return backend, waits


def test_host_returns_persistent_g1_view_slices_without_worker_requests() -> None:
    backend, waits = _backend_and_log()
    assert backend.get_sensor_view("pelvis_local_linvel").marker == ("sensor", (slice(None), 0))
    assert backend.get_sensor_view("track_pos_w_robot/pelvis").marker == (
        "body",
        (slice(None), 0, slice(0, 3)),
    )
    assert backend.get_sensor_view("track_quat_w_robot/torso_link").marker == (
        "body",
        (slice(None), 1, slice(3, 7)),
    )
    assert backend.get_sensor_view("track_linvel_w_robot/pelvis").marker == (
        "body",
        (slice(None), 0, slice(7, 10)),
    )
    assert backend.get_sensor_view("track_angvel_w_robot/torso_link").marker == (
        "body",
        (slice(None), 1, slice(10, 13)),
    )
    assert waits == ["state"] * 5


def test_host_sensor_view_fails_closed_for_unknown_names() -> None:
    backend, waits = _backend_and_log()
    with pytest.raises(KeyError, match="unknown IsaacSim CUDA IPC tensor sensor"):
        backend.get_sensor_view("pelvis_gyro")
    with pytest.raises(KeyError, match="unknown IsaacSim CUDA IPC tracked body"):
        backend.get_sensor_view("track_pos_w_left_foot")
    assert waits == []


def test_host_returns_inventory_and_aggregate_tracked_body_views() -> None:
    backend, waits = _backend_and_log()
    inventory = backend.get_sensor_inventory()
    assert inventory[0].name == "pelvis_local_linvel" and inventory[0].width == 3
    assert inventory[1].name == "torso_gyro" and inventory[1].width == 3
    tracked = {
        descriptor.name: descriptor.width
        for descriptor in inventory
        if descriptor.name.startswith("track_")
    }
    assert tracked == {
        "track_pos_w_robot/pelvis": 3,
        "track_quat_w_robot/pelvis": 4,
        "track_linvel_w_robot/pelvis": 3,
        "track_angvel_w_robot/pelvis": 3,
        "track_pos_w_robot/torso_link": 3,
        "track_quat_w_robot/torso_link": 4,
        "track_linvel_w_robot/torso_link": 3,
        "track_angvel_w_robot/torso_link": 3,
    }

    views = backend.get_tracked_body_views()
    assert views.body_names == ("robot/pelvis", "robot/torso_link")
    assert views.pos_w.marker == ("body", (slice(None), [0, 1], slice(0, 3)))
    assert views.quat_w.marker == ("body", (slice(None), [0, 1], slice(3, 7)))
    assert views.lin_vel_w.marker == ("body", (slice(None), [0, 1], slice(7, 10)))
    assert views.ang_vel_w.marker == ("body", (slice(None), [0, 1], slice(10, 13)))
    assert waits == ["state"]

    ordered = backend.get_tracked_body_views(("robot/torso_link",))
    assert ordered.body_names == ("robot/torso_link",)
    assert ordered.pos_w.marker == ("body", (slice(None), [1], slice(0, 3)))
    assert waits == ["state", "state"]


def test_host_tracked_body_view_name_selection_fails_closed() -> None:
    backend, waits = _backend_and_log()
    with pytest.raises(TypeError, match="sequence of strings"):
        backend.get_tracked_body_views("robot/pelvis")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="non-empty strings"):
        backend.get_tracked_body_views(("",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        backend.get_tracked_body_views(("robot/pelvis", "robot/pelvis"))
    with pytest.raises(ValueError, match="missing from the mapped namespace"):
        backend.get_tracked_body_views(("left_foot",))
    assert waits == []


def test_duplicate_unqualified_body_names_resolve_only_when_qualified() -> None:
    backend, waits = _backend_and_log()
    backend._cuda_tracked_body_inventory = ("robot/base", "object/base")
    backend._cuda_body_ids_by_name = {"robot/base": 1, "object/base": 3}
    with pytest.raises(KeyError, match="available bodies: robot/base, object/base"):
        backend.get_sensor_view("track_pos_w_base")
    assert backend.get_sensor_view("track_pos_w_object/base").marker == (
        "body",
        (slice(None), 3, slice(0, 3)),
    )
    views = backend.get_tracked_body_views(("object/base",))
    assert views.body_names == ("object/base",)
    assert views.pos_w.marker == ("body", (slice(None), [3], slice(0, 3)))
    assert waits == ["state", "state"]


def test_entity_qualified_scalar_alias_and_ambiguity_fail_closed() -> None:
    backend, waits = _backend_and_log()
    backend._cuda_sensor_aliases = {
        "robot/pelvis_local_linvel": {"slot": 0, "descriptor": {}, "ambiguous": False},
        "tool/pelvis_local_linvel": {"slot": 1, "descriptor": {}, "ambiguous": False},
        "pelvis_local_linvel": {"slot": 0, "descriptor": {}, "ambiguous": True},
    }
    view = backend.get_sensor_view("robot/pelvis_local_linvel")
    assert view.marker == ("sensor", (slice(None), 0))
    view = backend.get_sensor_view("tool/pelvis_local_linvel")
    assert view.marker == ("sensor", (slice(None), 1))
    with pytest.raises(KeyError, match="ambiguous IsaacSim CUDA IPC tensor sensor"):
        backend.get_sensor_view("pelvis_local_linvel")
    assert waits == ["state", "state"]
