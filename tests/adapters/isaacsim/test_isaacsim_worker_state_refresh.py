"""Selected-row IsaacSim state refresh preserves untouched shared-memory rows."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unisim.backend.isaacsim.worker import _WorkerContext


class _FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value, dtype=np.float32)

    def index_select(self, _dim: int, index: "_FakeTensor") -> "_FakeTensor":
        return _FakeTensor(self.value[index.value.astype(np.int64)])

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _FakeTorch:
    long = np.int64

    @staticmethod
    def as_tensor(value: np.ndarray, **_kwargs: object) -> _FakeTensor:
        return _FakeTensor(value)


def _make_context() -> _WorkerContext:
    num_envs = 4
    num_dof = 2
    num_bodies = 2
    root = np.zeros((num_envs, 13), dtype=np.float32)
    root[:, 0] = np.arange(num_envs, dtype=np.float32)
    dof_pos = np.arange(num_envs * num_dof, dtype=np.float32).reshape(num_envs, num_dof)
    dof_vel = dof_pos + 100.0
    body = np.zeros((num_envs, num_bodies, 13), dtype=np.float32)
    body[:, :, 0] = np.arange(num_envs, dtype=np.float32)[:, None]
    rigid = np.zeros((num_envs, 13), dtype=np.float32)
    rigid[:, 0] = np.arange(num_envs, dtype=np.float32) + 10.0

    ctx = _WorkerContext.__new__(_WorkerContext)
    ctx.num_envs = num_envs
    ctx.num_dof = num_dof
    ctx.num_bodies = num_bodies
    ctx.device = "cpu"
    ctx.torch = _FakeTorch()
    ctx.env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    ctx.native_joint_for_contract = np.array([0, 1], dtype=np.int64)
    ctx.native_body_for_contract = np.array([0, 1], dtype=np.int64)
    ctx.robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_state_w=_FakeTensor(root),
            joint_pos=_FakeTensor(dof_pos),
            joint_vel=_FakeTensor(dof_vel),
            body_link_state_w=_FakeTensor(body),
        )
    )
    ctx.rigid_objects = {
        "object": SimpleNamespace(data=SimpleNamespace(root_link_state_w=_FakeTensor(rigid)))
    }
    ctx.protocol = SimpleNamespace(entity_root_state_slot=lambda name: f"{name}_root")
    ctx.slots = {
        "root_state": np.full((num_envs, 13), -1.0, dtype=np.float32),
        "dof_state": np.full((num_envs, num_dof, 2), -1.0, dtype=np.float32),
        "body_state": np.full((num_envs, num_bodies, 13), -1.0, dtype=np.float32),
        "contact_force": np.full((num_envs, num_bodies, 3), -1.0, dtype=np.float32),
        "object_root": np.full((num_envs, 13), -1.0, dtype=np.float32),
    }
    return ctx


def test_selected_refresh_only_updates_selected_rows() -> None:
    ctx = _make_context()
    ctx.refresh_state_slots()
    before = {name: value.copy() for name, value in ctx.slots.items()}

    ctx.robot.data.root_link_state_w.value[[1, 3], 0] += 50.0
    ctx.robot.data.body_link_state_w.value[[1, 3], :, 0] += 50.0
    ctx.rigid_objects["object"].data.root_link_state_w.value[[1, 3], 0] += 50.0
    ctx.refresh_state_slots(np.array([1, 3], dtype=np.int32))

    for name, value in ctx.slots.items():
        np.testing.assert_array_equal(value[[0, 2]], before[name][[0, 2]])
    assert ctx.slots["root_state"][1, 0] == before["root_state"][1, 0] + 50.0
    assert ctx.slots["object_root"][3, 0] == before["object_root"][3, 0] + 50.0
