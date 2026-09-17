"""Instrumented host tensors for sparse native reset transfer contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from unisim.backend.isaacsim.scene_worker import SceneWorkerContext


class TransferProbe:
    def __init__(self) -> None:
        self.d2h: list[int] = []
        self.h2d: list[int] = []
        self.operations: list[dict[str, Any]] = []

    def as_tensor(self, value: Any, **kwargs: Any) -> np.ndarray:
        array = np.asarray(value, dtype=kwargs.get("dtype"))
        self.h2d.append(array.nbytes)
        return array.copy()


class TensorDouble:
    def __init__(self, values: np.ndarray, probe: TransferProbe) -> None:
        self.values, self.probe = values, probe

    def detach(self) -> TensorDouble:
        return self

    def cpu(self) -> TensorDouble:
        self.probe.d2h.append(self.values.nbytes)
        return self

    def numpy(self) -> np.ndarray:
        return self.values

    def __getitem__(self, key: Any) -> TensorDouble:
        return TensorDouble(self.values[key], self.probe)


def execute_case(commit, *, num_envs: int, num_joints: int, rows: int) -> dict[str, Any]:
    if not 1 <= rows <= num_envs or num_joints < 4:
        raise ValueError("require 1 <= rows <= num_envs and num_joints >= 4")
    probe = TransferProbe()
    worker = SceneWorkerContext.__new__(SceneWorkerContext)
    worker.device, worker.sim_dt, worker.faulted = "host-double", 0.002, False
    worker.origins = np.zeros((num_envs, 3), dtype=np.float32)
    worker.torch = SimpleNamespace(as_tensor=probe.as_tensor, long=np.int64, float32=np.float32)
    entities, assets, maps = [], [], []
    for entity_id in range(4):
        joint_count = num_joints if entity_id < 2 else 0
        offset = entity_id * num_joints
        joints = tuple(
            SimpleNamespace(qpos_indices=(offset + j,), qvel_indices=(offset + j,))
            for j in range(joint_count)
        )
        entities.append(
            SimpleNamespace(
                joints=joints, root_mode="fixed", kind="articulation" if joint_count else "rigid"
            )
        )
        values = np.arange(num_envs * num_joints, dtype=np.float32).reshape(num_envs, num_joints)

        def write(position, velocity, *, joint_ids, env_ids, entity=entity_id):
            probe.operations.append(
                {
                    "operation": "write_joint_state",
                    "entity": entity,
                    "position": position.tolist(),
                    "velocity": velocity.tolist(),
                    "joints": list(joint_ids),
                    "rows": env_ids.tolist(),
                }
            )

        def reset(ids, entity=entity_id):
            probe.operations.append({"operation": "reset", "entity": entity, "rows": ids.tolist()})

        def update(dt, entity=entity_id):
            probe.operations.append({"operation": "update", "entity": entity, "dt": dt})

        assets.append(
            SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos=TensorDouble(values, probe),
                    joint_vel=TensorDouble(-values - 0.25, probe),
                ),
                write_joint_state_to_sim=write,
                reset=reset,
                update=update,
            )
        )
        maps.append(
            {"envs": np.arange(num_envs - 1, -1, -1), "joints": np.arange(num_joints - 1, -1, -1)}
        )
    worker.layout = SimpleNamespace(entities=tuple(entities))
    worker.assets, worker.maps = assets, maps
    ids = np.arange(rows - 1, -1, -1)
    qpos = np.zeros((rows, 2 * num_joints), dtype=np.float32)
    qvel = qpos.copy()
    qpos[:, num_joints + 3] = np.arange(rows, dtype=np.float32) + 0.75
    pmask, vmask = (
        np.zeros(2 * num_joints, dtype=np.uint8),
        np.zeros(2 * num_joints, dtype=np.uint8),
    )
    pmask[num_joints + 3] = 1
    commit(
        worker,
        ids,
        qpos,
        qvel,
        np.zeros((rows, 4, 13)),
        pmask,
        vmask,
        np.zeros((4, 2), dtype=np.uint8),
    )
    assert not worker.faulted
    return {
        "d2h_calls": len(probe.d2h),
        "d2h_bytes": sum(probe.d2h),
        "h2d_calls": len(probe.h2d),
        "h2d_bytes": sum(probe.h2d),
        "d2h_bytes_each": probe.d2h,
        "h2d_bytes_each": probe.h2d,
        "operations": probe.operations,
    }


