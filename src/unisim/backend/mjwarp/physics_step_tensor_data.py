"""MJWarp-owned tensor samples at native physics substep boundaries.

The sampling kernels derive from UniSim commit
545c4a1ef614f0cff5088733dd93ada5966fef62, under Apache-2.0 (see LICENSE).
This adapter replaces that implementation's host snapshots with Torch aliases.
Ported from the SONIC fork commits 4f0a058f1e47 and 10fb1b3a68b8 to
UniSim 2.1.1, with backend-stream ordering and full-width mask reset support.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import warp as wp

from unisim.backend.base import PhysicsStepTensorDataView


# Warp compiles these bodies as device code; its Python stubs cannot model
# launch-dimensional tid() results or scalar array indexing in that DSL.
@wp.kernel
def _velocity_sample(qvel: wp.array2d[float], ids: wp.array[int], out: wp.array2d[float]):
    env, joint = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    out[env, joint] = qvel[env, ids[joint]]  # pyright: ignore[reportIndexIssue]


@wp.kernel
def _acceleration(
    qvel: wp.array2d[float],
    ids: wp.array[int],
    before: wp.array2d[float],
    dt: float,
    out: wp.array2d[float],
):
    env, joint = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    out[env, joint] = (qvel[env, ids[joint]] - before[env, joint]) / dt  # pyright: ignore[reportIndexIssue]


@wp.kernel
def _shift(history: wp.array4d[float], length: int):
    env, body, axis = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    for h in range(length - 1, 0, -1):
        history[env, h, body, axis] = history[env, h - 1, body, axis]  # pyright: ignore[reportIndexIssue]
    history[env, 0, body, axis] = 0.0  # pyright: ignore[reportIndexIssue]


@wp.kernel
def _accumulate(
    nacon: wp.array[int],
    worlds: wp.array[int],
    geoms: wp.array[wp.vec2i],
    frames: wp.array[wp.mat33],
    forces: wp.array[wp.spatial_vector],  # pyright: ignore[reportGeneralTypeIssues]
    geom_body: wp.array[int],
    selected: wp.array[int],
    history: wp.array4d[float],
):
    contact = wp.tid()
    if contact < nacon[0]:  # pyright: ignore[reportOperatorIssue]
        world = worlds[contact]
        for side in range(2):
            geom = geoms[contact][side]  # pyright: ignore[reportIndexIssue]
            if geom >= 0:  # pyright: ignore[reportOperatorIssue]
                body = selected[geom_body[geom]]
                if body >= 0:  # pyright: ignore[reportOperatorIssue]
                    sign = float(2 * side - 1)
                    for axis in range(3):
                        wp.atomic_add(
                            history,  # pyright: ignore[reportArgumentType]
                            world,  # pyright: ignore[reportArgumentType]
                            0,
                            body,  # pyright: ignore[reportArgumentType]
                            axis,
                            sign * forces[contact][0] * frames[contact][0, axis],  # pyright: ignore[reportIndexIssue]
                        )


def _binding_names(label: str, names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise ValueError(f"{label} must be a sequence of nonempty names")
    result = tuple(names)
    if any(not isinstance(name, str) or not name for name in result):
        raise ValueError(f"{label} must contain nonempty names")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicate names")
    return result


class MjwarpPhysicsStepTensorDataView(PhysicsStepTensorDataView):
    """Stable aliases sampled before and after each MJWarp native substep."""

    def __init__(
        self,
        backend: Any,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        history_length: int,
    ) -> None:
        self.joint_names = _binding_names("joint_names", joint_names)
        self.body_names = _binding_names("body_names", body_names)
        if not self.joint_names and not self.body_names:
            raise ValueError("physics-step tensor data requires at least one joint or body")
        if (
            isinstance(history_length, (bool, np.bool_))
            or not isinstance(history_length, (int, np.integer))
            or history_length < 1
        ):
            raise ValueError("history_length must be a positive integer")
        self.history_length = int(history_length)
        self._backend = backend
        self._n = backend._num_envs
        self._j = len(self.joint_names)
        self._b = len(self.body_names)
        joint_ids = (
            backend.get_joint_state_qvel_indices(self.joint_names) if self._j else []
        )
        if len(joint_ids) != self._j:
            raise ValueError("physics-step tensor data requires single-DoF joints")
        selected = np.full(backend._cpu_model.nbody, -1, dtype=np.int32)
        for index, name in enumerate(self.body_names):
            if name not in backend._body_ids:
                raise ValueError(f"Body {name!r} not found in mjwarp model")
            selected[backend._body_ids[name]] = index

        self._device = backend._device_data.qvel.device
        self._ids = wp.array(np.asarray(joint_ids, dtype=np.int32), dtype=int, device=self._device)
        self._selected = wp.array(selected, dtype=int, device=self._device)
        self._geom_body = wp.array(
            backend._cpu_model.geom_bodyid, dtype=int, device=self._device
        )
        self._before = wp.zeros((self._n, self._j), dtype=float, device=self._device)
        self._acc = wp.zeros((self._n, self._j), dtype=float, device=self._device)
        self._history = wp.zeros(
            (self._n, self.history_length, self._b, 3), dtype=float, device=self._device
        )
        capacity = backend._device_data.contact.geom.shape[0]
        self._contacts = wp.array(
            np.arange(capacity, dtype=np.int32), dtype=int, device=self._device
        )
        self._forces = wp.zeros(
            capacity,
            dtype=wp.spatial_vector,  # pyright: ignore[reportArgumentType]
            device=self._device,
        )
        self._acc_tensor = backend._torch_view(self._acc)
        self._history_tensor = backend._torch_view(self._history)
        self._before_tensor = backend._torch_view(self._before)

    def before_step(self) -> None:
        if self._j:
            wp.launch(
                _velocity_sample,
                (self._n, self._j),
                inputs=[self._backend._device_data.qvel, self._ids, self._before],
                device=self._device,
            )

    def after_step(self) -> None:
        backend = self._backend
        data = backend._device_data
        if self._j:
            wp.launch(
                _acceleration,
                (self._n, self._j),
                inputs=[data.qvel, self._ids, self._before, backend._sim_dt, self._acc],
                device=self._device,
            )
        if self._b:
            wp.launch(
                _shift,
                (self._n, self._b, 3),
                inputs=[self._history, self.history_length],
                device=self._device,
            )
            if self._contacts.size:
                backend._mujoco_warp.contact_force(
                    backend._device_model, data, self._contacts, False, self._forces
                )
                wp.launch(
                    _accumulate,
                    self._contacts.size,
                    inputs=[
                        data.nacon,
                        data.contact.worldid,
                        data.contact.geom,
                        data.contact.frame,
                        self._forces,
                        self._geom_body,
                        self._selected,
                        self._history,
                    ],
                    device=self._device,
                )

    @property
    def joint_acceleration(self) -> Any:
        self._backend._require_entity_healthy()
        return self._acc_tensor

    @property
    def contact_force_history(self) -> Any:
        self._backend._require_entity_healthy()
        return self._history_tensor

    def reset(self, env_ids: Any | slice | None = None) -> None:
        import torch

        self._backend._require_entity_healthy()
        device = self._acc_tensor.device
        if env_ids is None:
            rows = torch.arange(self._n, dtype=torch.int64, device=device)
        elif isinstance(env_ids, slice):
            for value in (env_ids.start, env_ids.stop, env_ids.step):
                if value is not None and (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                ):
                    raise ValueError("env_ids slice bounds must be integers")
            rows = torch.as_tensor(
                np.arange(self._n, dtype=np.int64)[env_ids].copy(), device=device
            )
        else:
            if not isinstance(env_ids, torch.Tensor):
                raise TypeError("env_ids must be a 1-D int32 or int64 Torch tensor")
            if env_ids.ndim != 1 or env_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("env_ids must be a 1-D int32 or int64 Torch tensor")
            if env_ids.device != device:
                raise ValueError(f"env_ids must live on {device}, got {env_ids.device}")
            rows = env_ids.to(dtype=torch.int64)
        if not rows.numel():
            return
        ordered = rows.sort().values
        if bool((ordered[0] < 0) | (ordered[-1] >= self._n)) or bool(
            (ordered[1:] == ordered[:-1]).any()
        ):
            raise ValueError(f"env_ids must contain unique values in [0, {self._n})")
        with self._reset_stream():
            self._acc_tensor.index_fill_(0, rows, 0.0)
            self._history_tensor.index_fill_(0, rows, 0.0)
            self._before_tensor.index_fill_(0, rows, 0.0)

    @contextmanager
    def _reset_stream(self):
        """Order reset writes with native sampling without a host barrier."""
        import torch

        device = self._acc_tensor.device
        if device.type != "cuda":
            yield
            return
        caller = torch.cuda.current_stream(device)
        owner = self._backend._torch_execution_stream(device)
        owner.wait_stream(caller)
        with torch.cuda.stream(owner):
            yield
        caller.wait_stream(owner)

    def _reset_mask(self, reset_mask: Any) -> None:
        """Clear an already validated backend mask without row compaction."""
        with self._reset_stream():
            self._acc_tensor.masked_fill_(reset_mask[:, None], 0.0)
            self._history_tensor.masked_fill_(reset_mask[:, None, None, None], 0.0)
            self._before_tensor.masked_fill_(reset_mask[:, None], 0.0)

    def _reset_host_rows(self, rows: np.ndarray) -> None:
        """Bridge only the selected host row identifiers at a reset barrier."""
        import torch

        self.reset(torch.as_tensor(rows, dtype=torch.int64, device=self._acc_tensor.device))
