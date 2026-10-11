"""Backend-owned packed reset transaction composition."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from unisim.dr.types import (
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_IQUAT,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_DOF_DAMPING,
    RESET_TERM_DOF_FRICTIONLOSS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GEOM_SIZE,
    RESET_TERM_GEOM_SOLIMP,
    RESET_TERM_GEOM_SOLREF,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    TensorResetRandomizationPayload,
    _validate_reset_term,
)

PACKED_RESET_MODEL_FIELDS = frozenset(
    {
        RESET_TERM_GRAVITY,
        RESET_TERM_BODY_IQUAT,
        RESET_TERM_BODY_INERTIA,
        RESET_TERM_BODY_IPOS,
        RESET_TERM_BODY_MASS,
        RESET_TERM_DOF_ARMATURE,
        RESET_TERM_DOF_DAMPING,
        RESET_TERM_DOF_FRICTIONLOSS,
        RESET_TERM_GEOM_FRICTION,
        RESET_TERM_GEOM_SIZE,
        RESET_TERM_GEOM_SOLIMP,
        RESET_TERM_GEOM_SOLREF,
        RESET_TERM_KP,
        RESET_TERM_KD,
    }
)
PACKED_RESET_HOST_ONLY_FIELDS = frozenset({RESET_TERM_BASE_MASS, RESET_TERM_BASE_COM})
PACKED_RESET_PERSISTENT_CHANNELS = frozenset(
    {"ctrl", "act", "qfrc_applied", "xfrc_applied", "qacc_warmstart"}
)
_PACKED_RESET_CANONICAL_NDIMS: dict[str, int] = {
    RESET_TERM_GRAVITY: 1,
    RESET_TERM_BODY_IQUAT: 2,
    RESET_TERM_BODY_INERTIA: 2,
    RESET_TERM_BODY_IPOS: 2,
    RESET_TERM_BODY_MASS: 1,
    RESET_TERM_DOF_ARMATURE: 1,
    RESET_TERM_DOF_DAMPING: 1,
    RESET_TERM_DOF_FRICTIONLOSS: 1,
    RESET_TERM_GEOM_FRICTION: 2,
    RESET_TERM_GEOM_SIZE: 2,
    RESET_TERM_GEOM_SOLIMP: 2,
    RESET_TERM_GEOM_SOLREF: 2,
    RESET_TERM_KP: 1,
    RESET_TERM_KD: 1,
}


class TorchPackedResetComposer:
    """Compose device-resident model-field writes and commit them once."""

    def __init__(self, backend: Any, *, device: Any | None = None) -> None:
        import torch

        capabilities = backend.get_tensor_capabilities()
        if not capabilities.packed_reset:
            raise NotImplementedError(
                f"{backend.backend_type} does not support packed reset transactions"
            )
        if not capabilities.device_reset_randomization:
            raise NotImplementedError(
                "packed reset transactions require device-resident reset randomization"
            )
        if device is None:
            accepted = capabilities.torch_devices
            if not accepted:
                raise ValueError("packed reset backend declared no Torch devices")
            family = accepted[0].split(":", maxsplit=1)[0]
            if family == "cpu":
                device = torch.device("cpu")
            elif family == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA packed reset requested but CUDA is unavailable")
                device = torch.device("cuda", index=torch.cuda.current_device())
            else:
                raise ValueError(f"unsupported packed reset Torch device family {family!r}")
        self.device = torch.device(device)
        self.backend = backend
        self._torch = torch
        self._baselines: dict[str, Any] = {}
        self._staged: dict[str, Any] = {}
        self._mocap_staged: dict[str, tuple[Any, Any]] = {}
        self._default_qpos: Any | None = None
        self._default_qvel: Any | None = None
        self._qpos: Any | None = None
        self._qvel: Any | None = None
        self._state_writes_staged = False

    def baseline(self, field: str) -> Any:
        """Return the committed full-width baseline for one model field."""
        self._validate_field(field)
        if field not in self._baselines:
            values = np.asarray(self.backend.get_reset_term_default(field), dtype=np.float32)
            canonical_ndim = _PACKED_RESET_CANONICAL_NDIMS[field]
            if values.ndim not in (canonical_ndim, canonical_ndim + 1):
                raise ValueError(
                    f"packed reset {field} default must have canonical rank "
                    f"{canonical_ndim} or per-environment rank {canonical_ndim + 1}; "
                    f"got {values.ndim}"
                )
            if values.ndim == canonical_ndim:
                values = np.broadcast_to(values, (int(self.backend.num_envs), *values.shape))
            values = np.array(values, copy=True)
            self._baselines[field] = self._torch.as_tensor(values, device=self.device)
        return self._baselines[field].detach()

    def write_model_field(
        self,
        field: str,
        rows: Any,
        values: Any,
        *,
        columns: Any | None = None,
    ) -> None:
        """Merge a selected-row (optionally selected-column) final-value write."""
        self._validate_field(field)
        rows = self._validate_rows(rows)
        values = self._validate_values(field, values, rows, columns)
        target = self._staged.get(field)
        if target is None:
            target = self.baseline(field).clone()
        selected = target.index_select(0, rows)
        if columns is None:
            selected.copy_(values)
        else:
            selected[:, columns] = values
        target.index_copy_(0, rows, selected)
        self._staged[field] = target

    def discard(self, field: str | None = None) -> None:
        """Discard staged writes before commit, or clear the whole transaction."""
        if field is None:
            self._staged.clear()
            self._mocap_staged.clear()
            return
        self._validate_field(field)
        self._staged.pop(field, None)

    def promote_baseline(
        self,
        field: str,
        rows: Any,
        values: Any,
        *,
        columns: Any | None = None,
    ) -> None:
        """Promote successfully committed final values into one field baseline."""
        self._validate_field(field)
        rows = self._validate_rows(rows)
        values = self._validate_values(field, values, rows, columns)
        target = self.baseline(field)
        selected = target.index_select(0, rows)
        if columns is None:
            selected.copy_(values)
        else:
            selected[:, columns] = values
        target.index_copy_(0, rows, selected)

    def set_state_default(self, qpos: Any, qvel: Any) -> None:
        """Install immutable one-row default state on the transaction device."""
        default_qpos = self._validate_state_vector(qpos, "default qpos")
        default_qvel = self._validate_state_vector(qvel, "default qvel")
        self._default_qpos = default_qpos
        self._default_qvel = default_qvel
        self._qpos = default_qpos.repeat(int(self.backend.num_envs), 1).clone()
        self._qvel = default_qvel.repeat(int(self.backend.num_envs), 1).clone()

    def begin_state(self, rows: Any) -> None:
        """Restore default state staging for the rows selected by one reset."""
        rows = self._validate_rows(rows)
        if self._qpos is None or self._qvel is None:
            raise RuntimeError("packed reset state defaults must be installed before begin_state")
        assert self._default_qpos is not None and self._default_qvel is not None
        count = int(rows.shape[0])
        self._qpos.index_copy_(0, rows, self._default_qpos.expand(count, -1))
        self._qvel.index_copy_(0, rows, self._default_qvel.expand(count, -1))
        self._state_writes_staged = False

    def state_views(self) -> tuple[Any, Any]:
        """Return the full-width device qpos/qvel staging tensors."""
        if self._qpos is None or self._qvel is None:
            raise RuntimeError("packed reset state defaults must be installed first")
        return self._qpos, self._qvel

    def write_state(
        self,
        rows: Any,
        *,
        qpos_columns: Any | None = None,
        qpos_values: Any | None = None,
        qvel_columns: Any | None = None,
        qvel_values: Any | None = None,
    ) -> None:
        """Merge selected-row (optionally selected-column) state writes."""
        rows = self._validate_rows(rows)
        qpos, qvel = self.state_views()
        if qpos_values is not None:
            qpos_columns_tensor = self._column_tensor(qpos_columns, qpos.shape[1], "qpos")
            qpos[rows[:, None], qpos_columns_tensor[None, :]] = self._validate_state_values(
                qpos_values, rows, qpos_columns, qpos.shape[1], "qpos"
            )
        if qvel_values is not None:
            qvel_columns_tensor = self._column_tensor(qvel_columns, qvel.shape[1], "qvel")
            qvel[rows[:, None], qvel_columns_tensor[None, :]] = self._validate_state_values(
                qvel_values, rows, qvel_columns, qvel.shape[1], "qvel"
            )
        self._state_writes_staged = True

    def commit(self, rows: Any) -> dict:
        """Commit staged state and model-field writes through one boundary."""
        rows = self._validate_rows(rows)
        qpos, qvel = self.state_views()
        return self.commit_state(rows, qpos.index_select(0, rows), qvel.index_select(0, rows))

    def commit_state(self, rows: Any, qpos: Any, qvel: Any) -> dict:
        """Commit state plus staged model-field writes for these rows.

        Rows outside this commit keep their staged writes until ``discard()``;
        baselines are promoted only for the committed rows.
        """
        rows = self._validate_rows(rows)
        started = time.perf_counter()
        payload = (
            None
            if not self._staged
            else TensorResetRandomizationPayload(
                **{
                    field: tensor.index_select(0, rows).contiguous()
                    for field, tensor in self._staged.items()
                }
            )
        )
        compose_ms = (time.perf_counter() - started) * 1000.0
        commit_started = time.perf_counter()
        result = self.backend.set_state_tensor(rows, qpos, qvel, payload)
        self._apply_mocap_poses()
        commit_ms = (time.perf_counter() - commit_started) * 1000.0
        for field, staged in self._staged.items():
            self.baseline(field).index_copy_(0, rows, staged.index_select(0, rows))
        timing = dict((result or {}).get("timing", {}))
        timing.update(
            {
                "packed_reset_compose_ms": compose_ms,
                "packed_reset_commit_ms": commit_ms,
                "packed_reset_total_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        return {"timing": timing}

    def write_mocap_pose(self, body_name: str, rows: Any, poses: Any) -> None:
        """Stage selected-row world-space poses for one mocap body."""
        capabilities = self.backend.get_tensor_capabilities()
        if not capabilities.packed_reset_mocap:
            raise NotImplementedError(
                f"{self.backend.backend_type} does not support packed reset mocap writes"
            )
        if not isinstance(body_name, str) or not body_name:
            raise TypeError("packed reset mocap body_name must be a non-empty string")
        rows = self._validate_rows(rows)
        poses = self._validate_mocap_poses(poses, rows)
        previous = self._mocap_staged.get(body_name)
        if previous is None:
            self._mocap_staged[body_name] = (rows, poses)
            return

        previous_rows, previous_poses = previous
        keep = ~self._torch.isin(previous_rows, rows)
        self._mocap_staged[body_name] = (
            self._torch.cat((previous_rows[keep], rows)),
            self._torch.cat((previous_poses[keep], poses)),
        )

    def commit_mocap_poses(self) -> dict:
        """Commit only staged mocap poses without a generalized-state reset."""
        started = time.perf_counter()
        self._apply_mocap_poses()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "timing": {
                "packed_reset_compose_ms": 0.0,
                "packed_reset_commit_ms": elapsed_ms,
                "packed_reset_total_ms": elapsed_ms,
            }
        }

    def _apply_mocap_poses(self) -> None:
        for body_name in sorted(self._mocap_staged):
            rows, poses = self._mocap_staged[body_name]
            self.backend.set_mocap_pose_tensor(body_name, rows, poses)
        self._mocap_staged.clear()

    def _validate_field(self, field: str) -> None:
        if field in PACKED_RESET_HOST_ONLY_FIELDS:
            raise NotImplementedError(
                f"reset field {field!r} is a host-only delta-merge term and is "
                "deliberately excluded from device packed reset transactions"
            )
        _validate_reset_term(field)
        if field not in PACKED_RESET_MODEL_FIELDS:
            raise NotImplementedError(f"packed reset transactions do not support {field!r}")
        capabilities = self.backend.get_tensor_capabilities()
        if field not in capabilities.packed_reset_fields:
            raise NotImplementedError(
                f"{self.backend.backend_type} does not support packed reset field {field!r}"
            )

    def _validate_rows(self, rows: Any) -> Any:
        if not isinstance(rows, self._torch.Tensor):
            raise TypeError("packed reset rows must be a torch.Tensor")
        if rows.ndim != 1 or rows.dtype != self._torch.int64 or not rows.is_contiguous():
            raise TypeError("packed reset rows must be a contiguous 1-D int64 tensor")
        if rows.device != self.device:
            raise ValueError(f"packed reset rows must live on {self.device}")
        return rows

    def _validate_values(
        self,
        field: str,
        values: Any,
        rows: Any,
        columns: Any | None,
    ) -> Any:
        if not isinstance(values, self._torch.Tensor):
            raise TypeError(f"packed reset {field} values must be a torch.Tensor")
        if values.dtype != self._torch.float32 or not values.is_contiguous():
            raise TypeError(f"packed reset {field} values must be contiguous float32")
        if values.device != self.device:
            raise ValueError(f"packed reset {field} values must live on {self.device}")
        baseline = self.baseline(field)
        expected = (
            (int(rows.shape[0]), *tuple(baseline.shape[1:]))
            if columns is None
            else (int(rows.shape[0]), int(len(columns)), *tuple(baseline.shape[2:]))
        )
        if tuple(values.shape) != expected:
            raise ValueError(f"packed reset {field} values must have shape {expected}")
        if not bool(self._torch.isfinite(values).all()):
            raise ValueError(f"packed reset {field} values must be finite")
        return values

    def _validate_mocap_poses(self, poses: Any, rows: Any) -> Any:
        if not isinstance(poses, self._torch.Tensor):
            raise TypeError("packed reset mocap poses must be a torch.Tensor")
        expected = (int(rows.shape[0]), 7)
        if poses.dtype != self._torch.float32 or not poses.is_contiguous():
            raise TypeError("packed reset mocap poses must be contiguous float32")
        if poses.device != self.device:
            raise ValueError(f"packed reset mocap poses must live on {self.device}")
        if tuple(poses.shape) != expected:
            raise ValueError(f"packed reset mocap poses must have shape {expected}")
        if not bool(self._torch.isfinite(poses).all()):
            raise ValueError("packed reset mocap poses must be finite")
        norms = self._torch.linalg.vector_norm(poses[:, 3:], dim=1)
        if not bool(
            self._torch.allclose(norms, self._torch.ones_like(norms), rtol=1e-5, atol=1e-6)
        ):
            raise ValueError("packed reset mocap poses require unit wxyz quaternions")
        return poses

    def _column_tensor(
        self, columns: Any | None, width: int, name: str
    ) -> Any:
        if columns is None:
            return self._torch.arange(width, dtype=self._torch.int64, device=self.device)
        column_tensor = self._torch.as_tensor(columns, dtype=self._torch.int64, device=self.device)
        if column_tensor.ndim != 1:
            raise TypeError(f"packed reset {name} columns must be one-dimensional")
        if column_tensor.numel() and (
            bool((column_tensor < 0).any()) or bool((column_tensor >= width).any())
        ):
            raise ValueError(f"packed reset {name} columns must be within [0, {width})")
        return column_tensor

    def _validate_state_vector(self, values: Any, name: str) -> Any:
        if not isinstance(values, self._torch.Tensor):
            raise TypeError(f"packed reset {name} must be a torch.Tensor")
        if values.ndim != 1 or values.dtype != self._torch.float32 or not values.is_contiguous():
            raise TypeError(f"packed reset {name} must be a contiguous 1-D float32 tensor")
        if values.device != self.device:
            raise ValueError(f"packed reset {name} must live on {self.device}")
        if not bool(self._torch.isfinite(values).all()):
            raise ValueError(f"packed reset {name} must be finite")
        return values

    def _validate_state_values(
        self,
        values: Any,
        rows: Any,
        columns: Any | None,
        width: int,
        name: str,
    ) -> Any:
        if not isinstance(values, self._torch.Tensor):
            raise TypeError(f"packed reset {name} values must be a torch.Tensor")
        if values.dtype != self._torch.float32 or not values.is_contiguous():
            raise TypeError(f"packed reset {name} values must be contiguous float32")
        if values.device != self.device:
            raise ValueError(f"packed reset {name} values must live on {self.device}")
        column_count = width if columns is None else int(len(columns))
        expected = (int(rows.shape[0]), column_count)
        if tuple(values.shape) != expected:
            raise ValueError(f"packed reset {name} values must have shape {expected}")
        if not bool(self._torch.isfinite(values).all()):
            raise ValueError(f"packed reset {name} values must be finite")
        return values
