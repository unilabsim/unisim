"""Backend-owned packed reset transaction composition."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from unisim.dr.types import (
    RESET_TERM_SPECS,
    TensorResetRandomizationPayload,
    _validate_reset_term,
)

# Field tables derive from the single authoritative RESET_TERM_SPECS source in
# unisim.dr.types; every reset term is a packable model field, including the
# delta terms whose merge with the baseline happens in Torch at commit time.
PACKED_RESET_MODEL_FIELDS = frozenset(spec.name for spec in RESET_TERM_SPECS)
PACKED_RESET_PERSISTENT_CHANNELS = frozenset(
    {"ctrl", "act", "qfrc_applied", "xfrc_applied", "qacc_warmstart"}
)
_PACKED_RESET_CANONICAL_NDIMS: dict[str, int] = {
    spec.name: spec.tail_ndim for spec in RESET_TERM_SPECS
}
_PACKED_RESET_DELTA_FIELDS = frozenset(spec.name for spec in RESET_TERM_SPECS if spec.is_delta)

# One verdict slot per validated write target keeps deferred failures
# attributable to a field without a per-write host readback. ``<name>`` latches
# nonfinite values; ``<name>:columns`` latches out-of-range column selections.
_PACKED_RESET_VALIDATION_NAMES = (*sorted(PACKED_RESET_MODEL_FIELDS), "qpos", "qvel")
PACKED_RESET_VALIDATION_SLOTS: tuple[str, ...] = tuple(
    label for name in _PACKED_RESET_VALIDATION_NAMES for label in (name, f"{name}:columns")
) + ("mocap_poses", "mocap_quaternion_norm")
_PACKED_RESET_SLOT_INDEX = {name: index for index, name in enumerate(PACKED_RESET_VALIDATION_SLOTS)}
PACKED_RESET_VALIDATION_MODES = ("deferred", "strict")


def _expand_mask(mask: Any, ndim: int) -> Any:
    return mask if ndim == 1 else mask.reshape(mask.shape + (1,) * (ndim - 1))


class PackedResetTransaction:
    """Compose device-resident model-field writes and commit them once.

    Validation of device-resident write payloads is fused: in the default
    ``deferred`` mode every ``write_*`` method only ORs its finite/shape
    verdict into a fixed device-side slot table (one slot per field), so a
    k-field reset performs zero host synchronizations. The owner polls at its
    own cycle boundary through :meth:`has_pending_verdict` /
    :meth:`take_verdict` / :meth:`raise_for_verdict`, which perform a single
    host read. ``strict`` mode restores the legacy eager behavior and raises
    at the offending write, for debugging and non-training consumers.
    """

    def __init__(
        self,
        backend: Any,
        *,
        device: Any | None = None,
        validation: str = "deferred",
    ) -> None:
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
        self._validation = self._validate_validation_mode(validation)
        self._verdict_slots = torch.zeros(
            (len(PACKED_RESET_VALIDATION_SLOTS),), dtype=torch.bool, device=self.device
        )
        self._baselines: dict[str, Any] = {}
        self._staged: dict[str, Any] = {}
        self._mocap_staged: dict[str, tuple[Any, Any]] = {}
        # Device column selections keyed by their host spelling: re-uploading
        # a host list every reset is a pageable H2D copy, which a CUDA graph
        # capture rejects.
        self._column_cache: dict[tuple[str, int, tuple[int, ...] | None], Any] = {}
        self._default_qpos: Any | None = None
        self._default_qvel: Any | None = None
        self._qpos: Any | None = None
        self._qvel: Any | None = None
        self._state_writes_staged = False

    @staticmethod
    def _validate_validation_mode(validation: str) -> str:
        if validation not in PACKED_RESET_VALIDATION_MODES:
            raise ValueError(
                f"packed reset validation must be one of {PACKED_RESET_VALIDATION_MODES}; "
                f"got {validation!r}"
            )
        return validation

    @property
    def validation(self) -> str:
        """Validation mode: ``deferred`` latches device verdicts, ``strict`` raises eagerly."""
        return self._validation

    @validation.setter
    def validation(self, mode: str) -> None:
        self._validation = self._validate_validation_mode(mode)

    @property
    def device_verdict(self) -> Any:
        """Persistent per-slot device verdict table, without a host transfer."""
        return self._verdict_slots

    def has_pending_verdict(self) -> bool:
        """Return whether any write latched a violation, with one host read."""
        return bool(self._verdict_slots.any())

    def take_verdict(self) -> tuple[str, ...]:
        """Return and clear the latched violation slots with a single host read."""
        slots = self._verdict_slots.detach().to(device="cpu", copy=True)
        if not bool(slots.any()):
            return ()
        self._verdict_slots.zero_()
        flagged = slots.tolist()
        return tuple(
            name for name, is_set in zip(PACKED_RESET_VALIDATION_SLOTS, flagged) if is_set
        )

    def raise_for_verdict(self) -> None:
        """Raise ValueError naming every latched violation slot, then clear them."""
        slots = self.take_verdict()
        if slots:
            raise ValueError("packed reset deferred validation failed for: " + ", ".join(slots))

    def _capturing(self) -> bool:
        return getattr(self.backend, "_external_capture", None) is not None

    def _record_violation(self, slot: str, violation: Any, message: str) -> None:
        """Latch one device-side verdict; strict mode raises at the write instead.

        Inside an external CUDA graph capture strict mode cannot read the
        device, so the verdict is latched exactly as in deferred mode.
        """
        if self._validation == "strict" and not self._capturing():
            if bool(violation):
                raise ValueError(message)
            return
        self._verdict_slots[_PACKED_RESET_SLOT_INDEX[slot]].logical_or_(violation)

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

    def _new_staged(self, field: str) -> Any:
        """Return a fresh staged table: zeros for delta terms, baseline copy else."""
        if field in _PACKED_RESET_DELTA_FIELDS:
            return self._torch.zeros_like(self.baseline(field))
        return self.baseline(field).clone()

    def _staged_final(self, field: str) -> Any:
        """Return the staged table as final absolute values.

        Delta terms merge in Torch at commit time: ``final = baseline + delta``.
        The baseline itself is cold NumPy metadata materialized from
        ``get_reset_term_default`` and promoted through prior commits.
        """
        staged = self._staged[field]
        if field in _PACKED_RESET_DELTA_FIELDS:
            return self.baseline(field) + staged
        return staged

    def write_model_field(
        self,
        field: str,
        rows: Any,
        values: Any,
        *,
        columns: Any | None = None,
    ) -> None:
        """Merge a selected-row (optionally selected-column) final-value write.

        Delta-term values are staged as deltas and merged with the baseline in
        Torch at commit time (``final = baseline + delta``).
        """
        self._validate_field(field)
        rows = self._validate_rows(rows)
        values = self._validate_values(field, values, rows, columns)
        target = self._staged.get(field)
        if target is None:
            target = self._new_staged(field)
        selected = target.index_select(0, rows)
        if columns is None:
            selected.copy_(values)
        else:
            selected[:, columns] = values
        target.index_copy_(0, rows, selected)
        self._staged[field] = target

    def write_model_field_mask(
        self,
        field: str,
        reset_mask: Any,
        values: Any,
        *,
        columns: Any | None = None,
    ) -> None:
        """Merge full-width final values selected by a fixed-width bool mask."""
        self._validate_field(field)
        mask = self._validate_state_mask(reset_mask)
        baseline = self.baseline(field)
        if values.ndim < 1 or values.shape[0] != int(baseline.shape[0]):
            raise ValueError(
                f"packed reset {field} values must have environment width "
                f"{int(baseline.shape[0])}; got {tuple(values.shape)}"
            )
        target = self._staged.get(field)
        if target is None:
            target = self._new_staged(field)
        if columns is None:
            target.copy_(self._torch.where(_expand_mask(mask, values.ndim), values, target))
        else:
            column_tensor = self._column_tensor(columns, target.shape[1], field)
            selected = target[:, column_tensor]
            target[:, column_tensor] = self._torch.where(
                _expand_mask(mask, values.ndim),
                values,
                selected,
            )
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

    def begin_state_mask(self, reset_mask: Any) -> None:
        """Restore default state staging for a fixed-width bool reset mask.

        It keeps the staging tensors full width and never materializes true-row
        indices. Backends still own selected-row publication at commit.
        """
        mask = self._validate_state_mask(reset_mask)
        if self._qpos is None or self._qvel is None:
            raise RuntimeError("packed reset state defaults must be installed before begin_state")
        assert self._default_qpos is not None and self._default_qvel is not None
        self._torch.where(
            mask[:, None],
            self._default_qpos.to(device=self.device),
            self._qpos,
            out=self._qpos,
        )
        self._torch.where(
            mask[:, None],
            self._default_qvel.to(device=self.device),
            self._qvel,
            out=self._qvel,
        )
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

    def write_state_mask(
        self,
        reset_mask: Any,
        *,
        qpos_columns: Any | None = None,
        qpos_values: Any,
        qvel_columns: Any | None = None,
        qvel_values: Any,
    ) -> None:
        """Merge full-width selected-column state values using a bool mask.

        Values have the full environment width. False rows preserve the staged
        state, while true rows consume the supplied columns. No true-row tensor
        is materialized by the transaction.
        """
        mask = self._validate_state_mask(reset_mask)
        qpos, qvel = self.state_views()
        if qpos_values is not None:
            self._write_state_columns_mask(
                qpos,
                mask,
                qpos_columns,
                qpos_values,
                width=qpos.shape[1],
                name="qpos",
            )
        if qvel_values is not None:
            self._write_state_columns_mask(
                qvel,
                mask,
                qvel_columns,
                qvel_values,
                width=qvel.shape[1],
                name="qvel",
            )
        self._state_writes_staged = True

    def commit(self, rows: Any) -> dict:
        """Commit staged state and model-field writes through one boundary."""
        rows = self._validate_rows(rows)
        qpos, qvel = self.state_views()
        return self.commit_state(rows, qpos.index_select(0, rows), qvel.index_select(0, rows))

    def commit_mask(self, reset_mask: Any) -> dict:
        """Commit the full-width staged state selected by a bool mask."""
        mask = self._validate_state_mask(reset_mask)
        # begin_state_mask marks defaults on selected rows even before an owner
        # issues a write; that is a valid state-only commit target.
        self._state_writes_staged = True
        qpos, qvel = self.state_views()
        return self.commit_state_mask(mask, qpos, qvel)

    def commit_state(self, rows: Any, qpos: Any, qvel: Any) -> dict:
        """Commit state plus staged model-field writes for these rows.

        Rows outside this commit keep their staged writes until ``discard()``;
        baselines are promoted only for the committed rows.
        """
        rows = self._validate_rows(rows)
        started = time.perf_counter()
        finals = {field: self._staged_final(field) for field in self._staged}
        payload = (
            None
            if not finals
            else TensorResetRandomizationPayload(
                **{
                    field: tensor.index_select(0, rows).contiguous()
                    for field, tensor in finals.items()
                }
            )
        )
        compose_ms = (time.perf_counter() - started) * 1000.0
        commit_started = time.perf_counter()
        result = self.backend.set_state_tensor(rows, qpos, qvel, payload)
        self._apply_mocap_poses()
        commit_ms = (time.perf_counter() - commit_started) * 1000.0
        for field, staged_final in finals.items():
            self.baseline(field).index_copy_(0, rows, staged_final.index_select(0, rows))
        timing = dict((result or {}).get("timing", {}))
        timing.update(
            {
                "packed_reset_compose_ms": compose_ms,
                "packed_reset_commit_ms": commit_ms,
                "packed_reset_total_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        return {"timing": timing}

    def commit_state_mask(self, reset_mask: Any, qpos: Any, qvel: Any) -> dict:
        """Commit full-width state plus staged writes selected by a bool mask."""
        mask = self._validate_state_mask(reset_mask)
        started = time.perf_counter()
        finals = {field: self._staged_final(field) for field in self._staged}
        payload = (
            None
            if not finals
            else TensorResetRandomizationPayload(
                **{
                    field: self._torch.where(
                        _expand_mask(mask, tensor.ndim), tensor, self.baseline(field)
                    ).contiguous()
                    for field, tensor in finals.items()
                }
            )
        )
        compose_ms = (time.perf_counter() - started) * 1000.0
        commit_started = time.perf_counter()
        result = self.backend.set_state_tensor_mask(mask, qpos, qvel, payload, full_refresh=False)
        self._apply_mocap_poses()
        commit_ms = (time.perf_counter() - commit_started) * 1000.0
        for field, staged_final in finals.items():
            self.baseline(field).copy_(
                self._torch.where(
                    _expand_mask(mask, staged_final.ndim), staged_final, self.baseline(field)
                )
            )
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

    def _validate_state_mask(self, reset_mask: Any) -> Any:
        if not isinstance(reset_mask, self._torch.Tensor):
            raise TypeError("packed reset mask must be a torch.Tensor")
        if (
            reset_mask.ndim != 1
            or reset_mask.dtype != self._torch.bool
            or not reset_mask.is_contiguous()
        ):
            raise TypeError("packed reset mask must be a contiguous 1-D bool tensor")
        if tuple(reset_mask.shape) != (int(self.backend.num_envs),):
            raise ValueError(f"packed reset mask must have shape {(int(self.backend.num_envs),)}")
        if reset_mask.device != self.device:
            raise ValueError(f"packed reset mask must live on {self.device}")
        return reset_mask

    def _write_state_columns_mask(
        self,
        state: Any,
        reset_mask: Any,
        columns: Any | None,
        values: Any,
        *,
        width: int,
        name: str,
    ) -> None:
        if not isinstance(values, self._torch.Tensor):
            raise TypeError(f"packed reset {name} values must be a torch.Tensor")
        if values.dtype != self._torch.float32 or not values.is_contiguous():
            raise TypeError(f"packed reset {name} values must be contiguous float32")
        if values.device != self.device:
            raise ValueError(f"packed reset {name} values must live on {self.device}")
        column_tensor = self._column_tensor(columns, width, name)
        expected = (int(reset_mask.shape[0]), int(column_tensor.numel()))
        if tuple(values.shape) != expected:
            raise ValueError(f"packed reset {name} values must have shape {expected}")
        self._record_violation(
            name,
            ~self._torch.isfinite(values).all(),
            f"packed reset {name} values must be finite",
        )
        expanded_mask = _expand_mask(reset_mask, state.ndim)
        selected = state[:, column_tensor]
        merged = self._torch.where(expanded_mask, values, selected)
        state[:, column_tensor] = merged

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
        self._record_violation(
            field,
            ~self._torch.isfinite(values).all(),
            f"packed reset {field} values must be finite",
        )
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
        self._record_violation(
            "mocap_poses",
            ~self._torch.isfinite(poses).all(),
            "packed reset mocap poses must be finite",
        )
        norms = self._torch.linalg.vector_norm(poses[:, 3:], dim=1)
        # Matches allclose(norms, ones, rtol=1e-5, atol=1e-6) without a host read.
        self._record_violation(
            "mocap_quaternion_norm",
            ((norms - 1.0).abs() > 1e-6 + 1e-5).any(),
            "packed reset mocap poses require unit wxyz quaternions",
        )
        return poses

    def _column_tensor(self, columns: Any | None, width: int, name: str) -> Any:
        if columns is None:
            key = (name, width, None)
            cached = self._column_cache.get(key)
            if cached is None:
                cached = self._torch.arange(width, dtype=self._torch.int64, device=self.device)
                self._column_cache[key] = cached
            return cached
        if isinstance(columns, self._torch.Tensor) and columns.device == self.device:
            column_tensor = columns.to(dtype=self._torch.int64)
        else:
            host = np.asarray(columns)
            if host.ndim != 1:
                raise TypeError(f"packed reset {name} columns must be one-dimensional")
            key = (name, width, tuple(int(c) for c in host.tolist()))
            column_tensor = self._column_cache.get(key)
            if column_tensor is None:
                column_tensor = self._torch.as_tensor(
                    host, dtype=self._torch.int64, device=self.device
                )
                self._column_cache[key] = column_tensor
        if column_tensor.ndim != 1:
            raise TypeError(f"packed reset {name} columns must be one-dimensional")
        if column_tensor.numel():
            self._record_violation(
                f"{name}:columns",
                ((column_tensor < 0) | (column_tensor >= width)).any(),
                f"packed reset {name} columns must be within [0, {width})",
            )
            if self._validation != "strict" or self._capturing():
                # Deferred mode must not feed out-of-range indices to the
                # device write; the latched verdict reports the violation.
                column_tensor = column_tensor.clamp(0, width - 1)
        return column_tensor

    def _validate_state_vector(self, values: Any, name: str) -> Any:
        if not isinstance(values, self._torch.Tensor):
            raise TypeError(f"packed reset {name} must be a torch.Tensor")
        if values.ndim != 1 or values.dtype != self._torch.float32 or not values.is_contiguous():
            raise TypeError(f"packed reset {name} must be a contiguous 1-D float32 tensor")
        if values.device != self.device:
            raise ValueError(f"packed reset {name} must live on {self.device}")
        if self._capturing():
            slot = "qpos" if "qpos" in name else "qvel"
            self._record_violation(
                slot, ~self._torch.isfinite(values).all(), f"packed reset {name} must be finite"
            )
        elif not bool(self._torch.isfinite(values).all()):
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
        self._record_violation(
            name,
            ~self._torch.isfinite(values).all(),
            f"packed reset {name} values must be finite",
        )
        return values
