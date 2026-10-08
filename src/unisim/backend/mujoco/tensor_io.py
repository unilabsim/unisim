"""Persistent packed tensor transfers for the MuJoCo/MJBatch host bridge."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from unisim.backend.base import (
    HostBridgeTransferPlan,
    TensorExecution,
    TensorIOSpec,
)

if TYPE_CHECKING:
    from .backend import MuJoCoBackend

_TRACKED_SENSOR_PREFIXES = (
    "track_pos_w",
    "track_quat_w",
    "track_linvel_w",
    "track_angvel_w",
)
_TRACKED_SENSOR_DIMS = {
    "track_pos_w": 3,
    "track_quat_w": 4,
    "track_linvel_w": 3,
    "track_angvel_w": 3,
}


class MuJoCoHostBridgeTransferPlan(HostBridgeTransferPlan):
    """One stable layout for MuJoCo's unavoidable CPU tensor boundaries.

    CPU physics remains authoritative. The plan removes per-field allocations,
    metadata lookup, and per-sensor device transfers from the hot path. Each
    public operation performs at most one packed transfer on the current Torch
    stream and synchronizes only that stream when crossing to or from CUDA.
    """

    def __init__(self, backend: MuJoCoBackend, spec: TensorIOSpec) -> None:
        import torch as _torch

        self._backend = backend
        self._spec = spec
        self.last_timing: dict[str, float] = {}
        self._transfer_stats = {
            "d2h_count": 0,
            "h2d_count": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
            "synchronization_count": 0,
        }

        backend._require_entity_healthy()
        if backend.tensor_execution() is not TensorExecution.HOST_BRIDGE:
            raise RuntimeError("MuJoCo packed tensor I/O requires the host-bridge lifecycle")

        requested_fields = tuple(spec.state_fields)
        if not requested_fields or set(requested_fields) - {"qpos", "qvel"}:
            raise ValueError("MuJoCo packed state I/O supports qpos and qvel only")
        sensor_names = tuple(spec.sensor_names)
        for name in sensor_names:
            if name not in backend._sensor_views:
                raise KeyError(f"unknown MuJoCo sensor {name!r}")

        target = _torch.device(spec.device) if spec.device is not None else _torch.device("cpu")
        if target.type not in {"cpu", "cuda"}:
            raise ValueError(f"MuJoCo tensor I/O device must be CPU or CUDA, got {target}")
        if target.type == "cuda":
            if not _torch.cuda.is_available():
                raise RuntimeError("CUDA MuJoCo tensor I/O requested but CUDA is unavailable")
            if target.index is None:
                target = _torch.device("cuda", index=_torch.cuda.current_device())
        self.device = target
        self._pin = target.type == "cuda"

        field_widths = {"qpos": backend.nq, "qvel": backend.nv}
        sensor_widths: dict[str, int] = {}
        cursor = 0
        offsets: dict[str, int] = {}
        for name in requested_fields:
            offsets[name] = cursor
            cursor += field_widths[name]
        for name in sensor_names:
            width = int(backend._sensor_views[name].reshape(backend.num_envs, -1).shape[1])
            sensor_widths[name] = width
            offsets[name] = cursor
            cursor += width
        self._field_widths = field_widths
        self._sensor_widths = sensor_widths
        self._offsets = dict(offsets)
        self._row_width = cursor

        num_envs = backend.num_envs
        self._host_packet = _torch.empty(
            (num_envs, cursor), dtype=_torch.float32, pin_memory=self._pin
        )
        self._selected_host = _torch.empty(
            (num_envs, cursor), dtype=_torch.float32, pin_memory=self._pin
        )
        self._device_packet = _torch.empty((num_envs, cursor), dtype=_torch.float32, device=target)
        self._selected_packet = _torch.empty_like(self._device_packet)
        self._host_ctrl = _torch.empty(
            (num_envs, backend.num_actuators), dtype=_torch.float32, pin_memory=self._pin
        )
        reset_width = 2 + backend.nq + backend.nv
        self._reset_device = _torch.empty(
            (num_envs, reset_width), dtype=_torch.int32, device=target
        )
        self._reset_host = _torch.empty(
            (num_envs, reset_width), dtype=_torch.int32, pin_memory=self._pin
        )

        aggregate_bodies: list[str] = []
        tracked_members: dict[str, list[tuple[int, str]]] = {}
        for index, name in enumerate(sensor_names):
            for prefix in _TRACKED_SENSOR_PREFIXES:
                marker = f"{prefix}_"
                if name.startswith(marker):
                    body_name = name[len(marker) :]
                    tracked_members.setdefault(prefix, []).append((index, body_name))
                    break
        aggregate_names: set[str] = set()
        tracked_sensor_groups: tuple[tuple[str, int, int], ...] = ()
        native_bodies = tuple(getattr(backend, "_valid_bnames", ()) or ())
        for prefix, members in tracked_members.items():
            member_names = tuple(name for _, name in members)
            indices = tuple(index for index, _ in members)
            contiguous = indices == tuple(range(indices[0], indices[0] + len(indices)))
            if member_names == native_bodies and contiguous:
                start_index = indices[0]
                start = offsets[sensor_names[start_index]]
                tracked_sensor_groups = (
                    *tracked_sensor_groups,
                    (prefix, start, len(members) * _TRACKED_SENSOR_DIMS[prefix]),
                )
                aggregate_names.update(sensor_names[index] for index in indices)
                for body_name in member_names:
                    if body_name not in aggregate_bodies:
                        aggregate_bodies.append(body_name)
        self._tracked_body_names = tuple(aggregate_bodies)
        self._tracked_body_ids = (
            np.asarray(backend.get_body_ids(self._tracked_body_names), dtype=np.intp)
            if aggregate_bodies
            else np.empty(0, dtype=np.intp)
        )
        self._tracked_sensor_groups = tracked_sensor_groups
        self._scalar_sensor_names = tuple(
            name for name in sensor_names if name not in aggregate_names
        )
        self._tracked_sensor_sources = (
            {
                "track_pos_w": backend._tracked_pos_w_all,
                "track_quat_w": backend._tracked_quat_w_all,
                "track_linvel_w": backend._tracked_linvel_w_all,
                "track_angvel_w": backend._tracked_angvel_w_all,
            }
            if tracked_sensor_groups
            else {}
        )

        self._control_ready = False
        self._last_reset_rows_host: np.ndarray | None = None
        self._last_reset_rows_device: torch.Tensor | None = None
        self._validate_layout()

    @property
    def spec(self) -> TensorIOSpec:
        return self._spec

    @property
    def transfer_stats(self) -> dict[str, int]:
        return dict(self._transfer_stats)

    def _validate_layout(self) -> None:
        if self._host_packet.shape != (self._backend.num_envs, self._row_width):
            raise RuntimeError("MuJoCo packed tensor host layout changed after compilation")
        if self._device_packet.shape != self._host_packet.shape:
            raise RuntimeError("MuJoCo packed tensor device layout is inconsistent")
        if self._selected_host.shape != self._host_packet.shape:
            raise RuntimeError("MuJoCo packed tensor selected host layout is inconsistent")
        if self._selected_packet.shape != self._device_packet.shape:
            raise RuntimeError("MuJoCo packed tensor selected device layout is inconsistent")
        for name, width in self._field_widths.items():
            if name not in self._spec.state_fields:
                continue
            expected_width = self._backend.nq if name == "qpos" else self._backend.nv
            if width != expected_width:
                raise RuntimeError(f"MuJoCo state field {name!r} layout changed after compilation")
        for name in self._spec.sensor_names:
            expected = (self._backend.num_envs, self._sensor_widths[name])
            source = self._backend._sensor_views.get(name)
            if source is None or source.reshape(expected[0], -1).shape != expected:
                raise RuntimeError(f"MuJoCo sensor {name!r} layout changed after compilation")

    def _record(
        self,
        direction: str,
        label: str,
        byte_count: int,
        elapsed_ms: float,
        synchronized: bool,
    ) -> None:
        key = f"{direction}_count"
        self._transfer_stats[key] += 1
        self._transfer_stats[f"{direction}_bytes"] += int(byte_count)
        if synchronized:
            self._transfer_stats["synchronization_count"] += 1
        prefix = f"tensor_{label}_packed_{direction}"
        self.last_timing.update(
            {
                f"{prefix}_ms": elapsed_ms,
                f"{prefix}_bytes": float(byte_count),
                f"{prefix}_count": 1.0,
            }
        )

    def _synchronize(self, tensor: torch.Tensor) -> None:
        if tensor.device.type == "cuda":
            torch.cuda.current_stream(tensor.device).synchronize()

    def _refresh_tracked_sensors(self, rows: np.ndarray | None = None) -> None:
        if not self._tracked_body_ids.size:
            return
        if rows is None:
            self._backend.get_body_pos_w(self._tracked_body_ids)
        else:
            self._backend.get_body_pose_w_rows(rows, self._tracked_body_ids)

    def _pack_host_packet(self, rows: np.ndarray | None = None) -> torch.Tensor:
        backend = self._backend
        packet = self._selected_host[: rows.shape[0]] if rows is not None else self._host_packet
        packet_np = packet.numpy()
        for name in self._spec.state_fields:
            value = backend._qpos_view if name == "qpos" else backend._qvel_view
            width = self._field_widths[name]
            destination = packet_np[:, self._offsets[name] : self._offsets[name] + width]
            if rows is None:
                destination[...] = value
            else:
                np.take(value, rows, axis=0, out=destination)
        self._refresh_tracked_sensors(rows)
        for name in self._spec.sensor_names:
            if name not in self._scalar_sensor_names:
                continue
            value = backend.get_sensor_data(name).reshape(backend.num_envs, -1)
            width = self._sensor_widths[name]
            destination = packet_np[:, self._offsets[name] : self._offsets[name] + width]
            if rows is None:
                destination[...] = value
            else:
                np.take(value, rows, axis=0, out=destination)
        for prefix, start, width in self._tracked_sensor_groups:
            source = self._tracked_sensor_sources[prefix].reshape(backend.num_envs, -1)
            destination = packet_np[:, start : start + width]
            if rows is None:
                destination[...] = source
            else:
                np.take(source, rows, axis=0, out=destination)
        return packet

    def _views(self) -> dict[str, torch.Tensor]:
        packet = self._device_packet
        views: dict[str, torch.Tensor] = {}
        for name in (*self._spec.state_fields, *self._spec.sensor_names):
            start = self._offsets[name]
            width = self._sensor_widths.get(name) or self._field_widths.get(name, 0)
            views[name] = packet[:, start : start + width]
        return views

    def write_control(self, ctrl: Any) -> None:
        self._backend._require_entity_healthy()
        if not isinstance(ctrl, torch.Tensor):
            raise TypeError("MuJoCo packed control must be a torch.Tensor")
        expected = (self._backend.num_envs, self._backend.num_actuators)
        if tuple(ctrl.shape) != expected:
            raise ValueError(f"MuJoCo packed ctrl must have shape {expected}")
        if ctrl.dtype != torch.float32 or not bool(ctrl.is_contiguous()):
            raise TypeError("MuJoCo packed ctrl must be contiguous float32")
        if ctrl.device != self.device:
            raise ValueError(f"MuJoCo packed ctrl must live on {self.device}, got {ctrl.device}")

        self.last_timing = {}
        started = time.perf_counter()
        self._host_ctrl.copy_(ctrl, non_blocking=self._pin)
        self._synchronize(ctrl)
        elapsed = (time.perf_counter() - started) * 1000.0
        if not np.isfinite(self._host_ctrl.numpy()).all():
            self._control_ready = False
            raise ValueError("MuJoCo packed ctrl must contain finite values")
        self._control_ready = True
        self._record(
            "d2h",
            "control",
            self._host_ctrl.numel() * self._host_ctrl.element_size(),
            elapsed,
            self._pin,
        )

    def step(self, nsteps: int = 1) -> dict | None:
        self._backend._require_entity_healthy()
        if not self._control_ready:
            raise RuntimeError("write_control() must complete before packed step()")
        if self._backend._pre_step_control_fn is not None:
            self._control_ready = False
            raise NotImplementedError(
                "MuJoCo packed tensor stepping does not support host pre-step control callbacks"
            )
        self._control_ready = False
        result = self._backend.step(self._host_ctrl.numpy(), nsteps)
        if result is None:
            result = {}
        timing = dict(result.get("timing", {}))
        timing.update(self.last_timing)
        result["timing"] = timing
        return result

    def read_state_sensors(self) -> Mapping[str, Any]:
        self._backend._require_entity_healthy()
        self._validate_layout()
        self.last_timing = {}
        started = time.perf_counter()
        self._pack_host_packet(None)
        self._device_packet.copy_(self._host_packet, non_blocking=self._pin)
        self._synchronize(self._device_packet)
        elapsed = (time.perf_counter() - started) * 1000.0
        self._record(
            "h2d",
            "state",
            self._host_packet.numel() * self._host_packet.element_size(),
            elapsed,
            self._pin,
        )
        return self._views()

    def apply_reset(
        self,
        env_indices: Any,
        qpos: Any,
        qvel: Any,
        randomization: Any | None = None,
    ) -> dict | None:
        self._backend._require_entity_healthy()
        self._last_reset_rows_host = None
        self._last_reset_rows_device = None
        self.last_timing = {}
        values = {"env_indices": env_indices, "qpos": qpos, "qvel": qvel}
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"MuJoCo packed reset {name} must be a torch.Tensor")
            if not bool(value.is_contiguous()):
                raise ValueError(f"MuJoCo packed reset {name} must be contiguous")
        if env_indices.ndim != 1 or env_indices.dtype != torch.int64:
            raise TypeError("MuJoCo packed reset env_indices must be a contiguous 1-D int64 tensor")
        count = int(env_indices.shape[0])
        if count > self._backend.num_envs:
            raise ValueError("MuJoCo packed reset contains more rows than environments")
        if tuple(qpos.shape) != (count, self._backend.nq):
            raise ValueError(
                f"MuJoCo packed reset qpos must have shape {(count, self._backend.nq)}"
            )
        if tuple(qvel.shape) != (count, self._backend.nv):
            raise ValueError(
                f"MuJoCo packed reset qvel must have shape {(count, self._backend.nv)}"
            )
        if qpos.dtype != torch.float32 or qvel.dtype != torch.float32:
            raise TypeError("MuJoCo packed reset qpos and qvel must be float32")
        if (
            env_indices.device != self.device
            or qpos.device != self.device
            or qvel.device != self.device
        ):
            raise ValueError(f"MuJoCo packed reset tensors must live on {self.device}")
        if count == 0:
            return {"timing": {}}

        qpos_offset = 2
        qvel_offset = qpos_offset + self._backend.nq
        finite = torch.isfinite(qpos).all(dim=1) & torch.isfinite(qvel).all(dim=1)
        packet = self._reset_device[:count]
        packet[:, 0] = env_indices.to(dtype=torch.int32)
        packet[:, 1] = finite.to(dtype=torch.int32)
        packet[:, qpos_offset:qvel_offset] = qpos.view(dtype=torch.int32)
        packet[:, qvel_offset:] = qvel.view(dtype=torch.int32)

        started = time.perf_counter()
        self._reset_host[:count].copy_(packet, non_blocking=self._pin)
        self._synchronize(packet)
        elapsed = (time.perf_counter() - started) * 1000.0
        host_packet = self._reset_host[:count].numpy()
        rows = host_packet[:, 0].astype(np.intp, copy=True)
        host_qpos = host_packet[:, qpos_offset:qvel_offset].view(np.float32)
        host_qvel = host_packet[:, qvel_offset:].view(np.float32)
        if not np.isfinite(host_qpos).all() or not np.isfinite(host_qvel).all():
            raise ValueError("MuJoCo packed reset qpos and qvel must contain finite values")
        if rows.size and (rows.min() < 0 or rows.max() >= self._backend.num_envs):
            raise ValueError(
                f"MuJoCo packed reset env_indices must be in [0, {self._backend.num_envs})"
            )
        if np.unique(rows).size != rows.size:
            raise ValueError("MuJoCo packed reset env_indices must contain unique values")

        result = self._backend.set_state(rows, host_qpos, host_qvel, randomization)
        self._last_reset_rows_host = rows
        self._last_reset_rows_device = env_indices.detach()
        self._record(
            "d2h",
            "reset",
            count * self._reset_host.stride(0) * self._reset_host.element_size(),
            elapsed,
            self._pin,
        )
        if result is None:
            result = {}
        timing = dict(result.get("timing", {}))
        timing.update(self.last_timing)
        result["timing"] = timing
        return result

    def read_selected_state_sensors(self) -> Mapping[str, Any]:
        self._backend._require_entity_healthy()
        rows = self._last_reset_rows_host
        rows_device = self._last_reset_rows_device
        if rows is None or rows_device is None or rows.size == 0:
            raise RuntimeError("apply_reset() must complete before a selected packed read")
        self._validate_layout()
        self.last_timing = {}
        started = time.perf_counter()
        count = int(rows.shape[0])
        selected_host = self._pack_host_packet(rows)
        self._selected_packet[:count].copy_(selected_host, non_blocking=self._pin)
        self._synchronize(self._selected_packet)
        self._device_packet.index_copy_(0, rows_device, self._selected_packet[:count])
        elapsed = (time.perf_counter() - started) * 1000.0
        self._record(
            "h2d",
            "post_reset",
            count * selected_host.stride(0) * selected_host.element_size(),
            elapsed,
            self._pin,
        )
        return self._views()

    def close(self) -> None:
        self._control_ready = False
        self._last_reset_rows_host = None
        self._last_reset_rows_device = None


__all__ = ["MuJoCoHostBridgeTransferPlan"]
