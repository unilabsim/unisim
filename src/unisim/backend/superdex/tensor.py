"""Persistent packed tensor transfers for the SuperDex CPU host bridge."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from unisim.backend.base import (
    HostBridgeTransferPlan,
    TensorDataPlane,
    TensorExecution,
    TensorIOSpec,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
    tensor_device_matches,
)

if TYPE_CHECKING:
    import torch

    from .backend import SuperDexBackend

_STREAM_OWNERSHIP = "caller-stream-per-packed-boundary; stream synchronized at CPU bridge"


def superdex_tensor_capabilities(backend: SuperDexBackend) -> TensorLifecycleCapabilities:
    """Return the initial, fail-closed SuperDex tensor capability matrix."""
    return TensorLifecycleCapabilities(
        execution=TensorExecution.HOST_BRIDGE,
        state_views=True,
        state_fields=frozenset({"qpos", "qvel", "ctrl"}),
        sensor_views=True,
        stepping=True,
        selected_reset=True,
        reset_randomization=False,
        fixed_variants=False,
        host_pre_step_control=False,
        packed_host_bridge=True,
        process_topology=TensorProcessTopology.IN_PROCESS,
        data_plane=TensorDataPlane.HOST_BRIDGE,
        stream_event_ownership=_STREAM_OWNERSHIP,
        torch_devices=("cpu", "cuda"),
    )


def require_superdex_tensor_runtime(
    backend: SuperDexBackend, *, require_callback_free: bool = False
) -> None:
    """Validate the intentionally narrow initial SuperDex tensor profile."""
    backend._check_open()
    if backend._plan is None:
        raise RuntimeError("SuperDex tensor I/O requires a materialized backend")
    if backend._variant_assignment is not None:
        raise NotImplementedError("SuperDex tensor I/O does not support fixed variants")
    if require_callback_free and backend._pre_step_control_fn is not None:
        raise NotImplementedError(
            "SuperDex tensor stepping does not support host pre-step control callbacks"
        )


@dataclass
class _TransferBuffers:
    """Stable staging buffers owned by one compiled plan."""

    host_packet: torch.Tensor
    selected_host: torch.Tensor
    device_packet: torch.Tensor
    selected_packet: torch.Tensor
    host_ctrl: torch.Tensor
    reset_device: torch.Tensor
    reset_host: torch.Tensor


class SuperDexHostBridgeTransferPlan(HostBridgeTransferPlan):
    """One stable SuperDex layout for explicit accelerator/CPU boundaries.

    SuperDex physics remains authoritative on CPU. Each semantic operation moves
    one contiguous packet at most, and CUDA transfers are synchronized on the
    caller's current Torch stream. The initial profile intentionally excludes
    fixed variants, reset randomization, and host pre-step callbacks.
    """

    def __init__(self, backend: SuperDexBackend, spec: TensorIOSpec) -> None:
        import torch

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

        require_superdex_tensor_runtime(backend, require_callback_free=True)
        if backend.tensor_execution() is not TensorExecution.HOST_BRIDGE:
            raise RuntimeError("SuperDex packed tensor I/O requires the host-bridge lifecycle")
        if set(spec.state_fields) != {"qpos", "qvel"}:
            raise ValueError("SuperDex packed state I/O supports exactly qpos and qvel")

        for name in spec.sensor_names:
            if name in backend._unsupported_sensors:
                raise NotImplementedError(
                    f"superdex sensor {name!r}: {backend._unsupported_sensors[name]}"
                )
            if name not in backend._sensor_values:
                raise KeyError(f"unknown SuperDex sensor {name!r}")

        target = torch.device(spec.device) if spec.device is not None else torch.device("cpu")
        if target.type not in {"cpu", "cuda"}:
            raise ValueError(f"SuperDex tensor I/O device must be CPU or CUDA, got {target}")
        if target.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA SuperDex tensor I/O requested but CUDA is unavailable")
            if target.index is None:
                target = torch.device("cuda", index=torch.cuda.current_device())
        self.device = target
        self._pin = target.type == "cuda"

        model = backend.model
        field_widths = {"qpos": model.nq, "qvel": model.nv}
        sensor_widths: dict[str, int] = {}
        offsets: dict[str, int] = {}
        cursor = 0
        for name in spec.state_fields:
            offsets[name] = cursor
            cursor += field_widths[name]
        for name in spec.sensor_names:
            width = int(backend._sensor_values[name].reshape(backend.num_envs, -1).shape[1])
            sensor_widths[name] = width
            offsets[name] = cursor
            cursor += width

        self._field_widths = field_widths
        self._sensor_widths = sensor_widths
        self._offsets = dict(offsets)
        self._row_width = cursor
        num_envs = backend.num_envs

        device_packet = torch.empty((num_envs, cursor), dtype=torch.float32, device=target)
        reset_width = 1 + model.nq + model.nv
        buffers = _TransferBuffers(
            host_packet=torch.empty((num_envs, cursor), dtype=torch.float32, pin_memory=self._pin),
            selected_host=torch.empty(
                (num_envs, cursor), dtype=torch.float32, pin_memory=self._pin
            ),
            device_packet=device_packet,
            selected_packet=torch.empty_like(device_packet),
            host_ctrl=torch.empty(
                (num_envs, backend.num_actuators), dtype=torch.float32, pin_memory=self._pin
            ),
            reset_device=torch.empty((num_envs, reset_width), dtype=torch.int32, device=target),
            reset_host=torch.empty(
                (num_envs, reset_width), dtype=torch.int32, pin_memory=self._pin
            ),
        )
        self._buffer_slots: list[_TransferBuffers] = [buffers]

        self._control_ready = False
        self._last_reset_rows_host: np.ndarray | None = None
        self._last_reset_rows_device: torch.Tensor | None = None
        self._closed = False
        self._validate_layout()

    @property
    def spec(self) -> TensorIOSpec:
        return self._spec

    @property
    def transfer_stats(self) -> dict[str, int]:
        return dict(self._transfer_stats)

    def _validate_layout(self) -> None:
        buffers = self._buffers()
        expected_packet = (self._backend.num_envs, self._row_width)
        if buffers.host_packet.shape != expected_packet:
            raise RuntimeError("SuperDex packed tensor host layout changed after compilation")
        if buffers.device_packet.shape != buffers.host_packet.shape:
            raise RuntimeError("SuperDex packed tensor device layout is inconsistent")
        if buffers.selected_host.shape != buffers.host_packet.shape:
            raise RuntimeError("SuperDex packed tensor selected host layout is inconsistent")
        if buffers.selected_packet.shape != buffers.device_packet.shape:
            raise RuntimeError("SuperDex packed tensor selected device layout is inconsistent")
        for name in self._spec.state_fields:
            expected = self._backend.model.nq if name == "qpos" else self._backend.model.nv
            if self._field_widths[name] != expected:
                raise RuntimeError(f"SuperDex state field {name!r} layout changed")
        for name in self._spec.sensor_names:
            expected_sensor = (self._backend.num_envs, self._sensor_widths[name])
            source = self._backend._sensor_values.get(name)
            if source is None or source.reshape(expected_sensor[0], -1).shape != expected_sensor:
                raise RuntimeError(f"SuperDex sensor {name!r} layout changed after compilation")

    def _buffers(self) -> _TransferBuffers:
        if not self._buffer_slots:
            raise RuntimeError("SuperDex packed tensor I/O plan is closed")
        return self._buffer_slots[0]

    def _record(
        self,
        direction: str,
        label: str,
        byte_count: int,
        elapsed_ms: float,
        synchronized: bool,
    ) -> None:
        self._transfer_stats[f"{direction}_count"] += 1
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
        import torch

        if tensor.device.type == "cuda":
            torch.cuda.current_stream(tensor.device).synchronize()

    def _validate_reset_rows(self, rows: np.ndarray) -> None:
        """Perform the one bounded host check required after reset D2H."""

        unique_rows = np.unique(rows)
        if rows.size != unique_rows.size:
            raise ValueError("SuperDex packed reset env_indices must contain unique values")
        if rows.size and (unique_rows[0] < 0 or unique_rows[-1] >= self._backend.num_envs):
            raise ValueError(
                f"SuperDex packed reset env_indices must be in [0, {self._backend.num_envs})"
            )

    def _pack_host_packet(self, rows: np.ndarray | None = None) -> torch.Tensor:
        buffers = self._buffers()
        packet = buffers.selected_host[: rows.shape[0]] if rows is not None else buffers.host_packet
        packet_np = packet.numpy()
        sources = {"qpos": self._backend._qpos, "qvel": self._backend._qvel}
        for name in self._spec.state_fields:
            source = sources[name]
            destination = packet_np[
                :, self._offsets[name] : self._offsets[name] + self._field_widths[name]
            ]
            if rows is None:
                destination[...] = source
            else:
                destination[...] = source[rows]
        for name in self._spec.sensor_names:
            source = self._backend._sensor_values[name].reshape(self._backend.num_envs, -1)
            destination = packet_np[
                :, self._offsets[name] : self._offsets[name] + self._sensor_widths[name]
            ]
            if rows is None:
                destination[...] = source
            else:
                destination[...] = source[rows]
        return packet

    def _views(self) -> dict[str, torch.Tensor]:
        packet = self._buffers().device_packet
        views: dict[str, torch.Tensor] = {}
        for name in (*self._spec.state_fields, *self._spec.sensor_names):
            start = self._offsets[name]
            width = self._sensor_widths.get(name) or self._field_widths.get(name, 0)
            views[name] = packet[:, start : start + width]
        return views

    def write_control(self, ctrl: Any) -> None:
        import torch

        self._require_open()
        require_superdex_tensor_runtime(self._backend, require_callback_free=True)
        if not isinstance(ctrl, torch.Tensor):
            raise TypeError("SuperDex packed control must be a torch.Tensor")
        expected = (self._backend.num_envs, self._backend.num_actuators)
        if tuple(ctrl.shape) != expected:
            raise ValueError(f"SuperDex packed ctrl must have shape {expected}")
        if ctrl.dtype != torch.float32 or not bool(ctrl.is_contiguous()):
            raise TypeError("SuperDex packed ctrl must be contiguous float32")
        if not tensor_device_matches(
            (str(self.device),), ctrl.device, current_device=self.device.index
        ):
            raise ValueError(f"SuperDex packed ctrl must live on {self.device}, got {ctrl.device}")

        self.last_timing = {}
        started = time.perf_counter()
        buffers = self._buffers()
        buffers.host_ctrl.copy_(ctrl, non_blocking=self._pin)
        self._synchronize(ctrl)
        elapsed = (time.perf_counter() - started) * 1000.0
        self._control_ready = True
        self._record(
            "d2h",
            "control",
            buffers.host_ctrl.numel() * buffers.host_ctrl.element_size(),
            elapsed,
            self._pin,
        )

    def step(self, nsteps: int = 1) -> dict | None:
        self._require_open()
        require_superdex_tensor_runtime(self._backend, require_callback_free=True)
        if not self._control_ready:
            raise RuntimeError("write_control() must complete before packed step()")
        self._control_ready = False
        self._backend.step(self._buffers().host_ctrl.numpy(), nsteps, _producer_owns_finite=True)
        return {"timing": dict(self.last_timing)}

    def read_state_sensors(self) -> Mapping[str, Any]:
        self._require_open()
        require_superdex_tensor_runtime(self._backend, require_callback_free=True)
        self._validate_layout()
        self.last_timing = {}
        started = time.perf_counter()
        self._pack_host_packet(None)
        buffers = self._buffers()
        buffers.device_packet.copy_(buffers.host_packet, non_blocking=self._pin)
        self._synchronize(buffers.device_packet)
        elapsed = (time.perf_counter() - started) * 1000.0
        self._record(
            "h2d",
            "state",
            buffers.host_packet.numel() * buffers.host_packet.element_size(),
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
        import torch

        self._require_open()
        require_superdex_tensor_runtime(self._backend)
        self._last_reset_rows_host = None
        self._last_reset_rows_device = None
        self.last_timing = {}
        if randomization is not None:
            raise NotImplementedError("SuperDex packed reset does not support randomization")

        values = {"env_indices": env_indices, "qpos": qpos, "qvel": qvel}
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"SuperDex packed reset {name} must be a torch.Tensor")
            if not bool(value.is_contiguous()):
                raise ValueError(f"SuperDex packed reset {name} must be contiguous")
        if env_indices.ndim != 1 or env_indices.dtype != torch.int64:
            raise TypeError(
                "SuperDex packed reset env_indices must be a contiguous 1-D int64 tensor"
            )
        count = int(env_indices.shape[0])
        if count > self._backend.num_envs:
            raise ValueError("SuperDex packed reset contains more rows than environments")
        model = self._backend.model
        if tuple(qpos.shape) != (count, model.nq):
            raise ValueError(f"SuperDex packed reset qpos must have shape {(count, model.nq)}")
        if tuple(qvel.shape) != (count, model.nv):
            raise ValueError(f"SuperDex packed reset qvel must have shape {(count, model.nv)}")
        if qpos.dtype != torch.float32 or qvel.dtype != torch.float32:
            raise TypeError("SuperDex packed reset qpos and qvel must be float32")
        if any(
            not tensor_device_matches(
                (str(self.device),), value.device, current_device=self.device.index
            )
            for value in (env_indices, qpos, qvel)
        ):
            raise ValueError(f"SuperDex packed reset tensors must live on {self.device}")
        if count == 0:
            return {"timing": {}}

        qpos_offset = 1
        qvel_offset = qpos_offset + model.nq
        buffers = self._buffers()
        packet = buffers.reset_device[:count]
        packet[:, 0] = env_indices.to(dtype=torch.int32)
        packet[:, qpos_offset:qvel_offset] = qpos.view(dtype=torch.int32)
        packet[:, qvel_offset:] = qvel.view(dtype=torch.int32)

        # Packing, transfer, and the bounded row check are all part of this
        # device-to-host boundary. Finiteness is intentionally producer-owned.
        started = time.perf_counter()
        buffers.reset_host[:count].copy_(packet, non_blocking=self._pin)
        self._synchronize(packet)
        host_packet = buffers.reset_host[:count].numpy()
        rows = host_packet[:, 0].astype(np.intp, copy=True)
        host_qpos = host_packet[:, qpos_offset:qvel_offset].view(np.float32)
        host_qvel = host_packet[:, qvel_offset:].view(np.float32)
        self._validate_reset_rows(rows)
        elapsed = (time.perf_counter() - started) * 1000.0

        self._backend.set_state(
            rows,
            host_qpos.astype(self._backend._dtype),
            host_qvel.astype(self._backend._dtype),
            _producer_owns_finite=True,
        )
        self._last_reset_rows_host = rows
        self._last_reset_rows_device = env_indices.detach().clone()
        self._record(
            "d2h",
            "reset",
            count * buffers.reset_host.stride(0) * buffers.reset_host.element_size(),
            elapsed,
            self._pin,
        )
        return {"timing": dict(self.last_timing)}

    def read_selected_state_sensors(self) -> Mapping[str, Any]:
        self._require_open()
        require_superdex_tensor_runtime(self._backend)
        rows = self._last_reset_rows_host
        rows_device = self._last_reset_rows_device
        if rows is None or rows_device is None or rows.size == 0:
            raise RuntimeError("apply_reset() must complete before a selected packed read")
        self._validate_layout()
        self.last_timing = {}
        started = time.perf_counter()
        count = int(rows.shape[0])
        selected_host = self._pack_host_packet(rows)
        buffers = self._buffers()
        buffers.selected_packet[:count].copy_(selected_host, non_blocking=self._pin)
        self._synchronize(buffers.selected_packet)
        buffers.device_packet.index_copy_(0, rows_device, buffers.selected_packet[:count])
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
        if self._closed:
            return
        self._control_ready = False
        self._last_reset_rows_host = None
        self._last_reset_rows_device = None
        self._buffer_slots.clear()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SuperDex packed tensor I/O plan is closed")


__all__ = [
    "SuperDexHostBridgeTransferPlan",
    "require_superdex_tensor_runtime",
    "superdex_tensor_capabilities",
]
