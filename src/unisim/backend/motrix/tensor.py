"""Persistent packed tensor transfers for the MotrixSim CPU host bridge."""

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

    from .backend import MotrixBackend

_STREAM_OWNERSHIP = "caller-stream-per-packed-boundary; stream synchronized at CPU bridge"
_SUPPORTED_STATE_FIELDS = frozenset({"qpos", "qvel", "ctrl"})
_BODY_VIEW_WIDTHS = {
    "track_pos_w_": 3,
    "track_quat_w_": 4,
    "track_linvel_w_": 3,
    "track_angvel_w_": 3,
}


def motrix_tensor_capabilities(backend: MotrixBackend) -> TensorLifecycleCapabilities:
    """Return the deliberately narrow MotrixSim tensor capability matrix."""
    del backend
    return TensorLifecycleCapabilities(
        execution=TensorExecution.HOST_BRIDGE,
        state_views=True,
        state_fields=_SUPPORTED_STATE_FIELDS,
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


def require_motrix_tensor_runtime(
    backend: MotrixBackend, *, require_callback_free: bool = False
) -> None:
    """Validate the intentionally narrow MotrixSim tensor profile."""
    if backend._closed:
        raise RuntimeError("MotrixSim tensor I/O requires an open backend")
    if backend._portable_mode:
        backend._require_portable_healthy("tensor I/O")
    if backend._portable_variant_assignment is not None:
        raise NotImplementedError("MotrixSim tensor I/O does not support fixed variants")
    if require_callback_free and backend._pre_step_control_fn is not None:
        raise NotImplementedError(
            "MotrixSim tensor stepping does not support host pre-step control callbacks"
        )


def _field_widths(backend: MotrixBackend) -> dict[str, int]:
    if backend._portable_mode:
        layout = backend.get_scene_layout()
        return {"qpos": int(layout.nq), "qvel": int(layout.nv), "ctrl": int(backend.num_actuators)}
    # MotrixSim model metadata can count disabled/fixed coordinates that are
    # absent from the public ``data.dof_pos`` / ``data.dof_vel`` blocks. Tensor
    # layouts must follow the public state actually returned by ``get_state``.
    public_state = _canonical_states(backend, ("qpos", "qvel"))
    return {
        "qpos": int(np.asarray(public_state["qpos"]).shape[1]),
        "qvel": int(np.asarray(public_state["qvel"]).shape[1]),
        "ctrl": int(backend.num_actuators),
    }


def _current_controls(backend: MotrixBackend) -> np.ndarray:
    if backend._portable_mode:
        values = backend._portable_current_controls()
    else:
        values = np.asarray(backend._data.actuator_ctrls)
    expected = (backend.num_envs, backend.num_actuators)
    values = np.asarray(values, dtype=np.float32).reshape(expected)
    return np.ascontiguousarray(values)


def _canonical_states(backend: MotrixBackend, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Return full public tensor state, including portable composed entities."""

    if backend._portable_mode:
        values: dict[str, np.ndarray] = {}
        if "qpos" in fields:
            values["qpos"] = backend._portable_state_qpos()
        if "qvel" in fields:
            values["qvel"] = backend._portable_state_qvel()
        return values
    return dict(backend.get_state(fields))


def _normalize_device(value: Any | None, *, label: str = "tensor I/O") -> Any:
    import torch

    target = torch.device(value) if value is not None else torch.device("cpu")
    if target.type not in {"cpu", "cuda"}:
        raise ValueError(f"MotrixSim {label} device must be CPU or CUDA, got {target}")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA MotrixSim {label} requested but CUDA is unavailable")
        if target.index is None:
            target = torch.device("cuda", index=torch.cuda.current_device())
    return target


def _synchronize_device(tensor: Any) -> None:
    import torch

    if tensor.device.type == "cuda":
        torch.cuda.current_stream(tensor.device).synchronize()


def _validate_reset_rows(rows: np.ndarray, num_envs: int, owner: str) -> None:
    """Perform the one bounded host check required after reset D2H."""

    unique_rows = np.unique(rows)
    if rows.size != unique_rows.size:
        raise ValueError(f"{owner} tensor reset env_indices must contain unique values")
    if rows.size and (unique_rows[0] < 0 or unique_rows[-1] >= num_envs):
        raise ValueError(f"{owner} tensor reset env_indices must be in [0, {num_envs})")


@dataclass(frozen=True)
class _ResolvedSensorNames:
    """Split native sensors from public body-state sensor aliases."""

    physical: tuple[str, ...]
    physical_widths: dict[str, int]
    body_names: tuple[str, ...]
    body_ids: np.ndarray
    body_slots: dict[str, tuple[int, str]]


def _resolve_sensor_names(
    backend: MotrixBackend, sensor_names: tuple[str, ...]
) -> _ResolvedSensorNames:
    """Resolve native sensors and canonical world-frame body sensor views.

    Motion-tracking consumers request stable public names such as
    ``track_pos_w_pelvis``. Motrix exposes those quantities through its public
    fused body-state API rather than native frame sensors, so the packed plan
    projects them into the same packet without exposing body-array conventions
    to task code.
    """

    physical: list[str] = []
    physical_widths: dict[str, int] = {}
    body_names: list[str] = []
    body_name_indices: dict[str, int] = {}
    body_slots: dict[str, tuple[int, str]] = {}
    for name in sensor_names:
        match = next(
            (
                (prefix, name[len(prefix) :])
                for prefix in _BODY_VIEW_WIDTHS
                if name.startswith(prefix)
            ),
            None,
        )
        if match is None:
            source = np.asarray(backend.get_sensor_data(name), dtype=np.float32)
            physical_widths[name] = int(source.reshape(backend.num_envs, -1).shape[1])
            physical.append(name)
            continue
        prefix, body_name = match
        if not body_name:
            raise KeyError(f"MotrixSim body sensor {name!r} has an empty body name")
        index = body_name_indices.get(body_name)
        if index is None:
            index = len(body_names)
            body_name_indices[body_name] = index
            body_names.append(body_name)
        body_slots[name] = (index, prefix)
    body_ids = (
        backend.get_body_ids(tuple(body_names)) if body_names else np.empty((0,), dtype=np.intp)
    )
    return _ResolvedSensorNames(
        physical=tuple(physical),
        physical_widths=physical_widths,
        body_names=tuple(body_names),
        body_ids=body_ids,
        body_slots=body_slots,
    )


def _read_sensor_sources(
    backend: MotrixBackend,
    sensor_names: tuple[str, ...],
    resolved: _ResolvedSensorNames,
) -> dict[str, np.ndarray]:
    """Read native sensors and requested body views on the CPU side."""

    sources: dict[str, np.ndarray] = {}
    if resolved.body_names:
        positions, quaternions, linear_velocities, angular_velocities = backend.get_body_state_w(
            resolved.body_ids
        )
        arrays_by_prefix = {
            "track_pos_w_": positions,
            "track_quat_w_": quaternions,
            "track_linvel_w_": linear_velocities,
            "track_angvel_w_": angular_velocities,
        }
        for name, (body_index, prefix) in resolved.body_slots.items():
            sources[name] = np.asarray(arrays_by_prefix[prefix][:, body_index, :], dtype=np.float32)
    if resolved.physical:
        values = np.asarray(
            backend.get_sensor_data_batch(resolved.physical), dtype=np.float32
        ).reshape(backend.num_envs, -1)
        cursor = 0
        for name in resolved.physical:
            width = resolved.physical_widths[name]
            sources[name] = values[:, cursor : cursor + width]
            cursor += width
    return sources


def _read_selected_sensor_sources(
    backend: MotrixBackend,
    sensor_names: tuple[str, ...],
    resolved: _ResolvedSensorNames,
    rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Read selected rows directly without full-batch body-state allocation.

    Native sensors already expose a public row-local reader. World-frame body
    views use MotrixSim's fused selected-link state read for positions and
    rotations, while velocities index the backend's authoritative full-batch
    velocity cache (link velocities are not exposed row-locally). Unlike the
    full read path, this never gathers state for every environment.
    """

    selected_rows = np.asarray(rows, dtype=np.intp)
    sources: dict[str, np.ndarray] = {}
    if resolved.body_names:
        count = int(selected_rows.shape[0])
        positions = np.empty((count, len(resolved.body_names), 3), dtype=np.float32)
        quaternions = np.empty((count, len(resolved.body_names), 4), dtype=np.float32)
        if count:
            velocities = np.empty_like(positions)
            angular_velocities = np.empty_like(positions)
            backend.copy_body_state_w_rows(
                selected_rows,
                resolved.body_ids,
                positions,
                quaternions,
                velocities,
                angular_velocities,
            )
        else:
            velocities = np.empty_like(positions)
            angular_velocities = np.empty_like(positions)
        arrays_by_prefix = {
            "track_pos_w_": positions,
            "track_quat_w_": quaternions,
            "track_linvel_w_": velocities,
            "track_angvel_w_": angular_velocities,
        }
        for name, (body_index, prefix) in resolved.body_slots.items():
            sources[name] = np.asarray(arrays_by_prefix[prefix][:, body_index, :], dtype=np.float32)
    if resolved.physical:
        for name in resolved.physical:
            sources[name] = backend.get_sensor_data_rows(name, selected_rows)
    del sensor_names
    return sources


def _read_selected_states(
    backend: MotrixBackend, fields: tuple[str, ...], rows: np.ndarray
) -> dict[str, np.ndarray]:
    """Read only selected canonical public state rows."""

    selected_rows = np.asarray(rows, dtype=np.intp)
    if backend._portable_mode:
        values = _canonical_states(backend, fields)
        return {name: np.asarray(values[name])[selected_rows] for name in fields}

    qpos_indices = (
        backend._actuator_joint_pos_indices
        if backend._actuator_joint_pos_indices is not None
        else backend._joint_dof_pos_indices
    )
    qvel_indices = (
        backend._actuator_joint_vel_indices
        if backend._actuator_joint_vel_indices is not None
        else backend._joint_dof_vel_indices
    )
    output: dict[str, np.ndarray] = {}
    if "qpos" in fields:
        dof_qpos = backend._data.dof_pos[np.ix_(selected_rows, qpos_indices)]
        qpos = np.empty((selected_rows.size, dof_qpos.shape[1] + 7), dtype=np.float32)
        qpos[:, :3] = backend.get_base_pos()[selected_rows]
        qpos[:, 3:7] = backend.get_base_quat()[selected_rows]
        qpos[:, 7:] = dof_qpos
        output["qpos"] = qpos
    if "qvel" in fields:
        dof_qvel = backend._data.dof_vel[np.ix_(selected_rows, qvel_indices)]
        qvel = np.empty((selected_rows.size, dof_qvel.shape[1] + 6), dtype=np.float32)
        qvel[:, :3] = backend.get_base_lin_vel()[selected_rows]
        qvel[:, 3:6] = backend.get_base_ang_vel()[selected_rows]
        qvel[:, 6:] = dof_qvel
        output["qvel"] = qvel
    return output


def _pack_state_sensors(
    backend: MotrixBackend,
    state_fields: tuple[str, ...],
    sensor_names: tuple[str, ...],
    destination: np.ndarray,
    offsets: Mapping[str, int],
    widths: Mapping[str, int],
    rows: np.ndarray | None = None,
    resolved: _ResolvedSensorNames | None = None,
) -> None:
    selected = slice(None) if rows is None else rows
    canonical_fields = tuple(name for name in ("qpos", "qvel") if name in state_fields)
    states = (
        _read_selected_states(backend, canonical_fields, rows)
        if rows is not None and canonical_fields
        else (_canonical_states(backend, canonical_fields) if canonical_fields else {})
    )
    if "ctrl" in state_fields:
        states["ctrl"] = _current_controls(backend)
    for name in state_fields:
        start = offsets[name]
        if rows is None:
            source = np.asarray(states[name], dtype=np.float32).reshape(
                backend.num_envs, widths[name]
            )
            destination[:, start : start + widths[name]] = source[selected]
        else:
            source = np.asarray(states[name], dtype=np.float32)
            destination[: source.shape[0], start : start + widths[name]] = source.reshape(
                source.shape[0], widths[name]
            )
    if not sensor_names:
        return
    resolved = resolved or _resolve_sensor_names(backend, sensor_names)
    sources = (
        _read_sensor_sources(backend, sensor_names, resolved)
        if rows is None
        else _read_selected_sensor_sources(backend, sensor_names, resolved, rows)
    )
    for name in sensor_names:
        width = widths[name]
        start = offsets[name]
        source = sources[name]
        if rows is None:
            destination[:, start : start + width] = source.reshape(backend.num_envs, width)[
                selected
            ]
        else:
            destination[: source.shape[0], start : start + width] = source.reshape(
                source.shape[0], width
            )


def motrix_state_views(
    backend: MotrixBackend, fields: tuple[str, ...] | str | None, device: Any | None
) -> dict[str, Any]:
    """Copy requested state blocks to one Torch device in one packed H2D."""
    import torch

    require_motrix_tensor_runtime(backend)
    names = (
        ("qpos", "qvel")
        if fields is None
        else ((fields,) if isinstance(fields, str) else tuple(fields))
    )
    unsupported = set(names) - _SUPPORTED_STATE_FIELDS
    if unsupported:
        raise KeyError(f"unknown MotrixSim tensor state field(s): {sorted(unsupported)}")
    target = _normalize_device(device, label="state view")
    canonical_fields = tuple(name for name in ("qpos", "qvel") if name in names)
    states = _canonical_states(backend, canonical_fields) if canonical_fields else {}
    if "ctrl" in names:
        states["ctrl"] = _current_controls(backend)
    widths = _field_widths(backend)
    offsets: dict[str, int] = {}
    cursor = 0
    for name in names:
        offsets[name] = cursor
        cursor += widths[name]
    host_packet = np.empty((backend.num_envs, cursor), dtype=np.float32)
    for name in names:
        source = np.asarray(states[name], dtype=np.float32).reshape(backend.num_envs, widths[name])
        start = offsets[name]
        host_packet[:, start : start + widths[name]] = source
    packet = torch.from_numpy(host_packet).to(target, non_blocking=bool(target.type == "cuda"))
    _synchronize_device(packet)
    return {name: packet[:, offsets[name] : offsets[name] + widths[name]] for name in names}


def motrix_sensor_view(backend: MotrixBackend, name: str, device: Any | None) -> Any:
    """Copy one authoritative CPU sensor block to a Torch device."""
    import torch

    require_motrix_tensor_runtime(backend)
    target = _normalize_device(device, label="sensor view")
    resolved = _resolve_sensor_names(backend, (name,))
    source = np.ascontiguousarray(
        _read_sensor_sources(backend, (name,), resolved)[name], dtype=np.float32
    )
    result = torch.from_numpy(source).to(target, non_blocking=bool(target.type == "cuda"))
    _synchronize_device(result)
    return result


def motrix_step_tensor(backend: MotrixBackend, ctrl: Any, nsteps: int = 1) -> dict | None:
    """Bridge control through one persistent packed direct-API plan."""
    import torch

    require_motrix_tensor_runtime(backend, require_callback_free=True)
    if not isinstance(ctrl, torch.Tensor):
        raise TypeError("MotrixSim tensor ctrl must be a torch.Tensor")
    if type(nsteps) is not int or nsteps < 1:
        raise ValueError("nsteps must be a positive integer")
    expected = (backend.num_envs, backend.num_actuators)
    if tuple(ctrl.shape) != expected:
        raise ValueError(f"MotrixSim tensor ctrl must have shape {expected}")
    if ctrl.dtype != torch.float32 or not bool(ctrl.is_contiguous()):
        raise TypeError("MotrixSim tensor ctrl must be contiguous float32")
    if ctrl.device.type not in {"cpu", "cuda"}:
        raise ValueError(f"MotrixSim tensor ctrl device must be CPU or CUDA, got {ctrl.device}")

    target = _normalize_device(ctrl.device, label="direct tensor I/O")
    plan = backend._direct_host_bridge_plan
    if plan is None or plan.device != target:
        if plan is not None:
            plan.close()
        plan = backend.compile_host_bridge_io(
            TensorIOSpec(state_fields=("qpos", "qvel"), device=target)
        )
        backend._direct_host_bridge_plan = plan
    plan.write_control(ctrl)
    return plan.step(nsteps)


def motrix_set_state_tensor(
    backend: MotrixBackend,
    env_indices: Any,
    qpos: Any,
    qvel: Any,
    randomization: Any | None = None,
) -> dict | None:
    """Bridge selected reset through one persistent packed direct-API plan."""
    import torch

    require_motrix_tensor_runtime(backend)
    if randomization is not None:
        raise NotImplementedError("MotrixSim tensor reset does not support randomization")
    values = {"env_indices": env_indices, "qpos": qpos, "qvel": qvel}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"MotrixSim tensor reset {name} must be a torch.Tensor")
        if not bool(value.is_contiguous()):
            raise ValueError(f"MotrixSim tensor reset {name} must be contiguous")
    if env_indices.ndim != 1 or env_indices.dtype != torch.int64:
        raise TypeError("MotrixSim tensor reset env_indices must be a contiguous 1-D int64 tensor")
    count = int(env_indices.shape[0])
    widths = _field_widths(backend)
    if tuple(qpos.shape) != (count, widths["qpos"]):
        raise ValueError(f"MotrixSim tensor reset qpos must have shape {(count, widths['qpos'])}")
    if tuple(qvel.shape) != (count, widths["qvel"]):
        raise ValueError(f"MotrixSim tensor reset qvel must have shape {(count, widths['qvel'])}")
    if qpos.dtype != torch.float32 or qvel.dtype != torch.float32:
        raise TypeError("MotrixSim tensor reset qpos and qvel must be float32")
    source_device = env_indices.device
    if source_device.type not in {"cpu", "cuda"}:
        raise ValueError(f"MotrixSim tensor reset device must be CPU or CUDA, got {source_device}")
    if qpos.device != source_device or qvel.device != source_device:
        raise ValueError("MotrixSim tensor reset tensors must share one device")
    target = _normalize_device(source_device, label="direct tensor I/O")
    plan = backend._direct_host_bridge_plan
    if plan is None or plan.device != target:
        if plan is not None:
            plan.close()
        plan = backend.compile_host_bridge_io(
            TensorIOSpec(state_fields=("qpos", "qvel"), device=target)
        )
        backend._direct_host_bridge_plan = plan
    return plan.apply_reset(env_indices, qpos, qvel)


@dataclass
class _TransferBuffers:
    """Stable staging buffers owned by one compiled MotrixSim plan."""

    host_packet: torch.Tensor
    selected_host: torch.Tensor
    device_packet: torch.Tensor
    selected_packet: torch.Tensor
    host_ctrl: torch.Tensor
    reset_device: torch.Tensor
    reset_host: torch.Tensor


class MotrixHostBridgeTransferPlan(HostBridgeTransferPlan):
    """One stable MotrixSim layout for explicit accelerator/CPU boundaries.

    MotrixSim remains CPU-authoritative. Control, reset, and packed reads each
    cross the device boundary at most once. CUDA transfers follow and
    synchronize the caller's current Torch stream.
    """

    def __init__(self, backend: MotrixBackend, spec: TensorIOSpec) -> None:
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

        require_motrix_tensor_runtime(backend, require_callback_free=True)
        if backend.tensor_execution() is not TensorExecution.HOST_BRIDGE:
            raise RuntimeError("MotrixSim packed tensor I/O requires the host-bridge lifecycle")
        unsupported = set(spec.state_fields) - _SUPPORTED_STATE_FIELDS
        if unsupported:
            raise ValueError(
                "MotrixSim packed state I/O supports qpos, qvel, and ctrl; "
                f"got {sorted(unsupported)}"
            )
        self.device = _normalize_device(spec.device)
        self._pin = self.device.type == "cuda"
        self._resolved_sensors = _resolve_sensor_names(backend, spec.sensor_names)
        sensor_sources = _read_sensor_sources(backend, spec.sensor_names, self._resolved_sensors)

        self._field_widths = _field_widths(backend)
        self._sensor_widths: dict[str, int] = {}
        self._offsets: dict[str, int] = {}
        cursor = 0
        for name in spec.state_fields:
            self._offsets[name] = cursor
            cursor += self._field_widths[name]
        for name in spec.sensor_names:
            source = np.asarray(sensor_sources[name], dtype=np.float32)
            width = int(source.reshape(backend.num_envs, -1).shape[1])
            self._sensor_widths[name] = width
            self._offsets[name] = cursor
            cursor += width
        self._row_width = cursor

        num_envs = backend.num_envs
        device_packet = torch.empty((num_envs, cursor), dtype=torch.float32, device=self.device)
        reset_width = 1 + self._field_widths["qpos"] + self._field_widths["qvel"]
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
            reset_device=torch.empty(
                (num_envs, reset_width), dtype=torch.int32, device=self.device
            ),
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
        # Selected reads publish only reset rows into the full-width packet.
        # Initialize every destination row from authoritative CPU state on this
        # cold path so an early selected read cannot expose allocator contents.
        # Compile-time initialization is not one of the four hot-path semantic
        # boundaries and therefore is not included in transfer_stats.
        self._pack_host_packet(None)
        initial_buffers = self._buffers()
        initial_buffers.device_packet.copy_(initial_buffers.host_packet, non_blocking=self._pin)
        _synchronize_device(initial_buffers.device_packet)

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
            raise RuntimeError("MotrixSim packed tensor host layout changed after compilation")
        if buffers.device_packet.shape != buffers.host_packet.shape:
            raise RuntimeError("MotrixSim packed tensor device layout is inconsistent")
        if buffers.selected_host.shape != buffers.host_packet.shape:
            raise RuntimeError("MotrixSim packed tensor selected host layout is inconsistent")
        if buffers.selected_packet.shape != buffers.device_packet.shape:
            raise RuntimeError("MotrixSim packed tensor selected device layout is inconsistent")
        for name, width in self._sensor_widths.items():
            if name in self._resolved_sensors.body_slots:
                prefix = next(
                    candidate for candidate in _BODY_VIEW_WIDTHS if name.startswith(candidate)
                )
                if width != _BODY_VIEW_WIDTHS[prefix]:
                    raise RuntimeError(f"MotrixSim body sensor {name!r} width changed")
            else:
                source = self._backend.get_sensor_data(name)
                expected_sensor = (self._backend.num_envs, width)
                if np.asarray(source).reshape(expected_sensor[0], -1).shape != expected_sensor:
                    raise RuntimeError(
                        f"MotrixSim sensor {name!r} layout changed after compilation"
                    )

    def _buffers(self) -> _TransferBuffers:
        if not self._buffer_slots:
            raise RuntimeError("MotrixSim packed tensor I/O plan is closed")
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

    def _pack_host_packet(self, rows: np.ndarray | None = None) -> Any:
        buffers = self._buffers()
        packet = buffers.selected_host[: rows.shape[0]] if rows is not None else buffers.host_packet
        packet_np = packet.numpy()
        _pack_state_sensors(
            self._backend,
            self._spec.state_fields,
            self._spec.sensor_names,
            packet_np,
            self._offsets,
            {**self._field_widths, **self._sensor_widths},
            rows,
            self._resolved_sensors,
        )
        return packet

    def _views(self) -> dict[str, Any]:
        packet = self._buffers().device_packet
        views: dict[str, Any] = {}
        for name in (*self._spec.state_fields, *self._spec.sensor_names):
            start = self._offsets[name]
            width = self._sensor_widths.get(name) or self._field_widths[name]
            views[name] = packet[:, start : start + width]
        return views

    def write_control(self, ctrl: Any) -> None:
        import torch

        self._require_open()
        require_motrix_tensor_runtime(self._backend, require_callback_free=True)
        if not isinstance(ctrl, torch.Tensor):
            raise TypeError("MotrixSim packed control must be a torch.Tensor")
        expected = (self._backend.num_envs, self._backend.num_actuators)
        if tuple(ctrl.shape) != expected:
            raise ValueError(f"MotrixSim packed ctrl must have shape {expected}")
        if ctrl.dtype != torch.float32 or not bool(ctrl.is_contiguous()):
            raise TypeError("MotrixSim packed ctrl must be contiguous float32")
        if not tensor_device_matches(
            (str(self.device),), ctrl.device, current_device=self.device.index
        ):
            raise ValueError(f"MotrixSim packed ctrl must live on {self.device}, got {ctrl.device}")

        self.last_timing = {}
        started = time.perf_counter()
        buffers = self._buffers()
        buffers.host_ctrl.copy_(ctrl, non_blocking=self._pin)
        _synchronize_device(ctrl)
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
        require_motrix_tensor_runtime(self._backend, require_callback_free=True)
        if not self._control_ready:
            raise RuntimeError("write_control() must complete before packed step()")
        self._control_ready = False
        self._backend.step(self._buffers().host_ctrl.numpy(), nsteps)
        return {"timing": dict(self.last_timing)}

    def read_state_sensors(self) -> Mapping[str, Any]:
        self._require_open()
        require_motrix_tensor_runtime(self._backend, require_callback_free=True)
        self._validate_layout()
        self.last_timing = {}
        started = time.perf_counter()
        self._pack_host_packet(None)
        buffers = self._buffers()
        buffers.device_packet.copy_(buffers.host_packet, non_blocking=self._pin)
        _synchronize_device(buffers.device_packet)
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
        require_motrix_tensor_runtime(self._backend)
        self._last_reset_rows_host = None
        self._last_reset_rows_device = None
        self.last_timing = {}
        if randomization is not None:
            raise NotImplementedError("MotrixSim packed reset does not support randomization")

        values = {"env_indices": env_indices, "qpos": qpos, "qvel": qvel}
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"MotrixSim packed reset {name} must be a torch.Tensor")
            if not bool(value.is_contiguous()):
                raise ValueError(f"MotrixSim packed reset {name} must be contiguous")
        if env_indices.ndim != 1 or env_indices.dtype != torch.int64:
            raise TypeError(
                "MotrixSim packed reset env_indices must be a contiguous 1-D int64 tensor"
            )
        count = int(env_indices.shape[0])
        nq = self._field_widths["qpos"]
        nv = self._field_widths["qvel"]
        if tuple(qpos.shape) != (count, nq):
            raise ValueError(f"MotrixSim packed reset qpos must have shape {(count, nq)}")
        if tuple(qvel.shape) != (count, nv):
            raise ValueError(f"MotrixSim packed reset qvel must have shape {(count, nv)}")
        if qpos.dtype != torch.float32 or qvel.dtype != torch.float32:
            raise TypeError("MotrixSim packed reset qpos and qvel must be float32")
        if any(
            not tensor_device_matches(
                (str(self.device),), value.device, current_device=self.device.index
            )
            for value in (env_indices, qpos, qvel)
        ):
            raise ValueError(f"MotrixSim packed reset tensors must live on {self.device}")
        if count == 0:
            return {"timing": {}}

        qpos_offset = 1
        qvel_offset = qpos_offset + nq
        buffers = self._buffers()
        # Packing, transfer, and the bounded row check are all part of this
        # device-to-host boundary. Finiteness is intentionally producer-owned.
        started = time.perf_counter()
        packet = buffers.reset_device[:count]
        packet[:, 0] = env_indices.to(dtype=torch.int32)
        packet[:, qpos_offset:qvel_offset] = qpos.view(dtype=torch.int32)
        packet[:, qvel_offset:] = qvel.view(dtype=torch.int32)

        buffers.reset_host[:count].copy_(packet, non_blocking=self._pin)
        _synchronize_device(packet)
        host_packet = buffers.reset_host[:count].numpy()
        rows = host_packet[:, 0].astype(np.intp, copy=True)
        host_qpos = host_packet[:, qpos_offset:qvel_offset].view(np.float32)
        host_qvel = host_packet[:, qvel_offset:].view(np.float32)
        _validate_reset_rows(rows, self._backend.num_envs, "MotrixSim packed")
        elapsed = (time.perf_counter() - started) * 1000.0
        # MotrixSim's native selected-row API materializes a sorted data slice.
        # Preserve the caller's row/value association by applying rows in sorted
        # order while retaining the original device rows for the later scatter.
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        host_qpos = np.ascontiguousarray(host_qpos[order])
        host_qvel = np.ascontiguousarray(host_qvel[order])
        rows_device = env_indices.detach().clone()
        rows_device = rows_device[torch.argsort(rows_device)]

        self._backend.set_state(
            rows,
            host_qpos.astype(self._backend._np_dtype),
            host_qvel.astype(self._backend._np_dtype),
        )
        self._last_reset_rows_host = rows
        self._last_reset_rows_device = rows_device
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
        require_motrix_tensor_runtime(self._backend)
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
        _synchronize_device(buffers.selected_packet)
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
            raise RuntimeError("MotrixSim packed tensor I/O plan is closed")


__all__ = [
    "MotrixHostBridgeTransferPlan",
    "motrix_tensor_capabilities",
    "require_motrix_tensor_runtime",
]
