"""Shared host-side behavior for the Isaac subprocess adapters.

IsaacGym and IsaacSim intentionally remain separate public backends because
they use different interpreters and native runtimes.  This module owns only
their duplicated mapped-scene host logic: body-wrench staging and projection
of frame sensors from public CUDA body state.  It never imports an Isaac SDK.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from unisim.dr._host import interval_payload_to_host
from unisim.dr.interval import INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE, IntervalTermOp


def initialize_staged_body_wrench(backend: Any) -> np.ndarray | None:
    """Allocate the mapped-scene wrench buffer after scene preparation."""
    scene = getattr(backend, "_entity_scene", None)
    if scene is None:
        return None
    return np.zeros((backend._num_envs, scene.layout.nbody, 6), dtype=np.float32)


def clear_staged_body_wrench(backend: Any) -> None:
    """Consume any staged wrench before a new interval plan is applied."""
    if getattr(backend, "_entity_scene", None) is None:
        return
    wrench = backend._staged_body_wrench
    if wrench is not None:
        wrench.fill(0.0)
    backend._body_wrench_pending = False


def interval_term_handlers(
    backend: Any, parent_handlers: Callable[[], dict[str, Callable[[IntervalTermOp], None]]]
) -> dict[str, Callable[[IntervalTermOp], None]]:
    """Return mapped-scene handlers, or the shared owner's model-file handlers."""
    if getattr(backend, "_entity_scene", None) is None:
        return parent_handlers()
    return {
        INTERVAL_TERM_BODY_FORCE: backend._apply_interval_body_force,
        INTERVAL_TERM_BODY_TORQUE: backend._apply_interval_body_torque,
    }


def apply_interval_body_force(backend: Any, op: IntervalTermOp) -> None:
    if op.body_ids is None:
        raise ValueError("interval body force requires body_ids")
    backend.apply_body_force(op.body_ids, interval_payload_to_host(op.payload))


def apply_interval_body_torque(
    backend: Any, op: IntervalTermOp, *, direct_torque: bool = False
) -> None:
    if op.body_ids is None:
        raise ValueError("interval body torque requires body_ids")
    torque = interval_payload_to_host(op.payload)
    if direct_torque:
        backend._apply_body_torque(op.body_ids, torque)
        return
    zero_force = np.zeros((backend._num_envs, len(op.body_ids), 3), dtype=np.float32)
    backend.apply_body_force(op.body_ids, zero_force, torque=torque)


def mapped_default_body_rows(
    backend: Any, canonical: np.ndarray, field: str, rows: np.ndarray, width: int | None = None
) -> np.ndarray:
    """Build per-row variant-assigned public body-property defaults."""
    scene = getattr(backend, "_entity_scene", None)
    if scene is None:
        raise backend._require_mapped_entity_scene()
    layout = scene.layout
    shape = (rows.size, layout.nbody) if width is None else (rows.size, layout.nbody, width)
    values = np.broadcast_to(canonical, shape).copy()
    for entity, entry in zip(layout.entities, scene.payload["scene_entities"]):
        defaults = np.asarray(
            [entry["variants"][int(entry["assignment"][int(env)])][field] for env in rows],
            dtype=np.float32,
        )
        expected = (
            (rows.size, len(entity.body_ids))
            if width is None
            else (rows.size, len(entity.body_ids), width)
        )
        if defaults.shape != expected or not np.isfinite(defaults).all():
            raise backend._worker_error(
                f"compiled variant {field} is malformed for entity {entity.name}"
            )
        if width is None:
            values[:, list(entity.body_ids)] = defaults
        else:
            values[:, list(entity.body_ids), :] = defaults
    return values


def mapped_default_full_geom_friction(backend: Any, rows: np.ndarray) -> np.ndarray:
    """Build full-width variant-assigned public geom-friction defaults."""
    scene = getattr(backend, "_entity_scene", None)
    if scene is None:
        raise backend._require_mapped_entity_scene()
    layout = scene.layout
    values = np.zeros((rows.size, layout.ngeom, 3), dtype=np.float32)
    offset = 0
    for entity, entry in zip(layout.entities, scene.payload["scene_entities"]):
        count = len(entity.geoms)
        if count:
            defaults = np.asarray(
                [
                    entry["variants"][int(entry["assignment"][int(env)])]["geom_friction"]
                    for env in rows
                ],
                dtype=np.float32,
            )
            values[:, offset : offset + count, 0] = defaults[:, :, 0]
            values[:, offset : offset + count, 1] = defaults[:, :, 0]
        offset += count
    return values


def mapped_default_actuator_rows(backend: Any, field: str, rows: np.ndarray) -> np.ndarray:
    """Build per-row variant-assigned public actuator-gain defaults."""
    scene = getattr(backend, "_entity_scene", None)
    if scene is None:
        raise backend._require_mapped_entity_scene()
    layout = scene.layout
    values = np.zeros((rows.size, layout.nu), dtype=np.float32)
    for entity, entry in zip(layout.entities, scene.payload["scene_entities"]):
        if not entity.actuator_indices:
            continue
        positions = [
            next(i for i, joint in enumerate(entity.joints) if joint.name == name)
            for name in entity.actuator_joint_names
        ]
        defaults = np.asarray(
            [entry["variants"][int(entry["assignment"][int(env)])][field] for env in rows],
            dtype=np.float32,
        )
        values[:, list(entity.actuator_indices)] = defaults[:, positions]
    return values


def mapped_default_dof_rows(backend: Any, field: str, rows: np.ndarray) -> np.ndarray:
    """Build per-row variant-assigned public DOF-property defaults."""
    scene = getattr(backend, "_entity_scene", None)
    if scene is None:
        raise backend._require_mapped_entity_scene()
    layout = scene.layout
    values = np.zeros((rows.size, layout.nv), dtype=np.float32)
    for entity, entry in zip(layout.entities, scene.payload["scene_entities"]):
        if not entity.joints:
            continue
        columns = [joint.qvel_indices[0] for joint in entity.joints]
        defaults = np.asarray(
            [entry["variants"][int(entry["assignment"][int(env)])][field] for env in rows],
            dtype=np.float32,
        )
        values[:, columns] = defaults
    return values


def apply_body_force(
    backend: Any,
    parent_apply_body_force: Callable[..., None],
    body_ids: np.ndarray,
    force: np.ndarray,
    torque: np.ndarray | None = None,
    *,
    reject_callback_writes: bool = False,
) -> None:
    """Stage a world-frame wrench for the next mapped-scene step."""
    if getattr(backend, "_entity_scene", None) is None:
        parent_apply_body_force(body_ids, force, torque)
        return
    if reject_callback_writes:
        backend._reject_wrench_write_inside_pre_step_control("apply_body_force")
    backend._require_state("interval body force perturbation")
    staged = backend._staged_body_wrench
    assert staged is not None
    body_ids_np = np.asarray(body_ids)
    if (
        body_ids_np.ndim != 1
        or body_ids_np.dtype.kind not in "iu"
        or np.issubdtype(body_ids_np.dtype, np.bool_)
    ):
        raise ValueError("body_ids must be a one-dimensional integer array")
    if np.any(body_ids_np < 0) or np.any(body_ids_np >= staged.shape[1]):
        raise ValueError(
            f"body ids must be in [0, {staged.shape[1]}), "
            f"got range [{body_ids_np.min(initial=0)}, {body_ids_np.max(initial=0)}]"
        )
    body_ids_np = body_ids_np.astype(np.intp, copy=False)
    expected_shape = (backend._num_envs, body_ids_np.size, 3)
    force_np = np.asarray(force, dtype=np.float32)
    if force_np.shape != expected_shape:
        raise ValueError(f"body force must have shape {expected_shape}, got {force_np.shape}")
    if not np.isfinite(force_np).all():
        raise ValueError("body force contains NaN or Inf")
    torque_np = None
    if torque is not None:
        torque_np = np.asarray(torque, dtype=np.float32)
        if torque_np.shape != expected_shape:
            raise ValueError(f"body torque must have shape {expected_shape}, got {torque_np.shape}")
        if not np.isfinite(torque_np).all():
            raise ValueError("body torque contains NaN or Inf")
    for body_offset, body_id in enumerate(body_ids_np):
        staged[:, int(body_id), 0:3] += force_np[:, body_offset, :]
        if torque_np is not None:
            staged[:, int(body_id), 3:6] += torque_np[:, body_offset, :]
    backend._body_wrench_pending = bool(np.any(staged) or backend._body_wrench_pending)


def append_step_payload_body_wrench(
    backend: Any, payload: dict[str, Any], body_wrench: np.ndarray | None = None
) -> dict[str, Any]:
    """Encode pending wrench with version-independent raw C-order bytes."""
    if body_wrench is not None:
        payload["body_wrench"] = body_wrench.tobytes(order="C")
    elif getattr(backend, "_entity_scene", None) is not None and backend._body_wrench_pending:
        staged = backend._staged_body_wrench
        assert staged is not None
        payload["body_wrench"] = staged.tobytes(order="C")
    return payload


def consume_step_payload_body_wrench(backend: Any, payload: dict[str, Any]) -> None:
    """Clear staged wrench after the worker consumes the STEP payload."""
    if "body_wrench" in payload:
        staged = backend._staged_body_wrench
        if staged is not None:
            staged.fill(0.0)
        backend._body_wrench_pending = False


def cuda_derived_sensor_specs(sensor_map: Any) -> dict[str, tuple[Any, int]]:
    """Return frame sensors projectable from public CUDA body state."""
    return {
        name: mapped
        for name, mapped in getattr(sensor_map, "items", lambda: [])()
        if mapped[0].kind in ("framepos", "framequat", "framezaxis")
    }


def cuda_derived_sensor_view(
    name: str,
    host: Any,
    specs: dict[str, tuple[Any, int]],
    torch: Any,
) -> Any:
    """Project one qualified frame sensor from a backend-owned body view."""
    candidates = [
        (qualified, mapped)
        for qualified, mapped in specs.items()
        if qualified == name or qualified.endswith(f"/{name}")
    ]
    if len(candidates) != 1:
        return None
    from unisim.utils import math as tensor_math

    _, (spec, _body_id) = candidates[0]
    body_name = spec.body_name
    body_pos = host.get_sensor_view(f"track_pos_w_{body_name}")
    body_quat = host.get_sensor_view(f"track_quat_w_{body_name}")
    if spec.kind == "framepos":
        local_pos = torch.as_tensor(
            spec.local_pos, dtype=torch.float32, device=body_pos.device
        ).expand(body_pos.shape[0], 3)
        return body_pos + tensor_math.quat_apply(body_quat, local_pos)
    local_quat = torch.as_tensor(
        spec.local_quat, dtype=torch.float32, device=body_quat.device
    ).expand(body_quat.shape[0], 4)
    world_quat = tensor_math.quat_mul(body_quat, local_quat)
    if spec.kind == "framequat":
        return world_quat
    world_z = torch.zeros_like(body_pos)
    world_z[:, 2] = 1.0
    return tensor_math.quat_apply(world_z, world_z)


__all__ = [
    "append_step_payload_body_wrench",
    "apply_body_force",
    "apply_interval_body_force",
    "apply_interval_body_torque",
    "clear_staged_body_wrench",
    "consume_step_payload_body_wrench",
    "cuda_derived_sensor_specs",
    "cuda_derived_sensor_view",
    "initialize_staged_body_wrench",
    "interval_term_handlers",
    "mapped_default_actuator_rows",
    "mapped_default_body_rows",
    "mapped_default_dof_rows",
    "mapped_default_full_geom_friction",
]
