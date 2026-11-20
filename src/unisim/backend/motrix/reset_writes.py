"""Batched reset randomization through MotrixSim ``WriteProgram`` plans.

MotrixSim 1.0.0 compiles per-object override operations (link mass/COM, geom
friction, geom size, actuator gains) into one reusable write program whose
buffer rows are filled with plain NumPy assignments and scattered with a single
``execute`` call (upstream ``motrixsim#20``).  This module builds one program per batched
``SceneData`` on the cold path and stages reset-randomization tables into its
field views on the hot path, replacing the historical per-link / per-geom /
per-actuator setter loops.

The program is compiled with ``reset=False``: the backend keeps owning the
``SceneData.reset`` call so override restoration, gravity ordering, and state
writes stay exactly on the historical path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

FIELD_MASS = "mass"
FIELD_COM = "com"
FIELD_FRICTION = "friction"
FIELD_SIZE = "size"
FIELD_KP = "kp"
FIELD_KD = "kd"

# Distinctive cold-path probe marker for mass-override eligibility.  Real link
# masses are positive and orders of magnitude away from this range.
_ELIGIBILITY_MARKER_BASE = 12345.0


@dataclass
class MotrixResetWriteProgram:
    """A reusable write program plus the native-index maps of its field columns."""

    program: Any
    mass_link_ids: np.ndarray  # (L,) native link index per mass/COM buffer column
    friction_geom_ids: np.ndarray  # (G,) native geom index per friction buffer column
    size_geom_ids: np.ndarray  # (S,) native geom index per size buffer column
    gain_actuator_ids: np.ndarray  # (A,) native actuator index per kp/kd buffer column
    default_mass: np.ndarray  # (L,)
    default_ipos: np.ndarray  # (L, 3)
    default_friction: np.ndarray  # (G, 3)
    default_size: np.ndarray  # (S, 3)
    size_required: np.ndarray  # (S, 3) bool: components the primitive requires positive
    default_kp: np.ndarray  # (A,)
    default_kd: np.ndarray  # (A,)

    def field_names(self) -> frozenset[str]:
        return frozenset(self.program.keys())


def _probe_mass_override_eligible(link: Any, scratch: Any, marker: float) -> bool:
    """Return whether the SDK applies a mass override to ``link``.

    World-fixed and mocap links silently ignore mass-property overrides (the
    historical setter loop no-ops on them); the write sources reject them at
    compile time instead.  Probe with a marker write on throwaway data so the
    compiled selection mirrors the setter semantics exactly.
    """

    try:
        link.set_mass_override(scratch, np.array([marker], dtype=np.float32))
        applied = float(
            np.asarray(link.get_mass_override(scratch), dtype=np.float32).reshape(-1)[0]
        )
    except Exception:
        return False
    return applied == marker


def build_reset_write_program(
    *,
    model: Any,
    data: Any,
    links_by_id: Mapping[int, Any],
    geom_friction_override_ids: Sequence[int],
    geom_size_override_ids: Sequence[int],
    geom_size_required: np.ndarray,
    position_actuators: Sequence[Any],
    supports_position_actuator_gains: bool,
    default_body_mass: np.ndarray,
    default_body_ipos: np.ndarray,
    default_geom_friction: np.ndarray,
    default_geom_size: np.ndarray,
    default_actuator_kp: np.ndarray,
    default_actuator_kd: np.ndarray,
) -> MotrixResetWriteProgram | None:
    """Compile one reset-randomization write program against ``data``.

    Returns ``None`` when the model exposes no writable override target, in
    which case callers keep the historical setter path.
    """

    import motrixsim as mtx
    from motrixsim import write

    ordered_links = sorted(links_by_id.items())
    scratch = mtx.SceneData(model, batch=[1])  # pyright: ignore[reportPossiblyUnbound]
    eligible_link_ids = [
        link_id
        for offset, (link_id, link) in enumerate(ordered_links)
        if _probe_mass_override_eligible(link, scratch, _ELIGIBILITY_MARKER_BASE + offset)
    ]
    gain_actuator_ids = sorted(int(actuator.index) for actuator in position_actuators)
    friction_geom_ids = sorted(int(geom_id) for geom_id in geom_friction_override_ids)
    size_geom_ids = sorted(int(geom_id) for geom_id in geom_size_override_ids)

    fields: dict[str, Any] = {}
    if eligible_link_ids:
        fields[FIELD_MASS] = write.LinkMassOverride(eligible_link_ids)
        fields[FIELD_COM] = write.LinkCenterOfMassOverride(eligible_link_ids)
    if friction_geom_ids:
        fields[FIELD_FRICTION] = write.GeomFrictionOverride(friction_geom_ids)
    if size_geom_ids:
        fields[FIELD_SIZE] = write.GeomSizeOverride(size_geom_ids)
    if supports_position_actuator_gains and gain_actuator_ids:
        fields[FIELD_KP] = write.ActuatorKpOverride(gain_actuator_ids)
        fields[FIELD_KD] = write.ActuatorDampingOverride(gain_actuator_ids)
    if not fields:
        return None

    plan = model.compile_write(fields, reset=False, reset_overrides=False, forward_kinematic=False)
    program = plan.allocate(data)
    mass_link_ids = np.asarray(eligible_link_ids, dtype=np.intp)
    friction_ids = np.asarray(friction_geom_ids, dtype=np.intp)
    size_ids = np.asarray(size_geom_ids, dtype=np.intp)
    gain_ids = np.asarray(gain_actuator_ids, dtype=np.intp)
    return MotrixResetWriteProgram(
        program=program,
        mass_link_ids=mass_link_ids,
        friction_geom_ids=friction_ids,
        size_geom_ids=size_ids,
        gain_actuator_ids=gain_ids,
        default_mass=np.asarray(default_body_mass, dtype=np.float32)[mass_link_ids].copy(),
        default_ipos=np.asarray(default_body_ipos, dtype=np.float32)[mass_link_ids].copy(),
        default_friction=np.asarray(default_geom_friction, dtype=np.float32)[friction_ids].copy(),
        default_size=np.asarray(default_geom_size, dtype=np.float32)[size_ids].copy(),
        size_required=np.asarray(geom_size_required, dtype=bool)[size_ids].copy(),
        default_kp=np.asarray(default_actuator_kp, dtype=np.float32)[gain_ids].copy(),
        default_kd=np.asarray(default_actuator_kd, dtype=np.float32)[gain_ids].copy(),
    )


def stage_reset_randomization(
    rwp: MotrixResetWriteProgram,
    env_ids: np.ndarray,
    randomization: Mapping[str, np.ndarray],
    *,
    num_reset: int,
    num_links: int,
    num_geoms: int,
    num_actuators: int,
    base_link_id: int,
    default_body_mass_rows: np.ndarray,
    default_body_ipos_rows: np.ndarray,
    default_geom_friction: np.ndarray,
    friction_override_ids: Sequence[int],
    default_geom_size: np.ndarray,
    geom_size_override_ids: Sequence[int],
    supports_position_actuator_gains: bool,
    coerce_field: Callable[..., np.ndarray],
) -> None:
    """Fill program buffer rows for ``env_ids`` from a reset-randomization payload.

    Fields absent from the payload are filled with model defaults so stale
    buffer content never leaks into worlds that a previous reset randomized.
    Non-collision geom friction stays fail-closed exactly like the setter path.
    """

    views = rwp.program
    fields = rwp.field_names()
    env_ids = np.asarray(env_ids, dtype=np.int64)

    mass_field = randomization.get("body_mass")
    mass_delta = randomization.get("base_mass_delta")
    if FIELD_MASS in fields:
        if mass_field is None and mass_delta is None:
            views[FIELD_MASS][env_ids] = rwp.default_mass
        else:
            if mass_field is None:
                body_mass = np.broadcast_to(
                    np.asarray(default_body_mass_rows, dtype=np.float32),
                    (num_reset, num_links),
                ).copy()
            else:
                body_mass = coerce_field(
                    mass_field, name="body_mass", num_reset=num_reset, shaped_tail=(num_links,)
                )
            base_selected = rwp.mass_link_ids == int(base_link_id)
            if mass_delta is not None and bool(np.any(base_selected)):
                body_mass[:, int(base_link_id)] += np.asarray(mass_delta, dtype=np.float32).reshape(
                    num_reset
                )
            views[FIELD_MASS][env_ids] = body_mass[:, rwp.mass_link_ids]
    elif mass_field is not None or mass_delta is not None:
        raise NotImplementedError("Motrix link mass override is not available")

    ipos_field = randomization.get("body_ipos")
    com_offset = randomization.get("base_com_offset")
    if FIELD_COM in fields:
        if ipos_field is None and com_offset is None:
            views[FIELD_COM][env_ids] = rwp.default_ipos
        else:
            if ipos_field is None:
                body_ipos = np.broadcast_to(
                    np.asarray(default_body_ipos_rows, dtype=np.float32),
                    (num_reset, num_links, 3),
                ).copy()
            else:
                body_ipos = coerce_field(
                    ipos_field,
                    name="body_ipos",
                    num_reset=num_reset,
                    shaped_tail=(num_links, 3),
                )
            base_selected = rwp.mass_link_ids == int(base_link_id)
            if com_offset is not None and bool(np.any(base_selected)):
                body_ipos[:, int(base_link_id), :] += np.asarray(
                    com_offset, dtype=np.float32
                ).reshape(num_reset, 3)
            views[FIELD_COM][env_ids] = body_ipos[:, rwp.mass_link_ids, :]
    elif ipos_field is not None or com_offset is not None:
        raise NotImplementedError("Motrix link center-of-mass override is not available")

    friction_field = randomization.get("geom_friction")
    if FIELD_FRICTION in fields:
        if friction_field is None:
            views[FIELD_FRICTION][env_ids] = rwp.default_friction
        else:
            geom_friction = coerce_field(
                friction_field,
                name="geom_friction",
                num_reset=num_reset,
                shaped_tail=(num_geoms, 3),
            )
            override_ids = frozenset(int(geom_id) for geom_id in friction_override_ids)
            unsupported_ids = sorted(set(range(num_geoms)) - override_ids)
            if unsupported_ids and not np.allclose(
                geom_friction[:, unsupported_ids, :],
                np.asarray(default_geom_friction, dtype=np.float32)[unsupported_ids, :][None],
            ):
                raise ValueError(
                    "Motrix geom friction override only supports collision geoms; "
                    f"non-collision geom ids were modified: {unsupported_ids}"
                )
            views[FIELD_FRICTION][env_ids] = geom_friction[:, rwp.friction_geom_ids, :]
    elif friction_field is not None:
        raise NotImplementedError("Motrix geom friction override is not available")

    size_field = randomization.get("geom_size")
    if FIELD_SIZE in fields:
        if size_field is None:
            views[FIELD_SIZE][env_ids] = rwp.default_size
        else:
            geom_size = coerce_field(
                size_field,
                name="geom_size",
                num_reset=num_reset,
                shaped_tail=(num_geoms, 3),
            )
            if not np.isfinite(geom_size).all():
                raise ValueError("geom_size must contain only finite values")
            override_ids = frozenset(int(geom_id) for geom_id in geom_size_override_ids)
            unsupported_ids = sorted(set(range(num_geoms)) - override_ids)
            if unsupported_ids and not np.allclose(
                geom_size[:, unsupported_ids, :],
                np.asarray(default_geom_size, dtype=np.float32)[unsupported_ids, :][None],
            ):
                raise ValueError(
                    "Motrix geom size override only supports primitive geoms; "
                    f"unsupported geom ids were modified: {unsupported_ids}"
                )
            selected = geom_size[:, rwp.size_geom_ids, :]
            required = rwp.size_required
            if not (selected[:, required] > 0.0).all():
                raise ValueError("geom_size required components must be positive")
            if not (selected[:, ~required] == 0.0).all():
                raise ValueError("geom_size unused components must stay zero")
            views[FIELD_SIZE][env_ids] = selected
    elif size_field is not None:
        raise NotImplementedError("Motrix geom size override is not available")

    for field_name, term in ((FIELD_KP, "kp"), (FIELD_KD, "kd")):
        field_values = randomization.get(term)
        if field_name in fields:
            if field_values is None:
                defaults = rwp.default_kp if field_name == FIELD_KP else rwp.default_kd
                views[field_name][env_ids] = defaults
            else:
                values = np.asarray(field_values, dtype=np.float32)
                expected = (num_reset, num_actuators)
                if values.shape != expected:
                    raise ValueError(f"{term} must have shape {expected}, got {values.shape}")
                views[field_name][env_ids] = values[:, rwp.gain_actuator_ids]
        elif field_values is not None:
            if not supports_position_actuator_gains:
                raise NotImplementedError(
                    "Motrix actuator gain override requires all-position-actuator models"
                )
            raise NotImplementedError(f"Motrix actuator {term} override is not available")
