"""Per-world model-field expansion for the independent ``mjwarp`` backend.

MuJoCo Warp declares the leading dimension of randomizable ``Model`` fields as
``"*"``: every kernel reads ``field[worldid % field.shape[0]]``, so tiling a
field from ``(1, ...)`` to ``(nworld, ...)`` yields per-world semantics without
kernel changes.  Expansion replaces the array allocation, so it is a cold-path
operation: the backend expands the declared set once during construction,
*before* CUDA graph capture, and all later DR writes are in-place ``assign``
uploads into the same fixed-address arrays (graph-safe).

The field list is ported from mjlab's ``expand_model_fields`` usage and checked
against the pinned mujoco-warp 3.11 ``Model`` dataclass.  Derived fields are
expanded alongside their inputs because the ``set_const*`` recompute kernels
size their launch grid from the *output* field's leading dimension.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Payload-writable fields that require ``mujoco_warp.set_const`` afterwards
# (mass/inertial family; superset of the set_const_0 level).
SET_CONST_FIELDS = ("body_mass", "body_ipos", "body_iquat")
# Payload-writable fields that require ``mujoco_warp.set_const_0`` afterwards.
SET_CONST_0_FIELDS = ("body_inertia", "dof_armature")
# Payload-writable fields that need no derived-quantity recomputation.
NO_RECOMPUTE_FIELDS = (
    "geom_friction",
    "actuator_gainprm",
    "actuator_biasprm",
    "geom_size",
    "geom_rbound",
    "geom_aabb",
    "geom_solref",
    "geom_solimp",
    "dof_damping",
    "dof_frictionloss",
)
# Derived fields recomputed by the ``set_const*`` family.  They must be
# per-world as well: the recompute kernels launch over ``field.shape[0]`` and
# the step/forward kernels index them per world.
DERIVED_FIELDS = (
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
    "actuator_acc0",
)

EXPANDED_MODEL_FIELDS: tuple[str, ...] = (
    SET_CONST_FIELDS + SET_CONST_0_FIELDS + NO_RECOMPUTE_FIELDS + DERIVED_FIELDS
)


class PrimitiveGeomBounds:
    """Cold-path geometry classification for batch-only bounds recomputation.

    Uses official MuJoCo primitive geometry formulas. Meshes, planes, height
    fields and SDFs may appear in dense payloads unchanged, but resizing them
    is unsupported. No model/XML metadata is consulted during reset.
    """

    def __init__(self, geom_types: np.ndarray, geom_enum: Any) -> None:
        count = geom_types.size
        self.halfsize_map = np.zeros((count, 3, 3), dtype=np.float32)
        self.radius_axes = np.zeros((count, 3), dtype=np.float32)
        self.required_positive = np.zeros((count, 3), dtype=bool)
        supported = np.zeros(count, dtype=bool)
        for kind, mapping, radius_axes, positive in (
            ("SPHERE", ((1, 0, 0),) * 3, (1, 0, 0), (True, False, False)),
            ("CAPSULE", ((1, 0, 0), (1, 0, 0), (1, 1, 0)), (1, 1, 0), (True, False, False)),
            ("ELLIPSOID", np.eye(3), (1, 1, 1), (True, True, True)),
            ("CYLINDER", ((1, 0, 0), (1, 0, 0), (0, 1, 0)), (1, 1, 0), (True, True, False)),
            ("BOX", np.eye(3), (1, 1, 1), (True, True, True)),
        ):
            ids = np.flatnonzero(geom_types == int(getattr(geom_enum, f"mjGEOM_{kind}")))
            self.halfsize_map[ids] = mapping
            self.radius_axes[ids] = radius_axes
            self.required_positive[ids] = positive
            supported[ids] = True
        self.supported_ids = np.flatnonzero(supported)
        self.unsupported_ids = np.flatnonzero(~supported)
        self.capsule_ids = np.flatnonzero(geom_types == int(geom_enum.mjGEOM_CAPSULE))
        self.ellipsoid_ids = np.flatnonzero(geom_types == int(geom_enum.mjGEOM_ELLIPSOID))

    def compute(
        self,
        sizes: np.ndarray,
        previous_sizes: np.ndarray,
        previous_rbound: np.ndarray,
        previous_aabb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate a dense size update and derive detached selected-row bounds."""
        unsupported = self.unsupported_ids
        if np.any(sizes[:, unsupported] != previous_sizes[:, unsupported]):
            raise NotImplementedError(
                "mjwarp geom_size supports resizing sphere, capsule, ellipsoid, cylinder and "
                "box primitives only; a non-primitive geometry was changed"
            )
        if np.any(sizes[:, self.supported_ids] < 0) or np.any(
            (sizes <= 0) & self.required_positive[None, :, :]
        ):
            raise ValueError(
                "geom_size primitive dimensions must be non-negative with positive radii/extents"
            )
        radius_vector = sizes * self.radius_axes[None, :, :]
        with np.errstate(over="ignore", invalid="ignore"):
            radii = np.linalg.norm(radius_vector, axis=-1)
        radii[:, self.capsule_ids] = radius_vector[:, self.capsule_ids].sum(axis=-1)
        radii[:, self.ellipsoid_ids] = sizes[:, self.ellipsoid_ids].max(axis=-1)
        halves = np.einsum("gij,rgj->rgi", self.halfsize_map, sizes)
        if not np.isfinite(radii).all() or not np.isfinite(halves).all():
            raise ValueError("geom_size derived bounds must be finite float32 values")
        rbound = previous_rbound.copy()
        aabb = previous_aabb.copy()
        rbound[:, self.supported_ids] = radii[:, self.supported_ids]
        aabb[:, self.supported_ids, 0] = 0
        aabb[:, self.supported_ids, 1] = halves[:, self.supported_ids]
        return rbound, aabb


def expand_model_fields(warp: Any, model: Any, nworld: int) -> tuple[str, ...]:
    """Tile the declared DR model fields from ``(1, ...)`` to ``(nworld, ...)``.

    Returns the names actually expanded.  Fields already per-world (e.g. from a
    prior expansion) are skipped, mirroring mjlab's guard.  A single-world
    backend keeps the shared arrays untouched.
    """
    if nworld <= 1:
        return ()
    model_fields = getattr(model, "__dataclass_fields__", None)
    if model_fields is None:
        raise TypeError(
            f"mjwarp DR expansion requires a mujoco_warp.Model dataclass, got {type(model)}"
        )
    expanded: list[str] = []
    for name in EXPANDED_MODEL_FIELDS:
        if name not in model_fields:
            raise RuntimeError(
                f"mujoco-warp Model no longer declares DR field {name!r}; update the "
                "mjwarp expansion field list for the pinned mujoco-warp version"
            )
        array = getattr(model, name)
        if array.shape[0] == nworld:
            continue
        if array.shape[0] != 1:
            raise RuntimeError(
                f"mujoco-warp Model field {name!r} has unexpected leading dim "
                f"{array.shape[0]}; expected 1 (shared) or {nworld} (per-world)"
            )
        host = np.asarray(array.numpy())
        tiled = np.ascontiguousarray(np.broadcast_to(host, (nworld, *host.shape[1:])))
        replacement = warp.array(
            shape=(nworld, *array.shape[1:]),
            dtype=array.dtype,
            device=array.device,
        )
        replacement.assign(tiled)
        setattr(model, name, replacement)
        expanded.append(name)
    return tuple(expanded)
