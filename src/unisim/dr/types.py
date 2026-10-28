from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

import numpy as np

from .interval import (  # noqa: F401
    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_TORQUE,
    INTERVAL_TERM_PUSH,
    IntervalTermOp,
    TensorIntervalPayload,
)
from .interval import (
    require_op_body_ids as require_op_body_ids,
)

RESET_TERM_BASE_COM = "base_com_offset"
RESET_TERM_BASE_MASS = "base_mass_delta"
RESET_TERM_GRAVITY = "gravity"
RESET_TERM_BODY_IQUAT = "body_iquat"
RESET_TERM_BODY_INERTIA = "body_inertia"
RESET_TERM_BODY_IPOS = "body_ipos"
RESET_TERM_BODY_MASS = "body_mass"
RESET_TERM_DOF_ARMATURE = "dof_armature"
RESET_TERM_DOF_DAMPING = "dof_damping"
RESET_TERM_DOF_FRICTIONLOSS = "dof_frictionloss"
RESET_TERM_GEOM_FRICTION = "geom_friction"
RESET_TERM_GEOM_SIZE = "geom_size"
RESET_TERM_GEOM_SOLREF = "geom_solref"
RESET_TERM_GEOM_SOLIMP = "geom_solimp"
RESET_TERM_KP = "kp"
RESET_TERM_KD = "kd"


@dataclass(frozen=True, slots=True)
class ResetTermSpec:
    """Declared payload contract for one reset randomization term.

    ``tail_ndim`` is the rank of the canonical model table (and equivalently
    of one selected row); a per-environment table has rank ``tail_ndim + 1``.
    ``is_delta`` marks terms whose payload values are deltas merged with the
    adapter baseline (``final = baseline + delta``) instead of final absolute
    values.
    """

    name: str
    tail_ndim: int
    is_delta: bool = False


RESET_TERM_SPECS: tuple[ResetTermSpec, ...] = (
    ResetTermSpec(name=RESET_TERM_BASE_COM, tail_ndim=1, is_delta=True),
    ResetTermSpec(name=RESET_TERM_BASE_MASS, tail_ndim=0, is_delta=True),
    ResetTermSpec(name=RESET_TERM_GRAVITY, tail_ndim=1),
    ResetTermSpec(name=RESET_TERM_BODY_IQUAT, tail_ndim=2),
    ResetTermSpec(name=RESET_TERM_BODY_INERTIA, tail_ndim=2),
    ResetTermSpec(name=RESET_TERM_BODY_IPOS, tail_ndim=2),
    ResetTermSpec(name=RESET_TERM_BODY_MASS, tail_ndim=1),
    ResetTermSpec(name=RESET_TERM_DOF_ARMATURE, tail_ndim=1),
    ResetTermSpec(name=RESET_TERM_DOF_DAMPING, tail_ndim=1),
    ResetTermSpec(name=RESET_TERM_DOF_FRICTIONLOSS, tail_ndim=1),
    ResetTermSpec(name=RESET_TERM_GEOM_FRICTION, tail_ndim=2),
    ResetTermSpec(name=RESET_TERM_GEOM_SIZE, tail_ndim=2),
    ResetTermSpec(name=RESET_TERM_GEOM_SOLREF, tail_ndim=2),
    ResetTermSpec(name=RESET_TERM_GEOM_SOLIMP, tail_ndim=2),
    ResetTermSpec(name=RESET_TERM_KP, tail_ndim=1),
    ResetTermSpec(name=RESET_TERM_KD, tail_ndim=1),
)

_RESET_TERM_NAMES = frozenset(spec.name for spec in RESET_TERM_SPECS)


def reset_term_spec(name: str) -> ResetTermSpec:
    """Return the spec for ``name`` or raise a stable ``KeyError``."""
    for spec in RESET_TERM_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown reset randomization term: {name!r}")


def _validate_reset_term(term: str) -> None:
    if term not in _RESET_TERM_NAMES:
        raise ValueError(f"unknown reset term {term!r}")


@dataclass(frozen=True)
class ModelSourceDescriptor:
    """A materialized, engine-loadable model source.

    ``model_file`` is deliberately a string path. UniLab materializes assets and
    relative mesh references before constructing this descriptor; adapters own
    cold-path loading. Live engine specs, executor handles, and device arrays
    are not valid source descriptors.
    """

    model_file: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_file, str) or not self.model_file:
            raise TypeError("ModelSourceDescriptor.model_file must be a non-empty string")


class FixedVariantLayout(str, Enum):
    """Public layout guarantee required by a fixed variant plan."""

    SAME_LAYOUT = "same_layout"
    UNIFORM_PUBLIC_LAYOUT = "uniform_public_layout"


@dataclass(frozen=True)
class FixedVariantPlan:
    """Immutable per-environment model identity selected before materialization.

    ``assignment`` contains final variant indices and is normalized to a
    read-only integer NumPy array. The plan is intentionally a catalog of
    complete sources within its binding scope: whole models for the model-file
    scene-level plan, or one physical entity for EntityVariantBinding. Slot
    merging, mesh/material pooling, per-world arrays, playback representation,
    and derived-field recomputation belong to backend adapters and executors.
    """

    assignment: np.ndarray
    variants: tuple[ModelSourceDescriptor, ...]
    layout: FixedVariantLayout = FixedVariantLayout.SAME_LAYOUT

    def __post_init__(self) -> None:
        if not isinstance(self.variants, tuple):
            raise TypeError("FixedVariantPlan.variants must be a tuple")
        if not self.variants:
            raise ValueError("FixedVariantPlan.variants cannot be empty")
        if not all(isinstance(variant, ModelSourceDescriptor) for variant in self.variants):
            raise TypeError("FixedVariantPlan.variants must contain ModelSourceDescriptor values")
        if not isinstance(self.layout, FixedVariantLayout):
            raise TypeError("FixedVariantPlan.layout must be a FixedVariantLayout")

        assignment = np.asarray(object.__getattribute__(self, "assignment"))
        if assignment.ndim != 1 or assignment.size == 0:
            raise ValueError("FixedVariantPlan.assignment must be a non-empty (num_envs,) array")
        if assignment.dtype.kind not in "iu":
            raise TypeError("FixedVariantPlan.assignment must contain integers")
        if np.any(assignment < 0) or np.any(assignment >= len(self.variants)):
            raise ValueError(
                f"FixedVariantPlan.assignment values must be in [0, {len(self.variants)})"
            )
        object.__setattr__(self, "_assignment_bytes", assignment.tobytes())
        object.__setattr__(self, "_assignment_dtype", assignment.dtype.str)
        object.__setattr__(self, "assignment", self.assignment)

    def __getattribute__(self, name: str) -> Any:
        # Retain the public dataclass field (including dataclasses.replace/asdict),
        # but never expose persistent ndarray metadata through it. Only immutable
        # bytes are shared; even changing a returned view's .base cannot alter identity.
        if name == "assignment":
            return np.frombuffer(
                object.__getattribute__(self, "_assignment_bytes"),
                dtype=object.__getattribute__(self, "_assignment_dtype"),
            )
        return object.__getattribute__(self, name)

    def validate(self, num_envs: int | None = None) -> None:
        """Validate the plan, optionally against a backend batch size."""
        if num_envs is not None:
            if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
                raise ValueError("num_envs must be a positive integer")
            if self.assignment.shape != (num_envs,):
                raise ValueError(
                    f"FixedVariantPlan.assignment must have shape ({num_envs},), "
                    f"got {self.assignment.shape}"
                )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FixedVariantPlan):
            return False
        return (
            self.layout is other.layout
            and self.variants == other.variants
            and np.array_equal(self.assignment, other.assignment)
        )

    def __hash__(self) -> int:
        return hash((self.layout, self.variants, tuple(int(i) for i in self.assignment)))

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Read historical state-dict pickles without trusting their array storage.

        Pickle compatibility window: releases 1.3.0-1.4.3 (which introduced this
        class) serialized the dataclass state dict keyed by the public field name
        ``assignment``. The 1.5.0 immutable-storage rework switched fresh pickles
        to the constructor form in ``__reduce__`` — new pickles only ever carry
        the public ``assignment`` value, never a state-dict key — and added the
        private ``_assignment`` key here as a reader fallback for state dicts
        written during that rework's window; no released writer ever emitted it.
        The ``_assignment`` fallback is retained until the next major release.
        """
        FixedVariantPlan.__init__(
            self,
            state["_assignment"] if "_assignment" in state else state["assignment"],
            state["variants"],
            state["layout"],
        )

    def __reduce__(self):
        return (type(self), (self.assignment, self.variants, self.layout))


@dataclass(frozen=True)
class DomainRandomizationCapabilities:
    """Backend domain-randomization capability declaration.

    Term support is independent of payload device. Adapters own any private
    host/device conversion or fallback needed to implement the declared terms.
    """

    supported_reset_terms: frozenset[str] = field(default_factory=frozenset)
    supported_interval_terms: frozenset[str] = field(default_factory=frozenset)
    supports_fixed_variants: bool = False
    supported_fixed_variant_layouts: frozenset[FixedVariantLayout] = field(
        default_factory=frozenset
    )
    supports_per_env_playback: bool = False

    def supports_reset_term(self, term: str) -> bool:
        """Return whether the backend declares ``term``."""
        return term in self.supported_reset_terms

    def supports_interval_term(self, term: str) -> bool:
        """Return whether the backend declares one interval term."""
        return term in self.supported_interval_terms

    def get_unsupported_interval_terms(self, terms: Iterable[str]) -> frozenset[str]:
        """Return requested interval terms not declared by the backend."""
        return frozenset(term for term in terms if not self.supports_interval_term(term))

    def get_unsupported_reset_terms(
        self, requested_terms: Iterable[str]
    ) -> frozenset[str]:
        """Return requested reset terms not declared by the backend."""
        return frozenset(term for term in requested_terms if not self.supports_reset_term(term))

    def fixed_variant_rejections(self, plan: FixedVariantPlan) -> tuple[str, ...]:
        """Return human-readable reasons why ``plan`` cannot be realized."""
        if not isinstance(plan, FixedVariantPlan):
            raise TypeError("plan must be a FixedVariantPlan")
        reasons: list[str] = []
        if not self.supports_fixed_variants:
            reasons.append("fixed variants are unsupported")
        if plan.layout not in self.supported_fixed_variant_layouts:
            reasons.append(f"fixed variant layout '{plan.layout.value}' is unsupported")
        return tuple(reasons)


@dataclass
class TensorResetRandomizationPayload:
    """The single public reset randomization payload, torch-first.

    Values are Torch tensors; the base contract deliberately stays Torch-free,
    so fields are typed ``Any`` and validated lazily by
    :func:`validate_tensor_reset_payload`. Each value is a dense selected-row
    table shaped ``(num_rows, *canonical_tail)``. Non-delta terms hold FINAL
    ABSOLUTE values. The delta terms ``base_mass_delta`` and
    ``base_com_offset`` hold deltas that the consuming adapter merges with its
    baseline (``final = baseline + delta``); the packed reset transaction performs
    that merge in Torch at commit time. Backends check requested terms against
    :class:`DomainRandomizationCapabilities` and own any payload-device handling.
    """

    base_mass_delta: Any | None = None
    base_com_offset: Any | None = None
    gravity: Any | None = None
    body_iquat: Any | None = None
    body_inertia: Any | None = None
    body_ipos: Any | None = None
    body_mass: Any | None = None
    dof_armature: Any | None = None
    geom_friction: Any | None = None
    kp: Any | None = None
    kd: Any | None = None
    # Dense model-column tables for selected reset rows. Geometry bounds are
    # derived by the adapter; callers must never supply independent bounds.
    geom_size: Any | None = None
    geom_solref: Any | None = None
    geom_solimp: Any | None = None
    dof_damping: Any | None = None
    dof_frictionloss: Any | None = None

    def requested_terms(self) -> frozenset[str]:
        return frozenset(
            payload_field.name
            for payload_field in fields(self)
            if getattr(self, payload_field.name) is not None
        )

    def is_empty(self) -> bool:
        return not self.requested_terms()


def validate_tensor_reset_payload(
    payload: TensorResetRandomizationPayload,
    *,
    device: Any | None = None,
    allowed_terms: Iterable[str] | None = None,
) -> None:
    """Validate a torch-first reset payload with a lazy Torch import.

    Every populated field must be a ``torch.Tensor``. When ``device`` is given,
    all tensors must live on that device; when ``allowed_terms`` is given, any
    requested term outside the allowlist fails closed naming the term. An empty
    payload is valid and never imports Torch.
    """
    if not isinstance(payload, TensorResetRandomizationPayload):
        raise TypeError("payload must be a TensorResetRandomizationPayload")
    requested = payload.requested_terms()
    if allowed_terms is not None:
        unsupported = requested - frozenset(allowed_terms)
        if unsupported:
            raise ValueError(f"reset terms not allowed here: {sorted(unsupported)}")
    if not requested:
        return
    import torch

    accepted_device = None if device is None else torch.device(device)
    for payload_field in fields(payload):
        value = getattr(payload, payload_field.name)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"reset field {payload_field.name!r} must be a torch.Tensor, "
                f"got {type(value).__name__}"
            )
        if accepted_device is not None and value.device != accepted_device:
            raise ValueError(
                f"reset field {payload_field.name!r} must live on {accepted_device}, "
                f"got {value.device}"
            )


@dataclass
class IntervalRandomizationPlan:
    """Scheduled interval randomization request.

    Populate ``ops`` with :class:`IntervalTermOp` entries.
    """

    ops: tuple[IntervalTermOp, ...] = ()

    def iter_ops(self) -> tuple[IntervalTermOp, ...]:
        """Return the explicit interval operations."""
        return self.ops

    def is_empty(self) -> bool:
        return not self.ops
