from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

import numpy as np

from .interval import (
    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_TORQUE,
    INTERVAL_TERM_PUSH,
    IntervalTermOp,
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


_RESET_TERM_NAMES = frozenset(
    (
        RESET_TERM_BASE_COM,
        RESET_TERM_BASE_MASS,
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
        RESET_TERM_GEOM_SOLREF,
        RESET_TERM_GEOM_SOLIMP,
        RESET_TERM_KP,
        RESET_TERM_KD,
    )
)


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
    complete model sources: slot merging, mesh/material pooling, per-world
    arrays, playback representation, and derived-field recomputation belong to
    backend adapters and their executors.
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

        assignment = np.asarray(self.assignment)
        if assignment.ndim != 1 or assignment.size == 0:
            raise ValueError("FixedVariantPlan.assignment must be a non-empty (num_envs,) array")
        if assignment.dtype.kind not in "iu":
            raise TypeError("FixedVariantPlan.assignment must contain integers")
        if np.any(assignment < 0) or np.any(assignment >= len(self.variants)):
            raise ValueError(
                f"FixedVariantPlan.assignment values must be in [0, {len(self.variants)})"
            )
        if not assignment.flags.writeable:
            assignment = assignment.copy()
        assignment.setflags(write=False)
        object.__setattr__(self, "assignment", assignment)

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
        return hash((self.layout, self.variants, self.assignment.tobytes()))

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore the assignment as read-only across process boundaries."""
        restored = dict(state)
        assignment = np.array(restored["assignment"], copy=True)
        assignment.setflags(write=False)
        restored["assignment"] = assignment
        for name, value in restored.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FixedVariantMetadata:
    """Backend-authoritative readback for one entity's fixed variant pool.

    ``mass`` is the measured mass of each environment's assigned variant,
    normalized to a read-only float64 ``(num_envs,)`` array; the backend
    expands the per-variant measurement table with the pool's assignment.
    ``variant_files`` is the diagnostic source list of the materialized
    pool.  Backend-authoritative; assignment/scale deliberately out of
    scope (entity-scoped variant attribute flow pends an upstream
    contract).
    """

    mass: np.ndarray
    variant_files: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.variant_files, tuple):
            raise TypeError("FixedVariantMetadata.variant_files must be a tuple")
        if not all(isinstance(name, str) and name for name in self.variant_files):
            raise TypeError(
                "FixedVariantMetadata.variant_files must contain non-empty strings"
            )

        mass = np.asarray(self.mass, dtype=np.float64)
        if mass.ndim != 1 or mass.size == 0:
            raise ValueError("FixedVariantMetadata.mass must be a non-empty (num_envs,) array")
        if not np.isfinite(mass).all():
            raise ValueError("FixedVariantMetadata.mass must be finite")
        if not mass.flags.writeable:
            mass = mass.copy()
        mass.setflags(write=False)
        object.__setattr__(self, "mass", mass)


@dataclass(frozen=True)
class DomainRandomizationCapabilities:
    """Backend domain-randomization capability declaration.

    ``supported_interval_terms`` is the authoritative set-based declaration of
    interval term support.  The five ``supports_interval_*`` bools are
    deprecated (kept for backward compatibility; removed in the next major
    release): :meth:`supports_interval_term` consults them only as a fallback
    when the term is absent from ``supported_interval_terms``, so old
    constructor call sites keep their meaning.
    """

    supported_reset_terms: frozenset[str] = field(default_factory=frozenset)
    supports_interval_push: bool = False
    supports_interval_body_velocity_delta: bool = False
    supports_interval_body_angular_velocity_delta: bool = False
    supports_interval_body_force: bool = False
    supports_interval_body_torque: bool = False
    supported_interval_terms: frozenset[str] = field(default_factory=frozenset)
    supports_fixed_variants: bool = False
    supported_fixed_variant_layouts: frozenset[FixedVariantLayout] = field(
        default_factory=frozenset
    )
    supports_per_env_playback: bool = False

    _LEGACY_INTERVAL_TERM_FLAGS: ClassVar[dict[str, str]] = {
        INTERVAL_TERM_PUSH: "supports_interval_push",
        INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA: "supports_interval_body_velocity_delta",
        INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA: (
            "supports_interval_body_angular_velocity_delta"
        ),
        INTERVAL_TERM_BODY_FORCE: "supports_interval_body_force",
        INTERVAL_TERM_BODY_TORQUE: "supports_interval_body_torque",
    }

    def supports_reset_term(self, term: str) -> bool:
        return term in self.supported_reset_terms

    def supports_interval_term(self, term: str) -> bool:
        """Return whether the backend declares support for one interval term.

        Set membership in ``supported_interval_terms`` wins; otherwise the
        deprecated legacy bool mapped to ``term`` decides.
        """
        if term in self.supported_interval_terms:
            return True
        flag = self._LEGACY_INTERVAL_TERM_FLAGS.get(term)
        return bool(getattr(self, flag)) if flag is not None else False

    def get_unsupported_interval_terms(self, terms: Iterable[str]) -> frozenset[str]:
        return frozenset(term for term in terms if not self.supports_interval_term(term))

    def get_unsupported_reset_terms(self, requested_terms: frozenset[str]) -> frozenset[str]:
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

    def filter_reset_payload(
        self, payload: ResetRandomizationPayload
    ) -> tuple[ResetRandomizationPayload | None, frozenset[str]]:
        unsupported = self.get_unsupported_reset_terms(payload.requested_terms())
        if not unsupported:
            return payload, frozenset()

        filtered = ResetRandomizationPayload(
            base_mass_delta=(
                payload.base_mass_delta if self.supports_reset_term(RESET_TERM_BASE_MASS) else None
            ),
            base_com_offset=(
                payload.base_com_offset if self.supports_reset_term(RESET_TERM_BASE_COM) else None
            ),
            gravity=payload.gravity if self.supports_reset_term(RESET_TERM_GRAVITY) else None,
            body_iquat=(
                payload.body_iquat if self.supports_reset_term(RESET_TERM_BODY_IQUAT) else None
            ),
            body_inertia=(
                payload.body_inertia if self.supports_reset_term(RESET_TERM_BODY_INERTIA) else None
            ),
            body_ipos=(
                payload.body_ipos if self.supports_reset_term(RESET_TERM_BODY_IPOS) else None
            ),
            body_mass=(
                payload.body_mass if self.supports_reset_term(RESET_TERM_BODY_MASS) else None
            ),
            dof_armature=(
                payload.dof_armature if self.supports_reset_term(RESET_TERM_DOF_ARMATURE) else None
            ),
            geom_friction=(
                payload.geom_friction
                if self.supports_reset_term(RESET_TERM_GEOM_FRICTION)
                else None
            ),
            kp=payload.kp if self.supports_reset_term(RESET_TERM_KP) else None,
            kd=payload.kd if self.supports_reset_term(RESET_TERM_KD) else None,
            geom_size=payload.geom_size if self.supports_reset_term(RESET_TERM_GEOM_SIZE) else None,
            geom_solref=(
                payload.geom_solref if self.supports_reset_term(RESET_TERM_GEOM_SOLREF) else None
            ),
            geom_solimp=(
                payload.geom_solimp if self.supports_reset_term(RESET_TERM_GEOM_SOLIMP) else None
            ),
            dof_damping=(
                payload.dof_damping if self.supports_reset_term(RESET_TERM_DOF_DAMPING) else None
            ),
            dof_frictionloss=(
                payload.dof_frictionloss
                if self.supports_reset_term(RESET_TERM_DOF_FRICTIONLOSS)
                else None
            ),
        )
        return (None if filtered.is_empty() else filtered), unsupported


@dataclass
class ResetRandomizationPayload:
    base_mass_delta: np.ndarray | None = None
    base_com_offset: np.ndarray | None = None
    gravity: np.ndarray | None = None
    body_iquat: np.ndarray | None = None
    body_inertia: np.ndarray | None = None
    body_ipos: np.ndarray | None = None
    body_mass: np.ndarray | None = None
    dof_armature: np.ndarray | None = None
    geom_friction: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    # Dense model-column tables for selected reset rows. Geometry bounds are
    # derived by the adapter; callers must never supply independent bounds.
    geom_size: np.ndarray | None = None
    geom_solref: np.ndarray | None = None
    geom_solimp: np.ndarray | None = None
    dof_damping: np.ndarray | None = None
    dof_frictionloss: np.ndarray | None = None

    def requested_terms(self) -> frozenset[str]:
        terms: set[str] = set()
        if self.base_mass_delta is not None:
            terms.add(RESET_TERM_BASE_MASS)
        if self.base_com_offset is not None:
            terms.add(RESET_TERM_BASE_COM)
        if self.gravity is not None:
            terms.add(RESET_TERM_GRAVITY)
        if self.body_iquat is not None:
            terms.add(RESET_TERM_BODY_IQUAT)
        if self.body_inertia is not None:
            terms.add(RESET_TERM_BODY_INERTIA)
        if self.body_ipos is not None:
            terms.add(RESET_TERM_BODY_IPOS)
        if self.body_mass is not None:
            terms.add(RESET_TERM_BODY_MASS)
        if self.dof_armature is not None:
            terms.add(RESET_TERM_DOF_ARMATURE)
        if self.geom_friction is not None:
            terms.add(RESET_TERM_GEOM_FRICTION)
        if self.kp is not None:
            terms.add(RESET_TERM_KP)
        if self.kd is not None:
            terms.add(RESET_TERM_KD)
        for term in (
            RESET_TERM_GEOM_SIZE,
            RESET_TERM_GEOM_SOLREF,
            RESET_TERM_GEOM_SOLIMP,
            RESET_TERM_DOF_DAMPING,
            RESET_TERM_DOF_FRICTIONLOSS,
        ):
            if getattr(self, term) is not None:
                terms.add(term)
        return frozenset(terms)

    def is_empty(self) -> bool:
        return not self.requested_terms()


@dataclass
class IntervalRandomizationPlan:
    """Scheduled interval randomization request.

    The five legacy fields (``push_perturbation_limit``, ``body_ids``,
    ``body_linear_velocity_delta``, ``body_angular_velocity_delta``,
    ``body_force``, ``body_torque``) are deprecated: they are kept for
    backward compatibility and will be removed in the next major release.
    New code should populate ``ops`` with :class:`IntervalTermOp` entries.
    :meth:`iter_ops` translates each set legacy field into one op; mixing
    legacy fields and explicit ops is allowed and both are yielded.
    """

    push_perturbation_limit: Sequence[float] | np.ndarray | None = None
    body_ids: np.ndarray | None = None
    body_linear_velocity_delta: np.ndarray | None = None
    body_angular_velocity_delta: np.ndarray | None = None
    body_force: np.ndarray | None = None
    body_torque: np.ndarray | None = None
    ops: tuple[IntervalTermOp, ...] = ()

    def iter_ops(self) -> tuple[IntervalTermOp, ...]:
        """Return ops derived 1:1 from set legacy fields, then explicit ops."""
        if (
            self.push_perturbation_limit is None
            and self.body_linear_velocity_delta is None
            and self.body_angular_velocity_delta is None
            and self.body_force is None
            and self.body_torque is None
        ):
            # Hot-path fast path: ops-only plans avoid per-call re-allocation.
            return self.ops
        derived: list[IntervalTermOp] = []
        if self.push_perturbation_limit is not None:
            derived.append(
                IntervalTermOp(INTERVAL_TERM_PUSH, np.asarray(self.push_perturbation_limit))
            )
        for term, payload in (
            (INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA, self.body_linear_velocity_delta),
            (INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA, self.body_angular_velocity_delta),
            (INTERVAL_TERM_BODY_FORCE, self.body_force),
            (INTERVAL_TERM_BODY_TORQUE, self.body_torque),
        ):
            if payload is not None:
                derived.append(IntervalTermOp(term, payload, body_ids=self.body_ids))
        return (*derived, *self.ops)

    def is_empty(self) -> bool:
        return not self.ops and (
            self.push_perturbation_limit is None
            and self.body_linear_velocity_delta is None
            and self.body_angular_velocity_delta is None
            and self.body_force is None
            and self.body_torque is None
        )


@dataclass
class ResetPlan:
    env_ids: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    info_updates: dict[str, Any]
    randomization: ResetRandomizationPayload | None = None
