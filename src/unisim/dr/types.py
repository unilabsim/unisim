from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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
RESET_TERM_GEOM_FRICTION = "geom_friction"
RESET_TERM_KP = "kp"
RESET_TERM_KD = "kd"


@dataclass(frozen=True)
class GeomSizeOverride:
    geom_name: str
    size: tuple[float, ...]


@dataclass(frozen=True)
class ModelVariantSpec:
    geom_size_overrides: tuple[GeomSizeOverride, ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        return not self.geom_size_overrides


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
class InitRandomizationPlan:
    model_assignments: np.ndarray
    model_variants: tuple[ModelVariantSpec, ...]

    def is_empty(self) -> bool:
        return len(self.model_variants) == 0


@dataclass
class ResetPlan:
    env_ids: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    info_updates: dict[str, Any]
    randomization: ResetRandomizationPayload | None = None
