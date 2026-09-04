"""Declarative interval domain-randomization term descriptors.

An interval randomization plan is a tuple of :class:`IntervalTermOp` entries:
a term name, a NumPy payload, and optional target body ids.  The builtin
specs below pin the payload contract for the terms shared between UniLab's
domain-randomization manager and the backend adapters.  Custom terms are
free-form strings owned by a backend and validated only against that
backend's declared capability set; there is intentionally no mutable global
registry for them.

The module is stdlib + NumPy only so ops and plans stay pickle-safe
(protocol 4) across spawn-based collector processes; no engine SDK may be
imported here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTERVAL_TERM_PUSH = "push"
INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA = "body_linear_velocity_delta"
INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA = "body_angular_velocity_delta"
INTERVAL_TERM_BODY_FORCE = "body_force"
INTERVAL_TERM_BODY_TORQUE = "body_torque"

__all__ = [
    "INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA",
    "INTERVAL_TERM_BODY_FORCE",
    "INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA",
    "INTERVAL_TERM_BODY_TORQUE",
    "INTERVAL_TERM_PUSH",
    "INTERVAL_TERM_SPECS",
    "IntervalTermOp",
    "IntervalTermSpec",
    "interval_term_spec",
    "validate_interval_op",
]


@dataclass(frozen=True, slots=True)
class IntervalTermSpec:
    """Declared payload contract for one interval randomization term."""

    name: str
    requires_body_ids: bool
    payload_ndim: int
    doc: str


INTERVAL_TERM_SPECS: tuple[IntervalTermSpec, ...] = (
    IntervalTermSpec(
        name=INTERVAL_TERM_PUSH,
        requires_body_ids=False,
        payload_ndim=1,
        doc=(
            "Per-axis push force limit with shape (3,); the backend samples "
            "the actual world-frame push force per environment."
        ),
    ),
    IntervalTermSpec(
        name=INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
        requires_body_ids=True,
        payload_ndim=3,
        doc=(
            "World-frame linear velocity delta with shape "
            "(num_envs, len(body_ids), 3)."
        ),
    ),
    IntervalTermSpec(
        name=INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
        requires_body_ids=True,
        payload_ndim=3,
        doc=(
            "World-frame angular velocity delta with shape "
            "(num_envs, len(body_ids), 3)."
        ),
    ),
    IntervalTermSpec(
        name=INTERVAL_TERM_BODY_FORCE,
        requires_body_ids=True,
        payload_ndim=3,
        doc="World-frame external force with shape (num_envs, len(body_ids), 3).",
    ),
    IntervalTermSpec(
        name=INTERVAL_TERM_BODY_TORQUE,
        requires_body_ids=True,
        payload_ndim=3,
        doc="World-frame external torque with shape (num_envs, len(body_ids), 3).",
    ),
)


def interval_term_spec(name: str) -> IntervalTermSpec:
    """Return the builtin spec for ``name`` or raise a stable ``KeyError``."""
    for spec in INTERVAL_TERM_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown interval randomization term: {name!r}")


@dataclass(frozen=True)
class IntervalTermOp:
    """One interval randomization operation: term, payload, optional bodies."""

    term: str
    payload: np.ndarray
    body_ids: np.ndarray | None = None

    def validate(self) -> None:
        """Check the builtin spec contract; custom terms pass through."""
        validate_interval_op(self)


def validate_interval_op(op: IntervalTermOp) -> None:
    """Validate ``op`` against its builtin spec, ignoring custom terms.

    Builtin terms must carry ``body_ids`` exactly when their spec requires
    them, and their payload must have the declared ndim.  Unknown (custom)
    terms are backend-owned and pass through untouched.
    """
    try:
        spec = interval_term_spec(op.term)
    except KeyError:
        return
    if spec.requires_body_ids and op.body_ids is None:
        raise ValueError(f"interval term '{op.term}' requires body_ids")
    if not spec.requires_body_ids and op.body_ids is not None:
        raise ValueError(f"interval term '{op.term}' does not accept body_ids")
    payload_ndim = np.asarray(op.payload).ndim
    if payload_ndim != spec.payload_ndim:
        raise ValueError(
            f"interval term '{op.term}' payload must have ndim "
            f"{spec.payload_ndim}, got {payload_ndim}"
        )
