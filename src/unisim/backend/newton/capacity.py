"""Fail-closed capacity calibration helpers for the Newton adapter.

Newton delegates its rigid solver to MuJoCo-Warp.  That solver can silently
truncate contact constraints when ``nconmax`` or ``njmax`` is too small, so
the adapter must validate the explicit capacities on the cold path and check
the observed counts at every sampled step.  This module contains only the
backend-neutral bookkeeping; all Newton/Warp object access stays in the
adapter-owned reader callback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from unisim.errors import BackendError


class NewtonCapacityError(BackendError):
    """Raised when configured Newton solver capacity is insufficient."""


@dataclass(frozen=True, slots=True)
class NewtonCapacitySample:
    """Peak solver counts observed during one cold-path calibration window."""

    samples: int
    peak_ncon: int
    peak_nefc: int

    def __post_init__(self) -> None:
        for name in ("samples", "peak_ncon", "peak_nefc"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")

    @property
    def peak_nj(self) -> int:
        """Compatibility spelling for the MuJoCo ``nefc`` constraint count."""
        return self.peak_nefc


@dataclass(frozen=True, slots=True)
class NewtonCapacityReport:
    """Configured limits and the largest counts observed against them."""

    nconmax: int
    njmax: int
    sample: NewtonCapacitySample


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def validate_capacity_limits(
    *,
    nconmax: int,
    njmax: int,
    peak_ncon: int | None = None,
    peak_nefc: int | None = None,
    context: str = "Newton materialization",
) -> None:
    """Validate configured capacities against optional observed solver counts."""
    nconmax = _positive_int("nconmax", nconmax)
    njmax = _positive_int("njmax", njmax)
    if peak_ncon is not None:
        peak_ncon = _non_negative_int("peak_ncon", peak_ncon)
        if peak_ncon > nconmax:
            raise NewtonCapacityError(
                f"{context}: nconmax={nconmax} is below observed ncon={peak_ncon}; "
                "increase newton_nconmax instead of allowing silent constraint truncation"
            )
    if peak_nefc is not None:
        peak_nefc = _non_negative_int("peak_nefc", peak_nefc)
        if peak_nefc > njmax:
            raise NewtonCapacityError(
                f"{context}: njmax={njmax} is below observed nefc={peak_nefc}; "
                "increase newton_njmax instead of allowing silent solver truncation"
            )


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def sample_capacity(
    advance_and_read_counts: Callable[[], tuple[int, int]],
    *,
    sample_steps: int,
) -> NewtonCapacitySample:
    """Sample ``(ncon, nefc)`` after each representative cold-path step.

    The callback owns engine-specific state access and must return host integer
    counts.  It is deliberately invoked only on the materialization/calibration
    path; hot getters never call it.
    """
    sample_steps = _positive_int("sample_steps", sample_steps)
    peak_ncon = 0
    peak_nefc = 0
    for _ in range(sample_steps):
        counts = advance_and_read_counts()
        if not isinstance(counts, tuple) or len(counts) != 2:
            raise TypeError("capacity reader must return a (ncon, nefc) tuple")
        ncon = _non_negative_int("ncon", counts[0])
        nefc = _non_negative_int("nefc", counts[1])
        peak_ncon = max(peak_ncon, ncon)
        peak_nefc = max(peak_nefc, nefc)
    return NewtonCapacitySample(sample_steps, peak_ncon, peak_nefc)


def calibrate_capacity(
    advance_and_read_counts: Callable[[], tuple[int, int]],
    *,
    nconmax: int,
    njmax: int,
    sample_steps: int,
    context: str = "Newton materialization",
) -> NewtonCapacityReport:
    """Sample and validate one explicit Newton solver capacity configuration."""
    sample = sample_capacity(advance_and_read_counts, sample_steps=sample_steps)
    validate_capacity_limits(
        nconmax=nconmax,
        njmax=njmax,
        peak_ncon=sample.peak_ncon,
        peak_nefc=sample.peak_nefc,
        context=context,
    )
    return NewtonCapacityReport(nconmax=int(nconmax), njmax=int(njmax), sample=sample)


__all__ = [
    "NewtonCapacityError",
    "NewtonCapacityReport",
    "NewtonCapacitySample",
    "calibrate_capacity",
    "sample_capacity",
    "validate_capacity_limits",
]
