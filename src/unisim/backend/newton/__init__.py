"""Independent optional Newton backend and its fail-closed support helpers."""

from .capacity import (
    NewtonCapacityError,
    NewtonCapacityReport,
    NewtonCapacitySample,
    calibrate_capacity,
    sample_capacity,
    validate_capacity_limits,
)
from .dependencies import NewtonDependencyError, newton_dependencies_available


def __getattr__(name: str):
    if name == "NewtonBackend":
        from .backend import NewtonBackend

        return NewtonBackend
    if name == "NEWTON_AVAILABLE":
        return newton_dependencies_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "NewtonCapacityError",
    "NewtonCapacityReport",
    "NewtonCapacitySample",
    "NewtonBackend",
    "NewtonDependencyError",
    "NEWTON_AVAILABLE",
    "calibrate_capacity",
    "sample_capacity",
    "validate_capacity_limits",
]
