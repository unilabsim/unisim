"""Newton adapter support modules.

The concrete adapter is added in Child 3.  Capacity helpers are exported here
so the implementation and its focused tests share one fail-closed contract.
"""

from .capacity import (
    NewtonCapacityError,
    NewtonCapacityReport,
    NewtonCapacitySample,
    calibrate_capacity,
    sample_capacity,
    validate_capacity_limits,
)

__all__ = [
    "NewtonCapacityError",
    "NewtonCapacityReport",
    "NewtonCapacitySample",
    "calibrate_capacity",
    "sample_capacity",
    "validate_capacity_limits",
]
