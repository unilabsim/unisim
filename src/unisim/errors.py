"""Package-level error and capability types shared by all adapters."""

from enum import Enum


class BackendError(RuntimeError):
    """Base error raised by a UniSim backend."""


class UnsupportedCapabilityError(BackendError):
    """Raised when a requested optional backend capability is unavailable."""


class BackendCapability(str, Enum):
    """Coarse lifecycle labels used by lightweight clients and benchmarks."""

    RESET = "reset"
    SELECTED_RESET = "selected_reset"
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"
    MUTATION = "mutation"


__all__ = ["BackendCapability", "BackendError", "UnsupportedCapabilityError"]
