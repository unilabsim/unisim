"""Independent SuperDex CPU adapter; SDK imports occur only at construction."""

from .backend import SuperDexBackend
from .dependencies import SuperDexDependencyError

__all__ = ["SuperDexBackend", "SuperDexDependencyError"]
