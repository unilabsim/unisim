"""Drake adapter for :mod:`unisim`."""
from __future__ import annotations

from .optional import RuntimeBackend


class DrakeBackend(RuntimeBackend):
    backend_type = "drake"
    module_names = ("drakeuni", "pydrake")
    install_hint = "Install `unisim-core[drake]` and the DrakeUni batch runtime."


__all__ = ["DrakeBackend"]
