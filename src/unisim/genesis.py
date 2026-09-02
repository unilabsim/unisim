"""Genesis adapter for the package-neutral UniSim contract."""
from __future__ import annotations

from .optional import RuntimeBackend


class GenesisBackend(RuntimeBackend):
    backend_type = "genesis"
    module_names = ("genesis",)
    install_hint = "Install `unisim-core[genesis]` (genesis-world==1.3.3)."


__all__ = ["GenesisBackend"]
