"""Adapter metadata shared by package tooling and future benchmark clients.

The manifest is intentionally descriptive. It does not claim runtime support;
an adapter is promoted only when its implementation and conformance child are
merged. Keeping identities here prevents benchmark and UniLab registries from
inventing divergent backend names during the staged migration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """Declared package identity and optional dependency for one engine."""

    name: str
    extra: str
    status: str


ADAPTER_SPECS: tuple[AdapterSpec, ...] = (
    AdapterSpec("mujoco", "mujoco", "available"),
    AdapterSpec("motrix", "motrix", "available"),
    # All current UniLab identities have a public UniSim adapter boundary.
    # Runtime support is resolved lazily and fails closed with an actionable
    # dependency/worker diagnostic when the optional SDK is not installed.
    AdapterSpec("drake", "drake", "available"),
    AdapterSpec("mjwarp", "mjwarp", "available"),
    AdapterSpec("newton", "newton", "available"),
    AdapterSpec("genesis", "genesis", "available"),
    AdapterSpec("isaacgym", "isaacgym", "available"),
    AdapterSpec("isaacsim", "isaacsim", "available"),
)


def adapter_spec(name: str) -> AdapterSpec:
    """Return one declared adapter identity or raise a stable ``KeyError``."""
    for spec in ADAPTER_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown UniSim adapter: {name!r}")


__all__ = ["ADAPTER_SPECS", "AdapterSpec", "adapter_spec"]
