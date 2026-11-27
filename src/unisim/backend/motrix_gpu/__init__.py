"""In-process CUDA-resident MotrixSim backend built on :mod:`motrixsim.gpu`.

This package owns its own runtime and never reads private state from the CPU
``motrix`` adapter.  It composes portable fragments through MSD and exposes a
narrow device-resident tensor profile that mirrors the MJWarp training
lifecycle.
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "MotrixGpuBackend":
        from .backend import MotrixGpuBackend

        return MotrixGpuBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MotrixGpuBackend"]
