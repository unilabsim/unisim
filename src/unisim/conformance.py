"""Reusable checks for adapter authors.

The helper intentionally tests only the public contract. Engine-specific
fixtures and expensive differential tests belong to adapter-owned suites.
"""

from __future__ import annotations

import numpy as np

from .contract import BackendCapability, SimBackend


def assert_backend_conformance(backend: SimBackend) -> None:
    """Run cheap shape/lifecycle checks against a materialized backend."""
    # Adapters may defer pool/worker allocation until the explicit cold-path
    # materialize hook. Conformance owns that lifecycle transition so a minimal
    # adapter test can construct, validate, and exercise the public contract.
    materialize = getattr(backend, "materialize", None)
    if callable(materialize):
        materialize()
    assert backend.num_envs > 0
    assert backend.num_actuators > 0
    assert BackendCapability.RESET in backend.capabilities
    assert BackendCapability.STATE_READ in backend.capabilities

    ctrl = np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float64)
    backend.step(ctrl)
    state = backend.get_state()
    assert state, "backend must expose at least one state field"
    for name, value in state.items():
        array = np.asarray(value)
        assert np.isfinite(array).all(), f"state field {name!r} contains non-finite values"
    if BackendCapability.SELECTED_RESET in backend.capabilities:
        backend.reset(np.arange(min(1, backend.num_envs), dtype=np.intp))
    else:
        backend.reset()
