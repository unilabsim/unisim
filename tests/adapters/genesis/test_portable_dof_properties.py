from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.genesis.backend import GenesisBackend


def _native(values: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        n_dofs=values.shape[1],
        get_dofs_damping=lambda: SimpleNamespace(cpu=lambda: values),
        get_dofs_frictionloss=lambda: SimpleNamespace(cpu=lambda: values * 0.1),
    )


def _owner() -> SimpleNamespace:
    return SimpleNamespace(name="entity", qvel_indices=np.asarray([8], dtype=np.intp))


def test_portable_dof_binding_captures_variant_rows_in_public_order() -> None:
    damping = np.asarray(
        [
            [0.1, 0.4],
            [0.1, 0.4],
            [0.1, 0.4],
            [0.2, 0.5],
            [0.2, 0.5],
        ],
        dtype=np.float64,
    )
    native_damping, native_frictionloss, damping_nonuniform, friction_nonuniform = (
        GenesisBackend._bind_portable_dof_properties(
            _native(damping),
            _owner(),
            (object(), object()),
            5,
            np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
            np.asarray([1], dtype=np.intp),
        )
    )

    np.testing.assert_allclose(native_damping, [[0.4], [0.5]])
    np.testing.assert_allclose(native_frictionloss, [[0.04], [0.05]])
    assert damping_nonuniform
    assert friction_nonuniform


def test_portable_dof_binding_rejects_nonuniform_active_rows() -> None:
    damping = np.full((5, 1), 0.1, dtype=np.float64)
    damping[1, 0] = 0.2

    with pytest.raises(RuntimeError, match="non-uniform within variant 0"):
        GenesisBackend._bind_portable_dof_properties(
            _native(damping),
            _owner(),
            (object(),),
            5,
            np.zeros((5,), dtype=np.int32),
            np.asarray([0], dtype=np.intp),
        )
