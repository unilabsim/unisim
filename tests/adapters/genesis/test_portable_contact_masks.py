from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unisim.backend.genesis.backend import GenesisBackend


def _owner() -> SimpleNamespace:
    return SimpleNamespace(
        geoms=(
            SimpleNamespace(name="base_geom", body_name="base"),
            SimpleNamespace(name="child_geom", body_name="child"),
        )
    )


def _native_geoms(
    masks: tuple[tuple[int, int], ...],
    frictions: tuple[tuple[float, float, float], ...] | None = None,
    solver_params: tuple[tuple[float, ...], ...] | None = None,
) -> list[SimpleNamespace]:
    assignment = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
    if frictions is None:
        frictions = ((0.4, 0.001, 0.002),) * len(masks)
    if solver_params is None:
        solver_params = ((0.02, 0.9, 0.9, 0.95, 0.001, 0.5, 2.0),) * len(masks)
    geoms: list[SimpleNamespace] = []
    for variant, ((contype, conaffinity), friction, solver_param) in enumerate(
        zip(masks, frictions, solver_params, strict=True)
    ):
        for _geom_index, body_name in enumerate(("base", "child")):
            geoms.append(
                SimpleNamespace(
                    metadata={"name": f"{body_name}_geom"},
                    link=SimpleNamespace(name=body_name),
                    active_envs_idx=np.flatnonzero(assignment == variant),
                    contype=contype,
                    conaffinity=conaffinity,
                    friction=friction[0],
                    friction_torsional=friction[1],
                    friction_rolling=friction[2],
                    sol_params=np.asarray(solver_param, dtype=np.float64),
                )
            )
    return geoms


def test_portable_collision_binding_returns_uniform_native_properties() -> None:
    masks, frictions, solver_params, mask_nonuniform, friction_nonuniform, solver_nonuniform = (
        GenesisBackend._bind_portable_collision_properties(
            SimpleNamespace(geoms=_native_geoms(((1, 2), (1, 2)))),
            _owner(),
            (object(), object()),
            5,
            np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
        )
    )
    np.testing.assert_array_equal(masks, ([1, 1], [2, 2]))
    np.testing.assert_allclose(frictions, [(0.4, 0.001, 0.002)] * 2)
    np.testing.assert_allclose(
        solver_params,
        [(0.02, 0.9, 0.9, 0.95, 0.001, 0.5, 2.0)] * 2,
    )
    assert not mask_nonuniform
    assert not friction_nonuniform
    assert not solver_nonuniform


def test_portable_collision_binding_requires_complete_native_identity() -> None:
    masks, frictions, solver_params, mask_nonuniform, friction_nonuniform, solver_nonuniform = (
        GenesisBackend._bind_portable_collision_properties(
            SimpleNamespace(geoms=[]),
            _owner(),
            (object(), object()),
            5,
            np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
        )
    )
    assert masks is None
    assert frictions is None
    assert solver_params is None
    assert not mask_nonuniform
    assert not friction_nonuniform
    assert not solver_nonuniform


def test_portable_collision_binding_rejects_nonuniform_masks() -> None:
    masks, _frictions, _solver_params, mask_nonuniform, friction_nonuniform, solver_nonuniform = (
        GenesisBackend._bind_portable_collision_properties(
            SimpleNamespace(geoms=_native_geoms(((1, 2), (3, 4)))),
            _owner(),
            (object(), object()),
            5,
            np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
        )
    )
    assert masks is None
    assert mask_nonuniform
    assert not friction_nonuniform
    assert not solver_nonuniform


def test_portable_collision_binding_rejects_nonuniform_friction() -> None:
    masks, frictions, _solver_params, mask_nonuniform, friction_nonuniform, solver_nonuniform = (
        GenesisBackend._bind_portable_collision_properties(
            SimpleNamespace(
                geoms=_native_geoms(
                    ((1, 2), (1, 2)),
                    ((0.4, 0.001, 0.002), (0.5, 0.001, 0.002)),
                )
            ),
            _owner(),
            (object(), object()),
            5,
            np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
        )
    )
    assert masks is not None
    assert frictions is None
    assert not mask_nonuniform
    assert friction_nonuniform
    assert not solver_nonuniform


def test_portable_collision_binding_rejects_nonuniform_solver_params() -> None:
    masks, _frictions, solver_params, mask_nonuniform, friction_nonuniform, solver_nonuniform = (
        GenesisBackend._bind_portable_collision_properties(
            SimpleNamespace(
                geoms=_native_geoms(
                    ((1, 2), (1, 2)),
                    solver_params=(
                        (0.02, 0.9, 0.9, 0.95, 0.001, 0.5, 2.0),
                        (0.03, 0.9, 0.9, 0.95, 0.001, 0.5, 2.0),
                    ),
                )
            ),
            _owner(),
            (object(), object()),
            5,
            np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
        )
    )
    assert masks is not None
    assert solver_params is None
    assert not mask_nonuniform
    assert not friction_nonuniform
    assert solver_nonuniform
