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


def _native_geoms(masks: tuple[tuple[int, int], ...]) -> list[SimpleNamespace]:
    assignment = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
    geoms: list[SimpleNamespace] = []
    for variant, (contype, conaffinity) in enumerate(masks):
        for geom_index, body_name in enumerate(("base", "child")):
            geoms.append(
                SimpleNamespace(
                    metadata={"name": f"{body_name}_geom"},
                    link=SimpleNamespace(name=body_name),
                    active_envs_idx=np.flatnonzero(assignment == variant),
                    contype=contype,
                    conaffinity=conaffinity,
                )
            )
    return geoms


def test_portable_contact_mask_binding_requires_complete_native_identity() -> None:
    masks, nonuniform = GenesisBackend._bind_portable_contact_masks(
        SimpleNamespace(geoms=[]),
        _owner(),
        (object(), object()),
        5,
        np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
    )
    assert masks is None
    assert not nonuniform


def test_portable_contact_mask_binding_rejects_nonuniform_variants() -> None:
    masks, nonuniform = GenesisBackend._bind_portable_contact_masks(
        SimpleNamespace(geoms=_native_geoms(((1, 2), (3, 4)))),
        _owner(),
        (object(), object()),
        5,
        np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
    )
    assert masks is None
    assert nonuniform
