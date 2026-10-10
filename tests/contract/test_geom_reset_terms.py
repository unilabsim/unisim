"""Engine-neutral host geom reset term negotiation and tensor fail-closed behavior."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from unisim.dr.types import (
    RESET_TERM_GEOM_ACTIVE,
    RESET_TERM_GEOM_MESH_VARIANT,
    RESET_TERM_GEOM_POS,
    RESET_TERM_GEOM_QUAT,
    RESET_TERM_GEOM_SHAPE,
    DomainRandomizationCapabilities,
    ResetRandomizationPayload,
    TensorResetRandomizationPayload,
    _validate_reset_term,
)

GEOM_TERMS = (
    RESET_TERM_GEOM_ACTIVE,
    RESET_TERM_GEOM_POS,
    RESET_TERM_GEOM_QUAT,
    RESET_TERM_GEOM_SHAPE,
    RESET_TERM_GEOM_MESH_VARIANT,
)


def _geom_values() -> dict[str, np.ndarray]:
    return {
        "geom_active": np.array([[True, False], [False, True]], dtype=np.bool_),
        "geom_pos": np.zeros((2, 2, 3), dtype=np.float32),
        "geom_quat": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (2, 2, 1)),
        "geom_shape": np.array([["box", "mesh"], ["sphere", "mesh"]]),
        "geom_mesh_variant": np.array([[0, 1], [0, 2]], dtype=np.int32),
    }


def test_geom_host_terms_are_distinct_known_reset_names() -> None:
    assert GEOM_TERMS == (
        "geom_active", "geom_pos", "geom_quat", "geom_shape", "geom_mesh_variant"
    )
    for term in GEOM_TERMS:
        _validate_reset_term(term)
    payload = ResetRandomizationPayload(**_geom_values())
    assert payload.requested_terms() == frozenset(GEOM_TERMS)
    assert not payload.is_empty()


@pytest.mark.parametrize("name", GEOM_TERMS)
def test_each_host_geom_term_filters_independently(name: str) -> None:
    values = _geom_values()
    payload = ResetRandomizationPayload(**values, geom_size=np.ones((2, 2, 3)))
    caps = DomainRandomizationCapabilities(supported_reset_terms=frozenset({name}))
    filtered, unsupported = caps.filter_reset_payload(payload)
    assert filtered is not None
    assert filtered.requested_terms() == frozenset({name})
    assert getattr(filtered, name) is values[name]
    assert all(
        getattr(filtered, other) is None for other in GEOM_TERMS if other != name
    )
    assert filtered.geom_size is None
    assert unsupported == frozenset({*GEOM_TERMS, "geom_size"} - {name})
    assert payload.requested_terms() == frozenset({*GEOM_TERMS, "geom_size"})


def test_geom_terms_fail_closed_without_capability_and_preserve_existing_terms() -> None:
    payload = ResetRandomizationPayload(**_geom_values(), geom_size=np.ones((2, 2, 3)))
    filtered, unsupported = DomainRandomizationCapabilities().filter_reset_payload(payload)
    assert filtered is None
    assert unsupported == frozenset({*GEOM_TERMS, "geom_size"})
    caps = DomainRandomizationCapabilities(supported_reset_terms=frozenset({"geom_size"}))
    filtered, unsupported = caps.filter_reset_payload(payload)
    assert filtered is not None
    assert filtered.requested_terms() == frozenset({"geom_size"})
    assert filtered.geom_size is payload.geom_size
    assert unsupported == frozenset(GEOM_TERMS)
    caps = DomainRandomizationCapabilities(supported_reset_terms=frozenset(GEOM_TERMS))
    filtered, unsupported = caps.filter_reset_payload(payload)
    assert filtered is not None
    assert filtered.requested_terms() == frozenset(GEOM_TERMS)
    assert unsupported == frozenset({"geom_size"})


def test_omitted_geom_terms_are_not_requested_or_filled_in() -> None:
    empty = ResetRandomizationPayload()
    assert empty.is_empty()
    assert empty.requested_terms() == frozenset()
    caps = DomainRandomizationCapabilities()
    assert caps.filter_reset_payload(empty) == (empty, frozenset())
    values = _geom_values()
    payload = ResetRandomizationPayload(geom_pos=values["geom_pos"])
    assert payload.requested_terms() == frozenset({RESET_TERM_GEOM_POS})
    assert payload.geom_quat is None
    assert payload.geom_shape is None
    assert payload.geom_mesh_variant is None


def test_new_geom_selection_is_host_only_not_tensor_payload() -> None:
    tensor_fields = {entry.name for entry in fields(TensorResetRandomizationPayload)}
    assert not tensor_fields.intersection(GEOM_TERMS)
    assert TensorResetRandomizationPayload().requested_terms() == frozenset()
    with pytest.raises(TypeError, match="geom_active"):
        TensorResetRandomizationPayload(geom_active=np.ones((1, 1), dtype=np.bool_))
