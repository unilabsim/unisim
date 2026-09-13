"""Backend-neutral fixed variant and reset payload metadata contract."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from unisim import FakeBackend, assert_backend_conformance
from unisim.dr.types import (
    DomainRandomizationCapabilities,
    FixedVariantLayout,
    FixedVariantPlan,
    ModelSourceDescriptor,
    ResetRandomizationPayload,
    ResetRecomputeObligation,
)


def _plan(
    assignment: np.ndarray | None = None,
    *,
    layout: FixedVariantLayout = FixedVariantLayout.SAME_LAYOUT,
) -> FixedVariantPlan:
    return FixedVariantPlan(
        assignment=np.array([0, 1, 0], dtype=np.int32) if assignment is None else assignment,
        variants=(
            ModelSourceDescriptor("/cache/tools/tool-0.xml"),
            ModelSourceDescriptor("/cache/tools/tool-1.xml"),
        ),
        layout=layout,
    )


def test_fixed_variant_plan_is_validated_and_pickle_safe() -> None:
    original = _plan()
    assert original.layout is FixedVariantLayout.SAME_LAYOUT
    assert original.assignment.shape == (3,)
    assert not original.assignment.flags.writeable

    restored = pickle.loads(pickle.dumps(original))
    assert restored == original
    assert not restored.assignment.flags.writeable
    with np.testing.assert_raises(ValueError):
        restored.assignment[0] = 1


def test_fixed_variant_plan_rejects_mutable_and_out_of_range_inputs() -> None:
    with pytest.raises(TypeError, match="variants must be a tuple"):
        FixedVariantPlan(
            assignment=np.array([0]),
            variants=[ModelSourceDescriptor("tool.xml")],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="assignment must contain integers"):
        _plan(np.array([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="values must be in"):
        _plan(np.array([0, 2, 0], dtype=np.int32))
    with pytest.raises(ValueError, match="non-empty"):
        _plan(np.array([], dtype=np.int32))
    with pytest.raises(ValueError, match="shape \\(2,\\)"):
        _plan().validate(2)


def test_model_source_descriptor_only_accepts_materialized_string_sources() -> None:
    with pytest.raises(TypeError, match="non-empty string"):
        ModelSourceDescriptor("")
    with pytest.raises(ValueError, match="source_format"):
        ModelSourceDescriptor("tool.urdf", "urdf")  # type: ignore[arg-type]


def test_reset_payload_reports_recompute_metadata() -> None:
    payload = ResetRandomizationPayload(
        body_mass=np.zeros((1, 1)),
        geom_size=np.zeros((1, 1, 3)),
        geom_friction=np.zeros((1, 1, 3)),
    )
    contracts = payload.term_contracts()
    assert tuple(contract.term for contract in contracts) == (
        "body_mass",
        "geom_friction",
        "geom_size",
    )
    assert payload.required_recompute_obligations() == frozenset(
        {ResetRecomputeObligation.MODEL_CONSTANTS, ResetRecomputeObligation.GEOMETRY}
    )


def test_capabilities_negotiate_fixed_variant_layout_and_sources() -> None:
    capabilities = DomainRandomizationCapabilities(
        supports_fixed_variants=True,
        supported_fixed_variant_layouts=frozenset({FixedVariantLayout.SAME_LAYOUT}),
        supported_fixed_variant_source_formats=frozenset({"mjcf"}),
    )
    assert capabilities.supports_fixed_variant_plan(_plan())
    assert capabilities.fixed_variant_rejections(_plan()) == ()

    uniform = _plan(layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT)
    assert not capabilities.supports_fixed_variant_plan(uniform)
    assert capabilities.fixed_variant_rejections(uniform) == (
        "fixed variant layout 'uniform_public_layout' is unsupported",
    )

    unsupported = DomainRandomizationCapabilities()
    assert unsupported.fixed_variant_rejections(_plan()) == (
        "fixed variants are unsupported",
        "fixed variant layout 'same_layout' is unsupported",
        "fixed variant source format(s) are unsupported: mjcf",
    )


def test_reset_recompute_obligations_are_not_advertised_implicitly() -> None:
    capabilities = DomainRandomizationCapabilities(supported_reset_terms=frozenset({"geom_size"}))
    requested = frozenset({"body_mass", "geom_size"})
    assert capabilities.get_unsupported_reset_terms(requested) == frozenset({"body_mass"})
    assert capabilities.get_unsupported_reset_recompute_obligations(requested) == frozenset(
        {ResetRecomputeObligation.MODEL_CONSTANTS}
    )


def test_fake_backend_fixed_variant_lifecycle_and_conformance() -> None:
    plan = _plan()
    unsupported_backend = FakeBackend(num_envs=3, num_actuators=1)
    with pytest.raises(NotImplementedError, match="fixed variants are unsupported"):
        unsupported_backend.apply_fixed_variant_plan(plan)

    backend = FakeBackend(
        num_envs=3,
        num_actuators=1,
        fixed_variants=True,
        per_env_playback=True,
    )
    assert backend.get_dr_capabilities().supports_fixed_variant_plan(plan)
    backend.apply_fixed_variant_plan(plan)
    backend.materialize()
    with pytest.raises(RuntimeError, match="after.*materialize"):
        backend.apply_fixed_variant_plan(plan)
    assert backend.get_playback_model(1) is plan.variants[1]

    assert_backend_conformance(
        FakeBackend(
            num_envs=3,
            num_actuators=1,
            fixed_variants=True,
            per_env_playback=True,
        ),
        fixed_variant_plan=plan,
    )
