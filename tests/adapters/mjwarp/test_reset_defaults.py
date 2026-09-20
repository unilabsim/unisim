"""CPU coverage of MJWarp's immutable compiler defaults and mutable mirrors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim.backend.mjwarp.backend import MjwarpBackend
from unisim.backend.mjwarp.variants import FixedVariantRealization, prepare_fixed_variants
from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor

mujoco = pytest.importorskip("mujoco")


def _mesh_variant(path: Path, mass: float, *, extra_mesh: bool = False) -> str:
    obj_path = path.with_suffix(".obj")
    if not obj_path.exists():
        obj_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\n")
    optional = (
        '<geom name="optional" type="mesh" mesh="primary" mass=".2"/>'
        if extra_mesh
        else ""
    )
    path.write_text(
        '<mujoco><asset><mesh name="primary" file='
        f'"{obj_path}"/></asset><worldbody><body name="base"><freejoint/>'
        f'<geom name="shape" type="mesh" mesh="primary" mass="{mass}"/>'
        f"{optional}</body></worldbody></mujoco>"
    )
    return str(path)


@pytest.mark.parametrize("fixed_variants", [False, True])
def test_reset_defaults_survive_mutable_mirror_updates(
    tmp_path: Path, fixed_variants: bool
) -> None:
    paths = []
    for index, mass in enumerate((1, 3)):
        path = tmp_path / f"variant-{index}.xml"
        path.write_text(
            "<mujoco><worldbody><body name='base'><joint name='joint'/>"
            f"<geom name='shape' type='sphere' size='.1' mass='{mass}'/>"
            "</body></worldbody><actuator><position joint='joint' kp='7' kv='2'/>"
            "</actuator></mujoco>"
        )
        paths.append(path)
    backend = object.__new__(MjwarpBackend)
    backend._num_envs = 3
    backend._push_body_id = None
    backend._interval_root_velocity_qvel_ids = None
    backend._fixed_variant_plan = None
    backend._fixed_variant_realization = None
    if fixed_variants:
        backend._fixed_variant_plan = FixedVariantPlan(
            assignment=np.array([1, 0, 1], dtype=np.int32),
            variants=tuple(ModelSourceDescriptor(str(path)) for path in paths),
            layout=FixedVariantLayout.SAME_LAYOUT,
        )
        backend._fixed_variant_realization = prepare_fixed_variants(
            backend._fixed_variant_plan, sim_dt=0.01
        )
        model = backend._fixed_variant_realization.canonical_model
    else:
        model = mujoco.MjModel.from_xml_path(str(paths[0]))
    backend._cpu_model = model
    backend._nbody, backend._nv, backend._nu = model.nbody, model.nv, model.nu
    backend._bind_dr_host_mirrors()

    terms = backend.get_dr_capabilities().supported_reset_terms
    defaults = {term: backend.get_reset_term_default(term) for term in terms}
    expected_mass = [[0, 3], [0, 1], [0, 3]] if fixed_variants else [0, 1]
    np.testing.assert_array_equal(defaults["body_mass"], expected_mass)
    np.testing.assert_array_equal(defaults["kp"], [[7], [7], [7]] if fixed_variants else [7])
    np.testing.assert_array_equal(defaults["kd"], [[2], [2], [2]] if fixed_variants else [2])

    # Reset writes update the selected mutable rows. Neither those updates nor
    # a caller modifying its detached query result may alter compiler defaults.
    for name, value in vars(backend).items():
        if name.startswith("_dr_") and isinstance(value, np.ndarray):
            value[0] += 10
    for term, expected in defaults.items():
        actual = backend.get_reset_term_default(term)
        np.testing.assert_array_equal(actual, expected, err_msg=term)
        assert not actual.flags.writeable
        actual.setflags(write=True)
        actual[...] = 100
        np.testing.assert_array_equal(backend.get_reset_term_default(term), expected, err_msg=term)

    mass_table = backend._reset_field_defaults["body_mass"]
    if fixed_variants:
        realization = backend._fixed_variant_realization
        assert realization is not None
        assert mass_table is realization.fields["body_mass"]
    else:
        assert mass_table.shape == (1, model.nbody)
    assert backend._reset_field_defaults["gravity"].strides[0] == 0


def test_sparse_assignment_retains_selected_and_unassigned_canonical(
    tmp_path: Path,
) -> None:
    sources = (
        _mesh_variant(tmp_path / "variant-0.obj.xml", 1.0),
        _mesh_variant(tmp_path / "variant-1.obj.xml", 2.0),
        _mesh_variant(tmp_path / "variant-2.obj.xml", 3.0),
        _mesh_variant(tmp_path / "variant-3.obj.xml", 4.0, extra_mesh=True),
    )
    plan = FixedVariantPlan(
        assignment=np.array([0, 2, 0, 2], dtype=np.int32),
        variants=tuple(ModelSourceDescriptor(source) for source in sources),
        layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
    )
    realization = prepare_fixed_variants(plan, sim_dt=0.01)

    assert realization.source_indices == (0, 2, 3)
    assert realization.playback_model_files == (sources[0], sources[2], sources[3])
    assert all(values.shape[0] == 3 for values in realization.fields.values())
    np.testing.assert_array_equal(realization.executor_rows(plan.assignment), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="fixed variant source 1 is not materialized"):
        realization.executor_rows(np.array([1]))

    backend = object.__new__(MjwarpBackend)
    backend._entity_faulted = False
    backend._entity_closed = False
    backend._num_envs = 4
    backend._fixed_variant_plan = plan
    backend._fixed_variant_realization = realization
    assert backend.get_playback_model(0) == sources[0]
    assert backend.get_playback_model(1) == sources[2]


def test_unassigned_catalog_variant_still_fails_closed(tmp_path: Path) -> None:
    def source(path: Path, friction: float) -> str:
        path.write_text(
            "<mujoco><worldbody><body name='base'><joint name='joint'/>"
            f"<geom name='shape' type='sphere' size='.1' friction='{friction} .005 .0001'/>"
            "</body></worldbody></mujoco>"
        )
        return str(path)

    sources = (
        source(tmp_path / "variant-0.xml", 0.9),
        source(tmp_path / "variant-1.xml", 0.9),
        source(tmp_path / "variant-2.xml", 0.1),
    )
    plan = FixedVariantPlan(
        assignment=np.array([0, 0], dtype=np.int32),
        variants=tuple(ModelSourceDescriptor(item) for item in sources),
    )

    with pytest.raises(ValueError, match="changes shared field geom_friction"):
        prepare_fixed_variants(plan, sim_dt=0.01)


def test_fixed_variant_realization_type_documents_compact_contract() -> None:
    realization = FixedVariantRealization(
        canonical_model=None,
        source_indices=(0, 2, 3),
        fields={},
        geom_dataid=np.empty((3, 2), dtype=np.int32),
        geom_matid=np.empty((3, 2), dtype=np.int32),
        playback_model_files=("zero", "two", "three"),
    )

    np.testing.assert_array_equal(realization.executor_rows(np.array([3, 0])), [2, 0])
