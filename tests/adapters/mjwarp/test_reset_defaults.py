"""CPU coverage of MJWarp's immutable compiler defaults and mutable mirrors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim.backend.mjwarp.backend import MjwarpBackend
from unisim.backend.mjwarp.variants import prepare_fixed_variants
from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor

mujoco = pytest.importorskip("mujoco")


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
        assert mass_table is backend._fixed_variant_realization.fields["body_mass"]
    else:
        assert mass_table.shape == (1, model.nbody)
    assert backend._reset_field_defaults["gravity"].strides[0] == 0
