"""Fixed-variant realization tests for the independent MJWarp backend."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

import warp  # noqa: E402

from unisim import MjwarpBackend
from unisim.backend.mjwarp.variants import (
    VARIANT_FIELDS,
    prepare_fixed_variants,
)
from unisim.dr.types import (
    FixedVariantLayout,
    FixedVariantPlan,
    ModelSourceDescriptor,
    ResetRandomizationPayload,
)
from unisim.scene import SceneCfg


def _write_variant(
    path: Path,
    shape: str,
    *,
    extra_mesh: bool = False,
    rgba: str = "1 0 0 1",
    mass: float = 1.0,
    friction: float = 0.9,
) -> str:
    spec = mujoco.MjSpec()
    mesh = spec.add_mesh(name="primary")
    if shape == "sphere":
        mesh.make_sphere(2)
    else:
        mesh.make_cone(8, 0.1)
    material = spec.add_material(name="tool_material")
    material.rgba[:] = np.asarray([float(value) for value in rgba.split()])

    floor = spec.worldbody.add_body(name="world_floor")
    floor.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1, 1, 0.1))
    tool = spec.worldbody.add_body(name="tool")
    tool.add_freejoint()
    geom = tool.add_geom(
        name="tool_collision",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="primary",
    )
    geom.material = "tool_material"
    geom.mass = mass
    geom.friction = (friction, 0.005, 0.0001)
    if extra_mesh:
        extra = spec.add_mesh(name="secondary")
        extra.make_sphere(1)
        extra_geom = tool.add_geom(
            name="tool_collision_extra",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="secondary",
        )
        extra_geom.mass = 0.1

    spec.compile()
    spec.to_file(str(path))
    return str(path)


def _plan(paths: tuple[str, str], layout: FixedVariantLayout) -> FixedVariantPlan:
    return FixedVariantPlan(
        assignment=np.array([0, 1, 0], dtype=np.int32),
        variants=tuple(ModelSourceDescriptor(path) for path in paths),
        layout=layout,
    )


def test_same_layout_preparation_matches_independent_compile_oracle(tmp_path: Path) -> None:
    sphere = _write_variant(tmp_path / "sphere.xml", "sphere", rgba="1 0 0 1")
    cone = _write_variant(tmp_path / "cone.xml", "cone", rgba="0 0 1 1", mass=2.0)
    realization = prepare_fixed_variants(
        _plan((sphere, cone), FixedVariantLayout.SAME_LAYOUT), sim_dt=0.01
    )

    assert realization.canonical_model.nmesh == 2
    assert realization.canonical_model.nmat == 2
    assert not np.array_equal(realization.geom_dataid[0], realization.geom_dataid[1])
    assert not np.array_equal(realization.geom_matid[0], realization.geom_matid[1])
    for variant, source in enumerate((sphere, cone)):
        oracle = mujoco.MjModel.from_xml_path(source)
        for field in VARIANT_FIELDS:
            actual = realization.fields[field][variant]
            expected = np.asarray(getattr(oracle, field), dtype=np.float32)
            if field == "geom_aabb":
                expected = expected.reshape(int(oracle.ngeom), 2, 3)
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_uniform_public_layout_uses_stable_optional_mesh_slots(tmp_path: Path) -> None:
    short = _write_variant(tmp_path / "short.xml", "sphere", rgba="1 0 0 1")
    long = _write_variant(
        tmp_path / "long.xml", "cone", extra_mesh=True, rgba="0 0 1 1"
    )
    realization = prepare_fixed_variants(
        _plan((short, long), FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT), sim_dt=0.01
    )

    short_oracle = mujoco.MjModel.from_xml_path(short)
    long_oracle = mujoco.MjModel.from_xml_path(long)
    for variant, oracle in enumerate((short_oracle, long_oracle)):
        np.testing.assert_allclose(
            realization.fields["body_mass"][variant],
            oracle.body_mass,
            rtol=2e-6,
            atol=2e-6,
        )
    optional = mujoco.mj_name2id(
        realization.canonical_model, mujoco.mjtObj.mjOBJ_GEOM, "tool_collision_extra"
    )
    assert realization.geom_dataid[0, optional] == -1
    assert realization.geom_dataid[1, optional] >= 0
    assert np.all(realization.fields["geom_size"][0, optional] == 0.0)
    assert np.all(realization.fields["geom_rbound"][0, optional] == 0.0)


def test_same_layout_rejects_different_geom_counts(tmp_path: Path) -> None:
    short = _write_variant(tmp_path / "short.xml", "sphere")
    long = _write_variant(tmp_path / "long.xml", "cone", extra_mesh=True)

    with pytest.raises(ValueError, match="changes geom layout"):
        prepare_fixed_variants(
            _plan((short, long), FixedVariantLayout.SAME_LAYOUT), sim_dt=0.01
        )


def test_variant_sources_cannot_change_shared_physics_parameters(tmp_path: Path) -> None:
    sphere = _write_variant(tmp_path / "sphere.xml", "sphere")
    slippery = _write_variant(tmp_path / "slippery.xml", "sphere", friction=0.1)

    with pytest.raises(ValueError, match="changes shared field geom_friction"):
        prepare_fixed_variants(
            _plan((sphere, slippery), FixedVariantLayout.SAME_LAYOUT), sim_dt=0.01
        )


def test_texture_backed_materials_are_pooled_per_variant(tmp_path: Path) -> None:
    sources: list[str] = []
    for name, rgb1 in (("red", "1 0 0"), ("blue", "0 0 1")):
        path = tmp_path / f"{name}.xml"
        path.write_text(
            f"""
            <mujoco>
              <asset>
                <texture name="tex" type="skybox" builtin="gradient" width="16"
                         rgb1="{rgb1}" rgb2="0 0 0"/>
                <material name="tool_material" texture="tex"/>
              </asset>
              <worldbody>
                <body name="tool"><freejoint/>
                  <geom name="tool_collision" type="sphere" size="0.1"
                        material="tool_material"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        sources.append(str(path))

    realization = prepare_fixed_variants(
        _plan((sources[0], sources[1]), FixedVariantLayout.SAME_LAYOUT), sim_dt=0.01
    )
    assert realization.canonical_model.ntex == 2
    assert realization.canonical_model.nmat == 2
    assert realization.geom_matid[0, -1] != realization.geom_matid[1, -1]


def test_mjwarp_fixed_variant_backend_defaults_playback_and_graph_safe_step(
    tmp_path: Path,
) -> None:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp fixed-variant runtime tests require CUDA")

    sphere = _write_variant(tmp_path / "sphere.xml", "sphere", rgba="1 0 0 1")
    cone = _write_variant(tmp_path / "cone.xml", "cone", rgba="0 0 1 1", mass=2.0)
    plan = _plan((sphere, cone), FixedVariantLayout.SAME_LAYOUT)
    backend = MjwarpBackend(
        SceneCfg(model_file=sphere, fixed_variant_plan=plan),
        num_envs=3,
        sim_dt=0.01,
        base_name="tool",
    )

    capabilities = backend.get_dr_capabilities()
    assert capabilities.supports_fixed_variant_plan(plan)
    assert capabilities.supports_per_env_playback
    assert backend.get_playback_model(0) == sphere
    assert backend.get_playback_model(1) == cone
    assert backend.get_playback_model(2) == sphere

    mass_default = backend.get_reset_term_default("body_mass")
    size_default = backend.get_reset_term_default("geom_size")
    assert mass_default.shape == (3, backend.model.body_mass.shape[1])
    assert size_default.shape == (3, *backend._dr_geom_size.shape[1:])
    assert not mass_default.flags.writeable
    assert not size_default.flags.writeable
    assert not np.allclose(mass_default[0], mass_default[1])

    rows = np.array([0, 1, 2], dtype=np.int32)
    qpos = np.tile(backend.get_default_qpos(), (3, 1))
    qvel = np.zeros((3, backend.get_init_qvel().size), dtype=np.float32)
    backend.set_state(
        rows,
        qpos,
        qvel,
    )
    original_mass = backend.model.body_mass.numpy().copy()
    requested_mass = backend.get_reset_term_default("body_mass")[1].copy()
    requested_mass[-1] *= 1.25
    backend.step(np.zeros((3, backend.num_actuators), dtype=np.float32), nsteps=2)
    backend.set_state(
        np.array([1], dtype=np.int32),
        qpos[1:2],
        qvel[1:2],
        randomization=ResetRandomizationPayload(body_mass=requested_mass[None]),
    )
    committed_mass = backend.model.body_mass.numpy()
    np.testing.assert_allclose(committed_mass[0], original_mass[0], rtol=2e-6)
    np.testing.assert_allclose(committed_mass[1, -1], requested_mass[-1], rtol=2e-6)
    np.testing.assert_allclose(committed_mass[2], original_mass[2], rtol=2e-6)
    backend.step(np.zeros((3, backend.num_actuators), dtype=np.float32), nsteps=2)
    assert np.isfinite(backend.get_physics_state()).all()
    assert backend._cuda_graph_enabled


def test_canonical_reset_term_default_has_model_tail(tmp_path: Path) -> None:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp canonical default test requires CUDA")
    source = _write_variant(tmp_path / "canonical.xml", "sphere")
    backend = MjwarpBackend(SceneCfg(model_file=source), num_envs=2, sim_dt=0.01)

    values = backend.get_reset_term_default("body_mass")
    assert values.shape == (int(backend.model.body_mass.shape[1]),)
    assert not values.flags.writeable
