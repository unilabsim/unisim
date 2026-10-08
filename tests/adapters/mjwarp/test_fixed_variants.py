"""Fixed-variant realization tests for the independent MJWarp backend."""

from __future__ import annotations

import gc
import hashlib
import weakref
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

import warp  # noqa: E402

from unisim import MjwarpBackend
from unisim.backend.mjwarp import variants as _variants
from unisim.backend.mjwarp.variants import (
    VARIANT_FIELDS,
    FixedVariantRealization,
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
    long = _write_variant(tmp_path / "long.xml", "cone", extra_mesh=True, rgba="0 0 1 1")
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


def test_uniform_public_layout_allows_optional_slots_to_change_body_simple(
    tmp_path: Path,
) -> None:
    def write_tool(path: Path, *, headed: bool) -> str:
        spec = mujoco.MjSpec()
        handle = spec.add_mesh(name="handle")
        handle.make_sphere(2)
        tool = spec.worldbody.add_body(name="tool", pos=(0, 0, 0))
        tool.add_freejoint(name="root")
        tool.add_geom(name="handle", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="handle")
        tool.explicitinertial = True
        if headed:
            head = spec.add_mesh(name="head")
            head.make_sphere(1)
            tool.add_geom(
                name="head",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="head",
                pos=(0.056, 0.0, 0.0),
            )
            tool.ipos = (0.0288, 0.0, 0.0)
            tool.mass = 0.0577
            tool.inertia = (2.7e-05, 6.9e-05, 6.3e-05)
        else:
            tool.mass = 0.0371
            tool.inertia = (4.7e-06, 4.3e-05, 4.1e-05)
        spec.compile()
        spec.to_file(str(path))
        return str(path)

    headed = write_tool(tmp_path / "headed.xml", headed=True)
    headless = write_tool(tmp_path / "headless.xml", headed=False)
    realization = prepare_fixed_variants(
        _plan((headed, headless), FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT), sim_dt=0.01
    )

    canonical = realization.canonical_model
    tool_body = mujoco.mj_name2id(canonical, mujoco.mjtObj.mjOBJ_BODY, "tool")
    headless_oracle = mujoco.MjModel.from_xml_path(headless)
    assert int(canonical.body_simple[tool_body]) == 0
    assert int(headless_oracle.body_simple[headless_oracle.body("tool").id]) == 1


def test_same_layout_rejects_different_geom_counts(tmp_path: Path) -> None:
    short = _write_variant(tmp_path / "short.xml", "sphere")
    long = _write_variant(tmp_path / "long.xml", "cone", extra_mesh=True)

    with pytest.raises(ValueError, match="changes geom layout"):
        prepare_fixed_variants(_plan((short, long), FixedVariantLayout.SAME_LAYOUT), sim_dt=0.01)


def test_same_layout_rejects_unnamed_variant_geoms_with_packaging_diagnostic(
    tmp_path: Path,
) -> None:
    named = _write_variant(tmp_path / "named.xml", "sphere")
    unnamed_path = tmp_path / "unnamed.xml"
    unnamed = _write_variant(unnamed_path, "cone")
    unnamed_path.write_text(
        unnamed_path.read_text().replace('name="tool_collision" ', "")
    )

    with pytest.raises(
        ValueError, match="fixed variant 1 geoms must have unique, non-empty names"
    ):
        prepare_fixed_variants(
            _plan((named, unnamed), FixedVariantLayout.SAME_LAYOUT), sim_dt=0.01
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
        add_body_sensors=True,
    )

    capabilities = backend.get_dr_capabilities()
    assert capabilities.fixed_variant_rejections(plan) == ()
    assert capabilities.supports_per_env_playback
    with pytest.raises(ValueError, match="explicit env_index"):
        backend.get_playback_model()
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

    for field in ("dof_invweight0", "actuator_acc0"):
        oracle = np.stack(
            [
                np.asarray(getattr(mujoco.MjModel.from_xml_path(source), field))
                for source in (sphere, cone)
            ]
        )
        device_values = np.asarray(getattr(backend.model, field).numpy())
        np.testing.assert_allclose(
            device_values,
            oracle[np.asarray(plan.assignment, dtype=np.intp)],
            rtol=2e-6,
            atol=2e-6,
        )

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


def _pool_assets_retained(
    canonical_spec: Any,
    specs: Mapping[int, Any],
    references: Mapping[int, Any],
    source_indices: tuple[int, ...],
    canonical_index: int,
) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, int]]]:
    """Pre-streaming asset pooling, kept verbatim as a bit-identity oracle."""

    mesh_pool: dict[tuple[Any, ...], str] = {}
    material_pool: dict[tuple[Any, ...], str] = {}
    texture_pool: dict[tuple[Any, ...], str] = {}
    for mesh in canonical_spec.meshes:
        mesh_pool[_variants._mesh_key(canonical_spec, mesh)] = mesh.name
    for material in canonical_spec.materials:
        material_pool[_variants._material_key(canonical_spec, material, ())] = material.name

    mesh_names_by_variant: list[dict[str, str]] = []
    material_names_by_variant: list[dict[str, str]] = []
    for source_index in source_indices:
        spec = specs[source_index]
        variant = source_index
        mesh_names: dict[str, str] = {}
        for mesh in spec.meshes:
            if variant == canonical_index:
                pooled_name = mesh.name
            else:
                key = _variants._mesh_key(spec, mesh)
                pooled_name = mesh_pool.get(key)
                if pooled_name is None:
                    pooled_name = _variants._copy_mesh(canonical_spec, spec, mesh, variant)
                    mesh_pool[key] = pooled_name
            if mesh.name in mesh_names:
                raise ValueError(f"fixed variant {variant} has duplicate mesh name {mesh.name!r}")
            mesh_names[mesh.name] = pooled_name
        mesh_names_by_variant.append(mesh_names)

        material_names: dict[str, str] = {}
        for material in spec.materials:
            if variant == canonical_index:
                pooled_name = material.name
            else:
                pooled_textures = tuple(
                    _variants._pool_texture(canonical_spec, spec, texture, texture_pool)
                    if texture
                    else ""
                    for texture in _variants._material_texture_names(material)
                )
                key = _variants._material_key(spec, material, pooled_textures)
                pooled_name = material_pool.get(key)
                if pooled_name is None:
                    pooled_name = _variants._copy_material(
                        canonical_spec,
                        spec,
                        material,
                        variant,
                        pooled_textures,
                    )
                    material_pool[key] = pooled_name
            if material.name in material_names:
                raise ValueError(
                    f"fixed variant {variant} has duplicate material name {material.name!r}"
                )
            material_names[material.name] = pooled_name
        material_names_by_variant.append(material_names)

    canonical = canonical_spec.compile()
    mesh_maps: dict[int, dict[int, int]] = {}
    material_maps: dict[int, dict[int, int]] = {}
    for source_index, reference, mesh_names, material_names in zip(
        source_indices,
        [references[source] for source in source_indices],
        mesh_names_by_variant,
        material_names_by_variant,
        strict=True,
    ):
        mesh_map = {
            int(reference.mesh(name).id): int(canonical.mesh(pooled_name).id)
            for name, pooled_name in mesh_names.items()
        }
        material_map = {
            int(reference.material(name).id): int(canonical.material(pooled_name).id)
            for name, pooled_name in material_names.items()
        }
        mesh_maps[source_index] = mesh_map
        material_maps[source_index] = material_map
    return mesh_maps, material_maps


def _prepare_fixed_variants_retained(
    plan: FixedVariantPlan,
    *,
    sim_dt: float,
    sensor_body_names: tuple[str, ...] = (),
) -> FixedVariantRealization:
    """Pre-streaming prepare_fixed_variants, kept verbatim as an oracle."""

    assigned_indices = {int(source) for source in plan.assignment}
    retained_specs: dict[int, Any] = {}
    retained_references: dict[int, Any] = {}
    canonical_index = 0
    canonical_ngeom = -1

    def compile_source(index: int, descriptor: Any) -> tuple[Any, Any]:
        path = Path(descriptor.model_file)
        if not path.is_file():
            raise ValueError(f"fixed variant {index} model source does not exist: {path}")
        try:
            spec = _variants._load_spec(path)
            _variants._inject_tracking_sensors(spec, sensor_body_names)
            reference = spec.copy().compile()
        except Exception as exc:
            raise ValueError(
                f"fixed variant {index} model source could not be loaded or compiled: {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return spec, reference

    for index, descriptor in enumerate(plan.variants):
        spec, reference = compile_source(index, descriptor)
        if int(reference.ngeom) > canonical_ngeom:
            previous_canonical = canonical_index
            canonical_index = index
            canonical_ngeom = int(reference.ngeom)
            if previous_canonical not in assigned_indices:
                retained_specs.pop(previous_canonical, None)
                retained_references.pop(previous_canonical, None)
        if index in assigned_indices or index == canonical_index:
            retained_specs[index] = spec
            retained_references[index] = reference
        else:
            del spec, reference

    selected_indices = tuple(sorted(assigned_indices | {canonical_index}))
    canonical_spec = retained_specs[canonical_index].copy()
    mesh_maps, material_maps = _pool_assets_retained(
        canonical_spec,
        retained_specs,
        retained_references,
        selected_indices,
        canonical_index,
    )
    canonical = canonical_spec.compile()
    canonical.opt.timestep = float(sim_dt)

    selected_geom_maps: dict[int, np.ndarray] = {}
    for index, descriptor in enumerate(plan.variants):
        _spec, reference = compile_source(index, descriptor)
        geom_map = _variants._validate_layout(plan.layout.value, index, reference, canonical)
        _variants._validate_shared_model_parameters(index, reference, canonical, geom_map)
        _variants._validate_shared_options(index, reference, canonical)
        if index in retained_references:
            selected_geom_maps[index] = geom_map
        del _spec, reference

    field_values = {
        name: np.broadcast_to(
            _variants._canonical_field(canonical, name),
            (len(selected_indices), *_variants._canonical_field(canonical, name).shape),
        ).copy()
        for name in VARIANT_FIELDS
    }
    dataids = np.full((len(selected_indices), int(canonical.ngeom)), -1, dtype=np.int32)
    matids = np.full((len(selected_indices), int(canonical.ngeom)), -1, dtype=np.int32)
    for row, source_index in enumerate(selected_indices):
        geom_map = selected_geom_maps[source_index]
        present = np.zeros(int(canonical.ngeom), dtype=bool)
        present[geom_map] = True
        for name in _variants._GEOM_FIELDS:
            field_values[name][row, ~present] = 0.0

    for row, source_index in enumerate(selected_indices):
        reference = retained_references[source_index]
        geom_map = selected_geom_maps[source_index]
        for name in VARIANT_FIELDS:
            values = np.asarray(getattr(reference, name), dtype=np.float32)
            if name == "geom_aabb":
                values = values.reshape(int(reference.ngeom), 2, 3)
            if name in _variants._GEOM_FIELDS:
                field_values[name][row, geom_map] = values
            elif name in _variants._BODY_FIELDS:
                if values.shape != field_values[name].shape[1:]:
                    raise ValueError(
                        f"fixed variant {source_index} changes body layout field {name}: "
                        f"{values.shape} != {field_values[name].shape[1:]}"
                    )
                field_values[name][row] = values
            else:
                raise AssertionError(name)

        source_dataids = np.asarray(reference.geom_dataid, dtype=np.int32)
        source_matids = np.asarray(reference.geom_matid, dtype=np.int32)
        for source_geom, canonical_geom in enumerate(geom_map):
            source_dataid = int(source_dataids[source_geom])
            if source_dataid >= 0:
                fallback = (
                    source_dataid
                    if _variants._same_non_mesh_asset(
                        reference, canonical, source_geom, canonical_geom
                    )
                    else -1
                )
                dataids[row, canonical_geom] = mesh_maps[source_index].get(source_dataid, fallback)
            source_matid = int(source_matids[source_geom])
            if source_matid >= 0:
                matids[row, canonical_geom] = material_maps[source_index][source_matid]

    for values in field_values.values():
        values.setflags(write=False)
    dataids.setflags(write=False)
    matids.setflags(write=False)
    return FixedVariantRealization(
        canonical_model=canonical,
        source_indices=selected_indices,
        fields=dict(field_values),
        geom_dataid=dataids,
        geom_matid=matids,
        playback_model_files=tuple(
            str(Path(plan.variants[source].model_file)) for source in selected_indices
        ),
        report_requested=tuple(
            _variants.mujoco_model_configuration(retained_references[source], mujoco)
            for source in selected_indices
        ),
    )


def _realization_digest(realization: FixedVariantRealization, path: Path) -> dict[str, Any]:
    """Hash every output of a realization, including the canonical model."""

    mujoco.mj_saveModel(realization.canonical_model, str(path))
    return {
        "canonical_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_indices": realization.source_indices,
        "fields": {
            name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for name, value in sorted(realization.fields.items())
        },
        "geom_dataid_sha256": hashlib.sha256(realization.geom_dataid.tobytes()).hexdigest(),
        "geom_matid_sha256": hashlib.sha256(realization.geom_matid.tobytes()).hexdigest(),
        "playback_model_files": realization.playback_model_files,
        "report_requested": realization.report_requested,
    }


def _texture_backed_sources(tmp_path: Path) -> tuple[str, str]:
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
    return sources[0], sources[1]


@pytest.mark.parametrize(
    ("catalog", "layout", "assignment"),
    [
        ("same_layout", FixedVariantLayout.SAME_LAYOUT, [0, 1, 0]),
        ("uniform_public", FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT, [0, 1, 0]),
        ("uniform_public_unassigned_canonical", FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT, [0]),
        ("texture_backed", FixedVariantLayout.SAME_LAYOUT, [0, 1, 1]),
    ],
)
def test_streaming_preparation_is_bit_identical_to_retained_oracle(
    tmp_path: Path,
    catalog: str,
    layout: FixedVariantLayout,
    assignment: list[int],
) -> None:
    """Streaming construction must reproduce the retained build bit-for-bit."""
    if catalog == "same_layout":
        paths = (
            _write_variant(tmp_path / "sphere.xml", "sphere", rgba="1 0 0 1"),
            _write_variant(tmp_path / "cone.xml", "cone", rgba="0 0 1 1", mass=2.0),
        )
    elif catalog.startswith("uniform_public"):
        paths = (
            _write_variant(tmp_path / "short.xml", "sphere", rgba="1 0 0 1"),
            _write_variant(tmp_path / "long.xml", "cone", extra_mesh=True, rgba="0 0 1 1"),
        )
    else:
        paths = _texture_backed_sources(tmp_path)
    plan = FixedVariantPlan(
        assignment=np.asarray(assignment, dtype=np.int32),
        variants=tuple(ModelSourceDescriptor(path) for path in paths),
        layout=layout,
    )

    expected = _realization_digest(
        _prepare_fixed_variants_retained(plan, sim_dt=0.01), tmp_path / "oracle.bin"
    )
    actual = _realization_digest(
        prepare_fixed_variants(plan, sim_dt=0.01), tmp_path / "streaming.bin"
    )
    assert actual == expected


@pytest.mark.parametrize(
    "fault",
    ["missing_source", "geom_layout", "shared_field"],
)
def test_streaming_faults_match_retained_oracle(tmp_path: Path, fault: str) -> None:
    """Fault injection: error type, message, and ordering must not change."""
    named = _write_variant(tmp_path / "named.xml", "sphere")
    if fault == "missing_source":
        other = str(tmp_path / "missing.xml")
        layout = FixedVariantLayout.SAME_LAYOUT
    elif fault == "geom_layout":
        other = _write_variant(tmp_path / "long.xml", "cone", extra_mesh=True)
        layout = FixedVariantLayout.SAME_LAYOUT
    else:
        other = _write_variant(tmp_path / "slippery.xml", "sphere", friction=0.1)
        layout = FixedVariantLayout.SAME_LAYOUT
    plan = FixedVariantPlan(
        assignment=np.asarray([0], dtype=np.int32),
        variants=(ModelSourceDescriptor(named), ModelSourceDescriptor(other)),
        layout=layout,
    )

    with pytest.raises(ValueError) as oracle_exc:
        _prepare_fixed_variants_retained(plan, sim_dt=0.01)
    with pytest.raises(ValueError) as streaming_exc:
        prepare_fixed_variants(plan, sim_dt=0.01)
    assert str(streaming_exc.value) == str(oracle_exc.value)


def test_streaming_build_releases_variant_specs_and_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Variant pool construction must not retain one spec/model per variant.

    A compiled MjSpec embeds the full mesh payload, so retaining one per
    assigned variant OOMed large catalogs on the MJWarp backend, mirroring
    unilabsim/unisim#280 on the CPU path.  The streaming build keeps only the
    canonical model plus small per-variant snapshot rows.
    """
    paths = tuple(
        _write_variant(tmp_path / f"variant_{index}.xml", "sphere", mass=1.0 + index * 0.1)
        for index in range(6)
    )
    plan = FixedVariantPlan(
        assignment=np.arange(6, dtype=np.int32),
        variants=tuple(ModelSourceDescriptor(path) for path in paths),
        layout=FixedVariantLayout.SAME_LAYOUT,
    )

    original_from_file = mujoco.MjSpec.from_file
    original_compile = mujoco.MjSpec.compile
    spec_refs: list[weakref.ref[mujoco.MjSpec]] = []
    model_refs: list[weakref.ref[mujoco.MjModel]] = []

    def tracked_from_file(*args: object, **kwargs: object) -> mujoco.MjSpec:
        spec = original_from_file(*args, **kwargs)
        spec_refs.append(weakref.ref(spec))
        return spec

    def tracked_compile(self: mujoco.MjSpec, *args: object, **kwargs: object) -> mujoco.MjModel:
        model = original_compile(self, *args, **kwargs)
        model_refs.append(weakref.ref(model))
        return model

    monkeypatch.setattr(mujoco.MjSpec, "from_file", staticmethod(tracked_from_file))
    monkeypatch.setattr(mujoco.MjSpec, "compile", tracked_compile)
    realization = prepare_fixed_variants(plan, sim_dt=0.01)
    gc.collect()
    alive_specs = sum(ref() is not None for ref in spec_refs)
    alive_models = sum(ref() is not None for ref in model_refs)
    assert alive_specs == 0
    # Only the canonical model stays; the per-variant realizations are gone.
    assert alive_models == 1
    assert realization.canonical_model.ngeom > 0
