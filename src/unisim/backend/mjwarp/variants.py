"""Compiler-coherent fixed model variants for the MJWarp adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from unisim.inspection import mujoco_model_configuration

VARIANT_FIELDS: tuple[str, ...] = (
    "geom_size",
    "geom_rbound",
    "geom_aabb",
    "geom_pos",
    "geom_quat",
    "body_mass",
    "body_subtreemass",
    "body_inertia",
    "body_invweight0",
    "body_ipos",
    "body_iquat",
)
_GEOM_FIELDS = frozenset(VARIANT_FIELDS[:5])
_BODY_FIELDS = frozenset(VARIANT_FIELDS[5:])
_PUBLIC_LAYOUT_SCALARS = (
    "nq",
    "nv",
    "na",
    "nu",
    "nbody",
    "njnt",
    "nsite",
    "ncam",
    "nlight",
    "npair",
    "nexclude",
    "neq",
    "ntendon",
    "nwrap",
    "nsensor",
    "nsensordata",
    "nmocap",
)
_NAMED_ENTITY_COUNTS = {
    "body": "nbody",
    "joint": "njnt",
    "site": "nsite",
    "actuator": "nu",
    "sensor": "nsensor",
    "tendon": "ntendon",
}
_ALLOWED_BODY_FIELDS = frozenset(VARIANT_FIELDS[5:])
_ALLOWED_GEOM_FIELDS = frozenset(VARIANT_FIELDS[:5]) | {"geom_dataid", "geom_matid"}
# Compiler diagnostic flags can change with mesh geometry but are not runtime
# per-world Model fields in the pinned mujoco-warp contract.
_IGNORED_COMPILER_FLAGS = frozenset({"body_sameframe", "geom_sameframe"})
_IGNORED_COMPILER_METADATA_PREFIXES = ("body_geom", "body_bvh", "geom_bvh")
_SHARED_PARAMETER_PREFIXES = (
    "body_",
    "geom_",
    "jnt_",
    "dof_",
    "actuator_",
    "sensor_",
    "site_",
    "pair_",
    "eq_",
    "wrap_",
    "light_",
    "cam_",
    "tendon_",
    "qpos0",
    "qpos_spring",
)
# Compiler outputs which follow variant mass/geometry but are not part of the
# eleven runtime Model fields installed by the MJWarp realization contract.
_ALLOWED_DERIVED_FIELDS = frozenset(
    {
        "dof_M0",
        "dof_invweight0",
        "dof_length",
        "actuator_acc0",
        "tendon_length0",
        "tendon_invweight0",
    }
)


@dataclass(frozen=True)
class FixedVariantRealization:
    """Backend-local canonical model and per-variant compiler outputs."""

    canonical_model: Any
    source_indices: tuple[int, ...]
    fields: Mapping[str, np.ndarray]
    geom_dataid: np.ndarray
    geom_matid: np.ndarray
    playback_model_files: tuple[str, ...]
    report_requested: tuple[dict[str, Any], ...] = ()

    def executor_rows(self, source_indices: np.ndarray) -> np.ndarray:
        """Map immutable catalog indices to this realization's compact rows."""

        rows = {source: row for row, source in enumerate(self.source_indices)}
        try:
            return np.asarray([rows[int(source)] for source in source_indices], dtype=np.intp)
        except KeyError as exc:
            raise ValueError(f"fixed variant source {exc.args[0]} is not materialized") from exc


def prepare_fixed_variants(
    plan: Any,
    *,
    sim_dt: float,
    sensor_body_names: tuple[str, ...] = (),
) -> FixedVariantRealization:
    """Compile and merge a construction-time :class:`FixedVariantPlan`."""

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
            spec = _load_spec(path)
            _inject_tracking_sensors(spec, sensor_body_names)
            # Compile a detached copy so the authoring spec retained for asset
            # pooling below does not acquire compiler-generated texture buffers.
            reference = spec.copy().compile()
        except Exception as exc:
            raise ValueError(
                f"fixed variant {index} model source could not be loaded or compiled: {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return spec, reference

    # Pass one determines the canonical source while retaining only assignment
    # rows and the canonical source.  The latter may be unassigned but defines
    # the optional-slot union of a uniform-public catalog.
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
    mesh_maps, material_maps = _pool_assets(
        canonical_spec,
        retained_specs,
        retained_references,
        selected_indices,
        canonical_index,
    )
    canonical = canonical_spec.compile()
    canonical.opt.timestep = float(sim_dt)

    # Pass two validates the complete catalog transiently.  Only selected
    # geom mappings are retained for executor field installation.
    selected_geom_maps: dict[int, np.ndarray] = {}
    for index, descriptor in enumerate(plan.variants):
        _spec, reference = compile_source(index, descriptor)
        geom_map = _validate_layout(plan.layout.value, index, reference, canonical)
        _validate_shared_model_parameters(index, reference, canonical, geom_map)
        _validate_shared_options(index, reference, canonical)
        if index in retained_references:
            selected_geom_maps[index] = geom_map
        del _spec, reference

    field_values = {
        name: np.broadcast_to(
            _canonical_field(canonical, name),
            (len(selected_indices), *_canonical_field(canonical, name).shape),
        ).copy()
        for name in VARIANT_FIELDS
    }
    dataids = np.full((len(selected_indices), int(canonical.ngeom)), -1, dtype=np.int32)
    matids = np.full((len(selected_indices), int(canonical.ngeom)), -1, dtype=np.int32)
    for row, source_index in enumerate(selected_indices):
        geom_map = selected_geom_maps[source_index]
        present = np.zeros(int(canonical.ngeom), dtype=bool)
        present[geom_map] = True
        for name in _GEOM_FIELDS:
            field_values[name][row, ~present] = 0.0

    for row, source_index in enumerate(selected_indices):
        reference = retained_references[source_index]
        geom_map = selected_geom_maps[source_index]
        for name in VARIANT_FIELDS:
            values = np.asarray(getattr(reference, name), dtype=np.float32)
            if name == "geom_aabb":
                values = values.reshape(int(reference.ngeom), 2, 3)
            if name in _GEOM_FIELDS:
                field_values[name][row, geom_map] = values
            elif name in _BODY_FIELDS:
                if values.shape != field_values[name].shape[1:]:
                    raise ValueError(
                        f"fixed variant {source_index} changes body layout field {name}: "
                        f"{values.shape} != {field_values[name].shape[1:]}"
                    )
                field_values[name][row] = values
            else:  # pragma: no cover - the sets above partition VARIANT_FIELDS.
                raise AssertionError(name)

        source_dataids = np.asarray(reference.geom_dataid, dtype=np.int32)
        source_matids = np.asarray(reference.geom_matid, dtype=np.int32)
        for source_geom, canonical_geom in enumerate(geom_map):
            source_dataid = int(source_dataids[source_geom])
            if source_dataid >= 0:
                fallback = (
                    source_dataid
                    if _same_non_mesh_asset(reference, canonical, source_geom, canonical_geom)
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
        fields=MappingProxyType(field_values),
        geom_dataid=dataids,
        geom_matid=matids,
        playback_model_files=tuple(
            str(Path(plan.variants[source].model_file)) for source in selected_indices
        ),
        report_requested=tuple(
            mujoco_model_configuration(retained_references[source], __import__("mujoco"))
            for source in selected_indices
        ),
    )


def install_fixed_variant_fields(
    warp: Any,
    device_model: Any,
    realization: FixedVariantRealization,
    assignment: np.ndarray,
) -> None:
    """Install fixed per-world rows into already expanded Warp Model arrays."""

    rows = realization.executor_rows(np.asarray(assignment, dtype=np.int64))
    for name in VARIANT_FIELDS:
        array = getattr(device_model, name)
        leading_dim = int(array.shape[0])
        if leading_dim == 1:
            expanded = warp.array(
                shape=(rows.size, *array.shape[1:]),
                dtype=array.dtype,
                device=array.device,
            )
            setattr(device_model, name, expanded)
            array = expanded
        elif leading_dim != rows.size:
            raise RuntimeError(
                f"mjwarp fixed-variant field {name} has shape {tuple(array.shape)}; "
                f"expected leading dimension {rows.size}"
            )
        array.assign(realization.fields[name][rows])

    for name, table in (
        ("geom_dataid", realization.geom_dataid),
        ("geom_matid", realization.geom_matid),
    ):
        shared = getattr(device_model, name)
        tail = int(realization.canonical_model.ngeom)
        if tuple(shared.shape) != (1, tail):
            raise RuntimeError(
                f"mjwarp fixed-variant field {name} has shape {tuple(shared.shape)}; "
                f"expected {(1, tail)} before per-world installation"
            )
        replacement = warp.array(
            shape=(rows.size, tail),
            dtype=shared.dtype,
            device=shared.device,
        )
        replacement.assign(table[rows])
        setattr(device_model, name, replacement)


def _load_spec(path: Path) -> Any:
    import mujoco

    return mujoco.MjSpec.from_file(str(path))


def _inject_tracking_sensors(spec: Any, body_names: tuple[str, ...]) -> None:
    """Mirror the backend's canonical sensor injection in every variant spec."""

    if not body_names:
        return
    if any(not isinstance(name, str) or not name for name in body_names):
        raise ValueError("sensor_body_names must contain non-empty strings")

    import mujoco

    for body_name in body_names:
        spec.add_sensor(
            name=f"track_pos_w_{body_name}",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=body_name,
        )
    for body_name in body_names:
        spec.add_sensor(
            name=f"track_quat_w_{body_name}",
            type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=body_name,
        )
    for body_name in body_names:
        spec.add_sensor(
            name=f"track_linvel_w_{body_name}",
            type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=body_name,
        )
    for body_name in body_names:
        spec.add_sensor(
            name=f"track_angvel_w_{body_name}",
            type=mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=body_name,
        )


def _canonical_field(model: Any, name: str) -> np.ndarray:
    values = np.asarray(getattr(model, name), dtype=np.float32)
    if name == "geom_aabb":
        return values.reshape(int(model.ngeom), 2, 3)
    return values


def _pool_assets(
    canonical_spec: Any,
    specs: Mapping[int, Any],
    references: Mapping[int, Any],
    source_indices: tuple[int, ...],
    canonical_index: int,
) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, int]]]:
    """Pool all source meshes/materials and return compiled-ID mappings."""

    mesh_pool: dict[tuple[Any, ...], str] = {}
    material_pool: dict[tuple[Any, ...], str] = {}
    texture_pool: dict[tuple[Any, ...], str] = {}
    for mesh in canonical_spec.meshes:
        mesh_pool[_mesh_key(canonical_spec, mesh)] = mesh.name
    for material in canonical_spec.materials:
        material_pool[_material_key(canonical_spec, material, ())] = material.name

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
                key = _mesh_key(spec, mesh)
                pooled_name = mesh_pool.get(key)
                if pooled_name is None:
                    pooled_name = _copy_mesh(canonical_spec, spec, mesh, variant)
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
                    _pool_texture(canonical_spec, spec, texture, texture_pool) if texture else ""
                    for texture in _material_texture_names(material)
                )
                key = _material_key(spec, material, pooled_textures)
                pooled_name = material_pool.get(key)
                if pooled_name is None:
                    pooled_name = _copy_material(
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


def _mesh_key(spec: Any, mesh: Any) -> tuple[Any, ...]:
    path = _asset_path(spec, mesh.file, "meshdir") if mesh.file else ""
    vector_names = (
        "refpos",
        "refquat",
        "scale",
        "uservert",
        "usernormal",
        "usertexcoord",
        "userface",
        "userfacenormal",
        "userfacetexcoord",
    )
    vectors = tuple(_vector_key(getattr(mesh, name)) for name in vector_names)
    return (
        mesh.content_type,
        path,
        vectors,
        int(mesh.inertia),
        bool(mesh.smoothnormal),
        bool(mesh.needsdf),
        int(mesh.maxhullvert),
        int(mesh.octree_maxdepth),
        mesh.material,
    )


def _copy_mesh(target: Any, source_spec: Any, source: Any, variant: int) -> str:
    base = source.name or "mesh"
    name = f"unisim_mjwarp_v{variant}_{base}"
    while any(mesh.name == name for mesh in target.meshes):
        name = f"_{name}"
    copied = target.add_mesh(name=name)
    copied.file = "" if not source.file else _asset_path(source_spec, source.file, "meshdir")
    copied.content_type = source.content_type
    for field in ("refpos", "refquat", "scale"):
        setattr(copied, field, np.asarray(getattr(source, field)))
    for field in ("inertia", "smoothnormal", "needsdf", "maxhullvert", "octree_maxdepth"):
        setattr(copied, field, getattr(source, field))
    copied.material = source.material
    for field in (
        "uservert",
        "usernormal",
        "usertexcoord",
        "userface",
        "userfacenormal",
        "userfacetexcoord",
    ):
        setattr(copied, field, list(getattr(source, field)))
    return name


def _material_texture_names(material: Any) -> tuple[str, ...]:
    """Return MuJoCo's fixed ten material texture slots, including empties."""

    names = tuple(str(name) for name in material.textures)
    if len(names) != 10:
        raise ValueError(f"material {material.name!r} has {len(names)} texture slots; expected 10")
    return names


def _material_key(spec: Any, material: Any, textures: tuple[str, ...]) -> tuple[Any, ...]:
    del spec
    return (
        _vector_key(material.rgba),
        float(material.specular),
        float(material.shininess),
        float(material.reflectance),
        float(material.metallic),
        float(material.roughness),
        float(material.emission),
        _vector_key(material.texrepeat),
        bool(material.texuniform),
        textures,
    )


def _copy_material(
    target: Any,
    source_spec: Any,
    source: Any,
    variant: int,
    textures: tuple[str, ...],
) -> str:
    del source_spec
    base = source.name or "material"
    name = f"unisim_mjwarp_v{variant}_{base}"
    while any(material.name == name for material in target.materials):
        name = f"_{name}"
    copied = target.add_material(name=name)
    copied.rgba[:] = np.asarray(source.rgba)
    for field in (
        "specular",
        "shininess",
        "reflectance",
        "metallic",
        "roughness",
        "emission",
    ):
        setattr(copied, field, getattr(source, field))
    copied.texrepeat[:] = np.asarray(source.texrepeat)
    copied.texuniform = source.texuniform
    copied.textures = list(textures)
    return name


def _pool_texture(
    target: Any,
    source_spec: Any,
    name: str,
    pool: dict[tuple[Any, ...], str],
) -> str:
    source = source_spec.texture(name)
    key = _texture_key(source_spec, source)
    pooled = pool.get(key)
    if pooled is not None:
        return pooled
    pooled = f"unisim_mjwarp_tex_{len(pool)}_{name}"
    while any(texture.name == pooled for texture in target.textures):
        pooled = f"_{pooled}"
    copied = target.add_texture(name=pooled)
    copied.file = "" if not source.file else _asset_path(source_spec, source.file, "texturedir")
    copied.content_type = source.content_type
    for field in (
        "type",
        "builtin",
        "colorspace",
        "gridlayout",
        "width",
        "height",
        "nchannel",
        "hflip",
        "vflip",
        "random",
    ):
        setattr(copied, field, getattr(source, field))
    for field in ("gridsize", "rgb1", "rgb2", "markrgb", "data", "cubefiles"):
        setattr(copied, field, getattr(source, field))
    copied.mark = source.mark
    pool[key] = pooled
    return pooled


def _texture_key(spec: Any, texture: Any) -> tuple[Any, ...]:
    path = _asset_path(spec, texture.file, "texturedir") if texture.file else ""
    return (
        texture.content_type,
        path,
        int(texture.type),
        str(texture.builtin),
        int(texture.colorspace),
        str(texture.gridlayout),
        _vector_key(texture.gridsize),
        int(texture.width),
        int(texture.height),
        int(texture.nchannel),
        bool(texture.hflip),
        bool(texture.vflip),
        float(texture.random),
        _vector_key(texture.rgb1),
        _vector_key(texture.rgb2),
        str(texture.mark),
        _vector_key(texture.markrgb),
        _vector_key(texture.data),
        _vector_key(texture.cubefiles),
    )


def _asset_path(spec: Any, file: str, directory_attr: str) -> str:
    path = Path(file)
    if path.is_absolute():
        return str(path.resolve())
    root = Path(spec.modelfiledir or ".")
    directory = getattr(spec, directory_attr, "")
    if directory:
        directory_path = Path(directory)
        root = directory_path if directory_path.is_absolute() else root / directory_path
    return str((root / path).resolve())


def _vector_key(value: Any) -> tuple[Any, ...]:
    array = np.asarray(value)
    if array.dtype.kind in "iuf":
        return (array.dtype.str, array.tobytes())
    return tuple(array.tolist())


def _validate_layout(
    layout: str,
    source_index: int,
    reference: Any,
    canonical: Any,
) -> np.ndarray:
    if layout not in ("same_layout", "uniform_public_layout"):
        raise ValueError(f"mjwarp fixed variants do not support layout {layout!r}")

    canonical_scalars = {
        name: int(getattr(canonical, name))
        for name in _PUBLIC_LAYOUT_SCALARS
        if hasattr(canonical, name)
    }
    canonical_named = {kind: _entity_names(canonical, kind) for kind in _NAMED_ENTITY_COUNTS}
    canonical_geoms = _entity_names(canonical, "geom")
    if len(set(canonical_geoms)) != len(canonical_geoms) or "" in canonical_geoms:
        raise ValueError("canonical fixed-variant geoms must have unique, non-empty names")

    canonical_geom_ids = {name: geom_id for geom_id, name in enumerate(canonical_geoms)}
    for name, expected in canonical_scalars.items():
        actual = int(getattr(reference, name))
        if actual != expected:
            raise ValueError(
                f"fixed variant {source_index} changes public layout field {name}: "
                f"{actual} != {expected}"
            )
    for kind, expected_names in canonical_named.items():
        actual_names = _entity_names(reference, kind)
        if actual_names != expected_names:
            raise ValueError(f"fixed variant {source_index} changes {kind} names or order")

    geoms = _entity_names(reference, "geom")
    if len(set(geoms)) != len(geoms) or "" in geoms:
        raise ValueError(
            f"fixed variant {source_index} geoms must have unique, non-empty names"
        )
    if layout == "same_layout" and int(reference.ngeom) != int(canonical.ngeom):
        raise ValueError(
            f"fixed variant {source_index} changes geom layout: {reference.ngeom} != "
            f"{canonical.ngeom}"
        )
    unknown = set(geoms) - set(canonical_geoms)
    if unknown:
        raise ValueError(
            f"fixed variant {source_index} has geoms absent from the canonical layout: "
            f"{sorted(unknown)}"
        )
    geom_map = np.asarray([canonical_geom_ids[name] for name in geoms], dtype=np.int32)
    canonical_types = np.asarray(canonical.geom_type, dtype=np.int32)
    reference_types = np.asarray(reference.geom_type, dtype=np.int32)
    if np.any(reference_types != canonical_types[geom_map]):
        raise ValueError(f"fixed variant {source_index} changes a present geom type")
    missing = np.setdiff1d(np.arange(int(canonical.ngeom)), geom_map)
    if missing.size and layout == "same_layout":
        raise ValueError(f"fixed variant {source_index} omits canonical geom slots: {missing}")
    if missing.size:
        mesh_type = _mesh_geom_type(canonical)
        if np.any(canonical_types[missing] != mesh_type):
            raise ValueError(
                f"fixed variant {source_index} omits non-mesh geom slots {missing.tolist()}; "
                "uniform_public_layout only permits optional mesh slots"
            )
    return geom_map


def _validate_shared_model_parameters(
    source_index: int,
    reference: Any,
    canonical: Any,
    geom_map: np.ndarray,
) -> None:
    """Fail closed when a source changes a field the canonical model shares."""

    for name in dir(canonical):
        if not name.startswith(_SHARED_PARAMETER_PREFIXES):
            continue
        if (
            name in _IGNORED_COMPILER_FLAGS
            or name.startswith(_IGNORED_COMPILER_METADATA_PREFIXES)
            or name in _ALLOWED_BODY_FIELDS
            or name in _ALLOWED_GEOM_FIELDS
            or name in _ALLOWED_DERIVED_FIELDS
        ):
            continue
        try:
            expected = getattr(canonical, name)
            actual = getattr(reference, name)
        except AttributeError:  # pragma: no cover - pybind optional metadata.
            continue
        if not isinstance(expected, np.ndarray) or not isinstance(actual, np.ndarray):
            continue
        if expected.shape != actual.shape:
            continue
        same = (
            np.array_equal(expected[geom_map], actual)
            if name.startswith("geom_")
            else np.array_equal(expected, actual)
        )
        if not same:
            raise ValueError(f"fixed variant {source_index} changes shared field {name}")


def _validate_shared_options(source_index: int, reference: Any, canonical: Any) -> None:
    """Reject variant sources that change solver or numerical options."""

    for name in dir(canonical.opt):
        if name.startswith("_") or name == "timestep":
            continue
        try:
            expected = getattr(canonical.opt, name)
            actual = getattr(reference.opt, name)
        except AttributeError:  # pragma: no cover - option schema drift.
            continue
        if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
            if not np.array_equal(expected, actual):
                raise ValueError(f"fixed variant {source_index} changes option {name}")
        elif isinstance(expected, (bool, int, float)) and isinstance(
            actual, (bool, int, float)
        ):
            if expected != actual:
                raise ValueError(f"fixed variant {source_index} changes option {name}")


def _same_non_mesh_asset(
    reference: Any,
    canonical: Any,
    source_geom: int,
    canonical_geom: int,
) -> bool:
    mesh_type = _mesh_geom_type(canonical)
    return (
        int(reference.geom_type[source_geom]) != mesh_type
        and int(canonical.geom_type[canonical_geom]) != mesh_type
    )


def _mesh_geom_type(model: Any) -> int:
    del model
    import mujoco

    return int(mujoco.mjtGeom.mjGEOM_MESH)


def _entity_names(model: Any, kind: str) -> tuple[str, ...]:
    accessor = getattr(model, kind)
    count = getattr(model, _NAMED_ENTITY_COUNTS.get(kind, f"n{kind}"))
    return tuple(str(accessor(index).name) for index in range(int(count)))
