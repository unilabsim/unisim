"""Cold-path Drake native property expectations and audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from unisim.scene_layout import CompiledSceneLayout


@dataclass(frozen=True)
class _ExpectedBodyProperties:
    name: str
    mass: float
    com: tuple[float, float, float]
    inertia: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class _ExpectedGeometryProperties:
    name: str
    body_name: str
    body_index: int
    geometry_type: str
    collision: bool
    parameters: tuple[float, float, float]


@dataclass(frozen=True)
class ExpectedNativeProperties:
    """Portable source values used only to audit Drake readback."""

    body_names: tuple[str, ...]
    body_masses: np.ndarray
    body_coms: np.ndarray
    body_inertias: np.ndarray
    geometry_names: tuple[str, ...]
    geometry_body_indices: np.ndarray
    geometry_types: tuple[str, ...]
    geometry_collision: tuple[bool, ...]
    geometry_parameters: np.ndarray


def scan_expected_native_properties(
    model_file: str, layout: CompiledSceneLayout
) -> ExpectedNativeProperties:
    """Scan one compiled portable source without loading DrakeUni."""

    import mujoco

    model = mujoco.MjModel.from_xml_path(model_file)
    global_body_names = ("world",) + tuple(
        f"{entity.name}/{body_name}"
        for entity in layout.entities
        for body_name in entity.body_names
    )
    if int(model.nbody) != len(global_body_names):
        raise ValueError(
            f"portable source {model_file!r} has {int(model.nbody)} bodies; "
            f"layout requires {len(global_body_names)}"
        )
    body_index_by_name = {
        str(model.body(index).name): index for index in range(int(model.nbody))
    }
    if tuple(body_index_by_name) != global_body_names:
        raise ValueError(f"portable source {model_file!r} body order differs from its layout")

    masses = np.zeros((len(global_body_names),), dtype=np.float64)
    coms = np.zeros((len(global_body_names), 3), dtype=np.float64)
    inertias = np.zeros((len(global_body_names), 3, 3), dtype=np.float64)
    for index, name in enumerate(global_body_names):
        model_body = int(body_index_by_name[name])
        mass = float(model.body_mass[model_body])
        com = np.asarray(model.body_ipos[model_body], dtype=np.float64).copy()
        orientation = np.asarray(model.body_iquat[model_body], dtype=np.float64)
        rotation = _quat_to_rotation_matrix(orientation)
        central_inertia = rotation @ np.diag(
            np.asarray(model.body_inertia[model_body], dtype=np.float64)
        ) @ rotation.T
        masses[index] = mass
        coms[index] = com
        inertias[index] = central_inertia + mass * (
            float(com @ com) * np.eye(3) - np.outer(com, com)
        )

    supported_types = {
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
        int(mujoco.mjtGeom.mjGEOM_PLANE): "half_space",
    }
    records: list[_ExpectedGeometryProperties] = []
    body_names_by_id = {
        int(body_index_by_name[name]): name for name in global_body_names
    }
    for geom_id in range(int(model.ngeom)):
        name = str(model.geom(geom_id).name)
        if not name:
            raise ValueError(f"portable source {model_file!r} contains an unnamed geometry")
        collision = (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
        if not collision:
            # DrakeUni's MJCF materializer intentionally drops visual-only
            # geometry before native model construction.
            continue
        model_type = int(model.geom_type[geom_id])
        if model_type not in supported_types:
            raise NotImplementedError(
                f"Drake native property audit does not support geometry {name!r} type "
                f"{model_type}"
            )
        geometry_type = supported_types[model_type]
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        parameters = _native_primitive_parameters(geometry_type, size)
        body_index = int(model.geom_bodyid[geom_id])
        records.append(
            _ExpectedGeometryProperties(
                name,
                body_names_by_id[body_index],
                body_index,
                geometry_type,
                True,
                parameters,
            )
        )
    records.sort(key=lambda record: (record.body_index, record.name))

    return ExpectedNativeProperties(
        body_names=global_body_names,
        body_masses=masses,
        body_coms=coms,
        body_inertias=inertias,
        geometry_names=tuple(record.name for record in records),
        geometry_body_indices=np.asarray(
            [record.body_index for record in records], dtype=np.int32
        ),
        geometry_types=tuple(record.geometry_type for record in records),
        geometry_collision=tuple(record.collision for record in records),
        geometry_parameters=np.asarray(
            [record.parameters for record in records], dtype=np.float64
        ).reshape((len(records), 3)),
    )


def validate_native_model_properties(
    expected: ExpectedNativeProperties, actual: Any, variant: int
) -> None:
    """Require Drake-native values to match the compiled portable source."""

    prefix = f"Drake native properties for variant {variant}"
    if int(getattr(actual, "contract_version", -1)) != 1:
        raise ValueError(f"{prefix} uses an unsupported readback contract version")
    if tuple(actual.body_names) != expected.body_names:
        raise ValueError(
            f"{prefix} body order differs: expected {expected.body_names}, "
            f"got {tuple(actual.body_names)}"
        )
    for field, actual_field, expected_values in (
        ("masses", "body_masses", expected.body_masses),
        ("COMs", "body_coms", expected.body_coms),
        ("inertias", "body_inertias", expected.body_inertias),
    ):
        actual_values = getattr(actual, actual_field)
        if not np.allclose(actual_values, expected_values, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(f"{prefix} body {field} differ from the portable source")
    if tuple(actual.geometry_names) != expected.geometry_names:
        raise ValueError(
            f"{prefix} geometry order differs: expected {expected.geometry_names}, "
            f"got {tuple(actual.geometry_names)}"
        )
    if tuple(actual.geometry_types) != expected.geometry_types:
        raise ValueError(
            f"{prefix} geometry types differ from the portable source"
        )
    if tuple(actual.geometry_collision) != expected.geometry_collision:
        raise ValueError(f"{prefix} geometry collision roles differ from the portable source")
    if not np.array_equal(
        np.asarray(actual.geometry_body_indices, dtype=np.int32),
        expected.geometry_body_indices,
    ):
        raise ValueError(f"{prefix} geometry body ownership differs from the portable source")
    if not np.allclose(
        np.asarray(actual.geometry_parameters, dtype=np.float64),
        expected.geometry_parameters,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(f"{prefix} geometry parameters differ from the portable source")


def _native_primitive_parameters(
    geometry_type: str, size: np.ndarray
) -> tuple[float, float, float]:
    if geometry_type == "sphere":
        return (float(size[0]), 0.0, 0.0)
    if geometry_type == "box":
        return (float(size[0] * 2.0), float(size[1] * 2.0), float(size[2] * 2.0))
    if geometry_type in {"capsule", "cylinder"}:
        return (float(size[0]), float(size[1] * 2.0), 0.0)
    if geometry_type == "ellipsoid":
        return (float(size[0]), float(size[1]), float(size[2]))
    return (0.0, 0.0, 0.0)


def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("portable source contains an invalid zero quaternion")
    w, x, y, z = values / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


__all__ = [
    "ExpectedNativeProperties",
    "scan_expected_native_properties",
    "validate_native_model_properties",
]
