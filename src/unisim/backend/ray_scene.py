"""Cold-path public ray-scene projections for MuJoCo-family adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..ray_query import (
    RayCollisionDescription,
    RayGeomType,
    RayMeshDescription,
    RaySceneDescription,
    RaySiteFrameDescription,
    ray_geom_type,
)


def ray_site_frames_from_mujoco_model(
    model: Any, site_names: Sequence[str]
) -> tuple[RaySiteFrameDescription, ...]:
    """Project named MuJoCo sites to public body-local frames."""

    frames = []
    for raw_name in site_names:
        name = str(raw_name)
        site = model.site(name) if name else None
        if site is None or site.id < 0:
            raise ValueError(f"Site '{name}' not found in MuJoCo model")
        site_id = int(site.id)
        body = model.body(int(model.site_bodyid[site_id]))
        if body is None or body.id < 0:
            raise ValueError(f"Site '{name}' has no owning MuJoCo body")
        frames.append(
            RaySiteFrameDescription(
                site_name=name,
                body_name=str(body.name),
                pos=np.asarray(model.site_pos[site_id], dtype=np.float64),
                quat=np.asarray(model.site_quat[site_id], dtype=np.float64),
            )
        )
    return tuple(frames)


def _selected_mujoco_axes(
    model: Any,
    body_names: Sequence[str] | None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Resolve public body aliases, native IDs, and the retained geom order."""

    native_names = [model.body(index).name for index in range(model.nbody)]
    aliases: list[str] = []
    occupied: set[str] = set()
    for index, name in enumerate(native_names):
        alias = name or f"body_{index}"
        if alias in occupied:
            raise ValueError(f"ray body identity {alias!r} is not unique")
        aliases.append(alias)
        occupied.add(alias)

    if body_names is None:
        selected_names = tuple(aliases)
        selected_ids = np.arange(model.nbody, dtype=np.intp)
    else:
        if isinstance(body_names, (str, bytes)) or not isinstance(body_names, Sequence):
            raise TypeError("ray body_names must be a sequence of strings or None")
        requested = tuple(body_names)
        if not requested:
            raise ValueError("ray body_names cannot be empty")
        if any(not isinstance(name, str) or not name for name in requested):
            raise TypeError("ray body_names must contain non-empty strings")
        if len(set(requested)) != len(requested):
            raise ValueError(f"ray body_names must be unique: {requested}")
        unknown = [name for name in requested if name not in occupied]
        if unknown:
            raise KeyError(f"ray body_names are missing from the scene: {unknown}")
        selected_names = requested
        alias_to_native = {alias: index for index, alias in enumerate(aliases)}
        selected_ids = np.asarray([alias_to_native[name] for name in selected_names], dtype=np.intp)

    native_to_selected = np.full(model.nbody, -1, dtype=np.intp)
    native_to_selected[selected_ids] = np.arange(selected_ids.size, dtype=np.intp)
    geom_bodies = np.asarray(model.geom_bodyid, dtype=np.intp)
    keep = geom_bodies >= 0
    selected_keep = native_to_selected[geom_bodies[keep]] >= 0
    geom_ids = np.flatnonzero(keep)[selected_keep]
    return selected_names, selected_ids, native_to_selected, geom_ids


def ray_scene_from_mujoco_model(
    model: Any,
    *,
    body_names: Sequence[str] | None = None,
) -> RaySceneDescription:
    """Project ray-visible primitive geometry and public body identity."""

    selected_names, _selected_ids, native_to_selected, geom_ids = _selected_mujoco_axes(
        model, body_names
    )
    return RaySceneDescription(
        num_bodies=len(selected_names),
        geom_types=tuple(ray_geom_type(int(model.geom_type[index])) for index in geom_ids),
        geom_sizes=np.asarray(model.geom_size, dtype=np.float64)[geom_ids],
        geom_local_pos=np.asarray(model.geom_pos, dtype=np.float64)[geom_ids],
        geom_local_quat=np.asarray(model.geom_quat, dtype=np.float64)[geom_ids],
        geom_body_ids=native_to_selected[np.asarray(model.geom_bodyid, dtype=np.intp)[geom_ids]],
        body_names=selected_names,
    )


def _mesh_asset_id(model: Any, kind: str, data_id: int) -> str:
    name = model.mesh(data_id).name if kind == "mesh" else model.hfield(data_id).name
    return f"mujoco:{kind}:{name if name else data_id}"


def _nondegenerate_faces(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Drop source triangles with zero area; they have no ray surface."""

    triangles = points[indices.reshape(-1, 3)]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    return indices.reshape(-1, 3)[areas > 0.0].reshape(-1)


def _triangulate_hfield(model: Any, hfield_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one MuJoCo hfield as the closed solid ``mj_ray`` intersects."""

    # The closed-solid triangulation strategy is adapted from MuJoCo-LiDAR's
    # MIT-licensed hfield mesh builder; see NOTICE for attribution.

    nrow = int(model.hfield_nrow[hfield_id])
    ncol = int(model.hfield_ncol[hfield_id])
    address = int(model.hfield_adr[hfield_id])
    heights = np.asarray(
        model.hfield_data[address : address + nrow * ncol], dtype=np.float64
    ).reshape(nrow, ncol)
    half_x, half_y, elevation_scale, base_depth = np.asarray(
        model.hfield_size[hfield_id], dtype=np.float64
    )
    x = np.linspace(-half_x, half_x, ncol, dtype=np.float32)
    y = np.linspace(-half_y, half_y, nrow, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    surface_z = (heights * elevation_scale).astype(np.float32)
    points = np.stack((grid_x, grid_y, surface_z), axis=-1).reshape(-1, 3)

    cells = np.arange((nrow - 1) * (ncol - 1), dtype=np.int32).reshape(nrow - 1, ncol - 1)
    row = cells // (ncol - 1)
    column = cells % (ncol - 1)
    lower_left = row * ncol + column
    lower_right = lower_left + 1
    upper_left = lower_left + ncol
    upper_right = upper_left + 1
    faces = np.stack(
        (
            np.stack((lower_left, lower_right, upper_right), axis=-1),
            np.stack((lower_left, upper_right, upper_left), axis=-1),
        ),
        axis=2,
    ).reshape(-1)

    additional_points: list[tuple[float, float, float]] = []
    additional_faces: list[tuple[int, int, int]] = []

    def add_vertex(x_value: float, y_value: float, z_value: float) -> int:
        additional_points.append((x_value, y_value, z_value))
        return len(points) + len(additional_points) - 1

    boundary_edges: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for column_index in range(ncol - 1):
        boundary_edges.extend(
            (
                (
                    (float(x[column_index]), -half_y, float(surface_z[0, column_index])),
                    (float(x[column_index + 1]), -half_y, float(surface_z[0, column_index + 1])),
                ),
                (
                    (float(x[column_index]), half_y, float(surface_z[-1, column_index])),
                    (float(x[column_index + 1]), half_y, float(surface_z[-1, column_index + 1])),
                ),
            )
        )
    for row_index in range(nrow - 1):
        boundary_edges.extend(
            (
                (
                    (-half_x, float(y[row_index]), float(surface_z[row_index, 0])),
                    (-half_x, float(y[row_index + 1]), float(surface_z[row_index + 1, 0])),
                ),
                (
                    (half_x, float(y[row_index]), float(surface_z[row_index, -1])),
                    (half_x, float(y[row_index + 1]), float(surface_z[row_index + 1, -1])),
                ),
            )
        )
    for first, second in boundary_edges:
        if first[2] == 0.0 and second[2] == 0.0:
            continue
        first_top = add_vertex(*first)
        second_top = add_vertex(*second)
        first_base = add_vertex(first[0], first[1], 0.0)
        second_base = add_vertex(second[0], second[1], 0.0)
        additional_faces.extend(
            ((first_base, second_base, second_top), (first_base, second_top, first_top))
        )

    if base_depth > 0.0:
        top_ring = [
            add_vertex(x_value, y_value, 0.0)
            for x_value, y_value in (
                (-half_x, -half_y),
                (half_x, -half_y),
                (half_x, half_y),
                (-half_x, half_y),
            )
        ]
        bottom_ring = [
            add_vertex(x_value, y_value, -base_depth)
            for x_value, y_value in (
                (-half_x, -half_y),
                (half_x, -half_y),
                (half_x, half_y),
                (-half_x, half_y),
            )
        ]
        top0, top1, top2, top3 = top_ring
        bottom0, bottom1, bottom2, bottom3 = bottom_ring
        additional_faces.extend(
            (
                (top0, top1, top2),
                (top0, top2, top3),
                (bottom0, bottom2, bottom1),
                (bottom0, bottom3, bottom2),
                (bottom0, bottom1, top1),
                (bottom0, top1, top0),
                (bottom1, bottom2, top2),
                (bottom1, top2, top1),
                (bottom2, bottom3, top3),
                (bottom2, top3, top2),
                (bottom3, bottom0, top0),
                (bottom3, top0, top3),
            )
        )

    if additional_points:
        points = np.concatenate(
            (points, np.asarray(additional_points, dtype=np.float32).reshape(-1, 3))
        )
        faces = np.concatenate((faces, np.asarray(additional_faces, dtype=np.int32).reshape(-1)))
    return points, _nondegenerate_faces(points, faces.astype(np.int32))


def ray_collision_from_mujoco_model(
    model: Any,
    *,
    body_names: Sequence[str] | None = None,
    body_parent_ids: np.ndarray | None = None,
) -> RayCollisionDescription:
    """Project detached collision geometry, identity, tags, and mesh topology."""

    selected_names, selected_ids, native_to_selected, geom_ids = _selected_mujoco_axes(
        model, body_names
    )
    geom_body_ids_native = np.asarray(model.geom_bodyid, dtype=np.intp)[geom_ids]
    geom_types = []
    for native_geom_id in geom_ids:
        native_type = int(model.geom_type[native_geom_id])
        geom_types.append(RayGeomType.MESH if native_type == 1 else ray_geom_type(native_type))

    scene = RaySceneDescription(
        num_bodies=len(selected_names),
        geom_types=tuple(geom_types),
        geom_sizes=np.asarray(model.geom_size, dtype=np.float64)[geom_ids],
        geom_local_pos=np.asarray(model.geom_pos, dtype=np.float64)[geom_ids],
        geom_local_quat=np.asarray(model.geom_quat, dtype=np.float64)[geom_ids],
        geom_body_ids=native_to_selected[geom_body_ids_native],
        body_names=selected_names,
    )

    geom_names = tuple(
        model.geom(native_geom_id).name or f"geom_{native_geom_id}" for native_geom_id in geom_ids
    )
    geom_tags = tuple(
        (f"mujoco:geom_group:{int(model.geom_group[native_geom_id])}",)
        for native_geom_id in geom_ids
    )

    meshes: list[RayMeshDescription] = []
    mesh_index_by_key: dict[tuple[int, int], int] = {}
    geom_mesh_ids = np.full(geom_ids.size, -1, dtype=np.intp)
    for public_geom_id, native_geom_id in enumerate(geom_ids):
        native_type = int(model.geom_type[native_geom_id])
        data_id = int(model.geom_dataid[native_geom_id])
        if native_type not in (1, 7) or data_id < 0:
            continue
        key = (native_type, data_id)
        if key not in mesh_index_by_key:
            if native_type == 1:
                points, indices = _triangulate_hfield(model, data_id)
                asset_id = _mesh_asset_id(model, "hfield", data_id)
            else:
                vertex_address = int(model.mesh_vertadr[data_id])
                vertex_count = int(model.mesh_vertnum[data_id])
                face_address = int(model.mesh_faceadr[data_id])
                face_count = int(model.mesh_facenum[data_id])
                points = np.asarray(
                    model.mesh_vert[vertex_address : vertex_address + vertex_count],
                    dtype=np.float32,
                )
                indices = np.asarray(
                    model.mesh_face[face_address * 3 : (face_address + face_count) * 3],
                    dtype=np.int32,
                ).reshape(-1)
                indices = _nondegenerate_faces(points, indices)
                asset_id = _mesh_asset_id(model, "mesh", data_id)
            mesh_index_by_key[key] = len(meshes)
            meshes.append(RayMeshDescription(asset_id=asset_id, points=points, indices=indices))
        geom_mesh_ids[public_geom_id] = mesh_index_by_key[key]

    native_parents = (
        np.asarray(body_parent_ids, dtype=np.intp)
        if body_parent_ids is not None
        else np.asarray(model.body_parentid, dtype=np.intp)
    )
    body_parent_ids = np.full(selected_ids.size, -1, dtype=np.intp)
    for public_body_id, native_body_id in enumerate(selected_ids):
        if native_body_id == 0:
            continue
        native_parent = int(native_parents[native_body_id])
        if native_parent < 0 or native_parent == native_body_id:
            continue
        public_parent = int(native_to_selected[native_parent])
        if public_parent >= 0:
            body_parent_ids[public_body_id] = public_parent

    return RayCollisionDescription(
        scene=scene,
        geom_names=geom_names,
        geom_tags=geom_tags,
        meshes=tuple(meshes),
        geom_mesh_ids=geom_mesh_ids,
        body_parent_ids=body_parent_ids,
    )


__all__ = ["ray_collision_from_mujoco_model", "ray_scene_from_mujoco_model"]
