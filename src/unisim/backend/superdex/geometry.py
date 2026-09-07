"""Cold construction of audited MJCF primitive collision meshes."""

from __future__ import annotations

from typing import Any

import numpy as np


def rotation_matrix(quat: Any) -> np.ndarray:
    """Convert a canonical wxyz quaternion without importing an engine."""
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def primitive_shape(physics: Any, kind: str, size: Any, pos: Any, quat: Any) -> Any:
    """Bake a primitive into body coordinates and build its collision SDF.

    Even spheres need a surface mesh for a dynamic SuperDex link. Mesh/SDF
    sampling approximates the analytic MJCF surface; it is never done at reset.
    """
    import trimesh

    size = np.asarray(size)
    if kind == "box":
        mesh = trimesh.creation.box(extents=2 * size[:3])
    elif kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    elif kind == "cylinder":
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=2 * size[1], sections=32)
    elif kind == "capsule":
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=2 * size[1], count=[16, 32])
    elif kind == "ellipsoid":
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.vertices *= size[:3]
    else:
        raise NotImplementedError(f"superdex collision geometry {kind!r} is unsupported")
    vertices = np.asarray(mesh.vertices) @ rotation_matrix(quat).T + np.asarray(pos)
    dtype = np.float64 if physics.uses_double_precision() else np.float32
    native_mesh = physics.MeshData(
        nodes_per_element=3,
        coordinates=np.asarray(vertices, dtype=dtype).ravel(),
        connectivity=np.asarray(mesh.faces, dtype=np.int32).ravel(),
    )
    model = physics.ModelData(mesh=native_mesh)
    physics.model.bake_sdf(model)
    return physics.create_model_shape(model)
