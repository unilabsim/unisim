"""MuJoCo geom-to-mesh extraction shared by Viser playback viewers.

``geom_mesh`` unifies the private ``_geom_mesh`` of UniLab's ``viser_scene``
with the ``geom_mesh`` of the downstream h2r_bench playback viewer, so both
consume one implementation.  Primitive geoms are tessellated with NumPy alone
and mesh geoms return copies of the compiled model arrays.  Nothing here
imports MuJoCo, trimesh, or Viser at module import: the optional ``mujoco``
package is loaded lazily inside ``geom_mesh``.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from unisim.optional import OptionalDependencyError

__all__ = ["geom_mesh"]

_MUJOCO_MISSING = (
    "geom_mesh requires the optional 'mujoco' package, which is not installed; "
    "install it with the MuJoCo extra (pip install 'unisim-core[mujoco]')"
)

# Tessellation of the primitive geoms.  The values reproduce the trimesh
# defaults UniLab's viewers used before the helper moved here (``icosphere(2)``,
# ``capsule()``, ``cylinder()``, and ``box()``), so every generated mesh keeps
# the vertex count, face count, and extents it had before.
_ICOSPHERE_SUBDIVISIONS = 2
_CYLINDER_SECTIONS = 32
# ``trimesh.creation.capsule`` revolves a half circle sampled at 32 latitudes
# through 64 sections; the odd latitude step leaves no vertex on the equator.
_CAPSULE_LATITUDE_POINTS = 32
_CAPSULE_SECTIONS = 64

# Unit icosahedron: the twelve golden-ratio vertices (0, +-a, +-b), (+-a, +-b, 0),
# and (+-b, 0, +-a) with their twenty outward-wound faces.
_ICOSAHEDRON_VERTICES = np.array(
    [
        [-0.5257311121191336, 0.85065080835204, 0.0],
        [0.5257311121191336, 0.85065080835204, 0.0],
        [-0.5257311121191336, -0.85065080835204, 0.0],
        [0.5257311121191336, -0.85065080835204, 0.0],
        [0.0, -0.5257311121191336, 0.85065080835204],
        [0.0, 0.5257311121191336, 0.85065080835204],
        [0.0, -0.5257311121191336, -0.85065080835204],
        [0.0, 0.5257311121191336, -0.85065080835204],
        [0.85065080835204, 0.0, -0.5257311121191336],
        [0.85065080835204, 0.0, 0.5257311121191336],
        [-0.85065080835204, 0.0, -0.5257311121191336],
        [-0.85065080835204, 0.0, 0.5257311121191336],
    ],
    dtype=np.float64,
)
_ICOSAHEDRON_FACES = np.array(
    [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ],
    dtype=np.int32,
)

# Cube corners of the unit cube (0/1 per axis), x varying slowest and z
# fastest, which the twelve outward-wound faces below index into.
_BOX_CORNERS = np.array(
    [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)], dtype=np.float64
)
_BOX_FACES = np.array(
    [
        [1, 3, 0],
        [4, 1, 0],
        [0, 3, 2],
        [2, 4, 0],
        [1, 7, 3],
        [5, 1, 4],
        [5, 7, 1],
        [3, 7, 2],
        [6, 4, 2],
        [2, 7, 6],
        [6, 5, 4],
        [7, 5, 6],
    ],
    dtype=np.int32,
)


def _load_mujoco() -> Any:
    """Import the optional MuJoCo runtime, or fail closed with an install hint."""
    try:
        return importlib.import_module("mujoco")
    except ImportError as error:
        raise OptionalDependencyError(_MUJOCO_MISSING) from error


def _subdivide(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split every triangle into four by adding one midpoint per edge."""
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    midpoints = (vertices[unique[:, 0]] + vertices[unique[:, 1]]) / 2.0
    # Index of the midpoint of the (a, b), (b, c), and (c, a) edge of each face.
    edge_midpoints = np.asarray(inverse).reshape(-1, 3) + len(vertices)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    ab, bc, ca = edge_midpoints[:, 0], edge_midpoints[:, 1], edge_midpoints[:, 2]
    faces = np.concatenate(
        (
            np.column_stack((a, ab, ca)),
            np.column_stack((ab, b, bc)),
            np.column_stack((ca, bc, c)),
            np.column_stack((ab, bc, ca)),
        )
    )
    return np.vstack((vertices, midpoints)), faces


def _icosphere(radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a subdivided icosahedron of the given radius, centered at the origin.

    Every subdivision level is pushed back onto the sphere along its vertex
    direction before the next level splits its edges, which is what places the
    vertices of the final level.
    """
    vertices, faces = _ICOSAHEDRON_VERTICES, _ICOSAHEDRON_FACES
    for _ in range(_ICOSPHERE_SUBDIVISIONS):
        vertices, faces = _subdivide(vertices, faces)
        distance = np.linalg.norm(vertices, axis=1)
        # A vertex already at the origin, which only a zero radius produces, has
        # no direction and stays where it is.
        direction = vertices / np.where(distance > 0.0, distance, 1.0)[:, None]
        vertices = vertices + direction * (radius - distance)[:, None]
    return vertices, np.asarray(faces, dtype=np.int32)


def _revolve(profile: np.ndarray, sections: int) -> tuple[np.ndarray, np.ndarray]:
    """Revolve a bottom-to-top ``(radius, z)`` profile around the local z axis.

    A profile point with zero radius collapses to a single vertex on the axis,
    so the same generator builds the capped cylinder (axis points at the flat
    caps) and the capsule (axis points at the hemisphere poles).  Faces are
    wound counter-clockwise seen from outside.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, sections + 1)[:-1]
    cosines, sines = np.cos(angles), np.sin(angles)
    blocks: list[np.ndarray] = []
    counts: list[int] = []
    for radius, height in profile:
        if radius == 0.0:
            blocks.append(np.array([[0.0, 0.0, height]]))
            counts.append(1)
        else:
            ring = np.column_stack((cosines * radius, sines * radius))
            blocks.append(np.column_stack((ring, np.full(sections, height))))
            counts.append(sections)
    vertices = np.vstack(blocks)
    # Vertex index of the first point of every profile ring.
    first = np.concatenate(([0], np.cumsum(counts)[:-1]))

    section = np.arange(sections)
    following = (section + 1) % sections
    faces: list[np.ndarray] = []
    for lower in range(len(profile) - 1):
        upper = lower + 1
        lower_first, upper_first = first[lower], first[upper]
        lower_radius, upper_radius = profile[lower, 0], profile[upper, 0]
        if lower_radius == 0.0 and upper_radius == 0.0:
            continue
        if lower_radius == 0.0:
            # Fan from the axis point up to the ring above it (flat cap or pole).
            axis = np.full(sections, lower_first)
            faces.append(np.column_stack((upper_first + section, axis, upper_first + following)))
        elif upper_radius == 0.0:
            # Fan from the ring below up to the axis point (flat cap or pole).
            axis = np.full(sections, upper_first)
            faces.append(np.column_stack((lower_first + section, lower_first + following, axis)))
        else:
            faces.append(
                np.column_stack(
                    (lower_first + section, lower_first + following, upper_first + section)
                )
            )
            faces.append(
                np.column_stack(
                    (upper_first + section, lower_first + following, upper_first + following)
                )
            )
    if not faces:
        return vertices, np.empty((0, 3), dtype=np.int32)
    return vertices, np.asarray(np.concatenate(faces), dtype=np.int32)


def _box(half_extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an axis-aligned box of half extents ``half_extents``."""
    vertices = (_BOX_CORNERS - 0.5) * (2.0 * np.asarray(half_extents, dtype=np.float64))
    return vertices, _BOX_FACES.copy()


def _capsule(radius: float, half_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a capsule along local z: cylinder half-length plus hemisphere caps."""
    latitudes = np.linspace(-np.pi / 2.0, np.pi / 2.0, _CAPSULE_LATITUDE_POINTS)[1:-1]
    offset = np.where(latitudes < 0.0, -half_length, half_length)
    rings = np.column_stack((np.cos(latitudes) * radius, np.sin(latitudes) * radius + offset))
    poles = np.array([[0.0, -half_length - radius], [0.0, half_length + radius]])
    return _revolve(np.vstack((poles[:1], rings, poles[1:])), _CAPSULE_SECTIONS)


def _cylinder(radius: float, half_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a capped cylinder along local z with the given half-length."""
    profile = np.array(
        [[0.0, -half_length], [radius, -half_length], [radius, half_length], [0.0, half_length]]
    )
    return _revolve(profile, _CYLINDER_SECTIONS)


def geom_mesh(model: Any, index: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a triangle mesh for one MuJoCo geom in the geom's local frame.

    It replaces the private ``_geom_mesh`` of UniLab's ``viser_scene`` and the
    ``geom_mesh`` of the downstream h2r_bench playback viewer.  Primitive geoms
    are tessellated with NumPy only, with the same vertex count, face count,
    and extents as UniLab's previous trimesh meshes; trimesh is never imported,
    neither at module import nor at call time.  Small geoms keep every face,
    where the trimesh meshes lost the triangles below an absolute area.

    Viewers map onto it in two ways.  A batched viewer calls it for every geom
    it draws as a mesh and skips the geom on ``None`` (UniLab's ``viser_scene``
    keeps its own grid for planes), and a strict viewer such as the h2r_bench
    playback turns ``None`` into its own ``ValueError`` at the call site.

    Args:
        model: ``mujoco.MjModel`` (duck-typed): reads ``ngeom``, ``geom_type``,
            ``geom_size``, ``geom_dataid``, and for mesh geoms ``mesh_vertadr``,
            ``mesh_vertnum``, ``mesh_faceadr``, ``mesh_facenum``, ``mesh_vert``,
            and ``mesh_face``.
        index: Integer geom index in ``[0, model.ngeom)``.  Booleans are
            rejected.

    Returns:
        ``(vertices, faces)`` in the geom's local frame, or ``None``.

        * ``mjGEOM_MESH``: copies of the compiled ``model.mesh_vert`` slice
          ``(n, 3)`` and ``model.mesh_face`` slice ``(m, 3)`` with the model's
          dtypes (``float32`` and ``int32`` in MuJoCo 4.0.1), in the compiled
          mesh frame that the geom pose locates.  ``None`` when ``geom_dataid``
          is negative.
        * ``mjGEOM_SPHERE``, ``mjGEOM_CAPSULE``, ``mjGEOM_ELLIPSOID``,
          ``mjGEOM_CYLINDER``, and ``mjGEOM_BOX``: a generated mesh with
          ``float64`` vertices and ``int32`` faces wound counter-clockwise seen
          from outside.  Capsule and cylinder are aligned with local z; the
          mesh is centered on the geom frame like the analytic geom.
        * ``None`` for every other geom type (plane, hfield, SDF, none), which
          has no mesh form.  Strict callers such as h2r_bench turn ``None``
          into their own error at the call site.

    Raises:
        ValueError: If ``index`` is not an integer or is outside
            ``[0, model.ngeom)``.
        OptionalDependencyError: If the optional ``mujoco`` package is not
            installed.
    """
    mujoco = _load_mujoco()
    ngeom = model.ngeom
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise ValueError(f"geom index must be an integer, got {index!r} for ngeom={ngeom}")
    if index < 0 or index >= ngeom:
        raise ValueError(f"geom index {index} is outside [0, ngeom) for ngeom={ngeom}")

    # Compare plain integers: MuJoCo's pybind enums only compare equal to NumPy
    # integers when the integer is on the left, which tuple membership violates.
    geom_type = int(model.geom_type[index])
    kinds = mujoco.mjtGeom
    if geom_type == int(kinds.mjGEOM_MESH):
        dataid = int(model.geom_dataid[index])
        if dataid < 0:
            return None
        vert_adr = int(model.mesh_vertadr[dataid])
        vert_num = int(model.mesh_vertnum[dataid])
        face_adr = int(model.mesh_faceadr[dataid])
        face_num = int(model.mesh_facenum[dataid])
        return (
            model.mesh_vert[vert_adr : vert_adr + vert_num].copy(),
            model.mesh_face[face_adr : face_adr + face_num].copy(),
        )

    size = np.asarray(model.geom_size[index], dtype=np.float64)
    if geom_type == int(kinds.mjGEOM_SPHERE):
        return _icosphere(float(size[0]))
    if geom_type == int(kinds.mjGEOM_CAPSULE):
        return _capsule(float(size[0]), float(size[1]))
    if geom_type == int(kinds.mjGEOM_ELLIPSOID):
        vertices, faces = _icosphere(1.0)
        return vertices * size[:3], faces
    if geom_type == int(kinds.mjGEOM_CYLINDER):
        return _cylinder(float(size[0]), float(size[1]))
    if geom_type == int(kinds.mjGEOM_BOX):
        return _box(size[:3])
    return None
