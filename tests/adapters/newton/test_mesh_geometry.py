"""Collision mesh geometry and Newton-owned convex hulls on the real runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unisim.backend.newton.backend import NewtonBackend
from unisim.backend.newton.dependencies import (
    NewtonDependencies,
    NewtonDependencyError,
    load_newton_dependencies,
)
from unisim.backend.newton.materialization import build_source_builder
from unisim.scene import SceneCfg

_WEDGE_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
    ],
    dtype=np.float32,
)
_WEDGE_FACES = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=np.int32)
_WEDGE_VERTEX_TEXT = "  ".join(
    " ".join(str(float(value)) for value in row) for row in _WEDGE_VERTICES
)
_WEDGE_FACE_TEXT = "0 2 1  0 1 3  0 3 2  1 2 3"
_GEOM_POS = np.array([0.01, 0.02, 0.03], dtype=np.float32)
# Authored MJCF quat is wxyz (0.7071068, 0.7071068, 0, 0); Newton transforms
# carry xyzw.
_GEOM_QUAT_XYZW = np.array([0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32)

_MESH_MODEL = f"""
<mujoco model="newton-mesh-test">
  <asset>
    <mesh name="wedge" vertex="{_WEDGE_VERTEX_TEXT}" face="{_WEDGE_FACE_TEXT}"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="mesh_body" pos="0 0 0.5">
      <joint name="root" type="free"/>
      <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
      <geom name="wedge_geom" type="mesh" mesh="wedge" pos="0.01 0.02 0.03"
            quat="0.7071068 0.7071068 0 0" density="0"/>
    </body>
  </worldbody>
  <sensor>
    <contact name="mesh_floor" geom1="wedge_geom" geom2="floor" data="found" num="1"/>
  </sensor>
</mujoco>
"""


def _dense_sphere_mesh(rings: int = 8, sectors: int = 12) -> tuple[str, str]:
    vertices = [(0.0, 0.0, 0.1)]
    for ring in range(1, rings):
        phi = np.pi * ring / rings
        for sector in range(sectors):
            theta = 2.0 * np.pi * sector / sectors
            vertices.append(
                (
                    0.1 * np.sin(phi) * np.cos(theta),
                    0.1 * np.sin(phi) * np.sin(theta),
                    0.1 * np.cos(phi),
                )
            )
    vertices.append((0.0, 0.0, -0.1))
    bottom = len(vertices) - 1
    faces: list[tuple[int, int, int]] = []
    for sector in range(sectors):
        faces.append((0, 1 + sector % sectors, 1 + (sector + 1) % sectors))
    for ring in range(rings - 2):
        upper = 1 + ring * sectors
        lower = upper + sectors
        for sector in range(sectors):
            a, b = upper + sector % sectors, upper + (sector + 1) % sectors
            c, d = lower + sector % sectors, lower + (sector + 1) % sectors
            faces.extend(((a, c, d), (a, d, b)))
    last = 1 + (rings - 2) * sectors
    for sector in range(sectors):
        faces.append((last + sector % sectors, bottom, last + (sector + 1) % sectors))
    vertex_text = "  ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    face_text = "  ".join(f"{a} {b} {c}" for a, b, c in faces)
    return vertex_text, face_text


def _newton_deps() -> NewtonDependencies:
    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    return deps


def _write_model(
    tmp_path: Path,
    *,
    vertices: str = _WEDGE_VERTEX_TEXT,
    faces: str = _WEDGE_FACE_TEXT,
    asset_attributes: str = "",
    geom_attributes: str = "",
    body_position: str = "0 0 0.5",
    contact_sensor: bool = True,
) -> str:
    model_file = tmp_path / "mesh_scene.xml"
    xml = (
        _MESH_MODEL.replace(_WEDGE_VERTEX_TEXT, vertices)
        .replace(_WEDGE_FACE_TEXT, faces)
        .replace('name="wedge"', f'name="wedge" {asset_attributes}')
        .replace('density="0"', f'density="0" {geom_attributes}')
        .replace('pos="0 0 0.5"', f'pos="{body_position}"')
    )
    if not contact_sensor:
        xml = xml.replace(
            '  <sensor>\n    <contact name="mesh_floor" geom1="wedge_geom" geom2="floor" '
            'data="found" num="1"/>\n  </sensor>\n',
            "",
        )
    model_file.write_text(xml, encoding="utf-8")
    return str(model_file)


def _mesh_shape_row(shape_types: np.ndarray, mesh_value: int) -> int:
    rows = np.flatnonzero(shape_types == mesh_value)
    assert rows.size == 1
    return int(rows[0])


def test_convex_mesh_vertices_pass_through_newton_import(tmp_path: Path) -> None:
    deps = _newton_deps()
    builder = build_source_builder(deps.newton, _write_model(tmp_path))
    mesh_value = int(deps.newton.GeoType.MESH)
    shape_types = np.asarray([int(value) for value in builder.shape_type], dtype=np.int64)
    assert shape_types.tolist() == [mesh_value, int(deps.newton.GeoType.PLANE)]
    row = _mesh_shape_row(shape_types, mesh_value)
    mesh = builder.shape_source[row]
    np.testing.assert_allclose(np.asarray(mesh.vertices), _WEDGE_VERTICES, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(mesh.indices), _WEDGE_FACES)
    transform = np.asarray(builder.shape_transform[row], dtype=np.float32).reshape(-1)
    np.testing.assert_allclose(transform[:3], _GEOM_POS, rtol=0, atol=1e-7)
    np.testing.assert_allclose(transform[3:7], _GEOM_QUAT_XYZW, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(builder.shape_scale[row], dtype=np.float32), np.ones(3), rtol=0, atol=1e-7
    )


def _cuda_backend(
    tmp_path: Path,
    *,
    vertices: str = _WEDGE_VERTEX_TEXT,
    faces: str = _WEDGE_FACE_TEXT,
    asset_attributes: str = "",
    geom_attributes: str = "",
    body_position: str = "0 0 0.5",
    contact_sensor: bool = True,
) -> tuple[NewtonBackend, Any]:
    deps = _newton_deps()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton materialization requires a CUDA Warp device")
    backend = NewtonBackend(
        SceneCfg(
            model_file=_write_model(
                tmp_path,
                vertices=vertices,
                faces=faces,
                asset_attributes=asset_attributes,
                geom_attributes=geom_attributes,
                body_position=body_position,
                contact_sensor=contact_sensor,
            )
        ),
        num_envs=2,
        sim_dt=0.005,
        device=str(device),
        capacity_check_steps=1,
    )
    backend.materialize()
    return backend, deps


def test_convex_mesh_materializes_with_authored_geometry(tmp_path: Path) -> None:
    backend, deps = _cuda_backend(tmp_path)
    try:
        model = backend._model
        mesh_value = int(deps.newton.GeoType.MESH)
        plane_value = int(deps.newton.GeoType.PLANE)
        shape_type = np.asarray(model.shape_type.numpy(), dtype=np.int64)
        shape_world = np.asarray(model.shape_world.numpy(), dtype=np.int64)
        shape_transform = np.asarray(model.shape_transform.numpy(), dtype=np.float32)
        assert int(model.shape_count) == 4
        for world_id in range(2):
            world_types = shape_type[shape_world == world_id]
            assert sorted(world_types.tolist()) == sorted([mesh_value, plane_value])
        mesh_rows = np.flatnonzero(shape_type == mesh_value)
        for row in mesh_rows.tolist():
            np.testing.assert_allclose(shape_transform[row, :3], _GEOM_POS, rtol=0, atol=1e-6)
            np.testing.assert_allclose(
                shape_transform[row, 3:7], _GEOM_QUAT_XYZW, rtol=0, atol=1e-5
            )
            mesh = model.shape_source[row]
            np.testing.assert_allclose(
                np.asarray(mesh.vertices), _WEDGE_VERTICES, rtol=0, atol=1e-7
            )
        assert backend._metadata.geom_contype.tolist() == [1, 1]
    finally:
        backend.close()


def _solver_hull_vertices(model: Any) -> np.ndarray:
    polygon_ids = range(
        int(model.mesh_polyadr[0]),
        int(model.mesh_polyadr[0]) + int(model.mesh_polynum[0]),
    )
    hull_ids = {
        int(vertex)
        for polygon in polygon_ids
        for vertex in model.mesh_polyvert[
            int(model.mesh_polyvertadr[polygon]) : int(model.mesh_polyvertadr[polygon])
            + int(model.mesh_polyvertnum[polygon])
        ]
    }
    vertex_start = int(model.mesh_vertadr[0])
    return np.asarray(model.mesh_vert[vertex_start:][sorted(hull_ids)])


@pytest.mark.parametrize(
    ("vertices", "faces"),
    [
        (
            "0 0 0  0.1 0 0  0 0.1 0  0 0 0.1  0.0167 0.0167 0.0167",
            "0 2 1  0 1 3  0 3 2  1 2 4  2 3 4  3 1 4",
        ),
        (_WEDGE_VERTEX_TEXT, "0 2 1  0 1 3  0 3 2"),
    ],
)
def test_raw_mesh_materializes_with_newton_generated_hull(
    tmp_path: Path, vertices: str, faces: str
) -> None:
    backend, deps = _cuda_backend(tmp_path, vertices=vertices, faces=faces)
    try:
        mesh_value = int(deps.newton.GeoType.MESH)
        shape_type = np.asarray(backend._model.shape_type.numpy(), dtype=np.int64)
        assert np.count_nonzero(shape_type == mesh_value) == 2
        solver_model = backend._solver.mj_model
        assert solver_model.nmesh == 1
        hull = _solver_hull_vertices(solver_model)
        assert hull.shape == (4, 3)
        pairs = np.triu_indices(4, 1)
        hull_distances = np.linalg.norm(hull[:, None] - hull[None, :], axis=-1)[pairs]
        authored_distances = np.linalg.norm(
            _WEDGE_VERTICES[:, None] - _WEDGE_VERTICES[None, :], axis=-1
        )[pairs]
        np.testing.assert_allclose(
            np.sort(hull_distances), np.sort(authored_distances), atol=1e-6
        )
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("asset_attributes", "expected"),
    [("", 64), ('maxhullvert="128"', None)],
)
def test_dense_mesh_uses_newton_hull_budget(
    tmp_path: Path, asset_attributes: str, expected: int | None
) -> None:
    vertices, faces = _dense_sphere_mesh()
    backend, _ = _cuda_backend(
        tmp_path, vertices=vertices, faces=faces, asset_attributes=asset_attributes
    )
    try:
        hull = _solver_hull_vertices(backend._solver.mj_model)
        if expected is None:
            assert len(hull) > 64
        else:
            assert len(hull) == expected
    finally:
        backend.close()


def test_colliding_mesh_reports_contact(tmp_path: Path) -> None:
    backend, _ = _cuda_backend(tmp_path, body_position="0 0 -0.05")
    try:
        backend._host_step(np.zeros((2, 0), dtype=np.float32))
        np.testing.assert_array_equal(backend._host_read_sensor("mesh_floor"), [[1.0], [1.0]])
    finally:
        backend.close()


def test_visual_only_mesh_does_not_collide(tmp_path: Path) -> None:
    backend, _ = _cuda_backend(
        tmp_path,
        geom_attributes='contype="0" conaffinity="0"',
        body_position="0 0 -0.05",
        contact_sensor=False,
    )
    try:
        backend._host_step(np.zeros((2, 0), dtype=np.float32))
        np.testing.assert_array_equal(backend._contacts.rigid_contact_count.numpy(), [0])
        assert np.all(backend.get_physics_state()[:, 3] < -0.05)
    finally:
        backend.close()
