"""Convex collision mesh geometry passthrough on the real Newton runtime."""

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
      <geom name="wedge_geom" type="mesh" mesh="wedge" pos="0.01 0.02 0.03"
            quat="0.7071068 0.7071068 0 0" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


def _newton_deps() -> NewtonDependencies:
    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    return deps


def _write_model(tmp_path: Path) -> str:
    model_file = tmp_path / "mesh_scene.xml"
    model_file.write_text(_MESH_MODEL, encoding="utf-8")
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


def _cuda_backend(tmp_path: Path) -> tuple[NewtonBackend, Any]:
    deps = _newton_deps()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton materialization requires a CUDA Warp device")
    backend = NewtonBackend(
        SceneCfg(model_file=_write_model(tmp_path)),
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
