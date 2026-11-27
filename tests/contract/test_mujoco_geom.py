"""Tests for the unified MuJoCo geom-to-mesh extraction primitive.

The MuJoCo-dependent tests skip individually through ``_scene`` so that the
SDK-free tests (module import and missing-runtime diagnostics) still run when
the optional runtime is absent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from unisim.visualization.mujoco_geom import geom_mesh

_SCENE_XML = """
<mujoco>
  <asset>
    <mesh name="tetra" vertex="0 0 0  0.3 0 0  0 0.4 0  0 0 0.5"
          face="0 2 1  0 1 3  0 3 2  1 2 3"/>
    <mesh name="wedge" vertex="0 0 0  0.2 0 0  0 0.3 0  0.1 0.1 0.4"
          face="0 2 1  0 1 3  0 3 2  1 2 3"/>
    <hfield name="terrain" size="4 4 0.1 0.01" nrow="2" ncol="2"/>
  </asset>
  <worldbody>
    <geom name="plane" type="plane" size="2 2 0.1"/>
    <geom name="hill" type="hfield" hfield="terrain"/>
    <geom name="sphere" type="sphere" size="0.1"/>
    <geom name="capsule" type="capsule" size="0.05 0.2"/>
    <geom name="ellipsoid" type="ellipsoid" size="0.1 0.2 0.3"/>
    <geom name="cylinder" type="cylinder" size="0.1 0.25"/>
    <geom name="box" type="box" size="0.1 0.2 0.3"/>
    <geom name="tetra" type="mesh" mesh="tetra"/>
    <geom name="wedge" type="mesh" mesh="wedge"/>
  </worldbody>
</mujoco>
"""

_MISSING_MESSAGE = (
    "geom_mesh requires the optional 'mujoco' package, which is not installed; "
    "install it with the MuJoCo extra (pip install 'unisim-core[mujoco]')"
)

_UNIT_SPHERE_VOLUME = 4.0 / 3.0 * np.pi * 0.1**3
_CAPSULE_RADIUS, _CAPSULE_HALF_LENGTH = 0.05, 0.2
# The capsule profile samples 32 latitudes on linspace(-pi/2, pi/2): the odd
# step leaves no vertex on the equator, so the radial extent is r*cos(pi/62),
# which is 0.13% inside the analytic radius.
_CAPSULE_RADIAL_EXTENT = _CAPSULE_RADIUS * np.cos(np.pi / 62)

# name -> (vertex count, face count, mesh extents, analytic volume, tolerance)
#
# Every vertex lies on the analytic surface, so each generated mesh is inscribed
# and its volume stays below the analytic volume.  The tolerances are the
# volume deficits of the tessellations: a 2-subdivision icosphere covers 96.62%
# of the sphere, a 32-section cylinder has the base area of a regular 32-gon
# (sin(pi/16) / (pi/16) = 99.36%), and the capsule combines that cylinder part
# with 32 x 64 hemispheres (99.58% overall).  The box is exact.
_PRIMITIVES: dict[str, tuple[tuple[int, int], tuple[float, ...], float, float]] = {
    "sphere": ((162, 320), (0.1, 0.1, 0.1), _UNIT_SPHERE_VOLUME, 0.04),
    "ellipsoid": ((162, 320), (0.1, 0.2, 0.3), 6.0 * _UNIT_SPHERE_VOLUME, 0.04),
    "capsule": (
        (1922, 3840),
        (_CAPSULE_RADIAL_EXTENT, _CAPSULE_RADIAL_EXTENT, _CAPSULE_HALF_LENGTH + _CAPSULE_RADIUS),
        np.pi * _CAPSULE_RADIUS**2 * (2.0 * _CAPSULE_HALF_LENGTH)
        + _UNIT_SPHERE_VOLUME * (_CAPSULE_RADIUS / 0.1) ** 3,
        0.01,
    ),
    "cylinder": (
        (66, 128),
        (0.1, 0.1, 0.25),
        np.pi * 0.1**2 * 0.5,
        0.01,
    ),
    "box": ((8, 12), (0.1, 0.2, 0.3), 8.0 * 0.1 * 0.2 * 0.3, 1.0e-12),
}


def _scene() -> tuple[Any, Any]:
    """Compile the mixed-geom scene, skipping when the optional MuJoCo is absent."""
    mujoco = pytest.importorskip("mujoco")
    return mujoco, mujoco.MjModel.from_xml_string(_SCENE_XML)


def _geom_index(model: Any, name: str) -> int:
    return int(model.geom(name).id)


def _fake_model(mujoco: Any, kind: str, size: tuple[float, ...], **arrays: Any) -> SimpleNamespace:
    """Build a one-geom duck-typed model for data MuJoCo cannot compile."""
    fields: dict[str, Any] = {
        "ngeom": 1,
        "geom_type": np.array([int(getattr(mujoco.mjtGeom, kind))], dtype=np.int32),
        "geom_size": np.array([size], dtype=np.float64),
        "geom_dataid": np.array([-1], dtype=np.int32),
    }
    fields.update(arrays)
    return SimpleNamespace(**fields)


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume by the divergence theorem: positive when faces point outward."""
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1], triangles[:, 2])
    return float(np.einsum("ij,ij->i", triangles[:, 0], cross).sum() / 6.0)


def _run_isolated(code: str, *, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=None if env is None else {**os.environ, **env},
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize(
    "name", ["sphere", "capsule", "ellipsoid", "cylinder", "box", "tetra", "wedge"]
)
def test_supported_geom_types_yield_a_triangle_mesh(name: str) -> None:
    """Verify every geom type with a mesh form returns finite vertices and indexed faces."""
    _, model = _scene()

    mesh = geom_mesh(model, _geom_index(model, name))

    assert mesh is not None
    vertices, faces = mesh
    assert vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) > 0
    assert faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
    assert np.issubdtype(vertices.dtype, np.floating)
    assert np.issubdtype(faces.dtype, np.integer)
    assert np.isfinite(vertices).all()
    assert faces.min() >= 0 and faces.max() < len(vertices)


@pytest.mark.parametrize("name", ["tetra", "wedge"])
def test_mesh_geom_returns_a_copy_of_the_compiled_model_arrays(name: str) -> None:
    """Verify mesh geoms yield their own model slice without aliasing model memory."""
    _, model = _scene()
    index = _geom_index(model, name)
    dataid = int(model.geom_dataid[index])
    vertex_adr, vertex_num = int(model.mesh_vertadr[dataid]), int(model.mesh_vertnum[dataid])
    face_adr, face_num = int(model.mesh_faceadr[dataid]), int(model.mesh_facenum[dataid])
    expected_vertices = model.mesh_vert[vertex_adr : vertex_adr + vertex_num].copy()
    expected_faces = model.mesh_face[face_adr : face_adr + face_num].copy()

    vertices, faces = geom_mesh(model, index)

    assert vertices.shape == (vertex_num, 3)
    assert faces.shape == (face_num, 3)
    assert vertices.dtype == model.mesh_vert.dtype
    assert faces.dtype == model.mesh_face.dtype
    np.testing.assert_array_equal(vertices, expected_vertices)
    np.testing.assert_array_equal(faces, expected_faces)
    assert not np.shares_memory(vertices, model.mesh_vert)
    assert not np.shares_memory(faces, model.mesh_face)

    vertices[:] = 0.0
    faces[:] = 0
    again_vertices, again_faces = geom_mesh(model, index)

    model_vertices = model.mesh_vert[vertex_adr : vertex_adr + vertex_num]
    model_faces = model.mesh_face[face_adr : face_adr + face_num]
    np.testing.assert_array_equal(model_vertices, expected_vertices)
    np.testing.assert_array_equal(model_faces, expected_faces)
    np.testing.assert_array_equal(again_vertices, expected_vertices)
    np.testing.assert_array_equal(again_faces, expected_faces)


def test_mesh_geoms_are_sliced_by_their_own_mesh_address() -> None:
    """Verify the per-mesh vertex and face addresses select the geom's own mesh."""
    _, model = _scene()

    first = geom_mesh(model, _geom_index(model, "tetra"))
    second = geom_mesh(model, _geom_index(model, "wedge"))

    assert first is not None and second is not None
    assert model.mesh_vertadr[0] != model.mesh_vertadr[1]
    assert model.mesh_faceadr[0] != model.mesh_faceadr[1]
    assert not np.array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[0], model.mesh_vert[0 : model.mesh_vertnum[0]])
    wedge_adr, wedge_num = model.mesh_vertadr[1], model.mesh_vertnum[1]
    np.testing.assert_array_equal(second[0], model.mesh_vert[wedge_adr : wedge_adr + wedge_num])


@pytest.mark.parametrize("name", ["plane", "hill"])
def test_geom_types_without_a_mesh_form_return_none(name: str) -> None:
    """Verify plane and hfield geoms report no mesh form instead of raising."""
    _, model = _scene()

    assert geom_mesh(model, _geom_index(model, name)) is None


def test_mesh_geom_without_compiled_mesh_data_returns_none() -> None:
    """Verify a negative dataid means no mesh, while a valid dataid reads the arrays."""
    mujoco = pytest.importorskip("mujoco")
    vertices = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.5]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    addresses = dict(
        mesh_vertadr=np.array([1], dtype=np.int32),
        mesh_vertnum=np.array([2], dtype=np.int32),
        mesh_faceadr=np.array([1], dtype=np.int32),
        mesh_facenum=np.array([2], dtype=np.int32),
        mesh_vert=vertices,
        mesh_face=faces,
    )

    missing = _fake_model(mujoco, "mjGEOM_MESH", (0.0, 0.0, 0.0))
    present = _fake_model(
        mujoco,
        "mjGEOM_MESH",
        (0.0, 0.0, 0.0),
        geom_dataid=np.array([0], dtype=np.int32),
        **addresses,
    )

    assert geom_mesh(missing, 0) is None
    mesh = geom_mesh(present, 0)
    assert mesh is not None
    np.testing.assert_array_equal(mesh[0], vertices[1:3])
    np.testing.assert_array_equal(mesh[1], faces[1:3])


@pytest.mark.parametrize("name", sorted(_PRIMITIVES))
def test_generated_primitive_matches_the_analytic_geom(name: str) -> None:
    """Verify counts, extents, outward winding, and inscribed volume of each primitive."""
    _, model = _scene()
    counts, extents, analytic_volume, tolerance = _PRIMITIVES[name]

    mesh = geom_mesh(model, _geom_index(model, name))

    assert mesh is not None
    vertices, faces = mesh
    assert (len(vertices), len(faces)) == counts
    assert vertices.dtype == np.float64
    assert faces.dtype == np.int32
    # Centered on the geom frame, with capsule and cylinder aligned to local z.
    np.testing.assert_allclose(vertices.min(axis=0), -np.asarray(extents), atol=1.0e-12)
    np.testing.assert_allclose(vertices.max(axis=0), np.asarray(extents), atol=1.0e-12)
    # Every face normal points away from the center of the convex geom.
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(np.einsum("ij,ij->i", normals, triangles.mean(axis=1)) > 0.0)
    # Inscribed, so the volume never exceeds the analytic one and misses it by
    # at most the tessellation deficit.
    volume = _signed_volume(vertices, faces)
    assert analytic_volume * (1.0 - tolerance) <= volume <= analytic_volume * (1.0 + 1.0e-12)


def test_capsule_profile_stays_just_inside_the_analytic_radius() -> None:
    """Verify the equator-less capsule profile stays within 0.13% of the analytic radius."""
    _, model = _scene()

    mesh = geom_mesh(model, _geom_index(model, "capsule"))

    assert mesh is not None
    radial_extent = np.linalg.norm(mesh[0][:, :2], axis=1).max()
    assert radial_extent == pytest.approx(_CAPSULE_RADIAL_EXTENT, abs=1.0e-12)
    assert 1.0 - radial_extent / _CAPSULE_RADIUS < 0.0013


def test_sphere_triangles_stay_near_equilateral() -> None:
    """Verify every subdivision level is projected back onto the sphere.

    Projecting each level before the next one splits its edges keeps the
    triangles close to equilateral (largest/smallest face area 1.269).  A single
    projection after the last subdivision stretches that ratio to 1.506 and
    moves vertices by up to 2.3% of the radius, which is a different sphere
    from the one UniLab rendered before.
    """
    _, model = _scene()

    mesh = geom_mesh(model, _geom_index(model, "sphere"))

    assert mesh is not None
    vertices, faces = mesh
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    assert areas.max() / areas.min() < 1.3
    # Every vertex sits on the analytic sphere of the geom.
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 0.1, atol=1.0e-12)


@pytest.mark.parametrize(
    ("kind", "counts", "analytic_volume", "tolerance"),
    [
        ("mjGEOM_CAPSULE", (1922, 3840), _UNIT_SPHERE_VOLUME, 0.04),
        ("mjGEOM_CYLINDER", (66, 128), 0.0, 0.0),
    ],
)
def test_zero_half_length_primitives_do_not_crash(
    kind: str, counts: tuple[int, int], analytic_volume: float, tolerance: float
) -> None:
    """Verify a zero half-length yields a finite sphere-like capsule or flat cylinder.

    MuJoCo refuses to compile these sizes, so the path is exercised through a
    duck-typed model; a viewer must never receive NaN vertices from one.
    """
    mujoco = pytest.importorskip("mujoco")
    model = _fake_model(mujoco, kind, (0.1, 0.0, 0.0))

    mesh = geom_mesh(model, 0)

    assert mesh is not None
    vertices, faces = mesh
    assert (len(vertices), len(faces)) == counts
    assert np.isfinite(vertices).all()
    assert np.isfinite(faces).all()
    volume = _signed_volume(vertices, faces)
    assert analytic_volume * (1.0 - tolerance) <= volume <= analytic_volume * (1.0 + tolerance)


@pytest.mark.parametrize(
    "kind",
    ["mjGEOM_SPHERE", "mjGEOM_CAPSULE", "mjGEOM_ELLIPSOID", "mjGEOM_CYLINDER", "mjGEOM_BOX"],
)
def test_zero_size_primitives_stay_finite(kind: str) -> None:
    """Verify an all-zero size collapses onto the origin instead of producing NaNs."""
    mujoco = pytest.importorskip("mujoco")
    model = _fake_model(mujoco, kind, (0.0, 0.0, 0.0))

    mesh = geom_mesh(model, 0)

    assert mesh is not None
    vertices, faces = mesh
    assert np.isfinite(vertices).all()
    assert np.isfinite(faces).all()
    assert np.all(vertices == 0.0)


def test_geom_index_must_be_an_integer_within_the_model() -> None:
    """Verify bad indices raise ValueError naming the index and ngeom, numpy integers pass."""
    _, model = _scene()
    ngeom = int(model.ngeom)

    for index in (-1, ngeom, 4 * ngeom):
        with pytest.raises(ValueError, match=rf"outside \[0, ngeom\) for ngeom={ngeom}") as raised:
            geom_mesh(model, index)
        assert str(index) in str(raised.value)

    for index in (True, False, np.bool_(True), 0.0, 2.5, "0", None, [0]):
        with pytest.raises(ValueError, match=r"must be an integer, got ") as raised:
            geom_mesh(model, index)
        assert repr(index) in str(raised.value)
        assert f"ngeom={ngeom}" in str(raised.value)

    box = _geom_index(model, "box")
    expected = geom_mesh(model, box)
    assert expected is not None
    for index in (np.int64(box), np.int32(box)):
        accepted = geom_mesh(model, index)
        assert accepted is not None
        np.testing.assert_array_equal(accepted[0], expected[0])
        np.testing.assert_array_equal(accepted[1], expected[1])


def test_missing_mujoco_runtime_raises_an_actionable_error() -> None:
    """Verify a missing MuJoCo package fails closed with the install hint and chained cause."""
    _run_isolated(
        f"""
import importlib.abc
import sys

class NoMujoco(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'mujoco':
            raise ImportError('mujoco blocked for this test')
        return None

sys.meta_path.insert(0, NoMujoco())
from unisim.optional import OptionalDependencyError
from unisim.visualization.mujoco_geom import geom_mesh
assert 'mujoco' not in sys.modules
try:
    geom_mesh(object(), 0)
except OptionalDependencyError as error:
    assert str(error) == {_MISSING_MESSAGE!r}, str(error)
    assert isinstance(error.__cause__, ImportError)
    assert 'mujoco blocked' in str(error.__cause__)
else:
    raise AssertionError('expected OptionalDependencyError')
assert 'mujoco' not in sys.modules
"""
    )


def test_module_import_loads_no_optional_runtime() -> None:
    """Verify importing the module loads neither MuJoCo nor a mesh library nor Viser."""
    _run_isolated(
        """
import importlib.abc
import sys

blocked = {'unilab', 'viser', 'trimesh', 'scipy', 'mujoco', 'mjbatch', 'mujoco_warp',
           'motrixsim', 'pydrake', 'warp', 'newton', 'genesis', 'superdex', 'isaacgym',
           'isaacsim', 'torch', 'omni'}

class NoOptionalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in blocked, fullname
        return None

sys.meta_path.insert(0, NoOptionalRuntime())
import unisim.visualization.mujoco_geom as module
assert module.__all__ == ['geom_mesh']
assert callable(module.geom_mesh)
assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)
"""
    )


def test_primitive_tessellation_needs_no_mesh_library_at_call_time() -> None:
    """Verify generation runs with trimesh blocked, so that dependency is gone at call time."""
    pytest.importorskip("mujoco")
    counts_by_kind = {name: value[0] for name, value in _PRIMITIVES.items()}
    # The child imports MuJoCo only to compile models, so rendering is disabled
    # there explicitly.  A test process that imported ``render_many`` exports
    # ``MUJOCO_GL=osmesa`` on a headless Linux host, and without an OSMesa
    # library that makes ``import mujoco`` fail before the code under test runs.
    _run_isolated(
        f"""
import importlib.abc
import sys

blocked = {{'unilab', 'viser', 'trimesh', 'scipy', 'torch', 'mjbatch'}}

class NoMeshLibrary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in blocked, fullname
        return None

sys.meta_path.insert(0, NoMeshLibrary())
import mujoco
from unisim.visualization.mujoco_geom import geom_mesh
for kind, counts in {counts_by_kind!r}.items():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="%s" size="0.1 0.2 0.3"/></worldbody></mujoco>' % kind
    )
    vertices, faces = geom_mesh(model, 0)
    assert (len(vertices), len(faces)) == counts, (kind, len(vertices), len(faces))
assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)
""",
        env={"MUJOCO_GL": "disable"},
    )
