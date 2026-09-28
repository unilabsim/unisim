"""SDK-free checks for the cold IsaacSim collision-geometry audit."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from unisim.backend.isaacsim.scene_worker import _native_collision_geometry_audit


class _FakeAttr:
    def __init__(self, value: Any):
        self.value = value

    def Get(self):  # noqa: N802 - mirrors USD
        return self.value

    def HasAuthoredValue(self) -> bool:  # noqa: N802 - mirrors USD
        return False


class _FakePrim:
    def __init__(self, path: str, attrs: dict[str, _FakeAttr], *, mesh: bool = True):
        self.path = path
        self.attrs = attrs
        self.mesh = mesh

    def GetPath(self):  # noqa: N802 - mirrors USD
        return self.path

    def GetTypeName(self):  # noqa: N802 - mirrors USD
        return "Mesh" if self.mesh else "Sphere"

    def GetPrimTypeInfo(self):  # noqa: N802 - mirrors USD
        return types.SimpleNamespace(GetSchemaTypeName=lambda: "Mesh" if self.mesh else "Sphere")

    def IsA(self, kind: Any) -> bool:  # noqa: N802 - mirrors USD
        return self.mesh and kind is _FAKE_USDGEOM.Mesh

    def HasAPI(self, kind: Any) -> bool:  # noqa: N802 - mirrors USD
        return kind is _FAKE_USDPHYSICS.MeshCollisionAPI

    def GetAttribute(self, name: str) -> _FakeAttr:  # noqa: N802 - mirrors USD
        return self.attrs[name]


class _FakeMesh:
    def __init__(self, prim: _FakePrim):
        self.prim = prim

    def GetPointsAttr(self):  # noqa: N802 - mirrors USD
        return self.prim.attrs["points"]

    def GetFaceVertexCountsAttr(self):  # noqa: N802 - mirrors USD
        return self.prim.attrs["faceVertexCounts"]

    def GetFaceVertexIndicesAttr(self):  # noqa: N802 - mirrors USD
        return self.prim.attrs["faceVertexIndices"]


class _FakeCollisionAPI:
    def __init__(self, prim: _FakePrim):
        self.prim = prim

    def GetCollisionEnabledAttr(self):  # noqa: N802 - mirrors USD
        return self.prim.attrs["collisionEnabled"]


class _FakeMeshCollisionAPI:
    def __init__(self, prim: _FakePrim):
        self.prim = prim

    def GetApproximationAttr(self):  # noqa: N802 - mirrors USD
        return self.prim.attrs["physics:approximation"]


class _FakePhysxCollisionAPI:
    def __init__(self, prim: _FakePrim):
        self.prim = prim

    def GetContactOffsetAttr(self):  # noqa: N802 - mirrors USD
        return self.prim.attrs["physxCollision:contactOffset"]

    def GetRestOffsetAttr(self):  # noqa: N802 - mirrors USD
        return self.prim.attrs["physxCollision:restOffset"]


class _FakeXformable:
    def __init__(self, _prim: _FakePrim):
        pass

    def ComputeLocalToWorldTransform(self, _time: Any) -> np.ndarray:  # noqa: N802
        return np.eye(4)

    def ComputeParentToWorldTransform(self, _time: Any) -> np.ndarray:  # noqa: N802
        return np.eye(4)


class _FakeRange:
    def __init__(self, minimum: list[float], maximum: list[float]):
        self.minimum = np.asarray(minimum)
        self.maximum = np.asarray(maximum)

    def IsEmpty(self) -> bool:  # noqa: N802 - mirrors USD
        return bool(np.any(self.maximum < self.minimum))

    def GetMin(self) -> np.ndarray:  # noqa: N802 - mirrors USD
        return self.minimum

    def GetMax(self) -> np.ndarray:  # noqa: N802 - mirrors USD
        return self.maximum


class _FakeBBox:
    def __init__(self, minimum: list[float], maximum: list[float]):
        self.range = _FakeRange(minimum, maximum)

    def ComputeAlignedRange(self) -> _FakeRange:  # noqa: N802 - mirrors USD
        return self.range


class _FakeBBoxCache:
    def __init__(self, *_args: Any, **_kwargs: Any):
        pass

    def ComputeLocalBound(self, prim: _FakePrim) -> _FakeBBox:  # noqa: N802 - mirrors USD
        return prim.attrs["localBounds"].value

    def ComputeWorldBound(self, prim: _FakePrim) -> _FakeBBox:  # noqa: N802 - mirrors USD
        return prim.attrs["worldBounds"].value


class _FakeUSD:
    class TimeCode:
        @staticmethod
        def Default():  # noqa: N802 - mirrors USD
            return None


_FAKE_USD = _FakeUSD()
_FAKE_USDGEOM = types.SimpleNamespace(
    Mesh=_FakeMesh,
    Plane=object(),
    Cube=object(),
    Sphere=object(),
    Capsule=object(),
    Cylinder=object(),
    Cone=object(),
    Xformable=_FakeXformable,
    BBoxCache=_FakeBBoxCache,
)
_FAKE_USDPHYSICS = types.SimpleNamespace(
    CollisionAPI=_FakeCollisionAPI,
    MeshCollisionAPI=_FakeMeshCollisionAPI,
)
_FAKE_PHYSXSCHEMA = types.SimpleNamespace(PhysxCollisionAPI=_FakePhysxCollisionAPI)


def _install_fake_pxr(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pxr = types.SimpleNamespace(
        Usd=_FAKE_USD,
        UsdGeom=_FAKE_USDGEOM,
        UsdPhysics=_FAKE_USDPHYSICS,
        PhysxSchema=_FAKE_PHYSXSCHEMA,
    )
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)


def _mesh_prim(points: list[list[float]], bounds: tuple[list[float], list[float]]) -> _FakePrim:
    local = _FakeBBox(*bounds)
    return _FakePrim(
        "/World/env_0/floor/collisions/floor",
        {
            "collisionEnabled": _FakeAttr(True),
            "physics:approximation": _FakeAttr("convexHull"),
            "physxCollision:contactOffset": _FakeAttr(-np.inf),
            "physxCollision:restOffset": _FakeAttr(-np.inf),
            "localBounds": _FakeAttr(local),
            "worldBounds": _FakeAttr(local),
            "points": _FakeAttr(points),
            "faceVertexCounts": _FakeAttr([len(points)]),
            "faceVertexIndices": _FakeAttr(list(range(len(points)))),
        },
    )


def test_collision_mesh_audit_distinguishes_finite_plane_from_degenerate_plane(monkeypatch):
    _install_fake_pxr(monkeypatch)

    finite = _native_collision_geometry_audit(
        _mesh_prim(
            [[-10, -10, 0], [-10, 10, 0], [10, 10, 0], [10, -10, 0]],
            ([-10, -10, 0], [10, 10, 0]),
        ),
        _FakeBBoxCache(),
    )
    assert finite["shape_kind"] == "mesh"
    assert finite["mesh_approximation"] == "convexHull"
    assert finite["mesh_vertex_count"] == 4
    assert finite["mesh_unique_vertex_count"] == 4
    assert finite["mesh_face_count"] == 1
    assert finite["mesh_all_points_coincident"] is False
    assert finite["mesh_points_extent"] == [20.0, 20.0, 0.0]
    assert finite["mesh_surface_area"] == 400.0
    assert finite["mesh_zero_area"] is False
    assert finite["zero_extent"] is False
    assert finite["zero_area"] is False
    assert finite["zero_volume_extent"] is True
    assert finite["planar_extent"] is True
    assert finite["world_extent"] == [20.0, 20.0, 0.0]

    degenerate = _native_collision_geometry_audit(
        _mesh_prim(
            [[0, 0, 0]] * 4,
            ([0, 0, 0], [0, 0, 0]),
        ),
        _FakeBBoxCache(),
    )
    assert degenerate["mesh_unique_vertex_count"] == 1
    assert degenerate["mesh_all_points_coincident"] is True
    assert degenerate["mesh_points_extent"] == [0.0, 0.0, 0.0]
    assert degenerate["mesh_surface_area"] == 0.0
    assert degenerate["mesh_zero_area"] is True
    assert degenerate["zero_extent"] is True
    assert degenerate["zero_area"] is True


def test_nonmesh_collision_audit_reports_explicit_absent_mesh_summary(monkeypatch):
    _install_fake_pxr(monkeypatch)
    prim = _mesh_prim(
        [[0, 0, 0]],
        ([-0.1] * 3, [0.1] * 3),
    )
    prim.mesh = False
    audit = _native_collision_geometry_audit(prim, _FakeBBoxCache())
    assert audit["shape_kind"] == "unknown"
    assert audit["mesh_vertex_count"] is None
    assert audit["mesh_all_points_coincident"] is None
    assert audit["zero_extent"] is False
