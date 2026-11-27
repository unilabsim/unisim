"""Tests for the shared DebugPrimitive-to-mjvScene conversion and its builders.

The NumPy-only builders and the subprocess-backed boundary tests run without
MuJoCo; the scene-conversion tests skip through ``pytest.importorskip`` in
their fixtures instead of at module level for that reason (the same convention
as ``tests/contract/test_debug_overlay.py``).
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from unisim.backend.base import DebugPrimitive
from unisim.utils.math import np_quat_apply
from unisim.visualization.debug_primitives import (
    append_debug_primitives,
    quat_from_z_axis,
    segment_arrow,
)

_PLANE_MODEL = '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>'


@dataclasses.dataclass(frozen=True)
class _DuckOverlay:
    """Duck-typed overlay: read like a DebugPrimitive without being one of its kinds."""

    kind: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] | None = None
    size: tuple[float, ...] = ()
    rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


def _rotate(quat, vector) -> np.ndarray:
    """Rotate one vector by a wxyz quaternion with NumPy only (no MuJoCo needed)."""
    return np_quat_apply(np.asarray(quat, dtype=np.float64), np.asarray(vector, dtype=np.float64))


def _run_isolated(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout


# --------------------------------------------------------------------------- builders


def test_quat_from_z_axis_rotates_z_onto_direction() -> None:
    """Verify the quaternion is unit length and maps the +z axis onto the direction."""
    quat = quat_from_z_axis([1.0, 0.0, 0.0])

    assert np.linalg.norm(quat) == pytest.approx(1.0)
    assert np.allclose(_rotate(quat, [0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-9)


def test_quat_from_z_axis_handles_opposite_and_zero() -> None:
    """Verify the antiparallel fallback still lands on the direction and zero is rejected."""
    quat = quat_from_z_axis([0.0, 0.0, -2.0])
    assert np.allclose(_rotate(quat, [0.0, 0.0, 1.0]), [0.0, 0.0, -1.0], atol=1e-9)
    with pytest.raises(ValueError, match="non-zero"):
        quat_from_z_axis([0.0, 0.0, 0.0])


def test_quat_from_z_axis_accepts_arrays_and_unnormalized_directions() -> None:
    """Verify numpy inputs and scaled directions yield the same quaternion as unit lists."""
    expected = quat_from_z_axis([0.0, 1.0, 0.0])

    np.testing.assert_allclose(quat_from_z_axis(np.array([0.0, 7.0, 0.0])), expected, atol=1e-12)
    assert np.allclose(_rotate(expected, [0.0, 0.0, 1.0]), [0.0, 1.0, 0.0], atol=1e-9)


def test_segment_arrow_spans_endpoints() -> None:
    """Verify the arrow primitive starts at p0 and its local +z reaches p1."""
    arrow = segment_arrow([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], rgba=(1.0, 0.0, 0.0, 1.0))

    assert arrow is not None
    assert arrow.kind == "arrow"
    assert arrow.pos == (0.0, 0.0, 0.0)
    assert arrow.rgba == (1.0, 0.0, 0.0, 1.0)
    assert arrow.size == pytest.approx((2.0,))
    direction = _rotate(arrow.quat, [0.0, 0.0, 1.0]) * arrow.size[0]
    assert np.allclose(arrow.pos + direction, [0.0, 0.0, 2.0], atol=1e-9)


def test_segment_arrow_accepts_array_endpoints() -> None:
    """Verify numpy endpoints build the same arrow as sequences."""
    arrow = segment_arrow(
        np.array([0.1, 0.2, 0.3]), np.array([0.6, 0.2, 0.3]), rgba=np.array([0.0, 1.0, 0.0, 1.0])
    )

    assert arrow is not None
    assert arrow.size == pytest.approx((0.5,))
    assert arrow.pos == (0.1, 0.2, 0.3)
    assert arrow.rgba == (0.0, 1.0, 0.0, 1.0)


def test_segment_arrow_returns_none_when_degenerate() -> None:
    """Verify coincident endpoints yield None."""
    assert (
        segment_arrow([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], rgba=(1.0, 1.0, 1.0, 1.0)) is None
    )


def test_segment_arrow_default_min_length() -> None:
    """Verify the default 1e-6 cutoff rejects shorter segments and keeps longer ones."""
    white = (1.0, 1.0, 1.0, 1.0)
    assert segment_arrow([0.0, 0.0, 0.0], [5e-7, 0.0, 0.0], rgba=white) is None
    assert segment_arrow([0.0, 0.0, 0.0], [1e-4, 0.0, 0.0], rgba=white) is not None
    # An explicit cutoff replaces the default.
    assert segment_arrow([0.0, 0.0, 0.0], [0.05, 0.0, 0.0], rgba=white, min_length=0.1) is None
    assert segment_arrow([0.0, 0.0, 0.0], [0.2, 0.0, 0.0], rgba=white, min_length=0.1) is not None


# --------------------------------------------------------------------------- conversion


class TestAppendDebugPrimitives:
    """Per-env entry point over a real mjvScene (skipped without MuJoCo)."""

    MESH_MATERIAL_MODEL = """<mujoco>
      <asset>
        <texture name="checker" type="2d" builtin="checker" width="8" height="8"
                 rgb1="1 0 0" rgb2="0 1 0"/>
        <material name="sticker" texture="checker"/>
        <mesh name="tri" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="tri" material="sticker"/>
      </worldbody>
    </mujoco>"""

    @pytest.fixture(autouse=True)
    def _mujoco(self):
        self.mujoco = pytest.importorskip("mujoco")
        self.model = self.mujoco.MjModel.from_xml_string(_PLANE_MODEL)

    def _scene(self, maxgeom: int = 64):
        return self.mujoco.MjvScene(self.model, maxgeom=maxgeom)

    def test_adds_expected_geom_counts(self) -> None:
        """Verify sphere, box, frame triad and arrow convert while text is a no-op."""
        scene = self._scene()
        primitives = [
            DebugPrimitive(kind="sphere", pos=(0.0, 0.0, 0.5), size=(0.05,)),
            DebugPrimitive(kind="box", pos=(0.2, 0.0, 0.5), size=(0.05, 0.05, 0.05)),
            DebugPrimitive(kind="frame", pos=(0.0, 0.2, 0.5), size=(0.1,)),
            DebugPrimitive(
                kind="arrow",
                pos=(0.0, 0.0, 0.5),
                quat=quat_from_z_axis([1.0, 0.0, 0.0]),
                size=(0.3,),
            ),
            DebugPrimitive(kind="text", pos=(0.0, 0.0, 0.8), text="label"),
        ]

        # sphere(1) + box(1) + frame(3 axes) + arrow(1); text is a documented no-op.
        assert append_debug_primitives(scene, [primitives]) == 6
        assert scene.ngeom == 6

    def test_none_overlays_inject_nothing(self) -> None:
        """Verify a frame without overlays is a pure no-op."""
        scene = self._scene()

        assert append_debug_primitives(scene, None) == 0
        assert scene.ngeom == 0

    def test_stops_at_maxgeom(self) -> None:
        """Verify injection stops once the caller-owned scene capacity is reached."""
        scene = self._scene(maxgeom=1)
        primitives = [
            DebugPrimitive(kind="sphere", pos=(0.0, 0.0, 0.5), size=(0.05,)),
            DebugPrimitive(kind="sphere", pos=(0.1, 0.0, 0.5), size=(0.05,)),
        ]

        assert append_debug_primitives(scene, [primitives]) == 1
        assert scene.ngeom == 1

    @pytest.mark.parametrize(
        ("maxgeom", "expected"),
        [(1, 1), (2, 2), (3, 3), (4, 4)],
        ids=["first-axis", "mid-triad", "triad-complete", "all"],
    )
    def test_maxgeom_truncates_mid_frame(self, maxgeom: int, expected: int) -> None:
        """Verify a frame triad is cut between axes and never overflows the scene."""
        scene = self._scene(maxgeom=maxgeom)
        primitives = [
            DebugPrimitive(kind="frame", pos=(0.0, 0.0, 0.0), size=(0.1,)),
            DebugPrimitive(kind="sphere", pos=(0.0, 0.0, 0.5), size=(0.05,)),
        ]

        assert append_debug_primitives(scene, [primitives]) == expected
        assert scene.ngeom == expected

    def test_ghost_geom_requires_resolvable_mesh(self) -> None:
        """Verify a ghost primitive without a resolved mesh fails closed."""
        scene = self._scene()
        ghost = DebugPrimitive(kind="ghost_geom", pos=(0.0, 0.0, 0.5), mesh_asset="missing_mesh")

        with pytest.raises(ValueError, match="missing_mesh"):
            append_debug_primitives(scene, [[ghost]])

    def test_ghost_geom_uses_mesh_dataid(self) -> None:
        """Verify a resolved ghost primitive draws the registered mesh."""
        scene = self._scene()
        ghost = DebugPrimitive(kind="ghost_geom", pos=(0.0, 0.0, 0.5), mesh_asset="goal")

        assert append_debug_primitives(scene, [[ghost]], mesh_ids={"goal": 3}) == 1
        assert int(scene.geoms[0].type) == int(self.mujoco.mjtGeom.mjGEOM_MESH)
        assert int(scene.geoms[0].dataid) == 3

    def test_mesh_materials_bind_texture_to_ghost_geom(self) -> None:
        """Verify mesh_materials binds the textured material; an unlisted asset stays flat."""
        mujoco = self.mujoco
        model = mujoco.MjModel.from_xml_string(self.MESH_MATERIAL_MODEL)
        mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "tri")
        mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "sticker")
        assert mesh_id >= 0 and mat_id >= 0
        overlays = [[DebugPrimitive(kind="ghost_geom", pos=(0.0, 0.0, 0.3), mesh_asset="goal")]]

        bound = mujoco.MjvScene(model, maxgeom=8)
        assert append_debug_primitives(
            bound, overlays, mesh_ids={"goal": mesh_id}, mesh_materials={"goal": mat_id}
        ) == 1
        assert int(bound.geoms[0].matid) == mat_id
        assert int(bound.geoms[0].texcoord) == 1

        flat = mujoco.MjvScene(model, maxgeom=8)
        append_debug_primitives(
            flat, overlays, mesh_ids={"goal": mesh_id}, mesh_materials={"other": mat_id}
        )
        assert int(flat.geoms[0].matid) == -1
        assert int(flat.geoms[0].texcoord) == 0

    def test_offsets_shift_each_env_in_xy(self) -> None:
        """Verify per-env grid offsets shift XY only, indexing overlays env by env."""
        scene = self._scene()
        sphere = DebugPrimitive(kind="sphere", pos=(0.0, 0.0, 0.5), size=(0.05,))
        overlays = [[sphere], None, [sphere], []]
        offsets = np.array([[0.0, 0.0], [9.0, 9.0], [1.0, 2.0], [5.0, 5.0]])

        assert append_debug_primitives(scene, overlays, offsets=offsets) == 2
        np.testing.assert_allclose(scene.geoms[0].pos, [0.0, 0.0, 0.5], atol=1e-7)
        np.testing.assert_allclose(scene.geoms[1].pos, [1.0, 2.0, 0.5], atol=1e-7)

        # A plain sequence of pairs is accepted as well, and no offsets keeps poses env-local.
        scene = self._scene()
        append_debug_primitives(scene, [[sphere]], offsets=[[0.25, -0.5]])
        np.testing.assert_allclose(scene.geoms[0].pos, [0.25, -0.5, 0.5], atol=1e-7)

    def test_offset_applies_to_oriented_arrow_endpoints(self) -> None:
        """Verify an offset translates a rotated arrow wholesale instead of rotating it."""
        scene = self._scene()
        arrow = DebugPrimitive(
            kind="arrow",
            pos=(0.0, 0.0, 0.5),
            quat=quat_from_z_axis([1.0, 0.0, 0.0]),
            size=(0.4,),
        )

        assert append_debug_primitives(scene, [[arrow]], offsets=[[1.0, 2.0]]) == 1
        geom = scene.geoms[0]
        rotation = np.asarray(geom.mat, dtype=np.float64).reshape(3, 3)
        np.testing.assert_allclose(geom.pos, [1.0, 2.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(
            geom.pos + rotation[:, 2] * geom.size[2], [1.4, 2.0, 0.5], atol=1e-6
        )

    def test_segment_arrow_lands_on_p1_in_the_scene(self) -> None:
        """Verify a built arrow converts to a geom whose local +z spans p0 to p1."""
        scene = self._scene()
        arrow = segment_arrow(
            [0.1, 0.2, 0.3], [0.1, 0.2, 0.9], rgba=(0.0, 0.0, 1.0, 1.0)
        )
        assert arrow is not None

        assert append_debug_primitives(scene, [[arrow]]) == 1
        geom = scene.geoms[0]
        rotation = np.asarray(geom.mat, dtype=np.float64).reshape(3, 3)
        assert int(geom.type) == int(self.mujoco.mjtGeom.mjGEOM_ARROW)
        np.testing.assert_allclose(geom.pos, [0.1, 0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(
            geom.pos + rotation[:, 2] * geom.size[2], [0.1, 0.2, 0.9], atol=1e-6
        )
        np.testing.assert_allclose(geom.rgba, [0.0, 0.0, 1.0, 1.0], atol=1e-7)

    def test_unknown_kind_is_a_noop(self) -> None:
        """Verify an undocumented duck-typed kind is skipped without stopping later ones."""
        scene = self._scene()
        unknown = _DuckOverlay(kind="sparkles", pos=(0.0, 0.0, 0.5), size=(0.1,))
        sphere = DebugPrimitive(kind="sphere", pos=(0.0, 0.0, 0.5), size=(0.05,))

        assert append_debug_primitives(scene, [[unknown]]) == 0
        assert scene.ngeom == 0
        assert append_debug_primitives(scene, [[unknown, sphere]]) == 1
        assert scene.ngeom == 1


class TestLineKind:
    """The duck-typed ``line`` kind shared with UniLab's interactive overlays."""

    @pytest.fixture(autouse=True)
    def _mujoco(self):
        self.mujoco = pytest.importorskip("mujoco")
        self.model = self.mujoco.MjModel.from_xml_string(_PLANE_MODEL)

    def _scene(self, maxgeom: int = 16):
        return self.mujoco.MjvScene(self.model, maxgeom=maxgeom)

    def test_line_is_not_a_debug_primitive_kind(self) -> None:
        """Verify ``line`` only exists as a duck-typed overlay kind."""
        with pytest.raises(ValueError, match="kind must be one of"):
            DebugPrimitive(kind="line", pos=(0.0, 0.0, 0.0), size=(0.5,))  # type: ignore[arg-type]

    def test_line_draws_plain_segment_along_local_z(self) -> None:
        """Verify a line geom starts at pos and runs size[0] along the local +z axis."""
        scene = self._scene()
        line = _DuckOverlay(
            kind="line",
            pos=(0.0, 0.0, 0.5),
            quat=quat_from_z_axis([1.0, 0.0, 0.0]),
            size=(0.4,),
            rgba=(0.0, 1.0, 1.0, 1.0),
        )

        assert append_debug_primitives(scene, [[line]]) == 1
        geom = scene.geoms[0]
        rotation = np.asarray(geom.mat, dtype=np.float64).reshape(3, 3)
        assert int(geom.type) == int(self.mujoco.mjtGeom.mjGEOM_LINE)
        np.testing.assert_allclose(geom.pos, [0.0, 0.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(rotation[:, 2], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(geom.size[:2], [1e-3, 1e-3], rtol=1e-6)
        np.testing.assert_allclose(geom.size[2], [0.4], atol=1e-6)
        np.testing.assert_allclose(
            geom.pos + rotation[:, 2] * geom.size[2], [0.4, 0.0, 0.5], atol=1e-6
        )
        np.testing.assert_allclose(geom.rgba, [0.0, 1.0, 1.0, 1.0], atol=1e-7)

    def test_line_without_quat_runs_along_world_z(self) -> None:
        """Verify an unrotated line points up the +z axis."""
        scene = self._scene()
        line = _DuckOverlay(kind="line", pos=(0.25, -0.25, 0.1), size=(2.0,))

        assert append_debug_primitives(scene, [[line]]) == 1
        geom = scene.geoms[0]
        rotation = np.asarray(geom.mat, dtype=np.float64).reshape(3, 3)
        np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(geom.size[0], [0.004], rtol=1e-6)  # 0.002 * length
        np.testing.assert_allclose(
            geom.pos + rotation[:, 2] * geom.size[2], [0.25, -0.25, 2.1], atol=1e-6
        )

    def test_line_with_empty_size_draws_zero_length(self) -> None:
        """Verify an empty size yields the documented 0.0 length."""
        scene = self._scene()
        line = _DuckOverlay(kind="line", pos=(0.0, 0.0, 0.0))

        assert append_debug_primitives(scene, [[line]]) == 1
        assert int(scene.geoms[0].type) == int(self.mujoco.mjtGeom.mjGEOM_LINE)
        assert float(scene.geoms[0].size[2]) == pytest.approx(0.0, abs=1e-7)

    def test_line_honors_offset(self) -> None:
        """Verify grid offsets shift a line's start and end together."""
        scene = self._scene()
        line = _DuckOverlay(
            kind="line", pos=(0.0, 0.0, 0.5), quat=quat_from_z_axis([1.0, 0.0, 0.0]), size=(0.5,)
        )

        assert append_debug_primitives(scene, [[line]], offsets=[[1.0, 2.0]]) == 1
        geom = scene.geoms[0]
        rotation = np.asarray(geom.mat, dtype=np.float64).reshape(3, 3)
        np.testing.assert_allclose(geom.pos, [1.0, 2.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(
            geom.pos + rotation[:, 2] * geom.size[2], [1.5, 2.0, 0.5], atol=1e-6
        )

    def test_line_stops_at_maxgeom(self) -> None:
        """Verify lines obey the same maxgeom rule as the other kinds."""
        scene = self._scene(maxgeom=1)
        lines = [
            _DuckOverlay(kind="line", pos=(0.0, 0.0, 0.0), size=(0.5,)),
            _DuckOverlay(kind="line", pos=(0.0, 0.0, 0.1), size=(0.5,)),
        ]

        assert append_debug_primitives(scene, [lines]) == 1
        assert scene.ngeom == 1

    def test_line_mixes_with_typed_primitives_in_order(self) -> None:
        """Verify lines and DebugPrimitive kinds share one scene in list order."""
        scene = self._scene()
        primitives = [
            DebugPrimitive(kind="sphere", pos=(0.0, 0.0, 0.5), size=(0.05,)),
            _DuckOverlay(kind="line", pos=(0.0, 0.0, 0.0), size=(0.5,)),
            DebugPrimitive(kind="text", pos=(0.0, 0.0, 0.8), text="skip-me"),
            _DuckOverlay(kind="line", pos=(0.0, 0.0, 0.1), size=(0.5,)),
        ]

        assert append_debug_primitives(scene, [primitives]) == 3
        types = [int(geom.type) for geom in scene.geoms[:3]]
        assert types == [
            int(self.mujoco.mjtGeom.mjGEOM_SPHERE),
            int(self.mujoco.mjtGeom.mjGEOM_LINE),
            int(self.mujoco.mjtGeom.mjGEOM_LINE),
        ]


# --------------------------------------------------------------------------- boundaries


def test_render_many_reexports_the_shared_conversion() -> None:
    """Verify render_many delegates to the one conversion implementation."""
    pytest.importorskip("mujoco")
    import unisim.visualization.debug_primitives as debug_primitives
    from unisim.visualization import render_many

    assert render_many.append_debug_primitives is debug_primitives.append_debug_primitives
    assert "_append_primitive" not in vars(render_many)


def test_module_imports_without_engine_sdks_or_the_gl_probe() -> None:
    """Verify the module needs no engine SDK and never triggers the offline GL probe."""
    _run_isolated(
        """
        import importlib.abc
        import os
        import sys

        import numpy as np

        blocked = {'unilab', 'viser', 'mujoco', 'mjbatch', 'motrixsim', 'pydrake', 'warp',
                   'newton', 'genesis', 'superdex', 'isaacgym', 'isaacsim', 'torch', 'omni'}

        class NoOptionalRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                assert fullname.split('.')[0] not in blocked, fullname
                return None

        os.environ.pop('MUJOCO_GL', None)
        sys.meta_path.insert(0, NoOptionalRuntime())
        import unisim.visualization.debug_primitives as debug_primitives
        assert 'unisim.visualization.render_many' not in sys.modules
        assert 'MUJOCO_GL' not in os.environ
        arrow = debug_primitives.segment_arrow([0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                                               rgba=(1.0, 0.0, 0.0, 1.0))
        assert arrow is not None and arrow.kind == 'arrow'
        # +z onto +y is a quarter turn about x: (cos 45deg, -sin 45deg, 0, 0).
        np.testing.assert_allclose(
            debug_primitives.quat_from_z_axis([0.0, 1.0, 0.0]),
            (2 ** -0.5, -(2 ** -0.5), 0.0, 0.0),
            atol=1e-12,
        )
        assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)
        """
    )


def test_missing_mujoco_raises_actionable_optional_dependency_error() -> None:
    """Verify converting a primitive without MuJoCo fails with the install hint."""
    _run_isolated(
        """
        import importlib.abc
        import sys
        from types import SimpleNamespace

        class NoMujoco(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == 'mujoco':
                    raise ImportError('mujoco blocked for this test')
                return None

        sys.meta_path.insert(0, NoMujoco())
        import unisim.visualization.debug_primitives as debug_primitives
        from unisim.backend.base import DebugPrimitive
        from unisim.optional import OptionalDependencyError

        scene = SimpleNamespace(ngeom=0, maxgeom=16)
        # Nothing to convert: the optional package is never needed.
        assert debug_primitives.append_debug_primitives(scene, None) == 0
        assert debug_primitives.append_debug_primitives(scene, [None, []]) == 0
        sphere = DebugPrimitive(kind='sphere', pos=(0.0, 0.0, 0.0), size=(0.1,))
        try:
            debug_primitives.append_debug_primitives(scene, [[sphere]])
        except OptionalDependencyError as error:
            assert isinstance(error, ImportError)
            assert isinstance(error.__cause__, ImportError)
            assert 'mujoco blocked' in str(error.__cause__)
            assert str(error) == (
                "append_debug_primitives requires the optional 'mujoco' package, which is not "
                "installed; install it with the MuJoCo extra (pip install 'unisim-core[mujoco]')"
            )
        else:
            raise AssertionError('expected OptionalDependencyError')
        assert 'mujoco' not in sys.modules
        """
    )
