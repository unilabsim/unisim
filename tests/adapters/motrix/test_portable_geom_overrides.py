"""Host geometry reset overrides against portable Motrix native row state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
mtx = pytest.importorskip("motrixsim")

from unisim.backend.motrix.backend import MotrixBackend  # noqa: E402
from unisim.dr.types import (  # noqa: E402
    RESET_TERM_GEOM_ACTIVE,
    RESET_TERM_GEOM_MESH_VARIANT,
    RESET_TERM_GEOM_POS,
    RESET_TERM_GEOM_QUAT,
    RESET_TERM_GEOM_SHAPE,
    ModelSourceDescriptor,
    ResetRandomizationPayload,
)
from unisim.entities import SceneEntitySpec  # noqa: E402
from unisim.scene import SceneCfg  # noqa: E402


def _backend(tmp_path: Path) -> MotrixBackend:
    source = tmp_path / "geoms.xml"
    source.write_text(
        """<mujoco><worldbody><body name="base" pos="0 0 .5">
        <freejoint name="root"/>
        <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
        <geom name="ball" type="sphere" size=".1" pos=".03 0 0" mass="0"/>
        <geom name="block" type="box" size=".1 .2 .3" pos="0 .04 0" mass="0"/>
        </body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("object", ModelSourceDescriptor(str(source)), kind="rigid"),)
    )
    return MotrixBackend(scene, 3, .002, base_name="object/base")


def _geom(backend: MotrixBackend, name: str):
    runtime = backend._portable_runtimes[0]
    public_id = backend.get_geom_id(f"object/{name}")
    native_id = int(runtime.binding.public_to_native_geom[public_id])
    return runtime.binding.geoms_by_id[native_id], runtime.data, public_id


def _set(backend: MotrixBackend, rows: np.ndarray, **overrides: np.ndarray) -> None:
    qpos = backend.get_state("qpos")["qpos"][rows].copy()
    qvel = backend.get_state("qvel")["qvel"][rows].copy()
    backend.set_state(rows, qpos, qvel, randomization=ResetRandomizationPayload(**overrides))


def _defaults(backend: MotrixBackend, term: str, rows: np.ndarray) -> np.ndarray:
    return backend.get_reset_term_default(term)[rows].copy()


def test_active_pose_selected_rows_readback_and_omitted_channels(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    try:
        terms = (RESET_TERM_GEOM_ACTIVE, RESET_TERM_GEOM_POS, RESET_TERM_GEOM_QUAT)
        assert all(backend.get_dr_capabilities().supports_reset_term(term) for term in terms)
        rows = np.array([2, 0], dtype=np.intp)
        active = _defaults(backend, RESET_TERM_GEOM_ACTIVE, rows)
        pos = _defaults(backend, RESET_TERM_GEOM_POS, rows)
        quat = _defaults(backend, RESET_TERM_GEOM_QUAT, rows)
        ball, data, index = _geom(backend, "ball")
        active[:, index] = [False, True]
        pos[:, index] = [[.1, .2, .3], [.4, .5, .6]]
        quat[:, index] = [[.5, .5, .5, .5], [0, 0, 0, 1]]  # public wxyz
        _set(backend, rows, geom_active=active, geom_pos=pos, geom_quat=quat)
        np.testing.assert_array_equal(ball.get_active_override(data), [True, True, False])
        np.testing.assert_allclose(ball.get_pos_override(data),
                                   [[.4, .5, .6], [.03, 0, 0], [.1, .2, .3]], atol=1e-6)
        np.testing.assert_allclose(ball.get_quat_override(data),
                                   [[0, 0, 1, 0], [0, 0, 0, 1], [.5, .5, .5, .5]], atol=1e-6)
        backend.reset(np.array([2], dtype=np.intp))
        backend.step(np.zeros((3, backend.num_actuators), dtype=np.float32))
        np.testing.assert_array_equal(ball.get_active_override(data), [True, True, False])
        # Only updating active must not revert either component of the pose.
        later = np.array([2], dtype=np.intp)
        _set(backend, later, geom_active=np.ones((1, backend.get_scene_layout().ngeom), dtype=bool))
        np.testing.assert_allclose(ball.get_pos_override(data)[2], [.1, .2, .3], atol=1e-6)
        np.testing.assert_allclose(ball.get_quat_override(data)[2], [.5, .5, .5, .5], atol=1e-6)
        new_pos = _defaults(backend, RESET_TERM_GEOM_POS, later)
        new_pos[0, index] = [.7, .8, .9]
        _set(backend, later, geom_pos=new_pos)
        np.testing.assert_allclose(ball.get_quat_override(data)[2], [.5, .5, .5, .5], atol=1e-6)
        assert not backend.get_reset_term_default(RESET_TERM_GEOM_ACTIVE).flags.writeable
        np.testing.assert_array_equal(backend.get_reset_term_default(RESET_TERM_GEOM_ACTIVE), True)
    finally:
        backend.close()


@pytest.mark.parametrize("term,kind", [
    (RESET_TERM_GEOM_ACTIVE, "dtype"), (RESET_TERM_GEOM_ACTIVE, "shape"),
    (RESET_TERM_GEOM_POS, "nan"), (RESET_TERM_GEOM_QUAT, "zero"),
    (RESET_TERM_GEOM_QUAT, "shape"),
])
def test_active_pose_invalid_payload_is_atomic(tmp_path: Path, term: str, kind: str) -> None:
    backend = _backend(tmp_path)
    try:
        rows = np.array([0, 2], dtype=np.intp)
        ball, data, index = _geom(backend, "ball")
        before = (ball.get_active_override(data).copy(), ball.get_pos_override(data).copy(),
                  ball.get_quat_override(data).copy())
        value = _defaults(backend, term, rows)
        if kind == "dtype":
            value = value.astype(np.int32)
        elif kind == "shape":
            value = value[:1]
        elif kind == "nan":
            value[1, index, 0] = np.nan
        else:
            value[1, index] = 0
        active = _defaults(backend, RESET_TERM_GEOM_ACTIVE, rows)
        active[0, index] = False
        payload = {"geom_active": active, term: value}
        with pytest.raises(ValueError, match=term):
            _set(backend, rows, **payload)
        for actual, original in zip(
            (ball.get_active_override(data), ball.get_pos_override(data),
             ball.get_quat_override(data)), before
        ):
            np.testing.assert_array_equal(actual, original)
    finally:
        backend.close()


def test_shape_and_mesh_variant_native_selection(tmp_path: Path) -> None:
    from tests.adapters.motrix.test_portable_mesh_variants import _scene

    backend = MotrixBackend(_scene(tmp_path, np.array([1, 0, 2])), 3, .002,
                            base_name="object/base")
    try:
        for term in (RESET_TERM_GEOM_SHAPE, RESET_TERM_GEOM_MESH_VARIANT):
            assert backend.get_dr_capabilities().supports_reset_term(term)
            assert not backend.get_reset_term_default(term).flags.writeable
        mesh, data, mesh_id = _geom(backend, "head")
        sphere, _, sphere_id = _geom(backend, "handle")
        rows = np.array([2, 0], dtype=np.intp)
        variants = _defaults(backend, RESET_TERM_GEOM_MESH_VARIANT, rows)
        variants[:, mesh_id] = [0, 2]
        _set(backend, rows, geom_mesh_variant=variants)
        np.testing.assert_array_equal(mesh.get_mesh_variant_override(data), [2, 0, 0])
        np.testing.assert_array_equal(sphere.get_mesh_variant_override(data), [-1, -1, -1])
        backend.reset(np.array([0], dtype=np.intp))
        backend.step(np.zeros((3, backend.num_actuators), dtype=np.float32))
        np.testing.assert_array_equal(mesh.get_mesh_variant_override(data), [2, 0, 0])
        # Switch a primitive to another category with all three dimensions valid.
        block_shapes = _defaults(backend, RESET_TERM_GEOM_SHAPE, np.array([1], dtype=np.intp))
        block_shapes[0, sphere_id] = "box"
        block_sizes = backend.get_reset_term_default("geom_size")[[1]].copy()
        block_sizes[0, sphere_id] = [.04, .05, .06]
        _set(backend, np.array([1], dtype=np.intp),
             geom_shape=block_shapes, geom_size=block_sizes)
        np.testing.assert_array_equal(sphere.get_shape_override(data)[[0, 2]],
                                      [int(mtx.Shape.Sphere)] * 2)
        assert sphere.get_shape_override(data)[1] == int(mtx.Shape.Cuboid)
        # GeomSphere's typed size getter exposes only its authored radius component.
        np.testing.assert_allclose(sphere.get_size_override(data)[1], [.04], atol=1e-6)
        np.testing.assert_array_equal(mesh.get_mesh_variant_override(data), [2, 0, 0])
        shapes = _defaults(backend, RESET_TERM_GEOM_SHAPE, np.array([2], dtype=np.intp))
        shapes[0, sphere_id] = "box"  # sphere size has no valid box y/z; reject before writing.
        before = mesh.get_mesh_variant_override(data).copy()
        with pytest.raises(ValueError, match="geom_shape"):
            _set(backend, np.array([2], dtype=np.intp), geom_shape=shapes)
        np.testing.assert_array_equal(mesh.get_mesh_variant_override(data), before)
    finally:
        backend.close()


@pytest.mark.parametrize("term,flag", [
    (RESET_TERM_GEOM_SHAPE, "_supports_geom_shape_override"),
    (RESET_TERM_GEOM_MESH_VARIANT, "_supports_geom_mesh_variant_override"),
])
def test_shape_variant_unavailable_fails_closed(
    tmp_path: Path, term: str, flag: str
) -> None:
    from tests.adapters.motrix.test_portable_mesh_variants import _scene

    backend = MotrixBackend(_scene(tmp_path, np.array([0, 1])), 2, .002,
                            base_name="object/base")
    try:
        value = _defaults(backend, term, np.array([0], dtype=np.intp))
        mesh, data, _ = _geom(backend, "head")
        before = mesh.get_mesh_variant_override(data).copy()
        setattr(backend, flag, False)
        assert not backend.get_dr_capabilities().supports_reset_term(term)
        with pytest.raises(NotImplementedError, match=term):
            backend.get_reset_term_default(term)
        with pytest.raises(NotImplementedError, match=term):
            _set(backend, np.array([0], dtype=np.intp), **{term: value})
        np.testing.assert_array_equal(mesh.get_mesh_variant_override(data), before)
    finally:
        backend.close()


@pytest.mark.parametrize("term,kind", [
    (RESET_TERM_GEOM_SHAPE, "category"),
    (RESET_TERM_GEOM_SHAPE, "shape"),
    (RESET_TERM_GEOM_MESH_VARIANT, "range"),
    (RESET_TERM_GEOM_MESH_VARIANT, "primitive"),
    (RESET_TERM_GEOM_MESH_VARIANT, "dtype"),
])
def test_invalid_shape_variant_prevalidation_preserves_all_rows(
    tmp_path: Path, term: str, kind: str
) -> None:
    from tests.adapters.motrix.test_portable_mesh_variants import _scene

    backend = MotrixBackend(_scene(tmp_path, np.array([1, 0, 2])), 3, .002,
                            base_name="object/base")
    try:
        rows = np.array([0, 2], dtype=np.intp)
        mesh, data, mesh_id = _geom(backend, "head")
        _, _, sphere_id = _geom(backend, "handle")
        original_shape = mesh.get_shape_override(data).copy()
        original_variant = mesh.get_mesh_variant_override(data).copy()
        value = _defaults(backend, term, rows)
        if kind == "category":
            value[1, mesh_id] = "invalid"
        elif kind == "shape":
            value = value[:1]
        elif kind == "range":
            value[1, mesh_id] = 99
        elif kind == "primitive":
            value[1, sphere_id] = 0
        else:
            value = value.astype(np.float32)
        with pytest.raises(ValueError, match=term):
            _set(backend, rows, **{term: value})
        np.testing.assert_array_equal(mesh.get_shape_override(data), original_shape)
        np.testing.assert_array_equal(mesh.get_mesh_variant_override(data), original_variant)
    finally:
        backend.close()
