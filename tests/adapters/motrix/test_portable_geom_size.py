"""Primitive geometry size reset overrides in portable Motrix scenes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("motrixsim")

from unisim.backend.motrix.backend import MotrixBackend
from unisim.dr.types import RESET_TERM_GEOM_SIZE, ModelSourceDescriptor, ResetRandomizationPayload
from unisim.entities import SceneEntitySpec
from unisim.scene import SceneCfg


def _backend(tmp_path: Path) -> MotrixBackend:
    source = tmp_path / "geom-size.xml"
    source.write_text(
        """<mujoco><worldbody><body name="base" pos="0 0 .5">
        <freejoint name="root"/>
        <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
        <geom name="ball" type="sphere" size=".1" mass="0"/>
        <geom name="block" type="box" size=".1 .2 .3" mass="0"/>
        </body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("object", ModelSourceDescriptor(str(source)), kind="rigid"),)
    )
    return MotrixBackend(scene, 3, .002, base_name="object/base")


def _size_payload(backend: MotrixBackend, rows: np.ndarray) -> np.ndarray:
    defaults = backend.get_reset_term_default(RESET_TERM_GEOM_SIZE)
    return defaults[rows].copy().astype(np.float32)


def test_size_reset_rows_readback_defaults_and_repeated_writes(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    try:
        assert backend.get_dr_capabilities().supports_reset_term(RESET_TERM_GEOM_SIZE)
        defaults = backend.get_reset_term_default(RESET_TERM_GEOM_SIZE)
        assert defaults.shape == (3, backend.get_scene_layout().ngeom, 3)
        assert not defaults.flags.writeable
        ball_id = backend.get_geom_id("object/ball")
        block_id = backend.get_geom_id("object/block")
        rows = np.array([2, 0], dtype=np.intp)
        sizes = _size_payload(backend, rows)
        sizes[:, ball_id, 0] = [0.23, 0.34]
        sizes[:, block_id] = [[.2, .3, .4], [.4, .5, .6]]
        qpos = backend.get_state("qpos")["qpos"][rows].copy()
        qvel = backend.get_state("qvel")["qvel"][rows].copy()
        backend.set_state(
            rows, qpos, qvel, randomization=ResetRandomizationPayload(geom_size=sizes)
        )
        runtime = backend._portable_runtimes[0]
        ball = runtime.binding.geoms_by_id[int(runtime.binding.public_to_native_geom[ball_id])]
        block = runtime.binding.geoms_by_id[int(runtime.binding.public_to_native_geom[block_id])]
        np.testing.assert_allclose(ball.get_size_override(runtime.data)[:, 0], [.34, .1, .23])
        np.testing.assert_allclose(
            block.get_size_override(runtime.data), [[.4, .5, .6], [.1, .2, .3], [.2, .3, .4]]
        )
        backend.step(np.zeros((3, backend.num_actuators), dtype=np.float32))
        np.testing.assert_allclose(ball.get_size_override(runtime.data)[:, 0], [.34, .1, .23])
        backend.reset(np.array([2], dtype=np.intp))
        np.testing.assert_allclose(ball.get_size_override(runtime.data)[:, 0], [.34, .1, .23])
        np.testing.assert_array_equal(
            backend.get_reset_term_default(RESET_TERM_GEOM_SIZE), defaults
        )
        restored = _size_payload(backend, np.array([0], dtype=np.intp))
        backend.set_state(
            np.array([0], dtype=np.intp), qpos[1:2], qvel[1:2],
            randomization=ResetRandomizationPayload(geom_size=restored),
        )
        np.testing.assert_allclose(ball.get_size_override(runtime.data)[:, 0], [.1, .1, .23])
    finally:
        backend.close()


@pytest.mark.parametrize("invalid", ["shape", "negative", "unused", "nan"])
def test_size_prevalidation_preserves_state(tmp_path: Path, invalid: str) -> None:
    backend = _backend(tmp_path)
    try:
        rows = np.array([0, 2], dtype=np.intp)
        sizes = _size_payload(backend, rows)
        ball_id = backend.get_geom_id("object/ball")
        sizes[0, ball_id, 0] = .2
        if invalid == "shape":
            sizes = sizes[:, :, :2]
        elif invalid == "negative":
            sizes[1, ball_id, 0] = -.1
        elif invalid == "unused":
            sizes[1, ball_id, 2] = .5
        else:
            sizes[1, ball_id, 0] = np.nan
        runtime = backend._portable_runtimes[0]
        ball = runtime.binding.geoms_by_id[
            int(runtime.binding.public_to_native_geom[ball_id])
        ]
        original = ball.get_size_override(runtime.data).copy()
        with pytest.raises(ValueError, match="geom_size"):
            backend._prepare_portable_reset_randomization(
                ResetRandomizationPayload(geom_size=sizes), rows
            )
        np.testing.assert_array_equal(ball.get_size_override(runtime.data), original)
    finally:
        backend.close()


def test_size_rejects_mesh_slot_without_changing_fixed_variant(tmp_path: Path) -> None:
    from tests.adapters.motrix.test_portable_mesh_variants import _scene

    backend = MotrixBackend(_scene(tmp_path, np.asarray((1, 0))), 2, .002,
                            base_name="object/base")
    try:
        rows = np.array([0, 1], dtype=np.intp)
        sizes = _size_payload(backend, rows)
        mesh_id = backend.get_geom_id("object/head")
        ball_id = backend.get_geom_id("object/handle")
        sizes[0, ball_id, 0] = .03
        sizes[1, mesh_id, 0] += .01
        runtime = backend._portable_runtimes[0]
        mesh = runtime.binding.geoms_by_id[int(runtime.binding.public_to_native_geom[mesh_id])]
        ball = runtime.binding.geoms_by_id[int(runtime.binding.public_to_native_geom[ball_id])]
        selection = mesh.get_mesh_variant_override(runtime.data).copy()
        initial = ball.get_size_override(runtime.data).copy()
        with pytest.raises(ValueError, match="non-primitive"):
            backend._prepare_portable_reset_randomization(
                ResetRandomizationPayload(geom_size=sizes), rows
            )
        np.testing.assert_array_equal(mesh.get_mesh_variant_override(runtime.data), selection)
        np.testing.assert_array_equal(ball.get_size_override(runtime.data), initial)
    finally:
        backend.close()


def test_size_unavailable_fails_closed(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    try:
        sizes = _size_payload(backend, np.array([0]))
        backend._supports_geom_size_override = False
        assert not backend.get_dr_capabilities().supports_reset_term(RESET_TERM_GEOM_SIZE)
        with pytest.raises(NotImplementedError, match="geom_size"):
            backend.get_reset_term_default(RESET_TERM_GEOM_SIZE)
        with pytest.raises(NotImplementedError, match="geom_size"):
            backend._prepare_portable_reset_randomization(
                ResetRandomizationPayload(geom_size=sizes),
                np.array([0], dtype=np.intp),
            )
    finally:
        backend.close()
