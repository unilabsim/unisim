from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("motrixsim")

from unisim.backend.motrix.backend import MotrixBackend
from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityVariantBinding, SceneEntitySpec
from unisim.scene import SceneCfg

TETRAHEDRON = """v 0 0 0
v .1 0 0
v 0 .1 0
v 0 0 .1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


def _mesh_source(
    tmp_path: Path,
    name: str,
    *,
    scale: str,
    body_mass: str = "1",
) -> ModelSourceDescriptor:
    obj = tmp_path / f"tetrahedron-{name}.obj"
    tmp_path.mkdir(parents=True, exist_ok=True)
    obj.write_text(TETRAHEDRON)
    path = tmp_path / f"{name}.xml"
    path.write_text(
        f"""<mujoco>
  <asset><mesh name="head" file="{obj}" scale="{scale}"/></asset>
  <worldbody>
    <body name="base" pos="0 0 .5">
      <freejoint name="root"/>
      <inertial mass="{body_mass}" pos="0 0 0" diaginertia=".01 .01 .01"/>
      <geom name="handle" type="sphere" size=".02" mass="0"/>
      <geom name="head" type="mesh" mesh="head" mass="0"/>
    </body>
  </worldbody>
</mujoco>
"""
    )
    return ModelSourceDescriptor(str(path))


def _scene(tmp_path: Path, assignment: np.ndarray) -> SceneCfg:
    sources = (
        _mesh_source(tmp_path, "small", scale=".8 .8 .8"),
        _mesh_source(tmp_path, "medium", scale="1.6 1.6 1.6"),
        _mesh_source(tmp_path, "large", scale="2.4 2.4 2.4"),
    )
    return SceneCfg(
        entity_assets=(SceneEntitySpec("object", sources[0], kind="rigid"),),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(
                np.asarray(assignment, dtype=np.int32),
                sources,
                layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
            ),
        ),
    )


def test_uniform_mesh_variants_persist_across_reset_and_step(tmp_path: Path) -> None:
    assignment = np.asarray((2, 0, 1, 1), dtype=np.int32)
    backend = MotrixBackend(_scene(tmp_path, assignment), 4, 0.002, base_name="object/base")
    try:
        assert len(backend._portable_runtimes) == 1
        assert backend._portable_runtimes[0].rows.tolist() == [0, 1, 2, 3]
        capabilities = backend.get_dr_capabilities()
        assert capabilities.supported_fixed_variant_layouts == frozenset(
            {FixedVariantLayout.SAME_LAYOUT, FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT}
        )
        assert capabilities.supports_per_env_playback
        assert set(backend._model.mesh_variant_sets) == {"__unisim_mesh_variant_0"}

        geom = backend._model.get_geom("object/head")
        assert len(backend._model.mesh_variant_sets[geom.mesh_variant_set]) == 3
        np.testing.assert_array_equal(geom.get_mesh_variant_override(backend._data), assignment)

        backend.reset()
        np.testing.assert_array_equal(geom.get_mesh_variant_override(backend._data), assignment)
        backend.step(np.zeros((4, backend.num_actuators), dtype=np.float32), nsteps=3)
        np.testing.assert_array_equal(geom.get_mesh_variant_override(backend._data), assignment)
    finally:
        backend.close()


def test_uniform_mesh_variants_match_single_env_runtimes(tmp_path: Path) -> None:
    batch = MotrixBackend(_scene(tmp_path, (2, 0, 1)), 3, 0.002, base_name="object/base")
    independent: list[MotrixBackend] = []
    try:
        qpos = batch.get_state("qpos")["qpos"].copy()
        qvel = np.zeros_like(batch.get_state("qvel")["qvel"])
        batch.reset()
        ctrl = np.zeros((3, batch.num_actuators), dtype=np.float32)
        batch.step(ctrl, nsteps=10)

        for variant in (2, 0, 1):
            single = MotrixBackend(
                _scene(tmp_path / f"single-{variant}", (variant,)),
                1,
                0.002,
                base_name="object/base",
            )
            independent.append(single)
            single.set_state(np.asarray((0,)), qpos[[variant]], qvel[[variant]])
            single.step(ctrl[[variant]], nsteps=10)

        np.testing.assert_allclose(
            batch.get_state("qpos")["qpos"],
            np.stack([single.get_state("qpos")["qpos"][0] for single in independent]),
            rtol=2e-6,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            batch.get_state("qvel")["qvel"],
            np.stack([single.get_state("qvel")["qvel"][0] for single in independent]),
            rtol=2e-6,
            atol=2e-7,
        )
    finally:
        batch.close()
        for single in independent:
            single.close()


def test_uniform_mesh_variant_optional_slot_fails_closed(tmp_path: Path) -> None:
    present = _mesh_source(tmp_path, "present", scale=".5 .5 .5")
    obj = tmp_path / "tetrahedron.obj"
    obj.write_text(TETRAHEDRON)
    missing_path = tmp_path / "missing.xml"
    missing_path.write_text(
        f"""<mujoco>
  <asset><mesh name="unused" file="{obj}"/></asset>
  <worldbody>
    <body name="base" pos="0 0 .5">
      <freejoint name="root"/>
      <inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
      <geom name="handle" type="sphere" size=".02" mass="0"/>
    </body>
  </worldbody>
</mujoco>
"""
    )
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("object", present, kind="rigid"),),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(
                np.asarray((0, 1), dtype=np.int32),
                (present, ModelSourceDescriptor(str(missing_path))),
                layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="every optional mesh slot"):
        MotrixBackend(scene, 2, 0.002, base_name="object/base")


def test_uniform_mesh_variant_renderer_shows_distinct_native_identities(tmp_path: Path) -> None:
    backend = MotrixBackend(_scene(tmp_path, (0, 1)), 2, 0.002, base_name="object/base")
    try:
        backend.init_renderer(
            spacing=1.0,
            headless=True,
            capture=True,
            width=96,
            height=96,
            camera_kwargs={
                "cam_tracking": True,
                "cam_tracking_env_idx": 0,
                "cam_distance": 0.8,
            },
        )
        assert backend._render_offsets_np is not None
        np.testing.assert_array_equal(backend._render_offsets_np[:, 0], [0.0, 1.0])
        for _ in range(4):
            frame = backend.capture_video_frame()
        assert frame.shape == (96, 96, 3)
        assert np.isfinite(frame).all()
        assert np.count_nonzero(frame != frame.flat[0]) > 32
        geom = backend._model.get_geom("object/head")

        # Change only the tracked row to the other compiled mesh ordinal. A pixel
        # delta demonstrates that per-row override reaches renderer synchronization,
        # rather than only surviving the collision-shape readback API.
        geom.set_shape_override(
            backend._data,
            np.full((2,), 7, dtype=np.int64),
            variant=np.asarray((1, 0), dtype=np.int64),
        )
        for _ in range(4):
            switched_frame = backend.capture_video_frame()
        assert np.count_nonzero(frame != switched_frame) > 32
        geom.set_shape_override(
            backend._data,
            np.asarray((7, 7), dtype=np.int64),
            variant=np.asarray((0, 1), dtype=np.int64),
        )

        np.testing.assert_array_equal(geom.get_mesh_variant_override(backend._data), (0, 1))
    finally:
        backend.close()


def test_uniform_mesh_variant_physical_delta_fails_closed(tmp_path: Path) -> None:
    light = _mesh_source(tmp_path, "light", scale=".5 .5 .5", body_mass="1")
    heavy = _mesh_source(tmp_path, "heavy", scale=".9 .9 .9", body_mass="2")
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("object", light, kind="rigid"),),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(
                np.asarray((0, 1), dtype=np.int32),
                (light, heavy),
                layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="body body_mass"):
        MotrixBackend(scene, 2, 0.002, base_name="object/base")
