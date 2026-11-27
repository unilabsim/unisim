"""Unit tests for generated-terrain materialization on the mjwarp backend."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.mjwarp.materialization import materialize_scene
from unisim.scene import SceneCfg, TerrainSceneCfg
from unisim.terrain.generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainHeightField,
    TerrainOutput,
)

_SIMPLE_ROBOT_XML = """\
<mujoco model="simple_robot">
  <option timestep="0.005">
    <flag multiccd="disable" nativeccd="disable"/>
  </option>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <freejoint/>
      <geom name="base_geom" type="sphere" size="0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _flat_terrain() -> SubTerrainCfg:
    class FlatTerrain(SubTerrainCfg):
        horizontal_scale: float = 0.2
        vertical_scale: float = 0.005

        def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
            del difficulty, rng
            rows = int(round(self.size[0] / self.horizontal_scale))
            cols = int(round(self.size[1] / self.horizontal_scale))
            noise = np.zeros((rows, cols), dtype=np.int16)
            heightfield = TerrainHeightField(
                noise, self.size, self.horizontal_scale, self.vertical_scale, 0, 0, 0, 0.2, 0.0
            )
            origin = np.asarray((self.size[0] * 0.5, self.size[1] * 0.5, 0.0))
            return TerrainOutput(origin, heightfield)

    return FlatTerrain()


def _make_terrain_cfg() -> TerrainSceneCfg:
    return TerrainSceneCfg(
        generator=TerrainGeneratorCfg(
            seed=42,
            curriculum=True,
            size=(4.0, 4.0),
            horizontal_scale=0.2,
            vertical_scale=0.005,
            border_width=2.0,
            num_rows=2,
            num_cols=1,
            sub_terrains={"flat": _flat_terrain()},
        )
    )


def test_mjwarp_materialize_generated_terrain_and_cleanup(tmp_path: Path) -> None:
    # The materialization writes the hfield PNG through imageio, which no extra
    # declares: only the ``viz`` extra installs it, through Viser.
    pytest.importorskip("imageio")
    robot_xml = tmp_path / "robot.xml"
    robot_xml.write_text(_SIMPLE_ROBOT_XML, encoding="utf-8")

    scene = SceneCfg(model_file=str(robot_xml), terrain=_make_terrain_cfg())
    ctx = materialize_scene(scene, add_body_sensors=True, tracked_body_names=("base",))
    try:
        assert Path(ctx.source_model_file).is_file()
        assert ctx.visual_model_file is not None and Path(ctx.visual_model_file).is_file()
        assert ctx.terrain_origins is not None
        assert ctx.terrain_origins.shape == (2, 1, 3)
        assert np.isfinite(ctx.terrain_origins).all()
        assert ctx.terrain_surface_sampler is not None
        sampled = ctx.terrain_surface_sampler.sample_height(
            np.array([[0.0, 0.0]], dtype=np.float32)
        )
        assert np.isfinite(sampled).all()
        assert ctx.tracked_body_names == ("base",)
        assert ctx.cleanup_handle is not None
    finally:
        assert ctx.cleanup_handle is not None
        ctx.cleanup_handle.cleanup()

    assert not Path(ctx.source_model_file).exists()
    assert not Path(ctx.visual_model_file).exists()


def test_mjwarp_materialize_terrain_rejects_missing_generator(tmp_path: Path) -> None:
    robot_xml = tmp_path / "robot.xml"
    robot_xml.write_text(_SIMPLE_ROBOT_XML, encoding="utf-8")

    scene = SceneCfg(model_file=str(robot_xml), terrain=TerrainSceneCfg(generator=None))
    with pytest.raises(ValueError, match="SceneCfg.terrain.generator"):
        materialize_scene(scene)


def test_mjwarp_materialize_rejects_fixed_variant_plan_with_terrain(tmp_path: Path) -> None:
    robot_xml = tmp_path / "robot.xml"
    robot_xml.write_text(_SIMPLE_ROBOT_XML, encoding="utf-8")

    fake_plan = SimpleNamespace(variants=(SimpleNamespace(model_file=str(robot_xml)),))
    scene = SceneCfg(
        model_file=str(robot_xml),
        terrain=_make_terrain_cfg(),
        fixed_variant_plan=fake_plan,  # type: ignore[arg-type]
    )
    with pytest.raises(NotImplementedError, match="fixed_variant_plan"):
        materialize_scene(scene)
