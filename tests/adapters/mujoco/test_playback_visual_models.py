"""Regression coverage for renderable MuJoCo playback models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mjbatch")

import mujoco  # noqa: E402

from unisim import MuJoCoBackend  # noqa: E402
from unisim.backend.mujoco.playback import (  # noqa: E402
    resolve_render_play_model_files,
)
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor  # noqa: E402
from unisim.scene import SceneCfg  # noqa: E402

_CUBE_OBJ = """\
v -1 -1 -1
v 1 -1 -1
v 1 1 -1
v -1 1 -1
v -1 -1 1
v 1 -1 1
v 1 1 1
v -1 1 1
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""


def _write_visual_model(path: Path, *, mass: str) -> None:
    path.with_suffix(".obj").write_text(_CUBE_OBJ, encoding="utf-8")
    path.write_text(
        f"""
<mujoco>
  <asset>
    <mesh name="visual_mesh" file="{path.with_suffix(".obj").name}" scale="0.1 0.1 0.1"/>
  </asset>
  <worldbody>
    <body name="base" pos="0 0 0.2">
      <freejoint name="root"/>
      <geom name="collision" type="sphere" size="0.08" mass="{mass}"/>
      <geom name="visual" type="mesh" mesh="visual_mesh" contype="0" conaffinity="0"
            mass="0" group="2"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )


class _PlaybackEnv:
    def __init__(self, backend: MuJoCoBackend) -> None:
        self._backend = backend

    def get_playback_model(self, env_index: int):
        return self._backend.get_playback_model(env_index)


def test_static_playback_preserves_visual_meshes(tmp_path: Path) -> None:
    model_path = tmp_path / "model.xml"
    _write_visual_model(model_path, mass="1")
    full_model = mujoco.MjModel.from_xml_path(str(model_path))
    backend = MuJoCoBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.002)

    assert backend.model.ngeom == full_model.ngeom - 1
    assert backend.model.nmesh == 0
    assert not backend.get_dr_capabilities().supports_per_env_playback
    assert resolve_render_play_model_files(
        _PlaybackEnv(backend), num_envs=2, tmp_dir=tmp_path / "render"
    ) == str(model_path)

    playback_model = backend.get_playback_model(0)
    assert playback_model.ngeom == full_model.ngeom
    assert playback_model.nmesh == full_model.nmesh
    assert backend.get_playback_model().nmesh == full_model.nmesh
    assert backend.get_playback_model(1) is playback_model


def test_fixed_variant_playback_preserves_visual_meshes(tmp_path: Path) -> None:
    model_paths = (tmp_path / "variant-0.xml", tmp_path / "variant-1.xml")
    for model_path, mass in zip(model_paths, ("1", "2"), strict=True):
        _write_visual_model(model_path, mass=mass)
    plan = FixedVariantPlan(
        np.array([0, 1], dtype=np.int32),
        tuple(ModelSourceDescriptor(str(path)) for path in model_paths),
    )
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(model_paths[0]), fixed_variant_plan=plan),
        num_envs=2,
        sim_dt=0.002,
    )

    assert backend.model.ngeom == 1
    assert backend.model.nmesh == 0
    assert backend.get_dr_capabilities().supports_per_env_playback
    assert backend.get_playback_model(0).ngeom == 2
    assert backend.get_playback_model(0).nmesh == 1
    assert backend.get_playback_model(1).ngeom == 2
    assert backend.get_playback_model(1).nmesh == 1

    output_dir = tmp_path / "render"
    output_dir.mkdir()
    model_files = resolve_render_play_model_files(
        _PlaybackEnv(backend), num_envs=2, tmp_dir=output_dir
    )
    assert [Path(model_file).suffix for model_file in model_files] == [".mjb", ".mjb"]
    for model_file in model_files:
        rendered_model = mujoco.MjModel.from_binary_path(model_file)
        assert rendered_model.ngeom == 2
        assert rendered_model.nmesh == 1
