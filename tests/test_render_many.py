"""Headless regression tests for multi-env grid offsets in render_many."""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

# Free-joint object as body 1 (apply_root_offset=False) plus a mocap palm
# with an articulated finger, mirroring the Wuji playback model.
WUJI_LIKE_XML = """<mujoco>
  <worldbody>
    <geom name='floor' type='plane' size='5 5 0.1'/>
    <body name='object/cube' pos='0.3 0.1 0.55'>
      <joint name='cube_free' type='free'/>
      <geom name='cube' type='box' size='0.04 0.04 0.04' rgba='0 0 1 1'/>
    </body>
    <body name='robot/palm' mocap='true' pos='0 0 0.5'>
      <geom name='palm' type='box' size='0.05 0.05 0.02' rgba='1 0 0 1'/>
      <body name='finger' pos='0 0 0.05'>
        <joint name='hinge' type='hinge' axis='0 1 0'/>
        <geom name='finger' type='box' size='0.02 0.02 0.02' rgba='0 1 0 1'/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# Fixed-root robot as body 1 (apply_root_offset=True) plus a mocap goal body;
# the base geom uses group 1 so the no-terrain static pass still draws it.
FIXED_ROOT_XML = """<mujoco>
  <worldbody>
    <geom name='floor' type='plane' size='5 5 0.1'/>
    <body name='robot/base' pos='0.1 0.2 0.3'>
      <geom name='base' type='box' size='0.03 0.03 0.03' rgba='0 0 1 1' group='1'/>
      <body name='finger' pos='0 0 0.1'>
        <joint name='hinge' type='hinge' axis='0 1 0'/>
        <geom name='finger' type='box' size='0.02 0.02 0.02' rgba='0 1 0 1'/>
      </body>
    </body>
    <body name='goal' mocap='true' pos='0.5 0.5 0.1'>
      <geom name='goal' type='sphere' size='0.02' rgba='1 0 0 1'/>
    </body>
  </worldbody>
</mujoco>
"""

_SCRIPT = textwrap.dedent(
    """
    import sys
    import numpy as np
    import mujoco
    from unisim.backend.base import DebugPrimitive
    from unisim.visualization import render_many

    model_path, mode = sys.argv[1:3]
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    states = np.zeros((2, 1 + model.nq + model.nv))
    for env in range(2):
        states[env, 1 : 1 + model.nq] = data.qpos
    if mode == "wuji_mocap":
        # Extended snapshot layout: [time, qpos, qvel, mocap_pos, mocap_quat].
        # The recorded mocap pose differs from the model default (0, 0, 0.5),
        # so the palm/finger must render at the recorded pose, not the default.
        nm = model.nmocap
        tail = np.zeros((2, 7 * nm))
        for env in range(2):
            tail[env, : 3 * nm] = np.tile([0.4, -0.3, 0.6], nm)
            tail[env, 3 * nm :] = np.tile([1.0, 0.0, 0.0, 0.0], nm)
        states = np.concatenate([states, tail], axis=1)
    offsets = np.array([[0.0, 0.0], [2.0, 3.0]])
    overlays = None
    if mode == "wuji":
        overlays = [
            None,
            [DebugPrimitive(
                kind="sphere", pos=(0, 0, 0.5), size=(0.05,), rgba=(1.0, 0.0, 1.0, 1.0)
            )],
        ]

    render_many.init_worker(model_path, (160, 120))
    try:
        render_frame_job = render_many.render_frame_job
        render_frame_job((states, offsets, False, 6.0, -30.0, 90.0, None, overlays))
        scene = render_many._worker_ctx["renderer"].scene
        for geom in list(scene.geoms)[: scene.ngeom]:
            print(
                "GEOM",
                *(round(float(c), 4) for c in geom.rgba[:3]),
                *(round(float(p), 4) for p in geom.pos),
            )
    finally:
        render_many._close_worker()
    """
)


def _parse_scene_geoms(stdout: str) -> dict[tuple[float, ...], list[np.ndarray]]:
    geoms: dict[tuple[float, ...], list[np.ndarray]] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 7 and parts[0] == "GEOM":
            rgba = tuple(float(v) for v in parts[1:4])
            geoms.setdefault(rgba, []).append(np.array([float(v) for v in parts[4:7]]))
    for positions in geoms.values():
        positions.sort(key=lambda p: p[0] + p[1])
    return geoms


class TestMultiEnvGridOffsets:
    """Multi-env grid rendering offsets env geometry exactly once."""

    @pytest.fixture(autouse=True)
    def _render(self, tmp_path: Path):
        pytest.importorskip("mujoco")
        from unisim.visualization import render_many

        if not render_many.render_backend_usable():
            pytest.skip("no usable MuJoCo off-screen GL backend on this host")
        self._tmp_path = tmp_path

    def _scene_geoms(
        self, xml: str, mode: str
    ) -> dict[tuple[float, ...], list[np.ndarray]]:
        model_path = self._tmp_path / f"{mode}.xml"
        model_path.write_text(xml)
        result = subprocess.run(
            [sys.executable, "-c", _SCRIPT, str(model_path), mode],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"render subprocess failed:\n{result.stdout}\n{result.stderr}"
        )
        return _parse_scene_geoms(result.stdout)

    def test_mocap_palm_follows_grid_with_freejoint_object(self) -> None:
        geoms = self._scene_geoms(WUJI_LIKE_XML, "wuji")
        # Free-joint cube moved via qpos; mocap palm and its articulated
        # finger must follow the same grid cell exactly once.
        np.testing.assert_allclose(
            geoms[(0.0, 0.0, 1.0)], [[0.3, 0.1, 0.55], [2.3, 3.1, 0.55]], atol=1e-4
        )
        np.testing.assert_allclose(
            geoms[(1.0, 0.0, 0.0)], [[0.0, 0.0, 0.5], [2.0, 3.0, 0.5]], atol=1e-4
        )
        np.testing.assert_allclose(
            geoms[(0.0, 1.0, 0.0)], [[0.0, 0.0, 0.55], [2.0, 3.0, 0.55]], atol=1e-4
        )
        # Overlay primitives (env-local poses) land on the same grid offset
        # as the env geometry: the env-1 marker coincides with the palm.
        np.testing.assert_allclose(geoms[(1.0, 0.0, 1.0)], [[2.0, 3.0, 0.5]], atol=1e-4)

    def test_fixed_root_robot_and_mocap_goal_offset_exactly_once(self) -> None:
        geoms = self._scene_geoms(FIXED_ROOT_XML, "fixedroot")
        np.testing.assert_allclose(
            geoms[(0.0, 0.0, 1.0)], [[0.1, 0.2, 0.3], [2.1, 3.2, 0.3]], atol=1e-4
        )
        np.testing.assert_allclose(
            geoms[(0.0, 1.0, 0.0)], [[0.1, 0.2, 0.4], [2.1, 3.2, 0.4]], atol=1e-4
        )
        # The mocap goal is translated via mocap_pos, not the geom_xpos
        # post-shift; both mechanisms together would double the offset.
        np.testing.assert_allclose(
            geoms[(1.0, 0.0, 0.0)], [[0.5, 0.5, 0.1], [2.5, 3.5, 0.1]], atol=1e-4
        )

    def test_mocap_snapshot_tail_replays_recorded_pose(self) -> None:
        geoms = self._scene_geoms(WUJI_LIKE_XML, "wuji_mocap")
        # The snapshot tail carries mocap (0.4, -0.3, 0.6) + identity quat;
        # the palm and finger must replay it (plus the grid offset for env 1)
        # instead of falling back to the model default (0, 0, 0.5).
        np.testing.assert_allclose(
            geoms[(1.0, 0.0, 0.0)], [[0.4, -0.3, 0.6], [2.4, 2.7, 0.6]], atol=1e-4
        )
        np.testing.assert_allclose(
            geoms[(0.0, 1.0, 0.0)], [[0.4, -0.3, 0.65], [2.4, 2.7, 0.65]], atol=1e-4
        )


class TestGridFitDistance:
    """Free-camera distance auto-fit for multi-env grid recording."""

    @pytest.fixture(autouse=True)
    def _mujoco(self):
        self.mujoco = pytest.importorskip("mujoco")
        from unisim.visualization import render_many

        self.render_many = render_many
        self.model = self.mujoco.MjModel.from_xml_string("<mujoco/>")

    def test_four_env_grid_vertical_fit_dominates(self) -> None:
        offsets = self.render_many.get_grid_offsets(4, spacing=1.0)
        fit = self.render_many._grid_fit_distance(offsets, self.model, (1280, 720))
        # fovy 45°, span 1.0 + margin 0.5 per side -> need_h = 1.0.
        expected = 1.0 / math.tan(math.radians(22.5))
        assert fit == pytest.approx(expected, rel=1e-6)

    def test_single_env_fit_stays_below_closeup_distance(self) -> None:
        offsets = self.render_many.get_grid_offsets(1, spacing=1.0)
        fit = self.render_many._grid_fit_distance(offsets, self.model, (1280, 720))
        assert fit == pytest.approx(0.5 / math.tan(math.radians(22.5)), rel=1e-6)
        assert fit < 2.0  # never pushes a single-env record beyond the default

    def test_cam_fov_widens_fit(self) -> None:
        self.model.vis.global_.fovy = 90.0
        offsets = self.render_many.get_grid_offsets(4, spacing=1.0)
        fit = self.render_many._grid_fit_distance(offsets, self.model, (1280, 720))
        assert fit == pytest.approx(1.0 / math.tan(math.radians(45.0)), rel=1e-6)
