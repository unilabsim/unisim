"""Genesis physics-state playback contract tests (#310).

Real-engine coverage for the playback contract surface; construction leaves
the process-wide Genesis session alive (multi-scene coexistence), matching the
other real-Genesis suites — ``close()`` would forbid any later session.
"""

# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("genesis")
pytest.importorskip("torch")

import mujoco

from unisim.backend.genesis.backend import GenesisBackend
from unisim.conformance import assert_physics_state_playback_conformance
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
from unisim.scene import SceneCfg

WHOLE_MODEL = """<mujoco model='unisim-genesis-playback'>
  <option gravity='0 0 -9.81'/>
  <worldbody>
    <geom name='floor' type='plane' size='1 1 0.1'/>
    <body name='base' pos='0 0 1'>
    <freejoint name='root'/>
    <inertial pos='0 0 0' mass='1' diaginertia='.2 .2 .2'/>
    <geom name='base_geom' type='sphere' size='.08'/>
    <body name='link' pos='0 0 .2'>
      <joint name='drive' type='hinge' axis='0 1 0'/>
      <inertial pos='0 0 0' mass='.2' diaginertia='.03 .03 .03'/>
      <geom name='link_geom' type='sphere' size='.03'/>
    </body>
  </body></worldbody>
  <actuator><position name='drive' joint='drive' kp='24' kv='3'/></actuator>
</mujoco>"""

ENTITY_MODEL = """<mujoco model='unisim-genesis-playback-entity'>
  <option gravity='0 0 -9.81'/>
  <worldbody><body name='base' pos='0 0 1'>
    <freejoint name='root'/>
    <inertial pos='0 0 0' mass='1' diaginertia='.2 .2 .2'/>
    <geom name='base_geom' type='sphere' size='.08'/>
    <body name='link' pos='0 0 .2'>
      <joint name='drive' type='hinge' axis='0 1 0'/>
      <inertial pos='0 0 0' mass='.2' diaginertia='.03 .03 .03'/>
      <geom name='link_geom' type='sphere' size='.03'/>
    </body>
  </body></worldbody>
  <actuator><position name='drive' joint='drive' kp='24' kv='3'/></actuator>
</mujoco>"""

FIXED_ROBOT = """<mujoco model='unisim-genesis-playback-fixed'>
  <worldbody><body name='base'>
    <inertial pos='0 0 0' mass='1' diaginertia='.2 .2 .2'/>
    <geom name='base_geom' type='sphere' size='.08'/>
    <body name='link' pos='0 0 .2'>
      <joint name='drive' type='hinge' axis='0 1 0'/>
      <inertial pos='0 0 0' mass='.2' diaginertia='.03 .03 .03'/>
      <geom name='link_geom' type='sphere' size='.03'/>
    </body>
  </body></worldbody>
  <actuator><position name='drive' joint='drive' kp='24' kv='3'/></actuator>
</mujoco>"""

OBJECT_VARIANT = """<mujoco model='unisim-genesis-playback-object'>
  <worldbody><body name='base'>
    <freejoint name='root'/>
    <inertial pos='0 0 0' mass='{mass}' diaginertia='.1 .1 .1'/>
    <geom name='base_geom' type='sphere' size='{size}'/>
  </body></worldbody>
</mujoco>"""

SIM_DT = 0.002


def _write(tmp_path: Path, name: str, xml: str) -> ModelSourceDescriptor:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{name}.xml"
    path.write_text(xml, encoding="utf-8")
    return ModelSourceDescriptor(str(path))


def _whole_model_backend(tmp_path: Path, num_envs: int = 2) -> GenesisBackend:
    source = _write(tmp_path, "model", WHOLE_MODEL)
    return GenesisBackend(SceneCfg(model_file=source.model_file), num_envs, SIM_DT)


def _portable_scene(tmp_path: Path) -> SceneCfg:
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                _write(tmp_path, "robot", ENTITY_MODEL),
                initial_state=EntityInitialState((0.0, 0.0, 1.0)),
            ),
        )
    )


def _variant_scene(tmp_path: Path, num_envs: int = 4) -> SceneCfg:
    light = _write(tmp_path, "object", OBJECT_VARIANT.format(mass="0.5", size=".08"))
    heavy = _write(
        tmp_path / "variant",
        "object-heavy",
        OBJECT_VARIANT.format(mass="1.5", size=".12"),
    )
    # Genesis 1.3.3 dispatches variants in contiguous balanced blocks, so the
    # assignment must equal the native balanced mapping.
    assignment = np.zeros((num_envs,), dtype=np.int32)
    assignment[num_envs // 2 :] = 1
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                _write(tmp_path, "robot", FIXED_ROBOT),
                root_mode="fixed",
                initial_state=EntityInitialState((0.0, 0.0, 1.0)),
            ),
            SceneEntitySpec(
                "object",
                light,
                kind="rigid",
                initial_state=EntityInitialState((1.0, 0.0, 1.0)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(assignment, (light, heavy)),
        ),
    )


def _zero_ctrl(backend: GenesisBackend) -> np.ndarray:
    return np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32)


def test_non_portable_layout_and_snapshot_shape(tmp_path: Path) -> None:
    backend = _whole_model_backend(tmp_path)
    layout = backend.get_physics_state_layout()
    assert (layout.nq, layout.nv, layout.nmocap) == (8, 7, 0)
    snapshot = backend.get_physics_state()
    assert snapshot.shape == (backend.num_envs, layout.state_width)
    assert snapshot.shape == (2, 16)


def test_non_portable_time_accumulates_and_zeroes_on_reset(tmp_path: Path) -> None:
    backend = _whole_model_backend(tmp_path, num_envs=3)
    np.testing.assert_array_equal(backend.get_physics_state()[:, 0], 0.0)
    ctrl = _zero_ctrl(backend)
    backend.step(ctrl, nsteps=3)
    np.testing.assert_allclose(backend.get_physics_state()[:, 0], 3 * SIM_DT, atol=1e-7)
    backend.step(ctrl)
    np.testing.assert_allclose(backend.get_physics_state()[:, 0], 4 * SIM_DT, atol=1e-7)
    backend.reset(np.asarray([1]))
    times = backend.get_physics_state()[:, 0]
    assert times[1] == 0.0
    np.testing.assert_allclose(times[[0, 2]], 4 * SIM_DT, atol=1e-7)
    backend.reset()
    np.testing.assert_array_equal(backend.get_physics_state()[:, 0], 0.0)


def test_non_portable_time_counts_pre_step_control_substeps(tmp_path: Path) -> None:
    backend = _whole_model_backend(tmp_path)
    backend.set_pre_step_control(lambda _backend, ctrl: ctrl)
    backend.step(_zero_ctrl(backend), nsteps=5)
    np.testing.assert_allclose(backend.get_physics_state()[:, 0], 5 * SIM_DT, atol=1e-7)


def test_non_portable_snapshot_qpos_is_wxyz(tmp_path: Path) -> None:
    backend = _whole_model_backend(tmp_path)
    layout = backend.get_physics_state_layout()
    # An identity free-base quaternion reads back as wxyz (1, 0, 0, 0); an
    # xyzw storage would surface as (0, 0, 0, 1).
    parts = layout.split_state(backend.get_physics_state())
    np.testing.assert_allclose(
        parts.qpos[:, 3:7],
        np.tile((1.0, 0.0, 0.0, 0.0), (backend.num_envs, 1)),
        atol=1e-7,
    )
    quat = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    qpos = np.broadcast_to(backend.get_default_qpos(), (backend.num_envs, layout.nq)).copy()
    qpos[:, 3:7] = quat
    backend.set_state(
        np.arange(backend.num_envs, dtype=np.intp),
        qpos,
        np.zeros((backend.num_envs, layout.nv), dtype=np.float32),
    )
    parts = layout.split_state(backend.get_physics_state())
    np.testing.assert_allclose(
        parts.qpos[:, 3:7], np.broadcast_to(quat, parts.qpos[:, 3:7].shape), atol=1e-7
    )


def test_non_portable_playback_conformance(tmp_path: Path) -> None:
    backend = _whole_model_backend(tmp_path)
    backend.step(_zero_ctrl(backend), nsteps=5)
    assert_physics_state_playback_conformance(backend)


def test_non_portable_playback_model_is_source_mjcf(tmp_path: Path) -> None:
    backend = _whole_model_backend(tmp_path)
    layout = backend.get_physics_state_layout()
    model_file = backend.get_playback_model()
    assert model_file == str(tmp_path / "model.xml")
    model = mujoco.MjModel.from_xml_path(model_file)
    assert (model.nq, model.nv) == (layout.nq, layout.nv)
    assert backend.get_playback_model(1) == model_file
    with pytest.raises(IndexError, match="env_index"):
        backend.get_playback_model(2)


def test_non_portable_set_physics_state_fails_closed_on_shape(tmp_path: Path) -> None:
    backend = _whole_model_backend(tmp_path)
    layout = backend.get_physics_state_layout()
    with pytest.raises(ValueError, match="shape"):
        backend.set_physics_state(np.zeros((backend.num_envs, layout.state_width + 1)))


def test_joint_order_validation_covers_multi_root_and_ball_joints(tmp_path: Path) -> None:
    """The construction-time validation inventories every joint kind.

    A whole-MJCF scene with two free roots plus hinge, slide, and ball joints
    must pass validation and expose a snapshot layout identical to MuJoCo's
    compiled generalized state.
    """
    scene = _write(
        tmp_path,
        "scene",
        """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="1" diaginertia=".2 .2 .2"/>
      <geom name="base_geom" type="sphere" size=".08"/>
      <body name="link" pos="0 0 .2">
        <joint name="drive" type="hinge" axis="0 1 0"/>
        <joint name="slider" type="slide" axis="1 0 0"/>
        <joint name="knuckle" type="ball"/>
        <inertial pos="0 0 0" mass=".2" diaginertia=".03 .03 .03"/>
        <geom name="link_geom" type="sphere" size=".03"/>
      </body>
    </body>
    <body name="ball" pos="1 0 .3">
      <freejoint name="ball_joint"/>
      <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
      <geom name="ball_geom" type="sphere" size=".03"/>
    </body>
  </worldbody>
  <actuator><position name="drive" joint="drive" kp="24" kv="3"/></actuator>
</mujoco>""",
    )

    backend = GenesisBackend(SceneCfg(model_file=scene.model_file), 2, SIM_DT)
    layout = backend.get_physics_state_layout()
    model = mujoco.MjModel.from_xml_path(scene.model_file)
    assert (layout.nq, layout.nv) == (int(model.nq), int(model.nv))
    snapshot = backend.get_physics_state()
    assert snapshot.shape == (2, layout.state_width)
    playback = mujoco.MjModel.from_xml_path(backend.get_playback_model())
    assert (playback.nq, playback.nv) == (layout.nq, layout.nv)


def test_include_split_articulations_fail_closed(tmp_path: Path) -> None:
    """Articulations split across include worldbodies fail closed at build time.

    Genesis orders native links with the main document's own worldbody first,
    while MuJoCo merges every included worldbody in document order; a scene
    whose articulated bodies span both would silently misalign snapshot
    columns against the playback model, so the adapter's MJCF import audit
    rejects it instead of replaying a wrong layout.
    """
    ball = tmp_path / "ball.xml"
    ball.write_text(
        """<mujoco>
  <worldbody>
    <body name="ball" pos="0 0 0.3">
      <freejoint name="ball_joint"/>
      <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
      <geom name="ball_geom" type="sphere" size="0.03"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    main = tmp_path / "scene.xml"
    main.write_text(
        '<mujoco>\n  <include file="ball.xml"/>\n'
        "  <worldbody>\n    <geom name='floor' type='plane' size='1 1 0.1'/>\n"
        "    <body name='base' pos='1 0 1'>\n"
        "    <freejoint name='root'/>\n"
        "    <inertial pos='0 0 0' mass='1' diaginertia='.2 .2 .2'/>\n"
        "    <geom name='base_geom' type='sphere' size='.08'/>\n"
        "    <body name='link' pos='0 0 .2'>\n"
        "      <joint name='drive' type='hinge' axis='0 1 0'/>\n"
        "      <inertial pos='0 0 0' mass='.2' diaginertia='.03 .03 .03'/>\n"
        "      <geom name='link_geom' type='sphere' size='.03'/>\n"
        "    </body></body>\n  </worldbody>\n"
        "  <actuator><position name='drive' joint='drive' kp='24' kv='3'/></actuator>\n"
        "</mujoco>\n",
        encoding="utf-8",
    )

    backend = GenesisBackend(SceneCfg(model_file=str(main)), 2, SIM_DT)
    with pytest.raises(RuntimeError, match="MJCF import mismatch"):
        backend.materialize()


def test_portable_layout_snapshot_time_and_reset(tmp_path: Path) -> None:
    backend = GenesisBackend(_portable_scene(tmp_path), 3, SIM_DT)
    layout = backend.get_physics_state_layout()
    assert (layout.nq, layout.nv, layout.nmocap) == (8, 7, 0)
    snapshot = backend.get_physics_state()
    assert snapshot.shape == (3, layout.state_width)
    np.testing.assert_array_equal(snapshot[:, 0], 0.0)
    ctrl = _zero_ctrl(backend)
    backend.step(ctrl, nsteps=4)
    np.testing.assert_allclose(backend.get_physics_state()[:, 0], 4 * SIM_DT, atol=1e-7)
    backend.reset(np.asarray([2]))
    times = backend.get_physics_state()[:, 0]
    assert times[2] == 0.0
    np.testing.assert_allclose(times[[0, 1]], 4 * SIM_DT, atol=1e-7)


def test_portable_snapshot_qpos_is_wxyz_and_playback_model(tmp_path: Path) -> None:
    backend = GenesisBackend(_portable_scene(tmp_path), 2, SIM_DT)
    layout = backend.get_physics_state_layout()
    parts = layout.split_state(backend.get_physics_state())
    np.testing.assert_allclose(
        parts.qpos[:, 3:7],
        np.tile((1.0, 0.0, 0.0, 0.0), (backend.num_envs, 1)),
        atol=1e-7,
    )
    model_file = backend.get_playback_model()
    assert model_file == backend.get_scene_model_file()
    model = mujoco.MjModel.from_xml_path(model_file)
    assert (model.nq, model.nv) == (layout.nq, layout.nv)
    assert backend.get_playback_model(1) == model_file
    with pytest.raises(IndexError, match="env_index"):
        backend.get_playback_model(2)


def test_portable_playback_conformance(tmp_path: Path) -> None:
    backend = GenesisBackend(_portable_scene(tmp_path), 2, SIM_DT)
    backend.step(_zero_ctrl(backend), nsteps=5)
    assert_physics_state_playback_conformance(backend)


def test_fixed_variant_playback_model_requires_env_index(tmp_path: Path) -> None:
    backend = GenesisBackend(_variant_scene(tmp_path), 4, SIM_DT)
    with pytest.raises(ValueError, match="explicit env_index"):
        backend.get_playback_model()
    light_file = backend.get_playback_model(0)
    heavy_file = backend.get_playback_model(2)
    assert light_file != heavy_file
    # Balanced assignment is (0, 0, 1, 1): rows 0/1 replay the light variant.
    assert backend.get_playback_model(1) == light_file
    assert backend.get_playback_model(3) == heavy_file
    layout = backend.get_physics_state_layout()
    for path in (light_file, heavy_file):
        model = mujoco.MjModel.from_xml_path(path)
        assert (model.nq, model.nv) == (layout.nq, layout.nv)
    with pytest.raises(IndexError, match="env_index"):
        backend.get_playback_model(4)


def test_fixed_variant_playback_conformance(tmp_path: Path) -> None:
    backend = GenesisBackend(_variant_scene(tmp_path), 4, SIM_DT)
    backend.step(_zero_ctrl(backend), nsteps=3)
    assert_physics_state_playback_conformance(backend)


def test_play_capabilities_declare_physics_state_playback(tmp_path: Path) -> None:
    backend = _whole_model_backend(tmp_path)
    capabilities = backend.get_play_capabilities()
    assert capabilities.supports_native_interactive_renderer
    assert capabilities.supports_native_video_capture
    assert capabilities.supports_physics_state_playback
    assert not capabilities.supports_mocap_playback
