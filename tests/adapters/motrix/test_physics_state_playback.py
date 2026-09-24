"""Motrix physics-state playback contract tests (#302)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco

from unisim import MotrixBackend
from unisim.backend.motrix.scene import extract_mjcf_joint_layout
from unisim.conformance import assert_physics_state_playback_conformance
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityVariantBinding, SceneEntitySpec
from unisim.scene import SceneCfg

WHOLE_MODEL = """<mujoco model='unisim-motrix-playback'>
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
  <actuator><motor name='drive' joint='drive'/></actuator>
</mujoco>"""

VARIANT_MODEL = """<mujoco model='unisim-motrix-playback-variant'>
  <option gravity='0 0 -9.81'/>
  <worldbody><body name='base' pos='0 0 1'>
    <freejoint name='root'/>
    <inertial pos='{ipos}' mass='{mass}' diaginertia='.2 .2 .2'/>
    <geom name='base_geom' type='sphere' size='{size}'/>
    <body name='link' pos='0 0 .2'>
      <joint name='drive' type='hinge' axis='0 1 0'/>
      <inertial pos='0 0 0' mass='.2' diaginertia='.03 .03 .03'/>
      <geom name='link_geom' type='sphere' size='.03'/>
    </body>
  </body></worldbody>
  <actuator><motor name='drive' joint='drive'/></actuator>
</mujoco>"""

SIM_DT = 0.002


def _write(tmp_path: Path, name: str, xml: str) -> ModelSourceDescriptor:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{name}.xml"
    path.write_text(xml, encoding="utf-8")
    return ModelSourceDescriptor(str(path))


def _whole_model_backend(tmp_path: Path, num_envs: int = 2) -> MotrixBackend:
    source = _write(tmp_path, "model", WHOLE_MODEL)
    return MotrixBackend(SceneCfg(model_file=source.model_file), num_envs, SIM_DT)


def _portable_scene(tmp_path: Path) -> SceneCfg:
    return SceneCfg(
        entity_assets=(SceneEntitySpec("robot", _write(tmp_path, "robot", WHOLE_MODEL)),)
    )


def _variant_scene(tmp_path: Path) -> SceneCfg:
    light = _write(
        tmp_path, "robot", VARIANT_MODEL.format(ipos="0 0 0", mass="1", size=".08")
    )
    heavy = _write(
        tmp_path / "variant",
        "robot-heavy",
        VARIANT_MODEL.format(ipos="-.04 0 .02", mass="1.7", size=".12"),
    )
    return SceneCfg(
        entity_assets=(SceneEntitySpec("robot", light),),
        entity_variant=EntityVariantBinding(
            "robot",
            FixedVariantPlan(np.asarray((1, 0, 1, 0), dtype=np.int32), (light, heavy)),
        ),
    )


def _zero_ctrl(backend: MotrixBackend) -> np.ndarray:
    return np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32)


def test_extract_mjcf_joint_layout_matches_mujoco(tmp_path: Path) -> None:
    part = tmp_path / "part.xml"
    part.write_text(
        """<mujoco>
  <worldbody>
    <body name="root" pos="0 0 1">
      <freejoint name="root_free"/>
      <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
      <body name="arm1" pos="1 0 0">
        <joint name="j1" type="hinge"/>
        <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
        <frame name="f0">
          <body name="forearm" pos="0.1 0 0">
            <joint name="j2" type="slide"/>
            <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
          </body>
        </frame>
        <joint name="j1b" type="ball"/>
      </body>
      <body name="arm2" pos="-1 0 0">
        <joint name="j3" type="hinge"/>
        <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    main = tmp_path / "main.xml"
    main.write_text('<mujoco>\n  <include file="part.xml"/>\n</mujoco>\n', encoding="utf-8")

    entries = extract_mjcf_joint_layout(str(main))
    model = mujoco.MjModel.from_xml_path(str(main))
    joint_object = mujoco.mjtObj.mjOBJ_JOINT
    assert tuple(entry.name for entry in entries) == tuple(
        mujoco.mj_id2name(model, joint_object, joint_id) for joint_id in range(model.njnt)
    )
    kind_by_type = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    assert tuple(entry.kind for entry in entries) == tuple(
        kind_by_type[int(model.jnt_type[joint_id])] for joint_id in range(model.njnt)
    )
    assert tuple(entry.qpos_address for entry in entries) == tuple(
        int(model.jnt_qposadr[joint_id]) for joint_id in range(model.njnt)
    )
    assert tuple(entry.qvel_address for entry in entries) == tuple(
        int(model.jnt_dofadr[joint_id]) for joint_id in range(model.njnt)
    )
    assert sum(entry.num_dof_pos for entry in entries) == model.nq
    assert sum(entry.num_dof_vel for entry in entries) == model.nv
    free_entry = entries[0]
    assert free_entry.kind == "free" and free_entry.body_name == "root"


def test_advance_playback_time_arithmetic() -> None:
    """Pin the float32 host clock's accumulation semantics without the engine."""
    backend = MotrixBackend.__new__(MotrixBackend)
    backend._time_view = np.zeros(3, dtype=np.float32)
    backend._sim_dt = SIM_DT
    backend._advance_playback_time(3)
    np.testing.assert_allclose(backend._time_view, 3 * SIM_DT, atol=1e-9)
    backend._advance_playback_time(1)
    np.testing.assert_allclose(backend._time_view, 4 * SIM_DT, atol=1e-9)


def test_extract_mjcf_joint_layout_fails_closed(tmp_path: Path) -> None:
    no_worldbody = _write(tmp_path, "empty", "<mujoco/>")
    with pytest.raises(ValueError, match="no <worldbody>"):
        extract_mjcf_joint_layout(no_worldbody.model_file)
    duplicate = _write(
        tmp_path,
        "duplicate",
        """<mujoco><worldbody>
          <body name="a"><joint name="j" type="hinge"/>
            <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
            <body name="b"><joint name="j" type="hinge"/>
              <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
            </body></body>
        </worldbody></mujoco>""",
    )
    with pytest.raises(ValueError, match="unique MJCF joint names"):
        extract_mjcf_joint_layout(duplicate.model_file)


def test_non_portable_layout_and_snapshot_shape(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = _whole_model_backend(tmp_path)
    try:
        layout = backend.get_physics_state_layout()
        assert (layout.nq, layout.nv, layout.nmocap) == (8, 7, 0)
        snapshot = backend.get_physics_state()
        assert snapshot.shape == (backend.num_envs, layout.state_width)
        assert snapshot.shape == (2, 16)
    finally:
        backend.close()


def test_non_portable_time_accumulates_and_zeroes_on_reset(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = _whole_model_backend(tmp_path, num_envs=3)
    try:
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
    finally:
        backend.close()


def test_non_portable_time_counts_pre_step_control_substeps(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = _whole_model_backend(tmp_path)
    try:
        backend.set_pre_step_control(lambda _backend, ctrl: ctrl)
        backend.step(_zero_ctrl(backend), nsteps=5)
        np.testing.assert_allclose(backend.get_physics_state()[:, 0], 5 * SIM_DT, atol=1e-7)
    finally:
        backend.close()


def test_non_portable_snapshot_qpos_is_wxyz(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = _whole_model_backend(tmp_path)
    try:
        layout = backend.get_physics_state_layout()
        # An identity free-base quaternion reads back as wxyz (1, 0, 0, 0); an
        # unconverted xyzw storage would surface as (0, 0, 0, 1).
        parts = layout.split_state(backend.get_physics_state())
        np.testing.assert_allclose(parts.qpos[:, 3:7], (1.0, 0.0, 0.0, 0.0), atol=1e-7)
        quat = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        qpos = np.broadcast_to(
            backend.get_default_qpos(), (backend.num_envs, layout.nq)
        ).copy()
        qpos[:, 3:7] = quat
        backend.set_state(
            np.arange(backend.num_envs, dtype=np.intp),
            qpos,
            np.zeros((backend.num_envs, layout.nv), dtype=np.float32),
        )
        parts = layout.split_state(backend.get_physics_state())
        np.testing.assert_allclose(parts.qpos[:, 3:7], quat, atol=1e-7)
    finally:
        backend.close()


def test_non_portable_playback_conformance(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = _whole_model_backend(tmp_path)
    try:
        backend.step(_zero_ctrl(backend), nsteps=5)
        assert_physics_state_playback_conformance(backend)
    finally:
        backend.close()


def test_non_portable_playback_model_is_source_mjcf(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = _whole_model_backend(tmp_path)
    try:
        layout = backend.get_physics_state_layout()
        model_file = backend.get_playback_model()
        assert model_file == str(tmp_path / "model.xml")
        model = mujoco.MjModel.from_xml_path(model_file)
        assert (model.nq, model.nv) == (layout.nq, layout.nv)
        assert backend.get_playback_model(1) == model_file
        with pytest.raises(IndexError, match="env_index"):
            backend.get_playback_model(2)
    finally:
        backend.close()


def test_non_portable_set_physics_state_fails_closed_on_shape(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = _whole_model_backend(tmp_path)
    try:
        layout = backend.get_physics_state_layout()
        with pytest.raises(ValueError, match="shape"):
            backend.set_physics_state(np.zeros((backend.num_envs, layout.state_width + 1)))
    finally:
        backend.close()


def test_portable_layout_snapshot_time_and_reset(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = MotrixBackend(_portable_scene(tmp_path), 3, SIM_DT, base_name="robot/base")
    try:
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
    finally:
        backend.close()


def test_portable_snapshot_qpos_is_wxyz_and_playback_model(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = MotrixBackend(_portable_scene(tmp_path), 2, SIM_DT, base_name="robot/base")
    try:
        layout = backend.get_physics_state_layout()
        parts = layout.split_state(backend.get_physics_state())
        np.testing.assert_allclose(parts.qpos[:, 3:7], (1.0, 0.0, 0.0, 0.0), atol=1e-7)
        model_file = backend.get_playback_model()
        assert model_file == backend.get_scene_model_file()
        model = mujoco.MjModel.from_xml_path(model_file)
        assert (model.nq, model.nv) == (layout.nq, layout.nv)
        assert backend.get_playback_model(1) == model_file
        with pytest.raises(IndexError, match="env_index"):
            backend.get_playback_model(2)
    finally:
        backend.close()


def test_portable_playback_conformance(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = MotrixBackend(_portable_scene(tmp_path), 2, SIM_DT, base_name="robot/base")
    try:
        backend.step(_zero_ctrl(backend), nsteps=5)
        assert_physics_state_playback_conformance(backend)
    finally:
        backend.close()


def test_fixed_variant_playback_model_requires_env_index(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = MotrixBackend(_variant_scene(tmp_path), 4, SIM_DT, base_name="robot/base")
    try:
        with pytest.raises(ValueError, match="explicit env_index"):
            backend.get_playback_model()
        heavy_file = backend.get_playback_model(0)
        light_file = backend.get_playback_model(1)
        assert heavy_file != light_file
        # assignment is (1, 0, 1, 0): rows 0/2 replay the heavy variant.
        assert backend.get_playback_model(2) == heavy_file
        assert backend.get_playback_model(3) == light_file
        layout = backend.get_physics_state_layout()
        for path in (heavy_file, light_file):
            model = mujoco.MjModel.from_xml_path(path)
            assert (model.nq, model.nv) == (layout.nq, layout.nv)
        with pytest.raises(IndexError, match="env_index"):
            backend.get_playback_model(4)
    finally:
        backend.close()


def test_fixed_variant_playback_conformance(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = MotrixBackend(_variant_scene(tmp_path), 4, SIM_DT, base_name="robot/base")
    try:
        backend.step(_zero_ctrl(backend), nsteps=3)
        assert_physics_state_playback_conformance(backend)
    finally:
        backend.close()


def test_play_capabilities_declare_physics_state_playback(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    backend = _whole_model_backend(tmp_path)
    try:
        capabilities = backend.get_play_capabilities()
        assert capabilities.supports_native_interactive_renderer
        assert capabilities.supports_native_video_capture
        assert capabilities.supports_physics_state_playback
        assert not capabilities.supports_mocap_playback
    finally:
        backend.close()
