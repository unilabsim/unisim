"""Cold authoring checks and optional native SuperDex numerical probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unisim.backend.superdex.materialization import (
    _actuators,
    _audit_model,
    _geom_pair_allowed,
    _pair_friction,
    materialize_model,
)
from unisim.scene import SceneCfg

MODEL = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <inertial mass="2" pos=".03 .02 0" diaginertia=".03 .04 .05"/>
      <geom name="left" type="box" size=".1 .1 .1" pos="-.15 0 0"/>
      <geom name="right" type="sphere" size=".1" pos=".15 0 0"/>
      <body name="arm" pos=".2 .1 0" quat=".9238795 0 0 .3826834">
        <joint name="hinge" axis="0 0 1" pos=".04 0 0" range="-70 70"
               damping=".2" armature=".01" frictionloss=".03"/>
        <geom type="box" size=".08 .03 .03"/>
        <body name="tip" pos=".15 0 0">
          <joint name="slide" type="slide" axis="1 0 0" range="-.1 .1"/>
          <geom type="sphere" size=".03"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide_motor" joint="slide" gear="2" ctrlrange="-3 3"/>
    <position name="hinge_servo" joint="hinge" kp="20" forcerange="-5 5"/>
  </actuator>
  <sensor>
    <jointpos name="angle" joint="hinge"/>
    <framepos name="body_pos" objtype="xbody" objname="arm"/>
    <contact name="left_contact" geom1="floor" geom2="left" data="found" num="1"/>
    <contact name="right_contact" geom1="floor" geom2="right" data="found" num="1"/>
  </sensor>
  <keyframe><key name="home" qpos="0 0 1 1 0 0 0 .2 .01"/></keyframe>
</mujoco>"""


def test_actuator_order_is_independent_of_joint_order():
    mj = pytest.importorskip("mujoco")
    m = mj.MjModel.from_xml_string(MODEL)
    result = _actuators(mj, m, ("root", "hinge", "slide"), None)
    assert result["actuator_joint_names"] == ("slide", "hinge")
    np.testing.assert_array_equal(result["actuator_qpos_indices"], [8, 7])
    np.testing.assert_array_equal(result["actuator_qvel_indices"], [7, 6])
    np.testing.assert_array_equal(result["actuator_gear"], [2, 1])
    np.testing.assert_array_equal(result["actuator_kp"], [0, 20])
    np.testing.assert_array_equal(result["actuator_force_ranges"][1], [-5, 5])


@pytest.mark.parametrize("floating", [False, True])
def test_welded_parent_contact_filter_matches_mujoco(floating):
    mj = pytest.importorskip("mujoco")
    root = '<freejoint name="root"/>' if floating else ""
    xml = f"""<mujoco><worldbody><body name="base">{root}
      <geom name="a" type="sphere" size=".1"/>
      <body name="spacer"><body name="child">
        <joint name="hinge"/>
        <geom name="b" type="sphere" size=".1" pos=".15 0 0"/>
      </body></body>
    </body></worldbody></mujoco>"""
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    assert bool(data.ncon) is (not floating)
    assert _geom_pair_allowed(model, 0, 1) is bool(data.ncon)


def test_massless_fixed_frame_does_not_gain_carrier_mass(tmp_path: Path, native):
    p, r = native
    source = tmp_path / "spacer.xml"
    source.write_text("""<mujoco><worldbody><body name="base"><freejoint/>
      <geom name="a" type="sphere" size=".1"/>
      <body name="spacer"><body name="child" pos=".3 0 0">
        <joint name="hinge"/>
        <geom name="b" type="sphere" size=".1"/>
      </body></body></body></worldbody>
      <actuator><motor joint="hinge"/></actuator></mujoco>""")
    plan = materialize_model(p, r, SceneCfg(str(source)))
    scene = p.create_scene("massless_frame")
    cleanup = None
    try:
        actor, cleanup = plan.spawn_actor(scene)
        spacer = scene.get_actor(actor.get_nested_link_actors()[plan.body_link_indices[2]])
        assert spacer.get_mass() == 0
        assert plan.body_mass[2] == 0
    finally:
        if cleanup is not None:
            cleanup()
        p.destroy_scene(scene)


def test_audit_rejects_springs_and_multijoint_bodies():
    mj = pytest.importorskip("mujoco")
    spring = MODEL.replace('damping=".2"', 'stiffness="1" damping=".2"')
    with pytest.raises(NotImplementedError, match="springs"):
        _audit_model(mj, mj.MjModel.from_xml_string(spring))
    multiple = MODEL.replace(
        '<joint name="slide"', '<joint name="extra" axis="0 1 0"/><joint name="slide"'
    )
    multiple = multiple[: multiple.index("<keyframe>")] + "</mujoco>"
    with pytest.raises(NotImplementedError, match="one joint"):
        _audit_model(mj, mj.MjModel.from_xml_string(multiple))


def test_non_factorable_pair_friction_uses_public_pair_overrides(tmp_path: Path, native):
    """Preserve an inconsistent friction graph with exact native pair overrides."""
    p, r = native
    source = tmp_path / "pair-friction.xml"
    source.write_text(
        """<mujoco><worldbody>
          <geom name="floor" type="plane" size="2 2 .1" friction="1"/>
          <body name="base" pos="0 0 1">
            <freejoint/>
            <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
            <geom name="base_geom" type="sphere" size=".1" friction=".8"/>
            <body name="left" pos=".3 0 -.5">
              <joint name="left" axis="0 1 0"/>
              <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
              <geom name="left_geom" type="sphere" size=".1" friction=".5"/>
            </body>
            <body name="right" pos="-.3 0 -.5">
              <joint name="right" axis="0 1 0"/>
              <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
              <geom name="right_geom" type="sphere" size=".1" friction=".25"/>
            </body>
          </body>
        </worldbody></mujoco>"""
    )
    plan = materialize_model(p, r, SceneCfg(str(source)), sim_dt=0.002)
    scene = p.create_scene("pair-friction")
    cleanup = None
    try:
        _, cleanup = plan.spawn_actor(scene)
        actors: dict[str, Any] = {}
        scene.for_each_actor(lambda actor: actors.__setitem__(actor.get_name(), actor))

        def override(first: str, second: str) -> float:
            return float(
                scene.get_contact_pair_params_override(
                    actors[first].get_handle(), actors[second].get_handle()
                ).coulomb_friction_coefficient
            )

        assert override("floor", "robot/left") == pytest.approx(1.0)
        assert override("floor", "robot/right") == pytest.approx(1.0)
        assert override("robot/left", "robot/right") == pytest.approx(0.5)
        assert not scene.has_contact_pair_params_override(
            actors["robot/base"].get_handle(), actors["robot/left"].get_handle()
        )
    finally:
        if cleanup is not None:
            cleanup()
        p.destroy_scene(scene)


@pytest.mark.parametrize(
    ("floor_priority", "geom_priority", "geom_condim", "expected"),
    [
        (2, 1, 3, 0.3),
        (1, 2, 3, 0.8),
        (1, 2, 1, 0.0),
    ],
    ids=["first-priority-wins", "second-priority-wins", "high-priority-condim-lt-3"],
)
def test_pair_friction_priority_selects_reviewed_geom(
    floor_priority: int, geom_priority: int, geom_condim: int, expected: float
):
    mj = pytest.importorskip("mujoco")
    xml = f"""<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .1"
            priority="{floor_priority}" friction=".3"/>
      <body name="base" pos="0 0 .1">
        <freejoint/>
        <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
        <geom name="body" type="sphere" size=".1" priority="{geom_priority}"
              condim="{geom_condim}" friction=".8"/>
      </body>
    </worldbody></mujoco>"""
    model = mj.MjModel.from_xml_string(xml)

    assert _pair_friction(model, 0, 1) == pytest.approx(expected)


def test_sensor_cutoff_support_is_narrow_and_fail_closed(tmp_path: Path, native):
    p, r = native
    source = tmp_path / "sensor-cutoff.xml"
    source.write_text("""<mujoco><worldbody>
      <body name="base" pos="0 0 1">
        <freejoint/>
        <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
        <geom name="geom" type="sphere" size=".1"/>
        <site name="site" pos=".1 .2 .3"/>
      </body></worldbody>
      <sensor><framepos name="position" objtype="site" objname="site" cutoff=".2"/></sensor>
    </mujoco>""")
    with pytest.raises(NotImplementedError, match="gyro/velocimeter"):
        materialize_model(p, r, SceneCfg(str(source)))


def test_accelerometer_is_declared_but_unused(tmp_path: Path, native):
    p, r = native
    source = tmp_path / "accelerometer-cutoff.xml"
    source.write_text("""<mujoco><worldbody>
      <body name="base" pos="0 0 1">
        <freejoint/>
        <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
        <geom name="geom" type="sphere" size=".1"/>
        <site name="site" pos=".1 .2 .3"/>
      </body></worldbody>
      <sensor><accelerometer name="acceleration" site="site" cutoff=".2"/></sensor>
    </mujoco>""")
    plan = materialize_model(p, r, SceneCfg(str(source)))
    accelerometer = next(sensor for sensor in plan.sensors if sensor.kind == "accelerometer")
    assert accelerometer.name == "acceleration"
    assert accelerometer.cutoff == pytest.approx(0.2)


@pytest.fixture
def native():
    p = pytest.importorskip("superdex.physics")
    r = pytest.importorskip("superdex.robotics")
    was_initialized = p.is_initialized()
    if not was_initialized:
        p.initialize(0)
    yield p, r
    if not was_initialized:
        p.shutdown()


def test_split_geometry_preserves_inertia_and_kinematics(tmp_path: Path, native):
    mj = pytest.importorskip("mujoco")
    p, r = native
    source = tmp_path / "scene.xml"
    source.write_text(MODEL)
    plan = materialize_model(p, r, SceneCfg(str(source)))
    assert plan.joint_names == ("hinge", "slide")
    assert plan.sensors[0].joint_index == 0
    assert plan.sensors[2].native_link_index != plan.sensors[3].native_link_index
    np.testing.assert_allclose(plan.keyframes["home"][-2:], [0.2, 0.01])
    scene = p.create_scene("materialization_test")
    cleanup = None
    try:
        actor, cleanup = plan.spawn_actor(scene)
        native_links = [scene.get_actor(h) for h in actor.get_nested_link_actors()]
        a = native_links[plan.body_link_indices[1]]
        b = native_links[plan.sensors[3].native_link_index]
        assert a.get_mass() + b.get_mass() == pytest.approx(2)
        np.testing.assert_allclose(a.get_rigid_center_of_mass_local(), [0.03, 0.02, 0])
        np.testing.assert_allclose(
            a.get_rigid_moment_of_inertia_local(), [0.015, 0, 0, 0.02, 0, 0.025], atol=1e-6
        )
        dtype = np.float64 if p.uses_double_precision() else np.float32
        q = np.array([0.1, 0.2, 1, 0.4, -0.2, 0.3, 0.2, 0.01], dtype=dtype)
        actor.set_articulated_pose_from_joints(q)
        m = mj.MjModel.from_xml_path(str(source))
        d = mj.MjData(m)
        d.qpos[:3] = q[:3]
        quat = np.asarray(p.Quaternion.from_rotation_vector(q[3:6]))
        d.qpos[3:7] = quat[[3, 0, 1, 2]]
        d.qpos[7:] = q[6:]
        mj.mj_forward(m, d)
        for body in range(1, m.nbody):
            link = native_links[plan.body_link_indices[body]]
            np.testing.assert_allclose(
                link.get_root_transform().translation, d.xpos[body], atol=2e-6
            )
            got = np.asarray(link.get_root_transform().rotation)[[3, 0, 1, 2]]
            assert abs(np.dot(got, d.xquat[body])) == pytest.approx(1, abs=2e-6)
    finally:
        if cleanup is not None:
            cleanup()
        p.destroy_scene(scene)
        plan.cleanup()


def test_contact_approximation_requires_explicit_opt_in(tmp_path: Path, native):
    pytest.importorskip("mujoco")
    p, r = native
    source = tmp_path / "contact.xml"
    source.write_text(MODEL.replace('name="left" type="box"', 'name="left" condim="6" type="box"'))
    with pytest.raises(NotImplementedError, match="allow_contact_approximation"):
        materialize_model(p, r, SceneCfg(str(source)))


def test_contact_queries_keep_separate_geoms_on_same_body(tmp_path: Path, native):
    pytest.importorskip("mujoco")
    p, r = native
    source = tmp_path / "scene.xml"
    source.write_text(
        MODEL.replace(
            'name="left" type="box"', 'name="left" priority="2" friction=".25" type="box"'
        )
    )
    plan = materialize_model(p, r, SceneCfg(str(source)))
    scene = p.create_scene("separate_contact_geoms")
    cleanup = None
    try:
        actor, cleanup = plan.spawn_actor(scene)
        actors = {}
        scene.for_each_actor(lambda a: actors.setdefault(a.get_name(), a))
        floor_actor = actors["floor"]
        links = [scene.get_actor(h) for h in actor.get_nested_link_actors()]
        left, right = [links[s.native_link_index] for s in plan.sensors[2:]]
        left_override = scene.get_contact_pair_params_override(
            left.get_handle(), floor_actor.get_handle()
        )
        right_override = scene.get_contact_pair_params_override(
            right.get_handle(), floor_actor.get_handle()
        )
        assert left_override.coulomb_friction_coefficient == pytest.approx(0.25)
        assert right_override.coulomb_friction_coefficient == pytest.approx(1.0)
        floor = floor_actor.get_handle()
        left.register_query(p.QueryType.CONTACT_POINTS)
        right.register_query(p.QueryType.CONTACT_POINTS)
        dtype = np.float64 if p.uses_double_precision() else np.float32
        # Tilt around Y: right sphere intersects the plane; left box stays above it.
        actor.set_articulated_pose_from_joints(np.array([0, 0, 0.16, 0, 0.5, 0, 0, 0], dtype=dtype))
        # step(0) refreshes state queries but does not rebuild this contact manifold.
        # A positive solver step is required before reading geometric contact results.
        scene.step(0.0001)

        def has_contact(link):
            own = link.get_handle()
            return any(
                cp.distance <= 0
                and (
                    (cp.actor_a == own and cp.actor_b == floor)
                    or (cp.actor_b == own and cp.actor_a == floor)
                )
                for cp in link.get_contact_points_world()
            )

        assert not has_contact(left)
        assert has_contact(right)
    finally:
        if cleanup is not None:
            cleanup()
        p.destroy_scene(scene)
