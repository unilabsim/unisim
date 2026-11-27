"""Real CUDA checks for fixed-base scenes: root layouts, force sensors and gravcomp."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from unisim import create_backend
from unisim.backend.motrix_gpu.dependencies import motrix_gpu_available
from unisim.scene import SceneCfg

FIXED_BASE_MODEL = """
<mujoco model="motrix_gpu_fixed_base">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <body name="arm" pos="0 0 0">
        <joint name="shoulder" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="0.2" diaginertia="0.01 0.01 0.01"/>
        <geom type="box" size="0.1 0.05 0.05"/>
      </body>
    </body>
    <body name="object" pos="0.5 0 0.5">
      <freejoint name="object_root"/>
      <inertial pos="0 0 0" mass="0.3" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
  <actuator><position name="drive" joint="shoulder" kp="10" kv="1"/></actuator>
  <sensor>
    <actuatorfrc name="drive_force" actuator="drive"/>
    <jointactuatorfrc name="joint_force" joint="shoulder"/>
  </sensor>
</mujoco>
"""

RESTING_BOX_MODEL = """
<mujoco model="motrix_gpu_resting_box">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="floor" pos="0 0 0">
      <geom name="floor_geom" type="box" size="0.5 0.5 0.05" pos="0 0 -0.05"/>
    </body>
    <body name="box" pos="0 0 0.05">
      <freejoint name="box_root"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
  <sensor>
    <contact name="body_net" body1="box" body2="floor" data="force" reduce="netforce"/>
    <contact name="geom_net" geom1="box_geom" geom2="floor_geom" data="force"
            reduce="netforce"/>
  </sensor>
</mujoco>
"""

GRAVCOMP_MODEL = """
<mujoco model="motrix_gpu_gravcomp">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <body name="arm" pos="0 0 0" gravcomp="{gravcomp}">
        <joint name="shoulder" type="hinge" axis="0 1 0"/>
        <inertial pos="0.2 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="drive" joint="shoulder" kp="2" kv="1"/></actuator>
  <sensor><jointpos name="angle" joint="shoulder"/></sensor>
</mujoco>
"""


def _gpu_backend(tmp_path, model: str, num_envs: int = 2, base_name: str = "base", **options):
    torch = pytest.importorskip("torch")
    if not motrix_gpu_available() or not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("Motrix GPU requires its optional NVIDIA CUDA runtime")
    path = tmp_path / "model.xml"
    path.write_text(model)
    backend = create_backend(
        "motrix_gpu",
        scene=SceneCfg(model_file=str(path)),
        num_envs=num_envs,
        sim_dt=0.002,
        base_name=base_name,
        max_iterations=20,
        device="cuda",
        **options,
    )
    return backend, torch


@pytest.fixture
def fixed_base(tmp_path):
    backend, torch = _gpu_backend(tmp_path, FIXED_BASE_MODEL)
    try:
        yield backend, torch
    finally:
        backend.close()


def test_fixed_base_root_layout_addresses_the_free_object(fixed_base):
    backend, _ = fixed_base
    layout = backend.get_root_state_layout("object")
    assert tuple(layout.qpos_indices) == tuple(range(1, 8))
    assert tuple(layout.qvel_indices) == tuple(range(1, 7))
    with pytest.raises(NotImplementedError, match="fixed"):
        backend.get_root_state_layout("base")
    np.testing.assert_array_equal(backend.get_joint_state_qpos_indices(("shoulder",)), [0])
    np.testing.assert_array_equal(backend.get_joint_dof_pos_indices(("shoulder",)), [0])
    np.testing.assert_array_equal(backend.get_joint_dof_vel_indices(("shoulder",)), [0])
    assert backend.get_default_dof_pos().shape == (1,)


def test_actuator_force_sensors_follow_the_affine_servo_law(fixed_base):
    backend, torch = fixed_base
    state = backend.get_state_views(device="cuda")
    ctrl = torch.full((2, 1), 0.3, device="cuda")
    backend.step_tensor(ctrl, 3)
    q = state["qpos"][:, 0]
    qd = state["qvel"][:, 0]
    expected = 10.0 * (ctrl[:, 0] - q) - 1.0 * qd
    torch.testing.assert_close(backend.get_sensor_view("drive_force", "cuda")[:, 0], expected)
    torch.testing.assert_close(
        backend.get_sensor_view("drive_force", "cuda"),
        backend.get_sensor_view("joint_force", "cuda"),
    )
    assert "drive_force" in backend.get_sensor_names()


@pytest.mark.parametrize("capacity", [{}, {"max_contacts": 512, "max_contact_constraints": 1024}])
def test_netforce_contact_sensors_report_the_resting_weight(tmp_path, capacity):
    # The runtime lists constraint rows in a different order at each capacity;
    # the sensors must not depend on it.
    backend, torch = _gpu_backend(tmp_path, RESTING_BOX_MODEL, 1, base_name="box", **capacity)
    try:
        backend.step_tensor(torch.zeros((1, 0), device="cuda"), 240)
        for name in ("body_net", "geom_net"):
            view = backend.get_sensor_view(name, "cuda")
            assert view.shape == (1, 3)
            np.testing.assert_allclose(view[0].cpu().numpy(), [0.0, 0.0, -9.81], atol=0.02)
    finally:
        backend.close()


TWO_BOXES_MODEL = """
<mujoco model="motrix_gpu_two_boxes">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="floor" pos="0 0 0">
      <geom name="floor_geom" type="box" size="1 0.5 0.05" pos="0 0 -0.05"/>
    </body>
    <body name="light" pos="-0.3 0 0.05">
      <freejoint name="light_root"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom name="light_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="heavy" pos="0.3 0 0.05">
      <freejoint name="heavy_root"/>
      <inertial pos="0 0 0" mass="2" diaginertia="0.01 0.01 0.01"/>
      <geom name="heavy_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
  <sensor>
    <contact name="light_net" body1="light" body2="floor" data="force" reduce="netforce"/>
    <contact name="heavy_net" body1="heavy" body2="floor" data="force" reduce="netforce"/>
  </sensor>
</mujoco>
"""


@pytest.mark.parametrize("capacity", [{}, {"max_contacts": 512, "max_contact_constraints": 1024}])
def test_netforce_contact_sensors_split_contact_pairs(tmp_path, capacity):
    backend, torch = _gpu_backend(tmp_path, TWO_BOXES_MODEL, 1, base_name="light", **capacity)
    try:
        backend.step_tensor(torch.zeros((1, 0), device="cuda"), 240)
        for name, weight in (("light_net", 9.81), ("heavy_net", 19.62)):
            view = backend.get_sensor_view(name, "cuda")
            np.testing.assert_allclose(view[0].cpu().numpy(), [0.0, 0.0, -weight], atol=0.02)
    finally:
        backend.close()


@pytest.mark.parametrize("gravcomp", ["1", "0"])
def test_gravcomp_links_hold_their_servo_target(tmp_path, gravcomp):
    backend, torch = _gpu_backend(tmp_path, GRAVCOMP_MODEL.format(gravcomp=gravcomp), 1)
    try:
        backend.step_tensor(torch.zeros((1, 1), device="cuda"), 300)
        angle = float(backend.get_sensor_view("angle", "cuda")[0, 0])
    finally:
        backend.close()
    if gravcomp == "1":
        # The compensated link holds the target: the feed-forward cancels its weight.
        assert abs(angle) < 0.02
    else:
        # Without compensation the 1 kg link at 0.2 m droops on the kp=2 servo.
        assert abs(angle) > 0.5


TINY_MESH_HALF_EXTENTS = (0.002, 0.0015, 0.00025)
BOX_FACES = (
    (1, 3, 4),
    (1, 4, 2),
    (5, 7, 8),
    (5, 8, 6),
    (1, 5, 6),
    (1, 6, 2),
    (3, 7, 8),
    (3, 8, 4),
    (1, 5, 7),
    (1, 7, 3),
    (2, 6, 8),
    (2, 8, 4),
)


def test_tiny_collision_mesh_keeps_a_collider(tmp_path):
    corners = itertools.product((-1, 1), repeat=3)
    (tmp_path / "tiny.obj").write_text(
        "".join(
            f"v {x * TINY_MESH_HALF_EXTENTS[0]:.6f} {y * TINY_MESH_HALF_EXTENTS[1]:.6f}"
            f" {z * TINY_MESH_HALF_EXTENTS[2]:.6f}\n"
            for x, y, z in corners
        )
        + "".join(f"f {a} {b} {c}\n" for a, b, c in BOX_FACES),
        encoding="utf-8",
    )
    model = """
<mujoco model="motrix_gpu_tiny_mesh">
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="0.1" diaginertia="1e-5 1e-5 1e-5"/>
      <geom name="tiny" type="mesh" mesh="tiny" density="0"/>
    </body>
  </worldbody>
  <asset><mesh name="tiny" file="tiny.obj"/></asset>
</mujoco>
"""
    backend, _ = _gpu_backend(tmp_path, model, 1)
    try:
        tiny = next(geom for geom in backend.model.geoms if geom.name == "tiny")
        assert len(list(tiny._collider_indices())) == 1
    finally:
        backend.close()
