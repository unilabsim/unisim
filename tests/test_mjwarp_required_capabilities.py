"""Real-CUDA reset effects for the Wuji-required public capabilities (#40)."""

from pathlib import Path

import numpy as np
import pytest

from unisim import MjwarpBackend
from unisim.backend.mjwarp.randomization import PrimitiveGeomBounds
from unisim.dr.types import ResetRandomizationPayload
from unisim.scene import SceneCfg

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")
warp = pytest.importorskip("warp")

MODEL = """<mujoco>
  <option timestep="0.005" gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="object" pos="0 0 0.045"><freejoint/>
      <geom name="ball" type="sphere" size="0.05" mass="1"/>
    </body>
    <body name="slider" pos="1 0 0.5">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="slide_geom" type="box" size="0.05 0.06 0.07" mass="1"/>
    </body>
    <body name="hand_root" mocap="true" pos="0 1 0.5">
      <body name="finger" pos="0.2 0 0">
        <joint name="hinge" axis="0 0 1"/>
        <geom name="finger_geom" type="capsule" size="0.02 0.1" mass="0.1"/>
        <site name="tip" pos="0.1 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="slide" ctrlrange="-10 10"/>
    <motor joint="hinge" ctrlrange="-10 10"/></actuator>
  <sensor>
    <framepos name="tip_position" objtype="site" objname="tip"/>
    <contact name="ball_found" geom1="ball" geom2="floor" data="found" num="1"/>
    <contact name="ball_force" geom1="ball" geom2="floor" data="force" reduce="netforce"/>
  </sensor>
</mujoco>"""


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    warp.init()
    if not warp.get_device().is_cuda:
        pytest.skip("required-capability numerical tests require CUDA")
    path = tmp_path_factory.mktemp("required-mjwarp") / "scene.xml"
    path.write_text(MODEL)
    return MjwarpBackend(SceneCfg(model_file=str(path)), 3, 0.005, base_name="object")


def _reset(backend, rows=None, payload=None, velocity=None, position=None):
    if rows is None:
        rows = np.arange(backend.num_envs, dtype=np.int32)
    qpos = np.tile(backend.get_default_qpos(), (len(rows), 1))
    qvel = np.tile(backend.get_init_qvel(), (len(rows), 1))
    if velocity is not None:
        qvel[:, 6] = velocity
    if position is not None:
        qpos[:, 2] = position
    backend.set_state(rows, qpos, qvel, randomization=payload)


def test_mocap_sparse_pose_effect_reset_and_other_worlds(backend) -> None:
    _reset(backend)
    binding = backend.bind_mocap_pose("hand_root")
    before = backend.get_sensor_data("tip_position").copy()
    qpos_before = backend.get_state(("qpos",))["qpos"].copy()
    poses = np.tile(binding.default_pose, (2, 1))
    poses[0, 0] += 0.4
    poses[1, 3:] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    binding.write(np.array([0, 2]), poses)
    after = backend.get_sensor_data("tip_position").copy()
    np.testing.assert_allclose(after[0], before[0] + [0.4, 0, 0], atol=1e-6)
    np.testing.assert_allclose(after[2], [0, 1.3, 0.5], atol=1e-6)
    np.testing.assert_array_equal(after[1], before[1])
    np.testing.assert_array_equal(backend.get_state(("qpos",))["qpos"], qpos_before)
    _reset(backend, np.array([0], dtype=np.int32))
    np.testing.assert_allclose(binding.read()[0], binding.default_pose)
    np.testing.assert_allclose(binding.read()[2], poses[1])
    np.testing.assert_allclose(backend.get_sensor_data("tip_position")[2], after[2])
    with pytest.raises(KeyError, match="does not exist"):
        backend.bind_mocap_pose("missing")
    with pytest.raises(NotImplementedError, match="not a mocap"):
        backend.bind_mocap_pose("object")


def test_extended_model_rows_bounds_and_contact_effect(backend) -> None:
    _reset(backend, position=0.08)
    sizes = backend.get_geom_sizes()[None].copy()
    ball = backend.get_geom_id("ball")
    sizes[0, ball, 0] = 0.1
    solref = backend.get_geom_solref()[None] * 1.2
    solimp = backend.get_geom_solimp()[None].copy()
    solimp[..., 2] *= 2
    damping = backend.get_dof_damping()[None] + 0.1
    frictionloss = backend.get_dof_frictionloss()[None] + 0.01
    payload = ResetRandomizationPayload(
        geom_size=sizes,
        geom_solref=solref,
        geom_solimp=solimp,
        dof_damping=damping,
        dof_frictionloss=frictionloss,
    )
    _reset(backend, np.array([1], dtype=np.int32), payload, position=0.08)
    for field in payload.requested_terms():
        values = getattr(backend._device_model, field).numpy()
        np.testing.assert_allclose(values[1], getattr(payload, field)[0])
        defaults = getattr(backend._cpu_model, field)
        np.testing.assert_allclose(
            values[[0, 2]], np.tile(defaults[None], (2, *([1] * defaults.ndim)))
        )
    np.testing.assert_allclose(backend._device_model.geom_rbound.numpy()[1, ball], 0.1)
    np.testing.assert_allclose(backend._device_model.geom_aabb.numpy()[1, ball, 1], [0.1] * 3)
    assert backend.get_sensor_data("ball_found")[1, 0] > 0
    np.testing.assert_array_equal(backend.get_sensor_data("ball_found")[[0, 2]], 0)
    backend.step(np.zeros((3, 2), dtype=np.float32), nsteps=2)
    assert np.isfinite(backend.get_sensor_data("ball_force")).all()


@pytest.mark.parametrize("field,amount", (("dof_damping", 10.0), ("dof_frictionloss", 5.0)))
def test_dof_randomization_changes_selected_world_motion(backend, field, amount) -> None:
    # Reset both fields explicitly because model DR persists across state resets.
    baseline = ResetRandomizationPayload(
        dof_damping=np.tile(backend.get_dof_damping(), (3, 1)),
        dof_frictionloss=np.tile(backend.get_dof_frictionloss(), (3, 1)),
    )
    _reset(backend, payload=baseline, velocity=1.0, position=1.0)
    values = np.zeros((1, backend.get_init_qvel().size), dtype=np.float32)
    values[0, 6] = amount
    _reset(
        backend,
        np.array([1], dtype=np.int32),
        ResetRandomizationPayload(**{field: values}),
        velocity=1.0,
        position=1.0,
    )
    for _ in range(10):
        backend.step(np.zeros((3, 2), dtype=np.float32), nsteps=4)
    velocity = backend.get_state(("qvel",))["qvel"][:, 6]
    assert abs(velocity[1]) < abs(velocity[0]) * 0.6
    np.testing.assert_allclose(velocity[[0, 2]], 1, atol=1e-5)


def test_new_field_validation_precedes_state_or_model_mutation(backend) -> None:
    before = backend.get_state(("qpos",))["qpos"].copy()
    size_before = backend._device_model.geom_size.numpy().copy()
    rows = np.array([0], dtype=np.int32)
    payload = ResetRandomizationPayload(geom_size=backend.get_geom_sizes()[None].copy())
    payload.geom_size[0, backend.get_geom_id("ball"), 0] = 0.2
    payload.dof_damping = np.full((1, backend.get_init_qvel().size), np.nan)
    with pytest.raises(ValueError, match="finite"):
        _reset(backend, rows, payload, position=10.0)
    np.testing.assert_array_equal(backend.get_state(("qpos",))["qpos"], before)
    np.testing.assert_array_equal(backend._device_model.geom_size.numpy(), size_before)
    payload.dof_damping = None
    payload.geom_size[0, backend.get_geom_id("floor"), 0] += 1
    with pytest.raises(NotImplementedError, match="non-primitive"):
        _reset(backend, rows, payload)


@pytest.mark.parametrize("field", ("geom_solref", "geom_solimp"))
def test_contact_parameters_change_selected_world_contact_force(backend, field) -> None:
    defaults = {
        "geom_size": backend.get_geom_sizes(),
        "geom_solref": backend.get_geom_solref(),
        "geom_solimp": backend.get_geom_solimp(),
    }
    payload = ResetRandomizationPayload(
        **{name: np.tile(value[None], (3, 1, 1)) for name, value in defaults.items()}
    )
    _reset(backend, payload=payload, position=0.045)
    original_force = backend.get_sensor_data("ball_force").copy()
    assert np.linalg.norm(original_force[0]) > 0
    values = defaults[field][None].copy()
    if field == "geom_solref":
        values[..., 0] = 0.08
    else:
        # Put this penetration inside a changed impedance transition rather
        # than changing both saturated endpoints equally (which cancels for
        # this one-contact zero-gravity system).
        values[..., 0] = 0.1
        values[..., 1] = 0.95
        values[..., 2] = 0.02
    _reset(
        backend,
        np.array([1], dtype=np.int32),
        ResetRandomizationPayload(**{field: values}),
        position=0.045,
    )
    changed_force = backend.get_sensor_data("ball_force")
    assert np.linalg.norm(changed_force[1] - original_force[1]) > 1e-3
    np.testing.assert_allclose(changed_force[[0, 2]], original_force[[0, 2]], atol=1e-6)


@pytest.mark.parametrize(
    "field,kind",
    (
        ("geom_size", "shape"),
        ("geom_size", "negative"),
        ("geom_solref", "mixed"),
        ("geom_solimp", "negative"),
        ("dof_damping", "negative"),
        ("dof_frictionloss", "negative"),
        ("geom_solref", "nan"),
        ("dof_damping", "integer"),
    ),
)
def test_extended_field_shapes_ranges_and_dtypes_fail_before_mutation(backend, field, kind) -> None:
    getter = backend.get_geom_sizes if field == "geom_size" else getattr(backend, f"get_{field}")
    values = getter()[None]
    if kind == "shape":
        values = values[:, :-1]
    elif kind == "mixed":
        values[..., 0] = -1
    elif kind == "nan":
        values[...] = np.nan
    elif kind == "integer":
        values = values.astype(np.int32)
    else:
        values[...] = -1
    before = backend.get_state(("qpos",))["qpos"].copy()
    error = TypeError if kind == "integer" else (ValueError, NotImplementedError)
    with pytest.raises(error):
        _reset(backend, np.array([0]), ResetRandomizationPayload(**{field: values}), position=3)
    np.testing.assert_array_equal(backend.get_state(("qpos",))["qpos"], before)


def test_mocap_large_coordinates_rejected_before_cast_mutation(backend) -> None:
    binding = backend.bind_mocap_pose("hand_root")
    before = binding.read()
    poses = binding.default_pose[None].astype(np.float64)
    poses[0, 0] = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="float32"):
        binding.write(np.array([1]), poses)
    np.testing.assert_array_equal(binding.read(), before)


def test_primitive_bounds_match_official_model_compilation(tmp_path: Path) -> None:
    xml = """<mujoco><worldbody>
      <geom type='plane' size='1 1 0.1'/>
      <geom type='sphere' size='0.2'/>
      <geom type='capsule' size='0.2 0.3'/>
      <geom type='ellipsoid' size='0.2 0.3 0.4'/>
      <geom type='cylinder' size='0.2 0.3'/>
      <geom type='box' size='0.2 0.3 0.4'/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    bounds = PrimitiveGeomBounds(model.geom_type, mujoco.mjtGeom)
    sizes = model.geom_size[None].astype(np.float32)
    rbound, aabb = bounds.compute(sizes, sizes, np.zeros((1, 6)), np.zeros((1, 6, 2, 3)))
    np.testing.assert_allclose(rbound[0, 1:], model.geom_rbound[1:], atol=1e-7)
    np.testing.assert_allclose(aabb[0, 1:], model.geom_aabb.reshape(6, 2, 3)[1:], atol=1e-7)
