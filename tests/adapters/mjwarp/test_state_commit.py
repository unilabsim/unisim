"""Preparation and real CUDA checks for unified whole/selected state commits."""

# ruff: noqa: E402
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from unisim.backend.mjwarp.backend import MjwarpBackend
from unisim.dr.types import ResetRandomizationPayload
from unisim.entities import EntityStatePatch, SceneResetRequest
from unisim.scene import SceneCfg

MODEL = """<mujoco><option gravity="0 0 0" integrator="Euler"/>
<worldbody><geom name="plane" type="plane" size="1 1 .1"/>
<body name="root" pos="0 0 2"><freejoint/><geom name="body" size=".1" mass="1"/>
 <body name="link" pos="0 0 .3"><joint name="hinge"/>
 <geom name="linkgeom" size=".03" mass=".2"/></body></body>
<body name="jointed" pos="2 0 2"><joint name="extra"/>
 <geom name="extra_geom" size=".1" mass="1"/></body>
<body name="target" mocap="true" pos="3 0 2"><geom name="targetgeom" size=".03"/></body>
</worldbody><actuator><general name="drive" joint="hinge" dyntype="filter" dynprm=".1"/>
</actuator><sensor><framepos name="position" objtype="body" objname="root"/></sensor></mujoco>"""


def _preparation_backend():
    backend = MjwarpBackend.__new__(MjwarpBackend)
    backend._cpu_model = mujoco.MjModel.from_xml_string(MODEL)
    backend._num_envs = 3
    backend._nbody = backend._cpu_model.nbody
    backend._nv, backend._nu = backend._cpu_model.nv, backend._cpu_model.nu
    backend._base_body_id = backend._cpu_model.body("root").id
    backend._push_body_id = None
    backend._interval_root_velocity_qvel_ids = None
    backend._fixed_variant_plan = None
    backend._fixed_variant_realization = None
    backend._bind_dr_host_mirrors()
    return backend


@pytest.mark.parametrize(
    "problem",
    [
        "late_gain_shape",
        "late_gain_nan",
        "late_iquat",
        "mass_negative",
        "inertia_negative",
        "armature_negative",
        "friction_negative",
        "mass_delta_negative",
        "offset_overflow",
        "body_mass_shape",
    ],
)
def test_model_preparation_is_pure_and_rejects_invalid_final_values(problem):
    backend = _preparation_backend()
    before = {
        name: value.copy()
        for name, value in vars(backend).items()
        if name.startswith("_dr_") and isinstance(value, np.ndarray)
    }
    rows = np.array([2, 0])
    mass = backend._dr_body_mass[rows].copy()
    mass[:, 1] *= 2
    payload = ResetRandomizationPayload(body_mass=mass)
    if problem == "late_gain_shape":
        payload.kd = np.zeros((2, 9))
    elif problem == "late_gain_nan":
        payload.kd = np.full((2, 1), np.nan)
    elif problem == "late_iquat":
        payload.body_iquat = np.zeros((2, backend._nbody, 4))
    elif problem == "mass_negative":
        payload.body_mass[:, 1] = -1
    elif problem == "inertia_negative":
        payload.body_inertia = np.full((2, backend._nbody, 3), -0.1)
    elif problem == "armature_negative":
        payload.dof_armature = np.full((2, backend._nv), -0.1)
    elif problem == "friction_negative":
        payload.geom_friction = np.full((2, backend._cpu_model.ngeom, 3), -0.1)
    elif problem == "mass_delta_negative":
        payload.base_mass_delta = np.array([-10.0, -10.0])
    elif problem == "offset_overflow":
        payload.base_com_offset = np.full((2, 3), np.finfo(np.float64).max)
    elif problem == "body_mass_shape":
        payload.body_mass = np.zeros((2, 99))
    with pytest.raises((ValueError, TypeError)):
        backend._prepare_reset_randomization(rows, payload)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(backend, name), value)


def test_prepared_dr_rows_do_not_alias_callers_or_live_mirrors():
    backend = _preparation_backend()
    mass = backend._dr_body_mass[[2, 0]].copy()
    payload = ResetRandomizationPayload(
        body_mass=mass, kp=np.ones((2, 1)) * 3, kd=np.ones((2, 1)) * 2
    )
    plan = backend._prepare_reset_randomization(np.array([2, 0]), payload)
    expected = plan.fields["body_mass"].copy()
    mass[:] = 7
    backend._dr_body_mass[:] = 8
    np.testing.assert_array_equal(plan.fields["body_mass"], expected)
    np.testing.assert_array_equal(plan.actuator_fields["actuator_biasprm"][[2, 0], :, 2], -2)
    assert plan.refresh == 2


@pytest.fixture
def gpu_backend(tmp_path):
    warp = pytest.importorskip("warp")
    warp.init()
    if not warp.get_device().is_cuda:
        pytest.skip("state commit acceptance requires CUDA")
    source = tmp_path / "legacy.xml"
    source.write_text(MODEL)
    backend = MjwarpBackend(
        SceneCfg(model_file=str(source)), 3, 0.002, base_name="root", nconmax=32, njmax=64
    )
    yield backend
    backend.close()


@pytest.mark.parametrize(
    "problem",
    ["shape", "nan", "duplicate", "negative", "overflow", "late_dr", "negative_mass", "quaternion"],
)
def test_invalid_legacy_commit_never_mutates_cache_model_or_device(
    gpu_backend, monkeypatch, problem
):
    backend = gpu_backend
    backend.step(np.ones((3, 1)))
    before = backend.get_state(("qpos", "qvel", "ctrl"))
    mass = backend._dr_body_mass.copy()
    rows = np.array([2, 0])
    qpos = before["qpos"][rows].copy()
    qvel = before["qvel"][rows].copy()
    payload = None
    if problem == "shape":
        qpos = qpos[:, :-1]
    elif problem == "nan":
        qvel[0, 0] = np.nan
    elif problem == "duplicate":
        rows = np.array([0, 0])
    elif problem == "negative":
        rows = np.array([-1, 0])
    elif problem == "overflow":
        qvel = qvel.astype(np.float64)
        qvel[0, 0] = 1e100
    elif problem == "late_dr":
        payload = ResetRandomizationPayload(body_mass=mass[rows] * 2, kd=np.zeros((2, 9)))
    elif problem == "negative_mass":
        payload = ResetRandomizationPayload(body_mass=-mass[rows])
    elif problem == "quaternion":
        payload = ResetRandomizationPayload(
            body_mass=mass[rows] * 2, body_iquat=np.zeros((2, backend._nbody, 4))
        )
    calls = []
    monkeypatch.setattr(backend, "_upload", lambda *args: calls.append("upload"))
    monkeypatch.setattr(backend, "_execute_device_reset", lambda: calls.append("reset"))
    monkeypatch.setattr(backend, "_execute_device_forward", lambda: calls.append("forward"))
    with pytest.raises((ValueError, TypeError)):
        backend.set_state(rows, qpos, qvel, payload)
    assert calls == []
    for name, values in before.items():
        np.testing.assert_array_equal(backend.get_state(name)[name], values)
    np.testing.assert_array_equal(backend._dr_body_mass, mass)
    assert not backend._entity_faulted


def test_legacy_whole_commit_preserves_unselected_channels_and_mocap_defaults(
    gpu_backend, monkeypatch
):
    backend = gpu_backend
    backend.step(np.ones((3, 1)))
    channels = backend._entity_persistent_channels()
    for name, values in channels.items():
        values[:] = 0.3
        backend._upload(getattr(backend._device_data, name), values)
    backend._xfrc_staging[:] = 0.2
    backend._xfrc_pending = True
    backend._mocap_pos[:] = 8
    backend._upload(backend._device_data.mocap_pos, backend._mocap_pos)
    before = backend.get_state()
    times = backend._time_cache.copy()
    rows = np.array([2, 0])
    qpos = before["qpos"][rows]
    qvel = before["qvel"][rows]
    qpos[:, 0] = [0.7, 0.9]
    commit = backend._commit_state
    plans = []

    def spy(plan):
        plans.append(plan)
        return commit(plan)

    monkeypatch.setattr(backend, "_commit_state", spy)
    backend.set_state(rows, qpos, qvel)
    assert len(plans) == 1 and plans[0].reset_world
    np.testing.assert_array_equal(backend.get_state()["qpos"][rows], qpos)
    after = backend._entity_persistent_channels()
    for name, values in after.items():
        np.testing.assert_array_equal(values[rows], 0)
        np.testing.assert_array_equal(values[1], channels[name][1])
    np.testing.assert_array_equal(backend.get_state("ctrl")["ctrl"][rows], 0)
    np.testing.assert_array_equal(backend._time_cache[rows], 0)
    assert backend._time_cache[1] == times[1]
    expected = np.broadcast_to(backend._default_mocap_pos, (2, *backend._default_mocap_pos.shape))
    np.testing.assert_array_equal(backend._device_data.mocap_pos.numpy()[rows], expected)
    np.testing.assert_array_equal(backend._device_data.mocap_pos.numpy()[1], 8)
    np.testing.assert_array_equal(backend._xfrc_staging[rows], 0)
    np.testing.assert_array_equal(backend._xfrc_staging[1], np.float32(0.2))


@pytest.mark.parametrize("failure", ["reset", "forward", "model"])
def test_legacy_native_failure_faults_all_state_reads(gpu_backend, monkeypatch, failure):
    backend = gpu_backend
    snapshot = backend.get_state()

    def broken(*args):
        raise RuntimeError("native write failed")

    payload = None
    if failure == "model":
        monkeypatch.setattr(backend, "_upload", broken)
        payload = ResetRandomizationPayload(body_mass=backend._dr_body_mass[[0]].copy())
    else:
        monkeypatch.setattr(backend, "_execute_device_" + failure, broken)
    with pytest.raises(RuntimeError, match="native write"):
        backend.set_state(np.array([0]), snapshot["qpos"][[0]], snapshot["qvel"][[0]], payload)
    for call in (
        backend.get_state,
        backend.get_dof_pos,
        backend.get_base_pos,
        lambda: backend.get_sensor_data("position"),
        lambda: backend.step(np.zeros((3, 1))),
    ):
        with pytest.raises(RuntimeError, match="faulted"):
            call()


def test_all_entity_and_legacy_entrypoints_share_submitter(tmp_path, monkeypatch):
    from .test_entities import _backend, _scene

    warp = pytest.importorskip("warp")
    warp.init()
    if not warp.get_device().is_cuda:
        pytest.skip("state commit acceptance requires CUDA")
    backend = _backend(_scene(tmp_path, fixed=True, key=True))
    try:
        plans = []
        commit = backend._commit_state

        def spy(plan):
            plans.append(plan)
            return commit(plan)

        monkeypatch.setattr(backend, "_commit_state", spy)
        backend.reset_entities(
            SceneResetRequest(
                (4, 1),
                (EntityStatePatch("robot", joint_positions=np.array([[0.2], [0.6]])),),
                restore_default_controls=True,
            )
        )
        assert len(plans) == 1 and not plans[-1].reset_world
        np.testing.assert_allclose(backend.get_state("ctrl")["ctrl"][[4, 1]], 0.3)
        snapshot = backend.get_state()
        backend.set_state(np.array([4, 1]), snapshot["qpos"][[4, 1]], snapshot["qvel"][[4, 1]])
        assert len(plans) == 2 and plans[-1].reset_world
        np.testing.assert_array_equal(backend.get_state("ctrl")["ctrl"][[4, 1]], 0)
        backend.reset(np.array([4, 1]))
        assert len(plans) == 3 and plans[-1].reset_world
        np.testing.assert_allclose(backend.get_state("ctrl")["ctrl"][[4, 1]], 0.3)
    finally:
        backend.close()


def test_legacy_preparation_owns_noncontiguous_input_aliases(gpu_backend, monkeypatch):
    backend = gpu_backend
    storage = np.array([2, 99, 0, 99], dtype=np.intp)
    rows = storage[::2]
    positions = np.repeat(backend.get_state()["qpos"][[2, 0]], 2, axis=1)
    velocities = np.repeat(backend.get_state()["qvel"][[2, 0]], 2, axis=1)
    qpos, qvel = positions[:, ::2], velocities[:, ::2]
    qpos[:, 0] = [0.7, 0.9]
    expected = qpos.copy()
    original = backend._commit_state

    def mutate_callers(plan):
        storage[:] = 1
        positions[:] = 3
        velocities[:] = 4
        return original(plan)

    monkeypatch.setattr(backend, "_commit_state", mutate_callers)
    backend.set_state(rows, qpos, qvel)
    np.testing.assert_array_equal(backend.get_state()["qpos"][[2, 0]], expected)
