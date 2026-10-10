"""Device-resident reset randomization for the CUDA ``mjwarp`` backend."""

# ruff: noqa: E402
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
warp = pytest.importorskip("warp")
torch = pytest.importorskip("torch")

from unisim import MjwarpBackend
from unisim.dr.types import ResetRandomizationPayload, TensorResetRandomizationPayload
from unisim.scene import SceneCfg

MODEL = """<mujoco model='unisim-test-mjwarp-device-dr'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
  <sensor><framepos name='base_pos' objtype='body' objname='base'/></sensor>
  <actuator><motor joint='slide' ctrlrange='-10 10'/></actuator>
</mujoco>"""

COM_MODEL = """<mujoco model='unisim-test-mjwarp-device-dr-com'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/>
    <body name='tip' pos='0 0 0.5'><geom type='sphere' size='0.05' mass='0.3'/></body>
  </body></worldbody>
  <sensor><subtreecom name='com' body='base'/></sensor>
  <actuator><motor joint='slide' ctrlrange='-10 10'/></actuator>
</mujoco>"""


def _make_backend(
    tmp_path: Path,
    xml: str = MODEL,
    *,
    num_envs: int = 3,
    add_body_sensors: bool = False,
) -> MjwarpBackend:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp runtime tests require an active CUDA Warp device")
    model_path = tmp_path / "model.xml"
    model_path.write_text(xml)
    return MjwarpBackend(
        SceneCfg(model_file=str(model_path)),
        num_envs=num_envs,
        sim_dt=0.01,
        add_body_sensors=add_body_sensors,
    )


def _rows_qpos_qvel(backend: MjwarpBackend, rows: list[int]) -> tuple:
    count = len(rows)
    return (
        torch.tensor(rows, dtype=torch.int64, device="cuda"),
        torch.zeros((count, backend._nq), dtype=torch.float32, device="cuda"),
        torch.zeros((count, backend._nv), dtype=torch.float32, device="cuda"),
    )


def test_device_reset_randomization_capability_and_diagnostics(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    capabilities = backend.get_tensor_capabilities()
    assert capabilities.device_reset_randomization
    diagnostic = backend.get_tensor_runtime_diagnostics()["device_reset_randomization"]
    assert diagnostic.requested and diagnostic.enabled and diagnostic.disable_reason is None
    backend.close()


def test_device_payload_scatter_selected_rows_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _make_backend(tmp_path)
    set_const_calls = []
    set_const = backend._mujoco_warp.set_const
    monkeypatch.setattr(
        backend._mujoco_warp,
        "set_const",
        lambda *args, **kwargs: (set_const_calls.append(kwargs), set_const(*args, **kwargs))[1],
    )

    rows, qpos, qvel = _rows_qpos_qvel(backend, [2, 0])
    nbody = backend._nbody
    ngeom = int(backend._cpu_model.ngeom)
    payload = TensorResetRandomizationPayload(
        body_mass=torch.tensor(
            [[5.0, 2.0], [5.0, 3.0]], dtype=torch.float32, device="cuda"
        ),
        body_ipos=torch.full((2, nbody, 3), 0.01, dtype=torch.float32, device="cuda"),
        geom_friction=torch.full((2, ngeom, 3), 0.7, dtype=torch.float32, device="cuda"),
        kp=torch.full((2, backend._nu), 11.0, dtype=torch.float32, device="cuda"),
        kd=torch.full((2, backend._nu), 0.5, dtype=torch.float32, device="cuda"),
    )
    before_mass = backend._device_model.body_mass.numpy().copy()
    result = backend.set_state_tensor(rows, qpos, qvel, payload)

    assert set(result["timing"]) == {
        "set_state_tensor_mask_ms",
        "set_state_tensor_commit_forward_ms",
        "set_state_tensor_host_cache_refresh_ms",
        "set_state_tensor_model_update_ms",
    }
    assert result["timing"]["set_state_tensor_model_update_ms"] >= 0.0

    # Selected rows carry the scattered values; the untouched row is intact.
    mass = backend._device_model.body_mass.numpy()
    np.testing.assert_array_equal(mass[2], [5.0, 2.0])
    np.testing.assert_array_equal(mass[0], [5.0, 3.0])
    np.testing.assert_array_equal(mass[1], before_mass[1])
    friction = backend._device_model.geom_friction.numpy()
    np.testing.assert_allclose(friction[[2, 0]], 0.7)
    np.testing.assert_array_equal(friction[1], backend._dr_geom_friction[1])
    gainprm = backend._device_model.actuator_gainprm.numpy()
    biasprm = backend._device_model.actuator_biasprm.numpy()
    np.testing.assert_allclose(gainprm[[2, 0], :, 0], 11.0)
    np.testing.assert_allclose(biasprm[[2, 0], :, 1], -11.0)
    np.testing.assert_allclose(biasprm[[2, 0], :, 2], -0.5)
    np.testing.assert_array_equal(gainprm[1], backend._dr_actuator_gainprm[1])
    np.testing.assert_array_equal(biasprm[1], backend._dr_actuator_biasprm[1])

    # Mass/COM scatters refresh derived constants eagerly, skipping the
    # redundant restore passes (the lazy full-width refresh and the next
    # forward recompute derived Data state before any consumer reads it).
    assert set_const_calls == [{"restore": False}]

    # Scattered mirrors stay stale until a host read refreshes them.
    assert backend._dr_mirror_stale == {
        "body_mass",
        "body_ipos",
        "geom_friction",
        "actuator_gainprm",
        "actuator_biasprm",
    }
    backend.close()


def test_device_payload_mirrors_refresh_on_host_read(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    rows, qpos, qvel = _rows_qpos_qvel(backend, [2, 0])
    nbody = backend._nbody
    ipos = np.zeros((2, nbody, 3), dtype=np.float32)
    ipos[:, 1, 0] = [0.02, 0.03]
    payload = TensorResetRandomizationPayload(
        body_ipos=torch.as_tensor(ipos, dtype=torch.float32, device="cuda")
    )
    backend.set_state_tensor(rows, qpos, qvel, payload)
    assert "body_ipos" in backend._dr_mirror_stale

    # A mirror-backed public getter refreshes its own mirror from the device.
    readback = backend.get_body_ipos(env_ids=np.array([0, 1, 2]))
    assert "body_ipos" not in backend._dr_mirror_stale
    np.testing.assert_allclose(readback[0, 1, 0], 0.03)
    np.testing.assert_allclose(readback[2, 1, 0], 0.02)
    np.testing.assert_allclose(readback[1], np.zeros((nbody, 3), dtype=np.float32))

    # The NumPy preparation path refreshes every remaining stale mirror first.
    backend.set_state_tensor(
        *_rows_qpos_qvel(backend, [1]),
        TensorResetRandomizationPayload(
            geom_friction=torch.full((1, 1, 3), 0.4, dtype=torch.float32, device="cuda")
        ),
    )
    assert "geom_friction" in backend._dr_mirror_stale
    backend._prepare_reset_randomization(np.array([1]), ResetRandomizationPayload())
    assert not backend._dr_mirror_stale
    np.testing.assert_allclose(backend._dr_geom_friction[1], 0.4)
    np.testing.assert_allclose(backend._dr_body_ipos[0, 1, 0], 0.03)
    backend.close()


def test_device_payload_accepts_negative_body_ipos(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    rows, qpos, qvel = _rows_qpos_qvel(backend, [2, 0])
    nbody = backend._nbody
    # body_ipos is a center-of-mass offset, not a magnitude: negative
    # components are legitimate (host path validates it finite-only).
    ipos = torch.full((2, nbody, 3), -0.03, dtype=torch.float32, device="cuda")
    payload = TensorResetRandomizationPayload(body_ipos=ipos)
    backend.set_state_tensor(rows, qpos, qvel, payload)
    device_ipos = backend._device_model.body_ipos.numpy()
    np.testing.assert_allclose(device_ipos[[2, 0]], -0.03)
    np.testing.assert_allclose(device_ipos[1], backend._dr_body_ipos[1])
    readback = backend.get_body_ipos(env_ids=np.array([0, 2]))
    np.testing.assert_allclose(readback, -0.03)
    backend.close()


def test_device_payload_rejects_unsupported_field(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    rows, qpos, qvel = _rows_qpos_qvel(backend, [0])
    ngeom = int(backend._cpu_model.ngeom)
    # geom_size stays fail closed: its derived geom_rbound/geom_aabb rows are
    # computed host-side, with no device-side derivation in this slice.
    payload = TensorResetRandomizationPayload(
        geom_size=torch.full((1, ngeom, 3), 0.1, dtype=torch.float32, device="cuda")
    )
    with pytest.raises(NotImplementedError, match="geom_size"):
        backend.set_state_tensor(rows, qpos, qvel, payload)
    backend.close()


@pytest.mark.parametrize("problem", ["negative_mass", "negative_friction", "shape", "nan", "dtype"])
def test_device_payload_validation_fails_closed(tmp_path: Path, problem: str) -> None:
    backend = _make_backend(tmp_path)
    rows, qpos, qvel = _rows_qpos_qvel(backend, [0])
    ngeom = int(backend._cpu_model.ngeom)
    payload = TensorResetRandomizationPayload()
    if problem == "negative_mass":
        payload.body_mass = torch.full((1, backend._nbody), -1.0, device="cuda")
    elif problem == "negative_friction":
        payload.geom_friction = torch.full((1, ngeom, 3), -0.1, device="cuda")
    elif problem == "shape":
        payload.body_mass = torch.ones((1, backend._nbody + 1), device="cuda")
    elif problem == "nan":
        payload.kp = torch.full((1, backend._nu), torch.nan, device="cuda")
    elif problem == "dtype":
        payload.kd = torch.zeros((1, backend._nu), dtype=torch.float64, device="cuda")
    before = backend._device_model.body_mass.numpy().copy()
    with pytest.raises((ValueError, TypeError)):
        backend.set_state_tensor(rows, qpos, qvel, payload)
    np.testing.assert_array_equal(backend._device_model.body_mass.numpy(), before)
    assert not backend._dr_mirror_stale
    backend.close()


def test_numpy_payload_path_unchanged(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    rows, qpos, qvel = _rows_qpos_qvel(backend, [1])
    mass = np.array([[5.0, 4.0]], dtype=np.float32)
    result = backend.set_state_tensor(rows, qpos, qvel, ResetRandomizationPayload(body_mass=mass))
    assert result["timing"]["set_state_tensor_model_update_ms"] == 0.0
    np.testing.assert_array_equal(backend._device_model.body_mass.numpy()[1], mass[0])
    # The host path writes its mirrors directly; nothing is marked stale and
    # no device scatter views were bound.
    assert not backend._dr_mirror_stale
    assert not backend._model_torch_views
    np.testing.assert_array_equal(backend._dr_body_mass[1], mass[0])
    backend.close()


def test_device_payload_skips_scratch_sensor_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _make_backend(tmp_path)
    published = []
    monkeypatch.setattr(
        backend,
        "_publish_selected_tensor_reset_sensors",
        lambda rows: (published.append(rows), False)[1],
    )
    rows, qpos, qvel = _rows_qpos_qvel(backend, [1])
    payload = TensorResetRandomizationPayload(
        body_mass=torch.tensor([[5.0, 4.0]], dtype=torch.float32, device="cuda")
    )
    backend.set_state_tensor(rows, qpos, qvel, payload)
    # The scratch replay reads model rows 0..n-1 for scratch worlds, so a
    # randomized model must fall back to the lazy full-model refresh.
    assert published == []
    assert backend._tracked_body_state_dirty is False  # no tracked bodies configured

    backend.set_state_tensor(*_rows_qpos_qvel(backend, [1]))
    assert len(published) == 1
    backend.close()


def test_device_payload_tracked_refresh_matches_host_reference(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path, xml=COM_MODEL, add_body_sensors=True)
    host_dir = tmp_path / "host"
    host_dir.mkdir()
    host = _make_backend(host_dir, xml=COM_MODEL, add_body_sensors=True)
    rows, qpos, qvel = _rows_qpos_qvel(backend, [0])
    mass = np.array([[0.0, 1.0, 5.0]], dtype=np.float32)
    payload = TensorResetRandomizationPayload(
        body_mass=torch.as_tensor(mass, dtype=torch.float32, device="cuda")
    )
    backend.set_state_tensor(rows, qpos, qvel, payload)
    host.set_state(
        np.array([0]),
        np.zeros((1, host._nq), dtype=np.float32),
        np.zeros((1, host._nv), dtype=np.float32),
        ResetRandomizationPayload(body_mass=mass),
    )
    np.testing.assert_allclose(
        backend.get_sensor_data("com"), host.get_sensor_data("com"), atol=1e-6
    )
    backend.close()
    host.close()


def test_device_payload_extended_scatter_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _make_backend(tmp_path)
    set_const_calls = []
    set_const_0_calls = []
    monkeypatch.setattr(
        backend._mujoco_warp,
        "set_const",
        lambda *args, **kwargs: set_const_calls.append(kwargs),
    )
    monkeypatch.setattr(
        backend._mujoco_warp,
        "set_const_0",
        lambda *args, **kwargs: set_const_0_calls.append(kwargs),
    )
    rows, qpos, qvel = _rows_qpos_qvel(backend, [2, 0])
    ngeom = int(backend._cpu_model.ngeom)
    nv = backend._nv
    payload = TensorResetRandomizationPayload(
        gravity=torch.tensor(
            [[0.0, 0.0, -4.9], [0.0, 0.0, -3.7]], dtype=torch.float32, device="cuda"
        ),
        geom_solref=torch.full((2, ngeom, 2), 0.02, dtype=torch.float32, device="cuda"),
        geom_solimp=torch.tensor(
            [0.9, 0.95, 0.001, 0.5, 2.0], dtype=torch.float32, device="cuda"
        ).repeat(2, ngeom, 1),
        dof_damping=torch.full((2, nv), 0.3, dtype=torch.float32, device="cuda"),
        dof_frictionloss=torch.full((2, nv), 0.1, dtype=torch.float32, device="cuda"),
    )
    before_gravity = backend._device_model.opt.gravity.numpy().copy()
    backend.set_state_tensor(rows, qpos, qvel, payload)

    gravity = backend._device_model.opt.gravity.numpy()
    np.testing.assert_allclose(gravity[2], [0.0, 0.0, -4.9])
    np.testing.assert_allclose(gravity[0], [0.0, 0.0, -3.7])
    np.testing.assert_array_equal(gravity[1], before_gravity[1])
    np.testing.assert_allclose(backend._device_model.geom_solref.numpy()[[2, 0]], 0.02)
    np.testing.assert_array_equal(
        backend._device_model.geom_solref.numpy()[1], backend._dr_geom_solref[1]
    )
    np.testing.assert_allclose(
        backend._device_model.geom_solimp.numpy()[[2, 0]],
        np.broadcast_to([0.9, 0.95, 0.001, 0.5, 2.0], (2, ngeom, 5)),
    )
    np.testing.assert_allclose(backend._device_model.dof_damping.numpy()[[2, 0]], 0.3)
    np.testing.assert_array_equal(
        backend._device_model.dof_damping.numpy()[1], backend._dr_dof_damping[1]
    )
    np.testing.assert_allclose(backend._device_model.dof_frictionloss.numpy()[[2, 0]], 0.1)

    # refresh=0 fields skip every derived-constant pass, and each scattered
    # field marks its host mirror stale.
    assert set_const_calls == []
    assert set_const_0_calls == []
    assert backend._dr_mirror_stale == {
        "gravity",
        "geom_solref",
        "geom_solimp",
        "dof_damping",
        "dof_frictionloss",
    }

    # The NumPy preparation path refreshes the stale mirrors from the device.
    backend._prepare_reset_randomization(np.array([1]), ResetRandomizationPayload())
    assert not backend._dr_mirror_stale
    np.testing.assert_allclose(backend._dr_gravity[2], [0.0, 0.0, -4.9])
    np.testing.assert_allclose(backend._dr_dof_damping[0], 0.3)
    backend.close()


def test_device_payload_refresh_level_one_uses_set_const_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _make_backend(tmp_path)
    set_const_calls = []
    set_const_0_calls = []
    set_const = backend._mujoco_warp.set_const
    set_const_0 = backend._mujoco_warp.set_const_0
    monkeypatch.setattr(
        backend._mujoco_warp,
        "set_const",
        lambda *args, **kwargs: (set_const_calls.append(kwargs), set_const(*args, **kwargs))[1],
    )
    monkeypatch.setattr(
        backend._mujoco_warp,
        "set_const_0",
        lambda *args, **kwargs: (
            set_const_0_calls.append(kwargs),
            set_const_0(*args, **kwargs),
        )[1],
    )
    rows, qpos, qvel = _rows_qpos_qvel(backend, [1])
    payload = TensorResetRandomizationPayload(
        body_inertia=torch.full((1, backend._nbody, 3), 0.01, dtype=torch.float32, device="cuda"),
        dof_armature=torch.full((1, backend._nv), 0.05, dtype=torch.float32, device="cuda"),
    )
    backend.set_state_tensor(rows, qpos, qvel, payload)

    np.testing.assert_allclose(backend._device_model.body_inertia.numpy()[1], 0.01)
    np.testing.assert_allclose(backend._device_model.dof_armature.numpy()[1], 0.05)
    # Inertia/armature rows need only the qpos0-dependent refresh, still
    # skipping the redundant restore passes.
    assert set_const_calls == []
    assert set_const_0_calls == [{"restore": False}]
    assert "body_inertia" in backend._dr_mirror_stale
    assert "dof_armature" in backend._dr_mirror_stale
    backend.close()


def test_device_payload_body_iquat_scatter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _make_backend(tmp_path)
    set_const_calls = []
    set_const = backend._mujoco_warp.set_const
    monkeypatch.setattr(
        backend._mujoco_warp,
        "set_const",
        lambda *args, **kwargs: (set_const_calls.append(kwargs), set_const(*args, **kwargs))[1],
    )
    rows, qpos, qvel = _rows_qpos_qvel(backend, [0])
    iquat = torch.zeros((1, backend._nbody, 4), dtype=torch.float32, device="cuda")
    iquat[..., 0] = 1.0
    backend.set_state_tensor(rows, qpos, qvel, TensorResetRandomizationPayload(body_iquat=iquat))
    np.testing.assert_allclose(backend._device_model.body_iquat.numpy()[0], iquat[0].cpu().numpy())
    # Inertial orientation is a refresh=2 field, with restore passes skipped.
    assert set_const_calls == [{"restore": False}]
    assert "body_iquat" in backend._dr_mirror_stale
    backend.close()


def test_device_payload_validation_single_fused_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _make_backend(tmp_path)
    fused_calls = []
    fused = backend._validate_torch_fused
    monkeypatch.setattr(
        backend,
        "_validate_torch_fused",
        lambda entries: (fused_calls.append(len(entries)), fused(entries))[1],
    )
    rows, qpos, qvel = _rows_qpos_qvel(backend, [2, 0])
    ngeom = int(backend._cpu_model.ngeom)
    payload = TensorResetRandomizationPayload(
        body_mass=torch.full((2, backend._nbody), 2.0, dtype=torch.float32, device="cuda"),
        geom_friction=torch.full((2, ngeom, 3), 0.7, dtype=torch.float32, device="cuda"),
        gravity=torch.tensor([[0.0, 0.0, -9.81], [0.0, 0.0, -9.81]], device="cuda"),
        dof_damping=torch.full((2, backend._nv), 0.3, dtype=torch.float32, device="cuda"),
        kp=torch.full((2, backend._nu), 11.0, dtype=torch.float32, device="cuda"),
        kd=torch.full((2, backend._nu), 0.5, dtype=torch.float32, device="cuda"),
    )
    backend.set_state_tensor(rows, qpos, qvel, payload)
    # env_indices + qpos + qvel + six payload fields aggregate into one fused
    # reduction (one D2H sync) instead of one sync per operand.
    assert fused_calls == [9]

    backend.set_state_tensor(*_rows_qpos_qvel(backend, [1]))
    assert fused_calls == [9, 3]
    backend.close()


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("geom_solref", "geom_solref"),
        ("geom_solimp", "geom_solimp"),
        ("dof_damping", "dof_damping"),
        ("dof_frictionloss", "dof_frictionloss"),
        ("body_inertia", "body_inertia"),
    ],
)
def test_device_payload_extended_validation_names_field(
    tmp_path: Path, field: str, match: str
) -> None:
    backend = _make_backend(tmp_path)
    rows, qpos, qvel = _rows_qpos_qvel(backend, [0])
    ngeom = int(backend._cpu_model.ngeom)
    tails = {
        "geom_solref": (1, ngeom, 2),
        "geom_solimp": (1, ngeom, 5),
        "dof_damping": (1, backend._nv),
        "dof_frictionloss": (1, backend._nv),
        "body_inertia": (1, backend._nbody, 3),
    }
    values = torch.zeros(tails[field], dtype=torch.float32, device="cuda")
    if field == "geom_solref":
        values[..., 0] = 0.02
        values[..., 1] = -1.0  # mixed-sign pair
    elif field == "geom_solimp":
        values[...] = torch.tensor(
            [0.9, 0.95, 0.001, 1.5, 2.0], dtype=torch.float32, device="cuda"
        )  # midpoint outside (0, 1)
    else:
        values.fill_(-0.1)
    before = backend._device_model.geom_solref.numpy().copy()
    with pytest.raises(ValueError, match=match):
        backend.set_state_tensor(
            rows, qpos, qvel, TensorResetRandomizationPayload(**{field: values})
        )
    np.testing.assert_array_equal(backend._device_model.geom_solref.numpy(), before)
    assert not backend._dr_mirror_stale
    backend.close()


def test_device_payload_rejects_non_unit_body_iquat(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    rows, qpos, qvel = _rows_qpos_qvel(backend, [0])
    iquat = torch.zeros((1, backend._nbody, 4), dtype=torch.float32, device="cuda")
    iquat[..., 0] = 2.0
    with pytest.raises(ValueError, match="body_iquat"):
        backend.set_state_tensor(
            rows, qpos, qvel, TensorResetRandomizationPayload(body_iquat=iquat)
        )
    assert not backend._dr_mirror_stale
    backend.close()
