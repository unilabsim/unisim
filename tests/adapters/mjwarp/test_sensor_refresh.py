"""Real-CUDA final-state complete sensor refresh for MJWarp."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

import warp

from unisim import BodyPairContactData, MjwarpBackend, SupportLevel
from unisim.dr.interval import TensorIntervalPayload
from unisim.scene import SceneCfg

_MODEL = """
<mujoco model='mjwarp-sensor-refresh'>
  <option timestep='0.005' iterations='50'/>
  <worldbody>
    <geom name='floor' type='plane' size='3 3 .1'/>
    <body name='ball' pos='0 0 .051'>
      <freejoint name='root'/>
      <geom name='ball_geom' type='sphere' size='.05' mass='1'/>
      <site name='ball_site'/>
    </body>
  </worldbody>
  <sensor>
    <framepos name='ball_pos' objtype='body' objname='ball'/>
    <framelinvel name='ball_vel' objtype='body' objname='ball'/>
    <accelerometer name='ball_acc' site='ball_site'/>
    <contact name='geom_ball_floor' geom1='ball_geom' geom2='floor'
             data='found force pos' num='1' reduce='none'/>
    <contact name='body_ball_floor' body1='ball' body2='world'
             data='found force pos' num='1' reduce='none'/>
  </sensor>
</mujoco>
"""
_FIELDS = ("active", "count", "position_w", "force_w")


def _backend(tmp_path: Path, name: str) -> MjwarpBackend:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp sensor refresh tests require an active CUDA Warp device")
    model_file = tmp_path / name
    model_file.write_text(_MODEL)
    return MjwarpBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=3,
        sim_dt=0.005,
        base_name="ball",
        add_body_sensors=True,
        nconmax=64,
        njmax=128,
    )


def _step(backend: MjwarpBackend) -> None:
    import torch

    ctrl = torch.zeros(
        (backend.num_envs, 0), dtype=torch.float32, device=backend._persistent_joint_f.device
    )
    result = backend.step_tensor(ctrl, nsteps=8)
    completion = result.get("completion")
    if completion is not None:
        completion.wait_before_read()


def _contact_values(view: BodyPairContactData) -> dict[str, Any]:
    return {field: getattr(view, field).detach().clone() for field in _FIELDS}


def _independent_forward(
    backend: MjwarpBackend, row: int, warmstart: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mujoco = backend._mujoco
    model = backend._cpu_model
    data = mujoco.MjData(model)
    data.qpos[:] = backend._device_data.qpos.numpy()[row]
    data.qvel[:] = backend._device_data.qvel.numpy()[row]
    data.ctrl[:] = backend._device_data.ctrl.numpy()[row]
    data.xfrc_applied[:] = backend._device_data.xfrc_applied.numpy()[row]
    data.qacc_warmstart[:] = warmstart[row]
    mujoco.mj_forward(model, data)

    ball_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"))
    floor_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    points = []
    force_sum = np.zeros(3)
    count = 0
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if {geom1, geom2} != {ball_geom, floor_geom}:
            continue
        native_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, native_force)
        world_force = contact.frame.reshape(3, 3).T @ native_force[:3]
        normal = np.asarray(contact.frame[:3])
        half_depth = 0.5 * float(contact.dist) * normal
        if geom1 == ball_geom:
            force_sum += world_force
            points.append(np.asarray(contact.pos) - half_depth)
        else:
            force_sum -= world_force
            points.append(np.asarray(contact.pos) + half_depth)
        count += 1
    aggregate = {
        "active": np.asarray(count > 0),
        "count": np.asarray(count, dtype=np.int32),
        "position_w": np.mean(points, axis=0) if points else np.zeros(3),
        "force_w": force_sum,
    }
    return data.sensordata.copy(), data.qacc_warmstart.copy(), aggregate


@pytest.mark.parametrize("staging", ["host", "device"])
def test_refresh_sensors_matches_independent_forward_and_preserves_other_rows(
    tmp_path: Path, staging: str
) -> None:
    backend = _backend(tmp_path, "refresh.xml")
    try:
        _step(backend)
        backend._sync_host_cache()
        backend.bind_body_pair_contact_views(("body_ball_floor",))
        view = backend.get_sensor_view("body_ball_floor")
        assert isinstance(view, BodyPairContactData)
        selected = 1
        unselected = np.array([0, 2], dtype=np.intp)
        qpos_before = backend._device_data.qpos.numpy().copy()
        qvel_before = backend._device_data.qvel.numpy().copy()
        ctrl_before = backend._device_data.ctrl.numpy().copy()
        sensors_before = backend._device_data.sensordata.numpy().copy()
        warm_before = backend._device_data.qacc_warmstart.numpy().copy()
        contacts_before = _contact_values(view)
        time_before = backend._time_cache.copy()
        expected_sensors, expected_warm, expected_contact = _independent_forward(
            backend, selected, warm_before
        )

        ball_id = backend.get_body_ids(["ball"])
        pending_force = np.zeros((backend.num_envs, 1, 3), dtype=np.float32)
        pending_force[:, 0, 0] = [4.0, -2.0, 1.0]
        if staging == "host":
            backend.apply_body_force(ball_id, pending_force)
        else:
            import torch

            backend.push_robots(
                TensorIntervalPayload(
                    push_force=torch.as_tensor(
                        pending_force[:, 0], device=backend._persistent_joint_f.device
                    )
                )
            )
        pending_staging_before = backend._xfrc_staging.copy()
        pending_native_xfrc = backend._device_data.xfrc_applied.numpy().copy()

        assert (
            backend.get_capabilities().get("state.sensor_refresh").support is SupportLevel.EXACT
        )
        backend.refresh_sensors([selected])

        np.testing.assert_allclose(
            backend._device_data.sensordata.numpy()[selected],
            expected_sensors,
            rtol=3e-3,
            atol=3e-4,
        )
        np.testing.assert_array_equal(
            backend._device_data.sensordata.numpy()[unselected], sensors_before[unselected]
        )
        np.testing.assert_allclose(
            backend._device_data.qacc_warmstart.numpy()[selected],
            expected_warm,
            rtol=3e-3,
            atol=3e-4,
        )
        np.testing.assert_array_equal(
            backend._device_data.qacc_warmstart.numpy()[unselected], warm_before[unselected]
        )
        for field in _FIELDS:
            actual = getattr(view, field).detach().cpu().numpy()
            if field in {"active", "count"}:
                assert actual[selected] == expected_contact[field]
                np.testing.assert_array_equal(
                    actual[unselected], contacts_before[field].cpu().numpy()[unselected]
                )
            else:
                np.testing.assert_allclose(
                    actual[selected], expected_contact[field], rtol=3e-3, atol=3e-4
                )
                np.testing.assert_array_equal(
                    actual[unselected], contacts_before[field].cpu().numpy()[unselected]
                )
        np.testing.assert_array_equal(backend._device_data.qpos.numpy(), qpos_before)
        np.testing.assert_array_equal(backend._device_data.qvel.numpy(), qvel_before)
        np.testing.assert_array_equal(backend._device_data.ctrl.numpy(), ctrl_before)
        np.testing.assert_array_equal(
            backend._device_data.xfrc_applied.numpy(), pending_native_xfrc
        )
        np.testing.assert_array_equal(backend._time_cache, time_before)
        np.testing.assert_array_equal(backend._xfrc_staging, pending_staging_before)
        assert backend._xfrc_pending == (staging == "host")
        assert backend._device_xfrc_pending == (staging == "device")
    finally:
        backend.close()


def test_refresh_sensors_cuda_graph_matches_eager(tmp_path: Path) -> None:
    graph = _backend(tmp_path, "graph.xml")
    eager = _backend(tmp_path, "eager.xml")
    if not graph._cuda_graph_enabled:
        reason = graph._cuda_graph_disable_reason or "unknown reason"
        graph.close()
        eager.close()
        pytest.skip(f"test requires an mjwarp CUDA forward graph; graphs disabled: {reason}")
    eager._cuda_graph_enabled = False
    try:
        _step(graph)
        _step(eager)
        graph_view = graph.get_sensor_view("body_ball_floor")
        eager_view = eager.get_sensor_view("body_ball_floor")
        graph.refresh_sensors([1])
        eager.refresh_sensors([1])
        np.testing.assert_allclose(
            graph._device_data.sensordata.numpy(),
            eager._device_data.sensordata.numpy(),
            rtol=3e-3,
            atol=3e-4,
        )
        np.testing.assert_allclose(
            graph._device_data.qacc_warmstart.numpy(),
            eager._device_data.qacc_warmstart.numpy(),
            rtol=3e-3,
            atol=3e-4,
        )
        for field in _FIELDS:
            actual = getattr(graph_view, field)
            expected = getattr(eager_view, field)
            if field in {"active", "count"}:
                np.testing.assert_array_equal(
                    actual.cpu().numpy(), expected.cpu().numpy()
                )
            else:
                np.testing.assert_allclose(
                    actual.cpu().numpy(), expected.cpu().numpy(), rtol=3e-3, atol=3e-4
                )
    finally:
        graph.close()
        eager.close()
