"""CPU coverage for MJWarp native and aggregate contact sensor layouts."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco

from unisim import MjwarpBackend

_MODEL = """
<mujoco model='mjwarp-contact-sensor-layouts'>
  <worldbody>
    <geom name='floor' type='plane' size='1 1 .1'/>
    <body name='box' pos='0 0 .2'>
      <freejoint/>
      <geom name='box_geom' type='box' size='.1 .1 .1'/>
    </body>
  </worldbody>
  <sensor>
    <contact name='aggregate' body1='box' body2='world'
             data='found force pos' num='1' reduce='none'/>
    <contact name='point_forces' body1='box' body2='world'
             data='found force' num='8' reduce='none'/>
    <contact name='full_points' body1='box' body2='world'
             data='found force dist pos normal tangent' num='8' reduce='none'/>
    <contact name='nearest' body1='box' body2='world'
             data='found pos' reduce='mindist'/>
    <contact name='net_force' body1='box' body2='world'
             data='force' reduce='netforce'/>
    <contact name='geom_points' geom1='box_geom' geom2='floor'
             data='found force' num='8' reduce='none'/>
  </sensor>
</mujoco>
"""

_NATIVE_BODY_PAIR_DIMENSIONS = {
    "point_forces": 32,
    "full_points": 112,
    "nearest": 4,
    "net_force": 3,
}


def _scan_fixture() -> tuple[MjwarpBackend, object]:
    model = mujoco.MjModel.from_xml_string(_MODEL)
    backend = object.__new__(MjwarpBackend)
    backend._mujoco = mujoco
    backend._cpu_model = model
    backend._sensor_slots, backend._sensor_id_by_name = backend._bind_sensor_slots()
    backend._body_pair_sensor_plans = backend._scan_body_pair_contact_sensors()
    return backend, model


def test_only_aggregate_contract_body_pair_sensors_get_decode_plans() -> None:
    backend, model = _scan_fixture()

    assert set(backend._body_pair_sensor_plans) == {"aggregate"}
    plan = backend._body_pair_sensor_plans["aggregate"]
    sensor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "aggregate"))
    assert plan.sensor_id == sensor_id
    assert plan.body1_id == int(model.body("box").id)
    assert plan.body2_id == 0


def test_non_aggregate_body_pair_views_use_native_sensordata_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, model = _scan_fixture()
    sensordata = np.arange(2 * model.nsensordata, dtype=np.float32).reshape(
        2, model.nsensordata
    )
    backend._device_data = SimpleNamespace(sensordata=sensordata)
    backend._sensor_torch_views = {}
    backend._tracked_sensor_names = set()
    backend._tracked_body_state_dirty = False
    backend._authored_sensor_state_dirty = False
    backend._require_entity_healthy = lambda: None
    backend._stable_tensor_field_view = lambda _name: None
    backend._torch_view = lambda value, **_kwargs: value
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())

    for name in (*_NATIVE_BODY_PAIR_DIMENSIONS, "geom_points"):
        sensor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name))
        address = int(model.sensor_adr[sensor_id])
        dimension = int(model.sensor_dim[sensor_id])
        if name in _NATIVE_BODY_PAIR_DIMENSIONS:
            assert dimension == _NATIVE_BODY_PAIR_DIMENSIONS[name]
        expected = sensordata[:, address : address + dimension]

        view = backend.get_sensor_view(name)

        assert view.shape == (2, dimension)
        np.testing.assert_array_equal(view, expected)
        assert backend.get_sensor_view(name) is view
