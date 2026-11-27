"""Issue #569 coverage: model-file actuator restores and the sensor public surface."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")

from unisim import MotrixBackend
from unisim.scene import SceneCfg

GENERAL_POS_MODEL = """<mujoco model='unisim-test'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='j1' type='hinge' axis='0 0 1'/>
    <geom type='box' size='0.05 0.05 0.05' mass='1'/></body></worldbody>
  <actuator>
    <general name='g1' joint='j1' input='pos' biastype='affine' ctrlrange='-1 1'
             gainprm='10' biasprm='0 -10 -1'/>
  </actuator>
</mujoco>"""

GENERAL_NON_ROUND_TRIP_MODEL = """<mujoco model='unisim-test'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='j1' type='hinge' axis='0 0 1'/>
    <geom type='box' size='0.05 0.05 0.05' mass='1'/></body></worldbody>
  <actuator>
    <general name='g1' joint='j1' input='pos' biastype='affine' gainprm='10' biasprm='0 -9 -1'/>
  </actuator>
</mujoco>"""

SENSOR_MODEL = """<mujoco model='unisim-test'>
  <option timestep='0.01'/>
  <worldbody>
    <body name='base'><joint name='j1' type='hinge' axis='0 0 1'/>
      <geom name='g1' type='sphere' size='0.05' mass='1'/></body>
    <body name='child'><joint name='j2' type='slide' axis='0 0 1' limited='false'/>
      <geom name='g2' type='sphere' size='0.05' mass='1'/></body>
  </worldbody>
  <actuator><motor name='m1' joint='j1' ctrlrange='-1 1'/></actuator>
  <sensor>
    <jointpos name='jp' joint='j1'/>
    <jointvel name='jv' joint='j1'/>
    <framepos name='fp' objtype='body' objname='child'/>
    <contact name='c_geom' geom1='g1' geom2='g2' data='found' num='1'/>
    <contact name='c_body' body1='base' body2='child'/>
    <contact name='c_sub' subtree1='base' subtree2='child'/>
    <actuatorfrc name='af' actuator='m1'/>
    <jointactuatorfrc name='jaf' joint='j1'/>
  </sensor>
</mujoco>"""

SITE_CONTACT_MODEL = """<mujoco model='unisim-test'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='j1' type='hinge'/>
    <geom type='sphere' size='0.05' mass='1'/><site name='s1'/></body></worldbody>
  <sensor><contact name='c_site' site1='s1'/></sensor>
</mujoco>"""


def _backend(tmp_path: Path, xml: str, name: str = "model.xml") -> MotrixBackend:
    model_path = tmp_path / name
    model_path.write_text(xml)
    return MotrixBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01)


def test_general_position_shorthand_restores_gain_surface(tmp_path: Path) -> None:
    backend = _backend(tmp_path, GENERAL_POS_MODEL)
    try:
        assert backend._supports_position_actuator_gains
        reset_terms = backend.get_dr_capabilities().supported_reset_terms
        assert {"kp", "kd"} <= reset_terms
        kp, kd = backend.get_actuator_gains()
        np.testing.assert_allclose(kp, [10.0], rtol=1e-6)
        np.testing.assert_allclose(kd, [1.0], rtol=1e-6)
    finally:
        backend.close()


def test_non_round_tripping_general_form_keeps_authored_semantics(tmp_path: Path) -> None:
    backend = _backend(tmp_path, GENERAL_NON_ROUND_TRIP_MODEL)
    try:
        assert not backend._supports_position_actuator_gains
        reset_terms = backend.get_dr_capabilities().supported_reset_terms
        assert "kp" not in reset_terms
    finally:
        backend.close()


def test_named_sensor_namespace_and_inventory(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    backend = _backend(tmp_path, SENSOR_MODEL)
    try:
        names = backend.get_sensor_names()
        assert names == tuple(sorted(names))
        assert {"jp", "jv", "fp", "c_geom", "c_body", "c_sub", "af", "jaf"} <= set(names)
        inventory = {descriptor.name: descriptor for descriptor in backend.get_sensor_inventory()}
        assert inventory["jp"].category == "position"
        assert inventory["jv"].category == "velocity"
        assert inventory["fp"].category == "position"
        assert inventory["c_geom"].category == "contact"
        assert inventory["c_body"].category == "contact"
        assert inventory["c_sub"].category == "contact"
        assert inventory["af"].category == "force"
        assert inventory["jaf"].category == "force"
        assert inventory["fp"].width == 3
        assert inventory["af"].width == 1
        with pytest.raises(KeyError):
            backend._get_sensor_category("missing_sensor")
    finally:
        backend.close()


def test_actuator_force_and_contact_filter_sensors_read_after_step(tmp_path: Path) -> None:
    backend = _backend(tmp_path, SENSOR_MODEL)
    try:
        backend._host_step(np.zeros((2, 1), dtype=np.float32))
        values = backend._host_read_sensor("af")
        assert values.shape == (2, 1)
        # jaf aggregates native actuator-force sensors with their joint gears.
        values = backend._host_read_sensor("jaf")
        assert values.shape == (2, 1)
        for name in ("c_geom", "c_body", "c_sub"):
            values = backend._host_read_sensor(name)
            assert values.shape[0] == 2
    finally:
        backend.close()


def test_site_filtered_contact_sensor_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="did not survive MJCF import"):
        _backend(tmp_path, SITE_CONTACT_MODEL).close()


def test_import_report_snapshots_effective_configuration(tmp_path: Path) -> None:
    backend = _backend(tmp_path, GENERAL_POS_MODEL)
    try:
        report = backend.get_import_report()
        assert report.backend == "motrix"
        assert report.profile == "single_mjcf"
        fields = {field.field: field for field in report.fields}
        assert set(fields) == {
            "solver",
            "solver_iterations",
            "solver_tolerance",
            "integrator",
            "cone",
            "impratio",
            "ls_iterations",
            "maxhullvert",
            "dt",
            "gravity",
            "actuator_mapping",
            "collision_filter",
            "body_mass",
            "body_inertia",
            "sensors",
        }
        assert fields["dt"].effective == pytest.approx(0.01, abs=1e-6)
        assert fields["actuator_mapping"].effective == {"position": 1}
        assert fields["sensors"].effective == {}
        assert fields["solver"].effective == "PGS"
        assert fields["solver"].difference == "unknown"
        assert fields["collision_filter"].difference == "unknown"
    finally:
        backend.close()


def test_import_report_profiles_portable(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    from unisim.dr.types import ModelSourceDescriptor
    from unisim.entities import EntityInitialState, SceneEntitySpec

    model_path = tmp_path / "arm.xml"
    model_path.write_text(
        """<mujoco model='unisim-arm'>
  <worldbody><body name='base'><geom type='box' size='0.05 0.05 0.05' mass='1'/></body>
  </worldbody>
</mujoco>"""
    )
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "arm",
                ModelSourceDescriptor(str(model_path)),
                root_mode="fixed",
                initial_state=EntityInitialState((0.0, 0.0, 1.0)),
            ),
        )
    )
    portable = MotrixBackend(scene, num_envs=1, sim_dt=0.01, base_name="arm/base")
    try:
        report = portable.get_import_report()
        assert report.profile == "portable"
        assert report.backend == "motrix"
    finally:
        portable.close()
