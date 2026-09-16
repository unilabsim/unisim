"""Real MuJoCo report readback, immutability and cold-path coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mjbatch")

from unisim import MuJoCoBackend
from unisim.inspection import ImportReport
from unisim.scene import SceneCfg

MODEL = """<mujoco><option timestep=".002" gravity="0 0 -2"/>
<worldbody><body name="arm"><joint name="joint"/><geom name="shape" size=".1" mass="2"/>
<site name="imu"/></body></worldbody><actuator><position name="drive" joint="joint" kp="3"/>
</actuator><sensor><gyro name="gyro" site="imu"/></sensor></mujoco>"""


def test_report_matches_model_and_remains_initial_snapshot(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "model.xml"
    source.write_text(MODEL)
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(source)), 2, 0.01, position_actuator_gains={"kp": 10.0, "kd": 1.0}
    )
    report = backend.get_import_report()
    fields = {item.field: item for item in report.fields}
    assert fields["add_body_sensors"].effective is False
    assert fields["refresh_pre_step_body_state"].effective is True
    assert fields["add_body_sensors"].provenance[-1].kind == "adapter_setting"
    assert fields["dt"].requested == 0.002
    assert fields["dt"].effective == 0.01
    assert fields["dt"].difference == "overridden"
    assert fields["gravity"].difference == "exact"
    assert fields["actuator_mapping"].requested["gainprm"][0][0] == 3.0
    assert fields["actuator_mapping"].effective["gainprm"][0][0] == 10.0
    assert fields["body_mass"].effective["values"][1] == 2.0
    assert fields["sensors"].effective["names"] == ("gyro",)
    assert ImportReport.from_dict(json.loads(json.dumps(report.to_dict()))) == report
    with pytest.raises(TypeError):
        fields["body_mass"].effective["values"] = ()
    backend.materialize()
    monkeypatch.setattr(backend, "_prepare_model_xml", lambda: pytest.fail("hot XML parse"))
    backend.step(np.zeros((2, 1)), nsteps=1)
    backend._model.body_mass[1] = 9.0
    assert backend.get_import_report() is report
    assert fields["body_mass"].effective["values"][1] == 2.0


def test_variant_report_does_not_claim_canonical_mass_for_all_envs(tmp_path: Path) -> None:
    from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor

    sources = []
    for index, mass in enumerate((2, 5)):
        source = tmp_path / f"variant-{index}.xml"
        source.write_text(MODEL.replace('mass="2"', f'mass="{mass}"'))
        sources.append(ModelSourceDescriptor(str(source)))
    plan = FixedVariantPlan(variants=tuple(sources), assignment=np.array([1, 0, 1]))
    backend = MuJoCoBackend(
        SceneCfg(model_file=sources[0].model_file, fixed_variant_plan=plan),
        3,
        0.01,
        position_actuator_gains={"kp": 11.0, "kd": 2.0},
    )
    report = backend.get_import_report()
    settings = {item.field: item for item in report.fields if item.scope.variant is None}
    assert settings["add_body_sensors"].effective is False
    assert settings["refresh_pre_step_body_state"].effective is True
    masses = [item for item in report.fields if item.field == "body_mass"]
    assert masses[0].scope.env_ids == (1,)
    assert masses[1].scope.env_ids == (0, 2)
    assert masses[0].effective["values"][1] == 2.0
    assert masses[1].effective["values"][1] == 5.0
    gains = [item for item in report.fields if item.field == "actuator_mapping"]
    assert all(item.requested["gainprm"][0][0] == 3.0 for item in gains)
    assert all(item.effective["gainprm"][0][0] == 11.0 for item in gains)


def test_factory_conditions_use_adapter_settings_snapshot(tmp_path: Path) -> None:
    from unisim import CapabilityCondition, SemanticRequirements, create_backend
    from unisim.validation import SemanticValidationError

    source = tmp_path / "settings.xml"
    source.write_text(MODEL)
    requirements = SemanticRequirements(configuration=(
        CapabilityCondition("add_body_sensors", "false"),
        CapabilityCondition("refresh_pre_step_body_state", "false"),
    ))
    backend = create_backend(
        "mujoco", SceneCfg(model_file=str(source)), 2, 0.01,
        refresh_pre_step_body_state=False, semantic_requirements=requirements,
    )
    try:
        assert backend.get_import_report().fields
    finally:
        backend.cleanup_scene_assets()
    with pytest.raises(SemanticValidationError, match="refresh_pre_step_body_state"):
        create_backend(
            "mujoco", SceneCfg(model_file=str(source)), 2, 0.01,
            refresh_pre_step_body_state=True, semantic_requirements=requirements,
        )
