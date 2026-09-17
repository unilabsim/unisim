"""SDK-free MJCF sensor metadata contract checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from unisim.backend.subprocess_ipc import sensors


def _model(sensor: str) -> str:
    return f"""<mujoco>
      <worldbody>
        <body name="finger"><geom name="finger_geom" size=".1"/></body>
        <body name="object"><geom name="object_geom" size=".1"/></body>
        <body name="table"><geom name="table_geom" size=".1"/></body>
      </worldbody>
      <sensor>{sensor}</sensor>
    </mujoco>
    """


def _scan(tmp_path: Path, sensor: str) -> sensors.SceneMetadata:
    path = tmp_path / "scene.xml"
    path.write_text(_model(sensor), encoding="utf-8")
    return sensors.scan_scene_metadata(str(path), backend_label="test")


def test_netforce_contact_sensor_preserves_collision_pair_and_row_order(tmp_path: Path) -> None:
    metadata = _scan(
        tmp_path,
        '<contact name="finger_object" geom1="finger_geom" geom2="object_geom" '
        'data="force" reduce="netforce"/>'
        '<contact name="finger_table" geom1="finger_geom" geom2="table_geom" '
        'data="force" reduce="netforce"/>',
    )
    first = metadata.sensors["finger_object"]
    second = metadata.sensors["finger_table"]
    assert (first.kind, first.body_name, first.target_body_name, first.sensor_index) == (
        sensors.KIND_CONTACT_FORCE,
        "finger",
        "object",
        0,
    )
    assert (second.kind, second.body_name, second.target_body_name, second.sensor_index) == (
        sensors.KIND_CONTACT_FORCE,
        "finger",
        "table",
        1,
    )
    assert first.dim == second.dim == 3
    assert not metadata.unsupported_sensors


@pytest.mark.parametrize(
    ("sensor", "reason"),
    [
        (
            '<contact name="bad" geom1="finger_geom" geom2="object_geom" '
            'data="force" reduce="sum"/>',
            "only reduce='netforce'",
        ),
        (
            '<contact name="bad" geom1="finger_geom" data="force" reduce="netforce"/>',
            "requires both geom1 and geom2",
        ),
        (
            '<contact name="bad" geom1="finger_geom" geom2="missing" '
            'data="force" reduce="netforce"/>',
            "unknown geom",
        ),
    ],
)
def test_contact_force_scanning_fails_closed_on_unsupported_semantics(
    tmp_path: Path, sensor: str, reason: str
) -> None:
    metadata = _scan(tmp_path, sensor)
    assert "bad" not in metadata.sensors
    assert reason in metadata.unsupported_sensors["bad"].reason
