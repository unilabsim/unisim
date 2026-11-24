"""Backend-neutral categories for MuJoCo sensor type names."""

from __future__ import annotations

from unisim.backend.base import SensorCategory

_SENSOR_NAMES_BY_CATEGORY: dict[SensorCategory, frozenset[str]] = {
    "position": frozenset(
        (
            "MAGNETOMETER",
            "RANGEFINDER",
            "CAMPROJECTION",
            "JOINTPOS",
            "TENDONPOS",
            "ACTUATORPOS",
            "BALLQUAT",
            "JOINTLIMITPOS",
            "TENDONLIMITPOS",
            "FRAMEPOS",
            "FRAMEQUAT",
            "FRAMEXAXIS",
            "FRAMEYAXIS",
            "FRAMEZAXIS",
            "SUBTREECOM",
            "INSIDESITE",
            "GEOMDIST",
            "GEOMNORMAL",
            "GEOMFROMTO",
            "E_POTENTIAL",
            "CLOCK",
        )
    ),
    "velocity": frozenset(
        (
            "VELOCIMETER",
            "GYRO",
            "JOINTVEL",
            "TENDONVEL",
            "ACTUATORVEL",
            "BALLANGVEL",
            "JOINTLIMITVEL",
            "TENDONLIMITVEL",
            "FRAMELINVEL",
            "FRAMEANGVEL",
            "SUBTREELINVEL",
            "SUBTREEANGMOM",
            "E_KINETIC",
        )
    ),
    "acceleration": frozenset(("ACCELEROMETER", "FRAMELINACC", "FRAMEANGACC")),
    "contact": frozenset(("TOUCH", "CONTACT", "TACTILE")),
    "force": frozenset(
        (
            "FORCE",
            "TORQUE",
            "ACTUATORFRC",
            "JOINTACTFRC",
            "TENDONACTFRC",
            "JOINTLIMITFRC",
            "TENDONLIMITFRC",
        )
    ),
    "other": frozenset(("PLUGIN", "USER")),
}


def classify_mujoco_sensor(sensor_type_name: str) -> SensorCategory:
    """Map one ``mjtSensor`` member name to its public metadata category."""
    sensor_name = sensor_type_name.removeprefix("mjSENS_")
    for category, names in _SENSOR_NAMES_BY_CATEGORY.items():
        if sensor_name in names:
            return category
    return "unknown"
