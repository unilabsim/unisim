"""Coverage for the lazy tactile plugin boundary."""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

import unisim
from unisim import (
    RayCollisionDescription,
    RayGeometryState,
    RayGeomType,
    RaySceneDescription,
    RaySiteFrameDescription,
    TactileFieldSpec,
    TactileSceneDescription,
    TactileSensor,
    TactileSensorCapabilities,
)
from unisim.errors import BackendError
from unisim.optional import OptionalDependencyError


def _scene() -> TactileSceneDescription:
    scene = RaySceneDescription(
        num_bodies=2,
        body_names=("object", "finger"),
        geom_types=(RayGeomType.BOX, RayGeomType.BOX),
        geom_sizes=np.array([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]]),
        geom_local_pos=np.zeros((2, 3)),
        geom_local_quat=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        geom_body_ids=np.array([0, 1], dtype=np.intp),
    )
    collision = RayCollisionDescription(
        scene=scene,
        geom_names=("object_geom", "finger_geom"),
        geom_tags=(("object",), ("finger",)),
        geom_mesh_ids=np.full(2, -1, dtype=np.intp),
        body_parent_ids=np.array([-1, -1], dtype=np.intp),
    )
    frame = RaySiteFrameDescription(
        "finger_pad", "finger", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    )
    return TactileSceneDescription(collision, ("object",), (frame,))


class _Sensor(TactileSensor):
    tactile_type = "test"
    _tactile_capabilities = TactileSensorCapabilities(
        pose_input_profiles=("numpy", "torch_cuda"),
        output_profiles=("numpy",),
        supports_selected_environments=True,
        supports_geometry_rebuild=True,
    )

    def __init__(self, num_envs: int = 1) -> None:
        self._num_envs = num_envs
        self._closed = False
        self._value = np.zeros((num_envs, 1), dtype=np.float32)
        self._pose: tuple[Any, Any] | None = None
        self._materialized: TactileSceneDescription | None = None
        self._specs = (
            TactileFieldSpec(
                "pressure",
                (num_envs, 1),
                "float32",
                "Pa",
                "sensor_local",
                ("environment", "node"),
                "sensor",
                "until the next advance(), reset(), rebuild(), or close()",
            ),
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def output_specs(self) -> tuple[TactileFieldSpec, ...]:
        return self._specs

    def materialize(self, scene: TactileSceneDescription) -> None:
        self._materialized = scene

    def update_pose(self, body_pos: Any, body_quat: Any, env_ids: Any = None) -> None:
        self._pose = body_pos, body_quat

    def advance(self) -> None:
        self._value += 1.0

    def read(self) -> Mapping[str, Any]:
        return {"pressure": self._value}

    def reset(self, env_ids: Any = None) -> None:
        if env_ids is None:
            self._value.fill(0.0)
        else:
            self._value[np.asarray(env_ids)] = 0.0

    def rebuild(self, scene: TactileSceneDescription) -> None:
        self._materialized = scene
        self._value.fill(0.0)

    def close(self) -> None:
        self._closed = True


def test_tactile_scene_reuses_public_collision_and_mount_descriptions():
    tactile_scene = _scene()
    assert tactile_scene.collision.scene.body_names == ("object", "finger")
    assert tactile_scene.object_body_names == ("object",)
    assert tactile_scene.sensor_frames[0].site_name == "finger_pad"


def test_tactile_scene_rejects_unbound_object_and_sensor_frames():
    scene = _scene()
    with pytest.raises(ValueError, match="object bodies are absent"):
        TactileSceneDescription(scene.collision, ("missing",), scene.sensor_frames)
    missing = RaySiteFrameDescription(
        "pad", "missing", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    )
    with pytest.raises(ValueError, match="sensor bodies are absent"):
        TactileSceneDescription(scene.collision, ("object",), (missing,))


def test_tactile_scene_selects_collision_geoms_and_carries_aligned_geometry_state():
    scene = _scene()
    state = RayGeometryState(
        num_envs=2,
        geom_names=scene.collision.geom_names,
        geom_sizes=np.tile(scene.collision.scene.geom_sizes, (2, 1, 1)),
        geom_local_pos=np.tile(scene.collision.scene.geom_local_pos, (2, 1, 1)),
        geom_local_quat=np.tile(scene.collision.scene.geom_local_quat, (2, 1, 1)),
        geom_enabled=np.ones((2, 2), dtype=np.bool_),
        geom_mesh_ids=np.tile(scene.collision.geom_mesh_ids, (2, 1)),
    )

    selected = TactileSceneDescription(
        scene.collision,
        ("object",),
        scene.sensor_frames,
        object_geom_names=("object_geom",),
        geometry_state=state,
    )

    assert selected.object_geom_names == ("object_geom",)
    assert selected.geometry_state is state
    assert _scene().object_geom_names is None


def test_tactile_scene_rejects_invalid_geom_selection_and_misaligned_geometry_state():
    scene = _scene()
    with pytest.raises(ValueError, match="absent from collision geometry"):
        TactileSceneDescription(
            scene.collision, ("object",), scene.sensor_frames, object_geom_names=("missing",)
        )
    with pytest.raises(ValueError, match="must belong to object_body_names"):
        TactileSceneDescription(
            scene.collision, ("object",), scene.sensor_frames, object_geom_names=("finger_geom",)
        )
    state = RayGeometryState(
        num_envs=1,
        geom_names=("other_geom",),
        geom_sizes=np.ones((1, 1, 3)),
        geom_local_pos=np.zeros((1, 1, 3)),
        geom_local_quat=np.asarray([[[1.0, 0.0, 0.0, 0.0]]]),
        geom_enabled=np.ones((1, 1), dtype=np.bool_),
        geom_mesh_ids=np.full((1, 1), -1, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="match collision geom_names in order"):
        TactileSceneDescription(
            scene.collision, ("object",), scene.sensor_frames, geometry_state=state
        )


def test_tactile_field_spec_preserves_unstructured_shape_metadata():
    field = TactileFieldSpec(
        "nodes",
        (4, None, 3),
        "float32",
        "mm",
        "sensor_local",
        ("environment", "node", "component"),
        "sensor",
        "until the next advance()",
    )
    assert field.shape == (4, None, 3)
    assert field.dimensions[-1] == "component"
    with pytest.raises(ValueError, match="dimensions must name"):
        TactileFieldSpec(
            "bad", (1, 2), "float32", "mm", "local", ("environment",),
            "sensor", "until next read",
        )


def test_factory_validates_before_import_and_forwards_opaque_tactile_config(monkeypatch):
    with pytest.raises(ValueError, match="tactile_type must be a non-empty string"):
        unisim.create_tactile_sensor(tactile_type="")
    with pytest.raises(TypeError, match="num_envs must be an integer"):
        unisim.create_tactile_sensor(tactile_type="hydroshear", num_envs=True)
    with pytest.raises(ValueError, match="num_envs must be positive"):
        unisim.create_tactile_sensor(tactile_type="hydroshear", num_envs=0)
    with pytest.raises(TypeError, match="config must be a mapping"):
        unisim.create_tactile_sensor(tactile_type="hydroshear", config=[])

    config = {"gel_triangles": "opaque to UniSim"}
    seen: dict[str, Any] = {}
    module = types.ModuleType("unitac")

    def tactile_factory(
        tactile_type: str, num_envs: int, config: Mapping[str, Any], **kwargs: Any
    ):
        seen.update(tactile_type=tactile_type, num_envs=num_envs, config=config, **kwargs)
        return _Sensor(num_envs)

    module.create_tactile_sensor = tactile_factory
    monkeypatch.setitem(sys.modules, "unitac", module)
    sensor = unisim.create_tactile_sensor(
        tactile_type="hydroshear", num_envs=4, config=config, device="cuda:0"
    )
    assert isinstance(sensor, TactileSensor)
    assert seen == {
        "tactile_type": "hydroshear",
        "num_envs": 4,
        "config": config,
        "device": "cuda:0",
    }
    sensor.materialize(_scene())
    sensor.update_pose(np.zeros((4, 2, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (4, 2, 1)))
    sensor.advance()
    result = sensor.read()
    assert result["pressure"].shape == (4, 1)
    assert sensor.output_specs[0].units == "Pa"
    sensor.reset([1])
    assert result["pressure"][1, 0] == 0.0
    sensor.close()


def test_tactile_factory_fails_closed_without_package_or_entry_point(monkeypatch):
    monkeypatch.setitem(sys.modules, "unitac", None)
    with pytest.raises(OptionalDependencyError, match="unitac"):
        unisim.create_tactile_sensor(tactile_type="hydroshear")

    module = types.ModuleType("unitac")
    monkeypatch.setitem(sys.modules, "unitac", module)
    with pytest.raises(OptionalDependencyError, match="create_tactile_sensor"):
        unisim.create_tactile_sensor(tactile_type="hydroshear")


def test_tactile_factory_checks_result_type(monkeypatch):
    module = types.ModuleType("unitac")
    module.create_tactile_sensor = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "unitac", module)
    with pytest.raises(BackendError, match="not a unisim.TactileSensor"):
        unisim.create_tactile_sensor(tactile_type="hydroshear")
