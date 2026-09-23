"""SDK-free checks for the bounded IsaacSim PhysX solver configuration."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from unisim import create_backend
from unisim.backend.isaacsim.backend import IsaacSimBackend, IsaacSimWorkerError
from unisim.backend.isaacsim.physx_solver import (
    PHYSX_SOLVER_AUTHORED_FIELDS,
    PHYSX_SOLVER_FIELDS,
    PhysxSolverConfig,
    apply_collision_offsets,
    apply_max_depenetration_velocity,
    build_isaaclab_physx_cfg,
    read_engine_solver_values,
    solver_value_matches,
)


def _backend(config: PhysxSolverConfig) -> IsaacSimBackend:
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._physx_solver = config
    return backend


def _meta(effective, readback):
    return {
        "configuration_report": {
            "schema_version": 1,
            "effective": effective,
            "engine_readback": readback,
        }
    }


def test_defaults_keep_existing_behavior():
    config = PhysxSolverConfig()
    assert config.configured_fields() == ()
    assert config.to_payload() == {}
    assert PhysxSolverConfig.from_payload(None) == config
    assert PhysxSolverConfig.from_payload({}) == config
    # An empty request skips the strict readback comparison entirely.
    _backend(config)._validate_solver_readback({})


def test_valid_values_normalize():
    config = PhysxSolverConfig(
        solver_position_iteration_count=8,
        # PhysX accepts zero velocity iterations (the SimToolReal default).
        solver_velocity_iteration_count=0,
        bounce_threshold_velocity=0,
        contact_offset=0.002,
        rest_offset=0.0,
        max_depenetration_velocity=1000,
        gpu_max_rigid_contact_count=2**24,
        gpu_max_rigid_patch_count=2**23,
    )
    assert config.configured_fields() == PHYSX_SOLVER_FIELDS
    assert config.to_payload() == {
        "solver_position_iteration_count": 8,
        "solver_velocity_iteration_count": 0,
        "bounce_threshold_velocity": 0.0,
        "contact_offset": 0.002,
        "rest_offset": 0.0,
        "max_depenetration_velocity": 1000.0,
        "gpu_max_rigid_contact_count": 16777216,
        "gpu_max_rigid_patch_count": 8388608,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver_position_iteration_count": 0},
        {"solver_position_iteration_count": -1},
        {"solver_position_iteration_count": 1.5},
        {"solver_position_iteration_count": True},
        {"solver_position_iteration_count": "8"},
        {"solver_velocity_iteration_count": -1},
        {"solver_velocity_iteration_count": 0.5},
        {"bounce_threshold_velocity": -0.5},
        {"bounce_threshold_velocity": float("nan")},
        {"bounce_threshold_velocity": float("inf")},
        {"bounce_threshold_velocity": True},
        {"contact_offset": 0},
        {"contact_offset": -0.002},
        {"contact_offset": "0.002"},
        {"rest_offset": -0.001, "contact_offset": 0.002},
        {"rest_offset": float("nan"), "contact_offset": 0.002},
        {"rest_offset": True, "contact_offset": 0.002},
        {"max_depenetration_velocity": -100},
        {"max_depenetration_velocity": float("inf")},
        {"max_depenetration_velocity": float("nan")},
        {"max_depenetration_velocity": "1000"},
        {"max_depenetration_velocity": True},
        {"gpu_max_rigid_contact_count": 0},
        {"gpu_max_rigid_contact_count": -1},
        {"gpu_max_rigid_contact_count": 1.5},
        {"gpu_max_rigid_contact_count": True},
        {"gpu_max_rigid_contact_count": "16777216"},
        {"gpu_max_rigid_patch_count": 0},
        {"gpu_max_rigid_patch_count": -1},
        {"gpu_max_rigid_patch_count": 1.5},
        {"gpu_max_rigid_patch_count": True},
        {"gpu_max_rigid_patch_count": "8388608"},
    ],
)
def test_invalid_values_fail_closed(kwargs):
    with pytest.raises((TypeError, ValueError)):
        PhysxSolverConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        # PhysX requires restOffset <= contactOffset; fail closed when the
        # relationship is violated or cannot be verified.
        ({"rest_offset": 0.003, "contact_offset": 0.002}, "must not exceed"),
        ({"rest_offset": 0.001}, "requires an explicit contact_offset"),
    ],
)
def test_rest_offset_contact_offset_relationship_fails_closed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PhysxSolverConfig(**kwargs)


def test_rest_offset_equal_to_contact_offset_is_accepted():
    config = PhysxSolverConfig(contact_offset=0.002, rest_offset=0.002)
    assert config.to_payload() == {"contact_offset": 0.002, "rest_offset": 0.002}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver_position_iteration_count": 0},
        {"solver_velocity_iteration_count": -2},
        {"bounce_threshold_velocity": float("nan")},
        {"contact_offset": -1e-3},
        {"rest_offset": 0.004, "contact_offset": 0.002},
        {"rest_offset": 0.001},
        {"max_depenetration_velocity": -1.0},
        {"gpu_max_rigid_contact_count": 0},
        {"gpu_max_rigid_patch_count": -1},
    ],
)
def test_backend_constructor_rejects_before_scene_use(kwargs):
    # Validation runs before scene composition or any worker spawn.
    with pytest.raises((TypeError, ValueError)):
        IsaacSimBackend(None, 1, 0.01, **kwargs)


def test_worker_payload_revalidation_fails_closed():
    with pytest.raises(TypeError):
        PhysxSolverConfig.from_payload([("contact_offset", 0.002)])
    with pytest.raises(ValueError, match="unsupported fields"):
        PhysxSolverConfig.from_payload({"solver_type": 0})
    with pytest.raises(ValueError):
        PhysxSolverConfig.from_payload({"solver_position_iteration_count": 0})


def test_build_isaaclab_physx_cfg_pins_scene_iteration_range():
    captured = {}

    class _PhysxCfg:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = PhysxSolverConfig(
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=1,
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=2**24,
        gpu_max_rigid_patch_count=2**23,
    )
    build_isaaclab_physx_cfg(SimpleNamespace(PhysxCfg=_PhysxCfg), config)
    assert captured == {
        "min_position_iteration_count": 8,
        "max_position_iteration_count": 8,
        "min_velocity_iteration_count": 1,
        "max_velocity_iteration_count": 1,
        "bounce_threshold_velocity": 0.2,
        "gpu_max_rigid_contact_count": 16777216,
        "gpu_max_rigid_patch_count": 8388608,
    }
    captured.clear()
    build_isaaclab_physx_cfg(SimpleNamespace(PhysxCfg=_PhysxCfg), PhysxSolverConfig())
    assert captured == {}


def test_solver_value_matches_strict_with_float32_tolerance():
    assert solver_value_matches("solver_position_iteration_count", 8, 8)
    assert not solver_value_matches("solver_position_iteration_count", 8, 7)
    assert not solver_value_matches("solver_position_iteration_count", 8, 8.0)
    assert not solver_value_matches("solver_position_iteration_count", 8, True)
    assert solver_value_matches("contact_offset", 0.002, 0.002)
    assert solver_value_matches("rest_offset", 0.001, float(np.float32(0.001)))
    assert solver_value_matches("max_depenetration_velocity", 1000.0, 1000)
    assert not solver_value_matches("max_depenetration_velocity", 1000.0, 100.0)
    assert solver_value_matches("gpu_max_rigid_contact_count", 2**24, 16777216)
    assert not solver_value_matches("gpu_max_rigid_contact_count", 2**24, 2**23)
    assert not solver_value_matches("gpu_max_rigid_contact_count", 2**24, float(2**24))
    assert not solver_value_matches("gpu_max_rigid_patch_count", 2**23, True)
    assert not solver_value_matches("rest_offset", 0.001, float("nan"))
    # USD float attributes store single precision; the readback differs only
    # by that quantization.
    assert solver_value_matches("contact_offset", 0.002, float(np.float32(0.002)))
    assert not solver_value_matches("contact_offset", 0.002, 0.0021)
    assert not solver_value_matches("contact_offset", 0.002, float("nan"))
    assert not solver_value_matches("bounce_threshold_velocity", 0.5, "0.5")
    with pytest.raises(ValueError, match="unknown PhysX solver field"):
        solver_value_matches("solver_type", 1, 1)


def test_host_accepts_matching_engine_readback():
    backend = _backend(
        PhysxSolverConfig(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            contact_offset=0.002,
            rest_offset=0.001,
            max_depenetration_velocity=1000,
        )
    )
    effective = {
        "solver_position_iteration_count": 8,
        "solver_velocity_iteration_count": 1,
        "bounce_threshold_velocity": 0.20000000298023224,
        "contact_offset": 0.0020000000949949026,
        "rest_offset": 0.0010000000474974513,
        "max_depenetration_velocity": 1000.0,
    }
    backend._validate_solver_readback(_meta(effective, list(PHYSX_SOLVER_FIELDS)))


@pytest.mark.parametrize(
    "breakage",
    ["no_envelope", "no_effective", "missing_readback", "missing_effective", "mismatch"],
)
def test_host_rejects_bad_engine_readback(breakage):
    backend = _backend(PhysxSolverConfig(solver_position_iteration_count=8))
    effective = {"solver_position_iteration_count": 8}
    readback = ["solver_position_iteration_count"]
    if breakage == "no_envelope":
        meta = {}
    elif breakage == "no_effective":
        meta = {"configuration_report": {"schema_version": 1}}
    else:
        if breakage == "missing_readback":
            readback = ["dt"]
        elif breakage == "missing_effective":
            effective = {"dt": 0.01}
        elif breakage == "mismatch":
            effective = {"solver_position_iteration_count": 4}
        meta = _meta(effective, readback)
    with pytest.raises(IsaacSimWorkerError):
        backend._validate_solver_readback(meta)


def test_host_accepts_authored_gpu_buffer_report():
    # The GPU buffer capacities are PhysX carb settings with no USD
    # attribute; the worker reports the authored value without claiming an
    # engine readback.
    backend = _backend(
        PhysxSolverConfig(
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**23,
        )
    )
    effective = {
        "gpu_max_rigid_contact_count": 16777216,
        "gpu_max_rigid_patch_count": 8388608,
    }
    backend._validate_solver_readback(_meta(effective, ["dt"]))


def test_host_rejects_forged_gpu_buffer_readback_fields():
    # Listing an authored-only field in engine_readback must not change the
    # strict comparison against the reported effective value.
    backend = _backend(PhysxSolverConfig(gpu_max_rigid_contact_count=2**24))
    meta = _meta({"gpu_max_rigid_contact_count": 2**23}, ["gpu_max_rigid_contact_count"])
    with pytest.raises(IsaacSimWorkerError):
        backend._validate_solver_readback(meta)


@pytest.mark.parametrize("breakage", ["missing_effective", "mismatch"])
def test_host_rejects_bad_authored_gpu_buffer_report(breakage):
    backend = _backend(
        PhysxSolverConfig(
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**23,
        )
    )
    effective = {
        "gpu_max_rigid_contact_count": 16777216,
        "gpu_max_rigid_patch_count": 8388608,
    }
    if breakage == "missing_effective":
        del effective["gpu_max_rigid_patch_count"]
    else:
        effective["gpu_max_rigid_patch_count"] = 2**22
    with pytest.raises(IsaacSimWorkerError):
        backend._validate_solver_readback(_meta(effective, ["dt"]))


def test_authored_field_set_matches_payload_contract():
    assert set(PHYSX_SOLVER_AUTHORED_FIELDS) == {
        "gpu_max_rigid_contact_count",
        "gpu_max_rigid_patch_count",
    }
    assert set(PHYSX_SOLVER_AUTHORED_FIELDS).issubset(PHYSX_SOLVER_FIELDS)


def test_init_payload_carries_physx_solver():
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._requested_render_mode = None
    backend._render_width = 1280
    backend._render_height = 720
    backend._entity_scene = None
    backend._physx_solver = PhysxSolverConfig(solver_position_iteration_count=8)
    payload = backend._worker_init_payload()
    assert payload["physx_solver"] == {"solver_position_iteration_count": 8}
    backend._physx_solver = PhysxSolverConfig()
    assert backend._worker_init_payload()["physx_solver"] == {}


def test_factory_forwards_solver_kwargs(monkeypatch):
    recorded = {}

    class _Recorder:
        def __init__(self, scene, num_envs, sim_dt, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(
        "unisim.backend.isaacsim.backend.IsaacSimBackend", _Recorder
    )
    create_backend(
        "isaacsim",
        None,
        2,
        1 / 60,
        worker_command=["true"],
        isaacsim_solver_position_iteration_count=8,
        isaacsim_solver_velocity_iteration_count=1,
        isaacsim_bounce_threshold_velocity=0.2,
        isaacsim_contact_offset=0.002,
        isaacsim_rest_offset=0.001,
        isaacsim_max_depenetration_velocity=1000.0,
        isaacsim_gpu_max_rigid_contact_count=2**24,
        isaacsim_gpu_max_rigid_patch_count=2**23,
    )
    assert recorded["solver_position_iteration_count"] == 8
    assert recorded["solver_velocity_iteration_count"] == 1
    assert recorded["bounce_threshold_velocity"] == 0.2
    assert recorded["contact_offset"] == 0.002
    assert recorded["rest_offset"] == 0.001
    assert recorded["max_depenetration_velocity"] == 1000.0
    assert recorded["gpu_max_rigid_contact_count"] == 16777216
    assert recorded["gpu_max_rigid_patch_count"] == 8388608


def test_factory_omits_unset_solver_kwargs(monkeypatch):
    recorded = {}

    class _Recorder:
        def __init__(self, scene, num_envs, sim_dt, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(
        "unisim.backend.isaacsim.backend.IsaacSimBackend", _Recorder
    )
    create_backend("isaacsim", None, 1, 0.01, worker_command=["true"])
    for key in (
        "solver_position_iteration_count",
        "solver_velocity_iteration_count",
        "bounce_threshold_velocity",
        "contact_offset",
        "rest_offset",
        "max_depenetration_velocity",
        "gpu_max_rigid_contact_count",
        "gpu_max_rigid_patch_count",
    ):
        assert key not in recorded


def test_import_report_fields_track_engine_readback():
    backend = _backend(PhysxSolverConfig(contact_offset=0.002))
    effective = {"contact_offset": 0.0020000000949949026}
    fields = backend._worker_configuration_fields(
        effective, {"contact_offset"}
    )
    (field,) = fields
    assert field.field == "contact_offset"
    assert field.difference == "approximate"
    assert [p.kind for p in field.provenance] == ["adapter_setting", "engine_readback"]

    (exact,) = backend._worker_configuration_fields(
        {"contact_offset": 0.002}, {"contact_offset"}
    )
    assert exact.difference == "exact"

    (overridden,) = backend._worker_configuration_fields(
        {"contact_offset": 0.004}, {"contact_offset"}
    )
    assert overridden.difference == "overridden"

    (unknown,) = backend._worker_configuration_fields({}, set())
    assert unknown.difference == "unknown"
    assert unknown.provenance[1].kind == "unverified"


def test_import_report_fields_mark_authored_gpu_buffers_unverified():
    # Carb settings have no engine readback; a matching authored report is
    # exact but carries the honest "unverified" provenance.
    backend = _backend(PhysxSolverConfig(gpu_max_rigid_contact_count=2**24))
    (field,) = backend._worker_configuration_fields(
        {"gpu_max_rigid_contact_count": 16777216}, {"dt"}
    )
    assert field.field == "gpu_max_rigid_contact_count"
    assert field.difference == "exact"
    assert [p.kind for p in field.provenance] == ["adapter_setting", "unverified"]


class _FakeAttr:
    def __init__(self, getter, setter=None):
        self._getter = getter
        self._setter = setter

    def Get(self):  # noqa: N802 - mirrors the USD attribute API
        return self._getter()

    def Set(self, value):  # noqa: N802 - mirrors the USD attribute API
        self._setter(value)


class _FakePrim:
    def __init__(
        self, apis, contact_offset=None, rest_offset=None, max_depenetration=None,
        scene_values=None,
    ):
        self._apis = apis
        self.contact_offset = contact_offset
        self.rest_offset = rest_offset
        self.max_depenetration = max_depenetration
        self.scene_values = scene_values or {}

    def HasAPI(self, api):  # noqa: N802 - mirrors the USD prim API
        return api in self._apis


def _fake_pxr():
    class _CollisionAPI:
        pass

    class _RigidBodyAPI:
        pass

    class _PhysxCollisionAPI:
        def __init__(self, prim):
            self._prim = prim

        def __bool__(self):
            return type(self) in self._prim._apis

        @classmethod
        def Apply(cls, prim):  # noqa: N802 - mirrors the USD API
            prim._apis.add(cls)
            return cls(prim)

        def CreateContactOffsetAttr(self):  # noqa: N802 - mirrors the USD API
            return _FakeAttr(
                lambda: self._prim.contact_offset,
                lambda value: setattr(self._prim, "contact_offset", value),
            )

        def GetContactOffsetAttr(self):  # noqa: N802 - mirrors the USD API
            return _FakeAttr(lambda: self._prim.contact_offset)

        def CreateRestOffsetAttr(self):  # noqa: N802 - mirrors the USD API
            return _FakeAttr(
                lambda: self._prim.rest_offset,
                lambda value: setattr(self._prim, "rest_offset", value),
            )

        def GetRestOffsetAttr(self):  # noqa: N802 - mirrors the USD API
            return _FakeAttr(lambda: self._prim.rest_offset)

    class _PhysxRigidBodyAPI:
        def __init__(self, prim):
            self._prim = prim

        def __bool__(self):
            return type(self) in self._prim._apis

        @classmethod
        def Apply(cls, prim):  # noqa: N802 - mirrors the USD API
            prim._apis.add(cls)
            return cls(prim)

        def CreateMaxDepenetrationVelocityAttr(self):  # noqa: N802 - mirrors the USD API
            return _FakeAttr(
                lambda: self._prim.max_depenetration,
                lambda value: setattr(self._prim, "max_depenetration", value),
            )

        def GetMaxDepenetrationVelocityAttr(self):  # noqa: N802 - mirrors the USD API
            return _FakeAttr(lambda: self._prim.max_depenetration)

    class _PhysxSceneAPI:
        def __init__(self, prim):
            self._prim = prim

        def _attr(self, name):
            return _FakeAttr(lambda: self._prim.scene_values[name])

        def GetMaxPositionIterationCountAttr(self):  # noqa: N802 - mirrors the USD API
            return self._attr("max_position_iteration_count")

        def GetMaxVelocityIterationCountAttr(self):  # noqa: N802 - mirrors the USD API
            return self._attr("max_velocity_iteration_count")

        def GetBounceThresholdAttr(self):  # noqa: N802 - mirrors the USD API
            return self._attr("bounce_threshold")

    return SimpleNamespace(
        UsdPhysics=_UsdPhysicsNS(_CollisionAPI, _RigidBodyAPI),
        PhysxSchema=_PhysxSchemaNS(_PhysxSceneAPI, _PhysxCollisionAPI, _PhysxRigidBodyAPI),
    )


class _UsdPhysicsNS:
    def __init__(self, collision_api, rigid_body_api):
        self.CollisionAPI = collision_api
        self.RigidBodyAPI = rigid_body_api


class _PhysxSchemaNS:
    def __init__(self, scene_api, collision_api, rigid_body_api):
        self.PhysxSceneAPI = scene_api
        self.PhysxCollisionAPI = collision_api
        self.PhysxRigidBodyAPI = rigid_body_api


def _fake_stage(monkeypatch):
    pxr = _fake_pxr()
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    scene_prim = _FakePrim(
        {pxr.PhysxSchema.PhysxSceneAPI},
        scene_values={
            "max_position_iteration_count": 8,
            "max_velocity_iteration_count": 1,
            "bounce_threshold": 0.20000000298023224,
        },
    )
    shapes = [
        _FakePrim({pxr.UsdPhysics.CollisionAPI}),
        _FakePrim({pxr.UsdPhysics.CollisionAPI}),
    ]
    bodies = [
        _FakePrim({pxr.UsdPhysics.RigidBodyAPI}),
        _FakePrim({pxr.UsdPhysics.RigidBodyAPI}),
    ]
    visual = _FakePrim(set())
    stage = SimpleNamespace(Traverse=lambda: [scene_prim, *shapes, *bodies, visual])
    return pxr, stage, shapes, bodies


def test_apply_and_read_back_contact_and_rest_offsets(monkeypatch):
    _pxr, stage, shapes, _bodies = _fake_stage(monkeypatch)
    assert apply_collision_offsets(stage, contact_offset=0.002, rest_offset=0.001) == 2
    values = read_engine_solver_values(
        stage, include_contact_offset=True, include_rest_offset=True
    )
    assert values == {
        "solver_position_iteration_count": 8,
        "solver_velocity_iteration_count": 1,
        "bounce_threshold_velocity": 0.20000000298023224,
        "contact_offset": 0.002,
        "rest_offset": 0.001,
    }
    assert all(shape.contact_offset == 0.002 for shape in shapes)
    assert all(shape.rest_offset == 0.001 for shape in shapes)


def test_apply_and_read_back_max_depenetration_velocity(monkeypatch):
    _pxr, stage, _shapes, bodies = _fake_stage(monkeypatch)
    assert apply_max_depenetration_velocity(stage, 1000.0) == 2
    values = read_engine_solver_values(
        stage, include_contact_offset=False, include_max_depenetration_velocity=True
    )
    assert values["max_depenetration_velocity"] == 1000.0
    assert "contact_offset" not in values
    assert "rest_offset" not in values
    assert all(body.max_depenetration == 1000.0 for body in bodies)


def test_apply_collision_offsets_requires_an_offset(monkeypatch):
    _pxr, stage, _shapes, _bodies = _fake_stage(monkeypatch)
    with pytest.raises(ValueError, match="at least one offset"):
        apply_collision_offsets(stage)


def test_read_back_rejects_nonuniform_contact_offsets(monkeypatch):
    _pxr, stage, shapes, _bodies = _fake_stage(monkeypatch)
    apply_collision_offsets(stage, contact_offset=0.002)
    shapes[1].contact_offset = 0.004
    with pytest.raises(RuntimeError, match="non-uniform contact offsets"):
        read_engine_solver_values(stage, include_contact_offset=True)


def test_read_back_rejects_missing_physx_collision_api(monkeypatch):
    pxr, stage, shapes, _bodies = _fake_stage(monkeypatch)
    apply_collision_offsets(stage, contact_offset=0.002)
    # A collision prim without the applied API would silently fall back to
    # the engine default; the readback must fail closed instead.
    shapes[1]._apis.discard(pxr.PhysxSchema.PhysxCollisionAPI)
    with pytest.raises(RuntimeError, match="PhysxCollisionAPI"):
        read_engine_solver_values(stage, include_contact_offset=True)


def test_read_back_rejects_unauthored_rest_offset(monkeypatch):
    _pxr, stage, shapes, _bodies = _fake_stage(monkeypatch)
    apply_collision_offsets(stage, contact_offset=0.002, rest_offset=0.001)
    shapes[1].rest_offset = None
    with pytest.raises(RuntimeError, match="no authored rest offset"):
        read_engine_solver_values(
            stage, include_contact_offset=True, include_rest_offset=True
        )


def test_read_back_rejects_nonuniform_max_depenetration_velocity(monkeypatch):
    _pxr, stage, _shapes, bodies = _fake_stage(monkeypatch)
    apply_max_depenetration_velocity(stage, 1000.0)
    bodies[1].max_depenetration = 100.0
    with pytest.raises(RuntimeError, match="non-uniform max depenetration velocities"):
        read_engine_solver_values(
            stage, include_contact_offset=False, include_max_depenetration_velocity=True
        )


def test_read_back_rejects_missing_physx_rigid_body_api(monkeypatch):
    pxr, stage, _shapes, bodies = _fake_stage(monkeypatch)
    apply_max_depenetration_velocity(stage, 1000.0)
    # A rigid body prim without the applied API would silently fall back to
    # the engine default; the readback must fail closed instead.
    bodies[1]._apis.discard(pxr.PhysxSchema.PhysxRigidBodyAPI)
    with pytest.raises(RuntimeError, match="PhysxRigidBodyAPI"):
        read_engine_solver_values(
            stage, include_contact_offset=False, include_max_depenetration_velocity=True
        )
