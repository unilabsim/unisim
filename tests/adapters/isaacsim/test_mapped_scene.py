"""Preflight and transaction checks independent of the optional Isaac SDK."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.isaacsim.backend import IsaacSimBackend, IsaacSimWorkerError
from unisim.backend.isaacsim.physx_solver import PhysxSolverConfig
from unisim.backend.isaacsim.scene_worker import (
    SceneWorkerContext,
    _assignment_groups,
    _consistent_collision_mask,
    _InitTelemetry,
    _native_geometry_columns,
    _prototype_spawn_paths,
    _record_collision_mask,
    _rotate,
    _validate_body_net_contact_entities,
    _validated_assignment,
    validate_scene_payload,
)
from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.backend import (
    MjcfSubprocessBackend,
    SubprocessWorkerError,
)
from unisim.backend.subprocess_ipc.scene_materialization import full_state_reset_patches
from unisim.dr.types import (
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_DOF_DAMPING,
    RESET_TERM_DOF_FRICTIONLOSS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_KD,
    RESET_TERM_KP,
    FixedVariantLayout,
    ResetRandomizationPayload,
)
from unisim.entities import SceneResetRequest
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, GeomLayout, JointLayout


def _payload():
    robot = EntityLayout(
        "robot", "articulation", "fixed", "base", ("base", "tip"), (0, 1), (None, "base"),
        (JointLayout("passive", "hinge", (0,), (0,), "tip"),), (), (), (),
        (),
        (),
        (GeomLayout("base::geom0", "base"), GeomLayout("tip::geom0", "tip")),
    )
    obj = EntityLayout(
        "object", "rigid", "floating", "box", ("box",), (2,), (None,), (), (), (), (),
        tuple(range(1, 8)), tuple(range(1, 7)),
        (GeomLayout("box", "box"),),
    )
    layout = CompiledSceneLayout((robot, obj), 8, 7, 0, 3, 3)
    entries = []
    for entity in layout.entities:
        n = len(entity.joints)
        record = {"joint_names": [joint.name for joint in entity.joints],
                  "body_names": list(entity.body_names), "actuator_names": [],
                  "actuator_joint_names": [],
                  "body_sphere_radii": [[] for _ in entity.body_names]}
        record["geom_names"] = [geom.name for geom in entity.geoms]
        record["geom_body_names"] = [geom.body_name for geom in entity.geoms]
        record["geom_contype"] = [1] * len(entity.geoms)
        record["geom_conaffinity"] = [1] * len(entity.geoms)
        record["geom_friction"] = [[0.5, 0.01, 0.0]] * len(entity.geoms)
        for field in ("dof_stiffness", "dof_damping", "dof_effort", "dof_armature",
                      "dof_friction", "dof_lower", "dof_upper"):
            record[field] = [0.0] * n
        record["body_mass"] = [1.0] * len(entity.body_names)
        record["body_ipos"] = [[0.0, 0.0, 0.0] for _ in entity.body_names]
        record["body_inertia"] = [[1.0, 1.0, 1.0] for _ in entity.body_names]
        record["body_iquat"] = [[1.0, 0.0, 0.0, 0.0] for _ in entity.body_names]
        entries.append({"name": entity.name, "kind": entity.kind,
                        "root_mode": entity.root_mode, "asset_format": "mjcf",
                        "collision_enabled": True, "mirror_of": None,
                        "self_collision": False,
                        "gravity_disabled": False,
                        "sources": ["source.xml"], "variants": [record], "assignment": [0, 0]})
    return {"scene_layout": layout.to_dict(), "num_envs": 2, "scene_entities": entries,
            "scene_content_identity": {
                "profile": "portable-mjcf-v1",
                "schema_version": 1,
                "source_identity": "a" * 64,
                "compiler_identity": "b" * 64,
                "canonical_identity": "c" * 64,
            },
            "initial_qpos": np.zeros((2, 8)).tolist(),
            "initial_qvel": np.zeros((2, 7)).tolist(),
            "initial_roots": np.zeros((2, 2, 13)).tolist()}


def test_passive_joint_has_state_but_no_control_and_unbounded_limits_are_valid():
    payload = _payload()
    record = payload["scene_entities"][0]["variants"][0]
    record["dof_lower"], record["dof_upper"] = [-np.inf], [np.inf]
    layout = validate_scene_payload(protocol, payload)
    assert layout.nv == 7 and layout.nu == 0


@pytest.mark.parametrize(
    "bad",
    [
        "format",
        "assignment_shape",
        "assignment_range",
        "assignment_type",
        "drive",
        "layout",
        "radii_shape",
        "radii_value",
        "geom_names",
        "geom_owner",
        "geom_mask",
        "geom_friction",
    ],
)
def test_unimplemented_or_inconsistent_requests_fail_before_kit(bad):
    payload = _payload()
    entity = payload["scene_entities"][0]
    if bad == "format":
        entity["asset_format"] = "urdf"
    elif bad == "assignment_shape":
        entity["sources"] *= 2
        entity["variants"] *= 2
        entity["assignment"] = [0, 1, 0]
    elif bad == "assignment_range":
        entity["sources"] *= 2
        entity["variants"] *= 2
        entity["assignment"] = [0, 2]
    elif bad == "assignment_type":
        entity["assignment"] = [0.5, 0]
    elif bad == "drive":
        entity["variants"][0]["dof_stiffness"] = [1.0]
    elif bad == "radii_shape":
        entity["variants"][0]["body_sphere_radii"] = [[]]
    elif bad == "radii_value":
        entity["variants"][0]["body_sphere_radii"] = [[-0.1]]
    elif bad == "geom_names":
        entity["variants"][0]["geom_names"] = ["wrong", "tip::geom0"]
    elif bad == "geom_owner":
        entity["variants"][0]["geom_body_names"] = ["tip", "tip"]
    elif bad == "geom_mask":
        entity["variants"][0]["geom_contype"] = [1, True]
    elif bad == "geom_friction":
        entity["variants"][0]["geom_friction"] = [[0.5, -0.1, 0.0], [0.5, 0.1, 0.0]]
    else:
        entity["variants"][0]["joint_names"] = ["wrong"]
    with pytest.raises((NotImplementedError, ValueError)):
        validate_scene_payload(protocol, payload)


def test_collision_pair_force_declarations_are_validated_before_kit():
    payload = _payload()
    payload["contact_force_sensors"] = [{
        "name": "tip_box", "source_entity": "robot", "source_body": "tip",
        "target_entity": "object", "target_body": "box",
    }]
    validate_scene_payload(protocol, payload)

    payload["contact_force_sensors"][0]["target_body"] = "missing"
    with pytest.raises(ValueError, match="unknown target entity/body"):
        validate_scene_payload(protocol, payload)


def test_self_collision_flag_is_validated_before_kit():
    payload = _payload()
    payload["scene_entities"][0]["self_collision"] = True
    validate_scene_payload(protocol, payload)

    for bad, error in (
        (1, "self_collision must be bool"),
        (None, "self_collision must be bool"),
    ):
        payload["scene_entities"][0]["self_collision"] = bad
        with pytest.raises(TypeError, match=error):
            validate_scene_payload(protocol, payload)

    payload["scene_entities"][0]["self_collision"] = True
    payload["scene_entities"][0]["collision_enabled"] = False
    with pytest.raises(ValueError, match="collision-enabled physical entity"):
        validate_scene_payload(protocol, payload)

    payload = _payload()
    payload["scene_entities"][1]["self_collision"] = True
    with pytest.raises(NotImplementedError, match="requires an articulation"):
        validate_scene_payload(protocol, payload)


def test_gravity_disabled_flag_is_validated_before_kit():
    payload = _payload()
    for requested in (True, False):
        payload["scene_entities"][0]["gravity_disabled"] = requested
        validate_scene_payload(protocol, payload)
    for bad in (1, None):
        payload["scene_entities"][0]["gravity_disabled"] = bad
        with pytest.raises(TypeError, match="gravity_disabled must be bool"):
            validate_scene_payload(protocol, payload)


def test_host_resolves_unset_gravity_requests_to_the_implicit_role_default():
    entries = [
        {"name": "robot", "kind": "articulation", "root_mode": "fixed",
         "gravity_disabled": None},
        {"name": "object", "kind": "rigid", "root_mode": "floating",
         "gravity_disabled": None},
        {"name": "table", "kind": "rigid", "root_mode": "fixed",
         "gravity_disabled": None},
        {"name": "mirror", "kind": "rigid", "root_mode": "kinematic",
         "gravity_disabled": None},
        {"name": "kept", "kind": "rigid", "root_mode": "fixed",
         "gravity_disabled": False},
        {"name": "requested", "kind": "articulation", "root_mode": "floating",
         "gravity_disabled": True},
    ]
    prepared = SimpleNamespace(payload={"scene_entities": entries})
    IsaacSimBackend._resolve_worker_entity_gravity(IsaacSimBackend.__new__(IsaacSimBackend),
                                                   prepared)
    assert [entry["gravity_disabled"] for entry in entries] == [
        False, False, True, True, False, True,
    ]


def test_host_strictly_compares_reported_per_entity_gravity_disabled():
    payload = _payload()
    payload["scene_entities"][0]["gravity_disabled"] = True
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._entity_scene = SimpleNamespace(payload=payload)
    meta = {
        "configuration_report": {
            "schema_version": 1,
            "effective": {
                "entity_gravity_disabled": {"robot": True, "object": False},
            },
        }
    }
    backend._validate_reported_entity_gravity_disabled(meta)

    for reported in (
        {"robot": False, "object": False},
        {"robot": True},
        {"robot": True, "object": 1},
        None,
    ):
        broken = copy.deepcopy(meta)
        broken["configuration_report"]["effective"]["entity_gravity_disabled"] = reported
        with pytest.raises(IsaacSimWorkerError, match="gravity"):
            backend._validate_reported_entity_gravity_disabled(broken)


def test_arbitrary_immutable_assignment_is_accepted_before_kit():
    payload = _payload()
    entity = payload["scene_entities"][0]
    entity["sources"] *= 2
    entity["variants"] *= 2
    entity["assignment"] = [1, 1, 0, 1, 0]
    payload["scene_entities"][1]["assignment"] = [0, 0, 0, 0, 0]
    payload["num_envs"] = 5
    payload["initial_qpos"] = np.zeros((5, 8)).tolist()
    payload["initial_qvel"] = np.zeros((5, 7)).tolist()
    payload["initial_roots"] = np.zeros((5, 2, 13)).tolist()
    validate_scene_payload(protocol, payload)
    np.testing.assert_array_equal(_validated_assignment(entity, 5), [1, 1, 0, 1, 0])


def test_exact_assignment_keeps_prototypes_and_copies_independent():
    component = "entity_object"
    env_paths = ["/World/envs/env_0", "/World/envs/env_1", "/World/envs/env_2"]
    assignment = np.asarray([1, 1, 0])
    prototypes = _prototype_spawn_paths(component, 2)
    groups = _assignment_groups(assignment, 2, env_paths)
    assert prototypes == [
        "/World/unisim_prototypes/entity_object/entity_object_0",
        "/World/unisim_prototypes/entity_object/entity_object_1",
    ]
    assert groups == (("/World/envs/env_2",), ("/World/envs/env_0", "/World/envs/env_1"))


class _FakeSdfAttributeSpec:
    def __init__(self, owner, name, _type):
        self.default = None
        owner.attrs[name] = self


class _FakeSdfPrimSpec:
    def __init__(self, path):
        self.path = path
        self.attrs = {}

    def GetAttributeAtPath(self, full_path):  # noqa: N802 - mirrors the USD API
        name = str(full_path).split(".")[-1]
        return self.attrs.get(name)


def _fake_pxr_modules(monkeypatch, translation=(1.0, 2.0, 3.0)):
    """Install minimal pxr/carb fakes that record the batched copy edits."""
    import sys
    import types

    local_to_world = np.eye(4)
    local_to_world[:3, 3] = translation
    state = {"copy_specs": [], "specs": {}, "removed": [], "op_order": None}

    class FakeVec3d:
        def __init__(self, *values):
            self.values = tuple(values)

    class FakeQuat:
        def __init__(self, value):
            self.value = value

    class FakeTransform:
        def SetMatrix(self, matrix):  # noqa: N802 - mirrors the USD API
            self._matrix = np.asarray(matrix.values, dtype=np.float64)

        def GetTranslation(self):  # noqa: N802 - mirrors the USD API
            return FakeVec3d(*self._matrix[:3, 3])

        def GetRotation(self):  # noqa: N802 - mirrors the USD API
            return types.SimpleNamespace(GetQuat=lambda: ("quat", 1.0, 0.0, 0.0, 0.0))

    class FakeMatrix4d:
        def __init__(self, values):
            self.values = values

    class FakeUsdAttr:
        def __init__(self, name):
            self.name = name
            self.value = None

    class FakePrim:
        def __init__(self):
            self.attrs = {
                "xformOp:translate": FakeUsdAttr("xformOp:translate"),
                "xformOp:orient": FakeUsdAttr("xformOp:orient"),
                "xformOp:transform": FakeUsdAttr("xformOp:transform"),
            }

        def __bool__(self):
            return True

        def GetPropertyNames(self):  # noqa: N802 - mirrors the USD API
            return list(self.attrs)

        def GetAttribute(self, name):  # noqa: N802 - mirrors the USD API
            return self.attrs[name]

        def RemoveProperty(self, name):  # noqa: N802 - mirrors the USD API
            state["removed"].append(name)
            del self.attrs[name]

    class FakeXformOp:
        TypeTranslate = "translate"
        TypeOrient = "orient"
        TypeScale = "scale"
        PrecisionDouble = "double"
        PrecisionFloat = "float"

        def __init__(self, attr=None, *_args):
            self._attr = attr if attr is not None else FakeUsdAttr("op")

        def Set(self, value):  # noqa: N802 - mirrors the USD API
            self._attr.value = value

        def GetPrecision(self):  # noqa: N802 - mirrors the USD API
            return self.PrecisionDouble

    class FakeXformable:
        def __init__(self, prim):
            self._prim = prim

        def ComputeParentToWorldTransform(self, _time):  # noqa: N802 - mirrors the USD API
            return np.eye(4)

        def ComputeLocalToWorldTransform(self, _time):  # noqa: N802 - mirrors the USD API
            return local_to_world

        def ClearXformOpOrder(self):  # noqa: N802 - mirrors the USD API
            pass

        def AddXformOp(self, op_type, _precision, _suffix):  # noqa: N802 - mirrors the USD API
            return FakeXformOp(FakeUsdAttr(op_type))

        def SetXformOpOrder(self, order):  # noqa: N802 - mirrors the USD API
            state["op_order"] = order

    class FakeSdfPath:
        def __init__(self, path):
            self.path = path

    class FakeLayer:
        pass

    root_layer = FakeLayer()

    pxr = types.ModuleType("pxr")
    gf = types.SimpleNamespace(
        Transform=FakeTransform,
        Matrix4d=FakeMatrix4d,
        Vec3d=FakeVec3d,
        Quatf=FakeQuat,
        Quatd=FakeQuat,
    )
    def _create_prim_in_layer(_layer, path):
        spec = _FakeSdfPrimSpec(path)
        state["specs"][path] = spec
        return spec

    sdf = types.SimpleNamespace(
        CreatePrimInLayer=_create_prim_in_layer,
        CopySpec=lambda src_layer, src, dst_layer, dst: state["copy_specs"].append(
            (src_layer, src.path, dst_layer, dst.path)
        ),
        Path=FakeSdfPath,
        AttributeSpec=_FakeSdfAttributeSpec,
        ValueTypeNames=types.SimpleNamespace(
            Double3="double3", Quatf="quatf", Quatd="quatd", TokenArray="token[]"
        ),
    )
    usd = types.SimpleNamespace(TimeCode=types.SimpleNamespace(Default=lambda: None))
    usd_geom = types.SimpleNamespace(
        Xformable=FakeXformable,
        XformOp=FakeXformOp,
        Tokens=types.SimpleNamespace(xformOpOrder="xformOpOrder"),
    )
    vt = types.SimpleNamespace(TokenArray=lambda values: list(values))
    pxr.Gf, pxr.Sdf, pxr.Usd, pxr.UsdGeom, pxr.Vt = gf, sdf, usd, usd_geom, vt
    carb = types.ModuleType("carb")
    carb_settings = types.ModuleType("carb.settings")
    carb_settings.get_settings = lambda: types.SimpleNamespace(get_as_string=lambda _k: "")
    carb.settings = carb_settings
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "carb.settings", carb_settings)

    class FakeStage:
        def GetPrimAtPath(self, path):  # noqa: N802 - mirrors the USD API
            return FakePrim() if path == "/proto" else None

        def GetRootLayer(self):  # noqa: N802 - mirrors the USD API
            return root_layer

    return state, FakeStage(), root_layer, FakeQuat


def test_batched_copy_spec_mirrors_cloner_per_destination_writes(monkeypatch):
    from unisim.backend.isaacsim.scene_worker import _copy_prims_from_source

    state, stage, root_layer, fake_quat = _fake_pxr_modules(monkeypatch)
    destinations = ("/World/envs/env_0/object", "/World/envs/env_1/object")
    _copy_prims_from_source(stage, "/proto", destinations)

    assert state["copy_specs"] == [
        (root_layer, "/proto", root_layer, "/World/envs/env_0/object"),
        (root_layer, "/proto", root_layer, "/World/envs/env_1/object"),
    ]
    assert state["removed"] == ["xformOp:transform"]
    assert len(state["op_order"]) == 3
    # Each destination keeps the source local transform and canonical op order.
    for destination in destinations:
        attrs = state["specs"][destination].attrs
        assert tuple(attrs["xformOp:translate"].default.values) == (1.0, 2.0, 3.0)
        assert isinstance(attrs["xformOp:orient"].default, fake_quat)
        assert tuple(attrs["xformOp:scale"].default.values) == (1.0, 1.0, 1.0)
        assert attrs["xformOpOrder"].default == [
            "xformOp:translate", "xformOp:orient", "xformOp:scale",
        ]


def test_batched_copy_spec_skips_source_and_fails_closed_on_missing_source(monkeypatch):
    from unisim.backend.isaacsim.scene_worker import _copy_prims_from_source

    state, stage, _, _ = _fake_pxr_modules(monkeypatch)
    _copy_prims_from_source(stage, "/proto", ("/proto", "/World/envs/env_0/object"))
    assert [entry[3] for entry in state["copy_specs"]] == ["/World/envs/env_0/object"]
    with pytest.raises(RuntimeError, match="prototype is missing"):
        _copy_prims_from_source(stage, "/missing", ("/World/envs/env_0/object",))


def test_host_rejects_corrupt_worker_sphere_geometry_readback():
    payload = _payload()
    backend = MjcfSubprocessBackend.__new__(MjcfSubprocessBackend)
    backend._entity_scene = SimpleNamespace(
        layout=CompiledSceneLayout.from_dict(payload["scene_layout"]), payload=payload
    )
    backend._num_envs = 2
    backend._base_name = None
    backend._bind_entity_query_maps()
    layout = backend._entity_scene.layout
    metadata = {
        "scene_layout": layout.to_dict(),
        "gravity": [0.0, 0.0, -9.81],
        "scene_entities_actual": [
            {
                "name": "robot",
                "assignment": [0, 0],
                "body_mass": [[1.0, 1.0], [1.0, 1.0]],
                "body_sphere_radii": [[[], []], [[], []]],
            },
            {
                "name": "object",
                "assignment": [0, 0],
                "body_mass": [[1.0], [1.0]],
                "body_sphere_radii": [[[0.2]], [[]]],
            },
        ],
    }
    with pytest.raises(SubprocessWorkerError, match="sphere radii differ.*object"):
        backend._bind_scene_metadata(metadata)


def test_host_strictly_compares_reported_per_entity_self_collision():
    payload = _payload()
    payload["scene_entities"][0]["self_collision"] = True
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._entity_scene = SimpleNamespace(payload=payload)
    meta = {
        "configuration_report": {
            "schema_version": 1,
            "effective": {
                "collision_filter": {
                    "self_collision": {"robot": True, "object": False},
                    "environment_isolation": True,
                    "implicit_ground": False,
                }
            },
        }
    }
    backend._validate_reported_entity_self_collision(meta)

    for reported in (
        {"robot": False, "object": False},
        {"robot": True},
        {"robot": True, "object": 1},
        None,
    ):
        broken = copy.deepcopy(meta)
        broken["configuration_report"]["effective"]["collision_filter"][
            "self_collision"
        ] = reported
        with pytest.raises(IsaacSimWorkerError, match="self-collision"):
            backend._validate_reported_entity_self_collision(broken)


def _context():
    ctx = SceneWorkerContext.__new__(SceneWorkerContext)
    ctx.layout = validate_scene_payload(protocol, _payload())
    ctx.num_envs = 2
    ctx.sim_dt = 0.002
    ctx.legacy_projection = None
    ctx.net_contact_entities = []
    ctx.net_contact_views = []
    ctx.net_contact_maps = []
    ctx.slots = {name: np.zeros(shape, dtype=protocol.slot_dtype(name))
                 for name, shape in protocol.scene_slot_shapes(2, ctx.layout).items()}
    ctx.slots["reset_env_ids"][:] = [1, 0]
    ctx.slots["reset_entity_root_state"][:, :, 3] = 1
    ctx.slots["reset_qpos"][:, 4] = 1
    ctx.slots["reset_root_mask"][1] = 1
    ctx.slots["reset_qpos_mask"][1:] = 1
    ctx.slots["reset_qvel_mask"][1:] = 1
    return ctx


def _property_context():
    ctx = _context()
    ctx.faulted = False
    ctx.device = "cpu"
    ctx.torch = SimpleNamespace(
        as_tensor=lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype),
        long=np.int64,
        int32=np.int32,
    )
    ctx._tensor = lambda value: value
    ctx._cpu_tensor = lambda value: value
    commits = []
    ctx._commit = lambda *args, **kwargs: commits.append(args)
    ctx.refresh_state_slots = lambda: None

    robot_mass = np.asarray([[10, 11], [20, 21]], dtype=np.float32)
    object_mass = np.asarray([[30], [40]], dtype=np.float32)
    robot_material = np.asarray(
        [
            [[0.1, 0.1, 0.0], [0.2, 0.2, 0.0]],
            [[0.3, 0.3, 0.0], [0.4, 0.4, 0.0]],
        ],
        dtype=np.float32,
    )
    object_material = np.asarray(
        [[[0.5, 0.5, 0.0]], [[0.6, 0.6, 0.0]]], dtype=np.float32
    )
    setter_ids = []

    def make_asset(masses, materials, name, links, joints=0):
        coms = np.zeros((2, links, 7), dtype=np.float32)
        coms[..., 3] = 1.0
        inertias = np.zeros((2, links, 9), dtype=np.float32)
        inertias[..., [0, 4, 8]] = 1.0
        view = SimpleNamespace(
            materials=materials, masses=masses, coms=coms, inertias=inertias
        )

        def set_rows(attr, label):
            def setter(values, *, indices):
                rows = np.asarray(indices)
                assert rows.dtype == np.int32
                values = np.asarray(values)
                getattr(view, attr)[rows] = values[rows]
                setter_ids.append((name, label, rows.tolist()))

            return setter

        def set_materials(values, *, indices):
            rows = np.asarray(indices)
            assert rows.dtype == np.int32
            values = np.asarray(values)
            view.materials[rows] = values[rows]
            setter_ids.append((name, "material", rows.tolist()))

        view.get_masses = lambda: view.masses.copy()
        view.set_masses = set_rows("masses", "mass")
        view.get_coms = lambda: view.coms.copy()
        view.set_coms = set_rows("coms", "com")
        view.get_inertias = lambda: view.inertias.copy()
        view.set_inertias = set_rows("inertias", "inertia")
        view.get_material_properties = lambda: view.materials.copy()
        view.set_material_properties = set_materials
        asset = SimpleNamespace(root_physx_view=view)
        for field, label, getter in (
            ("dof_stiffness", "stiffness", "get_dof_stiffnesses"),
            ("dof_damping", "damping", "get_dof_dampings"),
            ("dof_armature", "armature", "get_dof_armatures"),
            ("dof_friction", "friction", "get_dof_friction_coefficients"),
        ):
            store = np.zeros((2, joints), dtype=np.float32)
            setattr(view, field, store)
            setattr(view, getter, lambda store=store: store.copy())

            def write_joint(values, *, joint_ids, env_ids, store=store, label=label):
                rows = np.asarray(env_ids)
                store[np.ix_(rows, np.asarray(joint_ids))] = np.asarray(values)
                setter_ids.append((name, label, rows.tolist()))

            setattr(asset, f"write_joint_{label}_to_sim", write_joint)
        asset.write_joint_friction_coefficient_to_sim = (
            asset.write_joint_friction_to_sim
        )
        return asset

    ctx.assets = [
        make_asset(robot_mass, robot_material, "robot", 2, joints=1),
        make_asset(object_mass, object_material, "object", 1),
    ]
    ctx.maps = [
        {
            "envs": np.array([1, 0]),
            "bodies": np.array([0, 1]),
            "geoms": np.array([0, 1]),
            "joints": np.array([0]),
            "controls": np.array([], dtype=np.int64),
        },
        {
            "envs": np.array([0, 1]),
            "bodies": np.array([0]),
            "geoms": np.array([0]),
            "joints": np.array([], dtype=np.int64),
            "controls": np.array([], dtype=np.int64),
        },
    ]
    ctx.entries = _payload()["scene_entities"]
    ctx._drive_damping = [np.zeros((2, 1), dtype=np.float32), None]
    ctx._natural_damping = [np.zeros((2, 1), dtype=np.float32), None]
    ctx.actual = [
        {"name": "robot", "body_mass": [[20, 21], [10, 11]].copy()},
        {"name": "object", "body_mass": [[30], [40]].copy()},
    ]
    return ctx, commits, setter_ids


def test_filtered_contact_force_refresh_scatters_last_substep_rows_by_environment():
    ctx = _context()
    ctx.contact_force_sensors = [
        {"name": "first", "source_entity": "robot", "source_body": "tip",
         "target_entity": "object", "target_body": "box"},
        {"name": "second", "source_entity": "robot", "source_body": "tip",
         "target_entity": "object", "target_body": "box"},
    ]
    shapes = protocol.scene_slot_shapes(ctx.num_envs, ctx.layout, 2)
    ctx.slots["contact_sensor_force"] = np.zeros(
        shapes["contact_sensor_force"], dtype=protocol.slot_dtype("contact_sensor_force")
    )
    first = np.arange(6, dtype=np.float32).reshape(2, 1, 1, 3)
    second = np.arange(6, 12, dtype=np.float32).reshape(2, 1, 1, 3)
    ctx.contact_sensors = [
        SimpleNamespace(data=SimpleNamespace(force_matrix_w=first)),
        SimpleNamespace(data=SimpleNamespace(force_matrix_w=second)),
    ]
    ctx.contact_sensor_maps = [
        {"envs": np.array([1, 0])},
        {"envs": np.array([0, 1])},
    ]
    ctx._refresh_contact_sensor_forces()
    np.testing.assert_array_equal(
        ctx.slots["contact_sensor_force"][:, 0],
        np.array([[3, 4, 5], [0, 1, 2]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        ctx.slots["contact_sensor_force"][:, 1],
        np.array([[6, 7, 8], [9, 10, 11]], dtype=np.float32),
    )


def test_reset_clears_filtered_contact_forces_without_reporting_stale_contacts():
    ctx = _context()
    ctx.assets = []
    ctx.maps = []
    shape = protocol.scene_slot_shapes(ctx.num_envs, ctx.layout, 1)[
        "contact_sensor_force"
    ]
    ctx.slots["contact_sensor_force"] = np.full(shape, 7.0, dtype=np.float32)
    ctx.refresh_state_slots()
    assert np.all(ctx.slots["contact_sensor_force"] == 0.0)


def test_body_net_contact_entity_payload_validation():
    layout = validate_scene_payload(protocol, _payload())
    assert _validate_body_net_contact_entities({"body_net_contact_entities": []}, layout) == []
    assert _validate_body_net_contact_entities(
        {"body_net_contact_entities": ["object", "robot"]}, layout
    ) == ["object", "robot"]
    with pytest.raises(ValueError, match="non-empty strings"):
        _validate_body_net_contact_entities({"body_net_contact_entities": [""]}, layout)
    with pytest.raises(ValueError, match="duplicates"):
        _validate_body_net_contact_entities(
            {"body_net_contact_entities": ["robot", "robot"]}, layout
        )
    with pytest.raises(ValueError, match="unknown entities"):
        _validate_body_net_contact_entities({"body_net_contact_entities": ["ghost"]}, layout)


def test_contact_declarations_on_self_collision_entities_fail_closed():
    payload = _payload()
    payload["scene_entities"][0]["self_collision"] = True
    layout = validate_scene_payload(protocol, payload)
    with pytest.raises(ValueError, match="self-collision entities"):
        _validate_body_net_contact_entities(
            {**payload, "body_net_contact_entities": ["robot"]}, layout
        )
    # Unrelated entities keep their body-net coverage.
    assert _validate_body_net_contact_entities(
        {**payload, "body_net_contact_entities": ["object"]}, layout
    ) == ["object"]
    pair = {
        "name": "self",
        "source_entity": "robot",
        "source_body": "base",
        "target_entity": "robot",
        "target_body": "tip",
    }
    with pytest.raises(ValueError, match="same-entity pair on a self-collision entity"):
        validate_scene_payload(protocol, {**payload, "contact_force_sensors": [pair]})
    # Cross-entity pairs and same-entity pairs without self-collision pass.
    cross = {**pair, "name": "cross", "target_entity": "object", "target_body": "box"}
    validate_scene_payload(protocol, {**payload, "contact_force_sensors": [cross]})
    plain = _payload()
    validate_scene_payload(protocol, {**plain, "contact_force_sensors": [pair]})


def test_body_net_contact_force_refresh_scatters_flat_rows_by_environment_and_body():
    ctx = _context()
    # robot owns public body ids (0, 1) for bodies (base, tip); the fake
    # reporter exposes four flat rows in a permuted (env, body) order.
    native = np.array(
        [[10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]],
        dtype=np.float32,
    )
    ctx.net_contact_views = [
        SimpleNamespace(get_net_contact_forces=lambda dt: native)
    ]
    ctx.net_contact_maps = [
        {
            "env": np.array([1, 1, 0, 0]),
            "body": np.array([1, 0, 1, 0]),
            "entity": "robot",
            "count": 4,
        }
    ]
    ctx._refresh_body_net_contact_forces()
    np.testing.assert_array_equal(
        ctx.slots["contact_force"][:, 0, :], [[40, 0, 0], [20, 0, 0]],  # base
    )
    np.testing.assert_array_equal(
        ctx.slots["contact_force"][:, 1, :], [[30, 0, 0], [10, 0, 0]],  # tip
    )
    assert np.all(ctx.slots["contact_force"][:, 2, :] == 0.0)  # unrequested entity


def test_body_net_contact_force_refresh_rejects_malformed_reporter_output():
    ctx = _context()
    ctx.net_contact_maps = [
        {
            "env": np.array([0, 0, 1, 1]),
            "body": np.array([0, 1, 0, 1]),
            "entity": "robot",
            "count": 4,
        }
    ]
    ctx.net_contact_views = [
        SimpleNamespace(
            get_net_contact_forces=lambda dt: np.zeros((3, 3), dtype=np.float32)
        )
    ]
    with pytest.raises(RuntimeError, match="returned shape"):
        ctx._refresh_body_net_contact_forces()
    bad = np.zeros((4, 3), dtype=np.float32)
    bad[0, 0] = np.nan
    ctx.net_contact_views = [SimpleNamespace(get_net_contact_forces=lambda dt: bad)]
    with pytest.raises(RuntimeError, match="non-finite force"):
        ctx._refresh_body_net_contact_forces()


def test_body_net_contact_forces_poll_every_substep_and_publish_the_last():
    ctx = _context()
    ctx.assets = []
    ctx.maps = []
    ctx.sim = SimpleNamespace(step=lambda render: None)
    ctx.sim_dt = 0.002
    ctx.contact_force_sensors = []
    ctx.contact_sensors = []
    ctx.contact_sensor_maps = []
    calls = []
    values = [np.zeros((4, 3), dtype=np.float32)]

    class View:
        def get_net_contact_forces(self, dt: float):
            calls.append(dt)
            values[0] = values[0] + np.array([[1, 0, 0]], dtype=np.float32)
            return values[0]

    ctx.net_contact_views = [View()]
    ctx.net_contact_maps = [
        {
            "env": np.array([0, 0, 1, 1]),
            "body": np.array([0, 1, 0, 1]),
            "entity": "robot",
            "count": 4,
        }
    ]
    ctx.step({"nsteps": 2})
    # PhysX GPU zeroes a body's entry only on the exact substep where contact
    # is lost (IsaacLab #7613), so the worker polls every substep and then
    # reads once more to publish the final substep.
    assert calls == [0.002, 0.002, 0.002]
    np.testing.assert_array_equal(
        ctx.slots["contact_force"][:, 0, :], [[3, 0, 0], [3, 0, 0]]
    )


def test_refresh_state_slots_clears_contact_sensor_force_slot():
    ctx = _context()
    ctx.assets = []
    ctx.maps = []
    shape = protocol.scene_slot_shapes(ctx.num_envs, ctx.layout, 1)[
        "contact_sensor_force"
    ]
    ctx.slots["contact_sensor_force"] = np.full(shape, 7.0, dtype=np.float32)
    ctx.refresh_state_slots()
    assert np.all(ctx.slots["contact_sensor_force"] == 0.0)


def test_contact_sensors_update_after_each_physics_substep_and_publish_the_last():
    ctx = _context()
    ctx.assets = []
    ctx.maps = []
    ctx.sim = SimpleNamespace(step=lambda render: None)
    ctx.sim_dt = 0.002
    ctx.contact_force_sensors = [{
        "name": "tip_box", "source_entity": "robot", "source_body": "tip",
        "target_entity": "object", "target_body": "box",
    }]
    shape = protocol.scene_slot_shapes(ctx.num_envs, ctx.layout, 1)[
        "contact_sensor_force"
    ]
    ctx.slots["contact_sensor_force"] = np.zeros(shape, dtype=np.float32)
    values = [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32).reshape(2, 1, 1, 3)]

    class Sensor:
        def update(self, dt: float) -> None:
            values.append(
                (values[-1].reshape(2, 3) + np.array([[7, 8, 9]], dtype=np.float32)).reshape(
                    2, 1, 1, 3
                )
            )

        @property
        def data(self):
            return SimpleNamespace(force_matrix_w=values[-1])

    ctx.contact_sensors = [Sensor()]
    ctx.contact_sensor_maps = [{"envs": np.array([0, 1])}]
    ctx.step({"nsteps": 2})
    np.testing.assert_array_equal(
        ctx.slots["contact_sensor_force"][:, 0],
        [[15, 18, 21], [18, 21, 24]],
    )


def _readback_backend(records):
    payload = _payload()
    layout = validate_scene_payload(protocol, payload)
    robot = replace(layout.entities[0], body_ids=(1, 2))
    obj = replace(layout.entities[1], body_ids=(0,))
    layout = replace(layout, entities=(robot, obj), nbody=4)
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._num_envs = 2
    backend._model_info = object()
    backend._entity_scene = SimpleNamespace(
        layout=layout,
        payload=payload,
        owner=SimpleNamespace(
            variant_plan=None,
            model=SimpleNamespace(
                body_mass=np.array([100, 101, 102, 103], dtype=np.float32),
                body_ipos=np.arange(12, dtype=np.float32).reshape(4, 3) / 7,
                body_inertia=np.full((4, 3), 0.5, dtype=np.float32),
            ),
        ),
    )
    backend._entity_primary_index = 0
    backend._native_entity_records = records
    for entity in layout.entities:
        record = records.setdefault(entity.name, {})
        count = len(entity.geoms)
        record.setdefault(
            "geom_names", [[geom.name for geom in entity.geoms]] * backend._num_envs
        )
        record.setdefault(
            "geom_body_names",
            [[geom.body_name for geom in entity.geoms]] * backend._num_envs,
        )
        record.setdefault(
            "geom_contact_masks", [[[1, 1]] * count for _ in range(backend._num_envs)]
        )
        record.setdefault(
            "geom_friction",
            [[[0.5, 0.5, 0.0]] * count for _ in range(backend._num_envs)],
        )
    return backend


def test_mapped_native_body_mass_is_scattered_to_public_body_order_and_detached():
    records = {
        "robot": {"body_mass": [[10, 11], [20, 21]]},
        "object": {"body_mass": [[30], [40]]},
    }
    backend = _readback_backend(records)
    masses = backend.get_body_mass()
    np.testing.assert_array_equal(masses, [[30, 10, 11, 103], [40, 20, 21, 103]])
    masses[:] = 0
    np.testing.assert_array_equal(records["robot"]["body_mass"], [[10, 11], [20, 21]])


def test_mapped_native_body_ipos_selection_preserves_order_duplicates_and_empty_rows():
    entity_coms = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    public_coms = np.empty((2, 4, 3), dtype=np.float32)
    public_coms[:] = np.arange(12, dtype=np.float32).reshape(4, 3) / 7
    public_coms[:, (1, 2)] = entity_coms[:, :2]
    public_coms[:, 0] = entity_coms[:, 2]
    records = {
        "robot": {"body_com": entity_coms[:, :2].tolist()},
        "object": {"body_com": entity_coms[:, 2:].tolist()},
    }
    backend = _readback_backend(records)
    selected = backend.get_body_ipos(env_ids=[1, 0, 1])
    np.testing.assert_array_equal(selected, public_coms[[1, 0, 1]])
    assert backend.get_body_ipos(env_ids=[]).shape == (0, 4, 3)
    selected[:] = 0
    np.testing.assert_array_equal(
        np.asarray(records["robot"]["body_com"]), entity_coms[:, :2]
    )


def test_mapped_canonical_body_ipos_is_a_detached_compiled_default_table():
    source = np.arange(12, dtype=np.float32).reshape(4, 3) / 7
    backend = _readback_backend({})
    defaults = backend.get_body_ipos()
    np.testing.assert_allclose(defaults, source)
    defaults[:] = -1
    np.testing.assert_allclose(backend._entity_scene.owner.model.body_ipos, source)


def test_mapped_geometry_names_and_ownership_follow_noncontiguous_public_order():
    backend = _readback_backend({})
    assert backend.get_geom_names() == (
        "robot/base::geom0",
        "robot/tip::geom0",
        "object/box",
    )
    np.testing.assert_array_equal(backend.get_geom_body_ids(), [1, 2, 0])


def test_mapped_native_geometry_masks_and_friction_scatter_per_environment():
    records = {
        "robot": {
            "geom_contact_masks": [
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
            ],
            "geom_friction": [
                [[0.1, 0.2, 0.0], [0.3, 0.4, 0.0]],
                [[0.5, 0.6, 0.0], [0.7, 0.8, 0.0]],
            ],
        },
        "object": {
            "geom_contact_masks": [[[0, 0]], [[0, 0]]],
            "geom_friction": [[[0.9, 1.0, 0.0]], [[1.1, 1.2, 0.0]]],
        },
    }
    backend = _readback_backend(records)
    contype, conaffinity = backend.get_geom_contact_masks()
    np.testing.assert_array_equal(contype, [1, 1, 0])
    np.testing.assert_array_equal(conaffinity, [1, 1, 0])
    friction = backend.get_geom_friction()
    np.testing.assert_allclose(
        friction,
        [
            [[0.1, 0.2, 0.0], [0.3, 0.4, 0.0], [0.9, 1.0, 0.0]],
            [[0.5, 0.6, 0.0], [0.7, 0.8, 0.0], [1.1, 1.2, 0.0]],
        ],
    )
    friction[:] = 0.0
    np.testing.assert_allclose(
        records["robot"]["geom_friction"][1], [[0.5, 0.6, 0.0], [0.7, 0.8, 0.0]]
    )


def test_mapped_empty_geometry_layout_exposes_empty_readback_records():
    backend = _readback_backend({})
    scene = backend._entity_scene
    scene.layout = replace(  # type: ignore[attr-defined]
        scene.layout,
        entities=tuple(replace(entity, geoms=()) for entity in scene.layout.entities),
        ngeom=0,
    )
    backend._native_entity_records = {
        entity.name: {
            "geom_names": [[] for _ in range(2)],
            "geom_body_names": [[] for _ in range(2)],
            "geom_contact_masks": [[] for _ in range(2)],
            "geom_friction": [[] for _ in range(2)],
        }
        for entity in scene.layout.entities
    }
    assert backend.get_geom_names() == ()
    assert backend.get_geom_body_ids().shape == (0,)
    assert tuple(array.shape for array in backend.get_geom_contact_masks()) == ((0,), (0,))
    assert backend.get_geom_friction().shape == (2, 0, 3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("body_mass", None),
        ("body_mass", [[10, 11], [20]]),
        ("body_mass", [[10, np.nan], [20, 21]]),
        ("body_com", None),
        ("body_com", [[[0, 0, 0], [1, 1, 1]], [[2, 2, 2]]]),
        ("body_com", [[[0, 0, np.inf], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]]),
    ],
)
def test_mapped_native_property_records_fail_closed(field, value):
    records = {
        "robot": {
            "body_mass": [[10, 11], [20, 21]],
            "body_com": [[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]],
        },
        "object": {"body_mass": [[30], [40]], "body_com": [[[4, 4, 4]], [[5, 5, 5]]]},
    }
    if value is None:
        del records["object"][field]
    else:
        records["object"][field] = value
    backend = _readback_backend(records)
    with pytest.raises(IsaacSimWorkerError, match=f"native {field}.*object"):
        backend.get_body_mass() if field == "body_mass" else backend.get_body_ipos(env_ids=[0])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("geom_names", None),
        ("geom_names", ["wrong"]),
        ("geom_body_names", None),
        ("geom_body_names", ["wrong"]),
        ("geom_contact_masks", None),
        ("geom_contact_masks", [[[1, 1]]]),
        ("geom_contact_masks", [[[2, 1]], [[1, 1]]]),
        ("geom_contact_masks", [[[0, 0]], [[1, 1]]]),
        ("geom_friction", None),
        ("geom_friction", [[[0.5, 0.5, 0.0]]]),
        ("geom_friction", [[[0.5, -0.5, 0.0]], [[0.5, 0.5, 0.0]]]),
    ],
)
def test_mapped_native_geometry_records_fail_closed(field, value):
    backend = _readback_backend({})
    record = backend._native_entity_records["object"]
    if value is None:
        del record[field]
    else:
        record[field] = value
    method = backend.get_geom_friction if field == "geom_friction" else (
        backend.get_geom_contact_masks
    )
    with pytest.raises(IsaacSimWorkerError, match=f"{field}.*object"):
        method()


def test_body_property_readback_requires_mapped_scene_and_keeps_other_properties_unsupported():
    backend = _readback_backend({})
    backend._entity_scene = None
    with pytest.raises(NotImplementedError, match="explicit entity scene"):
        backend.get_body_mass()
    with pytest.raises(NotImplementedError, match="explicit entity scene"):
        backend.get_body_ipos()
    with pytest.raises(NotImplementedError, match="explicit entity scene"):
        backend.get_body_ipos(env_ids=[0])
    with pytest.raises(NotImplementedError, match="does not expose geom names"):
        backend.get_geom_names()
    with pytest.raises(NotImplementedError, match="does not expose geom friction"):
        backend.get_geom_friction()
    with pytest.raises(NotImplementedError, match="does not expose geom contact masks"):
        backend.get_geom_contact_masks()


def test_mapped_geometry_dimensions_and_contact_parameters_stay_unsupported():
    backend = _readback_backend({})
    for method in (
        backend.get_geom_sizes,
        backend.get_geom_solref,
        backend.get_geom_solimp,
    ):
        with pytest.raises(NotImplementedError, match="does not expose geom"):
            method()
    with pytest.raises(NotImplementedError, match="does not expose geom size"):
        backend.get_geom_size("object/box")


def test_mapped_native_geometry_identity_is_audited_per_environment():
    backend = _readback_backend({})
    backend._native_entity_records["object"]["geom_names"] = [["box"], ["wrong"]]
    with pytest.raises(IsaacSimWorkerError, match="geom_names.*object"):
        backend.get_geom_contact_masks()
    backend = _readback_backend({})
    backend._native_entity_records["object"]["geom_body_names"] = [["box"], ["wrong"]]
    with pytest.raises(IsaacSimWorkerError, match="geom_body_names.*object"):
        backend.get_geom_friction()


def test_mapped_capability_declares_exact_bounded_reset_terms():
    backend = _readback_backend({})
    capabilities = backend.get_dr_capabilities()
    assert capabilities.supported_reset_terms == {
        RESET_TERM_GEOM_FRICTION,
        RESET_TERM_KP,
        RESET_TERM_KD,
        RESET_TERM_BODY_MASS,
        RESET_TERM_BODY_INERTIA,
        RESET_TERM_BODY_IPOS,
        RESET_TERM_BASE_MASS,
        RESET_TERM_BASE_COM,
        RESET_TERM_DOF_DAMPING,
        RESET_TERM_DOF_ARMATURE,
        RESET_TERM_DOF_FRICTIONLOSS,
    }
    assert capabilities.supports_reset_term(RESET_TERM_BODY_MASS)
    assert capabilities.supports_reset_term(RESET_TERM_GEOM_FRICTION)
    assert not capabilities.supports_reset_term("gravity")
    assert not capabilities.supports_fixed_variants
    assert not capabilities.supported_fixed_variant_layouts


def test_mapped_capability_declares_fixed_variants_when_plan_is_bound():
    backend = _readback_backend({})
    backend._entity_scene.owner.variant_plan = SimpleNamespace()
    capabilities = backend.get_dr_capabilities()
    assert capabilities.supports_fixed_variants
    assert capabilities.supported_fixed_variant_layouts == {FixedVariantLayout.SAME_LAYOUT}


def _set_state_backend():
    backend = _readback_backend({})
    backend._worker_dead_error = None
    backend._slots = {"ready": True}
    commits = []

    def commit(request, controls=None, randomization=None):
        commits.append((request, controls, randomization))

    backend._commit_entity_reset = commit
    return backend, commits


def _full_state_rows(count):
    qpos = np.zeros((count, 8), dtype=np.float32)
    qpos[:, 4] = 1.0
    qvel = np.zeros((count, 7), dtype=np.float32)
    return qpos, qvel


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("geom_friction", [[[0.1, 0.1, 0.0], [0.2, 0.2, 0.0]]], "geom_friction must have shape"),
        (
            "geom_friction",
            [[[0.1, -0.1, 0.0], [0.2, 0.2, 0.0], [0.3, 0.3, 0.0]]],
            "nonnegative",
        ),
        (
            "geom_friction",
            [[[0.1, 0.2, 0.0], [0.2, 0.2, 0.0], [0.3, 0.3, 0.0]]],
            "static == dynamic",
        ),
        (
            "geom_friction",
            [[[0.1, 0.1, 0.1], [0.2, 0.2, 0.0], [0.3, 0.3, 0.0]]],
            "zero third column",
        ),
    ],
)
def test_mapped_reset_payloads_validate_before_worker_request(field, value, message):
    backend, commits = _set_state_backend()
    payload = ResetRandomizationPayload(**{field: np.asarray(value, dtype=np.float32)})
    with pytest.raises((TypeError, ValueError), match=message):
        backend._set_mapped_state(np.array([1]), *_full_state_rows(1), payload)
    assert commits == []


def test_mapped_reset_rejects_non_positive_body_mass_before_worker_request():
    backend, commits = _set_state_backend()
    with pytest.raises(ValueError, match="body_mass values must be strictly positive"):
        backend._set_mapped_state(
            np.array([1]),
            *_full_state_rows(1),
            ResetRandomizationPayload(
                body_mass=np.asarray([[30, 10, 0, 103]], dtype=np.float32)
            ),
        )
    assert commits == []


def test_mapped_reset_allows_zero_mass_on_unowned_world_row():
    backend, commits = _set_state_backend()
    backend._entity_scene.owner.model.body_mass = np.array(
        [100, 101, 102, 0], dtype=np.float32
    )
    backend._set_mapped_state(
        np.array([1]),
        *_full_state_rows(1),
        ResetRandomizationPayload(
            body_mass=np.asarray([[30, 10, 11, 0]], dtype=np.float32)
        ),
    )
    assert len(commits) == 1
    np.testing.assert_array_equal(commits[0][2].body_mass, [[30, 10, 11, 0]])


def test_mapped_reset_accepts_body_mass_and_folds_base_mass_delta():
    backend, commits = _set_state_backend()
    backend._set_mapped_state(
        np.array([1]),
        *_full_state_rows(1),
        ResetRandomizationPayload(
            body_mass=np.asarray([[35, 10, 11, 103]], dtype=np.float32),
            base_mass_delta=np.asarray([2.5], dtype=np.float32),
        ),
    )
    assert len(commits) == 1
    _request, _controls, randomization = commits[0]
    # The delta folds into the primary entity's root body (public body 1).
    np.testing.assert_allclose(randomization.body_mass, [[35, 12.5, 11, 103]])
    assert randomization.base_mass_delta is None
    assert randomization.body_ipos is None


def test_mapped_reset_folds_base_com_offset_into_variant_default_ipos():
    backend, commits = _set_state_backend()
    backend._set_mapped_state(
        np.array([0]),
        *_full_state_rows(1),
        ResetRandomizationPayload(
            base_com_offset=np.asarray([[0.1, -0.2, 0.3]], dtype=np.float32)
        ),
    )
    _request, _controls, randomization = commits[0]
    assert randomization.base_com_offset is None
    assert randomization.body_ipos is not None
    # Owned columns start from variant record defaults (zeros); the unowned
    # column keeps the canonical compiled default; only the primary entity's
    # root body (robot/base = public body 1) receives the offset.
    expected = np.zeros((4, 3), dtype=np.float32)
    expected[3] = np.arange(12, dtype=np.float32).reshape(4, 3)[3] / 7
    expected[1] += [0.1, -0.2, 0.3]
    np.testing.assert_allclose(randomization.body_ipos[0], expected, rtol=1e-6)


def _actuated_set_state_backend():
    backend = _readback_backend({})
    scene = backend._entity_scene
    robot = replace(
        scene.layout.entities[0],
        actuator_names=("motor",),
        actuator_joint_names=("passive",),
        actuator_indices=(0,),
    )
    scene.layout = replace(  # type: ignore[attr-defined]
        scene.layout, entities=(robot, scene.layout.entities[1]), nu=1
    )
    record = scene.payload["scene_entities"][0]["variants"][0]  # type: ignore[attr-defined]
    record["actuator_names"] = ["motor"]
    record["actuator_joint_names"] = ["passive"]
    record["dof_stiffness"] = [10.0]
    record["dof_damping"] = [1.0]
    backend._worker_dead_error = None
    backend._slots = {"ready": True}
    commits = []

    def commit(request, controls=None, randomization=None):
        commits.append((request, controls, randomization))

    backend._commit_entity_reset = commit
    return backend, commits


def test_mapped_reset_kp_kd_validate_and_pass_through():
    backend, commits = _actuated_set_state_backend()
    backend._set_mapped_state(
        np.array([1]),
        *_full_state_rows(1),
        ResetRandomizationPayload(
            kp=np.asarray([[20.0]], dtype=np.float32),
            kd=np.asarray([[3.0]], dtype=np.float32),
        ),
    )
    _request, _controls, randomization = commits[0]
    np.testing.assert_allclose(randomization.kp, [[20.0]])
    np.testing.assert_allclose(randomization.kd, [[3.0]])
    with pytest.raises(ValueError, match="kp values must be nonnegative"):
        backend._set_mapped_state(
            np.array([1]),
            *_full_state_rows(1),
            ResetRandomizationPayload(kp=np.asarray([[-1.0]], dtype=np.float32)),
        )
    with pytest.raises(ValueError, match="kd values must be nonnegative"):
        backend._set_mapped_state(
            np.array([1]),
            *_full_state_rows(1),
            ResetRandomizationPayload(kd=np.asarray([[-0.5]], dtype=np.float32)),
        )
    assert len(commits) == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "dof_damping",
            [[0.25, 0.1, 0, 0, 0, 0, 0]],
            "free-root columns must remain zero",
        ),
        ("dof_armature", [[-0.5, 0, 0, 0, 0, 0, 0]], "dof_armature values must be nonnegative"),
        (
            "dof_frictionloss",
            [[0.5, 0, 0, 0, 0, 0]],
            "dof_frictionloss must have shape",
        ),
        ("body_inertia", [[[1.0, 1.0, 1.0]] * 4], "body_inertia values must be strictly positive"),
    ],
)
def test_mapped_reset_dof_and_inertia_terms_validate_before_worker_request(
    field, value, message
):
    backend, commits = _set_state_backend()
    array = np.asarray(value, dtype=np.float32)
    if field == "body_inertia":
        array[0, 2, 1] = 0.0
    payload = ResetRandomizationPayload(**{field: array})
    with pytest.raises(ValueError, match=message):
        backend._set_mapped_state(np.array([1]), *_full_state_rows(1), payload)
    assert commits == []


def test_mapped_reset_rejects_unowned_body_columns():
    backend, commits = _set_state_backend()
    body_mass = np.asarray([[30, 10, 11, 55]], dtype=np.float32)
    with pytest.raises(ValueError, match="belong to no entity"):
        backend._set_mapped_state(
            np.array([1]),
            *_full_state_rows(1),
            ResetRandomizationPayload(body_mass=body_mass),
        )
    body_mass[0, 3] = 103.0
    backend._set_mapped_state(
        np.array([1]), *_full_state_rows(1), ResetRandomizationPayload(body_mass=body_mass)
    )
    assert len(commits) == 1


def test_mapped_reset_rejects_kinematic_mirror_mutation():
    backend, commits = _set_state_backend()
    scene = backend._entity_scene
    entity_cls = type(scene.layout.entities[0])
    mirror = entity_cls(
        "mirror", "rigid", "kinematic", "ghost", ("ghost",), (3,), (None,),
        (), (), (), (), (), (), (),
    )
    scene.layout = replace(  # type: ignore[attr-defined]
        scene.layout, entities=(*scene.layout.entities, mirror)
    )
    record = {
        "joint_names": [], "body_names": ["ghost"], "actuator_names": [],
        "actuator_joint_names": [], "body_sphere_radii": [[]],
        "geom_names": [], "geom_body_names": [], "geom_contype": [], "geom_conaffinity": [],
        "geom_friction": [],
        "dof_stiffness": [], "dof_damping": [], "dof_effort": [], "dof_armature": [],
        "dof_friction": [], "dof_lower": [], "dof_upper": [],
        "body_mass": [5.0], "body_ipos": [[0.1, 0.0, 0.0]],
        "body_inertia": [[2.0, 2.0, 2.0]], "body_iquat": [[1.0, 0.0, 0.0, 0.0]],
    }
    scene.payload["scene_entities"].append(  # type: ignore[attr-defined]
        {
            "name": "mirror", "kind": "rigid", "root_mode": "kinematic",
            "asset_format": "mjcf", "sources": ["mirror.xml"], "variants": [record],
            "assignment": [0, 0],
        }
    )
    body_mass = np.asarray([[30, 10, 11, 5.0]], dtype=np.float32)
    backend._set_mapped_state(
        np.array([1]), *_full_state_rows(1), ResetRandomizationPayload(body_mass=body_mass)
    )
    assert len(commits) == 1
    body_mass[0, 3] = 9.0
    with pytest.raises(ValueError, match="kinematic entity mirror"):
        backend._set_mapped_state(
            np.array([1]), *_full_state_rows(1), ResetRandomizationPayload(body_mass=body_mass)
        )
    assert len(commits) == 1


def test_mapped_reset_unsupported_terms_fail_closed_with_reasons():
    backend, commits = _set_state_backend()
    with pytest.raises(NotImplementedError, match="gravity: .*simulation-global"):
        backend._set_mapped_state(
            np.array([1]), *_full_state_rows(1), ResetRandomizationPayload(
                gravity=np.zeros((1, 3), dtype=np.float32)
            )
        )
    with pytest.raises(NotImplementedError, match="geom_size"):
        backend._set_mapped_state(
            np.array([1]), *_full_state_rows(1), ResetRandomizationPayload(
                geom_size=np.ones((1, 3, 3), dtype=np.float32)
            )
        )
    assert commits == []


def test_mapped_reset_term_defaults_are_per_env_variant_tables():
    backend = _readback_backend({})
    mass = backend.get_reset_term_default(RESET_TERM_BODY_MASS)
    np.testing.assert_array_equal(mass, [[1.0, 1.0, 1.0, 103.0], [1.0, 1.0, 1.0, 103.0]])
    assert not mass.flags.writeable
    ipos = backend.get_reset_term_default(RESET_TERM_BODY_IPOS)
    assert ipos.shape == (2, 4, 3)
    np.testing.assert_array_equal(ipos[:, :3], 0.0)
    np.testing.assert_allclose(
        ipos[:, 3],
        np.broadcast_to(
            np.arange(12, dtype=np.float32).reshape(4, 3)[3] / 7, (2, 3)
        ),
    )
    inertia = backend.get_reset_term_default(RESET_TERM_BODY_INERTIA)
    np.testing.assert_array_equal(inertia[:, :3], 1.0)
    np.testing.assert_array_equal(inertia[:, 3], 0.5)
    friction = backend.get_reset_term_default(RESET_TERM_GEOM_FRICTION)
    np.testing.assert_array_equal(friction, [[[0.5, 0.5, 0.0]] * 3] * 2)
    np.testing.assert_array_equal(
        backend.get_reset_term_default(RESET_TERM_DOF_DAMPING), np.zeros((2, 7))
    )
    np.testing.assert_array_equal(
        backend.get_reset_term_default(RESET_TERM_DOF_ARMATURE), np.zeros((2, 7))
    )
    np.testing.assert_array_equal(
        backend.get_reset_term_default(RESET_TERM_DOF_FRICTIONLOSS), np.zeros((2, 7))
    )
    np.testing.assert_array_equal(
        backend.get_reset_term_default(RESET_TERM_BASE_MASS), np.zeros(2)
    )
    np.testing.assert_array_equal(
        backend.get_reset_term_default(RESET_TERM_BASE_COM), np.zeros((2, 3))
    )
    with pytest.raises(NotImplementedError, match="gravity"):
        backend.get_reset_term_default("gravity")


def test_mapped_reset_term_defaults_scatter_actuator_gains():
    backend, _commits = _actuated_set_state_backend()
    np.testing.assert_array_equal(backend.get_reset_term_default(RESET_TERM_KP), [[10.0]] * 2)
    np.testing.assert_array_equal(backend.get_reset_term_default(RESET_TERM_KD), [[1.0]] * 2)


def test_mapped_reset_rejects_unsupported_and_duplicate_rows_before_worker_request():
    backend, commits = _set_state_backend()
    with pytest.raises(NotImplementedError, match="gravity"):
        backend._set_mapped_state(
            np.array([1]), *_full_state_rows(1), ResetRandomizationPayload(gravity=np.zeros((1, 3)))
        )
    with pytest.raises(ValueError, match="at least one environment"):
        backend._set_mapped_state(
            np.array([], dtype=np.intp),
            *_full_state_rows(0),
            ResetRandomizationPayload(
                geom_friction=np.ones((0, 3, 3), dtype=np.float32)
            ),
        )
    with pytest.raises(ValueError, match="duplicate"):
        backend._set_mapped_state(
            np.array([1, 1]),
            *_full_state_rows(2),
            ResetRandomizationPayload(
                geom_friction=np.ones((2, 3, 3), dtype=np.float32)
            ),
        )
    assert commits == []


def test_mapped_reset_rows_remain_reordered_and_randomization_is_detached():
    backend, commits = _set_state_backend()
    geom_friction = np.asarray(
        [
            [[0.7, 0.7, 0.0], [0.8, 0.8, 0.0], [0.9, 0.9, 0.0]],
            [[0.4, 0.4, 0.0], [0.5, 0.5, 0.0], [0.6, 0.6, 0.0]],
        ],
        dtype=np.float32,
    )
    payload = ResetRandomizationPayload(geom_friction=geom_friction)
    backend._set_mapped_state(np.array([1, 0]), *_full_state_rows(2), payload)
    request, _controls, randomization = commits[0]
    assert request.env_ids == (1, 0)
    np.testing.assert_array_equal(randomization.geom_friction, geom_friction)
    randomization.geom_friction[:] = 0
    np.testing.assert_array_equal(payload.geom_friction, geom_friction)


def test_entity_reset_wire_serializes_supported_terms_and_refreshes_records():
    records = {
        "robot": {
            "body_mass": [[10, 11], [20, 21]],
            "body_com": [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
            "body_inertia": [
                [[1, 0, 0, 0, 1, 0, 0, 0, 1]] * 2,
                [[1, 0, 0, 0, 1, 0, 0, 0, 1]] * 2,
            ],
            "geom_friction": [[[0.1, 0.1, 0.0], [0.2, 0.2, 0.0]]] * 2,
        },
        "object": {
            "body_mass": [[30], [40]],
            "body_com": [[[0, 0, 0]], [[0, 0, 0]]],
            "body_inertia": [
                [[1, 0, 0, 0, 1, 0, 0, 0, 1]],
                [[1, 0, 0, 0, 1, 0, 0, 0, 1]],
            ],
            "geom_friction": [[[0.3, 0.3, 0.0]]] * 2,
        },
    }
    backend = _readback_backend(records)
    backend._worker_dead_error = None
    backend._entity_query_names = tuple(
        entity.name for entity in backend._entity_scene.layout.entities
    )
    backend._slots = {
        name: np.zeros(shape, dtype=protocol.slot_dtype(name))
        for name, shape in protocol.scene_slot_shapes(2, backend._entity_scene.layout).items()
    }
    backend._staged_body_wrench = np.zeros((2, 4, 6), dtype=np.float32)
    requests = []

    def fake_request(command, payload, *, expect):
        assert command == protocol.CMD_RESET_ENTITIES and expect == protocol.CMD_READY
        requests.append(payload)
        if "randomization" not in payload:
            return None
        return {
            "native_entity_records": [
                {
                    "name": "robot",
                    "body_mass": [[10, 11], [50, 51]],
                    "body_com": [[[0, 0, 0], [0, 0, 0]], [[0.1, 0, 0], [0, 0, 0]]],
                    "body_inertia": [
                        [[1, 0, 0, 0, 1, 0, 0, 0, 1]] * 2,
                        [[2, 0, 0, 0, 2, 0, 0, 0, 2]] * 2,
                    ],
                    "geom_friction": [
                        [[0.2, 0.2, 0.0], [0.2, 0.2, 0.0]],
                        [[0.7, 0.7, 0.0], [0.2, 0.2, 0.0]],
                    ],
                },
                {
                    "name": "object",
                    "body_mass": [[30], [60]],
                    "body_com": [[[0, 0, 0]], [[0, 0.2, 0]]],
                    "body_inertia": [
                        [[1, 0, 0, 0, 1, 0, 0, 0, 1]],
                        [[3, 0, 0, 0, 3, 0, 0, 0, 3]],
                    ],
                    "geom_friction": [[[0.3, 0.3, 0.0]], [[0.8, 0.8, 0.0]]],
                },
            ]
        }

    backend._request = fake_request
    patches = full_state_reset_patches(
        backend._entity_scene.layout, *_full_state_rows(1)
    )
    geom_friction = np.asarray([[[0.7, 0.7, 0.0], [0.2, 0.2, 0.0], [0.8, 0.8, 0.0]]])
    body_mass = np.asarray([[30, 50, 51, 103]], dtype=np.float32)
    dof_armature = np.asarray([[0.5, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    backend._commit_entity_reset(
        SceneResetRequest((1,), patches),
        randomization=ResetRandomizationPayload(
            geom_friction=geom_friction, body_mass=body_mass, dof_armature=dof_armature
        ),
    )
    assert requests[-1]["randomization"] == {
        "geom_friction": geom_friction.tolist(),
        "body_mass": body_mass.tolist(),
        "dof_armature": dof_armature.tolist(),
    }
    np.testing.assert_allclose(
        backend.get_body_mass(), [[30, 10, 11, 103], [60, 50, 51, 103]]
    )
    np.testing.assert_allclose(
        backend.get_geom_friction()[1, 2], [0.8, 0.8, 0.0]
    )
    np.testing.assert_allclose(
        backend.get_body_ipos(env_ids=[1])[0, 0], [0, 0.2, 0]
    )

    backend._commit_entity_reset(SceneResetRequest((1,), patches))
    assert "randomization" not in requests[-1]


@pytest.mark.parametrize("bad", ["ids", "mask", "owner", "fixed", "quat", "nan", "root_mask"])
def test_entire_reset_is_rejected_before_first_native_write(bad):
    ctx = _context()
    writes = []
    ctx._commit = lambda *args, **kwargs: writes.append(args)
    ctx.refresh_state_slots = lambda: None
    if bad == "ids":
        ctx.slots["reset_env_ids"][:] = 1
    elif bad == "mask":
        ctx.slots["reset_qpos_mask"][1] = 2
    elif bad == "owner":
        ctx.slots["reset_qpos_mask"][0] = 1
    elif bad == "fixed":
        ctx.slots["reset_root_mask"][0] = 1
    elif bad == "quat":
        ctx.slots["reset_entity_root_state"][0, 1, 3:7] = 0
    elif bad == "nan":
        ctx.slots["reset_qpos"][0, 0] = np.nan
    else:
        ctx.slots["reset_qvel_mask"][1] = 0
    with pytest.raises(ValueError):
        ctx.reset_entities({"count": 2, "entity_names": ["object"]})
    assert writes == []


@pytest.mark.parametrize(
    ("randomization", "message"),
    [
        ({"gravity": np.zeros((1, 3), dtype=np.float32)}, "only supported property terms"),
        (
            {"body_mass": np.zeros((1, 3), dtype=np.float32)},
            "body_mass must be strictly positive",
        ),
        (
            {"dof_armature": np.asarray([[0.0, 1.0, 0, 0, 0, 0, 0]], dtype=np.float32)},
            "free-root columns must remain zero",
        ),
        (
            {"geom_friction": np.zeros((1, 2, 3), dtype=np.float32)},
            "randomization geom_friction must be finite nonnegative",
        ),
        (
            {
                "geom_friction": np.asarray(
                    [[[0.1, 0.2, 0.0], [0.3, 0.3, 0.0], [0.4, 0.4, 0.0]]],
                    dtype=np.float32,
                )
            },
            "equal static/dynamic",
        ),
    ],
)
def test_worker_randomization_validates_before_state_or_property_write(randomization, message):
    ctx, commits, setter_ids = _property_context()
    with pytest.raises(ValueError, match=message):
        ctx.reset_entities(
            {"count": 1, "entity_names": ["object"], "randomization": randomization}
        )
    assert commits == [] and setter_ids == [] and not ctx.faulted


def test_worker_randomization_accepts_ipc_wire_lists():
    ctx, _commits, _setter_ids = _property_context()
    result = ctx._validated_reset_randomization(
        {
            "randomization": {
                "geom_friction": [
                    [[0.1, 0.1, 0.0], [0.2, 0.2, 0.0], [0.3, 0.3, 0.0]]
                ],
            }
        },
        1,
    )
    assert result is not None
    assert result["geom_friction"].shape == (1, 3, 3)


def test_worker_property_writes_use_selected_native_rows_and_refresh_records():
    ctx, commits, setter_ids = _property_context()
    geom_friction = np.asarray(
        [[[0.7, 0.7, 0.0], [0.8, 0.8, 0.0], [0.9, 0.9, 0.0]]], dtype=np.float32
    )
    result = ctx.reset_entities(
        {
            "count": 1,
            "entity_names": ["object"],
            "randomization": {"geom_friction": geom_friction},
        }
    )
    assert commits and setter_ids == [
        ("robot", "material", [0]),
        ("object", "material", [1]),
    ]
    records = {record["name"]: record for record in result["native_entity_records"]}
    np.testing.assert_allclose(records["robot"]["body_mass"], [[20, 21], [10, 11]])
    np.testing.assert_allclose(records["object"]["body_mass"], [[30], [40]])
    np.testing.assert_allclose(
        records["robot"]["geom_friction"][1], [[0.7, 0.7, 0], [0.8, 0.8, 0]]
    )
    np.testing.assert_allclose(records["object"]["geom_friction"][1], [[0.9, 0.9, 0]])
    np.testing.assert_allclose(ctx.actual[0]["body_mass"][1], [10, 11])
    np.testing.assert_allclose(ctx.actual[1]["body_mass"][1], [40])


def test_worker_body_and_dof_writes_compose_and_refresh_records():
    ctx, commits, setter_ids = _property_context()
    ctx._drive_damping[0][:] = [[1.0], [2.0]]
    body_mass = np.asarray([[5.0, 6.0, 7.0]], dtype=np.float32)
    body_ipos = np.asarray(
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]], dtype=np.float32
    )
    body_inertia = np.asarray(
        [[[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]]], dtype=np.float32
    )
    dof_damping = np.asarray([[0.25, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    dof_armature = np.asarray([[0.5, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    dof_frictionloss = np.asarray([[0.75, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    result = ctx.reset_entities(
        {
            "count": 1,
            "entity_names": ["robot", "object"],
            "randomization": {
                "body_mass": body_mass,
                "body_ipos": body_ipos,
                "body_inertia": body_inertia,
                "dof_damping": dof_damping,
                "dof_armature": dof_armature,
                "dof_frictionloss": dof_frictionloss,
            },
        }
    )
    # Public env row 0 of the wire payload targets slot env 1; the robot view
    # maps it to native row 0 and the object view to native row 1.
    assert ("robot", "mass", [0]) in setter_ids
    assert ("robot", "com", [0]) in setter_ids
    assert ("robot", "inertia", [0]) in setter_ids
    assert ("robot", "damping", [0]) in setter_ids
    assert ("robot", "armature", [0]) in setter_ids
    assert ("robot", "friction", [0]) in setter_ids
    assert ("object", "mass", [1]) in setter_ids
    robot_view = ctx.assets[0].root_physx_view
    object_view = ctx.assets[1].root_physx_view
    np.testing.assert_allclose(robot_view.masses[0], [5.0, 6.0])
    np.testing.assert_allclose(object_view.masses[1], [7.0])
    np.testing.assert_allclose(robot_view.coms[0, 0, :3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(robot_view.inertias[0, 0].reshape(3, 3), np.diag([2, 3, 4]))
    # Drive damping composes with the natural dof_damping write.
    np.testing.assert_allclose(robot_view.dof_damping[0], [2.0 + 0.25])
    np.testing.assert_allclose(robot_view.dof_armature[0], [0.5])
    np.testing.assert_allclose(robot_view.dof_friction[0], [0.75])
    records = {record["name"]: record for record in result["native_entity_records"]}
    np.testing.assert_allclose(records["robot"]["body_mass"][1], [5.0, 6.0])
    np.testing.assert_allclose(records["robot"]["dof_damping"][1], [2.25])
    np.testing.assert_allclose(records["object"]["body_mass"][1], [7.0])
    np.testing.assert_allclose(ctx.actual[0]["body_mass"][1], [5.0, 6.0])


def test_worker_gain_writes_use_actuator_columns_and_compose_damping():
    ctx, commits, setter_ids = _property_context()
    robot = replace(
        ctx.layout.entities[0],
        actuator_names=("motor",),
        actuator_joint_names=("passive",),
        actuator_indices=(0,),
    )
    ctx.layout = replace(ctx.layout, entities=(robot, ctx.layout.entities[1]), nu=1)
    ctx.maps[0]["controls"] = np.array([0])
    ctx._drive_damping[0][:] = [[1.0], [2.0]]
    kp = np.asarray([[20.0]], dtype=np.float32)
    kd = np.asarray([[3.0]], dtype=np.float32)
    result = ctx.reset_entities(
        {
            "count": 1,
            "entity_names": ["robot", "object"],
            "randomization": {"kp": kp, "kd": kd},
        }
    )
    assert ("robot", "stiffness", [0]) in setter_ids
    assert ("robot", "damping", [0]) in setter_ids
    view = ctx.assets[0].root_physx_view
    np.testing.assert_allclose(view.dof_stiffness[0], [20.0])
    np.testing.assert_allclose(view.dof_damping[0], [3.0])
    np.testing.assert_allclose(ctx._drive_damping[0], [[1.0], [3.0]])
    records = {record["name"]: record for record in result["native_entity_records"]}
    np.testing.assert_allclose(records["robot"]["dof_stiffness"][1], [20.0])
    np.testing.assert_allclose(records["robot"]["dof_damping"][1], [3.0])


def test_worker_maps_multiple_geoms_within_reordered_native_body():
    ctx, _commits, setter_ids = _property_context()
    geom_type = ctx.layout.entities[0].geoms[0].__class__
    robot = replace(
        ctx.layout.entities[0],
        geoms=(
            geom_type("base::geom0", "base"),
            geom_type("base::geom1", "base"),
            geom_type("tip::geom0", "tip"),
        ),
    )
    object_entity = replace(ctx.layout.entities[1], geoms=())
    ctx.layout = replace(ctx.layout, entities=(robot, object_entity), ngeom=3)
    ctx.maps[0]["bodies"] = np.array([1, 0])
    np.testing.assert_array_equal(
        _native_geometry_columns(["tip", "base"], ctx.maps[0]["bodies"], robot),
        [1, 2, 0],
    )
    ctx.maps[0]["geoms"] = np.array([1, 2, 0])
    ctx.maps[1]["geoms"] = np.empty(0, dtype=np.int64)

    native_materials = np.asarray(
        [
            [[0.1, 0.1, 0.0], [0.2, 0.2, 0.0], [0.3, 0.3, 0.0]],
            [[0.4, 0.4, 0.0], [0.5, 0.5, 0.0], [0.6, 0.6, 0.0]],
        ],
        dtype=np.float32,
    )
    ctx.assets[0].root_physx_view.materials = native_materials
    geom_friction = np.asarray(
        [[[0.7, 0.7, 0.0], [0.8, 0.8, 0.0], [0.9, 0.9, 0.0]]],
        dtype=np.float32,
    )

    result = ctx.reset_entities(
        {
            "count": 1,
            "entity_names": ["object"],
            "randomization": {
                "geom_friction": geom_friction,
            },
        }
    )
    assert setter_ids == [
        ("robot", "material", [0]),
    ]
    records = {record["name"]: record for record in result["native_entity_records"]}
    np.testing.assert_allclose(records["robot"]["geom_friction"][1], geom_friction[0])
    np.testing.assert_allclose(records["robot"]["geom_friction"][0], native_materials[1, [1, 2, 0]])
    assert records["object"]["geom_friction"] == [[], []]
    np.testing.assert_allclose(native_materials[0, [1, 2, 0]], geom_friction[0])


def test_worker_rejects_extra_native_material_shapes():
    ctx, commits, setter_ids = _property_context()
    native_materials = np.asarray(
        [
            [[0.1, 0.1, 0.0], [0.2, 0.2, 0.0], [0.3, 0.3, 0.0]],
            [[0.4, 0.4, 0.0], [0.5, 0.5, 0.0], [0.6, 0.6, 0.0]],
        ],
        dtype=np.float32,
    )
    ctx.assets[0].root_physx_view.materials = native_materials
    with pytest.raises(RuntimeError, match="native material view does not match"):
        ctx.reset_entities(
            {
                "count": 1,
                "entity_names": ["object"],
                "randomization": {
                    "geom_friction": np.asarray(
                        [[[0.7, 0.7, 0.0], [0.8, 0.8, 0.0], [0.9, 0.9, 0.0]]],
                        dtype=np.float32,
                    )
                },
            }
        )
    assert commits and setter_ids == [] and ctx.faulted


def test_worker_native_property_failure_after_state_commit_faults_worker():
    ctx, commits, setter_ids = _property_context()
    attempts = []

    def fail_material(values, *, indices):
        attempts.append(np.asarray(indices).tolist())
        raise RuntimeError("native material setter failed")

    ctx.assets[0].root_physx_view.set_material_properties = fail_material
    with pytest.raises(RuntimeError, match="material setter failed"):
        ctx.reset_entities(
            {
                "count": 1,
                "entity_names": ["object"],
                "randomization": {
                    "geom_friction": np.asarray(
                        [[[0.7, 0.7, 0.0], [0.8, 0.8, 0.0], [0.9, 0.9, 0.0]]],
                        dtype=np.float32,
                    )
                },
            }
        )
    assert commits and attempts == [[0]] and setter_ids == [] and ctx.faulted


def test_worker_post_write_property_readback_failure_faults_worker():
    ctx, _commits, _setter_ids = _property_context()
    calls = 0
    original_materials = ctx.assets[0].root_physx_view.get_material_properties

    def failing_materials():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("native material readback failed")
        return original_materials()

    ctx.assets[0].root_physx_view.get_material_properties = failing_materials
    with pytest.raises(RuntimeError, match="material readback failed"):
        ctx.reset_entities(
            {
                "count": 1,
                "entity_names": ["object"],
                "randomization": {
                    "geom_friction": np.asarray(
                        [[[0.7, 0.7, 0.0], [0.8, 0.8, 0.0], [0.9, 0.9, 0.0]]],
                        dtype=np.float32,
                    )
                },
            }
        )
    assert ctx.faulted


def test_unsorted_selected_rows_remain_unsorted_and_detached_at_commit():
    ctx = _context()
    writes = []
    ctx._commit = lambda *args, **kwargs: writes.append(args)
    ctx.refresh_state_slots = lambda: None
    original = copy.deepcopy(ctx.slots)
    ctx.reset_entities({"count": 2, "entity_names": ["object"]})
    np.testing.assert_array_equal(writes[0][0], [1, 0])
    for values in writes[0]:
        values[...] = 0
    for name in original:
        np.testing.assert_array_equal(ctx.slots[name], original[name])


def test_world_body_rotation_has_independent_ninety_degree_oracle():
    q = np.array([[np.sqrt(.5), 0, 0, np.sqrt(.5)]])
    world = _rotate(q, np.array([[1., 2., 3.]]))
    np.testing.assert_allclose(world, [[-2, 1, 3]], atol=1e-6)
    np.testing.assert_allclose(_rotate(q, world, inverse=True), [[1, 2, 3]], atol=1e-6)


def test_readback_failure_after_native_reset_is_faulted():
    ctx = _context()
    ctx.faulted = False
    writes = []
    ctx._commit = lambda *args, **kwargs: writes.append(args)

    def broken_readback():
        raise RuntimeError("native state inaccessible after write")

    ctx.refresh_state_slots = broken_readback
    with pytest.raises(RuntimeError, match="inaccessible"):
        ctx.reset_entities({"count": 2, "entity_names": ["object"]})
    assert writes and ctx.faulted


def test_native_joint_commit_maps_reordered_envs_and_preserves_unselected_channel():
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

        def __getitem__(self, key):
            return Tensor(self.values[key])

    ctx = _context()
    ctx.device, ctx.sim_dt = "cpu", .01
    ctx.torch = SimpleNamespace(as_tensor=lambda value, **kwargs: np.asarray(value),
                                long=np.int64)
    ctx._tensor = lambda value: value
    writes, resets = [], []
    asset = SimpleNamespace(
        data=SimpleNamespace(joint_pos=Tensor([[10], [20]]),
                             joint_vel=Tensor([[1], [2]])),
        write_joint_state_to_sim=lambda p, v, **kw: writes.append((p.copy(), v.copy(), kw)),
        reset=lambda ids: resets.append(ids.copy()), update=lambda dt: None,
    )
    ctx.assets = [asset, asset]
    ctx.maps = [{"envs": np.array([1, 0]), "joints": np.array([0])},
                {"envs": np.array([0, 1]), "joints": np.array([], dtype=int)}]
    p = np.full((1, 8), 999, dtype=np.float32)
    v = np.zeros((1, 7), dtype=np.float32)
    v[0, 0] = 3
    pmask = np.zeros(8, dtype=np.uint8)
    vmask = np.zeros(7, dtype=np.uint8)
    vmask[0] = 1
    ctx._commit(np.array([1]), p, v, np.zeros((1, 2, 13)),
                pmask, vmask, np.zeros((2, 2), dtype=np.uint8))
    assert len(writes) == 1
    np.testing.assert_array_equal(writes[0][0], [[10]])
    np.testing.assert_array_equal(writes[0][1], [[3]])
    np.testing.assert_array_equal(writes[0][2]["env_ids"], [0])
    np.testing.assert_array_equal(resets, [[0]])


@pytest.mark.parametrize("rows", [1,8,256])
def test_sparse_joint_commit_downloads_only_selected_rows_and_keeps_native_lifecycle(rows):
    from tests.adapters.isaacsim.reset_transfer_fixture import execute_case

    result = execute_case(SceneWorkerContext._commit, num_envs=1024, num_joints=32, rows=rows)
    assert result["d2h_calls"] == 2
    assert result["d2h_bytes_each"] == [rows*4, rows*4]
    # One selected env-ID upload, one selected joint-ID upload, pos/vel payloads.
    # The other three entities must not cause native-ID construction.
    assert result["h2d_calls"] == 4
    assert result["h2d_bytes"] == rows*16 + 8
    assert [op["operation"] for op in result["operations"]] == [
        "write_joint_state", "reset", "update"]
    assert all(op["entity"] == 1 for op in result["operations"])
    native_rows = list(range(1024-rows,1024))
    assert result["operations"][0]["rows"] == native_rows
    assert result["operations"][0]["joints"] == [28]
    np.testing.assert_array_equal(result["operations"][0]["position"],
                                  (np.arange(rows)+.75)[:,None])
    expected_velocity = -(np.asarray(native_rows)*32+28)-.25
    np.testing.assert_array_equal(result["operations"][0]["velocity"],expected_velocity[:,None])


def test_initial_control_has_actuator_width_and_must_be_finite():
    payload = _payload()
    payload["initial_ctrl"] = [[], []]
    validate_scene_payload(protocol, payload)
    payload["initial_ctrl"] = [[1], [2]]
    with pytest.raises(ValueError, match="initial_ctrl"):
        validate_scene_payload(protocol, payload)


def test_keyframe_control_uses_control_columns_and_native_rows_not_joint_positions():
    payload = _payload()
    robot = payload["scene_layout"]["entities"][0]
    robot.update(actuator_names=["motor"], actuator_joint_names=["passive"],
                 actuator_indices=[0])
    payload["scene_layout"]["nu"] = 1
    record = payload["scene_entities"][0]["variants"][0]
    record.update(actuator_names=["motor"], actuator_joint_names=["passive"])
    payload["initial_ctrl"] = [[.25], [-.5]]
    layout = validate_scene_payload(protocol, payload)
    ctx = SceneWorkerContext.__new__(SceneWorkerContext)
    ctx.layout = layout
    writes = []
    ctx.assets = [SimpleNamespace(
        data=SimpleNamespace(joint_pos=np.array([[10], [20]])),
        set_joint_position_target=lambda value, **kw: writes.append((value.copy(), kw)),
    ), object()]
    ctx.maps = [{"public_for_native": np.array([1, 0]), "controls": np.array([0])}, {}]
    ctx._tensor = lambda value: value
    ctx._set_control_targets(np.asarray(payload["initial_ctrl"]))
    assert len(writes) == 1
    np.testing.assert_array_equal(writes[0][0], [[-.5], [.25]])
    assert writes[0][1]["joint_ids"] == [0]
    payload["initial_ctrl"] = [[np.nan], [0]]
    with pytest.raises(ValueError, match="initial_ctrl"):
        validate_scene_payload(protocol, payload)


def _controlled_context():
    from dataclasses import replace

    ctx = _context()
    robot = replace(ctx.layout.entities[0], actuator_names=("motor",),
                    actuator_joint_names=("passive",), actuator_indices=(0,))
    ctx.layout = replace(ctx.layout, entities=(robot, ctx.layout.entities[1]), nu=1)
    ctx.slots["ctrl"] = np.array([[.1], [.2]], dtype=np.float32)
    ctx.faulted, ctx.device = False, "cpu"
    ctx.torch = SimpleNamespace(as_tensor=lambda value, **kwargs: np.asarray(value), long=np.int64)
    ctx._tensor = lambda value: value
    writes = []
    ctx.assets = [SimpleNamespace(set_joint_position_target=lambda value, **kwargs:
                                 writes.append(("control", value.copy(), kwargs))), object()]
    ctx.maps = [{"envs": np.array([1, 0]), "controls": np.array([0])}, {}]
    ctx._commit = lambda *args, **kwargs: writes.append(("state",))
    ctx.refresh_state_slots = lambda: None
    return ctx, writes


def test_reset_keyframe_control_override_uses_selected_rows_and_independent_values():
    ctx, writes = _controlled_context()
    ctx.slots["reset_qpos"][0, 0] = 10
    ctx.reset_entities({"count": 1, "entity_names": ["robot", "object"],
                        "control_values": [[-.7]]})
    assert [write[0] for write in writes] == ["state", "control"]
    np.testing.assert_allclose(writes[1][1], [[-.7]])
    np.testing.assert_array_equal(writes[1][2]["env_ids"], [0])
    np.testing.assert_allclose(ctx.slots["ctrl"], [[.1], [-.7]])


@pytest.mark.parametrize("bad", [[[0, 1]], [[np.nan]], [[np.inf]], [[1e100]], [[True]]])
def test_reset_control_override_validation_precedes_native_state(bad):
    ctx, writes = _controlled_context()
    with pytest.raises(ValueError, match="control_values"):
        ctx.reset_entities({"count": 1, "entity_names": ["robot", "object"],
                            "control_values": bad})
    assert writes == [] and not ctx.faulted


def test_reset_control_override_cannot_change_an_unselected_entity():
    ctx, writes = _controlled_context()
    with pytest.raises(ValueError, match="unselected entity"):
        ctx.reset_entities({"count": 1, "entity_names": ["object"],
                            "control_values": [[-.7]]})
    assert writes == []


def test_normal_entity_patch_preserves_existing_hold_target_without_override():
    ctx, writes = _controlled_context()
    before = ctx.slots["ctrl"].copy()
    ctx.reset_entities({"count": 1, "entity_names": ["robot", "object"]})
    assert writes == [("state",)]
    np.testing.assert_array_equal(ctx.slots["ctrl"], before)


def test_reset_native_control_failure_after_state_commit_faults_worker():
    ctx, writes = _controlled_context()

    def fail(*args, **kwargs):
        raise RuntimeError("native target failure")

    ctx.assets[0].set_joint_position_target = fail
    before = ctx.slots["ctrl"].copy()
    with pytest.raises(RuntimeError, match="target failure"):
        ctx.reset_entities({"count": 1, "entity_names": ["robot", "object"],
                            "control_values": [[-.7]]})
    assert writes == [("state",)] and ctx.faulted
    np.testing.assert_array_equal(ctx.slots["ctrl"], before)


def test_entity_prim_components_are_valid_stable_and_injective():
    import re

    from unisim.backend.isaacsim.scene_worker import _entity_prim_component

    public_names = ("robot-arm", "robot_arm", "robot", "robot0", "entity_726f626f74")
    encoded = [_entity_prim_component(name) for name in public_names]
    assert len(set(encoded)) == len(public_names)
    assert all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in encoded)
    assert encoded == [_entity_prim_component(name) for name in public_names]
    assert [bytes.fromhex(name.removeprefix("entity_")).decode() for name in encoded] == list(
        public_names
    )


def test_native_environment_map_uses_exact_encoded_subtrees():
    from unisim.backend.isaacsim.scene_worker import (
        _entity_prim_component,
        _native_environment_order,
    )

    component = _entity_prim_component("robot-arm")
    roots = [f"/World/envs/env_{index}/{component}" for index in (1, 10)]
    actual = _native_environment_order([roots[1] + "/base", roots[0]], roots)
    np.testing.assert_array_equal(actual, [1, 0])
    wrong_entity = _entity_prim_component("robot_arm")
    for path in (roots[0] + "0/base", roots[0].replace(component, wrong_entity),
                 roots[0].replace("env_1/", "env_100/")):
        with pytest.raises(RuntimeError, match="unowned"):
            _native_environment_order([path, roots[1]], roots)
    with pytest.raises(RuntimeError, match="exactly one"):
        _native_environment_order([roots[0], roots[0] + "/base"], roots)


def _collision_record(names, owners, contype, conaffinity):
    return {
        "geom_names": names,
        "geom_body_names": owners,
        "geom_contype": contype,
        "geom_conaffinity": conaffinity,
        "geom_friction": [[0.5, 0.005, 0.0001]] * len(names),
    }


def test_record_collision_mask_marks_only_zero_zero_geoms_visual_only():
    record = _collision_record(
        ["col", "vis", "half", "vis2"], ["b", "b", "b", "b"], [1, 0, 0, 0], [1, 0, 1, 0]
    )
    np.testing.assert_array_equal(
        _record_collision_mask(record), [True, False, True, False]
    )


def test_consistent_collision_mask_requires_variant_agreement():
    colliding = _collision_record(["g"], ["b"], [1], [1])
    visual = _collision_record(["g"], ["b"], [0], [0])
    entry = {"variants": [colliding, colliding]}
    np.testing.assert_array_equal(_consistent_collision_mask(entry), [True])
    with pytest.raises(RuntimeError, match="collision geometry layout"):
        _consistent_collision_mask({"variants": [colliding, visual]})


def test_native_geometry_columns_skip_visual_only_geoms():
    geom_type = GeomLayout
    robot = EntityLayout(
        "robot", "articulation", "fixed", "base", ("base", "tip"), (0, 1), (None, "base"),
        (JointLayout("passive", "hinge", (0,), (0,), "tip"),), (), (), (),
        (),
        (),
        (
            geom_type("base::col", "base"),
            geom_type("base::vis", "base"),
            geom_type("tip::col", "tip"),
            geom_type("tip::vis", "tip"),
        ),
    )
    columns = _native_geometry_columns(
        ["tip", "base"],
        np.array([1, 0]),
        robot,
        np.array([True, False, True, False]),
    )
    # base::col -> native body 1 (base) first shape = 1; tip::col -> native
    # body 0 (tip) first shape = 0; visual-only geoms own no shape (-1).
    np.testing.assert_array_equal(columns, [1, -1, 0, -1])


def test_native_geometry_columns_without_mask_keeps_legacy_mapping():
    robot = EntityLayout(
        "robot", "articulation", "fixed", "base", ("base", "tip"), (0, 1), (None, "base"),
        (JointLayout("passive", "hinge", (0,), (0,), "tip"),), (), (), (),
        (),
        (),
        (GeomLayout("base::geom0", "base"), GeomLayout("tip::geom0", "tip")),
    )
    np.testing.assert_array_equal(
        _native_geometry_columns(["tip", "base"], np.array([1, 0]), robot), [1, 0]
    )


def test_visual_only_geom_record_passes_payload_validation():
    payload = _payload()
    robot_entry = payload["scene_entities"][0]
    robot_entry["variants"][0]["geom_contype"] = [1, 0]
    robot_entry["variants"][0]["geom_conaffinity"] = [1, 0]
    # The visual-only geom is valid input: only its native-shape consumers
    # skip it (mapped-worker visual-only geom support, #277).
    validate_scene_payload(protocol, payload)


def test_init_telemetry_accumulates_ordered_spans_and_snapshots():
    import time

    telemetry = _InitTelemetry()
    time.sleep(0.001)
    telemetry.mark("kit_startup_sim_context")
    started = time.perf_counter()
    time.sleep(0.001)
    telemetry.mark("entity.object.prototype_authoring")
    telemetry.mark("entity.object.prototype_authoring")  # repeated spans accumulate
    telemetry.span("entity.object.total", started)
    telemetry.count_prims(SimpleNamespace(Traverse=lambda: iter([1, 2, 3])), "final")
    telemetry.vram_used_mb = 1234.5
    report = telemetry.as_dict()

    assert report["schema_version"] == 1
    names = [phase["name"] for phase in report["phases"]]
    assert names == [
        "kit_startup_sim_context",
        "entity.object.prototype_authoring",
        "entity.object.total",
    ]
    for phase in report["phases"]:
        assert isinstance(phase["elapsed_ms"], float) and phase["elapsed_ms"] >= 0.0
        assert phase["rss_mb"] is None or phase["rss_mb"] > 0.0
    assert report["phases"][2]["elapsed_ms"] >= 1.0
    assert report["init_total_ms"] >= report["phases"][0]["elapsed_ms"] > 0.0
    assert report["stage_prims"] == {"final": 3}
    assert report["peak_rss_mb"] > 0.0
    assert report["vram_used_mb"] == 1234.5
    assert report["telemetry_overhead_ms"] >= 0.0


def test_get_meta_carries_additive_init_telemetry_key():
    ctx = SceneWorkerContext.__new__(SceneWorkerContext)
    ctx._legacy_metadata = None
    ctx.sim = SimpleNamespace(get_physics_dt=lambda: 0.002)
    ctx.gravity = np.asarray([0.0, 0.0, -9.81])
    ctx.entries = [
        {"name": "robot", "self_collision": False, "gravity_disabled": False},
        {"name": "object", "self_collision": False, "gravity_disabled": True},
    ]
    ctx.physx_solver = PhysxSolverConfig()
    ctx.layout = SimpleNamespace(to_dict=lambda: {"entities": []})
    ctx.actual = []
    ctx.origins = np.zeros((2, 3), dtype=np.float32)
    ctx.renderer = SimpleNamespace(render_mode="none", render_width=1280, render_height=720)
    ctx.num_envs = 2
    ctx._raw_usd_cache_persistent = False
    ctx._reported_raw_usd_source_digests = set()
    ctx._raw_usd_cache_reports = []
    ctx._role_usd_cache = None
    ctx._role_usd_cache_reports = []
    ctx._init_telemetry = {"schema_version": 1, "phases": []}

    meta = ctx.get_meta()
    assert meta["init_telemetry"] == {"schema_version": 1, "phases": []}
    # Pre-INIT (or legacy-run) contexts report the key as None; both are
    # additive over the pre-existing metadata shape.
    ctx._init_telemetry = None
    assert ctx.get_meta()["init_telemetry"] is None


def _reference_native_environment_order(native_paths, entity_paths):
    """The pre-#284 quadratic reference implementation (prefix scan)."""
    native_envs = []
    for path in native_paths:
        matches = [
            index
            for index, root in enumerate(entity_paths)
            if path == root or path.startswith(root + "/")
        ]
        if len(matches) != 1:
            raise RuntimeError("native view contains an unowned or ambiguous instance")
        native_envs.append(matches[0])
    if sorted(native_envs) != list(range(len(entity_paths))):
        raise RuntimeError("native view needs exactly one instance per environment")
    return np.asarray(native_envs, dtype=np.int64)


def test_native_environment_order_matches_quadratic_reference_byte_for_byte():
    from unisim.backend.isaacsim.scene_worker import _native_environment_order

    rng = np.random.default_rng(284)
    count = 1200  # crosses the env_1 vs env_11 prefix trap
    for component in ("entity_a", "entity_bb"):
        entity_paths = [f"/World/envs/env_{i}/{component}" for i in range(count)]
        suffixes = ("", "/link_0", "/link_0/geom_0/visual")
        for _ in range(20):
            order = rng.permutation(count)
            native = [
                entity_paths[i] + suffixes[int(rng.integers(len(suffixes)))] for i in order
            ]
            expected = _reference_native_environment_order(native, entity_paths)
            actual = _native_environment_order(native, entity_paths)
            assert actual.tobytes() == expected.tobytes()

    entity_paths = [f"/World/envs/env_{i}/entity_a" for i in range(count)]
    bad_paths = (
        "/World/envs/env_1200/entity_a",  # unknown environment
        "/World/envs/env_1x/entity_a",  # malformed number
        "/World/other/env_1/entity_a",  # wrong root
        "/World/envs/env_1/entity_ab",  # right env, wrong entity subtree
        "/World/envs/env_11x/entity_a/body",  # malformed deep path
    )
    for bad in bad_paths:
        for fn in (_reference_native_environment_order, _native_environment_order):
            with pytest.raises(RuntimeError, match="unowned or ambiguous"):
                fn([bad, *entity_paths[1:]], entity_paths)
    for fn in (_reference_native_environment_order, _native_environment_order):
        with pytest.raises(RuntimeError, match="exactly one"):
            fn([entity_paths[0], entity_paths[0] + "/body"], entity_paths[:2])


def test_assignment_representatives_pick_first_env_per_variant():
    from unisim.backend.isaacsim.scene_worker import _assignment_representatives

    observed = [2, 0, 2, 1, 0, 2]
    assert _assignment_representatives(observed) == {2: 0, 0: 1, 1: 3}
    # Every assigned variant is represented exactly once.
    assert sorted(_assignment_representatives(observed)) == sorted(set(observed))
    assert _assignment_representatives([0, 0, 0]) == {0: 0}
    assert _assignment_representatives([]) == {}
