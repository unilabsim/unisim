"""Mapped IsaacLab scene execution, loaded by path in the external SDK worker.

Asset parsing, USD edits and native-name discovery are cold-path operations.
Runtime operations use frozen entity/view maps, never actor creation order.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, cast

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _rotate(q: np.ndarray, v: np.ndarray, inverse: bool = False) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    if inverse:
        q[..., 1:] *= -1
    v = np.asarray(v, dtype=np.float64)
    t = 2 * np.cross(q[..., 1:], v)
    return (v + q[..., :1] * t + np.cross(q[..., 1:], t)).astype(np.float32)


def _entity_prim_component(name: str) -> str:
    """Injective USD identifier; public names never become native path syntax."""
    return "entity_" + name.encode("utf-8").hex()


def _native_environment_order(native_paths: list[str], entity_paths: list[str]) -> np.ndarray:
    """Resolve view rows against exact entity subtrees, never string prefixes alone."""
    native_envs = []
    for path in native_paths:
        matches = [index for index, root in enumerate(entity_paths)
                   if path == root or path.startswith(root + "/")]
        if len(matches) != 1:
            raise RuntimeError("native view contains an unowned or ambiguous instance")
        native_envs.append(matches[0])
    if sorted(native_envs) != list(range(len(entity_paths))):
        raise RuntimeError("native view needs exactly one instance per environment")
    return np.asarray(native_envs, dtype=np.int64)


def validate_scene_payload(protocol: Any, payload: dict[str, Any]) -> Any:
    """Reject unsupported combinations before launching Kit or converting assets."""
    layout = protocol.load_scene_layout(payload["scene_layout"])
    count = payload["num_envs"]
    contact_force_sensors = _validate_contact_force_sensors(payload, layout)
    protocol.scene_slot_shapes(count, layout, len(contact_force_sensors))
    entries = payload["scene_entities"]
    if [entry["name"] for entry in entries] != [entity.name for entity in layout.entities]:
        raise ValueError("scene entity order differs from the frozen layout")
    for entity, entry in zip(layout.entities, entries):
        if entry["asset_format"] != "mjcf":
            raise NotImplementedError("IsaacSim mapped scene currently requires MJCF sources")
        if entity.kind != entry["kind"] or entity.root_mode != entry["root_mode"]:
            raise ValueError("entity declaration differs from compiled layout")
        sources = entry["sources"]
        if not sources or len(sources) != len(entry["variants"]):
            raise ValueError("entity source and variant record counts differ")
        assignment = entry["assignment"]
        expected = [i % len(sources) for i in range(count)]
        if assignment != expected:
            raise NotImplementedError("IsaacSim mapped variants require round-robin assignment")
        if entity.kind == "rigid" and len(entity.body_names) != 1:
            raise NotImplementedError("IsaacSim rigid entity requires one physical body")
        if entity.root_mode == "kinematic" and entity.kind != "rigid":
            raise NotImplementedError("IsaacSim kinematic articulation is unsupported")
        if any(joint.kind == "ball" for joint in entity.joints):
            raise NotImplementedError("IsaacSim mapped scene supports scalar joints only")
        if len(set(entity.actuator_joint_names)) != len(entity.actuator_joint_names):
            raise NotImplementedError("IsaacSim requires one actuator per controlled joint")
        for record in entry["variants"]:
            if record["joint_names"] != [joint.name for joint in entity.joints]:
                raise ValueError("variant joint names differ from compiled layout")
            if record["body_names"] != list(entity.body_names):
                raise ValueError("variant body names differ from compiled layout")
            if record["actuator_joint_names"] != list(entity.actuator_joint_names):
                raise ValueError("variant actuator targets differ from compiled layout")
            for field in ("dof_stiffness", "dof_damping", "dof_effort", "dof_armature",
                          "dof_friction", "dof_lower", "dof_upper"):
                values = np.asarray(record[field])
                valid = (~np.isnan(values) if field in ("dof_lower", "dof_upper")
                         else np.isfinite(values))
                if values.shape != (len(entity.joints),) or not valid.all():
                    raise ValueError("invalid variant " + field)
            for index, joint in enumerate(entity.joints):
                if joint.name not in entity.actuator_joint_names and record["dof_stiffness"][index]:
                    raise NotImplementedError("passive joint stiffness requires explicit semantics")
        # One view uses one actuator configuration. Per-environment masses and
        # geometry may vary, but implicit drive settings must presently agree.
        for record in entry["variants"][1:]:
            for field in ("dof_stiffness", "dof_damping", "dof_effort", "dof_armature",
                          "dof_friction"):
                if record[field] != entry["variants"][0][field]:
                    raise NotImplementedError("IsaacSim entity variants require identical drives")
    for field, shape in (("initial_qpos", (count, layout.nq)),
                         ("initial_qvel", (count, layout.nv)),
                         ("initial_roots", (count, len(layout.entities), 13))):
        values = np.asarray(payload[field], dtype=np.float32)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError("invalid " + field)
    if "initial_ctrl" in payload:
        values = np.asarray(payload["initial_ctrl"], dtype=np.float32)
        if values.shape != (count, layout.nu) or not np.isfinite(values).all():
            raise ValueError("invalid initial_ctrl")
    return layout


def _validate_contact_force_sensors(payload: dict[str, Any], layout: Any) -> list[dict[str, str]]:
    records = payload.get("contact_force_sensors", [])
    if not isinstance(records, list):
        raise ValueError("contact_force_sensors must be a list")
    entities = {entity.name: entity for entity in layout.entities}
    names: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "name", "source_entity", "source_body", "target_entity", "target_body"
        }:
            raise ValueError("malformed contact force sensor declaration")
        if not all(isinstance(record[key], str) and record[key] for key in record):
            raise ValueError("contact force sensor fields must be non-empty strings")
        for role in ("source", "target"):
            entity = entities.get(record[f"{role}_entity"])
            body = record[f"{role}_body"]
            if entity is None or body not in entity.body_names:
                raise ValueError(
                    f"contact force sensor {record['name']!r} references unknown "
                    f"{role} entity/body: {record[f'{role}_entity']}/{body}"
                )
        if record["name"] in names:
            raise ValueError("duplicate contact force sensor name: " + record["name"])
        names.append(record["name"])
    return records


def _bake(
    usd_path: str,
    entity: Any,
    entry: dict[str, Any],
    variant: int,
    body_paths: dict[str, str],
    require_bodies: bool = False,
) -> str:
    """Author declared root/role semantics and immutable source identity on USD."""
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RuntimeError("converted entity USD has no default prim")
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    root.CreateAttribute("unisim:variantIndex", Sdf.ValueTypeNames.Int).Set(variant)
    articulation_roots = []
    rigid_bodies = []
    remove_joints = []
    root_path = str(root.GetPath())
    for prim in Usd.PrimRange(root):
        if entity.kind == "rigid" and prim.IsA(UsdPhysics.Joint):
            remove_joints.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            if entity.kind == "rigid":
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                    prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            elif prim.GetName() == entity.root_body:
                articulation_roots.append(str(prim.GetPath()))
            else:
                # Importer also tags its synthetic worldBody, outside this entity.
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim)
            body_name = prim.GetName()
            prim_path = str(prim.GetPath())
            if body_name in body_paths and body_paths[body_name] != prim_path[len(root_path):]:
                raise RuntimeError(f"entity {entity.name} has duplicate rigid body {body_name!r}")
            body_paths[body_name] = prim_path[len(root_path):]
            if entity.kind == "rigid":
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(
                    entity.root_mode != "floating"
                )
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(
                entity.root_mode == "kinematic" or entity.kind == "rigid"
                and entity.root_mode == "fixed"
            )
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(
                bool(entry["collision_enabled"])
            )
        # Disable converter-authored drives. IsaacLab owns declared control gains.
        for axis in ("angular", "linear"):
            if prim.HasAPI(UsdPhysics.DriveAPI, axis):
                drive = UsdPhysics.DriveAPI(prim, axis)
                drive.CreateStiffnessAttr().Set(0.0)
                drive.CreateDampingAttr().Set(0.0)
    if entity.kind == "rigid" and len(rigid_bodies) != 1:
        raise RuntimeError(f"rigid entity {entity.name} has {len(rigid_bodies)} native bodies")
    for path in remove_joints:
        # Converter prims may be authored in referenced layers. An inactive
        # override suppresses them; RemovePrim would merely reveal the reference.
        stage.GetPrimAtPath(path).SetActive(False)
    missing = [name for name in entity.body_names if name not in body_paths]
    if require_bodies and missing:
        raise RuntimeError(
            f"entity {entity.name} converted rigid-body paths are missing bodies: {missing}"
        )
    relative = ""
    if entity.kind == "articulation":
        if len(articulation_roots) != 1:
            raise RuntimeError(f"entity {entity.name} has ambiguous articulation roots: "
                               f"{articulation_roots}")
        if entity.root_mode == "fixed":
            # A world joint attached to an API-bearing rigid link remains a
            # floating articulation constrained by a maximal-coordinate joint.
            # PhysX recognizes a fixed-base tree when the root API is on the
            # encompassing prim instead (IsaacLab's fix_root_link convention).
            body_prim = stage.GetPrimAtPath(articulation_roots[0])
            body_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if body_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                body_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            UsdPhysics.ArticulationRootAPI.Apply(root)
            PhysxSchema.PhysxArticulationAPI.Apply(root)
        else:
            relative = articulation_roots[0][len(str(root.GetPath())):]
    stage.GetRootLayer().Save()
    return relative


class SceneWorkerContext:
    """One independent native asset view per declared entity."""

    def __init__(self, protocol: Any, renderer: Any) -> None:
        self.protocol = protocol
        self.renderer = renderer
        self.slots: dict[str, np.ndarray] = {}
        self._shm_handles: list[Any] = []
        self.assets: list[Any] = []
        self.maps: list[dict[str, Any]] = []
        self.contact_sensors: list[Any] = []
        self.contact_sensor_maps: list[dict[str, Any]] = []
        self.contact_force_sensors: list[dict[str, str]] = []
        self.faulted = False
        self.legacy_projection: Any = None
        self._legacy_metadata: dict[str, Any] | None = None
        self._temporary = tempfile.TemporaryDirectory(prefix="unisim-isaacsim-scene-")

    def _tensor(self, values: np.ndarray) -> Any:
        return self.torch.as_tensor(
            np.ascontiguousarray(values), dtype=self.torch.float32, device=self.device
        )

    def init_sim(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "scene_layout" not in payload:
            metadata = self.renderer.init_sim(payload)
            self.adopt_initialized_context(self.renderer, metadata, payload)
            return cast(dict[str, Any], metadata)
        self.layout = validate_scene_payload(self.protocol, payload)
        self.contact_force_sensors = _validate_contact_force_sensors(payload, self.layout)
        self.entity_components = {
            entity.name: _entity_prim_component(entity.name) for entity in self.layout.entities
        }
        self.num_envs = payload["num_envs"]
        self.entries = payload["scene_entities"]
        self.sim_dt = float(payload["sim_dt"])
        device_id = int(payload.get("device_id", 0))
        if device_id < 0:
            raise NotImplementedError("IsaacSim mapped scene requires CUDA")
        self.device = f"cuda:{device_id}"
        render_mode = payload.get("render_mode", "none")
        if render_mode not in ("none", "record", "interactive"):
            raise ValueError("invalid render_mode")
        os.environ["HEADLESS"] = "0" if render_mode == "interactive" else "1"
        os.environ["ENABLE_CAMERAS"] = "1" if render_mode == "record" else "0"
        os.environ["LIVESTREAM"] = "0"
        os.environ["XR"] = "0"
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
        from isaaclab.app import AppLauncher

        app = AppLauncher({"headless": render_mode != "interactive",
                           "enable_cameras": render_mode == "record", "device": self.device,
                           "multi_gpu": False}).app
        self.renderer.simulation_app = app
        import isaaclab.sim as sim_utils
        import isaacsim.core.utils.prims as prim_utils
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
        from isaacsim.core.cloner import GridCloner
        from isaacsim.core.utils.extensions import enable_extension

        self.torch = torch
        enable_extension("isaacsim.asset.importer.mjcf")
        self.gravity = np.asarray(payload["gravity"], dtype=np.float64)
        if self.gravity.shape != (3,) or not np.isfinite(self.gravity).all():
            raise ValueError("invalid gravity")
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.sim_dt, device=self.device, gravity=tuple(self.gravity.tolist())))
        cloner = GridCloner(spacing=2.0)
        cloner.define_base_env("/World/envs")
        self.env_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        self.entity_paths = {
            name: [path + "/" + component for path in self.env_paths]
            for name, component in self.entity_components.items()
        }
        prim_utils.create_prim(self.env_paths[0], "Xform")
        self.origins = np.asarray(cloner.clone(
            source_prim_path=self.env_paths[0], prim_paths=self.env_paths,
            replicate_physics=False, copy_from_source=True), dtype=np.float32)
        self.usd_paths = []
        self.entity_body_paths: list[list[dict[str, str]]] = []
        for entity, entry in zip(self.layout.entities, self.entries):
            component = self.entity_components[entity.name]
            paths, root_paths = [], []
            body_paths_by_variant: list[dict[str, str]] = []
            for index, source in enumerate(entry["sources"]):
                converter = MjcfConverter(MjcfConverterCfg(
                    asset_path=source, fix_base=entity.kind == "articulation"
                    and entity.root_mode == "fixed", import_sites=False,
                    import_inertia_tensor=True, make_instanceable=False, self_collision=False,
                    force_usd_conversion=True,
                    usd_dir=os.path.join(self._temporary.name, component, str(index)),
                    usd_file_name=f"{component}_{index}.usd"))
                paths.append(converter.usd_path)
                body_paths: dict[str, str] = {}
                root_paths.append(
                    _bake(
                        converter.usd_path,
                        entity,
                        entry,
                        index,
                        body_paths,
                        require_bodies=bool(self.contact_force_sensors),
                    )
                )
                body_paths_by_variant.append(body_paths)
            if len(set(root_paths)) != 1:
                raise RuntimeError("variant articulation root paths differ")
            if self.contact_force_sensors:
                canonical_body_paths = body_paths_by_variant[0]
                for body_name in entity.body_names:
                    for variant_body_paths in body_paths_by_variant[1:]:
                        if variant_body_paths.get(body_name) != canonical_body_paths[body_name]:
                            raise RuntimeError(
                                f"entity {entity.name} variant rigid-body prim paths differ"
                            )
            self.entity_body_paths.append(body_paths_by_variant)
            self.usd_paths.append(paths)
            spawn = sim_utils.MultiUsdFileCfg(usd_path=paths, random_choice=False)
            spawn.activate_contact_sensors = bool(self.contact_force_sensors)
            prim_path = "/World/envs/env_.*/" + component
            if entity.kind == "articulation":
                names = [joint.name for joint in entity.joints]
                gains = self.renderer._actuator_dicts(entry["variants"][0], names)
                actuators = {}
                if names:
                    actuators["declared"] = ImplicitActuatorCfg(
                        joint_names_expr=names, stiffness=gains["stiffness"],
                        damping=gains["damping"], effort_limit_sim=gains["effort"],
                        armature=gains["armature"], friction=gains["friction"])
                asset = Articulation(ArticulationCfg(
                    prim_path=prim_path, articulation_root_prim_path=root_paths[0],
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=tuple(entry["initial_pose"][:3]),
                        rot=tuple(entry["initial_pose"][3:])),
                    spawn=spawn, actuators=actuators))
            else:
                asset = RigidObject(RigidObjectCfg(
                    prim_path=prim_path, spawn=spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(entry["initial_pose"][:3]),
                        rot=tuple(entry["initial_pose"][3:]))))
            self.assets.append(asset)
        entity_indexes = {entity.name: index for index, entity in enumerate(self.layout.entities)}
        for record in self.contact_force_sensors:
            source_index = entity_indexes[record["source_entity"]]
            target_index = entity_indexes[record["target_entity"]]
            source_component = self.entity_components[record["source_entity"]]
            target_component = self.entity_components[record["target_entity"]]
            source_path = (
                "/World/envs/env_.*/" + source_component
                + self.entity_body_paths[source_index][0][record["source_body"]]
            )
            target_path = (
                "/World/envs/env_.*/" + target_component
                + self.entity_body_paths[target_index][0][record["target_body"]]
            )
            self.contact_sensors.append(ContactSensor(ContactSensorCfg(
                prim_path=source_path,
                filter_prim_paths_expr=[target_path],
                history_length=0,
            )))
        self._bind_fixed_anchors()
        if self.num_envs > 1:
            cloner.filter_collisions(self.sim.cfg.physics_prim_path, "/World/collisions",
                                     self.env_paths)
        self._setup_renderer(sim_utils, payload)
        self.sim.reset()
        for entity, asset in zip(self.layout.entities, self.assets):
            asset.update(self.sim_dt)
        for sensor, record in zip(self.contact_sensors, self.contact_force_sensors):
            sensor.update(self.sim_dt)
            native_paths = list(sensor.body_physx_view.prim_paths)
            native_envs = _native_environment_order(
                native_paths, self.entity_paths[record["source_entity"]]
            )
            self.contact_sensor_maps.append(
                {"envs": np.argsort(native_envs), "public_for_native": np.asarray(native_envs)}
            )
        for entity, asset in zip(self.layout.entities, self.assets):
            native_paths = list(asset.root_physx_view.prim_paths)
            native_envs = _native_environment_order(native_paths, self.entity_paths[entity.name])
            env_map = np.argsort(native_envs)
            native_bodies = list(asset.body_names)
            bodies = self.renderer._build_permutation(
                native_bodies, list(entity.body_names), "body")
            native_joints = list(asset.joint_names) if entity.kind == "articulation" else []
            if entity.kind == "articulation" and (
                bool(asset.is_fixed_base) != (entity.root_mode == "fixed")
            ):
                raise RuntimeError("native articulation root mode differs from declaration")
            joints = self.renderer._build_permutation(
                native_joints, [joint.name for joint in entity.joints], "joint")
            control_joints = np.asarray([native_joints.index(name)
                                        for name in entity.actuator_joint_names], dtype=np.int64)
            self.maps.append({"bodies": bodies, "joints": joints, "controls": control_joints,
                              "envs": env_map, "public_for_native": np.asarray(native_envs)})
        ids = np.arange(self.num_envs, dtype=np.int64)
        self._commit(ids, np.asarray(payload["initial_qpos"], dtype=np.float32),
                     np.asarray(payload["initial_qvel"], dtype=np.float32),
                     np.asarray(payload["initial_roots"], dtype=np.float32),
                     np.ones(self.layout.nq, dtype=np.uint8),
                     np.ones(self.layout.nv, dtype=np.uint8),
                     np.ones((len(self.assets), 2), dtype=np.uint8), initializing=True)
        if "initial_ctrl" in payload:
            self._set_control_targets(np.asarray(payload["initial_ctrl"], dtype=np.float32))
        self.actual = self._audit_instances()
        return self.get_meta()

    def adopt_initialized_context(
        self, renderer: Any, metadata: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        """Project the old wire onto this runtime after its existing cold importer.

        This adopts live views and the already initialized Kit context. The
        synthetic execution layout preserves the historical D-wide control
        surface; it is never published as a new physical entity capability.
        """
        bridge = self.protocol.load_legacy_projection()
        self.renderer = renderer
        self.num_envs, self.sim_dt = renderer.num_envs, renderer.sim_dt
        self.sim, self.torch, self.device = renderer.sim, renderer.torch, renderer.device
        self.origins = np.asarray(renderer.env_origins, dtype=np.float32).copy()
        if self.origins.shape != (self.num_envs, 3) or not np.isfinite(self.origins).all():
            raise RuntimeError("legacy native environment origins are malformed")
        self.layout = bridge.LegacyExecutionLayout(
            renderer.contract_joint_names, renderer.contract_body_names,
            root_body_name=payload.get("root_body_name"))
        self.legacy_projection = bridge.LegacySlotProjection(
            self.protocol, self.num_envs, self.layout)
        self.assets = [renderer.robot]
        # Resolve instance rows from the existing view rather than assuming
        # GridCloner creation order equals PhysX view order.
        roots = [path + "/Robot" for path in renderer.env_prim_paths]
        native_envs = _native_environment_order(
            list(renderer.robot.root_physx_view.prim_paths), roots)
        joints = np.asarray(renderer.native_joint_for_contract, dtype=np.int64).copy()
        bodies = np.asarray(renderer.native_body_for_contract, dtype=np.int64).copy()
        self.maps = [{"envs": np.argsort(native_envs), "public_for_native": native_envs,
                      "joints": joints, "bodies": bodies, "controls": joints.copy()}]
        self._legacy_metadata = metadata.copy()

    def _setup_renderer(self, sim_utils: Any, payload: dict[str, Any]) -> None:
        owner = self.renderer
        owner.sim, owner.torch, owner.device = self.sim, self.torch, self.device
        owner.num_envs, owner.sim_dt = self.num_envs, self.sim_dt
        owner.robot = self.assets[0]
        owner.render_mode = payload.get("render_mode", "none")
        owner.render_width = payload.get("render_width", 1280)
        owner.render_height = payload.get("render_height", 720)
        if owner.render_mode != "none":
            light = sim_utils.DomeLightCfg(intensity=100.0)
            light.func("/World/UniSimLight", light)
        if owner.render_mode == "record":
            from isaaclab.sensors.camera import Camera, CameraCfg

            owner.camera = Camera(CameraCfg(
                prim_path="/World/envs/env_0/UniSimCamera", update_period=0.0,
                data_types=["rgb"], width=owner.render_width, height=owner.render_height,
                spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.1, 1e5))))

    def _bind_fixed_anchors(self) -> None:
        """Place imported world joints at the actual cloned root world transform."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        for entity, asset in zip(self.layout.entities, self.assets):
            if entity.kind != "articulation" or entity.root_mode != "fixed":
                continue
            for path in self.entity_paths[entity.name]:
                prim = asset.stage.GetPrimAtPath(path)
                fixed = []
                for child in Usd.PrimRange(prim):
                    if not child.IsA(UsdPhysics.FixedJoint):
                        continue
                    joint = UsdPhysics.FixedJoint(child)
                    body0 = joint.GetBody0Rel().GetTargets()
                    body1 = joint.GetBody1Rel().GetTargets()
                    if not body0 and len(body1) == 1:
                        fixed.append((joint, body1[0]))
                if len(fixed) != 1:
                    raise RuntimeError("fixed entity requires one world anchor joint")
                joint, root_path = fixed[0]
                transform = UsdGeom.XformCache().GetLocalToWorldTransform(
                    asset.stage.GetPrimAtPath(root_path))
                position = transform.ExtractTranslation()
                quaternion = transform.ExtractRotationQuat()
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*position))
                joint.CreateLocalRot0Attr().Set(Gf.Quatf(quaternion))
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))

    def _audit_instances(self) -> list[dict[str, Any]]:
        """Read identity from spawned stage and mass from actual PhysX views."""
        from pxr import Usd, UsdPhysics

        result = []
        for entity, entry, asset, mapping in zip(
            self.layout.entities, self.entries, self.assets, self.maps
        ):
            paths = self.entity_paths[entity.name]
            native_paths = [asset.root_physx_view.prim_paths[i] for i in mapping["envs"]]
            if not np.array_equal(_native_environment_order(native_paths, paths),
                                  np.arange(self.num_envs)):
                raise RuntimeError("native view row order differs from environment order")
            observed = []
            for path in paths:
                prim = asset.stage.GetPrimAtPath(path)
                value = prim.GetAttribute("unisim:variantIndex").Get()
                if not isinstance(value, int):
                    raise RuntimeError("spawned asset has no observable variant identity")
                observed.append(value)
                if not entry["collision_enabled"]:
                    for child in Usd.PrimRange(prim):
                        if child.HasAPI(UsdPhysics.CollisionAPI) and (
                            UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get()
                        ):
                            raise RuntimeError("collision-disabled entity has an enabled collider")
            if observed != entry["assignment"]:
                raise RuntimeError("actual spawned variant assignment differs from requested")
            masses = _numpy(asset.root_physx_view.get_masses()).reshape(
                self.num_envs, -1)[mapping["envs"]]
            masses = masses[:, mapping["bodies"]]
            expected = np.asarray([entry["variants"][index]["body_mass"] for index in observed])
            if not np.allclose(masses, expected, rtol=2e-4, atol=1e-6):
                raise RuntimeError(f"entity {entity.name} native body masses differ: "
                                   f"actual={masses.tolist()}, requested={expected.tolist()}")
            coms = _numpy(asset.root_physx_view.get_coms()).reshape(
                self.num_envs, -1, 7)[mapping["envs"]]
            coms = coms[:, mapping["bodies"]]
            expected_coms = np.asarray([
                entry["variants"][variant]["body_ipos"] for variant in observed])
            if not np.allclose(coms[:, :, :3], expected_coms, rtol=1e-4, atol=1e-6):
                raise RuntimeError(f"entity {entity.name} native COM differs from source")
            inertias = _numpy(asset.root_physx_view.get_inertias()).reshape(
                self.num_envs, -1, 3, 3)[mapping["envs"]][:, mapping["bodies"]]
            expected_inertias = []
            for variant in observed:
                record = entry["variants"][variant]
                matrices = []
                for diagonal, quaternion in zip(record["body_inertia"], record["body_iquat"]):
                    # Columns of R are independently rotated unit basis vectors.
                    rotation = _rotate(np.broadcast_to(quaternion, (3, 4)), np.eye(3)).T
                    matrices.append((rotation * np.asarray(diagonal)) @ rotation.T)
                expected_inertias.append(matrices)
            if not np.allclose(inertias, expected_inertias, rtol=2e-4, atol=1e-6):
                raise RuntimeError(f"entity {entity.name} native inertia differs from source")
            if entity.joints:
                record = entry["variants"][0]
                kinds = _numpy(asset.root_physx_view.get_dof_types())[
                    mapping["envs"]][:, mapping["joints"]]
                expected_kinds = [0 if joint.kind == "hinge" else 1 for joint in entity.joints]
                if not np.all(kinds == expected_kinds):
                    raise RuntimeError(f"entity {entity.name} native joint types differ")
                for field, actual in (
                    ("dof_stiffness", asset.root_physx_view.get_dof_stiffnesses()),
                    ("dof_damping", asset.root_physx_view.get_dof_dampings()),
                ):
                    values = _numpy(actual)[mapping["envs"]][:, mapping["joints"]]
                    if not np.allclose(values, record[field], rtol=1e-5, atol=1e-6):
                        raise RuntimeError(f"entity {entity.name} native {field} differs")
                for env in range(self.num_envs):
                    metatype = asset.root_physx_view.get_metatype(int(mapping["envs"][env]))
                    parents = dict(zip(metatype.link_names, metatype.link_parents))
                    for body, parent in zip(entity.body_names, entity.body_parent_names):
                        if parent is not None and parents[body] != parent:
                            raise RuntimeError(f"entity {entity.name} native body topology differs")
            result.append({"name": entity.name, "assignment": observed,
                           "prim_paths": native_paths, "body_mass": masses.tolist(),
                           "body_com": coms[:, :, :3].tolist(),
                           "body_inertia": inertias.tolist()})
        return result

    def attach_slots(self, payload: dict[str, Any]) -> None:
        from multiprocessing import resource_tracker, shared_memory

        shapes = (self.protocol.slot_shapes(self.num_envs, len(self.layout.entities[0].joints),
                                           self.layout.nbody)
                  if self.legacy_projection is not None
                  else self.protocol.scene_slot_shapes(
                      self.num_envs, self.layout, len(self.contact_force_sensors)))
        self.protocol.validate_slot_specs(payload["slots"], shapes)
        attached = {}
        for name, spec in payload["slots"].items():
            handle = shared_memory.SharedMemory(name=spec["shm"], create=False)
            resource_tracker.unregister(handle._name, "shared_memory")  # type: ignore[attr-defined]
            self._shm_handles.append(handle)
            attached[name] = np.ndarray(tuple(spec["shape"]), dtype=spec["dtype"],
                                       buffer=handle.buf)
        self.slots = (attached if self.legacy_projection is None
                      else self.legacy_projection.attach(attached))
        self.refresh_state_slots()

    def refresh_state_slots(self) -> None:
        for field in ("qpos", "qvel", "entity_root_state", "body_state", "contact_force"):
            self.slots[field].fill(0)
        if "contact_sensor_force" in self.slots:
            self.slots["contact_sensor_force"].fill(0)
        # Unowned engine-world bodies still have a valid identity orientation.
        self.slots["body_state"][:, :, 3] = 1
        for index, (entity, asset, mapping) in enumerate(zip(
            self.layout.entities, self.assets, self.maps
        )):
            root = _numpy(asset.data.root_link_state_w)[mapping["envs"]].copy()
            root[:, :3] -= self.origins
            bodies = _numpy(asset.data.body_link_state_w)[mapping["envs"]].copy()
            bodies[:, :, :3] -= self.origins[:, None, :]
            self.slots["entity_root_state"][:, index] = root
            self.slots["body_state"][:, entity.body_ids] = bodies[:, mapping["bodies"]]
            if entity.root_mode == "floating":
                self.slots["qpos"][:, entity.root_qpos_indices] = root[:, :7]
                velocity = root[:, 7:13].copy()
                velocity[:, 3:] = _rotate(root[:, 3:7], velocity[:, 3:], inverse=True)
                self.slots["qvel"][:, entity.root_qvel_indices] = velocity
            if entity.joints:
                pos = _numpy(asset.data.joint_pos)[mapping["envs"]][:, mapping["joints"]]
                vel = _numpy(asset.data.joint_vel)[mapping["envs"]][:, mapping["joints"]]
                self.slots["qpos"][:, [j.qpos_indices[0] for j in entity.joints]] = pos
                self.slots["qvel"][:, [j.qvel_indices[0] for j in entity.joints]] = vel
        if self.legacy_projection is not None:
            self.legacy_projection.publish()

    def _refresh_contact_sensor_forces(self) -> None:
        """Publish final-substep filtered normal forces in world coordinates."""
        if not self.contact_sensors:
            return
        if "contact_sensor_force" not in self.slots:
            raise RuntimeError("contact sensors were initialized without their shared-memory slot")
        for index, (sensor, mapping) in enumerate(
            zip(self.contact_sensors, self.contact_sensor_maps)
        ):
            matrix = _numpy(sensor.data.force_matrix_w)
            if matrix.shape != (self.num_envs, 1, 1, 3):
                raise RuntimeError(
                    f"contact sensor {index} returned shape {matrix.shape}; expected "
                    f"{(self.num_envs, 1, 1, 3)}"
                )
            forces = matrix.reshape(self.num_envs, 3)[mapping["envs"]]
            if not np.isfinite(forces).all():
                raise RuntimeError(f"contact sensor {index} returned non-finite force")
            self.slots["contact_sensor_force"][:, index, :] = forces

    def set_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Translate only the old wire; native reset always uses reset_entities."""
        if self.legacy_projection is None:
            raise NotImplementedError("mapped scenes require RESET_ENTITIES")
        count = payload["count"]
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= self.num_envs:
            raise ValueError("invalid reset count")
        if count == 0:
            return {"timing": {}}
        return self.reset_entities(self.legacy_projection.prepare_reset(count))

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        count = payload["nsteps"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("nsteps must be a positive integer")
        if not np.isfinite(self.slots["ctrl"]).all():
            raise ValueError("control must be finite")
        started = time.perf_counter()
        try:
            self._set_control_targets(self.slots["ctrl"])
            for _ in range(count):
                for asset in self.assets:
                    asset.write_data_to_sim()
                self.sim.step(render=False)
                for asset in self.assets:
                    asset.update(self.sim_dt)
                for sensor in self.contact_sensors:
                    sensor.update(self.sim_dt)
            self.refresh_state_slots()
            self._refresh_contact_sensor_forces()
        except Exception:
            self.faulted = True
            raise
        return {"timing": {"physics_ms": (time.perf_counter() - started) * 1000}}

    def _set_control_targets(self, control: np.ndarray) -> None:
        """Apply actuator columns; keyframe controls are independent of joint positions."""
        for entity, asset, mapping in zip(self.layout.entities, self.assets, self.maps):
            if entity.actuator_indices:
                asset.set_joint_position_target(
                    self._tensor(control[mapping["public_for_native"]][:, entity.actuator_indices]),
                    joint_ids=mapping["controls"].tolist())

    def _commit(self, ids: np.ndarray, qpos: np.ndarray, qvel: np.ndarray,
                roots: np.ndarray, pmask: np.ndarray, vmask: np.ndarray,
                rmask: np.ndarray, *, initializing: bool = False) -> None:
        try:
            for index, (entity, asset, mapping) in enumerate(zip(
                self.layout.entities, self.assets, self.maps
            )):
                touched = bool(rmask[index].any())
                pcols = [j.qpos_indices[0] for j in entity.joints]
                vcols = [j.qvel_indices[0] for j in entity.joints]
                selected = np.flatnonzero(pmask[pcols] | vmask[vcols])
                if not touched and not selected.size:
                    continue
                native_rows = mapping["envs"][ids]
                native_ids = self.torch.as_tensor(
                    native_rows, dtype=self.torch.long, device=self.device)
                if rmask[index, 0] and (entity.root_mode != "fixed" or
                                       initializing and entity.kind == "rigid"):
                    pose = roots[:, index, :7].copy()
                    pose[:, :3] += self.origins[ids]
                    asset.write_root_pose_to_sim(self._tensor(pose), env_ids=native_ids)
                if rmask[index, 1] and entity.root_mode == "floating":
                    asset.write_root_link_velocity_to_sim(
                        self._tensor(roots[:, index, 7:]), env_ids=native_ids)
                if entity.joints:
                    if selected.size:
                        touched = True
                        joint_ids = mapping["joints"][selected].tolist()
                        native_joints = self.torch.as_tensor(
                            joint_ids, dtype=self.torch.long, device=self.device)
                        # Gather on-device before crossing the CPU boundary;
                        # a sparse reset must not download every environment.
                        positions = _numpy(asset.data.joint_pos[
                            native_ids[:, None], native_joints]).copy()
                        velocities = _numpy(asset.data.joint_vel[
                            native_ids[:, None], native_joints]).copy()
                        for column, public_index in enumerate(selected):
                            if pmask[pcols[public_index]]:
                                positions[:, column] = qpos[:, pcols[public_index]]
                            if vmask[vcols[public_index]]:
                                velocities[:, column] = qvel[:, vcols[public_index]]
                        asset.write_joint_state_to_sim(
                            self._tensor(positions), self._tensor(velocities),
                            joint_ids=joint_ids, env_ids=native_ids)
                if touched:
                    asset.reset(native_ids)
                    asset.update(self.sim_dt)
        except Exception:
            self.faulted = True
            raise

    def reset_entities(self, payload: dict[str, Any]) -> dict[str, Any]:
        count = payload["count"]
        if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= self.num_envs:
            raise ValueError("invalid reset count")
        ids = self.slots["reset_env_ids"][:count].astype(np.int64, copy=True)
        if np.unique(ids).size != count or np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise ValueError("invalid reset environment IDs")
        names = payload["entity_names"]
        if not isinstance(names, list) or len(names) != len(set(names)):
            raise ValueError("entity_names must be unique names")
        if not set(names).issubset({entity.name for entity in self.layout.entities}):
            raise ValueError("unknown reset entity")
        qpos = self.slots["reset_qpos"][:count].copy()
        qvel = self.slots["reset_qvel"][:count].copy()
        roots = self.slots["reset_entity_root_state"][:count].copy()
        pmask = self.slots["reset_qpos_mask"].copy()
        vmask = self.slots["reset_qvel_mask"].copy()
        rmask = self.slots["reset_root_mask"].copy()
        if any(not np.isfinite(values).all() for values in (qpos, qvel, roots)):
            raise ValueError("reset values must be finite")
        if any(np.any((mask != 0) & (mask != 1)) for mask in (pmask, vmask, rmask)):
            raise ValueError("reset masks must contain zero or one")
        control_values = None
        control_columns = tuple(column for entity in self.layout.entities if entity.name in names
                                for column in entity.actuator_indices)
        if "control_values" in payload:
            raw_control = np.asarray(payload["control_values"])
            if (raw_control.shape != (count, self.layout.nu)
                    or raw_control.dtype.kind not in "fiu"
                    or not np.isfinite(raw_control).all()
                    or np.any(np.abs(raw_control.astype(np.float64)) > np.finfo(np.float32).max)):
                raise ValueError("control_values must be finite (count, nu) float32 values")
            control_values = raw_control.astype(np.float32, copy=True)
            unselected = [column for column in range(self.layout.nu)
                          if column not in control_columns]
            if not np.array_equal(control_values[:, unselected],
                                  self.slots["ctrl"][ids][:, unselected]):
                raise ValueError("control_values changes controls of an unselected entity")
        for index, entity in enumerate(self.layout.entities):
            selected = bool(pmask[list(entity.qpos_indices)].any()
                            or vmask[list(entity.qvel_indices)].any() or rmask[index].any())
            if selected and entity.name not in names:
                raise ValueError("reset mask writes undeclared entity")
            if entity.root_mode == "fixed" and rmask[index].any():
                raise ValueError("fixed root cannot be reset")
            if entity.root_mode == "kinematic" and rmask[index, 1]:
                raise ValueError("kinematic root velocity cannot be reset")
            if rmask[index, 0] and not np.allclose(
                np.linalg.norm(roots[:, index, 3:7], axis=1), 1, rtol=0, atol=1e-5
            ):
                raise ValueError("root reset quaternion must be unit wxyz")
            if entity.root_mode == "floating":
                for cols, mask, channel in ((entity.root_qpos_indices, pmask, 0),
                                            (entity.root_qvel_indices, vmask, 1)):
                    bits = mask[list(cols)]
                    if np.any(bits) != bool(rmask[index, channel]) or len(set(bits.tolist())) > 1:
                        raise ValueError("root generalized and entity masks disagree")
                if rmask[index, 0] and not np.allclose(
                    qpos[:, entity.root_qpos_indices], roots[:, index, :7],
                    rtol=1e-5, atol=1e-6
                ):
                    raise ValueError("root pose differs between generalized and entity values")
                if rmask[index, 1]:
                    velocity = roots[:, index, 7:].copy()
                    velocity[:, 3:] = _rotate(roots[:, index, 3:7], velocity[:, 3:], inverse=True)
                    if not np.allclose(qvel[:, entity.root_qvel_indices], velocity,
                                       rtol=1e-5, atol=1e-6):
                        raise ValueError("generalized and entity root velocities differ")
        self._commit(ids, qpos, qvel, roots, pmask, vmask, rmask)
        try:
            if control_values is not None:
                for entity, asset, mapping in zip(self.layout.entities, self.assets, self.maps):
                    if entity.name in names and entity.actuator_indices:
                        native_ids = self.torch.as_tensor(
                            mapping["envs"][ids], dtype=self.torch.long, device=self.device)
                        asset.set_joint_position_target(
                            self._tensor(control_values[:, entity.actuator_indices]),
                            joint_ids=mapping["controls"].tolist(), env_ids=native_ids)
                self.slots["ctrl"][np.ix_(ids, control_columns)] = control_values[
                    :, control_columns]
            self.refresh_state_slots()
        except Exception:
            self.faulted = True
            raise
        return {"timing": {}}

    def get_meta(self) -> dict[str, Any]:
        if self._legacy_metadata is not None:
            return self._legacy_metadata.copy()
        return {"scene_layout": self.layout.to_dict(), "scene_entities_actual": self.actual,
                "gravity": self.gravity.tolist(), "use_gpu_pipeline": True,
                "env_origins": self.origins.tolist(),
                "collision_filtering_applied": self.num_envs > 1,
                "render_mode": self.renderer.render_mode,
                "render_width": self.renderer.render_width,
                "render_height": self.renderer.render_height,
                "graphics_enabled": self.renderer.render_mode != "none",
                "configuration_report": {
                    "schema_version": 1,
                    "effective": {"dt": float(self.sim.get_physics_dt()),
                                  "gravity": self.gravity.tolist(),
                                  "collision_filter": {"self_collision": False,
                                                       "environment_isolation": True,
                                                       "implicit_ground": False}},
                    "engine_readback": ["dt"],
                }}

    def shutdown(self) -> None:
        for handle in self._shm_handles:
            handle.close()
        self._shm_handles.clear()
        self.renderer.shutdown()
        self._temporary.cleanup()

    def init_renderer(self, payload: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], self.renderer.init_renderer(payload))

    def render_frame(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.renderer.render_frame())

    def capture_frame(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.renderer.capture_frame())
