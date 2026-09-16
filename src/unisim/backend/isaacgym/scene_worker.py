"""Native multi-actor execution for the versioned scene IPC profile.

Loaded by file path in the isolated Python 3.8 worker. Only stdlib and NumPy
are imported here; IsaacGym must be imported before Torch during initialization.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np


def finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fi" or not np.isfinite(array).all():
        raise ValueError("%s must be a finite real array of shape %s" % (label, shape))
    return np.array(array, dtype=np.float64, copy=True)


def unit_quaternion(value: np.ndarray, label: str) -> None:
    if not np.allclose(np.linalg.norm(value, axis=-1), 1.0, rtol=0, atol=1e-5):
        raise ValueError(label + " requires unit quaternions")


class SceneWorker:
    """Own one native scene, public/native maps and pending indexed writes."""

    def __init__(self, context: Any, payload: dict[str, Any]) -> None:
        self.ctx = context
        self.protocol = context.protocol
        self.layout = self.protocol.load_scene_layout(payload["scene_layout"])
        self.payload = payload
        count = payload["num_envs"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = count
        self.specs = payload["scene_entities"]
        self._validate_sources()
        self.qpos0 = finite_array(payload["initial_qpos"], (count, self.layout.nq), "initial_qpos")
        self.qvel0 = finite_array(payload["initial_qvel"], (count, self.layout.nv), "initial_qvel")
        self.roots0 = finite_array(
            payload["initial_roots"], (count, len(self.layout.entities), 13), "initial_roots"
        )
        unit_quaternion(self.roots0[..., 3:7], "initial_roots")
        for i, entity in enumerate(self.layout.entities):
            if entity.root_mode == "floating":
                unit_quaternion(self.qpos0[:, entity.root_qpos_indices[3:]], "initial_qpos")
                if not np.allclose(self.qpos0[:, entity.root_qpos_indices], self.roots0[:, i, :7]):
                    raise ValueError("initial root pose disagrees with generalized state")
                expected = self.roots0[:, i, 7:].copy()
                expected[:, 3:] = self.protocol.quat_rotate_inverse(
                    self.roots0[:, i, 3:7], expected[:, 3:]
                )
                if not np.allclose(self.qvel0[:, entity.root_qvel_indices], expected):
                    raise ValueError("initial root velocity disagrees with generalized state")
        self.gravity = finite_array(payload["gravity"], (3,), "gravity")
        raw_initial_ctrl = payload.get("initial_ctrl")
        self.initial_ctrl = (
            None
            if raw_initial_ctrl is None
            else finite_array(raw_initial_ctrl, (count, self.layout.nu), "initial_ctrl")
        )
        self.records: list[list[dict[str, Any]]] = []
        self.assets: list[list[Any]] = []
        self.actor_ids = np.empty((count, len(self.layout.entities)), dtype=np.int64)
        self.body_ids = np.full((count, self.layout.nbody), -1, dtype=np.int64)
        self.body_com = np.zeros((count, self.layout.nbody, 3))
        self.root_com = np.zeros((count, len(self.layout.entities), 3))
        self.control_dofs = np.empty((count, self.layout.nu), dtype=np.int64)
        self.pending_roots: dict[int, np.ndarray] = {}
        self.pending_dofs: dict[int, np.ndarray] = {}
        self.pending_dof_actors: set[int] = set()
        self.faulted = False
        self.metadata: dict[str, Any] = {}

    def _validate_sources(self) -> None:
        if not isinstance(self.specs, list) or len(self.specs) != len(self.layout.entities):
            raise ValueError("scene_entities must match scene_layout")
        if not self.specs or len(self.specs) > 30:
            raise ValueError(
                "IsaacGym scene profile supports 1..30 entities for collision filtering"
            )
        by_name = {entity.name: i for i, entity in enumerate(self.layout.entities)}
        for entity, spec in zip(self.layout.entities, self.specs):
            if not isinstance(spec, dict):
                raise TypeError("scene entity declaration must be a dictionary")
            if any(spec.get(key) != getattr(entity, key) for key in ("name", "kind", "root_mode")):
                raise ValueError("entity declaration differs from public layout")
            if spec.get("asset_format") != "mjcf":
                raise NotImplementedError("IsaacGym mapped scene profile currently requires MJCF")
            if any(joint.kind not in ("hinge", "slide") for joint in entity.joints):
                raise NotImplementedError("IsaacGym scene profile supports hinge/slide joints only")
            if len(set(entity.actuator_joint_names)) != len(entity.actuator_joint_names):
                raise NotImplementedError("IsaacGym position drives require one actuator per joint")
            if not isinstance(spec.get("collision_enabled"), bool):
                raise TypeError("collision_enabled must be bool")
            mirror = spec.get("mirror_of")
            if mirror is not None:
                if mirror not in by_name or mirror == entity.name:
                    raise ValueError("mirror must reference another physical entity")
                if (
                    entity.kind != "rigid"
                    or entity.root_mode != "kinematic"
                    or spec["collision_enabled"]
                ):
                    raise ValueError("mirror requires a collision-disabled kinematic rigid entity")
                target = self.specs[by_name[mirror]]
                if target.get("mirror_of") is not None:
                    raise ValueError("mirror chains are unsupported")
                if spec.get("assignment") != target.get("assignment"):
                    raise ValueError("mirror assignment differs from its consumer")
            sources, variants = spec.get("sources"), spec.get("variants")
            if (
                not isinstance(sources, list)
                or not sources
                or not isinstance(variants, list)
                or len(variants) != len(sources)
            ):
                raise ValueError("sources and variants must be aligned non-empty lists")
            for source in sources:
                if (
                    not isinstance(source, str)
                    or not os.path.isabs(source)
                    or not os.path.isfile(source)
                ):
                    raise ValueError("entity sources must be existing absolute paths")
                # Canonical MuJoCo XML rewrites position actuators as general;
                # Gym's importer can loop forever on unsupported actuator tags.
                xml = ET.parse(source).getroot()
                if xml.findall(".//include"):
                    raise ValueError("scene worker requires materialized self-contained MJCF")
                for actuator in xml.findall("actuator/*"):
                    if actuator.tag not in ("motor", "position", "velocity"):
                        raise ValueError("unsafe/unsupported IsaacGym actuator tag " + actuator.tag)
                source_parents: dict[str, str | None] = {}
                source_joints: dict[str, tuple[str, str]] = {}

                def scan(body: Any, parent: str | None) -> None:
                    name = body.get("name")
                    if not name or name in source_parents:
                        raise ValueError("entity source requires unique named bodies")
                    source_parents[name] = parent
                    for joint in body.findall("joint"):
                        kind = joint.get("type", "hinge")
                        if kind != "free":
                            joint_name = joint.get("name")
                            if not joint_name or joint_name in source_joints:
                                raise ValueError("entity source requires unique named joints")
                            source_joints[joint_name] = (name, kind)
                    for child in body.findall("body"):
                        scan(child, name)

                for root in xml.findall("worldbody/body"):
                    scan(root, None)
                if source_parents != dict(zip(entity.body_names, entity.body_parent_names)):
                    raise ValueError("entity source body topology differs from scene layout")
                if source_joints != {j.name: (j.body_name, j.kind) for j in entity.joints}:
                    raise ValueError("entity source joint topology differs from scene layout")
            assignment = spec.get("assignment")
            if (
                not isinstance(assignment, list)
                or len(assignment) != self.num_envs
                or any(
                    isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(sources)
                    for i in assignment
                )
            ):
                raise ValueError("entity assignment must contain one valid integer per environment")
            pose = finite_array(spec.get("initial_pose"), (7,), "initial_pose")
            unit_quaternion(pose[3:], "initial_pose")
            for variant in variants:
                if not isinstance(variant, dict):
                    raise TypeError("variant metadata must be a dictionary")
                if (
                    variant.get("joint_names") != [j.name for j in entity.joints]
                    or variant.get("actuator_names") != list(entity.actuator_names)
                    or variant.get("actuator_joint_names") != list(entity.actuator_joint_names)
                    or variant.get("body_names") != list(entity.body_names)
                ):
                    raise ValueError("variant names/actuation differ from public layout")
                nj, nb = len(entity.joints), len(entity.body_names)
                for name in ("stiffness", "damping", "effort", "armature", "friction"):
                    values = finite_array(variant.get("dof_" + name), (nj,), "dof_" + name)
                    if np.any(values < 0):
                        raise ValueError("negative dof_" + name)
                passive_damping = finite_array(
                    variant.get("dof_passive_damping", [0.0] * nj), (nj,), "dof_passive_damping"
                )
                if np.any(passive_damping != 0):
                    raise NotImplementedError(
                        "IsaacGym mapped profile has not validated joint passive damping"
                    )
                lower = np.asarray(variant.get("dof_lower"), dtype=float)
                upper = np.asarray(variant.get("dof_upper"), dtype=float)
                if (
                    lower.shape != (nj,)
                    or upper.shape != (nj,)
                    or np.isnan(lower).any()
                    or np.isnan(upper).any()
                    or np.any(lower > upper)
                ):
                    raise ValueError("invalid DOF limits")
                for name, shape in (
                    ("body_mass", (nb,)),
                    ("body_ipos", (nb, 3)),
                    ("body_inertia", (nb, 3)),
                    ("body_iquat", (nb, 4)),
                ):
                    finite_array(variant.get(name), shape, name)
                unit_quaternion(np.asarray(variant["body_iquat"]), "body_iquat")

    def initialize(self) -> dict[str, Any]:
        ctx = self.ctx
        sdk = self.payload["isaacgym_python"]
        if sdk not in sys.path:
            sys.path.insert(0, sdk)
        from isaacgym import gymapi, gymtorch  # noqa: I001 - SDK must precede Torch
        import torch

        ctx.gymapi, ctx.gymtorch, ctx.torch = gymapi, gymtorch, torch
        ctx.gym = gymapi.acquire_gym()
        ctx.num_envs = self.num_envs
        ctx.num_dof = sum(len(e.joints) for e in self.layout.entities)
        ctx.num_bodies = self.layout.nbody
        ctx.sim_dt = float(self.payload["sim_dt"])
        if not np.isfinite(ctx.sim_dt) or ctx.sim_dt <= 0:
            raise ValueError("sim_dt must be positive and finite")
        device_id = int(self.payload.get("device_id", 0))
        ctx.use_gpu_pipeline = device_id >= 0
        ctx.device = "cuda:%d" % device_id if ctx.use_gpu_pipeline else "cpu"
        params = gymapi.SimParams()
        params.dt, params.substeps = ctx.sim_dt, 1
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(*self.gravity)
        params.physx.solver_type = 1
        params.physx.num_position_iterations = 4
        params.physx.num_velocity_iterations = 1
        params.physx.use_gpu = ctx.use_gpu_pipeline
        params.use_gpu_pipeline = ctx.use_gpu_pipeline
        ctx.graphics_device_id = device_id if device_id >= 0 else -1
        ctx.sim = ctx.gym.create_sim(device_id, ctx.graphics_device_id, gymapi.SIM_PHYSX, params)
        if ctx.sim is None:
            raise RuntimeError("IsaacGym scene create_sim failed")
        for entity, spec in zip(self.layout.entities, self.specs):
            entity_assets = []
            for path, variant in zip(spec["sources"], spec["variants"]):
                options = gymapi.AssetOptions()
                options.fix_base_link = entity.root_mode != "floating"
                options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
                options.linear_damping = options.angular_damping = 0.0
                asset = ctx.gym.load_asset(
                    ctx.sim, os.path.dirname(path), os.path.basename(path), options
                )
                if asset is None:
                    raise RuntimeError("IsaacGym could not load entity source " + path)
                self._audit_asset(asset, entity)
                entity_assets.append(asset)
            self.assets.append(entity_assets)
        physical_bits = {
            e.name: 1 << i
            for i, (e, s) in enumerate(zip(self.layout.entities, self.specs))
            if s["collision_enabled"]
        }
        disabled_mask = sum(physical_bits.values())
        if not disabled_mask:
            disabled_mask = 1
        origins = []
        for env_id in range(self.num_envs):
            env = ctx.gym.create_env(
                ctx.sim,
                gymapi.Vec3(-2, -2, 0),
                gymapi.Vec3(2, 2, 2),
                max(1, int(np.ceil(np.sqrt(self.num_envs)))),
            )
            ctx.env_handles.append(env)
            origin = ctx.gym.get_env_origin(env)
            origins.append([origin.x, origin.y, origin.z])
            records = []
            for entity_id, (entity, spec) in enumerate(zip(self.layout.entities, self.specs)):
                source_id = spec["assignment"][env_id]
                asset = self.assets[entity_id][source_id]
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(*self.roots0[env_id, entity_id, :3])
                pose.r = gymapi.Quat(
                    *self.protocol.wxyz_to_xyzw(self.roots0[env_id, entity_id, 3:7])
                )
                filter_mask = physical_bits.get(entity.name, disabled_mask)
                actor = ctx.gym.create_actor(env, asset, pose, entity.name, env_id, filter_mask)
                observed_asset = ctx.gym.get_actor_asset(env, actor)
                actual_sources = [
                    i
                    for i, candidate in enumerate(self.assets[entity_id])
                    if candidate == observed_asset
                ]
                if actual_sources != [source_id]:
                    raise RuntimeError("native actor asset identity differs for " + entity.name)
                source_id = actual_sources[0]
                native_id = ctx.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)
                self.actor_ids[env_id, entity_id] = native_id
                variant = spec["variants"][source_id]
                native_joint_names = tuple(ctx.gym.get_asset_dof_names(asset))
                props = ctx.gym.get_actor_dof_properties(env, actor)
                self._apply_drives(props, entity, variant, native_joint_names)
                if len(props):
                    ctx.gym.set_actor_dof_properties(env, actor, props)
                    readback = ctx.gym.get_actor_dof_properties(env, actor)
                    for field in (
                        "driveMode",
                        "stiffness",
                        "damping",
                        "effort",
                        "armature",
                        "friction",
                        "hasLimits",
                        "lower",
                        "upper",
                    ):
                        if not np.allclose(readback[field], props[field], rtol=1e-5, atol=1e-7):
                            raise RuntimeError(
                                "native drive readback differs for %s field %s: %s != %s"
                                % (entity.name, field, readback[field], props[field])
                            )
                native_body_names = tuple(ctx.gym.get_asset_rigid_body_names(asset))
                body_props = ctx.gym.get_actor_rigid_body_properties(env, actor)
                native_body_ids = []
                masses = []
                inertia_matrices = []
                for local, public_id in enumerate(entity.body_ids):
                    native_local = native_body_names.index(entity.body_names[local])
                    native_body_id = ctx.gym.get_actor_rigid_body_index(
                        env, actor, native_local, gymapi.DOMAIN_SIM
                    )
                    self.body_ids[env_id, public_id] = native_body_id
                    native_body_ids.append(native_body_id)
                    prop = body_props[native_local]
                    com = np.array([prop.com.x, prop.com.y, prop.com.z])
                    self.body_com[env_id, public_id] = com
                    matrix = np.array(
                        [
                            [
                                getattr(getattr(prop.inertia, axis), coord)
                                for coord in ("x", "y", "z")
                            ]
                            for axis in ("x", "y", "z")
                        ]
                    )
                    masses.append(float(prop.mass))
                    inertia_matrices.append(matrix.tolist())
                    if not np.isclose(
                        prop.mass, variant["body_mass"][local], rtol=2e-4, atol=1e-6
                    ) or not np.allclose(com, variant["body_ipos"][local], rtol=2e-4, atol=1e-6):
                        raise RuntimeError(
                            "native mass/COM differs for "
                            + entity.name
                            + "/"
                            + entity.body_names[local]
                        )
                    # Inertia is audited in the body frame, not by principal values alone.
                    axes = self.protocol.quat_rotate(
                        np.asarray(variant["body_iquat"][local]), np.eye(3)
                    )
                    expected_inertia = axes.T @ np.diag(variant["body_inertia"][local]) @ axes
                    if not np.allclose(matrix, expected_inertia, rtol=5e-4, atol=1e-7):
                        raise RuntimeError(
                            "native inertia differs for "
                            + entity.name
                            + "/"
                            + entity.body_names[local]
                        )
                root_local = entity.body_names.index(entity.root_body)
                self.root_com[env_id, entity_id] = self.body_com[
                    env_id, entity.body_ids[root_local]
                ]
                dof_ids = tuple(
                    ctx.gym.get_actor_dof_index(
                        env, actor, native_joint_names.index(j.name), gymapi.DOMAIN_SIM
                    )
                    for j in entity.joints
                )
                for column, target in zip(entity.actuator_indices, entity.actuator_joint_names):
                    self.control_dofs[env_id, column] = dof_ids[
                        [j.name for j in entity.joints].index(target)
                    ]
                records.append(
                    {
                        "actor": actor,
                        "actor_id": native_id,
                        "source_id": source_id,
                        "source": spec["sources"][source_id],
                        "dof_ids": dof_ids,
                        "body_ids": native_body_ids,
                        "body_mass": masses,
                        "body_inertia": inertia_matrices,
                        "drive_modes": [int(value) for value in props["driveMode"]],
                        "native_joint_names": list(native_joint_names),
                    }
                )
            self.records.append(records)
            ctx.actor_handles.append(records[0]["actor"])
        ctx.gym.prepare_sim(ctx.sim)
        ctx._acquire_tensors()
        self.targets = ctx.torch.zeros_like(ctx._dof_state[:, 0])
        # prepare_sim may advance initialization: restore the authored complete state.
        self._stage_initial()
        self._submit_pending()
        actual_params = ctx.gym.get_sim_params(ctx.sim)
        actual_gravity = [
            float(actual_params.gravity.x),
            float(actual_params.gravity.y),
            float(actual_params.gravity.z),
        ]
        mass_table = np.zeros((self.num_envs, self.layout.nbody))
        inertia_table = np.zeros((self.num_envs, self.layout.nbody, 3, 3))
        for env in range(self.num_envs):
            for index, entity in enumerate(self.layout.entities):
                mass_table[env, entity.body_ids] = self.records[env][index]["body_mass"]
                inertia_table[env, entity.body_ids] = self.records[env][index]["body_inertia"]
        body_names = [
            next(
                (
                    e.name + "/" + name
                    for e in self.layout.entities
                    for name, index in zip(e.body_names, e.body_ids)
                    if index == b
                ),
                "world",
            )
            for b in range(self.layout.nbody)
        ]
        self.metadata = {
            "scene_layout": self.layout.to_dict(),
            "runtime": {
                "python": sys.version,
                "torch": str(torch.__version__),
                "isaacgym_module": getattr(gymapi, "__file__", None),
                "device": ctx.device,
                "device_name": torch.cuda.get_device_name(device_id)
                if ctx.use_gpu_pipeline
                else "cpu",
            },
            "scene_entities_actual": [
                {
                    "name": entity.name,
                    "assignment": [r[i]["source_id"] for r in self.records],
                    "sources": [r[i]["source"] for r in self.records],
                    "actor_ids": self.actor_ids[:, i].tolist(),
                    "body_mass": [r[i]["body_mass"] for r in self.records],
                    "body_inertia": [r[i]["body_inertia"] for r in self.records],
                    "body_ipos": self.body_com[:, entity.body_ids].tolist(),
                    "drive_modes": [r[i]["drive_modes"] for r in self.records],
                    "native_joint_names": [r[i]["native_joint_names"] for r in self.records],
                }
                for i, entity in enumerate(self.layout.entities)
            ],
            "num_dof": ctx.num_dof,
            "num_bodies": self.layout.nbody,
            "dof_names": [e.name + "/" + j.name for e in self.layout.entities for j in e.joints],
            "body_names": [
                next(
                    (
                        e.name + "/" + name
                        for e in self.layout.entities
                        for name, index in zip(e.body_names, e.body_ids)
                        if index == b
                    ),
                    "world",
                )
                for b in range(self.layout.nbody)
            ],
            "gravity": actual_gravity,
            "env_origins": origins,
            "use_gpu_pipeline": ctx.use_gpu_pipeline,
            "graphics_enabled": ctx.graphics_device_id >= 0,
            "state_freshness": {
                "root": "after_reset",
                "joint": "after_reset",
                "articulation_body": "after_step",
                "coordinates": "env_local",
            },
            "configuration_report": {
                "schema_version": 1,
                "effective": {
                    "dt": float(actual_params.dt),
                    "gravity": actual_gravity,
                    "solver": "PhysX solver_type=%d" % actual_params.physx.solver_type,
                    "body_mass": {"names": body_names, "per_env_values": mass_table.tolist()},
                    "body_inertia": {
                        "names": body_names,
                        "per_env_matrices": inertia_table.tolist(),
                    },
                    "collision_filter": {
                        "physical_entity_bits": physical_bits,
                        "disabled_mask": disabled_mask,
                    },
                    "body_linear_damping": 0.0,
                    "body_angular_damping": 0.0,
                },
                "engine_readback": ["dt", "gravity", "solver", "body_mass", "body_inertia"],
            },
        }
        return self.metadata

    def _audit_asset(self, asset: Any, entity: Any) -> None:
        gym, api = self.ctx.gym, self.ctx.gymapi
        bodies = tuple(gym.get_asset_rigid_body_names(asset))
        joints = tuple(gym.get_asset_dof_names(asset))
        if len(set(bodies)) != len(bodies) or set(bodies) != set(entity.body_names):
            raise RuntimeError("native body names differ for " + entity.name)
        if len(set(joints)) != len(joints) or set(joints) != {j.name for j in entity.joints}:
            raise RuntimeError("native joint names differ for " + entity.name)
        for joint in entity.joints:
            expected = api.DOF_ROTATION if joint.kind == "hinge" else api.DOF_TRANSLATION
            if gym.get_asset_dof_type(asset, joints.index(joint.name)) != expected:
                raise RuntimeError(
                    "native joint type differs for " + entity.name + "/" + joint.name
                )

    def _apply_drives(
        self, props: Any, entity: Any, variant: dict[str, Any], native_names: tuple
    ) -> None:
        controlled = set(entity.actuator_joint_names)
        for index, name in enumerate(native_names):
            source = variant["joint_names"].index(name)
            active = name in controlled
            props["driveMode"][index] = int(
                self.ctx.gymapi.DOF_MODE_POS if active else self.ctx.gymapi.DOF_MODE_NONE
            )
            for field in ("stiffness", "damping", "effort", "armature", "friction"):
                value = float(variant["dof_" + field][source])
                if not active and field in ("stiffness", "damping", "effort"):
                    value = 0.0
                props[field][index] = value
            low, high = variant["dof_lower"][source], variant["dof_upper"][source]
            props["hasLimits"][index] = bool(np.isfinite(low) and np.isfinite(high))
            if props["hasLimits"][index]:
                props["lower"][index], props["upper"][index] = low, high

    def _native_root(self, row: np.ndarray, com: np.ndarray) -> np.ndarray:
        result = row.astype(np.float32, copy=True)
        result[3:7] = self.protocol.wxyz_to_xyzw(row[3:7])
        result[7:10] += np.cross(row[10:13], self.protocol.quat_rotate(row[3:7], com))
        return result

    def _public_state(self, native: np.ndarray, com: np.ndarray) -> np.ndarray:
        result = native.copy()
        result[..., 3:7] = self.protocol.xyzw_to_wxyz(native[..., 3:7])
        offset = self.protocol.quat_rotate(result[..., 3:7], com)
        result[..., 7:10] -= np.cross(result[..., 10:13], offset)
        return result

    def _stage_initial(self) -> None:
        for env in range(self.num_envs):
            for index, entity in enumerate(self.layout.entities):
                record = self.records[env][index]
                self.pending_roots[record["actor_id"]] = self._native_root(
                    self.roots0[env, index], self.root_com[env, index]
                )
                for joint, dof in zip(entity.joints, record["dof_ids"]):
                    self.pending_dofs[dof] = np.array(
                        [
                            self.qpos0[env, joint.qpos_indices[0]],
                            self.qvel0[env, joint.qvel_indices[0]],
                        ],
                        dtype=np.float32,
                    )
                if record["dof_ids"]:
                    self.pending_dof_actors.add(record["actor_id"])
        for row, columns in enumerate(self.control_dofs):
            for index, dof in enumerate(columns):
                value = (
                    self.initial_ctrl[row, index]
                    if self.initial_ctrl is not None
                    else self.pending_dofs[int(dof)][0]
                )
                self.targets[dof] = float(value)
        if self.initial_ctrl is None:
            self.initial_ctrl = np.array(
                [[float(self.targets[dof]) for dof in columns] for columns in self.control_dofs],
                dtype=np.float64,
            ).reshape(self.num_envs, self.layout.nu)

    def _submit_pending(self) -> None:
        ctx = self.ctx
        try:
            if self.pending_roots:
                ids = np.array(sorted(self.pending_roots), dtype=np.int32)
                native_ids = ctx.torch.from_numpy(ids).to(ctx.device)
                values = np.array([self.pending_roots[int(i)] for i in ids], dtype=np.float32)
                ctx._root_state[native_ids.long()] = ctx.torch.from_numpy(values).to(ctx.device)
                if not ctx.gym.set_actor_root_state_tensor_indexed(
                    ctx.sim,
                    ctx.gymtorch.unwrap_tensor(ctx._root_state),
                    ctx.gymtorch.unwrap_tensor(native_ids),
                    len(ids),
                ):
                    raise RuntimeError("native root setter failed")
            if self.pending_dofs:
                dof_ids = ctx.torch.tensor(
                    sorted(self.pending_dofs), dtype=ctx.torch.long, device=ctx.device
                )
                values = np.array(
                    [self.pending_dofs[i] for i in sorted(self.pending_dofs)], dtype=np.float32
                )
                ctx._dof_state[dof_ids] = ctx.torch.from_numpy(values).to(ctx.device)
                actor_ids = ctx.torch.tensor(
                    sorted(self.pending_dof_actors), dtype=ctx.torch.int32, device=ctx.device
                )
                if not ctx.gym.set_dof_state_tensor_indexed(
                    ctx.sim,
                    ctx.gymtorch.unwrap_tensor(ctx._dof_state),
                    ctx.gymtorch.unwrap_tensor(actor_ids),
                    len(actor_ids),
                ):
                    raise RuntimeError("native DOF setter failed")
        except Exception:
            self.faulted = True
            raise

    def refresh(self) -> None:
        ctx = self.ctx
        if self.faulted:
            raise RuntimeError("IsaacGym scene is faulted")
        ctx._refresh_tensors()
        native_roots = ctx._root_state.cpu().numpy().copy()
        native_dofs = ctx._dof_state.cpu().numpy().copy()
        for index, values in self.pending_roots.items():
            native_roots[index] = values
        for index, values in self.pending_dofs.items():
            native_dofs[index] = values
        public_roots = self._public_state(native_roots[self.actor_ids], self.root_com)
        np.copyto(ctx.slots["entity_root_state"], public_roots)
        qpos, qvel = ctx.slots["qpos"], ctx.slots["qvel"]
        for entity_index, entity in enumerate(self.layout.entities):
            roots = public_roots[:, entity_index]
            if entity.root_mode == "floating":
                qpos[:, entity.root_qpos_indices] = roots[:, :7]
                velocity = roots[:, 7:13].copy()
                velocity[:, 3:] = self.protocol.quat_rotate_inverse(roots[:, 3:7], velocity[:, 3:])
                qvel[:, entity.root_qvel_indices] = velocity
            for env in range(self.num_envs):
                for joint, dof in zip(entity.joints, self.records[env][entity_index]["dof_ids"]):
                    qpos[env, joint.qpos_indices[0]] = native_dofs[dof, 0]
                    qvel[env, joint.qvel_indices[0]] = native_dofs[dof, 1]
        body = ctx.slots["body_state"]
        body.fill(0)
        body[..., 3] = 1
        contact = ctx.slots["contact_force"]
        contact.fill(0)
        native_body = ctx._body_state.cpu().numpy()
        native_contact = ctx._contact_force.cpu().numpy()
        for env in range(self.num_envs):
            present = self.body_ids[env] >= 0
            body[env, present] = self._public_state(
                native_body[self.body_ids[env, present]], self.body_com[env, present]
            )
            contact[env, present] = native_contact[self.body_ids[env, present]]
        # Root body has an authoritative actor state even when articulation FK is stale.
        for index, entity in enumerate(self.layout.entities):
            root_body = entity.body_ids[entity.body_names.index(entity.root_body)]
            body[:, root_body] = public_roots[:, index]

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        ctx = self.ctx
        if self.faulted:
            raise RuntimeError("IsaacGym scene is faulted")
        count = payload.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= self.num_envs:
            raise ValueError("invalid reset count")
        envs = ctx.slots["reset_env_ids"][:count].copy()
        if np.any(envs < 0) or np.any(envs >= self.num_envs) or len(set(envs.tolist())) != count:
            raise ValueError("reset environment IDs must be unique and in range")
        names = payload.get("entity_names")
        if (
            not isinstance(names, list)
            or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("reset entity_names must be a unique list")
        for name in names:
            self.layout.get_entity(name)
        control_values = None
        if "control_values" in payload:
            control_values = finite_array(
                payload["control_values"], (count, self.layout.nu), "control_values"
            )
            if np.any(np.abs(control_values) > np.finfo(np.float32).max):
                raise ValueError("control_values exceed native float32 range")
            unselected_controls = [
                column
                for entity in self.layout.entities
                if entity.name not in names
                for column in entity.actuator_indices
            ]
            if unselected_controls and not np.array_equal(
                control_values[:, unselected_controls],
                ctx.slots["ctrl"][np.ix_(envs, unselected_controls)],
            ):
                raise ValueError("control_values cannot modify an unselected controlled entity")
        qpos = finite_array(ctx.slots["reset_qpos"][:count], (count, self.layout.nq), "reset_qpos")
        qvel = finite_array(ctx.slots["reset_qvel"][:count], (count, self.layout.nv), "reset_qvel")
        roots = finite_array(
            ctx.slots["reset_entity_root_state"][:count],
            (count, len(self.layout.entities), 13),
            "reset roots",
        )
        pm, vm, rm = (
            ctx.slots[key].copy()
            for key in ("reset_qpos_mask", "reset_qvel_mask", "reset_root_mask")
        )
        if any(np.any((mask != 0) & (mask != 1)) for mask in (pm, vm, rm)):
            raise ValueError("reset masks must contain only zero or one")
        staged_roots: dict[int, np.ndarray] = {}
        staged_dofs: dict[int, np.ndarray] = {}
        staged_actors: set[int] = set()
        staged_controls: list[tuple[int, int, float]] = []
        # Validate every channel before touching native state or pending transactions.
        for index, entity in enumerate(self.layout.entities):
            changed = bool(
                np.any(pm[list(entity.qpos_indices)])
                or np.any(vm[list(entity.qvel_indices)])
                or np.any(rm[index])
            )
            if changed and entity.name not in names:
                raise ValueError("reset masks write an unselected entity")
            if entity.root_mode == "fixed" and np.any(rm[index]):
                raise ValueError("fixed root writes are unsupported")
            if entity.root_mode == "kinematic" and rm[index, 1]:
                raise ValueError("kinematic root velocity writes are unsupported")
            if entity.root_mode == "floating":
                rp, rv = pm[list(entity.root_qpos_indices)], vm[list(entity.root_qvel_indices)]
                if np.any(rp != rm[index, 0]) or np.any(rv != rm[index, 1]):
                    raise ValueError("root masks disagree with generalized column masks")
                if np.any(rm[index]):
                    unit_quaternion(roots[:, index, 3:7], "reset root pose")
                    if not np.allclose(
                        qpos[:, entity.root_qpos_indices], roots[:, index, :7], rtol=1e-5, atol=1e-6
                    ):
                        raise ValueError("reset root pose differs from generalized state")
                    expected = roots[:, index, 7:].copy()
                    expected[:, 3:] = self.protocol.quat_rotate_inverse(
                        roots[:, index, 3:7], expected[:, 3:]
                    )
                    if not np.allclose(
                        qvel[:, entity.root_qvel_indices], expected, rtol=1e-5, atol=1e-6
                    ):
                        raise ValueError("reset root velocity differs from generalized state")
            elif rm[index, 0]:
                unit_quaternion(roots[:, index, 3:7], "reset root pose")
            for row, env in enumerate(envs):
                record = self.records[int(env)][index]
                if np.any(rm[index]):
                    current = ctx.slots["entity_root_state"][env, index]
                    if not rm[index, 0] and not np.allclose(
                        roots[row, index, :7], current[:7], rtol=1e-5, atol=1e-6
                    ):
                        raise ValueError("reset attempted to change an unselected root pose")
                    if not rm[index, 1] and not np.allclose(
                        roots[row, index, 7:], current[7:], rtol=1e-5, atol=1e-6
                    ):
                        raise ValueError("reset attempted to change unselected root velocity")
                    staged_roots[record["actor_id"]] = self._native_root(
                        roots[row, index], self.root_com[env, index]
                    )
                for joint, dof in zip(entity.joints, record["dof_ids"]):
                    pi, vi = joint.qpos_indices[0], joint.qvel_indices[0]
                    if pm[pi] or vm[vi]:
                        staged_dofs[dof] = np.array(
                            [
                                qpos[row, pi] if pm[pi] else ctx.slots["qpos"][env, pi],
                                qvel[row, vi] if vm[vi] else ctx.slots["qvel"][env, vi],
                            ],
                            dtype=np.float32,
                        )
                        staged_actors.add(record["actor_id"])
                        if pm[pi] and joint.name in entity.actuator_joint_names:
                            column = entity.actuator_indices[
                                entity.actuator_joint_names.index(joint.name)
                            ]
                            staged_controls.append((int(env), column, float(qpos[row, pi])))
        self.pending_roots.update(staged_roots)
        self.pending_dofs.update(staged_dofs)
        self.pending_dof_actors.update(staged_actors)
        # Re-submit the union since the prior physics step. Gym drops earlier
        # disjoint indexed setters when a later call replaces their pending IDs.
        self._submit_pending()
        try:
            if control_values is not None:
                for row, env in enumerate(envs):
                    for column, dof in enumerate(self.control_dofs[env]):
                        self.targets[dof] = float(control_values[row, column])
                if self.layout.nu and count:
                    if not ctx.gym.set_dof_position_target_tensor(
                        ctx.sim, ctx.gymtorch.unwrap_tensor(self.targets)
                    ):
                        raise RuntimeError("native reset control setter failed")
                ctx.slots["ctrl"][envs] = control_values
            else:
                for env, column, value in staged_controls:
                    ctx.slots["ctrl"][env, column] = value
            self.refresh()
        except Exception:
            self.faulted = True
            raise
        return {"timing": {}}

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.faulted:
            raise RuntimeError("IsaacGym scene is faulted")
        nsteps = payload.get("nsteps")
        if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps <= 0:
            raise ValueError("nsteps must be a positive integer")
        ctx = self.ctx
        ctrl = finite_array(ctx.slots["ctrl"], (self.num_envs, self.layout.nu), "ctrl")
        try:
            if self.layout.nu:
                ids = ctx.torch.from_numpy(self.control_dofs.reshape(-1)).to(ctx.device)
                self.targets[ids] = ctx.torch.as_tensor(
                    ctrl.reshape(-1), dtype=self.targets.dtype, device=ctx.device
                )
            if len(self.targets):
                ctx.gym.set_dof_position_target_tensor(
                    ctx.sim, ctx.gymtorch.unwrap_tensor(self.targets)
                )
            self._submit_pending()
            for index in range(nsteps):
                ctx.gym.simulate(ctx.sim)
                ctx.gym.fetch_results(ctx.sim, True)
                if index == 0:
                    self.pending_roots.clear()
                    self.pending_dofs.clear()
                    self.pending_dof_actors.clear()
            self.refresh()
        except Exception:
            self.faulted = True
            raise
        return {"timing": {}}
