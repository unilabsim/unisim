"""Native multi-actor execution for the versioned scene IPC profile.

Loaded by file path in the isolated Python 3.8 worker. Only stdlib and NumPy
are imported here; IsaacGym must be imported before Torch during initialization.
"""

from __future__ import annotations

import os
import sys
import time
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

    projection: Any

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
        self._wrench_torch: Any = None
        raw_initial_ctrl = payload.get("initial_ctrl")
        self.initial_ctrl = (
            None
            if raw_initial_ctrl is None
            else finite_array(raw_initial_ctrl, (count, self.layout.nu), "initial_ctrl")
        )
        self.records: list[list[dict[str, Any]]] = []
        self.assets: list[dict[int, Any]] = []
        self.actor_ids = np.empty((count, len(self.layout.entities)), dtype=np.int64)
        self.body_ids = np.full((count, self.layout.nbody), -1, dtype=np.int64)
        self.body_com = np.zeros((count, self.layout.nbody, 3))
        self.root_com = np.zeros((count, len(self.layout.entities), 3))
        self.control_dofs = np.empty((count, self.layout.nu), dtype=np.int64)
        self.pending_roots: dict[int, np.ndarray] = {}
        self.pending_dofs: dict[int, np.ndarray] = {}
        self.pending_dof_actors: set[int] = set()
        # Legacy-only post-reset body-state overlay (issue #141): PhysX cannot
        # refresh link poses without stepping, so exact FK rows are published
        # until the first simulate.  None on the mapped scene path, which keeps
        # the host-side fail-closed staleness contract instead.
        self.pending_body_fk: dict[int, np.ndarray] | None = None
        self._fk: Any = None
        self.faulted = False
        self.metadata: dict[str, Any] = {}
        self.publish_actor_roots_as_body = True

    @classmethod
    def adopt_initialized_context(
        cls, context: Any, metadata: dict[str, Any], payload: dict[str, Any], bridge: Any
    ) -> SceneWorker:
        """Adopt a cold raw-asset loader's objects into the single scene runtime.

        The bridge's synthetic root and D control columns are compatibility
        projections, not assertions about the source's physical joint tree.
        Native indices below come from Gym, never actor-order arithmetic.
        """
        self = cls.__new__(cls)
        self.ctx, self.protocol, self.payload = context, context.protocol, payload
        ctx = context
        self.num_envs = ctx.num_envs
        self.layout = bridge.LegacyExecutionLayout(
            tuple(metadata["dof_names"]), tuple(metadata["body_names"])
        )
        self.specs = []
        self.assets = []
        self.records = []
        self.actor_ids = np.empty((self.num_envs, 1), dtype=np.int64)
        self.body_ids = np.empty((self.num_envs, self.layout.nbody), dtype=np.int64)
        self.body_com = np.zeros((self.num_envs, self.layout.nbody, 3))
        self.root_com = np.zeros((self.num_envs, 1, 3))
        self.control_dofs = np.empty((self.num_envs, self.layout.nu), dtype=np.int64)
        self.pending_roots = {}
        self.pending_dofs = {}
        self.pending_dof_actors = set()
        self.faulted = False
        self.metadata = metadata
        self._wrench_torch = None
        self.publish_actor_roots_as_body = False
        self.gravity = np.asarray(metadata["gravity"], dtype=np.float64)
        self._fk = self._bind_kinematics(payload)
        self.pending_body_fk = {}
        entity = self.layout.entities[0]
        expected_variant = payload.get("variant_assignment", [0] * self.num_envs)
        sources = payload.get("variant_model_files", [payload["model_file"]])
        for env_index, (env, actor) in enumerate(zip(ctx.env_handles, ctx.actor_handles)):
            asset = ctx.gym.get_actor_asset(env, actor)
            native_joints = tuple(ctx.gym.get_asset_dof_names(asset))
            native_bodies = tuple(ctx.gym.get_asset_rigid_body_names(asset))
            if native_joints != tuple(metadata["dof_names"]) or native_bodies != tuple(
                metadata["body_names"]
            ):
                raise RuntimeError("adopted Gym asset differs from initialized public metadata")
            actor_id = ctx.gym.get_actor_index(env, actor, ctx.gymapi.DOMAIN_SIM)
            self.actor_ids[env_index, 0] = actor_id
            dofs = tuple(
                ctx.gym.get_actor_dof_index(
                    env, actor, native_joints.index(j.name), ctx.gymapi.DOMAIN_SIM
                )
                for j in entity.joints
            )
            bodies = tuple(
                ctx.gym.get_actor_rigid_body_index(
                    env, actor, native_bodies.index(name), ctx.gymapi.DOMAIN_SIM
                )
                for name in entity.body_names
            )
            self.body_ids[env_index] = bodies
            self.control_dofs[env_index] = dofs
            properties = ctx.gym.get_actor_rigid_body_properties(env, actor)
            for index, name in enumerate(entity.body_names):
                com = properties[native_bodies.index(name)].com
                self.body_com[env_index, index] = [com.x, com.y, com.z]
            root = entity.body_names.index(entity.root_body)
            self.root_com[env_index, 0] = self.body_com[env_index, root]
            self.records.append(
                [
                    {
                        "actor": actor,
                        "actor_id": actor_id,
                        "dof_ids": dofs,
                        "body_ids": bodies,
                        "source_id": expected_variant[env_index],
                        "source": sources[expected_variant[env_index]],
                        "native_joint_names": list(native_joints),
                    }
                ]
            )
        self.targets = ctx.torch.zeros_like(ctx._dof_state[:, 0])
        # The cold loader already applied its keyframe. Preserve those exact
        # native writes until the first step consumes the pending actor IDs.
        roots = ctx._root_state.cpu().numpy()
        dof_state = ctx._dof_state.cpu().numpy()
        for records in self.records:
            record = records[0]
            self.pending_roots[record["actor_id"]] = roots[record["actor_id"]].copy()
            for dof in record["dof_ids"]:
                self.pending_dofs[dof] = dof_state[dof].copy()
            if record["dof_ids"]:
                self.pending_dof_actors.add(record["actor_id"])
        # The legacy slot starts at zero, and old set_state leaves its target
        # untouched. Do not infer new hold-position targets from a keyframe.
        self.initial_ctrl = None
        self.projection = bridge.LegacySlotProjection(
            self.protocol,
            self.num_envs,
            self.layout,
            root_com=self.root_com[:, 0],
            body_com=self.body_com,
        )
        self._bind_refresh_indices()
        return self

    def _bind_kinematics(self, payload: dict[str, Any]) -> Any:
        """Resolve the per-variant FK tables against the adopted public layout.

        PhysX refreshes rigid-body link poses only during ``simulate``; the FK
        tables let the legacy path publish exact post-reset body state (#141).
        """
        module = self.protocol.load_kinematics()
        variant_tables = payload.get("variant_mjcf_kinematics")
        if variant_tables is None:
            tables = payload.get("mjcf_kinematics")
            if tables is None:
                raise RuntimeError(
                    "IsaacGym INIT payload is missing mjcf_kinematics; the host must send "
                    "the MJCF kinematic tree so post-reset body state can be published"
                )
            variant_tables = [tables]
        entity = self.layout.entities[0]
        expected_bodies = list(entity.body_names)
        expected_joints = [joint.name for joint in entity.joints]
        for tables in variant_tables:
            if (
                not isinstance(tables, dict)
                or [str(name) for name in tables.get("body_names") or ()] != expected_bodies
                or [str(name) for name in tables.get("joint_names") or ()] != expected_joints
            ):
                raise RuntimeError(
                    "IsaacGym MJCF kinematics payload does not match the adopted public "
                    "layout; the host validates that the importer preserves body/joint "
                    "name order"
                )
            free_root = tables.get("free_root", -1)
            if isinstance(free_root, bool) or not isinstance(free_root, int) or free_root < -1:
                raise RuntimeError("IsaacGym MJCF kinematics free_root must be an index or -1")
            if free_root > 0:
                raise RuntimeError(
                    "IsaacGym MJCF kinematics requires the legacy freejoint root as the "
                    "first body"
                )
            if free_root == -1:
                # Fixed-base legacy assets have no floating root to FK from;
                # keep publishing native rows for them (overlay stays off).
                if len(variant_tables) > 1:
                    raise RuntimeError(
                        "IsaacGym fixed variants without a freejoint root cannot share "
                        "kinematics tables"
                    )
                return None
        assignment = np.asarray(
            payload.get("variant_assignment", [0] * self.num_envs), dtype=np.int64
        )
        if assignment.shape != (self.num_envs,) or np.any(
            (assignment < 0) | (assignment >= len(variant_tables))
        ):
            raise RuntimeError("IsaacGym kinematics variant assignment is out of range")
        prepared = tuple(module.prepare_kinematics(tables) for tables in variant_tables)
        return module, prepared, assignment

    def stage_fk_overlay_rows(self, env_ids: Any, qpos_rows: Any, qvel_rows: Any) -> None:
        """Overlay exact FK body state for freshly written envs until the first step."""
        if self._fk is None or self.pending_body_fk is None:
            return
        module, variant_tables, assignment = self._fk
        envs = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        qpos = np.asarray(qpos_rows, dtype=np.float64).reshape(len(envs), -1)
        qvel = np.asarray(qvel_rows, dtype=np.float64).reshape(len(envs), -1)
        row_variants = assignment[envs]
        for variant_index in np.unique(row_variants):
            rows = np.flatnonzero(row_variants == variant_index)
            states = module.forward_prepared_kinematics(
                variant_tables[variant_index], qpos[rows], qvel[rows]
            )
            for row, env in enumerate(envs[rows]):
                self.pending_body_fk[int(env)] = states[row]

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
            if not isinstance(spec.get("self_collision"), bool):
                raise TypeError("self_collision must be bool")
            if spec["self_collision"] and (
                entity.kind != "articulation" or not spec["collision_enabled"]
            ):
                raise ValueError(
                    "self_collision requires a collision-enabled articulation entity"
                )
            # The host resolves gravity_disabled=None to this backend's implicit
            # default (gravity enabled on every entity asset) before INIT; an
            # unset value here means the request never passed host resolution.
            if not isinstance(spec.get("gravity_disabled"), bool):
                raise TypeError("gravity_disabled must be bool")
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
                if spec["self_collision"] and xml.findall("contact/exclude"):
                    # PhysX exposes one 32-bit filter word per shape; honoring
                    # an authored exclusion while every other intra-actor body
                    # pair collides would require per-pair bit coloring. The
                    # mapped profile rejects instead of silently upgrading the
                    # exclusion to full self-collision.
                    raise NotImplementedError(
                        "IsaacGym self_collision cannot retain authored "
                        "<contact><exclude> pairs in " + source
                    )
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
                colors = finite_array(variant.get("body_visual_rgb"), (nb, 3), "body_visual_rgb")
                if np.any((colors < 0.0) | (colors > 1.0)):
                    raise ValueError("body_visual_rgb components must lie in [0, 1]")
                unit_quaternion(np.asarray(variant["body_iquat"]), "body_iquat")
                if entity.geoms:
                    public = [(geom.name, geom.body_name) for geom in entity.geoms]
                    variant_names = variant.get("geom_names")
                    variant_bodies = variant.get("geom_body_names")
                    if (
                        not isinstance(variant_names, list)
                        or not isinstance(variant_bodies, list)
                        or len(variant_names) != len(variant_bodies)
                    ):
                        raise ValueError("variant geometry identity differs from public layout")
                    # TEMP(unisimtoolreal local workaround): uniform_public_layout
                    # permits optional mesh-geom slots to be absent from a variant
                    # (e.g. eraser heads). Accept an order-preserving subset of the
                    # public layout until the worker grows first-class optional-slot
                    # handling.
                    cursor = 0
                    for pair in zip(variant_names, variant_bodies):
                        while cursor < len(public) and public[cursor] != pair:
                            cursor += 1
                        if cursor == len(public):
                            raise ValueError(
                                "variant geometry identity differs from public layout"
                            )
                        cursor += 1
                    friction = finite_array(
                        variant.get("geom_friction"), (len(variant_names), 3), "geom_friction"
                    )
                    if np.any(friction < 0):
                        raise ValueError("negative geom_friction")

    @staticmethod
    def _materialized_source_ids(spec: dict[str, Any]) -> tuple[int, ...]:
        """Return catalog rows that require a resident native asset."""
        return tuple(sorted({int(value) for value in spec["assignment"]}))

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
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        ctx.gym.add_ground(ctx.sim, plane_params)
        self._shaped_bodies: dict[str, set[str]] = {}
        for entity, spec in zip(self.layout.entities, self.specs):
            entity_assets: dict[int, Any] = {}
            for source_id in self._materialized_source_ids(spec):
                path = spec["sources"][source_id]
                options = gymapi.AssetOptions()
                options.fix_base_link = entity.root_mode != "floating"
                # Per-entity resolved request; IsaacGym offers no per-actor
                # gravity readback, so honoring is enforced at asset authoring.
                options.disable_gravity = bool(spec["gravity_disabled"])
                options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
                options.linear_damping = options.angular_damping = 0.0
                asset = ctx.gym.load_asset(
                    ctx.sim, os.path.dirname(path), os.path.basename(path), options
                )
                if asset is None:
                    raise RuntimeError("IsaacGym could not load entity source " + path)
                self._audit_asset(asset, entity)
                shaped = self._shaped_bodies.setdefault(entity.name, set())
                bodies = tuple(ctx.gym.get_asset_rigid_body_names(asset))
                ranges = ctx.gym.get_asset_rigid_body_shape_indices(asset)
                if len(ranges) != len(bodies):
                    raise RuntimeError("native asset shape ranges do not match bodies")
                shaped.update(
                    name for name, span in zip(bodies, ranges) if int(span.count) > 0
                )
                entity_assets[source_id] = asset
            self.assets.append(entity_assets)
        physical_bits, self_collision_bits, disabled_mask = self._collision_filter_plan(
            self.layout.entities, self.specs, self._shaped_bodies
        )
        origins = []
        env_spacing = self._env_spacing()
        for env_id in range(self.num_envs):
            half_spacing = env_spacing * 0.5
            env = ctx.gym.create_env(
                ctx.sim,
                gymapi.Vec3(-half_spacing, -half_spacing, 0.0),
                gymapi.Vec3(half_spacing, half_spacing, 2.0),
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
                filter_mask = (
                    0 if spec["self_collision"] else physical_bits.get(entity.name, disabled_mask)
                )
                actor = ctx.gym.create_actor(env, asset, pose, entity.name, env_id, filter_mask)
                observed_asset = ctx.gym.get_actor_asset(env, actor)
                actual_sources = [
                    i
                    for i, candidate in self.assets[entity_id].items()
                    if candidate == observed_asset
                ]
                if actual_sources != [source_id]:
                    raise RuntimeError("native actor asset identity differs for " + entity.name)
                source_id = actual_sources[0]
                if spec["self_collision"]:
                    self._author_self_collision_filters(
                        env, actor, entity, self_collision_bits[entity.name]
                    )
                native_id = ctx.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)
                self.actor_ids[env_id, entity_id] = native_id
                variant = spec["variants"][source_id]
                native_joint_names = tuple(ctx.gym.get_asset_dof_names(asset))
                for local, color in enumerate(variant["body_visual_rgb"]):
                    ctx.gym.set_rigid_body_color(
                        env,
                        actor,
                        local,
                        gymapi.MESH_VISUAL_AND_COLLISION,
                        gymapi.Vec3(*color),
                    )
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
                    # Preview 4's MJCF importer drops the inertial-frame
                    # quaternion. Restore the complete public inertia contract.
                    prop.mass = float(variant["body_mass"][local])
                    prop.com = gymapi.Vec3(*variant["body_ipos"][local])
                    axes = self.protocol.quat_rotate(
                        np.asarray(variant["body_iquat"][local]), np.eye(3)
                    )
                    expected_inertia = axes.T @ np.diag(variant["body_inertia"][local]) @ axes
                    prop.inertia.x = gymapi.Vec3(*expected_inertia[:, 0])
                    prop.inertia.y = gymapi.Vec3(*expected_inertia[:, 1])
                    prop.inertia.z = gymapi.Vec3(*expected_inertia[:, 2])
                ctx.gym.set_actor_rigid_body_properties(env, actor, body_props)
                body_props = ctx.gym.get_actor_rigid_body_properties(env, actor)
                for local, public_id in enumerate(entity.body_ids):
                    native_local = native_body_names.index(entity.body_names[local])
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
                            + f": expected={expected_inertia!r}, native={matrix!r}"
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
                        "body_sphere_radii": list(variant["body_sphere_radii"]),
                        "body_visual_rgb": list(variant["body_visual_rgb"]),
                        "drive_modes": [int(value) for value in props["driveMode"]],
                        "native_joint_names": list(native_joint_names),
                    }
                )
            self.records.append(records)
            ctx.actor_handles.append(records[0]["actor"])
        ctx.gym.prepare_sim(ctx.sim)
        ctx._acquire_tensors()
        self._bind_refresh_indices()
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
                    "body_sphere_radii": [
                        r[i]["body_sphere_radii"] for r in self.records
                    ],
                    "body_visual_rgb": [r[i]["body_visual_rgb"] for r in self.records],
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
            "env_spacing": env_spacing,
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
                        "self_collision": {
                            entity.name: bool(spec["self_collision"])
                            for entity, spec in zip(self.layout.entities, self.specs)
                        },
                        "self_collision_body_bits": {
                            name: dict(bits) for name, bits in self_collision_bits.items()
                        },
                    },
                    "entity_gravity_disabled": {
                        entity.name: bool(spec["gravity_disabled"])
                        for entity, spec in zip(self.layout.entities, self.specs)
                    },
                    "body_linear_damping": 0.0,
                    "body_angular_damping": 0.0,
                    "env_spacing": env_spacing,
                },
                "engine_readback": [
                    "dt",
                    "gravity",
                    "solver",
                    "body_mass",
                    "body_inertia",
                    "env_spacing",
                ],
            },
        }
        return self.metadata

    @staticmethod
    def _collision_filter_plan(
        entities: Any, specs: list[dict[str, Any]], shaped_bodies: dict[str, set[str]]
    ) -> tuple[dict[str, int], dict[str, dict[str, int]], int]:
        """Assign PhysX filter bits for the shared-bit-suppresses filter shader.

        Two shapes collide unless their filter words share a bit.  Colliding
        entities keep one entity bit each (bit-identical to the pre-self-
        collision scheme); a self-collision entity instead gets a zero actor
        filter plus one bit per collision-shaped body, so its distinct bodies
        share no bit and collide while every other physical entity still
        shares nothing with it.  Collision-disabled actors carry the union of
        every allocated bit, which keeps them excluded from self-collision
        bodies too.  Bits are signed-int32-safe positions below 30; entity
        indices reserve the low bits exactly as before.
        """
        physical_bits = {
            entity.name: 1 << index
            for index, (entity, spec) in enumerate(zip(entities, specs))
            if spec["collision_enabled"] and not spec["self_collision"]
        }
        body_bits: dict[str, dict[str, int]] = {}
        next_bit = len(entities)
        for entity, spec in zip(entities, specs):
            if not spec["self_collision"]:
                continue
            bits: dict[str, int] = {}
            for body in entity.body_names:
                if body not in shaped_bodies.get(entity.name, ()):
                    continue
                if next_bit >= 30:
                    raise NotImplementedError(
                        "IsaacGym collision filter budget exhausted: entity "
                        + entity.name
                        + " needs one filter bit per collision-shaped body for "
                        "self_collision, but the mapped profile shares 30 bits "
                        "with the per-entity filters"
                    )
                bits[body] = 1 << next_bit
                next_bit += 1
            body_bits[entity.name] = bits
        disabled_mask = sum(physical_bits.values()) + sum(
            bit for bits in body_bits.values() for bit in bits.values()
        )
        if not disabled_mask:
            disabled_mask = 1
        return physical_bits, body_bits, disabled_mask

    def _author_self_collision_filters(
        self, env: Any, actor: Any, entity: Any, body_bits: dict[str, int]
    ) -> None:
        """Write per-shape body filter bits and verify the native readback."""
        gym = self.ctx.gym
        native_bodies = tuple(gym.get_asset_rigid_body_names(gym.get_actor_asset(env, actor)))
        ranges = gym.get_actor_rigid_body_shape_indices(env, actor)
        props = gym.get_actor_rigid_shape_properties(env, actor)
        assigned = set()
        for local, name in enumerate(native_bodies):
            bit = body_bits.get(name)
            if bit is None:
                continue
            start, count = int(ranges[local].start), int(ranges[local].count)
            for shape in range(start, start + count):
                props[shape].filter = bit
                assigned.add(shape)
        if assigned != set(range(len(props))):
            raise RuntimeError(
                "native shape filter coverage differs for " + entity.name
            )
        gym.set_actor_rigid_shape_properties(env, actor, props)
        readback = gym.get_actor_rigid_shape_properties(env, actor)
        for index, prop in enumerate(readback):
            if int(prop.filter) != int(props[index].filter):
                raise RuntimeError(
                    "native shape filter readback differs for " + entity.name
                )

    def _env_spacing(self) -> float:
        value = self.payload.get("env_spacing", 4.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError("env_spacing must be positive and finite")
        return float(value)

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

    _RESET_RANDOMIZATION_TERMS = frozenset(
        {
            "kp",
            "kd",
            "body_mass",
            "body_inertia",
            "body_ipos",
            "dof_armature",
            "dof_frictionloss",
            "geom_friction",
        }
    )

    def _validated_reset_randomization(
        self, payload: dict[str, Any], count: int
    ) -> dict[str, np.ndarray] | None:
        """Validate the wire property tables before any native mutation."""
        if "randomization" not in payload:
            return None
        if self.pending_body_fk is not None:
            raise ValueError("reset randomization requires a mapped scene")
        raw = payload["randomization"]
        if not isinstance(raw, dict) or not set(raw) <= self._RESET_RANDOMIZATION_TERMS:
            raise ValueError("randomization must contain only supported property terms")

        def table(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
            if not isinstance(value, (list, np.ndarray)):
                raise ValueError("randomization property tables must be lists or arrays")
            try:
                array = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("randomization property tables must be numeric") from exc
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(
                    "randomization %s must be finite with shape %s" % (name, shape)
                )
            return array.copy()

        layout = self.layout
        result: dict[str, np.ndarray] = {}
        for name in ("kp", "kd"):
            if name in raw:
                values = table(raw[name], name, (count, layout.nu))
                if np.any(values < 0.0):
                    raise ValueError("randomization %s must be nonnegative" % name)
                result[name] = values
        if "body_mass" in raw:
            values = table(raw["body_mass"], "body_mass", (count, layout.nbody))
            if np.any(values <= 0.0):
                raise ValueError("randomization body_mass must be positive")
            result["body_mass"] = values
        if "body_ipos" in raw:
            result["body_ipos"] = table(
                raw["body_ipos"], "body_ipos", (count, layout.nbody, 3)
            )
        if "body_inertia" in raw:
            values = table(raw["body_inertia"], "body_inertia", (count, layout.nbody, 3))
            if np.any(values <= 0.0):
                raise ValueError("randomization body_inertia must be positive")
            result["body_inertia"] = values
        root_columns = sorted(
            {column for entity in layout.entities for column in entity.root_qvel_indices}
        )
        for name in ("dof_armature", "dof_frictionloss"):
            if name in raw:
                values = table(raw[name], name, (count, layout.nv))
                if np.any(values < 0.0):
                    raise ValueError("randomization %s must be nonnegative" % name)
                if root_columns and np.any(values[:, root_columns] != 0.0):
                    raise ValueError(
                        "randomization %s must be zero on floating-root columns" % name
                    )
                result[name] = values
        if "geom_friction" in raw:
            values = table(raw["geom_friction"], "geom_friction", (count, layout.ngeom, 3))
            if (
                np.any(values < 0.0)
                or np.any(values[..., 0] != values[..., 1])
                or np.any(values[..., 2] != 0.0)
            ):
                raise ValueError(
                    "randomization geom_friction must be nonnegative with equal "
                    "static/dynamic and zero torsion"
                )
            result["geom_friction"] = values
        return result or None

    @staticmethod
    def _shape_index_map(
        entity: Any, native_bodies: tuple, ranges: Any, shape_count: int
    ) -> list[int]:
        """Map public entity geoms onto native actor shape indices, fail closed."""
        if len(ranges) != len(native_bodies):
            raise RuntimeError("native shape ranges do not match bodies for " + entity.name)
        mapping: list[int] = []
        used: dict[str, int] = {}
        for geom in entity.geoms:
            native_local = native_bodies.index(geom.body_name)
            start, span = int(ranges[native_local].start), int(ranges[native_local].count)
            slot = used.get(geom.body_name, 0)
            if slot >= span:
                raise RuntimeError(
                    "native shape range does not cover public geometry for " + entity.name
                )
            mapping.append(start + slot)
            used[geom.body_name] = slot + 1
        if len(mapping) != shape_count:
            raise RuntimeError(
                "native shape count differs from public geometry for " + entity.name
            )
        return mapping

    def _native_body_names(self, env_handle: Any, actor: Any) -> tuple:
        gym = self.ctx.gym
        return tuple(
            gym.get_asset_rigid_body_names(gym.get_actor_asset(env_handle, actor))
        )

    def _apply_reset_randomization(
        self, envs: np.ndarray, randomization: dict[str, np.ndarray]
    ) -> None:
        """Write selected public property rows through the per-actor Gym API."""
        ctx = self.ctx
        gym, api = ctx.gym, ctx.gymapi
        kp = randomization.get("kp")
        kd = randomization.get("kd")
        armature = randomization.get("dof_armature")
        frictionloss = randomization.get("dof_frictionloss")
        mass = randomization.get("body_mass")
        ipos = randomization.get("body_ipos")
        inertia = randomization.get("body_inertia")
        geom_friction = randomization.get("geom_friction")
        for row, env in enumerate(envs.tolist()):
            env_handle = ctx.env_handles[env]
            geom_offset = 0
            for index, entity in enumerate(self.layout.entities):
                record = self.records[env][index]
                actor = record["actor"]
                native_joints = tuple(record["native_joint_names"])
                if (
                    kp is not None
                    or kd is not None
                    or armature is not None
                    or frictionloss is not None
                ):
                    props = gym.get_actor_dof_properties(env_handle, actor)
                    if kp is not None or kd is not None:
                        for column, joint_name in zip(
                            entity.actuator_indices, entity.actuator_joint_names
                        ):
                            native = native_joints.index(joint_name)
                            if kp is not None:
                                props["stiffness"][native] = float(kp[row, column])
                            if kd is not None:
                                props["damping"][native] = float(kd[row, column])
                    if armature is not None or frictionloss is not None:
                        for joint in entity.joints:
                            native = native_joints.index(joint.name)
                            column = joint.qvel_indices[0]
                            if armature is not None:
                                props["armature"][native] = float(armature[row, column])
                            if frictionloss is not None:
                                props["friction"][native] = float(frictionloss[row, column])
                    if len(props):
                        gym.set_actor_dof_properties(env_handle, actor, props)
                native_bodies: tuple | None = None
                if mass is not None or ipos is not None or inertia is not None:
                    native_bodies = self._native_body_names(env_handle, actor)
                    variant = self.specs[index]["variants"][record["source_id"]]
                    body_props = gym.get_actor_rigid_body_properties(env_handle, actor)
                    for local, public_id in enumerate(entity.body_ids):
                        prop = body_props[native_bodies.index(entity.body_names[local])]
                        if mass is not None:
                            prop.mass = float(mass[row, public_id])
                        if ipos is not None:
                            offset = np.asarray(ipos[row, public_id], dtype=np.float64)
                            prop.com = api.Vec3(
                                float(offset[0]), float(offset[1]), float(offset[2])
                            )
                        if inertia is not None:
                            axes = self.protocol.quat_rotate(
                                np.asarray(variant["body_iquat"][local], dtype=np.float64),
                                np.eye(3),
                            )
                            expected = (
                                axes.T
                                @ np.diag(np.asarray(inertia[row, public_id], dtype=np.float64))
                                @ axes
                            )
                            prop.inertia.x = api.Vec3(*[float(v) for v in expected[:, 0]])
                            prop.inertia.y = api.Vec3(*[float(v) for v in expected[:, 1]])
                            prop.inertia.z = api.Vec3(*[float(v) for v in expected[:, 2]])
                    gym.set_actor_rigid_body_properties(env_handle, actor, body_props)
                geom_count = len(entity.geoms)
                if geom_friction is not None and geom_count:
                    if native_bodies is None:
                        native_bodies = self._native_body_names(env_handle, actor)
                    shape_props = gym.get_actor_rigid_shape_properties(env_handle, actor)
                    mapping = self._shape_index_map(
                        entity,
                        native_bodies,
                        gym.get_actor_rigid_body_shape_indices(env_handle, actor),
                        len(shape_props),
                    )
                    for geom_local, shape_index in enumerate(mapping):
                        shape_props[shape_index].friction = float(
                            geom_friction[row, geom_offset + geom_local, 0]
                        )
                    gym.set_actor_rigid_shape_properties(env_handle, actor, shape_props)
                geom_offset += geom_count

    def _readback_reset_property_records(self) -> list[dict[str, Any]]:
        """Snapshot every entity's native property tables in public order."""
        gym = self.ctx.gym
        records: list[dict[str, Any]] = []
        for index, entity in enumerate(self.layout.entities):
            nb, nj, ng = len(entity.body_ids), len(entity.joints), len(entity.geoms)
            masses = np.zeros((self.num_envs, nb))
            coms = np.zeros((self.num_envs, nb, 3))
            inertias = np.zeros((self.num_envs, nb, 3, 3))
            stiffness = np.zeros((self.num_envs, nj))
            damping = np.zeros((self.num_envs, nj))
            armatures = np.zeros((self.num_envs, nj))
            frictions = np.zeros((self.num_envs, nj))
            geom_friction = np.zeros((self.num_envs, ng, 3))
            for env in range(self.num_envs):
                env_handle = self.ctx.env_handles[env]
                record = self.records[env][index]
                actor = record["actor"]
                native_bodies = self._native_body_names(env_handle, actor)
                body_props = gym.get_actor_rigid_body_properties(env_handle, actor)
                for local in range(nb):
                    prop = body_props[native_bodies.index(entity.body_names[local])]
                    masses[env, local] = prop.mass
                    coms[env, local] = [prop.com.x, prop.com.y, prop.com.z]
                    inertias[env, local] = [
                        [
                            getattr(getattr(prop.inertia, axis), coord)
                            for coord in ("x", "y", "z")
                        ]
                        for axis in ("x", "y", "z")
                    ]
                if nj:
                    native_joints = tuple(record["native_joint_names"])
                    props = gym.get_actor_dof_properties(env_handle, actor)
                    for public, joint in enumerate(entity.joints):
                        native = native_joints.index(joint.name)
                        stiffness[env, public] = props["stiffness"][native]
                        damping[env, public] = props["damping"][native]
                        armatures[env, public] = props["armature"][native]
                        frictions[env, public] = props["friction"][native]
                if ng:
                    shape_props = gym.get_actor_rigid_shape_properties(env_handle, actor)
                    mapping = self._shape_index_map(
                        entity,
                        native_bodies,
                        gym.get_actor_rigid_body_shape_indices(env_handle, actor),
                        len(shape_props),
                    )
                    for geom_local, shape_index in enumerate(mapping):
                        mu = float(shape_props[shape_index].friction)
                        geom_friction[env, geom_local] = [mu, mu, 0.0]
            tables = (
                masses,
                coms,
                inertias,
                stiffness,
                damping,
                armatures,
                frictions,
                geom_friction,
            )
            if not all(np.isfinite(table).all() for table in tables):
                raise RuntimeError(
                    "entity %s native property readback is invalid" % entity.name
                )
            records.append(
                {
                    "name": entity.name,
                    "body_mass": masses.tolist(),
                    "body_ipos": coms.tolist(),
                    "body_inertia": inertias.tolist(),
                    "dof_stiffness": stiffness.tolist(),
                    "dof_damping": damping.tolist(),
                    "dof_armature": armatures.tolist(),
                    "dof_friction": frictions.tolist(),
                    "geom_friction": geom_friction.tolist(),
                }
            )
        return records

    def _substitute_gpu_com_readback(
        self,
        records: list[dict[str, Any]],
        envs: np.ndarray,
        requested: np.ndarray,
    ) -> None:
        """Substitute the requested COM offsets on selected rows in place.

        PhysX honors rigid-body COM writes physically on the GPU pipeline,
        but Preview 4's property readback keeps the pre-write COM there, so
        the native table cannot confirm the selected rows.  Unselected rows
        still come from the native readback, and every other field keeps its
        full audit.
        """
        for index, entity in enumerate(self.layout.entities):
            coms = np.asarray(records[index]["body_ipos"], dtype=np.float64)
            coms[envs] = requested[:, list(entity.body_ids)]
            records[index]["body_ipos"] = coms.tolist()

    def _verify_reset_property_readback(
        self,
        envs: np.ndarray,
        randomization: dict[str, np.ndarray],
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
    ) -> None:
        """Audit the mutation on selected rows and no leakage onto other rows."""
        selected = np.zeros(self.num_envs, dtype=bool)
        selected[envs] = True
        geom_offset = 0
        for index, (entity, previous, record) in enumerate(
            zip(self.layout.entities, before, after)
        ):
            names = [joint.name for joint in entity.joints]
            expected: dict[str, tuple[np.ndarray, float, float]] = {}
            dof_tolerance = (1e-5, 1e-7)
            if "kp" in randomization or "kd" in randomization:
                stiffness = np.asarray(previous["dof_stiffness"], dtype=np.float64).copy()
                damping = np.asarray(previous["dof_damping"], dtype=np.float64).copy()
                for row, env in enumerate(envs):
                    for column, joint_name in zip(
                        entity.actuator_indices, entity.actuator_joint_names
                    ):
                        public = names.index(joint_name)
                        if "kp" in randomization:
                            stiffness[env, public] = randomization["kp"][row, column]
                        if "kd" in randomization:
                            damping[env, public] = randomization["kd"][row, column]
                expected["dof_stiffness"] = (stiffness, *dof_tolerance)
                expected["dof_damping"] = (damping, *dof_tolerance)
            for term, field in (
                ("dof_armature", "dof_armature"),
                ("dof_frictionloss", "dof_friction"),
            ):
                if term in randomization:
                    values = np.asarray(previous[field], dtype=np.float64).copy()
                    for row, env in enumerate(envs):
                        for public, joint in enumerate(entity.joints):
                            values[env, public] = randomization[term][
                                row, joint.qvel_indices[0]
                            ]
                    expected[field] = (values, *dof_tolerance)
            if "body_mass" in randomization:
                masses = np.asarray(previous["body_mass"], dtype=np.float64).copy()
                masses[envs] = randomization["body_mass"][:, list(entity.body_ids)]
                expected["body_mass"] = (masses, 2e-4, 1e-6)
            if "body_ipos" in randomization:
                coms = np.asarray(previous["body_ipos"], dtype=np.float64).copy()
                coms[envs] = randomization["body_ipos"][:, list(entity.body_ids)]
                expected["body_ipos"] = (coms, 2e-4, 1e-6)
            if "body_inertia" in randomization:
                inertias = np.asarray(previous["body_inertia"], dtype=np.float64).copy()
                for row, env in enumerate(envs):
                    variant = self.specs[index]["variants"][
                        self.records[env][index]["source_id"]
                    ]
                    for local, public_id in enumerate(entity.body_ids):
                        axes = self.protocol.quat_rotate(
                            np.asarray(variant["body_iquat"][local], dtype=np.float64),
                            np.eye(3),
                        )
                        inertias[env, local] = (
                            axes.T
                            @ np.diag(
                                np.asarray(
                                    randomization["body_inertia"][row, public_id],
                                    dtype=np.float64,
                                )
                            )
                            @ axes
                        )
                expected["body_inertia"] = (inertias, 5e-4, 1e-7)
            geom_count = len(entity.geoms)
            if "geom_friction" in randomization and geom_count:
                friction = np.asarray(previous["geom_friction"], dtype=np.float64).copy()
                friction[envs] = randomization["geom_friction"][
                    :, geom_offset : geom_offset + geom_count
                ]
                expected["geom_friction"] = (friction, 2e-5, 1e-6)
            geom_offset += geom_count
            for field, (values, rtol, atol) in expected.items():
                actual = np.asarray(record[field], dtype=np.float64)
                untouched = np.asarray(previous[field], dtype=np.float64)
                if not np.allclose(actual[selected], values[selected], rtol=rtol, atol=atol):
                    raise RuntimeError(
                        "entity %s native %s readback differs from reset" % (entity.name, field)
                    )
                if not np.allclose(actual[~selected], untouched[~selected], rtol=rtol, atol=atol):
                    raise RuntimeError(
                        "entity %s native %s write leaked outside selected rows"
                        % (entity.name, field)
                    )

    def _commit_reset_property_records(self, records: list[dict[str, Any]]) -> None:
        """Adopt verified native tables into the caches used by state refresh."""
        for index, (entity, record) in enumerate(zip(self.layout.entities, records)):
            coms = np.asarray(record["body_ipos"], dtype=np.float64)
            self.body_com[:, list(entity.body_ids)] = coms
            root_local = entity.body_names.index(entity.root_body)
            self.root_com[:, index] = coms[:, root_local]
            for env in range(self.num_envs):
                current = self.records[env][index]
                for field in (
                    "body_mass",
                    "body_inertia",
                    "dof_stiffness",
                    "dof_damping",
                    "dof_armature",
                    "dof_friction",
                    "geom_friction",
                ):
                    current[field] = record[field][env]
            actual = self.metadata.get("scene_entities_actual")
            if (
                isinstance(actual, list)
                and index < len(actual)
                and actual[index].get("name") == entity.name
            ):
                entry = actual[index]
                for field in (
                    "body_mass",
                    "body_ipos",
                    "body_inertia",
                    "dof_stiffness",
                    "dof_damping",
                    "dof_armature",
                    "dof_friction",
                    "geom_friction",
                ):
                    entry[field] = record[field]
        self._bind_refresh_indices()

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

    def _bind_refresh_indices(self) -> None:
        """Freeze native gathers once for both legacy and declared scenes."""
        self._joint_refresh = tuple(
            (
                np.asarray([record[index]["dof_ids"] for record in self.records], dtype=np.intp),
                tuple(joint.qpos_indices[0] for joint in entity.joints),
                tuple(joint.qvel_indices[0] for joint in entity.joints),
            )
            for index, entity in enumerate(self.layout.entities)
            if entity.joints
        )
        self._body_rows, self._body_columns = np.nonzero(self.body_ids >= 0)
        self._native_body_ids = self.body_ids[self._body_rows, self._body_columns]
        self._body_refresh_com = self.body_com[self._body_rows, self._body_columns]

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
        for dofs, pcols, vcols in self._joint_refresh:
            qpos[:, pcols] = native_dofs[dofs, 0]
            qvel[:, vcols] = native_dofs[dofs, 1]
        body = ctx.slots["body_state"]
        body.fill(0)
        body[..., 3] = 1
        contact = ctx.slots["contact_force"]
        contact.fill(0)
        native_body = ctx._body_state.cpu().numpy()
        native_contact = ctx._contact_force.cpu().numpy()
        body[self._body_rows, self._body_columns] = self._public_state(
            native_body[self._native_body_ids], self._body_refresh_com
        )
        contact[self._body_rows, self._body_columns] = native_contact[self._native_body_ids]
        # Root body has an authoritative actor state even when articulation FK is stale.
        if self.publish_actor_roots_as_body:
            for index, entity in enumerate(self.layout.entities):
                root_body = entity.body_ids[entity.body_names.index(entity.root_body)]
                body[:, root_body] = public_roots[:, index]
        if self.pending_body_fk:
            # Legacy path: PhysX keeps pre-write link poses until the first
            # simulate, so freshly reset envs publish exact FK rows (#141).
            # Stale contact forces from before the write are cleared too.
            for env, rows in self.pending_body_fk.items():
                body[env] = rows
                contact[env] = 0.0
        projection = getattr(self, "projection", None)
        if projection is not None:
            projection.publish()

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
        randomization = self._validated_reset_randomization(payload, count)
        property_records = None
        if randomization is not None:
            # Mutate model properties before staging state writes: the native
            # root-velocity conversion below must use post-mutation COM offsets.
            try:
                before_properties = self._readback_reset_property_records()
                self._apply_reset_randomization(envs, randomization)
                property_records = self._readback_reset_property_records()
                if ctx.use_gpu_pipeline and "body_ipos" in randomization:
                    self._substitute_gpu_com_readback(
                        property_records, envs, randomization["body_ipos"]
                    )
                self._verify_reset_property_readback(
                    envs, randomization, before_properties, property_records
                )
                self._commit_reset_property_records(property_records)
            except Exception:
                self.faulted = True
                raise
        for index, entity in enumerate(self.layout.entities):
            for row, env in enumerate(envs):
                record = self.records[int(env)][index]
                if np.any(rm[index]):
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
        if self._fk is not None and self.pending_body_fk is not None:
            # Publish the post-reset pose immediately: native link rows stay
            # stale until the first simulate (#141).  Compose the effective
            # generalized state so partial masks keep untouched columns.
            effective_qpos = ctx.slots["qpos"][envs].astype(np.float64, copy=True)
            effective_qvel = ctx.slots["qvel"][envs].astype(np.float64, copy=True)
            pos_columns = np.flatnonzero(pm)
            vel_columns = np.flatnonzero(vm)
            if len(pos_columns):
                effective_qpos[:, pos_columns] = qpos[:, pos_columns]
            if len(vel_columns):
                effective_qvel[:, vel_columns] = qvel[:, vel_columns]
            self.stage_fk_overlay_rows(envs, effective_qpos, effective_qvel)
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
        if property_records is not None:
            return {"timing": {}, "native_entity_records": property_records}
        return {"timing": {}}

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.faulted:
            raise RuntimeError("IsaacGym scene is faulted")
        nsteps = payload.get("nsteps")
        if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps <= 0:
            raise ValueError("nsteps must be a positive integer")
        ctx = self.ctx
        ctrl = finite_array(ctx.slots["ctrl"], (self.num_envs, self.layout.nu), "ctrl")
        wrench = None
        if "body_wrench" in payload:
            expected = (self.num_envs, self.layout.nbody, 6)
            encoded = payload["body_wrench"]
            nbytes = int(np.prod(expected, dtype=np.int64)) * np.dtype(np.float32).itemsize
            if not isinstance(encoded, bytes) or len(encoded) != nbytes:
                raise ValueError("body wrench payload must be C-order float32 bytes")
            wrench = np.frombuffer(encoded, dtype=np.float32).reshape(expected)
            if not np.isfinite(wrench).all():
                raise ValueError("body wrench contains NaN or Inf")
        timings = {}
        start = time.perf_counter()
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
            forces, torques = (
                self._wrench_tensors(wrench) if wrench is not None else (None, None)
            )
            timings["control_upload_ms"] = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            for index in range(nsteps):
                if wrench is not None:
                    # PhysX consumes external forces at each simulate; reapply so
                    # every substep of this STEP command sees the staged wrench.
                    if not ctx.gym.apply_rigid_body_force_tensors(
                        ctx.sim, forces, torques, ctx.gymapi.ENV_SPACE
                    ):
                        raise RuntimeError("native body wrench setter failed")
                ctx.gym.simulate(ctx.sim)
                ctx.gym.fetch_results(ctx.sim, True)
                if index == 0:
                    self.pending_roots.clear()
                    self.pending_dofs.clear()
                    self.pending_dof_actors.clear()
                    if self.pending_body_fk is not None:
                        self.pending_body_fk.clear()
            timings["physics_ms"] = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            self.refresh()
            timings["state_refresh_ms"] = (time.perf_counter() - start) * 1000.0
        except Exception:
            self.faulted = True
            raise
        return {"timing": timings}

    def _wrench_tensors(self, wrench: np.ndarray) -> tuple[Any, Any]:
        """Scatter the public (num_envs, nbody, 6) wrench into ENV_SPACE tensors."""
        ctx = self.ctx
        total = int(ctx._body_state.shape[0])
        per_env, remainder = divmod(total, self.num_envs)
        if remainder:
            raise RuntimeError("native rigid-body tensor is not env-aligned")
        valid = self.body_ids >= 0
        local = self.body_ids - np.arange(self.num_envs)[:, None] * per_env
        if np.any(valid & ((local < 0) | (local >= per_env))):
            raise RuntimeError("native rigid-body indices are not env-contiguous")
        target = np.arange(self.num_envs)[:, None] * per_env + np.clip(local, 0, None)
        forces = np.zeros((total, 3), dtype=np.float32)
        torques = np.zeros((total, 3), dtype=np.float32)
        forces[target[valid]] = wrench[..., 0:3][valid]
        torques[target[valid]] = wrench[..., 3:6][valid]
        unwrap = ctx.gymtorch.unwrap_tensor
        # unwrap_tensor only borrows the storage pointer: the Torch tensors
        # must outlive every apply call, or PhysX reads freed GPU memory.
        forces_t = ctx.torch.from_numpy(forces).to(ctx.device)
        torques_t = ctx.torch.from_numpy(torques).to(ctx.device)
        self._wrench_torch = (forces_t, torques_t)
        return unwrap(forces_t), unwrap(torques_t)
