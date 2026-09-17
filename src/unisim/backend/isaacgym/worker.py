"""Python 3.8 worker process for the out-of-process IsaacGym backend.

PYTHON 3.8 COMPATIBILITY: IsaacGym (Preview 4, EOL) only supports Python
3.6-3.8, so this file runs on the dedicated ``hsgym`` conda interpreter.  Keep
it stdlib + numpy + torch + isaacgym only, and never import ``unilab`` — the
shared protocol module is loaded by file path (``--protocol``) because the
worker interpreter has no access to the main environment's site-packages.

Message loop: read one framed command from stdin, dispatch, write one framed
reply to stdout.  Bulk state crosses the process boundary through shared
memory slots declared by the host (see ``protocol.slot_shapes``); the pipe
only carries commands, metadata, and error payloads.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Any, Dict, List, Tuple, cast

import numpy as np


def _load_protocol(path: str) -> Any:
    """Load the shared protocol module by file path (no package import)."""
    spec = importlib.util.spec_from_file_location("unisim_subprocess_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _WorkerContext:
    """Owns the IsaacGym sim, tensor views, and attached shared-memory slots."""

    def __init__(self, protocol: Any) -> None:
        self.protocol = protocol
        # IsaacGym/torch modules are imported inside init_sim; they do not
        # exist on the host interpreter, so these stay ``Any``.
        self.gymapi: Any = None
        self.gymtorch: Any = None
        self.torch: Any = None
        self.gym: Any = None
        self.sim: Any = None
        self.num_envs = 0
        self.num_dof = 0
        self.num_bodies = 0
        self.sim_dt = 0.0
        self.device = "cpu"
        self.use_gpu_pipeline = False
        self.env_handles: List[Any] = []
        self.actor_handles: List[Any] = []
        self.slots: Dict[str, np.ndarray] = {}
        self._shm_handles: List[Any] = []
        self._root_state: Any = None
        self._dof_state: Any = None
        self._body_state: Any = None
        self._contact_force: Any = None
        # Native rendering state (viewer and/or camera sensor).  Both live in
        # this process because the sim handle does.
        self.graphics_device_id = -1
        self.viewer: Any = None
        self.camera_handle: Any = None
        self.camera_env: Any = None
        self.camera_width = 0
        self.camera_height = 0
        # Defaults match the repository's MuJoCo playback camera convention
        # (elevation is the height angle above the horizon, in degrees).
        self.camera_distance = 2.0
        self.camera_elevation_deg = 20.0
        self.camera_azimuth_deg = 90.0
        self.scene_worker: Any = None

    # ------------------------------------------------------------------ #
    # INIT
    # ------------------------------------------------------------------ #

    def init_sim(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        path = os.path.join(os.path.dirname(__file__), "scene_worker.py")
        spec = importlib.util.spec_from_file_location("unisim_isaacgym_scene_worker", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load mapped IsaacGym scene worker")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if "scene_layout" in payload:
            self.scene_worker = module.SceneWorker(self, payload)
            return cast(Dict[str, Any], self.scene_worker.initialize())
        isaacgym_python = payload["isaacgym_python"]
        if isaacgym_python not in sys.path:
            sys.path.insert(0, isaacgym_python)
        # isaacgym must be imported before torch (it enforces this itself).
        from isaacgym import gymapi, gymtorch  # noqa: PLC0415, I001
        import torch  # noqa: PLC0415

        self.gymapi = gymapi
        self.gymtorch = gymtorch
        self.torch = torch

        gymapi = self.gymapi
        self.num_envs = int(payload["num_envs"])
        report_version = payload.get("configuration_report_version")
        if report_version not in (None, 1):
            raise RuntimeError("unsupported host configuration report schema version")
        self.sim_dt = float(payload["sim_dt"])
        device_id = int(payload.get("device_id", 0))
        self.use_gpu_pipeline = device_id >= 0
        self.device = "cuda:%d" % device_id if self.use_gpu_pipeline else "cpu"

        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = self.sim_dt
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = self.use_gpu_pipeline
        sim_params.use_gpu_pipeline = self.use_gpu_pipeline
        # The graphics context is enabled whenever the sim runs on a GPU
        # device.  It opens no window by itself (only create_viewer does) and
        # is required for both the interactive viewer and headless camera
        # capture; the cost for training-only runs is negligible.  CPU-pipeline
        # sims get no graphics context and fail closed on render requests.
        self.graphics_device_id = device_id if device_id >= 0 else -1
        self.sim = self.gym.create_sim(
            device_id, self.graphics_device_id, gymapi.SIM_PHYSX, sim_params
        )
        if self.sim is None:
            raise RuntimeError(
                "isaacgym create_sim failed (device_id=%d, gpu_pipeline=%s)"
                % (device_id, self.use_gpu_pipeline)
            )

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)

        model_file = os.fspath(payload["model_file"])
        raw_variant_files = payload.get("variant_model_files")
        if raw_variant_files is None:
            variant_model_files = [model_file]
            raw_assignment: Any = [0] * self.num_envs
            fixed_variants = False
        else:
            variant_model_files = [os.fspath(value) for value in raw_variant_files]
            if not variant_model_files:
                raise RuntimeError("isaacgym fixed variants require at least one source")
            raw_assignment = payload.get("variant_assignment")
            fixed_variants = True
        assignment = [int(value) for value in (raw_assignment or [])]
        if len(assignment) != self.num_envs:
            raise RuntimeError(
                "isaacgym fixed-variant assignment has %d entries; expected %d envs"
                % (len(assignment), self.num_envs)
            )
        if any(value < 0 or value >= len(variant_model_files) for value in assignment):
            raise RuntimeError("isaacgym fixed-variant assignment contains an invalid index")

        assets: List[Any] = []
        canonical_layout: Tuple[int, int, Tuple[str, ...], Tuple[str, ...]] | None = None
        for variant_file in variant_model_files:
            asset = self._load_mjcf_asset(variant_file, asset_options)
            layout = (
                int(self.gym.get_asset_dof_count(asset)),
                int(self.gym.get_asset_rigid_body_count(asset)),
                tuple(str(name) for name in self.gym.get_asset_dof_names(asset)),
                tuple(str(name) for name in self.gym.get_asset_rigid_body_names(asset)),
            )
            if canonical_layout is None:
                canonical_layout = layout
            elif layout != canonical_layout:
                raise RuntimeError(
                    "isaacgym fixed variant %r changes public layout: canonical=%r, "
                    "variant=%r; one actor slot requires identical dof/body counts and "
                    "name order" % (variant_file, canonical_layout, layout)
                )
            assets.append(asset)
        assert canonical_layout is not None
        asset = assets[0]

        self.num_dof, self.num_bodies, dof_names, body_names = canonical_layout
        self.dof_names: List[str] = list(dof_names)

        # Position-controlled dofs: ctrl is the per-dof position target,
        # matching MuJoCo <position kp kv forcerange> actuator semantics
        # (PhysX applies force = kp * (target - pos) - kv * vel, clamped to
        # the symmetric effort limit).  All parameters come from the host's
        # MJCF scan because the importer drops kv/frictionloss/joint ranges.
        variant_fields = payload.get("variant_dof_fields")
        if variant_fields is None:
            variant_fields = [self._legacy_dof_fields(payload)]
        if len(variant_fields) != len(assets):
            raise RuntimeError(
                "isaacgym INIT carried %d variant dof-field tables for %d assets"
                % (len(variant_fields), len(assets))
            )
        joint_names = [str(name) for name in (payload.get("mjcf_joint_names") or [])]
        variant_dof_props = []
        for variant_asset, fields in zip(assets, variant_fields):
            dof_props = self.gym.get_asset_dof_properties(variant_asset)
            self._apply_actuator_props(dof_props, fields, joint_names)
            variant_dof_props.append(dof_props)

        spacing = 2.0
        num_per_row = max(1, int(np.ceil(np.sqrt(self.num_envs))))
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, 0.0)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        for env_index in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            # collision_group=env_index isolates envs; filter=1 disables
            # self-collision.  The MJCF <contact><exclude> pairs (e.g. G1's
            # elbow/wrist and pelvis/hip overlaps) cannot be reproduced
            # per-link-pair through gymapi, and with self-collision on those
            # overlapping capsules generate permanent contact forces that
            # destabilize the drives.  Disabling self-collision is the
            # ecosystem-standard approximation (legged_gym, MetaSim) and a
            # superset of the MJCF exclusions.
            asset_index = assignment[env_index]
            actor_handle = self.gym.create_actor(
                env_handle, assets[asset_index], pose, "robot", env_index, 1
            )
            self.gym.set_actor_dof_properties(
                env_handle, actor_handle, variant_dof_props[asset_index]
            )
            self.env_handles.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.gym.prepare_sim(self.sim)
        self._acquire_tensors()

        variant_keyframes = payload.get("variant_keyframe_qpos")
        keyframe_qpos = payload.get("keyframe_qpos")
        if variant_keyframes is not None:
            if len(variant_keyframes) != len(assets):
                raise RuntimeError(
                    "isaacgym INIT carried %d variant keyframes for %d assets"
                    % (len(variant_keyframes), len(assets))
                )
            if any(value is not None for value in variant_keyframes):
                # Apply each variant's task-initial pose so post-INIT state
                # matches the host-side per-variant default-qpos contract.
                self._apply_variant_initial_keyframe(
                    variant_keyframes, joint_names, assignment, len(assets)
                )
        elif keyframe_qpos is not None:
            # Apply the scene's task-initial pose (AGENTS.md: the keyframe is
            # the task initial state) so the post-INIT state matches the
            # host-side get_default_qpos()/get_default_dof_pos() contract.
            self._apply_initial_keyframe(keyframe_qpos, joint_names)
        report_params = self.gym.get_sim_params(self.sim)
        report_bodies = [
            self.gym.get_actor_rigid_body_properties(env, actor)
            for env, actor in zip(self.env_handles, self.actor_handles)
        ]
        report_dofs = [
            self.gym.get_actor_dof_properties(env, actor)
            for env, actor in zip(self.env_handles, self.actor_handles)
        ]
        self._configuration_report = {
            "schema_version": 1,
            "effective": {
                "dt": float(report_params.dt),
                "gravity": [
                    float(report_params.gravity.x),
                    float(report_params.gravity.y),
                    float(report_params.gravity.z),
                ],
                "solver": "PhysX solver_type=%d" % report_params.physx.solver_type,
                "collision_filter": {"self_collision": False, "actor_filter": 1},
                "actuator_mapping": {
                    "joint_names": list(self.dof_names),
                    "per_env_stiffness": [row["stiffness"].tolist() for row in report_dofs],
                    "per_env_damping": [row["damping"].tolist() for row in report_dofs],
                    "per_env_effort": [row["effort"].tolist() for row in report_dofs],
                },
                "body_mass": {
                    "names": list(body_names),
                    "per_env_values": [[float(prop.mass) for prop in row] for row in report_bodies],
                },
                "body_inertia": {
                    "names": list(body_names),
                    "per_env_matrices": [
                        [
                            [
                                [
                                    float(getattr(getattr(prop.inertia, axis), coord))
                                    for coord in ("x", "y", "z")
                                ]
                                for axis in ("x", "y", "z")
                            ]
                            for prop in row
                        ]
                        for row in report_bodies
                    ],
                },
            },
            "engine_readback": [
                "dt",
                "gravity",
                "solver",
                "body_mass",
                "body_inertia",
                "actuator_mapping",
            ],
        }
        lower = np.asarray(variant_dof_props[0]["lower"], dtype=np.float64)
        upper = np.asarray(variant_dof_props[0]["upper"], dtype=np.float64)
        effort = np.asarray(variant_dof_props[0]["effort"], dtype=np.float64)
        metadata = {
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            "dof_names": list(self.dof_names),
            "body_names": list(body_names),
            "dof_lower": lower.tolist(),
            "dof_upper": upper.tolist(),
            "effort": effort.tolist(),
            "gravity": [0.0, 0.0, -9.81],
            "configuration_report": self._configuration_report,
            "use_gpu_pipeline": self.use_gpu_pipeline,
            "graphics_enabled": self.graphics_device_id >= 0,
            "fixed_variant_count": len(assets) if fixed_variants else 0,
            "fixed_variant_assignment": assignment if fixed_variants else [],
        }
        self.scene_worker = module.SceneWorker.adopt_initialized_context(
            self, metadata, payload, self.protocol.load_legacy_projection()
        )
        return metadata

    def _load_mjcf_asset(self, model_file: str, asset_options: Any) -> Any:
        asset_root, asset_file = os.path.split(model_file)
        if not asset_file.lower().endswith((".xml", ".mjcf")):
            raise RuntimeError(
                "isaacgym backend currently loads MJCF scenes only; got asset file "
                "%r. Convert the task scene or extend the worker asset loader." % asset_file
            )
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        if asset is None:
            raise RuntimeError(
                "isaacgym load_asset failed for %r. MJCF import requires the file to be "
                "self-contained for IsaacGym's importer (some MuJoCo elements are "
                "unsupported); run the worker command manually for the importer log." % model_file
            )
        return asset

    def _legacy_dof_fields(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "stiffness": payload["dof_stiffness"],
            "damping": payload["dof_damping"],
            "effort": payload["dof_effort"],
            "armature": payload["dof_armature"],
            "friction": payload["dof_friction"],
        }

    def _apply_actuator_props(
        self, dof_props: Any, fields: Dict[str, Any], joint_names_values: Any
    ) -> None:
        """Set per-dof PD/limit/dynamics properties from the host MJCF scan.

        The host sends arrays in MJCF joint document order
        (``mjcf_joint_names``); they are mapped onto the asset's dofs by NAME,
        because IsaacGym's MJCF importer is free to reorder joints.
        """
        gymapi = self.gymapi
        joint_names = [str(name) for name in (joint_names_values or [])]
        if len(joint_names) != self.num_dof:
            raise RuntimeError(
                "mjcf_joint_names has %d entries but the asset exposes %d dofs; "
                "IsaacGym's MJCF importer must preserve one dof per single-DoF joint"
                % (len(joint_names), self.num_dof)
            )
        index_by_name = {}
        for index, name in enumerate(joint_names):
            index_by_name[name] = index
        field_values = (
            ("stiffness", fields["stiffness"]),
            ("damping", fields["damping"]),
            ("effort", fields["effort"]),
            ("armature", fields["armature"]),
            ("friction", fields["friction"]),
        )
        for dof_index, dof_name in enumerate(self.dof_names):
            if dof_name not in index_by_name:
                raise RuntimeError(
                    "isaacgym asset dof %r is missing from mjcf_joint_names; the MJCF "
                    "importer may have dropped or renamed the joint" % dof_name
                )
            source = index_by_name[dof_name]
            dof_props["driveMode"][dof_index] = int(gymapi.DOF_MODE_POS)
            for field, values in field_values:
                dof_props[field][dof_index] = float(values[source])

    def _apply_initial_keyframe(self, qpos_values: Any, joint_names: Any) -> None:
        """Write the scene keyframe pose into every env via the tensor API.

        ``qpos_values`` follows the MJCF layout: 7 free-root columns
        (xyz + wxyz quat) plus one column per single-DoF joint in document
        order (``joint_names``).  DoF values are mapped onto the asset's dofs
        by NAME, because IsaacGym's MJCF importer is free to reorder joints.
        """
        self._apply_initial_rows([qpos_values] * self.num_envs, joint_names)

    def _apply_variant_initial_keyframe(
        self,
        qpos_by_variant: Any,
        joint_names: Any,
        assignment: List[int],
        asset_count: int,
    ) -> None:
        """Write each environment's assigned variant keyframe pose."""
        if len(qpos_by_variant) != asset_count:
            raise RuntimeError(
                "variant keyframe table has %d entries but %d assets were loaded"
                % (len(qpos_by_variant), asset_count)
            )
        if any(value is None for value in qpos_by_variant):
            raise RuntimeError(
                "isaacgym fixed variants require an initial keyframe for every variant"
            )
        rows = [qpos_by_variant[assignment[env_index]] for env_index in range(self.num_envs)]
        self._apply_initial_rows(rows, joint_names)

    def _apply_initial_rows(self, qpos_rows: Any, joint_names: Any) -> None:
        """Upload one canonical-layout qpos row per environment."""
        torch = self.torch
        expected = 7 + self.num_dof
        joint_names = [str(name) for name in joint_names]
        if len(joint_names) != self.num_dof:
            raise RuntimeError(
                "mjcf_joint_names has %d entries but the asset exposes %d dofs; "
                "IsaacGym's MJCF importer must preserve one dof per single-DoF joint"
                % (len(joint_names), self.num_dof)
            )
        index_by_name = {}
        for index, name in enumerate(joint_names):
            index_by_name[name] = index
        source_indices = np.empty((self.num_dof,), dtype=np.intp)
        for dof_index, dof_name in enumerate(self.dof_names):
            if dof_name not in index_by_name:
                raise RuntimeError(
                    "isaacgym asset dof %r is missing from mjcf_joint_names; the MJCF "
                    "importer may have dropped or renamed the joint" % dof_name
                )
            source_indices[dof_index] = index_by_name[dof_name]

        qpos_rows_matrix = np.zeros((self.num_envs, expected), dtype=np.float32)
        for env_index, qpos_values in enumerate(qpos_rows):
            qpos = np.asarray(qpos_values, dtype=np.float32).reshape(-1)
            if qpos.size != expected:
                raise RuntimeError(
                    "keyframe qpos has %d entries; expected %d (7 root + %d dofs)"
                    % (qpos.size, expected, self.num_dof)
                )
            qpos_rows_matrix[env_index, :] = qpos

        root = np.zeros((self.num_envs, 13), dtype=np.float32)
        root[:, 0:3] = qpos_rows_matrix[:, 0:3]
        root[:, 3:7] = self.protocol.wxyz_to_xyzw(qpos_rows_matrix[:, 3:7])
        dof_pos = qpos_rows_matrix[:, 7 + source_indices]

        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        root_view = self._root_state.view(self.num_envs, -1, 13)
        root_view[:, 0, :] = torch.from_numpy(root).to(self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            self.gymtorch.unwrap_tensor(self._root_state),
            self.gymtorch.unwrap_tensor(env_ids),
            self.num_envs,
        )
        dof = np.zeros((self.num_envs, self.num_dof, 2), dtype=np.float32)
        dof[:, :, 0] = dof_pos
        dof_view = self._dof_state.view(self.num_envs, self.num_dof, 2)
        dof_view[:, :, :] = torch.from_numpy(dof).to(self.device)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            self.gymtorch.unwrap_tensor(self._dof_state),
            self.gymtorch.unwrap_tensor(env_ids),
            self.num_envs,
        )
        # Root/dof tensors read back coherently without a physics step; rigid
        # body states stay at the spawn pose until the first STEP (the same
        # documented staleness as SET_STATE).
        self._refresh_tensors()

    def _acquire_tensors(self) -> None:
        gym = self.gym
        gymtorch = self.gymtorch
        self._root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(self.sim))
        if self.scene_worker is not None and self.num_dof == 0:
            self._dof_state = self.torch.empty((0, 2), dtype=self.torch.float32, device=self.device)
        else:
            self._dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(self.sim))
        self._body_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(self.sim))
        self._contact_force = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(self.sim))

    # ------------------------------------------------------------------ #
    # Shared-memory slots
    # ------------------------------------------------------------------ #

    def attach_slots(self, payload: Dict[str, Any]) -> None:
        """Attach host-created shm slots and detach them from resource tracking.

        Python's shared_memory resource tracker would otherwise unlink the
        host-owned segments when this worker exits (CPython issue 39959), so
        every attached name is unregistered here; the host owns unlinking.
        """
        from multiprocessing import resource_tracker, shared_memory  # noqa: PLC0415

        self.protocol.validate_slot_specs(
            payload["slots"],
            (
                self.protocol.scene_slot_shapes(self.num_envs, self.scene_worker.layout)
                if self.scene_worker is not None and not hasattr(self.scene_worker, "projection")
                else self.protocol.slot_shapes(self.num_envs, self.num_dof, self.num_bodies)
            ),
        )
        for name, spec in payload["slots"].items():
            handle = shared_memory.SharedMemory(name=spec["shm"], create=False)
            resource_tracker.unregister(handle._name, "shared_memory")  # type: ignore[attr-defined]
            array = np.ndarray(
                tuple(spec["shape"]), dtype=np.dtype(spec["dtype"]), buffer=handle.buf
            )
            self.slots[name] = array
            self._shm_handles.append(handle)
        if hasattr(self.scene_worker, "projection"):
            self.slots = self.scene_worker.projection.attach(self.slots)
        if self.scene_worker is not None and self.scene_worker.initial_ctrl is not None:
            np.copyto(
                self.slots["ctrl"],
                self.scene_worker.initial_ctrl.astype(self.slots["ctrl"].dtype),
            )
        self.refresh_state_slots()

    # ------------------------------------------------------------------ #
    # State exchange
    # ------------------------------------------------------------------ #

    def _refresh_tensors(self) -> None:
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def refresh_state_slots(self) -> None:
        """Copy the latest tensor state into every host-visible shm slot."""
        self.scene_worker.refresh()

    def step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.scene_worker.step(payload))

    def set_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self.scene_worker, "projection"):
            raise NotImplementedError("mapped scenes use RESET_ENTITIES with explicit masks")
        count = payload.get("count")
        if count == 0:
            return {"timing": {}}
        request = self.scene_worker.projection.prepare_reset(count)
        return cast(Dict[str, Any], self.scene_worker.reset(request))

    def get_meta(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.scene_worker.metadata)

    # ------------------------------------------------------------------ #
    # Native rendering (viewer + camera sensor)
    # ------------------------------------------------------------------ #

    def _require_graphics(self) -> None:
        if self.graphics_device_id < 0:
            raise RuntimeError(
                "isaacgym rendering requires a GPU sim (device_id >= 0); this sim was "
                "created without a graphics context"
            )

    def init_renderer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create the interactive viewer and/or the headless capture camera."""
        gym = self.gym
        gymapi = self.gymapi
        self._require_graphics()
        headless = bool(payload.get("headless", False))
        capture = bool(payload.get("capture", False))

        if not headless and self.viewer is None:
            viewer = gym.create_viewer(self.sim, gymapi.CameraProperties())
            if viewer is None:
                raise RuntimeError(
                    "isaacgym create_viewer failed (no display reachable); use "
                    "play_render_mode=record for headless video capture"
                )
            # Default view: env 0 area, slightly above the grid.
            gym.viewer_camera_look_at(
                viewer,
                None,
                gymapi.Vec3(2.5, 2.5, 1.8),
                gymapi.Vec3(0.0, 0.0, 0.5),
            )
            self.viewer = viewer

        if capture and self.camera_handle is None:
            camera = payload.get("camera") or {}
            self.camera_distance = float(camera.get("distance", 2.0))
            self.camera_elevation_deg = float(camera.get("elevation_deg", 20.0))
            self.camera_azimuth_deg = float(camera.get("azimuth_deg", 90.0))
            self.camera_width = int(payload.get("width", 1280))
            self.camera_height = int(payload.get("height", 720))
            cam_props = gymapi.CameraProperties()
            cam_props.width = self.camera_width
            cam_props.height = self.camera_height
            self.camera_env = self.env_handles[0]
            self.camera_handle = gym.create_camera_sensor(self.camera_env, cam_props)
            self._position_tracking_camera()

        return {
            "viewer": self.viewer is not None,
            "capture": self.camera_handle is not None,
        }

    def _position_tracking_camera(self) -> None:
        """Aim the capture camera at env 0's root on a spherical offset."""
        import math  # noqa: PLC0415

        gymapi = self.gymapi
        root = self._root_state.view(self.num_envs, -1, 13)[0, 0, :].cpu().numpy()
        target = np.asarray(root[0:3], dtype=np.float64)
        elevation = math.radians(self.camera_elevation_deg)
        azimuth = math.radians(self.camera_azimuth_deg)
        offset = self.camera_distance * np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )
        eye = target + offset
        self.gym.set_camera_location(
            self.camera_handle,
            self.camera_env,
            gymapi.Vec3(float(eye[0]), float(eye[1]), float(eye[2])),
            gymapi.Vec3(float(target[0]), float(target[1]), float(target[2])),
        )

    def render_frame(self) -> Dict[str, Any]:
        """Draw one viewer frame; report whether the user closed the window."""
        if self.viewer is None:
            raise RuntimeError("isaacgym viewer is not initialized; call INIT_RENDERER first")
        gym = self.gym
        if gym.query_viewer_has_closed(self.viewer):
            self._destroy_viewer()
            return {"closed": True}
        gym.step_graphics(self.sim)
        gym.draw_viewer(self.viewer, self.sim, True)
        if gym.query_viewer_has_closed(self.viewer):
            self._destroy_viewer()
            return {"closed": True}
        return {"closed": False}

    def capture_frame(self) -> Dict[str, Any]:
        """Render the capture camera and return one RGB uint8 frame."""
        if self.camera_handle is None:
            raise RuntimeError(
                "isaacgym capture camera is not initialized; call INIT_RENDERER first"
            )
        gym = self.gym
        self._position_tracking_camera()
        gym.step_graphics(self.sim)
        gym.render_all_camera_sensors(self.sim)
        image = np.asarray(
            gym.get_camera_image(
                self.sim, self.camera_env, self.camera_handle, self.gymapi.IMAGE_COLOR
            )
        )
        frame = np.ascontiguousarray(
            image.reshape(self.camera_height, self.camera_width, 4)[:, :, :3]
        )
        return {
            "frame": frame,
            "width": self.camera_width,
            "height": self.camera_height,
        }

    def _destroy_viewer(self) -> None:
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None

    def shutdown(self) -> None:
        if self.gym is not None:
            self._destroy_viewer()
        if self.gym is not None and self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None
        for handle in self._shm_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._shm_handles = []


def _dispatch(ctx: _WorkerContext, protocol: Any, cmd: str, payload: Any) -> Tuple[str, Any]:
    if cmd == protocol.CMD_INIT:
        return protocol.CMD_META, ctx.init_sim(payload)
    if cmd == protocol.CMD_ATTACH:
        ctx.attach_slots(payload)
        return protocol.CMD_READY, None
    if cmd == protocol.CMD_STEP:
        return protocol.CMD_READY, ctx.step(payload)
    if cmd == protocol.CMD_SET_STATE:
        return protocol.CMD_READY, ctx.set_state(payload)
    if cmd == protocol.CMD_RESET_ENTITIES:
        if ctx.scene_worker is None:
            raise NotImplementedError("entity reset requires a mapped scene")
        return protocol.CMD_READY, ctx.scene_worker.reset(payload)
    if cmd == protocol.CMD_REFRESH:
        ctx.refresh_state_slots()
        return protocol.CMD_READY, None
    if cmd == protocol.CMD_GET_META:
        return protocol.CMD_META, ctx.get_meta()
    if cmd == protocol.CMD_INIT_RENDERER:
        return protocol.CMD_META, ctx.init_renderer(payload)
    if cmd == protocol.CMD_RENDER_FRAME:
        return protocol.CMD_META, ctx.render_frame()
    if cmd == protocol.CMD_CAPTURE_FRAME:
        return protocol.CMD_META, ctx.capture_frame()
    raise ValueError(f"unknown command {cmd!r}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, help="path to protocol.py")
    args = parser.parse_args(argv)
    protocol = _load_protocol(args.protocol)
    ctx = _WorkerContext(protocol)

    stdin = sys.stdin.buffer
    # IsaacGym's native extension prints banners straight to fd 1, which would
    # corrupt the framed protocol. Keep a private copy of the original stdout
    # for protocol messages and reroute fd 1 (and with it sys.stdout) to
    # stderr, where the parent captures it for crash diagnostics.
    protocol_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    stdout = protocol_out
    while True:
        try:
            message = protocol.recv_message(stdin)
        except (EOFError, protocol.WorkerDisconnectedError):
            return 0
        cmd = message["cmd"]
        payload = message.get("payload")
        if cmd == protocol.CMD_SHUTDOWN:
            try:
                ctx.shutdown()
            finally:
                protocol.send_message(stdout, protocol.CMD_READY)
            return 0
        try:
            reply_cmd, reply_payload = _dispatch(ctx, protocol, cmd, payload)
        except Exception as exc:  # noqa: BLE001 - every worker error crosses the wire
            error = protocol.serialize_exception(exc)
            error["faulted"] = bool(ctx.scene_worker is not None and ctx.scene_worker.faulted)
            protocol.send_message(stdout, protocol.CMD_ERROR, error)
            continue
        protocol.send_message(stdout, reply_cmd, reply_payload)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
