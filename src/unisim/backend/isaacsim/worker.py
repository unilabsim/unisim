"""IsaacSim/IsaacLab Python 3.11 worker for the UniLab subprocess backend.

The worker is intentionally self-contained.  It imports Kit and IsaacLab only
after receiving ``INIT`` and communicates with the host through the canonical
``subprocess_ipc.protocol`` module loaded by path.  Control messages stay on
the pipe; all batched numeric state is copied into shared-memory slots.

This worker implements MJCF-backed articulation physics, masked root/joint
reset, implicit position-target control, and the eval-owned Kit viewer/RGB
camera commands.  Rendering is selected before Kit starts so a training
worker can remain on the inexpensive no-rendering experience.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HOST_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


class _HostUniSimFinder:
    """Load host UniSim modules without exposing sibling host distributions."""

    def __init__(self, package_root: Path) -> None:
        self._package_root = package_root

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != "unisim" and not fullname.startswith("unisim."):
            return None
        search_path = (
            [str(self._package_root)]
            if fullname == "unisim" or path is None
            else path
        )
        return importlib.machinery.PathFinder.find_spec(
            fullname,
            search_path,
            target,
        )


sys.meta_path.insert(0, _HostUniSimFinder(_HOST_PACKAGE_ROOT))


def _load_protocol(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("unisim_subprocess_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_numpy(value: Any) -> np.ndarray:
    """Detach one IsaacLab tensor at the worker/shm boundary."""
    # IsaacLab's ``Articulation.data`` quantities are torch tensors.  Keep the
    # conversion explicit at this one cold/IO boundary: probing arbitrary
    # backend objects with ``hasattr``/``getattr`` in the physics path can hide
    # API drift and violates the backend-isolation contract.  A non-tensor is
    # an implementation error and should fail loudly instead of being
    # silently coerced through NumPy.
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _to_tensor(torch: Any, value: np.ndarray, device: str) -> Any:
    return torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32, device=device)


def _resolve_articulation_root_prim_path(usd_path: str, root_name: str) -> str:
    """Resolve the imported articulation root relative to the asset prim.

    MJCF conversion does not promise a fixed nesting depth.  G1 currently
    yields ``/<asset>/<pelvis>/<pelvis>`` while other assets may expose the
    articulation root directly under the asset prim.  Discover the prim once
    during materialization and fail closed when the requested root is
    ambiguous or absent; never guess this path on a physics hot path.
    """
    if not root_name:
        raise ValueError("root_name must be a non-empty body name")
    try:
        from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - only runs in external worker
        raise RuntimeError("IsaacSim USD bindings are unavailable") from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"IsaacSim could not open converted USD asset {usd_path!r}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError(f"IsaacSim converted USD asset {usd_path!r} has no valid default prim")
    asset_path = str(default_prim.GetPath()).rstrip("/")
    candidates = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(asset_path + "/"):
            continue
        if path.rsplit("/", 1)[-1] != root_name:
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            "IsaacSim converted USD articulation root lookup for body "
            f"{root_name!r} expected one ArticulationRootAPI prim below "
            f"{asset_path!r}, found {candidates or '<none>'}"
        )
    relative = candidates[0][len(asset_path) :]
    if not relative.startswith("/"):
        raise RuntimeError(
            f"IsaacSim articulation root {candidates[0]!r} is not below {asset_path!r}"
        )
    return relative


class _WorkerContext:
    def __init__(self, protocol: Any) -> None:
        self.protocol = protocol
        self.num_envs = 0
        self.num_dof = 0
        self.num_bodies = 0
        self.sim_dt = 0.0
        self.device = "cuda:0"
        self.sim: Any = None
        self.robot: Any = None
        self.simulation_app: Any = None
        self.torch: Any = None
        self.render_mode = "none"
        self.render_width = 1280
        self.render_height = 720
        self.camera: Any = None
        self.camera_distance = 2.0
        self.camera_elevation_deg = 20.0
        self.camera_azimuth_deg = 90.0
        self.native_joint_names: list[str] = []
        self.native_body_names: list[str] = []
        self.contract_joint_names: list[str] = []
        self.contract_body_names: list[str] = []
        self.native_joint_for_contract: np.ndarray = np.empty(0, dtype=np.int64)
        self.native_body_for_contract: np.ndarray = np.empty(0, dtype=np.int64)
        # Physical clones are translated apart in the worker so that their
        # collision geometry does not overlap.  UniLab's flat-scene contract
        # exposes per-environment local coordinates (``env_origins`` is zero),
        # therefore these offsets stay private to the worker and are removed
        # at the shared-memory boundary.
        self.env_origins = np.empty((0, 3), dtype=np.float32)
        self.env_prim_paths: list[str] = []
        self.collision_filtering_applied = False
        self.slots: dict[str, np.ndarray] = {}
        self._shm_handles: list[Any] = []

    # ------------------------------------------------------------------
    # Cold-path materialization
    # ------------------------------------------------------------------

    def init_sim(self, payload: dict[str, Any]) -> dict[str, Any]:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
        self.num_envs = int(payload["num_envs"])
        report_version = payload.get("configuration_report_version")
        if report_version not in (None, 1):
            raise RuntimeError("unsupported host configuration report schema version")
        self.sim_dt = float(payload["sim_dt"])
        device_id = int(payload.get("device_id", 0))
        if device_id < 0:
            raise NotImplementedError(
                "isaacsim requires a CUDA device; CPU IsaacLab physics is outside the "
                "supported subprocess profile"
            )
        self.device = f"cuda:{device_id}"

        raw_render_mode = payload.get("render_mode", "none")
        if not isinstance(raw_render_mode, str):
            raise TypeError(
                "isaacsim worker render_mode must be a string; "
                f"got {type(raw_render_mode).__name__}"
            )
        render_mode = raw_render_mode.strip().lower()
        if render_mode not in {"none", "record", "interactive"}:
            raise ValueError(
                "isaacsim worker render_mode must be one of none, record, interactive; "
                f"got {render_mode!r}"
            )
        self.render_mode = render_mode
        raw_render_width = payload.get("render_width", 1280)
        raw_render_height = payload.get("render_height", 720)
        if (
            isinstance(raw_render_width, bool)
            or not isinstance(raw_render_width, int)
            or isinstance(raw_render_height, bool)
            or not isinstance(raw_render_height, int)
            or raw_render_width <= 0
            or raw_render_height <= 0
        ):
            raise ValueError(
                "isaacsim worker render dimensions must be positive integers; "
                f"got {raw_render_width!r}x{raw_render_height!r}"
            )
        self.render_width = raw_render_width
        self.render_height = raw_render_height
        headless = render_mode != "interactive"
        enable_cameras = render_mode == "record"
        # AppLauncher treats false/default values as "consult the environment".
        # Pin both variables explicitly so a user's shell cannot accidentally
        # turn a training worker into a GUI/camera process.
        os.environ["HEADLESS"] = "1" if headless else "0"
        os.environ["ENABLE_CAMERAS"] = "1" if enable_cameras else "0"
        os.environ["LIVESTREAM"] = "0"
        os.environ["XR"] = "0"

        # Kit must be launched before importing IsaacSim/IsaacLab modules.
        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

        self.simulation_app = AppLauncher(
            {
                "headless": headless,
                "enable_cameras": enable_cameras,
                "device": self.device,
                "multi_gpu": False,
                "width": self.render_width,
                "height": self.render_height,
                "window_width": self.render_width,
                "window_height": self.render_height,
            }
        ).app

        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        import isaacsim.core.utils.prims as prim_utils  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore[import-not-found]
        from isaaclab.assets import Articulation, ArticulationCfg  # type: ignore[import-not-found]
        from isaaclab.sim.converters import (  # type: ignore[import-not-found]
            MjcfConverter,
            MjcfConverterCfg,
        )
        from isaacsim.core.cloner import GridCloner  # type: ignore[import-not-found]
        from isaacsim.core.utils.extensions import (
            enable_extension,  # type: ignore[import-not-found]
        )

        self.torch = torch
        # The extension is enabled explicitly because IsaacSim 5.1 does not
        # guarantee the MJCF importer is active in a bare headless AppLauncher.
        enable_extension("isaacsim.asset.importer.mjcf")

        model_file = os.fspath(payload["model_file"])
        converter = MjcfConverter(
            MjcfConverterCfg(
                asset_path=model_file,
                fix_base=False,
                import_sites=True,
                make_instanceable=True,
                self_collision=False,
            )
        )

        # Build a deterministic environment grid.  The USD importer owns the
        # robot hierarchy; only these Xforms and the articulation wrapper are
        # created here, so no asset/XML parsing occurs on a hot path.  The
        # translations are private worker offsets; state is normalized back to
        # local coordinates before it is published to the host.
        cloner = GridCloner(spacing=2.0)
        cloner.define_base_env("/World/envs")
        self.env_prim_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        # The source Xform must exist before GridCloner.clone.  The returned
        # transforms are the authoritative world origins (a centered grid for
        # two or more environments), so no duplicate hand-written grid math is
        # needed here.
        prim_utils.create_prim(self.env_prim_paths[0], "Xform")
        self.env_origins = np.asarray(
            cloner.clone(
                source_prim_path=self.env_prim_paths[0],
                prim_paths=self.env_prim_paths,
                replicate_physics=False,
                copy_from_source=True,
            ),
            dtype=np.float32,
        )
        expected_origins = (self.num_envs, 3)
        if self.env_origins.shape != expected_origins or not np.isfinite(self.env_origins).all():
            raise RuntimeError(
                "IsaacSim GridCloner returned invalid environment origins: "
                f"shape={self.env_origins.shape}, expected={expected_origins}"
            )

        root_name = str(payload.get("root_body_name") or "")
        if not root_name:
            raise ValueError(
                "isaacsim INIT requires root_body_name so articulation_root_prim_path "
                "is explicit and importer discovery cannot choose a wrong root"
            )
        # IsaacLab resolves this path relative to each /Robot instance.  The
        # converter's nesting is asset-dependent, so discover it from the
        # converted USD stage rather than baking in the G1 layout.
        articulation_root = _resolve_articulation_root_prim_path(converter.usd_path, root_name)
        joint_names = [str(name) for name in (payload.get("mjcf_joint_names") or [])]
        if not joint_names:
            raise ValueError("isaacsim INIT requires the MJCF joint-name contract")
        gains = self._actuator_dicts(payload, joint_names)
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            articulation_root_prim_path=articulation_root,
            spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=gains["stiffness"],
                    damping=gains["damping"],
                    effort_limit_sim=gains["effort"],
                    armature=gains["armature"],
                    friction=gains["friction"],
                )
            },
        )
        sim_cfg = sim_utils.SimulationCfg(dt=self.sim_dt, device=self.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        if render_mode != "none":
            # Use IsaacSim's standard grid-world floor for rendered playback.
            # The MJCF floor is retained for the task/physics contract, while
            # this native floor supplies the normal IsaacSim visual ground.
            ground_cfg = sim_utils.GroundPlaneCfg()
            ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        # IsaacLab's SimulationContext owns the singleton simulation stage and
        # must be materialized before assets/articulations bind to it.  Keep
        # this ordering explicit so a real Kit worker does not accidentally
        # construct an Articulation against an uninitialized context.
        self.robot = Articulation(robot_cfg)
        if render_mode != "none":
            # MJCF scenes do not necessarily carry a renderer light.  This is
            # a real scene light (not a post-process or synthetic frame), and
            # is created only on the cold rendering path.
            light_cfg = sim_utils.DomeLightCfg(
                # MJCF scenes already provide a world light.  A 2500-lumen
                # dome on top of that light clips the converted materials on
                # RTX cameras (the RGB stream becomes nearly uniform white).
                # Keep a low fill light so the imported scene remains visible
                # without washing out its silver/black contrast.
                intensity=100.0,
                color=(0.75, 0.75, 0.75),
            )
            light_cfg.func("/World/UniLabDomeLight", light_cfg)
            if render_mode == "record":
                from isaaclab.sensors.camera import (  # type: ignore[import-not-found]
                    Camera,
                    CameraCfg,
                )

                camera_cfg = CameraCfg(
                    # Playback emits one video stream, so own one camera in
                    # env 0 rather than allocating an RTX render product for
                    # every policy-eval environment. This mirrors the
                    # IsaacGym capture path and keeps camera cost independent
                    # of ``training.play_env_num``.
                    prim_path="/World/envs/env_0/UniLabCamera",
                    update_period=0.0,
                    data_types=["rgb"],
                    width=self.render_width,
                    height=self.render_height,
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.1, 1.0e5),
                    ),
                )
                self.camera = Camera(camera_cfg)
        # Apply IsaacLab's PhysX collision-group filtering before the first
        # reset/step.  Without this stage operation, the translated clones
        # can still collide when a reset puts two local roots at the same pose.
        # Failing closed is important: an unfiltered batch is not equivalent
        # to the SimBackend's independent-environment contract.
        if self.num_envs > 1:
            cloner.filter_collisions(
                self._physics_scene_path(),
                "/World/collisions",
                self.env_prim_paths,
            )
            self.collision_filtering_applied = True
        self.sim.reset()
        self.robot.update(self.sim_dt)
        if self.camera is not None:
            if not self.camera.is_initialized:
                raise RuntimeError(
                    "IsaacSim RGB camera did not initialize; ensure the Kit experience "
                    "was launched with enable_cameras=True"
                )
            self.camera.reset()
            self.camera.update(self.sim_dt, force_recompute=True)

        self.native_joint_names = [str(name) for name in self.robot.joint_names]
        self.native_body_names = [str(name) for name in self.robot.body_names]
        self.num_dof = int(self.robot.num_joints)
        self.num_bodies = int(self.robot.num_bodies)
        self.contract_joint_names = joint_names
        self.contract_body_names = [
            str(name) for name in (payload.get("mjcf_body_names") or self.native_body_names)
        ]
        self.native_joint_for_contract = self._build_permutation(
            self.native_joint_names, self.contract_joint_names, "joint"
        )
        self.native_body_for_contract = self._build_permutation(
            self.native_body_names, self.contract_body_names, "body"
        )

        self._configuration_report = {
            "schema_version": 1,
            "effective": {
                "dt": float(self.sim.get_physics_dt()),
                "gravity": list(sim_cfg.gravity),
                "collision_filter": {"self_collision": False},
                "actuator_mapping": {"joint_names": list(joint_names),
                    "stiffness": gains["stiffness"], "damping": gains["damping"],
                    "effort": gains["effort"]},
                "body_mass": {"names": list(self.native_body_names), "per_env_values":
                              self.robot.root_physx_view.get_masses().cpu().tolist()},
                "body_inertia": {"names": list(self.native_body_names), "per_env_matrices":
                                 self.robot.root_physx_view.get_inertias().reshape(
                                     self.num_envs, self.num_bodies, 3, 3).cpu().tolist()},
            },
            "engine_readback": ["dt", "body_mass", "body_inertia"],
        }
        keyframe_qpos = payload.get("keyframe_qpos")
        if keyframe_qpos is not None:
            self._apply_keyframe(keyframe_qpos)
        return {
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            # Expose UniLab contract order, not the importer/native order.
            "dof_names": list(self.contract_joint_names),
            "body_names": list(self.contract_body_names),
            "dof_lower": self._joint_limits()[0],
            "dof_upper": self._joint_limits()[1],
            "effort": self._joint_limits()[2],
            "gravity": [0.0, 0.0, -9.81],
            "configuration_report": self._configuration_report,
            "use_gpu_pipeline": True,
            "graphics_enabled": render_mode != "none",
            "render_mode": render_mode,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "native_dof_names": list(self.native_joint_names),
            "native_body_names": list(self.native_body_names),
            "usd_path": str(converter.usd_path),
            "env_origins": self.env_origins.tolist(),
            "collision_filtering_applied": self.collision_filtering_applied,
        }

    def _physics_scene_path(self) -> str:
        """Find the stage's PhysX scene prim on the materialization path."""
        try:
            from pxr import PhysxSchema  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - external worker only
            raise RuntimeError("IsaacSim PhysX USD bindings are unavailable") from exc
        for prim in self.robot.stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                return str(prim.GetPath())
        raise RuntimeError(
            "IsaacSim stage has no PhysxSceneAPI; cannot filter environment collisions"
        )

    @staticmethod
    def _build_permutation(native: list[str], contract: list[str], kind: str) -> np.ndarray:
        if len(native) != len(contract) or len(set(native)) != len(native):
            raise RuntimeError(
                f"isaacsim importer returned invalid {kind} names: "
                f"native={native}, contract={contract}"
            )
        native_ids = {name: index for index, name in enumerate(native)}
        missing = [name for name in contract if name not in native_ids]
        extra = [name for name in native if name not in set(contract)]
        if missing or extra or len(set(contract)) != len(contract):
            raise RuntimeError(
                f"isaacsim importer {kind} mapping mismatch: missing={missing}, extra={extra}, "
                f"native={native}, contract={contract}"
            )
        return np.asarray([native_ids[name] for name in contract], dtype=np.int64)

    @staticmethod
    def _actuator_dicts(payload: dict[str, Any], names: list[str]) -> dict[str, dict[str, float]]:
        def values(key: str, default: float = 0.0) -> dict[str, float]:
            raw = list(payload.get(key) or [])
            if len(raw) != len(names):
                raise RuntimeError(
                    f"{key} has {len(raw)} values but the MJCF contract has {len(names)} joints"
                )
            return {name: float(raw[index]) for index, name in enumerate(names)}

        effort = values("dof_effort")
        # PhysX/IsaacLab reject an infinite or excessively large effort in
        # some releases.  The host uses 1e20 as the unlimited sentinel; use
        # the documented finite implicit-actuator ceiling in the worker.
        effort = {
            name: (1.0e9 if value <= 0.0 or value >= 1.0e19 else value)
            for name, value in effort.items()
        }
        return {
            "stiffness": values("dof_stiffness"),
            "damping": values("dof_damping"),
            "effort": effort,
            "armature": values("dof_armature"),
            "friction": values("dof_friction"),
        }

    def _joint_limits(self) -> tuple[list[float], list[float], list[float]]:
        limits = _tensor_numpy(self.robot.data.joint_pos_limits)[0]
        efforts = _tensor_numpy(self.robot.data.joint_effort_limits)[0]
        # Reorder native metadata to the public contract order.
        limits = limits[self.native_joint_for_contract]
        efforts = efforts[self.native_joint_for_contract]
        return limits[:, 0].tolist(), limits[:, 1].tolist(), efforts.tolist()

    def _apply_keyframe(self, qpos_values: Any) -> None:
        qpos = np.asarray(qpos_values, dtype=np.float32).reshape(-1)
        if qpos.size != 7 + self.num_dof:
            raise RuntimeError(
                f"keyframe qpos has {qpos.size} entries; expected {7 + self.num_dof}"
            )
        env_ids = self.torch.arange(self.num_envs, dtype=self.torch.long, device=self.device)
        root_pose_np = np.broadcast_to(qpos[:7], (self.num_envs, 7)).copy()
        root_pose_np[:, :3] += self.env_origins
        root_pose = _to_tensor(self.torch, root_pose_np, self.device)
        root_vel = self.torch.zeros(
            (self.num_envs, 6), dtype=self.torch.float32, device=self.device
        )
        native_pos = np.zeros((self.num_envs, self.num_dof), dtype=np.float32)
        native_pos[:, self.native_joint_for_contract] = qpos[7:][None, :]
        joint_pos = _to_tensor(self.torch, native_pos, self.device)
        joint_vel = self.torch.zeros_like(joint_pos)
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        # UniLab's root state is the link-frame state.  IsaacLab's similarly
        # named ``write_root_velocity_to_sim`` targets the COM frame, so use
        # the explicit link writer here.
        self.robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.reset(env_ids)
        self.robot.update(self.sim_dt)

    # ------------------------------------------------------------------
    # Native rendering (cold setup + eval/play commands)
    # ------------------------------------------------------------------

    def _require_render_mode(self, expected: str) -> None:
        if self.render_mode != expected:
            raise RuntimeError(
                "isaacsim renderer request is incompatible with the worker startup mode: "
                f"worker={self.render_mode!r}, requested={expected!r}"
            )

    def _camera_view(self) -> tuple[Any, Any]:
        """Return batched eye/target tensors for the spherical tracking view."""
        if self.robot is None:
            raise RuntimeError("isaacsim camera requested before articulation initialization")
        root_pos = self.robot.data.root_pos_w
        if tuple(root_pos.shape) != (self.num_envs, 3):
            raise RuntimeError(
                f"IsaacLab root positions have shape {root_pos.shape}; expected "
                f"({self.num_envs}, 3) for camera tracking"
            )
        elevation = math.radians(self.camera_elevation_deg)
        azimuth = math.radians(self.camera_azimuth_deg)
        offset = self.camera_distance * np.asarray(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float32,
        )
        offset_tensor = _to_tensor(self.torch, offset, self.device)
        targets = root_pos.clone()
        # Aim a little above the pelvis so playback is closer to eye level
        # instead of looking up from below.  Keeping the target above the root
        # also leaves enough vertical margin to keep the feet in frame.
        targets[:, 2] += 0.30
        eyes = targets + offset_tensor[None, :]
        return eyes, targets

    def _set_capture_camera(self) -> None:
        if self.camera is None:
            raise RuntimeError(
                "isaacsim capture camera is unavailable; worker was not started in record mode"
            )
        eyes, targets = self._camera_view()
        self.camera.set_world_poses_from_view(eyes[0:1], targets[0:1])

    @staticmethod
    def _app_is_running(app: Any) -> bool:
        """Read the documented SimulationApp lifecycle state."""
        try:
            return bool(app.is_running()) and not bool(app.is_exiting())
        except Exception:
            # A closed Kit app may invalidate the Python proxy before the
            # status methods can be queried. Treat that as a closed window.
            return False

    def init_renderer(self, payload: dict[str, Any]) -> dict[str, Any]:
        headless = bool(payload.get("headless", False))
        capture = bool(payload.get("capture", False))
        requested = "record" if (headless or capture) else "interactive"
        self._require_render_mode(requested)
        width = int(payload.get("width", self.render_width))
        height = int(payload.get("height", self.render_height))
        if width != self.render_width or height != self.render_height:
            raise ValueError(
                "isaacsim renderer dimensions differ from INIT: "
                f"requested={width}x{height}, configured={self.render_width}x{self.render_height}"
            )
        camera = payload.get("camera") or {}
        self.camera_distance = float(camera.get("distance", 2.0))
        self.camera_elevation_deg = float(camera.get("elevation_deg", 20.0))
        self.camera_azimuth_deg = float(camera.get("azimuth_deg", 90.0))
        if (
            not np.isfinite(
                [self.camera_distance, self.camera_elevation_deg, self.camera_azimuth_deg]
            ).all()
            or self.camera_distance <= 0.0
        ):
            raise ValueError(
                "isaacsim camera distance/elevation/azimuth must be finite and distance > 0"
            )

        if requested == "record":
            if not capture:
                raise RuntimeError("isaacsim record renderer requires capture=true")
            self._set_capture_camera()
            # Warm up Hydra/Replicator once on the cold path. Camera buffers
            # are then ready for the first playback frame.
            self.sim.render()
            self.camera.update(self.sim_dt, force_recompute=True)
            return {"viewer": False, "capture": True}

        if headless or capture:
            raise RuntimeError(
                "isaacsim interactive renderer cannot be headless or capture-enabled"
            )
        # Leave the Kit viewport camera under user control.  The interactive
        # viewer must not be re-aimed at the robot during startup or playback.
        self.sim.render()
        return {"viewer": self._app_is_running(self.simulation_app), "capture": False}

    def render_frame(self) -> dict[str, Any]:
        self._require_render_mode("interactive")
        if not self._app_is_running(self.simulation_app):
            return {"closed": True}
        self.sim.render()
        return {"closed": not self._app_is_running(self.simulation_app)}

    def capture_frame(self) -> dict[str, Any]:
        self._require_render_mode("record")
        if self.camera is None:
            raise RuntimeError(
                "isaacsim capture camera is not initialized; call INIT_RENDERER first"
            )
        self._set_capture_camera()
        self.sim.render()
        self.camera.update(self.sim_dt, force_recompute=True)
        output = self.camera.data.output
        if not isinstance(output, dict) or "rgb" not in output:
            raise RuntimeError(
                "IsaacSim camera did not return an rgb output; "
                f"available={list(output) if isinstance(output, dict) else output!r}"
            )
        image = output["rgb"]
        if not isinstance(image, self.torch.Tensor):
            raise RuntimeError("IsaacSim camera rgb output is not a torch tensor")
        frame = np.asarray(image[0].detach().cpu().numpy())
        if frame.ndim != 3 or frame.shape != (self.render_height, self.render_width, 3):
            raise RuntimeError(
                "IsaacSim camera rgb output has invalid shape: "
                f"got {frame.shape}, expected {(self.render_height, self.render_width, 3)}"
            )
        if frame.dtype != np.uint8:
            # IsaacLab's RGB annotator is uint8 by contract. Refuse lossy
            # coercion when an IsaacSim release changes that surface.
            raise RuntimeError(
                f"IsaacSim camera rgb output has dtype {frame.dtype}, expected uint8"
            )
        frame = np.ascontiguousarray(frame)
        if frame.size == 0 or int(np.ptp(frame)) == 0:
            raise RuntimeError("IsaacSim camera returned an empty or uniform RGB frame")
        return {
            "frame": frame,
            "width": self.render_width,
            "height": self.render_height,
        }

    def shutdown(self) -> None:
        self.camera = None
        for handle in self._shm_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._shm_handles = []
        if self.simulation_app is not None:
            try:
                self.simulation_app.close()
            except Exception:
                pass
            self.simulation_app = None


def _dispatch(ctx: Any, protocol: Any, cmd: str, payload: Any) -> tuple[str, Any]:
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
        return protocol.CMD_READY, ctx.reset_entities(payload)
    if cmd == protocol.CMD_REFRESH:
        ctx.refresh_state_slots()
        return protocol.CMD_READY, None
    if cmd == protocol.CMD_GET_META:
        return protocol.CMD_META, ctx.get_meta()
    if cmd == protocol.CMD_INIT_RENDERER:
        return protocol.CMD_META, ctx.init_renderer(payload or {})
    if cmd == protocol.CMD_RENDER_FRAME:
        return protocol.CMD_META, ctx.render_frame()
    if cmd == protocol.CMD_CAPTURE_FRAME:
        return protocol.CMD_META, ctx.capture_frame()
    raise NotImplementedError(f"isaacsim worker command {cmd!r} is unsupported")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)
    protocol = _load_protocol(args.protocol)
    scene_path = os.path.join(os.path.dirname(__file__), "scene_worker.py")
    spec = importlib.util.spec_from_file_location("unisim_isaacsim_scene", scene_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load IsaacSim scene worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx: Any = module.SceneWorkerContext(protocol, _WorkerContext(protocol))

    # Kit and extension startup can write banners to fd 1.  Preserve a private
    # protocol fd and route all incidental output to stderr before INIT.
    protocol_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    stdin = sys.stdin.buffer
    stdout = protocol_out
    while True:
        try:
            message = protocol.recv_message(stdin)
        except (EOFError, protocol.WorkerDisconnectedError):
            ctx.shutdown()
            return 0
        cmd = message["cmd"]
        if cmd == protocol.CMD_SHUTDOWN:
            try:
                ctx.shutdown()
            finally:
                protocol.send_message(stdout, protocol.CMD_READY)
            return 0
        try:
            reply_cmd, reply_payload = _dispatch(ctx, protocol, cmd, message.get("payload"))
        except Exception as exc:  # noqa: BLE001 - every worker error crosses the wire
            error = protocol.serialize_exception(exc)
            if getattr(ctx, "faulted", False):
                error["faulted"] = True
            protocol.send_message(stdout, protocol.CMD_ERROR, error)
            continue
        protocol.send_message(stdout, reply_cmd, reply_payload)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
