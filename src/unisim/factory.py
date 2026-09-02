"""Lazy backend factory for optional engine adapters."""

from __future__ import annotations

import warnings
from typing import Any

from .adapters import adapter_spec
from .contract import BackendError, SimBackend


def create_backend(
    backend_type: str,
    scene: Any | None = None,
    num_envs: int = 1,
    sim_dt: float = 0.01,
    **kwargs: Any,
) -> SimBackend:
    """Construct an optional backend without importing engine SDKs eagerly."""
    body_state_required = kwargs.pop("body_state_required", False)
    if not isinstance(body_state_required, bool):
        raise TypeError("body_state_required must be bool")
    if backend_type == "fake":
        from .fake import FakeBackend

        return FakeBackend(num_envs=num_envs, **kwargs)
    try:
        spec = adapter_spec(backend_type)
    except KeyError:
        raise ValueError(f"unknown UniSim backend: {backend_type!r}") from None
    if spec.status != "available":
        raise BackendError(f"backend '{backend_type}' is not currently available")
    if scene is None and backend_type not in {"isaacgym", "isaacsim"}:
        raise ValueError(f"backend '{backend_type}' requires a SceneCfg")

    position_actuator_gains = kwargs.pop("position_actuator_gains", None)
    motrix_max_iterations = kwargs.pop("motrix_max_iterations", None)
    post_step_forward_sensor = kwargs.pop("post_step_forward_sensor", None)
    iterations = kwargs.pop("iterations", None)
    chunk_size = kwargs.pop("chunk_size", None)
    adaptive_chunk_size = kwargs.pop("adaptive_chunk_size", False)
    cpu_ids = kwargs.pop("cpu_ids", None)
    bench_nsteps = kwargs.pop("bench_nsteps", 1)
    mjwarp_nconmax = kwargs.pop("mjwarp_nconmax", None)
    mjwarp_njmax = kwargs.pop("mjwarp_njmax", None)
    drake_backend_mode = kwargs.pop("drake_backend_mode", "batch")
    drake_nthread = kwargs.pop("drake_nthread", None)
    isaacgym_device_id = kwargs.pop("isaacgym_device_id", None)
    isaacgym_worker_timeout_s = kwargs.pop("isaacgym_worker_timeout_s", None)
    genesis_integrator = kwargs.pop("genesis_integrator", None)
    genesis_constraint_solver = kwargs.pop("genesis_constraint_solver", None)
    genesis_friction_cone = kwargs.pop("genesis_friction_cone", None)
    genesis_solver_iterations = kwargs.pop("genesis_solver_iterations", None)
    isaacsim_device_id = kwargs.pop("isaacsim_device_id", None)
    isaacsim_worker_timeout_s = kwargs.pop("isaacsim_worker_timeout_s", None)
    isaacsim_render_mode = kwargs.pop("isaacsim_render_mode", None)
    isaacsim_render_width = kwargs.pop("isaacsim_render_width", 1280)
    isaacsim_render_height = kwargs.pop("isaacsim_render_height", 720)

    if backend_type == "mujoco":
        from .backend.mujoco.backend import MuJoCoBackend

        if body_state_required:
            kwargs["add_body_sensors"] = True
        if position_actuator_gains is not None:
            kwargs["position_actuator_gains"] = position_actuator_gains
        if post_step_forward_sensor is not None:
            kwargs["post_step_forward_sensor"] = post_step_forward_sensor
        kwargs["iterations"] = iterations
        kwargs["chunk_size"] = chunk_size
        kwargs["adaptive_chunk_size"] = adaptive_chunk_size
        kwargs["cpu_ids"] = cpu_ids
        kwargs["bench_nsteps"] = bench_nsteps
        return MuJoCoBackend(scene, num_envs, sim_dt, **kwargs)
    if backend_type == "motrix":
        from .backend.motrix.backend import MOTRIX_AVAILABLE, MotrixBackend

        if not MOTRIX_AVAILABLE:
            raise ImportError("MotrixSim not available, install motrixsim package")
        if body_state_required:
            kwargs["add_body_sensors"] = True
        if motrix_max_iterations is not None:
            kwargs["max_iterations"] = motrix_max_iterations
        return MotrixBackend(scene, num_envs, sim_dt, **kwargs)
    if backend_type == "drake":
        from .backend.drake.backend import DrakeBackend

        kwargs.pop("base_name", None)
        kwargs.pop("push_body_name", None)
        kwargs.pop("add_body_sensors", None)
        kwargs["drake_backend_mode"] = drake_backend_mode
        if drake_nthread is not None:
            kwargs["nthread"] = drake_nthread
        return DrakeBackend(scene, num_envs, sim_dt, **kwargs)
    if backend_type == "mjwarp":
        from .backend.mjwarp.backend import MjwarpBackend

        if body_state_required:
            kwargs["add_body_sensors"] = True
        if position_actuator_gains is not None:
            raise ValueError(
                "mjwarp does not accept position_actuator_gains in the host compatibility "
                "profile; configure the model on the cold path instead."
            )
        _warn_ignored_mujoco_options(
            "mjwarp",
            post_step_forward_sensor=post_step_forward_sensor,
            iterations=iterations,
            chunk_size=chunk_size,
            adaptive_chunk_size=adaptive_chunk_size,
            cpu_ids=cpu_ids,
            bench_nsteps=bench_nsteps,
        )
        kwargs["nconmax"] = mjwarp_nconmax
        kwargs["njmax"] = mjwarp_njmax
        return MjwarpBackend(scene, num_envs, sim_dt, **kwargs)
    if backend_type == "genesis":
        from .backend.genesis.backend import GenesisBackend

        kwargs.pop("add_body_sensors", None)
        if position_actuator_gains is not None:
            raise ValueError(
                "genesis imports the MJCF position-actuator gains directly; "
                "position_actuator_gains has no Genesis equivalent."
            )
        _warn_ignored_mujoco_options(
            "genesis",
            post_step_forward_sensor=post_step_forward_sensor,
            iterations=iterations,
            chunk_size=chunk_size,
            adaptive_chunk_size=adaptive_chunk_size,
            cpu_ids=cpu_ids,
            bench_nsteps=bench_nsteps,
        )
        kwargs["integrator"] = genesis_integrator
        kwargs["constraint_solver"] = genesis_constraint_solver
        kwargs["friction_cone"] = genesis_friction_cone
        kwargs["solver_iterations"] = genesis_solver_iterations
        return GenesisBackend(scene, num_envs, sim_dt, **kwargs)
    if backend_type == "isaacgym":
        if scene is None and "runtime" not in kwargs and "worker_command" not in kwargs:
            from .backend.isaacgym.dependencies import IsaacGymDependencyError

            raise IsaacGymDependencyError(
                "IsaacGym backend requires a SceneCfg plus a configured external worker."
            )
        from .backend.isaacgym.backend import IsaacGymBackend

        kwargs.pop("add_body_sensors", None)
        if position_actuator_gains is not None:
            raise ValueError(
                "isaacgym runs torque-mode dofs only; position_actuator_gains has no "
                "IsaacGym equivalent in the subprocess profile."
            )
        _warn_ignored_mujoco_options(
            "isaacgym",
            post_step_forward_sensor=post_step_forward_sensor,
            iterations=iterations,
            chunk_size=chunk_size,
            adaptive_chunk_size=adaptive_chunk_size,
            cpu_ids=cpu_ids,
            bench_nsteps=bench_nsteps,
        )
        if isaacgym_device_id is not None:
            kwargs["device_id"] = isaacgym_device_id
        if isaacgym_worker_timeout_s is not None:
            kwargs["worker_timeout_s"] = isaacgym_worker_timeout_s
        return IsaacGymBackend(scene, num_envs, sim_dt, **kwargs)
    if backend_type == "isaacsim":
        if scene is None and "runtime" not in kwargs and "worker_command" not in kwargs:
            from .backend.isaacsim.dependencies import IsaacSimDependencyError

            raise IsaacSimDependencyError(
                "IsaacSim backend requires a SceneCfg plus a configured external worker."
            )
        from .backend.isaacsim.backend import IsaacSimBackend

        direct_render_width = kwargs.pop("render_width", None)
        direct_render_height = kwargs.pop("render_height", None)
        kwargs.pop("add_body_sensors", None)
        if position_actuator_gains is not None:
            raise ValueError(
                "isaacsim uses IsaacLab implicit position actuators; configure gains in the "
                "scene/owner contract rather than position_actuator_gains."
            )
        _warn_ignored_mujoco_options(
            "isaacsim",
            post_step_forward_sensor=post_step_forward_sensor,
            iterations=iterations,
            chunk_size=chunk_size,
            adaptive_chunk_size=adaptive_chunk_size,
            cpu_ids=cpu_ids,
            bench_nsteps=bench_nsteps,
        )
        if isaacsim_device_id is not None:
            kwargs["device_id"] = isaacsim_device_id
        if isaacsim_worker_timeout_s is not None:
            kwargs["worker_timeout_s"] = isaacsim_worker_timeout_s
        if isaacsim_render_mode is not None:
            kwargs["render_mode"] = isaacsim_render_mode
        kwargs["render_width"] = (
            isaacsim_render_width if direct_render_width is None else direct_render_width
        )
        kwargs["render_height"] = (
            isaacsim_render_height if direct_render_height is None else direct_render_height
        )
        return IsaacSimBackend(scene, num_envs, sim_dt, **kwargs)
    # Every backend in the manifest has a concrete public adapter.  Optional
    # SDK/worker availability is diagnosed by that adapter at construction;
    # this branch is retained only as a guard for future manifest mistakes.
    raise ValueError(f"unknown UniSim backend: {backend_type!r}")


def _warn_ignored_mujoco_options(
    backend_type: str,
    *,
    post_step_forward_sensor: Any,
    iterations: Any,
    chunk_size: Any,
    adaptive_chunk_size: Any,
    cpu_ids: Any,
    bench_nsteps: Any,
) -> None:
    values = {
        key: value
        for key, value, default in (
            ("post_step_forward_sensor", post_step_forward_sensor, None),
            ("iterations", iterations, None),
            ("chunk_size", chunk_size, None),
            ("adaptive_chunk_size", adaptive_chunk_size, False),
            ("cpu_ids", cpu_ids, None),
            ("bench_nsteps", bench_nsteps, 1),
        )
        if value != default
    }
    if not values:
        return
    rendered = ", ".join(f"{key}={value!r}" for key, value in values.items())
    warnings.warn(
        f"{backend_type} ignores non-default MuJoCo-only backend options: {rendered}",
        UserWarning,
        stacklevel=3,
    )
