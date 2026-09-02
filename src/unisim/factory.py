"""Lazy backend factory for optional engine adapters."""

from __future__ import annotations

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
    # An injected runtime is a deliberate integration hook for worker-owned
    # runtimes and contract tests.  Production construction below always uses
    # the real adapter under ``unisim.backend.<name>``.
    runtime = kwargs.get("runtime")
    if runtime is not None and backend_type in {"drake", "mjwarp", "genesis"}:
        from .optional import RuntimeBackend

        options = dict(kwargs)
        options.pop("runtime", None)
        backend = RuntimeBackend(runtime=runtime, num_envs=num_envs, **options)
        backend.backend_type = backend_type
        return backend  # type: ignore[return-value]
    if scene is None and backend_type not in {"isaacgym", "isaacsim"}:
        raise ValueError(f"backend '{backend_type}' requires a SceneCfg")
    if backend_type == "mujoco":
        from .backend.mujoco.backend import MuJoCoBackend

        if body_state_required:
            kwargs.setdefault("add_body_sensors", True)
        return MuJoCoBackend(scene, num_envs, sim_dt, **kwargs)
    if backend_type == "motrix":
        from .backend.motrix.backend import MotrixBackend

        if body_state_required:
            kwargs.setdefault("add_body_sensors", True)
        if "motrix_max_iterations" in kwargs:
            kwargs["max_iterations"] = kwargs.pop("motrix_max_iterations")
        return MotrixBackend(scene, num_envs, sim_dt, **kwargs)
    if backend_type == "drake":
        from .backend.drake.backend import DrakeBackend

        if "drake_nthread" in kwargs:
            kwargs["nthread"] = kwargs.pop("drake_nthread")
        for key in (
            "base_name",
            "push_body_name",
            "add_body_sensors",
            "position_actuator_gains",
            "post_step_forward_sensor",
            "iterations",
            "chunk_size",
            "adaptive_chunk_size",
            "cpu_ids",
            "bench_nsteps",
            "motrix_max_iterations",
        ):
            kwargs.pop(key, None)
        return DrakeBackend(scene, num_envs, sim_dt, **kwargs)
    if backend_type == "mjwarp":
        from .backend.mjwarp.backend import MjwarpBackend

        if "mjwarp_nconmax" in kwargs:
            kwargs["nconmax"] = kwargs.pop("mjwarp_nconmax")
        if "mjwarp_njmax" in kwargs:
            kwargs["njmax"] = kwargs.pop("mjwarp_njmax")
        for key in (
            "post_step_forward_sensor",
            "iterations",
            "chunk_size",
            "adaptive_chunk_size",
            "cpu_ids",
            "bench_nsteps",
            "motrix_max_iterations",
        ):
            kwargs.pop(key, None)
        if kwargs.pop("position_actuator_gains", None) is not None:
            raise ValueError("mjwarp does not accept position_actuator_gains")
        if body_state_required:
            kwargs.setdefault("add_body_sensors", True)
        return MjwarpBackend(scene, num_envs, sim_dt, **kwargs)
    if backend_type == "genesis":
        from .backend.genesis.backend import GenesisBackend

        mapping = {
            "genesis_integrator": "integrator",
            "genesis_constraint_solver": "constraint_solver",
            "genesis_friction_cone": "friction_cone",
            "genesis_solver_iterations": "solver_iterations",
        }
        for source, target in mapping.items():
            if source in kwargs:
                kwargs[target] = kwargs.pop(source)
        for key in (
            "post_step_forward_sensor",
            "iterations",
            "chunk_size",
            "adaptive_chunk_size",
            "cpu_ids",
            "bench_nsteps",
            "add_body_sensors",
            "motrix_max_iterations",
        ):
            kwargs.pop(key, None)
        if kwargs.pop("position_actuator_gains", None) is not None:
            raise ValueError("genesis does not accept position_actuator_gains")
        return GenesisBackend(scene, num_envs, sim_dt, **kwargs)
    if backend_type == "isaacgym":
        if scene is None and "runtime" not in kwargs and "worker_command" not in kwargs:
            from .backend.isaacgym.dependencies import IsaacGymDependencyError

            raise IsaacGymDependencyError(
                "IsaacGym backend requires a SceneCfg plus a configured external worker."
            )
        from .backend.isaacgym.backend import IsaacGymBackend

        if "isaacgym_device_id" in kwargs:
            kwargs["device_id"] = kwargs.pop("isaacgym_device_id")
        if "isaacgym_worker_timeout_s" in kwargs:
            kwargs["worker_timeout_s"] = kwargs.pop("isaacgym_worker_timeout_s")
        for key in (
            "add_body_sensors",
            "position_actuator_gains",
            "post_step_forward_sensor",
            "iterations",
            "chunk_size",
            "adaptive_chunk_size",
            "cpu_ids",
            "bench_nsteps",
            "motrix_max_iterations",
        ):
            kwargs.pop(key, None)
        return IsaacGymBackend(scene, num_envs, sim_dt, **kwargs)
    if backend_type == "isaacsim":
        if scene is None and "runtime" not in kwargs and "worker_command" not in kwargs:
            from .backend.isaacsim.dependencies import IsaacSimDependencyError

            raise IsaacSimDependencyError(
                "IsaacSim backend requires a SceneCfg plus a configured external worker."
            )
        from .backend.isaacsim.backend import IsaacSimBackend

        mapping = {
            "isaacsim_device_id": "device_id",
            "isaacsim_worker_timeout_s": "worker_timeout_s",
            "isaacsim_render_mode": "render_mode",
            "isaacsim_render_width": "render_width",
            "isaacsim_render_height": "render_height",
        }
        for source, target in mapping.items():
            if source in kwargs:
                kwargs[target] = kwargs.pop(source)
        for key in (
            "add_body_sensors",
            "position_actuator_gains",
            "post_step_forward_sensor",
            "iterations",
            "chunk_size",
            "adaptive_chunk_size",
            "cpu_ids",
            "bench_nsteps",
            "motrix_max_iterations",
        ):
            kwargs.pop(key, None)
        return IsaacSimBackend(scene, num_envs, sim_dt, **kwargs)
    try:
        spec = adapter_spec(backend_type)
    except KeyError:
        raise ValueError(f"unknown UniSim backend: {backend_type!r}") from None
    # Every backend in the manifest has a concrete public adapter.  Optional
    # SDK/worker availability is diagnosed by that adapter at construction;
    # this branch is retained only as a guard for future manifest mistakes.
    if spec.status != "available":
        raise BackendError(f"backend '{backend_type}' is not currently available")
    raise ValueError(f"unknown UniSim backend: {backend_type!r}")
