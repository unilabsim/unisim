"""Profile hidden host transfers in tensor runtimes.

This is an M9 diagnostic, not an RL throughput benchmark.  It constructs one
small real CUDA articulation (or a caller-supplied model), warms its tensor
step/reset path, and separately means wall time and profiles CUDA memcpy
activity.  Profiling changes runtime overhead, so timing and transfer evidence
come from independent loops.  Genesis and Newton use device-resident phases;
SuperDex instead profiles each packed host-bridge semantic boundary.

Genesis example:

.. code-block:: console

   PYTHONPATH=src uv run --no-project --with genesis-world==1.3.3 --with torch \\
     python scripts/benchmarks/m9_tensor_runtime_profile.py --backend genesis

Newton example:

   PYTHONPATH=src:$PWD/../UniLab/.venv/lib/python3.13/site-packages \\
   uv run --project ../UniLab --no-sync \\
    python scripts/benchmarks/m9_tensor_runtime_profile.py --backend newton

SuperDex example:

.. code-block:: console

   PYTHONPATH=src:$PWD/../UniLab/.venv/lib/python3.13/site-packages \\
     uv run --project ../UniLab --no-sync \\
     python scripts/benchmarks/m9_tensor_runtime_profile.py --backend superdex \\
       --model-file ../UniLab/src/unilab/assets/robots/g1/scene_flat.xml \\
       --sensor-name pelvis_local_linvel
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from unisim.backend.base import TensorExecution, TensorIOSpec
from unisim.scene import SceneCfg

ROOT = Path(__file__).resolve().parents[2]
_MODEL = """<mujoco model="unisim-m9-tensor-profile">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.5">
      <joint name="root" type="free"/>
      <geom name="base_geom" type="sphere" size="0.08" mass="1"/>
      <body name="arm" pos="0 0 0.12">
        <joint name="hinge" axis="0 1 0"/>
        <geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 0.2"
              size="0.03" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="hinge_motor" joint="hinge" kp="10" kv="1"/></actuator>
</mujoco>
"""
_DISTRIBUTIONS = {
    "genesis": ("genesis-world", "torch"),
    "newton": ("newton", "mujoco-warp", "mujoco", "warp-lang", "torch"),
    "superdex": ("superdex-physics-uni", "superdex-robotics-uni", "mujoco", "torch"),
}
_DEVICE_RESIDENT_PHASES = ("step", "reset", "sensors")
_SUPERDEX_PHASES = (
    "compile_host_bridge_io",
    "write_control",
    "step",
    "read_state_sensors",
    "apply_reset",
    "read_selected_state_sensors",
)


def _write_default_model(directory: Path) -> Path:
    model_file = directory / "model.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    return model_file


def _default_phases(backend: str) -> tuple[str, ...]:
    if backend == "superdex":
        return _SUPERDEX_PHASES
    return _DEVICE_RESIDENT_PHASES[:2]


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions(backend: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in _DISTRIBUTIONS[backend]:
        try:
            result[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result[distribution] = "missing"
    return result


def _tensor_runtime_diagnostics(backend: Any) -> dict[str, dict[str, bool | str | None]]:
    return {
        name: {
            "requested": diagnostic.requested,
            "enabled": diagnostic.enabled,
            "disable_reason": diagnostic.disable_reason,
        }
        for name, diagnostic in backend.get_tensor_runtime_diagnostics().items()
    }


def _cuda_graph_replay_enabled(backend: Any) -> bool:
    diagnostic = backend.get_tensor_runtime_diagnostics().get("cuda_graph")
    return diagnostic is not None and diagnostic.enabled


def _make_backend(
    name: str,
    model_file: Path,
    num_envs: int,
    *,
    sim_dt: float,
    integrator: str | None = None,
    base_name: str | None = None,
    use_cuda_graph: bool = False,
) -> Any:
    if name == "genesis":
        from unisim.backend.genesis.backend import GenesisBackend

        return GenesisBackend(
            SceneCfg(model_file=str(model_file)),
            num_envs=num_envs,
            sim_dt=sim_dt,
            integrator=integrator,
            base_name=base_name,
        )

    if name == "superdex":
        from unisim.backend.superdex.backend import SuperDexBackend

        return SuperDexBackend(
            SceneCfg(model_file=str(model_file)),
            num_envs=num_envs,
            sim_dt=sim_dt,
        )

    from unisim.backend.newton.backend import NewtonBackend
    from unisim.backend.newton.dependencies import load_newton_dependencies

    dependencies = load_newton_dependencies()
    dependencies.warp.init()
    device = dependencies.warp.get_device()
    if not bool(device.is_cuda):
        raise RuntimeError(f"Newton tensor profiling requires CUDA Warp, found {device}")
    return NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=num_envs,
        sim_dt=sim_dt,
        device=str(device),
        capacity_check_steps=1,
        use_cuda_graph=use_cuda_graph,
    )


def _mean_ms(operation: Callable[[], Any], iterations: int) -> dict[str, float]:
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _trace_summary(profiler: Any) -> dict[str, Any]:
    categories = {
        "h2d": "Memcpy HtoD",
        "dtoh": "Memcpy DtoH",
        "dtoh_pinned": "Memcpy DtoH (Device -> Pinned)",
        "dtoh_pageable": "Memcpy DtoH (Device -> Pageable)",
        "h2d_pageable": "Memcpy HtoD (Pageable -> Device)",
        "d2d": "Memcpy DtoD",
    }
    result: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(suffix="-trace") as trace_directory:
        trace_path = Path(trace_directory) / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = trace.get("traceEvents", [])
    for label, name_fragment in categories.items():
        matches = [event for event in events if name_fragment in event.get("name", "")]
        result[f"{label}_transfers"] = len(matches)
        result[f"{label}_bytes"] = sum(
            int(event.get("args", {}).get("bytes", 0)) for event in matches
        )

    keys = profiler.key_averages()
    for event_name in ("cudaMemcpyAsync", "cudaStreamSynchronize", "cudaDeviceSynchronize"):
        matches = [event for event in keys if event.key == event_name]
        result[event_name] = sum(int(event.count) for event in matches)
    return result


def _scalar_reads(operation: Callable[[], Any]) -> tuple[dict[str, int], Any]:
    original_item = torch.Tensor.item
    original_tolist = torch.Tensor.tolist
    counts = {"tensor_item_calls": 0, "tensor_tolist_calls": 0}
    counting_nested_call = False

    def counted_item(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal counting_nested_call
        outer_call = not counting_nested_call
        if outer_call:
            counts["tensor_item_calls"] += 1
            counting_nested_call = True
        try:
            return original_item(self, *args, **kwargs)
        finally:
            if outer_call:
                counting_nested_call = False

    def counted_tolist(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal counting_nested_call
        outer_call = not counting_nested_call
        if outer_call:
            counts["tensor_tolist_calls"] += 1
            counting_nested_call = True
        try:
            return original_tolist(self, *args, **kwargs)
        finally:
            if outer_call:
                counting_nested_call = False

    torch.Tensor.item = counted_item  # type: ignore[method-assign]
    torch.Tensor.tolist = counted_tolist  # type: ignore[method-assign]
    try:
        value = operation()
    finally:
        torch.Tensor.item = original_item  # type: ignore[method-assign]
        torch.Tensor.tolist = original_tolist  # type: ignore[method-assign]
    return counts, value


def _profile(
    operation: Callable[[], Any], iterations: int
) -> tuple[dict[str, Any], dict[str, int]]:
    def repeated() -> None:
        for _ in range(iterations):
            operation()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        counts, _ = _scalar_reads(repeated)
    return _trace_summary(profiler), counts


def _record_semantic_transfer(
    operation: Callable[[], Any], stats: Callable[[], dict[str, int]]
) -> tuple[dict[str, Any], dict[str, int]]:
    before = stats()
    value = operation()
    after = stats()
    return value, {name: after[name] - before[name] for name in after}


def _profile_semantic_phase(
    operation: Callable[[], Any], stats: Callable[[], dict[str, int]], iterations: int
) -> tuple[dict[str, Any], dict[str, int], list[dict[str, int]]]:
    """Profile one semantic operation and capture its transfer-counter deltas."""

    semantic_deltas: list[dict[str, int]] = []

    def measured() -> Any:
        _, delta = _record_semantic_transfer(operation, stats)
        semantic_deltas.append(delta)

    cuda_profile, scalar_reads = _profile(measured, iterations)
    return cuda_profile, scalar_reads, semantic_deltas


def _semantic_phase_result(
    operation: Callable[[], Any], stats_owner: Any, iterations: int
) -> dict[str, Any]:
    def stats() -> dict[str, int]:
        return dict(stats_owner.transfer_stats)

    cuda_profile, scalar_reads, deltas = _profile_semantic_phase(operation, stats, iterations)
    return {
        **_mean_ms(operation, iterations),
        "cuda_profile": cuda_profile,
        "scalar_reads": scalar_reads,
        "semantic_transfer_deltas": deltas,
        "semantic_transfer_totals": {
            name: sum(delta[name] for delta in deltas) for name in next(iter(deltas), {})
        },
        "iterations": iterations,
    }


def _superdex_operations(
    backend: Any,
    plan: Any,
    *,
    ctrl: torch.Tensor,
    rows: torch.Tensor,
    reset_qpos: torch.Tensor,
    reset_qvel: torch.Tensor,
    sensor_names: tuple[str, ...],
) -> dict[str, Callable[[], Any]]:
    """Build isolated probes for each packed SuperDex semantic boundary."""

    spec = TensorIOSpec(
        state_fields=("qpos", "qvel"),
        sensor_names=sensor_names,
        device="cuda",
    )

    def compile_host_bridge_io() -> Any:
        compiled = backend.compile_host_bridge_io(spec)
        compiled.close()
        return None

    def packed_step() -> Any:
        result = plan.step()
        # A packed step consumes control readiness.  Profiling the solver phase
        # in isolation requires restoring only that gate; no tensor data is
        # copied or mutated by this diagnostic-only transition.
        plan._control_ready = True
        return result

    return {
        "compile_host_bridge_io": compile_host_bridge_io,
        "write_control": lambda: plan.write_control(ctrl),
        "step": packed_step,
        "read_state_sensors": plan.read_state_sensors,
        "apply_reset": lambda: plan.apply_reset(rows, reset_qpos, reset_qvel),
        "read_selected_state_sensors": plan.read_selected_state_sensors,
    }


def _run_superdex(args: argparse.Namespace, model_file: Path) -> dict[str, Any]:
    phases = tuple(args.phases)
    unknown = set(phases) - set(_SUPERDEX_PHASES)
    if unknown:
        raise ValueError(
            f"SuperDex supports phases {list(_SUPERDEX_PHASES)}, got {sorted(unknown)}"
        )
    if args.profile_stages:
        raise ValueError("--profile-stages is not supported by --backend superdex")

    profiler_environment = {
        key: os.environ.get(key)
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "KINETO_CONFIG",
            "KINETO_LOG_DONE",
            "PYTORCH_PROFILER_PROFILE_MEM",
        )
    }
    backend = _make_backend(
        "superdex",
        model_file,
        args.num_envs,
        sim_dt=args.sim_dt,
    )
    plan = None
    try:
        if backend.tensor_execution() is not TensorExecution.HOST_BRIDGE:
            raise RuntimeError(
                f"superdex did not expose HOST_BRIDGE tensors; got {backend.tensor_execution()}"
            )
        runtime_diagnostics = _tensor_runtime_diagnostics(backend)
        plan = backend.compile_host_bridge_io(
            TensorIOSpec(
                state_fields=("qpos", "qvel"),
                sensor_names=tuple(args.sensor_name),
                device="cuda",
            )
        )
        views = plan.read_state_sensors()
        device = views["qpos"].device
        if device.type != "cuda":
            raise RuntimeError(f"SuperDex packed state views are not CUDA-resident: {device}")

        ctrl = torch.zeros(
            (args.num_envs, backend.num_actuators), dtype=torch.float32, device=device
        )
        rows = torch.arange(args.reset_rows, dtype=torch.int64, device=device)
        reset_qpos = views["qpos"][: args.reset_rows].clone().contiguous()
        reset_qvel = views["qvel"][: args.reset_rows].clone().contiguous()
        operations = _superdex_operations(
            backend,
            plan,
            ctrl=ctrl,
            rows=rows,
            reset_qpos=reset_qpos,
            reset_qvel=reset_qvel,
            sensor_names=tuple(args.sensor_name),
        )

        # One setup cycle is always required so reset/read probes have valid
        # packed state even when the caller explicitly requests --warmup 0.
        for _ in range(args.warmup + 1):
            plan.write_control(ctrl)
            plan.step(args.nsteps)
            plan.read_state_sensors()
            plan.apply_reset(rows, reset_qpos, reset_qvel)
            plan.read_selected_state_sensors()

        phase_results: dict[str, Any] = {}
        for phase in phases:
            phase_results[phase] = _semantic_phase_result(operations[phase], plan, args.iterations)
    finally:
        if plan is not None:
            plan.close()
        backend.close()

    return {
        "schema": "unisim.m9.tensor-runtime-profile/2",
        "scope": (
            "custom or tiny built-in articulation; packed SuperDex host-bridge "
            "transfer attribution only, not cross-engine parity or RL throughput"
        ),
        "tensor_execution": TensorExecution.HOST_BRIDGE.value,
        "model_source": str(model_file),
        "nsteps_per_step_phase": args.nsteps,
        "profiler_environment": profiler_environment,
        "backend": "superdex",
        "tensor_runtime_diagnostics": runtime_diagnostics,
        "git_commit": _git(["git", "rev-parse", "HEAD"]),
        "git_status_short": _git(["git", "status", "--short"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
        "runtime_versions": _versions("superdex"),
        "num_envs": args.num_envs,
        "reset_rows": args.reset_rows,
        "sensor_names": tuple(args.sensor_name),
        "packet_row_width": sum(int(value.shape[1]) for value in views.values()),
        "setup_iterations": 1,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "phases": phase_results,
    }


def _stage_operations(
    backend: Any,
    phase: str,
    *,
    ctrl: torch.Tensor,
    rows: torch.Tensor,
    reset_qpos: torch.Tensor,
    reset_qvel: torch.Tensor,
    sensor_names: tuple[str, ...] = (),
) -> dict[str, Callable[[], Any]]:
    """Build isolated attribution probes around adapter and public runtime calls."""

    if backend.backend_type == "genesis":
        if phase == "step":
            return {
                "adapter_control": lambda: backend._entity.control_dofs_position(
                    ctrl, dofs_idx_local=backend._actuated_dofs
                ),
                "vendor_scene_step": lambda: backend._scene.step(),
                "adapter_state_refresh": lambda: backend._tensor_refresh_state(),
            }
        if phase == "sensors":
            prefixes = {
                "track_pos_w_",
                "track_quat_w_",
                "track_linvel_w_",
                "track_angvel_w_",
            }
            tracked_body_names = tuple(
                dict.fromkeys(
                    name[len(prefix) :]
                    for name in sensor_names
                    for prefix in prefixes
                    if name.startswith(prefix)
                )
            )
            named_sensor_names = tuple(
                name for name in sensor_names if not name.startswith(tuple(prefixes))
            )
            for name in sensor_names:
                backend.get_sensor_view(name)
            stacked_outputs = {
                prefix: torch.stack(
                    tuple(
                        backend.get_sensor_view(f"{prefix}_{suffix}")
                        for suffix in tracked_body_names
                    ),
                    dim=1,
                )
                for prefix in (
                    "track_pos_w",
                    "track_quat_w",
                    "track_linvel_w",
                    "track_angvel_w",
                )
                if tracked_body_names
            }

            def repeated_sensor_reads() -> None:
                for name in sensor_names:
                    backend.get_sensor_view(name)

            def repeated_sensor_reads_and_stack() -> None:
                for name in named_sensor_names:
                    backend.get_sensor_view(name)
                for prefix, output in stacked_outputs.items():
                    torch.stack(
                        tuple(
                            backend.get_sensor_view(f"{prefix}_{body_name}")
                            for body_name in tracked_body_names
                        ),
                        dim=1,
                        out=output,
                    )

            return {
                "state_and_body_publication": lambda: backend._tensor_refresh_state(force=True),
                "repeated_sensor_requests": repeated_sensor_reads,
                "repeated_sensor_reads_and_stack": repeated_sensor_reads_and_stack,
            }
        full_qpos = reset_qpos.new_zeros((backend.num_envs, reset_qpos.shape[1]))
        full_qvel = reset_qvel.new_zeros((backend.num_envs, reset_qvel.shape[1]))
        full_qpos[rows] = reset_qpos
        full_qvel[rows] = reset_qvel
        mask = torch.zeros((backend.num_envs,), dtype=torch.bool, device=rows.device)
        mask.scatter_(0, rows, torch.ones_like(rows, dtype=torch.bool))
        return {
            "adapter_row_validation": lambda: backend._validate_torch_rows(rows),
            "vendor_set_qpos": lambda: backend._entity.set_qpos(
                full_qpos, envs_idx=mask, zero_velocity=False
            ),
            "vendor_set_dofs_velocity": lambda: backend._entity.set_dofs_velocity(
                full_qvel, envs_idx=mask
            ),
            "adapter_state_refresh": lambda: backend._tensor_refresh_state(),
        }

    if phase == "step":
        solver_step = (
            backend._replay_cuda_graph_substep
            if _cuda_graph_replay_enabled(backend)
            else backend._physics_substep_current_control
        )
        return {
            "adapter_control_upload": lambda: backend._tensor_upload_control(ctrl),
            "vendor_solver_step": solver_step,
            "adapter_state_refresh": lambda: backend._tensor_refresh_state(),
        }

    views = backend.get_state_views()
    _, _, raw_qpos, raw_qvel, mask, solver_mask = backend._prepare_tensor_reset(
        rows, reset_qpos, reset_qvel
    )
    return {
        "adapter_row_validation": lambda: backend._validate_torch_reset_rows(rows),
        "adapter_state_prepare": lambda: backend._prepare_tensor_reset(
            rows, reset_qpos, reset_qvel
        ),
        "vendor_set_dof_positions": lambda: backend._view.set_dof_positions(
            backend._state,
            backend._deps.warp.from_torch(raw_qpos.unsqueeze(1).contiguous()),
            mask=mask,
        ),
        "vendor_set_dof_velocities": lambda: backend._view.set_dof_velocities(
            backend._state,
            backend._deps.warp.from_torch(raw_qvel.unsqueeze(1).contiguous()),
            mask=mask,
        ),
        "vendor_eval_fk": lambda: backend._deps.newton.eval_fk(
            backend._model,
            backend._state.joint_q,
            backend._state.joint_qd,
            backend._state,
        ),
        "vendor_solver_reset": lambda: backend._solver.reset(
            backend._state, backend._deps.warp.from_torch(solver_mask), flags=0
        ),
        "adapter_state_publish": lambda: backend._publish_tensor_reset(
            views["qpos"].clone(), views["qvel"].clone()
        ),
    }


def _profile_stages(
    backend: Any,
    phase: str,
    *,
    ctrl: torch.Tensor,
    rows: torch.Tensor,
    reset_qpos: torch.Tensor,
    reset_qvel: torch.Tensor,
    iterations: int,
    sensor_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if phase == "sensors":
        if backend.backend_type != "genesis":
            raise ValueError("sensor attribution stages require --backend genesis")
    for label, operation in _stage_operations(
        backend,
        phase,
        ctrl=ctrl,
        rows=rows,
        reset_qpos=reset_qpos,
        reset_qvel=reset_qvel,
        sensor_names=sensor_names,
    ).items():
        transfers, scalar_reads = _profile(operation, iterations)
        result[label] = {
            **_mean_ms(operation, iterations),
            "cuda_profile": transfers,
            "scalar_reads": scalar_reads,
        }
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.phases is None:
        args.phases = list(_default_phases(args.backend))
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("--iterations must be positive and --warmup must be nonnegative")
    if args.reset_rows < 1 or args.reset_rows > args.num_envs:
        raise ValueError("--reset-rows must be in [1, --num-envs]")
    if args.sim_dt <= 0.0:
        raise ValueError("--sim-dt must be positive")
    if args.nsteps <= 0:
        raise ValueError("--nsteps must be positive")
    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            f"{args.backend.capitalize()} tensor profiling requires CUDA Torch; "
            "torch.cuda.is_available() returned False"
        )
    if args.backend != "newton" and args.use_cuda_graph:
        raise ValueError("--use-cuda-graph is only supported by --backend newton")
    if args.backend == "superdex":
        with tempfile.TemporaryDirectory(prefix="unisim-m9-tensor-") as directory:
            model_file = (
                args.model_file.resolve()
                if args.model_file is not None
                else _write_default_model(Path(directory))
            )
            return _run_superdex(args, model_file)

    profiler_environment = {
        key: os.environ.get(key)
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "KINETO_CONFIG",
            "KINETO_LOG_DONE",
            "PYTORCH_PROFILER_PROFILE_MEM",
            "UNISIM_ISAAC_WORKER_PROFILE_TRACE",
            "UNISIM_ISAAC_WORKER_PROFILE_START_COMMAND",
            "UNISIM_ISAAC_WORKER_PROFILE_STOP_COMMAND",
        )
    }
    with tempfile.TemporaryDirectory(prefix="unisim-m9-tensor-") as directory:
        model_file = (
            args.model_file.resolve()
            if args.model_file is not None
            else _write_default_model(Path(directory))
        )
        backend = _make_backend(
            args.backend,
            model_file,
            args.num_envs,
            sim_dt=args.sim_dt,
            integrator=args.genesis_integrator,
            base_name=args.genesis_base_name,
            use_cuda_graph=args.use_cuda_graph,
        )
        try:
            if backend.tensor_execution() is not TensorExecution.DEVICE_RESIDENT:
                raise RuntimeError(
                    f"{args.backend} did not expose DEVICE_RESIDENT tensors; "
                    f"got {backend.tensor_execution()}"
                )
            backend.materialize()
            runtime_diagnostics = _tensor_runtime_diagnostics(backend)
            views = backend.get_state_views()
            device = views["qpos"].device
            if device.type != "cuda":
                raise RuntimeError(f"{args.backend} state views are not CUDA-resident: {device}")
            ctrl = torch.zeros(
                (args.num_envs, backend.num_actuators), dtype=torch.float32, device=device
            )
            rows = torch.arange(args.reset_rows, dtype=torch.int64, device=device)
            reset_qpos = views["qpos"][: args.reset_rows].contiguous()
            reset_qvel = views["qvel"][: args.reset_rows].contiguous()

            def step() -> Any:
                return backend.step_tensor(ctrl, nsteps=args.nsteps)

            def reset() -> Any:
                return backend.set_state_tensor(rows, reset_qpos, reset_qvel)

            for _ in range(args.warmup):
                step()
                reset()

            phase_results: dict[str, Any] = {}
            if "step" in args.phases:
                step_profile, step_scalars = _profile(step, args.iterations)
                phase_results["step"] = {
                    **_mean_ms(step, args.iterations),
                    "cuda_profile": step_profile,
                    "scalar_reads": step_scalars,
                }
            if "reset" in args.phases:
                reset_profile, reset_scalars = _profile(reset, args.iterations)
                phase_results["reset"] = {
                    **_mean_ms(reset, args.iterations),
                    "cuda_profile": reset_profile,
                    "scalar_reads": reset_scalars,
                }
            if "sensors" in args.phases:
                if not args.sensor_name:
                    raise ValueError("--phases sensors requires at least one --sensor-name")
                for name in args.sensor_name:
                    backend.get_sensor_view(name)
            stage_profiles: dict[str, dict[str, Any]] = {}
            if args.profile_stages:
                for phase in args.phases:
                    phase_stages = _profile_stages(
                        backend,
                        phase,
                        ctrl=ctrl,
                        rows=rows,
                        reset_qpos=reset_qpos,
                        reset_qvel=reset_qvel,
                        iterations=args.iterations,
                        sensor_names=tuple(args.sensor_name),
                    )
                    if phase_stages:
                        stage_profiles[phase] = phase_stages
        finally:
            backend.close()

    return {
        "schema": "unisim.m9.tensor-runtime-profile/2",
        "scope": (
            "custom or tiny built-in articulation; diagnostic transfer attribution "
            "only, not generalized-task parity or RL throughput"
        ),
        "model_source": str(model_file),
        "nsteps_per_step_phase": args.nsteps,
        "profiler_environment": profiler_environment,
        "backend": args.backend,
        **({"newton_use_cuda_graph": args.use_cuda_graph} if args.backend == "newton" else {}),
        "tensor_runtime_diagnostics": runtime_diagnostics,
        "git_commit": _git(["git", "rev-parse", "HEAD"]),
        "git_status_short": _git(["git", "status", "--short"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
        "runtime_versions": _versions(args.backend),
        "num_envs": args.num_envs,
        "reset_rows": args.reset_rows,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "phases": phase_results,
        **(
            {
                "stage_profiles": stage_profiles,
                "stage_scope": (
                    "separate attribution loops; useful for transfer ownership, "
                    "not for end-to-end timing"
                ),
            }
            if args.profile_stages
            else {}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("genesis", "newton", "superdex"), required=True)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--reset-rows", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--phases",
        choices=(*_DEVICE_RESIDENT_PHASES, *_SUPERDEX_PHASES),
        nargs="+",
        default=None,
        help=(
            "device-resident backends default to 'step reset'; SuperDex defaults "
            "to all six packed host-bridge semantic phases"
        ),
    )
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--genesis-integrator")
    parser.add_argument("--genesis-base-name")
    parser.add_argument("--nsteps", type=int, default=1)
    parser.add_argument("--sensor-name", action="append", default=[])
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="profile adapter/public-runtime stages separately for transfer attribution",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="profile Newton CUDA-graph replay (the G1 owner default) instead of eager execution",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phases is None:
        args.phases = list(_default_phases(args.backend))
    result = _run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
