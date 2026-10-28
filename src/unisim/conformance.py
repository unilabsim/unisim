"""Reusable checks for adapter authors.

The helper intentionally tests only the public contract. Engine-specific
fixtures and expensive differential tests belong to adapter-owned suites.
"""

from __future__ import annotations

from os import PathLike

import numpy as np

from .contract import BackendCapability, SimBackend
from .dr.types import RESET_TERM_SPECS, TensorResetRandomizationPayload
from .errors import BackendError, UnsupportedCapabilityError
from .ray_query import (
    RayCaster,
    RayGeomType,
    RayPoseTensorView,
    RaySceneDescription,
    RayTraceFilter,
    RayTraceOutputs,
)

_RESET_DELTA_TERMS = frozenset(spec.name for spec in RESET_TERM_SPECS if spec.is_delta)
_RESET_TERM_TAIL_NDIMS = {spec.name: spec.tail_ndim for spec in RESET_TERM_SPECS}


def assert_ray_caster_conformance(caster: RayCaster) -> None:
    """Run cheap lifecycle/geometry checks against a fresh ray caster.

    The canonical scene is one infinite ground plane at ``z = 0`` owned by a
    single identity-posed body, so every caster must report the same analytic
    distances. The helper covers lifecycle ordering, capability-declared
    outputs, the shared and per-environment ray profiles, selected-row
    queries, and fail-closed rejection of undeclared requests.
    """
    capabilities = caster.get_ray_capabilities()
    assert capabilities.supports_host_readback, (
        "ray caster conformance requires supports_host_readback; device-only output "
        "is reserved for a future contract revision"
    )
    assert caster.num_envs > 0 and caster.num_rays > 0
    scene = RaySceneDescription(
        num_bodies=1,
        geom_types=(RayGeomType.PLANE,),
        geom_sizes=np.zeros((1, 3), dtype=np.float64),
        geom_local_pos=np.zeros((1, 3), dtype=np.float64),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.zeros(1, dtype=np.intp),
    )

    down_origin = np.zeros((caster.num_rays, 3), dtype=np.float64)
    down_origin[:, 2] = 1.0
    down_direction = np.tile(np.array([0.0, 0.0, -1.0]), (caster.num_rays, 1))

    def expect_raises(error_type: type[Exception], fn) -> None:
        try:
            fn()
        except error_type:
            return
        raise AssertionError(f"expected {error_type.__name__}")

    expect_raises(BackendError, lambda: caster.trace(down_origin, down_direction, 10.0))
    caster.materialize(scene)
    expect_raises(BackendError, lambda: caster.materialize(scene))

    result = caster.trace(down_origin, down_direction, 10.0)
    assert result.distance.shape == (caster.num_envs, caster.num_rays)
    assert result.hit.shape == (caster.num_envs, caster.num_rays)
    assert result.hit.all(), "downward rays must hit the ground plane"
    np.testing.assert_allclose(result.distance, 1.0, rtol=1e-6, atol=1e-9)

    miss = caster.trace(down_origin, -down_direction, 10.0)
    assert not miss.hit.any(), "upward rays must miss the ground plane"
    np.testing.assert_allclose(miss.distance, 10.0)

    if capabilities.supports_min_distance:
        inclusive = caster.trace(
            down_origin + np.array([0.0, 0.0, 1.0]),
            down_direction,
            10.0,
            min_distance=2.0,
        )
        assert inclusive.hit.all()
        np.testing.assert_allclose(inclusive.distance, 2.0, rtol=1e-6, atol=1e-9)
        rejected = caster.trace(down_origin, down_direction, 10.0, min_distance=1.5)
        assert not rejected.hit.any()
        np.testing.assert_allclose(rejected.distance, 10.0)
    else:
        expect_raises(
            UnsupportedCapabilityError,
            lambda: caster.trace(down_origin, down_direction, 10.0, min_distance=1.5),
        )

    if capabilities.supports_static_filter:
        allowed = caster.trace(
            down_origin, down_direction, 10.0, ray_filter=RayTraceFilter(allowed_geom_ids=(0,))
        )
        assert allowed.hit.all()
        excluded = caster.trace(
            down_origin,
            down_direction,
            10.0,
            ray_filter=RayTraceFilter(excluded_body_ids=(0,)),
        )
        assert not excluded.hit.any()
        np.testing.assert_allclose(excluded.distance, 10.0)
    else:
        expect_raises(
            UnsupportedCapabilityError,
            lambda: caster.trace(
                down_origin,
                down_direction,
                10.0,
                ray_filter=RayTraceFilter(allowed_geom_ids=(0,)),
            ),
        )

    selected = caster.trace(down_origin, down_direction, 10.0, env_ids=np.arange(1))
    assert selected.distance.shape == (1, caster.num_rays)

    if capabilities.supports_pose_sync:
        identity_pos = np.zeros((caster.num_envs, scene.num_bodies, 3), dtype=np.float64)
        identity_quat = np.zeros((caster.num_envs, scene.num_bodies, 4), dtype=np.float64)
        identity_quat[..., 0] = 1.0
        caster.update_pose(identity_pos, identity_quat)
        synced = caster.trace(down_origin, down_direction, 10.0)
        np.testing.assert_allclose(synced.distance, result.distance)
    else:
        expect_raises(
            UnsupportedCapabilityError,
            lambda: caster.update_pose(
                np.zeros((caster.num_envs, scene.num_bodies, 3)),
                np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (caster.num_envs, scene.num_bodies, 1)),
            ),
        )

    if capabilities.supports_per_env_rays:
        per_env = caster.trace(
            np.broadcast_to(down_origin, (caster.num_envs, caster.num_rays, 3)),
            np.broadcast_to(down_direction, (caster.num_envs, caster.num_rays, 3)),
            10.0,
        )
        np.testing.assert_allclose(per_env.distance, result.distance)
    else:
        expect_raises(
            UnsupportedCapabilityError,
            lambda: caster.trace(
                np.broadcast_to(down_origin, (caster.num_envs, caster.num_rays, 3)),
                np.broadcast_to(down_direction, (caster.num_envs, caster.num_rays, 3)),
                10.0,
            ),
        )

    declared = (
        ("hit_point", capabilities.supports_hit_point),
        ("normal", capabilities.supports_host_normal),
        ("geom_id", capabilities.supports_geom_id),
        ("body_id", capabilities.supports_body_id),
    )
    requested = RayTraceOutputs(**{name: flag for name, flag in declared})
    full = caster.trace(down_origin, down_direction, 10.0, outputs=requested)
    for name, flag in declared:
        assert (getattr(full, name) is not None) == flag, name
    if not all(flag for _, flag in declared):
        undeclared = next(name for name, flag in declared if not flag)
        expect_raises(
            UnsupportedCapabilityError,
            lambda: caster.trace(
                down_origin,
                down_direction,
                10.0,
                outputs=RayTraceOutputs(**{undeclared: True}),
            ),
        )

    if capabilities.supports_host_normal:
        normal_result = caster.trace(
            down_origin,
            down_direction,
            10.0,
            outputs=RayTraceOutputs(normal=True),
        )
        assert normal_result.normal is not None
        np.testing.assert_allclose(
            normal_result.normal,
            np.broadcast_to(np.array([0.0, 0.0, 1.0]), normal_result.normal.shape),
            atol=1e-6,
        )

    if (
        capabilities.supports_tensor_pose_sync
        and capabilities.supports_tensor_rays
        and capabilities.supports_device_output
    ):
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None:
            candidates = []
            if "cpu" in capabilities.torch_devices:
                candidates.append(torch.device("cpu"))
            if (
                any(label.startswith("cuda") for label in capabilities.torch_devices)
                and torch.cuda.is_available()
            ):
                candidates.append(torch.device("cuda", index=torch.cuda.current_device()))
            tensor_device = None
            caster.bind_pose_layout(tuple(scene.body_names))
            for candidate in candidates:
                pose = RayPoseTensorView(
                    body_names=tuple(scene.body_names),
                    pos_w=torch.zeros(
                        (caster.num_envs, scene.num_bodies, 3),
                        dtype=torch.float32,
                        device=candidate,
                    ),
                    quat_w=torch.zeros(
                        (caster.num_envs, scene.num_bodies, 4),
                        dtype=torch.float32,
                        device=candidate,
                    ),
                )
                pose.quat_w[..., 0] = 1.0
                try:
                    caster.update_pose_tensor(pose)
                except ValueError:
                    continue
                tensor_device = candidate
                break
            if tensor_device is not None:
                tensor_origins = torch.as_tensor(down_origin, dtype=torch.float32).to(tensor_device)
                tensor_directions = torch.as_tensor(down_direction, dtype=torch.float32).to(
                    tensor_device
                )
                tensor_outputs = RayTraceOutputs(
                    hit_point=capabilities.supports_hit_point,
                    normal=capabilities.supports_tensor_normal,
                    geom_id=capabilities.supports_geom_id,
                    body_id=capabilities.supports_body_id,
                )
                tensor_filter = (
                    RayTraceFilter(allowed_geom_ids=(0,))
                    if capabilities.supports_static_filter
                    else None
                )
                tensor_minimum = 0.5 if capabilities.supports_min_distance else 0.0
                tensor_result = caster.trace_tensor(
                    tensor_origins,
                    tensor_directions,
                    10.0,
                    outputs=tensor_outputs,
                    min_distance=tensor_minimum,
                    ray_filter=tensor_filter,
                )
                host_result = caster.trace(
                    down_origin,
                    down_direction,
                    10.0,
                    outputs=RayTraceOutputs(
                        hit_point=capabilities.supports_hit_point,
                        normal=capabilities.supports_host_normal,
                        geom_id=capabilities.supports_geom_id,
                        body_id=capabilities.supports_body_id,
                    ),
                    min_distance=tensor_minimum,
                    ray_filter=tensor_filter,
                )
                torch.testing.assert_close(
                    tensor_result.distance,
                    torch.as_tensor(host_result.distance, dtype=torch.float32).to(tensor_device),
                )
                torch.testing.assert_close(
                    tensor_result.hit,
                    torch.as_tensor(host_result.hit).to(tensor_device),
                )
                if tensor_result.normal is not None and host_result.normal is not None:
                    torch.testing.assert_close(
                        tensor_result.normal,
                        torch.as_tensor(host_result.normal, dtype=torch.float32).to(tensor_device),
                        atol=2e-6,
                        rtol=2e-6,
                    )
                repeated = caster.trace_tensor(
                    tensor_origins,
                    tensor_directions,
                    10.0,
                    outputs=tensor_outputs,
                    min_distance=tensor_minimum,
                    ray_filter=tensor_filter,
                )
                assert repeated.distance is tensor_result.distance
                assert repeated.hit is tensor_result.hit

    caster.close()
    caster.close()
    expect_raises(BackendError, lambda: caster.trace(down_origin, down_direction, 10.0))


def assert_backend_conformance(backend: SimBackend) -> None:
    """Run cheap shape/lifecycle checks against a materialized backend."""
    # Adapters may defer pool/worker allocation until the explicit cold-path
    # materialize hook. Conformance owns that lifecycle transition so a minimal
    # adapter test can construct, validate, and exercise the public contract.
    materialize = getattr(backend, "materialize", None)
    if callable(materialize):
        materialize()
    assert backend.num_envs > 0
    assert backend.num_actuators > 0
    assert BackendCapability.RESET in backend.capability_flags
    assert BackendCapability.STATE_READ in backend.capability_flags
    if backend.get_play_capabilities().supports_physics_state_playback:
        assert_physics_state_playback_conformance(backend)
    if backend.get_tensor_capabilities().packed_reset:
        assert_packed_reset_conformance(backend)
    assert_reset_randomization_conformance(backend)


def assert_reset_randomization_conformance(backend: SimBackend) -> None:
    """Assert one declared reset term round-trips through the torch-first payload.

    Write non-baseline values for one declared non-delta term through
    ``set_state_tensor`` on a declared Torch device, then re-read the term
    baseline in a fresh transaction — a new packed reset transaction when
    declared, otherwise a fresh ``get_reset_term_default`` read — and assert
    the written values come back. Torch is optional here; without it the check
    skips, matching the packed reset conformance pattern.
    """
    capabilities = backend.get_dr_capabilities()
    if not capabilities.supported_reset_terms:
        return
    try:
        import torch
    except ImportError:
        return

    tensor_capabilities = backend.get_tensor_capabilities()
    if "cpu" in tensor_capabilities.torch_devices:
        device = torch.device("cpu")
    elif any(
        label.split(":", maxsplit=1)[0] == "cuda" for label in tensor_capabilities.torch_devices
    ):
        if not torch.cuda.is_available():
            return
        device = torch.device("cuda", index=torch.cuda.current_device())
    else:
        return

    terms = sorted(capabilities.supported_reset_terms - _RESET_DELTA_TERMS)
    if not terms:
        return
    # Prefer kp: actuator gains accept arbitrary finite values on every adapter.
    term = "kp" if "kp" in terms else terms[0]
    tail_ndim = _RESET_TERM_TAIL_NDIMS[term]
    default = np.asarray(backend.get_reset_term_default(term), dtype=np.float32)
    if default.size == 0 or default.ndim not in (tail_ndim, tail_ndim + 1):
        return
    row = default if default.ndim == tail_ndim else default[0]
    written = np.asarray(row, dtype=np.float32) + np.float32(0.05)
    values = torch.as_tensor(written[None], dtype=torch.float32, device=device).contiguous()

    default_qpos = np.asarray(backend.get_default_qpos(), dtype=np.float32)
    default_qvel = np.asarray(backend.get_init_qvel(), dtype=np.float32)
    qpos = torch.as_tensor(default_qpos[None], dtype=torch.float32, device=device).contiguous()
    qvel = torch.as_tensor(default_qvel[None], dtype=torch.float32, device=device).contiguous()
    rows = torch.tensor([0], dtype=torch.int64, device=device)
    backend.set_state_tensor(rows, qpos, qvel, TensorResetRandomizationPayload(**{term: values}))

    expected = torch.as_tensor(written, dtype=torch.float32)
    if tensor_capabilities.packed_reset:
        transaction = backend.bind_packed_reset(device=device)
        baseline = transaction.baseline(term)
        assert baseline.shape[0] == backend.num_envs
        torch.testing.assert_close(baseline[0].detach().cpu(), expected, rtol=1e-4, atol=1e-5)
        return
    refreshed = np.asarray(backend.get_reset_term_default(term), dtype=np.float32)
    read_row = refreshed if refreshed.ndim == tail_ndim else refreshed[0]
    np.testing.assert_allclose(read_row, written, rtol=1e-4, atol=1e-5)


def assert_packed_reset_conformance(backend: SimBackend) -> None:
    """Validate the staged packed reset contract of a declaring backend."""
    capabilities = backend.get_tensor_capabilities()
    assert capabilities.device_reset_randomization, (
        "packed_reset requires device_reset_randomization"
    )
    assert capabilities.packed_reset_fields, "packed_reset requires at least one field"

    try:
        import torch
    except ImportError:
        return

    if "cpu" in capabilities.torch_devices:
        device = torch.device("cpu")
    elif any(label.split(":", maxsplit=1)[0] == "cuda" for label in capabilities.torch_devices):
        if not torch.cuda.is_available():
            return
        device = torch.device("cuda", index=torch.cuda.current_device())
    else:
        return

    transaction = backend.bind_packed_reset(device=device)
    state = backend.get_state_views(("qpos", "qvel"), device=device)
    qpos = state["qpos"][0].detach().to(device=device, dtype=torch.float32).clone()
    qvel = state["qvel"][0].detach().to(device=device, dtype=torch.float32).clone()
    transaction.set_state_default(qpos, qvel)
    rows = torch.tensor([0], dtype=torch.int64, device=device)
    transaction.begin_state(rows)
    written_qpos = torch.full((1, qpos.numel()), 0.5, dtype=torch.float32, device=device)
    written_qvel = torch.full((1, qvel.numel()), -0.25, dtype=torch.float32, device=device)
    transaction.write_state(
        rows,
        qpos_values=written_qpos,
        qvel_values=written_qvel,
    )

    # Delta terms (base_mass_delta/base_com_offset) merge against a baseline
    # and require a base body, so exercise a plain absolute field here.
    fields = sorted(capabilities.packed_reset_fields - _RESET_DELTA_TERMS)
    assert fields, "packed_reset requires at least one non-delta field"
    field = fields[0]
    baseline = transaction.baseline(field)
    assert baseline.shape[0] == backend.num_envs
    field_rows = baseline.index_select(0, rows)
    transaction.write_model_field(field, rows, field_rows)
    result = transaction.commit(rows)
    timing = result.get("timing", {})
    assert {"packed_reset_compose_ms", "packed_reset_commit_ms", "packed_reset_total_ms"} <= set(
        timing
    )

    views = backend.get_state_views(("qpos", "qvel"), device=device)
    torch.testing.assert_close(views["qpos"][rows], written_qpos)
    torch.testing.assert_close(views["qvel"][rows], written_qvel)

    reset_mask = torch.zeros(backend.num_envs, dtype=torch.bool, device=device)
    reset_mask[rows] = True
    transaction.begin_state_mask(reset_mask)
    transaction.write_state_mask(
        reset_mask,
        qpos_values=written_qpos.expand(backend.num_envs, -1).contiguous(),
        qvel_values=written_qvel.expand(backend.num_envs, -1).contiguous(),
    )
    result = transaction.commit_mask(reset_mask)
    timing = result.get("timing", {})
    assert {"packed_reset_compose_ms", "packed_reset_commit_ms", "packed_reset_total_ms"} <= set(
        timing
    )
    views = backend.get_state_views(("qpos", "qvel"), device=device)
    torch.testing.assert_close(views["qpos"][rows], written_qpos)
    if backend.num_envs > 1:
        false_rows = torch.ones(backend.num_envs, dtype=torch.bool, device=device)
        false_rows[rows] = False
        torch.testing.assert_close(views["qpos"][false_rows], state["qpos"][false_rows])


def assert_physics_state_playback_conformance(backend: SimBackend) -> None:
    """Validate the physics-state playback contract of a materialized backend."""
    layout = backend.get_physics_state_layout()
    snapshot = np.asarray(backend.get_physics_state())
    expected = (backend.num_envs, layout.state_width)
    assert snapshot.shape == expected, (
        f"physics-state snapshot must have shape {expected}, got {snapshot.shape}"
    )
    parts = layout.split_state(snapshot)
    assert np.isfinite(parts.qpos).all(), "physics-state snapshot qpos is not finite"
    assert np.isfinite(parts.qvel).all(), "physics-state snapshot qvel is not finite"

    _assert_playback_model_loadable(backend)

    if type(backend).set_physics_state is not SimBackend.set_physics_state:
        backend.set_physics_state(snapshot)
        restored = np.asarray(backend.get_physics_state())
        assert restored.shape == snapshot.shape
        np.testing.assert_allclose(restored, snapshot, rtol=1e-5, atol=1e-6)

    if layout.nmocap > 0:
        assert backend.get_play_capabilities().supports_mocap_playback, (
            "backend snapshots carry a mocap tail but do not declare supports_mocap_playback"
        )
        mocap_pos, mocap_quat = backend.get_playback_mocap_state(0)
        assert np.asarray(mocap_pos).shape == (layout.nmocap, 3)
        assert np.asarray(mocap_quat).shape == (layout.nmocap, 4)
        assert np.isfinite(mocap_pos).all() and np.isfinite(mocap_quat).all()
        assert parts.mocap_pos is not None and parts.mocap_quat is not None


def _assert_playback_model_loadable(backend: SimBackend) -> None:
    """Check that the playback model is an ``MjModel`` or a compilable file."""
    try:
        model = backend.get_playback_model()
    except (ValueError, IndexError):
        # Fixed-variant backends require an explicit per-env index.
        model = backend.get_playback_model(0)
    try:
        import mujoco
    except ImportError:
        # The base package must not hard-depend on MuJoCo; adapter suites with
        # the runtime installed re-check loadability through their own tests.
        return
    if isinstance(model, mujoco.MjModel):
        return
    assert isinstance(model, (str, PathLike)), (
        f"playback model must be an MjModel or a model file path, got {type(model).__name__}"
    )
    path = str(model)
    if path.endswith(".mjb"):
        mujoco.MjModel.from_binary_path(path)
    else:
        mujoco.MjModel.from_xml_path(path)
