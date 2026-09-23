"""Reusable checks for adapter authors.

The helper intentionally tests only the public contract. Engine-specific
fixtures and expensive differential tests belong to adapter-owned suites.
"""

from __future__ import annotations

from os import PathLike

import numpy as np

from .contract import BackendCapability, SimBackend
from .errors import BackendError, UnsupportedCapabilityError
from .ray_query import (
    RayCaster,
    RayGeomType,
    RaySceneDescription,
    RayTraceOutputs,
)


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
        ("normal", capabilities.supports_normal),
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
    assert BackendCapability.RESET in backend.capabilities
    assert BackendCapability.STATE_READ in backend.capabilities
    ctrl = np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float64)
    backend.step(ctrl)
    state = backend.get_state()
    assert state, "backend must expose at least one state field"
    for name, value in state.items():
        array = np.asarray(value)
        assert np.isfinite(array).all(), f"state field {name!r} contains non-finite values"
    if BackendCapability.SELECTED_RESET in backend.capabilities:
        backend.reset(np.arange(min(1, backend.num_envs), dtype=np.intp))
    else:
        backend.reset()
    if backend.get_play_capabilities().supports_physics_state_playback:
        assert_physics_state_playback_conformance(backend)


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
            "backend snapshots carry a mocap tail but do not declare "
            "supports_mocap_playback"
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

