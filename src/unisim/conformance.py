"""Reusable checks for adapter authors.

The helper intentionally tests only the public contract. Engine-specific
fixtures and expensive differential tests belong to adapter-owned suites.
"""

from __future__ import annotations

from os import PathLike

import numpy as np

from .contract import BackendCapability, SimBackend


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
    if backend.get_tensor_capabilities().packed_reset:
        assert_packed_reset_conformance(backend)


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

    transaction = backend.begin_packed_reset(device=device)
    state = backend.get_state()
    qpos = torch.as_tensor(np.asarray(state["qpos"][0], dtype=np.float32), device=device)
    qvel = torch.as_tensor(np.asarray(state["qvel"][0], dtype=np.float32), device=device)
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

    field = next(iter(sorted(capabilities.packed_reset_fields)))
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
