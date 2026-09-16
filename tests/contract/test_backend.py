import numpy as np
import pytest

from unisim import (
    ADAPTER_SPECS,
    BackendCapability,
    BenchmarkCase,
    FakeBackend,
    assert_backend_conformance,
    create_backend,
)
from unisim.backend.base import PreStepControlOutput


def test_fake_backend_conforms() -> None:
    backend = FakeBackend(num_envs=3, num_actuators=2)
    assert_backend_conformance(backend)
    backend.step(np.ones((3, 2)), nsteps=2)
    np.testing.assert_allclose(backend.get_state(("qpos",))["qpos"], 2.0)
    assert BackendCapability.STATE_WRITE in backend.capabilities


def test_benchmark_api_is_only_data() -> None:
    case = BenchmarkCase("future-step-case")
    assert case.schema_version == "0.1"
    assert case.name == "future-step-case"


def test_factory_keeps_optional_backend_lazy() -> None:
    backend = create_backend("fake", num_envs=1, num_actuators=1)
    assert isinstance(backend, FakeBackend)


def test_adapter_manifest_covers_roadmap_backends() -> None:
    assert {spec.name for spec in ADAPTER_SPECS} == {
        "mujoco",
        "motrix",
        "drake",
        "mjwarp",
        "newton",
        "superdex",
        "genesis",
        "isaacgym",
        "isaacsim",
    }


def test_unknown_backend_fails_closed() -> None:
    with np.testing.assert_raises_regex(ValueError, "unknown UniSim backend"):
        create_backend("missing")


def test_factory_rejects_invalid_body_state_required() -> None:
    with np.testing.assert_raises_regex(TypeError, "body_state_required must be bool"):
        create_backend("fake", body_state_required=1)  # type: ignore[arg-type]


def test_factory_rejects_invalid_pre_step_body_state_refresh() -> None:
    with np.testing.assert_raises_regex(TypeError, "refresh_pre_step_body_state must be bool"):
        create_backend("fake", refresh_pre_step_body_state=1)  # type: ignore[arg-type]


def test_factory_scopes_pre_step_body_state_refresh_to_mujoco() -> None:
    with np.testing.assert_raises_regex(
        TypeError, "refresh_pre_step_body_state is only supported by the mujoco backend"
    ):
        create_backend("fake", refresh_pre_step_body_state=False)


def test_fake_factory_path_is_engine_independent() -> None:
    backend = create_backend("fake", num_envs=2, num_actuators=1)
    assert isinstance(backend, FakeBackend)
    assert backend.backend_type == "fake"


def test_ctrl_only_pre_step_backends_fail_closed_on_wrench() -> None:
    backend = FakeBackend(num_envs=2, num_actuators=1)
    ctrl = np.zeros((2, 1))
    body_ids = np.zeros(2, dtype=np.intp)
    force = np.zeros((2, 2, 3))
    backend.set_pre_step_control(
        lambda owner, c: PreStepControlOutput(ctrl=c, body_ids=body_ids, force=force)
    )
    with pytest.raises(
        NotImplementedError, match="FakeBackend does not support pre-step control wrenches"
    ):
        backend._apply_pre_step_control(ctrl)


def test_pre_step_wrench_output_validation() -> None:
    backend = FakeBackend(num_envs=2, num_actuators=1)
    ctrl = np.zeros((2, 1))
    body_ids = np.zeros(2, dtype=np.intp)

    # A wrench without body_ids names no target.
    backend.set_pre_step_control(
        lambda owner, c: PreStepControlOutput(ctrl=c, force=np.zeros((2, 1, 3)))
    )
    with pytest.raises(ValueError, match="wrench requires body_ids"):
        backend._apply_pre_step_control(ctrl)

    # body_ids without force/torque is equally meaningless.
    backend.set_pre_step_control(lambda owner, c: PreStepControlOutput(ctrl=c, body_ids=body_ids))
    with pytest.raises(ValueError, match="requires force and/or torque"):
        backend._apply_pre_step_control(ctrl)

    # Shape and finiteness are validated before any backend consumes it.
    backend.set_pre_step_control(
        lambda owner, c: PreStepControlOutput(
            ctrl=c, body_ids=body_ids, force=np.zeros((2, 3, 3))
        )
    )
    with pytest.raises(ValueError, match="force must have shape"):
        backend._apply_pre_step_control(ctrl)
    bad = np.full((2, 2, 3), np.nan)
    backend.set_pre_step_control(
        lambda owner, c: PreStepControlOutput(ctrl=c, body_ids=body_ids, force=bad)
    )
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        backend._apply_pre_step_control(ctrl)
