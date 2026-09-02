import numpy as np

from unisim import (
    ADAPTER_SPECS,
    BackendCapability,
    BenchmarkCase,
    FakeBackend,
    assert_backend_conformance,
    create_backend,
)


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
        "genesis",
        "isaacgym",
        "isaacsim",
    }


def test_unknown_backend_fails_closed() -> None:
    with np.testing.assert_raises_regex(ValueError, "unknown UniSim backend"):
        create_backend("missing")
