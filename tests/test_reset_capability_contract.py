"""Engine-independent negative paths for selected-row reset capabilities."""

import numpy as np
import pytest

from unisim import FakeBackend
from unisim.backend.base import BackendMocapPoseBinding
from unisim.dr.types import DomainRandomizationCapabilities, ResetRandomizationPayload


def test_new_reset_terms_are_never_silently_advertised_or_filtered() -> None:
    fields = ("geom_size", "geom_solref", "geom_solimp", "dof_damping", "dof_frictionloss")
    payload = ResetRandomizationPayload(**{name: np.zeros((1, 1)) for name in fields})
    assert payload.requested_terms() == frozenset(fields)
    assert not payload.is_empty()
    filtered, unsupported = DomainRandomizationCapabilities().filter_reset_payload(payload)
    assert filtered is None
    assert unsupported == frozenset(fields)
    caps = DomainRandomizationCapabilities(supported_reset_terms=frozenset(fields))
    assert caps.filter_reset_payload(payload) == (payload, frozenset())
    partial_caps = DomainRandomizationCapabilities(supported_reset_terms=frozenset({"geom_size"}))
    filtered, unsupported = partial_caps.filter_reset_payload(payload)
    assert filtered.requested_terms() == frozenset({"geom_size"})
    assert unsupported == frozenset(fields[1:])


@pytest.mark.parametrize(
    "name",
    (
        "get_geom_sizes",
        "get_geom_solref",
        "get_geom_solimp",
        "get_dof_damping",
        "get_dof_frictionloss",
        "bind_mocap_pose",
    ),
)
def test_unimplemented_backends_fail_explicitly(name: str) -> None:
    backend = FakeBackend(num_envs=2, num_actuators=1)
    with pytest.raises(NotImplementedError):
        getattr(backend, name)(*(("root",) if name == "bind_mocap_pose" else ()))


@pytest.mark.parametrize("adapter", ("mujoco", "motrix", "genesis"))
@pytest.mark.parametrize(
    "field", ("geom_size", "geom_solref", "geom_solimp", "dof_damping", "dof_frictionloss")
)
def test_legacy_adapter_rejects_new_payload_before_any_state_access(adapter, field) -> None:
    # Construction requires optional SDKs. Calling the unbound method on an
    # object with only the documented capability query proves the rejection
    # precedes every state/native-runtime access, without mocking an engine.
    import importlib

    class CapabilityOnly:
        def get_dr_capabilities(self):
            return DomainRandomizationCapabilities()

    module = importlib.import_module(f"unisim.backend.{adapter}.backend")
    class_name = {
        "mujoco": "MuJoCoBackend",
        "motrix": "MotrixBackend",
        "genesis": "GenesisBackend",
    }[adapter]
    backend_class = getattr(module, class_name)
    with pytest.raises(NotImplementedError, match=field):
        backend_class.set_state(
            CapabilityOnly(),
            np.array([0]),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            ResetRandomizationPayload(**{field: np.zeros((1, 1))}),
        )


def _binding():
    state = np.tile(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32), (3, 1))
    writes = []

    def write(rows, poses):
        writes.append(rows.copy())
        state[rows] = poses

    return BackendMocapPoseBinding("test", "root", 3, state[0], lambda: state, write), writes


@pytest.mark.parametrize(
    "rows,poses,error",
    (
        (np.array([0.5]), np.zeros((1, 7)), TypeError),
        (np.array([True]), np.zeros((1, 7)), TypeError),
        (np.array([0, 0]), np.zeros((2, 7)), ValueError),
        (np.array([-1]), np.zeros((1, 7)), IndexError),
        (np.array([3]), np.zeros((1, 7)), IndexError),
        (np.array([0]), np.zeros((1, 6)), ValueError),
        (np.array([0]), np.zeros((1, 7), dtype=np.int32), TypeError),
        (np.array([0]), np.full((1, 7), np.nan), ValueError),
        (np.array([0]), np.zeros((1, 7)), ValueError),
    ),
)
def test_mocap_invalid_writes_do_not_call_adapter(rows, poses, error) -> None:
    binding, writes = _binding()
    before = binding.read()
    with pytest.raises(error):
        binding.write(rows, poses)
    assert not writes
    np.testing.assert_array_equal(binding.read(), before)


def test_mocap_sparse_write_default_snapshot_and_empty_ids() -> None:
    binding, writes = _binding()
    pose = binding.default_pose.copy()
    pose[:3] = [1, 2, 3]
    binding.write(np.array([2]), pose[None])
    np.testing.assert_array_equal(binding.read()[2], pose)
    np.testing.assert_array_equal(binding.read()[:2], np.tile(binding.default_pose, (2, 1)))
    binding.read()[:] = 8  # Detached read must not let consumers mutate physics.
    assert binding.read()[2, 0] == 1
    with pytest.raises(ValueError):
        binding.default_pose[0] = 2
    binding.write(np.array([], dtype=np.int32), np.zeros((0, 7), dtype=np.float32))
    assert len(writes) == 1
