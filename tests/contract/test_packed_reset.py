from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from unisim import FakeBackend  # noqa: E402
from unisim.backend.base import (  # noqa: E402
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)
from unisim.backend.packed_reset import (  # noqa: E402
    PACKED_RESET_HOST_ONLY_FIELDS,
    PACKED_RESET_MODEL_FIELDS,
    PACKED_RESET_PERSISTENT_CHANNELS,
    TorchPackedResetComposer,
)
from unisim.dr.types import (  # noqa: E402
    RESET_TERM_KD,
    RESET_TERM_KP,
    TensorResetRandomizationPayload,
)


class _ComposerBackend:
    backend_type = "composer-stub"
    num_envs = 4

    def __init__(self, *, packed_reset_mocap: bool = False) -> None:
        self.device = torch.device("cpu")
        self.commits: list[tuple[object, object, object, TensorResetRandomizationPayload]] = []
        self.mocap_writes: list[tuple[str, object, object]] = []
        self.packed_reset_mocap = packed_reset_mocap

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        return TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            state_views=True,
            state_fields=frozenset({"qpos", "qvel"}),
            selected_reset=True,
            reset_randomization=True,
            device_reset_randomization=True,
            packed_reset=True,
            packed_reset_fields=PACKED_RESET_MODEL_FIELDS,
            packed_reset_mocap=self.packed_reset_mocap,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="stub-completes-reset",
            torch_devices=("cpu",),
        )

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if term == RESET_TERM_KP:
            return np.full((4, 3), 20.0, dtype=np.float32)
        if term == RESET_TERM_KD:
            return np.full((4, 3), 2.0, dtype=np.float32)
        if term == "body_ipos":
            return np.zeros((2, 3), dtype=np.float32)
        raise AssertionError(term)

    def set_state_tensor(self, rows, qpos, qvel, randomization=None):
        self.commits.append((rows, qpos, qvel, randomization))
        return {"timing": {"set_state_tensor_commit_forward_ms": 1.0}}

    def set_mocap_pose_tensor(self, body_name, rows, poses):
        self.mocap_writes.append((body_name, rows, poses))


def test_packed_reset_defaults_fail_closed() -> None:
    backend = FakeBackend()
    with pytest.raises(NotImplementedError, match="does not support packed reset"):
        backend.begin_packed_reset()
    capabilities = backend.get_tensor_capabilities()
    assert not capabilities.packed_reset
    assert not capabilities.packed_reset_fields


def test_unsupported_tensor_lifecycle_rejects_packed_reset_declaration() -> None:
    with pytest.raises(ValueError, match="unsupported tensor lifecycle"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.UNSUPPORTED,
            packed_reset=True,
        )
    with pytest.raises(ValueError, match="packed reset fields"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.UNSUPPORTED,
            packed_reset_fields=frozenset({RESET_TERM_KP}),
        )
    with pytest.raises(ValueError, match="unsupported tensor lifecycle"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.UNSUPPORTED,
            packed_reset_mocap=True,
        )


def test_composer_merges_columns_and_promotes_baselines() -> None:
    backend = _ComposerBackend()
    transaction = TorchPackedResetComposer(backend)
    assert transaction.baseline(RESET_TERM_KP).shape == (4, 3)

    rows = torch.tensor([1, 3], dtype=torch.int64)
    transaction.write_model_field(
        RESET_TERM_KP,
        rows,
        torch.full((2, 2), 40.0, dtype=torch.float32),
        columns=torch.tensor([0, 2], dtype=torch.int64),
    )
    expected = torch.full((4, 3), 20.0)
    expected[rows, 0] = 40.0
    expected[rows, 2] = 40.0
    torch.testing.assert_close(transaction.baseline(RESET_TERM_KP), torch.full((4, 3), 20.0))

    qpos = torch.zeros((2, 2), dtype=torch.float32)
    qvel = torch.zeros((2, 2), dtype=torch.float32)
    result = transaction.commit_state(rows, qpos, qvel)
    assert result["timing"]["set_state_tensor_commit_forward_ms"] == 1.0
    assert result["timing"]["packed_reset_compose_ms"] >= 0.0
    assert result["timing"]["packed_reset_commit_ms"] >= 0.0
    assert result["timing"]["packed_reset_total_ms"] >= 0.0
    payload = backend.commits[0][3]
    assert isinstance(payload, TensorResetRandomizationPayload)
    torch.testing.assert_close(payload.kp, expected[rows])
    torch.testing.assert_close(transaction.baseline(RESET_TERM_KP), expected)


def test_composer_discard_and_host_only_fail_closed() -> None:
    backend = _ComposerBackend()
    transaction = TorchPackedResetComposer(backend)
    transaction.write_model_field(
        RESET_TERM_KD,
        torch.tensor([0], dtype=torch.int64),
        torch.full((1, 3), 4.0, dtype=torch.float32),
    )
    transaction.discard(RESET_TERM_KD)
    torch.testing.assert_close(
        transaction.baseline(RESET_TERM_KD), torch.full((4, 3), 2.0)
    )

    for field in PACKED_RESET_HOST_ONLY_FIELDS:
        with pytest.raises(NotImplementedError, match="host-only delta-merge"):
            transaction.write_model_field(field, torch.empty(0, dtype=torch.int64), None)

    assert PACKED_RESET_PERSISTENT_CHANNELS == frozenset(
        {"ctrl", "act", "qfrc_applied", "xfrc_applied", "qacc_warmstart"}
    )


def test_composer_stages_merges_and_commits_mocap_poses() -> None:
    backend = _ComposerBackend(packed_reset_mocap=True)
    transaction = TorchPackedResetComposer(backend)

    with pytest.raises(NotImplementedError, match="does not support packed reset mocap"):
        TorchPackedResetComposer(_ComposerBackend()).write_mocap_pose(
            "palm", torch.empty(0, dtype=torch.int64), torch.empty(0, 7)
        )

    pose = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])
    transaction.write_mocap_pose("palm", torch.tensor([0]), pose)
    transaction.write_mocap_pose(
        "palm", torch.tensor([2]), torch.tensor([[4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0]])
    )
    transaction.write_mocap_pose(
        "palm", torch.tensor([0]), torch.tensor([[7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0.0]])
    )

    rows = torch.tensor([1])
    result = transaction.commit_state(
        rows,
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
    )
    assert result["timing"]["packed_reset_total_ms"] >= 0.0
    assert len(backend.commits) == 1
    assert len(backend.mocap_writes) == 1
    body_name, mocap_rows, mocap_poses = backend.mocap_writes[0]
    assert body_name == "palm"
    torch.testing.assert_close(mocap_rows, torch.tensor([2, 0]))
    torch.testing.assert_close(mocap_poses[0, :3], torch.tensor([4.0, 5.0, 6.0]))
    torch.testing.assert_close(mocap_poses[1, :3], torch.tensor([7.0, 8.0, 9.0]))

    transaction.write_mocap_pose("palm", torch.tensor([3]), pose)
    result = transaction.commit_mocap_poses()
    assert result["timing"]["packed_reset_total_ms"] >= 0.0
    assert len(backend.commits) == 1
    assert backend.mocap_writes[-1][1].tolist() == [3]


def test_composer_promotes_only_committed_rows() -> None:
    backend = _ComposerBackend()
    transaction = TorchPackedResetComposer(backend)
    transaction.write_model_field(
        RESET_TERM_KP,
        torch.tensor([3], dtype=torch.int64),
        torch.full((1, 3), 60.0, dtype=torch.float32),
    )
    committed_rows = torch.tensor([0], dtype=torch.int64)
    transaction.commit_state(
        committed_rows,
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
    )
    torch.testing.assert_close(
        transaction.baseline(RESET_TERM_KP), torch.full((4, 3), 20.0)
    )

    transaction.commit_state(
        torch.tensor([3], dtype=torch.int64),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
    )
    promoted = transaction.baseline(RESET_TERM_KP)
    torch.testing.assert_close(promoted[3], torch.full((3,), 60.0))
    torch.testing.assert_close(promoted[:3], torch.full((3, 3), 20.0))


def test_composer_merges_tailed_columns_and_stages_state() -> None:
    backend = _ComposerBackend()
    transaction = TorchPackedResetComposer(backend)
    rows = torch.tensor([0, 2], dtype=torch.int64)
    transaction.write_model_field(
        "body_ipos",
        rows,
        torch.full((2, 1, 3), 0.25, dtype=torch.float32),
        columns=torch.tensor([1], dtype=torch.int64),
    )

    transaction.set_state_default(
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([3.0, 4.0], dtype=torch.float32),
    )
    transaction.begin_state(rows)
    qpos, qvel = transaction.state_views()
    torch.testing.assert_close(qpos, torch.tensor([[1.0, 2.0]] * 4))
    transaction.write_state(
        rows,
        qpos_columns=torch.tensor([1], dtype=torch.int64),
        qpos_values=torch.tensor([[20.0], [22.0]], dtype=torch.float32),
        qvel_columns=torch.tensor([0], dtype=torch.int64),
        qvel_values=torch.tensor([[30.0], [32.0]], dtype=torch.float32),
    )
    result = transaction.commit(rows)
    assert result["timing"]["packed_reset_total_ms"] >= 0.0
    committed_rows, committed_qpos, committed_qvel, payload = backend.commits[-1]
    assert torch.equal(committed_rows, rows)
    torch.testing.assert_close(committed_qpos[:, 1], torch.tensor([20.0, 22.0]))
    torch.testing.assert_close(committed_qvel[:, 0], torch.tensor([30.0, 32.0]))
    torch.testing.assert_close(payload.body_ipos[0, 1], torch.full((3,), 0.25))
