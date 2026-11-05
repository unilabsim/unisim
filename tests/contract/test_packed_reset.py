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
    PACKED_RESET_MODEL_FIELDS,
    PACKED_RESET_PERSISTENT_CHANNELS,
    PackedResetTransaction,
)
from unisim.dr.types import (  # noqa: E402
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
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
        self.mask_commits: list[tuple[object, object, object, TensorResetRandomizationPayload]] = []
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
        if term == RESET_TERM_BASE_MASS:
            return np.full((4,), 10.0, dtype=np.float32)
        if term == RESET_TERM_BASE_COM:
            return np.zeros((4, 3), dtype=np.float32)
        if term == "body_ipos":
            return np.zeros((2, 3), dtype=np.float32)
        raise AssertionError(term)

    def set_state_tensor(self, rows, qpos, qvel, randomization=None):
        self.commits.append((rows, qpos, qvel, randomization))
        return {"timing": {"set_state_tensor_commit_forward_ms": 1.0}}

    def set_state_tensor_mask(self, reset_mask, qpos, qvel, randomization=None, **_kwargs):
        self.mask_commits.append((reset_mask, qpos, qvel, randomization))
        return {"timing": {"set_state_tensor_mask_ms": 1.0}}

    def set_mocap_pose_tensor(self, body_name, rows, poses):
        self.mocap_writes.append((body_name, rows, poses))


def test_packed_reset_defaults_fail_closed() -> None:
    backend = FakeBackend()
    with pytest.raises(NotImplementedError, match="does not support packed reset"):
        backend.bind_packed_reset()
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
    transaction = PackedResetTransaction(backend)
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


def test_composer_discard_and_persistent_channels() -> None:
    backend = _ComposerBackend()
    transaction = PackedResetTransaction(backend)
    transaction.write_model_field(
        RESET_TERM_KD,
        torch.tensor([0], dtype=torch.int64),
        torch.full((1, 3), 4.0, dtype=torch.float32),
    )
    transaction.discard(RESET_TERM_KD)
    torch.testing.assert_close(transaction.baseline(RESET_TERM_KD), torch.full((4, 3), 2.0))

    assert PACKED_RESET_PERSISTENT_CHANNELS == frozenset(
        {"ctrl", "act", "qfrc_applied", "xfrc_applied", "qacc_warmstart"}
    )


def test_composer_stages_delta_terms_and_merges_in_torch_at_commit() -> None:
    backend = _ComposerBackend()
    transaction = PackedResetTransaction(backend)
    rows = torch.tensor([1, 3], dtype=torch.int64)
    transaction.write_model_field(
        RESET_TERM_BASE_MASS,
        rows,
        torch.full((2,), 1.5, dtype=torch.float32),
    )
    # Staging holds deltas; the baseline is untouched before commit.
    torch.testing.assert_close(transaction.baseline(RESET_TERM_BASE_MASS), torch.full((4,), 10.0))
    transaction.commit_state(
        rows,
        torch.zeros((2, 2), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.float32),
    )
    payload = backend.commits[-1][3]
    assert isinstance(payload, TensorResetRandomizationPayload)
    torch.testing.assert_close(payload.base_mass_delta, torch.full((2,), 11.5))
    torch.testing.assert_close(
        transaction.baseline(RESET_TERM_BASE_MASS),
        torch.tensor([10.0, 11.5, 10.0, 11.5]),
    )


def test_composer_mask_commit_merges_delta_terms() -> None:
    backend = _ComposerBackend()
    transaction = PackedResetTransaction(backend)
    mask = torch.tensor([False, True, False, True])
    transaction.write_model_field_mask(
        RESET_TERM_BASE_COM,
        mask,
        torch.full((4, 3), 0.25, dtype=torch.float32),
    )
    transaction.commit_state_mask(
        mask,
        torch.zeros((4, 2), dtype=torch.float32),
        torch.zeros((4, 2), dtype=torch.float32),
    )
    payload = backend.mask_commits[-1][3]
    assert isinstance(payload, TensorResetRandomizationPayload)
    torch.testing.assert_close(payload.base_com_offset[1], torch.full((3,), 0.25))
    torch.testing.assert_close(payload.base_com_offset[0], torch.zeros(3))
    torch.testing.assert_close(transaction.baseline(RESET_TERM_BASE_COM)[1], torch.full((3,), 0.25))


def test_composer_mask_commit_emits_no_completion_receipt() -> None:
    backend = _ComposerBackend()
    transaction = PackedResetTransaction(backend)
    mask = torch.tensor([False, True, False, True])
    qpos = torch.zeros((4, 2), dtype=torch.float32)
    qvel = torch.zeros((4, 2), dtype=torch.float32)

    result = transaction.commit_state_mask(mask, qpos, qvel)

    # Reset operations emit no completion receipt under the async step-only
    # completion contract.
    assert "completion" not in result
    assert result["timing"]["set_state_tensor_mask_ms"] == 1.0


def test_composer_stages_merges_and_commits_mocap_poses() -> None:
    backend = _ComposerBackend(packed_reset_mocap=True)
    transaction = PackedResetTransaction(backend)

    with pytest.raises(NotImplementedError, match="does not support packed reset mocap"):
        PackedResetTransaction(_ComposerBackend()).write_mocap_pose(
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
    transaction = PackedResetTransaction(backend)
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
    torch.testing.assert_close(transaction.baseline(RESET_TERM_KP), torch.full((4, 3), 20.0))

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
    transaction = PackedResetTransaction(backend)
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


def test_composer_mask_lifecycle_stays_full_width_and_preserves_false_rows() -> None:
    backend = _ComposerBackend()
    transaction = PackedResetTransaction(backend)
    transaction.set_state_default(
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([3.0, 4.0], dtype=torch.float32),
    )
    reset_mask = torch.tensor([False, True, False, True])
    original_qpos, original_qvel = transaction.state_views()
    original_qpos.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2))
    original_qvel.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2) + 10.0)
    transaction.begin_state_mask(reset_mask)

    qpos, qvel = transaction.state_views()
    torch.testing.assert_close(qpos[reset_mask], torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    torch.testing.assert_close(qvel[reset_mask], torch.tensor([[3.0, 4.0], [3.0, 4.0]]))
    torch.testing.assert_close(qpos[~reset_mask], original_qpos[~reset_mask])
    torch.testing.assert_close(qvel[~reset_mask], original_qvel[~reset_mask])

    transaction.write_state_mask(
        reset_mask,
        qpos_columns=torch.tensor([1], dtype=torch.int64),
        qpos_values=torch.tensor([[20.0], [21.0], [22.0], [23.0]], dtype=torch.float32),
        qvel_columns=torch.tensor([0], dtype=torch.int64),
        qvel_values=torch.tensor([[30.0], [31.0], [32.0], [33.0]], dtype=torch.float32),
    )
    result = transaction.commit_mask(reset_mask)
    assert result["timing"]["packed_reset_total_ms"] >= 0.0
    assert result["timing"]["set_state_tensor_mask_ms"] == 1.0

    committed_mask, committed_qpos, committed_qvel, payload = backend.mask_commits[0]
    assert committed_mask is reset_mask
    assert tuple(committed_qpos.shape) == (4, 2)
    assert tuple(committed_qvel.shape) == (4, 2)
    assert payload is None
    torch.testing.assert_close(committed_qpos[reset_mask, 1], torch.tensor([21.0, 23.0]))
    torch.testing.assert_close(committed_qpos[~reset_mask, 1], original_qpos[~reset_mask, 1])
    torch.testing.assert_close(committed_qvel[reset_mask, 0], torch.tensor([31.0, 33.0]))
    torch.testing.assert_close(committed_qvel[~reset_mask, 0], original_qvel[~reset_mask, 0])


def test_composer_mask_lifecycle_validates_layout() -> None:
    backend = _ComposerBackend()
    transaction = PackedResetTransaction(backend)
    transaction.set_state_default(
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([3.0, 4.0], dtype=torch.float32),
    )
    with pytest.raises(TypeError, match="contiguous 1-D bool"):
        transaction.begin_state_mask(torch.zeros(4, dtype=torch.float32))
    with pytest.raises(ValueError, match="must have shape"):
        transaction.begin_state_mask(torch.zeros(3, dtype=torch.bool))


def test_composer_deferred_validation_latches_per_field_slots() -> None:
    backend = _ComposerBackend()
    transaction = PackedResetTransaction(backend)
    assert transaction.validation == "deferred"

    rows = torch.tensor([0, 1], dtype=torch.int64)
    transaction.write_model_field(RESET_TERM_KP, rows, torch.full((2, 3), float("nan")))
    transaction.write_model_field(RESET_TERM_KD, rows, torch.full((2, 3), float("inf")))
    assert transaction.has_pending_verdict()

    verdict = transaction.take_verdict()
    assert verdict == (RESET_TERM_KD, RESET_TERM_KP)
    assert not transaction.has_pending_verdict()
    assert transaction.take_verdict() == ()


def test_composer_deferred_validation_survives_commit_until_polled() -> None:
    backend = _ComposerBackend()
    transaction = PackedResetTransaction(backend)
    rows = torch.tensor([0, 1], dtype=torch.int64)
    transaction.write_model_field(RESET_TERM_KP, rows, torch.full((2, 3), float("nan")))
    result = transaction.commit_state(
        rows,
        torch.zeros((2, 2), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.float32),
    )
    assert result["timing"]["packed_reset_total_ms"] >= 0.0
    assert transaction.has_pending_verdict()
    with pytest.raises(ValueError, match=r"deferred validation failed for: kp"):
        transaction.raise_for_verdict()
    assert not transaction.has_pending_verdict()


def test_composer_deferred_validation_covers_state_writes() -> None:
    backend = _ComposerBackend()
    transaction = PackedResetTransaction(backend)
    transaction.set_state_default(
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([3.0, 4.0], dtype=torch.float32),
    )
    rows = torch.tensor([0, 2], dtype=torch.int64)
    transaction.begin_state(rows)
    transaction.write_state(
        rows,
        qpos_values=torch.tensor([[float("nan"), 1.0], [2.0, 3.0]], dtype=torch.float32),
    )
    transaction.write_state_mask(
        torch.tensor([True, True, True, True]),
        qpos_values=None,
        qvel_values=torch.full((4, 2), float("inf"), dtype=torch.float32),
    )
    assert transaction.take_verdict() == ("qpos", "qvel")


def test_composer_deferred_validation_covers_mocap_and_columns() -> None:
    backend = _ComposerBackend(packed_reset_mocap=True)
    transaction = PackedResetTransaction(backend)
    rows = torch.tensor([0], dtype=torch.int64)
    transaction.write_mocap_pose(
        "palm",
        rows,
        torch.tensor([[1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    transaction.write_model_field_mask(
        RESET_TERM_KP,
        torch.tensor([True, False, False, False]),
        torch.full((4, 1), 30.0, dtype=torch.float32),
        columns=torch.tensor([7], dtype=torch.int64),
    )
    assert transaction.take_verdict() == (f"{RESET_TERM_KP}:columns", "mocap_quaternion_norm")


def test_composer_strict_validation_raises_at_the_write() -> None:
    backend = _ComposerBackend(packed_reset_mocap=True)
    transaction = PackedResetTransaction(backend, validation="strict")
    assert transaction.validation == "strict"
    rows = torch.tensor([0, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match="kp values must be finite"):
        transaction.write_model_field(RESET_TERM_KP, rows, torch.full((2, 3), float("nan")))
    with pytest.raises(ValueError, match="columns must be within"):
        transaction.write_model_field_mask(
            RESET_TERM_KP,
            torch.tensor([True, True, False, False]),
            torch.full((4, 1), 30.0, dtype=torch.float32),
            columns=torch.tensor([9], dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="unit wxyz quaternions"):
        transaction.write_mocap_pose(
            "palm",
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([[1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
    transaction.set_state_default(
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([3.0, 4.0], dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="qpos values must be finite"):
        transaction.write_state(
            rows,
            qpos_values=torch.tensor([[float("nan"), 1.0], [2.0, 3.0]], dtype=torch.float32),
        )
    assert transaction.take_verdict() == ()


def test_composer_validation_mode_is_settable_and_validated() -> None:
    transaction = PackedResetTransaction(_ComposerBackend())
    transaction.validation = "strict"
    assert transaction.validation == "strict"
    transaction.validation = "deferred"
    assert transaction.validation == "deferred"
    with pytest.raises(ValueError, match="validation must be one of"):
        transaction.validation = "eager"
    with pytest.raises(ValueError, match="validation must be one of"):
        PackedResetTransaction(_ComposerBackend(), validation="eager")


def test_composer_deferred_columns_stay_memory_safe() -> None:
    backend = _ComposerBackend()
    transaction = PackedResetTransaction(backend)
    transaction.write_model_field_mask(
        RESET_TERM_KP,
        torch.tensor([True, False, False, False]),
        torch.full((4, 1), 42.0, dtype=torch.float32),
        columns=torch.tensor([5], dtype=torch.int64),
    )
    assert transaction.has_pending_verdict()
    staged = transaction._staged[RESET_TERM_KP]
    assert bool(torch.isfinite(staged).all())
