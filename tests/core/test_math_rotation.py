"""Numerical coverage for the unified ``unisim.utils.math`` rotation primitives."""

from __future__ import annotations

import numpy as np
import pytest

from unisim.utils import math as um


def _random_unit_quats(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(count, 4))
    return quats / np.linalg.norm(quats, axis=-1, keepdims=True)


def _reference_quat_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Pre-migration reference formula (kept local to pin numerical equivalence)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * (w * uv + uuv)


def test_quat_apply_matches_reference_formula_and_inverts() -> None:
    quats = _random_unit_quats(8)
    vectors = np.random.default_rng(1).normal(size=(8, 3))
    np.testing.assert_allclose(
        um.np_quat_apply(quats, vectors), _reference_quat_rotate(quats, vectors)
    )
    np.testing.assert_allclose(
        um.np_quat_apply_inverse(quats, um.np_quat_apply(quats, vectors)), vectors, atol=1e-12
    )


def test_quat_mul_compose_known_rotation() -> None:
    angle = np.pi / 3
    quat = np.asarray((np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0))
    vector = np.asarray((1.0, 0.0, 0.0))
    np.testing.assert_allclose(
        um.np_quat_apply(quat, vector), (0.5, 0.0, -np.sqrt(3) / 2), atol=1e-12
    )
    np.testing.assert_allclose(
        um.np_quat_mul(quat, quat), (np.cos(angle), 0.0, np.sin(angle), 0.0), atol=1e-12
    )


def test_wxyz_convention_and_batch_broadcast_semantics() -> None:
    quats = _random_unit_quats(5)
    single = _random_unit_quats(1)[0]

    # (4,) x (4,) stays non-batched; trailing dimension carries wxyz.
    assert um.np_quat_mul(single, single).shape == (4,)
    assert um.np_quat_apply(single, np.zeros(3)).shape == (3,)

    # Mixed scalar/batch inputs broadcast over leading dimensions.
    assert um.np_quat_mul(single, quats).shape == (5, 4)
    assert um.np_quat_mul(quats, single).shape == (5, 4)
    np.testing.assert_allclose(
        um.np_quat_mul(single, quats), um.np_quat_mul(np.stack([single] * 5), quats)
    )

    # Multi-rank broadcast: (2, 1, 4) x (3, 4) -> (2, 3, 4).
    left = _random_unit_quats(2, seed=3)[:, None, :]
    right = _random_unit_quats(3, seed=4)
    assert um.np_quat_mul(left, right).shape == (2, 3, 4)
    assert um.np_quat_apply(left, np.zeros(3)).shape == (2, 1, 3)
    assert um.np_quat_apply(single, np.zeros((2, 3, 3))).shape == (2, 3, 3)

    # Quaternion order is w-first (wxyz), not x-first.
    assert um.np_yaw_to_quat(np.pi / 2)[0] == pytest.approx(np.cos(np.pi / 4))
    with pytest.raises(ValueError):
        um.np_quat_mul(np.zeros(5), quats)
    with pytest.raises(ValueError):
        um.np_quat_apply(quats, np.zeros((5, 4)))


def test_quat_conjugate_canonicalize_and_error_magnitudes() -> None:
    quats = _random_unit_quats(6)
    np.testing.assert_allclose(um.np_quat_conjugate(quats)[..., 0], quats[..., 0])
    np.testing.assert_allclose(
        um.np_quat_mul(quats, um.np_quat_conjugate(quats)),
        np.broadcast_to(np.asarray((1.0, 0.0, 0.0, 0.0)), (6, 4)),
        atol=1e-12,
    )

    flipped = -quats
    canonical = um.np_quat_canonicalize(flipped)
    assert np.all(canonical[..., 0] >= 0.0)
    np.testing.assert_allclose(canonical, um.np_quat_canonicalize(quats))
    np.testing.assert_allclose(um.np_quat_error_magnitude(quats, flipped), np.zeros(6), atol=1e-12)

    quarter_turn = um.np_yaw_to_quat(np.full(6, np.pi / 2))
    np.testing.assert_allclose(
        um.np_quat_error_magnitude(quats, um.np_quat_mul(quarter_turn, quats)),
        np.full(6, np.pi / 2),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        um.np_quat_error_magnitude_squared(quats, um.np_quat_mul(quarter_turn, quats)),
        np.full(6, (np.pi / 2) ** 2),
        rtol=1e-12,
    )
    # Scalar pairs return 0-d results.
    assert um.np_quat_error_magnitude(quats[0], quats[1]).shape == ()


def test_euler_matrix_and_axis_angle_roundtrips() -> None:
    roll, pitch, yaw = 0.3, -0.2, 1.1
    quat = um.np_quat_from_euler_xyz(roll, pitch, yaw)
    assert quat.shape == (4,)

    matrix = um.np_matrix_from_quat(quat)
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    vector = np.asarray((0.4, -0.5, 0.7))
    np.testing.assert_allclose(um.np_quat_apply(quat, vector), matrix @ vector, atol=1e-12)

    batch = um.np_quat_from_euler_xyz(
        np.asarray([roll, 0.0]), np.asarray([pitch, 0.1]), np.asarray([yaw, -0.4])
    )
    assert batch.shape == (2, 4)
    np.testing.assert_allclose(batch[0], quat, atol=1e-12)
    assert um.np_matrix_from_quat(batch).shape == (2, 3, 3)

    quats = _random_unit_quats(4)
    axis_angle = um.np_quat_to_axis_angle(quats)
    angle = np.linalg.norm(axis_angle, axis=-1)
    axis = axis_angle / np.where(angle < 1e-12, 1.0, angle)[:, None]
    rebuilt = um.np_quat_from_angle_axis(angle, axis)
    # Quaternion sign is ambiguous; compare through the relative rotation.
    np.testing.assert_allclose(um.np_quat_error_magnitude(quats, rebuilt), np.zeros(4), atol=1e-10)

    cols = um.np_matrix_first_two_cols_from_quat(quats)
    np.testing.assert_allclose(
        cols, um.np_matrix_from_quat(quats)[:, :, :2].reshape(4, 6), atol=1e-12
    )
    assert um.np_matrix_first_two_cols_from_quat(quats[0]).shape == (6,)


def test_yaw_helpers_and_wrap_to_pi() -> None:
    quats = um.np_quat_from_euler_xyz(
        np.asarray([0.2, -0.1]), np.asarray([0.1, 0.3]), np.asarray([0.7, -1.2])
    )
    np.testing.assert_allclose(um.np_yaw_from_quat(quats), (0.7, -1.2), atol=1e-12)
    yaw_only = um.np_yaw_quat(quats)
    np.testing.assert_allclose(yaw_only, um.np_yaw_to_quat(np.asarray([0.7, -1.2])), atol=1e-12)
    np.testing.assert_allclose(um.np_yaw_from_quat(um.np_yaw_to_quat(0.3)), 0.3, atol=1e-12)

    np.testing.assert_allclose(um.np_wrap_to_pi(np.asarray([3.0 * np.pi, -0.5])), (-np.pi, -0.5))
    np.testing.assert_allclose(um.np_wrap_to_pi(0.25), 0.25, atol=1e-12)


def test_sequence_helpers_continuity_and_angular_velocity() -> None:
    dt = 0.01
    rate = 2.0
    times = np.arange(8) * dt
    seq = um.np_yaw_to_quat(rate * times)
    flipped = seq.copy()
    flipped[3:5] *= -1.0
    continuous = um.np_quat_ensure_continuity(flipped)
    assert np.all((continuous * seq).sum(axis=-1) > 0.0)

    omega = um.np_quat_angular_velocity(flipped, dt)
    assert omega.shape == (8, 3)
    np.testing.assert_allclose(omega[:, 0], 0.0, atol=1e-9)
    np.testing.assert_allclose(omega[:, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(omega[:, 2], rate, atol=1e-9)

    np.testing.assert_allclose(um.np_quat_angular_velocity(seq[:1], dt), np.zeros((1, 3)))
    np.testing.assert_allclose(
        um.np_quat_angular_velocity(seq[:2], dt),
        np.broadcast_to((0.0, 0.0, rate), (2, 3)),
        atol=1e-9,
    )
    with pytest.raises(ValueError):
        um.np_quat_angular_velocity(seq, 0.0)
    with pytest.raises(ValueError):
        um.np_quat_ensure_continuity(seq[0])


def test_subtract_frame_transforms() -> None:
    quats = _random_unit_quats(3)
    pos = np.random.default_rng(5).normal(size=(3, 3))
    other_quats = _random_unit_quats(3, seed=6)
    other_pos = np.random.default_rng(7).normal(size=(3, 3))

    rel_pos, rel_quat = um.np_subtract_frame_transforms(pos, quats, other_pos, other_quats)
    np.testing.assert_allclose(
        rel_pos, um.np_quat_apply_inverse(quats, other_pos - pos), atol=1e-12
    )
    np.testing.assert_allclose(
        rel_quat, um.np_quat_mul(um.np_quat_conjugate(quats), other_quats), atol=1e-12
    )

    anchor_pos = pos[:, None, :]
    body_pos = np.broadcast_to(pos[:, None, :] + 1.0, (3, 2, 3)).copy()
    body_quat = np.broadcast_to(other_quats[:, None, :], (3, 2, 4)).copy()
    anchor_rel_pos, anchor_rel_quat = um.np_subtract_anchor_frame_transforms(
        pos, quats, body_pos, body_quat
    )
    assert anchor_pos.shape == (3, 1, 3)
    assert anchor_rel_pos.shape == (3, 2, 3)
    assert anchor_rel_quat.shape == (3, 2, 4)
    np.testing.assert_allclose(
        anchor_rel_pos, um.np_quat_apply_inverse(quats[:, None, :], body_pos - pos[:, None, :])
    )
    np.testing.assert_allclose(
        anchor_rel_quat,
        um.np_quat_mul(um.np_quat_conjugate(quats)[:, None, :], body_quat),
        atol=1e-12,
    )


def test_torch_primitives_match_numpy() -> None:
    torch = pytest.importorskip("torch")

    quats = _random_unit_quats(7)
    others = _random_unit_quats(7, seed=11)
    vectors = np.random.default_rng(12).normal(size=(7, 3))
    t_quats = torch.as_tensor(quats, dtype=torch.float64)
    t_others = torch.as_tensor(others, dtype=torch.float64)
    t_vectors = torch.as_tensor(vectors, dtype=torch.float64)

    def as_numpy(value: "object") -> np.ndarray:
        return np.asarray(value.detach().cpu())

    np.testing.assert_allclose(
        as_numpy(um.quat_mul(t_quats, t_others)), um.np_quat_mul(quats, others)
    )
    np.testing.assert_allclose(
        as_numpy(um.quat_mul(t_quats[0], t_others)), um.np_quat_mul(quats[0], others)
    )
    np.testing.assert_allclose(as_numpy(um.quat_conjugate(t_quats)), um.np_quat_conjugate(quats))
    np.testing.assert_allclose(
        as_numpy(um.quat_apply(t_quats, t_vectors)), um.np_quat_apply(quats, vectors)
    )
    np.testing.assert_allclose(
        as_numpy(um.quat_apply_inverse(t_quats, t_vectors)),
        um.np_quat_apply_inverse(quats, vectors),
    )
    np.testing.assert_allclose(
        as_numpy(um.quat_canonicalize(-t_quats)), um.np_quat_canonicalize(-quats)
    )
    np.testing.assert_allclose(
        as_numpy(um.axis_angle_from_quat(t_quats)), um.np_quat_to_axis_angle(quats), atol=1e-10
    )

    angles = np.linalg.norm(um.np_quat_to_axis_angle(quats), axis=-1)
    axes = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    np.testing.assert_allclose(
        as_numpy(
            um.quat_from_angle_axis(
                torch.as_tensor(angles, dtype=torch.float64),
                torch.as_tensor(axes, dtype=torch.float64),
            )
        ),
        um.np_quat_from_angle_axis(angles, axes),
        atol=1e-10,
    )

    yaws = np.asarray([0.3, -1.2, 2.5, 0.0, 1.1, -0.7, 3.0])
    t_yaws = torch.as_tensor(yaws, dtype=torch.float64)
    np.testing.assert_allclose(as_numpy(um.yaw_to_quat(t_yaws)), um.np_yaw_to_quat(yaws))
    np.testing.assert_allclose(as_numpy(um.yaw_from_quat(t_quats)), um.np_yaw_from_quat(quats))
    np.testing.assert_allclose(as_numpy(um.yaw_quat(t_quats)), um.np_yaw_quat(quats))
    np.testing.assert_allclose(as_numpy(um.wrap_to_pi(t_yaws * 5)), um.np_wrap_to_pi(yaws * 5))

    np.testing.assert_allclose(
        as_numpy(um.matrix_from_quat(t_quats)), um.np_matrix_from_quat(quats)
    )
    np.testing.assert_allclose(
        as_numpy(um.matrix_first_two_cols_from_quat(t_quats)),
        um.np_matrix_first_two_cols_from_quat(quats),
    )
    np.testing.assert_allclose(
        as_numpy(um.quat_to_rot6(t_quats)), um.np_matrix_first_two_cols_from_quat(quats)
    )
    np.testing.assert_allclose(
        as_numpy(um.quat_error_magnitude(t_quats, t_others)),
        um.np_quat_error_magnitude(quats, others),
    )

    roll = torch.as_tensor(np.asarray([0.2, -0.4]), dtype=torch.float64)
    pitch = torch.as_tensor(np.asarray([0.1, 0.5]), dtype=torch.float64)
    yaw = torch.as_tensor(np.asarray([0.7, -1.2]), dtype=torch.float64)
    np.testing.assert_allclose(
        as_numpy(um.quat_from_euler_xyz(roll, pitch, yaw)),
        um.np_quat_from_euler_xyz(roll.numpy(), pitch.numpy(), yaw.numpy()),
    )
    euler_quats = um.quat_from_euler_xyz(roll, pitch, yaw)
    t_roll, t_pitch = um.quat_roll_pitch(euler_quats)
    np.testing.assert_allclose(as_numpy(t_roll), roll.numpy(), atol=1e-12)
    np.testing.assert_allclose(as_numpy(t_pitch), pitch.numpy(), atol=1e-12)

    seq = um.np_yaw_to_quat(np.arange(6) * 0.05)
    seq[2:4] *= -1.0
    t_seq = torch.as_tensor(seq, dtype=torch.float64)
    np.testing.assert_allclose(
        as_numpy(um.quat_ensure_continuity(t_seq)), um.np_quat_ensure_continuity(seq)
    )
    np.testing.assert_allclose(
        as_numpy(um.quat_angular_velocity(t_seq, 0.01)), um.np_quat_angular_velocity(seq, 0.01)
    )
