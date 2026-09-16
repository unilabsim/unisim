from __future__ import annotations

import numpy as np
import pytest

from unisim import FakeBackend


@pytest.mark.parametrize(
    "indices",
    [[0.5], [-0.1], [True], [[0, 1]], 0, ["1"], [-1], [3], np.array([2**63], dtype=np.uint64)],
)
def test_environment_query_rejects_invalid_indices(indices) -> None:
    backend = FakeBackend(num_envs=3)
    with pytest.raises(ValueError, match="env_ids"):
        backend._validate_env_ids(indices)


def test_environment_query_preserves_order_duplicates_and_empty_selection() -> None:
    backend = FakeBackend(num_envs=3)
    np.testing.assert_array_equal(backend._validate_env_ids([2, 0, 2]), [2, 0, 2])
    assert backend._validate_env_ids([]).shape == (0,)
