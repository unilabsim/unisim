from __future__ import annotations

import numpy as np
import pytest

from unisim.backend.body_state import copy_selected_body_state


@pytest.mark.parametrize(
    "selected_ids",
    [
        np.array([2, 3, 4], dtype=np.intp),
        np.array([4, 1, 3], dtype=np.intp),
        np.array([], dtype=np.intp),
    ],
)
def test_copy_selected_body_state_preserves_selection(selected_ids: np.ndarray) -> None:
    widths = (3, 4, 3, 3)
    sources = tuple(
        np.arange(2 * 6 * width, dtype=np.float32).reshape(2, 6, width) + index
        for index, width in enumerate(widths)
    )
    outputs = tuple(np.empty((2, selected_ids.size, width), dtype=np.float32) for width in widths)

    copy_selected_body_state(*sources, selected_ids, *outputs)

    for source, output in zip(sources, outputs, strict=True):
        np.testing.assert_array_equal(output, source[:, selected_ids])


def test_copy_selected_body_state_retains_assignment_casting() -> None:
    source = np.array([[[1.25, 2.5, 3.75]]], dtype=np.float64)
    output = np.empty((1, 1, 3), dtype=np.float32)

    copy_selected_body_state(
        source, source, source, source, np.array([0]), output, output, output, output
    )

    assert output.dtype == np.float32
    np.testing.assert_array_equal(output, source.astype(np.float32))
