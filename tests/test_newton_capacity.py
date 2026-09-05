from __future__ import annotations

import pytest

from unisim.backend.newton.capacity import (
    NewtonCapacityError,
    calibrate_capacity,
    sample_capacity,
    validate_capacity_limits,
)


def test_capacity_sample_tracks_peak_counts() -> None:
    values = iter(((3, 7), (12, 9), (8, 11)))
    sample = sample_capacity(lambda: next(values), sample_steps=3)
    assert sample.samples == 3
    assert sample.peak_ncon == 12
    assert sample.peak_nefc == 11


def test_capacity_overflow_fails_closed() -> None:
    with pytest.raises(NewtonCapacityError, match="nconmax=8.*ncon=9"):
        calibrate_capacity(lambda: (9, 2), nconmax=8, njmax=4, sample_steps=1)

    with pytest.raises(NewtonCapacityError, match="njmax=4.*nefc=5"):
        calibrate_capacity(lambda: (2, 5), nconmax=8, njmax=4, sample_steps=1)


@pytest.mark.parametrize(
    ("name", "value"),
    [("nconmax", 0), ("njmax", -1), ("nconmax", True)],
)
def test_capacity_limits_require_positive_integers(name: str, value: object) -> None:
    kwargs = {"nconmax": 1, "njmax": 1}
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        validate_capacity_limits(**kwargs)


def test_capacity_reader_shape_is_validated() -> None:
    with pytest.raises(TypeError, match=r"\(ncon, nefc\)"):
        sample_capacity(lambda: (1, 2, 3), sample_steps=1)  # type: ignore[return-value]

    with pytest.raises(ValueError, match="ncon"):
        sample_capacity(lambda: (-1, 0), sample_steps=1)
