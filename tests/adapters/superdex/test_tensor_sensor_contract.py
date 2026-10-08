"""SDK-free checks for SuperDex's public deterministic sensor namespace."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from unisim.backend.superdex.backend import SuperDexBackend


def _backend_view() -> SuperDexBackend:
    backend = SuperDexBackend.__new__(SuperDexBackend)
    untyped_backend: Any = backend
    backend._pid = os.getpid()
    backend._closed = False
    backend._entity_faulted = False
    backend._unsupported_sensors = {
        "acceleration": (
            "SuperDex declares accelerometers but does not expose "
            "instantaneous point acceleration; no substitute is published"
        )
    }
    backend._sensor_values = {"authored": np.array([[1.0, 2.0, 3.0]])}
    backend._body_lookup = {"world": 0, "base": 1}
    untyped_backend._pos = np.arange(6, dtype=np.float64).reshape(1, 2, 3)
    untyped_backend._quat = np.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    untyped_backend._lin = untyped_backend._pos + 10
    untyped_backend._ang = untyped_backend._pos + 20
    return backend


def test_tracked_body_sensors_are_public_body_state_views():
    backend = _backend_view()

    np.testing.assert_allclose(backend.get_sensor_data("track_pos_w_base"), [[3, 4, 5]])
    np.testing.assert_allclose(backend.get_sensor_data("track_quat_w_base"), [[0, 1, 0, 0]])
    np.testing.assert_allclose(backend.get_sensor_data("track_linvel_w_base"), [[13, 14, 15]])
    np.testing.assert_allclose(backend.get_sensor_data("track_angvel_w_base"), [[23, 24, 25]])
    np.testing.assert_allclose(backend.get_sensor_data("authored"), [[1, 2, 3]])


def test_tracked_body_and_accelerometer_names_fail_closed():
    backend = _backend_view()

    with pytest.raises(KeyError, match="unknown SuperDex tracked body sensor"):
        backend.get_sensor_data("track_pos_w_world")
    with pytest.raises(KeyError, match="unknown SuperDex tracked body sensor"):
        backend.get_sensor_data("track_pos_w_missing")
    with pytest.raises(NotImplementedError, match="no substitute is published"):
        backend.get_sensor_data("acceleration")
