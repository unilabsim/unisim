"""Layout tests for MotrixSim packed tensor I/O."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unisim.backend.motrix.tensor import _field_widths


class _NativeStateBackend:
    _portable_mode = False
    _whole_model_variant_mode = False
    num_actuators = 3
    _model = SimpleNamespace(num_dof_pos=9, num_dof_vel=8)

    def _host_get_state(self, fields: tuple[str, ...] | str | None) -> dict[str, np.ndarray]:
        raise AssertionError("single-MJCF widths must not require a state read")


def test_field_widths_follow_single_mjcf_native_dimensions() -> None:
    assert _field_widths(_NativeStateBackend()) == {  # type: ignore[arg-type]
        "qpos": 9,
        "qvel": 8,
        "ctrl": 3,
    }
