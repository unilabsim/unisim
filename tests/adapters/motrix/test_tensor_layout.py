"""Layout tests for MotrixSim packed tensor I/O."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from unisim.backend.motrix.tensor import _field_widths


class _PublicStateBackend:
    """Expose public state narrower than native model coordinate metadata."""

    _portable_mode = False
    num_actuators = 3
    _model = SimpleNamespace(num_dof_pos=9, num_dof_vel=9)

    def get_state(self, fields: tuple[str, ...] | str | None) -> dict[str, Any]:
        assert fields == ("qpos", "qvel")
        return {
            "qpos": np.empty((2, 5), dtype=np.float32),
            "qvel": np.empty((2, 4), dtype=np.float32),
        }


def test_field_widths_follow_public_state_blocks_not_model_metadata() -> None:
    assert _field_widths(_PublicStateBackend()) == {  # type: ignore[arg-type]
        "qpos": 5,
        "qvel": 4,
        "ctrl": 3,
    }
