"""Fail-closed Genesis DR capability checks for portable CPU sessions."""
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("genesis")
torch = pytest.importorskip("torch")

from tests.adapters.genesis.test_portable_entities import _scene
from unisim.backend.genesis.backend import GenesisBackend
from unisim.dr.types import DomainRandomizationCapabilities


def test_portable_entities_body_torque_mapping_and_reset_cancellation(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    backend = GenesisBackend(scene, 5, 0.002)
    backend.materialize()
    layout = backend.get_scene_layout()
    object_body = layout.get_entity("object").body_ids[0]

    capabilities = backend.get_dr_capabilities()
    if backend._device_dr_supported():
        assert capabilities.supported_reset_terms == frozenset()
        assert capabilities.supported_interval_terms == frozenset()
        assert capabilities.supported_device_reset_terms == {
            "body_mass",
            "body_ipos",
            "base_mass_delta",
            "base_com_offset",
            "kp",
            "kd",
        }
        assert capabilities.supported_device_interval_terms == {
            "body_force",
            "body_torque",
            "body_linear_velocity_delta",
            "body_angular_velocity_delta",
        }
    else:
        assert capabilities == DomainRandomizationCapabilities()
    zero_force = torch.zeros((5, 1, 3), dtype=torch.float32)
    body_ids = torch.asarray((object_body,), dtype=torch.int64)
    if backend._device_dr_supported():
        with pytest.raises(TypeError, match="genesis tensor body_ids must be a torch.Tensor"):
            backend.apply_body_force(
                np.asarray((object_body,)),
                zero_force,
                torque=torch.zeros((5, 3), dtype=torch.float32),
            )
        with pytest.raises(ValueError, match="backend-bound CUDA"):
            backend.apply_body_force(
                body_ids,
                zero_force,
                torque=torch.full((5, 1, 3), float("inf")),
            )
    else:
        with pytest.raises(
            NotImplementedError,
            match="genesis runtime interval randomization requires the exact CUDA",
        ):
            backend.apply_body_force(body_ids, zero_force)
    assert backend._portable_pending_body_forces is not None
    assert backend._portable_pending_body_torques is not None
    assert not bool(backend._portable_pending_body_forces.any().item())
    assert not bool(backend._portable_pending_body_torques.any().item())
