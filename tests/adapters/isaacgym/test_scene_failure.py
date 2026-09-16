"""Framed worker failures distinguish validation from an incomplete native commit."""

from pathlib import Path

import numpy as np
import pytest

from tests.adapters.isaacgym.test_fixed_variants import _make_backend, _write_variants
from unisim.backend.subprocess_ipc.backend import SubprocessWorkerError


@pytest.mark.parametrize("phase", ["validation", "native"])
def test_failed_reset_poisoning_is_explicit_and_prevents_followup_step(
    tmp_path: Path, phase
) -> None:
    backend = _make_backend(_write_variants(tmp_path), (0, 1, 2), tmp_path / "init.json")
    backend._worker_command += ["--reset-error", phase]
    try:
        backend.materialize()
        ids = np.array([1], dtype=np.int32)
        qpos = backend.get_default_qpos()[None, :]
        qvel = backend.get_init_qvel()[None, :]
        with pytest.raises(SubprocessWorkerError, match="injected reset " + phase):
            backend.set_state(ids, qpos, qvel)
        control = np.zeros((3, backend.num_actuators), dtype=np.float32)
        if phase == "native":
            with pytest.raises(SubprocessWorkerError, match="earlier failure"):
                backend.step(control)
            assert backend._proc.poll() is not None
        else:
            backend.step(control)
            assert backend._proc.poll() is None
    finally:
        backend.close()
