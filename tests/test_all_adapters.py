import numpy as np
import pytest

import unisim
from unisim.optional import OptionalDependencyError


class _Runtime:
    num_envs = 2
    num_actuators = 3

    def __init__(self):
        self.qpos = np.zeros((2, 3), dtype=np.float64)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.qpos.fill(0)
        else:
            self.qpos[np.asarray(env_ids)] = 0

    def step(self, ctrl, nsteps=1):
        self.qpos += np.asarray(ctrl) * nsteps

    def get_state(self):
        return {"qpos": self.qpos}


def test_manifest_covers_all_current_backends():
    names = {spec.name for spec in unisim.ADAPTER_SPECS}
    assert names == {"mujoco", "motrix", "drake", "mjwarp", "genesis", "isaacgym", "isaacsim"}
    assert all(spec.status == "available" for spec in unisim.ADAPTER_SPECS)


@pytest.mark.parametrize(
    "backend_type",
    ["drake", "mjwarp", "genesis"],
)
def test_optional_adapter_runtime_bridge(backend_type):
    backend = unisim.create_backend(backend_type, runtime=_Runtime())
    backend.step(np.ones((2, 3)), nsteps=2)
    assert backend.get_state("qpos")["qpos"].shape == (2, 3)


@pytest.mark.parametrize("backend_type", ["isaacgym", "isaacsim"])
def test_worker_adapters_fail_closed_without_runtime(backend_type, monkeypatch):
    monkeypatch.delenv("UNISIM_ISAACGYM_PYTHON", raising=False)
    monkeypatch.delenv("UNISIM_ISAACSIM_PYTHON", raising=False)
    with pytest.raises(OptionalDependencyError):
        unisim.create_backend(backend_type)


def test_protocol_round_trip():
    from unisim.backend.subprocess_ipc.protocol import decode_message, pack_message

    assert decode_message(pack_message("PING", {"value": 1})) == {
        "cmd": "PING",
        "payload": {"value": 1},
    }
