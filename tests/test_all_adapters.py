import pytest

import unisim
from unisim.optional import OptionalDependencyError


def test_manifest_covers_all_current_backends():
    names = {spec.name for spec in unisim.ADAPTER_SPECS}
    assert names == {"mujoco", "motrix", "drake", "mjwarp", "genesis", "isaacgym", "isaacsim"}
    assert all(spec.status == "available" for spec in unisim.ADAPTER_SPECS)


@pytest.mark.parametrize("backend_type", ["mujoco", "motrix", "drake", "mjwarp", "genesis"])
def test_in_process_adapters_require_scene(backend_type):
    with pytest.raises(ValueError, match="requires a SceneCfg"):
        unisim.create_backend(backend_type)


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
