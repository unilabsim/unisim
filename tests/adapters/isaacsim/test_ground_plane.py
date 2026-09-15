"""Tests for the declarative ground-plane channel (Fix-A, review I-1).

Pure-Python coverage only: the ``GroundPlaneSceneCfg`` contract (scene.py),
the ``SceneCfg.ground_plane`` default, the host-side INIT serialization in
``MjcfSubprocessBackend.materialize()`` (both the undeclared ``None`` branch
and the declared dict branch), and the worker's wire-boundary re-validation.
The USD spawn itself runs inside the external Kit worker and is exercised by
the Kit probes.
"""

from __future__ import annotations

import io
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest

from unisim.backend.isaacsim.worker import parse_ground_plane_declaration
from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.backend import MjcfSubprocessBackend
from unisim.scene import GroundPlaneSceneCfg, SceneCfg

ROBOT_URDF = """<?xml version="1.0"?>
<robot name="robot">
  <link name="base_link"/>
  <link name="arm"/>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/><child link="arm"/>
    <limit lower="-1.57" upper="1.57" effort="300" velocity="10"/>
  </joint>
</robot>
"""

NUM_ENVS = 2
# Robot scan: one revolute joint; bodies base_link/arm (declaration order).
NUM_DOF = 1
NUM_BODIES = 2


@pytest.fixture()
def robot_file(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text(ROBOT_URDF)
    return path


# ---------------------------------------------------------------------------
# GroundPlaneSceneCfg contract
# ---------------------------------------------------------------------------


def test_ground_plane_cfg_defaults_mirror_ground_plane_cfg_material():
    cfg = GroundPlaneSceneCfg()
    assert cfg.friction == (0.5, 0.5, 0.0)
    assert cfg.restitution == 0.0
    assert cfg.size_m == 200.0


def test_ground_plane_cfg_normalizes_triple_and_scalars():
    cfg = GroundPlaneSceneCfg(friction=[1, "0.5", 0.0], restitution="0.2", size_m="40")
    assert cfg.friction == (1.0, 0.5, 0.0)
    assert cfg.restitution == 0.2
    assert cfg.size_m == 40.0


def test_ground_plane_cfg_friction_triple_fails_closed():
    with pytest.raises(TypeError, match="triple"):
        GroundPlaneSceneCfg(friction="slippery")
    with pytest.raises(ValueError, match="exactly 3"):
        GroundPlaneSceneCfg(friction=(0.5, 0.5))
    with pytest.raises(ValueError, match="non-negative"):
        GroundPlaneSceneCfg(friction=(0.5, -0.5, 0.0))
    with pytest.raises(ValueError, match="finite"):
        GroundPlaneSceneCfg(friction=(0.5, math.nan, 0.0))


def test_ground_plane_cfg_restitution_and_size_fail_closed():
    with pytest.raises(ValueError, match="restitution"):
        GroundPlaneSceneCfg(restitution=-0.1)
    with pytest.raises(ValueError, match="finite"):
        GroundPlaneSceneCfg(restitution=math.inf)
    with pytest.raises(ValueError, match="size_m"):
        GroundPlaneSceneCfg(size_m=0.0)
    with pytest.raises(ValueError, match="size_m"):
        GroundPlaneSceneCfg(size_m=-200.0)
    with pytest.raises(ValueError, match="finite"):
        GroundPlaneSceneCfg(size_m=math.nan)


def test_scene_cfg_ground_plane_defaults_to_none(robot_file):
    assert SceneCfg(model_file=str(robot_file)).ground_plane is None


# ---------------------------------------------------------------------------
# Host serialization: the INIT payload carries the ground_plane key
# ---------------------------------------------------------------------------


class _FakeWorkerProcess:
    """Pipe-bearing ``Popen`` stand-in so materialize() spawns no real worker."""

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        self.returncode = 0

    def poll(self):
        return None

    def wait(self, timeout=None):
        del timeout
        return 0

    def terminate(self):
        return None

    def kill(self):
        return None


class _FakeWorkerBackend(MjcfSubprocessBackend):
    """Record every worker request and answer the INIT/ATTACH handshake."""

    def _supports_ground_plane(self) -> bool:
        # The serialization under test belongs to the workers that consume
        # the declaration; the family default rejects declared scenes.
        return True

    def __init__(self, scene: SceneCfg, num_envs: int = NUM_ENVS):
        self.requests: list[tuple[str, dict]] = []
        super().__init__(scene, num_envs=num_envs, sim_dt=0.01)

    def _worker_entrypoint(self) -> Path:
        return Path(__file__)

    def _resolve_worker_runtime(self):
        return types.SimpleNamespace(python=sys.executable)

    def _build_worker_environment(self, runtime):
        del runtime
        return {}

    def _request(self, cmd, payload, *, expect):
        del expect
        self.requests.append((cmd, payload))
        if cmd == protocol.CMD_INIT:
            return {
                "num_dof": NUM_DOF,
                "num_bodies": NUM_BODIES,
                "dof_names": ["shoulder"],
                "body_names": ["base_link", "arm"],
                "gravity": [0.0, 0.0, -9.81],
            }
        return {}


def _init_ground_plane(robot_file, monkeypatch, ground_plane):
    monkeypatch.setattr(subprocess, "Popen", _FakeWorkerProcess)
    backend = _FakeWorkerBackend(
        SceneCfg(model_file=str(robot_file), ground_plane=ground_plane)
    )
    backend.materialize()
    for cmd, payload in backend.requests:
        if cmd == protocol.CMD_INIT:
            return payload
    raise AssertionError("materialize() never sent INIT")


def test_init_payload_carries_ground_plane_none_when_undeclared(robot_file, monkeypatch):
    payload = _init_ground_plane(robot_file, monkeypatch, None)
    assert payload["ground_plane"] is None


def test_init_payload_serializes_declared_ground_plane(robot_file, monkeypatch):
    payload = _init_ground_plane(
        robot_file,
        monkeypatch,
        GroundPlaneSceneCfg(friction=(0.6, 0.4, 0.0), restitution=0.1, size_m=50.0),
    )
    assert payload["ground_plane"] == {
        "friction": [0.6, 0.4, 0.0],
        "restitution": 0.1,
        "size_m": 50.0,
    }


def test_init_payload_serializes_default_declaration(robot_file, monkeypatch):
    payload = _init_ground_plane(robot_file, monkeypatch, GroundPlaneSceneCfg())
    assert payload["ground_plane"] == {
        "friction": [0.5, 0.5, 0.0],
        "restitution": 0.0,
        "size_m": 200.0,
    }


# ---------------------------------------------------------------------------
# Worker wire re-validation
# ---------------------------------------------------------------------------


def test_parse_ground_plane_declaration_none_is_legacy():
    assert parse_ground_plane_declaration(None) is None


def test_parse_ground_plane_declaration_valid_entry():
    parsed = parse_ground_plane_declaration(
        {"friction": [0.5, 0.5, 0.0], "restitution": 0.0, "size_m": 200.0}
    )
    assert parsed == ((0.5, 0.5, 0.0), 0.0, 200.0)


def test_parse_ground_plane_declaration_malformed_fails_closed():
    with pytest.raises(TypeError, match="must be a dict"):
        parse_ground_plane_declaration([0.5, 0.5, 0.0])
    with pytest.raises(ValueError, match="exactly"):
        parse_ground_plane_declaration({"friction": [0.5, 0.5, 0.0]})
    with pytest.raises(ValueError, match="friction/restitution/size_m"):
        parse_ground_plane_declaration(
            {"friction": [0.5, 0.5, 0.0], "restitution": 0.0, "size_m": 200.0, "extra": 1}
        )
    with pytest.raises(ValueError, match="ground_plane friction"):
        parse_ground_plane_declaration(
            {"friction": [0.5, -0.5, 0.0], "restitution": 0.0, "size_m": 200.0}
        )
    with pytest.raises(ValueError, match="restitution"):
        parse_ground_plane_declaration(
            {"friction": [0.5, 0.5, 0.0], "restitution": -1.0, "size_m": 200.0}
        )
    with pytest.raises(ValueError, match="size_m"):
        parse_ground_plane_declaration(
            {"friction": [0.5, 0.5, 0.0], "restitution": 0.0, "size_m": 0.0}
        )
