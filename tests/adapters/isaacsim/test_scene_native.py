"""Opt-in real-worker acceptance; this is not public host integration coverage.

Run with UNISIM_TEST_ISAACSIM_SCENE=1 after provisioning the dedicated SDK.
Three N=5 scenes run sequentially; no SDK installation occurs in this test.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest

from unisim.backend.isaacsim.dependencies import build_worker_env, resolve_isaacsim_runtime
from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.backend import _read_exactly_with_deadline
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout

pytestmark = pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACSIM_SCENE") != "1",
    reason="set UNISIM_TEST_ISAACSIM_SCENE=1 for real IsaacSim scene acceptance",
)

_ROBOT = """<mujoco><worldbody><body name="base">
<inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
<geom type="sphere" size=".1"/><body name="tip" pos="0 0 .3">
<joint name="hinge" type="hinge" range="-90 90"/>
<inertial pos="0 0 .1" mass=".5" diaginertia=".005 .005 .005"/>
<geom type="capsule" size=".04 .1"/></body></body></worldbody>
<actuator><position name="motor" joint="hinge" kp="10" kv="1"
forcerange="-100 100"/></actuator></mujoco>"""
_PASSIVE = """<mujoco><worldbody><body name="anchor">
<inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
<geom type="sphere" size=".03"/><body name="pendulum" pos=".15 0 0">
<joint name="passive" type="hinge" axis="0 1 0" range="-170 170"/>
<inertial pos=".1 0 0" mass=".2" diaginertia=".001 .002 .003"/>
<geom type="sphere" size=".03" pos=".1 0 0"/></body></body></worldbody></mujoco>"""
_NUM_ENVS = 5
_OBJECT_ASSIGNMENT = [1, 1, 0, 1, 0]


def _record(mujoco, source, entity):
    model = mujoco.MjModel.from_xml_path(source)
    bodies = [model.body(name).id for name in entity.body_names]
    joints = [model.joint(joint.name).id for joint in entity.joints]
    count = len(joints)
    controlled = bool(entity.actuator_names)
    return {
        "joint_names": [joint.name for joint in entity.joints],
        "actuator_names": list(entity.actuator_names),
        "actuator_joint_names": list(entity.actuator_joint_names),
        "dof_stiffness": [10.0 if controlled else 0.0] * count,
        "dof_damping": [1.0 if controlled else 0.0] * count,
        "dof_effort": [100.0 if controlled else 0.0] * count,
        "dof_armature": [0.0] * count,
        "dof_friction": [0.0] * count,
        "dof_lower": model.jnt_range[joints, 0].tolist(),
        "dof_upper": model.jnt_range[joints, 1].tolist(),
        "body_names": list(entity.body_names),
        "body_mass": model.body_mass[bodies].tolist(),
        "body_ipos": model.body_ipos[bodies].tolist(),
        "body_inertia": model.body_inertia[bodies].tolist(),
        "body_iquat": model.body_iquat[bodies].tolist(),
    }


def _scene(directory: Path, mode: str):
    # Independent source oracle; never use the native importer's own metadata
    # as the expected mass/inertia/topology of the imported asset.
    mujoco = pytest.importorskip("mujoco")

    def write(name, text):
        path = directory / (name + ".xml")
        path.write_text(text, encoding="utf-8")
        return str(path)

    floating_robot = mode == "floating"
    floating_passive = mode == "passive_float"
    robot_xml = (
        _ROBOT.replace('<body name="base">', '<body name="base"><freejoint/>')
        if floating_robot
        else _ROBOT
    )
    sources = [[write("robot", robot_xml)]]
    objects = [
        write(
            f"object{index}",
            f"""<mujoco><worldbody><body name="box"><freejoint/>
    <inertial pos=".02 0 0" mass="{mass}" diaginertia=".01 .02 .03"/>
    <geom type="box" size=".08 .08 .08"/></body></worldbody></mujoco>""",
        )
        for index, mass in enumerate((1, 2))
    ]
    sources.extend(
        [
            objects,
            [
                write(
                    "table",
                    """<mujoco><worldbody><body name="table">
    <inertial pos="0 0 0" mass="5" diaginertia="1 1 1"/>
    <geom type="box" size=".5 .5 .05"/></body></worldbody></mujoco>""",
                )
            ],
            objects,
        ]
    )
    entities = [
        EntityLayout(
            "robot",
            "articulation",
            "floating" if floating_robot else "fixed",
            "base",
            ("base", "tip"),
            (0, 1),
            (None, "base"),
            (JointLayout("hinge", "hinge", (0,), (0,), "tip"),),
            ("motor",),
            ("hinge",),
            (0,),
            tuple(range(8, 15)) if floating_robot else (),
            tuple(range(7, 13)) if floating_robot else (),
        ),
        EntityLayout(
            "object",
            "rigid",
            "floating",
            "box",
            ("box",),
            (2,),
            (None,),
            (),
            (),
            (),
            (),
            tuple(range(1, 8)),
            tuple(range(1, 7)),
        ),
        EntityLayout("table", "rigid", "fixed", "table", ("table",), (3,), (None,), (), (), (), ()),
        EntityLayout(
            "mirror", "rigid", "kinematic", "box", ("box",), (4,), (None,), (), (), (), ()
        ),
    ]
    poses = [
        [0, 0, 0.5, 1, 0, 0, 0],
        [0.3, 0, 1, 1, 0, 0, 0],
        [0.3, 0, 0.3, 1, 0, 0, 0],
        [-0.4, 0, 1, 1, 0, 0, 0],
    ]
    if not floating_robot:
        passive_xml = (
            _PASSIVE.replace('<body name="anchor">', '<body name="anchor"><freejoint/>')
            if floating_passive
            else _PASSIVE
        )
        sources.append([write("passive", passive_xml)])
        poses.append([0.7, 0.2, 0.8, 1, 0, 0, 0])
        entities.append(
            EntityLayout(
                "passive",
                "articulation",
                "floating" if floating_passive else "fixed",
                "anchor",
                ("anchor", "pendulum"),
                (5, 6),
                (None, "anchor"),
                (JointLayout("passive", "hinge", (8,), (7,), "pendulum"),),
                (),
                (),
                (),
                tuple(range(9, 16)) if floating_passive else (),
                tuple(range(8, 14)) if floating_passive else (),
            )
        )
    layout = CompiledSceneLayout(
        tuple(entities),
        15 if floating_robot else 16 if floating_passive else 9,
        13 if floating_robot else 14 if floating_passive else 8,
        1,
        5 if floating_robot else 7,
    )
    entries = [
        {
            "name": entity.name,
            "kind": entity.kind,
            "root_mode": entity.root_mode,
            "asset_format": "mjcf",
            "collision_enabled": entity.name != "mirror",
            "mirror_of": "object" if entity.name == "mirror" else None,
            "initial_pose": poses[index],
            "sources": sources[index],
            "assignment": (
                list(_OBJECT_ASSIGNMENT) if len(sources[index]) > 1 else [0] * _NUM_ENVS
            ),
            "variants": [_record(mujoco, source, entity) for source in sources[index]],
        }
        for index, entity in enumerate(entities)
    ]
    roots = np.zeros((_NUM_ENVS, len(entities), 13), dtype=np.float32)
    roots[:, :, :7] = poses
    qpos, qvel = (
        np.zeros((_NUM_ENVS, layout.nq)),
        np.zeros((_NUM_ENVS, layout.nv)),
    )
    qpos[:, 1:8] = poses[1]
    if floating_robot:
        qpos[:, 8:15] = poses[0]
    if floating_passive:
        qpos[:, 9:16], qvel[:, 7] = poses[-1], 0.2
    return layout, {
        "num_envs": _NUM_ENVS,
        "sim_dt": 0.005,
        "device_id": 0,
        "render_mode": "none",
        "scene_layout": layout.to_dict(),
        "scene_content_identity": {
            "profile": "portable-mjcf-v1",
            "schema_version": 1,
            "source_identity": "a" * 64,
            "compiler_identity": "b" * 64,
            "canonical_identity": "c" * 64,
        },
        "scene_entities": entries,
        "initial_qpos": qpos.tolist(),
        "initial_qvel": qvel.tolist(),
        "initial_roots": roots.tolist(),
        "gravity": [0, 0, -9.81],
    }


class _NativeWorker:
    def __init__(self, directory: Path):
        runtime = resolve_isaacsim_runtime()
        worker = Path(__file__).resolve().parents[3] / "src" / "unisim" / "backend"
        command = [
            str(runtime.python),
            str((worker / "isaacsim" / "worker.py").resolve()),
            "--protocol",
            str(Path(protocol.__file__).resolve()),
        ]
        (directory / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        self.log_path = directory / "stderr.log"
        self.log = self.log_path.open("wb")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            env=build_worker_env(runtime),
            bufsize=0,
        )
        self.memory = []

    def request(self, command, payload=None, timeout=60):
        protocol.send_message(self.process.stdin, command, payload)
        deadline = time.monotonic() + timeout
        header = _read_exactly_with_deadline(self.process.stdout, protocol.HEADER_SIZE, deadline)
        body = _read_exactly_with_deadline(
            self.process.stdout, protocol.unpack_header(header), deadline
        )
        message = protocol.decode_message(body)
        if message["cmd"] == protocol.CMD_ERROR:
            raise AssertionError(f"{message['payload']}\nWorker log: {self.log_path}")
        expected = protocol.CMD_META if command == protocol.CMD_INIT else protocol.CMD_READY
        assert message["cmd"] == expected
        return message.get("payload")

    def attach(self, layout, num_contact_force_sensors: int = 0):
        slots, specs = {}, {}
        for name, shape in protocol.scene_slot_shapes(
            _NUM_ENVS, layout, num_contact_force_sensors
        ).items():
            memory = shared_memory.SharedMemory(
                create=True, size=protocol.slot_allocation_nbytes(name, shape)
            )
            self.memory.append(memory)
            slots[name] = np.ndarray(shape, dtype=protocol.slot_dtype(name), buffer=memory.buf)
            slots[name].fill(0)
            specs[name] = {
                "shm": memory.name,
                "shape": list(shape),
                "dtype": str(protocol.slot_dtype(name)),
            }
        self.request(protocol.CMD_ATTACH, {"slots": specs})
        return slots

    def close(self):
        if self.process.poll() is None:
            try:
                protocol.send_message(self.process.stdin, protocol.CMD_SHUTDOWN)
                self.process.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=10)
        self.process.stdin.close()
        self.process.stdout.close()
        for memory in self.memory:
            memory.close()
            memory.unlink()
        self.log.close()


def test_real_collision_pair_sensor_reports_static_support_force(tmp_path: Path):
    layout, payload = _scene(tmp_path, "passive")
    payload["contact_force_sensors"] = [{
        "name": "object_table",
        "source_entity": "object",
        "source_body": "box",
        "target_entity": "table",
        "target_body": "table",
    }]
    (tmp_path / "init.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    worker = _NativeWorker(tmp_path)
    try:
        worker.request(protocol.CMD_INIT, payload, timeout=240)
        slots = worker.attach(layout, 1)
        worker.request(protocol.CMD_STEP, {"nsteps": 600})
        force = slots["contact_sensor_force"][:, 0].copy()
        assert np.all(np.isfinite(force))
        expected_force = np.asarray([9.81, 19.62])[_OBJECT_ASSIGNMENT]
        np.testing.assert_allclose(force[:, 2], expected_force, rtol=0.15, atol=0.05)
        np.testing.assert_allclose(force[:, :2], 0.0, atol=0.5)
        np.testing.assert_allclose(
            slots["entity_root_state"][:, 1, 2], 0.43, atol=0.02
        )
        (tmp_path / "result.json").write_text(
            json.dumps(
                {
                    "result": "passed",
                    "force": force.tolist(),
                    "object_z": slots["entity_root_state"][:, 1, 2].tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        worker.close()


@pytest.mark.parametrize("mode", ["floating", "passive", "passive_float"])
def test_real_mapped_scene_identity_reset_and_physics(tmp_path: Path, mode: str):
    layout, payload = _scene(tmp_path, mode)
    (tmp_path / "init.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    worker = _NativeWorker(tmp_path)
    try:
        meta = worker.request(protocol.CMD_INIT, payload, timeout=240)
        (tmp_path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        actual = {entry["name"]: entry for entry in meta["scene_entities_actual"]}
        assert (
            actual["object"]["assignment"]
            == actual["mirror"]["assignment"]
            == _OBJECT_ASSIGNMENT
        )
        expected_masses = [[2.0], [2.0], [1.0], [2.0], [1.0]]
        np.testing.assert_allclose(
            actual["object"]["body_mass"], expected_masses, atol=1e-6
        )
        assert meta["scene_layout"]["nu"] == 1
        slots = worker.attach(layout)
        before = slots["entity_root_state"].copy()
        worker.request(protocol.CMD_STEP, {"nsteps": 5})
        fallen = slots["entity_root_state"].copy()
        np.testing.assert_allclose(fallen[:, 1, 9], -9.81 * 0.005 * 5, atol=1e-5)
        assert np.all(fallen[:, 1, 2] < before[:, 1, 2])
        if mode == "floating":
            assert np.all(fallen[:, 0, 2] < before[:, 0, 2])
        else:
            np.testing.assert_allclose(fallen[:, 0, :7], before[:, 0, :7], atol=1e-5)
            assert np.all(np.abs(slots["qvel"][:, 7]) > 0.001)
        slots["reset_env_ids"][0] = 1
        slots["reset_qpos"][0] = slots["qpos"][1]
        slots["reset_qvel"][0] = slots["qvel"][1]
        slots["reset_entity_root_state"][0] = fallen[1]
        slots["reset_entity_root_state"][0, 1, :3] = [0.1, 0.2, 1.7]
        slots["reset_entity_root_state"][0, 1, 7:] = 0
        slots["reset_qpos"][0, 1:8] = [0.1, 0.2, 1.7, 1, 0, 0, 0]
        slots["reset_qvel"][0, 1:7] = 0
        slots["reset_qpos_mask"][1:8] = 1
        slots["reset_qvel_mask"][1:7] = 1
        slots["reset_root_mask"][1] = 1
        worker.request(protocol.CMD_RESET_ENTITIES, {"count": 1, "entity_names": ["object"]})
        after = slots["entity_root_state"].copy()
        np.testing.assert_allclose(after[1, 1, :3], [0.1, 0.2, 1.7], atol=1e-5)
        np.testing.assert_array_equal(after[0], fallen[0])
        other_entities = [i for i in range(len(layout.entities)) if i != 1]
        np.testing.assert_array_equal(after[:, other_entities], fallen[:, other_entities])
        worker.request(protocol.CMD_STEP, {"nsteps": 1})
        assert slots["entity_root_state"][1, 1, 2] > 1.69
        # Nonidentity orientation and offset COM distinguish link and COM velocity.
        slots["reset_entity_root_state"][0, 1, :7] = [
            0.1,
            0.2,
            1.7,
            np.sqrt(0.5),
            0,
            0,
            np.sqrt(0.5),
        ]
        slots["reset_entity_root_state"][0, 1, 7:] = [0.1, 0.2, 0.3, 1, 2, 3]
        slots["reset_qpos"][0, 1:8] = slots["reset_entity_root_state"][0, 1, :7]
        slots["reset_qvel"][0, 1:7] = [0.1, 0.2, 0.3, 2, -1, 3]
        worker.request(protocol.CMD_RESET_ENTITIES, {"count": 1, "entity_names": ["object"]})
        np.testing.assert_allclose(
            slots["entity_root_state"][1, 1, 7:], [0.1, 0.2, 0.3, 1, 2, 3], atol=1e-5
        )
        np.testing.assert_allclose(slots["qvel"][1, 1:7], [0.1, 0.2, 0.3, 2, -1, 3], atol=1e-5)
        # Co-locate the collision-free mirror and compare against the far baseline.
        slots["reset_env_ids"][:] = np.arange(slots["reset_env_ids"].shape[0])
        slots["reset_qpos"][:] = slots["qpos"]
        slots["reset_qvel"][:] = slots["qvel"]
        slots["reset_entity_root_state"][:] = slots["entity_root_state"]
        slots["reset_entity_root_state"][:, 1] = before[:, 1]
        slots["reset_entity_root_state"][:, 3, :7] = before[:, 1, :7]
        slots["reset_qpos"][:, 1:8] = before[:, 1, :7]
        slots["reset_qvel"][:, 1:7] = 0
        slots["reset_root_mask"][3, 0] = 1
        worker.request(
            protocol.CMD_RESET_ENTITIES, {"count": 5, "entity_names": ["object", "mirror"]}
        )
        worker.request(protocol.CMD_STEP, {"nsteps": 5})
        near = slots["entity_root_state"][:, 1].copy()
        np.testing.assert_allclose(near, fallen[:, 1], atol=1e-5)
        slots["ctrl"][:, 0] = 0.4
        worker.request(protocol.CMD_STEP, {"nsteps": 40})
        assert np.all(slots["qpos"][:, 0] > 0.01)
        result = {
            "initial": before.tolist(),
            "fallen": fallen.tolist(),
            "reset": after.tolist(),
            "mirror_near": near.tolist(),
            "controlled_hinge": slots["qpos"][:, 0].tolist(),
        }
        (tmp_path / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        worker.close()
