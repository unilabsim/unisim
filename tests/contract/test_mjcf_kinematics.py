"""MJCF kinematic-tree scan and worker FK kernel coverage (issue #141).

The subprocess Isaac workers publish forward-kinematics body state between a
reset and the first physics step because PhysX keeps pre-write link poses
until ``simulate``.  These tests pin the host-side scan order, the wire
payload validation, and the kernel's MuJoCo semantics (offset before joint
rotation, body-reference-frame axes, world-origin/body-local root velocity).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from unisim.backend.subprocess_ipc import protocol, sensors
from unisim.backend.subprocess_ipc.kinematics import (
    forward_kinematics,
    forward_prepared_kinematics,
    prepare_kinematics,
)
from unisim.backend.subprocess_ipc.sensors import scan_scene_kinematics

_MODEL = """<mujoco model="fk_tree">
  <compiler angle="radian"/>
  <default>
    <default class="roll">
      <joint axis="1 0 0"/>
    </default>
  </default>
  <worldbody>
    <body name="root" pos="0 0 1">
      <freejoint name="root_free"/>
      <geom name="root_geom" size="0.05" mass="1"/>
      <body name="shoulder" pos="0.1 0.05 0.2" quat="0.9800666 0 0.1986693 0">
        <joint name="pitch" type="hinge" axis="0 1 0"/>
        <geom name="shoulder_geom" size="0.05" mass="0.5"/>
        <body name="elbow" pos="0.3 0 0" quat="0.9238795 0 0 0.3826834">
          <joint name="hinge" type="hinge" class="roll"/>
          <geom name="elbow_geom" size="0.05" mass="0.4"/>
          <body name="rail" pos="0 0.1 -0.05">
            <joint name="slide" type="slide" axis="0 0 1"/>
            <geom name="rail_geom" size="0.05" mass="0.3"/>
            <body name="tip" pos="0.02 -0.03 0.04" quat="0.7071068 0.7071068 0 0">
              <geom name="tip_geom" size="0.05" mass="0.2"/>
            </body>
          </body>
        </body>
      </body>
      <body name="antenna" pos="-0.1 0 0.1">
        <geom name="antenna_geom" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _write_model(tmp_path: Path, text: str = _MODEL) -> Path:
    path = tmp_path / "fk_tree.xml"
    path.write_text(text, encoding="utf-8")
    return path


def test_scan_order_matches_metadata_and_tables_are_wire_safe(tmp_path: Path) -> None:
    path = _write_model(tmp_path)
    tables = scan_scene_kinematics(str(path), backend_label="test")
    metadata = sensors.scan_scene_metadata(str(path), backend_label="test")
    assert tables["body_names"] == list(metadata.body_names)
    assert tables["joint_names"] == list(metadata.joint_names)
    assert tables["body_names"] == ["root", "shoulder", "elbow", "rail", "tip", "antenna"]
    assert tables["joint_names"] == ["pitch", "hinge", "slide"]
    assert tables["free_root"] == 0
    assert tables["body_parent"] == [-1, 0, 1, 2, 3, 0]
    assert tables["body_joint_kind"] == [0, 1, 1, 2, 0, 0]
    assert tables["body_joint_column"] == [-1, 7, 8, 9, -1, -1]
    # The joint axis resolves through the default class.
    np.testing.assert_allclose(tables["body_joint_axis"][2], [1.0, 0.0, 0.0], atol=1e-12)
    # Payload survives a JSON-style round trip (lists of plain floats).
    assert all(isinstance(value, list) for value in tables["body_pos"])


def test_combined_cold_scan_matches_independent_owners(tmp_path: Path) -> None:
    path = _write_model(tmp_path)
    metadata, kinematics = sensors.scan_scene_metadata_with_kinematics(
        str(path), backend_label="test"
    )
    independent_metadata = sensors.scan_scene_metadata(str(path), backend_label="test")
    independent_kinematics = scan_scene_kinematics(str(path), backend_label="test")
    assert metadata == independent_metadata
    assert kinematics == independent_kinematics


@pytest.mark.parametrize("seed", range(4))
def test_forward_kinematics_matches_mujoco(tmp_path: Path, seed: int) -> None:
    mujoco = pytest.importorskip("mujoco")
    path = _write_model(tmp_path)
    tables = scan_scene_kinematics(str(path), backend_label="test")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    qpos = np.zeros(model.nq)
    qpos[:3] = rng.normal(scale=0.4, size=3)
    random_quat = rng.normal(size=4)
    qpos[3:7] = random_quat / np.linalg.norm(random_quat)
    qpos[7:] = rng.normal(scale=0.6, size=model.nq - 7)
    qvel = rng.normal(scale=0.8, size=model.nv)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)

    state = forward_kinematics(tables, qpos[None, :], qvel[None, :])[0]
    prepared_state = forward_prepared_kinematics(
        prepare_kinematics(tables), qpos[None, :], qvel[None, :]
    )[0]
    np.testing.assert_array_equal(state, prepared_state)
    order = [name for name in tables["body_names"]]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in order]
    np.testing.assert_allclose(
        state[:, 0:3], np.asarray([data.xpos[b] for b in body_ids]), atol=1e-10
    )
    for row, b in zip(state, body_ids):
        dot = abs(float(np.dot(row[3:7] / np.linalg.norm(row[3:7]), data.xquat[b])))
        assert 2 * np.arccos(np.clip(dot, -1, 1)) < 1e-7
    expected_lin = []
    expected_ang = []
    for b in body_ids:
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)
        expected_ang.append(vel6[:3].copy())
        # mj_objectVelocity reports the COM velocity; the canonical body-state
        # contract publishes the link-origin velocity.
        expected_lin.append(vel6[3:] - np.cross(vel6[:3], data.xipos[b] - data.xpos[b]))
    np.testing.assert_allclose(state[:, 7:10], np.asarray(expected_lin), atol=1e-10)
    np.testing.assert_allclose(state[:, 10:13], np.asarray(expected_ang), atol=1e-10)


def test_scan_fails_closed_on_unsupported_constructs(tmp_path: Path) -> None:
    cases = {
        "ball joint": (
            '<body name="root" pos="0 0 1"><freejoint name="f"/>'
            '<body name="ball"><joint name="ball_j" type="ball"/></body></body>',
            "only hinge and slide",
        ),
        "compound joints": (
            '<body name="root" pos="0 0 1"><freejoint name="f"/>'
            '<body name="two"><joint name="a" type="hinge"/><joint name="b" type="hinge"/>'
            "</body></body>",
            "one single-DoF joint per body",
        ),
        "euler orientation": (
            '<body name="root" pos="0 0 1"><freejoint name="f"/>'
            '<body name="e" euler="0.1 0 0"/></body>',
            "only parses the quat attribute",
        ),
        "nested freejoint": (
            '<body name="root" pos="0 0 1"><freejoint name="f"/>'
            '<body name="nested"><freejoint name="inner"/></body></body>',
            "not a direct worldbody child",
        ),
        "two freejoints": (
            '<body name="root" pos="0 0 1"><freejoint name="f"/></body>'
            '<body name="second" pos="1 0 0"><freejoint name="g"/></body>',
            "one freejoint body",
        ),
        "unnamed joint": (
            '<body name="root" pos="0 0 1"><freejoint name="f"/>'
            '<body name="u"><joint type="hinge"/></body></body>',
            "unnamed joint",
        ),
        "mixed freejoint and hinge": (
            '<body name="root" pos="0 0 1"><freejoint name="f"/><joint name="h" type="hinge"/>'
            "</body>",
            "mixes a freejoint with scalar joints",
        ),
        "nonzero joint ref": (
            '<body name="root" pos="0 0 1"><freejoint name="f"/>'
            '<body name="r"><joint name="h" type="hinge" ref="0.3"/></body></body>',
            "reference position is zero",
        ),
        "zero joint axis": (
            '<body name="root" pos="0 0 1"><freejoint name="f"/>'
            '<body name="z"><joint name="h" type="hinge" axis="0 0 0"/></body></body>',
            "axis must be nonzero",
        ),
    }
    for label, (body_xml, match) in cases.items():
        path = tmp_path / f"bad_{label.replace(' ', '_')}.xml"
        path.write_text(f"<mujoco><worldbody>{body_xml}</worldbody></mujoco>", encoding="utf-8")
        with pytest.raises((ValueError, NotImplementedError), match=match):
            scan_scene_kinematics(str(path), backend_label="test")


def test_payload_validation_fails_closed(tmp_path: Path) -> None:
    tables = scan_scene_kinematics(str(_write_model(tmp_path)), backend_label="test")
    qpos = np.zeros((1, 10))
    qpos[0, 3] = 1.0
    qvel = np.zeros((1, 9))

    bad_schema = dict(tables, schema_version=99)
    with pytest.raises(ValueError, match="schema"):
        forward_kinematics(bad_schema, qpos, qvel)

    swapped = dict(tables)
    swapped["body_parent"] = list(tables["body_parent"])
    swapped["body_parent"][1] = 3  # child precedes its parent
    with pytest.raises(ValueError, match="parents must precede children"):
        forward_kinematics(swapped, qpos, qvel)

    nested_free = dict(tables, free_root=2)
    with pytest.raises(ValueError, match="top-level jointless body"):
        forward_kinematics(nested_free, qpos, qvel)

    no_free = dict(tables, free_root=-1)
    with pytest.raises(ValueError, match="free root"):
        forward_kinematics(no_free, qpos, qvel)

    bad_column = dict(tables)
    bad_column["body_joint_column"] = [12 if c == 7 else c for c in tables["body_joint_column"]]
    with pytest.raises(ValueError, match="column out of range"):
        forward_kinematics(bad_column, qpos, qvel)

    with pytest.raises(ValueError, match="unit wxyz"):
        forward_kinematics(tables, np.zeros((1, 10)), qvel)
    with pytest.raises(ValueError, match="qvel must have shape"):
        forward_kinematics(tables, qpos, np.zeros((1, 8)))


def test_kinematics_kernel_is_py38_syntax_and_loads_standalone() -> None:
    path = Path(protocol.__file__).with_name("kinematics.py")
    ast.parse(path.read_text(), feature_version=(3, 8))
    code = """import importlib.util,sys
spec=importlib.util.spec_from_file_location('wire',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
k=m.load_kinematics()
assert k.SCHEMA_VERSION==1
assert 'unisim' not in sys.modules
assert not {'torch','mujoco','warp'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(Path(protocol.__file__))], check=True)
