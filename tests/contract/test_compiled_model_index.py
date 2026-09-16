"""Compiled addresses have independent native topology and column oracles."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.model_index import CompiledModelIndex
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout


def _legacy_model():
    mujoco = pytest.importorskip("mujoco")
    return mujoco.MjModel.from_xml_string("""<mujoco>
      <worldbody><geom type="plane" size="1 1 .1"/>
        <body name="rotor:base"><joint name="hinge/axis"/><geom size=".1"/>
          <body pos="0 0 .3"><geom size=".05"/></body>
        </body>
        <body name="free/A"><joint name="free:A" type="free"/><geom size=".1"/>
          <site name="drive:site"/>
        </body>
        <body name="free:B"><joint name="free/B" type="free"/><geom size=".1"/>
          <body name="ball/child"><joint name="ball:joint" type="ball"/><geom size=".05"/>
          </body>
        </body>
        <body name="mocap/target" mocap="true"><geom size=".03"/></body>
      </worldbody>
      <tendon><fixed name="tendon:drive"><joint joint="hinge/axis" coef="1"/></fixed></tendon>
      <actuator><motor name="drive/tendon" tendon="tendon:drive"/>
        <general name="drive:site" site="drive:site" gear="1 0 0 0 0 0"/>
        <motor name="drive/free" joint="free:A" gear="0 0 0 0 0 1"/>
      </actuator>
    </mujoco>""")


def _simple_model():
    mujoco = pytest.importorskip("mujoco")
    return mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="robot/base"><geom size=".1"/>
        <body name="robot/tip"><joint name="robot/slide" type="slide"/><geom size=".1"/></body>
      </body>
      <body name="object/base"><joint name="object/free" type="free"/><geom size=".1"/></body>
      <body name="target/base" mocap="true"><geom size=".1"/></body>
      </worldbody><actuator><motor name="robot/drive" joint="robot/slide"/></actuator>
    </mujoco>""")


def _layout():
    robot = EntityLayout(
        "robot",
        "articulation",
        "fixed",
        "base",
        ("base", "tip"),
        (1, 2),
        (None, "base"),
        (JointLayout("slide", "slide", (0,), (0,), "tip"),),
        ("drive",),
        ("slide",),
        (0,),
    )
    obj = EntityLayout(
        "object",
        "rigid",
        "floating",
        "base",
        ("base",),
        (3,),
        (None,),
        (),
        (),
        (),
        (),
        tuple(range(1, 8)),
        tuple(range(1, 7)),
    )
    target = EntityLayout(
        "target", "rigid", "kinematic", "base", ("base",), (4,), (None,), (), (), (), ()
    )
    return CompiledSceneLayout((robot, obj, target), 8, 7, 1, 5)


def _fake(model):
    fields = (
        "nq",
        "nv",
        "nu",
        "nbody",
        "njnt",
        "nmocap",
        "nsite",
        "ntendon",
        "body_parentid",
        "body_mocapid",
        "jnt_bodyid",
        "jnt_type",
        "jnt_qposadr",
        "jnt_dofadr",
        "body_jntadr",
        "body_jntnum",
        "actuator_trntype",
        "actuator_trnid",
    )
    values = {
        name: value.copy() if isinstance(value := getattr(model, name), np.ndarray) else value
        for name in fields
    }
    for accessor, count in (("body", model.nbody), ("joint", model.njnt), ("actuator", model.nu)):
        names = tuple(getattr(model, accessor)(i).name for i in range(count))
        values[accessor] = lambda index, names=names: SimpleNamespace(name=names[index])
    return SimpleNamespace(**values)


def test_legacy_partitions_and_columns_are_not_a_single_articulation_assumption():
    model = _legacy_model()
    before = (model.nq, model.nv, model.nu, model.names, model.qpos0.copy())
    index = CompiledModelIndex.from_model(model)
    assert (index.nq, index.nv, index.nu, index.nbody) == (19, 16, 3, 7)
    assert index.root_ids == (1, 3, 4, 6)
    assert index.partitions == ((1, 2), (3,), (4, 5), (6,))
    assert index.bodies[0].parent_id == index.bodies[0].root_id == 0
    assert index.bodies[2].name == "" and index.bodies[2].id == 2
    assert index.bodies[6].mocap_id == 0
    assert index.body_id("free/A") == 3
    assert index.joint_id("hinge/axis") == 0
    assert index.free_root_layout("free/A") == (tuple(range(1, 8)), tuple(range(1, 7)))
    assert index.free_root_layout("free:B") == (tuple(range(8, 15)), tuple(range(7, 13)))
    assert index.joint_qpos_indices(["ball:joint", "hinge/axis"]) == (15, 16, 17, 18, 0)
    assert index.joint_qvel_indices(["ball:joint", "hinge/axis"]) == (13, 14, 15, 0)
    assert [(a.trntype, a.trnid) for a in index.actuators] == [
        (3, (0, -1)),
        (4, (0, -1)),
        (0, (1, -1)),
    ]
    assert tuple(a.name for a in index.actuators) == ("drive/tendon", "drive:site", "drive/free")
    assert (model.nq, model.nv, model.nu, model.names) == before[:4]
    np.testing.assert_array_equal(model.qpos0, before[4])
    with pytest.raises(NotImplementedError, match="exactly one free"):
        index.free_root_layout("rotor:base")
    with pytest.raises(ValueError, match="anonymous"):
        index.body_id("")
    with pytest.raises(FrozenInstanceError):
        index.bodies[2].name = "invented"


def test_native_metadata_is_detached_and_body_order_need_not_be_topological():
    fake = _fake(_simple_model())
    index = CompiledModelIndex.from_model(fake)
    fake.body_parentid[:] = 0
    fake.jnt_qposadr[:] = 0
    assert index.bodies[2].parent_id == 1
    assert index.joints[1].qpos_indices == tuple(range(1, 8))
    reordered = _fake(_simple_model())
    reordered.body_parentid[1:3] = [2, 0]
    changed = CompiledModelIndex.from_model(reordered)
    assert changed.bodies[1].root_id == 2
    assert changed.partitions[0] == (1, 2)


def test_declared_layout_crosschecks_native_partition_names_and_addresses():
    index = CompiledModelIndex.from_model(_simple_model())
    index.validate_entity_layout(_layout())
    robot, obj, target = _layout().entities
    tampered = (
        replace(
            _layout(),
            entities=(replace(robot, body_ids=(1, 3)), replace(obj, body_ids=(2,)), target),
        ),
        replace(
            _layout(),
            entities=(
                replace(robot, joints=(replace(robot.joints[0], kind="hinge"),)),
                obj,
                target,
            ),
        ),
        replace(_layout(), entities=(replace(robot, actuator_names=("wrong",)), obj, target)),
        replace(_layout(), entities=(robot, obj, replace(target, root_mode="fixed"))),
        replace(
            _layout(),
            entities=(
                replace(robot, joints=(replace(robot.joints[0], qpos_indices=(1,)),)),
                replace(obj, root_qpos_indices=(0, 2, 3, 4, 5, 6, 7)),
                target,
            ),
        ),
    )
    for layout in tampered:
        with pytest.raises(ValueError):
            index.validate_entity_layout(layout)
    # Same names and dimensions, but robot/tip is a different physical root.
    wrong_parent = _fake(_simple_model())
    wrong_parent.body_parentid[2] = 0
    with pytest.raises(ValueError, match="partition"):
        CompiledModelIndex.from_model(wrong_parent).validate_entity_layout(_layout())


def test_new_public_layout_does_not_absorb_complex_native_transmissions():
    fake = _fake(_simple_model())
    fake.ntendon = 1
    fake.actuator_trntype[0] = 3
    fake.actuator_trnid[0] = [0, -1]
    index = CompiledModelIndex.from_model(fake)
    assert index.actuators[0].trntype == 3
    with pytest.raises(ValueError, match="transmission"):
        index.validate_entity_layout(_layout())


def test_native_multichannel_actuator_controls_are_preserved_without_public_schema_expansion():
    fake = _fake(_simple_model())
    fake.nactuator = 1
    fake.nu = 3
    fake.actuator_trntype[:] = 6
    fake.actuator_ctrladr = np.array([0])
    fake.actuator_ctrlnum = np.array([3])
    index = CompiledModelIndex.from_model(fake)
    assert len(index.actuators) == 1
    assert index.actuators[0].control_indices == (0, 1, 2)
    assert index.nu == 3


def test_site_transmission_reference_and_jointinparent_are_recorded_as_native_targets():
    fake = _fake(_legacy_model())
    fake.nsite = 2
    fake.actuator_trnid[1] = [0, 1]
    fake.actuator_trntype[2] = 1
    index = CompiledModelIndex.from_model(fake)
    assert index.actuators[1].trnid == (0, 1)
    assert index.actuators[2].trntype == 1
    assert index.actuators[2].trnid == (1, -1)


def test_same_counts_cannot_relabel_a_cross_partition_actuator_target():
    fake = _fake(_simple_model())
    # A motor on the object's free joint is valid native topology, but it is
    # not the robot/slide target advertised by the public entity layout.
    fake.actuator_trnid[0, 0] = 1
    index = CompiledModelIndex.from_model(fake)
    assert index.actuators[0].trnid == (1, -1)
    with pytest.raises(ValueError, match="transmission"):
        index.validate_entity_layout(_layout())


@pytest.mark.parametrize(
    "corruption",
    [
        "world_parent",
        "world_joint",
        "cycle",
        "parent_bound",
        "qpos_overlap",
        "qvel_gap",
        "unknown_kind",
        "joint_body",
        "joint_count",
        "joint_address",
        "mocap_duplicate",
        "mocap_gap",
        "mocap_joint",
        "float_indices",
        "bool_dimension",
        "body_duplicate_name",
        "joint_duplicate_name",
        "actuator_target",
        "actuator_unused_target",
        "site_reference",
        "slidercrank_reference",
        "undefined_transmission",
        "actuator_width",
        "control_overlap",
    ],
)
def test_corrupt_compiled_metadata_fails_before_any_executor(corruption):
    fake = _fake(_legacy_model())
    if corruption == "world_parent":
        fake.body_parentid[0] = 1
    elif corruption == "world_joint":
        fake.jnt_bodyid[0] = 0
    elif corruption == "cycle":
        fake.body_parentid[1:3] = [2, 1]
    elif corruption == "parent_bound":
        fake.body_parentid[1] = 7
    elif corruption == "qpos_overlap":
        fake.jnt_qposadr[1] = 0
    elif corruption == "qvel_gap":
        fake.nv += 1
    elif corruption == "unknown_kind":
        fake.jnt_type[0] = 99
    elif corruption == "joint_body":
        fake.jnt_bodyid[0] = 7
    elif corruption == "joint_count":
        fake.body_jntnum[1] = 0
    elif corruption == "joint_address":
        fake.body_jntadr[1] = 1
    elif corruption == "mocap_duplicate":
        fake.body_mocapid[2] = 0
    elif corruption == "mocap_gap":
        fake.nmocap = 2
    elif corruption == "mocap_joint":
        fake.body_mocapid[6], fake.body_mocapid[1] = -1, 0
    elif corruption == "float_indices":
        fake.body_parentid = fake.body_parentid.astype(float)
    elif corruption == "bool_dimension":
        fake.nq = True
    elif corruption == "body_duplicate_name":
        fake.body = lambda index: SimpleNamespace(name="same")
    elif corruption == "joint_duplicate_name":
        fake.joint = lambda index: SimpleNamespace(name="same")
    elif corruption == "actuator_target":
        fake.actuator_trnid[0, 0] = 1
    elif corruption == "actuator_unused_target":
        fake.actuator_trnid[0, 1] = 0
    elif corruption == "site_reference":
        fake.actuator_trnid[1, 1] = 1
    elif corruption == "slidercrank_reference":
        fake.actuator_trntype[0] = 2
    elif corruption == "undefined_transmission":
        fake.actuator_trntype[0] = 1000
    elif corruption == "actuator_width":
        fake.actuator_ctrlnum = np.array([1, 0, 1])
    else:
        fake.actuator_ctrladr = np.array([0, 0, 2])
    with pytest.raises(ValueError):
        CompiledModelIndex.from_model(fake)
