"""Small compiler-backed scene payload for mock and optional native worker tests."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout


def scene_payload(
    directory: Path,
    *,
    mirror_overlap: bool = True,
    gravity=(0.0, 0.0, 0.0),
    env_spacing: float = 4.0,
):
    import mujoco

    directory.mkdir(parents=True, exist_ok=True)
    robot = """<mujoco><worldbody><body name="base">
    <inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
    <geom type="sphere" size=".1"/><body name="finger" pos="0 0 .3">
    <joint name="drive_joint" type="hinge" axis="0 1 0" range="-1 1"/>
    <inertial pos="0 0 0" mass=".2" diaginertia=".001 .001 .001"/>
    <geom type="sphere" size=".05"/></body></body></worldbody>
    <actuator><position name="drive" joint="drive_joint" kp="20" kv="1"/></actuator></mujoco>"""
    object_xml = """<mujoco><worldbody><body name="base"><freejoint/>
    <inertial pos=".05 0 0" mass="{mass}" diaginertia=".01 .02 .03"/>
    <geom type="box" size=".1 .1 .1"/><body name="lid" pos="0 0 .2">
    <joint name="passive" type="hinge" axis="0 1 0"/>
    <inertial pos=".05 0 0" mass=".2" diaginertia=".001 .002 .003"/>
    <geom type="box" size=".08 .08 .02"/></body></body></worldbody></mujoco>"""
    rigid = """<mujoco><worldbody><body name="base">
    <inertial pos="0 0 0" mass="{mass}" diaginertia=".01 .01 .01"/>
    <geom type="box" size="{size}"/></body></worldbody></mujoco>"""
    sources = {
        "robot": [robot],
        "object": [object_xml.format(mass=m) for m in (1, 3)],
        "table": [rigid.format(mass=10, size="2 2 .1")],
        "target": [rigid.format(mass=m, size=".1 .1 .1") for m in (1, 3)],
    }
    layout = CompiledSceneLayout(
        (
            EntityLayout(
                "robot",
                "articulation",
                "fixed",
                "base",
                ("base", "finger"),
                (1, 2),
                (None, "base"),
                (JointLayout("drive_joint", "hinge", (0,), (0,), "finger"),),
                ("drive",),
                ("drive_joint",),
                (0,),
            ),
            EntityLayout(
                "object",
                "articulation",
                "floating",
                "base",
                ("base", "lid"),
                (3, 4),
                (None, "base"),
                (JointLayout("passive", "hinge", (8,), (7,), "lid"),),
                (),
                (),
                (),
                tuple(range(1, 8)),
                tuple(range(1, 7)),
            ),
            EntityLayout(
                "table", "rigid", "fixed", "base", ("base",), (5,), (None,), (), (), (), ()
            ),
            EntityLayout(
                "target", "rigid", "kinematic", "base", ("base",), (6,), (None,), (), (), (), ()
            ),
        ),
        nq=9,
        nv=8,
        nu=1,
        nbody=7,
    )
    specs = []
    count = 5
    qpos = np.zeros((count, layout.nq))
    qvel = np.zeros((count, layout.nv))
    roots = np.zeros((count, 4, 13))
    roots[..., 3] = 1
    roots[:, 0, :3] = [-1, 0, 0.4]
    roots[:, 1, :3] = [0, 0, 1]
    roots[:, 1, 3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    roots[:, 3, :3] = [0 if mirror_overlap else 3, 0, 0.5]
    qpos[:, 1:8] = roots[:, 1, :7]
    for index, entity in enumerate(layout.entities):
        paths, variants = [], []
        for number, text in enumerate(sources[entity.name]):
            model = mujoco.MjModel.from_xml_string(text)
            path = directory / (entity.name + str(number) + ".xml")
            mujoco.mj_saveLastXML(str(path), model)
            # Native drives are restored from the explicit variant record; the
            # Gym importer rejects canonical MuJoCo <general> actuator elements.
            xml = ET.parse(path)
            actuator = xml.getroot().find("actuator")
            if actuator is not None:
                xml.getroot().remove(actuator)
            for joint in xml.findall(".//joint"):
                if joint.get("type") == "free" or not joint.get("name"):
                    continue
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.get("name"))
                joint.set("limited", "true" if model.jnt_limited[joint_id] else "false")
            xml.write(path)
            paths.append(str(path.resolve()))
            body_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in entity.body_names
            ]
            body_sphere_radii = []
            body_visual_rgb = []
            for body_name in entity.body_names:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                body_rgba = None
                body_sphere_radii.append(
                    [
                        float(model.geom_size[geom_id, 0])
                        for geom_id in range(model.ngeom)
                        if model.geom_bodyid[geom_id] == body_id
                        and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
                    ]
                )
                for geom_id in range(model.ngeom):
                    if model.geom_bodyid[geom_id] != body_id:
                        continue
                    if body_rgba is None and model.geom_rgba[geom_id, 3] > 0:
                        body_rgba = model.geom_rgba[geom_id, :3]
                body_visual_rgb.append(
                    body_rgba.tolist() if body_rgba is not None else [0.5, 0.5, 0.5]
                )
            active = bool(entity.actuator_names)
            nj = len(entity.joints)
            joint_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name) for j in entity.joints
            ]
            variants.append(
                {
                    "joint_names": [j.name for j in entity.joints],
                    "actuator_names": list(entity.actuator_names),
                    "actuator_joint_names": list(entity.actuator_joint_names),
                    "dof_stiffness": [20.0] * nj if active else [0.0] * nj,
                    "dof_damping": [1.0] * nj if active else [0.0] * nj,
                    "dof_effort": [100.0] * nj if active else [0.0] * nj,
                    "dof_armature": [0.0] * nj,
                    "dof_friction": [0.0] * nj,
                    "dof_lower": [
                        float(model.jnt_range[j, 0]) if model.jnt_limited[j] else -np.inf
                        for j in joint_ids
                    ],
                    "dof_upper": [
                        float(model.jnt_range[j, 1]) if model.jnt_limited[j] else np.inf
                        for j in joint_ids
                    ],
                    "body_names": list(entity.body_names),
                    "body_mass": model.body_mass[body_ids].tolist(),
                    "body_ipos": model.body_ipos[body_ids].tolist(),
                    "body_inertia": model.body_inertia[body_ids].tolist(),
                    "body_iquat": model.body_iquat[body_ids].tolist(),
                    "body_sphere_radii": body_sphere_radii,
                    "body_visual_rgb": body_visual_rgb,
                }
            )
        specs.append(
            {
                "name": entity.name,
                "kind": entity.kind,
                "root_mode": entity.root_mode,
                "asset_format": "mjcf",
                "collision_enabled": entity.name != "target",
                "mirror_of": "object" if entity.name == "target" else None,
                "self_collision": False,
                "initial_pose": roots[0, index, :7].tolist(),
                "sources": paths,
                "assignment": [1, 1, 0, 1, 0] if len(paths) == 2 else [0] * count,
                "variants": variants,
            }
        )
    return {
        "num_envs": count,
        "sim_dt": 0.001,
        "device_id": 0,
        "scene_layout": layout.to_dict(),
        "scene_entities": specs,
        "initial_qpos": qpos.tolist(),
        "initial_qvel": qvel.tolist(),
        "initial_ctrl": [[0.37] for _ in range(count)],
        "initial_roots": roots.tolist(),
        "gravity": list(gravity),
        "env_spacing": env_spacing,
    }


PUBLIC_GEOMS = {
    "robot": (("base::geom0", "base"), ("finger::geom0", "finger")),
    "object": (("base::geom0", "base"), ("lid::geom0", "lid")),
    "table": (("base::geom0", "base"),),
    "target": (("base::geom0", "base"),),
}


def add_public_geoms(payload):
    """Declare the public geom layout matching the MJCF source geoms in order."""
    layout = payload["scene_layout"]
    total = 0
    for entry in layout["entities"]:
        geoms = PUBLIC_GEOMS[entry["name"]]
        entry["geoms"] = [{"name": name, "body_name": body} for name, body in geoms]
        total += len(geoms)
    layout["ngeom"] = total
    for spec in payload["scene_entities"]:
        geoms = PUBLIC_GEOMS[spec["name"]]
        for variant in spec["variants"]:
            variant["geom_names"] = [name for name, _ in geoms]
            variant["geom_body_names"] = [body for _, body in geoms]
            variant["geom_friction"] = [[0.5, 0.5, 0.0]] * len(geoms)
    return payload
