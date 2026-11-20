"""Unnamed-geom addressing for portable Motrix scenes (upstream ``motrixsim#22``).

MJCF geoms may be unnamed.  The public layout compiler synthesizes
``<body>::geom<ordinal>`` names whose ordinal counts every geom of the body in
MJCF definition order, and MotrixSim preserves that definition order inside
``link.geoms``.  These tests lock the adapter's (link, ordinal) resolution for
named/unnamed mixes, multiple unnamed geoms per link, shared meshes across
links, and repeated loads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")
pytest.importorskip("mujoco")

from unisim import MotrixBackend  # noqa: E402
from unisim.dr.types import ModelSourceDescriptor  # noqa: E402
from unisim.entities import EntityInitialState, SceneEntitySpec  # noqa: E402
from unisim.scene import SceneCfg  # noqa: E402

TETRAHEDRON = """v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


def _robot(tmp_path: Path) -> ModelSourceDescriptor:
    obj = tmp_path / "shared-tetrahedron.obj"
    tmp_path.mkdir(parents=True, exist_ok=True)
    obj.write_text(TETRAHEDRON, encoding="utf-8")
    path = tmp_path / "robot.xml"
    path.write_text(
        f"""<mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/>
        <asset><mesh name="shared" file="{obj}" scale="1 1 1"/></asset>
        <worldbody><body name="base" pos="0 0 1">
          <inertial pos="0 0 0" mass="1" diaginertia=".2 .2 .2"/>
          <geom type="sphere" size=".08"/>
          <geom name="base_box" type="box" size=".05 .06 .07"/>
          <geom type="capsule" size=".02 .09"/>
          <body name="link" pos="0 0 .2">
            <joint name="drive" axis="0 1 0" ref="0.1"/>
            <inertial pos="0 0 0" mass=".2" diaginertia=".03 .03 .03"/>
            <geom type="sphere" size=".03"/>
            <geom type="sphere" size=".04"/>
          </body>
          <body name="finger_a" pos=".2 0 0">
            <joint name="joint_a" axis="1 0 0"/>
            <inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/>
            <geom type="mesh" mesh="shared"/>
          </body>
          <body name="finger_b" pos="-.2 0 0">
            <joint name="joint_b" axis="1 0 0"/>
            <inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/>
            <geom type="mesh" mesh="shared"/>
          </body>
        </body></worldbody>
        <actuator><motor name="drive" joint="drive"/></actuator></mujoco>
        """,
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _scene(tmp_path: Path) -> SceneCfg:
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                _robot(tmp_path),
                root_mode="fixed",
                initial_state=EntityInitialState((0.0, 0.0, 1.0)),
            ),
        )
    )


def test_portable_unnamed_geoms_bind_by_link_ordinal(tmp_path: Path) -> None:
    backend = MotrixBackend(_scene(tmp_path), 3, 0.002, base_name="robot/base")
    try:
        assert backend.get_geom_names() == (
            "robot/base::geom0",  # unnamed sphere, ordinal 0
            "robot/base_box",  # named geom keeps its authored name
            "robot/base::geom2",  # unnamed capsule, ordinal 2 (ordinal 1 is the box)
            "robot/link::geom0",
            "robot/link::geom1",
            "robot/finger_a::geom0",
            "robot/finger_b::geom0",
        )
        base_sphere = backend.get_geom_size("robot/base::geom0")
        base_box = backend.get_geom_size("robot/base_box")
        base_capsule = backend.get_geom_size("robot/base::geom2")
        np.testing.assert_allclose(base_sphere[:1], [0.08], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(base_box, [0.05, 0.06, 0.07], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(base_capsule[:2], [0.02, 0.09], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(
            backend.get_geom_size("robot/link::geom0")[:1], [0.03], rtol=0.0, atol=1e-6
        )
        np.testing.assert_allclose(
            backend.get_geom_size("robot/link::geom1")[:1], [0.04], rtol=0.0, atol=1e-6
        )
        # The shared mesh asset backs two distinct unnamed geoms on two links.
        mesh_a = backend.get_geom_size("robot/finger_a::geom0")
        mesh_b = backend.get_geom_size("robot/finger_b::geom0")
        np.testing.assert_allclose(mesh_a, mesh_b, rtol=0.0, atol=0.0)

        geom_body_ids = backend.get_geom_body_ids()
        layout = backend.get_scene_layout()
        body_id_by_name = {
            name: body_id
            for entity in layout.entities
            for name, body_id in zip(entity.body_names, entity.body_ids, strict=True)
        }
        names = backend.get_geom_names()
        assert geom_body_ids[names.index("robot/base::geom2")] == body_id_by_name["base"]
        assert geom_body_ids[names.index("robot/link::geom1")] == body_id_by_name["link"]
        assert geom_body_ids[names.index("robot/finger_b::geom0")] == body_id_by_name["finger_b"]
    finally:
        backend.close()


def test_portable_unnamed_geoms_repeated_loads_keep_binding(tmp_path: Path) -> None:
    first = MotrixBackend(_scene(tmp_path), 2, 0.002, base_name="robot/base")
    try:
        first_names = first.get_geom_names()
        first_sizes = [first.get_geom_size(name) for name in first_names]
    finally:
        first.close()
    second = MotrixBackend(_scene(tmp_path), 2, 0.002, base_name="robot/base")
    try:
        assert second.get_geom_names() == first_names
        for name, expected in zip(first_names, first_sizes, strict=True):
            np.testing.assert_allclose(
                second.get_geom_size(name), expected, rtol=0.0, atol=0.0
            )
    finally:
        second.close()


def test_msd_repeated_loads_keep_link_geom_order_and_shared_mesh(tmp_path: Path) -> None:
    import motrixsim.msd as msd

    source = str(Path(_robot(tmp_path).model_file).resolve())

    def snapshot() -> tuple:
        world = msd.from_file(source)
        records = []
        for body in world.hierarchy.bodies:
            pending = [body.link]
            while pending:
                link = pending.pop()
                pending.extend(link.children)
                records.append((str(link.name), tuple(str(geom.name) for geom in link.geoms)))
        meshes = {str(name) for name in world.assets.meshes}
        return tuple(records), meshes

    first, first_meshes = snapshot()
    second, second_meshes = snapshot()
    assert first == second
    # Unnamed geoms stay unnamed and distinguishable by (link, ordinal); the
    # shared mesh asset is loaded once and referenced from both fingers.  The
    # source file carries unprefixed names; the portable compiler adds the
    # entity prefix when it composes the scene.
    base_record = next(record for record in first if record[0] == "base")
    assert base_record[1] == ("None", "base_box", "None")
    assert first_meshes == second_meshes == {"shared"}
