"""Portable MJCF structural-oracle and identity integration checks."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
from unisim.scene import SceneCfg
from unisim.scene_compiler import compile_portable_scene


def _png(rgb: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(
            ">I", zlib.crc32(kind + data) & 0xFFFFFFFF
        )

    return b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ) + chunk(b"IDAT", zlib.compress(bytes([0, *rgb]))) + chunk(b"IEND", b"")


def _robot(path: Path) -> ModelSourceDescriptor:
    target = path / "robot.xml"
    target.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="base">
              <freejoint name="root"/>
              <geom name=" torso" type="sphere" size=".12" mass="1"/>
              <body name="link"><joint name="hinge"/><geom type="sphere" size=".05"/></body>
            </body>
          </worldbody>
          <actuator><motor name="drive" joint="hinge"/></actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(target))


def _object(path: Path, *, mass: float, texture: bytes | None = None) -> ModelSourceDescriptor:
    path.mkdir(parents=True, exist_ok=True)
    (path / "stripe.png").write_bytes(texture or _png())
    target = path / f"object-{mass}.xml"
    target.write_text(
        f"""
        <mujoco>
          <asset>
            <texture name="stripe" type="2d" file="stripe.png"/>
            <material name="stripe" texture="stripe"/>
          </asset>
          <worldbody>
            <body name="base">
              <freejoint name="root"/>
              <geom name="shape" type="box" size=".05 .04 .03" mass="{mass}" material="stripe"/>
              <body name="lid" pos="0 0 .04">
                <joint name="hinge"/><geom type="sphere" size=".02" mass=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(target))


def _table(path: Path) -> ModelSourceDescriptor:
    target = path / "table.xml"
    target.write_text(
        """
        <mujoco><worldbody>
          <body name="top"><geom name="surface" type="box" size=".5 .5 .02" mass="8"/>
        </body></worldbody></mujoco>
        """,
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(target))


def _scene(tmp_path: Path, *, texture: bytes | None = None) -> SceneCfg:
    texture = texture or _png()
    a = _object(tmp_path, mass=0.5, texture=texture)
    b = _object(tmp_path, mass=0.7, texture=texture)
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", _robot(tmp_path)),
            SceneEntitySpec(
                "object", a, kind="articulation", initial_state=EntityInitialState((1, 0, 0.4))
            ),
            SceneEntitySpec("table", _table(tmp_path), kind="rigid", root_mode="fixed"),
            SceneEntitySpec(
                "mirror",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="object",
                initial_state=EntityInitialState((-1, 0, 0.4)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object", FixedVariantPlan(np.array([1, 1, 0, 1, 0]), (a, b))
        ),
    )


def test_golden_portable_scene_freezes_layout_report_and_variant_identity(tmp_path):
    scene = _scene(tmp_path)
    with compile_portable_scene(scene, 5, 0.002) as composed:
        assert composed.variant_plan is not None
        np.testing.assert_array_equal(composed.variant_plan.assignment, [1, 1, 0, 1, 0])
        assert len(composed.variant_layouts) == 2
        assert composed.layout.get_entity("robot").actuator_names == ("drive",)
        assert composed.layout.get_entity("object").joints[0].name == "hinge"
        assert composed.layout.get_entity("table").joints == ()
        assert composed.layout.get_entity("mirror").joints == ()
        expected_modes = {
            "robot": "floating",
            "object": "floating",
            "table": "fixed",
            "mirror": "kinematic",
        }
        assert {e.name: e.root_mode for e in composed.layout.entities} == expected_modes

        report = composed.intent_report
        assert report.profile == "portable-mjcf-v1"
        assert report.content_identity == composed.content_identity
        # The declared target source is audited first, then each catalog variant
        # and each mirror role that consumes that variant.
        records = [(source.entity, source.variant) for source in report.sources]
        assert len(records) == 8
        assert records.count(("robot", None)) == 1
        assert records.count(("table", None)) == 1
        assert records.count(("object", None)) == 1
        assert records.count(("mirror", None)) == 1
        assert records.count(("object", 0)) == 1
        assert records.count(("mirror", 0)) == 1
        assert records.count(("object", 1)) == 1
        assert records.count(("mirror", 1)) == 1
        assert any(
            r.kind == "texture" and r.logical_path == "stripe.png"
            for r in report.sources[1].resources
        )
        assert all(field.effective is None for field in report.fields)

        for descriptor, expected_mass in zip(
            composed.variant_plan.variants, (0.5, 0.7), strict=True
        ):
            model = mujoco.MjModel.from_xml_path(descriptor.model_file)
            assert model.body("object/base").mass == pytest.approx(expected_mass)
            assert model.actuator("robot/drive").id == 0
            assert model.geom("mirror/shape").contype == 0
            assert model.geom("mirror/shape").conaffinity == 0


def test_content_identity_tracks_referenced_resource_content(tmp_path):
    first = compile_portable_scene(_scene(tmp_path / "first"), 5, 0.002)
    changed_texture = _png((0, 255, 0))
    second_scene = _scene(tmp_path / "second", texture=changed_texture)
    second = compile_portable_scene(second_scene, 5, 0.002)
    try:
        assert first.content_identity != second.content_identity
    finally:
        first.close()
        second.close()


def test_compile_wrapper_surfaces_missing_compiler_dependency(monkeypatch: pytest.MonkeyPatch):
    from unisim import scene_compiler

    def missing() -> None:
        raise ImportError("install unisim-core[scene-compiler]")

    monkeypatch.setattr(scene_compiler, "load_portable_mjcf_compiler", missing)
    with pytest.raises(ImportError, match=r"scene-compiler"):
        compile_portable_scene(object(), 1, 0.002)


def test_portable_profile_rejects_included_sources_before_composition(tmp_path):
    (tmp_path / "part.xml").write_text(
        "<mujoco><worldbody><body name='part'><geom type='sphere' size='.1'/></body>"
        "</worldbody></mujoco>",
        encoding="utf-8",
    )
    source = tmp_path / "source.xml"
    source.write_text(
        "<mujoco><include file='part.xml'/><worldbody><body name='base'>"
        "<geom type='sphere' size='.1'/></body></worldbody></mujoco>",
        encoding="utf-8",
    )
    scene = SceneCfg(entity_assets=(SceneEntitySpec("object", ModelSourceDescriptor(str(source))),))
    with pytest.raises(NotImplementedError, match="MJCF includes"):
        compile_portable_scene(scene, 1, 0.002)
