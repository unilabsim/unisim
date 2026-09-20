"""Public entity mesh-variant contract coverage for the IsaacGym host."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityVariantBinding, SceneEntitySpec
from unisim.factory import create_backend
from unisim.scene import SceneCfg

_MOCK_WORKER = Path(__file__).resolve().parent / "mock_worker.py"
_TETRAHEDRON = """v 0 0 0
v .1 0 0
v 0 .1 0
v 0 0 .1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


def _mesh_source(
    root: Path,
    name: str,
    *,
    include_head: bool,
    head_scale: str = "1 1 1",
) -> ModelSourceDescriptor:
    obj = root / "tetrahedron.obj"
    obj.write_text(_TETRAHEDRON, encoding="utf-8")
    head_mesh = (
        f'<mesh name="head" file="{obj}" scale="{head_scale}"/>' if include_head else ""
    )
    head_geom = '<geom name="head" type="mesh" mesh="head" mass="0"/>' if include_head else ""
    path = root / f"{name}.xml"
    path.write_text(
        f"""<mujoco>
  <asset><mesh name="handle" file="{obj}"/>{head_mesh}</asset>
  <worldbody><body name="base"><freejoint/>
    <inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
    <geom name="handle" type="mesh" mesh="handle" mass="0"/>{head_geom}
  </body></worldbody>
</mujoco>""",
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _table_source(root: Path) -> ModelSourceDescriptor:
    path = root / "table.xml"
    path.write_text(
        """<mujoco><worldbody><body name="base">
  <inertial mass="5" pos="0 0 0" diaginertia=".1 .1 .1"/>
  <geom name="surface" type="box" size="2 2 .1" mass="0"/>
</body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _scene(root: Path, *, layout: FixedVariantLayout) -> SceneCfg:
    missing = _mesh_source(root, "missing", include_head=False)
    present = _mesh_source(root, "present", include_head=True, head_scale=".5 .5 .5")
    plan = FixedVariantPlan(
        np.asarray((0, 1, 1, 0), dtype=np.int32),
        (missing, present),
        layout=layout,
    )
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec("object", missing, kind="rigid"),
            SceneEntitySpec(
                "mirror",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="object",
            ),
            SceneEntitySpec(
                "table", _table_source(root), kind="rigid", root_mode="fixed"
            ),
        ),
        entity_variant=EntityVariantBinding("object", plan),
    )


def _backend(scene: SceneCfg, record: Path):
    return create_backend(
        "isaacgym",
        scene,
        4,
        0.002,
        base_name="object",
        worker_command=[sys.executable, str(_MOCK_WORKER), "--record", str(record)],
        worker_timeout_s=30.0,
    )


def test_entity_mesh_variants_use_public_contracts_and_reach_worker(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    import mujoco

    scene = _scene(tmp_path, layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT)
    record = tmp_path / "init.json"
    backend = _backend(scene, record)
    try:
        capabilities = backend.get_dr_capabilities()
        assert capabilities.supports_fixed_variants
        assert capabilities.supported_fixed_variant_layouts == frozenset(
            {FixedVariantLayout.SAME_LAYOUT, FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT}
        )
        assert capabilities.supports_per_env_playback

        backend.materialize()
        payload: dict[str, Any] = json.loads(record.read_text(encoding="utf-8"))
        entries = {entry["name"]: entry for entry in payload["scene_entities"]}
        assert entries["object"]["assignment"] == [0, 1, 1, 0]
        assert entries["mirror"]["assignment"] == entries["object"]["assignment"]
        assert entries["table"]["assignment"] == [0, 0, 0, 0]
        assert len(entries["object"]["sources"]) == len(entries["mirror"]["sources"]) == 2
        assert len(entries["table"]["sources"]) == 1

        for source, has_head in zip(entries["object"]["sources"], (False, True)):
            geoms = {geom.get("name") for geom in ET.parse(source).findall(".//geom")}
            assert ("head" in geoms) is has_head
            assert "handle" in geoms

        playback = [backend.get_playback_model(index) for index in range(4)]
        assert [mujoco.MjModel.from_xml_path(path).ngeom for path in playback] == [3, 5, 5, 3]
    finally:
        backend.close()


def test_entity_mesh_variant_same_layout_claims_fail_closed(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    with pytest.raises(ValueError, match="scene layouts differ"):
        _backend(
            _scene(tmp_path, layout=FixedVariantLayout.SAME_LAYOUT),
            tmp_path / "unused-init.json",
        )
