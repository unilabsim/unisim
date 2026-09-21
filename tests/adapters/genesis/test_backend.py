"""Adapter tests for the Genesis backend cold-path body-id contract."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unisim.backend.genesis import dependencies, materialization
from unisim.backend.genesis.backend import GenesisBackend
from unisim.backend.genesis.dependencies import GenesisDependencies
from unisim.scene import SceneCfg

_MODEL = """
<mujoco model="unisim-genesis-test">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.5">
      <joint name="root" type="free"/>
      <geom name="base_geom" type="sphere" size="0.08" mass="1"/>
      <body name="arm" pos="0 0 0.12">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="hinge_motor" joint="hinge" ctrlrange="-1 1"/></actuator>
</mujoco>
"""


class _StubCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _StubTorch:
    cuda = _StubCuda()

    @staticmethod
    def device(name: str) -> str:
        return name


class _StubMorphs:
    class MJCF:
        def __init__(self, *, file: str) -> None:
            self.file = file


class _StubGenesis:
    morphs = _StubMorphs


class _StubEntity:
    pass


class _StubScene:
    def add_entity(self, morph: Any) -> _StubEntity:
        del morph
        return _StubEntity()


def test_genesis_motion_body_ids_follow_mjcf_worldbody_zero_convention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.setattr(
        dependencies,
        "load_genesis_dependencies",
        lambda: GenesisDependencies(
            genesis=_StubGenesis, torch=_StubTorch, mujoco=mujoco
        ),
    )
    monkeypatch.setattr(materialization, "init_genesis_session", lambda *a, **k: None)
    monkeypatch.setattr(
        materialization, "build_genesis_scene", lambda *a, **k: _StubScene()
    )
    model_file = tmp_path / "genesis.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    backend = GenesisBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=1,
        sim_dt=0.005,
    )

    model = mujoco.MjModel.from_xml_path(str(model_file))
    names = ["base", "arm"]
    expected = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names],
        dtype=np.int32,
    )
    assert expected.tolist() == [1, 2]  # MJCF body ids, worldbody is id 0
    np.testing.assert_array_equal(backend.get_motion_body_ids(names), expected)
    np.testing.assert_array_equal(
        backend.get_motion_body_ids(names), backend.get_body_ids(names)
    )


def test_genesis_found_contact_sensor_accepts_mindist_reduce(tmp_path: Path) -> None:
    """data="found" ignores the reduce mode; accept the mindist-compiled form."""
    mujoco = pytest.importorskip("mujoco")
    model_file = tmp_path / "contact.xml"
    model_file.write_text(
        """<mujoco model="unisim-genesis-contact-test">
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.5">
      <joint name="root" type="free"/>
      <geom name="base_geom" type="sphere" size="0.08" mass="1"/>
    </body>
  </worldbody>
  <sensor>
    <contact name="ground_base_none" geom1="ground" geom2="base_geom"
             data="found" num="1"/>
    <contact name="ground_base_mindist" geom1="ground" geom2="base_geom"
             data="found" num="1" reduce="mindist"/>
  </sensor>
</mujoco>
""",
        encoding="utf-8",
    )
    metadata = materialization.scan_genesis_model_metadata(
        mujoco, SceneCfg(model_file=str(model_file))
    )
    plans = {plan.name: plan for plan in metadata.sensor_plans}
    assert plans["ground_base_none"].kind == "contact"
    mindist = plans["ground_base_mindist"]
    assert mindist.kind == "contact"
    assert mindist.dim == 1
    # The found flag is a scalar contact-presence read, so the reduce mode is
    # unobservable: both compiled forms must map to the same plan.
    assert dataclasses.replace(mindist, name="") == dataclasses.replace(
        plans["ground_base_none"], name=""
    )
