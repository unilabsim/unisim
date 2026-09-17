"""Host-side fixed-variant contract tests for the IsaacGym subprocess adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unisim.backend.isaacgym.backend import IsaacGymBackend
from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
from unisim.factory import create_backend
from unisim.scene import SceneCfg

_MOCK_WORKER = Path(__file__).resolve().parent / "mock_worker.py"
_SIM_DT = 0.005


def _variant_xml(
    *,
    tool_mass: float,
    tool_size: float,
    key_dof: float,
    kp: float,
) -> str:
    return f"""<mujoco model="IsaacGymFixedVariant">
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="tool" pos="0 0 0.3">
      <joint name="tool_pitch" type="hinge" axis="0 1 0"/>
      <geom name="handle" type="box" size="{tool_size} {tool_size} {tool_size}"
            mass="{tool_mass}"/>
    </body>
  </worldbody>
  <actuator>
    <position name="tool_motor" joint="tool_pitch" kp="{kp}" kv="0.1"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0.3 1 0 0 0 {key_dof}"/>
  </keyframe>
</mujoco>
"""


def _write_variants(root: Path, *, count: int = 3) -> tuple[Path, ...]:
    files: list[Path] = []
    for index in range(count):
        path = root / f"tool_{index}.xml"
        path.write_text(
            _variant_xml(
                tool_mass=0.4 + 0.2 * index,
                tool_size=0.05 + 0.01 * index,
                key_dof=0.1 * index,
                kp=20.0 + 10.0 * index,
            ),
            encoding="utf-8",
        )
        files.append(path)
    return tuple(files)


def _make_backend(
    sources: tuple[Path, ...],
    assignment: tuple[int, ...],
    record: Path,
    *,
    num_envs: int | None = None,
    extra_worker_args: tuple[str, ...] = (),
) -> IsaacGymBackend:
    plan = FixedVariantPlan(
        assignment=np.asarray(assignment, dtype=np.int32),
        variants=tuple(ModelSourceDescriptor(str(path)) for path in sources),
    )
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=str(sources[0]), fixed_variant_plan=plan),
        len(assignment) if num_envs is None else num_envs,
        _SIM_DT,
        base_name="tool",
        worker_command=[
            sys.executable,
            str(_MOCK_WORKER),
            "--record",
            str(record),
            *extra_worker_args,
        ],
        worker_timeout_s=30.0,
    )
    assert isinstance(backend, IsaacGymBackend)
    return backend


def test_fixed_variant_plan_reaches_worker_and_advertises_capabilities(
    tmp_path: Path,
) -> None:
    sources = _write_variants(tmp_path)
    record = tmp_path / "init.json"
    backend = _make_backend(sources, (0, 1, 2, 0), record)
    try:
        capabilities = backend.get_dr_capabilities()
        assert capabilities.supports_fixed_variants
        assert capabilities.supported_fixed_variant_layouts == frozenset(
            {FixedVariantLayout.SAME_LAYOUT, FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT}
        )
        assert capabilities.supports_per_env_playback

        backend.materialize()
        payload: dict[str, Any] = json.loads(record.read_text(encoding="utf-8"))
        # Ablation guard: the variant path must not also carry the duplicated
        # legacy single-model actuation/keyframe fields.
        assert "keyframe_qpos" not in payload
        assert "dof_stiffness" not in payload
        assert "dof_damping" not in payload
        assert [Path(value).resolve() for value in payload["variant_model_files"]] == [
            path.resolve() for path in sources
        ]
        assert payload["variant_assignment"] == [0, 1, 2, 0]
        # The worker-side FK tables (#141) ship per variant and share the
        # public name/column contract with the metadata echo.
        variant_kinematics = payload["variant_mjcf_kinematics"]
        assert len(variant_kinematics) == 3
        for tables in variant_kinematics:
            assert tables["schema_version"] == 1
            assert tables["body_names"] == payload["mjcf_body_names"] == ["tool"]
            assert tables["joint_names"] == payload["mjcf_joint_names"] == ["tool_pitch"]
            assert tables["free_root"] == -1  # the synthetic tool asset has no freejoint
            assert tables["body_joint_column"] == [7]
        fields = payload["variant_dof_fields"]
        assert [row["stiffness"] for row in fields] == [[20.0], [30.0], [40.0]]
        np.testing.assert_allclose(
            payload["variant_keyframe_qpos"],
            [
                [0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2],
            ],
            rtol=1e-6,
        )

        assert [backend.get_playback_model(index) for index in range(4)] == [
            str(sources[0]),
            str(sources[1]),
            str(sources[2]),
            str(sources[0]),
        ]
        np.testing.assert_allclose(
            backend.get_default_dof_pos(), np.asarray([0.0], dtype=np.float32)
        )
    finally:
        backend.close()


def test_init_payload_carries_mjcf_kinematics(tmp_path: Path) -> None:
    sources = _write_variants(tmp_path, count=1)
    record = tmp_path / "init.json"
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=str(sources[0])),
        2,
        _SIM_DT,
        base_name="tool",
        worker_command=[sys.executable, str(_MOCK_WORKER), "--record", str(record)],
        worker_timeout_s=30.0,
    )
    try:
        backend.materialize()
        payload: dict[str, Any] = json.loads(record.read_text(encoding="utf-8"))
        tables = payload["mjcf_kinematics"]
        assert tables["schema_version"] == 1
        assert tables["body_names"] == payload["mjcf_body_names"] == ["tool"]
        assert tables["joint_names"] == payload["mjcf_joint_names"] == ["tool_pitch"]
        assert tables["body_parent"] == [-1]
        assert tables["body_joint_column"] == [7]
        assert "variant_mjcf_kinematics" not in payload
    finally:
        backend.close()


def test_public_layout_drift_fails_closed_before_worker_spawn(tmp_path: Path) -> None:
    sources = list(_write_variants(tmp_path, count=2))
    bad = tmp_path / "bad.xml"
    bad.write_text(
        _variant_xml(tool_mass=1.0, tool_size=0.2, key_dof=0.3, kp=50.0).replace(
            "</body>\n  </worldbody>",
            '<body name="extra"><joint name="extra_joint" type="hinge"/></body>'
            "</body>\n  </worldbody>",
        ),
        encoding="utf-8",
    )
    sources.append(bad)
    record = tmp_path / "init.json"
    backend = _make_backend(tuple(sources), (0, 1, 0), record)
    try:
        with pytest.raises(ValueError, match="changes public joint_names"):
            backend.materialize()
        assert not record.exists()
    finally:
        backend.close()


def test_worker_must_echo_immutable_assignment(tmp_path: Path) -> None:
    sources = _write_variants(tmp_path, count=2)
    record = tmp_path / "init.json"
    backend = _make_backend(
        sources,
        (0, 1),
        record,
        extra_worker_args=("--omit-variant-echo",),
    )
    try:
        with pytest.raises(Exception, match="fixed-variant handshake"):
            backend.materialize()
    finally:
        backend.close()


def test_playback_model_validates_env_index(tmp_path: Path) -> None:
    sources = _write_variants(tmp_path, count=2)
    backend = _make_backend(sources, (0, 1), tmp_path / "init.json")
    try:
        with pytest.raises(ValueError, match="explicit env_index"):
            backend.get_playback_model(None)
        with pytest.raises(IndexError):
            backend.get_playback_model(2)
    finally:
        backend.close()
