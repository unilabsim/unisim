"""Mapped-scene reset term default tables for the IsaacGym host (no SDK)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.isaacgym.backend import IsaacGymBackend, IsaacGymWorkerError
from unisim.dr.types import (
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_DOF_FRICTIONLOSS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_KD,
    RESET_TERM_KP,
)
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, GeomLayout, JointLayout

_MOCK_WORKER = Path(__file__).resolve().parent / "mock_worker.py"

_ROBOT = EntityLayout(
    "robot",
    "articulation",
    "fixed",
    "base",
    ("base", "finger"),
    (1, 2),
    (None, "base"),
    (JointLayout("drive", "hinge", (0,), (0,), "finger"),),
    ("drive",),
    ("drive",),
    (0,),
    (),
    (),
    (GeomLayout("base::geom0", "base"), GeomLayout("finger::geom0", "finger")),
)
_OBJECT = EntityLayout(
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
    (GeomLayout("base::geom0", "base"),),
)
# Body column 0 is not owned by any entity; defaults keep the canonical table there.
_LAYOUT = CompiledSceneLayout((_ROBOT, _OBJECT), nq=9, nv=8, nu=1, nbody=5, ngeom=3)
_NUM_ENVS = 3
_ASSIGNMENT = [1, 0, 1]


def _variant_records() -> tuple[dict, dict]:
    return (
        {
            "body_mass": [1.0, 2.0],
            "body_ipos": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "body_inertia": [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
            "dof_stiffness": [0.0],
            "dof_damping": [0.0],
            "dof_armature": [0.3],
            "dof_friction": [0.4],
            "geom_friction": [[0.7, 0.01, 0.0]],
        },
        {
            "body_mass": [3.0, 4.0],
            "body_ipos": [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
            "body_inertia": [[0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
            "dof_stiffness": [0.0],
            "dof_damping": [0.0],
            "dof_armature": [0.5],
            "dof_friction": [0.6],
            "geom_friction": [[0.9, 0.01, 0.0]],
        },
    )


def _scene_entries() -> list[dict]:
    object_v0, object_v1 = _variant_records()
    return [
        {
            "name": "robot",
            "assignment": [0] * _NUM_ENVS,
            "variants": [
                {
                    "body_mass": [10.0, 11.0],
                    "body_ipos": [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
                    "body_inertia": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    "dof_stiffness": [20.0],
                    "dof_damping": [2.0],
                    "dof_armature": [0.1],
                    "dof_friction": [0.2],
                    "geom_friction": [[0.5, 0.01, 0.0], [0.6, 0.01, 0.0]],
                }
            ],
        },
        {
            "name": "object",
            "assignment": list(_ASSIGNMENT),
            "variants": [object_v0, object_v1],
        },
    ]


def _mapped_backend() -> IsaacGymBackend:
    backend = IsaacGymBackend.__new__(IsaacGymBackend)
    backend._num_envs = _NUM_ENVS
    backend._fixed_variant_plan = None
    backend._entity_scene = SimpleNamespace(
        layout=_LAYOUT,
        payload={"scene_entities": _scene_entries()},
        owner=SimpleNamespace(
            variant_plan=None,
            model=SimpleNamespace(
                body_mass=np.arange(100, 105, dtype=np.float32),
                body_ipos=np.arange(15, dtype=np.float32).reshape(5, 3) / 7,
                body_inertia=np.full((5, 3), 0.5, dtype=np.float32),
            ),
        ),
    )
    return backend


def test_mapped_body_defaults_are_per_env_variant_tables() -> None:
    backend = _mapped_backend()
    mass = backend.get_reset_term_default(RESET_TERM_BODY_MASS)
    assert mass.dtype == np.float32
    assert not mass.flags.writeable
    np.testing.assert_array_equal(
        mass,
        [
            [100.0, 10.0, 11.0, 3.0, 4.0],
            [100.0, 10.0, 11.0, 1.0, 2.0],
            [100.0, 10.0, 11.0, 3.0, 4.0],
        ],
    )
    ipos = backend.get_reset_term_default(RESET_TERM_BODY_IPOS)
    assert ipos.shape == (3, 5, 3)
    canonical_row = np.arange(15, dtype=np.float32).reshape(5, 3)[0] / 7
    np.testing.assert_array_equal(ipos[:, 0], np.broadcast_to(canonical_row, (3, 3)))
    np.testing.assert_allclose(ipos[:, 1], [[0.1, 0.0, 0.0]] * 3, rtol=1e-6)
    np.testing.assert_allclose(ipos[0, 3:], [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]], rtol=1e-6)
    np.testing.assert_array_equal(ipos[1, 3:], 0.0)
    inertia = backend.get_reset_term_default(RESET_TERM_BODY_INERTIA)
    np.testing.assert_array_equal(inertia[:, 0], 0.5)
    np.testing.assert_array_equal(inertia[:, 1], 1.0)
    np.testing.assert_array_equal(inertia[:, 2], 2.0)
    np.testing.assert_allclose(inertia[0, 3:], [[0.3, 0.3, 0.3], [0.4, 0.4, 0.4]], rtol=1e-6)
    np.testing.assert_allclose(inertia[1, 3:], [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], rtol=1e-6)


def test_mapped_actuator_dof_and_geom_defaults_follow_assignment() -> None:
    backend = _mapped_backend()
    np.testing.assert_array_equal(backend.get_reset_term_default(RESET_TERM_KP), [[20.0]] * 3)
    np.testing.assert_array_equal(backend.get_reset_term_default(RESET_TERM_KD), [[2.0]] * 3)
    armature = backend.get_reset_term_default(RESET_TERM_DOF_ARMATURE)
    assert armature.shape == (3, 8)
    np.testing.assert_allclose(armature[:, 0], 0.1, rtol=1e-6)
    np.testing.assert_allclose(armature[:, 7], [0.5, 0.3, 0.5], rtol=1e-6)
    np.testing.assert_array_equal(armature[:, 1:7], 0.0)
    friction = backend.get_reset_term_default(RESET_TERM_DOF_FRICTIONLOSS)
    np.testing.assert_allclose(friction[:, 0], 0.2, rtol=1e-6)
    np.testing.assert_allclose(friction[:, 7], [0.6, 0.4, 0.6], rtol=1e-6)
    np.testing.assert_array_equal(friction[:, 1:7], 0.0)
    geom = backend.get_reset_term_default(RESET_TERM_GEOM_FRICTION)
    assert geom.shape == (3, 3, 3)
    np.testing.assert_allclose(geom[:, 0], [[0.5, 0.5, 0.0]] * 3, rtol=1e-6)
    np.testing.assert_allclose(geom[:, 1], [[0.6, 0.6, 0.0]] * 3, rtol=1e-6)
    np.testing.assert_allclose(
        geom[:, 2], [[0.9, 0.9, 0.0], [0.7, 0.7, 0.0], [0.9, 0.9, 0.0]], rtol=1e-6
    )


def test_mapped_reset_term_default_fails_closed() -> None:
    backend = _mapped_backend()
    with pytest.raises(ValueError, match="unknown reset term"):
        backend.get_reset_term_default("not_a_reset_term")
    for term in ("gravity", "dof_damping", "base_mass_delta", "base_com_offset", "body_iquat"):
        with pytest.raises(NotImplementedError, match="does not support reset term"):
            backend.get_reset_term_default(term)


def test_legacy_reset_term_default_fails_closed() -> None:
    backend = _mapped_backend()
    backend._entity_scene = None
    with pytest.raises(NotImplementedError, match="does not support reset term"):
        backend.get_reset_term_default(RESET_TERM_BODY_MASS)


def test_mapped_reset_term_default_rejects_malformed_variant_records() -> None:
    backend = _mapped_backend()
    entries = backend._entity_scene.payload["scene_entities"]
    entries[0]["variants"][0]["body_mass"] = [10.0]
    with pytest.raises(IsaacGymWorkerError, match="compiled variant body_mass is malformed"):
        backend.get_reset_term_default(RESET_TERM_BODY_MASS)
    entries[1]["variants"][1]["body_mass"][0] = float("nan")
    with pytest.raises(IsaacGymWorkerError, match="compiled variant body_mass is malformed"):
        backend.get_reset_term_default(RESET_TERM_BODY_MASS)


def _variant_scene(root: Path):
    pytest.importorskip("mujoco")
    from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
    from unisim.entities import EntityVariantBinding, SceneEntitySpec
    from unisim.scene import SceneCfg

    root.mkdir(parents=True, exist_ok=True)
    template = """<mujoco><worldbody><body name="base">
  <inertial mass="{mass}" pos="0 0 0" diaginertia=".01 .01 .01"/>
  <geom name="handle" type="sphere" size=".1"/>
  <body name="tip" pos="0 0 .2"><joint name="pitch" type="hinge" axis="0 1 0"/>
    <inertial mass=".5" pos="0 0 0" diaginertia=".001 .001 .001"/>
    <geom name="tip_geom" type="sphere" size=".05"/></body></body></worldbody>
  <actuator><position name="drive" joint="pitch" kp="{kp}" kv="1"/></actuator></mujoco>"""
    sources = []
    for index, (mass, kp) in enumerate(((1.0, 10.0), (2.0, 30.0))):
        path = root / f"tool_{index}.xml"
        path.write_text(template.format(mass=mass, kp=kp), encoding="utf-8")
        sources.append(ModelSourceDescriptor(str(path)))
    plan = FixedVariantPlan(np.asarray((1, 0, 1), dtype=np.int32), tuple(sources))
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec("tool", sources[0], kind="articulation", root_mode="fixed"),
        ),
        entity_variant=EntityVariantBinding("tool", plan),
    )


def test_mock_worker_defaults_match_spawned_variant_assignment(tmp_path: Path) -> None:
    from unisim.factory import create_backend

    backend = create_backend(
        "isaacgym",
        _variant_scene(tmp_path),
        3,
        0.002,
        base_name="tool",
        worker_command=[sys.executable, str(_MOCK_WORKER)],
        worker_timeout_s=30.0,
    )
    try:
        backend.materialize()
        layout = backend._entity_scene.layout
        assert layout.nbody == 3 and layout.nu == 1
        mass = backend.get_reset_term_default(RESET_TERM_BODY_MASS)
        np.testing.assert_allclose(mass[:, 1], [2.0, 1.0, 2.0])
        np.testing.assert_allclose(mass[:, 2], 0.5)
        np.testing.assert_allclose(
            backend.get_reset_term_default(RESET_TERM_KP), [[30.0], [10.0], [30.0]]
        )
        np.testing.assert_allclose(backend.get_reset_term_default(RESET_TERM_KD), [[1.0]] * 3)
        np.testing.assert_allclose(
            backend.get_reset_term_default(RESET_TERM_GEOM_FRICTION),
            np.tile(np.array([1.0, 1.0, 0.0]), (3, layout.ngeom, 1)),
        )
        np.testing.assert_allclose(
            backend.get_reset_term_default(RESET_TERM_DOF_ARMATURE),
            np.zeros((3, layout.nv)),
        )
        for term in (
            RESET_TERM_BODY_MASS,
            RESET_TERM_BODY_IPOS,
            RESET_TERM_BODY_INERTIA,
            RESET_TERM_GEOM_FRICTION,
            RESET_TERM_KP,
            RESET_TERM_KD,
            RESET_TERM_DOF_ARMATURE,
            RESET_TERM_DOF_FRICTIONLOSS,
        ):
            default = backend.get_reset_term_default(term)
            assert default.dtype == np.float32
            assert not default.flags.writeable
    finally:
        backend.close()
