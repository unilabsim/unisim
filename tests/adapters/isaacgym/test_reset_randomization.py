"""Mapped-scene reset randomization and interval wrench coverage (no SDK)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.base import BackendPlayCapabilities
from unisim.backend.isaacgym.backend import IsaacGymBackend, IsaacGymWorkerError
from unisim.capabilities import SupportLevel, backend_capabilities
from unisim.dr.interval import (
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_TORQUE,
    IntervalTermOp,
)
from unisim.dr.types import (
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_DOF_FRICTIONLOSS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_KD,
    RESET_TERM_KP,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, GeomLayout, JointLayout

_MOCK_WORKER = Path(__file__).resolve().parent / "mock_worker.py"

_ROBOT = EntityLayout(
    "robot",
    "articulation",
    "fixed",
    "base",
    ("base", "finger"),
    (0, 1),
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
    (2, 3),
    (None, "base"),
    (JointLayout("passive", "hinge", (8,), (7,), "lid"),),
    (),
    (),
    (),
    tuple(range(1, 8)),
    tuple(range(1, 7)),
    (GeomLayout("base::geom0", "base"),),
)
_LAYOUT = CompiledSceneLayout((_ROBOT, _OBJECT), nq=9, nv=8, nu=1, nbody=4, ngeom=3)
_NUM_ENVS = 3


def _default_records() -> dict[str, dict]:
    records: dict[str, dict] = {}
    for entity in _LAYOUT.entities:
        nb, nj, ng = len(entity.body_ids), len(entity.joints), len(entity.geoms)
        records[entity.name] = {
            "name": entity.name,
            "body_mass": [[1.0] * nb for _ in range(_NUM_ENVS)],
            "body_ipos": [[[0.0, 0.0, 0.0]] * nb for _ in range(_NUM_ENVS)],
            "body_inertia": [
                [np.diag([0.01, 0.01, 0.01]).tolist()] * nb for _ in range(_NUM_ENVS)
            ],
            "dof_stiffness": [[20.0] * nj for _ in range(_NUM_ENVS)],
            "dof_damping": [[1.0] * nj for _ in range(_NUM_ENVS)],
            "dof_armature": [[0.0] * nj for _ in range(_NUM_ENVS)],
            "dof_friction": [[0.0] * nj for _ in range(_NUM_ENVS)],
            "geom_friction": [[[0.5, 0.5, 0.0]] * ng for _ in range(_NUM_ENVS)],
        }
    return records


def _mapped_backend() -> IsaacGymBackend:
    backend = IsaacGymBackend.__new__(IsaacGymBackend)
    backend.backend_type = "isaacgym"
    backend._num_envs = _NUM_ENVS
    backend._fixed_variant_plan = None
    backend._entity_scene = SimpleNamespace(
        layout=_LAYOUT, owner=SimpleNamespace(variant_plan=None)
    )
    backend._native_entity_records = _default_records()
    backend._staged_body_wrench = np.zeros((_NUM_ENVS, _LAYOUT.nbody, 6), dtype=np.float32)
    backend._body_wrench_pending = False
    backend._model_info = object()
    backend._worker_dead_error = None
    backend._slots = {"ctrl": np.zeros((_NUM_ENVS, _LAYOUT.nu), dtype=np.float32)}
    backend._import_report = SimpleNamespace(profile="default")
    backend._play_capabilities = BackendPlayCapabilities()
    return backend


def test_mapped_capabilities_declare_bounded_reset_and_interval_terms() -> None:
    capabilities = _mapped_backend().get_dr_capabilities()
    assert capabilities.supported_reset_terms == frozenset(
        {
            RESET_TERM_KP,
            RESET_TERM_KD,
            RESET_TERM_BODY_MASS,
            RESET_TERM_BODY_INERTIA,
            RESET_TERM_BODY_IPOS,
            RESET_TERM_DOF_ARMATURE,
            RESET_TERM_DOF_FRICTIONLOSS,
            RESET_TERM_GEOM_FRICTION,
        }
    )
    assert capabilities.supports_interval_term(INTERVAL_TERM_BODY_FORCE)
    assert capabilities.supports_interval_term(INTERVAL_TERM_BODY_TORQUE)
    # PhysX gravity is sim-wide, passive joint damping has no PhysX channel, and
    # a multi-root scene has no single base target: those terms stay unsupported.
    for term in ("gravity", "dof_damping", "base_mass_delta", "base_com_offset", "body_iquat"):
        assert not capabilities.supports_reset_term(term)
    # No fixed-variant plan is bound to this scene.
    assert not capabilities.supports_fixed_variants


def test_derived_capability_report_marks_mapped_reset_terms_exact() -> None:
    report = backend_capabilities(_mapped_backend())
    for term in ("kp", "kd", "body_mass", "body_ipos", "body_inertia", "geom_friction"):
        declaration = report.get("dr.reset." + term)
        assert declaration is not None
        assert declaration.support == SupportLevel.EXACT
    assert report.get("dr.reset.gravity").support == SupportLevel.UNSUPPORTED
    assert report.get("dr.interval.body_force").support == SupportLevel.EXACT
    assert report.get("dr.interval.body_torque").support == SupportLevel.EXACT
    fixed = report.get("root.fixed", configuration={"entity.asset_format": "mjcf"})
    assert fixed.support == SupportLevel.EXACT


def test_legacy_capabilities_stay_fail_closed() -> None:
    backend = _mapped_backend()
    backend._entity_scene = None
    backend._staged_body_wrench = None
    capabilities = backend.get_dr_capabilities()
    assert not capabilities.supported_reset_terms
    assert not capabilities.supported_interval_terms
    assert not capabilities.supports_fixed_variants


def _set_state_backend():
    backend = _mapped_backend()
    commits = []

    def commit(request, controls=None, randomization=None):
        commits.append((request, controls, randomization))

    backend._commit_entity_reset = commit
    return backend, commits


def _full_state_rows(count):
    qpos = np.zeros((count, _LAYOUT.nq), dtype=np.float32)
    qpos[:, 4] = 1.0
    qvel = np.zeros((count, _LAYOUT.nv), dtype=np.float32)
    return qpos, qvel


def test_mapped_reset_randomization_validates_and_forwards_all_terms() -> None:
    backend, commits = _set_state_backend()
    payload = ResetRandomizationPayload(
        kp=np.full((2, 1), 30.0),
        kd=np.full((2, 1), 2.0),
        body_mass=np.full((2, 4), 2.5),
        body_ipos=np.full((2, 4, 3), 0.01),
        body_inertia=np.full((2, 4, 3), 0.02),
        dof_armature=np.zeros((2, 8)),
        dof_frictionloss=np.zeros((2, 8)),
        geom_friction=np.tile(np.array([0.7, 0.7, 0.0]), (2, 3, 1)),
    )
    payload.dof_armature[:, 7] = 0.1
    payload.dof_frictionloss[:, 0] = 0.2
    backend._set_mapped_state(np.array([0, 2]), *_full_state_rows(2), payload)
    assert len(commits) == 1
    validated = commits[0][2]
    assert validated.requested_terms() == payload.requested_terms()
    for field in (
        "kp",
        "kd",
        "body_mass",
        "body_ipos",
        "body_inertia",
        "dof_armature",
        "dof_frictionloss",
        "geom_friction",
    ):
        assert getattr(validated, field).dtype == np.float32


@pytest.mark.parametrize(
    "term",
    ["gravity", "dof_damping", "base_mass_delta", "base_com_offset", "body_iquat", "geom_size"],
)
def test_mapped_reset_unsupported_terms_fail_closed(term: str) -> None:
    backend, commits = _set_state_backend()
    tail = {
        "gravity": (3,),
        "dof_damping": (8,),
        "base_mass_delta": (),
        "base_com_offset": (3,),
        "body_iquat": (4, 4),
        "geom_size": (3, 3),
    }[term]
    payload = ResetRandomizationPayload(**{term: np.zeros((1, *tail), dtype=np.float32)})
    with pytest.raises(NotImplementedError, match=term):
        backend._set_mapped_state(np.array([1]), *_full_state_rows(1), payload)
    assert commits == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kp", [[[30.0]]], "kp must have shape"),
        ("kp", [[-1.0]], "kp values must be nonnegative"),
        ("kd", [[-0.5]], "kd values must be nonnegative"),
        ("body_mass", [[0.0, 1.0, 1.0, 1.0]], "body_mass values must be positive"),
        ("body_inertia", [[[0.0, 0.1, 0.1]] * 4], "body_inertia values must be positive"),
        ("dof_armature", [[0.0] * 8], None),
        ("geom_friction", [[[0.1, 0.2, 0.0]] * 3], "static == dynamic"),
        ("geom_friction", [[[0.1, 0.1, 0.5]] * 3], "zero third column"),
        ("geom_friction", [[[-0.1, -0.1, 0.0]] * 3], "nonnegative"),
    ],
)
def test_mapped_reset_field_validation(field, value, message) -> None:
    backend, commits = _set_state_backend()
    if field == "dof_armature":
        value = np.asarray(value, dtype=np.float32)
        value[0, 3] = 0.5  # floating-root column
        message = "zero on floating-root columns"
    payload = ResetRandomizationPayload(**{field: np.asarray(value, dtype=np.float32)})
    with pytest.raises(ValueError, match=message):
        backend._set_mapped_state(np.array([1]), *_full_state_rows(1), payload)
    assert commits == []


def test_consume_reset_randomization_merges_worker_records() -> None:
    backend = _mapped_backend()
    records = _default_records()
    for entity in _LAYOUT.entities:
        records[entity.name]["body_mass"] = [[5.0] * len(entity.body_ids)] * _NUM_ENVS
    backend._consume_entity_reset_randomization({"native_entity_records": list(records.values())})
    assert backend._native_entity_records["robot"]["body_mass"][0] == [5.0, 5.0]
    assert backend._native_entity_records["object"]["dof_stiffness"][2] == [20.0]


@pytest.mark.parametrize("tamper", ["missing_entity", "bad_shape", "nan", "not_a_list"])
def test_consume_reset_randomization_rejects_malformed_records(tamper: str) -> None:
    backend = _mapped_backend()
    records = _default_records()
    if tamper == "missing_entity":
        del records["object"]
    elif tamper == "bad_shape":
        records["robot"]["body_mass"] = [[1.0]]
    elif tamper == "nan":
        records["robot"]["dof_stiffness"][0][0] = float("nan")
    response = {"native_entity_records": list(records.values())}
    if tamper == "not_a_list":
        response = {"native_entity_records": {"robot": {}}}
    with pytest.raises(IsaacGymWorkerError):
        backend._consume_entity_reset_randomization(response)


def test_interval_body_wrench_stages_into_step_payload_and_clears() -> None:
    backend = _mapped_backend()
    force = np.full((_NUM_ENVS, 1, 3), 4.0, dtype=np.float32)
    torque = np.full((_NUM_ENVS, 1, 3), 0.5, dtype=np.float32)
    backend.apply_body_force(np.asarray([2]), force, torque=torque)
    assert backend._body_wrench_pending
    payload = backend._step_payload(1)
    wrench = np.frombuffer(payload["body_wrench"], dtype=np.float32).reshape(
        _NUM_ENVS, _LAYOUT.nbody, 6
    )
    np.testing.assert_allclose(wrench[:, 2], [[4.0, 4.0, 4.0, 0.5, 0.5, 0.5]] * _NUM_ENVS)
    backend._after_step(payload)
    assert not backend._body_wrench_pending
    assert not np.any(backend._staged_body_wrench)
    assert "body_wrench" not in backend._step_payload(1)


def test_interval_ops_accumulate_and_new_plan_replaces_staging() -> None:
    backend = _mapped_backend()
    plan = IntervalRandomizationPlan(
        ops=(
            IntervalTermOp(
                INTERVAL_TERM_BODY_TORQUE,
                np.full((_NUM_ENVS, 1, 3), 2.0, dtype=np.float32),
                body_ids=np.asarray([1]),
            ),
        )
    )
    backend.apply_interval_randomization(plan)
    np.testing.assert_allclose(backend._staged_body_wrench[:, 1, 3:6], 2.0)
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(
                IntervalTermOp(
                    INTERVAL_TERM_BODY_FORCE,
                    np.full((_NUM_ENVS, 1, 3), 3.0, dtype=np.float32),
                    body_ids=np.asarray([0]),
                ),
            )
        )
    )
    assert not np.any(backend._staged_body_wrench[:, 1])
    np.testing.assert_allclose(backend._staged_body_wrench[:, 0, 0:3], 3.0)


def test_interval_body_force_validates_targets_and_values() -> None:
    backend = _mapped_backend()
    with pytest.raises(ValueError, match="one-dimensional integer"):
        backend.apply_body_force(np.asarray([[0]]), np.zeros((_NUM_ENVS, 1, 3)))
    with pytest.raises(ValueError, match="body ids must be in"):
        backend.apply_body_force(np.asarray([_LAYOUT.nbody]), np.zeros((_NUM_ENVS, 1, 3)))
    with pytest.raises(ValueError, match="body force must have shape"):
        backend.apply_body_force(np.asarray([0]), np.zeros((_NUM_ENVS, 2, 3)))
    with pytest.raises(ValueError, match="NaN or Inf"):
        backend.apply_body_force(
            np.asarray([0]), np.full((_NUM_ENVS, 1, 3), np.nan, dtype=np.float32)
        )
    with pytest.raises(ValueError, match="requires body_ids"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(
                ops=(
                    IntervalTermOp(
                        INTERVAL_TERM_BODY_FORCE,
                        np.zeros((_NUM_ENVS, 1, 3), dtype=np.float32),
                    ),
                )
            )
        )


def test_legacy_interval_wrench_fails_closed() -> None:
    backend = _mapped_backend()
    backend._entity_scene = None
    backend._staged_body_wrench = None
    with pytest.raises(NotImplementedError, match="does not support interval body force"):
        backend.apply_body_force(np.asarray([0]), np.zeros((_NUM_ENVS, 1, 3)))
    plan = IntervalRandomizationPlan(
        ops=(
            IntervalTermOp(
                INTERVAL_TERM_BODY_FORCE,
                np.zeros((_NUM_ENVS, 1, 3), dtype=np.float32),
                body_ids=np.asarray([0]),
            ),
        )
    )
    with pytest.raises(NotImplementedError, match="does not support interval term"):
        backend.apply_interval_randomization(plan)


def _mock_scene(root: Path):
    pytest.importorskip("mujoco")
    from unisim.dr.types import ModelSourceDescriptor
    from unisim.entities import SceneEntitySpec
    from unisim.scene import SceneCfg

    root.mkdir(parents=True, exist_ok=True)
    path = root / "robot.xml"
    path.write_text(
        """<mujoco><worldbody><body name="base"><freejoint/>
  <inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
  <geom name="handle" type="sphere" size=".1"/>
  <body name="tip" pos="0 0 .2"><joint name="pitch" type="hinge" axis="0 1 0"/>
    <inertial mass=".5" pos="0 0 0" diaginertia=".001 .001 .001"/>
    <geom name="tip_geom" type="sphere" size=".05"/></body></body></worldbody>
  <actuator><position name="drive" joint="pitch" kp="10" kv="1"/></actuator></mujoco>""",
        encoding="utf-8",
    )
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", ModelSourceDescriptor(str(path)), kind="articulation"),
        )
    )


def test_mock_worker_roundtrip_applies_reset_randomization(tmp_path: Path) -> None:
    from unisim.factory import create_backend

    num_envs = 4
    backend = create_backend(
        "isaacgym",
        _mock_scene(tmp_path),
        num_envs,
        0.002,
        base_name="robot",
        worker_command=[sys.executable, str(_MOCK_WORKER)],
        worker_timeout_s=30.0,
    )
    try:
        backend.materialize()
        layout = backend._entity_scene.layout
        assert layout.nbody == 3 and layout.nu == 1
        capabilities = backend.get_dr_capabilities()
        assert RESET_TERM_KP in capabilities.supported_reset_terms
        qpos = np.zeros((num_envs, layout.nq), dtype=np.float32)
        qpos[:, 3] = 1.0
        qvel = np.zeros((num_envs, layout.nv), dtype=np.float32)
        randomization = ResetRandomizationPayload(
            kp=np.full((num_envs, 1), 42.0),
            body_mass=np.full((num_envs, layout.nbody), 3.0),
            dof_frictionloss=np.zeros((num_envs, layout.nv)),
            geom_friction=np.tile(
                np.array([0.9, 0.9, 0.0]), (num_envs, layout.ngeom, 1)
            ),
        )
        backend.set_state(np.arange(num_envs), qpos, qvel, randomization=randomization)
        record = backend._native_entity_records["robot"]
        np.testing.assert_allclose(record["body_mass"], 3.0)
        np.testing.assert_allclose(record["dof_stiffness"], 42.0)
        np.testing.assert_allclose(
            record["geom_friction"],
            np.tile(np.array([0.9, 0.9, 0.0]), (num_envs, layout.ngeom, 1)),
        )
        assert "body_ipos" in record and "body_inertia" in record

        backend.apply_interval_randomization(
            IntervalRandomizationPlan(
                ops=(
                    IntervalTermOp(
                        INTERVAL_TERM_BODY_FORCE,
                        np.ones((num_envs, 1, 3), dtype=np.float32),
                        body_ids=np.asarray([0]),
                    ),
                )
            )
        )
        backend.step(np.zeros((num_envs, layout.nu), dtype=np.float32))
        assert not backend._body_wrench_pending
    finally:
        backend.close()


def test_mock_worker_rejects_unsupported_term_before_wire(tmp_path: Path) -> None:
    from unisim.factory import create_backend

    num_envs = 2
    backend = create_backend(
        "isaacgym",
        _mock_scene(tmp_path / "scene"),
        num_envs,
        0.002,
        base_name="robot",
        worker_command=[sys.executable, str(_MOCK_WORKER)],
        worker_timeout_s=30.0,
    )
    try:
        backend.materialize()
        layout = backend._entity_scene.layout
        qpos = np.zeros((num_envs, layout.nq), dtype=np.float32)
        qpos[:, 3] = 1.0
        qvel = np.zeros((num_envs, layout.nv), dtype=np.float32)
        with pytest.raises(NotImplementedError, match="gravity"):
            backend.set_state(
                np.arange(num_envs),
                qpos,
                qvel,
                randomization=ResetRandomizationPayload(
                    gravity=np.zeros((num_envs, 3), dtype=np.float32)
                ),
            )
    finally:
        backend.close()
