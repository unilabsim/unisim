"""Tests for the declarative interval domain-randomization term contract."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from unisim.dr import (
    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_TORQUE,
    INTERVAL_TERM_PUSH,
    INTERVAL_TERM_SPECS,
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    IntervalTermOp,
    IntervalTermSpec,
    interval_term_spec,
    validate_interval_op,
)
from unisim.fake import FakeBackend

BUILTIN_TERMS = (
    INTERVAL_TERM_PUSH,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_TORQUE,
)

BODY_IDS = np.asarray([1], dtype=np.int32)


def test_interval_term_spec_registry_covers_builtin_terms() -> None:
    assert tuple(spec.name for spec in INTERVAL_TERM_SPECS) == BUILTIN_TERMS
    for name in BUILTIN_TERMS:
        assert interval_term_spec(name) is not None
    assert isinstance(interval_term_spec(INTERVAL_TERM_PUSH), IntervalTermSpec)
    with pytest.raises(KeyError):
        interval_term_spec("no_such_term")


def test_interval_term_op_validation_builtin_contract() -> None:
    payload = np.zeros((2, 1, 3))
    validate_interval_op(IntervalTermOp(INTERVAL_TERM_BODY_FORCE, payload, body_ids=BODY_IDS))
    with pytest.raises(ValueError, match="body_force"):
        IntervalTermOp(INTERVAL_TERM_BODY_FORCE, payload).validate()
    with pytest.raises(ValueError, match="push"):
        IntervalTermOp(INTERVAL_TERM_PUSH, np.zeros(3), body_ids=BODY_IDS).validate()
    with pytest.raises(ValueError, match="body_force"):
        IntervalTermOp(INTERVAL_TERM_BODY_FORCE, np.zeros(3), body_ids=BODY_IDS).validate()
    with pytest.raises(ValueError, match="push"):
        IntervalTermOp(INTERVAL_TERM_PUSH, np.zeros((2, 3))).validate()


def test_interval_term_op_validation_passes_custom_terms() -> None:
    # Custom terms are backend-owned: no spec, no validation.
    op = IntervalTermOp("custom_spin", np.zeros((7, 2)))
    op.validate()
    validate_interval_op(op)


def test_interval_term_op_and_plan_pickle_round_trip() -> None:
    op = IntervalTermOp(INTERVAL_TERM_BODY_FORCE, np.ones((2, 1, 3)), body_ids=BODY_IDS)
    plan = IntervalRandomizationPlan(ops=(op, IntervalTermOp("custom", np.zeros(4))))
    restored = pickle.loads(pickle.dumps(plan, protocol=4))
    assert [restored_op.term for restored_op in restored.ops] == ["body_force", "custom"]
    np.testing.assert_array_equal(restored.ops[0].payload, op.payload)
    np.testing.assert_array_equal(restored.ops[0].body_ids, BODY_IDS)


def test_plan_iter_ops_derives_legacy_fields() -> None:
    lin = np.zeros((2, 1, 3))
    force = np.ones((2, 1, 3))
    plan = IntervalRandomizationPlan(
        push_perturbation_limit=[1.0, 2.0, 3.0],
        body_ids=BODY_IDS,
        body_linear_velocity_delta=lin,
        body_force=force,
    )
    ops = plan.iter_ops()
    assert [op.term for op in ops] == [
        INTERVAL_TERM_PUSH,
        INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
        INTERVAL_TERM_BODY_FORCE,
    ]
    np.testing.assert_array_equal(ops[0].payload, np.asarray([1.0, 2.0, 3.0]))
    assert ops[0].body_ids is None
    assert ops[1].payload is lin
    assert ops[1].body_ids is BODY_IDS
    assert ops[2].payload is force
    assert ops[2].body_ids is BODY_IDS


def test_plan_is_empty_semantics() -> None:
    assert IntervalRandomizationPlan().is_empty()
    # body_ids alone does not make a plan non-empty.
    assert IntervalRandomizationPlan(body_ids=BODY_IDS).is_empty()
    assert not IntervalRandomizationPlan(push_perturbation_limit=[1.0, 1.0, 1.0]).is_empty()
    ops_only = IntervalRandomizationPlan(ops=(IntervalTermOp("custom", np.zeros(1)),))
    assert not ops_only.is_empty()
    mixed = IntervalRandomizationPlan(
        push_perturbation_limit=[1.0, 1.0, 1.0],
        ops=(IntervalTermOp("custom", np.zeros(1)),),
    )
    assert not mixed.is_empty()
    assert [op.term for op in mixed.iter_ops()] == [INTERVAL_TERM_PUSH, "custom"]


def test_capabilities_supported_interval_terms_and_legacy_fallback() -> None:
    legacy = DomainRandomizationCapabilities(
        supports_interval_push=True,
        supports_interval_body_force=True,
    )
    assert legacy.supports_interval_term(INTERVAL_TERM_PUSH)
    assert legacy.supports_interval_term(INTERVAL_TERM_BODY_FORCE)
    assert not legacy.supports_interval_term(INTERVAL_TERM_BODY_TORQUE)
    assert not legacy.supports_interval_term("custom")

    declared = DomainRandomizationCapabilities(
        supported_interval_terms=frozenset({INTERVAL_TERM_PUSH, "custom"})
    )
    assert declared.supports_interval_term(INTERVAL_TERM_PUSH)
    assert declared.supports_interval_term("custom")
    assert not declared.supports_interval_term(INTERVAL_TERM_BODY_FORCE)

    unsupported = legacy.get_unsupported_interval_terms(
        [INTERVAL_TERM_PUSH, INTERVAL_TERM_BODY_TORQUE, "custom"]
    )
    assert unsupported == frozenset({INTERVAL_TERM_BODY_TORQUE, "custom"})


def test_base_dispatch_fails_closed_and_empty_plan_is_noop() -> None:
    backend = FakeBackend()
    backend.apply_interval_randomization(IntervalRandomizationPlan())
    plan = IntervalRandomizationPlan(ops=(IntervalTermOp("custom_spin", np.zeros(3)),))
    with pytest.raises(NotImplementedError) as excinfo:
        backend.apply_interval_randomization(plan)
    message = str(excinfo.value)
    assert "FakeBackend" in message
    assert "custom_spin" in message


class _CustomTermBackend(FakeBackend):
    """Fake backend that registers one custom interval term handler."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[IntervalTermOp] = []
        self._handlers = {"custom_spin": self.received.append}

    def _interval_term_handlers(self):
        return self._handlers

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities(
            supported_interval_terms=frozenset({"custom_spin"})
        )


def test_backend_custom_term_handler_dispatch() -> None:
    backend = _CustomTermBackend()
    assert backend.get_dr_capabilities().supports_interval_term("custom_spin")
    op = IntervalTermOp("custom_spin", np.zeros(3))
    backend.apply_interval_randomization(IntervalRandomizationPlan(ops=(op,)))
    assert backend.received == [op]
    # A builtin term without a handler still fails closed.
    with pytest.raises(NotImplementedError, match="push"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(push_perturbation_limit=[1.0, 1.0, 1.0])
        )


MUJOCO_MODEL = """<mujoco model='unisim-test'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
  <actuator><motor joint='slide' ctrlrange='-1 1'/></actuator>
</mujoco>"""

MUJOCO_FREE_MODEL = """<mujoco model='unisim-free'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='free' type='free'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
</mujoco>"""


def _mujoco_backend(tmp_path: Path, model: str, num_envs: int = 2):
    pytest.importorskip("mujoco")
    from unisim import MuJoCoBackend
    from unisim.scene import SceneCfg

    model_path = tmp_path / "model.xml"
    model_path.write_text(model)
    return MuJoCoBackend(
        SceneCfg(model_file=str(model_path)), num_envs=num_envs, sim_dt=0.01, base_name="base"
    )


def test_mujoco_interval_term_capabilities(tmp_path: Path) -> None:
    backend = _mujoco_backend(tmp_path, MUJOCO_FREE_MODEL)
    caps = backend.get_dr_capabilities()
    assert caps.supported_interval_terms == frozenset(BUILTIN_TERMS)
    for term in BUILTIN_TERMS:
        assert caps.supports_interval_term(term)


def test_mujoco_force_and_torque_legacy_ops_parity(tmp_path: Path) -> None:
    backend = _mujoco_backend(tmp_path, MUJOCO_MODEL)
    force = np.arange(6, dtype=np.float64).reshape(2, 1, 3)
    torque = np.full((2, 1, 3), 0.5)

    backend.apply_interval_randomization(
        IntervalRandomizationPlan(body_ids=BODY_IDS, body_force=force, body_torque=torque)
    )
    legacy = backend._pending_xfrc_applied.copy()
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(
                IntervalTermOp(INTERVAL_TERM_BODY_FORCE, force, body_ids=BODY_IDS),
                IntervalTermOp(INTERVAL_TERM_BODY_TORQUE, torque, body_ids=BODY_IDS),
            )
        )
    )
    np.testing.assert_array_equal(legacy, backend._pending_xfrc_applied)

    # A torque-only plan matches the legacy zero-force single-call form.
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(body_ids=BODY_IDS, body_torque=torque)
    )
    legacy_torque_only = backend._pending_xfrc_applied.copy()
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(IntervalTermOp(INTERVAL_TERM_BODY_TORQUE, torque, body_ids=BODY_IDS),)
        )
    )
    np.testing.assert_array_equal(legacy_torque_only, backend._pending_xfrc_applied)


def test_mujoco_push_legacy_ops_parity(tmp_path: Path) -> None:
    backend = _mujoco_backend(tmp_path, MUJOCO_MODEL)
    limit = np.asarray([1.0, 2.0, 3.0])
    np.random.seed(0)
    backend.apply_interval_randomization(IntervalRandomizationPlan(push_perturbation_limit=limit))
    legacy = backend._pending_xfrc_applied.copy()
    np.random.seed(0)
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(ops=(IntervalTermOp(INTERVAL_TERM_PUSH, limit),))
    )
    np.testing.assert_array_equal(legacy, backend._pending_xfrc_applied)


def test_mujoco_velocity_delta_legacy_ops_parity(tmp_path: Path) -> None:
    legacy_backend = _mujoco_backend(tmp_path, MUJOCO_FREE_MODEL)
    legacy_backend.materialize()
    lin = np.zeros((2, 1, 3))
    lin[:, 0, 0] = 0.5
    ang = np.zeros((2, 1, 3))
    ang[:, 0, 2] = 0.25
    legacy_backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            body_ids=BODY_IDS,
            body_linear_velocity_delta=lin,
            body_angular_velocity_delta=ang,
        )
    )
    legacy_state = legacy_backend.get_state(("qpos", "qvel"))

    ops_backend = _mujoco_backend(tmp_path, MUJOCO_FREE_MODEL)
    ops_backend.materialize()
    ops_backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(
                IntervalTermOp(
                    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA, lin, body_ids=BODY_IDS
                ),
                IntervalTermOp(
                    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA, ang, body_ids=BODY_IDS
                ),
            )
        )
    )
    ops_state = ops_backend.get_state(("qpos", "qvel"))
    np.testing.assert_allclose(legacy_state["qpos"], ops_state["qpos"])
    np.testing.assert_allclose(legacy_state["qvel"], ops_state["qvel"])


def test_mujoco_unknown_term_fails_closed(tmp_path: Path) -> None:
    backend = _mujoco_backend(tmp_path, MUJOCO_MODEL)
    plan = IntervalRandomizationPlan(ops=(IntervalTermOp("custom_spin", np.zeros(3)),))
    with pytest.raises(NotImplementedError) as excinfo:
        backend.apply_interval_randomization(plan)
    message = str(excinfo.value)
    assert "MuJoCoBackend" in message
    assert "custom_spin" in message
