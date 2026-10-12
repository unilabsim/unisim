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


def test_plan_iter_ops_preserves_explicit_order() -> None:
    lin = np.zeros((2, 1, 3))
    force = np.ones((2, 1, 3))
    push = IntervalTermOp(INTERVAL_TERM_PUSH, np.asarray([1.0, 2.0, 3.0]))
    linear = IntervalTermOp(INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA, lin, body_ids=BODY_IDS)
    wrench = IntervalTermOp(INTERVAL_TERM_BODY_FORCE, force, body_ids=BODY_IDS)
    plan = IntervalRandomizationPlan(ops=(push, linear, wrench))
    assert plan.iter_ops() == (push, linear, wrench)


def test_plan_is_empty_semantics() -> None:
    assert IntervalRandomizationPlan().is_empty()
    assert not IntervalRandomizationPlan(ops=(IntervalTermOp("custom", np.zeros(1)),)).is_empty()


def test_capabilities_use_supported_interval_terms_only() -> None:
    declared = DomainRandomizationCapabilities(
        supported_interval_terms=frozenset({INTERVAL_TERM_PUSH, "custom"})
    )
    assert declared.supports_interval_term(INTERVAL_TERM_PUSH)
    assert declared.supports_interval_term("custom")
    assert not declared.supports_interval_term(INTERVAL_TERM_BODY_FORCE)

    unsupported = declared.get_unsupported_interval_terms(
        [INTERVAL_TERM_PUSH, INTERVAL_TERM_BODY_TORQUE, "custom"]
    )
    assert unsupported == frozenset({INTERVAL_TERM_BODY_TORQUE})


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
        return DomainRandomizationCapabilities(supported_interval_terms=frozenset({"custom_spin"}))


def test_backend_custom_term_handler_dispatch() -> None:
    backend = _CustomTermBackend()
    assert backend.get_dr_capabilities().supports_interval_term("custom_spin")
    op = IntervalTermOp("custom_spin", np.zeros(3))
    backend.apply_interval_randomization(IntervalRandomizationPlan(ops=(op,)))
    assert backend.received == [op]
    # A builtin term without a handler still fails closed.
    with pytest.raises(NotImplementedError, match="push"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(ops=(IntervalTermOp(INTERVAL_TERM_PUSH, np.zeros(3)),))
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


def test_mujoco_unknown_term_fails_closed(tmp_path: Path) -> None:
    backend = _mujoco_backend(tmp_path, MUJOCO_MODEL)
    plan = IntervalRandomizationPlan(ops=(IntervalTermOp("custom_spin", np.zeros(3)),))
    with pytest.raises(NotImplementedError) as excinfo:
        backend.apply_interval_randomization(plan)
    message = str(excinfo.value)
    assert "MuJoCoBackend" in message
    assert "custom_spin" in message
