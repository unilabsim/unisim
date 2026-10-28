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
    TensorIntervalPayload,
    interval_term_spec,
    validate_interval_op,
)
from unisim.dr._host import interval_payload_to_host
from unisim.fake import FakeBackend

BUILTIN_TERMS = (
    INTERVAL_TERM_PUSH,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_TORQUE,
)

BODY_IDS = np.asarray([1], dtype=np.int32)


class _FakeTensor:
    """Duck-typed Torch tensor stand-in: carries ``ndim`` without importing Torch."""

    def __init__(self, ndim: int) -> None:
        self.ndim = ndim


def test_interval_term_spec_registry_covers_builtin_terms() -> None:
    assert tuple(spec.name for spec in INTERVAL_TERM_SPECS) == BUILTIN_TERMS
    for name in BUILTIN_TERMS:
        assert interval_term_spec(name) is not None
    assert isinstance(interval_term_spec(INTERVAL_TERM_PUSH), IntervalTermSpec)
    with pytest.raises(KeyError):
        interval_term_spec("no_such_term")


def test_interval_term_op_validation_builtin_contract() -> None:
    payload = _FakeTensor(3)
    validate_interval_op(IntervalTermOp(INTERVAL_TERM_BODY_FORCE, payload, body_ids=BODY_IDS))
    with pytest.raises(ValueError, match="body_force"):
        IntervalTermOp(INTERVAL_TERM_BODY_FORCE, payload).validate()
    with pytest.raises(ValueError, match="push"):
        IntervalTermOp(INTERVAL_TERM_PUSH, _FakeTensor(1), body_ids=BODY_IDS).validate()
    with pytest.raises(ValueError, match="body_force"):
        IntervalTermOp(INTERVAL_TERM_BODY_FORCE, _FakeTensor(1), body_ids=BODY_IDS).validate()
    validate_interval_op(IntervalTermOp(INTERVAL_TERM_PUSH, _FakeTensor(2)))
    with pytest.raises(ValueError, match="push"):
        IntervalTermOp(INTERVAL_TERM_PUSH, _FakeTensor(1)).validate()


def test_interval_term_op_validation_rejects_numpy_payloads() -> None:
    # The op path is torch-first: NumPy payloads fail closed with TypeError
    # instead of being silently duck-typed through ``.ndim``.
    with pytest.raises(TypeError, match="torch-first"):
        validate_interval_op(IntervalTermOp(INTERVAL_TERM_PUSH, np.zeros((2, 3))))
    with pytest.raises(TypeError, match="torch-first"):
        validate_interval_op(
            IntervalTermOp(INTERVAL_TERM_BODY_FORCE, np.zeros((2, 1, 3)), body_ids=BODY_IDS)
        )
    # Non-tensor array-likes no longer fall back to np.asarray either.
    with pytest.raises(TypeError, match="torch-first"):
        validate_interval_op(IntervalTermOp(INTERVAL_TERM_PUSH, [[0.0, 0.0, 0.0]]))
    # Custom terms remain backend-owned and pass through untouched.
    validate_interval_op(IntervalTermOp("custom_spin", np.zeros(3)))


def test_tensor_interval_payload_validation_defers_engine_rules() -> None:
    class FakeTensor:
        pass

    tensor = FakeTensor()
    payload = TensorIntervalPayload(push_force=tensor)
    op = IntervalTermOp(INTERVAL_TERM_PUSH, payload)
    validate_interval_op(op)
    op.validate()
    assert payload.requested_terms() == frozenset({INTERVAL_TERM_PUSH})
    assert not payload.is_empty()

    with pytest.raises(ValueError, match="exactly the interval term"):
        validate_interval_op(IntervalTermOp(INTERVAL_TERM_BODY_FORCE, payload, body_ids=BODY_IDS))
    with pytest.raises(ValueError, match="does not accept body_ids"):
        validate_interval_op(IntervalTermOp(INTERVAL_TERM_PUSH, payload, body_ids=BODY_IDS))
    assert TensorIntervalPayload().is_empty()


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


def test_host_payload_conversion_is_adapter_private() -> None:
    # NumPy payloads are a torch-first contract violation, not a conversion
    # leaf: the adapter-private helper fails closed instead of passing through.
    host_op = IntervalTermOp(INTERVAL_TERM_BODY_FORCE, np.ones((2, 1, 3)), body_ids=BODY_IDS)
    with pytest.raises(TypeError, match="torch-first"):
        interval_payload_to_host(host_op.payload)
    device_op = IntervalTermOp(INTERVAL_TERM_PUSH, TensorIntervalPayload(push_force=object()))
    with pytest.raises(TypeError, match="torch-first"):
        interval_payload_to_host(device_op.payload)


def test_host_payload_conversion_of_torch_tensors() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    converted = interval_payload_to_host(tensor)
    assert isinstance(converted, np.ndarray)
    np.testing.assert_array_equal(
        converted, np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    )
    # Torch tensors validate by duck-typed ndim without importing Torch here.
    op = IntervalTermOp(INTERVAL_TERM_PUSH, tensor)
    validate_interval_op(op)
    with pytest.raises(ValueError, match="push"):
        validate_interval_op(IntervalTermOp(INTERVAL_TERM_PUSH, torch.zeros(3)))


def test_reset_payload_to_host_dict_converts_populated_fields() -> None:
    torch = pytest.importorskip("torch")
    from unisim.dr._host import reset_payload_to_host_dict
    from unisim.dr.types import TensorResetRandomizationPayload

    payload = TensorResetRandomizationPayload(kp=torch.ones(2, 3), base_mass_delta=torch.zeros(2))
    host = reset_payload_to_host_dict(payload)
    assert set(host) == {"kp", "base_mass_delta"}
    np.testing.assert_array_equal(host["kp"], np.ones((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(host["base_mass_delta"], np.zeros(2, dtype=np.float32))
    with pytest.raises(TypeError, match="TensorResetRandomizationPayload"):
        reset_payload_to_host_dict(object())  # pyright: ignore[reportArgumentType]


def test_plan_iter_ops_preserves_explicit_order() -> None:
    lin = np.zeros((2, 1, 3))
    force = np.ones((2, 1, 3))
    push = IntervalTermOp(INTERVAL_TERM_PUSH, np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
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
    assert not declared.supports_interval_term(INTERVAL_TERM_BODY_TORQUE)


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
            IntervalRandomizationPlan(ops=(IntervalTermOp(INTERVAL_TERM_PUSH, _FakeTensor(2)),))
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
