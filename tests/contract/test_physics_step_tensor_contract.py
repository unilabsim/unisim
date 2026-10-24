"""SDK-free coverage of the optional physics-step tensor contract."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

import unisim
from unisim import contract
from unisim.backend.base import (
    PhysicsStepTensorDataView,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
)
from unisim.fake import FakeBackend


def test_physics_step_tensor_view_is_one_public_contract() -> None:
    assert unisim.PhysicsStepTensorDataView is PhysicsStepTensorDataView
    assert contract.PhysicsStepTensorDataView is PhysicsStepTensorDataView
    assert "PhysicsStepTensorDataView" in unisim.__all__
    assert "PhysicsStepTensorDataView" in contract.__all__
    with pytest.raises(TypeError):
        PhysicsStepTensorDataView()


def test_physics_step_binding_signature_is_keyword_only() -> None:
    parameters = inspect.signature(SimBackend.bind_physics_step_tensor_data).parameters
    assert tuple(parameters) == ("self", "joint_names", "body_names", "history_length", "device")
    for name in ("joint_names", "body_names", "history_length", "device"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["joint_names"].default == ()
    assert parameters["body_names"].default == ()
    assert parameters["history_length"].default == 3
    assert parameters["device"].default is None


def test_physics_step_tensor_data_fails_closed_by_default() -> None:
    backend = FakeBackend(num_envs=2, num_actuators=1)
    assert backend.get_tensor_capabilities().physics_step_data is False
    with pytest.raises(NotImplementedError, match="physics-step tensor data"):
        backend.bind_physics_step_tensor_data(joint_names=("ankle",))


def test_unsupported_lifecycle_cannot_declare_physics_step_data() -> None:
    with pytest.raises(ValueError):
        TensorLifecycleCapabilities(
            execution=TensorExecution.UNSUPPORTED,
            physics_step_data=True,
        )


@pytest.mark.parametrize(
    ("execution", "data_plane", "stepping"),
    [
        (TensorExecution.DEVICE_RESIDENT, TensorDataPlane.DIRECT, False),
        (TensorExecution.HOST_BRIDGE, TensorDataPlane.HOST_BRIDGE, True),
    ],
    ids=["no-tensor-stepping", "host-bridge"],
)
def test_physics_step_data_requires_device_resident_stepping(
    execution: TensorExecution, data_plane: TensorDataPlane, stepping: bool
) -> None:
    with pytest.raises(ValueError, match="physics-step tensor data"):
        TensorLifecycleCapabilities(
            execution=execution,
            data_plane=data_plane,
            stepping=stepping,
            physics_step_data=True,
            stream_event_ownership="backend completes sampling before return",
            torch_devices=("cuda",),
        )


def test_device_resident_stepping_can_declare_physics_step_data() -> None:
    capabilities = TensorLifecycleCapabilities(
        execution=TensorExecution.DEVICE_RESIDENT,
        data_plane=TensorDataPlane.DIRECT,
        stepping=True,
        physics_step_data=True,
        stream_event_ownership="backend completes sampling before return",
        torch_devices=("cuda",),
    )
    assert capabilities.physics_step_data is True


def test_public_physics_step_contract_keeps_torch_and_engines_lazy() -> None:
    code = """
import importlib.abc
import sys

blocked = {'torch', 'warp', 'mujoco', 'mujoco_warp', 'unilab'}

class NoRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in blocked, fullname
        return None

sys.meta_path.insert(0, NoRuntimeImports())
from unisim import PhysicsStepTensorDataView, FakeBackend
from unisim.contract import PhysicsStepTensorDataView as ContractView
assert PhysicsStepTensorDataView is ContractView
assert FakeBackend().get_tensor_capabilities().physics_step_data is False
assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or result.stdout
