from __future__ import annotations

import pytest

from unisim import (
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
    TensorRuntimeDiagnostic,
    tensor_device_matches,
)
from unisim.support import FEATURES, get_adapter_capabilities


def test_default_tensor_capability_matrix_fails_closed() -> None:
    capabilities = TensorLifecycleCapabilities(execution=TensorExecution.UNSUPPORTED)

    assert capabilities.process_topology is TensorProcessTopology.IN_PROCESS
    assert capabilities.data_plane is TensorDataPlane.NONE
    assert capabilities.stream_event_ownership is None
    assert capabilities.torch_devices == ()


def test_invalid_tensor_capability_matrix_fails_closed() -> None:
    valid_fields = {
        "stream_event_ownership": "caller",
        "torch_devices": ("cuda",),
    }
    with pytest.raises(ValueError, match="fail closed"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.UNSUPPORTED,
            stepping=True,
        )
    with pytest.raises(ValueError, match="stream/event ownership"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            data_plane=TensorDataPlane.DIRECT,
            torch_devices=("cuda",),
        )
    with pytest.raises(ValueError, match="Torch devices"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="caller",
        )
    with pytest.raises(ValueError, match="Torch devices must be unique"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="caller",
            torch_devices=("cuda", "cuda"),
        )
    with pytest.raises(ValueError, match="process/data-plane combination"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            data_plane=TensorDataPlane.CUDA_IPC,
            **valid_fields,
        )
    with pytest.raises(ValueError, match="process/data-plane combination"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.HOST_BRIDGE,
            data_plane=TensorDataPlane.DIRECT,
            **valid_fields,
        )


@pytest.mark.parametrize(
    "fields",
    (
        {"requested": True, "enabled": False, "disable_reason": None},
        {"requested": True, "enabled": False, "disable_reason": "  "},
        {"requested": False, "enabled": True, "disable_reason": None},
        {"requested": True, "enabled": True, "disable_reason": "stale reason"},
    ),
    ids=["missing-reason", "blank-reason", "enabled-without-request", "enabled-with-reason"],
)
def test_tensor_runtime_diagnostic_state_is_fail_closed(fields: dict) -> None:
    with pytest.raises((TypeError, ValueError), match="runtime diagnostic"):
        TensorRuntimeDiagnostic(**fields)  # type: ignore[arg-type]


def test_tensor_runtime_diagnostic_represents_requested_disablement() -> None:
    diagnostic = TensorRuntimeDiagnostic(
        requested=False, enabled=False, disable_reason="not requested"
    )

    assert diagnostic.requested is False
    assert diagnostic.enabled is False
    assert diagnostic.disable_reason == "not requested"


@pytest.mark.parametrize(
    "metadata",
    [
        {"state_fields": frozenset({"qpos"})},
        {"stream_event_ownership": "caller"},
        {"torch_devices": ("cuda",)},
    ],
    ids=["state-fields", "stream-ownership", "torch-devices"],
)
def test_unsupported_tensor_metadata_fails_closed(metadata: dict) -> None:
    with pytest.raises(ValueError, match="unsupported tensor lifecycle"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.UNSUPPORTED,
            **metadata,  # type: ignore[arg-type]
        )


def test_tensor_feature_dependencies_fail_closed() -> None:
    base = {
        "execution": TensorExecution.DEVICE_RESIDENT,
        "process_topology": TensorProcessTopology.IN_PROCESS,
        "data_plane": TensorDataPlane.DIRECT,
        "stream_event_ownership": "caller",
        "torch_devices": ("cuda",),
    }
    invalid_matrices = (
        base | {"state_views": True},
        base | {"selected_reset": True, "state_fields": frozenset({"qpos", "ctrl"})},
        base | {"selected_reset": True, "state_fields": frozenset({"qvel", "ctrl"})},
        base
        | {
            "state_fields": frozenset({"qpos", "qvel"}),
            "reset_randomization": True,
        },
    )
    messages = (
        "state views require at least one declared state field",
        "selected reset requires qpos and qvel",
        "selected reset requires qpos and qvel",
        "reset randomization requires selected reset",
    )

    for matrix, message in zip(invalid_matrices, messages, strict=True):
        with pytest.raises(ValueError, match=message):
            TensorLifecycleCapabilities(**matrix)


@pytest.mark.parametrize(
    ("execution", "topology", "data_plane"),
    [
        (TensorExecution.DEVICE_RESIDENT, TensorProcessTopology.IN_PROCESS, TensorDataPlane.DIRECT),
        (
            TensorExecution.DEVICE_RESIDENT,
            TensorProcessTopology.EXTERNAL_WORKER,
            TensorDataPlane.CUDA_IPC,
        ),
        (
            TensorExecution.HOST_BRIDGE,
            TensorProcessTopology.EXTERNAL_WORKER,
            TensorDataPlane.HOST_SHARED_MEMORY,
        ),
        (TensorExecution.HOST_BRIDGE, TensorProcessTopology.IN_PROCESS, TensorDataPlane.DIRECT),
    ],
    ids=[
        "device-direct",
        "device-cuda-ipc",
        "host-external-shm",
        "host-wrong-data-plane",
    ],
)
def test_packed_host_bridge_requires_in_process_host_bridge(
    execution: TensorExecution,
    topology: TensorProcessTopology,
    data_plane: TensorDataPlane,
) -> None:
    with pytest.raises(
        ValueError,
        match="packed host bridge requires|invalid tensor process/data-plane combination",
    ):
        TensorLifecycleCapabilities(
            execution=execution,
            state_views=True,
            state_fields=frozenset({"qpos", "qvel"}),
            selected_reset=True,
            packed_host_bridge=True,
            process_topology=topology,
            data_plane=data_plane,
            stream_event_ownership="caller",
            torch_devices=("cpu", "cuda"),
        )


@pytest.mark.parametrize(
    ("execution", "topology", "data_plane", "packed"),
    [
        (
            TensorExecution.DEVICE_RESIDENT,
            TensorProcessTopology.IN_PROCESS,
            TensorDataPlane.DIRECT,
            False,
        ),
        (
            TensorExecution.DEVICE_RESIDENT,
            TensorProcessTopology.EXTERNAL_WORKER,
            TensorDataPlane.CUDA_IPC,
            False,
        ),
        (
            TensorExecution.HOST_BRIDGE,
            TensorProcessTopology.IN_PROCESS,
            TensorDataPlane.HOST_BRIDGE,
            True,
        ),
        (
            TensorExecution.HOST_BRIDGE,
            TensorProcessTopology.EXTERNAL_WORKER,
            TensorDataPlane.HOST_SHARED_MEMORY,
            False,
        ),
    ],
    ids=["device-direct", "device-cuda-ipc", "packed-host-bridge", "host-external-shm"],
)
def test_valid_tensor_capability_matrices_remain_supported(
    execution: TensorExecution,
    topology: TensorProcessTopology,
    data_plane: TensorDataPlane,
    packed: bool,
) -> None:
    capabilities = TensorLifecycleCapabilities(
        execution=execution,
        state_views=True,
        state_fields=frozenset({"qpos", "qvel"}),
        stepping=True,
        selected_reset=True,
        packed_host_bridge=packed,
        process_topology=topology,
        data_plane=data_plane,
        stream_event_ownership="caller",
        torch_devices=("cpu", "cuda"),
    )

    assert capabilities.execution is execution
    assert capabilities.process_topology is topology
    assert capabilities.data_plane is data_plane
    assert capabilities.packed_host_bridge is packed


@pytest.mark.parametrize(
    ("accepted", "requested", "current_device", "expected"),
    [
        (("cpu", "cuda"), "cuda", None, True),
        (("cpu", "cuda"), "cuda:2", None, True),
        (("cuda:0",), "cuda:0", None, True),
        (("cuda:1",), "cuda", 1, True),
        (("cuda:0",), "cuda", 1, False),
        (("cuda:0",), "cuda:1", None, False),
        (("cuda",), "cpu", None, False),
        (("cuda",), "cuda:not-an-index", None, False),
    ],
)
def test_tensor_device_labels_match_family_and_exact_cuda_semantics(
    accepted, requested, current_device, expected
):
    assert tensor_device_matches(accepted, requested, current_device=current_device) is expected


def test_tensor_capability_device_labels_fail_closed() -> None:
    valid_matrix = {
        "stream_event_ownership": "caller",
    }
    for label in ("cuda:", "cuda:-1", "cpu:0", "cuda:0:1", 0):
        with pytest.raises(ValueError, match="Torch device label"):
            TensorLifecycleCapabilities(
                execution=TensorExecution.DEVICE_RESIDENT,
                data_plane=TensorDataPlane.DIRECT,
                torch_devices=(label,),
                **valid_matrix,
            )


def test_static_tensor_matrix_covers_every_declared_feature() -> None:
    for name in (
        "mujoco",
        "mjwarp",
        "superdex",
        "motrix",
        "drake",
        "newton",
        "genesis",
        "isaacgym",
        "isaacsim",
    ):
        report = get_adapter_capabilities(name)
        assert {item.feature for item in report.declarations} == set(FEATURES)


def test_mjwarp_and_mujoco_tensor_classifications_are_explicit() -> None:
    mjwarp = get_adapter_capabilities("mjwarp")
    assert mjwarp.get("tensor.execution").support.value == "exact"
    assert "DEVICE_RESIDENT" in mjwarp.get("tensor.execution").reason
    assert mjwarp.get("tensor.data_plane").reason == "direct"

    mujoco = get_adapter_capabilities("mujoco")
    assert mujoco.get("tensor.execution").support.value == "exact"
    assert "HOST_BRIDGE" in mujoco.get("tensor.execution").reason
    assert mujoco.get("tensor.data_plane").reason == "host_bridge"

    superdex = get_adapter_capabilities("superdex")
    assert superdex.get("tensor.execution").support.value == "exact"
    assert "HOST_BRIDGE" in superdex.get("tensor.execution").reason
    assert superdex.get("tensor.data_plane").reason == "host_bridge"
    assert superdex.get("tensor.packed_host_bridge").support.value == "exact"
    assert superdex.get("tensor.reset_randomization").support.value == "unsupported"

    motrix = get_adapter_capabilities("motrix")
    assert motrix.get("tensor.execution").support.value == "exact"
    assert "HOST_BRIDGE" in motrix.get("tensor.execution").reason
    assert motrix.get("tensor.data_plane").reason == "host_bridge"
    assert motrix.get("tensor.packed_host_bridge").support.value == "exact"
    assert motrix.get("tensor.fixed_variants").support.value == "unsupported"

    drake = get_adapter_capabilities("drake")
    assert drake.get("tensor.execution").support.value == "exact"
    assert "HOST_BRIDGE" in drake.get("tensor.execution").reason
    assert drake.get("tensor.data_plane").reason == "host_bridge"
    assert drake.get("tensor.packed_host_bridge").support.value == "exact"
    assert drake.get("tensor.host_pre_step_control").support.value == "unsupported"

    newton = get_adapter_capabilities("newton")
    assert newton.get("tensor.execution").support.value == "exact"
    assert "DEVICE_RESIDENT" in newton.get("tensor.execution").reason
    assert newton.get("tensor.data_plane").reason == "direct"
    assert newton.get("tensor.selected_reset").support.value == "exact"
    assert newton.get("tensor.packed_host_bridge").support.value == "unsupported"

    genesis = get_adapter_capabilities("genesis")
    assert genesis.get("tensor.execution").support.value == "exact"
    assert "DEVICE_RESIDENT" in genesis.get("tensor.execution").reason
    assert genesis.get("tensor.data_plane").reason == "direct"
    assert genesis.get("tensor.state_fields").reason == "qpos and qvel"
    assert genesis.get("tensor.selected_reset").support.value == "exact"
    assert genesis.get("tensor.packed_host_bridge").support.value == "unsupported"

    isaacsim = get_adapter_capabilities("isaacsim")
    assert isaacsim.get("tensor.execution").support.value == "exact"
    assert "DEVICE_RESIDENT" in isaacsim.get("tensor.execution").reason
    assert isaacsim.get("tensor.data_plane").reason == "cuda_ipc"
    assert isaacsim.get("tensor.process_topology").reason == "external_worker"
    assert isaacsim.get("tensor.state_fields").reason == "qpos and qvel"
    assert isaacsim.get("tensor.selected_reset").support.value == "exact"
    assert isaacsim.get("tensor.reset_randomization").support.value == "unsupported"
    assert isaacsim.get("tensor.packed_host_bridge").support.value == "unsupported"


def test_isaacgym_tensor_profile_declares_reviewed_cuda_ipc_boundary() -> None:
    report = get_adapter_capabilities("isaacgym")
    assert report.get("tensor.execution").support.value == "exact"
    assert "DEVICE_RESIDENT" in report.get("tensor.execution").reason
    assert report.get("tensor.data_plane").reason == "cuda_ipc"
    assert report.get("tensor.process_topology").reason == "external_worker"
    assert report.get("tensor.state_fields").reason == "qpos and qvel"
    assert "after a tensor step" in report.get("tensor.sensor_views").reason
    assert report.get("tensor.selected_reset").support.value == "exact"
    assert report.get("tensor.reset_randomization").support.value == "unsupported"
    assert report.get("tensor.fixed_variants").support.value == "unsupported"
    assert report.get("tensor.host_pre_step_control").support.value == "unsupported"
    assert report.get("tensor.packed_host_bridge").support.value == "unsupported"
