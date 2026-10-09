"""Static audit for accidental host detours in IsaacGym CUDA IPC hot paths."""

from __future__ import annotations

import ast
from pathlib import Path

_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "unisim"
    / "backend"
    / "isaacgym"
    / "tensor.py"
)


def _method(class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    return item
    raise AssertionError(f"missing method {class_name}.{method_name}")


def _attribute_names(node: ast.AST) -> set[str]:
    return {
        item.attr
        for item in ast.walk(node)
        if isinstance(item, ast.Attribute)
    }


def _has_numpy_reference(node: ast.AST) -> bool:
    return any(
        isinstance(item, ast.Name) and item.id in {"np", "numpy"}
        for item in ast.walk(node)
    )


def _request_calls(node: ast.AST) -> int:
    return sum(
        1
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == "_request"
    )


def _attribute_count(node: ast.AST, name: str) -> int:
    return sum(
        1
        for item in ast.walk(node)
        if isinstance(item, ast.Attribute) and item.attr == name
    )


def _request_payload_keys(node: ast.AST) -> set[str]:
    keys: set[str] = set()
    for item in ast.walk(node):
        if not (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "_request"
        ):
            continue
        for argument in item.args:
            if isinstance(argument, ast.Dict):
                keys.update(
                    key.value
                    for key in argument.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                )
    return keys


def test_isaacgym_hot_paths_have_no_hidden_host_detours() -> None:
    methods = {
        "worker.step": _method("IsaacGymCudaIpcWorkerRuntime", "step"),
        "worker.reset": _method("IsaacGymCudaIpcWorkerRuntime", "set_state"),
        "host.control": _method("IsaacGymCudaIpcPlan", "write_control"),
        "host.command-step": _method("IsaacGymCudaIpcPlan", "step"),
        "host.step": _method("IsaacGymCudaIpcPlan", "step_tensor"),
        "host.reset": _method("IsaacGymCudaIpcPlan", "set_state_tensor"),
        "host.state": _method("IsaacGymCudaIpcPlan", "get_state_views"),
        "host.sensor-view": _method("IsaacGymCudaIpcPlan", "get_sensor_view"),
        "worker.body-projection": _method(
            "IsaacGymCudaIpcWorkerRuntime", "_publish_body_state"
        ),
        "worker.sensor-projection": _method(
            "IsaacGymCudaIpcWorkerRuntime", "_publish_scalar_sensors"
        ),
        "worker.selected-reset-fk": _method(
            "IsaacGymCudaIpcWorkerRuntime", "_publish_selected_body_fk"
        ),
    }
    for label, node in methods.items():
        attrs = _attribute_names(node)
        assert not attrs & {"cpu", "numpy", "item", "synchronize", "from_numpy"}, label
        assert not _has_numpy_reference(node), label

    host_reset = methods["host.reset"]
    host_reset_attrs = _attribute_names(host_reset)
    assert not host_reset_attrs & {"cpu", "numpy", "item", "synchronize", "from_numpy"}
    assert not _has_numpy_reference(host_reset)

    assert _attribute_count(methods["worker.reset"], "tolist") == 0
    assert _attribute_count(methods["host.reset"], "tolist") == 1
    assert _attribute_count(methods["worker.reset"], "sort") == 2


def test_isaacgym_tensor_commands_are_metadata_only_and_bounded() -> None:
    step = _method("IsaacGymCudaIpcPlan", "step")
    reset = _method("IsaacGymCudaIpcPlan", "set_state_tensor")
    sensor_view = _method("IsaacGymCudaIpcPlan", "get_sensor_view")
    assert _request_calls(step) == 1
    assert _request_calls(reset) == 1
    assert _request_calls(sensor_view) == 0

    step_payload = _request_payload_keys(step)
    reset_payload = _request_payload_keys(reset)
    assert step_payload == {"nsteps"}
    assert reset_payload == {"count", "sequence"}
