"""Static audit for accidental host detours in IsaacSim CUDA IPC hot paths."""

from __future__ import annotations

import ast
from pathlib import Path

_BACKEND_SOURCE = (
    Path(__file__).resolve().parents[3] / "src" / "unisim" / "backend" / "isaacsim" / "backend.py"
)
_ARENA_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "unisim"
    / "backend"
    / "isaacsim"
    / "tensor_ipc.py"
)
_WORKER_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "unisim"
    / "backend"
    / "isaacsim"
    / "scene_worker.py"
)


def _method(source: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"missing method {class_name}.{method_name}")


def _attribute_names(node: ast.AST) -> set[str]:
    return {item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)}


def _has_numpy_reference(node: ast.AST) -> bool:
    return any(isinstance(item, ast.Name) and item.id in {"np", "numpy"} for item in ast.walk(node))


def _attribute_count(node: ast.AST, name: str) -> int:
    return sum(
        1 for item in ast.walk(node) if isinstance(item, ast.Attribute) and item.attr == name
    )


def _request_calls(node: ast.AST) -> int:
    return sum(
        1
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == "_request"
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


def test_isaacsim_host_and_worker_hot_paths_have_no_hidden_host_detours() -> None:
    methods = {
        "host.state": _method(_BACKEND_SOURCE, "IsaacSimBackend", "get_state_views"),
        "host.sensor-view": _method(_BACKEND_SOURCE, "IsaacSimBackend", "get_sensor_view"),
        "host.control-write": _method(_ARENA_SOURCE, "HostCudaIpcArena", "write_control"),
        "host.reset-write": _method(_ARENA_SOURCE, "HostCudaIpcArena", "write_reset"),
        "host.step": _method(_BACKEND_SOURCE, "IsaacSimBackend", "step_tensor"),
        "worker.control": _method(
            _WORKER_SOURCE, "SceneWorkerContext", "_set_control_tensor_targets"
        ),
        "worker.publish": _method(_WORKER_SOURCE, "SceneWorkerContext", "_publish_cuda_state"),
        "worker.reset-publish": _method(
            _WORKER_SOURCE, "SceneWorkerContext", "_publish_cuda_reset_state"
        ),
        "worker.body-projection": _method(
            _WORKER_SOURCE, "SceneWorkerContext", "_publish_cuda_body_state"
        ),
        "worker.sensor-projection": _method(
            _WORKER_SOURCE, "SceneWorkerContext", "_publish_cuda_scalar_sensors"
        ),
        "worker.step": _method(_WORKER_SOURCE, "SceneWorkerContext", "step_cuda_ipc"),
        "worker.reset": _method(_WORKER_SOURCE, "SceneWorkerContext", "reset_cuda_ipc"),
    }
    for label, node in methods.items():
        attrs = _attribute_names(node)
        assert not attrs & {"cpu", "numpy", "item", "tolist", "synchronize", "from_numpy"}, label
        assert not _has_numpy_reference(node), label

    backend_reset = _method(_BACKEND_SOURCE, "IsaacSimBackend", "set_state_tensor")
    backend_reset_attrs = _attribute_names(backend_reset)
    assert not backend_reset_attrs & {"cpu", "numpy", "item", "synchronize", "from_numpy"}
    assert not _has_numpy_reference(backend_reset)

    # The backend reset performs exactly one bounded scalar reduction download for
    # row range/uniqueness closure; the worker never detours reset data.
    # The backend performs one bounded scalar sync (covered below); worker reset
    # remains device-resident and has no host conversion.
    worker_reset = methods["worker.reset"]
    assert _attribute_count(worker_reset, "tolist") == 0


def test_isaacsim_tensor_commands_are_metadata_only_and_bounded() -> None:
    step = _method(_BACKEND_SOURCE, "IsaacSimBackend", "step_tensor")
    reset = _method(_BACKEND_SOURCE, "IsaacSimBackend", "set_state_tensor")
    sensor_view = _method(_BACKEND_SOURCE, "IsaacSimBackend", "get_sensor_view")
    assert _request_calls(step) == 1
    assert _request_calls(reset) == 1
    assert _request_calls(sensor_view) == 0
    assert _attribute_count(step, "tolist") == 0
    # One `.tolist()` is three scalar checks packed into one device reduction.
    assert _attribute_count(reset, "tolist") == 1

    step_payload = _request_payload_keys(step)
    reset_payload = _request_payload_keys(reset)
    assert step_payload == {"nsteps"}
    assert reset_payload == {"count", "sequence"}
