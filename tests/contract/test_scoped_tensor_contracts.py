"""Contract coverage for the scoped tensor-manager backend surface."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

from unisim.backend.base import (
    PublicStateWidths,
    SelectedResetPublication,
    SensorDescriptor,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)


def test_public_state_widths_are_positive_integers() -> None:
    assert PublicStateWidths(nq=1, nv=2) == PublicStateWidths(nq=1, nv=2)
    for kwargs in ({"nq": 0, "nv": 1}, {"nq": 1, "nv": 0}):
        with pytest.raises(ValueError):
            PublicStateWidths(**kwargs)  # pyright: ignore[reportArgumentType]


def test_sensor_descriptor_requires_name_and_positive_width() -> None:
    assert SensorDescriptor(name="imu", width=3).width == 3
    with pytest.raises(ValueError):
        SensorDescriptor(name="", width=3)
    with pytest.raises(TypeError):
        SensorDescriptor(name="imu", width=True)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError):
        SensorDescriptor(name="imu", width=0)
    for kwargs in ({"nq": True, "nv": 1}, {"nq": 1, "nv": False}):
        with pytest.raises(TypeError):
            PublicStateWidths(**kwargs)  # pyright: ignore[reportArgumentType]


def test_selected_reset_publication_requires_selected_reset() -> None:
    kwargs = {
        "execution": TensorExecution.DEVICE_RESIDENT,
        "state_views": True,
        "state_fields": frozenset(("qpos", "qvel")),
        "sensor_views": True,
        "stepping": True,
        "selected_reset": False,
        "stream_event_ownership": "test",
        "torch_devices": ("cuda",),
        "process_topology": TensorProcessTopology.IN_PROCESS,
        "data_plane": TensorDataPlane.DIRECT,
        "selected_reset_publication": SelectedResetPublication.AUTHORITATIVE_VIEWS,
    }
    with pytest.raises(ValueError, match="selected-reset publication requires selected reset"):
        TensorLifecycleCapabilities(**kwargs)


def test_post_construction_barrier_requires_supported_lifecycle() -> None:
    with pytest.raises(ValueError, match="unsupported tensor lifecycle"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.UNSUPPORTED,
            requires_post_construction_publication_barrier=True,
        )


def test_base_public_width_method_is_fail_closed() -> None:
    from unisim.backend.base import SimBackend

    with pytest.raises(NotImplementedError, match="does not expose public tensor state widths"):
        SimBackend.get_public_state_widths(object())


def test_backend_base_exports_contract_names_stably() -> None:
    import unisim.backend.base as base

    assert base.SelectedResetPublication.AUTHORITATIVE_VIEWS.value == "authoritative_views"
    assert set(("nq", "nv")).issubset({field.name for field in fields(base.PublicStateWidths)})


def test_default_get_sensor_names_rejects_incomplete_namespace_diagnostics() -> None:
    class Backend:
        backend_type = "fake-no-namespace"

        def get_sensor_data(self, name: str) -> object:
            raise ValueError(f"Sensor {name!r} not found; available: a, b")

    from unisim.backend.base import SimBackend

    with pytest.raises(ValueError, match="not found"):
        SimBackend.get_sensor_names(Backend())  # pyright: ignore[reportArgumentType]


def test_contract_source_does_not_import_optional_engines() -> None:
    source = Path("src/unisim/backend/base.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0]
            in {
                "mujoco",
                "mujoco_warp",
                "warp",
                "genesis",
            }
        ):
            raise AssertionError(f"backend base imports optional engine: {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {
                    "mujoco",
                    "mujoco_warp",
                    "warp",
                    "genesis",
                }
