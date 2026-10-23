"""Contract coverage for the scoped tensor-manager backend surface."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

import unisim.dr
from unisim.backend.base import (
    PublicStateWidths,
    SelectedResetPublication,
    SensorDescriptor,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
    TrackedBodyStateViews,
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


def test_tracked_body_views_require_sensor_views_and_selected_reset() -> None:
    kwargs = {
        "execution": TensorExecution.DEVICE_RESIDENT,
        "state_views": True,
        "state_fields": frozenset(("qpos", "qvel")),
        "stepping": True,
        "stream_event_ownership": "test",
        "torch_devices": ("cuda",),
        "process_topology": TensorProcessTopology.IN_PROCESS,
        "data_plane": TensorDataPlane.DIRECT,
        "tracked_body_views": True,
    }
    with pytest.raises(ValueError, match="tracked-body views require sensor views"):
        TensorLifecycleCapabilities(**kwargs)


def _device_reset_randomization_kwargs() -> dict:
    return {
        "execution": TensorExecution.DEVICE_RESIDENT,
        "state_fields": frozenset(("qpos", "qvel")),
        "selected_reset": True,
        "reset_randomization": True,
        "device_reset_randomization": True,
        "stream_event_ownership": "test",
        "torch_devices": ("cuda",),
        "process_topology": TensorProcessTopology.IN_PROCESS,
        "data_plane": TensorDataPlane.DIRECT,
    }


def test_device_reset_randomization_valid_on_device_resident_direct_matrix() -> None:
    capabilities = TensorLifecycleCapabilities(**_device_reset_randomization_kwargs())
    assert capabilities.device_reset_randomization


def test_device_reset_randomization_valid_on_in_process_host_bridge() -> None:
    capabilities = TensorLifecycleCapabilities(
        **{
            **_device_reset_randomization_kwargs(),
            "execution": TensorExecution.HOST_BRIDGE,
            "data_plane": TensorDataPlane.HOST_BRIDGE,
        }
    )
    assert capabilities.device_reset_randomization


def test_device_reset_randomization_unsupported_lifecycle_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported tensor lifecycle must remain fail closed"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.UNSUPPORTED,
            device_reset_randomization=True,
        )


def test_generalized_force_injection_requires_stepping() -> None:
    with pytest.raises(ValueError, match="generalized-force injection requires tensor stepping"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            generalized_force_injection=True,
            stream_event_ownership="test",
            torch_devices=("cuda",),
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.DIRECT,
        )


def test_generalized_force_injection_unsupported_lifecycle_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported tensor lifecycle must remain fail closed"):
        TensorLifecycleCapabilities(
            execution=TensorExecution.UNSUPPORTED,
            generalized_force_injection=True,
        )


def test_base_set_generalized_force_tensor_is_fail_closed() -> None:
    from unisim.backend.base import SimBackend

    class Backend:
        backend_type = "fake"

        def get_tensor_execution(self) -> TensorExecution:
            return TensorExecution.UNSUPPORTED

    with pytest.raises(
        NotImplementedError, match="fake does not support generalized-force injection"
    ):
        SimBackend.set_generalized_force_tensor(Backend())  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "override",
    [{"reset_randomization": False}],
)
def test_device_reset_randomization_requires_reset_randomization_and_direct_device_matrix(
    override: dict,
) -> None:
    kwargs = {**_device_reset_randomization_kwargs(), **override}
    with pytest.raises(ValueError, match="device reset randomization requires"):
        TensorLifecycleCapabilities(**kwargs)


def test_device_reset_randomization_accepts_external_cuda_ipc_matrix() -> None:
    capabilities = TensorLifecycleCapabilities(
        **{
            **_device_reset_randomization_kwargs(),
            "process_topology": TensorProcessTopology.EXTERNAL_WORKER,
            "data_plane": TensorDataPlane.CUDA_IPC,
        }
    )
    assert capabilities.device_reset_randomization


def test_tensor_reset_randomization_payload_terms_and_public_export() -> None:
    import unisim.dr as dr
    from unisim.dr.types import TensorResetRandomizationPayload

    assert dr.TensorResetRandomizationPayload is TensorResetRandomizationPayload

    empty = TensorResetRandomizationPayload()
    assert empty.is_empty()
    assert not empty.requested_terms()

    payload = TensorResetRandomizationPayload(
        body_mass=object(), kd=object(), geom_friction=object()
    )
    assert not payload.is_empty()
    assert payload.requested_terms() == frozenset({"body_mass", "kd", "geom_friction"})

    # Delta-merge terms are first-class fields of the single payload family.
    delta = TensorResetRandomizationPayload(base_mass_delta=object(), base_com_offset=object())
    assert delta.requested_terms() == frozenset({"base_mass_delta", "base_com_offset"})


def test_reset_term_specs_are_the_single_field_table_source() -> None:
    from unisim.backend.packed_reset import (
        _PACKED_RESET_CANONICAL_NDIMS,
        PACKED_RESET_MODEL_FIELDS,
    )
    from unisim.dr.types import (
        _RESET_TERM_NAMES,
        RESET_TERM_SPECS,
        TensorResetRandomizationPayload,
    )

    assert len(RESET_TERM_SPECS) == 16
    names = {spec.name for spec in RESET_TERM_SPECS}
    assert names == set(TensorResetRandomizationPayload.__dataclass_fields__)
    assert names == _RESET_TERM_NAMES == PACKED_RESET_MODEL_FIELDS
    assert _PACKED_RESET_CANONICAL_NDIMS == {spec.name: spec.tail_ndim for spec in RESET_TERM_SPECS}
    assert {spec.name for spec in RESET_TERM_SPECS if spec.is_delta} == frozenset(
        {"base_mass_delta", "base_com_offset"}
    )
    assert not hasattr(unisim.dr, "ResetRandomizationPayload")
    assert not hasattr(unisim.dr.types, "ResetRandomizationPayload")


def test_validate_tensor_reset_payload_lazy_torch() -> None:
    torch = pytest.importorskip("torch")
    from unisim.dr.types import TensorResetRandomizationPayload, validate_tensor_reset_payload

    payload = TensorResetRandomizationPayload(kp=torch.ones(2, 3))
    validate_tensor_reset_payload(payload, device="cpu", allowed_terms={"kp"})
    with pytest.raises(ValueError, match="kp"):
        validate_tensor_reset_payload(payload, allowed_terms={"kd"})
    with pytest.raises(TypeError, match="torch.Tensor"):
        validate_tensor_reset_payload(TensorResetRandomizationPayload(kp=object()))
    with pytest.raises(ValueError, match="cuda"):
        validate_tensor_reset_payload(payload, device="cuda")
    with pytest.raises(TypeError, match="TensorResetRandomizationPayload"):
        validate_tensor_reset_payload(object())  # pyright: ignore[reportArgumentType]
    # An empty payload validates without importing Torch.
    validate_tensor_reset_payload(TensorResetRandomizationPayload())


def test_dr_package_has_no_star_import_leakage() -> None:
    assert not hasattr(unisim.dr, "np")
    assert not hasattr(unisim.dr, "Any")
    assert not hasattr(unisim.dr, "require_op_host_payload")
    assert not hasattr(unisim.dr, "ResetRandomizationPayload")


def test_dr_capabilities_split_host_and_device_planes() -> None:
    from unisim.capabilities import (
        get_unsupported_interval_terms,
        get_unsupported_reset_terms,
    )
    from unisim.dr.types import DomainRandomizationCapabilities

    caps = DomainRandomizationCapabilities(
        supported_reset_terms=frozenset({"kp", "kd"}),
        supported_device_reset_terms=frozenset({"kp"}),
        supported_interval_terms=frozenset({"push"}),
    )
    assert caps.supported_device_interval_terms == frozenset()
    assert caps.supports_reset_term("kd")
    assert caps.supports_reset_term("kp", plane="device")
    assert not caps.supports_reset_term("kd", plane="device")
    assert caps.get_unsupported_reset_terms({"kp", "kd"}, plane="device") == frozenset({"kd"})
    assert caps.get_unsupported_interval_terms(["push"], plane="device") == frozenset({"push"})
    assert get_unsupported_reset_terms(caps, ["kp", "kd"]) == frozenset()
    assert get_unsupported_interval_terms(caps, ["push"]) == frozenset()
    with pytest.raises(ValueError, match="plane"):
        caps.get_unsupported_reset_terms({"kp"}, plane="mps")  # pyright: ignore[reportArgumentType]


def test_tracked_body_state_views_validate_body_names() -> None:
    values = {"pos_w": None, "quat_w": None, "lin_vel_w": None, "ang_vel_w": None}

    assert TrackedBodyStateViews(("b", "a"), **values).body_names == ("b", "a")
    for body_names in ("b", (), ("",), ("a", "a")):
        with pytest.raises((TypeError, ValueError)):
            TrackedBodyStateViews(body_names, **values)


def test_base_tracked_body_views_is_fail_closed() -> None:
    from unisim.backend.base import SimBackend

    class Backend:
        backend_type = "fake"

        def get_tensor_execution(self) -> TensorExecution:
            return TensorExecution.UNSUPPORTED

    with pytest.raises(NotImplementedError, match="fake does not support tracked-body views"):
        SimBackend.get_tracked_body_views(Backend())  # pyright: ignore[reportArgumentType]


def test_base_public_width_method_is_fail_closed() -> None:
    from unisim.backend.base import SimBackend

    with pytest.raises(NotImplementedError, match="does not expose public tensor state widths"):
        SimBackend.get_public_state_widths(object())


def test_backend_base_exports_contract_names_stably() -> None:
    import unisim.backend.base as base

    assert base.SelectedResetPublication.AUTHORITATIVE_VIEWS.value == "authoritative_views"
    assert set(("nq", "nv")).issubset({field.name for field in fields(base.PublicStateWidths)})


def test_default_sensor_namespace_is_fail_closed_without_private_probe() -> None:
    class Backend:
        backend_type = "fake-no-namespace"

        def get_sensor_view(self, name: str) -> object:
            raise KeyError(name)

    from unisim.backend.base import SimBackend

    with pytest.raises(NotImplementedError, match="does not expose its named sensor namespace"):
        SimBackend.get_sensor_names(Backend())  # pyright: ignore[reportArgumentType]


def test_sim_backend_training_numpy_surface_is_removed() -> None:
    from unisim.backend.base import SimBackend

    legacy_names = (
        "step",
        "reset",
        "get_state",
        "set_state",
        "get_base_pos",
        "get_base_quat",
        "get_base_lin_vel",
        "get_base_ang_vel",
        "get_dof_pos",
        "get_dof_vel",
        "get_body_pos_w",
        "get_body_quat_w",
        "get_body_pose_w",
        "get_body_lin_vel_w",
        "get_body_vel_w",
        "get_body_ang_vel_w",
        "get_body_state_w",
        "copy_body_state_w",
        "get_body_pose_w_rows",
        "get_body_lin_vel_w_rows",
        "get_body_ang_vel_w_rows",
        "get_body_pos_b",
        "get_body_quat_b",
        "get_body_lin_vel_b",
        "get_body_ang_vel_b",
        "get_sensor_data",
        "get_sensor_data_rows",
        "get_sensor_data_batch",
    )
    assert not any(hasattr(SimBackend, name) for name in legacy_names)


def test_reset_plan_dead_code_is_removed() -> None:
    import unisim.dr.types as dr_types

    assert not hasattr(dr_types, "ResetPlan")
    assert not hasattr(unisim.dr, "ResetPlan")


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
