"""CUDA-resident Newton tensor lifecycle contract tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unisim.backend.newton.backend import NewtonBackend
from unisim.backend.newton.dependencies import (
    NewtonDependencyError,
    load_newton_dependencies,
)
from unisim.scene import SceneCfg

from .test_contract import _MODEL

_SENSOR_MODEL = (
    _MODEL.replace(
        '<geom name="base_geom" type="sphere" size="0.08" mass="1"/>',
        '<geom name="base_geom" type="sphere" size="0.08" mass="1"/>'
        '<site name="imu_in_pelvis" pos="0.01 0.02 0.03"/>',
    )
    .replace(
        '<geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" mass="0.2"/>',
        '<geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" mass="0.2"/>'
        '<site name="imu_in_torso" pos="-0.01 0.02 0.04"/>',
    )
    .replace(
        "<actuator>",
        "<sensor>"
        '<velocimeter site="imu_in_pelvis" name="pelvis_local_linvel"/>'
        '<gyro site="imu_in_torso" name="torso_gyro"/>'
        '<gyro site="imu_in_pelvis" name="pelvis_gyro"/>'
        "</sensor><actuator>",
    )
)


def _portable_fake(entity_count: int) -> NewtonBackend:
    backend = cast(Any, NewtonBackend.__new__(NewtonBackend))
    backend._portable_mode = True
    backend._device = "cuda:0"
    backend._entity_layout = None
    backend._entity_runtimes = {f"entity_{index}": object() for index in range(entity_count)}
    return backend


def _cuda_backend(tmp_path: Path, xml: str = _MODEL) -> NewtonBackend:
    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton tensor lifecycle requires a CUDA Warp device")
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_file = tmp_path / "newton.xml"
    model_file.write_text(xml, encoding="utf-8")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=2,
        sim_dt=0.005,
        device=str(device),
        capacity_check_steps=1,
    )
    backend.materialize()
    return backend


def test_newton_declares_partial_device_resident_tensor_lifecycle(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        assert backend.tensor_execution().value == "device_resident"
        capabilities = backend.get_tensor_capabilities()
        assert capabilities.process_topology.value == "in_process"
        assert capabilities.data_plane.value == "direct"
        assert capabilities.torch_devices == ("cuda",)
        assert capabilities.state_fields == frozenset({"qpos", "qvel"})
        assert capabilities.stepping
        assert capabilities.selected_reset
        assert capabilities.sensor_views

        with pytest.raises(NotImplementedError, match="unsupported: 'unused'"):
            backend.get_sensor_view("unused", device=torch.device(backend._device))
    finally:
        backend.close()


def test_newton_portable_multi_entity_tensor_lifecycle_fails_closed() -> None:
    torch = pytest.importorskip("torch")
    backend = _portable_fake(2)

    assert backend.tensor_execution().value == "unsupported"
    capabilities = backend.get_tensor_capabilities()
    assert capabilities.execution.value == "unsupported"
    assert not capabilities.state_views
    assert not capabilities.stepping
    assert not capabilities.selected_reset

    with pytest.raises(NotImplementedError, match="one physical articulation.*2"):
        backend.get_state_views(device=torch.device("cuda:0"))
    with pytest.raises(NotImplementedError, match="one physical articulation.*2"):
        backend.step_tensor(torch.zeros((1, 1), device="cpu"))
    with pytest.raises(NotImplementedError, match="one physical articulation.*2"):
        backend.set_state_tensor(
            torch.zeros(1, dtype=torch.int64),
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
        )


def test_newton_portable_single_entity_declares_narrow_tensor_lifecycle() -> None:
    backend = _portable_fake(1)

    capabilities = backend.get_tensor_capabilities()
    assert capabilities.execution.value == "device_resident"
    assert capabilities.state_views
    assert capabilities.stepping
    assert not capabilities.selected_reset


def test_newton_tensor_sensor_routing_is_device_only_without_sdk() -> None:
    torch = pytest.importorskip("torch")
    backend = cast(Any, NewtonBackend.__new__(NewtonBackend))
    backend._portable_mode = False
    backend._device = "cpu"
    backend._entity_layout = None
    backend._entity_runtimes = {}
    backend._body_names = ("base", "arm")
    backend._body_ids = {"base": 0, "arm": 1}
    backend._metadata = SimpleNamespace(
        sensor_plans=(
            SimpleNamespace(
                name="pelvis_local_linvel",
                kind="velocimeter",
                dim=3,
                body_id=1,
                site_pos=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                site_quat=np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32),
            ),
            SimpleNamespace(
                name="torso_gyro",
                kind="gyro",
                dim=3,
                body_id=2,
                site_pos=np.zeros(3, dtype=np.float32),
                site_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
            SimpleNamespace(
                name="pelvis_gyro",
                kind="gyro",
                dim=3,
                body_id=1,
                site_pos=np.zeros(3, dtype=np.float32),
                site_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
        )
    )
    backend._tensor_body_state_stale = False
    backend._tensor_body_pos = torch.zeros((2, 2, 3), dtype=torch.float32)
    backend._tensor_body_pos[:, 1] = torch.tensor((9.0, 8.0, 7.0))
    backend._tensor_body_quat = torch.zeros((2, 2, 4), dtype=torch.float32)
    backend._tensor_body_quat[..., 0] = 1.0
    backend._tensor_body_lin_vel = torch.zeros((2, 2, 3), dtype=torch.float32)
    backend._tensor_body_lin_vel[:, 0] = torch.tensor((1.0, 2.0, 3.0))
    backend._tensor_body_lin_vel[:, 1] = torch.tensor((4.0, 5.0, 6.0))
    backend._tensor_body_ang_vel = torch.zeros((2, 2, 3), dtype=torch.float32)
    backend._tensor_body_ang_vel[:, 0] = torch.tensor((0.0, 0.0, 1.0))
    backend._tensor_body_ang_vel[:, 1] = torch.tensor((0.1, 0.2, 0.3))
    backend._tensor_sensor_views = {}
    backend._tensor_sensor_constants = {}
    backend._require_tensor_lifecycle = lambda operation: None
    backend._tensor_device = lambda requested=None: torch.device("cpu")
    backend._tensor_ensure_state = lambda: None

    linvel = backend.get_sensor_view("pelvis_local_linvel")
    gyro = backend.get_sensor_view("torso_gyro")
    assert linvel.shape == (2, 3)
    assert gyro.shape == (2, 3)
    torch.testing.assert_close(
        linvel, torch.tensor(((3.0, -1.0, 3.0), (3.0, -1.0, 3.0)), dtype=torch.float32)
    )
    torch.testing.assert_close(
        gyro, torch.tensor(((0.1, 0.2, 0.3), (0.1, 0.2, 0.3)), dtype=torch.float32)
    )
    assert torch.equal(backend.get_sensor_view("track_pos_w_arm"), backend._tensor_body_pos[:, 1])
    assert torch.equal(
        backend.get_sensor_view("track_quat_w_base"), backend._tensor_body_quat[:, 0]
    )
    assert torch.equal(
        backend.get_sensor_view("track_linvel_w_base"), backend._tensor_body_lin_vel[:, 0]
    )
    assert torch.equal(
        backend.get_sensor_view("track_angvel_w_arm"), backend._tensor_body_ang_vel[:, 1]
    )
    with pytest.raises(NotImplementedError, match="unsupported: 'pelvis_gyro'"):
        backend.get_sensor_view("pelvis_gyro")
    with pytest.raises(KeyError, match="unknown Newton tensor tracked body"):
        backend.get_sensor_view("track_pos_w_missing")


def test_newton_tensor_selected_reset_preserves_legacy_updated_rows(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    reference = _cuda_backend(tmp_path)
    try:
        views = backend.get_state_views()
        ctrl = torch.zeros(
            (backend.num_envs, backend.num_actuators),
            dtype=torch.float32,
            device=views["qpos"].device,
        )
        backend.step_tensor(ctrl)
        reference.step(ctrl.cpu().numpy())

        legacy_views = reference.get_state_views()
        legacy_qpos = legacy_views["qpos"].clone()
        legacy_qvel = legacy_views["qvel"].clone()
        legacy_qpos[0, 2] += 0.2
        legacy_qvel[0, 0] = 0.3
        reference.set_state(
            np.array([0]),
            legacy_qpos[:1].cpu().numpy(),
            legacy_qvel[:1].cpu().numpy(),
        )
        backend.set_state(
            np.array([0]),
            legacy_qpos[:1].cpu().numpy(),
            legacy_qvel[:1].cpu().numpy(),
        )

        refreshed = backend.get_state_views()
        torch.testing.assert_close(refreshed["qpos"][0], legacy_qpos[0])
        reset_qpos = refreshed["qpos"].clone()
        reset_qvel = refreshed["qvel"].clone()
        reset_qpos[1, 2] += 0.1
        reset_qvel[1, 0] = 0.25
        backend.set_state_tensor(
            torch.tensor([1], device=reset_qpos.device),
            reset_qpos[1:2],
            reset_qvel[1:2],
        )
        reference.set_state(
            np.array([1]),
            reset_qpos[1:2].cpu().numpy(),
            reset_qvel[1:2].cpu().numpy(),
        )

        result = backend.get_state_views()
        expected = reference.get_state_views()
        torch.testing.assert_close(result["qpos"], expected["qpos"])
        torch.testing.assert_close(result["qvel"], expected["qvel"])
        torch.testing.assert_close(result["qpos"][0], legacy_qpos[0])
        torch.testing.assert_close(result["qvel"][0], legacy_qvel[0])
    finally:
        backend.close()
        reference.close()


def test_newton_tensor_hot_path_has_bounded_scalar_synchronization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        views = backend.get_state_views()
        ctrl = torch.zeros(
            (backend.num_envs, backend.num_actuators),
            dtype=torch.float32,
            device=views["qpos"].device,
        )

        scalar_reads: list[str] = []
        original_item = torch.Tensor.item
        original_tolist = torch.Tensor.tolist

        def counted_item(self: Any) -> Any:
            scalar_reads.append("item")
            return original_item(self)

        def counted_tolist(self: Any) -> Any:
            scalar_reads.append("tolist")
            return original_tolist(self)

        monkeypatch.setattr(torch.Tensor, "item", counted_item)
        monkeypatch.setattr(torch.Tensor, "tolist", counted_tolist)
        backend.step_tensor(ctrl)
        assert scalar_reads == []

        rows = torch.tensor([1], dtype=torch.int64, device=views["qpos"].device)
        qpos = views["qpos"][1:2].clone()
        qvel = views["qvel"][1:2].clone()
        backend.set_state_tensor(rows, qpos, qvel)
        assert scalar_reads == ["item"]
    finally:
        backend.close()


def test_newton_selected_reset_avoids_scalar_index_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        views = backend.get_state_views()
        rows = torch.tensor([1], dtype=torch.int64, device=views["qpos"].device)
        scalar_writes: list[object] = []
        original_setitem = torch.Tensor.__setitem__

        def counting_setitem(self: Any, key: Any, value: Any) -> None:
            if isinstance(value, (bool, int, float)):
                scalar_writes.append(value)
            return original_setitem(self, key, value)

        monkeypatch.setattr(torch.Tensor, "__setitem__", counting_setitem)
        backend.set_state_tensor(rows, views["qpos"][1:2], views["qvel"][1:2])

        assert scalar_writes == []
    finally:
        backend.close()


def test_newton_tensor_sensor_views_match_host_and_selected_reset(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path, _SENSOR_MODEL)
    host = _cuda_backend(tmp_path / "host", _SENSOR_MODEL)
    try:
        device = backend.get_state_views()["qpos"].device
        ctrl = torch.full(
            (backend.num_envs, backend.num_actuators),
            0.2,
            dtype=torch.float32,
            device=device,
        )
        backend.step_tensor(ctrl, nsteps=2)
        host.step(ctrl.detach().cpu().numpy(), nsteps=2)

        def assert_sensor_parity() -> None:
            for name in ("pelvis_local_linvel", "torso_gyro"):
                actual = backend.get_sensor_view(name, device=device)
                expected = host.get_sensor_data(name)
                np.testing.assert_allclose(actual.detach().cpu().numpy(), expected, atol=2e-5)
            body_ids = host.get_body_ids(("base", "arm"))
            expected = (
                host.get_body_pos_w(body_ids),
                host.get_body_quat_w(body_ids),
                host.get_body_lin_vel_w(body_ids),
                host.get_body_ang_vel_w(body_ids),
            )
            for body_name, body_id in zip(("base", "arm"), body_ids.tolist(), strict=True):
                actual = (
                    backend.get_sensor_view(f"track_pos_w_{body_name}", device=device),
                    backend.get_sensor_view(f"track_quat_w_{body_name}", device=device),
                    backend.get_sensor_view(f"track_linvel_w_{body_name}", device=device),
                    backend.get_sensor_view(f"track_angvel_w_{body_name}", device=device),
                )
                expected_body = tuple(value[:, body_id] for value in expected)
                for index, (tensor_value, host_value) in enumerate(
                    zip(actual, expected_body, strict=True)
                ):
                    np.testing.assert_allclose(
                        tensor_value.detach().cpu().numpy(), host_value, atol=2e-5
                    )
                    del index

        assert_sensor_parity()

        views = backend.get_state_views()
        rows = torch.tensor([1], dtype=torch.int64, device=device)
        reset_qpos = views["qpos"].clone()
        reset_qvel = views["qvel"].clone()
        reset_qpos[1, 2] += 0.15
        reset_qpos[1, 7] = 0.4
        reset_qvel[1, 3] = -0.2
        backend.set_state_tensor(rows, reset_qpos[1:2], reset_qvel[1:2])
        host.set_state(
            rows.detach().cpu().numpy(),
            reset_qpos[1:2].detach().cpu().numpy(),
            reset_qvel[1:2].detach().cpu().numpy(),
        )
        assert_sensor_parity()
    finally:
        backend.close()
        host.close()


def test_newton_tensor_step_and_reset_avoid_host_cache_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    host = _cuda_backend(tmp_path)
    try:
        monkeypatch.setattr(
            backend,
            "_refresh_host_cache",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("hidden host cache refresh")),
        )
        views = backend.get_state_views(device=torch.device(backend._device))
        ctrl = torch.full(
            (2, backend.num_actuators),
            0.2,
            dtype=torch.float32,
            device=views["qpos"].device,
        )
        result = backend.step_tensor(ctrl, nsteps=2)
        assert result is not None
        assert result["timing"]["tensor_host_cache_refresh_ms"] == 0.0

        host.step(np.full((2, host.num_actuators), 0.2, dtype=np.float32), nsteps=2)
        host_state = host.get_physics_state()
        host_qpos = host_state[:, 1 : 1 + host._metadata.nq]
        host_qvel = host_state[:, 1 + host._metadata.nq :]
        torch.testing.assert_close(views["qpos"], torch.tensor(host_qpos, device=ctrl.device))
        torch.testing.assert_close(views["qvel"], torch.tensor(host_qvel, device=ctrl.device))

        rows = torch.tensor([1], dtype=torch.int64, device=ctrl.device)
        reset_qpos = views["qpos"].clone()
        reset_qvel = views["qvel"].clone()
        reset_qpos[0, 2] += 0.1
        reset_qvel[0, 0] = 0.25
        backend.set_state_tensor(rows, reset_qpos[:1], reset_qvel[:1])
        assert views["qpos"][1, 2].item() == pytest.approx(reset_qpos[0, 2].item(), abs=1e-6)
        assert views["qvel"][1, 0].item() == pytest.approx(0.25, abs=1e-6)
        host.set_state(
            rows.cpu().numpy(), reset_qpos[:1].cpu().numpy(), reset_qvel[:1].cpu().numpy()
        )
        host_state = host.get_physics_state()
        torch.testing.assert_close(
            views["qpos"],
            torch.tensor(host_state[:, 1 : 1 + host._metadata.nq], device=ctrl.device),
        )
        torch.testing.assert_close(
            views["qvel"],
            torch.tensor(host_state[:, 1 + host._metadata.nq :], device=ctrl.device),
        )
    finally:
        backend.close()
        host.close()


def test_newton_tensor_inputs_fail_closed(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        device = torch.device(backend._device)
        views = backend.get_state_views(device=device)
        with pytest.raises(ValueError, match="shape"):
            backend.step_tensor(torch.zeros((1, backend.num_actuators), device=device))
        with pytest.raises(ValueError, match="device"):
            backend.get_state_views(device="cpu")
        with pytest.raises(ValueError, match="unique"):
            backend.set_state_tensor(
                torch.tensor([0, 0], device=device),
                views["qpos"][:2],
                views["qvel"][:2],
            )
    finally:
        backend.close()
