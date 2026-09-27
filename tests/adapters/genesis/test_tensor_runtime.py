"""Partial device-lifecycle coverage for the Genesis owner adapter."""

# ruff: noqa: E402

from __future__ import annotations

from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from unisim.backend.base import TensorExecution
from unisim.backend.genesis.backend import GenesisBackend
from unisim.scene import SceneCfg


class _NoHostTensor(torch.Tensor):
    """Reject accidental bulk ``.cpu()`` calls in the fake Genesis lane."""

    def cpu(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("Genesis tensor hot path unexpectedly copied to host")


class _FakeGenesisBackend(Enum):
    CUDA = "cuda"
    AMDGPU = "amdgpu"
    CPU = "cpu"


class _FakeEntity:
    def __init__(self, qpos: torch.Tensor, qvel: torch.Tensor, device: torch.device) -> None:
        self.qpos = qpos.as_subclass(_NoHostTensor)
        self.qvel = qvel.as_subclass(_NoHostTensor)
        self.device = device
        self.controls: list[torch.Tensor] = []
        self.reset_masks: list[torch.Tensor] = []

    def get_qpos(self) -> torch.Tensor:
        return self.qpos.clone()

    def get_dofs_velocity(self) -> torch.Tensor:
        return self.qvel.clone()

    def control_dofs_position(
        self, position: torch.Tensor, dofs_idx_local: list[int] | None = None
    ) -> None:
        assert position.is_cuda and position.dtype == torch.float32
        self.controls.append(position)

    def set_qpos(
        self,
        qpos: torch.Tensor,
        envs_idx: torch.Tensor | None = None,
        *,
        zero_velocity: bool = False,
    ) -> None:
        del zero_velocity
        assert envs_idx is not None and envs_idx.dtype == torch.bool
        self.reset_masks.append(envs_idx.clone())
        self.qpos = torch.where(envs_idx[:, None], qpos, self.qpos).as_subclass(_NoHostTensor)

    def set_dofs_velocity(
        self, velocity: torch.Tensor, envs_idx: torch.Tensor | None = None
    ) -> None:
        assert envs_idx is not None and envs_idx.dtype == torch.bool
        self.qvel = torch.where(envs_idx[:, None], velocity, self.qvel).as_subclass(_NoHostTensor)


class _FakeScene:
    def __init__(self, entity: _FakeEntity) -> None:
        self.entity = entity
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        self.entity.qpos.add_(float(self.steps))


def _backend(device: torch.device) -> tuple[GenesisBackend, _FakeEntity, _FakeScene]:
    backend = object.__new__(GenesisBackend)
    entity = _FakeEntity(
        torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4),
        torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3),
        device,
    )
    scene = _FakeScene(entity)
    backend._torch = torch
    backend._gs = SimpleNamespace(
        backend=_FakeGenesisBackend.CUDA,
        cuda=_FakeGenesisBackend.CUDA,
        amdgpu=_FakeGenesisBackend.AMDGPU,
        cpu=_FakeGenesisBackend.CPU,
        use_zerocopy=True,
    )
    backend._device = device
    backend._scene = scene
    backend._entity = entity
    backend._portable_mode = False
    backend._metadata = cast(
        Any,
        SimpleNamespace(nq=4, nv=3, actuator_names=("a",)),
    )
    backend._num_envs = 2
    backend._sim_dt = 0.5
    backend._actuated_dofs = [0]
    backend._materialized = True
    backend._closed = False
    backend._entity_faulted = False
    backend._host_cache_stale = False
    backend._tensor_time = None
    backend._tensor_state_mirrors = {}
    backend._time_cache = np.asarray((1.0, 2.0), dtype=np.float32)
    backend._contact_sensor_rows_valid = np.zeros((2,), dtype=np.bool_)
    backend._pre_step_control_fn = None
    backend._viewer = None
    backend._portable_pending_body_forces = None
    backend._portable_pending_body_torques = None
    return backend, entity, scene


def _cuda_backend() -> tuple[GenesisBackend, _FakeEntity, _FakeScene]:
    if not torch.cuda.is_available():
        pytest.skip("Genesis fake tensor-lifecycle checks require CUDA")
    return _backend(torch.device("cuda", torch.cuda.current_device()))


def test_genesis_partial_cuda_capability_profile() -> None:
    backend, _, _ = _cuda_backend()
    capabilities = backend.get_tensor_capabilities()

    assert backend.tensor_execution() is TensorExecution.DEVICE_RESIDENT
    assert capabilities.state_views
    assert set(capabilities.state_fields) == {"qpos", "qvel"}
    assert capabilities.stepping
    assert capabilities.selected_reset
    assert not capabilities.sensor_views
    assert not capabilities.reset_randomization
    assert capabilities.torch_devices == ("cuda",)

    cpu_backend, _, _ = _backend(torch.device("cpu"))
    assert cpu_backend.tensor_execution() is TensorExecution.UNSUPPORTED
    assert not cpu_backend.get_tensor_capabilities().state_views

    assert backend._genesis_backend_is_exact_cuda()
    backend._gs.backend = _FakeGenesisBackend.AMDGPU
    assert backend.tensor_execution() is TensorExecution.UNSUPPORTED
    backend._gs.backend = _FakeGenesisBackend.CPU
    assert backend.tensor_execution() is TensorExecution.UNSUPPORTED
    backend._gs.backend = None
    assert backend.tensor_execution() is TensorExecution.UNSUPPORTED
    backend._gs.backend = _FakeGenesisBackend.CUDA

    backend._gs = SimpleNamespace(use_zerocopy=False)
    assert backend.tensor_execution() is TensorExecution.UNSUPPORTED

    exact_cuda, _, _ = _cuda_backend()
    exact_cuda._portable_mode = True
    assert exact_cuda.tensor_execution() is TensorExecution.UNSUPPORTED
    exact_cuda._portable_mode = False
    assert exact_cuda.tensor_execution() is TensorExecution.DEVICE_RESIDENT


def test_genesis_state_views_are_stable_cuda_mirrors_without_host_copy() -> None:
    backend, entity, _ = _cuda_backend()
    views = backend.get_state_views()
    qpos_pointer = views["qpos"].data_ptr()

    assert views["qpos"].is_cuda
    assert views["qvel"].is_cuda
    assert torch.equal(views["qpos"], entity.qpos)

    entity.qpos.add_(3.0)
    refreshed = backend.get_state_views()
    assert refreshed["qpos"].data_ptr() == qpos_pointer
    assert torch.equal(refreshed["qpos"], entity.qpos)


def test_genesis_tensor_step_and_selected_reset_stay_on_cuda() -> None:
    backend, entity, scene = _cuda_backend()
    original_qvel = entity.qvel.clone()
    ctrl = torch.full((2, 1), 7.0, dtype=torch.float32, device=backend._device)

    result = backend.step_tensor(ctrl)
    assert result is not None
    assert result["timing"]["tensor_host_cache_refresh_ms"] == 0.0
    assert backend._host_cache_stale
    assert entity.controls[0].is_cuda
    assert torch.equal(entity.controls[0], ctrl)
    assert scene.steps == 1
    assert torch.allclose(backend.get_state_views()["qvel"], original_qvel)

    rows = torch.asarray((1, 0), dtype=torch.int64, device=backend._device)
    qpos = torch.full((2, 4), 11.0, dtype=torch.float32, device=backend._device)
    qvel = torch.full((2, 3), -2.0, dtype=torch.float32, device=backend._device)
    backend.set_state_tensor(rows, qpos, qvel)
    assert entity.reset_masks and bool(entity.reset_masks[-1].all())
    assert torch.equal(entity.qpos, qpos.as_subclass(_NoHostTensor))
    assert torch.equal(entity.qvel, qvel.as_subclass(_NoHostTensor))
    assert torch.all(backend._tensor_time == 0)


def test_genesis_tensor_hot_path_does_not_refresh_host_cache() -> None:
    backend, _, _ = _cuda_backend()

    def reject_refresh() -> None:
        raise AssertionError("tensor hot path must not refresh the legacy host cache")

    backend._refresh_host_cache = reject_refresh  # type: ignore[method-assign]
    backend.step_tensor(torch.zeros((2, 1), dtype=torch.float32, device=backend._device))


def test_genesis_tensor_hot_path_scalar_sync_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, entity, _ = _cuda_backend()
    scalar_reads: list[str] = []
    original_item = torch.Tensor.item
    original_tolist = torch.Tensor.tolist

    def counting_item(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        scalar_reads.append("item")
        return original_item(self, *args, **kwargs)

    def counting_tolist(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        scalar_reads.append("tolist")
        return original_tolist(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counting_item)
    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)

    ctrl = torch.full((2, 1), 7.0, dtype=torch.float32, device=backend._device)
    ctrl[0, 0] = torch.nan
    backend.step_tensor(ctrl)
    assert scalar_reads == []
    assert torch.isnan(entity.controls[0][0, 0])

    backend.set_state_tensor(
        torch.empty((0,), dtype=torch.int64, device=backend._device),
        torch.empty((0, 4), dtype=torch.float32, device=backend._device),
        torch.empty((0, 3), dtype=torch.float32, device=backend._device),
    )
    assert scalar_reads == []

    invalid_rows = torch.asarray((-1, 1), dtype=torch.int64, device=backend._device)
    with pytest.raises(ValueError, match="range"):
        backend.set_state_tensor(
            invalid_rows,
            torch.empty((2, 4), dtype=torch.float32, device=backend._device),
            torch.empty((2, 3), dtype=torch.float32, device=backend._device),
        )
    assert scalar_reads == ["tolist"]

    scalar_reads.clear()
    duplicate_rows = torch.asarray((1, 1), dtype=torch.int64, device=backend._device)
    with pytest.raises(ValueError, match="unique"):
        backend.set_state_tensor(
            duplicate_rows,
            torch.empty((2, 4), dtype=torch.float32, device=backend._device),
            torch.empty((2, 3), dtype=torch.float32, device=backend._device),
        )
    assert scalar_reads == ["tolist"]

    scalar_reads.clear()
    rows = torch.asarray((1, 0), dtype=torch.int64, device=backend._device)
    qpos = torch.full((2, 4), 11.0, dtype=torch.float32, device=backend._device)
    qpos[0, 0] = torch.nan
    qvel = torch.full((2, 3), -2.0, dtype=torch.float32, device=backend._device)
    backend.set_state_tensor(rows, qpos, qvel)
    assert scalar_reads == ["tolist"]


def test_genesis_legacy_reads_use_one_explicit_mixed_boundary() -> None:
    backend, entity, _ = _cuda_backend()
    qpos_pinned = torch.empty((2, 4), dtype=torch.float32, pin_memory=True)
    qvel_pinned = torch.empty((2, 3), dtype=torch.float32, pin_memory=True)
    backend._qpos_cache = (qpos_pinned, qpos_pinned.numpy())
    backend._qvel_cache = (qvel_pinned, qvel_pinned.numpy())
    refreshes = 0

    def refresh() -> None:
        nonlocal refreshes
        refreshes += 1
        qpos_pinned.copy_(entity.get_qpos())
        qvel_pinned.copy_(entity.get_dofs_velocity())

    backend._refresh_host_cache = refresh  # type: ignore[method-assign]
    backend.step_tensor(torch.zeros((2, 1), dtype=torch.float32, device=backend._device))
    snapshot = backend.get_physics_state()

    assert refreshes == 1
    assert not backend._host_cache_stale
    np.testing.assert_allclose(snapshot[:, 0], (1.5, 2.5), atol=1e-6)
    np.testing.assert_allclose(snapshot[:, 1:5], qpos_pinned.numpy(), atol=1e-6)
    np.testing.assert_allclose(snapshot[:, 5:], qvel_pinned.numpy(), atol=1e-6)


def test_genesis_tensor_operands_and_sensor_views_fail_closed() -> None:
    backend, _, _ = _cuda_backend()
    device = backend._device

    with pytest.raises(KeyError, match="unknown"):
        backend.get_state_views(("qpos", "ctrl"))
    with pytest.raises(NotImplementedError, match="sensor views"):
        backend.get_sensor_view("gyro")
    with pytest.raises(TypeError, match="torch.Tensor"):
        backend.step_tensor(np.zeros((2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        backend.step_tensor(torch.zeros((3, 1), dtype=torch.float32, device=device))
    with pytest.raises(ValueError, match="unique"):
        rows = torch.asarray((1, 1), dtype=torch.int64, device=device)
        backend.set_state_tensor(
            rows,
            torch.zeros((2, 4), dtype=torch.float32, device=device),
            torch.zeros((2, 3), dtype=torch.float32, device=device),
        )


def test_real_genesis_cuda_public_state_and_partial_lifecycle(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    genesis = pytest.importorskip("genesis")
    if not torch.cuda.is_available():
        pytest.skip("real Genesis device-residency check requires CUDA zero-copy mode")

    model_file = tmp_path / "genesis-cuda.xml"
    model_file.write_text(
        """
        <mujoco model="unisim-genesis-cuda">
          <option timestep="0.005" gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="ground" type="plane" size="2 2 0.1"/>
            <body name="base" pos="0 0 0.5">
              <joint name="root" type="free"/>
              <geom name="base_geom" type="sphere" size="0.08" mass="1"/>
              <body name="arm" pos="0 0 0.12">
                <joint name="hinge" axis="0 1 0"/>
                <geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 0.2"
                      size="0.03" mass="0.2"/>
              </body>
            </body>
          </worldbody>
          <actuator><position name="hinge_motor" joint="hinge" kp="10" kv="1"/></actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    backend = GenesisBackend(SceneCfg(model_file=str(model_file)), num_envs=2, sim_dt=0.005)
    if backend.tensor_execution() is not TensorExecution.DEVICE_RESIDENT:
        pytest.skip("real Genesis session did not expose the CUDA zero-copy lane")
    assert bool(getattr(genesis, "use_zerocopy", False))
    backend.materialize()

    native_qpos = backend.model.get_qpos()
    native_qvel = backend.model.get_dofs_velocity()
    assert native_qpos.is_cuda and native_qvel.is_cuda

    views = backend.get_state_views()
    assert views["qpos"].is_cuda and views["qvel"].is_cuda
    backend.step_tensor(
        torch.zeros((2, backend.num_actuators), dtype=torch.float32, device=backend._device)
    )
    assert views["qpos"].data_ptr() == backend.get_state_views()["qpos"].data_ptr()

    qpos = views["qpos"].clone()
    qvel = views["qvel"].clone()
    qvel[:, -1] = 0.25
    backend.set_state_tensor(
        torch.asarray((1,), dtype=torch.int64, device=backend._device),
        qpos[1:2].contiguous(),
        qvel[1:2].contiguous(),
    )
    refreshed = backend.get_state_views()
    assert torch.equal(refreshed["qvel"][1], qvel[1])
