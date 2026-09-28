"""SDK-free coverage for the Genesis M9 profiler's sensor-attribution stages."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


def _load_profiler() -> Any:
    path = Path(__file__).resolve().parents[3] / "scripts/benchmarks/m9_tensor_runtime_profile.py"
    spec = importlib.util.spec_from_file_location("m9_tensor_runtime_profile_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sensor_stage_profiles_publication_repeated_reads_and_stacking() -> None:
    profiler = _load_profiler()
    requests: list[str] = []
    refreshes: list[bool | None]

    class FakeGenesisBackend:
        backend_type = "genesis"

        def get_sensor_view(self, name: str) -> Any:
            requests.append(name)
            width = 4 if "quat" in name else 3
            return torch.empty((2, width), dtype=torch.float32, device="cpu")

        def _tensor_refresh_state(self, *, force: bool = False) -> None:
            refreshes.append(force)

    device = torch.device("cpu")
    backend = FakeGenesisBackend()
    refreshes = []
    sensor_names = (
        "track_pos_w_pelvis",
        "track_quat_w_pelvis",
        "track_linvel_w_pelvis",
        "track_angvel_w_pelvis",
        "pelvis_local_linvel",
        "torso_gyro",
    )
    operations = profiler._stage_operations(
        backend,
        "sensors",
        ctrl=torch.empty((2, 1), device=device),
        rows=torch.empty((1,), dtype=torch.int64, device=device),
        reset_qpos=torch.empty((1, 2), device=device),
        reset_qvel=torch.empty((1, 1), device=device),
        sensor_names=sensor_names,
    )

    assert set(operations) == {
        "state_and_body_publication",
        "repeated_sensor_requests",
        "repeated_sensor_reads_and_stack",
    }
    requests.clear()
    operations["state_and_body_publication"]()
    assert requests == []
    assert refreshes == [True]

    requests.clear()
    operations["repeated_sensor_requests"]()
    assert requests == list(sensor_names)

    requests.clear()
    operations["repeated_sensor_reads_and_stack"]()
    assert sorted(requests) == sorted(sensor_names)


def test_scalar_sensor_stage_does_not_stack_an_empty_body_list() -> None:
    profiler = _load_profiler()
    requests: list[str] = []

    class FakeGenesisBackend:
        backend_type = "genesis"

        def get_sensor_view(self, name: str) -> Any:
            requests.append(name)
            return torch.empty((2, 3), dtype=torch.float32, device="cpu")

        def _tensor_refresh_state(self, *, force: bool = False) -> None:
            return None

    operations = profiler._stage_operations(
        FakeGenesisBackend(),
        "sensors",
        ctrl=torch.empty((2, 1)),
        rows=torch.empty((1,), dtype=torch.int64),
        reset_qpos=torch.empty((1, 2)),
        reset_qvel=torch.empty((1, 1)),
        sensor_names=("pelvis_local_linvel", "torso_gyro"),
    )
    requests.clear()
    operations["repeated_sensor_reads_and_stack"]()
    assert requests == ["pelvis_local_linvel", "torso_gyro"]


def test_profiler_cli_accepts_custom_model_and_step_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = _load_profiler()
    assert "model_file" in profiler._make_backend.__annotations__
    assert "sim_dt" in profiler._make_backend.__annotations__
    assert "integrator" in profiler._make_backend.__annotations__
    assert "base_name" in profiler._make_backend.__annotations__
    captured: list[Any] = []
    monkeypatch.setattr(profiler, "_run", lambda args: captured.append(args) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "m9_tensor_runtime_profile.py",
            "--backend",
            "genesis",
            "--model-file",
            "/tmp/g1.xml",
            "--sim-dt",
            "0.006666666666666667",
            "--genesis-integrator",
            "implicitfast",
            "--genesis-base-name",
            "pelvis",
            "--nsteps",
            "3",
        ],
    )
    assert profiler.main() == 0
    assert captured[0].model_file == Path("/tmp/g1.xml")
    assert captured[0].sim_dt == 0.006666666666666667
    assert captured[0].genesis_integrator == "implicitfast"
    assert captured[0].genesis_base_name == "pelvis"
    assert captured[0].nsteps == 3
