"""SDK-free checks for the SuperDex M9 host-bridge profiler model."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


def _load_profiler() -> Any:
    path = Path(__file__).resolve().parents[3] / "scripts/benchmarks/m9_tensor_runtime_profile.py"
    spec = importlib.util.spec_from_file_location("m9_tensor_runtime_profile_superdex", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakePlan:
    def __init__(self) -> None:
        self.transfer_stats = {
            "d2h_count": 0,
            "h2d_count": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
            "synchronization_count": 0,
        }
        self._control_ready = True

    def close(self) -> None:
        return None

    def step(self) -> Any:
        self._control_ready = False
        return None

    def read_state_sensors(self) -> Any:
        return None

    def apply_reset(self, *_args: Any, **_kwargs: Any) -> Any:
        return None

    def read_selected_state_sensors(self) -> Any:
        return None


def test_superdex_phase_model_exposes_all_packed_boundaries() -> None:
    profiler = _load_profiler()
    plan = _FakePlan()
    backend = type("FakeSuperDexBackend", (), {"compile_host_bridge_io": lambda _, __: plan})()
    operations = profiler._superdex_operations(
        backend,
        plan,
        ctrl=torch.empty((2, 1)),
        rows=torch.tensor([0], dtype=torch.int64),
        reset_qpos=torch.empty((1, 2)),
        reset_qvel=torch.empty((1, 1)),
        sensor_names=(),
    )

    assert tuple(operations) == profiler._SUPERDEX_PHASES
    operations["compile_host_bridge_io"]()
    assert operations["step"]() is None
    assert plan._control_ready is True


def test_semantic_transfer_delta_reads_live_owner_counters() -> None:
    profiler = _load_profiler()
    stats = {
        "d2h_count": 3,
        "h2d_count": 0,
        "d2h_bytes": 12,
        "h2d_bytes": 0,
        "synchronization_count": 3,
    }

    def operation() -> None:
        stats["d2h_count"] += 1
        stats["d2h_bytes"] += 4
        stats["synchronization_count"] += 1

    _, delta = profiler._record_semantic_transfer(operation, lambda: dict(stats))
    assert delta == {
        "d2h_count": 1,
        "h2d_count": 0,
        "d2h_bytes": 4,
        "h2d_bytes": 0,
        "synchronization_count": 1,
    }


def test_cli_selects_backend_specific_default_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    profiler = _load_profiler()
    captured: list[Any] = []
    monkeypatch.setattr(profiler, "_run", lambda args: captured.append(args) or {})

    for backend, expected in (
        ("genesis", ("step", "reset")),
        ("superdex", profiler._SUPERDEX_PHASES),
    ):
        captured.clear()
        monkeypatch.setattr(sys, "argv", ["profile.py", "--backend", backend])
        assert profiler.main() == 0
        assert tuple(captured[0].phases) == expected


def test_superdex_rejects_device_resident_stage_profiling() -> None:
    profiler = _load_profiler()
    args = argparse.Namespace(phases=["step"], profile_stages=True)
    with pytest.raises(ValueError, match="not supported by --backend superdex"):
        profiler._run_superdex(args, Path("/unused.xml"))
