"""SDK-free checks for the opt-in Isaac worker profiler."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from unisim.backend.isaacgym.worker import _load_worker_profiler as load_isaacgym_profiler
from unisim.backend.isaacsim.worker import _load_worker_profiler as load_isaacsim_profiler
from unisim.backend.subprocess_ipc.worker_profile import (
    PROFILE_START_COMMAND_ENV,
    PROFILE_STOP_COMMAND_ENV,
    PROFILE_TRACE_ENV,
    NullWorkerProfiler,
    WorkerProfiler,
    profiler_from_environment,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROTOCOL_PATH = _REPO_ROOT / "src" / "unisim" / "backend" / "subprocess_ipc" / "protocol.py"


class FakeProfile:
    def __init__(
        self, *, enter_error: Exception | None = None, export_error: Exception | None = None
    ) -> None:
        self.enter_error = enter_error
        self.export_error = export_error
        self.entered = 0
        self.exited = 0
        self.exported: list[str] = []

    def __enter__(self) -> "FakeProfile":
        self.entered += 1
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.exited += 1

    def export_chrome_trace(self, path: str) -> None:
        if self.export_error is not None:
            raise self.export_error
        self.exported.append(path)
        Path(path).write_text('{"traceEvents": []}', encoding="utf-8")


class FakeRecordFunction:
    def __init__(self, name: str) -> None:
        self.name = name
        self.enter_error: Exception | None = None
        self.exit_error: Exception | None = None
        self.entered = 0
        self.exited = 0
        self.exit_args: tuple[Any, Any, Any] | None = None

    def __enter__(self) -> "FakeRecordFunction":
        self.entered += 1
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.exited += 1
        self.exit_args = (exc_type, exc_value, traceback)
        if self.exit_error is not None:
            raise self.exit_error


def test_environment_defaults_are_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROFILE_TRACE_ENV, raising=False)
    monkeypatch.delenv(PROFILE_START_COMMAND_ENV, raising=False)
    monkeypatch.delenv(PROFILE_STOP_COMMAND_ENV, raising=False)

    real_import = builtins.__import__

    def reject_torch(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("default worker profiling must not import PyTorch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    profiler = profiler_from_environment()

    assert isinstance(profiler, NullWorkerProfiler)
    profiler.before_dispatch("STEP")
    with profiler.command_scope("STEP"):
        pass
    profiler.finish()


@pytest.mark.parametrize(
    "environment",
    [
        {PROFILE_START_COMMAND_ENV: "STEP"},
        {PROFILE_TRACE_ENV: "/tmp/trace.json"},
        {PROFILE_TRACE_ENV: "", PROFILE_START_COMMAND_ENV: "STEP"},
    ],
)
def test_incomplete_profiler_environment_fails_closed(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]
) -> None:
    for name in (PROFILE_TRACE_ENV, PROFILE_START_COMMAND_ENV, PROFILE_STOP_COMMAND_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(PROFILE_STOP_COMMAND_ENV, "SHUTDOWN")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises((RuntimeError, ValueError), match=PROFILE_TRACE_ENV):
        profiler_from_environment()


def test_profiler_rejects_ambiguous_start_and_stop_commands() -> None:
    with pytest.raises(ValueError, match="start and stop commands must differ"):
        WorkerProfiler("/tmp/worker-trace.json", "STEP", "STEP")


def test_profiler_starts_once_and_stops_on_requested_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "worker-trace.json"
    error_path = Path(str(trace_path) + ".error")
    error_path.write_text("stale profiler failure", encoding="utf-8")
    profiler = WorkerProfiler(str(trace_path), "STEP", "STOP")
    profile = FakeProfile()
    monkeypatch.setattr(profiler, "_create_profile", lambda: profile)

    profiler.before_dispatch("INIT")
    assert profile.entered == 0
    profiler.before_dispatch("STEP")
    profiler.before_dispatch("STEP")
    profiler.before_dispatch("OTHER")
    profiler.before_dispatch("STOP")
    profiler.before_dispatch("STEP")
    profiler.finish()

    assert profile.entered == 1
    assert profile.exited == 1
    assert profile.exported == [str(trace_path) + ".tmp.json"]
    assert trace_path.is_file()
    assert not error_path.exists()


def test_start_failure_is_attempted_once_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "worker-trace.json"
    profiler = WorkerProfiler(str(trace_path), "STEP", "STOP")
    profile = FakeProfile(enter_error=RuntimeError("profiler unavailable"))
    monkeypatch.setattr(profiler, "_create_profile", lambda: profile)

    with pytest.raises(RuntimeError, match="profiler unavailable"):
        profiler.before_dispatch("STEP")

    profiler.before_dispatch("STEP")
    profiler.finish()

    error_path = Path(str(trace_path) + ".error")
    assert "profiler unavailable" in error_path.read_text(encoding="utf-8")
    assert profile.entered == 1
    assert not profile.exported


def test_export_failure_does_not_break_worker_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "worker-trace.json"
    profiler = WorkerProfiler(str(trace_path), "STEP", "STOP")
    profile = FakeProfile(export_error=RuntimeError("trace export failed"))
    monkeypatch.setattr(profiler, "_create_profile", lambda: profile)

    profiler.before_dispatch("STEP")
    profiler.finish()
    profiler.finish()

    assert profile.exited == 1
    assert "trace export failed" in Path(str(trace_path) + ".error").read_text(encoding="utf-8")


def test_command_markers_cover_active_dispatches_and_skip_stop_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = WorkerProfiler("/tmp/worker-trace.json", "STEP", "STOP")
    profile = FakeProfile()
    markers: list[FakeRecordFunction] = []

    def fake_record_function(command: str) -> FakeRecordFunction:
        marker = FakeRecordFunction(f"unisim_worker_command/{command}")
        markers.append(marker)
        return marker

    monkeypatch.setattr(profiler, "_create_profile", lambda: profile)
    monkeypatch.setattr(profiler, "_create_record_function", fake_record_function)

    profiler.before_dispatch("INIT")
    with profiler.command_scope("INIT"):
        pass
    profiler.before_dispatch("STEP")
    with profiler.command_scope("STEP"):
        pass
    profiler.before_dispatch("STOP")
    with profiler.command_scope("STOP"):
        pass

    assert [marker.name for marker in markers] == ["unisim_worker_command/STEP"]
    assert markers[0].entered == 1
    assert markers[0].exited == 1
    assert markers[0].exit_args == (None, None, None)
    assert profile.exited == 1


def test_command_marker_closes_scope_on_dispatch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = WorkerProfiler("/tmp/worker-trace.json", "STEP", "STOP")
    profile = FakeProfile()
    marker = FakeRecordFunction("unisim_worker_command/STEP")
    monkeypatch.setattr(profiler, "_create_profile", lambda: profile)
    monkeypatch.setattr(profiler, "_create_record_function", lambda command: marker)
    profiler.before_dispatch("STEP")

    with pytest.raises(RuntimeError, match="dispatch failed"):
        with profiler.command_scope("STEP"):
            raise RuntimeError("dispatch failed")

    assert marker.exited == 1
    assert marker.exit_args is not None
    assert issubclass(marker.exit_args[0], RuntimeError)
    profiler.finish()
    assert profile.exited == 1


def test_record_function_start_failure_closes_profiler_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = tmp_path / "worker-trace.json"
    profiler = WorkerProfiler(str(trace_path), "STEP", "STOP")
    profile = FakeProfile()
    marker = FakeRecordFunction("unisim_worker_command/STEP")
    marker.enter_error = RuntimeError("marker unavailable")
    monkeypatch.setattr(profiler, "_create_profile", lambda: profile)
    monkeypatch.setattr(profiler, "_create_record_function", lambda command: marker)
    profiler.before_dispatch("STEP")

    with pytest.raises(RuntimeError, match="marker unavailable"):
        with profiler.command_scope("STEP"):
            pass

    assert profile.exited == 1
    assert "marker unavailable" in Path(str(trace_path) + ".error").read_text(encoding="utf-8")


def test_record_function_exit_failure_closes_profiler_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = tmp_path / "worker-trace.json"
    profiler = WorkerProfiler(str(trace_path), "STEP", "STOP")
    profile = FakeProfile()
    marker = FakeRecordFunction("unisim_worker_command/STEP")
    marker.exit_error = RuntimeError("marker exit failed")
    monkeypatch.setattr(profiler, "_create_profile", lambda: profile)
    monkeypatch.setattr(profiler, "_create_record_function", lambda command: marker)
    profiler.before_dispatch("STEP")

    with profiler.command_scope("STEP"):
        pass

    assert marker.exited == 1
    assert profile.exited == 1
    assert "marker exit failed" in Path(str(trace_path) + ".error").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("loader", "worker_path"),
    [
        (load_isaacgym_profiler, "isaacgym"),
        (load_isaacsim_profiler, "isaacsim"),
    ],
)
def test_worker_loaders_skip_profiler_module_by_default(
    monkeypatch: pytest.MonkeyPatch, loader: Any, worker_path: str
) -> None:
    for name in (PROFILE_TRACE_ENV, PROFILE_START_COMMAND_ENV, PROFILE_STOP_COMMAND_ENV):
        monkeypatch.delenv(name, raising=False)

    def reject_load(path: str, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("default worker startup must not load worker_profile.py")

    monkeypatch.setattr(importlib.util, "spec_from_file_location", reject_load)
    assert loader(str(_PROTOCOL_PATH)) is None
    assert worker_path in {"isaacgym", "isaacsim"}


@pytest.mark.parametrize(
    ("loader", "command"),
    [
        (load_isaacgym_profiler, "ISAACGYM_CUDA_IPC_STEP"),
        (load_isaacsim_profiler, "TENSOR_CUDA_STEP"),
    ],
)
def test_worker_loaders_construct_environment_profiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    command: str,
) -> None:
    trace_path = tmp_path / "worker-trace.json"
    for name in (PROFILE_TRACE_ENV, PROFILE_START_COMMAND_ENV, PROFILE_STOP_COMMAND_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(PROFILE_TRACE_ENV, str(trace_path))
    monkeypatch.setenv(PROFILE_START_COMMAND_ENV, command)
    monkeypatch.setenv(PROFILE_STOP_COMMAND_ENV, "SHUTDOWN")

    profiler = loader(str(_PROTOCOL_PATH))

    assert type(profiler).__name__ == "WorkerProfiler"
    assert profiler.start_command == command
    assert profiler.stop_command == "SHUTDOWN"
    assert profiler.trace_path == str(trace_path)
