"""Opt-in PyTorch profiling for an isolated Isaac worker process.

The environment-controlled helper keeps diagnostics out of the production data
plane.  A maintainer sets a trace destination and the first CUDA IPC command to
instrument; the worker starts Kineto immediately before dispatching that command
and exports a Chrome trace when its framed protocol loop exits.
"""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

PROFILE_TRACE_ENV = "UNISIM_ISAAC_WORKER_PROFILE_TRACE"
PROFILE_START_COMMAND_ENV = "UNISIM_ISAAC_WORKER_PROFILE_START_COMMAND"
PROFILE_STOP_COMMAND_ENV = "UNISIM_ISAAC_WORKER_PROFILE_STOP_COMMAND"


class WorkerProfiler:
    """Own one lazily entered profiler context in the worker process."""

    def __init__(self, trace_path: str, start_command: str, stop_command: str) -> None:
        if not trace_path or not start_command or not stop_command:
            raise ValueError(
                f"worker profiling requires non-empty {PROFILE_TRACE_ENV}, "
                f"{PROFILE_START_COMMAND_ENV}, and {PROFILE_STOP_COMMAND_ENV}"
            )
        if start_command == stop_command:
            raise ValueError("worker profiling start and stop commands must differ")
        self.trace_path = os.path.abspath(trace_path)
        self.start_command = start_command
        self.stop_command = stop_command
        self._profiler: Optional[Any] = None
        self._closed = False

    def before_dispatch(self, command: str) -> None:
        if command == self.stop_command:
            self.finish()
            return
        if self._profiler is not None or self._closed or command != self.start_command:
            return
        # Import through the worker interpreter only when profiling is enabled.
        # Start is attempted once.  A failed diagnostic propagates the command
        # error, is recorded beside the requested trace, and never leaves the
        # worker loop with a partially entered profiler.
        try:
            profile = self._create_profile()
        except Exception as exc:
            self._closed = True
            self._record_failure(exc)
            raise

        self._profiler = profile

        try:
            profile.__enter__()
        except Exception as exc:
            self._closed = True
            self._record_failure(exc)
            raise

    @contextmanager
    def command_scope(self, command: str) -> Iterator[None]:
        """Annotate one dispatch while profiling is active."""

        if self._profiler is None or self._closed:
            yield
            return

        marker = self._create_record_function(command)
        try:
            marker.__enter__()
        except Exception as exc:
            self.finish()
            self._record_failure(exc)
            raise
        try:
            yield
        finally:
            try:
                marker.__exit__(*sys.exc_info())
            except Exception as exc:
                self.finish()
                self._record_failure(exc)

    def _create_profile(self) -> Any:
        from torch.profiler import ProfilerActivity, profile

        return profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])

    @staticmethod
    def _create_record_function(command: str) -> Any:
        from torch.profiler import record_function

        return record_function(f"unisim_worker_command/{command}")

    def finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        profiler = self._profiler
        self._profiler = None
        if profiler is None:
            return
        try:
            profiler.__exit__(None, None, None)
            self._export_trace(profiler)
        except Exception as exc:
            self._record_failure(exc)

    def _export_trace(self, profiler: Any) -> None:
        parent = os.path.dirname(self.trace_path)
        temporary_path = self.trace_path + ".tmp.json"
        os.makedirs(parent, exist_ok=True)
        try:
            profiler.export_chrome_trace(temporary_path)
            os.replace(temporary_path, self.trace_path)
            try:
                os.unlink(self.trace_path + ".error")
            except FileNotFoundError:
                pass
        finally:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass

    def _record_failure(self, exc: Exception) -> None:
        """Persist profiler diagnostics without turning shutdown into a crash."""

        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            path = self.trace_path + ".error"
            parent = os.path.dirname(path)
            os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(details)
        except OSError:
            print(f"UniSim worker profiler failed:\n{details}", file=sys.stderr)


class NullWorkerProfiler:
    """No-op fallback used unless profiling is explicitly requested."""

    def before_dispatch(self, command: str) -> None:
        return None

    def command_scope(self, command: str) -> Any:
        from contextlib import nullcontext

        return nullcontext()

    def finish(self) -> None:
        return None


def profiler_from_environment() -> Any:
    trace_path = os.environ.get(PROFILE_TRACE_ENV)
    start_command = os.environ.get(PROFILE_START_COMMAND_ENV)
    stop_command = os.environ.get(PROFILE_STOP_COMMAND_ENV, "SHUTDOWN")
    if trace_path is None and start_command is None:
        return NullWorkerProfiler()
    if trace_path is None or start_command is None:
        raise RuntimeError(
            f"worker profiling requires both {PROFILE_TRACE_ENV} and {PROFILE_START_COMMAND_ENV}"
        )
    return WorkerProfiler(trace_path, start_command, stop_command)


__all__ = [
    "PROFILE_START_COMMAND_ENV",
    "PROFILE_STOP_COMMAND_ENV",
    "PROFILE_TRACE_ENV",
    "NullWorkerProfiler",
    "WorkerProfiler",
    "profiler_from_environment",
]
