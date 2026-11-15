"""External CUDA graph capture of backend tensor operations.

A Torch owner (``torch.cuda.graph``) can record a backend's device work into
its own CUDA graph only when, for the duration of the capture,

1. every backend kernel launches on the Torch capture stream,
2. the backend does not replay its own pre-captured graphs (a nested graph
   launch invalidates a Torch-owned capture, and Warp's child-graph insertion
   rejects it),
3. nothing in the captured region reads device memory from the host, and
4. host-side bookkeeping the region normally performs is re-applied after
   every replay, because a replay does not execute Python.

Backends implement (1) and (2) inside ``SimBackend.torch_graph_capture_scope``.
:class:`DeviceVerdicts` implements (3): fail-closed checks accumulate into
fixed-address device flags and are published by one host read at a boundary
the owner chooses. :class:`ExternalGraphCapture` implements (4): backend code
declares each host transition explicitly with
:meth:`ExternalGraphCapture.record_host_transition`, and the owner replays the
declared transitions with :meth:`ExternalGraphCapture.replay_host_transitions`
after each ``graph.replay()``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


class GraphCaptureError(RuntimeError):
    """An operation that cannot be recorded ran inside an external capture."""


class DeviceVerdicts:
    """Named fail-closed device checks published with one host read.

    Each named check owns one persistent 0-d ``torch.bool`` flag that starts
    ``True`` and is AND-ed with every verdict recorded under that name. Flags
    keep a fixed address once created, so recording is CUDA-graph safe: a
    captured ``record`` becomes a device ``logical_and_`` on the same storage
    on every replay. :meth:`flush` performs the only host read.

    Create every flag before a capture begins (one eager pass suffices);
    :meth:`ensure` exists for owners that want to pre-declare names.
    """

    def __init__(self) -> None:
        self._flags: dict[str, Any] = {}
        self._messages: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._flags)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._flags)

    def ensure(self, name: str, *, device: Any, message: str) -> Any:
        """Return the persistent flag for ``name``, creating it if needed."""
        flag = self._flags.get(name)
        if flag is None:
            import torch

            flag = torch.ones((), dtype=torch.bool, device=device)
            self._flags[name] = flag
        elif flag.device != _torch_device(device):
            raise ValueError(
                f"device verdict {name!r} lives on {flag.device}, got a verdict on {device}"
            )
        self._messages[name] = message
        return flag

    def record(self, name: str, ok: Any, message: str) -> None:
        """AND a 0-d bool device verdict into the persistent flag ``name``."""
        flag = self.ensure(name, device=ok.device, message=message)
        flag.logical_and_(ok.reshape(()))

    def device_flags(self) -> tuple[Any, ...]:
        """Persistent flags, for owners that fold them into their own verdict."""
        return tuple(self._flags.values())

    def failed(self) -> tuple[str, ...]:
        """Return failed check names with one host read; does not clear them."""
        if not self._flags:
            return ()
        import torch

        values = torch.stack(list(self._flags.values())).cpu().tolist()
        return tuple(name for name, ok in zip(self._flags, values, strict=True) if not ok)

    def clear(self) -> None:
        """Reset every flag to ``True`` on the device (no host read)."""
        for flag in self._flags.values():
            flag.fill_(True)

    def flush(self) -> None:
        """Raise ``ValueError`` naming every failed check, then clear all flags."""
        failed = self.failed()
        if not failed:
            return
        self.clear()
        raise ValueError(
            "deferred tensor validation failed: "
            + "; ".join(self._messages.get(name, name) for name in failed)
        )


def _torch_device(device: Any) -> Any:
    import torch

    return torch.device(device)


@dataclass
class ExternalGraphCapture:
    """Handle for one backend recording into a Torch-owned CUDA graph.

    Returned by ``SimBackend.torch_graph_capture_scope``. The handle outlives
    the scope: after each replay of the owner's graph, call
    :meth:`replay_host_transitions` so the backend's host-side state matches
    what an eager execution of the captured calls would have produced.
    """

    torch_stream: Any
    backend_stream: Any = None
    # Backend-native capture state that must outlive the owner's graph (for
    # MJWarp, Warp's external-capture handle owning capture-time temporaries).
    backend_graph: Any = None
    _transitions: list[tuple[str, Callable[[], None]]] = field(default_factory=list)
    _guards: list[tuple[str, Callable[[], str | None]]] = field(default_factory=list)
    _closed: bool = False

    @property
    def host_transition_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._transitions)

    def record_host_transition(self, name: str, transition: Callable[[], None]) -> None:
        """Run ``transition`` now and replay it after every graph replay.

        A transition is a host-only state update (it must not read device
        memory). It runs once now, for the capture pass, and once after every
        replay, so per-call counters advance exactly as in eager execution.
        """
        if self._closed:
            raise GraphCaptureError(
                f"host transition {name!r} recorded after the capture scope closed"
            )
        transition()
        self._transitions.append((name, transition))

    def record_replay_guard(self, name: str, guard: Callable[[], str | None]) -> None:
        """Declare a host precondition the captured work was specialized on.

        Captured device work is fixed at capture time, so a backend whose
        captured path depends on host state (for example "no interval wrench is
        staged") records a guard returning ``None`` when the precondition still
        holds, or a reason string when replaying would diverge from eager
        execution. :meth:`check_replay_guards` runs every guard.
        """
        if self._closed:
            raise GraphCaptureError(
                f"replay guard {name!r} recorded after the capture scope closed"
            )
        reason = guard()
        if reason is not None:
            raise GraphCaptureError(f"cannot capture: {name}: {reason}")
        self._guards.append((name, guard))

    def check_replay_guards(self) -> None:
        """Raise :class:`GraphCaptureError` if a recorded precondition no longer holds."""
        for name, guard in self._guards:
            reason = guard()
            if reason is not None:
                raise GraphCaptureError(f"captured graph cannot replay: {name}: {reason}")

    def replay_host_transitions(self) -> None:
        """Re-apply every recorded host transition, in recording order."""
        for _, transition in self._transitions:
            transition()

    def close(self) -> None:
        self._closed = True


@contextmanager
def closing_capture(capture: ExternalGraphCapture) -> Iterator[ExternalGraphCapture]:
    """Yield ``capture`` and mark it closed when the scope exits."""
    try:
        yield capture
    finally:
        capture.close()


__all__ = [
    "DeviceVerdicts",
    "ExternalGraphCapture",
    "GraphCaptureError",
    "closing_capture",
]
