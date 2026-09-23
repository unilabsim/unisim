"""Dependency-free terminal progress reporting for cold-path scene builds.

Long multi-variant scene builds (``compose_scene``, worker-source
materialization, native worker initialization) previously ran silently for
minutes.  These helpers render a single-line carriage-return progress bar on
stderr when stderr is a terminal.  Set ``UNISIM_PROGRESS`` to ``always`` (or
``1``/``on``/``true``) to force output, or to ``never`` (or ``0``/``off``/
``false``) to disable it; the default ``auto`` follows terminal detection.
"""

from __future__ import annotations

import os
import sys
import time
from typing import IO

ENV_PROGRESS = "UNISIM_PROGRESS"

_ON = {"1", "on", "always", "true", "yes"}
_OFF = {"0", "off", "never", "false", "no"}

_BAR_WIDTH = 28
_MIN_RENDER_INTERVAL_S = 0.1


def progress_enabled() -> bool:
    """Return True when terminal progress output should be rendered."""
    value = os.environ.get(ENV_PROGRESS, "auto").strip().lower()
    if value in _OFF:
        return False
    if value in _ON:
        return True
    return sys.stderr.isatty()


class ProgressBar:
    """Render one throttled stderr progress line; a no-op when disabled."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        stream: IO[str] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._label = label
        self._total = max(0, int(total))
        self._stream = stream if stream is not None else sys.stderr
        self._enabled = progress_enabled() if enabled is None else enabled
        self._started = time.monotonic()
        self._last_render = 0.0
        self._rendered_done: int | None = None
        self._done = 0
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def label(self) -> str:
        return self._label

    def update(self, done: int, *, force: bool = False) -> None:
        """Record ``done`` completed units and redraw at most ~10 times/s."""
        if self._closed:
            raise RuntimeError("progress bar is closed")
        self._done = min(max(0, int(done)), self._total) if self._total else int(done)
        if not self._enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_render < _MIN_RENDER_INTERVAL_S:
            return
        self._last_render = now
        self._render()

    def advance(self, step: int = 1, *, force: bool = False) -> None:
        self.update(self._done + step, force=force)

    def _render(self) -> None:
        elapsed = time.monotonic() - self._started
        if self._total > 0:
            fraction = min(1.0, self._done / self._total)
            filled = round(_BAR_WIDTH * fraction)
            bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
            count = f"{self._done}/{self._total}"
            if 0 < self._done < self._total and elapsed > 0.5:
                eta = f" ETA {elapsed * (self._total - self._done) / self._done:.0f}s"
            else:
                eta = ""
        else:
            bar = "-" * _BAR_WIDTH
            count = str(self._done)
            eta = ""
        line = f"\r{self._label}: [{bar}] {count} ({elapsed:.0f}s{eta})"
        self._stream.write(line.ljust(100)[:100])
        self._stream.flush()
        self._rendered_done = self._done

    def close(self) -> None:
        """Draw the final state and end the line. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if not self._enabled:
            return
        if self._total:
            self._done = self._total
        if self._done != self._rendered_done:
            self._render()
        self._stream.write("\n")
        self._stream.flush()

    def __enter__(self) -> ProgressBar:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["ENV_PROGRESS", "ProgressBar", "progress_enabled"]
