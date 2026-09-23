"""Terminal progress reporting gating and rendering."""

from __future__ import annotations

import io

import pytest

from unisim.progress import ProgressBar, progress_enabled


def test_progress_disabled_for_non_tty_stream(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("UNISIM_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    assert not progress_enabled()


def test_progress_enabled_for_tty_stream(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("UNISIM_PROGRESS", raising=False)
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    assert progress_enabled()


@pytest.mark.parametrize("value", ["0", "off", "never", "false", "NO"])
def test_progress_env_forces_off(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("UNISIM_PROGRESS", value)
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    assert not progress_enabled()


@pytest.mark.parametrize("value", ["1", "on", "always", "TRUE"])
def test_progress_env_forces_on(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("UNISIM_PROGRESS", value)
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    assert progress_enabled()


def test_disabled_bar_emits_nothing():
    stream = io.StringIO()
    bar = ProgressBar("build", 10, stream=stream, enabled=False)
    for index in range(10):
        bar.update(index + 1)
    bar.close()
    assert stream.getvalue() == ""


def test_enabled_bar_renders_single_progressing_line():
    stream = io.StringIO()
    bar = ProgressBar("build", 4, stream=stream, enabled=True)
    for index in range(4):
        bar.update(index + 1, force=True)
    bar.close()
    output = stream.getvalue()
    assert output.count("\r") >= 4
    assert output.endswith("\n")
    assert "4/4" in output


def test_bar_close_is_idempotent_and_update_after_close_raises():
    stream = io.StringIO()
    bar = ProgressBar("build", 2, stream=stream, enabled=True)
    bar.update(2, force=True)
    bar.close()
    bar.close()
    with pytest.raises(RuntimeError):
        bar.update(1)
