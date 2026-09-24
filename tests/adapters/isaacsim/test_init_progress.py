"""SDK-free checks for IsaacSim worker INIT progress reporting."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from unisim import IsaacSimBackend
from unisim.backend.isaacsim.scene_worker import _InitProgress
from unisim.backend.subprocess_ipc import protocol
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import SceneEntitySpec
from unisim.scene import SceneCfg


class _Context:
    def __init__(self, stream: io.BytesIO | None) -> None:
        self.protocol = protocol
        if stream is not None:
            self.progress_out = stream


def _frames(stream: io.BytesIO) -> list[dict]:
    reader = io.BytesIO(stream.getvalue())
    messages = []
    while True:
        try:
            messages.append(protocol.recv_message(reader))
        except EOFError:
            return messages


def test_init_progress_frames_throttle_and_force():
    stream = io.BytesIO()
    progress = _InitProgress(_Context(stream), min_interval=60.0)
    progress.report("build", 1, 4, force=True)
    progress.report("build", 2, 4)
    progress.report("build", 3, 4)
    progress.report("build", 4, 4, force=True)
    messages = _frames(stream)
    assert [message["cmd"] for message in messages] == [protocol.CMD_PROGRESS] * 2
    assert [message["payload"]["done"] for message in messages] == [1, 4]
    assert all(message["payload"] == {
        "label": "build",
        "done": message["payload"]["done"],
        "total": 4,
    } for message in messages)


def test_init_progress_honors_interval_after_it_expires():
    stream = io.BytesIO()
    progress = _InitProgress(_Context(stream), min_interval=0.0)
    progress.report("build", 1, 2)
    progress.report("build", 2, 2)
    assert len(_frames(stream)) == 2


def test_init_progress_disabled_or_streamless_emits_nothing():
    stream = io.BytesIO()
    disabled = _InitProgress(_Context(stream), enabled=False)
    disabled.report("build", 1, 1, force=True)
    assert stream.getvalue() == b""
    streamless = _InitProgress(_Context(None))
    streamless.report("build", 1, 1, force=True)


def test_init_progress_never_masks_a_write_failure():
    class BrokenStream:
        def write(self, data: bytes) -> int:
            raise OSError("closed")

    progress = _InitProgress(_Context(BrokenStream()))  # type: ignore[arg-type]
    progress.report("build", 1, 1, force=True)


def test_worker_init_payload_declares_progress_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "object.xml"
    source.write_text(
        '<mujoco><worldbody><body name="base"><freejoint/>'
        '<geom name="object_geom" size=".1" mass="1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    config = SceneCfg(
        entity_assets=(
            SceneEntitySpec("object", ModelSourceDescriptor(str(source)), kind="rigid"),
        )
    )
    backend = IsaacSimBackend(config, 2, 0.002)
    monkeypatch.setenv("UNISIM_PROGRESS", "always")
    assert backend._worker_init_payload()["init_progress"] is True
    monkeypatch.setenv("UNISIM_PROGRESS", "never")
    assert backend._worker_init_payload()["init_progress"] is False
