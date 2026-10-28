"""Real Viser smoke for the explicit Newton renderer selection."""

# ruff: noqa: E402
from __future__ import annotations

import socket
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("newton")
viser = pytest.importorskip("viser")
warp = pytest.importorskip("warp")

from .test_replay_public_apis import _backend, _single_scene


@pytest.fixture(scope="module", autouse=True)
def _cuda():
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("Newton Viser smoke requires real CUDA")


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_real_viser_model_state_gui_render_and_teardown(tmp_path: Path):
    backend = _backend(_single_scene(tmp_path), num_envs=2)
    viewer = None
    try:
        port = _unused_port()
        backend.bind_renderer(
            renderer="viser",
            viser_kwargs={"port": port, "label": "UniSim smoke", "verbose": False},
        )
        viewer = backend._viewer
        server = backend.get_viser_server()
        assert isinstance(server, viser.ViserServer)
        assert server.get_port() == port
        button = server.gui.add_button("Replay smoke")
        assert button is not None

        backend.render()
        assert viewer._scene_handles
        backend.render()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=5) as response:
            assert response.status == 200
        with pytest.raises(NotImplementedError, match="does not support GL frame capture"):
            backend.capture_video_frame()
        with pytest.raises(RuntimeError, match="already initialized"):
            backend.bind_renderer(
                renderer="viser",
                viser_kwargs={"port": port + 1, "label": "UniSim smoke", "verbose": False},
            )
    finally:
        backend.close()
    assert viewer is not None
    assert not viewer.is_running()
    with pytest.raises(RuntimeError, match="requires bind_renderer first"):
        backend.get_viser_server()
