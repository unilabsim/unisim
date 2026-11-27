"""Dispatch of the MJWarp final-state and derived-constant refresh.

Both refreshes replay one backend-owned CUDA graph when backend graphs are
active and launch the same kernels eagerly otherwise.  These tests build
constructor-free backends, so they need neither a GPU, nor Torch, nor the
optional MJWarp runtime; the CUDA behaviour lives in
``test_refresh_cuda_graphs.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unisim import MjwarpBackend
from unisim.backend.graph_capture import ExternalGraphCapture

_FINAL_STATE_KERNELS = ("kinematics", "com_pos", "com_vel", "sensor_pos", "sensor_vel")
_REFRESH_VARIANTS = ((2, False), (2, True), (1, False), (1, True))


def _dispatch_shell() -> tuple[MjwarpBackend, dict[str, list[tuple[str, dict]]]]:
    """Constructor-free backend whose kernels and graph launches only record."""
    backend = MjwarpBackend.__new__(MjwarpBackend)
    recorded: dict[str, list[tuple[str, dict]]] = {"kernel": [], "launch": []}
    backend._device_model = object()
    backend._device_data = object()
    backend._external_capture = None
    backend._refresh_graph = object()
    backend._set_const_graphs = {variant: object() for variant in _REFRESH_VARIANTS}

    def recording(name: str):
        def launch(*args: object, **kwargs: object) -> None:
            recorded["kernel"].append((name, kwargs))

        return launch

    backend._mujoco_warp = SimpleNamespace(
        SetConstGroup=SimpleNamespace(ALL=7, KIN0=2, INERT0=4),
        **{name: recording(name) for name in (*_FINAL_STATE_KERNELS, "set_const")},
    )
    backend._warp = SimpleNamespace(capture_launch=recorded["launch"].append)
    return backend, recorded


def _force_eager(backend: MjwarpBackend, reason: str) -> None:
    """Reach one of the states where both refresh dispatchers launch eagerly.

    A constructor-free instance never sets the graph switch, a disabled or
    externally captured backend cannot replay, and a backend whose refresh
    graphs are missing must fall back even while its step graphs stay active.
    """
    if reason != "constructor-free":
        backend._cuda_graph_enabled = reason != "disabled"
    if reason == "external capture":
        backend._external_capture = ExternalGraphCapture(torch_stream=None)
    elif reason == "missing graph":
        backend._refresh_graph = None
        backend._set_const_graphs = {}


_EAGER_REASONS = ["disabled", "external capture", "constructor-free", "missing graph"]


def test_final_state_refresh_replays_one_graph_while_backend_graphs_are_active() -> None:
    backend, recorded = _dispatch_shell()
    backend._cuda_graph_enabled = True

    backend._launch_final_state_refresh()

    assert recorded["launch"] == [backend._refresh_graph]
    assert recorded["kernel"] == []


@pytest.mark.parametrize("eager_reason", _EAGER_REASONS)
def test_final_state_refresh_launches_the_same_kernels_eagerly(eager_reason: str) -> None:
    backend, recorded = _dispatch_shell()
    _force_eager(backend, eager_reason)

    backend._launch_final_state_refresh()

    assert recorded["launch"] == []
    assert [name for name, _ in recorded["kernel"]] == list(_FINAL_STATE_KERNELS)


@pytest.mark.parametrize(("level", "groups"), [(2, 7), (1, 6)], ids=["level-2", "level-1"])
@pytest.mark.parametrize("restore", [False, True], ids=["no-restore", "restore"])
def test_derived_constant_refresh_replays_its_own_graph_variant(
    level: int, groups: int, restore: bool
) -> None:
    backend, recorded = _dispatch_shell()
    backend._cuda_graph_enabled = True

    backend._refresh_derived_constants(level, restore=restore)

    assert recorded["launch"] == [backend._set_const_graphs[(level, restore)]]
    assert recorded["kernel"] == []


@pytest.mark.parametrize(("level", "groups"), [(2, 7), (1, 6)], ids=["level-2", "level-1"])
@pytest.mark.parametrize("restore", [False, True], ids=["no-restore", "restore"])
@pytest.mark.parametrize("eager_reason", _EAGER_REASONS)
def test_derived_constant_refresh_launches_one_set_const_eagerly(
    level: int, groups: int, restore: bool, eager_reason: str
) -> None:
    backend, recorded = _dispatch_shell()
    _force_eager(backend, eager_reason)

    backend._refresh_derived_constants(level, restore=restore)

    assert recorded["launch"] == []
    # One eager pass: level 2 recomputes every group and level 1 only the
    # qpos0-dependent ones, through the non-deprecated set_const spelling.
    assert recorded["kernel"] == [("set_const", {"groups": groups, "restore": restore})]


def test_disabling_backend_graphs_retires_every_graph_handle() -> None:
    backend = MjwarpBackend.__new__(MjwarpBackend)
    handles = {
        "_step_graph": object(),
        "_forward_graph": object(),
        "_reset_graph": object(),
        "_reset_scratch_reset_graph": object(),
        "_reset_scratch_forward_graph": object(),
        "_refresh_graph": object(),
    }
    for name, handle in handles.items():
        setattr(backend, name, handle)
    backend._set_const_graphs = {variant: object() for variant in _REFRESH_VARIANTS}
    backend._cuda_graph_enabled = True

    backend._disable_cuda_graphs("forced")

    assert backend._cuda_graph_enabled is False
    assert backend._cuda_graph_disable_reason == "forced"
    assert backend._replay_backend_graphs is False
    assert all(getattr(backend, name) is None for name in handles)
    assert backend._set_const_graphs == {}
