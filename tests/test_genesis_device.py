"""Tests for the Genesis CUDA_VISIBLE_DEVICES pinning contract.

Quadrants binds its CUDA runtime to the first visible device, so a non-zero
Genesis device request is honored by shrinking CUDA_VISIBLE_DEVICES before
any CUDA context exists (issue #1508).  Torch CUDA entry points are stubbed
so the lane stays host-free and independent of test-order CUDA state.
"""

from __future__ import annotations

import os

import pytest

from unisim.backend.genesis.materialization import (
    _pin_cuda_visible_devices,
    _resolve_genesis_device_id,
)


class _StubCuda:
    def __init__(self, *, initialized: bool = False, count: int = 2) -> None:
        self.initialized = initialized
        self.count = count
        self.set_calls: list[int] = []

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self.initialized

    def device_count(self) -> int:
        return self.count

    def current_device(self) -> int:
        return 0

    def set_device(self, index: int) -> None:
        self.set_calls.append(index)


class _StubTorch:
    def __init__(self, *, initialized: bool = False, count: int = 2) -> None:
        self.cuda = _StubCuda(initialized=initialized, count=count)


@pytest.fixture
def clean_visible_devices(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(os.environ, "CUDA_VISIBLE_DEVICES", raising=False)
    return monkeypatch


def test_nonzero_device_pins_visibility_without_existing_cvd(clean_visible_devices) -> None:
    torch = _StubTorch()

    assert _resolve_genesis_device_id(torch, 1) == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert torch.cuda.set_calls == [0]


def test_nonzero_device_translates_existing_cvd(clean_visible_devices) -> None:
    clean_visible_devices.setitem(os.environ, "CUDA_VISIBLE_DEVICES", "4,5")
    torch = _StubTorch(count=2)

    assert _resolve_genesis_device_id(torch, 1) == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
    assert torch.cuda.set_calls == [0]


def test_device_zero_binds_without_pinning(clean_visible_devices) -> None:
    torch = _StubTorch()

    assert _resolve_genesis_device_id(torch, 0) == 0
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert torch.cuda.set_calls == [0]


def test_pin_fails_closed_after_cuda_init(clean_visible_devices) -> None:
    torch = _StubTorch(initialized=True)

    with pytest.raises(RuntimeError, match="before any CUDA context"):
        _resolve_genesis_device_id(torch, 1)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_pin_rejects_out_of_range_host_index(clean_visible_devices) -> None:
    torch = _StubTorch(count=2)

    with pytest.raises(ValueError, match="out of range"):
        _resolve_genesis_device_id(torch, 2)


def test_pin_rejects_index_beyond_existing_cvd(clean_visible_devices) -> None:
    clean_visible_devices.setitem(os.environ, "CUDA_VISIBLE_DEVICES", "3")
    torch = _StubTorch(count=1)

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        _pin_cuda_visible_devices(torch, 1)


def test_unset_device_keeps_current_device(clean_visible_devices) -> None:
    torch = _StubTorch()

    assert _resolve_genesis_device_id(torch, None) == 0
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert torch.cuda.set_calls == []
