"""CPU coverage for MJWarp tensor-reader hot-path reuse."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

from unisim import MjwarpBackend


def _sensor_reader_fixture() -> tuple[MjwarpBackend, list[Any], object]:
    qpos = SimpleNamespace(label="qpos")
    sensor_slice = object()

    class _Sensordata:
        def __getitem__(self, key: slice) -> object:
            return sensor_slice

    sensordata = _Sensordata()
    backend = object.__new__(MjwarpBackend)
    backend._device_data = SimpleNamespace(qpos=qpos, sensordata=sensordata)
    backend._state_torch_views = {}
    backend._sensor_torch_views = {}
    backend._sensor_slots = {"frame": (0, 3)}
    backend._tracked_sensor_names = set()
    backend._body_pair_sensor_plans = {}
    backend._body_pair_contact_views = {}
    backend._authored_sensor_state_dirty = False
    backend._require_entity_healthy = lambda: None

    wrapped: list[Any] = []

    def torch_view(device_array: Any, *, require_contiguous: bool = True) -> Any:
        del require_contiguous
        wrapped.append(device_array)
        return device_array

    backend._torch_view = torch_view  # type: ignore[method-assign]
    return backend, wrapped, sensor_slice


def test_sensor_reads_reuse_the_stable_device_reference() -> None:
    backend, wrapped, sensor_slice = _sensor_reader_fixture()

    first = backend.get_sensor_view("frame")
    second = backend.get_sensor_view("frame")

    assert first is second
    assert wrapped == [
        backend._device_data.qpos,
        sensor_slice,
    ]


def test_fresh_batch_refresh_publishes_without_decoding_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(MjwarpBackend)
    backend._require_entity_healthy = lambda: None
    backend._body_pair_sensor_plans = {"pair": object()}
    backend._body_pair_contact_views = {"pair": object()}
    backend._body_pair_contacts_stale = False
    backend._body_pair_contacts_allow_decode = False

    decoded = 0
    published = 0

    def fail_decode() -> None:
        nonlocal decoded
        decoded += 1

    def publish() -> None:
        nonlocal published
        published += 1

    monkeypatch.setattr(backend, "_refresh_body_pair_contacts_device_only", fail_decode)
    monkeypatch.setattr(backend, "_publish_body_pair_contact_batches", publish)

    backend.refresh_body_pair_contact_views(("pair",))

    assert decoded == 0
    assert published == 1


def test_vectorized_decode_does_not_sanitize_the_full_contact_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    backend = object.__new__(MjwarpBackend)
    backend._num_envs = 1
    backend._body_pair_contact_views = {"pair": object()}
    backend._body_pair_contact_index = object()
    backend._body_pair_contacts_stale = True
    backend._body_pair_contacts_allow_decode = True

    capacity = 8

    class _ShapedArray:
        shape = (capacity,)

    raw = {
        "worldid": torch.tensor([0, -1, -1, -1, -1, -1, -1, -1], dtype=torch.int64),
        "geom": torch.tensor([[1, 2]] + [[-1, -1]] * 7, dtype=torch.int64),
        "pos": torch.full((capacity, 3), float("nan")),
        "dist": torch.full((capacity,), float("nan")),
        "frame": torch.full((capacity, 3, 3), float("nan")),
        "force": torch.full((capacity, 6), float("nan")),
        "nacon": torch.tensor(1, dtype=torch.int64),
    }
    raw["pos"][0].fill_(1.0)
    raw["dist"][0].fill_(0.0)
    raw["frame"][0].fill_(0.0)
    raw["force"][0].fill_(0.0)
    worldid = object()
    geom = object()
    pos = object()
    dist = _ShapedArray()
    frame = object()
    nacon = object()
    raw.update(
        {
            worldid: raw["worldid"],
            geom: raw["geom"],
            pos: raw["pos"],
            dist: raw["dist"],
            frame: raw["frame"],
            nacon: raw["nacon"],
        }
    )
    backend._device_data = SimpleNamespace(
        contact=SimpleNamespace(
            worldid=worldid,
            geom=geom,
            pos=pos,
            dist=dist,
            frame=frame,
        ),
        nacon=nacon,
    )
    backend._body_pair_contact_geom_body = torch.arange(16, dtype=torch.int64)
    backend._body_pair_contact_ordinals = torch.arange(capacity, dtype=torch.int64)
    backend._body_pair_contact_scratch_ids = object()
    scratch_force = object()
    backend._body_pair_contact_scratch_force = scratch_force
    raw[scratch_force] = raw["force"]
    backend._device_model = object()
    backend._mujoco_warp = SimpleNamespace(contact_force=lambda *args, **kwargs: None)
    backend._warp = SimpleNamespace(to_torch=lambda value: raw[value])

    snapshots: list[dict[str, Any]] = []
    monkeypatch.setattr(
        backend,
        "_ensure_body_pair_contact_scratch",
        lambda _capacity: None,
    )
    monkeypatch.setattr(
        backend,
        "_refresh_vectorized_body_pair_contacts",
        lambda snapshot: snapshots.append(snapshot),
    )
    monkeypatch.setattr(backend, "_publish_body_pair_contact_batches", lambda: None)

    original_where = torch.where
    full_capacity_where: list[int] = []

    def counted_where(condition: Any, input: Any, other: Any) -> Any:
        if condition.ndim == 1 and condition.numel() == capacity:
            full_capacity_where.append(int(condition.numel()))
        return original_where(condition, input, other)

    monkeypatch.setattr(torch, "where", counted_where)

    backend._refresh_body_pair_contacts_device_only()

    assert full_capacity_where == []
    assert len(snapshots) == 1
