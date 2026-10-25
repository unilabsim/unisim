"""Shared packed selected-body pose blocks for host-bridge transfer plans."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from unisim.backend.base import HostBodyPoseViews, TensorIOSpec

_BODY_FIELD_WIDTHS = {"pos_w": 3, "quat_w": 4}
_BODY_FIELD_PACKET_NAMES = {"pos_w": "body_pos_w", "quat_w": "body_quat_w"}


@dataclass(frozen=True)
class BodyPosePacketBlock:
    """Compile-time layout for one pose-only selected-body request."""

    body_names: tuple[str, ...]
    body_fields: tuple[str, ...]
    packet_names: tuple[str, ...]
    offsets: Mapping[str, int]
    row_width: int

    @property
    def view_names(self) -> tuple[str, ...]:
        return tuple(_BODY_FIELD_PACKET_NAMES[field] for field in self.body_fields)


def compile_body_pose_packet_block(spec: TensorIOSpec, *, owner: str) -> BodyPosePacketBlock | None:
    """Resolve the request order/widths, failing closed before hot-path work."""

    if not spec.body_names:
        if spec.body_fields:
            raise ValueError(f"{owner} body_fields require body_names")
        return None
    if not spec.body_fields:
        raise ValueError(f"{owner} body request must name at least one body field")
    offsets: dict[str, int] = {}
    cursor = 0
    for field in spec.body_fields:
        packet_name = _BODY_FIELD_PACKET_NAMES[field]
        offsets[packet_name] = cursor
        cursor += _BODY_FIELD_WIDTHS[field] * len(spec.body_names)
    return BodyPosePacketBlock(
        body_names=tuple(spec.body_names),
        body_fields=tuple(spec.body_fields),
        packet_names=tuple(_BODY_FIELD_PACKET_NAMES[field] for field in spec.body_fields),
        offsets=offsets,
        row_width=cursor,
    )


def validate_known_body_names(
    requested: tuple[str, ...], available: tuple[str, ...], *, owner: str
) -> None:
    """Fail closed on missing bodies after backend-owned name resolution."""

    namespace = set(available)
    missing = [name for name in requested if name not in namespace]
    if missing:
        raise KeyError(
            f"{owner} body pose blocks request bodies missing from the tracked namespace: "
            f"{missing}; available={list(available)}"
        )


def body_pose_packet_views(
    packet: Any, block: BodyPosePacketBlock, *, owner: str
) -> HostBodyPoseViews:
    """Return stable E x B x 3/4 views into one contiguous packet."""

    import torch

    if not isinstance(packet, torch.Tensor):
        raise TypeError(f"{owner} packed body pose views require a Torch packet")
    count = len(block.body_names)
    values: dict[str, Any] = {}
    for field, name in zip(block.body_fields, block.packet_names, strict=True):
        start = block.offsets[name]
        width = _BODY_FIELD_WIDTHS[field] * count
        values[field] = packet[:, start : start + width].reshape(
            -1, count, _BODY_FIELD_WIDTHS[field]
        )
    return HostBodyPoseViews(
        body_names=block.body_names,
        pos_w=values.get("pos_w"),
        quat_w=values.get("quat_w"),
    )


def pack_body_pose_packet_block(
    destination: np.ndarray,
    block: BodyPosePacketBlock,
    read_poses: Callable[[np.ndarray | None], tuple[np.ndarray, np.ndarray]],
    rows: np.ndarray | None,
) -> None:
    """Pack selected rows from an authoritative adapter pose read."""

    positions, quaternions = read_poses(rows)
    count = len(block.body_names)
    sources = {"pos_w": positions, "quat_w": quaternions}
    for field, name in zip(block.body_fields, block.packet_names, strict=True):
        source = np.asarray(sources[field], dtype=np.float32)
        expected = (
            (-1, count, _BODY_FIELD_WIDTHS[field])
            if rows is None
            else (np.asarray(rows).shape[0], count, _BODY_FIELD_WIDTHS[field])
        )
        source_matches = source.ndim == 3 and source.shape[1:] == expected[1:]
        if rows is not None:
            source_matches = source_matches and source.shape[0] == expected[0]
        if not source_matches:
            received = tuple(source.shape)
            raise RuntimeError(
                f"authoritative body pose source for {field} changed shape; "
                f"expected {expected if rows is not None else ('*', *expected[1:])}, got {received}"
            )
        stop = block.offsets[name] + _BODY_FIELD_WIDTHS[field] * count
        destination[:, block.offsets[name] : stop] = source.reshape(source.shape[0], -1)


__all__ = [
    "BodyPosePacketBlock",
    "body_pose_packet_views",
    "compile_body_pose_packet_block",
    "pack_body_pose_packet_block",
    "validate_known_body_names",
]
