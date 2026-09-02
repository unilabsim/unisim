"""Compatibility surface for shared MJCF subprocess metadata helpers."""

from unisim.backend.subprocess_ipc import sensors as _sensors
from unisim.backend.subprocess_ipc.sensors import (
    SceneMetadata,
)


def scan_scene_metadata(model_file: str, *, backend_label: str = "isaacgym") -> SceneMetadata:
    """Preserve the historical IsaacGym diagnostic label for direct callers."""
    return _sensors.scan_scene_metadata(model_file, backend_label=backend_label)


__all__ = _sensors.__all__
