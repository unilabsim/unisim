"""Shared scene-materialization helpers."""

from __future__ import annotations

import os


class TemporarySceneCleanup:
    """Own the temporary XMLs created while materializing one scene."""

    def __init__(self, *paths: str) -> None:
        self._paths = paths
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        for path in self._paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
