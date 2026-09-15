from __future__ import annotations

__all__ = [
    "DrakeBackend",
    "run_drake_playback",
]


def __getattr__(name: str):
    if name == "DrakeBackend":
        from .backend import DrakeBackend

        return DrakeBackend
    if name == "run_drake_playback":
        from .playback import run_drake_playback

        return run_drake_playback
    raise AttributeError(name)
