"""Backend-owned offline rendering and simulator-neutral playback helpers.

``playback_clock``, ``viser_playback_controls``, and ``viser_utils`` are moved
from UniLab with their public names and semantics unchanged.  They duck-type
the Viser GUI surface and never import Viser, so the base wheel stays
NumPy-only; the optional ``viz`` extra installs Viser for viewers that host a
real server.

``fit_camera`` is the static MuJoCo free-camera fit shared by the downstream
video tools; it is NumPy-only and reproduces MuJoCo's camera convention in
closed form.

``mujoco_geom`` is the MuJoCo geom-to-mesh extraction shared by Viser playback
viewers; it is NumPy-only (no trimesh) and loads the optional ``mujoco``
package lazily inside ``geom_mesh``.

``render_many`` resolves lazily on first access because importing
it probes the GL backend and imports MuJoCo.
"""

from typing import TYPE_CHECKING

from . import fit_camera, mujoco_geom, playback_clock, viser_playback_controls, viser_utils

if TYPE_CHECKING:
    from . import render_many

__all__ = [
    "fit_camera",
    "mujoco_geom",
    "playback_clock",
    "render_many",
    "viser_playback_controls",
    "viser_utils",
]


def __getattr__(name: str):
    """Resolve the MuJoCo-backed offline renderer without importing it eagerly."""
    if name != "render_many":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = importlib.import_module(".render_many", __name__)
    globals()[name] = value
    return value
