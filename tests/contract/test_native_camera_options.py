"""Native renderer limits reject before materialization or worker mutation."""

from __future__ import annotations

import pytest

from unisim import CameraCfg
from unisim.backend.subprocess_ipc.backend import MjcfSubprocessBackend, _normalize_camera_kwargs


@pytest.mark.parametrize(
    "override",
    [
        {"cam_lookat": (1, 2, 3)},
        {"cam_tracking": True},
        {"cam_tracking_env_idx": 1},
        {"cam_tracking_extra_envs": 0},
        {"cam_fov": 45},
    ],
)
@pytest.mark.parametrize("initialized", [False, True])
def test_unsupported_camera_override_rejects_before_any_worker_access(override, initialized):
    # No worker or model is needed to reject an unsupported camera request.
    owner = MjcfSubprocessBackend.__new__(MjcfSubprocessBackend)
    owner._render_config = (True, True) if initialized else None
    with pytest.raises(NotImplementedError, match=next(iter(override))):
        owner.init_renderer(headless=True, capture=True, camera_kwargs=override)


def test_native_capture_camera_preserves_supported_angles_and_distance():
    assert _normalize_camera_kwargs(
        CameraCfg(cam_distance=3, cam_elevation=-35, cam_azimuth=120)
    ) == {"distance": 3, "elevation_deg": 35, "azimuth_deg": 120}


@pytest.mark.parametrize("override", [{"cam_distance": 3}, {"cam_elevation": -45},
                                      {"cam_azimuth": 10}])
def test_interactive_viewer_rejects_ignored_spherical_options(override):
    owner = MjcfSubprocessBackend.__new__(MjcfSubprocessBackend)
    with pytest.raises(NotImplementedError, match="interactive viewers"):
        owner.init_renderer(camera_kwargs=override)
