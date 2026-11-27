"""Static MuJoCo free-camera framing for world-space point clouds.

``fit_camera`` is the owner-layer home of the camera fit that the UniLab
downstream video tools ``leanretarget`` and ``h2r_bench`` each carried as a
private copy; it unifies them on the superset semantics of the
``leanretarget`` variant (perspective frustum fit with 10% image margins and
an optional header band).  The module is NumPy-only and never imports MuJoCo,
Viser, or an engine SDK: MuJoCo's free-camera convention is reproduced in
closed form, so the result does not depend on the viewer that renders it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _real(name: str, value: object) -> float:
    """Validate one finite real scalar and normalize it to ``float``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_integer(name: str, value: object) -> int:
    """Validate one positive integer and normalize it to ``int``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _real_array(name: str, value: object) -> np.ndarray:
    """Validate a finite real array-like and return it as a float64 array."""
    array = np.asarray(value)
    if array.dtype.kind not in "fi":
        raise TypeError(f"{name} must be an array of real numbers")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def fit_camera(
    points: np.ndarray | Sequence[Sequence[float]],
    *,
    azimuth: float,
    elevation: float,
    fovy: float,
    width: int = 1280,
    height: int = 720,
    header_height: float = 0.0,
    near: float = 0.01,
    lookat: np.ndarray | Sequence[float] | None = None,
) -> tuple[list[float], float]:
    """Fit a static MuJoCo free camera to world-space points, keeping image margins.

    A MuJoCo free camera is defined by ``lookat`` (a world-space target),
    ``distance`` and the angles ``azimuth``/``elevation`` in degrees.  The eye
    sits at ``lookat - distance * forward`` and looks along ``forward``.  With
    ``ca = cos(azimuth)``, ``sa = sin(azimuth)``, ``ce = cos(elevation)`` and
    ``se = sin(elevation)``, the camera axes in world coordinates are
    ``forward = (ce * ca, ce * sa, se)``, ``up = (-se * ca, -se * sa, ce)`` and
    ``right = (sa, -ca, 0)``.  ``azimuth`` is the direction of ``forward`` in
    the xy plane, counter-clockwise from +x; ``elevation`` tilts ``forward``
    up from the xy plane, so negative values look down.  A point ``p`` has
    camera coordinates ``(x, y, z) = (right . p, up . p, forward . p)``: +x is
    image right, +y image up and +z the viewing direction.  The vertical
    half-angle is ``fovy / 2`` and square pixels scale the horizontal one by
    ``width / height``.

    The fit is a perspective one.  Every point keeps 10% of the image free at
    the left, right and bottom edges.  The top edge keeps
    ``max(0.1, header_height / height + 0.03)`` of the image height, so a
    header band additionally reserves its own share plus 3% above the subject.
    Every point also stays at least ``near`` in front of the eye.

    Arguments are validated before any geometry is computed: a wrong type
    raises ``TypeError`` and an out-of-range value raises ``ValueError``.

    The explicit-distance override is not an argument: a distance configured
    by the caller always wins over the fitted value, and both downstream
    callers apply it themselves.  ``leanretarget``'s ``fit_reference_camera``
    is a direct call with identical keyword arguments.  ``h2r_bench``'s
    ``fit_camera(points, camera, width, height, fovy, near)`` calls this
    function with ``header_height=0`` (the default), passing
    ``camera["azimuth"]``, ``camera["elevation"]`` and ``camera["lookat"]``,
    and keeps its result-dict merge and distance override in the caller.

    Args:
        points: Finite world-space points of shape ``(n, 3)`` with ``n >= 1``.
        azimuth: Camera azimuth in degrees.
        elevation: Camera elevation in degrees.
        fovy: Vertical field of view in degrees, in ``(0, 180)``.
        width: Image width in pixels.
        height: Image height in pixels.
        header_height: Image rows reserved at the top, in pixels;
            ``0`` gives the symmetric margins.
        near: Distance in the unit of ``points`` that points keep in front of
            the eye.
        lookat: Optional world-space target; ``None`` fits both the target and
            the distance.

    Returns:
        Tuple ``(lookat, distance)``.  ``lookat`` is always a 3-element list of
        floats: the fitted target, or the normalized ``lookat`` argument.
        ``distance`` is always the fitted value in the unit of ``points``.  With
        ``lookat=None`` the target sits at the mid-depth of the points and the
        eye stays at least ``near`` in front of the nearest one, up to rounding.
        In both branches the eye is the closest one on the viewing axis through
        the target that satisfies every constraint, so at least one of them is
        exactly active.  With an explicit ``lookat``, ``distance`` is signed: it
        is negative when that closest eye lies in front of the target, i.e. when
        the target is farther from the viewer than every point.  The eye still
        sits at ``lookat - distance * forward``.
    """
    azimuth = _real("azimuth", azimuth)
    elevation = _real("elevation", elevation)
    fovy = _real("fovy", fovy)
    if not 0.0 < fovy < 180.0:
        raise ValueError("fovy must be in (0, 180) degrees")
    width = _positive_integer("width", width)
    height = _positive_integer("height", height)
    header_height = _real("header_height", header_height)
    if header_height < 0.0:
        raise ValueError("header_height must be non-negative")
    near = _real("near", near)
    if near <= 0.0:
        raise ValueError("near must be positive")
    cloud = _real_array("points", points)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or cloud.shape[0] < 1:
        raise ValueError(f"points must have shape (n, 3) with n >= 1, got {cloud.shape}")
    target = None if lookat is None else _real_array("lookat", lookat)
    if target is not None and target.shape != (3,):
        raise ValueError(f"lookat must be a 3-vector, got shape {target.shape}")

    azimuth_rad, elevation_rad = np.radians([azimuth, elevation])
    ca, sa = np.cos(azimuth_rad), np.sin(azimuth_rad)
    ce, se = np.cos(elevation_rad), np.sin(elevation_rad)
    basis = np.array([[sa, -ca, 0], [-se * ca, -se * sa, ce], [ce * ca, ce * sa, se]])
    tangent = math.tan(math.radians(fovy) / 2)
    # Leave 10% at each edge; the header needs its own space above the subject.
    horizontal = tangent * width / height * 0.8
    bottom = tangent * 0.8
    top = tangent * (1 - 2 * max(0.1, header_height / height + 0.03))
    if top <= 0:
        raise ValueError("header_height leaves no vertical room for automatic framing")
    projected = cloud @ basis.T

    if target is None:
        x, y, z = projected.T
        # Each side of the frustum bounds the eye's position. Intersect those
        # intervals to find the closest camera that fits every selected point.
        left = np.max(x - horizontal * z)
        right = np.min(x + horizontal * z)
        lower = np.max(y - top * z)
        upper = np.min(y + bottom * z)
        eye_z = min(
            (right - left) / (2 * horizontal),
            (upper - lower) / (top + bottom),
            z.min() - near,
        )
        center = np.array(
            [
                (left + right) / 2,
                (lower + upper + (top - bottom) * eye_z) / 2,
                (z.min() + z.max()) / 2,
            ]
        )
        fitted = [float(value) for value in (center @ basis).tolist()]
        distance = center[2] - eye_z
    else:
        x, y, z = (projected - target @ basis.T).T
        distance = max(
            np.max(np.abs(x) / horizontal - z),
            np.max(y / top - z),
            np.max(-y / bottom - z),
            near - z.min(),
        )
        fitted = [float(value) for value in target]
    return fitted, float(distance)
