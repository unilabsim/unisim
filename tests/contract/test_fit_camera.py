"""Tests for the static MuJoCo free-camera fit over world-space point clouds."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from unisim.visualization.fit_camera import fit_camera

# The golden outputs below were produced by the downstream ``leanretarget``
# ``fit_reference_camera`` reference on the same arguments.  The fit keeps that
# function's math and operation order, so only floating-point noise may differ.
_GOLDEN = {"rtol": 1e-12, "atol": 1e-12}
# The fit is exact up to rounding; loose clouds lose a few digits to cancellation.
_TIGHT = 1e-9
_ASPECTS = [(1280, 720), (720, 1280), (1920, 1080), (640, 480)]


def _box_points() -> np.ndarray:
    """Return the box of the ``h2r_bench`` camera-fit scenario."""
    return np.array([[x, y, z] for x in [-1, 1] for y in [-0.4, 0.4] for z in [0.0, 0.7]])


def _two_boxes_points() -> np.ndarray:
    """Return two offset boxes with different extents."""
    corners = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64
    )
    return np.concatenate((corners * [0.2, 0.1, 0.05], corners * [0.1, 0.3, 0.2] + [0.4, 0.2, 0.5]))


def _frame(azimuth: float, elevation: float) -> np.ndarray:
    """Return MuJoCo's free-camera axes (right, up, forward) as rows."""
    az, el = np.radians([azimuth, elevation])
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    return np.array([[sa, -ca, 0], [-se * ca, -se * sa, ce], [ce * ca, ce * sa, se]])


def _vertical_window(header_height: float, height: int) -> tuple[float, float]:
    """Return the bottom and top limits of the margined frame in half-image units."""
    return 0.8, 1 - 2 * max(0.1, header_height / height + 0.03)


def _limits(
    points: np.ndarray,
    lookat: list[float],
    distance: float,
    *,
    frame: np.ndarray,
    fovy: float,
    width: int,
    height: int,
    near: float,
    top: float,
) -> np.ndarray:
    """Return how much of each margined limit the points consume (1.0 is active).

    The order is left/right, bottom, top and near plane.  Image coordinates are
    in half-image units, and every value grows as the eye moves toward the points.
    """
    local = (np.asarray(points, dtype=np.float64) - np.asarray(lookat)) @ frame.T
    depth = local[:, 2] + distance
    tangent = math.tan(math.radians(fovy) / 2)
    horizontal = local[:, 0] / (depth * tangent * width / height)
    vertical = local[:, 1] / (depth * tangent)
    return np.array(
        [
            np.max(np.abs(horizontal)) / 0.8,
            -np.min(vertical) / 0.8,
            np.max(vertical) / top,
            near / np.min(depth),
        ]
    )


def _assert_framed(
    points: np.ndarray,
    result: tuple[list[float], float],
    *,
    frame: np.ndarray,
    fovy: float,
    width: int,
    height: int,
    header_height: float,
    near: float,
    tolerance: float = _TIGHT,
) -> np.ndarray:
    """Assert the points fit the margined frame exactly and return the limit usage."""
    lookat, distance = result
    bottom, top = _vertical_window(header_height, height)
    assert bottom == 0.8 and top <= 0.8
    shape = {"fovy": fovy, "width": width, "height": height, "near": near, "top": top}
    usage = _limits(points, lookat, distance, frame=frame, **shape)
    assert np.all(usage <= 1.0 + tolerance), usage
    assert np.max(usage) >= 1.0 - tolerance, usage
    # The fit is the closest admissible eye: moving it toward the points by a
    # millionth of the largest depth breaks a limit.  The step is absolute because
    # the fitted distance is signed for an explicit target.
    local = (np.asarray(points, dtype=np.float64) - np.asarray(lookat)) @ frame.T
    depths = local[:, 2] + distance
    step = min(1e-6 * float(np.max(depths)), 0.5 * float(np.min(depths)))
    closer = _limits(points, lookat, distance - step, frame=frame, **shape)
    assert np.max(closer) > 1.0 + 5e-7, closer
    return usage


def _h2r_symmetric_fit(
    points: np.ndarray,
    azimuth: float,
    elevation: float,
    fovy: float,
    width: int,
    height: int,
    near: float,
    lookat: list[float] | None,
) -> tuple[list[float], float]:
    """Reproduce ``h2r_bench``'s symmetric camera fit with its own formulas."""
    az, el = np.radians([azimuth, elevation])
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    basis = np.array([[sa, -ca, 0], [-se * ca, -se * sa, ce], [ce * ca, ce * sa, se]])
    projected = np.asarray(points) @ basis.T
    vertical = math.tan(math.radians(fovy) / 2) * 0.8
    horizontal = vertical * width / height
    if lookat is None:
        x, y, z = projected.T
        left, right = np.max(x - horizontal * z), np.min(x + horizontal * z)
        lower, upper = np.max(y - vertical * z), np.min(y + vertical * z)
        eye = min(
            (right - left) / (2 * horizontal),
            (upper - lower) / (2 * vertical),
            z.min() - near,
        )
        center = np.array([(left + right) / 2, (lower + upper) / 2, (z.min() + z.max()) / 2])
        return (center @ basis).tolist(), float(center[2] - eye)
    x, y, z = (projected - np.asarray(lookat) @ basis.T).T
    distance = max(
        np.max(np.abs(x) / horizontal - z), np.max(np.abs(y) / vertical - z), near - z.min()
    )
    return list(lookat), float(distance)


def _random_cloud(rng: np.random.Generator) -> np.ndarray:
    """Return a cloud with varied size, structure, anisotropy and world offset."""
    size = int(rng.integers(1, 601))
    scale = 10.0 ** rng.uniform(-2.0, 2.0, 3)
    centre = rng.uniform(-20.0, 20.0, 3)
    kind = int(rng.integers(0, 5))
    if kind == 0:
        cloud = rng.normal(size=(size, 3)) * scale
    elif kind == 1:
        cloud = rng.uniform(-1.0, 1.0, (size, 3)) * scale
    elif kind == 2:  # planar patch
        cloud = rng.uniform(-1.0, 1.0, (size, 3)) * scale
        cloud[:, int(rng.integers(0, 3))] = 0.0
    elif kind == 3:  # collinear samples
        cloud = rng.uniform(-1.0, 1.0, (size, 1)) * rng.normal(size=(1, 3))
    else:  # clustered blobs, which yields duplicated points
        blobs = rng.normal(size=(3, 3)) * scale
        cloud = blobs[rng.integers(0, 3, size)]
    return cloud + centre


def test_fit_camera_matches_golden_values() -> None:
    """Verify representative fits reproduce the downstream reference outputs."""
    clouds = {
        "box": _box_points(),
        "two boxes": _two_boxes_points(),
        "single": np.array([[0.3, -0.2, 0.7]]),
        "line": np.array([[0.0, 0.0, z] for z in np.linspace(-0.5, 1.5, 9)]),
    }
    # Parallel tables: cloud name, (elevation, azimuth, fovy, width, height, header,
    # near, lookat), the fitted lookat and the fitted distance.
    names = [
        "box",
        "box",
        "box",
        "box",
        "single",
        "single",
        "two boxes",
        "two boxes",
        "two boxes",
        "line",
    ]
    arguments = [
        (-30, 90, 45, 1280, 720, 0, 0.01, None),
        (-30, 90, 45, 1280, 720, 106, 0.01, None),
        (-65, 35, 45, 720, 1280, 106, 0.01, None),
        (-30, 90, 45, 1280, 720, 106, 0.01, [0.2, -0.1, 0.8]),
        (-30, 90, 45, 1280, 720, 0, 0.01, None),
        (-30, 90, 45, 1280, 720, 106, 0.01, None),
        (-30, 90, 45, 1280, 720, 106, 0.01, None),
        (-65, 35, 45, 720, 1280, 106, 0.01, None),
        (20, -150, 60, 1920, 1080, 0, 0.004, None),
        (-30, 90, 45, 1280, 720, 106, 0.02, None),
    ]
    expected_lookat = [
        [-1.7390085857428008e-18, -0.02840016545102745, 0.3008094704954582],
        [6.018353912771548e-19, 0.009828717826171209, 0.3670238386481865],
        [-0.036832419355193345, 0.03092068873118749, 0.34420101047936613],
        [0.2, -0.1, 0.8],
        [0.3, -0.19999999999999998, 0.7000000000000001],
        [0.3, -0.1998400675411948, 0.70027701114443],
        [0.2457293566373375, 0.13888431174519225, 0.39414452280313245],
        [0.27426439283837745, 0.10328889175673793, 0.3618944024795531],
        [0.2560013731624925, 0.1927724998474618, 0.41728940615507965],
        [7.363506453028186e-18, 0.12025518636320194, 0.7082880926547298],
    ]
    expected_distance = [
        2.2189040725573586,
        2.2189040725573586,
        5.152272618092995,
        2.403242939256315,
        0.010000000000000009,
        0.010000000000000009,
        1.5060114196174987,
        1.8655807832344238,
        1.0916191705672083,
        2.8392670986294006,
    ]
    tables = (names, arguments, expected_lookat, expected_distance)
    for name, args, exp_lookat, exp_distance in zip(*tables, strict=True):
        elevation, azimuth, fovy, width, height, header, near, lookat = args
        result = fit_camera(
            clouds[name],
            azimuth=azimuth,
            elevation=elevation,
            fovy=fovy,
            width=width,
            height=height,
            header_height=header,
            near=near,
            lookat=lookat,
        )
        case = (name, args)
        assert type(result[0]) is list and len(result[0]) == 3, case
        assert all(type(value) is float for value in result[0]), case
        assert type(result[1]) is float, case
        np.testing.assert_allclose(result[0], exp_lookat, **_GOLDEN, err_msg=repr(case))
        np.testing.assert_allclose(result[1], exp_distance, **_GOLDEN, err_msg=repr(case))


def test_fit_camera_contains_points_with_asymmetric_margins_and_is_tight() -> None:
    """Verify every point fits the margined frame and one limit is exactly active."""
    rng = np.random.default_rng(20261127)
    active = np.zeros(4, dtype=int)
    signed = 0
    for _ in range(600):
        points = _random_cloud(rng)
        width, height = _ASPECTS[int(rng.integers(0, len(_ASPECTS)))]
        azimuth = float(rng.uniform(-360.0, 360.0))
        elevation = float(rng.uniform(-89.0, 89.0))
        fovy = float(rng.uniform(20.0, 90.0))
        near = float(10.0 ** rng.uniform(-3.0, 0.0))
        header = 0.0 if rng.uniform() < 0.5 else float(rng.uniform(0.0, 0.25 * height))
        lookat = None if rng.uniform() < 0.5 else points[rng.integers(0, len(points))] + 0.1
        result = fit_camera(
            points,
            azimuth=azimuth,
            elevation=elevation,
            fovy=fovy,
            width=width,
            height=height,
            header_height=header,
            near=near,
            lookat=lookat,
        )
        usage = _assert_framed(
            points,
            result,
            frame=_frame(azimuth, elevation),
            fovy=fovy,
            width=width,
            height=height,
            header_height=header,
            near=near,
        )
        active[np.abs(usage - 1.0) <= _TIGHT] += 1
        if lookat is None:
            # A fitted target is never in front of its own eye.  The bound is a
            # rounding slack of the coordinate magnitude, not of ``near``.
            assert result[1] > 0.0 and result[1] >= near * (1.0 - 1e-9)
        else:
            # An explicit target farther from the viewer than every point yields
            # the signed distance documented by ``fit_camera``.
            signed += result[1] < 0.0
    # Left/right, bottom, top and near plane are each the active limit sometimes,
    # and both signs of an explicitly targeted distance are exercised.
    assert np.all(active > 0), active
    assert signed > 0


def test_fit_camera_collapses_to_the_symmetric_h2r_bench_margins_without_header() -> None:
    """Verify ``header_height=0`` reproduces the symmetric ``h2r_bench`` fit."""
    rng = np.random.default_rng(106)
    for index in range(200):
        points = _random_cloud(rng)
        width, height = _ASPECTS[index % len(_ASPECTS)]
        azimuth = float(rng.uniform(-180.0, 180.0))
        elevation = float(rng.uniform(-80.0, 80.0))
        fovy = float(rng.uniform(20.0, 90.0))
        near = float(10.0 ** rng.uniform(-3.0, 0.0))
        lookat = None if rng.uniform() < 0.5 else rng.uniform(-1.0, 1.0, 3).tolist()
        result = fit_camera(
            points,
            azimuth=azimuth,
            elevation=elevation,
            fovy=fovy,
            width=width,
            height=height,
            near=near,
            lookat=lookat,
        )
        expected_lookat, expected_distance = _h2r_symmetric_fit(
            points, azimuth, elevation, fovy, width, height, near, lookat
        )
        # ``h2r_bench`` associates ``horizontal`` differently, so compare below the
        # last bits; the two fits are the same function of the inputs.
        np.testing.assert_allclose(result[0], expected_lookat, rtol=1e-12, atol=1e-11)
        np.testing.assert_allclose(result[1], expected_distance, rtol=1e-12, atol=1e-11)
        bottom, top = _vertical_window(0.0, height)
        assert top == bottom
        _assert_framed(
            points,
            result,
            frame=_frame(azimuth, elevation),
            fovy=fovy,
            width=width,
            height=height,
            header_height=0.0,
            near=near,
        )


def test_fit_camera_symmetric_window_is_centered_for_a_planar_cloud() -> None:
    """Verify a depth-flat cloud fills the same room above and below its target."""
    # A plane perpendicular to the viewing direction (azimuth 90, elevation 0).
    tangent = math.tan(math.radians(22.5))
    horizontal = 0.8 * tangent * 1280 / 720
    vertical = 0.8 * tangent
    # A wide, low cloud: the lateral extent needs the deeper eye, and the vertical
    # window is centred although it is not full.
    points = np.array([[x, 0.3, z] for x in np.linspace(-1.0, 2.0, 7) for z in (0.0, 0.5)])
    lookat, distance = fit_camera(points, azimuth=90.0, elevation=0.0, fovy=45.0)
    offset = (points[:, 2] - lookat[2]) / (0.3 - (lookat[1] - distance))
    assert np.max(offset) == pytest.approx(-np.min(offset), rel=1e-12, abs=1e-15)
    assert distance == pytest.approx(max(1.5 / horizontal, 0.5 / (2 * vertical)), rel=1e-12)
    np.testing.assert_allclose(lookat, [0.5, 0.3, 0.25], rtol=0.0, atol=1e-12)
    # A tall, narrow cloud: the vertical extent needs the deeper eye, so both
    # vertical limits are exactly full and symmetric.
    points = np.array([[x, 0.3, z] for x in np.linspace(-0.25, 0.25, 7) for z in (0.0, 0.5)])
    lookat, distance = fit_camera(points, azimuth=90.0, elevation=0.0, fovy=45.0)
    offset = (points[:, 2] - lookat[2]) / (0.3 - (lookat[1] - distance))
    assert np.max(offset) == pytest.approx(-np.min(offset), rel=1e-12, abs=1e-15)
    assert np.max(offset) == pytest.approx(vertical, rel=1e-12)
    assert distance == pytest.approx(max(0.25 / horizontal, 0.5 / (2 * vertical)), rel=1e-12)
    np.testing.assert_allclose(lookat, [0.0, 0.3, 0.25], rtol=0.0, atol=1e-12)


def test_fit_camera_explicit_lookat_is_returned_and_only_distance_is_fitted() -> None:
    """Verify a given target passes through normalized while the distance is fitted."""
    points = _two_boxes_points()
    for lookat in ([0.2, -0.1, 0.8], (0.2, -0.1, 0.8), np.array([0.2, -0.1, 0.8]), [1, 2, 3]):
        result = fit_camera(
            points,
            azimuth=35.0,
            elevation=-65.0,
            fovy=45.0,
            header_height=106,
            lookat=lookat,
        )
        assert result[0] == [float(value) for value in lookat]
        assert type(result[0]) is list and all(type(value) is float for value in result[0])
        assert type(result[1]) is float
        _assert_framed(
            points,
            result,
            frame=_frame(35.0, -65.0),
            fovy=45.0,
            width=1280,
            height=720,
            header_height=106.0,
            near=0.01,
        )
    # The fitted target centres the cloud and therefore needs the closer eye.
    fitted = fit_camera(points, azimuth=35.0, elevation=-65.0, fovy=45.0, header_height=106)
    fixed = fit_camera(
        points, azimuth=35.0, elevation=-65.0, fovy=45.0, header_height=106, lookat=[1, 2, 3]
    )
    assert fitted[1] < fixed[1]


def test_fit_camera_signed_distance_for_a_target_in_front_of_the_points() -> None:
    """Verify a target between the viewer and the points yields a signed distance."""
    shape = {"fovy": 45.0, "width": 1280, "height": 720, "header_height": 0.0}
    frame = _frame(90.0, 0.0)
    point = np.array([[0.0, 0.0, 0.0]])
    # The camera looks along +y from -y.  The target lies between the viewer and
    # the point, farther from the point than the eye needs to be, so the closest
    # admissible eye is in front of the target and the distance is negative.
    lookat, distance = fit_camera(
        point, azimuth=90.0, elevation=0.0, fovy=45.0, near=0.01, lookat=[0.0, -0.5, 0.0]
    )
    assert lookat == [0.0, -0.5, 0.0]
    assert distance == pytest.approx(-0.49, rel=1e-12)
    usage = _assert_framed(point, (lookat, distance), frame=frame, near=0.01, **shape)
    assert np.abs(usage[3] - 1.0) <= _TIGHT, usage
    # The eye is still ``lookat - distance * forward``: on the near plane.
    eye = np.asarray(lookat) - distance * np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(eye, [0.0, -0.01, 0.0], rtol=0.0, atol=1e-12)
    # Behind the point, the same target gives the usual positive distance.
    behind = fit_camera(
        point, azimuth=90.0, elevation=0.0, fovy=45.0, near=0.01, lookat=[0.0, 0.5, 0.0]
    )
    assert behind[1] == pytest.approx(0.51, rel=1e-12)
    _assert_framed(point, behind, frame=frame, near=0.01, **shape)


def test_fit_camera_header_shifts_the_vertical_window_downwards() -> None:
    """Verify the header shrinks the top limit and moves the subject down in frame."""
    # A tall, narrow plane perpendicular to the viewing direction: only its
    # vertical extent limits the eye, which is the limit the header acts on.
    points = np.array([[x, 0.3, z] for x in np.linspace(-0.25, 0.25, 5) for z in (0.0, 0.5)])
    tangent = math.tan(math.radians(22.5))
    horizontal = 0.8 * tangent * 1280 / 720
    frame = _frame(90.0, 0.0)
    plain = fit_camera(points, azimuth=90.0, elevation=0.0, fovy=45.0)
    with_header = fit_camera(points, azimuth=90.0, elevation=0.0, fovy=45.0, header_height=106)
    bottom, top = _vertical_window(106.0, 720)
    assert top < bottom == 0.8
    shape = {"frame": frame, "fovy": 45.0, "width": 1280, "height": 720, "near": 0.01}
    plain_usage = _assert_framed(points, plain, header_height=0.0, **shape)
    header_usage = _assert_framed(points, with_header, header_height=106.0, **shape)
    # The same vertical extent fills the symmetric window and the asymmetric one.
    assert np.all(np.abs(plain_usage[1:3] - 1.0) <= _TIGHT), plain_usage
    assert np.all(np.abs(header_usage[1:3] - 1.0) <= _TIGHT), header_usage
    assert plain_usage[0] == pytest.approx(0.25 / horizontal / plain[1], rel=1e-12)
    extent = points[:, 2].max() - points[:, 2].min()
    assert plain[1] == pytest.approx(extent / (2 * 0.8 * tangent), rel=1e-12)
    assert with_header[1] == pytest.approx(extent / ((bottom + top) * tangent), rel=1e-12)
    # The target rises above the cloud centre so the subject sits below the band.
    assert with_header[0][2] == pytest.approx(
        points[:, 2].min() + bottom * tangent * with_header[1], rel=1e-12
    )
    assert np.mean(points[:, 2]) == pytest.approx(plain[0][2], rel=1e-12)
    assert np.mean(points[:, 2]) < with_header[0][2]
    # The eye backs away; lateral framing is untouched.
    assert with_header[1] > plain[1]
    np.testing.assert_allclose(with_header[0][:2], plain[0][:2], rtol=0.0, atol=1e-12)


def test_fit_camera_handles_single_collinear_and_planar_clouds() -> None:
    """Verify degenerate clouds stay finite, contained and tight."""
    single = np.array([[0.3, -0.2, 0.7]])
    shape = {"fovy": 45.0, "width": 1280, "height": 720}
    lookat, distance = fit_camera(single, azimuth=90.0, elevation=-30.0, fovy=45.0)
    np.testing.assert_allclose(lookat, single[0], rtol=0.0, atol=1e-13)
    assert distance == pytest.approx(0.01, rel=1e-12)
    assert np.isfinite(lookat).all() and math.isfinite(distance)

    # A single point sits on the near plane; a header lifts the target so the point
    # lands below the image centre by half the window's asymmetry.
    with_header = fit_camera(single, azimuth=90.0, elevation=-30.0, fovy=45.0, header_height=106)
    bottom, top = _vertical_window(106.0, 720)
    usage = _assert_framed(
        single,
        with_header,
        frame=_frame(90.0, -30.0),
        header_height=106.0,
        near=0.01,
        **shape,
    )
    assert np.abs(usage[3] - 1.0) <= _TIGHT and np.abs(usage[0]) <= _TIGHT, usage
    offset = (bottom - top) / 2 * math.tan(math.radians(22.5)) * 0.01
    assert np.linalg.norm(np.asarray(with_header[0]) - np.asarray(lookat)) == pytest.approx(
        offset, rel=1e-6
    )

    # Collinear along the viewing direction: the near plane limits the eye and the
    # target is the midpoint of the depth range.
    line = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = fit_camera(line, azimuth=0.0, elevation=-90.0, fovy=45.0, near=0.05)
    np.testing.assert_allclose(result[0], [0.0, 0.0, 0.5], rtol=0.0, atol=1e-13)
    assert result[1] == pytest.approx(0.55, rel=1e-12)
    usage = _assert_framed(
        line, result, frame=_frame(0.0, -90.0), header_height=0.0, near=0.05, **shape
    )
    assert np.abs(usage[3] - 1.0) <= _TIGHT, usage

    # Collinear across the viewing direction: both ends touch the lateral limits.
    across = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = fit_camera(across, azimuth=90.0, elevation=0.0, fovy=45.0)
    np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], rtol=0.0, atol=1e-13)
    usage = _assert_framed(
        across, result, frame=_frame(90.0, 0.0), header_height=0.0, near=0.01, **shape
    )
    assert np.abs(usage[0] - 1.0) <= _TIGHT, usage

    # A planar cloud seen edge-on, i.e. lying in a plane containing the view axis.
    edge_on = np.array([[x, y, 0.0] for x in (-1.0, 1.0) for y in (-0.5, 0.5)])
    result = fit_camera(edge_on, azimuth=0.0, elevation=0.0, fovy=45.0)
    np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], rtol=0.0, atol=1e-13)
    usage = _assert_framed(
        edge_on, result, frame=_frame(0.0, 0.0), header_height=0.0, near=0.01, **shape
    )
    assert np.abs(usage[0] - 1.0) <= _TIGHT, usage


def test_fit_camera_follows_the_mujoco_free_camera_convention() -> None:
    """Verify the fitted frame matches hand-derived MuJoCo free-camera axes."""
    # Literal axes from MuJoCo's convention: the azimuth is the compass direction
    # of the viewing direction in the xy plane and the elevation tilts it up from
    # that plane; ``right`` is ``forward x up``.
    half = math.sqrt(0.5)
    literal = [
        (0.0, 0.0, [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        (90.0, 0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        (180.0, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),
        (270.0, 0.0, [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
        (0.0, -90.0, [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        (90.0, -90.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
        (90.0, -30.0, [1.0, 0.0, 0.0], [0.0, 0.5, math.sqrt(3) / 2], [0.0, math.sqrt(3) / 2, -0.5]),
        (45.0, 0.0, [half, -half, 0.0], [0.0, 0.0, 1.0], [half, half, 0.0]),
    ]
    # The formula-derived frame used by the other tests is this documented convention.
    for azimuth, elevation, right, up, forward in literal:
        np.testing.assert_allclose(
            _frame(azimuth, elevation), [right, up, forward], rtol=0.0, atol=1e-15
        )

    # An asymmetric 3-D box framed through the literal axes fits exactly for every
    # heading, with and without a header (the header makes ``up`` observable).
    box = np.array([[x, y, z] for x in (-1.0, 2.0) for y in (-0.5, 1.5) for z in (0.0, 0.7)])
    for azimuth, elevation, right, up, forward in literal:
        for header in (0.0, 106.0):
            result = fit_camera(
                box, azimuth=azimuth, elevation=elevation, fovy=45.0, header_height=header
            )
            _assert_framed(
                box,
                result,
                frame=np.array([right, up, forward]),
                fovy=45.0,
                width=1280,
                height=720,
                header_height=header,
                near=0.01,
            )

    tangent = math.tan(math.radians(22.5))
    horizontal = 0.8 * tangent * 1280 / 720
    # A plane perpendicular to the viewing direction (azimuth 90, elevation 0):
    # world +x spans the image width and world +z its height.
    flat = np.array([[x, 0.3, z] for x in np.linspace(-1.0, 2.0, 7) for z in (0.0, 0.5)])
    lookat, distance = fit_camera(flat, azimuth=90.0, elevation=0.0, fovy=45.0)
    assert distance == pytest.approx(max(1.5 / horizontal, 0.5 / (2 * 0.8 * tangent)), rel=1e-12)
    np.testing.assert_allclose(lookat, [0.5, 0.3, 0.25], rtol=0.0, atol=1e-12)

    # A square ground footprint seen from straight above (elevation -90).  Only
    # its extent along ``up`` limits the eye, so the header lifts the target
    # along ``up`` by a closed-form amount; ``up`` is the compass direction of
    # the azimuth, which the literal vectors below pin for each heading.
    ground = np.array([[x, y, 0.3] for x in np.linspace(-0.25, 0.25, 5) for y in (-0.25, 0.25)])
    bottom, top = _vertical_window(106.0, 720)
    limited = 0.5 / ((bottom + top) * tangent)
    lift = -0.25 + bottom * tangent * limited
    assert lift > 0.0 and 0.25 / horizontal < limited
    headings = ((0.0, 1.0, 0.0), (90.0, 0.0, 1.0), (180.0, -1.0, 0.0), (270.0, 0.0, -1.0))
    for azimuth, up_x, up_y in headings:
        lookat, distance = fit_camera(
            ground, azimuth=azimuth, elevation=-90.0, fovy=45.0, header_height=106
        )
        assert distance == pytest.approx(limited, rel=1e-12)
        np.testing.assert_allclose(lookat, [lift * up_x, lift * up_y, 0.3], **_GOLDEN)


@pytest.mark.parametrize(
    ("name", "value", "error", "match"),
    [
        ("points", None, TypeError, "points must be an array of real numbers"),
        ("points", [[0.0, 0.0, complex(0.0, 0.0)]], TypeError, "points must be an array of real"),
        ("points", np.zeros((2, 3), dtype=bool), TypeError, "points must be an array of real"),
        ("points", [["0.0", "0.0", "0.0"]], TypeError, "points must be an array of real"),
        ("points", [[0.0, 0.0, np.nan]], ValueError, "points must be finite"),
        ("points", [[0.0, 0.0, np.inf]], ValueError, "points must be finite"),
        ("points", [[0.0, 0.0], [1.0, 1.0]], ValueError, r"shape \(n, 3\) with n >= 1"),
        ("points", [[0.0, 0.0, 0.0, 0.0]], ValueError, r"shape \(n, 3\) with n >= 1"),
        ("points", [0.0, 0.0, 0.0], ValueError, r"shape \(n, 3\) with n >= 1"),
        ("points", np.zeros((0, 3)), ValueError, r"shape \(n, 3\) with n >= 1"),
        ("points", np.zeros((4, 1, 3)), ValueError, r"shape \(n, 3\) with n >= 1"),
        ("points", [[0.0, 0.0], [1.0, 1.0, 2.0]], ValueError, None),
        ("azimuth", "90", TypeError, "azimuth must be a real number"),
        ("azimuth", True, TypeError, "azimuth must be a real number"),
        ("azimuth", np.bool_(True), TypeError, "azimuth must be a real number"),
        ("azimuth", None, TypeError, "azimuth must be a real number"),
        ("azimuth", 90 + 0j, TypeError, "azimuth must be a real number"),
        ("azimuth", np.nan, ValueError, "azimuth must be finite"),
        ("azimuth", np.inf, ValueError, "azimuth must be finite"),
        ("elevation", "-30", TypeError, "elevation must be a real number"),
        ("elevation", False, TypeError, "elevation must be a real number"),
        ("elevation", np.nan, ValueError, "elevation must be finite"),
        ("fovy", "45", TypeError, "fovy must be a real number"),
        ("fovy", True, TypeError, "fovy must be a real number"),
        ("fovy", np.nan, ValueError, "fovy must be finite"),
        ("fovy", 0, ValueError, r"fovy must be in \(0, 180\) degrees"),
        ("fovy", 0.0, ValueError, r"fovy must be in \(0, 180\) degrees"),
        ("fovy", -10.0, ValueError, r"fovy must be in \(0, 180\) degrees"),
        ("fovy", 180, ValueError, r"fovy must be in \(0, 180\) degrees"),
        ("fovy", 180.0, ValueError, r"fovy must be in \(0, 180\) degrees"),
        ("fovy", 270.0, ValueError, r"fovy must be in \(0, 180\) degrees"),
        ("width", 0, ValueError, "width must be positive"),
        ("width", -1280, ValueError, "width must be positive"),
        ("width", 1280.0, TypeError, "width must be an integer"),
        ("width", True, TypeError, "width must be an integer"),
        ("width", np.bool_(True), TypeError, "width must be an integer"),
        ("width", "1280", TypeError, "width must be an integer"),
        ("width", None, TypeError, "width must be an integer"),
        ("height", 0, ValueError, "height must be positive"),
        ("height", -720, ValueError, "height must be positive"),
        ("height", 720.0, TypeError, "height must be an integer"),
        ("height", False, TypeError, "height must be an integer"),
        ("height", None, TypeError, "height must be an integer"),
        ("header_height", -1, ValueError, "header_height must be non-negative"),
        ("header_height", -0.5, ValueError, "header_height must be non-negative"),
        ("header_height", np.nan, ValueError, "header_height must be finite"),
        ("header_height", np.inf, ValueError, "header_height must be finite"),
        ("header_height", "0", TypeError, "header_height must be a real number"),
        ("header_height", True, TypeError, "header_height must be a real number"),
        ("header_height", 339, ValueError, "header_height leaves no vertical room"),
        ("header_height", 360, ValueError, "header_height leaves no vertical room"),
        ("header_height", 720, ValueError, "header_height leaves no vertical room"),
        ("near", 0, ValueError, "near must be positive"),
        ("near", 0.0, ValueError, "near must be positive"),
        ("near", -0.01, ValueError, "near must be positive"),
        ("near", np.inf, ValueError, "near must be finite"),
        ("near", np.nan, ValueError, "near must be finite"),
        ("near", "0.01", TypeError, "near must be a real number"),
        ("near", True, TypeError, "near must be a real number"),
        ("lookat", [0.0, 0.0], ValueError, "lookat must be a 3-vector"),
        ("lookat", [0.0, 0.0, 0.0, 0.0], ValueError, "lookat must be a 3-vector"),
        ("lookat", 0.0, ValueError, "lookat must be a 3-vector"),
        ("lookat", np.zeros((1, 3)), ValueError, "lookat must be a 3-vector"),
        ("lookat", [0.0, 0.0, np.nan], ValueError, "lookat must be finite"),
        ("lookat", [0.0, 0.0, np.inf], ValueError, "lookat must be finite"),
        ("lookat", "000", TypeError, "lookat must be an array of real"),
        ("lookat", [0.0, 0.0, 1j], TypeError, "lookat must be an array of real"),
        ("lookat", [True, False, True], TypeError, "lookat must be an array of real"),
    ],
)
def test_fit_camera_rejects_invalid_inputs(
    name: str, value: object, error: type[Exception], match: str | None
) -> None:
    """Verify malformed or out-of-range arguments fail closed before any geometry."""
    arguments: dict[str, object] = {
        "points": [[0.0, 0.0, 0.0], [1.0, 0.5, 0.2]],
        "azimuth": 90.0,
        "elevation": -30.0,
        "fovy": 45.0,
    }
    arguments[name] = value
    with pytest.raises(error, match=match):
        fit_camera(**arguments)


def test_fit_camera_is_deterministic_and_input_agnostic() -> None:
    """Verify equal inputs give equal outputs whatever their container or order."""
    rng = np.random.default_rng(7)
    points = rng.uniform(-1.0, 1.0, (400, 3))
    pristine = points.copy()

    def fit(cloud: object, **overrides: object) -> tuple[list[float], float]:
        options: dict[str, object] = {
            "azimuth": 90.0,
            "elevation": -30.0,
            "fovy": 45.0,
            "header_height": 106.0,
            "near": 0.01,
        }
        options.update(overrides)
        return fit_camera(cloud, **options)

    baseline = fit(points)
    assert fit(points) == baseline

    # Point order does not matter.
    shuffled = fit(points[rng.permutation(len(points))])
    np.testing.assert_allclose(shuffled[0], baseline[0], **_GOLDEN)
    np.testing.assert_allclose(shuffled[1], baseline[1], **_GOLDEN)

    # Plain sequences and read-only arrays carry the same values.
    frozen = points.copy()
    frozen.setflags(write=False)
    for cloud in (points.tolist(), tuple(map(tuple, points.tolist())), frozen):
        assert fit(cloud) == baseline

    # Integer and single-precision scalars are accepted at their exact values.
    whole = {"azimuth": 90, "elevation": -30, "fovy": 45, "header_height": 106, "near": 1}
    assert fit(points, **whole) == fit(
        points, azimuth=90.0, elevation=-30.0, fovy=45.0, header_height=106.0, near=1.0
    )
    single_precision = {
        "azimuth": np.float32(90),
        "elevation": np.float32(-30),
        "fovy": np.float32(45),
        "near": np.float32(0.01),
    }
    widened = {name: float(value) for name, value in single_precision.items()}
    assert fit(points, **single_precision) == fit(points, **widened)
    wide = fit(points, width=np.int64(1280), height=np.int64(720))
    assert wide == fit(points, width=1280, height=720)

    # Integer-valued coordinates give the same fit as float64 ones.
    grid = np.round(points * 20).astype(np.int64)
    assert fit(grid) == fit(grid.astype(np.float64))

    # An explicit target is normalized into a new list and never aliases its input.
    lookat = [0.25, -0.5, 0.125]
    result = fit(points, lookat=lookat)
    assert result[0] == lookat and result[0] is not lookat
    assert fit(points, lookat=np.array(lookat))[0] == lookat
    assert fit(points, lookat=(1, 2, 3))[0] == [1.0, 2.0, 3.0]
    assert lookat == [0.25, -0.5, 0.125]

    # No input is modified.
    np.testing.assert_array_equal(points, pristine)
    np.testing.assert_array_equal(grid, np.round(points * 20).astype(np.int64))


def test_fit_camera_loads_without_engine_or_viewer_sdks() -> None:
    """Verify the module imports and runs without MuJoCo, Viser, or any optional SDK."""
    code = """
import importlib.abc
import sys

blocked = {'unilab', 'viser', 'mujoco', 'mjbatch', 'motrixsim', 'pydrake', 'warp',
           'newton', 'genesis', 'superdex', 'isaacgym', 'isaacsim', 'torch', 'omni',
           'scipy', 'trimesh'}

class NoOptionalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in blocked, fullname
        return None

sys.meta_path.insert(0, NoOptionalRuntime())
from unisim.visualization.fit_camera import fit_camera
points = [[0.0, 0.0, 0.0], [1.0, 0.5, 0.2]]
lookat, distance = fit_camera(points, azimuth=90, elevation=-30, fovy=45)
assert len(lookat) == 3 and distance > 0.0, (lookat, distance)
import unisim.visualization
assert 'unisim.visualization.render_many' not in sys.modules
assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or result.stdout
