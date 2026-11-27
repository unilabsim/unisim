"""DebugPrimitive-to-mjvScene conversion shared by offline and interactive paths.

This is the single ``DebugPrimitive`` conversion shared by the offline render
workers (:mod:`unisim.visualization.render_many`) and interactive viewers
(``mujoco.viewer.launch_passive`` with ``viewer.user_scn``).  The module is
importable without MuJoCo, so an interactive viewer never triggers the offline
renderer's GL probe; MuJoCo itself is imported lazily when a primitive is
converted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from unisim.backend.base import DebugPrimitive
from unisim.optional import OptionalDependencyError


def _import_mujoco() -> Any:
    """Return the lazily imported ``mujoco`` module or fail with an actionable error."""
    try:
        import mujoco
    except ImportError as error:
        raise OptionalDependencyError(
            "append_debug_primitives requires the optional 'mujoco' package, which is not "
            "installed; install it with the MuJoCo extra (pip install 'unisim-core[mujoco]')"
        ) from error
    return mujoco


def quat_from_z_axis(
    direction: np.ndarray | Sequence[float],
) -> tuple[float, float, float, float]:
    """Return the unit wxyz quaternion rotating the +z axis onto ``direction``.

    Args:
        direction: Non-zero 3-vector.

    Returns:
        Unit ``(w, x, y, z)`` quaternion.

    Raises:
        ValueError: ``direction`` is (numerically) zero.
    """
    vec = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError("direction must be non-zero")
    vec /= norm
    # axis = z × d, w = 1 + z·d (normalized); degenerates when d ≈ -z.
    w = 1.0 + float(vec[2])
    if w < 1e-9:
        return (0.0, 1.0, 0.0, 0.0)
    quat = np.array([w, -vec[1], vec[0], 0.0], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def segment_arrow(
    p0: np.ndarray | Sequence[float],
    p1: np.ndarray | Sequence[float],
    *,
    rgba: np.ndarray | Sequence[float],
    min_length: float = 1e-6,
) -> DebugPrimitive | None:
    """Build an arrow primitive spanning ``p0`` → ``p1`` (``None`` when degenerate).

    Args:
        p0: Arrow tail.
        p1: Arrow tip.
        rgba: ``(r, g, b, a)`` arrow color.
        min_length: Segments shorter than this are degenerate and yield ``None``.

    Returns:
        ``arrow`` :class:`DebugPrimitive` from ``p0`` to ``p1``, or ``None``.
    """
    start = np.asarray(p0, dtype=np.float64).reshape(3)
    end = np.asarray(p1, dtype=np.float64).reshape(3)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < min_length:
        return None
    # Both tuples are built as in UniLab; ``DebugPrimitive.__post_init__``
    # re-validates their arity, range, and finiteness, so the casts only state
    # the declared field types (they have no runtime effect).
    return DebugPrimitive(
        kind="arrow",
        pos=cast("tuple[float, float, float]", tuple(float(v) for v in start)),
        quat=quat_from_z_axis(delta),
        size=(length,),
        rgba=cast("tuple[float, float, float, float]", tuple(float(v) for v in rgba)),
    )


def _append_primitive(
    scene: Any,
    primitive: Any,
    offset_xy: Sequence[float] | None,
    mesh_ids: Mapping[str, int],
    mesh_materials: Mapping[str, int] | None = None,
) -> int:
    """Convert one :class:`DebugPrimitive` into user-scene geoms (env-local pos).

    ``mesh_materials`` optionally maps ``ghost_geom`` ``mesh_asset`` keys to
    material ids in the model that owns this scene; a resolved material binds
    the mesh's texture to the ghost geom.  Returns the number of geoms
    injected (0 for the documented text no-op or when ``scene.maxgeom`` is
    exhausted mid-frame).
    """
    mujoco = _import_mujoco()
    pos = np.array(primitive.pos, dtype=np.float64)
    if offset_xy is not None:
        pos[0] += float(offset_xy[0])
        pos[1] += float(offset_xy[1])
    if primitive.quat is not None:
        quat = np.array(primitive.quat, dtype=np.float64)
        quat /= np.linalg.norm(quat)
        mat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)
    else:
        mat = np.eye(3, dtype=np.float64).flatten()
    rgba = np.array(primitive.rgba, dtype=np.float64)

    kind = primitive.kind
    if kind == "sphere":
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([primitive.size[0], 0.0, 0.0]),
            pos=pos,
            mat=mat,
            rgba=rgba,
        )
        scene.ngeom += 1
        return 1
    if kind == "box":
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=np.array(primitive.size),
            pos=pos,
            mat=mat,
            rgba=rgba,
        )
        scene.ngeom += 1
        return 1
    if kind == "ghost_geom":
        assert primitive.mesh_asset is not None
        mesh_id = mesh_ids.get(primitive.mesh_asset, -1)
        if mesh_id < 0:
            raise ValueError(
                f"ghost_geom mesh asset {primitive.mesh_asset!r} is not resolved; the mesh "
                "must be registered in the model that owns this scene (resolve it with "
                "mj_name2id and pass the id via mesh_ids)"
            )
        scale = primitive.size[0] if primitive.size else 1.0
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            size=np.array([scale, scale, scale]),
            pos=pos,
            mat=mat,
            rgba=rgba,
        )
        geom.dataid = mesh_id
        matid = (mesh_materials or {}).get(primitive.mesh_asset, -1)
        if matid >= 0:
            # Bind the inherited material (e.g. a cube's sticker texture);
            # texcoord enables the mesh's UV coordinates for texturing.
            geom.matid = matid
            geom.texcoord = 1
        scene.ngeom += 1
        return 1
    if kind in ("frame", "arrow"):
        length = float(primitive.size[0])
        width = max(1e-3, 0.02 * length)
        rotation = np.asarray(mat, dtype=np.float64).reshape(3, 3)
        axes: tuple[tuple[np.ndarray, tuple[Any, ...]], ...]
        if kind == "arrow":
            axes = ((rotation[:, 2], tuple(rgba)),)
        else:
            alpha = rgba[3]
            axes = tuple(
                (rotation[:, axis], (*color, alpha))
                for axis, color in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
            )
        added = 0
        for direction, color in axes:
            if scene.ngeom >= scene.maxgeom:
                return added
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                width,
                pos,
                pos + direction * length,
            )
            geom.rgba[:] = color
            scene.ngeom += 1
            added += 1
        return added
    if kind == "line":
        # Plain straight segment without an arrowhead (mjGEOM_LINE) along the
        # local +z of the primitive frame.  A duck-typed kind emitted by
        # interactive/task-owned overlays, not a DebugPrimitive kind.
        length = float(primitive.size[0]) if primitive.size else 0.0
        rotation = np.asarray(mat, dtype=np.float64).reshape(3, 3)
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            max(1e-3, 0.002 * length),
            pos,
            pos + rotation[:, 2] * length,
        )
        scene.geoms[scene.ngeom].rgba[:] = rgba
        scene.ngeom += 1
        return 1
    # kind == "text" (and any other kind without a scene channel): mjvScene has
    # no text channel; text primitives are a documented no-op here (see
    # DebugPrimitive).
    return 0


def append_debug_primitives(
    scene: Any,
    overlays: Sequence[Sequence[DebugPrimitive] | None] | None,
    *,
    offsets: np.ndarray | Sequence[Sequence[float]] | None = None,
    mesh_ids: Mapping[str, int] | None = None,
    mesh_materials: Mapping[str, int] | None = None,
) -> int:
    """Inject per-env :class:`DebugPrimitive` overlays into a caller-owned scene.

    This is the single conversion implementation shared by the offline render
    workers and interactive viewers (e.g. ``mujoco.viewer.launch_passive`` with
    ``viewer.user_scn``).  ``overlays`` follows the ``debug_overlay_getter``
    shape convention: one entry per environment, each a sequence of
    :class:`DebugPrimitive` (``None``/empty marks an env without overlay);
    a single-env caller passes ``[primitives]``.  Primitive poses are
    env-local; ``offsets`` (optional ``(num_envs, 2)`` grid XY offsets, same
    indexing as ``overlays``) shifts them into world space — pass ``None`` for
    a single already-placed env.

    ``mesh_ids`` maps ``ghost_geom`` ``mesh_asset`` keys to mesh ids in the
    model that owns this scene's render context.  Unlike the offline pipeline
    (which can inject mesh files into a recompiled render model), an
    interactive viewer cannot be recompiled, so ghost meshes must already be
    registered in the loaded model; unresolved assets fail closed with
    ``ValueError``.  ``mesh_materials`` optionally maps the same keys to
    material ids so ghost meshes render with their textured material instead
    of the flat primitive rgba.

    ``text`` is a documented no-op (``mjvScene`` has no text channel), as is
    any other unknown ``kind``.  One duck-typed kind is honoured on purpose:
    an object whose ``kind`` is ``"line"`` and whose ``pos``/``quat``/``size``/
    ``rgba`` are read like a :class:`DebugPrimitive` draws a plain straight
    segment without an arrowhead (``mjGEOM_LINE``) that starts at ``pos`` and
    runs ``size[0]`` along the local +z of its frame.  ``"line"`` is not a
    :class:`DebugPrimitive` kind (the constructor rejects it); it is emitted
    by interactive/task-owned overlays (e.g. ray visualization) and is drawn
    whenever it reaches this function.  ``validate_debug_overlays`` still
    admits only real :class:`DebugPrimitive` objects on the validated playback
    paths, so recorded videos are unchanged.

    The caller owns the scene's capacity (``maxgeom``) and must reset
    ``scene.ngeom`` between frames.  MuJoCo is imported lazily: only a call
    that converts at least one primitive needs the optional package and
    raises :class:`~unisim.optional.OptionalDependencyError` when it is
    missing.

    Args:
        scene: Caller-owned ``mjvScene`` (e.g. ``renderer.scene`` or
            ``viewer.user_scn``); geoms are appended at ``scene.ngeom``.
        overlays: Per-env overlay primitives, or ``None`` for no overlay.
        offsets: Optional ``(num_envs, 2)`` XY grid offsets indexed like
            ``overlays``; ``None`` keeps env-local poses.
        mesh_ids: Optional ``mesh_asset`` key to mesh id map resolving
            ``ghost_geom`` primitives.
        mesh_materials: Optional ``mesh_asset`` key to material id map binding
            textured materials to ``ghost_geom`` primitives.

    Returns:
        Number of geoms injected.

    Raises:
        ValueError: A ``ghost_geom`` mesh asset is not resolved by
            ``mesh_ids``.
        OptionalDependencyError: MuJoCo is not installed.
    """
    if overlays is None:
        return 0
    resolved_mesh_ids = mesh_ids or {}
    added = 0
    for env_idx, env_primitives in enumerate(overlays):
        if not env_primitives:
            continue
        offset_xy = offsets[env_idx] if offsets is not None else None
        for primitive in env_primitives:
            if scene.ngeom >= scene.maxgeom:
                return added
            added += _append_primitive(
                scene, primitive, offset_xy, resolved_mesh_ids, mesh_materials
            )
    return added


__all__ = ["append_debug_primitives", "quat_from_z_axis", "segment_arrow"]
