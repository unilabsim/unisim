"""Cold-path public ray-scene projection for MuJoCo-family adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..ray_query import RaySceneDescription, ray_geom_type


def ray_scene_from_mujoco_model(
    model: Any,
    *,
    body_names: Sequence[str] | None = None,
) -> RaySceneDescription:
    """Project ray-visible geometry and public body identity from one model.

    Anonymous native bodies are addressed as stable non-empty public aliases;
    this identity is local to the ray descriptor and never rewrites native
    model names. Optional ``body_names`` selects and orders that public axis
    after identity validation.
    """

    native_names = [model.body(i).name for i in range(model.nbody)]
    aliases: list[str] = []
    occupied: set[str] = set()
    for index, name in enumerate(native_names):
        alias = name or f"body_{index}"
        if alias in occupied:
            raise ValueError(f"ray body identity {alias!r} is not unique")
        aliases.append(alias)
        occupied.add(alias)

    if body_names is None:
        selected_names = tuple(aliases)
        selected_ids = np.arange(model.nbody, dtype=np.intp)
    else:
        if isinstance(body_names, (str, bytes)) or not isinstance(body_names, Sequence):
            raise TypeError("ray body_names must be a sequence of strings or None")
        requested = tuple(body_names)
        if not requested:
            raise ValueError("ray body_names cannot be empty")
        if any(not isinstance(name, str) or not name for name in requested):
            raise TypeError("ray body_names must contain non-empty strings")
        if len(set(requested)) != len(requested):
            raise ValueError(f"ray body_names must be unique: {requested}")
        unknown = [name for name in requested if name not in occupied]
        if unknown:
            raise KeyError(f"ray body_names are missing from the scene: {unknown}")
        selected_names = requested
        alias_to_native = {alias: index for index, alias in enumerate(aliases)}
        selected_ids = np.asarray([alias_to_native[name] for name in requested], dtype=np.intp)

    native_to_selected = np.full(model.nbody, -1, dtype=np.intp)
    native_to_selected[selected_ids] = np.arange(selected_ids.size, dtype=np.intp)
    geom_bodies = np.asarray(model.geom_bodyid, dtype=np.intp)
    keep = geom_bodies >= 0
    selected_keep = native_to_selected[geom_bodies[keep]] >= 0
    geom_ids = np.flatnonzero(keep)[selected_keep]

    # Preserve the descriptor's public geom order while remapping body ids.
    return RaySceneDescription(
        num_bodies=int(selected_ids.size),
        geom_types=tuple(ray_geom_type(int(model.geom_type[index])) for index in geom_ids),
        geom_sizes=np.asarray(model.geom_size, dtype=np.float64)[geom_ids],
        geom_local_pos=np.asarray(model.geom_pos, dtype=np.float64)[geom_ids],
        geom_local_quat=np.asarray(model.geom_quat, dtype=np.float64)[geom_ids],
        geom_body_ids=native_to_selected[geom_bodies[geom_ids]],
        body_names=selected_names,
    )


__all__ = ["ray_scene_from_mujoco_model"]
