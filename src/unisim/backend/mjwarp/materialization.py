"""Cold-path scene materialization for the independent ``mjwarp`` backend."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from unisim.backend.scene_cleanup import TemporarySceneCleanup
from unisim.scene import SceneCfg


class _CombinedSceneCleanup:
    """Clean up temporary XML files and generated terrain directories together."""

    def __init__(
        self,
        temp_paths: Sequence[str],
        temp_dir: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        self._file_cleanup = TemporarySceneCleanup(*temp_paths) if temp_paths else None
        self._temp_dir = temp_dir
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        if self._file_cleanup is not None:
            self._file_cleanup.cleanup()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()


@dataclass(frozen=True)
class MjwarpSceneContext:
    """Cold-path scene source and cleanup ownership for one backend instance."""

    source_model_file: str
    diagnostic_model_file: str
    visual_model_file: str | None = None
    terrain_origins: np.ndarray | None = None
    terrain_surface_sampler: Any | None = None
    cleanup_handle: Any | None = None
    tracked_body_names: tuple[str, ...] = ()


def materialize_scene(
    scene: SceneCfg,
    *,
    add_body_sensors: bool = False,
    tracked_body_names: Sequence[str] | None = None,
) -> MjwarpSceneContext:
    """Resolve a flat, fragment, or generated-terrain scene before CUDA model upload.

    When ``add_body_sensors`` is set, world-frame body tracking sensors are
    injected into the resolved model on the cold path (same helper as the
    MuJoCo backend) so the host profile can serve body kinematics from its
    per-step sensor cache without extra device transfers.
    """
    if scene is None or not scene.model_file:
        raise ValueError("MjwarpBackend requires SceneCfg.model_file")

    resolved_tracked_body_names: tuple[str, ...] = ()
    if scene.fixed_variant_plan is not None:
        if scene.terrain is not None:
            raise NotImplementedError(
                "mjwarp backend does not yet support fixed_variant_plan combined with "
                "generated terrain."
            )
        source_model_file = str(scene.fixed_variant_plan.variants[0].model_file)
        if add_body_sensors:
            from unisim.backend.mujoco.xml import get_named_bodies

            _body_ids, valid_bnames = get_named_bodies(source_model_file)
            if tracked_body_names is not None:
                requested = tuple(tracked_body_names)
                known = set(valid_bnames)
                missing = [name for name in requested if name not in known]
                if missing:
                    raise ValueError(
                        "tracked_body_names reference bodies missing from the model: "
                        f"{missing}"
                    )
                valid_bnames = list(requested)
            resolved_tracked_body_names = tuple(valid_bnames)
        return MjwarpSceneContext(
            source_model_file=source_model_file,
            diagnostic_model_file=str(scene.model_file),
            visual_model_file=str(scene.visual_model_file or scene.model_file),
            cleanup_handle=None,
            tracked_body_names=resolved_tracked_body_names,
        )

    temp_paths: list[str] = []
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    terrain_origins: np.ndarray | None = None
    terrain_surface_sampler: Any | None = None

    if scene.terrain is not None:
        if scene.terrain.generator is None:
            raise ValueError("SceneCfg.terrain.generator must be configured for terrain scenes")
        from unisim.backend.mujoco.xml import materialize_hfield_attached_scene

        temp_dir = tempfile.TemporaryDirectory(prefix="unilab_mjwarp_scene_")
        try:
            _, terrain_origins, terrain_surface_sampler = materialize_hfield_attached_scene(
                model_file=str(scene.model_file),
                terrain_cfg=scene.terrain.generator,
                output_dir=temp_dir.name,
                fragment_files=scene.fragment_files,
                hfield_name=scene.terrain.hfield_name,
                geom_name=scene.terrain.geom_name or "floor",
                return_surface_sampler=True,
            )
        except Exception:
            temp_dir.cleanup()
            raise
        source_model_file = os.path.join(temp_dir.name, "scene.xml")
        visual_model_file = source_model_file
    elif not scene.fragment_files:
        source_model_file = str(scene.model_file)
        visual_model_file = str(scene.visual_model_file or scene.model_file)
    else:
        # This is intentionally in a cold-path-only module.  The shared XML
        # composition helper is not a sibling runtime backend dependency.
        from unisim.backend.mujoco.xml import materialize_scene_fragments

        source_model_file = materialize_scene_fragments(
            str(scene.model_file),
            fragment_files=scene.fragment_files,
        )
        temp_paths.append(source_model_file)
        visual_model_file = source_model_file

    if add_body_sensors:
        from unisim.backend.mujoco.xml import bind_tracking_sensors

        source_model_file, _tracked_body_ids, valid_bnames = bind_tracking_sensors(
            source_model_file,
            tracked_body_names=tracked_body_names,
        )
        temp_paths.append(source_model_file)
        resolved_tracked_body_names = tuple(valid_bnames)

    cleanup_handle: Any | None = None
    if temp_dir is not None:
        cleanup_handle = _CombinedSceneCleanup(temp_paths, temp_dir)
    elif temp_paths:
        cleanup_handle = TemporarySceneCleanup(*temp_paths)

    return MjwarpSceneContext(
        source_model_file=source_model_file,
        diagnostic_model_file=str(scene.model_file),
        visual_model_file=visual_model_file,
        terrain_origins=terrain_origins,
        terrain_surface_sampler=terrain_surface_sampler,
        cleanup_handle=cleanup_handle,
        tracked_body_names=resolved_tracked_body_names,
    )
