from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from unisim.dr.types import FixedVariantPlan
from unisim.entities import EntityVariantBinding, SceneEntitySpec, validate_entity_declarations
from unisim.terrain.generator import TerrainGeneratorCfg

if TYPE_CHECKING:
    from unisim.backend.base import SimBackend


def resolve_scene_fragment_path(fragment_file: str, model_file: Path) -> Path:
    """Resolve a ``SceneCfg.fragment_files`` entry against the scene model file.

    Single resolution rule shared by the MuJoCo and Motrix scene
    materializers: absolute paths pass through; relative paths that exist
    resolve against the CWD; anything else resolves relative to the model
    file's directory.
    """
    path = Path(fragment_file)
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    return (model_file.parent / path).resolve()


@dataclass
class TerrainSceneCfg:
    """Backend-agnostic terrain slot declaration for a scene."""

    generator: TerrainGeneratorCfg | None = None
    hfield_name: str = "terrain_hfield"
    geom_name: str | None = None


@dataclass
class SceneCfg:
    """Scene source and optional cold-path composition configuration."""

    model_file: str = ""
    fragment_files: list[str] = field(default_factory=list)
    terrain: TerrainSceneCfg | None = None
    entities: dict[str, object] = field(default_factory=dict)
    """Logical entity partitions materialized by the base-owned manager facade."""
    # Optional render-only model override. When set, offline playback/video
    # export renders this XML instead of ``model_file`` while physics keeps
    # using ``model_file``. Used to give the renderer a visual twin of the
    # scene (e.g. a per-env replicable obstacle) without touching the trained
    # collision model. ``None`` => render with ``model_file`` (unchanged).
    visual_model_file: str | None = None
    default_keyframe_name: str | None = None
    """Optional named keyframe used as the Manager-Based default state."""
    fixed_variant_plan: FixedVariantPlan | None = None
    """Immutable fixed model identities realized by a backend at construction."""
    entity_assets: tuple[SceneEntitySpec, ...] = ()
    """Physical sources, distinct from the task-owned logical entity selectors."""
    entity_variant: EntityVariantBinding | None = None
    """One entity-bound catalog; cannot coexist with a whole-model variant plan."""

    def __post_init__(self) -> None:
        self.validate_composition()

    def validate_composition(self, num_envs: int | None = None) -> None:
        validate_entity_declarations(self.entity_assets, self.entity_variant, num_envs)
        if self.entity_assets:
            if self.model_file:
                raise ValueError("use either model_file or entity_assets, not both")
            if self.fixed_variant_plan is not None:
                raise ValueError("entity assets use entity_variant, not fixed_variant_plan")
        elif self.entity_variant is not None:
            raise ValueError("entity_variant requires entity_assets")


def require_scene_composition_support(scene: SceneCfg | None, backend: str) -> None:
    """Negotiate the existing M1 declaration before consuming scene sources.

    Called by concrete constructors as well as the factory: direct adapter
    construction must not silently discard an entity or its fixed identity.
    """
    if not isinstance(scene, SceneCfg):
        # Optional-runtime probes historically reach dependency diagnostics
        # before consuming a scene. Only typed scene declarations are gated here.
        return
    scene.validate_composition()
    if scene.entity_assets:
        from unisim.capabilities import SupportLevel, get_adapter_capabilities

        formats = {entity.asset_format for entity in scene.entity_assets}
        configuration: dict[str, str] = {
            "entity.asset_format": next(iter(formats)) if len(formats) == 1 else "mixed"
        }
        if len(formats) == 1:
            configuration["entity.variant"] = (
                "none" if scene.entity_variant is None else "fixed"
            )
            has_physical_kinematic = any(
                entity.root_mode == "kinematic" and entity.mirror_of is None
                for entity in scene.entity_assets
            )
            configuration["entity.kinematic"] = (
                "none_or_physical" if has_physical_kinematic else "none"
            )
        if backend != "fake":
            capabilities = get_adapter_capabilities(backend)
            declaration = capabilities.get("entity.multiple", configuration=configuration)
            if declaration.support is SupportLevel.EXACT:
                requested = [
                    entity.name for entity in scene.entity_assets if entity.self_collision
                ]
                if requested:
                    # The per-entity toggle is a converter-level request; backends
                    # that only retain source-authored contact behavior declare
                    # collision.self for entity.self_collision="authored" and fail
                    # closed here instead of silently ignoring the request.
                    self_collision = capabilities.get(
                        "collision.self",
                        configuration={
                            "entity.self_collision": "true",
                            "scene.profile": "mapped_entities",
                        },
                    )
                    if self_collision.support is not SupportLevel.EXACT:
                        raise NotImplementedError(
                            f"{backend} has not implemented per-entity self_collision for "
                            f"{requested}; see the adapter's collision.self capability (#251)"
                        )
                gravity_requested = [
                    entity.name
                    for entity in scene.entity_assets
                    if entity.gravity_disabled is not None
                ]
                if gravity_requested:
                    # An explicit per-entity gravity request must be honored
                    # exactly by the mapped-scene worker; backends without an
                    # exact entity.gravity_disable declaration fail closed here
                    # instead of silently keeping their implicit behavior.
                    gravity = capabilities.get(
                        "entity.gravity_disable",
                        configuration={
                            "entity.gravity_disabled": "explicit",
                            "scene.profile": "mapped_entities",
                        },
                    )
                    if gravity.support is not SupportLevel.EXACT:
                        raise NotImplementedError(
                            f"{backend} has not implemented per-entity gravity_disabled "
                            f"for {gravity_requested}; see the adapter's "
                            "entity.gravity_disable capability (#265)"
                        )
                return
        raise NotImplementedError(
            f"{backend} has not implemented entity_assets materialization for {sorted(formats)}; "
            "see the adapter's entity.multiple capability and roadmap #108"
        )


def resolve_scene_default_qpos(cfg: SceneCfg, backend: SimBackend) -> np.ndarray | None:
    """Resolve one named default-qpos snapshot without changing the qpos0 path."""
    keyframe_name = cfg.default_keyframe_name
    if keyframe_name is not None and not isinstance(keyframe_name, str):
        raise TypeError(
            "SceneCfg default_keyframe_name must be a non-empty string or None, "
            f"got {type(keyframe_name).__name__}"
        )
    if keyframe_name == "":
        raise ValueError("SceneCfg default_keyframe_name must be a non-empty string or None")
    if keyframe_name is None:
        return None

    capability = f"default keyframe {keyframe_name!r} qpos"
    try:
        value = backend.get_keyframe_qpos(keyframe_name)
    except (AttributeError, NotImplementedError) as exc:
        raise NotImplementedError(
            f"Manager scene default keyframe {keyframe_name!r} is unavailable on "
            f"backend '{backend.backend_type}': {exc}"
        ) from exc
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Manager scene could not resolve default keyframe {keyframe_name!r} on "
            f"backend '{backend.backend_type}': {exc}"
        ) from exc

    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"Manager scene {capability} on backend '{backend.backend_type}' must return "
            f"np.ndarray, got {type(value).__name__}"
        )
    if value.ndim != 1:
        raise ValueError(
            f"Manager scene {capability} on backend '{backend.backend_type}' returned shape "
            f"{value.shape}; expected 1-D"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(
            f"Manager scene {capability} on backend '{backend.backend_type}' must be "
            f"floating, got {value.dtype}"
        )
    if not np.isfinite(value).all():
        raise ValueError(
            f"Manager scene {capability} on backend '{backend.backend_type}' returned NaN or Inf"
        )
    resolved = np.array(value, copy=True)
    resolved.setflags(write=False)
    return resolved
