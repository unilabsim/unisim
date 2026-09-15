from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from unisim.dr.types import FixedVariantPlan
from unisim.terrain.generator import TerrainGeneratorCfg

if TYPE_CHECKING:
    from unisim.backend.base import SimBackend

MODEL_FORMAT_URDF = "urdf"
MODEL_FORMAT_MJCF = "mjcf"
SUPPORTED_MODEL_FORMATS = frozenset({MODEL_FORMAT_URDF, MODEL_FORMAT_MJCF})
"""Asset source formats the cold-path contract understands.

Any other format tag fails closed at contract validation time; adapters must
never silently probe or sniff an undeclared format.
"""

ENTITY_MATERIALIZATION_ARTICULATION = "articulation"
ENTITY_MATERIALIZATION_RIGID = "rigid"
SUPPORTED_ENTITY_MATERIALIZATIONS = frozenset(
    {ENTITY_MATERIALIZATION_ARTICULATION, ENTITY_MATERIALIZATION_RIGID}
)

ENTITY_ROOT_FIXED = "fixed"
ENTITY_ROOT_FLOATING = "floating"
ENTITY_ROOT_KINEMATIC = "kinematic"
SUPPORTED_ENTITY_ROOT_MODES = frozenset(
    {ENTITY_ROOT_FIXED, ENTITY_ROOT_FLOATING, ENTITY_ROOT_KINEMATIC}
)


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


@dataclass(frozen=True)
class ActuatorGainOverride:
    """Owner-supplied PD/dynamics override for one named joint (cold path).

    URDF assets carry no actuator gains, so the owner (task configuration)
    supplies them per joint name; the host validates the names against the
    scanned asset and pushes the values to the worker in the INIT payload.
    ``armature``/``frictionloss`` of ``None`` keep the scanned value.
    """

    joint_name: str
    stiffness: float
    damping: float
    armature: float | None = None
    frictionloss: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.joint_name, str) or not self.joint_name:
            raise ValueError(
                f"ActuatorGainOverride joint_name must be a non-empty string, "
                f"got {self.joint_name!r}"
            )
        for field_name, value in (
            ("stiffness", self.stiffness),
            ("damping", self.damping),
            ("armature", self.armature),
            ("frictionloss", self.frictionloss),
        ):
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"ActuatorGainOverride {field_name} for joint {self.joint_name!r} must be "
                    f"a finite non-negative number, got {value!r}"
                )


def _validate_friction_triple(value: object, label: str) -> tuple[float, float, float]:
    """Validate one PhysX material triple: (static friction, dynamic friction, restitution).

    The original repository writes ``[f, f, 0.0]`` per material
    (simtoolreal/isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py:1558-1564).
    Anything but three finite non-negative numbers fails closed.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a (static, dynamic, restitution) triple, got {value!r}")
    if len(value) != 3:
        raise ValueError(f"{label} must have exactly 3 components, got {value!r}")
    triple: list[float] = []
    for component in value:
        if isinstance(component, bool):
            raise TypeError(f"{label} components must be numbers, got {value!r}")
        numeric = float(component)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                f"{label} components must be finite non-negative numbers, got {value!r}"
            )
        triple.append(numeric)
    return (triple[0], triple[1], triple[2])


@dataclass(frozen=True)
class BodyFrictionOverride:
    """Owner-supplied contact-material override for one named body (cold path).

    Same shape as :class:`ActuatorGainOverride`: URDF assets carry no contact
    materials beyond converter defaults, so the owner (task configuration)
    supplies per-body values by body name; the host validates the names
    against the scanned asset and pushes the values to the worker in the
    INIT payload.  ``friction`` is the PhysX material triple (static
    friction, dynamic friction, restitution); the original repository's
    fingertip rule is ``[1.5, 1.5, 0.0]`` on the five DP links
    (scene_utils.py:52-55 and 1579-1592).
    """

    body_name: str
    friction: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.body_name, str) or not self.body_name:
            raise ValueError(
                f"BodyFrictionOverride body_name must be a non-empty string, "
                f"got {self.body_name!r}"
            )
        object.__setattr__(
            self,
            "friction",
            _validate_friction_triple(
                self.friction, f"BodyFrictionOverride friction for body {self.body_name!r}"
            ),
        )


@dataclass(frozen=True)
class ScenePhysxCfg:
    """Scene-level PhysX solver declaration consumed by PhysX backends.

    Field defaults mirror IsaacLab's ``PhysxCfg`` so a partial declaration
    overrides exactly the authored fields.  ``SceneCfg.physx=None`` (the
    scene default) keeps the backend's own defaults; declaring this block is
    what opts a scene into explicit scene-level solver tuning (iteration
    clamps, bounce threshold, GPU contact stream buffers).  Values are the
    raw PhysX quantities: ``solver_type`` is the PhysX enum (0 = PGS,
    1 = TGS).
    """

    solver_type: int = 1
    min_position_iteration_count: int = 1
    max_position_iteration_count: int = 255
    min_velocity_iteration_count: int = 0
    max_velocity_iteration_count: int = 255
    bounce_threshold_velocity: float = 0.5
    friction_offset_threshold: float = 0.04
    friction_correlation_distance: float = 0.025
    gpu_max_rigid_contact_count: int = 2**23
    gpu_max_rigid_patch_count: int = 5 * 2**15

    _INT_FIELDS = (
        "min_position_iteration_count",
        "max_position_iteration_count",
        "min_velocity_iteration_count",
        "max_velocity_iteration_count",
        "gpu_max_rigid_contact_count",
        "gpu_max_rigid_patch_count",
    )
    _FLOAT_FIELDS = (
        "bounce_threshold_velocity",
        "friction_offset_threshold",
        "friction_correlation_distance",
    )

    def __post_init__(self) -> None:
        solver = self.solver_type
        if (
            isinstance(solver, bool)
            or not isinstance(solver, (int, np.integer))
            or int(solver) not in (0, 1)
        ):
            raise ValueError(
                f"ScenePhysxCfg solver_type must be 0 (PGS) or 1 (TGS), got {solver!r}"
            )
        object.__setattr__(self, "solver_type", int(solver))
        for field_name in self._INT_FIELDS:
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise ValueError(
                    f"ScenePhysxCfg {field_name} must be a non-negative integer, "
                    f"got {value!r}"
                )
            object.__setattr__(self, field_name, int(value))
        for field_name in self._FLOAT_FIELDS:
            value = getattr(self, field_name)
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"ScenePhysxCfg {field_name} must be a finite non-negative number, "
                    f"got {value!r}"
                )
            object.__setattr__(self, field_name, numeric)
        for field_name in ("gpu_max_rigid_contact_count", "gpu_max_rigid_patch_count"):
            count = getattr(self, field_name)
            if count <= 0:
                raise ValueError(
                    f"ScenePhysxCfg {field_name} must be positive, got {count!r}"
                )

    def as_kwargs(self) -> dict[str, float | int]:
        """Return the validated field mapping for wire serialization."""
        return {
            "solver_type": self.solver_type,
            "min_position_iteration_count": self.min_position_iteration_count,
            "max_position_iteration_count": self.max_position_iteration_count,
            "min_velocity_iteration_count": self.min_velocity_iteration_count,
            "max_velocity_iteration_count": self.max_velocity_iteration_count,
            "bounce_threshold_velocity": self.bounce_threshold_velocity,
            "friction_offset_threshold": self.friction_offset_threshold,
            "friction_correlation_distance": self.friction_correlation_distance,
            "gpu_max_rigid_contact_count": self.gpu_max_rigid_contact_count,
            "gpu_max_rigid_patch_count": self.gpu_max_rigid_patch_count,
        }


@dataclass(frozen=True)
class EntityInitStateCfg:
    """Articulation spawn pose for one declared entity (cold path).

    A fixed-base articulation's root pose has no other write channel (root
    writes are reset-event-owned and fixed-base root writes are fail-closed),
    so the owner declares the spawn pose here and the worker applies it as
    the articulation's spawn-time initial state.  Defaults mirror IsaacLab's
    spawn defaults (origin, identity ``wxyz`` quaternion).
    """

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if isinstance(self.pos, (str, bytes)) or not isinstance(self.pos, (tuple, list)):
            raise TypeError(f"EntityInitStateCfg pos must be an xyz triple, got {self.pos!r}")
        if len(self.pos) != 3 or not all(math.isfinite(float(v)) for v in self.pos):
            raise ValueError(
                f"EntityInitStateCfg pos must be three finite numbers, got {self.pos!r}"
            )
        object.__setattr__(
            self, "pos", (float(self.pos[0]), float(self.pos[1]), float(self.pos[2]))
        )
        rot = self.rot_wxyz
        if isinstance(rot, (str, bytes)) or not isinstance(rot, (tuple, list)):
            raise TypeError(f"EntityInitStateCfg rot_wxyz must be a wxyz quaternion, got {rot!r}")
        if len(rot) != 4 or not all(math.isfinite(float(v)) for v in rot):
            raise ValueError(
                f"EntityInitStateCfg rot_wxyz must be four finite numbers, got {rot!r}"
            )
        quat = tuple(float(v) for v in rot)
        if math.sqrt(sum(v * v for v in quat)) <= 0.0:
            raise ValueError(f"EntityInitStateCfg rot_wxyz must be non-zero, got {rot!r}")
        object.__setattr__(self, "rot_wxyz", quat)


@dataclass(frozen=True)
class SceneEntitySpec:
    """Typed cold-path declaration of one logical scene asset.

    Multi-asset scenes (e.g. SimToolReal's robot/table/object/goalviz) declare
    each logical role here: the asset source file, its format tag, how the
    backend materializes it, and how its root moves.  UniSim never parses
    UniLab-private entity types; this is the backend-facing contract.

    ``asset_format`` is the explicit format tag for ``model_file``; only
    ``urdf``/``mjcf`` are supported and anything else fails closed at
    construction.  ``root_mode`` declares the root motion per role:
    ``fixed`` welds the root to the world (fixed-base articulation),
    ``floating`` materializes a dynamic free root, and ``kinematic``
    materializes a pose-driven root without dynamics (goal visualization).
    For URDF assets the fixed/floating choice is a converter flag, so the
    declaration is authoritative; for MJCF assets the declaration is
    cross-checked against the scanned free joint and a mismatch fails closed
    (``kinematic`` is a worker-side simulation flag and is not derivable from
    MJCF content, so it is not cross-checked).

    Composition is declaration-driven — the backend never infers task
    semantics from entity names: ``collision_enabled`` is the USD-bake
    collision flag, ``replace_cylinders_with_capsules`` is the URDF converter
    flag, ``init_state`` is the articulation spawn pose, and
    ``mirrors_fixed_variant_pool`` binds a kinematic visual twin to the
    scene's variant pool.
    """

    name: str
    model_file: str
    asset_format: str
    materialization: str
    root_mode: str
    actuator_gain_overrides: tuple[ActuatorGainOverride, ...] = ()
    """Owner gain table for this entity's actuated joints, by joint name."""
    contact_friction: tuple[float, float, float] | None = None
    """Default PhysX contact material (static, dynamic, restitution) applied to
    every collision shape of this entity after spawn.  ``None`` leaves the
    converted-USD materials untouched."""
    contact_friction_by_body: tuple[BodyFrictionOverride, ...] = ()
    """Per-body contact-material overrides layered on ``contact_friction``.

    Articulation entities only (the original repository's rule set is the
    robot's five fingertip DP links, scene_utils.py:1579-1592); declaring
    overrides on a rigid entity, or without a ``contact_friction`` default,
    fails closed at construction.
    """
    consumes_fixed_variant_pool: bool = False
    """Whether this entity's asset is drawn from ``SceneCfg.fixed_variant_plan``.

    Exactly one entity per scene may declare it; plan-binding consistency
    (pool presence, entity shape, assignment) is validated fail-closed on
    the host cold path, not at construction.
    """
    init_state: EntityInitStateCfg | None = None
    """Spawn-time articulation pose (see :class:`EntityInitStateCfg`).

    Articulation entities only: the worker applies it as the articulation's
    ``InitialStateCfg``.  ``None`` keeps the backend's default spawn pose.
    """
    collision_enabled: bool | None = None
    """USD-bake collision flag for this entity's collision prims.

    ``None`` leaves the converted-USD collision state untouched (converter
    default: enabled); ``False`` authors ``collisionEnabled=False`` for
    non-physical visual twins; ``True`` pins collision explicitly.
    """
    replace_cylinders_with_capsules: bool | None = None
    """URDF converter flag for this entity's asset.

    ``None`` keeps the materialization-based backend default: floating rigid
    entities convert with capsule replacement (the dynamic object contract),
    every other role keeps the converter default (no replacement).  A
    declared value overrides the default for this entity; a fixed variant
    pool target's flag also applies to every pool variant source.
    """
    mirrors_fixed_variant_pool: bool = False
    """Whether this kinematic entity visualizes the pool target's variants.

    The entity's per-environment visuals mirror the scene's fixed variant
    pool (same per-env source, kinematic non-physical bake) instead of using
    its own single asset.  Rigid kinematic entities only, mutually exclusive
    with ``consumes_fixed_variant_pool``, and it requires the scene to carry
    a plan (validated fail-closed on the host cold path).
    """

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"SceneEntitySpec name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.model_file, str) or not self.model_file:
            raise ValueError(
                f"SceneEntitySpec {self.name!r} model_file must be a non-empty string, "
                f"got {self.model_file!r}"
            )
        if self.asset_format not in SUPPORTED_MODEL_FORMATS:
            raise ValueError(
                f"SceneEntitySpec {self.name!r} asset_format must be one of "
                f"{sorted(SUPPORTED_MODEL_FORMATS)}, got {self.asset_format!r}; other formats "
                "fail closed at contract validation time"
            )
        if self.materialization not in SUPPORTED_ENTITY_MATERIALIZATIONS:
            raise ValueError(
                f"SceneEntitySpec {self.name!r} materialization must be one of "
                f"{sorted(SUPPORTED_ENTITY_MATERIALIZATIONS)}, got {self.materialization!r}"
            )
        if self.root_mode not in SUPPORTED_ENTITY_ROOT_MODES:
            raise ValueError(
                f"SceneEntitySpec {self.name!r} root_mode must be one of "
                f"{sorted(SUPPORTED_ENTITY_ROOT_MODES)}, got {self.root_mode!r}"
            )
        override_names = [override.joint_name for override in self.actuator_gain_overrides]
        duplicates = sorted({name for name in override_names if override_names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"SceneEntitySpec {self.name!r} has duplicate actuator gain overrides "
                f"for joints: {duplicates}"
            )
        if self.contact_friction is not None:
            object.__setattr__(
                self,
                "contact_friction",
                _validate_friction_triple(
                    self.contact_friction, f"SceneEntitySpec {self.name!r} contact_friction"
                ),
            )
        body_names = [override.body_name for override in self.contact_friction_by_body]
        duplicate_bodies = sorted({name for name in body_names if body_names.count(name) > 1})
        if duplicate_bodies:
            raise ValueError(
                f"SceneEntitySpec {self.name!r} has duplicate contact friction overrides "
                f"for bodies: {duplicate_bodies}"
            )
        if self.contact_friction_by_body:
            if self.materialization != ENTITY_MATERIALIZATION_ARTICULATION:
                raise ValueError(
                    f"SceneEntitySpec {self.name!r} declares per-body contact friction "
                    f"overrides but materialization={self.materialization!r}; only "
                    "articulation entities support per-body overrides"
                )
            if self.contact_friction is None:
                raise ValueError(
                    f"SceneEntitySpec {self.name!r} declares per-body contact friction "
                    "overrides without a contact_friction default; the worker tiles the "
                    "default across all shapes before applying per-body overrides "
                    "(scene_utils.py:1576-1592)"
                )
        if self.init_state is not None:
            if not isinstance(self.init_state, EntityInitStateCfg):
                raise TypeError(
                    f"SceneEntitySpec {self.name!r} init_state must be an "
                    f"EntityInitStateCfg, got {type(self.init_state).__name__}"
                )
            if self.materialization != ENTITY_MATERIALIZATION_ARTICULATION:
                raise ValueError(
                    f"SceneEntitySpec {self.name!r} declares init_state but "
                    f"materialization={self.materialization!r}; spawn poses apply to "
                    "articulation entities (rigid roots are reset-event-owned)"
                )
        if self.collision_enabled is not None and not isinstance(self.collision_enabled, bool):
            raise TypeError(
                f"SceneEntitySpec {self.name!r} collision_enabled must be a boolean or "
                f"None, got {self.collision_enabled!r}"
            )
        if self.replace_cylinders_with_capsules is not None and not isinstance(
            self.replace_cylinders_with_capsules, bool
        ):
            raise TypeError(
                f"SceneEntitySpec {self.name!r} replace_cylinders_with_capsules must be "
                f"a boolean or None, got {self.replace_cylinders_with_capsules!r}"
            )
        if self.mirrors_fixed_variant_pool:
            if not isinstance(self.mirrors_fixed_variant_pool, bool):
                raise TypeError(
                    f"SceneEntitySpec {self.name!r} mirrors_fixed_variant_pool must be "
                    f"a boolean, got {self.mirrors_fixed_variant_pool!r}"
                )
            if self.consumes_fixed_variant_pool:
                raise ValueError(
                    f"SceneEntitySpec {self.name!r} declares both consumes_fixed_variant_"
                    "pool and mirrors_fixed_variant_pool; one entity either draws its "
                    "asset from the pool or mirrors it, not both"
                )
            if (
                self.materialization != ENTITY_MATERIALIZATION_RIGID
                or self.root_mode != ENTITY_ROOT_KINEMATIC
            ):
                raise ValueError(
                    f"SceneEntitySpec {self.name!r} declares mirrors_fixed_variant_pool "
                    f"but materialization={self.materialization!r}/root_mode="
                    f"{self.root_mode!r}; pool mirrors are kinematic rigid visual twins"
                )

    @property
    def fixed_base(self) -> bool:
        """Whether the entity root is welded to the world (URDF ``fix_base``)."""
        return self.root_mode == ENTITY_ROOT_FIXED


@dataclass(frozen=True)
class GroundPlaneSceneCfg:
    """Declarative world-level ground plane for offline-safe scenes.

    Composition declaration consumed by adapters whose scene sources do not
    carry a floor (e.g. the IsaacSim URDF multi-asset path); backends that do
    not consume a declarative ground plane reject the scene at construction
    rather than silently dropping it.  Defaults mirror IsaacLab
    ``GroundPlaneCfg``'s physics material.

    ``friction`` is the repo's PhysX material triple (static friction,
    dynamic friction, restitution slot) validated like every other contact
    declaration; the dedicated ``restitution`` field is the authoritative
    material restitution, and ``size_m`` is the ground's full side length.
    """

    friction: tuple[float, float, float] = (0.5, 0.5, 0.0)
    restitution: float = 0.0
    size_m: float = 200.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "friction",
            _validate_friction_triple(self.friction, "GroundPlaneSceneCfg friction"),
        )
        for field_name, value in (("restitution", self.restitution), ("size_m", self.size_m)):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(
                    f"GroundPlaneSceneCfg {field_name} must be a finite number, got {value!r}"
                )
        if float(self.restitution) < 0.0:
            raise ValueError(
                f"GroundPlaneSceneCfg restitution must be a finite non-negative number, "
                f"got {self.restitution!r}"
            )
        if float(self.size_m) <= 0.0:
            raise ValueError(
                f"GroundPlaneSceneCfg size_m must be a finite positive number, "
                f"got {self.size_m!r}"
            )
        object.__setattr__(self, "restitution", float(self.restitution))
        object.__setattr__(self, "size_m", float(self.size_m))


@dataclass
class SceneCfg:
    """Scene source and optional cold-path composition configuration."""

    model_file: str
    fragment_files: list[str] = field(default_factory=list)
    terrain: TerrainSceneCfg | None = None
    ground_plane: GroundPlaneSceneCfg | None = None
    """Declarative world-level ground plane (composition declaration).

    The IsaacSim worker consumes it as a world-level local collision ground
    in every runtime mode (task-level scene composition in the original
    repository: ``scene_utils.py`` ``setup_scene`` step 5); an undeclared
    scene keeps the backend's native ground behavior.  The declaration is
    scene content, not a hint: backends that do not consume a declarative
    ground plane fail closed at construction
    (:func:`validate_scene_composition_support`).
    """
    entities: dict[str, object] = field(default_factory=dict)
    """Logical entity partitions materialized by the base-owned manager facade."""
    entity_assets: tuple[SceneEntitySpec, ...] = ()
    """Typed backend-facing asset declarations for multi-asset scenes.

    Unlike ``entities`` (an owner-level passthrough that UniLab materializes
    into its private ``EntityCfg`` records for the manager facade), this tuple
    is the typed contract backends consume on the cold path: each entry pairs
    an asset role with its source file, format tag, materialization type, and
    root mode.  Scenes that declare ``entity_assets`` still keep ``model_file``
    as the primary asset (typically the actuated robot).  The declaration is
    scene content, not a hint: backends that do not materialize declared
    entity assets fail closed at construction
    (:func:`validate_scene_composition_support`).
    """
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
    physx: ScenePhysxCfg | None = None
    """Scene-level PhysX solver declaration (see :class:`ScenePhysxCfg`).

    ``None`` (the default) keeps the backend's own solver defaults; a
    declaration is consumed by PhysX-backed adapters (IsaacSim) and rejected
    at construction by backends that cannot consume it.
    """
    env_grid_spacing: float | None = None
    """Spacing in meters of the environment clone grid.

    Layout only (every environment is its own collision-filtered subtree),
    but world-frame environment origins derive from it.  ``None`` (the
    default) keeps the backend's native layout — IsaacSim's ``GridCloner``
    spacing of 2.0 m, matching the upstream single-asset behavior; owners
    that need a tighter grid declare it, and backends that own their layout
    reject the declaration at construction.  Must be a finite positive
    number when declared.
    """


def validate_scene_composition_support(
    scene: SceneCfg,
    backend_label: str,
    *,
    supports_entity_assets: bool = False,
    supports_ground_plane: bool = False,
    supports_scene_physx: bool = False,
    supports_env_grid_spacing: bool = False,
) -> None:
    """Fail closed when a backend cannot consume declared scene composition.

    ``SceneCfg.entity_assets``, ``SceneCfg.ground_plane``, ``SceneCfg.physx``,
    and ``SceneCfg.env_grid_spacing`` are composition/tuning declarations: a
    backend that cannot materialize them must reject the scene at
    construction instead of silently dropping content and degrading to its
    default behavior.  Each adapter declares what it consumes; the subprocess
    family routes the flags through the ``_supports_*`` hooks so pooled
    specializations own their declaration.

    The fields are read defensively: a duck-typed scene stand-in without them
    (dependency-probe tests construct backends with bare objects to reach the
    backend's own missing-runtime error) carries no declarations, so the gate
    passes through; real ``SceneCfg`` instances always define the fields.
    """
    entity_assets = getattr(scene, "entity_assets", None)
    ground_plane = getattr(scene, "ground_plane", None)
    scene_physx = getattr(scene, "physx", None)
    env_grid_spacing = getattr(scene, "env_grid_spacing", None)
    if entity_assets and not supports_entity_assets:
        raise NotImplementedError(
            f"{backend_label} backend does not consume multi-asset scene composition "
            "(SceneCfg.entity_assets); select a backend that materializes "
            "declared entity assets"
        )
    if ground_plane is not None and not supports_ground_plane:
        raise NotImplementedError(
            f"{backend_label} backend does not consume the declarative ground plane "
            "(SceneCfg.ground_plane); this backend's scene floor comes from the "
            "scene model itself"
        )
    if scene_physx is not None and not supports_scene_physx:
        raise NotImplementedError(
            f"{backend_label} backend does not consume the scene-level PhysX "
            "declaration (SceneCfg.physx); this backend's solver configuration "
            "comes from the scene model itself"
        )
    if env_grid_spacing is not None and not supports_env_grid_spacing:
        raise NotImplementedError(
            f"{backend_label} backend does not consume the environment grid "
            "spacing declaration (SceneCfg.env_grid_spacing); this backend owns "
            "its environment layout"
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
