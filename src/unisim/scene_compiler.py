"""Versioned contracts for the portable, MJCF-first cold-path scene compiler.

The base module deliberately contains no SDK import.  Compilation is owned by
the structural oracle selected in the governing ADR and is loaded lazily.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from unisim.inspection import (
    ConfigurationField,
    ConfigurationProvenance,
    ConfigurationScope,
)

if TYPE_CHECKING:
    from unisim.mjcf_compiler import ComposedScene
    from unisim.scene import SceneCfg

PORTABLE_MJCF_PROFILE_NAME = "portable-mjcf"
PORTABLE_MJCF_PROFILE_VERSION = 1
PORTABLE_MJCF_PROFILE_ID = "portable-mjcf-v1"
PORTABLE_MJCF_STRUCTURAL_ORACLE = "mujoco:MjSpec"
SCENE_CONTENT_IDENTITY_SCHEMA_VERSION = 1
_IDENTITY_RE = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"scene identity values must be finite JSON data: {exc}") from exc


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PortableMJCFProfile:
    """The frozen v1 authoring boundary consumed by the common compiler."""

    name: str = PORTABLE_MJCF_PROFILE_NAME
    version: int = PORTABLE_MJCF_PROFILE_VERSION
    structural_oracle: str = PORTABLE_MJCF_STRUCTURAL_ORACLE

    def __post_init__(self) -> None:
        if (
            self.name != PORTABLE_MJCF_PROFILE_NAME
            or self.version != PORTABLE_MJCF_PROFILE_VERSION
            or self.structural_oracle != PORTABLE_MJCF_STRUCTURAL_ORACLE
        ):
            raise ValueError(f"unsupported portable scene profile {self.name}-v{self.version}")

    @property
    def identity(self) -> str:
        return PORTABLE_MJCF_PROFILE_ID


PORTABLE_MJCF_PROFILE = PortableMJCFProfile()


@dataclass(frozen=True)
class SceneResourceProvenance:
    """Content and location provenance for one referenced binary resource."""

    kind: str
    logical_path: str
    resolved_path: str
    content_digest: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in self.__dict__.values()):
            raise ValueError("scene resource provenance requires non-empty strings")
        if not _IDENTITY_RE.fullmatch(self.content_digest):
            raise ValueError("scene resource content digest must be 64 hexadecimal characters")

    def identity_payload(self) -> tuple[str, str, str]:
        return self.kind, self.logical_path, self.content_digest

    def to_dict(self) -> dict[str, str]:
        return dict(self.__dict__)

    @classmethod
    def from_file(
        cls, kind: str, logical_path: str, path: Path, *, digest: str | None = None
    ) -> SceneResourceProvenance:
        resolved = path.resolve(strict=True)
        return cls(
            kind,
            logical_path,
            str(resolved),
            digest or _file_digest(resolved),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneResourceProvenance:
        return cls(**dict(value))


@dataclass(frozen=True)
class SceneSourceProvenance:
    """Cold-path provenance for one logical or catalog source."""

    entity: str
    format: str
    source_path: str
    source_digest: str
    resources: tuple[SceneResourceProvenance, ...] = ()
    entity_kind: str = "articulation"
    root_mode: str = "floating"
    collision_enabled: bool = True
    initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mirror_of: str | None = None
    variant: int | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.entity, self.format, self.source_path)
        ):
            raise ValueError("scene source provenance requires non-empty strings")
        if (
            self.variant is not None
            and (type(self.variant) is not int or self.variant < -1)
        ):
            raise ValueError("scene source variant index must be None or an integer >= -1")
        if self.mirror_of is not None and (
            not isinstance(self.mirror_of, str) or not self.mirror_of
        ):
            raise ValueError("scene source mirror_of must be None or a non-empty string")
        if not _IDENTITY_RE.fullmatch(self.source_digest):
            raise ValueError("scene source digest must be 64 hexadecimal characters")
        if self.entity_kind not in {"articulation", "rigid"}:
            raise ValueError("invalid scene source entity kind")
        if self.root_mode not in {"fixed", "floating", "kinematic"}:
            raise ValueError("invalid scene source root mode")
        if type(self.collision_enabled) is not bool:
            raise TypeError("scene source collision_enabled must be boolean")
        if (
            len(self.initial_position) != 3
            or len(self.initial_quaternion) != 4
            or not all(
                type(value) in (int, float) and isfinite(value)
                for value in (*self.initial_position, *self.initial_quaternion)
            )
        ):
            raise ValueError("initial pose must contain finite numeric values")
        object.__setattr__(self, "resources", tuple(self.resources))
        if any(not isinstance(item, SceneResourceProvenance) for item in self.resources):
            raise TypeError("scene source resources must contain SceneResourceProvenance records")

    def identity_payload(self) -> tuple[Any, ...]:
        return (
            self.entity,
            self.format,
            self.mirror_of,
            self.variant,
            self.source_digest,
            self.entity_kind,
            self.root_mode,
            self.collision_enabled,
            self.initial_position,
            self.initial_quaternion,
            tuple(sorted(resource.identity_payload() for resource in self.resources)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "format": self.format,
            "source_path": self.source_path,
            "source_digest": self.source_digest,
            "resources": [resource.to_dict() for resource in self.resources],
            "entity_kind": self.entity_kind,
            "root_mode": self.root_mode,
            "collision_enabled": self.collision_enabled,
            "initial_position": list(self.initial_position),
            "initial_quaternion": list(self.initial_quaternion),
            "mirror_of": self.mirror_of,
            "variant": self.variant,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneSourceProvenance:
        data = dict(value)
        data["resources"] = tuple(
            SceneResourceProvenance.from_dict(item) for item in data.get("resources", ())
        )
        data["initial_position"] = tuple(data.get("initial_position", (0.0, 0.0, 0.0)))
        data["initial_quaternion"] = tuple(
            data.get("initial_quaternion", (1.0, 0.0, 0.0, 0.0))
        )
        return cls(**data)


@dataclass(frozen=True)
class SceneCompilerParameters:
    """Versioned inputs, excluding source content, to the canonical compiler."""

    profile: str
    compiler_name: str
    compiler_version: str
    sim_dt: float
    default_keyframe_name: str | None = None
    assignment: tuple[int, ...] = ()
    sensor_fragment_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.profile, self.compiler_name, self.compiler_version)
        ):
            raise ValueError("scene compiler identity requires non-empty strings")
        if self.profile != PORTABLE_MJCF_PROFILE_ID:
            raise ValueError(f"unsupported portable scene compiler profile {self.profile!r}")
        if type(self.sim_dt) not in (int, float) or not isfinite(self.sim_dt) or self.sim_dt <= 0:
            raise ValueError("scene compiler sim_dt must be finite and positive")
        if self.default_keyframe_name is not None and (
            not isinstance(self.default_keyframe_name, str) or not self.default_keyframe_name
        ):
            raise ValueError("default_keyframe_name must be None or a non-empty string")
        raw_assignment = tuple(self.assignment)
        if any(type(item) is not int or item < 0 for item in raw_assignment):
            raise ValueError("assignment values must be non-negative integers")
        object.__setattr__(self, "assignment", raw_assignment)
        sensor_fragments = tuple(self.sensor_fragment_digests)
        if any(not _IDENTITY_RE.fullmatch(item) for item in sensor_fragments):
            raise ValueError("sensor fragment digests must be 64 hexadecimal characters")
        object.__setattr__(self, "sensor_fragment_digests", sensor_fragments)

    def identity_payload(self) -> tuple[Any, ...]:
        return (
            self.profile,
            self.compiler_name,
            self.compiler_version,
            float(self.sim_dt),
            self.default_keyframe_name,
            self.assignment,
            self.sensor_fragment_digests,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "compiler_name": self.compiler_name,
            "compiler_version": self.compiler_version,
            "sim_dt": self.sim_dt,
            "default_keyframe_name": self.default_keyframe_name,
            "assignment": list(self.assignment),
            "sensor_fragment_digests": list(self.sensor_fragment_digests),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneCompilerParameters:
        data = dict(value)
        data["assignment"] = tuple(data.get("assignment", ()))
        data["sensor_fragment_digests"] = tuple(data.get("sensor_fragment_digests", ()))
        return cls(**data)


@dataclass(frozen=True)
class SceneContentIdentity:
    """The stable canonical identity used before adapter-specific extension."""

    profile: str
    schema_version: int
    source_identity: str
    compiler_identity: str
    canonical_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError("scene content identity requires a profile")
        if type(self.schema_version) is not int or (
            self.schema_version != SCENE_CONTENT_IDENTITY_SCHEMA_VERSION
        ):
            raise ValueError("unsupported scene content identity schema version")
        for name in ("source_identity", "compiler_identity", "canonical_identity"):
            if not _IDENTITY_RE.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be 64 hexadecimal characters")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneContentIdentity:
        return cls(**dict(value))


def compute_scene_content_identity(
    sources: tuple[SceneSourceProvenance, ...], parameters: SceneCompilerParameters
) -> SceneContentIdentity:
    """Hash canonical source/content and compiler/profile inputs.

    Absolute source locations and adapter runtime settings are intentionally
    excluded.  They remain available in provenance or are appended by downstream
    artifact owners through :func:`derive_scene_artifact_identity`.
    """
    source_payload = tuple(
        sorted((source.identity_payload() for source in sources), key=_canonical_bytes)
    )
    source_identity = _digest(
        {
            "schema_version": SCENE_CONTENT_IDENTITY_SCHEMA_VERSION,
            "sources": source_payload,
        }
    )
    compiler_identity = _digest(
        {
            "schema_version": SCENE_CONTENT_IDENTITY_SCHEMA_VERSION,
            "compiler": parameters.identity_payload(),
        }
    )
    canonical_identity = _digest(
        {
            "schema_version": SCENE_CONTENT_IDENTITY_SCHEMA_VERSION,
            "source_identity": source_identity,
            "compiler_identity": compiler_identity,
        }
    )
    return SceneContentIdentity(
        parameters.profile,
        SCENE_CONTENT_IDENTITY_SCHEMA_VERSION,
        source_identity,
        compiler_identity,
        canonical_identity,
    )


def derive_scene_artifact_identity(
    identity: SceneContentIdentity,
    stage: str,
    parameters: Mapping[str, Any],
) -> str:
    """Extend, never replace, canonical scene identity for one artifact stage.

    Importer parameters and native runtime versions belong here.  They must not
    leak into the canonical identity because non-MuJoCo adapters consume it.
    """
    if not isinstance(stage, str) or not stage:
        raise ValueError("artifact identity stage must be a non-empty string")
    return _digest(
        {
            "schema_version": SCENE_CONTENT_IDENTITY_SCHEMA_VERSION,
            "canonical_identity": identity.canonical_identity,
            "stage": stage,
            "parameters": parameters,
        }
    )


@dataclass(frozen=True)
class SceneIntentReport:
    """Serializable source-intent observations with no native effective claims."""

    profile: str
    compiler: SceneCompilerParameters
    sources: tuple[SceneSourceProvenance, ...]
    content_identity: SceneContentIdentity
    fields: tuple[ConfigurationField, ...] = field(default_factory=tuple)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError("scene intent report requires a profile")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported scene intent report schema version")
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "fields", tuple(self.fields))
        if any(not isinstance(value, SceneSourceProvenance) for value in self.sources):
            raise TypeError("scene intent sources must contain SceneSourceProvenance records")
        if any(not isinstance(value, ConfigurationField) for value in self.fields):
            raise TypeError("scene intent fields must contain ConfigurationField records")
        if any(value.effective is not None for value in self.fields):
            raise ValueError("scene intent reports cannot contain native effective values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "compiler": self.compiler.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
            "content_identity": self.content_identity.to_dict(),
            "fields": [item.to_dict() for item in self.fields],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneIntentReport:
        data = dict(value)
        data["compiler"] = SceneCompilerParameters.from_dict(data["compiler"])
        data["sources"] = tuple(
            SceneSourceProvenance.from_dict(item) for item in data.get("sources", ())
        )
        data["content_identity"] = SceneContentIdentity.from_dict(data["content_identity"])
        fields = []
        for item in data.get("fields", ()):
            record = dict(item)
            record["scope"] = ConfigurationScope(**record["scope"])
            record["provenance"] = tuple(
                ConfigurationProvenance(**provenance) for provenance in record["provenance"]
            )
            fields.append(ConfigurationField(**record))
        data["fields"] = tuple(fields)
        return cls(**data)


def load_portable_mjcf_compiler() -> Any:
    """Import only the compiler dependency required by the portable profile."""
    try:
        return importlib.import_module("mujoco")
    except ModuleNotFoundError as exc:
        try:
            installed = version("mujoco")
        except PackageNotFoundError:
            installed = None
        if installed is not None:
            raise
        raise ImportError(
            "Portable MJCF scene compilation requires the scene-compiler extra; "
            "install unisim-core[scene-compiler] (or a compatible MuJoCo runtime)"
        ) from exc


def compile_portable_scene(
    scene: SceneCfg, num_envs: int, sim_dt: float
) -> ComposedScene:
    """Compile one portable scene through the declared structural oracle.

    This owner boundary intentionally does not expose adapter-native objects to
    consumers.  The concrete adapter consumes the returned generated sources,
    layout, report and identity on its cold materialization path.
    """
    load_portable_mjcf_compiler()
    from unisim.mjcf_compiler import compose_scene

    return compose_scene(scene, num_envs, sim_dt)
