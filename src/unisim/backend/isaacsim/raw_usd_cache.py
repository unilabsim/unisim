"""Content-addressed cold-path cache for IsaacSim raw MJCF-to-USD artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

RAW_USD_ARTIFACT_STAGE = "isaacsim.raw-usd"
RAW_USD_CACHE_SCHEMA_VERSION = 1
ROLE_USD_ARTIFACT_STAGE = "isaacsim.role-usd"
ROLE_USD_CACHE_SCHEMA_VERSION = 2
ENV_RAW_USD_CACHE = "UNISIM_ISAACSIM_RAW_USD_CACHE"
_LEGACY_ENV_RAW_USD_CACHE = "UNILAB_ISAACSIM_RAW_USD_CACHE"
_DEFAULT_RAW_USD_CACHE = Path("~/.cache/unisim/isaacsim/raw-usd").expanduser()
ENV_ROLE_USD_CACHE = "UNISIM_ISAACSIM_ROLE_USD_CACHE"
_LEGACY_ENV_ROLE_USD_CACHE = "UNILAB_ISAACSIM_ROLE_USD_CACHE"
_DEFAULT_ROLE_USD_CACHE = Path("~/.cache/unisim/isaacsim/role-usd").expanduser()
_DISABLED_VALUES = frozenset({"", "0", "false", "no", "off", "none", "disabled"})


def resolve_raw_usd_cache_root() -> Path | None:
    """Resolve the persistent cache root; an explicit disabled value opts out."""
    value = os.environ.get(ENV_RAW_USD_CACHE)
    if value is None:
        value = os.environ.get(_LEGACY_ENV_RAW_USD_CACHE)
    if value is None:
        return _DEFAULT_RAW_USD_CACHE
    if value.strip().lower() in _DISABLED_VALUES:
        return None
    return Path(value).expanduser().resolve()


def resolve_role_usd_cache_root() -> Path | None:
    """Resolve the derived-role cache root; an explicit disabled value opts out."""
    value = os.environ.get(ENV_ROLE_USD_CACHE)
    if value is None:
        value = os.environ.get(_LEGACY_ENV_ROLE_USD_CACHE)
    if value is None:
        return _DEFAULT_ROLE_USD_CACHE
    if value.strip().lower() in _DISABLED_VALUES:
        return None
    return Path(value).expanduser().resolve()


def file_sha256(path: Path) -> str:
    """Hash one cold-path input without loading the whole file into memory."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class RawUSDCacheFile:
    """One immutable file in a published raw artifact."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class RawUSDArtifactRequest:
    """All inputs that distinguish one raw conversion entry."""

    identity: str
    source_digest: str
    parameters: Mapping[str, Any]
    runtime_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        if not _is_sha256(self.identity):
            raise ValueError("raw USD artifact identity must be 64 hexadecimal characters")
        if not _is_sha256(self.source_digest):
            raise ValueError("raw USD source digest must be 64 hexadecimal characters")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("raw USD parameters must be a mapping")
        if not isinstance(self.runtime_versions, Mapping):
            raise TypeError("raw USD runtime versions must be a mapping")
        if not self.runtime_versions or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in self.runtime_versions.items()
        ):
            raise ValueError("raw USD runtime versions require non-empty string records")
        _canonical_json(
            {
                "parameters": dict(self.parameters),
                "runtime_versions": dict(self.runtime_versions),
            }
        )


@dataclass(frozen=True)
class RawUSDCacheRecord:
    """A validated, immutable complete cache entry."""

    identity: str
    source_digest: str
    parameters: dict[str, Any]
    runtime_versions: dict[str, str]
    usd_relative_path: str
    files: tuple[RawUSDCacheFile, ...]
    entry_path: Path
    artifact_path: Path
    usd_path: Path

    @property
    def size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)


def raw_artifact_fingerprint(record: RawUSDCacheRecord) -> str:
    """Fingerprint the immutable converter output associated with a raw entry."""
    payload = {
        "schema_version": ROLE_USD_CACHE_SCHEMA_VERSION,
        "usd_path": record.usd_relative_path,
        "files": [
            (item.path, item.size_bytes, item.sha256)
            for item in sorted(record.files, key=lambda item: item.path)
        ],
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RawUSDCacheResult:
    """Outcome of one cache materialization request."""

    record: RawUSDCacheRecord
    hit: bool
    materialize_ms: float


class _InvalidCacheEntryError(ValueError):
    """Internal signal for a missing, malformed, or corrupted entry."""


@dataclass(frozen=True)
class _Manifest:
    source_digest: str
    parameters: dict[str, Any]
    runtime_versions: dict[str, str]
    usd_relative_path: str
    files: tuple[RawUSDCacheFile, ...]


class RawUSDCache:
    """Publish and read complete raw converter output directories atomically."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser()
        if not self.root.is_absolute():
            raise ValueError("raw USD cache root must be absolute")
        self.entries_root = self.root / "entries"
        self.staging_root = self.root / "staging"
        self.quarantine_root = self.root / "quarantine"

    def load(self, identity: str) -> RawUSDCacheRecord | None:
        """Return one validated entry, or ``None`` for any miss/corruption."""
        try:
            return self._load(identity)
        except (_InvalidCacheEntryError, OSError):
            return None

    def materialize(
        self,
        request: RawUSDArtifactRequest,
        convert: Callable[[Path, str], Path],
    ) -> RawUSDCacheResult:
        """Return a validated entry, converting and publishing only on a miss."""
        started = time.perf_counter()
        existing = self.load(request.identity)
        if existing is not None and self._matches_request(existing, request):
            return RawUSDCacheResult(existing, True, (time.perf_counter() - started) * 1000.0)

        self.root.mkdir(parents=True, exist_ok=True)
        self.entries_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{request.identity[:16]}-", dir=str(self.staging_root))
        )
        try:
            artifact = staging / "artifact"
            artifact.mkdir()
            usd_path = convert(artifact, "artifact.usd")
            self._validate_usd_output(artifact, usd_path)
            manifest = _Manifest(
                request.source_digest,
                dict(request.parameters),
                dict(request.runtime_versions),
                _artifact_relative_path(artifact, Path(usd_path)),
                self._inventory(artifact),
            )
            self._write_manifest(staging, request.identity, manifest)
            record = self._publish(staging, request)
            elapsed = (time.perf_counter() - started) * 1000.0
            return RawUSDCacheResult(record, False, elapsed)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def copy_artifact(self, record: RawUSDCacheRecord, destination: Path) -> Path:
        """Copy one validated immutable artifact for mutable downstream baking."""
        current = self._load(record.identity)
        if current != record:
            raise RuntimeError("raw USD cache entry changed while preparing a role copy")
        if destination.exists():
            raise FileExistsError(f"role artifact destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(record.artifact_path, destination, copy_function=shutil.copy2)
            if self._inventory(destination) != record.files:
                raise RuntimeError("copied raw USD artifact differs from its cache record")
        except (OSError, _InvalidCacheEntryError) as exc:
            shutil.rmtree(destination, ignore_errors=True)
            raise RuntimeError("failed to copy a complete raw USD artifact") from exc
        return destination / PurePosixPath(record.usd_relative_path)

    def _load(self, identity: str) -> RawUSDCacheRecord:
        if not _is_sha256(identity):
            raise _InvalidCacheEntryError("invalid raw cache identity")
        entry = self.entries_root / identity
        if not entry.is_dir() or entry.is_symlink():
            raise _InvalidCacheEntryError("raw cache entry is missing or malformed")
        manifest_path = entry / "manifest.json"
        try:
            value = json.loads(
                manifest_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise _InvalidCacheEntryError("raw cache manifest is unreadable") from exc
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "identity",
            "source_digest",
            "parameters",
            "runtime_versions",
            "usd_path",
            "files",
        }:
            raise _InvalidCacheEntryError("raw cache manifest has unexpected fields")
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != RAW_USD_CACHE_SCHEMA_VERSION
            or value["identity"] != identity
            or not _is_sha256(value["source_digest"])
            or not isinstance(value["parameters"], dict)
            or not isinstance(value["runtime_versions"], dict)
        ):
            raise _InvalidCacheEntryError("raw cache manifest identity is malformed")
        if not value["runtime_versions"] or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in value["runtime_versions"].items()
        ):
            raise _InvalidCacheEntryError("raw cache runtime versions are malformed")
        if not isinstance(value["usd_path"], str):
            raise _InvalidCacheEntryError("raw cache USD path is malformed")
        raw_files = value["files"]
        if not isinstance(raw_files, list) or not raw_files:
            raise _InvalidCacheEntryError("raw cache file inventory is empty")
        files: list[RawUSDCacheFile] = []
        for item in raw_files:
            if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
                raise _InvalidCacheEntryError("raw cache file record is malformed")
            if not isinstance(item["path"], str) or not isinstance(item["size"], int):
                raise _InvalidCacheEntryError("raw cache file path or size is malformed")
            if isinstance(item["size"], bool) or item["size"] < 0:
                raise _InvalidCacheEntryError("raw cache file size is malformed")
            if not _is_sha256(item["sha256"]):
                raise _InvalidCacheEntryError("raw cache file digest is malformed")
            files.append(RawUSDCacheFile(item["path"], item["size"], item["sha256"]))
        artifact = entry / "artifact"
        usd_path = _resolve_entry_file(artifact, value["usd_path"])
        actual_paths: set[str] = set()
        for record in files:
            path = _resolve_entry_file(artifact, record.path)
            if path.stat().st_size != record.size_bytes or file_sha256(path) != record.sha256:
                raise _InvalidCacheEntryError(f"raw cache file is corrupted: {record.path}")
            actual_paths.add(record.path)
        if len(actual_paths) != len(files):
            raise _InvalidCacheEntryError("raw cache file inventory contains duplicate paths")
        all_files = self._inventory(artifact)
        if actual_paths != {item.path for item in all_files}:
            raise _InvalidCacheEntryError("raw cache file inventory differs from artifact")
        if value["usd_path"] not in actual_paths:
            raise _InvalidCacheEntryError("raw cache USD path is absent from inventory")
        return RawUSDCacheRecord(
            identity,
            value["source_digest"],
            value["parameters"],
            value["runtime_versions"],
            value["usd_path"],
            tuple(files),
            entry,
            artifact,
            usd_path,
        )

    def _publish(
        self, staging: Path, request: RawUSDArtifactRequest
    ) -> RawUSDCacheRecord:
        identity = request.identity
        entry = self.entries_root / identity
        while True:
            if entry.exists():
                existing = self.load(identity)
                if existing is not None and self._matches_request(existing, request):
                    return existing
                self._quarantine(entry)
            try:
                os.rename(staging, entry)
            except OSError:
                existing = self.load(identity)
                if existing is not None and self._matches_request(existing, request):
                    return existing
                continue
            record = self._load(identity)
            if record is None or not self._matches_request(
                record, request
            ):  # pragma: no cover - staging is validated before rename
                raise RuntimeError("newly published raw USD cache entry is invalid")
            return record

    @staticmethod
    def _matches_request(
        record: RawUSDCacheRecord, request: RawUSDArtifactRequest
    ) -> bool:
        return (
            record.identity == request.identity
            and record.source_digest == request.source_digest
            and _canonical_json(
                {
                    "parameters": record.parameters,
                    "runtime_versions": record.runtime_versions,
                }
            )
            == _canonical_json(
                {
                    "parameters": dict(request.parameters),
                    "runtime_versions": dict(request.runtime_versions),
                }
            )
        )

    def _quarantine(self, entry: Path) -> None:
        self.quarantine_root.mkdir(parents=True, exist_ok=True)
        quarantine = self.quarantine_root / f"{entry.name}-{uuid.uuid4().hex}"
        try:
            os.rename(entry, quarantine)
        except FileNotFoundError:
            return
        shutil.rmtree(quarantine, ignore_errors=True)

    def _write_manifest(self, staging: Path, identity: str, manifest: _Manifest) -> None:
        payload = {
            "schema_version": RAW_USD_CACHE_SCHEMA_VERSION,
            "identity": identity,
            "source_digest": manifest.source_digest,
            "parameters": manifest.parameters,
            "runtime_versions": manifest.runtime_versions,
            "usd_path": manifest.usd_relative_path,
            "files": [
                {"path": item.path, "size": item.size_bytes, "sha256": item.sha256}
                for item in manifest.files
            ],
        }
        try:
            text = json.dumps(
                payload, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=False
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(f"raw USD cache manifest is not finite JSON: {exc}") from exc
        (staging / "manifest.json").write_text(text + "\n", encoding="utf-8")

    @staticmethod
    def _inventory(artifact: Path) -> tuple[RawUSDCacheFile, ...]:
        records: list[RawUSDCacheFile] = []
        for path in sorted(artifact.rglob("*")):
            if path.is_symlink():
                raise _InvalidCacheEntryError("raw USD artifacts cannot contain symlinks")
            if not path.is_file():
                continue
            relative = _artifact_relative_path(artifact, path)
            records.append(
                RawUSDCacheFile(relative, path.stat().st_size, file_sha256(path))
            )
        if not records:
            raise _InvalidCacheEntryError("raw USD artifact directory is empty")
        return tuple(records)

    @staticmethod
    def _validate_usd_output(artifact: Path, usd_path: Path) -> None:
        try:
            relative = _artifact_relative_path(artifact, usd_path)
        except _InvalidCacheEntryError as exc:
            raise RuntimeError(
                "MJCF converter wrote its USD outside the raw cache stage"
            ) from exc
        if PurePosixPath(relative).suffix.lower() not in {".usd", ".usda"}:
            raise RuntimeError("MJCF converter did not return a USD file")


class RoleUSDCache(RawUSDCache):
    """Publish immutable role-derived USD using the complete-entry protocol.

    A role request's identity already includes its raw artifact identity and
    fingerprint plus collision/visual/mirror/bake parameters.  A miss copies
    and bakes the raw artifact inside staging; a hit is inventory-validated and
    must never be mutated by native materialization or role inspection.
    """


def _artifact_relative_path(artifact: Path, path: Path) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(artifact.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise _InvalidCacheEntryError("artifact path escapes its raw cache directory") from exc
    return _safe_relative_path(relative.as_posix())


def _safe_relative_path(relative: str) -> str:
    if (
        not relative
        or "\\" in relative
        or PurePosixPath(relative).is_absolute()
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise _InvalidCacheEntryError("raw cache path is not a safe relative path")
    return relative


def _resolve_entry_file(entry: Path, relative: str) -> Path:
    _safe_relative_path(relative)
    pure = PurePosixPath(relative)
    path = entry / Path(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise _InvalidCacheEntryError(f"raw cache file is missing: {relative}")
    try:
        path.resolve(strict=True).relative_to(entry.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise _InvalidCacheEntryError("raw cache file escapes its entry") from exc
    return path


__all__ = [
    "ENV_RAW_USD_CACHE",
    "ENV_ROLE_USD_CACHE",
    "RAW_USD_ARTIFACT_STAGE",
    "RAW_USD_CACHE_SCHEMA_VERSION",
    "ROLE_USD_ARTIFACT_STAGE",
    "ROLE_USD_CACHE_SCHEMA_VERSION",
    "RawUSDArtifactRequest",
    "RawUSDCache",
    "RawUSDCacheFile",
    "RawUSDCacheRecord",
    "RawUSDCacheResult",
    "RoleUSDCache",
    "file_sha256",
    "raw_artifact_fingerprint",
    "resolve_raw_usd_cache_root",
    "resolve_role_usd_cache_root",
]
