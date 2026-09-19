"""SDK-free semantic declarations and version-scoped verification evidence.

Declarations describe an adapter's advertised semantics, not successful runtime
execution. Evidence only verifies the exact recorded runtime scope. Missing
features, versions and configuration conditions remain explicitly unknown.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .backend.base import SimBackend


class SupportLevel(str, Enum):
    """Semantic support independently of the presence of runtime evidence."""

    EXACT = "exact"
    APPROXIMATE = "approximate"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


def _nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class CapabilityScope:
    """Identity of a declaration or a concrete verification environment.

    Versions are opaque exact identifiers (a version or a commit), never ranges.
    ``None`` means unrecorded; it never acts as a wildcard for verification.
    ``runtime_available`` is supplied by the caller, never discovered here.
    """

    adapter: str
    profile: str = "default"
    unisim_version: str | None = None
    adapter_version: str | None = None
    engine_version: str | None = None
    platform: str | None = None
    device: str | None = None
    runtime_available: bool | None = None

    def __post_init__(self) -> None:
        for name in ("adapter", "profile"):
            _nonempty(getattr(self, name), name)
        for name in ("unisim_version", "adapter_version", "engine_version", "platform", "device"):
            value = getattr(self, name)
            if value is not None:
                _nonempty(value, name)
        if self.runtime_available is not None and not isinstance(self.runtime_available, bool):
            raise TypeError("runtime_available must be bool or None")

    def matches_runtime(self, other: CapabilityScope) -> bool:
        """Require complete, identical identities and a present runtime at both ends."""
        if self.runtime_available is not True or other.runtime_available is not True:
            return False
        names = (
            "adapter",
            "profile",
            "unisim_version",
            "adapter_version",
            "engine_version",
            "platform",
            "device",
        )
        return all(
            getattr(self, name) is not None and getattr(self, name) == getattr(other, name)
            for name in names
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityScope:
        return cls(**dict(value))


@dataclass(frozen=True)
class CapabilityEvidence:
    """One traceable source review or runtime result, optionally withdrawn.

    Source review, skipped runs and rejection tests do not verify implemented
    semantics. Use runtime evidence only for an exercised implementation path.
    """

    kind: Literal["source", "runtime"]
    source: str
    scope: CapabilityScope
    result: Literal["passed", "failed", "skipped"] = "passed"
    revoked: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("source", "runtime"):
            raise ValueError("evidence kind must be source or runtime")
        _nonempty(self.source, "evidence source")
        if not isinstance(self.scope, CapabilityScope):
            raise TypeError("evidence scope must be CapabilityScope")
        if self.result not in ("passed", "failed", "skipped"):
            raise ValueError("evidence result must be passed, failed or skipped")
        if not isinstance(self.revoked, bool):
            raise TypeError("evidence revoked must be bool")

    def verifies(self, scope: CapabilityScope) -> bool:
        return (
            self.kind == "runtime"
            and self.result == "passed"
            and not self.revoked
            and self.scope.matches_runtime(scope)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityEvidence:
        payload = dict(value)
        payload["scope"] = CapabilityScope.from_dict(payload["scope"])
        return cls(**payload)


@dataclass(frozen=True)
class CapabilityCondition:
    """An exact or named-union configuration requirement; unknown inputs do not match."""

    key: str
    value: str

    def __post_init__(self) -> None:
        _nonempty(self.key, "condition key")
        _nonempty(self.value, "condition value")

    def matches(self, configuration: Mapping[str, str]) -> bool:
        if self.value == "none_or_physical":
            return configuration.get(self.key) in {"none", "none_or_physical"}
        return configuration.get(self.key) == self.value


@dataclass(frozen=True)
class CapabilityDeclaration:
    """One named semantic feature, its constraints and independent evidence."""

    feature: str
    support: SupportLevel
    reason: str
    conditions: tuple[CapabilityCondition, ...] = ()
    evidence: tuple[CapabilityEvidence, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.feature, "feature")
        _nonempty(self.reason, "reason")
        object.__setattr__(self, "support", SupportLevel(self.support))
        for name, kind in (("conditions", CapabilityCondition), ("evidence", CapabilityEvidence)):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, kind) for item in values):
                raise TypeError(f"{name} must be a tuple of {kind.__name__}")
        if len({condition.key for condition in self.conditions}) != len(self.conditions):
            raise ValueError("condition keys must be unique")

    def applies(self, configuration: Mapping[str, str] | None = None) -> bool:
        return all(condition.matches(configuration or {}) for condition in self.conditions)

    def runtime_verified(self, scope: CapabilityScope) -> bool:
        """Verify supported semantics without promoting approximate to exact."""
        return self.support in (SupportLevel.EXACT, SupportLevel.APPROXIMATE) and any(
            item.verifies(scope) for item in self.evidence
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "support": self.support.value,
            "reason": self.reason,
            "conditions": [asdict(item) for item in self.conditions],
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityDeclaration:
        payload = dict(value)
        payload["conditions"] = tuple(
            CapabilityCondition(**item) for item in payload.get("conditions", ())
        )
        payload["evidence"] = tuple(
            CapabilityEvidence.from_dict(item) for item in payload.get("evidence", ())
        )
        return cls(**payload)


@dataclass(frozen=True)
class CapabilityReport:
    """Deterministically ordered declarations for one adapter profile."""

    scope: CapabilityScope
    declarations: tuple[CapabilityDeclaration, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.scope, CapabilityScope):
            raise TypeError("report scope must be CapabilityScope")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported capability schema_version; expected 1")
        if not isinstance(self.declarations, tuple) or any(
            not isinstance(item, CapabilityDeclaration) for item in self.declarations
        ):
            raise TypeError("declarations must be a tuple of CapabilityDeclaration")
        names = [item.feature for item in self.declarations]
        if len(names) != len(set(names)):
            raise ValueError("capability feature names must be unique")
        object.__setattr__(
            self, "declarations", tuple(sorted(self.declarations, key=lambda item: item.feature))
        )

    def get(
        self, feature: str, *, configuration: Mapping[str, str] | None = None
    ) -> CapabilityDeclaration:
        """Resolve a feature conservatively against explicit configuration."""
        _nonempty(feature, "feature")
        context = dict(configuration or {})
        context["profile"] = self.scope.profile
        for item in self.declarations:
            if item.feature == feature:
                if item.applies(context):
                    return item
                return replace(
                    item,
                    support=SupportLevel.UNKNOWN,
                    reason="Configuration conditions are not satisfied: " + item.reason,
                )
        return CapabilityDeclaration(feature, SupportLevel.UNKNOWN, "No declaration recorded")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.to_dict(),
            "declarations": [item.to_dict() for item in self.declarations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityReport:
        payload = dict(value)
        payload["scope"] = CapabilityScope.from_dict(payload["scope"])
        payload["declarations"] = tuple(
            CapabilityDeclaration.from_dict(item) for item in payload["declarations"]
        )
        return cls(**payload)


def get_adapter_capabilities(name: str, *, profile: str = "default") -> CapabilityReport:
    """Read the audited static inventory without discovering or importing SDKs."""
    from .support import get_adapter_capabilities as query

    return query(name, profile=profile)


def backend_capabilities(backend: SimBackend, *, profile: str = "default") -> CapabilityReport:
    """Aggregate current authoritative DR/play/variant declarations on a cold path."""
    from .adapters import ADAPTER_SPECS
    from .dr.types import _RESET_TERM_NAMES, DomainRandomizationCapabilities, FixedVariantLayout

    if backend.backend_type in {spec.name for spec in ADAPTER_SPECS}:
        report = get_adapter_capabilities(backend.backend_type, profile=profile)
    else:
        report = CapabilityReport(CapabilityScope(backend.backend_type, profile))
    actual_profile = backend.get_import_report().profile
    if profile != actual_profile:
        return replace(
            report,
            declarations=tuple(
                replace(
                    item,
                    support=SupportLevel.UNKNOWN,
                    reason=f"Instance profile {actual_profile!r} does not match {profile!r}",
                )
                for item in report.declarations
            ),
        )
    declarations = {item.feature: item for item in report.declarations}
    dr = backend.get_dr_capabilities()
    play = backend.get_play_capabilities()
    source = CapabilityEvidence("source", "SimBackend.get_dr_capabilities", report.scope)

    def declare(feature: str, supported: bool, provenance: CapabilityEvidence = source) -> None:
        declarations[feature] = CapabilityDeclaration(
            feature,
            SupportLevel.EXACT if supported else SupportLevel.UNSUPPORTED,
            "Derived from authoritative " + provenance.source,
            evidence=(provenance,),
        )

    for term in sorted(_RESET_TERM_NAMES | dr.supported_reset_terms):
        declare("dr.reset." + term, dr.supports_reset_term(term))
    interval_terms = (
        set(DomainRandomizationCapabilities._LEGACY_INTERVAL_TERM_FLAGS)
        | dr.supported_interval_terms
    )
    for term in sorted(interval_terms):
        declare("dr.interval." + term, dr.supports_interval_term(term))
    declare("variant.fixed", dr.supports_fixed_variants)
    for layout in FixedVariantLayout:
        declare(
            "variant." + layout.value,
            dr.supports_fixed_variants and layout in dr.supported_fixed_variant_layouts,
        )
    declare("play.per_env", dr.supports_per_env_playback)
    play_source = CapabilityEvidence("source", "SimBackend.get_play_capabilities", report.scope)
    for field in fields(play):
        declare(
            "play." + field.name.removeprefix("supports_"),
            bool(getattr(play, field.name)),
            play_source,
        )
    return replace(report, declarations=tuple(declarations.values()))
