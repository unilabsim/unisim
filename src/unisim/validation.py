"""Explicit cold-path semantic requirements and fail-closed binding checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from .capabilities import CapabilityCondition, CapabilityReport, SupportLevel
from .errors import BackendError

if TYPE_CHECKING:
    from .inspection import ImportReport


class SemanticValidationError(BackendError):
    """A requested semantic feature or materialized setting cannot be guaranteed."""


@dataclass(frozen=True)
class SemanticRequirements:
    """Opt into strict construction checks for a concrete adapter profile.

    ``features`` selects capability keys. ``settings`` selects import-report
    field names and requires a known effective value in every reported scope.
    Approximation consent names individual feature/field keys and applies only
    to this exact profile. Overrides are visible configuration choices, not
    approximations; callers can inspect their values in the retained report.
    Existing adapter validation always applies, even without these requirements.
    """

    features: tuple[str, ...] = ()
    settings: tuple[str, ...] = ()
    profile: str = "default"
    configuration: tuple[CapabilityCondition, ...] = ()
    approximations: tuple[str, ...] = ()
    require_runtime_verified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("profile must be a non-empty string")
        for name in ("features", "settings", "approximations"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in values
            ):
                raise TypeError(f"{name} must be a tuple of non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        if not isinstance(self.configuration, tuple) or any(
            not isinstance(item, CapabilityCondition) for item in self.configuration
        ):
            raise TypeError("configuration must be a tuple of CapabilityCondition")
        keys = [item.key for item in self.configuration]
        if len(keys) != len(set(keys)) or "profile" in keys:
            raise ValueError("configuration keys must be unique; use the profile field for profile")
        if type(self.require_runtime_verified) is not bool:
            raise TypeError("require_runtime_verified must be bool")
        if not set(self.approximations) <= set(self.features) | set(self.settings):
            raise ValueError(
                "approximation consent must name an explicitly required feature/setting"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticRequirements:
        payload = dict(value)
        for name in ("features", "settings", "approximations"):
            payload[name] = tuple(payload.get(name, ()))
        payload["configuration"] = tuple(
            CapabilityCondition(**item) for item in payload.get("configuration", ())
        )
        return cls(**payload)


def validate_semantic_requirements(
    capabilities: CapabilityReport,
    requirements: SemanticRequirements,
    import_report: ImportReport | None = None,
) -> None:
    """Validate declarations and optional construction readback without touching an SDK.

    Use without a report for declaration-only preflight (``settings`` must then
    be empty). A successful source declaration check is not runtime evidence.
    """
    if not isinstance(requirements, SemanticRequirements):
        raise TypeError("requirements must be SemanticRequirements")
    backend = capabilities.scope.adapter

    def reject(field: str, reason: str) -> None:
        raise SemanticValidationError(
            f"{backend}/{requirements.profile}: semantic request {field!r} rejected: {reason}. "
            "Select a declared supported profile/feature or provide matching verification; "
            "authorize only documented approximations by their specific key."
        )

    if capabilities.scope.profile != requirements.profile:
        reject("profile", f"declarations describe {capabilities.scope.profile!r}")
    if import_report is not None and (
        import_report.backend != backend or import_report.profile != requirements.profile
    ):
        reject(
            "import_report",
            f"actual backend/profile {import_report.backend}/{import_report.profile} "
            "does not match the declaration",
        )
    configuration = {item.key: item.value for item in requirements.configuration}
    for feature in requirements.features:
        declaration = capabilities.get(feature, configuration=configuration)
        if declaration.support in (SupportLevel.UNKNOWN, SupportLevel.UNSUPPORTED):
            reject(feature, f"{declaration.support.value}: {declaration.reason}")
        if (
            declaration.support == SupportLevel.APPROXIMATE
            and feature not in requirements.approximations
        ):
            reject(feature, "approximation requires explicit consent: " + declaration.reason)
        if requirements.require_runtime_verified and not declaration.runtime_verified(
            capabilities.scope
        ):
            reject(feature, "no passing runtime evidence matches the complete version/device scope")
    if not requirements.settings:
        return
    if import_report is None:
        reject("import_report", "required materialization settings have no report")
        return
    for name in requirements.settings:
        matches = [item for item in import_report.fields if item.field == name]
        if not matches:
            reject(name, "no materialization field recorded")
        for item in matches:
            location = (
                f"entity={item.scope.entity!r}, env_ids={item.scope.env_ids!r}, "
                f"variant={item.scope.variant!r}; sources="
                + repr(tuple(source.source for source in item.provenance))
            )
            if item.difference in ("unknown", "not_applicable") or item.effective is None:
                reject(name, f"effective value is {item.difference}: {item.reason}; {location}")
            if item.difference == "approximate" and name not in requirements.approximations:
                reject(
                    name,
                    "materialized approximation requires explicit consent: "
                    + item.reason
                    + "; "
                    + location,
                )
