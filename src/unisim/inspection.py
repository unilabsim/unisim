"""Detached, cold-path configuration snapshots; no optional SDK imports.

Values describe construction/materialization, never the current state after DR.
A compiled source model includes compiler defaults and adapter scene composition;
its provenance deliberately does not claim to be the literal original XML.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import Any, Literal

Difference = Literal["exact", "overridden", "approximate", "unknown", "not_applicable"]
ProvenanceKind = Literal["source", "engine_readback", "adapter_setting", "unverified"]


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("configuration value keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    raise TypeError("configuration values must contain finite JSON primitives, not SDK objects")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ConfigurationScope:
    """Entity labels and explicit environment/variant applicability.

    ``env_ids=None`` means all environments; an empty tuple means no assigned
    environments. Entity labels use the engine model's ordering when applicable.
    """

    entity: str = "scene"
    env_ids: tuple[int, ...] | None = None
    variant: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.entity, str) or not self.entity:
            raise ValueError("configuration scope entity must be nonempty")
        if self.variant is not None and not isinstance(self.variant, str):
            raise TypeError("configuration variant must be a string or None")
        if self.env_ids is not None:
            object.__setattr__(self, "env_ids", tuple(self.env_ids))
            if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.env_ids):
                raise ValueError("configuration environment ids must be nonnegative integers")
            if len(set(self.env_ids)) != len(self.env_ids):
                raise ValueError("configuration environment ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {"entity": self.entity, "env_ids": _plain(self.env_ids), "variant": self.variant}


@dataclass(frozen=True)
class ConfigurationProvenance:
    """Origin of a value, distinct from capability test/verification evidence."""

    kind: ProvenanceKind
    source: str

    def __post_init__(self) -> None:
        if self.kind not in {"source", "engine_readback", "adapter_setting", "unverified"}:
            raise ValueError(f"unknown configuration provenance kind: {self.kind!r}")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("configuration provenance requires a source")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "source": self.source}


@dataclass(frozen=True)
class ConfigurationField:
    """One requested/effective comparison, with explicit unresolved semantics."""

    field: str
    requested: Any = None
    effective: Any = None
    difference: Difference = "unknown"
    provenance: tuple[ConfigurationProvenance, ...] = ()
    scope: ConfigurationScope = dataclass_field(default_factory=ConfigurationScope)
    unit: str | None = None
    frame: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field:
            raise ValueError("configuration field name must be nonempty")
        if self.difference not in {
            "exact",
            "overridden",
            "approximate",
            "unknown",
            "not_applicable",
        }:
            raise ValueError(f"unknown configuration difference: {self.difference!r}")
        if not isinstance(self.scope, ConfigurationScope):
            raise TypeError("configuration scope must be a ConfigurationScope")
        if not isinstance(self.reason, str):
            raise TypeError("configuration reason must be a string")
        if any(
            value is not None and not isinstance(value, str) for value in (self.unit, self.frame)
        ):
            raise TypeError("configuration unit and frame must be strings or None")
        same_value = self.requested is self.effective
        object.__setattr__(self, "requested", _freeze(self.requested))
        object.__setattr__(
            self, "effective", self.requested if same_value else _freeze(self.effective)
        )
        object.__setattr__(self, "provenance", tuple(self.provenance))
        if any(not isinstance(p, ConfigurationProvenance) for p in self.provenance):
            raise TypeError("configuration provenance must contain ConfigurationProvenance records")
        if self.difference in {"exact", "overridden", "approximate"}:
            if self.requested is None or self.effective is None:
                raise ValueError("known differences require both requested and effective values")
            if not any(p.kind in {"engine_readback", "adapter_setting"} for p in self.provenance):
                raise ValueError("effective values require engine or adapter provenance")
        if self.difference == "exact" and self.requested != self.effective:
            raise ValueError("exact configuration values must match")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "requested": _plain(self.requested),
            "effective": _plain(self.effective),
            "difference": self.difference,
            "provenance": [p.to_dict() for p in self.provenance],
            "scope": self.scope.to_dict(),
            "unit": self.unit,
            "frame": self.frame,
            "reason": self.reason,
        }


REPORT_FIELDS = (
    "solver",
    "integrator",
    "dt",
    "gravity",
    "actuator_mapping",
    "collision_filter",
    "body_mass",
    "body_inertia",
    "sensors",
)
_FIELD_UNITS = {"dt": "s", "gravity": "m/s^2", "body_mass": "kg", "body_inertia": "kg*m^2"}
_FIELD_FRAMES = {"gravity": "world", "body_inertia": "body principal inertia frame"}


@dataclass(frozen=True)
class ImportReport:
    """Serializable initial configuration snapshot, detached from native models."""

    backend: str
    fields: tuple[ConfigurationField, ...] = ()
    profile: str = "default"
    lifecycle: Literal["construction", "materialization"] = "construction"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(f"unsupported import report schema version: {self.schema_version}")
        if any(not isinstance(value, str) or not value for value in (self.backend, self.profile)):
            raise ValueError("import report requires backend and profile")
        if self.lifecycle not in {"construction", "materialization"}:
            raise ValueError("invalid import report lifecycle")
        object.__setattr__(self, "fields", tuple(self.fields))
        if any(not isinstance(item, ConfigurationField) for item in self.fields):
            raise TypeError("import report fields must contain ConfigurationField records")

    @classmethod
    def unknown(cls, backend: str, profile: str = "default") -> ImportReport:
        return cls(
            backend,
            tuple(
                ConfigurationField(
                    name, reason="Adapter has not provided an effective configuration snapshot."
                )
                for name in REPORT_FIELDS
            ),
            profile=profile,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "profile": self.profile,
            "lifecycle": self.lifecycle,
            "fields": [item.to_dict() for item in self.fields],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ImportReport:
        fields = []
        for item in value["fields"]:
            record = dict(item)
            record["scope"] = ConfigurationScope(**record["scope"])
            record["provenance"] = tuple(ConfigurationProvenance(**p) for p in record["provenance"])
            fields.append(ConfigurationField(**record))
        return cls(
            backend=value["backend"],
            fields=tuple(fields),
            profile=value["profile"],
            lifecycle=value["lifecycle"],
            schema_version=value["schema_version"],
        )


def compare_configuration(
    backend: str,
    requested: Mapping[str, Any],
    effective: Mapping[str, Any],
    *,
    source: str,
    effective_source: str,
    scope: ConfigurationScope | None = None,
    effective_kind: ProvenanceKind = "engine_readback",
    lifecycle: Literal["construction", "materialization"] = "construction",
) -> ImportReport:
    """Assemble observations already obtained on the adapter's cold path."""
    fields = []
    for name in REPORT_FIELDS:
        before, after = requested.get(name), effective.get(name)
        difference: Difference = "unknown"
        reason = "Requested value or engine adoption could not be verified."
        if before is not None and after is not None:
            # Native readers already return JSON-shaped values. Avoid allocating
            # two discarded frozen trees for the common identical/equal case;
            # the field below still validates and detaches every input. The
            # fallback preserves list/tuple normalization for mixed inputs.
            equal = before is after or before == after or _freeze(before) == _freeze(after)
            difference = "exact" if equal else "overridden"
            reason = "" if difference == "exact" else "Adapter configuration differs from source."
        provenance = [ConfigurationProvenance("source", source)]
        provenance.append(
            ConfigurationProvenance(
                effective_kind if after is not None else "unverified", effective_source
            )
        )
        fields.append(
            ConfigurationField(
                name,
                before,
                after,
                difference,
                tuple(provenance),
                scope or ConfigurationScope(),
                _FIELD_UNITS.get(name),
                _FIELD_FRAMES.get(name),
                reason,
            )
        )
    return ImportReport(backend, tuple(fields), lifecycle=lifecycle)


def mujoco_actuator_configuration(model: Any) -> dict[str, Any]:
    """Snapshot only actuator tables when applying an actuator-only override."""
    return {
        "names": [str(model.actuator(i).name or f"#{i}") for i in range(model.nu)],
        "trntype": model.actuator_trntype.tolist(),
        "trnid": model.actuator_trnid.tolist(),
        "gear": model.actuator_gear.tolist(),
        "gainprm": model.actuator_gainprm.tolist(),
        "biasprm": model.actuator_biasprm.tolist(),
    }


def mujoco_model_configuration(model: Any, sdk: Any) -> dict[str, Any]:
    """Read an existing compiled MuJoCo model; never parse or load an SDK."""

    def names(kind: str, count: int) -> list[str]:
        return [str(getattr(model, kind)(i).name or f"#{i}") for i in range(count)]

    return {
        "solver": str(sdk.mjtSolver(int(model.opt.solver)).name),
        "integrator": str(sdk.mjtIntegrator(int(model.opt.integrator)).name),
        "dt": float(model.opt.timestep),
        "gravity": model.opt.gravity.tolist(),
        "actuator_mapping": mujoco_actuator_configuration(model),
        "collision_filter": {
            "geom_names": names("geom", model.ngeom),
            "contype": model.geom_contype.tolist(),
            "conaffinity": model.geom_conaffinity.tolist(),
            "exclude_signature": model.exclude_signature.tolist(),
            "pair_geom1": model.pair_geom1.tolist(),
            "pair_geom2": model.pair_geom2.tolist(),
            "disableflags": int(model.opt.disableflags),
        },
        "body_mass": {"names": names("body", model.nbody), "values": model.body_mass.tolist()},
        "body_inertia": {
            "names": names("body", model.nbody),
            "values": model.body_inertia.tolist(),
            "ipos": model.body_ipos.tolist(),
            "iquat_wxyz": model.body_iquat.tolist(),
        },
        "sensors": {
            "names": names("sensor", model.nsensor),
            "type": model.sensor_type.tolist(),
            "objtype": model.sensor_objtype.tolist(),
            "objid": model.sensor_objid.tolist(),
            "dim": model.sensor_dim.tolist(),
        },
    }
