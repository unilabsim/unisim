"""Serializable report schema and fail-closed provenance contracts."""

from __future__ import annotations

import json

import pytest

from unisim.inspection import (
    ConfigurationField,
    ConfigurationProvenance,
    ImportReport,
    compare_configuration,
)


def test_report_preserves_unknown_and_authorized_approximation() -> None:
    report = ImportReport(
        "example",
        (
            ConfigurationField(
                "collision_filter",
                {"pair_exclusions": ["a:b"]},
                {"self_collision": False},
                "approximate",
                (
                    ConfigurationProvenance("source", "asset/contact/exclude"),
                    ConfigurationProvenance("adapter_setting", "actor collision flag"),
                ),
                reason="Self-collision disabled instead of pair filtering",
            ),
            ConfigurationField("body_inertia", reason="Runtime cannot read inertia"),
        ),
    )
    assert ImportReport.from_dict(json.loads(json.dumps(report.to_dict()))) == report
    assert report.fields[1].effective is None


def test_report_rejects_handles_nonfinite_values_and_unsupported_versions() -> None:
    for value in (object(), float("nan"), float("inf")):
        with pytest.raises(TypeError):
            ConfigurationField("dt", effective=value)
    with pytest.raises(ValueError, match="schema version"):
        ImportReport("example", schema_version=2)
    with pytest.raises(ValueError, match="provenance"):
        ConfigurationField("dt", 1.0, 1.0, "exact")


def test_report_validates_nested_records_and_boolean_schema_version() -> None:
    with pytest.raises(ValueError, match="schema version"):
        ImportReport("example", schema_version=True)
    with pytest.raises(TypeError, match="ConfigurationField"):
        ImportReport("example", fields=({},))
    with pytest.raises(TypeError, match="ConfigurationScope"):
        ConfigurationField("dt", scope={})
    with pytest.raises(TypeError, match="ConfigurationProvenance"):
        ConfigurationField("dt", provenance=({},))


def test_equal_inputs_still_detach_and_normalize_nested_sequences() -> None:
    source = {"values": [[1.0, 2.0]]}
    report = compare_configuration(
        "fixture", {"body_mass": source}, {"body_mass": source},
        source="compiled source", effective_source="native readback",
    )
    item = next(field for field in report.fields if field.field == "body_mass")
    source["values"][0][0] = 99.0
    assert item.requested["values"] == ((1.0, 2.0),)
    assert item.effective["values"] == ((1.0, 2.0),)
    assert item.difference == "exact"
    mixed = compare_configuration(
        "fixture", {"gravity": [0, 0, -9.81]}, {"gravity": (0, 0, -9.81)},
        source="source", effective_source="readback",
    )
    assert next(field for field in mixed.fields if field.field == "gravity").difference == "exact"
