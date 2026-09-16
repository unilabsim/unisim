"""Worker-origin report structure; mock tests make no runtime support claims."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.adapters.isaacgym.test_fixed_variants import _make_backend, _write_variants
from unisim.backend.subprocess_ipc.backend import SubprocessWorkerError


def test_legacy_worker_unknown_and_versioned_worker_origin(tmp_path: Path) -> None:
    backend = _make_backend(_write_variants(tmp_path), (0, 1, 2, 0), tmp_path / "init.json")
    try:
        backend.materialize()
        assert all(item.effective is None for item in backend.get_import_report().fields)
        backend._capture_import_report(
            {
                "configuration_report": {
                    "schema_version": 1,
                    "effective": {
                        "dt": 0.01,
                        "gravity": [0.0, 0.0, -9.81],
                        "body_mass": {
                            "names": ["body"],
                            "per_env_values": [[1.0], [2.0], [3.0], [4.0]],
                        },
                    },
                    "engine_readback": ["gravity"],
                }
            }
        )
        report = backend.get_import_report()
        gravity = [item for item in report.fields if item.field == "gravity"]
        assert gravity[0].scope.env_ids == (0, 3)
        assert gravity[1].scope.env_ids == (1,)
        assert gravity[0].provenance[-1].kind == "engine_readback"
        masses = [item for item in report.fields if item.field == "body_mass"]
        assert masses[0].effective["per_env_values"] == ((1.0,), (4.0,))
        assert masses[0].effective["env_ids"] == (0, 3)
        assert masses[1].effective["per_env_values"] == ((2.0,),)
        with pytest.raises(SubprocessWorkerError, match="schema version"):
            backend._capture_import_report({"configuration_report": {"schema_version": 99}})
    finally:
        backend.close()


def test_mismatched_report_version_closes_worker(tmp_path: Path, monkeypatch) -> None:
    backend = _make_backend(_write_variants(tmp_path), (0, 1, 2), tmp_path / "init.json")
    request = backend._request

    def changed_meta(command, payload=None, **kwargs):
        response = request(command, payload, **kwargs)
        if command == "INIT":
            response["configuration_report"] = {"schema_version": 9}
        return response

    monkeypatch.setattr(backend, "_request", changed_meta)
    with pytest.raises(SubprocessWorkerError, match="schema version"):
        backend.materialize()
    assert backend._closed
    assert backend._proc is None
    assert backend._shm_handles == {}


def test_fixed_worker_gravity_requires_consent_and_solver_stays_unknown(tmp_path: Path) -> None:
    from dataclasses import replace

    from unisim.capabilities import CapabilityReport, CapabilityScope
    from unisim.validation import (
        SemanticRequirements,
        SemanticValidationError,
        validate_semantic_requirements,
    )

    backend = _make_backend(_write_variants(tmp_path), (0, 1, 2), tmp_path / "init.json")
    try:
        backend.materialize()
        for metadata in backend._get_fixed_variant_metadata():
            metadata.source_options.update(gravity="0 0 -2", solver="Newton")
        backend._capture_import_report(
            {
                "configuration_report": {
                    "schema_version": 1,
                    "effective": {"gravity": [0.0, 0.0, -9.81], "solver": "PhysX solver_type=1"},
                }
            }
        )
        report = backend.get_import_report()
        capabilities = CapabilityReport(CapabilityScope("isaacgym"), ())
        requirements = SemanticRequirements(settings=("gravity",))
        with pytest.raises(SemanticValidationError, match="approximation"):
            validate_semantic_requirements(capabilities, requirements, report)
        validate_semantic_requirements(
            capabilities, replace(requirements, approximations=("gravity",)), report
        )
        with pytest.raises(SemanticValidationError, match="unknown"):
            validate_semantic_requirements(
                capabilities,
                SemanticRequirements(settings=("solver",), approximations=("solver",)),
                report,
            )
    finally:
        backend.close()


def test_authored_inertials_preserved_without_inference_or_false_equivalence(
    tmp_path: Path,
) -> None:
    sources = _write_variants(tmp_path)
    for index, source in enumerate(sources):
        source.write_text(
            source.read_text().replace(
                '<joint name="tool_pitch"',
                f'<inertial mass="{index + 1}" diaginertia="1 2 3" pos="0 0 .1"/>'
                '<joint name="tool_pitch"',
            )
        )
    backend = _make_backend(sources, (0, 1, 2), tmp_path / "init.json")
    try:
        backend.materialize()
        metadata = backend._get_fixed_variant_metadata()
        assert metadata[1].source_inertials[0]["attributes"]["mass"] == "2"
        backend._capture_import_report(
            {
                "configuration_report": {
                    "schema_version": 1,
                    "effective": {
                        "body_mass": {"names": ["tool"], "per_env_values": [[1.0], [2.0], [3.0]]}
                    },
                    "engine_readback": ["body_mass"],
                }
            }
        )
        masses = [
            field for field in backend.get_import_report().fields if field.field == "body_mass"
        ]
        assert masses[1].requested["authored_inertials"][0]["body"] == "tool"
        assert masses[1].requested["authored_inertials"][0]["attributes"]["mass"] == "2"
        assert masses[1].effective["per_env_values"] == ((2.0,),)
        assert masses[1].difference == "unknown"
        assert masses[1].provenance[-1].kind == "engine_readback"
    finally:
        backend.close()
