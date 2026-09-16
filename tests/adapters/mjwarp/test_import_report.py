"""Actual device configuration reporting, with optional CUDA runtime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.adapters.mjwarp.test_backend import _make_backend


def test_report_reads_uploaded_device_configuration_once(tmp_path: Path, monkeypatch) -> None:
    backend = _make_backend(tmp_path)
    report = backend.get_import_report()
    fields = {item.field: item for item in report.fields}
    assert report.lifecycle == "materialization"
    assert fields["dt"].effective == pytest.approx(.01)
    assert fields["dt"].scope.env_ids == (0, 1)
    assert fields["body_mass"].effective["values"] == pytest.approx(
        backend._device_model.body_mass.numpy()[0].tolist()
    )
    assert fields["actuator_mapping"].effective is not None
    assert fields["collision_filter"].effective is not None
    assert fields["sensors"].effective is not None
    monkeypatch.setattr(backend, "_capture_import_report", lambda: pytest.fail("report reread"))
    assert backend.get_import_report() is report
    json.dumps(report.to_dict(), allow_nan=False)
