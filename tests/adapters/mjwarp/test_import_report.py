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


@pytest.mark.parametrize("assignment", [(0,) * 8, (0, 1, 0, 1), (0, 1)])
def test_report_transfers_representative_rows_and_preserves_variants(
    tmp_path: Path, monkeypatch, assignment: tuple[int, ...],
) -> None:
    import numpy as np
    import warp

    from tests.adapters.mjwarp.test_backend import MODEL
    from unisim import MjwarpBackend
    from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
    from unisim.scene import SceneCfg

    warp.init()
    if not warp.get_device().is_cuda:
        pytest.skip("report readback requires CUDA")
    variants = []
    for index in range(max(assignment) + 1):
        path = tmp_path / f"variant-{index}.xml"
        path.write_text(MODEL.replace("size='0.05 0.05 0.05'",
                                      f"size='0.05 0.05 0.05' mass='{index + 1}'")
                        .replace("<geom ", "<geom name='shape' ")
                        .replace("<motor ", "<motor name='drive' "))
        variants.append(ModelSourceDescriptor(str(path)))
    plan = FixedVariantPlan(variants=tuple(variants), assignment=np.array(assignment))
    capture = MjwarpBackend._capture_import_report
    mass_rows = []

    def capture_with_transfer_check(owner, requested):
        mass = owner._device_model.body_mass
        array_type = type(mass)
        numpy = array_type.numpy

        def observe(array):
            if array.dtype == mass.dtype and array.ndim == 2 and array.shape[1:] == mass.shape[1:]:
                mass_rows.append(array.shape[0])
            return numpy(array)

        with monkeypatch.context() as context:
            context.setattr(array_type, "numpy", observe)
            capture(owner, requested)

    monkeypatch.setattr(MjwarpBackend, "_capture_import_report", capture_with_transfer_check)
    backend = MjwarpBackend(
        SceneCfg(model_file=variants[0].model_file, fixed_variant_plan=plan),
        len(assignment), 0.01,
    )
    assert sum(mass_rows) == len(variants)
    fields = [item for item in backend.get_import_report().fields if item.field == "body_mass"]
    for item in fields:
        variant = int(item.scope.variant)
        assert item.scope.env_ids == tuple(i for i, value in enumerate(assignment)
                                           if value == variant)
        assert item.effective["values"][1] == pytest.approx(variant + 1)
    assert backend._fixed_variant_realization.report_requested == ()
