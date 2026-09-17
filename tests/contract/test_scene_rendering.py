"""Explicit native camera acceptance, separate from headless physics support."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from tests.contract.test_worker_scene_materialization import scene
from unisim import EntityStatePatch, SceneResetRequest, create_backend
from unisim.dr.types import FixedVariantPlan
from unisim.entities import EntityVariantBinding


@pytest.mark.parametrize("backend", ["isaacgym", "isaacsim"])
def test_native_capture_preserves_state_and_does_not_introduce_ground(tmp_path, backend):
    if os.environ.get("UNISIM_TEST_" + backend.upper() + "_RENDER") != "1":
        pytest.skip("native camera acceptance requires an explicit renderer opt-in")
    config = scene(tmp_path)
    config.entity_variant = EntityVariantBinding(
        "object", FixedVariantPlan(np.array([0, 1]), config.entity_variant.plan.variants)
    )
    records = []
    for mode in ("none", "record"):
        options = (
            dict(
                isaacsim_render_mode=mode,
                isaacsim_render_width=320,
                isaacsim_render_height=240,
                isaacsim_worker_timeout_s=120,
            )
            if backend == "isaacsim"
            else {}
        )
        owner = create_backend(backend, replace(config), num_envs=2, sim_dt=0.002, **options)
        try:
            owner.materialize()
            pose = owner.get_entity_state("object")["root_pose"][[1]]
            # No authored floor here. A hidden renderer ground would intersect
            # the object immediately and change its downward trajectory.
            pose[0, :3] = [3.0, 0.0, 0.03]
            owner.reset_entities(
                SceneResetRequest((1,), (EntityStatePatch("object", root_pose=pose),))
            )
            if mode == "record":
                owner.init_renderer(headless=True, capture=True, width=320, height=240)
            states = []
            for step in range(20):
                owner.step(np.zeros((2, 1), dtype=np.float32))
                before = owner.get_state(("qpos", "qvel", "ctrl"))
                if mode == "record" and step in (0, 5, 19):
                    frame = owner.capture_video_frame()
                    assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8
                    assert np.ptp(frame) > 0 and np.count_nonzero(frame) > 100
                    after = owner.get_state(("qpos", "qvel", "ctrl"))
                    for field in before:
                        np.testing.assert_array_equal(before[field], after[field])
                states.append(before)
            assert owner.get_entity_state("object")["root_pose"][1, 2] < 0.03
            records.append(states)
        finally:
            owner.close()
    for first, second in zip(*records, strict=True):
        for field in first:
            np.testing.assert_allclose(first[field], second[field], rtol=0, atol=1e-6)
