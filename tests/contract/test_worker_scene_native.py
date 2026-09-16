"""Public factory-to-native scene acceptance; optional vendor runtimes are explicit."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.contract.test_worker_scene_materialization import scene
from unisim import EntityStatePatch, SceneResetRequest, create_backend
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec


@pytest.mark.parametrize("backend", ["isaacgym", "isaacsim"])
def test_public_entity_factory_native_state_identity_and_reset(tmp_path: Path, backend: str):
    if os.environ.get("UNISIM_TEST_" + backend.upper() + "_SCENE") != "1":
        pytest.skip("set UNISIM_TEST_" + backend.upper() + "_SCENE=1 for vendor acceptance")
    n = 5 if backend == "isaacgym" else 2
    config = scene(tmp_path)
    assignment = np.array([1, 1, 0, 1, 0]) if n == 5 else np.array([0, 1])
    config.entity_variant = EntityVariantBinding(
        "object", FixedVariantPlan(assignment, config.entity_variant.plan.variants)
    )
    # A keyframe control deliberately differs from its joint position.
    robot_file = Path(config.entity_assets[0].source.model_file)
    robot_file.write_text(
        robot_file.read_text().replace(
            "</mujoco>", '<keyframe><key name="start" qpos=".1" ctrl=".35"/></keyframe></mujoco>'
        )
    )
    config.default_keyframe_name = "start"
    table = tmp_path / "table.xml"
    table.write_text(
        '<mujoco><worldbody><body name="base"><geom name="floor" type="box" '
        'size="2 2 .1" mass="5"/></body></worldbody></mujoco>'
    )
    config.entity_assets = (
        replace(
            config.entity_assets[0], initial_state=EntityInitialState(position=(0.0, 0.0, 0.5))
        ),
        config.entity_assets[1],
        SceneEntitySpec(
            "table",
            ModelSourceDescriptor(str(table)),
            kind="rigid",
            root_mode="fixed",
            initial_state=EntityInitialState(position=(0.0, 0.0, -0.1)),
        ),
        config.entity_assets[2],
    )
    options = {"isaacsim_worker_timeout_s": 240.0} if backend == "isaacsim" else {}
    owner = create_backend(backend, config, num_envs=n, sim_dt=0.002, **options)
    try:
        owner.materialize()
        assert owner.get_entity_names() == ("robot", "object", "table", "target")
        assert owner.num_actuators == 1
        np.testing.assert_allclose(owner.get_state("ctrl")["ctrl"], 0.35, atol=1e-6)
        initial = owner.get_state()
        np.testing.assert_allclose(
            owner.get_entity_state("object")["root_pose"][:, 2], 1.0, atol=1e-5
        )
        assert initial["qpos"].shape == (n, 8) and initial["qvel"].shape == (n, 7)
        if backend == "isaacgym":
            with pytest.raises(NotImplementedError, match="after joint reset"):
                owner.get_body_pos_w(owner.get_body_ids(["robot/tip"]))
        owner.step(np.full((n, 1), 0.6, dtype=np.float32), nsteps=3)
        before = owner.get_state()
        row = n - 1
        root = owner.get_entity_state("object")["root_pose"][[row]]
        root[:, 2] = 2.0
        owner.reset_entities(
            SceneResetRequest((row,), (EntityStatePatch("object", root_pose=root),))
        )
        after = owner.get_state()
        for key in before:
            np.testing.assert_array_equal(after[key][:row], before[key][:row])
        np.testing.assert_allclose(owner.get_state("ctrl")["ctrl"], 0.6, atol=1e-6)
        owner.step(np.full((n, 1), 0.6, dtype=np.float32), nsteps=1)
        assert owner.get_entity_state("object")["root_pose"][row, 2] > 1.99
        owner.reset(np.array([row], dtype=np.int32))
        control = owner.get_state("ctrl")["ctrl"]
        np.testing.assert_allclose(control[row], 0.35, atol=1e-6)
        np.testing.assert_allclose(control[:row], 0.6, atol=1e-6)
        owner.step(control, nsteps=1)
        assert owner.get_entity_state("object")["root_pose"][row, 2] < 1.001
        import mujoco

        playback = mujoco.MjModel.from_xml_path(owner.get_playback_model(row))
        assert playback.body("table/base").id > 0 and playback.body("target/base").id > 0
        np.testing.assert_allclose(
            playback.body("object/base").mass, (1.0, 3.0)[int(assignment[row])]
        )
        (tmp_path / "result.json").write_text(
            json.dumps(
                {
                    "backend": backend,
                    "result": "passed",
                    "assignment": assignment.tolist(),
                    "layout": owner.get_scene_layout().to_dict(),
                    "report": owner.get_import_report().to_dict(),
                },
                indent=2,
            )
        )
    finally:
        owner.close()
