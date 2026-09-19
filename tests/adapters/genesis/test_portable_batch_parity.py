"""Real Genesis CPU acceptance for portable batch and single-row parity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("genesis")
pytest.importorskip("torch")

from tests.adapters.genesis.test_portable_entities import _scene
from unisim.backend.genesis.backend import GenesisBackend
from unisim.entities import EntityStatePatch, SceneResetRequest


def _single_variant_scene(tmp_path: Path, variant: int):
    scene = _scene(tmp_path, assignment=(0,))
    binding = scene.entity_variant
    scene.entity_variant = replace(
        binding,
        plan=replace(
            binding.plan,
            variants=(binding.plan.variants[variant],),
        ),
    )
    return scene


def test_portable_batch_matches_independent_single_env_runtimes(tmp_path: Path) -> None:
    assignment = (0, 0, 0, 1, 1)
    batch = GenesisBackend(_scene(tmp_path / "batch", assignment=assignment), 5, 0.002)
    singles = tuple(
        GenesisBackend(_single_variant_scene(tmp_path / f"single-{row}", variant), 1, 0.002)
        for row, variant in enumerate(assignment)
    )
    try:
        batch.materialize()
        for single in singles:
            single.materialize()

        batch_layout = batch.get_scene_layout()
        object_body = batch_layout.get_entity("object").body_ids[0]
        batch_mass = batch.get_body_mass()[:, object_body]
        for row, single in enumerate(singles):
            single_mass = float(single.get_body_mass()[0, object_body])
            np.testing.assert_allclose(
                batch_mass[row], single_mass, rtol=2e-6, atol=1e-7
            )

        object_pose = np.asarray(
            [
                [2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 1.0, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
                [2.0, 0.0, 1.0, np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)],
            ],
            dtype=np.float32,
        )
        object_velocity = np.asarray(
            [[0.03, -0.02, 0.01, 0.2, -0.1, 0.3]] * 5, dtype=np.float32
        )

        batch.reset_entities(
            SceneResetRequest(
                tuple(range(5)),
                (
                    EntityStatePatch(
                        "robot",
                        joint_positions=np.full((5, 1), 0.35, dtype=np.float32),
                        joint_velocities=np.full((5, 1), -0.05, dtype=np.float32),
                    ),
                    EntityStatePatch(
                        "passive",
                        joint_positions=np.full((5, 1), 0.4, dtype=np.float32),
                        joint_velocities=np.full((5, 1), 0.7, dtype=np.float32),
                    ),
                    EntityStatePatch(
                        "object",
                        root_pose=object_pose,
                        root_velocity=object_velocity,
                    ),
                ),
            )
        )
        for row, backend in enumerate(singles):
            backend.reset_entities(
                SceneResetRequest(
                    (0,),
                    (
                        EntityStatePatch(
                            "robot",
                            joint_positions=np.full((1, 1), 0.35, dtype=np.float32),
                            joint_velocities=np.full((1, 1), -0.05, dtype=np.float32),
                        ),
                        EntityStatePatch(
                            "passive",
                            joint_positions=np.full((1, 1), 0.4, dtype=np.float32),
                            joint_velocities=np.full((1, 1), 0.7, dtype=np.float32),
                        ),
                        EntityStatePatch(
                            "object",
                            root_pose=object_pose[[row]],
                            root_velocity=object_velocity[[row]],
                        ),
                    ),
                )
            )

        batch_control = np.full((5, 1), 0.45, dtype=np.float32)
        single_control = np.full((1, 1), 0.45, dtype=np.float32)
        for _ in range(10):
            batch.step(batch_control)
            for single in singles:
                single.step(single_control)

        for row, single in enumerate(singles):
            for entity in ("robot", "passive", "object", "table"):
                batch_state = batch.get_entity_state(entity)
                single_state = single.get_entity_state(entity)
                for field, expected in single_state.items():
                    np.testing.assert_allclose(
                        np.asarray(batch_state[field])[row],
                        np.asarray(expected)[0],
                        rtol=2e-6,
                        atol=2e-6,
                    )
    finally:
        # Genesis permits one process-wide session; the final test in
        # test_portable_entities.py owns teardown so later native constructions
        # remain valid.
        pass
