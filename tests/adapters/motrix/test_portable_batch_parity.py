"""Real Motrix acceptance for portable batch and single-row parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("motrixsim")

from tests.adapters.motrix.test_portable_entities import _heavy_passive, _scene
from unisim.backend.motrix.backend import MotrixBackend
from unisim.dr.types import FixedVariantPlan
from unisim.entities import EntityStatePatch, EntityVariantBinding, SceneResetRequest
from unisim.scene import SceneCfg


def _fixed_variant_scene(tmp_path: Path, assignment: tuple[int, ...]) -> SceneCfg:
    scene = _scene(tmp_path)
    passive = next(entity for entity in scene.entity_assets if entity.name == "passive")
    scene.entity_variant = EntityVariantBinding(
        "passive",
        FixedVariantPlan(
            np.asarray(assignment, dtype=np.int32),
            (passive.source, _heavy_passive(tmp_path / "variant")),
        ),
    )
    return scene


def _unit_quaternion(wxyz: tuple[float, float, float, float]) -> np.ndarray:
    quaternion = np.asarray(wxyz, dtype=np.float32)
    return quaternion / np.linalg.norm(quaternion)


def test_fixed_variant_batch_matches_independent_single_env_runtimes(
    tmp_path: Path,
) -> None:
    assignment = (1, 1, 0, 1, 0)
    batch = MotrixBackend(
        _fixed_variant_scene(tmp_path / "batch", assignment),
        5,
        0.002,
        base_name="robot/base",
    )
    singles: list[MotrixBackend] = []
    try:
        for row, variant in enumerate(assignment):
            singles.append(
                MotrixBackend(
                    _fixed_variant_scene(tmp_path / f"single-{row}", (variant,)),
                    1,
                    0.002,
                    base_name="robot/base",
                )
            )

        layout = batch.get_scene_layout()
        passive_body = layout.get_entity("passive").body_ids[0]
        batch_mass = batch.get_body_mass()[:, passive_body]
        for row, single in enumerate(singles):
            np.testing.assert_allclose(
                batch_mass[row],
                single.get_body_mass()[0, passive_body],
                rtol=2e-6,
                atol=1e-7,
            )

        robot_joint_positions = np.asarray(
            [[0.31], [0.27], [0.35], [0.23], [0.29]], dtype=np.float32
        )
        robot_joint_velocities = np.asarray(
            [[-0.04], [-0.08], [0.02], [0.11], [-0.06]], dtype=np.float32
        )
        passive_positions = np.asarray(
            [
                [1.10, -0.15, 2.05],
                [1.18, 0.12, 1.96],
                [0.92, -0.08, 2.11],
                [1.26, 0.21, 1.89],
                [0.84, 0.05, 2.18],
            ],
            dtype=np.float32,
        )
        passive_orientations = np.stack(
            [
                _unit_quaternion(quaternion)
                for quaternion in (
                    (0.9, 0.1, 0.2, 0.3),
                    (0.8, -0.2, 0.4, 0.2),
                    (0.7, 0.3, -0.2, 0.5),
                    (0.85, 0.1, 0.3, -0.25),
                    (0.75, -0.35, 0.2, 0.3),
                )
            ]
        )
        passive_root_pose = np.concatenate((passive_positions, passive_orientations), axis=1)
        passive_root_velocities = np.asarray(
            [
                [0.12, -0.07, 0.05, 0.22, -0.13, 0.31],
                [-0.09, 0.14, 0.08, 0.18, 0.25, -0.22],
                [0.06, 0.11, -0.13, -0.27, 0.16, 0.24],
                [0.18, -0.05, 0.11, 0.31, -0.19, 0.08],
                [-0.14, 0.09, 0.16, 0.12, 0.28, -0.31],
            ],
            dtype=np.float32,
        )
        passive_joint_positions = np.asarray(
            [[0.37], [-0.21], [0.44], [0.12], [-0.38]], dtype=np.float32
        )
        passive_joint_velocities = np.asarray(
            [[0.62], [-0.48], [0.75], [0.29], [-0.57]], dtype=np.float32
        )
        object_positions = np.asarray(
            [
                [2.20, 0.15, 1.90],
                [2.32, -0.12, 2.04],
                [2.08, 0.24, 1.82],
                [2.41, 0.06, 2.15],
                [1.96, -0.21, 1.94],
            ],
            dtype=np.float32,
        )
        object_orientations = np.stack(
            [
                _unit_quaternion(quaternion)
                for quaternion in (
                    (0.8, 0.2, 0.4, 0.2),
                    (0.9, -0.1, 0.3, -0.2),
                    (0.75, 0.25, -0.3, 0.4),
                    (0.85, -0.25, 0.15, 0.35),
                    (0.7, 0.4, 0.2, -0.3),
                )
            ]
        )
        object_root_pose = np.concatenate((object_positions, object_orientations), axis=1)
        object_root_velocities = np.asarray(
            [
                [0.21, -0.13, 0.07, 0.14, -0.24, 0.33],
                [-0.17, 0.22, 0.11, 0.28, 0.09, -0.19],
                [0.09, 0.16, -0.21, -0.22, 0.27, 0.14],
                [0.26, -0.08, 0.18, 0.34, -0.12, 0.07],
                [-0.19, 0.12, 0.24, 0.11, 0.31, -0.28],
            ],
            dtype=np.float32,
        )

        batch.reset_entities(
            SceneResetRequest(
                tuple(range(5)),
                (
                    EntityStatePatch(
                        "robot",
                        joint_positions=robot_joint_positions,
                        joint_velocities=robot_joint_velocities,
                    ),
                    EntityStatePatch(
                        "passive",
                        root_pose=passive_root_pose,
                        root_velocity=passive_root_velocities,
                        joint_positions=passive_joint_positions,
                        joint_velocities=passive_joint_velocities,
                    ),
                    EntityStatePatch(
                        "object",
                        root_pose=object_root_pose,
                        root_velocity=object_root_velocities,
                    ),
                ),
            )
        )
        for row, single in enumerate(singles):
            single.reset_entities(
                SceneResetRequest(
                    (0,),
                    (
                        EntityStatePatch(
                            "robot",
                            joint_positions=robot_joint_positions[row : row + 1],
                            joint_velocities=robot_joint_velocities[row : row + 1],
                        ),
                        EntityStatePatch(
                            "passive",
                            root_pose=passive_root_pose[row : row + 1],
                            root_velocity=passive_root_velocities[row : row + 1],
                            joint_positions=passive_joint_positions[row : row + 1],
                            joint_velocities=passive_joint_velocities[row : row + 1],
                        ),
                        EntityStatePatch(
                            "object",
                            root_pose=object_root_pose[row : row + 1],
                            root_velocity=object_root_velocities[row : row + 1],
                        ),
                    ),
                )
            )

        controls = np.asarray(
            [[0.41], [0.33], [0.27], [0.48], [0.36]], dtype=np.float32
        )
        for _ in range(10):
            batch.step(controls)
            for row, single in enumerate(singles):
                single.step(controls[row : row + 1])

        for row, single in enumerate(singles):
            for entity in batch.get_entity_names():
                batch_state = batch.get_entity_state(entity)
                single_state = single.get_entity_state(entity)
                assert batch_state.keys() == single_state.keys()
                for field, expected in single_state.items():
                    np.testing.assert_allclose(
                        np.asarray(batch_state[field])[row],
                        np.asarray(expected)[0],
                        rtol=2e-6,
                        atol=2e-6,
                    )
    finally:
        for single in singles:
            single.close()
        batch.close()
