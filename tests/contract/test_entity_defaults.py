"""Consumer-facing defaults are per-variant snapshots, never environment-zero broadcasts."""

from pathlib import Path

import numpy as np
import pytest

from unisim import create_backend
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
from unisim.scene import SceneCfg


def _scene(tmp_path: Path) -> SceneCfg:
    variants = []
    for index, ref in enumerate((0.2, 0.4)):
        path = tmp_path / f"variant{index}.xml"
        path.write_text(
            '<mujoco><compiler angle="radian"/><worldbody><body name="base">'
            '<freejoint/><geom name="base_geom" type="sphere" size=".1" mass="1"/>'
            '<body name="tip"><joint name="hinge"/>'
            '<geom name="tip_geom" type="sphere" size=".1" mass=".1"/>'
            '</body></body></worldbody><keyframe>'
            f'<key name="start" qpos="0 0 0 1 0 0 0 {ref}"/></keyframe></mujoco>'
        )
        variants.append(ModelSourceDescriptor(str(path)))
    return SceneCfg(
        default_keyframe_name="start",
        entity_assets=(
            SceneEntitySpec(
                "object", variants[0], initial_state=EntityInitialState(position=(0.0, 0.0, 1.0))
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object", FixedVariantPlan(np.array([1, 1, 0, 1, 0]), tuple(variants))
        ),
    )


@pytest.mark.parametrize("name", ["mujoco", "mjwarp", "isaacgym", "isaacsim"])
def test_defaults_follow_selected_variant_rows_without_reset_or_aliasing(tmp_path, name):
    pytest.importorskip("mujoco")
    if name == "mujoco":
        mjbatch = pytest.importorskip("mjbatch")
        if not hasattr(mjbatch.VariantPack, "builder"):
            pytest.skip("mjbatch VariantPack builder API is required")
    if name == "mjwarp":
        pytest.importorskip("mujoco_warp")
        warp = pytest.importorskip("warp")
        warp.init()
        if not warp.is_cuda_available():
            pytest.skip("CUDA required for actual MJWarp construction")
    owner = create_backend(name, _scene(tmp_path), num_envs=5, sim_dt=0.002)
    try:
        # Isaac defaults use cold source compilation: no worker starts just to
        # read them. This test does not establish native IsaacSim evidence.
        defaults = owner.get_entity_default_state("object", [4, 1])
        np.testing.assert_allclose(defaults["joint_positions"][:, 0], [0.2, 0.4], atol=1e-6)
        np.testing.assert_allclose(defaults["root_pose"][:, 2], 1)
        defaults["joint_positions"][:] = 9
        np.testing.assert_allclose(
            owner.get_entity_default_state("object")["joint_positions"][:, 0],
            [0.4, 0.4, 0.2, 0.4, 0.2],
            atol=1e-6,
        )
        assert owner.get_entity_default_state("object", [])["root_pose"].shape == (0, 7)
        for invalid in ([True], [-1], [5], [1, 1], [1.5]):
            with pytest.raises(ValueError):
                owner.get_entity_default_state("object", invalid)
        if name == "mujoco":
            owner.materialize()
            owner.step(np.zeros((5, 0)), nsteps=2)
            np.testing.assert_allclose(
                owner.get_entity_default_state("object")["joint_positions"][:, 0],
                [0.4, 0.4, 0.2, 0.4, 0.2],
                atol=1e-6,
            )
    finally:
        close = getattr(owner, "close", None)
        if close is not None:
            close()
        owner.cleanup_scene_assets()
