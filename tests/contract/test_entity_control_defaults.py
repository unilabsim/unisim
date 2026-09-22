"""One entity reset can restore keyframe controls without resetting other rows."""

import numpy as np
import pytest

from unisim import EntityStatePatch, SceneResetRequest, create_backend


@pytest.mark.parametrize("name", ["mujoco", "mjwarp"])
def test_selected_default_controls_use_same_native_commit_and_preserve_other_rows(tmp_path, name):
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
            pytest.skip("CUDA required")
    from tests.adapters.mujoco.test_entity_runtime import _scene

    owner = create_backend(
        name, _scene(tmp_path, n=2, fixed_robot=True, key=True), num_envs=2, sim_dt=0.002
    )
    try:
        if name == "mujoco":
            owner.materialize()
        owner.step(np.full((2, 1), 0.8), nsteps=2)
        before = owner.get_state()
        patch = EntityStatePatch("robot", joint_positions=np.array([[0.4]]))
        owner.reset_entities(SceneResetRequest((1,), (patch,), restore_default_controls=True))
        np.testing.assert_allclose(owner.get_state("ctrl")["ctrl"], [[0.8], [0.3]], atol=1e-6)
        for field, values in before.items():
            np.testing.assert_array_equal(owner.get_state()[field][0], values[0])
        if name == "mujoco":
            np.testing.assert_allclose(owner._act_view[1], [0.1], atol=1e-6)
        else:
            np.testing.assert_allclose(owner._device_data.act.numpy()[1], [0.1], atol=1e-6)
        copy = owner.get_state("ctrl")["ctrl"]
        copy[:] = -99
        np.testing.assert_allclose(owner.get_state("ctrl")["ctrl"], [[0.8], [0.3]], atol=1e-6)
        owner.reset_entities(SceneResetRequest((1,), (patch,)))
        np.testing.assert_allclose(owner.get_state("ctrl")["ctrl"], [[0.8], [0]], atol=1e-6)
    finally:
        close = getattr(owner, "close", None)
        if close is not None:
            close()
        owner.cleanup_scene_assets()


@pytest.mark.parametrize("invalid", [1, "true", np.bool_(True)])
def test_default_control_intent_must_be_explicit_boolean(invalid):
    with pytest.raises(TypeError, match="restore_default_controls"):
        SceneResetRequest(
            (0,),
            (EntityStatePatch("robot", joint_positions=np.array([[0.0]])),),
            restore_default_controls=invalid,
        )
