"""Opt-in real-worker acceptance for the legacy isaacsim scene path (#141).

The arbitrary-qpos ``set_state`` reset path broke on 1.4.3 and was fixed by
the 1.5.0 legacy-runtime refactor; this test locks the published post-reset
body state against MuJoCo forward kinematics.

Run with UNISIM_TEST_ISAACSIM_SCENE=1 after provisioning the dedicated SDK.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACSIM_SCENE") != "1",
    reason="set UNISIM_TEST_ISAACSIM_SCENE=1 for real IsaacSim scene acceptance",
)

_MODEL = """<mujoco><worldbody><body name="base" pos="0 0 1"><freejoint name="root"/>
<inertial pos=".1 0 0" mass="1" diaginertia=".1 .1 .1"/>
<geom name="base_geom" type="box" size=".08 .08 .08" mass="1"/>
<body name="active_link" pos="0 0 .2"><joint name="active" axis="0 1 0"/>
<inertial pos=".05 0 0" mass=".3" diaginertia=".01 .01 .01"/>
<geom name="active_geom" size=".05" mass=".3"/></body>
<body name="passive_link" pos="0 0 .4"><joint name="passive" axis="0 0 1"/>
<inertial pos="0 .05 0" mass=".3" diaginertia=".01 .01 .01"/>
<geom name="passive_geom" size=".05" mass=".3"/></body>
</body></worldbody><actuator><position name="drive" joint="active" kp="20" kv="2"/>
</actuator><keyframe><key name="home" qpos="0 0 1 1 0 0 0 .15 -.2"/></keyframe></mujoco>"""


def test_native_post_reset_body_state_matches_mujoco_fk(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from unisim import create_backend
    from unisim.scene import SceneCfg

    path = tmp_path / "legacy.xml"
    path.write_text(_MODEL)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    bodies = ["base", "active_link", "passive_link"]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in bodies]

    def reference(qpos, qvel):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        lin_origin, ang = [], []
        for b in body_ids:
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)
            ang.append(vel6[:3].copy())
            # mj_objectVelocity reports the COM velocity; the isaacsim legacy
            # channel publishes the link-origin velocity.
            lin_origin.append(vel6[3:] - np.cross(vel6[:3], data.xipos[b] - data.xpos[b]))
        return (
            np.asarray([data.xpos[b] for b in body_ids]),
            np.asarray([data.xquat[b] for b in body_ids]),
            np.asarray(lin_origin),
            np.asarray(ang),
        )

    backend = create_backend(
        "isaacsim",
        SceneCfg(model_file=str(path)),
        num_envs=2,
        sim_dt=0.002,
        base_name="base",
        worker_timeout_s=300,
    )
    try:
        backend.materialize()
        ids = backend.get_body_ids(bodies)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key)
        pos, quat, _, _ = reference(data.qpos.copy(), np.zeros(model.nv))
        np.testing.assert_allclose(backend.get_body_pos_w(ids), np.tile(pos, (2, 1, 1)), atol=1e-4)
        np.testing.assert_allclose(
            np.abs(np.sum(backend.get_body_quat_w(ids) * quat[None, :, :], axis=-1)),
            1.0,
            atol=1e-4,
        )

        # Arbitrary motion-frame state with nonzero root velocities.
        qpos = np.array([0.1, -0.05, 1.2, np.sqrt(0.5), 0, np.sqrt(0.5), 0, 0.35, -0.5])
        qvel = np.array([0.2, -0.1, 0.05, 0.3, -0.2, 0.4, 0.6, -0.7])
        backend.set_state(
            np.array([0, 1], dtype=np.int32),
            np.tile(qpos, (2, 1)).astype(np.float32),
            np.tile(qvel, (2, 1)).astype(np.float32),
        )
        pos, quat, lin_origin, ang = reference(qpos, qvel)
        np.testing.assert_allclose(backend.get_body_pos_w(ids), np.tile(pos, (2, 1, 1)), atol=1e-4)
        np.testing.assert_allclose(
            np.abs(np.sum(backend.get_body_quat_w(ids) * quat[None, :, :], axis=-1)),
            1.0,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            backend.get_body_lin_vel_w(ids), np.tile(lin_origin, (2, 1, 1)), atol=1e-4
        )
        np.testing.assert_allclose(
            backend.get_body_ang_vel_w(ids), np.tile(ang, (2, 1, 1)), atol=1e-4
        )

        # The published pose must stay continuous across the first step.
        before = backend.get_body_pos_w(ids).copy()
        backend.step(np.zeros((2, 2), dtype=np.float32))
        np.testing.assert_allclose(backend.get_body_pos_w(ids), before, atol=0.02)
    finally:
        backend.close()
