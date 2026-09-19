"""Public factory-to-native scene acceptance; optional vendor runtimes are explicit."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.contract.test_worker_scene_materialization import scene
from unisim import EntityStatePatch, SceneResetRequest, create_backend
from unisim.backend.base import PreStepControlOutput
from unisim.backend.isaacsim.dependencies import resolve_isaacsim_runtime
from unisim.backend.isaacsim.raw_usd_cache import (
    RawUSDCache,
    RoleUSDCache,
    resolve_raw_usd_cache_root,
)
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor, ResetRandomizationPayload
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
from unisim.scene import SceneCfg


def _final_operation_scene(tmp_path: Path) -> SceneCfg:
    """Author the final MJCF-only operation scene once for both backends."""

    robot = tmp_path / "robot.xml"
    robot.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="base">
              <geom name="base_geom" type="box" size=".06 .06 .05" mass=".8"/>
              <body name="finger" pos=".06 0 0">
                <joint name="hinge" axis="0 1 0" range="-1.2 1.2"/>
                <geom name="finger_geom" type="capsule" fromto="0 0 0 .18 0 0"
                      size=".018" mass=".3"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position name="drive" joint="hinge" kp="100" kv="5" forcerange="-40 40"/>
          </actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    object_sources = []
    for name, mass, half_size, inertia in (
        ("light", ".5", ".08", ".006"),
        ("heavy", "1.0", ".12", ".015"),
    ):
        source = tmp_path / f"{name}.xml"
        source.write_text(
            f"""
            <mujoco>
              <worldbody>
                <body name="base">
                  <freejoint/>
                  <inertial pos=".002 0 0" mass="{mass}"
                            diaginertia="{inertia} {inertia} {inertia}"/>
                  <geom name="shape" type="box" size="{half_size} {half_size} {half_size}"/>
                </body>
              </worldbody>
            </mujoco>
            """,
            encoding="utf-8",
        )
        object_sources.append(ModelSourceDescriptor(str(source)))

    table = tmp_path / "table.xml"
    table.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="base">
              <inertial pos="0 0 0" mass="5" diaginertia=".5 .5 .1"/>
              <geom name="surface" type="box" size=".6 .6 .05"/>
            </body>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    sensors = tmp_path / "contact-pairs.xml"
    sensors.write_text(
        """
        <mujoco>
          <sensor>
            <contact name="object_table" geom1="object/shape"
                     geom2="table/surface" data="force" reduce="netforce"/>
            <contact name="object_mirror" geom1="object/shape"
                     geom2="mirror/shape" data="force" reduce="netforce"/>
          </sensor>
        </mujoco>
        """,
        encoding="utf-8",
    )

    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                ModelSourceDescriptor(str(robot)),
                root_mode="fixed",
                initial_state=EntityInitialState(position=(-0.25, 0.0, 0.25)),
            ),
            SceneEntitySpec(
                "object",
                object_sources[0],
                kind="rigid",
                initial_state=EntityInitialState(position=(0.15, 0.0, 0.13)),
            ),
            SceneEntitySpec(
                "table",
                ModelSourceDescriptor(str(table)),
                kind="rigid",
                root_mode="fixed",
                initial_state=EntityInitialState(position=(0.0, 0.0, -0.05)),
            ),
            SceneEntitySpec(
                "mirror",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="object",
                initial_state=EntityInitialState(position=(1.0, 0.0, 0.13)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(np.array([1, 1, 0, 1, 0]), tuple(object_sources)),
        ),
        fragment_files=[str(sensors)],
    )


@pytest.mark.parametrize("backend", ["mujoco", "isaacsim"])
def test_final_integrated_mjcf_operation_scene_acceptance(tmp_path: Path, backend: str):
    if backend == "isaacsim" and os.environ.get("UNISIM_TEST_ISAACSIM_SCENE") != "1":
        pytest.skip("set UNISIM_TEST_ISAACSIM_SCENE=1 for final IsaacSim acceptance")

    assignment = np.asarray([1, 1, 0, 1, 0], dtype=np.int32)
    config = _final_operation_scene(tmp_path)
    options = {"isaacsim_worker_timeout_s": 240.0} if backend == "isaacsim" else {}
    owner = create_backend(backend, config, num_envs=5, sim_dt=0.005, **options)
    worker_metadata: dict = {}
    if backend == "isaacsim":
        original_bind = owner._bind_scene_metadata

        def bind_metadata(metadata):
            worker_metadata.update(metadata)
            original_bind(metadata)

        owner._bind_scene_metadata = bind_metadata

    try:
        owner.materialize()
        owner.reset()
        layout = owner.get_scene_layout()
        robot = layout.get_entity("robot")
        movable = layout.get_entity("object")
        mirror = layout.get_entity("mirror")
        object_body = movable.body_ids[0]

        # Source and native identity: the public layout retains independent
        # entities, the unbalanced assignment, one controlled joint and a
        # collision-disabled mirror.
        assert owner.get_entity_names() == ("robot", "object", "table", "mirror")
        assert tuple(config.entity_variant.plan.assignment) == tuple(assignment)
        assert (layout.nq, layout.nv, layout.nu, owner.num_actuators) == (8, 7, 1, 1)
        assert robot.kind == "articulation" and movable.kind == "rigid"
        assert mirror.root_mode == "kinematic"
        assert layout.ngeom == 5

        if backend == "isaacsim":
            assert owner.get_geom_names() == (
                "robot/base_geom",
                "robot/finger_geom",
                "object/shape",
                "table/surface",
                "mirror/shape",
            )
            np.testing.assert_array_equal(
                owner.get_geom_body_ids(),
                [
                    robot.body_ids[0],
                    robot.body_ids[1],
                    object_body,
                    layout.get_entity("table").body_ids[0],
                    mirror.body_ids[0],
                ],
            )
            contype, conaffinity = owner.get_geom_contact_masks()
            np.testing.assert_array_equal(contype, [1, 1, 1, 1, 0])
            np.testing.assert_array_equal(conaffinity, [1, 1, 1, 1, 0])
            np.testing.assert_allclose(
                owner.get_geom_friction()[:, 2, :],
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
            )

        import mujoco

        mass_table = owner.get_body_mass()
        coms = owner.get_body_ipos(env_ids=np.arange(5))
        assert np.all(np.isfinite(mass_table)) and np.all(np.isfinite(coms))
        row_masses = np.empty(5)
        for row in range(5):
            playback_source = owner.get_playback_model(row)
            playback = (
                mujoco.MjModel.from_xml_path(playback_source)
                if isinstance(playback_source, (str, Path))
                else playback_source
            )
            row_masses[row] = np.asarray(playback.body("object/base").mass).reshape(-1)[0]
            assert playback.geom("mirror/shape").contype == 0
            assert playback.geom("mirror/shape").conaffinity == 0
            if mass_table.ndim == 2:
                np.testing.assert_allclose(
                    mass_table[row, object_body],
                    row_masses[row],
                    rtol=2e-4,
                    atol=1e-6,
                )
            np.testing.assert_allclose(
                coms[row, object_body], playback.body_ipos[object_body], atol=1e-6
            )

        # Let both engines settle the same source-authored box/table pair.
        owner.step(np.zeros((5, 1), dtype=np.float32), nsteps=500)
        support = owner.get_sensor_data("object_table")
        no_contact = owner.get_sensor_data("object_mirror")
        assert support.shape == no_contact.shape == (5, 3)
        np.testing.assert_allclose(no_contact, 0.0, atol=0.25)
        support_tolerance = {"rtol": 0.06, "atol": 0.02} if backend == "mujoco" else {
            "rtol": 0.2,
            "atol": 0.1,
        }
        np.testing.assert_allclose(
            np.abs(support[:, 2]), row_masses * 9.81, **support_tolerance
        )
        np.testing.assert_allclose(
            support[:, :2], 0.0, atol=0.1 if backend == "mujoco" else 0.5
        )

        # Selected reset must teleport only row four to the mirror location;
        # the collision-free mirror cannot arrest its subsequent free fall.
        before = owner.get_state()
        mirror_pose = owner.get_entity_state("mirror")["root_pose"][[4]].copy()
        owner.reset_entities(
            SceneResetRequest(
                (4,),
                (
                    EntityStatePatch(
                        "object", root_pose=mirror_pose, root_velocity=np.zeros((1, 6))
                    ),
                )
            )
        )
        for field in before:
            np.testing.assert_array_equal(owner.get_state()[field][:4], before[field][:4])
        np.testing.assert_allclose(
            owner.get_entity_state("object")["root_pose"][4, :3], mirror_pose[0, :3], atol=1e-5
        )
        np.testing.assert_allclose(owner.get_sensor_data("object_table")[4], 0.0, atol=0.25)

        # A public pre-step callback observes fresh robot state once per physics
        # substep and recomputes control without widening the action contract.
        observed_joint_positions: list[np.ndarray] = []
        observed_object_velocities: list[np.ndarray] = []

        def controller(owner_, policy_ctrl):
            observed_joint_positions.append(
                owner_.get_entity_state("robot")["joint_positions"][:, 0].copy()
            )
            observed_object_velocities.append(
                owner_.get_entity_state("object")["root_velocity"][:, 2].copy()
            )
            return policy_ctrl + np.float32(0.02 * len(observed_joint_positions))

        owner.set_pre_step_control(controller)
        owner.step(np.full((5, 1), 0.5, dtype=np.float32), nsteps=4)
        owner.set_pre_step_control(None)
        assert len(observed_joint_positions) == 4
        assert observed_joint_positions[0].shape == (5,)
        assert len(observed_object_velocities) == 4
        np.testing.assert_allclose(observed_object_velocities[0][4], 0.0, atol=1e-5)
        assert all(values[4] < -1e-4 for values in observed_object_velocities[1:])
        np.testing.assert_allclose(
            owner.get_state("ctrl")["ctrl"], 0.5 + np.arange(1, 5)[-1] * 0.02, atol=1e-6
        )

        # State identity is public-view-consistent after reset and callback work.
        object_qpos = movable.root_qpos_indices
        object_qvel = movable.root_qvel_indices
        robot_qpos = robot.joints[0].qpos_indices
        object_state = owner.get_entity_state("object")
        np.testing.assert_allclose(
            object_state["root_pose"], owner.get_state()["qpos"][:, object_qpos], atol=1e-6
        )
        np.testing.assert_allclose(
            object_state["root_velocity"], owner.get_state()["qvel"][:, object_qvel], atol=1e-6
        )
        np.testing.assert_allclose(
            owner.get_entity_state("robot")["joint_positions"],
            owner.get_state()["qpos"][:, robot_qpos],
            atol=1e-6,
        )

        # Hover all objects away from the table. Exact per-row native mass readback
        # feeds gravity compensation; the staged plan is consumed after one step.
        hover_pose = np.tile((0.15, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0), (5, 1))
        owner.reset_entities(
            SceneResetRequest(
                tuple(range(5)),
                (
                    EntityStatePatch(
                        "object", root_pose=hover_pose, root_velocity=np.zeros((5, 6))
                    ),
                ),
            )
        )
        hover_force = np.zeros((5, 1, 3), dtype=np.float32)
        hover_force[:, 0, 2] = row_masses * 9.81
        owner.apply_body_force(np.asarray([object_body]), hover_force)
        hover_ctrl = np.full((5, 1), 0.58, dtype=np.float32)
        owner.step(hover_ctrl, nsteps=5)
        compensated = owner.get_entity_state("object")["root_velocity"][:, 2]
        np.testing.assert_allclose(
            compensated, 0.0, atol=0.012 if backend == "mujoco" else 0.02
        )

        owner.step(hover_ctrl, nsteps=1)
        first_free = owner.get_entity_state("object")["root_velocity"][:, 2]
        np.testing.assert_allclose(first_free, -9.81 * 0.005, rtol=0.12, atol=0.004)
        owner.step(hover_ctrl, nsteps=1)
        second_free = owner.get_entity_state("object")["root_velocity"][:, 2]
        np.testing.assert_allclose(second_free, -9.81 * 0.01, rtol=0.12, atol=0.006)

        if backend == "isaacsim":
            # The active PhysX GPU solver does not consume a reset-time mass
            # write until a later nonzero solver step. A reset must not advance
            # physics silently, so body mass fails closed before worker access.
            mass_table = owner.get_body_mass().copy()
            current = owner.get_state()
            with pytest.raises(NotImplementedError, match="body_mass"):
                owner.set_state(
                    np.array([4], dtype=np.intp),
                    np.asarray(current["qpos"])[[4]],
                    np.asarray(current["qvel"])[[4]],
                    ResetRandomizationPayload(body_mass=mass_table[[4]]),
                )
            np.testing.assert_array_equal(owner.get_body_mass(), mass_table)

            # Low friction on row zero and the source 1.0 material on row four
            # share the same variant geometry and initial sliding state; only
            # the selected material row may differ physically.
            friction_table = owner.get_geom_friction().copy()
            friction_table[0, 0, :2] = 0.11
            friction_table[0, 1, :2] = 0.12
            friction_table[0, 2, :2] = 0.05
            friction_table[0, 3, :2] = 0.05
            current = owner.get_state()
            owner.set_state(
                np.array([0], dtype=np.intp),
                np.asarray(current["qpos"])[[0]],
                np.asarray(current["qvel"])[[0]],
                ResetRandomizationPayload(geom_friction=friction_table[[0]]),
            )
            np.testing.assert_allclose(
                owner.get_geom_friction()[0, 2:4, :2], 0.05, rtol=2e-5, atol=1e-6
            )
            np.testing.assert_allclose(
                owner.get_geom_friction()[0, 0:2, :2], [[0.11, 0.11], [0.12, 0.12]]
            )
            np.testing.assert_allclose(owner.get_geom_friction()[4, 2:4, :2], 1.0)
            slip_pose = np.tile((0.15, 0.0, 0.13, 1.0, 0.0, 0.0, 0.0), (2, 1))
            slip_velocity = np.zeros((2, 6))
            slip_velocity[:, 0] = 0.8
            before_slip = owner.get_entity_state("object")["root_pose"][:, 0].copy()
            owner.reset_entities(
                SceneResetRequest(
                    (0, 4),
                    (
                        EntityStatePatch(
                            "object", root_pose=slip_pose, root_velocity=slip_velocity
                        ),
                    ),
                )
            )
            owner.step(hover_ctrl, nsteps=150)
            slip_state = owner.get_entity_state("object")
            assert slip_state["root_velocity"][0, 0] > slip_state["root_velocity"][4, 0] + 0.05
            assert slip_state["root_pose"][0, 0] - before_slip[0] > (
                slip_state["root_pose"][4, 0] - before_slip[4] + 0.05
            )
            owner.set_state(
                np.array([0], dtype=np.intp),
                np.asarray(owner.get_state()["qpos"])[[0]],
                np.asarray(owner.get_state()["qvel"])[[0]],
            )
            np.testing.assert_allclose(owner.get_geom_friction()[0, 2, :2], 0.05)

        # Full selected reset restores source defaults and control while leaving
        # other rows at their post-step values.
        owner.reset(np.array([4], dtype=np.int32))
        np.testing.assert_allclose(
            owner.get_entity_state("object")["root_pose"][4],
            (0.15, 0.0, 0.13, 1.0, 0.0, 0.0, 0.0),
            atol=1e-5,
        )
        np.testing.assert_allclose(owner.get_state("ctrl")["ctrl"][4], 0.0, atol=1e-6)
        np.testing.assert_allclose(owner.get_state("ctrl")["ctrl"][:4], 0.58, atol=1e-6)

        if backend == "isaacsim":
            try:
                commit = subprocess.check_output(
                    ("git", "rev-parse", "HEAD"), text=True, cwd=Path(__file__).parents[2]
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                commit = "unavailable"
            cache_entries = worker_metadata.get("raw_usd_cache", {}).get("entries", ())
            assert cache_entries
            raw_cache_root = resolve_raw_usd_cache_root()
            assert raw_cache_root is not None
            raw_cache = RawUSDCache(raw_cache_root)
            raw_record = raw_cache.load(cache_entries[0]["identity"])
            assert raw_record is not None
            runtime = resolve_isaacsim_runtime()
            gpu = subprocess.check_output(
                ("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"),
                text=True,
            ).strip()
            evidence = {
                "result": "passed",
                "commit_head": commit,
                "backend": "isaacsim",
                "assignment": assignment.tolist(),
                "runtime_versions": raw_record.runtime_versions,
                "worker_python": str(runtime.python),
                "gpu_and_driver": gpu.splitlines(),
                "readback": {
                    "pair_force": "IsaacLab ContactSensor force_matrix_w",
                    "body_mass_and_com": "worker-native materialization records",
                    "property_mutation": "post-write worker-native material readback",
                    "state": "public entity/generalized state slots",
                    "playback": "selected expanded MJCF source",
                },
                "tolerances": {
                    "support_force": support_tolerance,
                    "no_contact_atol": 0.25,
                    "compensated_velocity_atol": 0.02,
                    "free_fall_rtol": 0.12,
                    "friction_velocity_delta": 0.05,
                    "property_readback_rtol": 2e-5,
                },
                "unverified": [
                    "#133 native camera/RGB recording",
                    "training quality or performance",
                    "cross-backend numerical trajectory equality",
                ],
                "unsupported": [
                    "body-mass reset mutation (PhysX requires a later nonzero solver step)",
                ],
            }
            (tmp_path / "isaacsim-final-integrated-acceptance.json").write_text(
                json.dumps(evidence, indent=2), encoding="utf-8"
            )
    finally:
        owner.close()


@pytest.mark.parametrize("backend", ["isaacgym", "isaacsim"])
def test_public_entity_factory_native_state_identity_and_reset(tmp_path: Path, backend: str):
    if os.environ.get("UNISIM_TEST_" + backend.upper() + "_SCENE") != "1":
        pytest.skip("set UNISIM_TEST_" + backend.upper() + "_SCENE=1 for vendor acceptance")
    n = 5
    config = scene(tmp_path)
    source_assignment = config.entity_variant.plan.assignment
    assignment = 1 - source_assignment
    config.entity_variant = EntityVariantBinding(
        "object",
        FixedVariantPlan(assignment, tuple(reversed(config.entity_variant.plan.variants))),
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
    robot_name = "robot-arm" if backend == "isaacsim" else "robot"
    config.entity_assets = (
        replace(config.entity_assets[0], name=robot_name),
        *config.entity_assets[1:],
    )
    options = {"isaacsim_worker_timeout_s": 240.0} if backend == "isaacsim" else {}
    owner = create_backend(backend, config, num_envs=n, sim_dt=0.002, **options)
    try:
        owner.materialize()
        assert owner.get_entity_names() == (robot_name, "object", "table", "target")
        assert owner.num_actuators == 1
        if backend == "isaacsim":
            import mujoco

            layout = owner.get_scene_layout()
            masses = owner.get_body_mass()
            selected = owner.get_body_ipos(env_ids=[n - 1, 0, n - 1])
            assert masses.shape == (n, layout.nbody)
            assert selected.shape == (3, layout.nbody, 3)
            canonical = owner.get_body_ipos()
            canonical_model = mujoco.MjModel.from_xml_path(owner.get_playback_model(0))
            np.testing.assert_allclose(
                canonical, canonical_model.body_ipos, rtol=1e-4, atol=1e-6
            )
            for result_row, env_index in enumerate((n - 1, 0, n - 1)):
                playback = mujoco.MjModel.from_xml_path(owner.get_playback_model(env_index))
                for entity in layout.entities:
                    ids = np.asarray(entity.body_ids)
                    np.testing.assert_allclose(
                        masses[env_index, ids],
                        playback.body_mass[ids],
                        rtol=2e-4,
                        atol=1e-6,
                    )
                    np.testing.assert_allclose(
                        selected[result_row, ids],
                        playback.body_ipos[ids],
                        rtol=1e-4,
                        atol=1e-6,
                    )
            native_radii = owner._native_entity_records["object"]["body_sphere_radii"]
            expected_radii = [[[0.15 if value == 0 else 0.1]] for value in assignment]
            np.testing.assert_allclose(native_radii, expected_radii, rtol=0.0, atol=1e-8)
        result = {
            "backend": backend,
            "result": "passed",
            "assignment": assignment.tolist(),
            "layout": owner.get_scene_layout().to_dict(),
            "report": owner.get_import_report().to_dict(),
        }
        if backend == "isaacsim":
            result.update(
                {
                    "native_object_sphere_radii": native_radii,
                    "sphere_radius_tolerance": {"atol": 1e-8, "readback": "USD Sphere.radius"},
                }
            )
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
            playback.body("object/base").mass, (3.0, 1.0)[int(assignment[row])]
        )
        (tmp_path / "result.json").write_text(json.dumps(result, indent=2))
    finally:
        owner.close()


def test_isaacsim_native_staged_body_wrench_lifecycle(tmp_path: Path):
    if os.environ.get("UNISIM_TEST_ISAACSIM_SCENE") != "1":
        pytest.skip("set UNISIM_TEST_ISAACSIM_SCENE=1 for vendor acceptance")

    num_envs = 3
    assignment = np.arange(num_envs) % 2
    config = scene(tmp_path)
    config.entity_variant = EntityVariantBinding(
        "object", FixedVariantPlan(assignment, config.entity_variant.plan.variants)
    )
    owner = create_backend(
        "isaacsim", config, num_envs=num_envs, sim_dt=0.002, isaacsim_worker_timeout_s=240.0
    )
    try:
        owner.materialize()
        owner.reset()
        body_ids = owner.get_body_ids(["object/base"])
        masses = np.asarray([1.0, 3.0, 1.0])
        control = np.zeros((num_envs, owner.num_actuators), dtype=np.float32)

        # Two half submissions must accumulate into one gravity-compensating
        # wrench.  Resetting row zero consumes only that row before the step.
        half_hover = np.zeros((num_envs, 1, 3), dtype=np.float32)
        half_hover[:, 0, 2] = masses * 9.81 / 2.0
        owner.apply_body_force(body_ids, half_hover)
        owner.apply_body_force(body_ids, half_hover)
        owner.reset(np.array([0], dtype=np.int32))
        owner.step(control, nsteps=10)
        # The target is the rigid object's sole root body; entity root velocity is
        # the authoritative mapped-scene readback for this wrench target.
        first_velocity = owner.get_entity_state("object")["root_velocity"]
        assert first_velocity[0, 2] < -0.15
        np.testing.assert_allclose(first_velocity[1:, 2], 0.0, atol=0.005)

        # The wrench was consumed: previously compensated rows now fall freely,
        # while the already-falling control row continues from its velocity.
        owner.step(control, nsteps=10)
        idle_velocity = owner.get_entity_state("object")["root_velocity"]
        assert idle_velocity[0, 2] < -0.35
        np.testing.assert_allclose(idle_velocity[1:, 2], -0.1962, atol=0.035)

        owner.reset()
        force = np.zeros((num_envs, 1, 3), dtype=np.float32)
        force[2, 0, 2] = masses[2] * 9.81
        torque = np.zeros((num_envs, 1, 3), dtype=np.float32)
        torque[0, 0, 2] = 0.2
        torque[2, 0, 2] = -0.2
        owner.apply_body_force(body_ids, force, torque=torque)
        owner.step(control, nsteps=10)
        combined_velocity = owner.get_entity_state("object")["root_velocity"]
        np.testing.assert_allclose(combined_velocity[2, 2], 0.0, atol=0.005)
        np.testing.assert_allclose(combined_velocity[1, 2], -0.1962, atol=0.035)
        assert combined_velocity[0, 5] > 0.5
        assert combined_velocity[2, 5] < -0.5
        np.testing.assert_allclose(combined_velocity[1, 5], 0.0, atol=0.005)

        (tmp_path / "isaacsim-body-wrench.json").write_text(
            json.dumps(
                {
                    "result": "passed",
                    "commit_head": "pending-local-run",
                    "assignment": assignment.tolist(),
                    "tolerances": {
                        "hover_linear_velocity_z": 0.005,
                        "free_fall_linear_velocity_z": 0.035,
                        "idle_angular_velocity_z": 0.005,
                    },
                },
                indent=2,
            )
        )
    finally:
        owner.close()


def test_isaacsim_native_pre_step_control_composes_fresh_state_and_interval_wrench(
    tmp_path: Path,
):
    if os.environ.get("UNISIM_TEST_ISAACSIM_SCENE") != "1":
        pytest.skip("set UNISIM_TEST_ISAACSIM_SCENE=1 for vendor acceptance")

    num_envs = 3
    assignment = np.arange(num_envs) % 2
    masses = np.asarray([1.0, 3.0, 1.0])
    config = scene(tmp_path)
    config.entity_variant = EntityVariantBinding(
        "object", FixedVariantPlan(assignment, config.entity_variant.plan.variants)
    )
    owner = create_backend(
        "isaacsim", config, num_envs=num_envs, sim_dt=0.002, isaacsim_worker_timeout_s=240.0
    )
    observed_velocities: list[np.ndarray] = []
    observed_body_positions: list[np.ndarray] = []
    callback_ctrl_values: list[np.ndarray] = []
    fixed_force_z = 0.5

    try:
        owner.materialize()
        owner.reset()
        body_ids = owner.get_body_ids(["object/base"])
        fixed = np.zeros((num_envs, 1, 3), dtype=np.float32)
        fixed[:, 0, 2] = fixed_force_z
        owner.apply_body_force(body_ids, fixed)

        def controller(owner_, policy_ctrl):
            velocity = owner_.get_entity_state("object")["root_velocity"][:, 2].copy()
            observed_velocities.append(velocity)
            observed_body_positions.append(owner_.get_body_pos_w(body_ids)[:, 0, 2].copy())
            converted_ctrl = policy_ctrl + np.float32(0.01 * len(callback_ctrl_values))
            callback_ctrl_values.append(converted_ctrl.copy())
            # The fixed interval channel contributes +0.5 N.  Return enough
            # dynamic force for a +1 N first impulse when the observed state is
            # still at rest, then only gravity compensation once velocity is
            # nonzero.  This makes the next callback depend on actual worker
            # state rather than a constant zero-order-hold wrench.
            impulse = np.where(np.abs(velocity) < 1e-5, 1.0, 0.0)
            dynamic = np.zeros((num_envs, 1, 3), dtype=np.float32)
            dynamic[:, 0, 2] = masses * 9.81 - fixed_force_z + impulse
            return PreStepControlOutput(
                ctrl=converted_ctrl,
                body_ids=body_ids,
                force=dynamic,
            )

        owner.set_pre_step_control(controller)
        policy_ctrl = np.full((num_envs, owner.num_actuators), 0.1, dtype=np.float32)
        owner.step(policy_ctrl, nsteps=6)
        compensated_velocity = owner.get_entity_state("object")["root_velocity"][:, 2]
        assert len(observed_velocities) == 6
        np.testing.assert_allclose(observed_velocities[0], 0.0, atol=1e-5)
        assert np.all(observed_velocities[1] > 0.0005)
        assert np.all(observed_velocities[2:] > observed_velocities[1] - 1e-5)
        assert np.all(observed_body_positions[1] > observed_body_positions[0])
        assert np.all(observed_body_positions[-1] > observed_body_positions[1])
        np.testing.assert_allclose(compensated_velocity, observed_velocities[-1], atol=1e-6)
        expected_impulse_velocity = np.asarray([1.0, 1.0 / 3.0, 1.0]) * 0.002
        np.testing.assert_allclose(
            compensated_velocity, expected_impulse_velocity, rtol=0.15, atol=2e-4
        )
        np.testing.assert_allclose(owner._slots["ctrl"], callback_ctrl_values[-1], atol=0.0)
        assert not owner._body_wrench_pending
        assert not np.any(owner._staged_body_wrench)

        owner.set_pre_step_control(None)
        owner.step(policy_ctrl, nsteps=1)
        free_step_velocity = owner.get_entity_state("object")["root_velocity"][:, 2]
        np.testing.assert_allclose(
            free_step_velocity - compensated_velocity,
            np.full(num_envs, -9.81 * 0.002),
            rtol=0.05,
            atol=2e-4,
        )

        try:
            commit = subprocess.check_output(
                ("git", "rev-parse", "HEAD"), text=True, cwd=Path(__file__).parents[2]
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unavailable"
        (tmp_path / "isaacsim-pre-step-control.json").write_text(
            json.dumps(
                {
                    "result": "passed",
                    "commit_head": commit,
                    "assignment": assignment.tolist(),
                    "callback_count": len(observed_velocities),
                    "observed_velocities": [values.tolist() for values in observed_velocities],
                    "observed_body_positions": [
                        values.tolist() for values in observed_body_positions
                    ],
                    "compensated_velocity": compensated_velocity.tolist(),
                    "free_step_velocity": free_step_velocity.tolist(),
                    "fixed_force_z": fixed_force_z,
                    "tolerances": {
                        "initial_velocity_atol": 1e-5,
                        "impulse_velocity_rtol": 0.15,
                        "impulse_velocity_atol": 2e-4,
                        "free_fall_delta_rtol": 0.05,
                    },
                },
                indent=2,
            )
        )
    finally:
        owner.close()


def test_isaacsim_native_raw_and_role_usd_cache_cold_warm_semantics_and_immutability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    if os.environ.get("UNISIM_TEST_ISAACSIM_SCENE") != "1":
        pytest.skip("set UNISIM_TEST_ISAACSIM_SCENE=1 for vendor acceptance")

    cache_root = tmp_path / "raw-usd-cache"
    role_cache_root = tmp_path / "role-usd-cache"
    monkeypatch.setenv("UNISIM_ISAACSIM_RAW_USD_CACHE", str(cache_root))
    monkeypatch.setenv("UNISIM_ISAACSIM_ROLE_USD_CACHE", str(role_cache_root))
    config = scene(tmp_path)
    config.entity_variant = EntityVariantBinding(
        "object",
        FixedVariantPlan(
            np.arange(3) % 2, config.entity_variant.plan.variants
        ),
    )

    reports: list[dict] = []

    def bind(owner):
        original = owner._bind_scene_metadata

        def capture(value):
            original(value)
            if not reports:
                reports.append(value)

        owner._bind_scene_metadata = capture  # type: ignore[method-assign]

    cold_started = time.perf_counter()
    cold = create_backend(
        "isaacsim", config, num_envs=3, sim_dt=0.002, isaacsim_worker_timeout_s=240.0
    )
    cold_report: dict = {}
    cold_role_report: dict = {}
    try:
        bind(cold)
        cold.materialize()
        cold_init_s = time.perf_counter() - cold_started
        assert len(reports) == 1
        cold_report = reports[0]["raw_usd_cache"]
        cold_role_report = reports[0]["role_usd_cache"]
        assert cold_report["enabled"] is True
        assert cold_report["unique_sources"] > 0
        assert cold_report["conversions"] > 0 and cold_report["hits"] == 0
        assert cold_role_report["enabled"] is True
        assert cold_role_report["bakes"] > 0 and cold_role_report["hits"] == 0
        cold_masses = cold.get_body_mass().copy()
        cold_ipos = cold.get_body_ipos().copy()
        cold_state = {key: value.copy() for key, value in cold.get_state().items()}
        control = np.zeros((3, cold.num_actuators), dtype=np.float32)
        cold.step(control, nsteps=10)
        cold_response = {key: value.copy() for key, value in cold.get_state().items()}
    finally:
        cold.close()

    cache = RawUSDCache(cache_root)
    identities = [entry["identity"] for entry in cold_report["entries"]]
    assert len(identities) == len(set(identities))
    role_cache = RoleUSDCache(role_cache_root)
    role_identities = [entry["identity"] for entry in cold_role_report["entries"]]
    assert len(role_identities) == len(set(role_identities))
    records = [cache.load(identity) for identity in identities]
    assert all(record is not None for record in records)
    role_records = [role_cache.load(identity) for identity in role_identities]
    assert all(record is not None for record in role_records)
    cold_hashes = {
        file.path: file.sha256
        for record in records
        for file in record.files  # type: ignore[union-attr]
    }
    assert cold_hashes
    cold_role_hashes = {
        file.path: file.sha256
        for record in role_records
        for file in record.files  # type: ignore[union-attr]
    }
    assert cold_role_hashes

    reports.clear()
    warm_started = time.perf_counter()
    warm = create_backend(
        "isaacsim", config, num_envs=3, sim_dt=0.002, isaacsim_worker_timeout_s=240.0
    )
    try:
        bind(warm)
        warm.materialize()
        warm_init_s = time.perf_counter() - warm_started
        assert len(reports) == 1
        warm_report = reports[0]["raw_usd_cache"]
        warm_role_report = reports[0]["role_usd_cache"]
        assert warm_report["enabled"] is True
        assert warm_report["unique_sources"] == cold_report["unique_sources"]
        assert warm_report["hits"] == cold_report["conversions"]
        assert warm_report["conversions"] == 0
        assert [entry["identity"] for entry in warm_report["entries"]] == identities
        assert warm_role_report["enabled"] is True
        assert warm_role_report["hits"] == cold_role_report["bakes"]
        assert warm_role_report["bakes"] == 0
        assert [entry["identity"] for entry in warm_role_report["entries"]] == role_identities
        np.testing.assert_allclose(warm.get_body_mass(), cold_masses, rtol=2e-4, atol=1e-6)
        np.testing.assert_allclose(warm.get_body_ipos(), cold_ipos, rtol=1e-4, atol=1e-6)
        for key, value in warm.get_state().items():
            np.testing.assert_allclose(value, cold_state[key], rtol=1e-5, atol=1e-5)
        warm.step(np.zeros((3, warm.num_actuators), dtype=np.float32), nsteps=10)
        for key, value in warm.get_state().items():
            np.testing.assert_allclose(value, cold_response[key], rtol=1e-4, atol=2e-4)
    finally:
        warm.close()

    warm_records = [cache.load(identity) for identity in identities]
    assert all(record is not None for record in warm_records)
    warm_hashes = {
        file.path: file.sha256
        for record in warm_records
        for file in record.files  # type: ignore[union-attr]
    }
    assert warm_hashes == cold_hashes
    warm_role_records = [role_cache.load(identity) for identity in role_identities]
    assert all(record is not None for record in warm_role_records)
    warm_role_hashes = {
        file.path: file.sha256
        for record in warm_role_records
        for file in record.files  # type: ignore[union-attr]
    }
    assert warm_role_hashes == cold_role_hashes
    try:
        commit = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    runtime_versions = warm_records[0].runtime_versions  # type: ignore[union-attr]
    (tmp_path / "isaacsim-usd-caches.json").write_text(
        json.dumps(
            {
                "result": "passed",
                "commit_head": commit,
                "runtime_versions": runtime_versions,
                "cache_bytes": sum(
                    record.size_bytes for record in warm_records if record is not None
                ),
                "cold": {
                    "init_s": cold_init_s,
                    "conversions": cold_report["conversions"],
                    "hits": cold_report["hits"],
                    "materialize_ms": [
                        entry["materialize_ms"] for entry in cold_report["entries"]
                    ],
                },
                "warm": {
                    "init_s": warm_init_s,
                    "conversions": warm_report["conversions"],
                    "hits": warm_report["hits"],
                    "materialize_ms": [
                        entry["materialize_ms"] for entry in warm_report["entries"]
                    ],
                },
                "readback": {
                    "fields": ["body_mass", "body_ipos", "qpos", "qvel"],
                    "mass_tolerance": {"rtol": 2e-4, "atol": 1e-6},
                    "com_tolerance": {"rtol": 1e-4, "atol": 1e-6},
                    "state_response_tolerance": {"rtol": 1e-4, "atol": 2e-4},
                },
                "raw_immutability": "all cached file SHA-256 digests matched after warm hit",
                "role_cache_bytes": sum(
                    record.size_bytes for record in warm_role_records if record is not None
                ),
                "role": {
                    "cold": {
                        "bakes": cold_role_report["bakes"],
                        "hits": cold_role_report["hits"],
                        "materialize_ms": [
                            entry["materialize_ms"] for entry in cold_role_report["entries"]
                        ],
                    },
                    "warm": {
                        "bakes": warm_role_report["bakes"],
                        "hits": warm_role_report["hits"],
                        "materialize_ms": [
                            entry["materialize_ms"] for entry in warm_role_report["entries"]
                        ],
                    },
                    "immutability": "all cached role-file SHA-256 digests matched after warm hit",
                },
                "unverified": "memory usage and 1,200-variant scale are not measured here",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
