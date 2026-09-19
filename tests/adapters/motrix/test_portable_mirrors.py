"""Real Motrix acceptance for portable collision-disabled mirrors."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("motrixsim")

from tests.adapters.motrix.test_portable_entities import _scene, _write
from unisim.backend.motrix.backend import MotrixBackend
from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import (
    EntityInitialState,
    EntityStatePatch,
    EntityVariantBinding,
    SceneEntitySpec,
    SceneResetRequest,
)


def _mirror_contact_fragment(tmp_path: Path) -> Path:
    target = tmp_path / "object-mirror-contact.xml"
    target.write_text(
        "<mujoco><sensor>"
        "<contact name='object_mirror_force' geom1='object/object_geom' "
        "geom2='mirror/object_geom' data='force' reduce='netforce'/>"
        "<contact name='object_mirror_found' geom1='object/object_geom' "
        "geom2='mirror/object_geom' data='found' num='1'/>"
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    return target


def _mirror_scene(tmp_path: Path):
    scene = _scene(tmp_path)
    mirror = SceneEntitySpec(
        "mirror",
        kind="rigid",
        root_mode="kinematic",
        collision_enabled=False,
        mirror_of="object",
        initial_state=EntityInitialState((20.0, 0.0, 10.0)),
    )
    scene.entity_assets = scene.entity_assets + (mirror,)
    scene.fragment_files = (str(_mirror_contact_fragment(tmp_path)),)
    return mirror, scene


def _write_object_variant(
    tmp_path: Path, name: str, *, mass: float, size: float
) -> ModelSourceDescriptor:
    return _write(
        tmp_path,
        name,
        f"""
        <mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/>
        <worldbody><body name="base" pos="2 0 2">
          <freejoint name="root"/><inertial pos=".01 0 0" mass="{mass}"
            diaginertia=".02 .03 .04"/>
          <geom name="object_geom" type="sphere" size="{size}"/>
        </body></worldbody></mujoco>
        """,
    )


def test_mirror_pose_writes_are_row_local_and_collision_suppression_is_physical(tmp_path):
    _, far_scene = _mirror_scene(tmp_path / "far")
    _, overlap_scene = _mirror_scene(tmp_path / "overlap")
    far_backend = MotrixBackend(far_scene, 3, 0.002, base_name="robot/base")
    overlap_backend = MotrixBackend(overlap_scene, 3, 0.002, base_name="robot/base")
    try:
        layout = overlap_backend.get_scene_layout()
        mirror_layout = layout.get_entity("mirror")
        assert mirror_layout.root_mode == "kinematic"
        assert mirror_layout.joints == ()
        assert mirror_layout.actuator_names == ()

        native_mirror = overlap_backend._model.get_body("mirror/base")
        assert native_mirror is not None
        assert native_mirror.is_mocap
        assert native_mirror.mocap is not None
        mirror_geom_id = overlap_backend.get_geom_id("mirror/object_geom")
        contype, conaffinity = overlap_backend.get_geom_contact_masks()
        assert (contype[mirror_geom_id], conaffinity[mirror_geom_id]) == (0, 0)

        object_pose = np.tile(
            np.asarray((0.5, 0.0, 0.5, 0.9, 0.1, 0.2, 0.3), dtype=np.float32),
            (3, 1),
        )
        object_pose[:, 3:] /= np.linalg.norm(object_pose[:, 3:], axis=1, keepdims=True)
        object_velocity = np.tile(
            np.asarray((0.03, 0.0, -1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (3, 1),
        )
        for backend in (far_backend, overlap_backend):
            backend.reset_entities(
                SceneResetRequest(
                    tuple(range(3)),
                    (
                        EntityStatePatch(
                            "object", root_pose=object_pose, root_velocity=object_velocity
                        ),
                    ),
                )
            )

        selected_rows = np.asarray((1, 2), dtype=np.intp)
        default_mirror_state = overlap_backend.get_entity_state("mirror")
        unrelated_before = {
            name: {
                field: np.asarray(values).copy()
                for field, values in overlap_backend.get_entity_state(name).items()
            }
            for name in overlap_backend.get_entity_names()
            if name != "mirror"
        }
        selected_pose = np.asarray(
            (
                (3.0, 0.1, 2.5, 0.8, 0.6, 0.0, 0.0),
                (0.5, 0.0, 0.5, 0.8, -0.6, 0.0, 0.0),
            ),
            dtype=np.float32,
        )
        selected_pose[:, 3:] /= np.linalg.norm(selected_pose[:, 3:], axis=1, keepdims=True)
        overlap_backend.reset_entities(
            SceneResetRequest((1, 2), (EntityStatePatch("mirror", root_pose=selected_pose),))
        )
        mirror_after_write = overlap_backend.get_entity_state("mirror")
        np.testing.assert_allclose(
            mirror_after_write["root_pose"][selected_rows],
            selected_pose,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            mirror_after_write["root_pose"][0], default_mirror_state["root_pose"][0]
        )
        np.testing.assert_array_equal(mirror_after_write["root_velocity"], 0.0)
        for name, state in unrelated_before.items():
            for field, values in state.items():
                np.testing.assert_array_equal(
                    np.asarray(overlap_backend.get_entity_state(name)[field]), values
                )

        overlap_object_history = []
        overlap_unrelated_history = []
        found_history = []
        force_history = []
        controls = np.zeros((3, overlap_backend.num_actuators), dtype=np.float32)
        for step in range(60):
            object_state = overlap_backend.get_entity_state("object")
            far_target = np.repeat(default_mirror_state["root_pose"][:1], 2, axis=0)
            overlap_target = np.vstack(
                (
                    selected_pose[:1],
                    np.asarray(object_state["root_pose"][2], dtype=np.float32),
                )
            )
            far_target[0, 0] += 0.001 * step
            far_backend.reset_entities(
                SceneResetRequest((1, 2), (EntityStatePatch("mirror", root_pose=far_target),))
            )
            overlap_backend.reset_entities(
                SceneResetRequest(
                    (1, 2), (EntityStatePatch("mirror", root_pose=overlap_target),)
                )
            )
            far_backend.step(controls)
            overlap_backend.step(controls)

            overlap_object = overlap_backend.get_entity_state("object")
            overlap_object_history.append(overlap_object)
            overlap_unrelated_history.append(
                {
                    name: overlap_backend.get_entity_state(name)
                    for name in overlap_backend.get_entity_names()
                    if name not in ("object", "mirror")
                }
            )
            found_history.append(
                np.asarray(overlap_backend.get_sensor_data("object_mirror_found")).copy()
            )
            force_history.append(
                np.asarray(overlap_backend.get_sensor_data("object_mirror_force")).copy()
            )

            mirror_state = overlap_backend.get_entity_state("mirror")
            np.testing.assert_allclose(
                mirror_state["root_pose"][0],
                default_mirror_state["root_pose"][0],
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                mirror_state["root_pose"][1], selected_pose[0], rtol=0.0, atol=0.0
            )
            np.testing.assert_allclose(
                mirror_state["root_pose"][2], overlap_target[1], rtol=0.0, atol=1e-7
            )

        found = np.stack(found_history)
        force = np.stack(force_history)
        assert found.shape == (60, 3, 1)
        assert force.shape == (60, 3, 3)
        np.testing.assert_array_equal(found, 0.0)
        np.testing.assert_array_equal(np.linalg.norm(force, axis=2), 0.0)

        far_object = far_backend.get_entity_state("object")
        overlap_object_final = overlap_backend.get_entity_state("object")
        for field in far_object:
            np.testing.assert_allclose(
                far_object[field], overlap_object_final[field], rtol=2e-7, atol=2e-7
            )
            np.testing.assert_allclose(
                overlap_object_final[field][0],
                overlap_object_final[field][1],
                rtol=2e-7,
                atol=2e-7,
            )
            np.testing.assert_allclose(
                overlap_object_final[field][0],
                overlap_object_final[field][2],
                rtol=2e-7,
                atol=2e-7,
            )
        for old_object in overlap_object_history:
            for field in old_object:
                np.testing.assert_allclose(
                    old_object[field][0], old_object[field][2], rtol=2e-7, atol=2e-7
                )
        for states in overlap_unrelated_history:
            for name, state in states.items():
                for field in state:
                    np.testing.assert_allclose(
                        state[field][0], state[field][2], rtol=2e-7, atol=2e-7
                    )
    finally:
        far_backend.close()
        overlap_backend.close()


def test_fixed_variant_mirror_identity_and_pose_routing(tmp_path):
    light = _write_object_variant(tmp_path / "light", "light", mass=0.5, size=0.1)
    heavy = _write_object_variant(tmp_path / "heavy", "heavy", mass=1.7, size=0.12)
    mirror = SceneEntitySpec(
        "mirror",
        kind="rigid",
        root_mode="kinematic",
        collision_enabled=False,
        mirror_of="object",
        initial_state=EntityInitialState((4.0, 0.0, 3.0)),
    )
    scene = replace(
        _scene(tmp_path / "base"),
        entity_assets=(SceneEntitySpec("object", light, kind="rigid"), mirror),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(
                np.asarray((1, 0), dtype=np.int32),
                (light, heavy),
                FixedVariantLayout.SAME_LAYOUT,
            ),
        ),
    )
    backend = MotrixBackend(scene, 2, 0.002, base_name="object/base")
    try:
        assert len(backend._portable_runtimes) == 2
        default_pose = np.asarray(
            backend.get_entity_state("mirror")["root_pose"].copy(), dtype=np.float32
        )
        for runtime in backend._portable_runtimes:
            body = runtime.model.get_body("mirror/base")
            assert body is not None and body.is_mocap and body.mocap is not None
            geom = runtime.model.get_geom("mirror/object_geom")
            assert geom is not None
            assert (int(geom.collision_group), int(geom.collision_affinity)) == (0, 0)

        poses = np.asarray(
            (
                (5.0, 0.2, 3.2, 0.8, 0.6, 0.0, 0.0),
                (6.0, -0.2, 3.4, 0.8, 0.0, 0.6, 0.0),
            ),
            dtype=np.float32,
        )
        poses[:, 3:] /= np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)
        backend.reset_entities(
            SceneResetRequest((0, 1), (EntityStatePatch("mirror", root_pose=poses),))
        )
        backend.step(np.zeros((2, backend.num_actuators), dtype=np.float32), 5)
        mirror_pose = backend.get_entity_state("mirror")["root_pose"]
        np.testing.assert_allclose(mirror_pose, poses, rtol=0.0, atol=0.0)
        backend.reset((0,))
        mirror_pose = backend.get_entity_state("mirror")["root_pose"]
        np.testing.assert_allclose(mirror_pose[0], default_pose[0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(mirror_pose[1], poses[1], rtol=0.0, atol=0.0)
    finally:
        backend.close()
