"""Real Genesis acceptance for portable fixed variants and visual mirrors."""

# ruff: noqa: E402
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("genesis")
pytest.importorskip("torch")

from tests.adapters.genesis.test_portable_entities import (
    _enable_source_gravity,
    _scene,
)
from unisim.backend.genesis.backend import GenesisBackend
from unisim.dr.types import ResetRandomizationPayload
from unisim.entities import (
    EntityInitialState,
    EntityStatePatch,
    SceneEntitySpec,
    SceneResetRequest,
)
from unisim.scene import SceneCfg


def _enable_contact_masks_only(scene: SceneCfg) -> None:
    """Enable physical collision identity without Kinematic-incompatible condim."""

    masks = {
        "robot": (1, 16),
        "passive": (2, 32),
        "object": (4, 64),
        "table": (8, 128),
    }
    for entity in scene.entity_assets:
        sources = [Path(entity.source.model_file)]
        if scene.entity_variant is not None and scene.entity_variant.target_entity == entity.name:
            sources.extend(
                Path(variant.model_file) for variant in scene.entity_variant.plan.variants
            )
        for source in dict.fromkeys(sources):
            text = source.read_text(encoding="utf-8")
            source.write_text(
                text.replace(
                    'contype="0" conaffinity="0"',
                    f'contype="{masks[entity.name][0]}" '
                    f'conaffinity="{masks[entity.name][1]}"',
                ),
                encoding="utf-8",
            )


def _scene_with_mirror(
    tmp_path: Path,
    *,
    assignment: tuple[int, ...] = (0, 0, 0, 1, 1),
    position: tuple[float, float, float] = (20.0, 0.0, 10.0),
    gravity: bool = False,
    contact_masks: bool = False,
) -> SceneCfg:
    """Build a physical scene and append a collision-disabled visual mirror."""

    scene = _scene(tmp_path, assignment=assignment)
    if gravity:
        _enable_source_gravity(scene)
    if contact_masks:
        _enable_contact_masks_only(scene)
    scene.entity_assets = scene.entity_assets + (
        SceneEntitySpec(
            "mirror",
            kind="rigid",
            root_mode="kinematic",
            collision_enabled=False,
            mirror_of="object",
            initial_state=EntityInitialState(position),
        ),
    )
    return scene


def _object_state_snapshot(backend: GenesisBackend) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: {
            field: np.asarray(values).copy()
            for field, values in backend.get_entity_state(name).items()
        }
        for name in backend.get_entity_names()
    }


def test_reversed_variant_declaration_controls_native_identity(tmp_path: Path) -> None:
    scene = _scene_with_mirror(tmp_path, assignment=(0, 1), position=(4.0, 0.0, 3.0))
    binding = scene.entity_variant
    assert binding is not None
    scene.entity_variant = replace(
        binding,
        plan=replace(binding.plan, variants=tuple(reversed(binding.plan.variants))),
    )

    backend = GenesisBackend(scene, 2, 0.002)
    try:
        backend.materialize()
        object_runtime = backend._entity_runtimes["object"]
        mirror_runtime = backend._entity_runtimes["mirror"]
        assert object_runtime.source_metadata is not mirror_runtime.source_metadata
        assert tuple(item.body_mass[1] for item in object_runtime.source_metadata) == (1.5, 0.5)

        solver = backend._scene.sim.rigid_solver
        assert solver is not None
        native_link = int(object_runtime.entity.get_link("base").idx)
        native_mass = (
            solver.get_links_inertial_mass(links_idx=[native_link])
            .cpu()
            .numpy()
            .reshape(2, 1)[:, 0]
        )
        native_com = (
            solver.get_links_root_COM(links_idx=[native_link])
            .cpu()
            .numpy()
            .reshape(2, 3)
        )
        np.testing.assert_allclose(native_mass, (1.5, 0.5), rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(native_com[:, 0], (2.03, 2.01), rtol=2e-6, atol=1e-7)

        native_dof_start = int(object_runtime.entity.dof_start)
        mass_matrix = solver.get_mass_mat().cpu().numpy()
        rotational_inertia = mass_matrix[
            np.arange(2)[:, None],
            native_dof_start + 3 + np.arange(3)[None, :],
            native_dof_start + 3 + np.arange(3)[None, :],
        ]
        np.testing.assert_allclose(
            rotational_inertia,
            ((0.03, 0.04, 0.05), (0.02, 0.03, 0.04)),
            rtol=2e-5,
            atol=2e-7,
        )

        for runtime in (object_runtime, mirror_runtime):
            vgeoms = list(runtime.entity.vgeoms)
            assert len(vgeoms) == 2
            assert vgeoms[0].active_envs_idx is not None
            assert vgeoms[1].active_envs_idx is not None
            assert vgeoms[0].active_envs_idx.tolist() == [0]
            assert vgeoms[1].active_envs_idx.tolist() == [1]
            first_vertices = np.asarray(vgeoms[0].init_vverts, dtype=np.float64)
            second_vertices = np.asarray(vgeoms[1].init_vverts, dtype=np.float64)
            np.testing.assert_allclose(np.min(first_vertices, axis=0), -0.15, atol=2e-6)
            np.testing.assert_allclose(np.max(first_vertices, axis=0), 0.15, atol=2e-6)
            np.testing.assert_allclose(np.min(second_vertices, axis=0), -0.1, atol=2e-6)
            np.testing.assert_allclose(np.max(second_vertices, axis=0), 0.1, atol=2e-6)

        assert not list(getattr(mirror_runtime.entity, "geoms", ()))
        assert int(mirror_runtime.entity.n_qs) == 0
        assert int(mirror_runtime.entity.n_dofs) == 0
        np.testing.assert_array_equal(mirror_runtime.collision_geom_indices, -1)
    finally:
        # Genesis permits one process-wide session; teardown is owned by the
        # process boundary so later native constructions remain valid.
        pass


def test_fixed_variant_mirror_pose_routing_and_reset(tmp_path: Path) -> None:
    scene = _scene_with_mirror(tmp_path, position=(4.0, 0.0, 3.0), contact_masks=True)
    backend = GenesisBackend(scene, 5, 0.002)
    try:
        backend.materialize()
        mirror_runtime = backend._entity_runtimes["mirror"]
        default_pose = backend.get_entity_state("mirror")["root_pose"].copy()
        contype, conaffinity = backend.get_geom_contact_masks()
        assert (contype[-1], conaffinity[-1]) == (0, 0)
        assert mirror_runtime.contact_masks is not None
        np.testing.assert_array_equal(mirror_runtime.contact_masks, 0)
        with pytest.raises(
            NotImplementedError,
            match="native collision identity is unavailable or ambiguous for geometry friction",
        ):
            backend.get_geom_friction()
        with pytest.raises(
            NotImplementedError,
            match="native collision identity is unavailable or ambiguous for geometry solref",
        ):
            backend.get_geom_solref()

        selected_rows = np.asarray((4, 1), dtype=np.intp)
        selected_pose = np.asarray(
            (
                (5.0, 0.2, 3.2, 0.8, 0.6, 0.0, 0.0),
                (6.0, -0.2, 3.4, 0.8, 0.0, 0.6, 0.0),
            ),
            dtype=np.float32,
        )
        selected_pose[:, 3:] /= np.linalg.norm(
            selected_pose[:, 3:], axis=1, keepdims=True
        )
        backend.reset_entities(
            SceneResetRequest(
                (4, 1),
                (EntityStatePatch("mirror", root_pose=selected_pose),),
            )
        )
        mirror_pose = backend.get_entity_state("mirror")["root_pose"]
        np.testing.assert_allclose(mirror_pose[selected_rows], selected_pose, atol=1e-6)
        untouched_rows = np.asarray((0, 2, 3), dtype=np.intp)
        np.testing.assert_array_equal(mirror_pose[untouched_rows], default_pose[untouched_rows])
        np.testing.assert_array_equal(backend.get_entity_state("mirror")["root_velocity"], 0.0)

        backend.step(np.zeros((5, backend.num_actuators), dtype=np.float32), 5)
        mirror_pose = backend.get_entity_state("mirror")["root_pose"]
        np.testing.assert_allclose(mirror_pose[selected_rows], selected_pose, atol=1e-6)

        backend.reset((1,))
        mirror_pose = backend.get_entity_state("mirror")["root_pose"]
        np.testing.assert_allclose(mirror_pose[1], default_pose[1], atol=1e-6)
        np.testing.assert_allclose(mirror_pose[4], selected_pose[0], atol=1e-6)
        assert len(list(mirror_runtime.entity.vgeoms)) == 2
    finally:
        # Keep the process-wide Genesis session alive for later native tests.
        pass


def test_unsorted_object_only_reset_isolates_entities_and_controls(tmp_path: Path) -> None:
    scene = _scene_with_mirror(tmp_path)
    backend = GenesisBackend(scene, 5, 0.002)
    robot_entity = next(entity for entity in backend._scene.entities if str(entity.name) == "robot")
    original_control = robot_entity.control_dofs_position
    control_calls: list[tuple[np.ndarray, list[int] | None, list[int] | None]] = []

    def capture_control(position, dofs_idx_local=None, envs_idx=None):
        control_calls.append(
            (
                position.detach().cpu().numpy().copy(),
                dofs_idx_local,
                envs_idx,
            )
        )
        return original_control(position, dofs_idx_local=dofs_idx_local, envs_idx=envs_idx)

    robot_entity.control_dofs_position = capture_control
    try:
        backend.materialize()
        control_calls.clear()
        backend.step(np.asarray([[0.4], [0.2], [0.1], [-0.2], [-0.4]], dtype=np.float32))
        control_calls.clear()

        states_before = _object_state_snapshot(backend)
        selected_rows = np.asarray((4, 1), dtype=np.intp)
        untouched_rows = np.asarray((0, 2, 3), dtype=np.intp)
        selected_pose = np.asarray(
            (
                (2.4, 0.1, 1.2, 0.8, 0.6, 0.0, 0.0),
                (1.7, -0.1, 0.9, 0.8, 0.0, 0.6, 0.0),
            ),
            dtype=np.float32,
        )
        selected_pose[:, 3:] /= np.linalg.norm(
            selected_pose[:, 3:], axis=1, keepdims=True
        )
        selected_velocity = np.asarray(
            ((0.2, -0.1, 0.3, 0.0, 0.1, -0.2), (-0.2, 0.1, -0.3, 0.0, -0.1, 0.2)),
            dtype=np.float32,
        )
        backend.reset_entities(
            SceneResetRequest(
                (4, 1),
                (
                    EntityStatePatch(
                        "object",
                        root_pose=selected_pose,
                        root_velocity=selected_velocity,
                    ),
                ),
            )
        )
        assert control_calls == []

        object_state = backend.get_entity_state("object")
        np.testing.assert_allclose(
            object_state["root_pose"][selected_rows], selected_pose, atol=1e-6
        )
        np.testing.assert_allclose(
            object_state["root_velocity"][selected_rows], selected_velocity, atol=1e-6
        )
        np.testing.assert_array_equal(
            object_state["root_pose"][untouched_rows],
            states_before["object"]["root_pose"][untouched_rows],
        )
        np.testing.assert_array_equal(
            object_state["root_velocity"][untouched_rows],
            states_before["object"]["root_velocity"][untouched_rows],
        )
        for name in backend.get_entity_names():
            if name == "object":
                continue
            for field, values in states_before[name].items():
                np.testing.assert_array_equal(backend.get_entity_state(name)[field], values)

        backend.step(np.zeros((5, backend.num_actuators), dtype=np.float32))
        assert len(control_calls) == 1
        np.testing.assert_array_equal(control_calls[0][0], np.zeros((5, 1), dtype=np.float32))
        assert control_calls[0][2] is None
    finally:
        robot_entity.control_dofs_position = original_control


def test_mirror_overlap_preserves_physical_trajectory(tmp_path: Path) -> None:
    far_scene = _scene_with_mirror(
        tmp_path / "far",
        assignment=(0, 1),
        position=(20.0, 0.0, 10.0),
        gravity=True,
    )
    overlap_scene = _scene_with_mirror(
        tmp_path / "overlap",
        assignment=(0, 1),
        position=(2.0, 0.0, 2.0),
        gravity=True,
    )

    far_backend = GenesisBackend(far_scene, 2, 0.002)
    overlap_backend = GenesisBackend(overlap_scene, 2, 0.002)
    try:
        far_backend.materialize()
        overlap_backend.materialize()
        object_pose = np.asarray(
            (
                (2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0),
                (2.0, 0.0, 2.0, 0.9, 0.1, 0.2, 0.3),
            ),
            dtype=np.float32,
        )
        object_pose[:, 3:] /= np.linalg.norm(object_pose[:, 3:], axis=1, keepdims=True)
        object_velocity = np.asarray(
            (
                (-0.1, 0.05, -1.2, 0.2, -0.1, 0.3),
                (0.1, -0.05, -1.0, -0.2, 0.1, 0.3),
            ),
            dtype=np.float32,
        )
        for backend in (far_backend, overlap_backend):
            backend.reset_entities(
                SceneResetRequest(
                    (0, 1),
                    (
                        EntityStatePatch(
                            "object",
                            root_pose=object_pose,
                            root_velocity=object_velocity,
                        ),
                    ),
                )
            )

        far_pose = np.tile(
            np.asarray((20.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (2, 1),
        )
        for _ in range(60):
            overlap_target = overlap_backend.get_entity_state("object")["root_pose"].copy()
            far_backend.reset_entities(
                SceneResetRequest((0, 1), (EntityStatePatch("mirror", root_pose=far_pose),))
            )
            overlap_backend.reset_entities(
                SceneResetRequest(
                    (0, 1),
                    (EntityStatePatch("mirror", root_pose=overlap_target),),
                )
            )
            far_backend.step(np.zeros((2, far_backend.num_actuators), dtype=np.float32))
            overlap_backend.step(
                np.zeros((2, overlap_backend.num_actuators), dtype=np.float32)
            )
            far_state = far_backend.get_entity_state("object")
            overlap_state = overlap_backend.get_entity_state("object")
            for field, far_values in far_state.items():
                np.testing.assert_allclose(
                    far_values,
                    overlap_state[field],
                    rtol=2e-6,
                    atol=2e-6,
                )

        overlap_mirror_pose = overlap_backend.get_entity_state("mirror")["root_pose"]
        object_pose_after = overlap_backend.get_entity_state("object")["root_pose"]
        assert np.max(np.linalg.norm(overlap_mirror_pose - object_pose_after, axis=1)) > 1e-3
        np.testing.assert_array_equal(
            overlap_backend.get_entity_state("mirror")["root_velocity"], 0.0
        )
    finally:
        # Keep the process-wide Genesis session alive for other native tests.
        pass


def test_visual_mirror_physical_mutation_and_contact_fragments_fail_closed(
    tmp_path: Path,
) -> None:
    scene = _scene_with_mirror(tmp_path)
    backend = GenesisBackend(scene, 5, 0.002)
    try:
        backend.materialize()
        layout = backend.get_scene_layout()
        mirror_body = layout.get_entity("mirror").body_ids[0]
        default_mass = backend.get_body_mass().copy()
        requested_mass = default_mass.copy()
        requested_mass[1, mirror_body] += 0.25
        qpos = backend._qpos_cache[1].copy()
        qvel = backend._qvel_cache[1].copy()
        states_before = _object_state_snapshot(backend)
        with pytest.raises(
            ValueError,
            match="body_mass cannot randomize public columns without native Genesis links",
        ):
            backend.set_state(
                np.asarray((1,), dtype=np.intp),
                qpos[[1]],
                qvel[[1]],
                randomization=ResetRandomizationPayload(body_mass=requested_mass[[1]]),
            )
        np.testing.assert_array_equal(backend.get_body_mass(), default_mass)
        for name in backend.get_entity_names():
            for field, values in states_before[name].items():
                np.testing.assert_array_equal(backend.get_entity_state(name)[field], values)
    finally:
        # Keep the process-wide Genesis session alive for the rejection probe.
        pass

    contact_fragment = tmp_path / "mirror-contact.xml"
    contact_fragment.write_text(
        "<mujoco><sensor>"
        "<contact name='object_mirror_force' geom1='object/object_geom' "
        "geom2='mirror/object_geom' data='force' reduce='netforce'/>"
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    scene.fragment_files = (str(contact_fragment),)
    rejection_backend = GenesisBackend(scene, 5, 0.002)
    with pytest.raises(
        RuntimeError,
        match="absent or ambiguous native collision geom identity",
    ):
        rejection_backend.materialize()
