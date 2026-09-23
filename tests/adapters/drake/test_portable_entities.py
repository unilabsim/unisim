"""Native Drake acceptance for the bounded portable-entity profile."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from unisim import create_backend
from unisim.dr.types import (
    FixedVariantLayout,
    FixedVariantPlan,
    ModelSourceDescriptor,
)
from unisim.entities import (
    EntityInitialState,
    EntityStatePatch,
    EntityVariantBinding,
    SceneEntitySpec,
    SceneResetRequest,
)
from unisim.scene import SceneCfg

pytest.importorskip("drake_uni")


def _native_runtime() -> None:
    from drake_uni.runtime import batch_diagnostics

    if not batch_diagnostics().batch_available:
        pytest.skip("Drake native batch extension is not available")


def _articulation(
    path: Path,
    *,
    fixed: bool = False,
    controlled: bool = False,
    position: tuple[float, float, float] = (0.0, 0.0, 0.5),
) -> ModelSourceDescriptor:
    root = "" if fixed else '<freejoint name="root"/>'
    actuator = (
        '<actuator><motor name="drive" joint="hinge"/></actuator>' if controlled else ""
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<mujoco><option gravity="0 0 -9.81"/><worldbody>'
        f'<body name="base" pos="0 0 0">{root}'
        '<inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>'
        '<geom name="base_geom" type="sphere" size=".08" contype="0" conaffinity="0"/>'
        '<body name="link" pos="0 0 .2"><joint name="hinge"/>'
        '<inertial pos="0 0 0" mass=".2" diaginertia=".02 .02 .02"/>'
        '<geom name="link_geom" type="sphere" size=".04" contype="0" conaffinity="0"/>'
        f"</body></body></worldbody>{actuator}</mujoco>",
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _sphere(
    path: Path,
    position: tuple[float, float, float],
    *,
    mass: float = 0.5,
    inertia: float = 0.02,
    radius: float = 0.05,
    collision: bool = False,
) -> ModelSourceDescriptor:
    collision_flags = 'contype="1" conaffinity="1"' if collision else 'contype="0" conaffinity="0"'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<mujoco><option gravity="0 0 -9.81"/><worldbody>'
        '<body name="base"><freejoint name="root"/>'
        f'<inertial pos="0 0 0" mass="{mass}" diaginertia="{inertia} {inertia} {inertia}"/>'
        f'<geom name="shape" type="sphere" size="{radius}" {collision_flags}/>'
        "</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _table(path: Path) -> ModelSourceDescriptor:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<mujoco><option gravity="0 0 -9.81"/><worldbody>'
        '<body name="base"><geom name="surface" type="box" size="1 1 .1"/>'
        "</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _scene(
    tmp_path: Path,
    *,
    reverse: bool = False,
) -> SceneCfg:
    robot = _articulation(
        tmp_path / "robot.xml",
        fixed=True,
        controlled=True,
        position=(0.0, 0.0, 0.1),
    )
    obj = _articulation(
        tmp_path / "object.xml", position=(0.5, 0.0, 1.0)
    )
    ball = _sphere(tmp_path / "ball.xml", (-0.5, 0.0, 1.2))
    table = _table(tmp_path / "table.xml")
    entities = (
        SceneEntitySpec(
            "robot",
            robot,
            root_mode="fixed",
            initial_state=EntityInitialState((0, 0, 0.1)),
        ),
        SceneEntitySpec(
            "object",
            obj,
            root_mode="floating",
            initial_state=EntityInitialState((0.5, 0.0, 1.0)),
        ),
        SceneEntitySpec(
            "ball",
            ball,
            kind="rigid",
            root_mode="floating",
            initial_state=EntityInitialState((-0.5, 0.0, 1.2)),
        ),
        SceneEntitySpec(
            "table",
            table,
            kind="rigid",
            root_mode="fixed",
            initial_state=EntityInitialState((0, 0, -0.1)),
        ),
    )
    return SceneCfg(entity_assets=tuple(reversed(entities)) if reverse else entities)


def _backend(scene: SceneCfg, num_envs: int):
    return create_backend("drake", scene, num_envs=num_envs, sim_dt=0.002)


def _fixed_variant_scene(tmp_path: Path) -> tuple[SceneCfg, tuple[Path, Path]]:
    scene = _scene(tmp_path)
    variant_paths = (tmp_path / "ball-v0.xml", tmp_path / "ball-v1.xml")
    variants = (
        _sphere(
            variant_paths[0],
            (-0.5, 0.0, 1.2),
            mass=0.7,
            inertia=0.025,
            radius=0.06,
            collision=True,
        ),
        _sphere(
            variant_paths[1],
            (-0.5, 0.0, 1.2),
            mass=1.3,
            inertia=0.037,
            radius=0.075,
            collision=True,
        ),
    )
    scene.entity_variant = EntityVariantBinding(
        "ball", FixedVariantPlan(np.array([1, 1, 0, 1, 0]), variants)
    )
    return scene, variant_paths


def test_bounded_layout_passive_state_and_two_floating_roots(tmp_path):
    _native_runtime()
    backend = _backend(_scene(tmp_path), 5)
    try:
        layout = backend.get_scene_layout()
        robot, obj, ball, table = layout.entities
        assert (layout.nq, layout.nv, layout.nu) == (16, 14, 1)
        assert robot.body_names == obj.body_names == ("base", "link")
        assert backend.get_body_ids(
            ("robot/base", "object/base", "ball/base", "table/base")
        ).tolist() == [1, 3, 5, 6]
        assert obj.qpos_indices == (1, 2, 3, 4, 5, 6, 7, 8)
        assert obj.joints[0].qpos_indices == (8,)
        assert ball.root_qpos_indices[0] == 9
        assert table.qpos_indices == ()
        assert backend.num_actuators == 1
        assert backend.num_dof_vel == 2
        assert backend.get_actuator_names() == ("robot/drive",)
        assert backend.get_joint_dof_pos_indices(("object/hinge", "robot/hinge")).tolist() == [1, 0]
        assert backend.get_entity_state("object")["joint_positions"].shape == (5, 1)

        object_root = backend.get_scene_layout().get_entity("object").body_ids[0]
        body = backend.get_body_pos_w(np.array([object_root]))
        np.testing.assert_allclose(
            backend.get_entity_state("object")["root_pose"][:, :3], body[:, 0], atol=1e-12
        )
    finally:
        backend.close()


def test_selected_reset_preserves_entities_and_unselected_worlds(tmp_path):
    _native_runtime()
    backend = _backend(_scene(tmp_path), 5)
    try:
        controls = np.linspace(0.1, 0.5, 5).reshape(5, 1)
        backend.step(controls, nsteps=3)
        before = {
            name: backend.get_entity_state(name) for name in backend.get_entity_names()
        }
        physics_before = backend.get_physics_state()
        layout = backend.get_physics_state_layout()
        assert layout.state_width == physics_before.shape[1]
        assert layout.nmocap == 0
        pose = np.tile([1.5, 2.0, 1.4, 0.5, 0.5, 0.5, 0.5], (2, 1))
        velocity = np.tile([0.2, -0.1, 0.3, 0.4, -0.2, 0.1], (2, 1))
        ids = (3, 0)
        backend.reset_entities(
            SceneResetRequest(
                ids,
                (
                    EntityStatePatch(
                        "object",
                        root_pose=pose,
                        root_velocity=velocity,
                        joint_positions=np.array([[0.7], [-0.4]]),
                    ),
                ),
            )
        )
        state = backend.get_entity_state("object")
        np.testing.assert_allclose(state["root_pose"][list(ids)], pose, atol=1e-12)
        np.testing.assert_allclose(state["root_velocity"][list(ids)], velocity, atol=1e-12)
        np.testing.assert_allclose(state["joint_positions"][list(ids), 0], [0.7, -0.4])
        for name in ("robot", "ball", "table"):
            for field, values in before[name].items():
                np.testing.assert_array_equal(backend.get_entity_state(name)[field], values)
        untouched = [i for i in range(5) if i not in ids]
        np.testing.assert_array_equal(
            backend.get_physics_state()[untouched], physics_before[untouched]
        )

        invalid = SceneResetRequest(
            (0,),
            (EntityStatePatch("object", joint_positions=np.array([[0.2]])),),
            restore_default_controls=True,
        )
        current = backend.get_physics_state()
        with pytest.raises(NotImplementedError, match="restore_default_controls"):
            backend.reset_entities(invalid)
        np.testing.assert_array_equal(backend.get_physics_state(), current)
    finally:
        backend.close()


def test_batched_rows_match_independent_drake_runtimes_and_permutation(tmp_path):
    _native_runtime()
    batch = _backend(_scene(tmp_path / "batch"), 5)
    singles = [
        _backend(_scene(tmp_path / f"single-{i}"), 1) for i in range(5)
    ]
    reverse = _backend(_scene(tmp_path / "reverse", reverse=True), 5)
    try:
        pose = np.array(
            [[0.4 + i * 0.05, 0.1 * i, 1.3, 1.0, 0.0, 0.0, 0.0] for i in range(5)]
        )
        velocity = np.array([[0.05 * i, -0.02, 0.01, 0.2, -0.1, 0.3] for i in range(5)])
        for runtime in (batch, reverse):
            runtime.reset_entities(
                SceneResetRequest(
                    tuple(range(5)),
                    (
                        EntityStatePatch(
                            "object", root_pose=pose, root_velocity=velocity,
                            joint_positions=np.full((5, 1), 0.3),
                        ),
                    ),
                )
            )
        for index, single in enumerate(singles):
            single.reset_entities(
                SceneResetRequest(
                    (0,),
                    (
                        EntityStatePatch(
                            "object",
                            root_pose=pose[index : index + 1],
                            root_velocity=velocity[index : index + 1],
                            joint_positions=np.array([[0.3]]),
                        ),
                    ),
                )
            )
        controls = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        for _ in range(10):
            batch.step(controls)
            reverse.step(controls)
            for index, single in enumerate(singles):
                single.step(controls[index : index + 1])
        for env, single in enumerate(singles):
            for name in ("robot", "object", "ball", "table"):
                actual = batch.get_entity_state(name)
                expected = single.get_entity_state(name)
                for field, values in expected.items():
                    np.testing.assert_allclose(
                        np.asarray(actual[field])[env],
                        np.asarray(values)[0],
                        rtol=2e-6,
                        atol=2e-7,
                    )
                    np.testing.assert_allclose(
                        np.asarray(reverse.get_entity_state(name)[field])[env],
                        np.asarray(values)[0],
                        rtol=2e-6,
                        atol=2e-7,
                    )
    finally:
        batch.close()
        reverse.close()
        for single in singles:
            single.close()


def test_passive_joint_remains_physical_state_not_control(tmp_path):
    _native_runtime()
    backend = _backend(_scene(tmp_path), 2)
    try:
        backend.reset_entities(
            SceneResetRequest(
                (0,),
                (EntityStatePatch("object", joint_velocities=np.array([[2.0]])),),
            )
        )
        before = backend.get_entity_state("object")["joint_positions"][0, 0]
        backend.step(np.zeros((2, 1)), nsteps=2)
        after = backend.get_entity_state("object")["joint_positions"][0, 0]
        assert after != before
        np.testing.assert_array_equal(
            backend.get_entity_state("object")["joint_positions"][1],
            backend.get_entity_default_state("object", env_ids=[1])["joint_positions"][0],
        )
    finally:
        backend.close()


def test_same_layout_variant_native_identity_state_and_cleanup(tmp_path):
    _native_runtime()
    scene, source_paths = _fixed_variant_scene(tmp_path)
    backend = _backend(scene, 5)
    try:
        groups = backend._runtime_groups
        assert [(group.variant, group.public_ids, group.count) for group in groups] == [
            (0, (2, 4), 2),
            (1, (0, 1, 3), 3),
        ]
        capabilities = backend.get_dr_capabilities()
        assert capabilities.supports_fixed_variants
        assert capabilities.supported_fixed_variant_layouts == {
            FixedVariantLayout.SAME_LAYOUT
        }
        assert capabilities.supports_per_env_playback
        actual = {
            group.variant: group.runtime.native_model_properties()
            for group in groups
        }
        assert actual[0].body_names == actual[1].body_names
        assert actual[0].body_masses[5] == pytest.approx(0.7)
        assert actual[1].body_masses[5] == pytest.approx(1.3)
        assert actual[0].body_inertias[5, 0, 0] == pytest.approx(0.025)
        assert actual[1].body_inertias[5, 0, 0] == pytest.approx(0.037)
        assert actual[0].geometry_parameters[-2, 0] == pytest.approx(0.06)
        assert actual[1].geometry_parameters[-2, 0] == pytest.approx(0.075)
        np.testing.assert_allclose(
            actual[0].body_masses[[1, 2, 3, 4, 6]],
            actual[1].body_masses[[1, 2, 3, 4, 6]],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            actual[0].body_inertias[[1, 2, 3, 4, 6]],
            actual[1].body_inertias[[1, 2, 3, 4, 6]],
            atol=1.0e-12,
        )

        before = backend.get_physics_state()
        selected_ids = (2, 0, 3)
        pose = np.array(
            [
                [1.1, -0.2, 1.4, 1.0, 0.0, 0.0, 0.0],
                [1.7, 0.3, 1.6, 0.0, 1.0, 0.0, 0.0],
                [2.3, -0.4, 1.8, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        backend.reset_entities(
            SceneResetRequest(
                selected_ids,
                (EntityStatePatch("ball", root_pose=pose),),
            )
        )
        ball_state = backend.get_entity_state("ball")["root_pose"]
        np.testing.assert_allclose(
            ball_state[list(selected_ids), :3],
            pose[:, :3],
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_array_equal(
            backend.get_physics_state()[[1, 4]], before[[1, 4]]
        )
        backend.step(np.linspace(-0.2, 0.2, 5).reshape(5, 1), nsteps=2)
        for group in groups:
            local_state = group.runtime.physics_state()
            for local_id, public_id in enumerate(group.public_ids):
                np.testing.assert_array_equal(
                    backend.get_physics_state()[public_id], local_state[local_id]
                )

        with pytest.raises(ValueError, match="explicit env_index"):
            backend.get_playback_model()
        assignment = (1, 1, 0, 1, 0)
        playback_paths = [
            backend.get_playback_model(env_index=index) for index in range(5)
        ]
        assert [Path(path).name for path in playback_paths] == [
            f"scene-{variant}.xml" for variant in assignment
        ]
    finally:
        backend.close()
    assert all(not Path(path).exists() for path in set(playback_paths))


def test_old_drakeuni_native_readback_fails_closed_and_cleans_sources(
    tmp_path, monkeypatch
):
    _native_runtime()
    from unisim.backend.drake import backend as module

    original_compile = module.compile_portable_scene
    original_create = module.create_drake_runtime
    variant_paths: list[Path] = []

    class OldRuntimeProxy:
        closed = False

        def __init__(self, runtime):
            self._runtime = runtime

        def close(self):
            self.closed = True
            self._runtime.close()

        def __getattr__(self, name):
            if name == "native_model_properties":
                raise AttributeError("old DrakeUni runtime lacks native_model_properties")
            return getattr(self._runtime, name)

    proxies: list[OldRuntimeProxy] = []

    def compile_and_record(scene, num_envs, sim_dt):
        composed = original_compile(scene, num_envs, sim_dt)
        if composed.variant_plan is not None:
            variant_paths.extend(
                Path(variant.model_file) for variant in composed.variant_plan.variants
            )
        return composed

    def create_old_runtime(config):
        proxy = OldRuntimeProxy(original_create(config))
        proxies.append(proxy)
        return proxy

    monkeypatch.setattr(module, "compile_portable_scene", compile_and_record)
    monkeypatch.setattr(module, "create_drake_runtime", create_old_runtime)
    with pytest.raises(RuntimeError, match="native_model_properties"):
        module.DrakeBackend(_fixed_variant_scene(tmp_path)[0], 5, 0.002)
    assert len(variant_paths) == 2
    assert all(not path.exists() for path in variant_paths)
    assert len(proxies) == 2
    assert all(proxy.closed for proxy in proxies)


def test_fixed_variant_pre_step_callback_runs_once_per_public_substep(tmp_path):
    _native_runtime()
    backend = _backend(_fixed_variant_scene(tmp_path)[0], 5)
    callbacks: list[np.ndarray] = []
    backend.set_pre_step_control(
        lambda owner, ctrl: callbacks.append(ctrl.copy()) or ctrl
    )
    try:
        controls = np.linspace(-0.2, 0.2, 5).reshape(5, 1)
        backend.step(controls, nsteps=2)
    finally:
        backend.set_pre_step_control(None)
        backend.close()
    assert len(callbacks) == 2
    assert all(callback.shape == (5, 1) for callback in callbacks)


def test_unsupported_variant_layouts_and_mirrors_fail_before_materialization(
    tmp_path, monkeypatch
):
    scene = _scene(tmp_path)
    variant_source = ModelSourceDescriptor(str(tmp_path / "robot.xml"))
    scene.entity_variant = EntityVariantBinding(
        "object",
        FixedVariantPlan(
            np.array([0, 1]),
            (variant_source, variant_source),
            layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
        ),
    )
    from unisim.backend.drake import backend as module

    def unavailable():
        raise AssertionError("unsupported profile must not load DrakeUni")

    monkeypatch.setattr(module, "_load_drake_uni_symbols", unavailable)
    with pytest.raises(NotImplementedError, match="same_layout fixed variants"):
        module.DrakeBackend(scene, 2, 0.002)

    scene = _scene(tmp_path / "mirror")
    mirror = SceneEntitySpec(
        "target",
        kind="rigid",
        root_mode="kinematic",
        collision_enabled=False,
        mirror_of="object",
    )
    scene.entity_assets = scene.entity_assets + (mirror,)
    with pytest.raises(NotImplementedError, match="kinematic mirrors"):
        module.DrakeBackend(scene, 2, 0.002)


def test_close_removes_portable_model_and_rejects_use(tmp_path):
    _native_runtime()
    backend = _backend(_scene(tmp_path), 1)
    path = Path(backend.get_playback_model(0))
    backend.close()
    assert not path.exists()
    with pytest.raises(RuntimeError, match="closed"):
        backend.get_entity_state("object")
    backend.close()


def test_layout_mismatch_closes_runtime_and_portable_model(tmp_path, monkeypatch):
    _native_runtime()
    from unisim.backend.drake import backend as module

    original_compile = module.compile_portable_scene
    original_create = module.create_drake_runtime
    composed_paths: list[Path] = []

    def compile_and_record(scene, num_envs, sim_dt):
        composed = original_compile(scene, num_envs, sim_dt)
        composed_paths.append(Path(composed.model_file))
        return composed

    def create_with_mismatched_layout(config):
        runtime = original_create(config)
        info = runtime.model_info()
        mismatched_info = replace(info, nq=int(info.nq) + 1)
        monkeypatch.setattr(runtime, "model_info", lambda: mismatched_info)
        return runtime

    monkeypatch.setattr(module, "compile_portable_scene", compile_and_record)
    monkeypatch.setattr(module, "create_drake_runtime", create_with_mismatched_layout)
    with pytest.raises(ValueError, match="scene nq"):
        module.DrakeBackend(_scene(tmp_path), 2, 0.002)

    assert len(composed_paths) == 1
    assert not composed_paths[0].exists()
