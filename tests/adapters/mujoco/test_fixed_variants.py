"""Fixed variant construction and equivalence tests for the MuJoCo adapter."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
mjbatch = pytest.importorskip("mjbatch")

if not hasattr(mjbatch.Batch, "from_variant_pack") or not hasattr(mjbatch.VariantPack, "builder"):
    pytest.skip("mjbatch VariantPack builder API is required", allow_module_level=True)

import mujoco  # noqa: E402

from unisim import MuJoCoBackend  # noqa: E402
from unisim.backend.mujoco.playback import (  # noqa: E402
    resolve_render_play_model_files,
)
from unisim.dr.types import (  # noqa: E402
    FixedVariantLayout,
    FixedVariantPlan,
    ModelSourceDescriptor,
    ResetRandomizationPayload,
)  # noqa: E402
from unisim.entities import (  # noqa: E402
    EntityVariantBinding,
    SceneEntitySpec,
)
from unisim.scene import SceneCfg  # noqa: E402


def _primitive_xml(radius: str, mass: str, height: str, gravity: str = "-9.81") -> str:
    return f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 {gravity}"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 {height}">
      <freejoint name="root"/>
      <geom name="ball" type="sphere" size="{radius}" mass="{mass}"/>
    </body>
  </worldbody>
</mujoco>
"""


def _actuator_xml(mass: str, ctrlrange: str = "-1 1") -> str:
    return f"""
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 0.2">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="arm" type="capsule" size="0.02 0.08" mass="{mass}"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="hinge" name="hinge_pos" kp="2" ctrlrange="{ctrlrange}"/>
  </actuator>
</mujoco>
"""


def _write_sources(tmp_path: Path, sources: list[str]) -> list[ModelSourceDescriptor]:
    descriptors: list[ModelSourceDescriptor] = []
    for index, source in enumerate(sources):
        path = tmp_path / f"variant-{index}.xml"
        path.write_text(source)
        descriptors.append(ModelSourceDescriptor(str(path)))
    return descriptors


def _step_reference(
    model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, steps: int
) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    state = np.empty(model.nq + model.nv, dtype=np.float64)
    state[: model.nq] = data.qpos
    state[model.nq :] = data.qvel
    return state


def _reference_model(path: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(path)
    spec.option.timestep = 0.002
    return spec.compile()


def test_fixed_variants_use_compiler_defaults_and_persist_per_world(
    tmp_path: Path,
) -> None:
    descriptors = _write_sources(
        tmp_path,
        [
            _primitive_xml("0.08", "1", "0.7"),
            _primitive_xml("0.12", "2", "0.7"),
        ],
    )
    assignment = np.array([0, 1, 0, 1], dtype=np.int32)
    plan = FixedVariantPlan(assignment, tuple(descriptors))
    backend = MuJoCoBackend(
        SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
        num_envs=4,
        sim_dt=0.002,
        base_name="base",
        np_dtype=np.float64,
    )

    capabilities = backend.get_dr_capabilities()
    assert capabilities.fixed_variant_rejections(plan) == ()
    assert capabilities.supports_per_env_playback
    with pytest.raises(ValueError, match="explicit env_index"):
        backend.get_playback_model()
    for term in sorted(capabilities.supported_reset_terms):
        assert backend.get_reset_term_default(term).shape[0] == 4
    default_mass = backend.get_reset_term_default("body_mass")
    assert default_mass.shape == (4, backend.model.nbody)
    assert not default_mass.flags.writeable
    np.testing.assert_allclose(default_mass[:, 1], [1.0, 2.0, 1.0, 2.0])

    backend.materialize()
    backend.reset()
    np.testing.assert_allclose(backend._qpos_view[:, 2], [0.7, 0.7, 0.7, 0.7])
    np.testing.assert_allclose(backend._pool.expand("body_mass")[:, 1], default_mass[:, 1])

    qpos = backend._qpos_view.copy()
    qvel = backend._qvel_view.copy()
    backend.set_state(
        np.array([1, 3]),
        qpos[[1, 3]],
        qvel[[1, 3]],
        randomization=ResetRandomizationPayload(
            base_mass_delta=np.array([0.25, -0.5]),
            gravity=np.tile(np.array([0.0, 0.0, -9.5]), (2, 1)),
        ),
    )
    np.testing.assert_allclose(backend._pool.expand("body_mass")[[1, 3], 1], [2.25, 1.5])
    np.testing.assert_allclose(backend._pool.expand("gravity")[[1, 3], 2], -9.5, rtol=0.0, atol=0.0)

    # A plain reset does not erase fixed identities or reset-time model writes.
    backend.reset()
    np.testing.assert_allclose(backend._pool.expand("body_mass")[[1, 3], 1], [2.25, 1.5])


def test_fixed_variants_reject_shared_actuator_parameter_changes(
    tmp_path: Path,
) -> None:
    descriptors = _write_sources(
        tmp_path,
        [_actuator_xml("1"), _actuator_xml("2", ctrlrange="-2 2")],
    )
    plan = FixedVariantPlan(np.array([0, 1], dtype=np.int32), tuple(descriptors))

    with pytest.raises(ValueError, match="changes shared field actuator_ctrlrange"):
        MuJoCoBackend(
            SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
            num_envs=2,
            sim_dt=0.002,
        )


def test_same_layout_variant_build_releases_variant_specs_and_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Variant pool construction must not retain one spec/model per variant.

    A compiled MjSpec embeds the full mesh payload, so retaining one per
    variant is what OOMed large SAME_LAYOUT pools (unilabsim/unisim#280). The
    streaming build keeps only the canonical model plus small per-variant rows.
    """
    descriptors = _write_sources(
        tmp_path,
        [_primitive_xml(f"0.0{index + 5}", str(index + 1), "0.7") for index in range(6)],
    )
    assignment = np.arange(6, dtype=np.int32)
    plan = FixedVariantPlan(assignment, tuple(descriptors))

    original_from_file = mujoco.MjSpec.from_file
    original_compile = mujoco.MjSpec.compile
    spec_refs: list[weakref.ref[mujoco.MjSpec]] = []
    model_refs: list[weakref.ref[mujoco.MjModel]] = []

    def tracked_from_file(*args: object, **kwargs: object) -> mujoco.MjSpec:
        spec = original_from_file(*args, **kwargs)
        spec_refs.append(weakref.ref(spec))
        return spec

    def tracked_compile(self: mujoco.MjSpec, *args: object, **kwargs: object) -> mujoco.MjModel:
        model = original_compile(self, *args, **kwargs)
        model_refs.append(weakref.ref(model))
        return model

    monkeypatch.setattr(mujoco.MjSpec, "from_file", staticmethod(tracked_from_file))
    monkeypatch.setattr(mujoco.MjSpec, "compile", tracked_compile)
    backend = MuJoCoBackend(
        SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
        num_envs=6,
        sim_dt=0.002,
        base_name="base",
        np_dtype=np.float64,
    )
    gc.collect()
    alive_specs = sum(ref() is not None for ref in spec_refs)
    alive_models = sum(ref() is not None for ref in model_refs)
    assert alive_specs == 0
    # The canonical pack model stays; the per-variant realizations are gone.
    assert alive_models <= 2
    assert backend._fixed_variant_build is not None
    assert backend._fixed_variant_build.pack.num_variants == 6


def test_all_advertised_reset_defaults_are_canonical_and_read_only(
    tmp_path: Path,
) -> None:
    descriptor = _write_sources(tmp_path, [_primitive_xml("0.08", "1", "0.7")])[0]
    backend = MuJoCoBackend(
        SceneCfg(model_file=descriptor.model_file),
        num_envs=2,
        sim_dt=0.002,
    )
    terms = backend.get_dr_capabilities().supported_reset_terms
    assert terms
    for term in sorted(terms):
        default = backend.get_reset_term_default(term)
        assert not default.flags.writeable
        assert default.ndim >= 0


def test_same_layout_primitive_variants_match_independent_compiles(
    tmp_path: Path,
) -> None:
    descriptors = _write_sources(
        tmp_path,
        [
            _primitive_xml("0.08", "1", "0.7"),
            _primitive_xml("0.12", "2", "0.7"),
        ],
    )
    assignment = np.array([0, 1], dtype=np.int32)
    plan = FixedVariantPlan(assignment, tuple(descriptors))
    backend = MuJoCoBackend(
        SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
        num_envs=2,
        sim_dt=0.002,
        base_name="base",
        np_dtype=np.float64,
    )
    backend.materialize()
    backend.reset()
    qpos_before = backend._qpos_view.copy()
    qvel_before = backend._qvel_view.copy()
    ctrl = np.zeros((2, backend.num_actuators))
    for _ in range(20):
        backend.step(ctrl)

    build = backend._fixed_variant_build
    assert build is not None
    for env_index, variant in enumerate(assignment):
        expected = _step_reference(
            _reference_model(build.plan.variants[int(variant)].model_file),
            qpos_before[env_index],
            qvel_before[env_index],
            steps=20,
        )
        actual = np.concatenate((backend._qpos_view[env_index], backend._qvel_view[env_index]))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


TETRAHEDRON_OBJ = """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
"""


def _mesh_xml(obj_path: Path, *, include_b: bool, scale: str = "1 1 1") -> str:
    mesh_b = f'<mesh name="b" file="{obj_path}" scale="{scale}"/>' if include_b else ""
    geom_b = '<geom name="b" type="mesh" mesh="b" mass="0.5"/>' if include_b else ""
    return f"""
<mujoco>
  <option timestep="0.002"/>
  <asset>
    <mesh name="a" file="{obj_path}" scale="1 1 1"/>{mesh_b}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 0.8">
      <freejoint name="root"/>
      <geom name="a" type="mesh" mesh="a" mass="1"/>{geom_b}
    </body>
  </worldbody>
</mujoco>
"""


def test_uniform_public_layout_pads_optional_mesh_slots(tmp_path: Path) -> None:
    obj_path = tmp_path / "tetrahedron.obj"
    obj_path.write_text(TETRAHEDRON_OBJ)
    descriptors = _write_sources(
        tmp_path,
        [_mesh_xml(obj_path, include_b=True), _mesh_xml(obj_path, include_b=False)],
    )
    assignment = np.array([0, 1], dtype=np.int32)
    plan = FixedVariantPlan(
        assignment,
        tuple(descriptors),
        layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
    )
    backend = MuJoCoBackend(
        SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
        num_envs=2,
        sim_dt=0.002,
        base_name="base",
        np_dtype=np.float64,
    )
    backend.materialize()
    backend.reset()
    qpos_before = backend._qpos_view.copy()
    qvel_before = backend._qvel_view.copy()
    ctrl = np.zeros((2, backend.num_actuators))
    for _ in range(20):
        backend.step(ctrl)

    build = backend._fixed_variant_build
    assert build is not None
    for env_index, variant in enumerate(assignment):
        expected = _step_reference(
            _reference_model(build.plan.variants[int(variant)].model_file),
            qpos_before[env_index],
            qvel_before[env_index],
            steps=20,
        )
        actual = np.concatenate((backend._qpos_view[env_index], backend._qvel_view[env_index]))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    assert backend.model.ngeom == 3
    assert backend.get_reset_term_default("geom_size").shape == (2, 3, 3)
    assert backend.get_playback_model(0).ngeom == 3
    assert backend.get_playback_model(1).ngeom == 2
    np.testing.assert_array_equal(
        backend._pool.expand("geom_type")[1, backend.get_geom_id("b")],
        int(mujoco.mjtGeom.mjGEOM_NONE),
    )


def _entity_mesh_xml(
    obj_path: Path,
    *,
    include_head: bool,
    scale: str = "1 1 1",
    derived_inertia: bool = False,
) -> str:
    head_mesh = f'<mesh name="head" file="{obj_path}" scale="{scale}"/>' if include_head else ""
    head_geom = '<geom name="head" type="mesh" mesh="head" mass="0"/>' if include_head else ""
    handle_mass = "1" if derived_inertia else "0"
    inertial = (
        "" if derived_inertia else '<inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>'
    )
    return f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    <mesh name="handle" file="{obj_path}"/>{head_mesh}
  </asset>
  <worldbody>
    <body name="base" pos="0 0 0.8">
      {inertial}
      <freejoint name="root"/>
      <geom name="handle" type="mesh" mesh="handle" mass="{handle_mass}"/>{head_geom}
    </body>
  </worldbody>
</mujoco>
"""


def _table_xml() -> str:
    return """
<mujoco>
  <worldbody>
    <body name="base">
      <geom type="plane" size="1 1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_uniform_public_entity_mesh_variants_execute_and_play_back(
    tmp_path: Path,
) -> None:
    obj_path = tmp_path / "tetrahedron.obj"
    obj_path.write_text(TETRAHEDRON_OBJ)
    descriptors = _write_sources(
        tmp_path,
        [
            _entity_mesh_xml(obj_path, include_head=False),
            _entity_mesh_xml(obj_path, include_head=True),
            _entity_mesh_xml(obj_path, include_head=True, scale=".5 .5 .5"),
        ],
    )
    assignment = np.array([0, 1, 2], dtype=np.int32)
    plan = FixedVariantPlan(
        assignment,
        tuple(descriptors),
        layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
    )
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec("object", descriptors[0], kind="rigid"),
            SceneEntitySpec(
                "mirror",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="object",
            ),
        ),
        entity_variant=EntityVariantBinding("object", plan),
    )
    backend = MuJoCoBackend(
        scene,
        num_envs=3,
        sim_dt=0.002,
        base_name="object/base",
        np_dtype=np.float64,
    )
    backend.materialize()
    backend.reset()
    qpos_before = backend._qpos_view.copy()
    qvel_before = backend._qvel_view.copy()
    ctrl = np.zeros((3, backend.num_actuators))
    for _ in range(20):
        backend.step(ctrl)

    build = backend._fixed_variant_build
    assert build is not None
    for env_index, variant in enumerate(assignment):
        expected = _step_reference(
            _reference_model(build.plan.variants[int(variant)].model_file),
            qpos_before[env_index],
            qvel_before[env_index],
            steps=20,
        )
        actual = np.concatenate((backend._qpos_view[env_index], backend._qvel_view[env_index]))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    assert backend.get_dr_capabilities().fixed_variant_rejections(plan) == ()
    assert backend.model.ngeom == 4
    assert [geom.name for geom in backend.get_scene_layout().get_entity("object").geoms] == [
        "handle",
        "head",
    ]
    head_id = backend.get_geom_id("object/head")
    np.testing.assert_array_equal(
        backend._pool.expand("geom_type")[0, head_id], int(mujoco.mjtGeom.mjGEOM_NONE)
    )
    assert backend._pool.expand("geom_dataid")[0, head_id] == -1
    assert backend._pool.expand("geom_dataid")[1:, head_id].min() >= 0
    assert [backend.get_playback_model(index).ngeom for index in range(3)] == [2, 4, 4]


def test_uniform_entity_mesh_variants_compare_public_body_names_across_entities(
    tmp_path: Path,
) -> None:
    obj_path = tmp_path / "tetrahedron.obj"
    obj_path.write_text(TETRAHEDRON_OBJ)
    robot_path = tmp_path / "robot.xml"
    robot_path.write_text(
        '<mujoco><worldbody><body name="base">'
        '<geom name="link" type="box" size=".1 .1 .1" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    descriptors = _write_sources(
        tmp_path,
        [
            _entity_mesh_xml(obj_path, include_head=False),
            _entity_mesh_xml(obj_path, include_head=True),
        ],
    )
    plan = FixedVariantPlan(
        np.array([0, 1], dtype=np.int32),
        tuple(descriptors),
        layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
    )
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                ModelSourceDescriptor(str(robot_path)),
                root_mode="fixed",
            ),
            SceneEntitySpec("object", descriptors[0], kind="rigid"),
        ),
        entity_variant=EntityVariantBinding("object", plan),
    )
    backend = MuJoCoBackend(
        scene,
        num_envs=2,
        sim_dt=0.002,
        base_name="object/base",
        np_dtype=np.float64,
    )

    backend.materialize()

    assert [backend.get_playback_model(index).ngeom for index in range(2)] == [2, 3]


def test_uniform_entity_mesh_variants_ignore_unrelated_anonymous_geoms(
    tmp_path: Path,
) -> None:
    obj_path = tmp_path / "tetrahedron.obj"
    obj_path.write_text(TETRAHEDRON_OBJ)
    descriptors = _write_sources(
        tmp_path,
        [
            _entity_mesh_xml(obj_path, include_head=False),
            _entity_mesh_xml(obj_path, include_head=True),
        ],
    )
    table_path = tmp_path / "table.xml"
    table_path.write_text(_table_xml())
    table = ModelSourceDescriptor(str(table_path))
    plan = FixedVariantPlan(
        np.array([0, 1], dtype=np.int32),
        tuple(descriptors),
        layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
    )
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec("object", descriptors[0], kind="rigid"),
            SceneEntitySpec("table", table, kind="rigid", root_mode="fixed"),
        ),
        entity_variant=EntityVariantBinding("object", plan),
    )
    backend = MuJoCoBackend(
        scene,
        num_envs=2,
        sim_dt=0.002,
        base_name="object/base",
        np_dtype=np.float64,
    )
    backend.materialize()
    backend.reset()

    assert backend.model.ngeom == 3
    assert backend.model.geom("table/base::geom0").id >= 0
    assert [backend.get_playback_model(index).ngeom for index in range(2)] == [2, 3]


def test_uniform_entity_mesh_variants_normalize_derived_body_simple(
    tmp_path: Path,
) -> None:
    obj_path = tmp_path / "tetrahedron.obj"
    obj_path.write_text(TETRAHEDRON_OBJ)
    descriptors = _write_sources(
        tmp_path,
        [
            _entity_mesh_xml(obj_path, include_head=False),
            _entity_mesh_xml(obj_path, include_head=True, derived_inertia=True),
        ],
    )
    source_body_simple = [
        mujoco.MjModel.from_xml_path(descriptor.model_file).body_simple[-1]
        for descriptor in descriptors
    ]
    assert source_body_simple == [1, 0]
    plan = FixedVariantPlan(
        np.array([0, 1], dtype=np.int32),
        tuple(descriptors),
        layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
    )
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("object", descriptors[0], kind="rigid"),),
        entity_variant=EntityVariantBinding("object", plan),
    )
    backend = MuJoCoBackend(
        scene,
        num_envs=2,
        sim_dt=0.002,
        base_name="object/base",
        np_dtype=np.float64,
    )
    backend.materialize()
    backend.reset()

    assert backend.model.ngeom == 2
    assert backend.model.body_simple[backend.model.body("object/base").id] == 0


def test_uniform_entity_variant_executor_materializes_assigned_and_canonical(
    tmp_path: Path,
) -> None:
    obj_path = tmp_path / "tetrahedron.obj"
    obj_path.write_text(TETRAHEDRON_OBJ)
    descriptors = _write_sources(
        tmp_path,
        [
            _entity_mesh_xml(obj_path, include_head=False),
            *(_entity_mesh_xml(obj_path, include_head=True) for _ in range(3)),
        ],
    )
    plan = FixedVariantPlan(
        np.array([0, 1, 0], dtype=np.int32),
        tuple(descriptors),
        layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
    )
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("object", descriptors[0], kind="rigid"),),
        entity_variant=EntityVariantBinding("object", plan),
    )
    backend = MuJoCoBackend(
        scene,
        num_envs=3,
        sim_dt=0.002,
        base_name="object/base",
        np_dtype=np.float64,
    )

    build = backend._fixed_variant_build
    assert build is not None
    assert len(plan.variants) == 4
    assert build.pack.num_variants == 2
    np.testing.assert_array_equal(build.executor_assignment, [0, 1, 0])
    assert backend.get_playback_model(0).ngeom == 1
    assert backend.get_playback_model(1).ngeom == 2


def test_unassigned_catalog_variant_still_fails_closed(tmp_path: Path) -> None:
    descriptors = _write_sources(
        tmp_path,
        [
            _actuator_xml("1"),
            _actuator_xml("1"),
            _actuator_xml("1", ctrlrange="-2 2"),
        ],
    )
    plan = FixedVariantPlan(np.array([0, 0], dtype=np.int32), tuple(descriptors))

    with pytest.raises(ValueError, match="changes shared field actuator_ctrlrange"):
        MuJoCoBackend(
            SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
            num_envs=2,
            sim_dt=0.002,
        )


def test_uniform_entity_composition_releases_noncanonical_loaded_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unisim.mjcf_compiler import compose_scene

    obj_path = tmp_path / "tetrahedron.obj"
    obj_path.write_text(TETRAHEDRON_OBJ)
    descriptors = _write_sources(
        tmp_path,
        [_entity_mesh_xml(obj_path, include_head=index % 2 == 1) for index in range(8)],
    )
    robot_path = tmp_path / "robot.xml"
    robot_path.write_text(
        '<mujoco><worldbody><body name="base">'
        '<geom name="link" type="box" size=".1 .1 .1" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", ModelSourceDescriptor(str(robot_path)), root_mode="fixed"),
            SceneEntitySpec("object", descriptors[0], kind="rigid"),
        ),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(
                np.arange(8, dtype=np.int32),
                tuple(descriptors),
                layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
            ),
        ),
    )
    original_load = mujoco.MjModel.from_xml_path
    model_refs: list[weakref.ref[mujoco.MjModel]] = []

    def tracked_load(path: str, *args: object, **kwargs: object):
        model = original_load(path, *args, **kwargs)
        model_refs.append(weakref.ref(model))
        return model

    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", staticmethod(tracked_load))
    with compose_scene(scene, 8, 0.002) as composed:
        gc.collect()
        assert composed.model.ngeom > 0
        assert len(model_refs) == len(descriptors) + 1
        assert sum(ref() is not None for ref in model_refs) == 1


def test_fixed_variants_preserve_injected_body_sensors(tmp_path: Path) -> None:
    descriptors = _write_sources(
        tmp_path,
        [
            _primitive_xml("0.08", "1", "0.7"),
            _primitive_xml("0.12", "2", "0.7"),
        ],
    )
    plan = FixedVariantPlan(np.array([0, 1], dtype=np.int32), tuple(descriptors))
    backend = MuJoCoBackend(
        SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
        num_envs=2,
        sim_dt=0.002,
        base_name="base",
        np_dtype=np.float64,
        add_body_sensors=True,
        tracked_body_names=("base",),
    )
    backend.materialize()
    backend.reset()

    assert backend._tracked_pos_w_all.shape == (2, 1, 3)
    np.testing.assert_allclose(backend._tracked_pos_w_all[:, 0, 2], [0.7, 0.7])


def test_fixed_variants_apply_constructor_actuator_gain_configuration(
    tmp_path: Path,
) -> None:
    descriptors = _write_sources(tmp_path, [_actuator_xml("1"), _actuator_xml("2")])
    plan = FixedVariantPlan(np.array([0, 1], dtype=np.int32), tuple(descriptors))
    backend = MuJoCoBackend(
        SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
        num_envs=2,
        sim_dt=0.002,
        base_name="base",
        np_dtype=np.float64,
        position_actuator_gains={"kp": 7.0, "kd": 0.3},
    )
    backend.materialize()

    gain = backend._pool.expand("actuator_gainprm")
    bias = backend._pool.expand("actuator_biasprm")
    np.testing.assert_allclose(gain[:, 0, 0], 7.0)
    np.testing.assert_allclose(bias[:, 0, 1], -7.0)
    np.testing.assert_allclose(bias[:, 0, 2], -0.3)
    np.testing.assert_allclose(backend.get_reset_term_default("kp"), [[7.0], [7.0]])
    np.testing.assert_allclose(backend.get_reset_term_default("kd"), [[0.3], [0.3]])


def test_fixed_variant_compiler_defaults_reach_expanded_model_fields(
    tmp_path: Path,
) -> None:
    descriptors = _write_sources(
        tmp_path,
        [
            _primitive_xml("0.08", "1", "0.7"),
            _primitive_xml("0.12", "2", "0.7"),
        ],
    )
    plan = FixedVariantPlan(np.array([0, 1, 1, 0], dtype=np.int32), tuple(descriptors))
    backend = MuJoCoBackend(
        SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
        num_envs=4,
        sim_dt=0.002,
        base_name="base",
        np_dtype=np.float64,
    )
    backend.materialize()

    np.testing.assert_allclose(backend._pool.expand("geom_size")[:, 1, 0], [0.08, 0.12, 0.12, 0.08])


def test_optional_mesh_slots_reject_same_layout_claim(tmp_path: Path) -> None:
    obj_path = tmp_path / "tetrahedron.obj"
    obj_path.write_text(TETRAHEDRON_OBJ)
    descriptors = _write_sources(
        tmp_path,
        [_mesh_xml(obj_path, include_b=True), _mesh_xml(obj_path, include_b=False)],
    )
    plan = FixedVariantPlan(np.array([0, 1], dtype=np.int32), tuple(descriptors))
    with pytest.raises(ValueError, match="uniform_public_layout"):
        MuJoCoBackend(
            SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
            num_envs=2,
            sim_dt=0.002,
        )


def test_public_topology_changes_fail_closed(tmp_path: Path) -> None:
    changed = _primitive_xml("0.12", "2", "0.7").replace(
        '<geom name="ball"', '<site name="extra"/><geom name="ball"'
    )
    descriptors = _write_sources(
        tmp_path,
        [_primitive_xml("0.08", "1", "0.7"), changed],
    )
    plan = FixedVariantPlan(
        np.array([0, 1], dtype=np.int32),
        tuple(descriptors),
        layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
    )
    with pytest.raises(ValueError, match="same public topology"):
        MuJoCoBackend(
            SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
            num_envs=2,
            sim_dt=0.002,
        )


def test_missing_non_mesh_geom_slot_fails_closed(tmp_path: Path) -> None:
    changed = _primitive_xml("0.12", "2", "0.7").replace(
        '<geom name="ball" type="sphere" size="0.12" mass="2"/>',
        '<inertial mass="2" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>',
    )
    descriptors = _write_sources(
        tmp_path,
        [_primitive_xml("0.08", "1", "0.7"), changed],
    )
    plan = FixedVariantPlan(
        np.array([0, 1], dtype=np.int32),
        tuple(descriptors),
        layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
    )
    with pytest.raises(ValueError, match="only optional mesh-geom slots"):
        MuJoCoBackend(
            SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
            num_envs=2,
            sim_dt=0.002,
        )


def test_construction_time_plan_resolves_per_env_playback(tmp_path: Path) -> None:
    descriptors = _write_sources(
        tmp_path,
        [_primitive_xml("0.08", "1", "0.7"), _primitive_xml("0.12", "2", "0.7")],
    )
    plan = FixedVariantPlan(np.array([0, 1], dtype=np.int32), tuple(descriptors))
    backend = MuJoCoBackend(
        SceneCfg(model_file=descriptors[0].model_file, fixed_variant_plan=plan),
        num_envs=2,
        sim_dt=0.002,
    )
    backend.materialize()

    class Env:
        def __init__(self, value: MuJoCoBackend) -> None:
            self._backend = value

        def get_playback_model(self, env_index: int):
            return self._backend.get_playback_model(env_index)

    output = Path(tmp_path) / "playback"
    output.mkdir()
    model_files = resolve_render_play_model_files(Env(backend), num_envs=2, tmp_dir=output)
    assert len(model_files) == 2
    assert all(Path(path).is_file() for path in model_files)
    assert mujoco.MjModel.from_binary_path(model_files[0]).ngeom == backend.model.ngeom
