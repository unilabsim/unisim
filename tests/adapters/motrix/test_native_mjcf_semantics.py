"""Issue #573 native MSD materialization and single-MJCF state regressions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")
pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from unisim import MotrixBackend, TensorIOSpec  # noqa: E402
from unisim.scene import SceneCfg, TerrainSceneCfg  # noqa: E402
from unisim.terrain.generator import (  # noqa: E402
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainHeightField,
    TerrainOutput,
)

GENERAL_POS_MODEL = """<mujoco model='motrix-native-position'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='j1' type='hinge' axis='0 0 1'/>
    <geom type='box' size='0.05 0.05 0.05' mass='1'/></body></worldbody>
  <actuator><general name='g1' joint='j1' input='pos' biastype='affine'
    gainprm='10' biasprm='0 -10 -1'/></actuator>
</mujoco>"""

FIXED_OBJECT_MODEL = """<mujoco model='motrix-fixed-object'>
  <option timestep='0.002' gravity='0 0 0'/>
  <worldbody>
    <body name='hand'>
      <joint name='driven' type='hinge' axis='1 0 0'/>
      <joint name='passive' type='hinge' axis='0 1 0'/>
      <geom type='sphere' size='.05'/>
    </body>
    <body name='object' pos='1 2 3'>
      <freejoint name='object_free'/><geom type='sphere' size='.05'/>
    </body>
  </worldbody>
  <actuator><motor name='drive' joint='driven'/></actuator>
</mujoco>"""

# Two actuators share the hinge and one drives the slide, so both aggregate
# shapes (multi-source and single-source with a non-unit gear) are covered.
FORCE_MODEL = """<mujoco model='motrix-aggregate-force'>
  <option timestep='0.01' gravity='0 0 0'/>
  <worldbody><body name='base'>
    <joint name='hinge' type='hinge' axis='0 0 1'/>
    <joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05' mass='1'/></body></worldbody>
  <actuator>
    <motor name='m1' joint='hinge' gear='2' ctrlrange='-1 1'/>
    <motor name='m2' joint='hinge' gear='3' ctrlrange='-1 1'/>
    <motor name='m3' joint='slide' gear='0.5' ctrlrange='-1 1'/>
  </actuator>
  <sensor>
    <actuatorfrc name='a1' actuator='m1'/><actuatorfrc name='a2' actuator='m2'/>
    <actuatorfrc name='a3' actuator='m3'/>
    <jointactuatorfrc name='jaf_hinge' joint='hinge'/>
    <jointactuatorfrc name='jaf_slide' joint='slide'/>
  </sensor>
</mujoco>"""


def _flat_terrain() -> SubTerrainCfg:
    class FlatTerrain(SubTerrainCfg):
        horizontal_scale: float = 0.05
        vertical_scale: float = 0.005

        def function(self, difficulty: float, rng: np.random.Generator) -> TerrainOutput:
            del difficulty, rng
            noise = np.zeros((int(self.size[0] / 0.05), int(self.size[1] / 0.05)), dtype=np.int16)
            heightfield = TerrainHeightField(noise, self.size, 0.05, 0.005, 0, 0, 0, 0.05, 0.0)
            return TerrainOutput(np.asarray((0.1, 0.1, 0.0)), heightfield)

    return FlatTerrain()


@pytest.mark.parametrize("with_terrain", [False, True])
def test_read_only_source_with_general_position_and_terrain(
    tmp_path: Path, with_terrain: bool
) -> None:
    source_dir = tmp_path / "read-only-assets"
    source_dir.mkdir()
    model_path = source_dir / "model.xml"
    model_path.write_text(GENERAL_POS_MODEL, encoding="utf-8")
    fragment_path = source_dir / "task.xml"
    fragment_path.write_text(
        "<mujoco><keyframe><key name='home' qpos='0.25'/></keyframe></mujoco>",
        encoding="utf-8",
    )
    scene = SceneCfg(str(model_path), fragment_files=(str(fragment_path),))
    if with_terrain:
        scene = SceneCfg(
            str(model_path),
            fragment_files=(str(fragment_path),),
            terrain=TerrainSceneCfg(
                generator=TerrainGeneratorCfg(
                    size=(0.2, 0.2), sub_terrains={"flat": _flat_terrain()}, seed=7
                )
            ),
        )
    source_dir.chmod(0o555)
    backend = None
    try:
        backend = MotrixBackend(scene, num_envs=1, sim_dt=0.01)
        kp, kd = backend.get_actuator_gains()
        np.testing.assert_allclose(kp, [10.0])
        np.testing.assert_allclose(kd, [1.0])
        np.testing.assert_allclose(backend.get_keyframe_qpos("home"), [0.25])
        assert {path.name for path in source_dir.iterdir()} == {"model.xml", "task.xml"}
    finally:
        if backend is not None:
            backend.close()
        source_dir.chmod(0o755)


def test_joint_actuator_force_matches_mujoco_with_gear_and_multiple_actuators(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "force.xml"
    model_path.write_text(
        """<mujoco><option timestep='0.01' gravity='0 0 0'/>
  <worldbody><body name='base'><joint name='j1' type='hinge'/>
    <geom type='sphere' size='.1' mass='1'/></body></worldbody>
  <actuator><motor name='m1' joint='j1' gear='2' ctrlrange='-1 1'/>
    <motor name='m2' joint='j1' gear='3' ctrlrange='-1 1'/></actuator>
  <sensor><actuatorfrc name='a1' actuator='m1'/><actuatorfrc name='a2' actuator='m2'/>
    <jointactuatorfrc name='jaf' joint='j1'/></sensor>
</mujoco>""",
        encoding="utf-8",
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.ctrl[:] = [0.5, 0.25]
    mujoco.mj_step(model, data)
    expected = data.sensor("jaf").data.copy()

    backend = MotrixBackend(SceneCfg(str(model_path)), num_envs=2, sim_dt=0.01)
    try:
        backend._host_step(np.tile(np.asarray([0.5, 0.25], dtype=np.float32), (2, 1)))
        np.testing.assert_allclose(
            backend._host_read_sensor("jaf"), np.tile(expected, (2, 1)), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            backend._host_read_sensor_rows_batch(("jaf",), np.asarray([1, 0])),
            np.tile(expected, (2, 1)),
            rtol=1e-6,
            atol=1e-6,
        )
    finally:
        backend.close()


def test_joint_actuator_force_requires_scalar_actuated_joint(tmp_path: Path) -> None:
    model_path = tmp_path / "unsupported-force.xml"
    model_path.write_text(
        "<mujoco><worldbody><body name='base'><joint name='ball' type='ball'/>"
        "<geom type='sphere' size='.1'/></body></worldbody>"
        "<actuator><motor name='m' joint='ball' gear='1 0 0'/></actuator>"
        "<sensor><jointactuatorfrc name='jaf' joint='ball'/></sensor></mujoco>",
        encoding="utf-8",
    )
    with pytest.raises(NotImplementedError, match="single-DoF"):
        MotrixBackend(SceneCfg(str(model_path)), num_envs=1, sim_dt=0.01)

    model_path.write_text(
        "<mujoco><worldbody><body name='base'><joint name='j1'/>"
        "<geom type='sphere' size='.1'/></body></worldbody>"
        "<sensor><jointactuatorfrc name='jaf' joint='j1'/></sensor></mujoco>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no joint-targeting actuator"):
        MotrixBackend(SceneCfg(str(model_path)), num_envs=1, sim_dt=0.01)


def test_authored_solver_options_survive_and_explicit_iterations_override(tmp_path: Path) -> None:
    model_path = tmp_path / "solver.xml"
    model_path.write_text(
        "<mujoco><option solver='Newton' iterations='20' tolerance='1e-8'/>"
        "<worldbody><body name='base'><joint name='j1'/>"
        "<geom type='sphere' size='.1'/></body></worldbody></mujoco>",
        encoding="utf-8",
    )
    backend = MotrixBackend(SceneCfg(str(model_path)), num_envs=1, sim_dt=0.002)
    try:
        assert str(backend._scene_world.simulate_option.solver_type) == "SolverType.Newton"
        assert backend.model.options.max_iterations == 20
        assert backend.model.options.solver_tolerance == pytest.approx(1e-8)
        fields = {field.field: field for field in backend.get_import_report().fields}
        assert fields["solver"].effective == "Newton"
        assert fields["solver_iterations"].effective == 20
        assert fields["solver_tolerance"].effective == pytest.approx(1e-8)
        assert "no integrator selection" in fields["integrator"].reason
        for name in ("cone", "impratio", "ls_iterations", "maxhullvert"):
            assert "ignores" in fields[name].reason
        assert "VolumeTooSmall" in fields["collision_filter"].reason
    finally:
        backend.close()

    overridden = MotrixBackend(
        SceneCfg(str(model_path)), num_envs=1, sim_dt=0.002, max_iterations=7
    )
    try:
        assert overridden.model.options.max_iterations == 7
    finally:
        overridden.close()


def test_fixed_base_tensor_and_packed_reset_include_object_and_passive_joint(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    model_path = tmp_path / "fixed-object.xml"
    model_path.write_text(FIXED_OBJECT_MODEL, encoding="utf-8")
    backend = MotrixBackend(
        SceneCfg(str(model_path)), num_envs=2, sim_dt=0.002, base_name="hand"
    )
    try:
        widths = backend.get_public_state_widths()
        qpos = torch.zeros((1, widths.nq), dtype=torch.float32)
        qpos[0] = torch.tensor([0.2, -0.3, 4, 5, 6, 0.5, 0.5, 0.5, -0.5])
        qvel = torch.arange(widths.nv, dtype=torch.float32).reshape(1, -1) / 10
        backend.set_state_tensor(torch.tensor([1]), qpos, qvel)
        state = backend.get_state_views(device="cpu")
        torch.testing.assert_close(state["qpos"][1], qpos[0])
        torch.testing.assert_close(state["qvel"][1], qvel[0])

        transaction = backend.bind_packed_reset(device="cpu")
        default_qpos = torch.tensor(backend.get_default_qpos(), dtype=torch.float32)
        default_qvel = torch.tensor(backend.get_init_qvel(), dtype=torch.float32)
        transaction.set_state_default(default_qpos, default_qvel)
        packed_qpos = default_qpos.expand(2, -1).clone()
        packed_qvel = default_qvel.expand(2, -1).clone()
        packed_qpos[1] = qpos[0]
        packed_qvel[1] = qvel[0]
        reset_mask = torch.tensor([False, True])
        transaction.begin_state_mask(reset_mask)
        transaction.commit_state_mask(reset_mask, packed_qpos, packed_qvel)
        state = backend.get_state_views(device="cpu")
        torch.testing.assert_close(state["qpos"][1], packed_qpos[1])
        torch.testing.assert_close(state["qvel"][1], packed_qvel[1])
    finally:
        backend.close()


class _CountingModel:
    """Forward to the native model while counting batched and single sensor reads."""

    def __init__(self, model: object) -> None:
        self._model = model
        self.batched_reads = 0
        self.single_reads = 0

    def __getattr__(self, name: str) -> object:
        attribute = getattr(self._model, name)
        if name == "get_sensor_values":

            def batched(*args: object, **kwargs: object) -> object:
                self.batched_reads += 1
                return attribute(*args, **kwargs)

            return batched
        if name == "get_sensor_value":

            def single(*args: object, **kwargs: object) -> object:
                self.single_reads += 1
                return attribute(*args, **kwargs)

            return single
        return attribute


@contextmanager
def _count_native_sensor_reads(backend: MotrixBackend) -> Iterator[_CountingModel]:
    """Count the native sensor calls the backend issues inside the context."""
    counting = _CountingModel(backend._model)
    backend._model = counting
    try:
        yield counting
    finally:
        backend._model = counting._model


def test_packed_sensor_reads_use_one_native_call_with_aggregates(tmp_path: Path) -> None:
    """Aggregates must not turn a packed read into one native call per sensor."""
    torch = pytest.importorskip("torch")
    model_path = tmp_path / "force.xml"
    model_path.write_text(FORCE_MODEL, encoding="utf-8")
    backend = MotrixBackend(SceneCfg(str(model_path)), num_envs=4, sim_dt=0.01)
    try:
        backend._host_step(np.full((4, 3), [0.5, 0.25, 0.8], dtype=np.float32))
        names = backend.get_sensor_names()
        assert {"a1", "a2", "a3", "jaf_hinge", "jaf_slide"} <= set(names)
        plan = backend.bind_host_bridge_io(
            TensorIOSpec(state_fields=("qpos", "qvel"), sensor_names=names)
        )

        with _count_native_sensor_reads(backend) as counting:
            packet = plan.read_state_sensors()
        assert (counting.batched_reads, counting.single_reads) == (1, 0)
        for name in names:
            np.testing.assert_allclose(
                packet[name].numpy(), backend._host_read_sensor(name), rtol=1e-6, atol=1e-6
            )

        rows = torch.tensor((2, 0))
        plan.apply_reset(rows, packet["qpos"][rows].clone(), packet["qvel"][rows].clone())
        with _count_native_sensor_reads(backend) as counting:
            selected = plan.read_selected_state_sensors()
        assert (counting.batched_reads, counting.single_reads) == (1, 0)
        for name in names:
            np.testing.assert_allclose(
                selected[name].numpy()[[2, 0]],
                backend._host_read_sensor(name)[[2, 0]],
                rtol=1e-6,
                atol=1e-6,
            )
    finally:
        backend.close()
