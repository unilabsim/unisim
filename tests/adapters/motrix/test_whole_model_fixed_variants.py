"""Whole-MJCF fixed-variant HOST_BRIDGE acceptance for MotrixSim."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")
pytest.importorskip("mujoco")

from unisim import MotrixBackend
from unisim.backend.base import TensorIOSpec
from unisim.dr.types import (
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    FixedVariantLayout,
    FixedVariantPlan,
    ModelSourceDescriptor,
    TensorResetRandomizationPayload,
)
from unisim.scene import SceneCfg

_ASSIGNMENT = (1, 0, 1, 0, 1)
_SENSOR_NAMES = ("angle", "speed", "object_quat", "object_floor", "object_found", "touch")


def _model(
    *,
    radius: float,
    half_height: float,
    mass: float,
    com: tuple[float, float, float],
    friction: float,
) -> str:
    return f"""<mujoco model='whole-fixed-variant'>
  <option timestep='.002' gravity='0 0 -9.81'/>
  <worldbody>
    <body name='hand' pos='0 0 .5'>
      <joint name='joint0' type='hinge' axis='1 0 0'/>
      <joint name='joint1' type='hinge' axis='0 1 0'/>
      <inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>
      <geom name='hand_geom' type='sphere' size='.04' contype='0' conaffinity='0'/>
    </body>
    <body name='object' pos='.4 0 1'>
      <freejoint name='object_free'/>
      <inertial pos='{com[0]} {com[1]} {com[2]}' mass='{mass}' diaginertia='.1 .1 .1'/>
      <geom name='object_geom' type='cylinder' size='{radius} {half_height}'
            friction='{friction} .005 .0001'/>
      <site name='object_touch' pos='0 0 0'/>
    </body>
    <geom name='floor' type='plane' size='2 2 .1'/>
  </worldbody>
  <actuator>
    <position name='drive0' joint='joint0' kp='18' dampratio='1'/>
    <position name='drive1' joint='joint1' kp='22' dampratio='1'/>
  </actuator>
  <sensor>
    <jointpos name='angle' joint='joint0'/>
    <jointvel name='speed' joint='joint0'/>
    <framequat name='object_quat' objtype='body' objname='object'/>
    <contact name='object_floor' geom1='object_geom' geom2='floor' data='force' reduce='netforce'/>
    <contact name='object_found' geom1='object_geom' geom2='floor' data='found' num='1'/>
    <touch name='touch' site='object_touch'/>
  </sensor>
</mujoco>"""


def _sources(tmp_path: Path) -> tuple[ModelSourceDescriptor, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("small", 0.04, 0.06, 1.0, (0.0, 0.0, 0.0), 0.5),
        ("large", 0.08, 0.12, 1.4, (0.01, 0.02, 0.03), 0.8),
        ("unused", 0.12, 0.16, 1.8, (0.02, 0.03, 0.04), 0.6),
    )
    sources: list[ModelSourceDescriptor] = []
    for name, radius, half_height, mass, com, friction in specifications:
        path = tmp_path / f"{name}.xml"
        path.write_text(
            _model(
                radius=radius,
                half_height=half_height,
                mass=mass,
                com=com,
                friction=friction,
            )
        )
        sources.append(ModelSourceDescriptor(str(path)))
    return tuple(sources)


def _scene(tmp_path: Path, assignment: tuple[int, ...] = _ASSIGNMENT) -> SceneCfg:
    sources = _sources(tmp_path)
    return SceneCfg(
        model_file=sources[0].model_file,
        fixed_variant_plan=FixedVariantPlan(
            np.asarray(assignment, dtype=np.int32),
            sources,
            layout=FixedVariantLayout.SAME_LAYOUT,
        ),
    )


def _single_scene(tmp_path: Path, variant: int) -> SceneCfg:
    source = _sources(tmp_path)[variant]
    return SceneCfg(source.model_file)


def test_whole_model_fixed_variant_layout_and_selected_packed_reset(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    backend = MotrixBackend(_scene(tmp_path), len(_ASSIGNMENT), 0.002, base_name="hand")
    try:
        assert len(backend._whole_model_runtimes) == 2
        assert [runtime.rows.tolist() for runtime in backend._whole_model_runtimes] == [
            [1, 3],
            [0, 2, 4],
        ]
        capabilities = backend.get_dr_capabilities()
        assert capabilities.supports_fixed_variants
        assert capabilities.supported_fixed_variant_layouts == frozenset(
            {FixedVariantLayout.SAME_LAYOUT}
        )
        assert backend.get_tensor_capabilities().fixed_variants
        assert backend.get_tensor_capabilities().packed_reset
        np.testing.assert_array_equal(
            backend.get_joint_state_qpos_indices(("joint0", "joint1")), [7, 8]
        )
        np.testing.assert_array_equal(
            backend.get_joint_state_qvel_indices(("joint0", "joint1")), [6, 7]
        )
        object_geom = backend.get_geom_id("object_geom")
        np.testing.assert_allclose(
            backend._whole_model_geom_sizes[:, object_geom, 0],
            [0.08, 0.04, 0.08, 0.04, 0.08],
        )
        object_body = int(backend.get_body_ids(("object",))[0])
        np.testing.assert_allclose(
            backend.get_body_mass()[:, object_body], [1.4, 1.0, 1.4, 1.0, 1.4]
        )
        with pytest.raises(NotImplementedError, match="non-uniform public geometry sizes"):
            backend.get_geom_size("object_geom")

        plan = backend.bind_host_bridge_io(
            TensorIOSpec(
                state_fields=("qpos", "qvel", "ctrl"),
                sensor_names=_SENSOR_NAMES,
                body_names=("object",),
                body_fields=("pos_w", "quat_w"),
            )
        )
        before = plan.read_state_sensors()
        before_poses = plan.get_body_pose_views()
        assert before_poses.pos_w is not None
        before_state = {name: value.clone() for name, value in before.items()}
        before_pose = before_poses.pos_w.clone()
        qpos = before["qpos"].clone()
        qvel = before["qvel"].clone()
        submitted_rows = (4, 1, 0)
        for offset, row in enumerate(submitted_rows):
            qpos[row, 7:] = torch.tensor([0.2 + offset, -0.3 - offset])
            qvel[row, 6:] = torch.tensor([0.4 + offset, -0.5 - offset])
        plan.apply_reset(
            torch.tensor(submitted_rows, dtype=torch.int64),
            qpos[list(submitted_rows)].contiguous(),
            qvel[list(submitted_rows)].contiguous(),
        )
        after = plan.read_selected_state_sensors()
        after_poses = plan.get_body_pose_views()
        assert after_poses.pos_w is not None
        selected_rows = [0, 1, 4]
        untouched = [2, 3]
        for name in ("qpos", "qvel", "ctrl"):
            np.testing.assert_allclose(
                after[name].numpy()[selected_rows],
                before[name].numpy()[selected_rows]
                if name == "ctrl"
                else locals()[name].numpy()[selected_rows],
                atol=2e-6,
            )
            np.testing.assert_allclose(
                after[name].numpy()[untouched], before_state[name].numpy()[untouched], atol=2e-6
            )
        for name in _SENSOR_NAMES:
            np.testing.assert_allclose(
                after[name].numpy()[untouched], before_state[name].numpy()[untouched], atol=2e-6
            )
        np.testing.assert_allclose(
            after_poses.pos_w.numpy()[untouched], before_pose.numpy()[untouched], atol=2e-6
        )
        np.testing.assert_allclose(
            after["angle"].numpy()[selected_rows], qpos.numpy()[selected_rows, 7:8], atol=2e-6
        )
        np.testing.assert_allclose(
            after["speed"].numpy()[selected_rows], qvel.numpy()[selected_rows, 6:7], atol=2e-6
        )
    finally:
        backend.close()


def test_whole_model_fixed_variant_rollout_matches_independent_backends(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    batch = MotrixBackend(_scene(tmp_path / "batch"), len(_ASSIGNMENT), 0.002, base_name="hand")
    references = [
        MotrixBackend(
            _single_scene(tmp_path / f"single-{row}", variant), 1, 0.002, base_name="hand"
        )
        for row, variant in enumerate(_ASSIGNMENT)
    ]
    try:
        initial = batch._host_get_state()
        qpos = initial["qpos"].copy()
        qvel = initial["qvel"].copy()
        qpos[:, 7:] = np.asarray([[0.3, -0.2], [0.1, 0.4], [-0.2, 0.5], [0.4, 0.2], [-0.3, -0.1]])
        qvel[:, 6:] = np.asarray([[0.2, -0.1], [0.3, 0.2], [-0.1, 0.4], [0.5, 0.1], [-0.2, -0.3]])
        rows = np.arange(len(_ASSIGNMENT), dtype=np.intp)
        batch._host_set_state(rows, qpos, qvel)
        for row, reference in enumerate(references):
            reference._host_set_state(np.asarray([0]), qpos[[row]], qvel[[row]])
        ctrl = np.asarray(
            [[0.2, -0.1], [0.4, 0.3], [-0.3, 0.2], [0.1, -0.4], [0.3, 0.5]], dtype=np.float32
        )
        plan = batch.bind_host_bridge_io(
            TensorIOSpec(state_fields=("qpos", "qvel", "ctrl"), sensor_names=_SENSOR_NAMES)
        )
        for _ in range(12):
            plan.write_control(torch.from_numpy(ctrl.copy()))
            plan.step()
            for row, reference in enumerate(references):
                reference._host_step(ctrl[[row]])
        packed = plan.read_state_sensors()
        batch_state = batch._host_get_state()
        for name in ("qpos", "qvel"):
            expected = np.stack([reference._host_get_state()[name][0] for reference in references])
            np.testing.assert_allclose(batch_state[name], expected, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(
                packed[name].numpy(), expected, rtol=2e-5, atol=2e-5
            )
        for sensor in _SENSOR_NAMES:
            expected = np.stack(
                [reference._host_read_sensor(sensor)[0] for reference in references]
            )
            np.testing.assert_allclose(
                batch._host_read_sensor(sensor), expected, rtol=2e-5, atol=2e-5
            )
            np.testing.assert_allclose(
                packed[sensor].numpy(), expected, rtol=2e-5, atol=2e-5
            )
        batch_snapshot = batch.get_physics_state().copy()
        batch._host_set_state(rows, qpos, qvel)
        batch.set_physics_state(batch_snapshot)
        np.testing.assert_allclose(batch.get_physics_state(), batch_snapshot, atol=2e-6)
        assert batch.get_playback_model(0).endswith("large.xml")
        assert batch.get_playback_model(1).endswith("small.xml")
    finally:
        batch.close()
        for reference in references:
            reference.close()


def test_whole_model_fixed_variant_reset_dr_composes_by_assignment(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    backend = MotrixBackend(_scene(tmp_path), len(_ASSIGNMENT), 0.002, base_name="hand")
    try:
        state = backend.get_state_views(device="cpu")
        rows = torch.tensor([3, 0], dtype=torch.int64)
        body_mass = torch.as_tensor(
            np.array(backend.get_reset_term_default(RESET_TERM_BODY_MASS)), dtype=torch.float32
        )
        body_ipos = torch.as_tensor(
            np.array(backend.get_reset_term_default(RESET_TERM_BODY_IPOS)), dtype=torch.float32
        )
        friction = torch.as_tensor(
            np.array(backend.get_reset_term_default(RESET_TERM_GEOM_FRICTION)), dtype=torch.float32
        )
        gravity = torch.as_tensor(
            np.array(backend.get_reset_term_default(RESET_TERM_GRAVITY)), dtype=torch.float32
        )
        kp = torch.as_tensor(
            np.array(backend.get_reset_term_default(RESET_TERM_KP)), dtype=torch.float32
        )
        kd = torch.as_tensor(
            np.array(backend.get_reset_term_default(RESET_TERM_KD)), dtype=torch.float32
        )
        object_body = int(backend.get_body_ids(("object",))[0])
        object_geom = backend.get_geom_id("object_geom")
        body_mass[[3, 0], object_body] = torch.tensor([2.2, 2.8])
        body_ipos[[3, 0], object_body] = torch.tensor([[0.03, 0.02, 0.01], [0.04, 0.03, 0.02]])
        friction[[3, 0], object_geom] = torch.tensor([[0.9, 0.02, 0.0], [0.7, 0.01, 0.0]])
        gravity[[3, 0]] = torch.tensor([[0.0, 0.0, -7.0], [0.0, 0.0, -8.0]])
        kp[[3, 0]] = torch.tensor([[30.0, 31.0], [32.0, 33.0]])
        kd[[3, 0]] = torch.tensor([[3.0, 3.1], [3.2, 3.3]])
        payload = TensorResetRandomizationPayload(
            body_mass=body_mass[rows],
            body_ipos=body_ipos[rows],
            base_mass_delta=torch.tensor([0.2, 0.3]),
            base_com_offset=torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]),
            geom_friction=friction[rows],
            gravity=gravity[rows],
            kp=kp[rows],
            kd=kd[rows],
        )
        backend.set_state_tensor(
            rows, state["qpos"][rows].contiguous(), state["qvel"][rows].contiguous(), payload
        )
        hand_body = int(backend.get_body_ids(("hand",))[0])
        body_mass[[3, 0], hand_body] += torch.tensor([0.2, 0.3])
        body_ipos[[3, 0], hand_body] += torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
        for term, expected in (
            (RESET_TERM_BODY_MASS, body_mass),
            (RESET_TERM_BODY_IPOS, body_ipos),
            (RESET_TERM_GEOM_FRICTION, friction),
            (RESET_TERM_GRAVITY, gravity),
            (RESET_TERM_KP, kp),
            (RESET_TERM_KD, kd),
        ):
            actual = backend.get_reset_term_default(term)
            np.testing.assert_allclose(
                actual[rows.numpy()], expected[rows].numpy(), rtol=2e-6, atol=2e-6
            )
        assert np.asarray(backend.get_reset_term_default(RESET_TERM_BASE_MASS)).shape == (
            len(_ASSIGNMENT),
        )
        assert np.asarray(backend.get_reset_term_default(RESET_TERM_BASE_COM)).shape == (
            len(_ASSIGNMENT),
            3,
        )
    finally:
        backend.close()


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_whole_model_fixed_variant_tensor_collector(tmp_path: Path, device_name: str) -> None:
    torch = pytest.importorskip("torch")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    backend = MotrixBackend(_scene(tmp_path), len(_ASSIGNMENT), 0.002, base_name="hand")
    try:
        device = torch.device(device_name)
        plan = backend.bind_host_bridge_io(
            TensorIOSpec(
                state_fields=("qpos", "qvel", "ctrl"), sensor_names=_SENSOR_NAMES, device=device
            )
        )
        plan.write_control(
            torch.full((len(_ASSIGNMENT), 2), 0.1, dtype=torch.float32, device=device)
        )
        plan.step(2)
        views = plan.read_state_sensors()
        expected = batch_state = backend._host_get_state()
        for name in ("qpos", "qvel"):
            np.testing.assert_allclose(views[name].cpu().numpy(), batch_state[name], atol=2e-6)
        mask = torch.tensor([True, False, True, False, False], device=device)
        qpos = views["qpos"].clone()
        qvel = views["qvel"].clone()
        qpos[mask, 7] = 0.6
        backend.set_state_tensor_mask(mask, qpos.contiguous(), qvel.contiguous())
        selected = plan.read_selected_state_sensors()
        np.testing.assert_allclose(
            selected["qpos"].cpu().numpy()[mask.cpu()], qpos.cpu().numpy()[mask.cpu()], atol=2e-6
        )
        np.testing.assert_allclose(
            selected["qpos"].cpu().numpy()[~mask.cpu()],
            expected["qpos"][~mask.cpu().numpy()],
            atol=2e-6,
        )
    finally:
        backend.close()


def test_whole_model_fixed_variant_layout_mismatch_fails_closed(tmp_path: Path) -> None:
    sources = list(_sources(tmp_path))
    mismatched = tmp_path / "mismatched.xml"
    mismatched.write_text(
        sources[1].model_file
        and Path(sources[1].model_file).read_text().replace("joint1", "other_joint")
    )
    sources[1] = ModelSourceDescriptor(str(mismatched))
    scene = SceneCfg(
        model_file=sources[0].model_file,
        fixed_variant_plan=FixedVariantPlan(np.asarray([0, 1], dtype=np.int32), tuple(sources)),
    )
    with pytest.raises(ValueError, match="mismatched.xml.*joints layout"):
        MotrixBackend(scene, 2, 0.002, base_name="hand")


def test_whole_model_fixed_variant_uniform_layout_fails_closed(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    scene = SceneCfg(
        model_file=sources[0].model_file,
        fixed_variant_plan=FixedVariantPlan(
            np.asarray([0, 1], dtype=np.int32),
            sources,
            layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
        ),
    )
    with pytest.raises(NotImplementedError, match="only same_layout"):
        MotrixBackend(scene, 2, 0.002, base_name="hand")


def test_whole_model_fixed_variant_sensor_mismatch_fails_closed(tmp_path: Path) -> None:
    sources = list(_sources(tmp_path))
    mismatched = tmp_path / "sensor-mismatched.xml"
    mismatched.write_text(
        Path(sources[1].model_file)
        .read_text()
        .replace(
            "<jointpos name='angle' joint='joint0'/>",
            "<jointpos name='angle' joint='joint1'/>",
        )
    )
    sources[1] = ModelSourceDescriptor(str(mismatched))
    scene = SceneCfg(
        model_file=sources[0].model_file,
        fixed_variant_plan=FixedVariantPlan(np.asarray([0, 1], dtype=np.int32), tuple(sources)),
    )
    with pytest.raises(ValueError, match="sensor-mismatched.xml.*sensor identity"):
        MotrixBackend(scene, 2, 0.002, base_name="hand")


def test_whole_model_fixed_variant_packed_transaction_uses_variant_baselines(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    backend = MotrixBackend(_scene(tmp_path), len(_ASSIGNMENT), 0.002, base_name="hand")
    try:
        state = backend.get_state_views(device="cpu")
        transaction = backend.bind_packed_reset(device="cpu")
        transaction.set_state_default(state["qpos"][0], state["qvel"][0])
        baseline = transaction.baseline(RESET_TERM_BODY_MASS).clone()
        object_body = int(backend.get_body_ids(("object",))[0])
        np.testing.assert_allclose(
            baseline[:, object_body].numpy(), [1.4, 1.0, 1.4, 1.0, 1.4], rtol=2e-6
        )
        rows = torch.tensor([3, 0], dtype=torch.int64)
        requested = baseline.clone()
        requested[rows, object_body] = torch.tensor([2.4, 2.9])
        transaction.write_model_field(
            RESET_TERM_BODY_MASS,
            rows,
            requested[rows][:, [object_body]],
            columns=torch.tensor([object_body], dtype=torch.int64),
        )
        transaction.begin_state(rows)
        transaction.commit_state(rows, state["qpos"][rows], state["qvel"][rows])
        actual = np.asarray(backend.get_reset_term_default(RESET_TERM_BODY_MASS))
        np.testing.assert_allclose(actual[rows.numpy(), object_body], [2.4, 2.9], rtol=2e-6)
        np.testing.assert_allclose(
            actual[[1, 2, 4], object_body], baseline[[1, 2, 4], object_body].numpy(), rtol=2e-6
        )
        torch.testing.assert_close(transaction.baseline(RESET_TERM_BODY_MASS), requested)
    finally:
        backend.close()
