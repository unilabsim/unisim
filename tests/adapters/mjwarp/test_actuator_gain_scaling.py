"""Real-CUDA MJWarp paired actuator gain-scaling coverage (#502)."""

# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")
warp = pytest.importorskip("warp")

from unisim import MjwarpBackend
from unisim.dr.types import (
    FixedVariantPlan,
    ModelSourceDescriptor,
    TensorResetRandomizationPayload,  # noqa: F401
)
from unisim.scene import SceneCfg

_NUM_ENVS = 3
_DT = 0.002

_MODEL = """<mujoco model="unisim-mjwarp-gain-scaling">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="base">
      <freejoint name="root"/>
      <geom name="base_geom" type="sphere" size="0.08" mass="1.0"/>
      <body name="link" pos="0 0 0.2">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="link_geom" type="sphere" size="0.05" mass="0.2"/>
      </body>
      <body name="other" pos="0 0.2 0">
        <joint name="other_hinge" type="hinge" axis="0 1 0"/>
        <geom name="other_geom" type="sphere" size="0.05" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="other_motor" joint="other_hinge"/>
    <position name="hinge_position" joint="hinge" kp="24"/>
    <velocity name="hinge_velocity" joint="hinge" kv="3"/>
  </actuator>
</mujoco>"""

_UNSUPPORTED_TRANSMISSION_MODEL = _MODEL.replace(
    "  <actuator>\n",
    (
        '  <tendon><fixed name="hinge_tendon">'
        '<joint joint="hinge" coef="1"/></fixed></tendon>\n'
        "  <actuator>\n"
    ),
).replace('joint="other_hinge"', 'tendon="hinge_tendon"')


_MALFORMED_MODEL = _MODEL.replace(
    '<velocity name="hinge_velocity" joint="hinge" kv="3"/>',
    (
        '<general name="hinge_velocity" joint="hinge" gaintype="fixed" gainprm="5 0 0" '
        'biastype="affine" biasprm="0 0 -4"/>'
    ),
)


def _write(tmp_path: Path, name: str, xml: str) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(xml, encoding="utf-8")
    return str(path)


def _mjwarp(
    tmp_path: Path,
    xml: str = _MODEL,
    name: str = "model.xml",
    *,
    plan: FixedVariantPlan | None = None,
) -> MjwarpBackend:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp actuator gain scaling requires real CUDA")
    return MjwarpBackend(
        SceneCfg(model_file=_write(tmp_path, name, xml), fixed_variant_plan=plan),
        num_envs=_NUM_ENVS,
        sim_dt=_DT,
        nconmax=32,
        njmax=32,
    )


def _newton(tmp_path: Path) -> Any:
    from unisim.backend.newton.backend import NewtonBackend
    from unisim.backend.newton.dependencies import (
        NewtonDependencyError,
        load_newton_dependencies,
    )

    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    if not bool(deps.warp.get_device().is_cuda):
        pytest.skip("Newton gain-scaling parity requires a CUDA Warp device")
    backend = NewtonBackend(
        SceneCfg(model_file=_write(tmp_path, "newton.xml", _MODEL)),
        num_envs=_NUM_ENVS,
        sim_dt=_DT,
        device=str(deps.warp.get_device()),
        nconmax=32,
        njmax=32,
        capacity_check_steps=1,
    )
    backend.materialize()
    return backend


def _gain_tables(backend: MjwarpBackend) -> tuple[np.ndarray, np.ndarray]:
    backend._synchronize()
    return (
        np.asarray(backend._device_model.actuator_gainprm.numpy()).copy(),
        np.asarray(backend._device_model.actuator_biasprm.numpy()).copy(),
    )


def _assert_pair(
    backend: MjwarpBackend,
    baseline: tuple[np.ndarray, np.ndarray],
    *,
    scale: float,
    rows: tuple[int, ...],
    pair: tuple[int, int] = (1, 2),
    motor_row: int = 0,
) -> None:
    gain, bias = _gain_tables(backend)
    for row in rows:
        np.testing.assert_array_equal(gain[row, pair, :3], baseline[0][0, pair, :3] * scale)
        np.testing.assert_array_equal(bias[row, pair, :3], baseline[1][0, pair, :3] * scale)
        np.testing.assert_array_equal(gain[row, motor_row], baseline[0][0, motor_row])
        np.testing.assert_array_equal(bias[row, motor_row], baseline[1][0, motor_row])
    untouched = tuple(index for index in range(_NUM_ENVS) if index not in rows)
    if untouched:
        np.testing.assert_array_equal(gain[untouched, pair], baseline[0][0, (1, 2)])
        np.testing.assert_array_equal(bias[untouched, pair], baseline[1][0, (1, 2)])


def test_gain_scale_zero_one_and_absolute_idempotence(tmp_path: Path) -> None:
    backend = _mjwarp(tmp_path)
    try:
        assert backend.get_tensor_capabilities().actuator_gain_scaling
        assert backend._actuator_target_kinds == ("direct", "position", "velocity")
        baseline = _gain_tables(backend)

        backend.set_actuator_gain_scale((1, 2), 0.0, env_ids=(1,))
        _assert_pair(backend, baseline, scale=0.0, rows=(1,))
        backend.set_actuator_gain_scale((1, 2), 1.0, env_ids=(1,))
        _assert_pair(backend, baseline, scale=1.0, rows=(1,))

        backend.set_actuator_gain_scale((1, 2), 0.25, env_ids=(1, 2))
        _assert_pair(backend, baseline, scale=0.25, rows=(1, 2))
        backend.set_actuator_gain_scale((2, 1), 0.25, env_ids=(1, 2))
        backend.set_actuator_gain_scale((1, 2), 0.25, env_ids=(1, 2))
        _assert_pair(backend, baseline, scale=0.25, rows=(1, 2))
    finally:
        backend.close()


def test_zero_scale_disables_physical_drive_and_restore_enables_it(tmp_path: Path) -> None:
    backend = _mjwarp(tmp_path)
    try:
        views = backend.get_state_views()
        device = views["qpos"].device
        ctrl = torch.zeros((_NUM_ENVS, backend.num_actuators), dtype=torch.float32, device=device)
        ctrl[:, 1] = 0.5

        backend.set_state_tensor(
            torch.arange(_NUM_ENVS, dtype=torch.int64, device=device),
            views["qpos"].clone(),
            torch.zeros_like(views["qvel"]),
        )
        backend.set_actuator_gain_scale((1, 2), 0.0, env_ids=(1,))
        backend.step_tensor(ctrl, nsteps=4)
        disabled = backend.get_state_views()["qvel"][:, 6].clone()

        backend.set_state_tensor(
            torch.tensor([1], dtype=torch.int64, device=device),
            views["qpos"][1:2].clone(),
            torch.zeros((1, backend._nv), dtype=torch.float32, device=device),
        )
        backend.set_actuator_gain_scale((1, 2), 1.0, env_ids=(1,))
        backend.step_tensor(ctrl, nsteps=4)
        restored = backend.get_state_views()["qvel"][1, 6]
        assert disabled[1].abs().item() < 1e-7
        assert restored.abs().item() > 1e-4
    finally:
        backend.close()


def test_selected_reset_and_gain_randomization_use_correct_baselines(tmp_path: Path) -> None:
    backend = _mjwarp(tmp_path)
    try:
        views = backend.get_state_views()
        device = views["qpos"].device
        baseline = _gain_tables(backend)
        requested_kp = torch.full((2, backend._nu), 50.0, dtype=torch.float32, device=device)
        requested_kd = torch.full((2, backend._nu), 6.0, dtype=torch.float32, device=device)
        rows = torch.tensor([2, 0], dtype=torch.int64, device=device)
        payload = TensorResetRandomizationPayload(kp=requested_kp, kd=requested_kd)
        backend.set_state_tensor(
            rows,
            views["qpos"].index_select(0, rows).contiguous(),
            torch.zeros((2, backend._nv), dtype=torch.float32, device=device),
            payload,
        )

        randomized = _gain_tables(backend)
        np.testing.assert_allclose(randomized[0][[2, 0], :, 0], 50.0)
        np.testing.assert_allclose(randomized[1][[2, 0], :, 1], -50.0)
        np.testing.assert_allclose(randomized[1][[2, 0], :, 2], -6.0)
        np.testing.assert_array_equal(randomized[0][1], baseline[0][1])

        # Absolute scale must use defaults, not the just-written randomized row.
        backend.set_actuator_gain_scale((1, 2), 0.0, env_ids=(0, 2))
        gain, bias = _gain_tables(backend)
        np.testing.assert_array_equal(gain[[0, 2], (1, 2), :3], 0.0)
        np.testing.assert_array_equal(bias[[0, 2], (1, 2), :3], 0.0)
        np.testing.assert_array_equal(gain[1, (1, 2)], baseline[0][0, (1, 2)])
        np.testing.assert_array_equal(bias[1, (1, 2)], baseline[1][0, (1, 2)])

        # A subsequent reset without gain terms preserves that selection.
        backend.set_state_tensor(
            rows,
            views["qpos"].index_select(0, rows).contiguous(),
            torch.zeros((2, backend._nv), dtype=torch.float32, device=device),
        )
        gain, bias = _gain_tables(backend)
        np.testing.assert_array_equal(gain[[0, 2], (1, 2), :3], 0.0)
        np.testing.assert_array_equal(bias[[0, 2], (1, 2), :3], 0.0)
    finally:
        backend.close()


def test_fixed_variant_default_baseline(tmp_path: Path) -> None:
    sources: list[str] = []
    for index, mass in enumerate((0.4, 0.8)):
        xml = _MODEL.replace('mass="0.2"', f'mass="{mass}"', 1)
        sources.append(_write(tmp_path, f"variant-{index}.xml", xml))
    plan = FixedVariantPlan(
        assignment=np.array([1, 0, 1], dtype=np.int32),
        variants=tuple(ModelSourceDescriptor(item) for item in sources),
    )
    backend = _mjwarp(tmp_path, plan=plan)
    try:
        assert backend.get_tensor_capabilities().actuator_gain_scaling
        baseline = _gain_tables(backend)
        backend.set_actuator_gain_scale((1, 2), 0.5, env_ids=(0, 2))
        gain, bias = _gain_tables(backend)
        np.testing.assert_array_equal(gain[0, (1, 2), :3], baseline[0][0, (1, 2), :3] * 0.5)
        np.testing.assert_array_equal(bias[0, (1, 2), :3], baseline[1][0, (1, 2), :3] * 0.5)
        np.testing.assert_array_equal(gain[1, (1, 2)], baseline[0][0, (1, 2)])
        np.testing.assert_array_equal(bias[1, (1, 2)], baseline[1][0, (1, 2)])
    finally:
        backend.close()


def test_validation_and_layout_fail_closed(tmp_path: Path) -> None:
    backend = _mjwarp(tmp_path)
    malformed = _mjwarp(tmp_path / "malformed", _MALFORMED_MODEL)
    try:
        with pytest.raises(ValueError, match="one-dimensional integer"):
            backend.set_actuator_gain_scale(np.array([[1, 2]]), 1.0)
        with pytest.raises(ValueError, match="distinct and in range"):
            backend.set_actuator_gain_scale((1, 1), 1.0)
        with pytest.raises(ValueError, match="distinct and in range"):
            backend.set_actuator_gain_scale((3,), 1.0)
        for scale in (-1.0, float("nan"), float("inf"), True, "x"):
            with pytest.raises(ValueError, match="finite nonnegative scalar"):
                backend.set_actuator_gain_scale((1, 2), scale)
        with pytest.raises(ValueError, match="complete position/velocity pair"):
            backend.set_actuator_gain_scale((1,), 1.0)
        with pytest.raises(NotImplementedError, match="direct"):
            backend.set_actuator_gain_scale((0,), 1.0)
        with pytest.raises(NotImplementedError, match="mixed direct/target"):
            backend.set_actuator_gain_scale((0, 1, 2), 1.0)
        assert backend._actuator_target_kinds == ("direct", "position", "velocity")

        with pytest.raises(NotImplementedError, match="canonical fixed-gain affine"):
            malformed.set_actuator_gain_scale((2,), 1.0)
    finally:
        backend.close()
        malformed.close()


def test_authoritative_baseline_and_validation_match_newton(tmp_path: Path) -> None:
    mjwarp = _mjwarp(tmp_path)
    newton = _newton(tmp_path / "newton")
    try:
        assert mjwarp.get_tensor_capabilities().actuator_gain_scaling
        assert newton.get_tensor_capabilities().actuator_gain_scaling
        np.testing.assert_array_equal(mjwarp.get_actuator_gains(), newton.get_actuator_gains())

        for backend in (mjwarp, newton):
            with pytest.raises(ValueError, match="complete position/velocity pair"):
                backend.set_actuator_gain_scale((1,), 0.5, env_ids=(0,))
            with pytest.raises(ValueError, match="finite nonnegative scalar"):
                backend.set_actuator_gain_scale((1, 2), -1.0, env_ids=(0,))
            with pytest.raises(NotImplementedError):
                backend.set_actuator_gain_scale((0,), 0.5, env_ids=(0,))
    finally:
        mjwarp.close()
        newton.close()


def test_unsupported_transmission_fails_only_when_selected(tmp_path: Path) -> None:
    backend = _mjwarp(tmp_path, _UNSUPPORTED_TRANSMISSION_MODEL, "transmission.xml")
    try:
        assert backend._actuator_target_kinds[0] == "unsupported"
        backend.set_actuator_gain_scale((1, 2), 0.5, env_ids=(0,))
        with pytest.raises(NotImplementedError, match="canonical fixed-gain affine"):
            backend.set_actuator_gain_scale((0,), 0.5, env_ids=(0,))
    finally:
        backend.close()
