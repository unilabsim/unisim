"""Portable and whole-model interval operations with native runtime support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")
pytest.importorskip("mujoco")

import torch  # noqa: E402

from unisim import MotrixBackend  # noqa: E402
from unisim.dr.types import (  # noqa: E402
    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    INTERVAL_TERM_PUSH,
    FixedVariantLayout,
    FixedVariantPlan,
    IntervalRandomizationPlan,
    IntervalTermOp,
    ModelSourceDescriptor,
    TensorIntervalPayload,
)
from unisim.entities import EntityInitialState, SceneEntitySpec  # noqa: E402
from unisim.scene import SceneCfg  # noqa: E402


def _source(path: Path) -> ModelSourceDescriptor:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<mujoco><option timestep='.01' gravity='0 0 0'/>"
        "<worldbody><body name='base'><freejoint name='root'/>"
        "<inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>"
        "<geom type='sphere' size='.1'/></body></worldbody>"
        "<sensor><framepos name='pos' objtype='body' objname='base'/></sensor></mujoco>",
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _portable_backend(path: Path) -> MotrixBackend:
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "object",
                _source(path),
                root_mode="floating",
                initial_state=EntityInitialState((0.0, 0.0, 1.0)),
            ),
        )
    )
    return MotrixBackend(scene, num_envs=2, sim_dt=0.01, base_name="object/base")


def test_portable_push_applies_to_each_runtime_row(tmp_path: Path) -> None:
    backend = _portable_backend(tmp_path / "object.xml")
    try:
        assert INTERVAL_TERM_PUSH in backend.get_dr_capabilities().supported_interval_terms
        backend.push_robots(
            TensorIntervalPayload(
                push_force=torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
            )
        )
        backend._host_step(np.zeros((2, 0), dtype=np.float32))
        np.testing.assert_allclose(backend.data.dof_vel[:, 0], [0.1, 0.1], atol=1e-4)
    finally:
        backend.close()


def test_portable_velocity_deltas_use_public_body_and_native_rows(tmp_path: Path) -> None:
    backend = _portable_backend(tmp_path / "object.xml")
    try:
        capabilities = backend.get_dr_capabilities()
        assert INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA in capabilities.supported_interval_terms
        assert INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA in capabilities.supported_interval_terms
        body_id = int(backend.get_body_ids(("object/base",))[0])
        linear = np.zeros((2, 1, 3), dtype=np.float32)
        linear[1, 0, 0] = 1.5
        angular = np.zeros((2, 1, 3), dtype=np.float32)
        angular[1, 0, 2] = 0.7
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(
                ops=(
                    IntervalTermOp(
                        INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
                        body_ids=torch.tensor([body_id]),
                        payload=torch.from_numpy(linear),
                    ),
                    IntervalTermOp(
                        INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
                        body_ids=torch.tensor([body_id]),
                        payload=torch.from_numpy(angular),
                    ),
                )
            )
        )
        np.testing.assert_allclose(backend.data.dof_vel[0], np.zeros(6), atol=1e-7)
        np.testing.assert_allclose(
            backend.data.dof_vel[1], [1.5, 0.0, 0.0, 0.0, 0.0, 0.7], atol=1e-6
        )
    finally:
        backend.close()


def test_whole_model_variant_push_is_supported(tmp_path: Path) -> None:
    sources = tuple(_source(tmp_path / f"variant-{index}.xml") for index in range(2))
    scene = SceneCfg(
        model_file=sources[0].model_file,
        fixed_variant_plan=FixedVariantPlan(
            np.asarray([0, 1], dtype=np.int32),
            sources,
            layout=FixedVariantLayout.SAME_LAYOUT,
        ),
    )
    backend = MotrixBackend(scene, num_envs=2, sim_dt=0.01, base_name="base")
    try:
        assert INTERVAL_TERM_PUSH in backend.get_dr_capabilities().supported_interval_terms
        backend.push_robots(
            TensorIntervalPayload(
                push_force=torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
            )
        )
        backend._host_step(np.zeros((2, 0), dtype=np.float32))
        for runtime in backend._whole_model_runtimes:
            np.testing.assert_allclose(runtime.data.dof_vel[:, 0], [0.1], atol=1e-4)
    finally:
        backend.close()


def test_portable_refresh_sensors_preserves_public_state(tmp_path: Path) -> None:
    backend = _portable_backend(tmp_path / "object.xml")
    try:
        before_qpos = backend.data.dof_pos.copy()
        before_qvel = backend.data.dof_vel.copy()
        before_unselected = backend._host_read_sensor("object/pos")[0].copy()
        backend.refresh_sensors([1])
        np.testing.assert_array_equal(backend.data.dof_pos, before_qpos)
        np.testing.assert_array_equal(backend.data.dof_vel, before_qvel)
        np.testing.assert_array_equal(backend._host_read_sensor("object/pos")[0], before_unselected)
        np.testing.assert_allclose(
            backend._host_read_sensor("object/pos")[1], [0.0, 0.0, 1.0], atol=1e-6
        )
    finally:
        backend.close()
