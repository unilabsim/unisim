"""Drake physics-configuration pass-through and effective-settings tests.

The typed physics API lands after the released ``drake-uni==0.1.0``, so these
tests fake the ``drake_uni.runtime`` symbols instead of importing the real
runtime. Native acceptance remains with the import-skipped portable-entity
tests.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from unisim import create_backend
from unisim.backend.drake import backend as drake_backend
from unisim.scene import SceneCfg


@dataclass(frozen=True)
class _FakeGeometryMaterial:
    hydroelastic_type: str
    hydroelastic_modulus: float | None = None
    resolution_hint: float | None = None
    hunt_crossley_dissipation: float | None = None


@dataclass(frozen=True)
class _FakePhysicsConfig:
    contact_model: str = "point"
    discrete_contact_approximation: str = "sap"
    penetration_allowance: float = 1.0e-4
    geometry_materials: Mapping[str, _FakeGeometryMaterial] = field(default_factory=dict)
    frictionloss_regularization_speed: float | None = None
    preserve_gravity_compensation: bool = False


@dataclass(frozen=True)
class _FakeBatchConfig:
    model_file: str
    num_envs: int
    sim_dt: float
    nthread: int
    physics: Any | None = None


@dataclass(frozen=True)
class _FakeEffectivePhysics:
    contact_model: str
    discrete_contact_approximation: str
    penetration_allowance: float
    geometry_materials: tuple[tuple[str, _FakeGeometryMaterial], ...]
    field_status: tuple[tuple[str, str], ...]
    ignored_mjcf_fields: tuple[str, ...]
    drake_version: str


class _FakeModelInfo:
    nq = 7
    nv = 6
    nu = 0
    num_bodies = 2
    nsensordata = 3
    home_qpos = np.zeros(7)
    home_qvel = np.zeros(6)
    ctrl_limits = np.zeros((0, 2))
    joint_ranges = np.zeros((0, 2))
    actuator_stiffness = np.zeros(0)
    actuator_damping = np.zeros(0)
    actuator_qpos_adr = np.zeros(0, dtype=np.intp)
    actuator_qvel_adr = np.zeros(0, dtype=np.intp)
    actuator_names: tuple[str, ...] = ()
    sensor_names = ("base_imu",)
    sensor_adr = np.array([0], dtype=np.intp)
    sensor_dim = np.array([3], dtype=np.intp)
    site_names: tuple[str, ...] = ()
    joint_names: tuple[str, ...] = ()
    joint_qpos_adr: tuple[int, ...] = ()
    joint_qvel_adr: tuple[int, ...] = ()
    joint_qpos_dim: tuple[int, ...] = ()
    joint_qvel_dim: tuple[int, ...] = ()


class _FakeRuntime:
    """Minimal DrakeUni runtime double; records the received batch config."""

    nthread = 1

    def __init__(self, config: _FakeBatchConfig) -> None:
        self.config = config
        self.effective_physics_report = _FakeEffectivePhysics(
            contact_model=config.physics.contact_model if config.physics else "point",
            discrete_contact_approximation=(
                config.physics.discrete_contact_approximation if config.physics else "sap"
            ),
            penetration_allowance=(
                config.physics.penetration_allowance if config.physics else 1.0e-4
            ),
            geometry_materials=(
                tuple(sorted(config.physics.geometry_materials.items()))
                if config.physics
                else ()
            ),
            field_status=(
                (
                    ("contact_model", "applied"),
                    ("discrete_contact_approximation", "applied"),
                    ("frictionloss_regularization_speed", "approximated"),
                    ("preserve_gravity_compensation", "ignored"),
                )
                if config.physics
                else ()
            ),
            ignored_mjcf_fields=("gravcomp", "frictionloss") if config.physics else (),
            drake_version="1.51.1",
        )
        self.closed = False

    def model_info(self) -> _FakeModelInfo:
        return _FakeModelInfo()

    def physics_state(self) -> np.ndarray:
        return np.zeros((self.config.num_envs, 1 + _FakeModelInfo.nq + _FakeModelInfo.nv))

    def sensor_data(self) -> np.ndarray:
        return np.zeros((self.config.num_envs, _FakeModelInfo.nsensordata))

    def body_ids(self, names: tuple[str, ...]) -> np.ndarray:
        return np.zeros(len(names), dtype=np.int32)

    def effective_physics(self) -> _FakeEffectivePhysics:
        return self.effective_physics_report

    def diagnostics(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        self.closed = True


def _install_fake_drake_uni(
    monkeypatch: pytest.MonkeyPatch, *, with_physics: bool = True
) -> list[_FakeBatchConfig]:
    """Register a fake ``drake_uni.runtime`` and reset the adapter globals."""
    configs: list[_FakeBatchConfig] = []

    def create_runtime(config: _FakeBatchConfig) -> _FakeRuntime:
        configs.append(config)
        return _FakeRuntime(config)

    runtime_module = types.ModuleType("drake_uni.runtime")
    runtime_module.DrakeBatchConfig = _FakeBatchConfig  # type: ignore[attr-defined]
    runtime_module.create_runtime = create_runtime  # type: ignore[attr-defined]
    runtime_module.batch_diagnostics = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
        batch_available=True, batch_import_error=None
    )
    if with_physics:
        runtime_module.DrakePhysicsConfig = _FakePhysicsConfig  # type: ignore[attr-defined]
    package = types.ModuleType("drake_uni")
    package.runtime = runtime_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "drake_uni", package)
    monkeypatch.setitem(sys.modules, "drake_uni.runtime", runtime_module)
    # ``_load_drake_uni_symbols`` writes module globals; register them with
    # monkeypatch so the fakes cannot leak into other tests.
    monkeypatch.setattr(drake_backend, "_DRAKE_UNI_SYMBOLS_LOADED", False)
    for name in (
        "DRAKE_AVAILABLE",
        "DRAKE_BATCH_AVAILABLE",
        "DRAKE_BATCH_IMPORT_ERROR",
        "DrakeBatchConfig",
        "create_drake_runtime",
    ):
        monkeypatch.setattr(drake_backend, name, getattr(drake_backend, name))
    return configs


def _scene() -> SceneCfg:
    return SceneCfg(model_file="scene.xml")


def _physics() -> _FakePhysicsConfig:
    return _FakePhysicsConfig(
        contact_model="hydroelastic",
        discrete_contact_approximation="lagged",
        penetration_allowance=2.0e-4,
        geometry_materials={
            "gripper": _FakeGeometryMaterial(
                "compliant", hydroelastic_modulus=1.0e5, resolution_hint=0.01
            )
        },
        frictionloss_regularization_speed=1.0e-6,
        preserve_gravity_compensation=True,
    )


def test_physics_config_reaches_every_batch_config(monkeypatch: pytest.MonkeyPatch) -> None:
    configs = _install_fake_drake_uni(monkeypatch)
    physics = _physics()
    backend = create_backend(
        "drake", _scene(), num_envs=2, sim_dt=1 / 240, drake_physics=physics
    )
    try:
        assert configs and all(config.physics is physics for config in configs)
        assert all(config.sim_dt == pytest.approx(1 / 240) for config in configs)
        assert backend._physics is physics
    finally:
        backend.close()


def test_omitted_physics_keeps_default_batch_config(monkeypatch: pytest.MonkeyPatch) -> None:
    configs = _install_fake_drake_uni(monkeypatch)
    backend = create_backend("drake", _scene(), num_envs=1, sim_dt=0.01)
    try:
        assert configs and all(config.physics is None for config in configs)
        # The drake-uni 0.1.0-compatible four-field call must not even mention
        # ``physics``; the fake dataclass default makes the omission observable
        # indirectly.
        assert backend._physics is None
    finally:
        backend.close()


def test_physics_config_requires_current_drake_uni(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_drake_uni(monkeypatch, with_physics=False)
    with pytest.raises(ImportError, match="drake-uni >= 0.2.0"):
        create_backend("drake", _scene(), num_envs=1, sim_dt=0.01, drake_physics=object())


def test_physics_config_rejects_foreign_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_drake_uni(monkeypatch)
    with pytest.raises(TypeError, match="DrakePhysicsConfig"):
        create_backend("drake", _scene(), num_envs=1, sim_dt=0.01, drake_physics=object())


def test_factory_rejects_drake_physics_for_other_backends() -> None:
    with pytest.raises(TypeError, match="only supported by the drake backend"):
        create_backend("mujoco", _scene(), num_envs=1, sim_dt=0.01, drake_physics=object())
    with pytest.raises(TypeError, match="only supported by the drake backend"):
        create_backend("fake", num_envs=1, drake_physics=object())


def test_effective_physics_reports_each_runtime_group(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_drake_uni(monkeypatch)
    physics = _physics()
    backend = create_backend("drake", _scene(), num_envs=3, sim_dt=0.01, drake_physics=physics)
    try:
        reports = backend.get_effective_physics()
        assert len(reports) == len(backend._runtime_groups) == 1
        report = reports[0]
        assert report.contact_model == "hydroelastic"
        assert report.discrete_contact_approximation == "lagged"
        assert dict(report.field_status)["preserve_gravity_compensation"] == "ignored"
        assert "frictionloss" in report.ignored_mjcf_fields
    finally:
        backend.close()


def test_effective_physics_fails_closed_on_old_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_drake_uni(monkeypatch)
    backend = create_backend("drake", _scene(), num_envs=1, sim_dt=0.01)
    try:
        monkeypatch.delattr(_FakeRuntime, "effective_physics")
        with pytest.raises(NotImplementedError, match="effective_physics"):
            backend.get_effective_physics()
    finally:
        backend.close()


def test_sensor_namespace_is_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_drake_uni(monkeypatch)
    backend = create_backend("drake", _scene(), num_envs=1, sim_dt=0.01)
    try:
        assert backend.get_sensor_names() == ("base_imu",)
        inventory = backend.get_sensor_inventory()
        assert [(item.name, item.width) for item in inventory] == [("base_imu", 3)]
    finally:
        backend.close()


def test_effective_physics_rejects_closed_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_drake_uni(monkeypatch)
    backend = create_backend("drake", _scene(), num_envs=1, sim_dt=0.01, drake_physics=_physics())
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.get_effective_physics()
