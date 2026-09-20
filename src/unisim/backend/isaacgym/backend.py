"""IsaacGym specialization of the shared MJCF subprocess host adapter.

IsaacGym Preview 4 runs in an external Python 3.8 process. The shared owner
layer supplies pipe/shm lifecycle, MJCF metadata, selected reset, NumPy state
views, and native-render command plumbing; this module only selects the
IsaacGym runtime and worker entrypoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from unisim.backend.subprocess_ipc.backend import (
    MjcfSubprocessBackend,
    SubprocessModelInfo,
    SubprocessWorkerError,
    _normalize_camera_kwargs,
)
from unisim.dr.types import DomainRandomizationCapabilities, FixedVariantLayout
from unisim.inspection import ConfigurationField, ConfigurationProvenance

from .dependencies import build_worker_env, resolve_isaacgym_runtime

_WORKER_PATH = Path(__file__).resolve().parent / "worker.py"


class IsaacGymWorkerError(SubprocessWorkerError):
    """Raised when the external IsaacGym worker fails."""


@dataclass(frozen=True)
class IsaacGymModelInfo(SubprocessModelInfo):
    """Opaque metadata returned by the IsaacGym worker handshake."""


class IsaacGymBackend(MjcfSubprocessBackend):
    """NumPy-facing ``SimBackend`` client for the IsaacGym worker."""

    _BACKEND_TYPE = "isaacgym"
    _BACKEND_LABEL = "isaacgym"
    _WORKER_ERROR_CLS = IsaacGymWorkerError
    _MODEL_INFO_CLS = IsaacGymModelInfo

    def __init__(
        self,
        scene: Any,
        num_envs: int,
        sim_dt: float,
        *,
        env_spacing: float = 4.0,
        **kwargs: Any,
    ) -> None:
        if isinstance(env_spacing, bool) or not isinstance(env_spacing, (int, float)):
            raise ValueError(f"env_spacing must be positive and finite, got {env_spacing!r}")
        if not math.isfinite(float(env_spacing)) or env_spacing <= 0:
            raise ValueError(f"env_spacing must be positive and finite, got {env_spacing!r}")
        self._env_spacing = float(env_spacing)
        super().__init__(scene, num_envs, sim_dt, **kwargs)

    def _worker_init_payload(self) -> dict[str, Any]:
        return {"env_spacing": self._env_spacing}

    def _bind_scene_metadata(self, meta: dict[str, Any]) -> None:
        super()._bind_scene_metadata(meta)
        reported_spacing = meta.get("env_spacing")
        if (
            isinstance(reported_spacing, bool)
            or not isinstance(reported_spacing, (int, float))
            or not math.isfinite(float(reported_spacing))
            or not math.isclose(
                float(reported_spacing), self._env_spacing, rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise self._worker_error(
                f"worker env spacing differs from the requested {self._env_spacing}: "
                f"{reported_spacing!r}"
            )
        origins = np.asarray(meta.get("env_origins"), dtype=float)
        columns = max(1, math.ceil(math.sqrt(self._num_envs)))
        indexes = np.arange(self._num_envs)
        expected = (
            np.column_stack(
                (
                    indexes % columns,
                    indexes // columns,
                    np.zeros(self._num_envs, dtype=float),
                )
            )
            * self._env_spacing
        )
        if (
            origins.shape != (self._num_envs, 3)
            or not np.isfinite(origins).all()
            or not np.allclose(origins, expected, rtol=0.0, atol=1e-6)
        ):
            raise self._worker_error(
                f"worker env origins do not match a {self._env_spacing:g} m native grid: "
                f"{origins.tolist()!r}"
            )

    def _capture_entity_report(self, meta: dict[str, Any]) -> None:
        super()._capture_entity_report(meta)
        report = self.get_import_report()
        spacing = ConfigurationField(
            "env_spacing",
            self._env_spacing,
            float(meta["env_spacing"]),
            "exact",
            (
                ConfigurationProvenance("adapter_setting", "Native worker scene profile"),
                ConfigurationProvenance("engine_readback", "Worker-reported native env origins"),
            ),
            unit="m",
            reason="Adjacent environment origins are audited against the requested grid.",
        )
        self._import_report = replace(report, fields=(*report.fields, spacing))

    def _supports_fixed_variant_plans(self) -> bool:
        return True

    def _worker_entrypoint(self) -> Path:
        return _WORKER_PATH

    def _resolve_worker_runtime(self) -> Any:
        return resolve_isaacgym_runtime()

    def _build_worker_environment(self, runtime: Any) -> dict[str, str]:
        return build_worker_env(runtime)

    def _runtime_payload(self, runtime: Any) -> dict[str, str]:
        return {"isaacgym_python": str(runtime.isaacgym_python)}

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Advertise actor/entity fixed variants without reset-time model DR."""
        entity_plan = None if self._entity_scene is None else self._entity_scene.owner.variant_plan
        if self._fixed_variant_plan is None and entity_plan is None:
            return DomainRandomizationCapabilities()
        return DomainRandomizationCapabilities(
            supports_fixed_variants=True,
            supported_fixed_variant_layouts=frozenset(
                {
                    FixedVariantLayout.SAME_LAYOUT,
                    FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
                }
            ),
            supports_per_env_playback=True,
        )

    def get_playback_model(self, env_index: int | None = None) -> Any:
        """Return the assigned variant source for one environment."""
        plan = self._fixed_variant_plan
        if plan is None:
            return super().get_playback_model(env_index)
        if env_index is None:
            raise ValueError("fixed-variant playback requires an explicit env_index")
        if isinstance(env_index, bool) or not isinstance(env_index, int):
            raise TypeError("env_index must be an integer or None")
        if env_index < 0 or env_index >= self._num_envs:
            raise IndexError(f"env_index must be in [0, {self._num_envs - 1}]")
        variant_index = int(plan.assignment[env_index])
        return plan.variants[variant_index].model_file


__all__ = [
    "IsaacGymBackend",
    "IsaacGymModelInfo",
    "IsaacGymWorkerError",
    "_normalize_camera_kwargs",
]
