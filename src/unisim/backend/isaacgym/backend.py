"""IsaacGym specialization of the shared MJCF subprocess host adapter.

IsaacGym Preview 4 runs in an external Python 3.8 process. The shared owner
layer supplies pipe/shm lifecycle, MJCF metadata, selected reset, NumPy state
views, and native-render command plumbing; this module only selects the
IsaacGym runtime and worker entrypoint.
"""

from __future__ import annotations

import math
from collections.abc import Callable
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
from unisim.dr.interval import (
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_TORQUE,
    IntervalTermOp,
)
from unisim.dr.types import (
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_DOF_FRICTIONLOSS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    FixedVariantLayout,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
    _validate_reset_term,
)
from unisim.inspection import ConfigurationField, ConfigurationProvenance

from .dependencies import build_worker_env, resolve_isaacgym_runtime

_WORKER_PATH = Path(__file__).resolve().parent / "worker.py"

_MAPPED_RESET_TERMS = frozenset(
    {
        RESET_TERM_KP,
        RESET_TERM_KD,
        RESET_TERM_BODY_MASS,
        RESET_TERM_BODY_INERTIA,
        RESET_TERM_BODY_IPOS,
        RESET_TERM_DOF_ARMATURE,
        RESET_TERM_DOF_FRICTIONLOSS,
        RESET_TERM_GEOM_FRICTION,
    }
)
_MAPPED_INTERVAL_TERMS = frozenset({INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE})
_FIXED_VARIANT_LAYOUTS = frozenset(
    {
        FixedVariantLayout.SAME_LAYOUT,
        FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
    }
)


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
        self._staged_body_wrench = (
            None
            if self._entity_scene is None
            else np.zeros(
                (self._num_envs, self._entity_scene.layout.nbody, 6), dtype=np.float32
            )
        )
        self._body_wrench_pending = False

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
        """Advertise mapped-scene reset DR and interval wrenches.

        The legacy raw-MJCF path keeps the fail-closed empty declaration; mapped
        entity scenes declare the reset terms the worker writes and reads back
        natively.  Per-env ``gravity``/``dof_damping`` have no PhysX channel
        (sim-wide parameter, drive-only joint damping) and the multi-root scene
        has no single ``base`` target, so those terms stay unsupported.
        """
        entity_plan = None if self._entity_scene is None else self._entity_scene.owner.variant_plan
        has_plan = self._fixed_variant_plan is not None or entity_plan is not None
        variant_kwargs: dict[str, Any] = (
            {
                "supports_fixed_variants": True,
                "supported_fixed_variant_layouts": _FIXED_VARIANT_LAYOUTS,
                "supports_per_env_playback": True,
            }
            if has_plan
            else {}
        )
        if self._entity_scene is None:
            return DomainRandomizationCapabilities(**variant_kwargs)
        return DomainRandomizationCapabilities(
            supported_reset_terms=_MAPPED_RESET_TERMS,
            supports_interval_body_force=True,
            supports_interval_body_torque=True,
            supported_interval_terms=_MAPPED_INTERVAL_TERMS,
            **variant_kwargs,
        )

    @staticmethod
    def _coerce_mapped_reset_field(
        values: Any, name: str, shape: tuple[int, ...]
    ) -> np.ndarray:
        if not isinstance(values, np.ndarray) or not np.issubdtype(values.dtype, np.floating):
            raise TypeError(f"isaacgym {name} must be a floating NumPy array")
        array = np.asarray(values, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"isaacgym {name} must have shape {shape}, got {array.shape}")
        if (
            not np.isfinite(array).all()
            or np.any(np.abs(array.astype(np.float64)) > np.finfo(np.float32).max)
        ):
            raise ValueError(f"isaacgym {name} must contain finite float32 values")
        return array.copy()

    def _validated_mapped_reset_randomization(
        self, randomization: ResetRandomizationPayload | None, rows: np.ndarray
    ) -> ResetRandomizationPayload | None:
        if randomization is None or randomization.is_empty():
            return None
        if self._entity_scene is None:
            raise NotImplementedError(
                "isaacgym reset domain randomization requires a mapped entity scene"
            )
        unsupported = self.get_dr_capabilities().get_unsupported_reset_terms(
            randomization.requested_terms()
        )
        if unsupported:
            requested = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                f"isaacgym does not support reset domain randomization terms: {requested}."
            )
        layout = self._entity_scene.layout
        count = int(rows.size)
        coerce = self._coerce_mapped_reset_field

        def field(name: str, shape: tuple[int, ...]) -> np.ndarray | None:
            values = getattr(randomization, name)
            return None if values is None else coerce(values, name, shape)

        kp = field(RESET_TERM_KP, (count, layout.nu))
        kd = field(RESET_TERM_KD, (count, layout.nu))
        for name, gains in ((RESET_TERM_KP, kp), (RESET_TERM_KD, kd)):
            if gains is not None and np.any(gains < 0.0):
                raise ValueError(f"isaacgym {name} values must be nonnegative")
        body_mass = field(RESET_TERM_BODY_MASS, (count, layout.nbody))
        if body_mass is not None and np.any(body_mass <= 0.0):
            raise ValueError("isaacgym body_mass values must be positive")
        body_ipos = field(RESET_TERM_BODY_IPOS, (count, layout.nbody, 3))
        body_inertia = field(RESET_TERM_BODY_INERTIA, (count, layout.nbody, 3))
        if body_inertia is not None and np.any(body_inertia <= 0.0):
            raise ValueError("isaacgym body_inertia values must be positive")
        root_columns = sorted(
            {column for entity in layout.entities for column in entity.root_qvel_indices}
        )
        dof_armature = field(RESET_TERM_DOF_ARMATURE, (count, layout.nv))
        dof_frictionloss = field(RESET_TERM_DOF_FRICTIONLOSS, (count, layout.nv))
        for name, values in (
            (RESET_TERM_DOF_ARMATURE, dof_armature),
            (RESET_TERM_DOF_FRICTIONLOSS, dof_frictionloss),
        ):
            if values is None:
                continue
            if np.any(values < 0.0):
                raise ValueError(f"isaacgym {name} values must be nonnegative")
            if root_columns and np.any(values[:, root_columns] != 0.0):
                raise ValueError(
                    f"isaacgym {name} must be zero on floating-root columns; "
                    "PhysX root state is not a driven DOF"
                )
        geom_friction = field(RESET_TERM_GEOM_FRICTION, (count, layout.ngeom, 3))
        if geom_friction is not None:
            if np.any(geom_friction < 0.0):
                raise ValueError("isaacgym geom_friction values must be nonnegative")
            if (
                np.any(geom_friction[..., 0] != geom_friction[..., 1])
                or np.any(geom_friction[..., 2] != 0.0)
            ):
                raise ValueError(
                    "isaacgym geom_friction requires static == dynamic and a zero third column"
                )
        return ResetRandomizationPayload(
            kp=kp,
            kd=kd,
            body_mass=body_mass,
            body_ipos=body_ipos,
            body_inertia=body_inertia,
            dof_armature=dof_armature,
            dof_frictionloss=dof_frictionloss,
            geom_friction=geom_friction,
        )

    def _consume_entity_reset_randomization(self, response: Any) -> None:
        """Replace current native property records from the reset barrier."""
        if self._entity_scene is None:
            raise self._worker_error("isaacgym property mutation requires a mapped scene")
        layout = self._entity_scene.layout
        raw_records = response.get("native_entity_records") if isinstance(response, dict) else None
        if not isinstance(raw_records, list):
            raise self._worker_error(
                "isaacgym worker omitted native records after property mutation"
            )
        records = {record.get("name"): record for record in raw_records if isinstance(record, dict)}
        names = {entity.name for entity in layout.entities}
        if len(records) != len(raw_records) or set(records) != names:
            raise self._worker_error(
                "isaacgym worker property-mutation records do not match the frozen entity layout"
            )
        for entity in layout.entities:
            current = self._native_entity_records.get(entity.name)
            if current is None:
                raise self._worker_error(
                    "worker omitted native entity properties: " + entity.name
                )
            reported = records[entity.name]
            shapes = {
                "body_mass": (self._num_envs, len(entity.body_ids)),
                "body_ipos": (self._num_envs, len(entity.body_ids), 3),
                "body_inertia": (self._num_envs, len(entity.body_ids), 3, 3),
                "dof_stiffness": (self._num_envs, len(entity.joints)),
                "dof_damping": (self._num_envs, len(entity.joints)),
                "dof_armature": (self._num_envs, len(entity.joints)),
                "dof_friction": (self._num_envs, len(entity.joints)),
                "geom_friction": (self._num_envs, len(entity.geoms), 3),
            }
            for field, shape in shapes.items():
                try:
                    values = np.asarray(reported.get(field), dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise self._worker_error(
                        f"worker native {field} is malformed for entity {entity.name}: "
                        f"expected shape {shape}"
                    ) from exc
                if values.shape != shape or not np.isfinite(values).all():
                    raise self._worker_error(
                        f"worker native {field} is malformed for entity {entity.name}: "
                        f"got shape {values.shape}, expected {shape}"
                    )
                current[field] = reported[field]

    # ------------------------------------------------------------------ #
    # Mapped reset term defaults
    # ------------------------------------------------------------------ #

    def get_reset_term_default(self, term: str) -> np.ndarray:
        """Return per-environment variant-assigned mapped reset defaults."""
        if self._entity_scene is None:
            return super().get_reset_term_default(term)
        _validate_reset_term(term)
        if not self.get_dr_capabilities().supports_reset_term(term):
            raise NotImplementedError(f"IsaacGymBackend does not support reset term {term!r}")
        rows = np.arange(self._num_envs, dtype=np.intp)
        if term == RESET_TERM_BODY_MASS:
            value = self._mapped_default_body_rows("body_mass", rows)
        elif term == RESET_TERM_BODY_IPOS:
            value = self._mapped_default_body_rows("body_ipos", rows, width=3)
        elif term == RESET_TERM_BODY_INERTIA:
            value = self._mapped_default_body_rows("body_inertia", rows, width=3)
        elif term == RESET_TERM_GEOM_FRICTION:
            value = self._mapped_default_geom_friction(rows)
        elif term == RESET_TERM_KP:
            value = self._mapped_default_actuator_rows("dof_stiffness", rows)
        elif term == RESET_TERM_KD:
            value = self._mapped_default_actuator_rows("dof_damping", rows)
        elif term == RESET_TERM_DOF_ARMATURE:
            value = self._mapped_default_dof_rows("dof_armature", rows)
        elif term == RESET_TERM_DOF_FRICTIONLOSS:
            value = self._mapped_default_dof_rows("dof_friction", rows)
        else:  # pragma: no cover - guarded by supports_reset_term
            raise NotImplementedError(f"IsaacGymBackend does not support reset term {term!r}")
        result = np.array(value, dtype=np.float32, copy=True)
        result.setflags(write=False)
        return result

    def _canonical_body_table(self, field: str, width: int | None = None) -> np.ndarray:
        assert self._entity_scene is not None
        expected = (
            (self._entity_scene.layout.nbody, 3)
            if width == 3
            else (self._entity_scene.layout.nbody,)
        )
        try:
            canonical = np.asarray(
                getattr(self._entity_scene.owner.model, field), dtype=np.float32
            )
        except (TypeError, ValueError) as exc:
            raise self._worker_error(
                f"compiled canonical {field} is malformed: expected shape {expected}"
            ) from exc
        if canonical.shape != expected or not np.isfinite(canonical).all():
            raise self._worker_error(
                f"compiled canonical {field} is malformed: got shape {canonical.shape}, "
                f"expected {expected}"
            )
        return canonical.copy()

    def _mapped_default_body_rows(
        self, field: str, rows: np.ndarray, width: int | None = None
    ) -> np.ndarray:
        """Build per-row variant-assigned public body-property defaults."""
        assert self._entity_scene is not None
        scene = self._entity_scene
        layout = scene.layout
        shape = (rows.size, layout.nbody) if width is None else (rows.size, layout.nbody, width)
        values = np.broadcast_to(self._canonical_body_table(field, width), shape).copy()
        for entity, entry in zip(layout.entities, scene.payload["scene_entities"]):
            defaults = np.asarray(
                [
                    entry["variants"][int(entry["assignment"][int(env)])][field]
                    for env in rows
                ],
                dtype=np.float32,
            )
            expected = (
                (rows.size, len(entity.body_ids))
                if width is None
                else (rows.size, len(entity.body_ids), width)
            )
            if defaults.shape != expected or not np.isfinite(defaults).all():
                raise self._worker_error(
                    f"compiled variant {field} is malformed for entity {entity.name}"
                )
            if width is None:
                values[:, list(entity.body_ids)] = defaults
            else:
                values[:, list(entity.body_ids), :] = defaults
        return values

    def _mapped_default_geom_friction(self, rows: np.ndarray) -> np.ndarray:
        """Build per-row variant-assigned public geom-friction defaults."""
        assert self._entity_scene is not None
        scene = self._entity_scene
        layout = scene.layout
        values = np.zeros((rows.size, layout.ngeom, 3), dtype=np.float32)
        offset = 0
        for entity, entry in zip(layout.entities, scene.payload["scene_entities"]):
            count = len(entity.geoms)
            if count:
                defaults = np.asarray(
                    [
                        entry["variants"][int(entry["assignment"][int(env)])]["geom_friction"]
                        for env in rows
                    ],
                    dtype=np.float32,
                )
                values[:, offset : offset + count, 0] = defaults[:, :, 0]
                values[:, offset : offset + count, 1] = defaults[:, :, 0]
            offset += count
        return values

    def _mapped_default_actuator_rows(self, field: str, rows: np.ndarray) -> np.ndarray:
        """Build per-row variant-assigned public actuator-gain defaults.

        ``field`` is the per-joint variant record (``dof_stiffness`` for
        ``kp``, ``dof_damping`` for the drive part of ``kd``).
        """
        assert self._entity_scene is not None
        scene = self._entity_scene
        layout = scene.layout
        values = np.zeros((rows.size, layout.nu), dtype=np.float32)
        for entity, entry in zip(layout.entities, scene.payload["scene_entities"]):
            if not entity.actuator_indices:
                continue
            positions = [
                next(i for i, joint in enumerate(entity.joints) if joint.name == name)
                for name in entity.actuator_joint_names
            ]
            defaults = np.asarray(
                [
                    entry["variants"][int(entry["assignment"][int(env)])][field]
                    for env in rows
                ],
                dtype=np.float32,
            )
            values[:, list(entity.actuator_indices)] = defaults[:, positions]
        return values

    def _mapped_default_dof_rows(self, field: str, rows: np.ndarray) -> np.ndarray:
        """Build per-row variant-assigned public DOF-property defaults."""
        assert self._entity_scene is not None
        scene = self._entity_scene
        layout = scene.layout
        values = np.zeros((rows.size, layout.nv), dtype=np.float32)
        for entity, entry in zip(layout.entities, scene.payload["scene_entities"]):
            if not entity.joints:
                continue
            columns = [joint.qvel_indices[0] for joint in entity.joints]
            defaults = np.asarray(
                [
                    entry["variants"][int(entry["assignment"][int(env)])][field]
                    for env in rows
                ],
                dtype=np.float32,
            )
            values[:, columns] = defaults
        return values

    # ------------------------------------------------------------------ #
    # Interval body wrenches (mapped scenes)
    # ------------------------------------------------------------------ #

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.is_empty():
            return
        if self._entity_scene is not None:
            assert self._staged_body_wrench is not None
            self._staged_body_wrench.fill(0.0)
            self._body_wrench_pending = False
        super().apply_interval_randomization(plan)

    def _interval_term_handlers(self) -> dict[str, Callable[[IntervalTermOp], None]]:
        if self._entity_scene is None:
            return super()._interval_term_handlers()
        return {
            INTERVAL_TERM_BODY_FORCE: self._apply_interval_body_force,
            INTERVAL_TERM_BODY_TORQUE: self._apply_interval_body_torque,
        }

    def _apply_interval_body_force(self, op: IntervalTermOp) -> None:
        if op.body_ids is None:
            raise ValueError("interval body force requires body_ids")
        self.apply_body_force(op.body_ids, op.payload)

    def _apply_interval_body_torque(self, op: IntervalTermOp) -> None:
        if op.body_ids is None:
            raise ValueError("interval body torque requires body_ids")
        zero_force = np.zeros((self._num_envs, len(op.body_ids), 3), dtype=np.float32)
        self.apply_body_force(op.body_ids, zero_force, torque=op.payload)

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray | None = None,
    ) -> None:
        """Stage a world-frame wrench for the next mapped-scene step.

        The force acts at each target body's center of mass and the optional
        torque is about that center. Repeated submissions accumulate until the
        next successful step consumes them.
        """
        if self._entity_scene is None:
            super().apply_body_force(body_ids, force, torque)
            return
        self._require_state("interval body force perturbation")
        assert self._staged_body_wrench is not None
        body_ids_np = np.asarray(body_ids)
        if (
            body_ids_np.ndim != 1
            or body_ids_np.dtype.kind not in "iu"
            or np.issubdtype(body_ids_np.dtype, np.bool_)
        ):
            raise ValueError("body_ids must be a one-dimensional integer array")
        if np.any(body_ids_np < 0) or np.any(body_ids_np >= self._staged_body_wrench.shape[1]):
            raise ValueError(
                f"body ids must be in [0, {self._staged_body_wrench.shape[1]}), "
                f"got range [{body_ids_np.min(initial=0)}, {body_ids_np.max(initial=0)}]"
            )
        body_ids_np = body_ids_np.astype(np.intp, copy=False)
        expected_shape = (self._num_envs, body_ids_np.size, 3)
        force_np = np.asarray(force, dtype=np.float32)
        if force_np.shape != expected_shape:
            raise ValueError(f"body force must have shape {expected_shape}, got {force_np.shape}")
        if not np.isfinite(force_np).all():
            raise ValueError("body force contains NaN or Inf")
        torque_np = None
        if torque is not None:
            torque_np = np.asarray(torque, dtype=np.float32)
            if torque_np.shape != expected_shape:
                raise ValueError(
                    f"body torque must have shape {expected_shape}, got {torque_np.shape}"
                )
            if not np.isfinite(torque_np).all():
                raise ValueError("body torque contains NaN or Inf")
        for body_offset, body_id in enumerate(body_ids_np):
            self._staged_body_wrench[:, int(body_id), 0:3] += force_np[:, body_offset, :]
            if torque_np is not None:
                self._staged_body_wrench[:, int(body_id), 3:6] += torque_np[:, body_offset, :]
        self._body_wrench_pending = bool(
            np.any(self._staged_body_wrench) or self._body_wrench_pending
        )

    def _step_payload(self, nsteps: int) -> dict[str, Any]:
        payload = super()._step_payload(nsteps)
        if self._entity_scene is not None and self._body_wrench_pending:
            assert self._staged_body_wrench is not None
            # NumPy pickle internals are not stable across the host and IsaacGym
            # worker interpreter versions; raw C-order bytes are.
            payload["body_wrench"] = self._staged_body_wrench.tobytes(order="C")
        return payload

    def _after_step(self, payload: dict[str, Any]) -> None:
        if "body_wrench" in payload:
            assert self._staged_body_wrench is not None
            self._staged_body_wrench.fill(0.0)
            self._body_wrench_pending = False

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
