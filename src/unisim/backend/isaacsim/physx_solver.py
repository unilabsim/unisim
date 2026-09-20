"""Bounded PhysX solver configuration shared by the IsaacSim host and worker.

The host validates constructor kwargs fail-closed, forwards the bounded set
through the cold INIT payload, and strictly compares the worker's engine
readback against the request.  The external worker re-validates the payload,
maps the values onto IsaacLab's ``sim_utils.PhysxCfg`` plus per-collision-shape
contact offsets, and reads the applied settings back from the USD stage for
its configuration report.  No engine SDK is imported at module scope, so the
host interpreter stays SDK-free; the ``stage`` helpers run worker-side only.

SimToolReal/MuJoCo-style substeps are intentionally not part of this set: the
subprocess contract already expresses them as the ``step(ctrl, nsteps)``
decimation, where one host step walks the worker through ``n`` physics
sub-steps of ``sim_dt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SOLVER_POSITION_ITERATION_FIELD = "solver_position_iteration_count"
SOLVER_VELOCITY_ITERATION_FIELD = "solver_velocity_iteration_count"
BOUNCE_THRESHOLD_FIELD = "bounce_threshold_velocity"
CONTACT_OFFSET_FIELD = "contact_offset"

PHYSX_SOLVER_ITERATION_FIELDS = (
    SOLVER_POSITION_ITERATION_FIELD,
    SOLVER_VELOCITY_ITERATION_FIELD,
)
PHYSX_SOLVER_FLOAT_FIELDS = (BOUNCE_THRESHOLD_FIELD, CONTACT_OFFSET_FIELD)
PHYSX_SOLVER_FIELDS = PHYSX_SOLVER_ITERATION_FIELDS + PHYSX_SOLVER_FLOAT_FIELDS


def _validated_iteration_count(name: str, value: Any, *, allow_zero: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if result < 0 or (result == 0 and not allow_zero):
        bound = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {bound} integer, got {value!r}")
    return result


def _validated_float(name: str, value: Any, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if allow_zero:
        if result < 0.0:
            raise ValueError(f"{name} must be nonnegative, got {value!r}")
    elif result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


@dataclass(frozen=True)
class PhysxSolverConfig:
    """The bounded, validated PhysX solver override set; ``None`` keeps defaults."""

    solver_position_iteration_count: int | None = None
    solver_velocity_iteration_count: int | None = None
    bounce_threshold_velocity: float | None = None
    contact_offset: float | None = None

    def __post_init__(self) -> None:
        if self.solver_position_iteration_count is not None:
            object.__setattr__(
                self,
                "solver_position_iteration_count",
                _validated_iteration_count(
                    SOLVER_POSITION_ITERATION_FIELD,
                    self.solver_position_iteration_count,
                    allow_zero=False,
                ),
            )
        if self.solver_velocity_iteration_count is not None:
            object.__setattr__(
                self,
                "solver_velocity_iteration_count",
                _validated_iteration_count(
                    SOLVER_VELOCITY_ITERATION_FIELD,
                    self.solver_velocity_iteration_count,
                    # PhysX accepts zero velocity iterations (its scene default
                    # minimum is 0), matching the SimToolReal default.
                    allow_zero=True,
                ),
            )
        if self.bounce_threshold_velocity is not None:
            object.__setattr__(
                self,
                "bounce_threshold_velocity",
                _validated_float(
                    BOUNCE_THRESHOLD_FIELD,
                    self.bounce_threshold_velocity,
                    allow_zero=True,
                ),
            )
        if self.contact_offset is not None:
            object.__setattr__(
                self,
                "contact_offset",
                _validated_float(CONTACT_OFFSET_FIELD, self.contact_offset, allow_zero=False),
            )

    def configured_fields(self) -> tuple[str, ...]:
        return tuple(
            field for field in PHYSX_SOLVER_FIELDS if getattr(self, field) is not None
        )

    def to_payload(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.configured_fields()}

    @classmethod
    def from_payload(cls, payload: Any) -> PhysxSolverConfig:
        """Re-validate the worker-side INIT copy; unknown keys fail closed."""
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise TypeError(
                "isaacsim physx_solver INIT payload must be a dict or null, "
                f"got {type(payload).__name__}"
            )
        unknown = sorted(set(payload).difference(PHYSX_SOLVER_FIELDS))
        if unknown:
            raise ValueError(
                "isaacsim physx_solver INIT payload contains unsupported fields: "
                + ", ".join(unknown)
            )
        return cls(**payload)


def solver_value_matches(field: str, requested: Any, reported: Any) -> bool:
    """Strict host comparison; only the engine's float32 storage is tolerated."""
    if field in PHYSX_SOLVER_ITERATION_FIELDS:
        return (
            not isinstance(reported, bool)
            and isinstance(reported, (int, np.integer))
            and int(reported) == requested
        )
    if field in PHYSX_SOLVER_FLOAT_FIELDS:
        if isinstance(reported, bool) or not isinstance(
            reported, (int, float, np.integer, np.floating)
        ):
            return False
        value = float(reported)
        return bool(np.isfinite(value)) and (
            value == requested
            or float(np.float32(value)) == float(np.float32(requested))
        )
    raise ValueError(f"unknown PhysX solver field: {field!r}")


def build_isaaclab_physx_cfg(sim_utils: Any, config: PhysxSolverConfig) -> Any:
    """Map the bounded set onto IsaacLab's ``PhysxCfg`` (worker-side only).

    PhysX clamps every actor's solver iteration counts to the scene's
    ``[min, max]`` range, so pinning both bounds to the requested count makes
    it the effective count for each actor without editing per-actor USD
    attributes.  An unconfigured field keeps IsaacLab's default.
    """
    kwargs: dict[str, Any] = {}
    if config.solver_position_iteration_count is not None:
        kwargs["min_position_iteration_count"] = config.solver_position_iteration_count
        kwargs["max_position_iteration_count"] = config.solver_position_iteration_count
    if config.solver_velocity_iteration_count is not None:
        kwargs["min_velocity_iteration_count"] = config.solver_velocity_iteration_count
        kwargs["max_velocity_iteration_count"] = config.solver_velocity_iteration_count
    if config.bounce_threshold_velocity is not None:
        kwargs["bounce_threshold_velocity"] = config.bounce_threshold_velocity
    return sim_utils.PhysxCfg(**kwargs)


def apply_contact_offset(stage: Any, contact_offset: float) -> int:
    """Author one contact offset on every collision shape (worker-side only).

    The contact offset lives on ``PhysxSchema.PhysxCollisionAPI`` (plain
    ``UsdPhysics.CollisionAPI`` has no such attribute), so the API is applied
    to every collision prim before the value is authored.
    """
    from pxr import PhysxSchema, UsdPhysics  # type: ignore[import-not-found]

    applied = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        physx_api = PhysxSchema.PhysxCollisionAPI(prim)
        if not physx_api:
            physx_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_api.CreateContactOffsetAttr().Set(float(contact_offset))
        applied += 1
    if applied == 0:
        raise RuntimeError(
            "isaacsim contact_offset was requested but the stage has no collision prims"
        )
    return applied


def read_engine_solver_values(
    stage: Any, *, include_contact_offset: bool
) -> dict[str, Any]:
    """Read the applied solver settings back from the USD stage (worker-side only)."""
    from pxr import PhysxSchema, UsdPhysics  # type: ignore[import-not-found]

    scene_api = None
    offsets: set[float] = set()
    for prim in stage.Traverse():
        if scene_api is None and prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            scene_api = PhysxSchema.PhysxSceneAPI(prim)
        if include_contact_offset and prim.HasAPI(UsdPhysics.CollisionAPI):
            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                raise RuntimeError(
                    "isaacsim collision prim is missing its PhysxCollisionAPI; "
                    "the contact offset readback would silently use engine defaults"
                )
            value = PhysxSchema.PhysxCollisionAPI(prim).GetContactOffsetAttr().Get()
            if value is None or not np.isfinite(float(value)):
                raise RuntimeError(
                    "isaacsim collision prim has no authored contact offset; "
                    "the readback would silently use engine defaults"
                )
            offsets.add(float(value))
    if scene_api is None:
        raise RuntimeError(
            "isaacsim stage has no PhysxSceneAPI; solver settings cannot be read back"
        )
    result: dict[str, Any] = {
        # PhysX clamps actor counts to [min, max]; the maximum is reported
        # because the worker pins both bounds to the requested count.
        SOLVER_POSITION_ITERATION_FIELD: int(
            scene_api.GetMaxPositionIterationCountAttr().Get()
        ),
        SOLVER_VELOCITY_ITERATION_FIELD: int(
            scene_api.GetMaxVelocityIterationCountAttr().Get()
        ),
        BOUNCE_THRESHOLD_FIELD: float(scene_api.GetBounceThresholdAttr().Get()),
    }
    if include_contact_offset:
        if not offsets:
            raise RuntimeError("isaacsim stage has no authored collision contact offsets")
        if len(offsets) != 1:
            raise RuntimeError(
                "isaacsim collision shapes report non-uniform contact offsets: "
                f"{sorted(offsets)}"
            )
        result[CONTACT_OFFSET_FIELD] = offsets.pop()
    return result
