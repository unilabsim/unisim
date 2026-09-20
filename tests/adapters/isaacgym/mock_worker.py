"""Deterministic protocol mock for IsaacGym host-side fixed-variant tests.

The process speaks the real framed worker protocol but does not import
IsaacGym.  It validates the cold-path INIT contract and records the payload so
tests can assert exactly what the host sent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

from unisim.backend.subprocess_ipc import protocol

_RANDOMIZATION_TERMS = frozenset(
    {
        "kp",
        "kd",
        "body_mass",
        "body_inertia",
        "body_ipos",
        "dof_armature",
        "dof_frictionloss",
        "geom_friction",
    }
)


def _scene_state(payload: dict[str, Any]) -> dict[str, dict[str, list]]:
    """Build per-entity mock property tables from the INIT entity payload."""
    state: dict[str, dict[str, list]] = {}
    for entry, spec in zip(payload["scene_layout"]["entities"], payload["scene_entities"]):
        variants = spec["variants"]
        assignment = spec["assignment"]
        nb = len(entry["body_ids"])
        ng = len(entry.get("geoms") or ())

        def per_env(field: str, default: Any) -> list:
            return [variants[a].get(field, default) for a in assignment]

        state[entry["name"]] = {
            "spec": spec,
            "entry": entry,
            "body_mass": per_env("body_mass", [1.0] * nb),
            "body_ipos": per_env("body_ipos", [[0.0, 0.0, 0.0]] * nb),
            "body_inertia": [
                [
                    np.diag(np.asarray(triple, dtype=float)).tolist()
                    for triple in variants[a]["body_inertia"]
                ]
                for a in assignment
            ],
            "dof_stiffness": per_env("dof_stiffness", []),
            "dof_damping": per_env("dof_damping", []),
            "dof_armature": per_env("dof_armature", []),
            "dof_friction": per_env("dof_friction", []),
            "geom_friction": per_env("geom_friction", [[0.5, 0.5, 0.0]] * ng),
        }
    return state


def _apply_randomization(
    state: dict[str, dict[str, list]],
    randomization: dict[str, Any],
    env_ids: list[int],
) -> None:
    if not isinstance(randomization, dict) or not set(randomization) <= _RANDOMIZATION_TERMS:
        raise ValueError("randomization must contain only supported property terms")
    for row, env in enumerate(env_ids):
        geom_offset = 0
        for name, tables in state.items():
            entry = tables["entry"]
            body_ids = entry["body_ids"]
            if "body_mass" in randomization:
                tables["body_mass"][env] = [randomization["body_mass"][row][b] for b in body_ids]
            if "body_ipos" in randomization:
                tables["body_ipos"][env] = [randomization["body_ipos"][row][b] for b in body_ids]
            if "body_inertia" in randomization:
                tables["body_inertia"][env] = [
                    np.diag(np.asarray(randomization["body_inertia"][row][b], float)).tolist()
                    for b in body_ids
                ]
            joint_names = [joint["name"] for joint in entry["joints"]]
            for term, field in (("kp", "dof_stiffness"), ("kd", "dof_damping")):
                if term in randomization:
                    row_values = list(tables[field][env])
                    for column, joint_name in zip(
                        entry["actuator_indices"], entry["actuator_joint_names"]
                    ):
                        row_values[joint_names.index(joint_name)] = randomization[term][row][
                            column
                        ]
                    tables[field][env] = row_values
            for term, field in (
                ("dof_armature", "dof_armature"),
                ("dof_frictionloss", "dof_friction"),
            ):
                if term in randomization:
                    row_values = list(tables[field][env])
                    for public, joint in enumerate(entry["joints"]):
                        row_values[public] = randomization[term][row][joint["qvel_indices"][0]]
                    tables[field][env] = row_values
            geom_count = len(entry.get("geoms") or ())
            if "geom_friction" in randomization and geom_count:
                tables["geom_friction"][env] = randomization["geom_friction"][row][
                    geom_offset : geom_offset + geom_count
                ]
            geom_offset += geom_count


def _property_records(state: dict[str, dict[str, list]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "body_mass": tables["body_mass"],
            "body_ipos": tables["body_ipos"],
            "body_inertia": tables["body_inertia"],
            "dof_stiffness": tables["dof_stiffness"],
            "dof_damping": tables["dof_damping"],
            "dof_armature": tables["dof_armature"],
            "dof_friction": tables["dof_friction"],
            "geom_friction": tables["geom_friction"],
        }
        for name, tables in state.items()
    ]


def _meta_for_init(
    payload: dict[str, Any],
    *,
    omit_variant_echo: bool,
    spacing_error: str | None,
) -> dict[str, Any]:
    env_spacing = float(payload.get("env_spacing", 4.0))
    columns = max(1, math.ceil(math.sqrt(payload["num_envs"])))
    env_origins = [
        [index % columns * env_spacing, index // columns * env_spacing, 0.0]
        for index in range(payload["num_envs"])
    ]
    reported_spacing = env_spacing + 1.0 if spacing_error == "reported" else env_spacing
    if spacing_error == "origin" and env_origins:
        env_origins[0][0] += 0.25
    if "scene_entities" in payload:
        entities = []
        for entry in payload["scene_entities"]:
            assignment = entry["assignment"]
            entities.append(
                {
                    "name": entry["name"],
                    "assignment": assignment,
                    "body_mass": [
                        entry["variants"][variant]["body_mass"] for variant in assignment
                    ],
                    "body_sphere_radii": [
                        entry["variants"][variant]["body_sphere_radii"]
                        for variant in assignment
                    ],
                }
            )
        return {
            "scene_layout": payload["scene_layout"],
            "scene_entities_actual": entities,
            "gravity": payload["gravity"],
            "use_gpu_pipeline": False,
            "graphics_enabled": False,
            "configuration_report": {
                "schema_version": 1,
                "effective": {
                    "dt": payload["sim_dt"],
                    "gravity": payload["gravity"],
                    "solver": "mock",
                    "env_spacing": env_spacing,
                    "entity_gravity_disabled": {
                        entry["name"]: bool(entry["gravity_disabled"])
                        for entry in payload["scene_entities"]
                    },
                    "collision_filter": {
                        "self_collision": {
                            entry["name"]: bool(entry.get("self_collision", False))
                            for entry in payload["scene_entities"]
                        },
                    },
                },
            },
            "env_spacing": reported_spacing,
            "env_origins": env_origins,
        }

    joint_names = [str(name) for name in payload.get("mjcf_joint_names") or []]
    body_names = [str(name) for name in payload.get("mjcf_body_names") or []]
    meta: dict[str, Any] = {
        "num_dof": len(joint_names),
        "num_bodies": len(body_names),
        "dof_names": joint_names,
        "body_names": body_names,
        "dof_lower": [0.0] * len(joint_names),
        "dof_upper": [0.0] * len(joint_names),
        "effort": [0.0] * len(joint_names),
        "gravity": [0.0, 0.0, -9.81],
        "env_spacing": reported_spacing,
        "env_origins": env_origins,
        "use_gpu_pipeline": False,
        "graphics_enabled": False,
    }
    if "variant_model_files" in payload and not omit_variant_echo:
        meta["fixed_variant_count"] = len(payload["variant_model_files"])
        meta["fixed_variant_assignment"] = list(payload["variant_assignment"])
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--omit-variant-echo", action="store_true")
    parser.add_argument("--spacing-error", choices=("reported", "origin"), default=None)
    parser.add_argument("--reset-error", choices=("validation", "native"), default=None)
    # The host always appends the canonical protocol path. This mock imports
    # the installed package directly, so the argument is accepted and ignored.
    parser.add_argument("--protocol", default=None)
    args = parser.parse_args(argv)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    shm_handles: list[Any] = []
    slots: dict[str, np.ndarray] = {}
    state: dict[str, dict[str, list]] = {}
    while True:
        message = protocol.recv_message(stdin)
        command = message["cmd"]
        payload: Any = message.get("payload")
        if command == protocol.CMD_INIT:
            if not isinstance(payload, dict):
                raise TypeError("INIT payload must be a dict")
            if "scene_entities" in payload:
                state = _scene_state(payload)
            if args.record is not None:
                args.record.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            protocol.send_message(
                stdout,
                protocol.CMD_META,
                _meta_for_init(
                    payload,
                    omit_variant_echo=args.omit_variant_echo,
                    spacing_error=args.spacing_error,
                ),
            )
        elif command == protocol.CMD_ATTACH:
            for name, spec in payload["slots"].items():
                handle = shared_memory.SharedMemory(name=spec["shm"], create=False)
                slots[name] = np.ndarray(
                    tuple(spec["shape"]), dtype=np.dtype(spec["dtype"]), buffer=handle.buf
                )
                shm_handles.append(handle)
            protocol.send_message(stdout, protocol.CMD_READY, None)
        elif command == protocol.CMD_RESET_ENTITIES:
            if "randomization" in payload:
                count = int(payload["count"])
                env_ids = [int(i) for i in slots["reset_env_ids"][:count]]
                _apply_randomization(state, payload["randomization"], env_ids)
                protocol.send_message(
                    stdout,
                    protocol.CMD_READY,
                    {"timing": {}, "native_entity_records": _property_records(state)},
                )
            else:
                protocol.send_message(stdout, protocol.CMD_READY, {"timing": {}})
        elif command == protocol.CMD_SET_STATE and args.reset_error is not None:
            protocol.send_message(
                stdout,
                protocol.CMD_ERROR,
                {
                    "type": "RuntimeError",
                    "message": "injected reset " + args.reset_error,
                    "traceback": "mock reset",
                    "faulted": args.reset_error == "native",
                },
            )
        elif command in (
            protocol.CMD_STEP,
            protocol.CMD_SET_STATE,
            protocol.CMD_REFRESH,
            protocol.CMD_SHUTDOWN,
        ):
            protocol.send_message(stdout, protocol.CMD_READY, None)
            if command == protocol.CMD_SHUTDOWN:
                for handle in shm_handles:
                    handle.close()
                return 0
        else:
            raise ValueError(f"mock worker does not implement {command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
