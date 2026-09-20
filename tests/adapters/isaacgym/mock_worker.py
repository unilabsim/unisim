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
from pathlib import Path
from typing import Any

from unisim.backend.subprocess_ipc import protocol


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
    while True:
        message = protocol.recv_message(stdin)
        command = message["cmd"]
        payload: Any = message.get("payload")
        if command == protocol.CMD_INIT:
            if not isinstance(payload, dict):
                raise TypeError("INIT payload must be a dict")
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
            protocol.CMD_ATTACH,
            protocol.CMD_STEP,
            protocol.CMD_SET_STATE,
            protocol.CMD_REFRESH,
            protocol.CMD_SHUTDOWN,
        ):
            protocol.send_message(stdout, protocol.CMD_READY, None)
            if command == protocol.CMD_SHUTDOWN:
                return 0
        else:
            raise ValueError(f"mock worker does not implement {command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
