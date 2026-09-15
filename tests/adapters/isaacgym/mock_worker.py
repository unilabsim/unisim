"""Deterministic protocol mock for IsaacGym host-side fixed-variant tests.

The process speaks the real framed worker protocol but does not import
IsaacGym.  It validates the cold-path INIT contract and records the payload so
tests can assert exactly what the host sent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from unisim.backend.subprocess_ipc import protocol


def _meta_for_init(payload: dict[str, Any], *, omit_variant_echo: bool) -> dict[str, Any]:
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
                _meta_for_init(payload, omit_variant_echo=args.omit_variant_echo),
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
