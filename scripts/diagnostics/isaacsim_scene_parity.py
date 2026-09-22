#!/usr/bin/env python3
"""Dump/diff parity harness for the mapped IsaacSim scene materialization path.

Purpose: reproduce the bit-exact equivalence evidence used to validate
unilabsim/unisim#283 (batched prototype copies + prototype scope removal)
whenever the mapped-worker materialization path (prototype authoring, batched
``Sdf.CopySpec``, scope removal, collision filtering, reset, audit) changes.

``dump`` builds a small mapped scene (K tool variants x N envs, round-robin
assignment) through the full public host-to-worker path
(``create_backend("isaacsim", ...)``), captures the worker's INIT metadata —
including the additive ``init_telemetry`` phase report and the
``scene_entities_actual`` native audit records (per-env variant identity,
body masses, sphere radii, geometry rows) — plus the post-reset state and a
deterministic zero-control rollout (qpos/qvel and per-entity root pose and
velocity at every step), and writes one JSON.

``diff`` compares two dumps fail-closed: configuration, environment origins,
audit records, and stage prim counts must be identical; rollout arrays must
agree within ``--tolerance`` (default 1e-6, absolute + relative). On the same
host and GPU the rollout is expected to be bit-identical (the #283 baseline
showed max |diff| = 0.0); any nonzero drift beyond the tolerance means the
materialization change altered training semantics — stop and investigate, do
not widen the tolerance.

Prerequisites: the vendored IsaacSim worker runtime (scripts/setup-isaacsim
or the host's conventional worker install), a CUDA GPU, and the unisimtoolreal
asset tree (robot/table MJCFs + tools manifest). Run both dumps on the same
machine; PhysX results are not portable across hosts/GPUs.

Interpretation: "PARITY OK" means the change is semantics-preserving on this
scene; informational lines report stage prim counts and peak RSS (prototype
removal legitimately reduces both). "PARITY FAIL" lists every mismatch.

Limit: worker-internal USD details (composed destination xforms, prim
hierarchy) are not host-observable; the #283 forensics captured those with a
scratch worker-side hook. This script covers the full host-observable surface:
native audit records, INIT telemetry, and physics rollout.

Example:

    OMNI_KIT_ACCEPT_EULA=yes uv run --no-sync python \
      scripts/diagnostics/isaacsim_scene_parity.py dump \
      --assets-root /path/to/unisimtoolreal --output /tmp/parity_a.json
    # ... apply the change under test ...
    OMNI_KIT_ACCEPT_EULA=yes uv run --no-sync python \
      scripts/diagnostics/isaacsim_scene_parity.py dump \
      --assets-root /path/to/unisimtoolreal --output /tmp/parity_b.json
    uv run --no-sync python scripts/diagnostics/isaacsim_scene_parity.py \
      diff /tmp/parity_a.json /tmp/parity_b.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_NUM_ENVS = 64
DEFAULT_NUM_VARIANTS = 12
DEFAULT_STEPS = 50


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dump = sub.add_parser("dump", help="capture one parity dump JSON")
    dump.add_argument("--assets-root", type=Path, required=True)
    dump.add_argument("--output", type=Path, required=True)
    dump.add_argument("--num-envs", type=_positive_int, default=DEFAULT_NUM_ENVS)
    dump.add_argument("--num-variants", type=_positive_int, default=DEFAULT_NUM_VARIANTS)
    dump.add_argument("--steps", type=_positive_int, default=DEFAULT_STEPS)
    dump.add_argument("--sim-dt", type=float, default=1.0 / 60.0)
    dump.add_argument("--nsteps", type=_positive_int, default=2)
    dump.add_argument("--worker-timeout-s", type=float, default=1800.0)
    diff = sub.add_parser("diff", help="fail-closed comparison of two dumps")
    diff.add_argument("baseline", type=Path)
    diff.add_argument("candidate", type=Path)
    diff.add_argument("--tolerance", type=float, default=1e-6)
    return parser


def _build_scene(assets_root: Path, assignment: np.ndarray, variants: tuple[Any, ...]) -> Any:
    from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
    from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
    from unisim.scene import SceneCfg

    robot = assets_root / "assets" / "robot" / "robot.xml"
    table = assets_root / "assets" / "table" / "table.xml"
    for path in (robot, table):
        if not path.is_file():
            raise ValueError(f"scene asset missing: {path}")
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                ModelSourceDescriptor(str(robot)),
                root_mode="fixed",
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.15)),
            ),
            SceneEntitySpec(
                "table",
                ModelSourceDescriptor(str(table)),
                kind="rigid",
                root_mode="fixed",
            ),
            SceneEntitySpec(
                "tool",
                variants[0],
                kind="rigid",
                root_mode="floating",
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.3)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            "tool",
            FixedVariantPlan(assignment, variants, layout=FixedVariantLayout.SAME_LAYOUT),
        ),
    )


def _load_tool_sources(assets_root: Path, num_variants: int) -> tuple[Any, ...]:
    """Return the name-sorted two-geom tool pool (same rule as the scale bench)."""
    import re

    from unisim.dr.types import ModelSourceDescriptor

    manifest_path = assets_root / "manifests" / "tools.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "unitoolreal.tools.v1":
        raise ValueError(f"unexpected tools manifest schema in {manifest_path}")
    entries = sorted(
        (entry for entry in manifest["tools"] if "head_mesh" in entry),
        key=lambda entry: entry["name"],
    )
    sources = []
    for entry in entries:
        model = assets_root / entry["model"]
        geom_names = tuple(re.findall(r'<geom[^>]*name="([^"]+)"', model.read_text()))
        if geom_names != ("handle", "head"):
            raise ValueError(f"{model} geoms {geom_names} != ('handle', 'head')")
        sources.append(ModelSourceDescriptor(str(model)))
    if num_variants > len(sources):
        raise ValueError(
            f"requested {num_variants} variants but only {len(sources)} two-geom tools exist"
        )
    return tuple(sources[:num_variants])


def _entity_states(backend: Any) -> dict[str, Any]:
    states = {}
    for name in backend.get_entity_names():
        state = backend.get_entity_state(name)
        states[name] = {key: np.asarray(value).tolist() for key, value in state.items()}
    return states


def run_dump(args: argparse.Namespace) -> None:
    from unisim import create_backend

    assignment = np.arange(args.num_envs, dtype=np.int64) % args.num_variants
    variants = _load_tool_sources(args.assets_root.expanduser().resolve(), args.num_variants)
    scene = _build_scene(args.assets_root.expanduser().resolve(), assignment, variants)

    backend = create_backend(
        "isaacsim",
        scene,
        num_envs=args.num_envs,
        sim_dt=args.sim_dt,
        isaacsim_worker_timeout_s=args.worker_timeout_s,
    )
    worker_metadata: dict[str, Any] = {}
    original_bind = backend._bind_scene_metadata

    def bind_metadata(metadata: dict[str, Any]) -> None:
        worker_metadata.update(metadata)
        original_bind(metadata)

    backend._bind_scene_metadata = bind_metadata
    try:
        backend.materialize()
        backend.reset()
        state = {
            "qpos": np.asarray(backend.get_state("qpos")["qpos"]).tolist(),
            "qvel": np.asarray(backend.get_state("qvel")["qvel"]).tolist(),
            "entities": _entity_states(backend),
        }
        ctrl = np.zeros((args.num_envs, backend.num_actuators), dtype=np.float32)
        rollout = [state]
        for _ in range(args.steps):
            backend.step(ctrl, args.nsteps)
            rollout.append(
                {
                    "qpos": np.asarray(backend.get_state("qpos")["qpos"]).tolist(),
                    "qvel": np.asarray(backend.get_state("qvel")["qvel"]).tolist(),
                    "entities": _entity_states(backend),
                }
            )
    finally:
        backend.close()

    telemetry = worker_metadata.get("init_telemetry")
    payload = {
        "schema_version": 1,
        "config": {
            "num_envs": args.num_envs,
            "num_variants": args.num_variants,
            "steps": args.steps,
            "sim_dt": args.sim_dt,
            "nsteps": args.nsteps,
            "assignment": assignment.tolist(),
        },
        "meta": {
            "dt": worker_metadata.get("dt"),
            "gravity": worker_metadata.get("gravity"),
            "env_origins": worker_metadata.get("env_origins"),
            "collision_filtering_applied": worker_metadata.get("collision_filtering_applied"),
            "configuration_report": worker_metadata.get("configuration_report"),
            "scene_entities_actual": worker_metadata.get("scene_entities_actual"),
            "init_telemetry": (
                {
                    "schema_version": telemetry.get("schema_version"),
                    "stage_prims": telemetry.get("stage_prims"),
                    "peak_rss_mb": telemetry.get("peak_rss_mb"),
                    "vram_used_mb": telemetry.get("vram_used_mb"),
                    "phases": [
                        {"name": phase["name"], "elapsed_ms": phase["elapsed_ms"]}
                        for phase in telemetry.get("phases", [])
                    ],
                }
                if isinstance(telemetry, dict)
                else None
            ),
        },
        "rollout": rollout,
    }
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[parity] dump written to {args.output}")


def _flatten(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, sub in value.items():
            yield from _flatten(sub, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, sub in enumerate(value):
            yield from _flatten(sub, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def run_diff(args: argparse.Namespace) -> int:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    tolerance = args.tolerance

    mismatches: list[str] = []

    # Phase timings and resource snapshots are workload-dependent; report the
    # interesting deltas informationally instead of gating on them.
    b_telemetry = baseline["meta"].get("init_telemetry") or {}
    c_telemetry = candidate["meta"].get("init_telemetry") or {}
    for key in ("peak_rss_mb", "vram_used_mb"):
        if b_telemetry.get(key) != c_telemetry.get(key):
            print(f"INFO init_telemetry.{key}: {b_telemetry.get(key)} -> {c_telemetry.get(key)}")

    # Exact sections: configuration, audit records, origins, prim counts.
    exact_sections = {
        "config": (baseline["config"], candidate["config"]),
        "meta.dt": (baseline["meta"]["dt"], candidate["meta"]["dt"]),
        "meta.gravity": (baseline["meta"]["gravity"], candidate["meta"]["gravity"]),
        "meta.env_origins": (baseline["meta"]["env_origins"], candidate["meta"]["env_origins"]),
        "meta.collision_filtering_applied": (
            baseline["meta"]["collision_filtering_applied"],
            candidate["meta"]["collision_filtering_applied"],
        ),
        "meta.configuration_report": (
            baseline["meta"]["configuration_report"],
            candidate["meta"]["configuration_report"],
        ),
        "meta.scene_entities_actual": (
            baseline["meta"]["scene_entities_actual"],
            candidate["meta"]["scene_entities_actual"],
        ),
        "meta.init_telemetry.stage_prims": (
            b_telemetry.get("stage_prims"),
            c_telemetry.get("stage_prims"),
        ),
        "meta.init_telemetry.phase_names": (
            [phase["name"] for phase in b_telemetry.get("phases", [])],
            [phase["name"] for phase in c_telemetry.get("phases", [])],
        ),
    }
    for section, (b_sec, c_sec) in exact_sections.items():
        b_flat = dict(_flatten(b_sec, section))
        c_flat = dict(_flatten(c_sec, section))
        for key in b_flat.keys() | c_flat.keys():
            if key not in b_flat:
                mismatches.append(f"{key} missing in baseline")
                continue
            if key not in c_flat:
                mismatches.append(f"{key} missing in candidate")
                continue
            b_value, c_value = b_flat[key], c_flat[key]
            if _is_number(b_value) and _is_number(c_value):
                same = (
                    math.isnan(b_value) and math.isnan(c_value)
                ) or b_value == c_value
                if not same:
                    mismatches.append(f"{key}: {b_value} != {c_value}")
            elif b_value != c_value:
                mismatches.append(f"{key}: {b_value!r} != {c_value!r}")

    # Rollout: per-step float arrays under the tolerance.
    b_rollout, c_rollout = baseline["rollout"], candidate["rollout"]
    if len(b_rollout) != len(c_rollout):
        mismatches.append(f"rollout length {len(b_rollout)} != {len(c_rollout)}")
    else:
        worst = 0.0
        worst_key = ""
        for step, (b_step, c_step) in enumerate(zip(b_rollout, c_rollout)):
            b_flat = dict(_flatten(b_step, f"step{step}"))
            c_flat = dict(_flatten(c_step, f"step{step}"))
            for key in b_flat.keys() | c_flat.keys():
                if key not in b_flat or key not in c_flat:
                    mismatches.append(f"rollout {key} missing on one side")
                    continue
                b_value, c_value = b_flat[key], c_flat[key]
                difference = abs(b_value - c_value)
                if difference > worst:
                    worst, worst_key = difference, key
                if difference > tolerance + tolerance * abs(b_value):
                    mismatches.append(f"rollout {key}: {b_value} vs {c_value}")
        print(f"INFO rollout max |diff| = {worst:.3e} at {worst_key}")

    if mismatches:
        print(f"PARITY FAIL ({len(mismatches)} mismatches)")
        for line in mismatches[:50]:
            print("  " + line)
        return 1
    print("PARITY OK")
    return 0


def main() -> int:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    # Auto-detect the conventional worker install, mirroring downstream
    # training entries; an explicit UNISIM_ISAACSIM_PYTHON always wins.
    conventional = Path.home() / ".cache" / "unisim" / "isaacsim" / "venv" / "bin" / "python"
    if not os.environ.get("UNISIM_ISAACSIM_PYTHON") and conventional.is_file():
        os.environ["UNISIM_ISAACSIM_PYTHON"] = str(conventional)
    args = _parser().parse_args()
    if args.command == "dump":
        run_dump(args)
        return 0
    return run_diff(args)


if __name__ == "__main__":
    sys.exit(main())
