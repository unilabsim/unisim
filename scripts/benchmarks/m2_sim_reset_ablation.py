"""Host-double A/B accounting of IsaacSim sparse reset transfers, without SDKs.

The baseline _commit method is extracted from a fixed Git revision. Tensor
doubles execute both methods and count explicit H2D/D2H payload bytes while
checking identical setter/reset/update calls. This does not measure native
GPU bandwidth, implicit IsaacLab transfers, or end-to-end reset throughput.
"""

from __future__ import annotations

import argparse
import ast
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.adapters.isaacsim.reset_transfer_fixture import execute_case  # noqa: E402

from unisim.backend.isaacsim.scene_worker import SceneWorkerContext, _numpy  # noqa: E402

BASELINE = "e952419"
OWNER = "src/unisim/backend/isaacsim/scene_worker.py"


def baseline_commit(revision: str = BASELINE):
    source = subprocess.check_output(["git", "show", f"{revision}:{OWNER}"], cwd=ROOT, text=True)
    module = ast.parse(source)
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "SceneWorkerContext"
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_commit"
    )
    extracted = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace: dict[str, Any] = {"np": np, "_numpy": _numpy}
    exec(compile(extracted, f"{revision}:{OWNER}::_commit", "exec"), namespace)
    return namespace["_commit"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-joints", type=int, default=32)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 256])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = baseline_commit(args.baseline)
    cases = []
    for rows in args.rows:
        parameters = {"num_envs": args.num_envs, "num_joints": args.num_joints, "rows": rows}
        before = execute_case(baseline, **parameters)
        after = execute_case(SceneWorkerContext._commit, **parameters)
        assert before["operations"] == after["operations"], "native-call values/order changed"
        for record in (before, after):
            record["native_operation_count"] = len(record.pop("operations"))
        cases.append(
            {
                **parameters,
                "baseline": before,
                "candidate": after,
                "native_call_values_and_order_equal": True,
            }
        )
    report = {
        "kind": "executed host-double transfer accounting; not native throughput",
        "baseline_revision": args.baseline,
        "candidate_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "candidate_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        ),
        "command": "uv run --no-sync python " + " ".join(sys.argv),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cases": cases,
    }
    output = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
