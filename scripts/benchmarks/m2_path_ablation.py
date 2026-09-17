"""Bounded host mapping A/B; native physics and transfer costs are excluded."""

from __future__ import annotations

import argparse
import ast
import json
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.adapters.isaacgym.test_refresh_mapping import refresh_fixture  # noqa: E402


def historical_refresh(revision):
    filename = "src/unisim/backend/isaacgym/scene_worker.py"
    source = subprocess.check_output(["git", "show", f"{revision}:{filename}"],
                                     cwd=ROOT, text=True)
    tree = ast.parse(source)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "SceneWorker")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                  and node.name == "refresh")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), filename, "exec"), namespace)
    return namespace["refresh"]


def measure(function, repeats):
    function()
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - start) / 1e6)
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"median_ms": statistics.median(values), "peak_python_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="e952419326d96d2e89faa087c030469c0ebd0adc")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = historical_refresh(args.baseline)
    results = []
    for count in (2, 512, 4096):
        worker = refresh_fixture(count=count, joints=32)
        old = types.MethodType(baseline, worker)
        old()
        expected = {name: value.copy() for name, value in worker.ctx.slots.items()}
        worker.refresh()
        for name, value in expected.items():
            np.testing.assert_array_equal(worker.ctx.slots[name], value)
        results.append({"envs": count, "joints": 32, "all_slots_equal": True,
                        "A_loop": measure(old, 7), "B_gather": measure(worker.refresh, 7)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "baseline": args.baseline,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                        text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "python": platform.python_version(), "numpy": np.__version__,
        "scope": "actual refresh method with host-backed tensor doubles; excludes SDK/GPU/IPC",
        "repeats": 7, "gym_refresh": results,
    }, indent=2) + "\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
