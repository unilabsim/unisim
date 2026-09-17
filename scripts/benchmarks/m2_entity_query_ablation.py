"""Bounded host-query A/B on NumPy fixtures, not native simulation throughput.

The baseline method is read from git; the candidate calls the current owner.
No SDK is initialized and no GPU is used. Fixture and production behavior live
in their owners; this script only orchestrates parity and timing measurements.
"""

from __future__ import annotations

import argparse
import ast
import json
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


def baseline_method(revision, backend, module):
    path = f"src/unisim/backend/{backend}/backend.py"
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)
    tree = ast.parse(source)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == ("MuJoCoBackend" if backend == "mujoco" else "MjwarpBackend"))
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                  and node.name == "get_entity_state")
    namespace = dict(vars(module))
    shared_source = subprocess.check_output(
        ["git", "show", f"{revision}:src/unisim/entity_state.py"], cwd=ROOT, text=True)
    shared = types.ModuleType("unisim._query_ablation_entity_state")
    sys.modules[shared.__name__] = shared
    exec(compile(shared_source, f"{revision}/entity_state.py", "exec"), shared.__dict__)
    namespace["entity_state_snapshot"] = shared.entity_state_snapshot
    roots = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_entity_roots")
    wrapper = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), method, roots], type_ignores=[])
    exec(compile(ast.fix_missing_locations(wrapper), f"{revision}/{path}", "exec"), namespace)
    return namespace["get_entity_state"], namespace["_entity_roots"]


def compare(before, after, rounds, iterations):
    for name, values in before().items():
        np.testing.assert_array_equal(values, after()[name])
    for _ in range(10):
        before()
        after()
    elapsed = {"baseline": [], "candidate": []}
    functions = (("baseline", before), ("candidate", after))
    for index in range(rounds):
        for label, function in functions[::1 if index % 2 == 0 else -1]:
            start = time.perf_counter_ns()
            for _ in range(iterations):
                function()
            elapsed[label].append((time.perf_counter_ns() - start) / iterations / 1000)
    peaks = {}
    for label, function in functions:
        tracemalloc.start()
        function()
        peaks[label] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return {"median_us": {name: statistics.median(values) for name, values in elapsed.items()},
            "peak_python_bytes": peaks, "exact_output_parity": True}


def main():
    from tests.contract.test_entity_query_cost import query_owner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="e952419")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds <= 0 or args.iterations <= 0:
        parser.error("rounds and iterations must be positive")
    results = []
    for backend in ("mujoco", "mjwarp"):
        for count in (2, 4096):
            module, cls, owner, _ = query_owner(backend, count)
            original, original_roots = baseline_method(args.baseline, backend, module)
            owner._entity_roots = lambda: original_roots(owner)
            result = compare(lambda: original(owner, "object"),
                             lambda: cls.get_entity_state(owner, "object"),
                             args.rounds, args.iterations)
            results.append({"backend": backend, "num_envs": count, **result})
    record = {
        "scope": "NumPy host query fixture only; no native simulation or GPU",
        "baseline": subprocess.check_output(["git", "rev-parse", args.baseline],
                                             cwd=ROOT, text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=ROOT, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "rounds": args.rounds, "iterations": args.iterations, "results": results,
    }
    text = json.dumps(record, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
