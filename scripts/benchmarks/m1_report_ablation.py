"""Bounded A/B measurements of M1 cold reporting; never selects production behavior."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
import types
from pathlib import Path
from unittest.mock import patch

from unisim import get_adapter_capabilities, inspection

ROOT = Path(__file__).resolve().parents[2]


def historical_module(revision: str, filename: str):
    source = subprocess.check_output(
        ["git", "show", f"{revision}:src/unisim/{filename}.py"], cwd=ROOT, text=True,
    )
    name = "unisim._ablation_" + filename
    module = types.ModuleType(name)
    module.__package__ = "unisim"
    sys.modules[name] = module
    exec(compile(source, f"{revision}/{filename}.py", "exec"), module.__dict__)
    return module


def measure(fn, repeats=9):
    fn()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed.append((time.perf_counter() - start) * 1000)
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"median_ms": statistics.median(elapsed), "peak_python_bytes": peak}


def mujoco_paths():
    import numpy as np

    import unisim.backend.mujoco.backend as owner
    from unisim import SemanticRequirements, create_backend
    from unisim.scene import SceneCfg

    n = 128
    xml = "<mujoco><worldbody>" + "".join(
        f'<body name="b{i}" pos="{i * .3} 0 1"><joint name="j{i}"/>'
        '<geom size=".05" mass="1"/></body>' for i in range(n)
    ) + "</worldbody><actuator>" + "".join(
        f'<motor joint="j{i}"/>' for i in range(n)
    ) + "</actuator></mujoco>"
    results, states = {}, {}
    original = owner.mujoco_model_configuration, owner.compare_configuration
    with tempfile.TemporaryDirectory() as directory:
        model = Path(directory) / "model.xml"
        model.write_text(xml)
        for mode in ("report_off", "report_on", "strict"):
            readers = original if mode != "report_off" else (
                lambda *a, **kw: {}, lambda *a, **kw: inspection.ImportReport("mujoco"),
            )
            with patch.object(owner, "mujoco_model_configuration", readers[0]), patch.object(
                owner, "compare_configuration", readers[1]
            ):
                timings, steps = [], []
                for _ in range(8):
                    start = time.perf_counter()
                    backend = create_backend(
                        "mujoco", SceneCfg(str(model)), num_envs=32, sim_dt=.002,
                        semantic_requirements=(SemanticRequirements(settings=("dt",))
                                               if mode == "strict" else None),
                    )
                    if mode != "strict":
                        backend.materialize()
                    timings.append((time.perf_counter() - start) * 1000)
                    ctrl = np.zeros((32, n))
                    backend.step(ctrl, nsteps=2)
                    start = time.perf_counter()
                    backend.step(ctrl, nsteps=10)
                    steps.append((time.perf_counter() - start) * 1000)
                    states[mode] = backend.get_state(("qpos", "qvel"))
                    backend.cleanup_scene_assets()
                    del backend
                results[mode] = {"construct_materialize_ms": statistics.median(timings[1:]),
                                 "ten_substeps_ms": statistics.median(steps[1:])}
    assert all(np.array_equal(states[mode][key], states["report_off"][key])
               for mode in states for key in states[mode])
    return {"paths": results, "states_equal": True, "envs": 32, "joints": n, "repeats": 7}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="b0f77ab")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mujoco", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = historical_module(args.baseline, "inspection")
    old_capabilities = historical_module(args.baseline, "capabilities")
    values = {"body_inertia": {"names": [f"b{i}" for i in range(512)],
                              "per_env_values": [[[.1, .2, .3] for _ in range(512)]
                                                 for _ in range(32)]}}
    adopted = copy.deepcopy(values)
    options = {"source": "source", "effective_source": "readback"}
    old_report = baseline.compare_configuration("fixture", values, adopted, **options)
    new_report = inspection.compare_configuration("fixture", values, adopted, **options)
    assert old_report.to_dict() == new_report.to_dict()
    declaration = get_adapter_capabilities("mujoco")
    old_declaration = old_capabilities.CapabilityReport.from_dict(declaration.to_dict())
    assert old_declaration.to_dict() == declaration.to_dict()
    result = {
        "baseline": subprocess.check_output(
            ["git", "rev-parse", args.baseline], cwd=ROOT, text=True,
        ).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "python": sys.version.split()[0], "repeats": 9,
        "report_shape": [32, 512, 3], "separate_equal_inputs": True,
        "same_serialized_values": True,
        "report_before": measure(lambda: baseline.compare_configuration(
            "fixture", values, adopted, **options)),
        "report_after": measure(lambda: inspection.compare_configuration(
            "fixture", values, adopted, **options)),
        "shared_input_before": measure(lambda: baseline.compare_configuration(
            "fixture", values, values, **options)),
        "shared_input_after": measure(lambda: inspection.compare_configuration(
            "fixture", values, values, **options)),
        "serialization_before": measure(old_declaration.to_dict),
        "serialization_after": measure(declaration.to_dict),
    }
    if args.cuda:
        import numpy as np
        import warp as wp

        wp.init()
        array = wp.array(np.ones((4096, 128, 3), dtype=np.float32), device="cuda:0")
        assert np.array_equal(array.numpy()[0], array[0:1].numpy()[0])
        result["cuda_device"] = wp.get_device("cuda:0").name
        result["readback_shape"] = [4096, 128, 3]
        result["full_readback"] = measure(lambda: array.numpy()[0])
        result["selected_readback"] = measure(lambda: array[0:1].numpy()[0])
    if args.mujoco:
        result["mujoco"] = mujoco_paths()
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
