"""Generate the semantic inventory or run an explicit tiny real-runtime diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from unisim.adapters import ADAPTER_SPECS
from unisim.support import FEATURES, get_adapter_capabilities

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- semantic-inventory:start -->"
END = "<!-- semantic-inventory:end -->"


def inventory_markdown() -> str:
    names = [item.name for item in ADAPTER_SPECS]
    reports = [get_adapter_capabilities(name) for name in names]
    rows = [
        "| Feature | " + " | ".join(names) + " |",
        "| --- | " + " | ".join(["---"] * len(names)) + " |",
    ]
    for feature in FEATURES:
        cells = []
        for report in reports:
            declaration = next(item for item in report.declarations if item.feature == feature)
            cells.append(declaration.support.value + ("*" if declaration.conditions else ""))
        rows.append("| `" + feature + "` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def runtime_identity(name: str) -> dict:
    if name in {"isaacgym", "isaacsim"}:
        if name == "isaacgym":
            from unisim.backend.isaacgym.dependencies import (
                build_worker_env,
                resolve_isaacgym_runtime,
            )

            runtime = resolve_isaacgym_runtime()
        else:
            from unisim.backend.isaacsim.dependencies import (
                build_worker_env,
                resolve_isaacsim_runtime,
            )

            runtime = resolve_isaacsim_runtime()
        probe = (
            "import json,platform,importlib.metadata as m; "
            "packages={d.metadata['Name']:d.version for d in m.distributions() "
            "if d.metadata['Name'].lower() in ('isaacgym','isaacsim','isaaclab','torch')}; "
            "print(json.dumps({'python':platform.python_version(),'packages':packages}))"
        )
        completed = subprocess.run(
            [str(runtime.python), "-c", probe],
            env=build_worker_env(runtime),
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        identity = json.loads(completed.stdout.strip().splitlines()[-1])
    else:
        identity = {"python": platform.python_version()}
    if name != "mujoco":
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        identity["device"] = gpu.stdout.strip()
    else:
        identity["device"] = "CPU"
    return identity


def check_configuration(name: str, fields: list[dict]) -> tuple[list[str], list[str]]:
    import numpy as np

    values = {field["field"]: field["effective"] for field in fields}
    checks = []
    np.testing.assert_allclose(values["dt"], 0.002, atol=1e-9, rtol=0)
    np.testing.assert_allclose(values["gravity"], [0, 0, -9.81], atol=1e-5, rtol=0)
    checks.extend(["effective_dt_matches_factory", "gravity_world_si"])
    mass = values["body_mass"]
    inertia = values["body_inertia"]
    if name in {"mujoco", "mjwarp"}:
        assert values["solver"] == "mjSOL_NEWTON"
        assert values["integrator"] == "mjINT_EULER"
        checks.extend(["native_newton_solver", "native_euler_integrator"])
        for body, expected_mass, diagonal in (("base", 2, 0.02), ("arm", 0.5, 0.005)):
            index = mass["names"].index(body)
            np.testing.assert_allclose(mass["values"][index], expected_mass, atol=1e-6)
            np.testing.assert_allclose(inertia["values"][index], [diagonal] * 3, atol=1e-6)
        assert values["actuator_mapping"]["names"] == ["drive"]
        np.testing.assert_allclose(values["actuator_mapping"]["gear"][0][0], 2)
        assert values["collision_filter"]["exclude_signature"]
        assert values["sensors"]["names"] == ["gyro", "accel"]
        checks.extend(["motor_gear", "authored_collision_exclusion", "native_sensor_map"])
    else:
        for body, expected_mass, diagonal in (("base", 2, 0.02), ("arm", 0.5, 0.005)):
            index = mass["names"].index(body)
            np.testing.assert_allclose(
                np.array(mass["per_env_values"])[:, index], expected_mass, atol=1e-6
            )
            matrices = np.array(inertia["per_env_matrices"])[:, index].reshape(-1, 3, 3)
            np.testing.assert_allclose(
                matrices, np.broadcast_to(np.eye(3) * diagonal, matrices.shape), atol=1e-6
            )
        assert values["collision_filter"]["self_collision"] is False
        actuator = values["actuator_mapping"]
        assert actuator["joint_names"] == ["hinge"]
        gains = actuator.get("per_env_stiffness", list(actuator.get("stiffness", {}).values()))
        np.testing.assert_allclose(gains, 10, atol=1e-6)
        checks.extend(["worker_self_collision_disabled", "worker_position_drive_gain"])
    checks.extend(["body_mass_si_all_envs", "body_principal_inertia_si_all_envs"])
    unknown = [field["field"] for field in fields if field["effective"] is None]
    return checks, unknown


def runtime_diagnostic(name: str) -> dict:
    import numpy as np

    from unisim import create_backend
    from unisim.scene import SceneCfg

    fixture_name = "m1_position.xml" if name in {"isaacgym", "isaacsim"} else "m1_semantics.xml"
    fixture = ROOT / "tests/contract/fixtures" / fixture_name
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    packages = {}
    for package in ("unisim-core", "mujoco", "mjbatch-uni", "mujoco-warp", "warp-lang"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    result = {
        "backend": name,
        "profile": "default",
        "commit": commit,
        "dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host_packages": packages,
        "command": "uv run --no-sync python " + " ".join(sys.argv),
        "fixture": str(fixture.relative_to(ROOT)),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "checks": [],
        "tolerances": {
            "dt_atol_s": 1e-9,
            "gravity_atol": 1e-5,
            "mass_inertia_atol": 1e-6,
            "sensor_shape": [2, 3],
        },
    }
    backend = None
    try:
        result["runtime"] = runtime_identity(name)
        if name in {"isaacgym", "isaacsim"}:
            try:
                rejected = create_backend(
                    name,
                    SceneCfg(model_file=str(ROOT / "tests/contract/fixtures/m1_semantics.xml")),
                    num_envs=2,
                )
                try:
                    rejected.materialize()
                finally:
                    rejected.close()
            except NotImplementedError as exc:
                result["motor_rejection"] = str(exc)
                result["checks"].append("unsupported_motor_rejected_before_worker")
            else:
                raise AssertionError("Expected unsupported motor refusal")
        backend = create_backend(name, SceneCfg(model_file=str(fixture)), num_envs=2, sim_dt=0.002)
        backend.materialize()
        report = backend.get_import_report()
        result["import_report"] = report.to_dict()
        checks, unknown = check_configuration(name, result["import_report"]["fields"])
        result["checks"].extend(checks)
        result["unverified_effective_fields"] = unknown
        backend.reset()
        backend.step(np.zeros((2, backend.num_actuators), dtype=np.float32), nsteps=2)
        sensors = ("gyro",) if name in {"isaacgym", "isaacsim"} else ("gyro", "accel")
        for sensor in sensors:
            values = backend.get_sensor_data(sensor)
            assert values.shape == (2, 3) and np.isfinite(values).all()
        result["checks"].append("real_step_and_finite_" + "_".join(sensors))
        if name in {"isaacgym", "isaacsim"}:
            try:
                backend.get_sensor_data("accel")
            except NotImplementedError as exc:
                result["accelerometer_rejection"] = str(exc)
                result["checks"].append("unsupported_accelerometer_rejected")
            else:
                raise AssertionError("Expected unsupported accelerometer refusal")
        snapshot = backend.get_import_report().to_dict()
        assert snapshot == result["import_report"]
        result["checks"].append("construction_snapshot_unchanged_after_reset_step")
        result["result"] = "passed"
    except Exception as exc:
        result["result"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if backend is not None and callable(getattr(backend, "close", None)):
            backend.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=[item.name for item in ADAPTER_SPECS])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-docs", action="store_true")
    parser.add_argument("--check-docs", action="store_true")
    args = parser.parse_args()
    if args.runtime:
        result = runtime_diagnostic(args.runtime)
        rendered = json.dumps(result, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
        print(rendered)
        return 0 if result["result"] == "passed" else 1
    table = inventory_markdown()
    if args.write_docs or args.check_docs:
        for language in ("en", "zh"):
            path = ROOT / "docs" / language / "support-matrix.md"
            content = path.read_text()
            if START not in content or END not in content:
                raise ValueError(f"Missing generated inventory markers: {path}")
            before, rest = content.split(START, 1)
            current, after = rest.split(END, 1)
            expected = "\n" + table + "\n"
            if args.write_docs:
                path.write_text(before + START + expected + END + after)
            elif current != expected:
                raise ValueError(f"Outdated generated inventory: {path}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
