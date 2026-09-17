"""Bounded A/B for the legacy IsaacGym post-reset FK paths.

The baseline methods are extracted from the #141 fix commit. Fixtures use
NumPy-backed tensors and synthetic MJCF chains, so measurements exclude the
IsaacGym SDK, GPU simulation, and process IPC while retaining the production
scan, FK, grouping, and shared-slot publication code.
"""

from __future__ import annotations

import argparse
import ast
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.adapters.isaacgym.test_legacy_adoption import _Tensor  # noqa: E402

from unisim.backend.isaacgym.scene_worker import SceneWorker  # noqa: E402
from unisim.backend.subprocess_ipc import protocol  # noqa: E402
from unisim.backend.subprocess_ipc.kinematics import (  # noqa: E402
    forward_kinematics,
    forward_prepared_kinematics,
    prepare_kinematics,
)
from unisim.backend.subprocess_ipc.sensors import (  # noqa: E402
    scan_scene_kinematics,
    scan_scene_metadata,
    scan_scene_metadata_with_kinematics,
)
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout  # noqa: E402

BASELINE = "bdd3ab4"
OWNER = "src/unisim/backend/isaacgym/scene_worker.py"


def historical_methods(revision: str) -> tuple[Callable[..., None], Callable[..., None]]:
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{OWNER}"], cwd=ROOT, text=True
    )
    tree = ast.parse(source)
    owner = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SceneWorker"
    )
    names = ("stage_fk_overlay_rows", "refresh")
    methods = [
        next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
        for name in names
    ]
    extracted = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)],
        type_ignores=[],
    )
    extracted.body.extend(methods)
    namespace: dict[str, Any] = {"np": np}
    exec(compile(ast.fix_missing_locations(extracted), f"{revision}:{OWNER}", "exec"), namespace)
    return namespace["stage_fk_overlay_rows"], namespace["refresh"]


def write_chain_model(directory: Path, bodies: int) -> Path:
    children = []
    for index in range(1, bodies):
        axis = "1 0 0" if index % 3 else "0 1 0"
        children.append(
            f'<body name="link{index}" pos="0.01 -0.02 0.03">'
            f'<joint name="joint{index}" axis="{axis}"/></body>'
        )
    text = (
        '<mujoco><worldbody><body name="root" pos="0 0 1"><freejoint name="root"/>'
        + "".join(children)
        + "</body></worldbody></mujoco>"
    )
    path = directory / f"chain-{bodies}.xml"
    path.write_text(text, encoding="utf-8")
    return path


def measure_pair(
    baseline: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    rounds: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(3):
        baseline()
        candidate()
    elapsed: dict[str, list[float]] = {"baseline": [], "candidate": []}
    functions = (("baseline", baseline), ("candidate", candidate))
    for index in range(rounds):
        for label, function in functions[:: 1 if index % 2 == 0 else -1]:
            start = time.perf_counter_ns()
            for _ in range(iterations):
                function()
            elapsed[label].append((time.perf_counter_ns() - start) / iterations / 1e6)
    peaks = {}
    for label, function in functions:
        tracemalloc.start()
        function()
        peaks[label] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return {
        "median_ms": {name: statistics.median(values) for name, values in elapsed.items()},
        "peak_python_bytes": peaks,
    }


def state_rows(num_envs: int, joints: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    qpos = rng.normal(size=(num_envs, 7 + joints))
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    qvel = rng.normal(size=(num_envs, 6 + joints))
    return qpos, qvel


def overlay_fixture(
    num_envs: int,
    bodies: int,
    tables: dict[str, Any],
    variant_count: int,
    *,
    legacy: bool,
) -> SceneWorker:
    joints = bodies - 1
    entity = EntityLayout(
        "robot",
        "articulation",
        "floating",
        "root",
        tuple("root" if index == 0 else f"link{index}" for index in range(bodies)),
        tuple(range(bodies)),
        (None, *("root" for _ in range(joints))),
        tuple(
            JointLayout(
                f"joint{index}",
                "hinge",
                (7 + index,),
                (6 + index,),
                "root",
            )
            for index in range(joints)
        ),
        (),
        (),
        (),
        tuple(range(7)),
        tuple(range(6)),
    )
    layout = CompiledSceneLayout((entity,), 7 + joints, 6 + joints, 0, bodies)
    rng = np.random.default_rng(19)
    native_roots = rng.normal(size=(num_envs + 5, 13)).astype(np.float32)
    native_dofs = rng.normal(size=(num_envs * joints + 5, 2)).astype(np.float32)
    native_bodies = rng.normal(size=(num_envs * bodies, 13)).astype(np.float32)
    native_contact = rng.normal(size=(num_envs * bodies, 3)).astype(np.float32)
    ctx = types.SimpleNamespace(
        _root_state=_Tensor(native_roots),
        _dof_state=_Tensor(native_dofs),
        _body_state=_Tensor(native_bodies),
        _contact_force=_Tensor(native_contact),
        _refresh_tensors=lambda: None,
        slots={
            name: np.zeros(shape, dtype=protocol.slot_dtype(name))
            for name, shape in protocol.scene_slot_shapes(num_envs, layout).items()
        },
    )
    worker = SceneWorker.__new__(SceneWorker)
    worker.ctx, worker.protocol, worker.layout = ctx, protocol, layout
    worker.num_envs, worker.faulted = num_envs, False
    worker.actor_ids = np.arange(num_envs, dtype=np.int64)[:, None]
    worker.body_ids = np.arange(num_envs * bodies, dtype=np.int64).reshape(num_envs, bodies)
    worker.body_com = rng.normal(size=(num_envs, bodies, 3)) * 0.05
    worker.root_com = rng.normal(size=(num_envs, 1, 3)) * 0.05
    worker.records = [
        [{"dof_ids": list(range(env * joints, (env + 1) * joints))}] for env in range(num_envs)
    ]
    worker.pending_roots, worker.pending_dofs = {}, {}
    worker.pending_dof_actors = set()
    worker.publish_actor_roots_as_body = True
    worker.projection = None
    worker._bind_refresh_indices()
    assignment = np.arange(num_envs, dtype=np.int64) % variant_count
    if legacy:
        worker._fk = (protocol.load_kinematics(), (tables,) * variant_count, assignment.tolist())
        worker.pending_body_fk = {}
    else:
        worker._fk = (
            protocol.load_kinematics(),
            (prepare_kinematics(tables),) * variant_count,
            assignment,
        )
        worker.pending_body_fk = {}
    return worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline_stage, baseline_refresh = historical_methods(args.baseline)

    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        model = write_chain_model(directory, bodies=128)
        model_file = str(model)

        def cold_baseline() -> None:
            scan_scene_metadata(model_file, backend_label="ablation")
            scan_scene_kinematics(model_file, backend_label="ablation")

        def cold_candidate() -> tuple[Any, dict]:
            return scan_scene_metadata_with_kinematics(model_file, backend_label="ablation")

        metadata_a, kinematics_a = scan_scene_metadata(
            model_file, backend_label="ablation"
        ), scan_scene_kinematics(model_file, backend_label="ablation")
        metadata_b, kinematics_b = cold_candidate()
        assert metadata_a == metadata_b
        assert kinematics_a == kinematics_b
        results["cold_scan"] = {
            "bodies": 128,
            "exact_output_parity": True,
            **measure_pair(cold_baseline, cold_candidate, rounds=7, iterations=1),
        }

        tables = kinematics_b
        qpos, qvel = state_rows(num_envs=512, joints=127)
        prepared = prepare_kinematics(tables)

        def fk_baseline() -> np.ndarray:
            return forward_kinematics(tables, qpos, qvel)

        def fk_candidate() -> np.ndarray:
            return forward_prepared_kinematics(prepared, qpos, qvel)

        np.testing.assert_array_equal(fk_baseline(), fk_candidate())
        results["fk_kernel"] = {
            "envs": 512,
            "bodies": 128,
            "exact_output_parity": True,
            **measure_pair(fk_baseline, fk_candidate, rounds=7, iterations=2),
        }

        old = overlay_fixture(512, 128, tables, 4, legacy=True)
        new = overlay_fixture(512, 128, tables, 4, legacy=False)
        envs = np.arange(512)
        baseline_stage(old, envs, qpos, qvel)
        new.stage_fk_overlay_rows(envs, qpos, qvel)
        baseline_refresh(old)
        new.refresh()
        for name in ("entity_root_state", "qpos", "qvel", "body_state", "contact_force"):
            np.testing.assert_array_equal(old.ctx.slots[name], new.ctx.slots[name])

        def stage_baseline() -> None:
            baseline_stage(old, envs, qpos, qvel)

        results["stage_and_refresh"] = {
            "envs": 512,
            "bodies": 128,
            "fixed_variants": 4,
            "all_published_slots_equal": True,
            "stage": measure_pair(
                stage_baseline, lambda: new.stage_fk_overlay_rows(envs, qpos, qvel),
                rounds=7,
                iterations=1,
            ),
            "refresh": measure_pair(
                lambda: baseline_refresh(old), new.refresh, rounds=7, iterations=1
            ),
        }

    report = {
        "kind": "bounded host-double path ablation; no IsaacGym SDK, GPU, or IPC",
        "baseline_revision": subprocess.check_output(
            ["git", "rev-parse", args.baseline], cwd=ROOT, text=True
        ).strip(),
        "candidate_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "candidate_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "results": results,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
