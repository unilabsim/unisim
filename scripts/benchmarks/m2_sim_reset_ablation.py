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
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from unisim.backend.isaacsim.scene_worker import SceneWorkerContext, _numpy  # noqa: E402

BASELINE = "e952419"
OWNER = "src/unisim/backend/isaacsim/scene_worker.py"


class TransferProbe:
    def __init__(self) -> None:
        self.d2h: list[int] = []
        self.h2d: list[int] = []
        self.operations: list[dict[str, Any]] = []

    def as_tensor(self, value: Any, **kwargs: Any) -> np.ndarray:
        array = np.asarray(value, dtype=kwargs.get("dtype"))
        self.h2d.append(array.nbytes)
        return array.copy()


class TensorDouble:
    def __init__(self, values: np.ndarray, probe: TransferProbe) -> None:
        self.values, self.probe = values, probe

    def detach(self) -> TensorDouble:
        return self

    def cpu(self) -> TensorDouble:
        self.probe.d2h.append(self.values.nbytes)
        return self

    def numpy(self) -> np.ndarray:
        return self.values

    def __getitem__(self, key: Any) -> TensorDouble:
        return TensorDouble(self.values[key], self.probe)


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


def execute_case(commit, *, num_envs: int, num_joints: int, rows: int) -> dict[str, Any]:
    if not 1 <= rows <= num_envs or num_joints < 4:
        raise ValueError("require 1 <= rows <= num_envs and num_joints >= 4")
    probe = TransferProbe()
    worker = SceneWorkerContext.__new__(SceneWorkerContext)
    worker.device, worker.sim_dt, worker.faulted = "host-double", 0.002, False
    worker.origins = np.zeros((num_envs, 3), dtype=np.float32)
    worker.torch = SimpleNamespace(as_tensor=probe.as_tensor, long=np.int64, float32=np.float32)
    entities, assets, maps = [], [], []
    for entity_id in range(4):
        joint_count = num_joints if entity_id < 2 else 0
        offset = entity_id * num_joints
        joints = tuple(
            SimpleNamespace(qpos_indices=(offset + j,), qvel_indices=(offset + j,))
            for j in range(joint_count)
        )
        entities.append(
            SimpleNamespace(
                joints=joints, root_mode="fixed", kind="articulation" if joint_count else "rigid"
            )
        )
        values = np.arange(num_envs * num_joints, dtype=np.float32).reshape(num_envs, num_joints)

        def write(position, velocity, *, joint_ids, env_ids, entity=entity_id):
            probe.operations.append(
                {
                    "operation": "write_joint_state",
                    "entity": entity,
                    "position": position.tolist(),
                    "velocity": velocity.tolist(),
                    "joints": list(joint_ids),
                    "rows": env_ids.tolist(),
                }
            )

        def reset(ids, entity=entity_id):
            probe.operations.append({"operation": "reset", "entity": entity, "rows": ids.tolist()})

        def update(dt, entity=entity_id):
            probe.operations.append({"operation": "update", "entity": entity, "dt": dt})

        assets.append(
            SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos=TensorDouble(values, probe),
                    joint_vel=TensorDouble(-values - 0.25, probe),
                ),
                write_joint_state_to_sim=write,
                reset=reset,
                update=update,
            )
        )
        maps.append(
            {"envs": np.arange(num_envs - 1, -1, -1), "joints": np.arange(num_joints - 1, -1, -1)}
        )
    worker.layout = SimpleNamespace(entities=tuple(entities))
    worker.assets, worker.maps = assets, maps
    ids = np.arange(rows - 1, -1, -1)
    qpos = np.zeros((rows, 2 * num_joints), dtype=np.float32)
    qvel = qpos.copy()
    qpos[:, num_joints + 3] = np.arange(rows, dtype=np.float32) + 0.75
    pmask, vmask = (
        np.zeros(2 * num_joints, dtype=np.uint8),
        np.zeros(2 * num_joints, dtype=np.uint8),
    )
    pmask[num_joints + 3] = 1
    commit(
        worker,
        ids,
        qpos,
        qvel,
        np.zeros((rows, 4, 13)),
        pmask,
        vmask,
        np.zeros((4, 2), dtype=np.uint8),
    )
    assert not worker.faulted
    return {
        "d2h_calls": len(probe.d2h),
        "d2h_bytes": sum(probe.d2h),
        "h2d_calls": len(probe.h2d),
        "h2d_bytes": sum(probe.h2d),
        "d2h_bytes_each": probe.d2h,
        "h2d_bytes_each": probe.h2d,
        "operations": probe.operations,
    }


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
