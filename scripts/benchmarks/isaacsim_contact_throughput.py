"""Maintainer probe: mapped IsaacSim contact data path throughput at scale.

Measures the public step rate and worker GPU memory of a mapped scene with
the contact sensor forms added for per-body net force / contact-found queries
(body-net wildcard) and the pre-existing collision-pair reporter.  Requires a
provisioned IsaacSim runtime (resolve_isaacsim_runtime) and a CUDA GPU.

Example:
    python scripts/benchmarks/isaacsim_contact_throughput.py \
        --num-envs 2048 --links 5 --steps 200 --mode both

The probe is a maintainer benchmark only; it never selects production
behavior.  GPU memory is sampled from the worker process via nvidia-smi.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from unisim import create_backend  # noqa: E402
from unisim.dr.types import ModelSourceDescriptor  # noqa: E402
from unisim.entities import EntityInitialState, SceneEntitySpec  # noqa: E402
from unisim.scene import SceneCfg  # noqa: E402


def _worker_gpu_bytes(pid: int) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and int(parts[0]) == pid:
            return int(parts[1]) * 1024 * 1024
    return None


def _scene(directory: Path, links: int, mode: str) -> SceneCfg:
    chain = []
    for index in range(links):
        parent_pos = "0 0 0" if index == 0 else ".12 0 0"
        chain.append(
            f'<body name="link{index}" pos="{parent_pos}">'
            + (
                ""
                if index == 0
                else f'<joint name="hinge{index}" axis="0 1 0" range="-1.5 1.5"/>'
            )
            + f'<geom name="geom{index}" type="capsule" fromto="0 0 0 .1 0 0" '
            f'size=".02" mass=".1"/>'
        )
    fragment_sensors = []
    if mode in ("net", "both"):
        fragment_sensors.append(
            '<contact name="object_net" geom1="object/shape" '
            'data="force" reduce="netforce"/>'
            '<contact name="object_touch" geom1="object/shape" data="found" num="1"/>'
        )
    if mode in ("pair", "both"):
        fragment_sensors.append(
            '<contact name="object_table" geom1="object/shape" '
            'geom2="table/surface" data="force" reduce="netforce"/>'
        )
    robot_xml = (
        "<mujoco><worldbody>"
        + "".join(chain)
        + "</body>" * (links - 1)
        + "</body></worldbody>"
        + (
            "<actuator>"
            + "".join(
                f'<position name="drive{index}" joint="hinge{index}" kp="20" kv="2"/>'
                for index in range(1, links)
            )
            + "</actuator>"
            if links > 1
            else ""
        )
        + "</mujoco>"
    )
    robot = directory / "robot.xml"
    robot.write_text(robot_xml, encoding="utf-8")
    box = directory / "object.xml"
    box.write_text(
        '<mujoco><worldbody><body name="base">'
        "<freejoint/>"
        '<inertial pos="0 0 0" mass="1" diaginertia=".002 .002 .002"/>'
        '<geom name="shape" type="box" size=".05 .05 .05"/>'
        "</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    table = directory / "table.xml"
    table.write_text(
        '<mujoco><worldbody><body name="base">'
        '<inertial pos="0 0 0" mass="5" diaginertia=".5 .5 .1"/>'
        '<geom name="surface" type="box" size="2 2 .05"/>'
        "</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    fragment = directory / "sensors.xml"
    fragment.write_text(
        "<mujoco><sensor>" + "".join(fragment_sensors) + "</sensor></mujoco>",
        encoding="utf-8",
    )
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                ModelSourceDescriptor(str(robot)),
                root_mode="fixed",
                # The driven chain hovers above the table; it exists to give
                # each environment a realistic articulation step cost.
                initial_state=EntityInitialState(
                    position=(0.12 * (links - 1) + 0.1, 0.0, 0.07)
                ),
            ),
            SceneEntitySpec(
                "object",
                ModelSourceDescriptor(str(box)),
                kind="rigid",
                # Rests on the tabletop so the contact data path carries real
                # forces every step.
                initial_state=EntityInitialState(position=(0.15, 0.0, 0.05)),
            ),
            SceneEntitySpec(
                "table",
                ModelSourceDescriptor(str(table)),
                kind="rigid",
                root_mode="fixed",
                initial_state=EntityInitialState(position=(0.0, 0.0, -0.05)),
            ),
        ),
        fragment_files=[str(fragment)] if fragment_sensors else [],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--links", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--mode", choices=("none", "net", "pair", "both"), default="both"
    )
    args = parser.parse_args()
    if args.links < 1:
        raise ValueError("--links must be positive")

    with tempfile.TemporaryDirectory(prefix="unisim-contact-bench-") as tmp:
        config = _scene(Path(tmp), args.links, args.mode)
        backend = create_backend(
            "isaacsim",
            config,
            num_envs=args.num_envs,
            sim_dt=0.005,
            isaacsim_worker_timeout_s=900.0,
        )
        try:
            started = time.perf_counter()
            backend.materialize()
            materialize_s = time.perf_counter() - started
            backend.reset()
            ctrl = np.zeros((args.num_envs, backend.num_actuators), dtype=np.float32)
            for _ in range(args.warmup):
                backend.step(ctrl, nsteps=1)
            pid = backend._proc.pid if backend._proc is not None else -1
            gpu_bytes = _worker_gpu_bytes(pid)
            started = time.perf_counter()
            for _ in range(args.steps):
                backend.step(ctrl, nsteps=1)
            elapsed = time.perf_counter() - started
            # Exercise the observation reads as a consumer would each step.
            report = {
                "mode": args.mode,
                "num_envs": args.num_envs,
                "links": args.links,
                "steps": args.steps,
                "materialize_s": round(materialize_s, 2),
                "ms_per_step": round(elapsed / args.steps * 1000.0, 3),
                "env_steps_per_s": round(args.num_envs * args.steps / elapsed),
                "worker_gpu_mib": (
                    None if gpu_bytes is None else round(gpu_bytes / 1024 / 1024)
                ),
            }
            if args.mode in ("net", "both"):
                net = backend.get_sensor_data("object_net")
                touch = backend.get_sensor_data("object_touch")
                report["net_force_shape"] = list(net.shape)
                report["touch_shape"] = list(touch.shape)
                report["rows_in_contact"] = int(np.count_nonzero(touch))
                report["net_force_z_mean"] = float(np.mean(np.abs(net[:, 2])))
            print(json.dumps(report, indent=2))
        finally:
            backend.close()


if __name__ == "__main__":
    main()
