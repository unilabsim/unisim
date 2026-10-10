"""Demonstrate portable Motrix per-environment geometry randomization.

Run with the repository environment:

    uv run --extra motrix python examples/motrix_geom_randomization.py

The interactive window is the default. Press ``r`` to resample dimensions,
pose, and the mesh variant, ``a`` to toggle the ball for one selected
environment, ``s`` to switch between sphere and box, and ``q`` or ``esc`` to
quit. Use ``--no-interactive`` for a non-window smoke run. The keyboard
handling follows MotrixSim's
``RenderApp.input.is_key_just_pressed`` API.

The example uses a temporary MJCF source, three mesh-scale variants, and a
portable entity scene. Geometry randomization is a selected-row reset operation:
each table is ordered as
``(selected_envs, public_geoms, ...)`` and omitted fields are left unchanged.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from unisim.backend.motrix.backend import MotrixBackend
from unisim.dr.types import (
    RESET_TERM_GEOM_ACTIVE,
    RESET_TERM_GEOM_MESH_VARIANT,
    RESET_TERM_GEOM_POS,
    RESET_TERM_GEOM_QUAT,
    RESET_TERM_GEOM_SHAPE,
    RESET_TERM_GEOM_SIZE,
    FixedVariantLayout,
    FixedVariantPlan,
    ModelSourceDescriptor,
    ResetRandomizationPayload,
)
from unisim.entities import EntityVariantBinding, SceneEntitySpec
from unisim.scene import SceneCfg

TETRAHEDRON = """\
v 0 0 0
v .1 0 0
v 0 .1 0
v 0 0 .1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


def write_source(directory: Path, name: str, scale: str) -> ModelSourceDescriptor:
    mesh = directory / f"{name}.obj"
    mesh.write_text(TETRAHEDRON, encoding="utf-8")
    source = directory / f"{name}.xml"
    source.write_text(
        f"""<mujoco>
  <asset><mesh name="head" file="{mesh}" scale="{scale}"/></asset>
  <worldbody><body name="base" pos="0 0 0.5">
    <freejoint name="root"/>
    <inertial mass="1" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
    <geom name="ball" type="sphere" size="0.10" pos="0.15 0 0" mass="0"/>
    <geom name="block" type="box" size="0.08 0.12 0.16" pos="-0.15 0 0" mass="0"/>
    <geom name="mesh" type="mesh" mesh="head" pos="0 0 0.2" mass="0"/>
  </body></worldbody>
</mujoco>""",
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(source))


def make_backend(directory: Path, num_envs: int = 4) -> MotrixBackend:
    sources = tuple(write_source(directory, name, scale) for name, scale in (
        ("small", ".8 .8 .8"),
        ("medium", "1.4 1.4 1.4"),
        ("large", "2.0 2.0 2.0"),
    ))
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("object", sources[0], kind="rigid"),),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(
                np.asarray((0, 1, 2, 1), dtype=np.int32),
                sources,
                layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
            ),
        ),
    )
    return MotrixBackend(scene, num_envs, sim_dt=0.002, base_name="object/base")


def apply_randomization(
    backend: MotrixBackend,
    rows: np.ndarray,
    geom_ids: dict[str, int],
    *,
    seed: int | None,
) -> None:
    """Apply one new randomized payload to the selected rows."""
    rng = np.random.default_rng(seed)
    capabilities = backend.get_dr_capabilities().supported_reset_terms
    terms = {
        RESET_TERM_GEOM_SIZE,
        RESET_TERM_GEOM_ACTIVE,
        RESET_TERM_GEOM_POS,
        RESET_TERM_GEOM_QUAT,
        RESET_TERM_GEOM_SHAPE,
        RESET_TERM_GEOM_MESH_VARIANT,
    } & capabilities
    payload: dict[str, np.ndarray] = {
        term: backend.get_reset_term_default(term)[rows].copy() for term in terms
    }
    if RESET_TERM_GEOM_SIZE in payload:
        sizes = payload[RESET_TERM_GEOM_SIZE].astype(np.float32, copy=False)
        sizes[:, geom_ids["ball"]] = rng.uniform(.04, .12, size=(rows.size, 3))
        payload[RESET_TERM_GEOM_SIZE] = sizes
    if RESET_TERM_GEOM_ACTIVE in payload:
        active = payload[RESET_TERM_GEOM_ACTIVE].astype(bool, copy=False)
        active[:, geom_ids["block"]] = True
        payload[RESET_TERM_GEOM_ACTIVE] = active
    if RESET_TERM_GEOM_POS in payload:
        pos = payload[RESET_TERM_GEOM_POS].astype(np.float32, copy=False)
        pos[:, geom_ids["ball"]] += rng.uniform(-.08, .08, size=(rows.size, 3))
        payload[RESET_TERM_GEOM_POS] = pos
    if RESET_TERM_GEOM_QUAT in payload:
        quat = payload[RESET_TERM_GEOM_QUAT].astype(np.float32, copy=False)
        quat[:, geom_ids["ball"]] = (.7071068, 0.0, 0.0, .7071068)
        payload[RESET_TERM_GEOM_QUAT] = quat
    if RESET_TERM_GEOM_SHAPE in payload:
        shapes = payload[RESET_TERM_GEOM_SHAPE].astype("<U16", copy=False)
        shapes[:, geom_ids["ball"]] = "box"
        payload[RESET_TERM_GEOM_SHAPE] = shapes
    if RESET_TERM_GEOM_MESH_VARIANT in payload:
        variants = payload[RESET_TERM_GEOM_MESH_VARIANT].astype(np.int64, copy=False)
        variants[:, geom_ids["mesh"]] = rng.integers(0, 3, size=rows.size)
        payload[RESET_TERM_GEOM_MESH_VARIANT] = variants
    qpos = backend.get_state("qpos")["qpos"][rows].copy()
    qvel = backend.get_state("qvel")["qvel"][rows].copy()
    backend.set_state(rows, qpos, qvel, randomization=ResetRandomizationPayload(**payload))
    print(f"Applied a new randomization to environments {rows.tolist()}.")


def toggle_ball_activity(
    backend: MotrixBackend, rows: np.ndarray, geom_ids: dict[str, int]
) -> None:
    """Toggle the ball's active flag on the selected rows."""
    if RESET_TERM_GEOM_ACTIVE not in backend.get_dr_capabilities().supported_reset_terms:
        print("geom_active is unavailable in this MotrixSim runtime")
        return
    state = getattr(toggle_ball_activity, "state", True)
    payload = backend.get_reset_term_default(RESET_TERM_GEOM_ACTIVE)[rows].copy()
    payload[:, geom_ids["ball"]] = not state
    toggle_ball_activity.state = not state
    qpos = backend.get_state("qpos")["qpos"][rows].copy()
    qvel = backend.get_state("qvel")["qvel"][rows].copy()
    backend.set_state(
        rows, qpos, qvel, randomization=ResetRandomizationPayload(geom_active=payload)
    )


def toggle_ball_shape(backend: MotrixBackend, rows: np.ndarray, geom_ids: dict[str, int]) -> None:
    """Toggle the ball between a sphere and a box."""
    if RESET_TERM_GEOM_SHAPE not in backend.get_dr_capabilities().supported_reset_terms:
        print("geom_shape is unavailable in this MotrixSim runtime")
        return
    box = getattr(toggle_ball_shape, "box", True)
    payload = backend.get_reset_term_default(RESET_TERM_GEOM_SHAPE)[rows].copy()
    next_shape = "sphere" if box else "box"
    payload[:, geom_ids["ball"]] = next_shape
    toggle_ball_shape.box = not box
    randomization: dict[str, np.ndarray] = {RESET_TERM_GEOM_SHAPE: payload}
    if (
        next_shape == "box"
        and RESET_TERM_GEOM_SIZE in backend.get_dr_capabilities().supported_reset_terms
    ):
        sizes = backend.get_reset_term_default(RESET_TERM_GEOM_SIZE)[rows].copy().astype(np.float32)
        sizes[:, geom_ids["ball"]] = (.08, .08, .08)
        randomization[RESET_TERM_GEOM_SIZE] = sizes
    qpos = backend.get_state("qpos")["qpos"][rows].copy()
    qvel = backend.get_state("qvel")["qvel"][rows].copy()
    backend.set_state(rows, qpos, qvel, randomization=ResetRandomizationPayload(**randomization))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Run without opening the MotrixSim window.",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        metavar="PNG",
        help="Capture a headless RGB frame as a PPM image after applying the overrides.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="unisim-motrix-example-") as directory:
        backend = make_backend(Path(directory))
        try:
            capabilities = backend.get_dr_capabilities()
            print("MotrixSim geometry reset capabilities:")
            print("  ", sorted(capabilities.supported_reset_terms))

            requested = {
                RESET_TERM_GEOM_SIZE,
                RESET_TERM_GEOM_ACTIVE,
                RESET_TERM_GEOM_POS,
                RESET_TERM_GEOM_QUAT,
                RESET_TERM_GEOM_SHAPE,
                RESET_TERM_GEOM_MESH_VARIANT,
            }
            supported = requested & capabilities.supported_reset_terms
            print("Supported terms in this runtime:", sorted(supported))

            # Only request fields supported by the installed MotrixSim runtime.
            # 0.10.2.dev126386 supports active and pose overrides; older wheels
            # fail closed for those terms while retaining primitive size support.
            rows = np.asarray([1, 3], dtype=np.intp)
            defaults = {
                term: backend.get_reset_term_default(term)[rows].copy()
                for term in supported
            }
            geom_ids = {
                "ball": backend.get_geom_id("object/ball"),
                "block": backend.get_geom_id("object/block"),
                "mesh": backend.get_geom_id("object/mesh"),
            }

            if RESET_TERM_GEOM_SIZE in supported:
                sizes = defaults[RESET_TERM_GEOM_SIZE].astype(np.float32, copy=False)
                sizes[:, geom_ids["ball"]] = (.04, .05, .06)
                defaults[RESET_TERM_GEOM_SIZE] = sizes

            if RESET_TERM_GEOM_ACTIVE in supported:
                active = defaults[RESET_TERM_GEOM_ACTIVE].astype(bool, copy=False)
                active[0, geom_ids["block"]] = False
                defaults[RESET_TERM_GEOM_ACTIVE] = active

            if RESET_TERM_GEOM_POS in supported:
                positions = defaults[RESET_TERM_GEOM_POS].astype(np.float32, copy=False)
                positions[:, geom_ids["ball"]] += (.02, .04, .03)
                defaults[RESET_TERM_GEOM_POS] = positions

            if RESET_TERM_GEOM_QUAT in supported:
                quaternions = defaults[RESET_TERM_GEOM_QUAT].astype(np.float32, copy=False)
                # Public quaternion order is wxyz.  Rotate the ball 90 degrees about Z.
                quaternions[:, geom_ids["ball"]] = (.7071068, 0.0, 0.0, .7071068)
                defaults[RESET_TERM_GEOM_QUAT] = quaternions

            if RESET_TERM_GEOM_SHAPE in supported:
                shapes = defaults[RESET_TERM_GEOM_SHAPE].astype("<U16", copy=False)
                shapes[:, geom_ids["ball"]] = "box"
                defaults[RESET_TERM_GEOM_SHAPE] = shapes

            if RESET_TERM_GEOM_MESH_VARIANT in supported:
                variants = defaults[RESET_TERM_GEOM_MESH_VARIANT].astype(np.int64, copy=False)
                variants[:, geom_ids["mesh"]] = np.asarray([0, 2], dtype=np.int64)
                defaults[RESET_TERM_GEOM_MESH_VARIANT] = variants

            qpos = backend.get_state("qpos")["qpos"][rows].copy()
            qvel = backend.get_state("qvel")["qvel"][rows].copy()
            backend.set_state(
                rows,
                qpos,
                qvel,
                randomization=ResetRandomizationPayload(**defaults),
            )

            print(f"Applied selected-row randomization to environments {rows.tolist()}.")
            for term in sorted(defaults):
                print(f"  {term}: shape={defaults[term].shape}")

            # The selected-row state remains valid after a normal physics step.
            backend.step(np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32))
            print("Geometry overrides remain active after one physics step.")

            if args.capture is not None:
                backend.init_renderer(
                    spacing=0.8,
                    headless=True,
                    capture=True,
                    width=960,
                    height=540,
                    camera_kwargs={
                        "cam_tracking": True,
                        "cam_tracking_env_idx": int(rows[0]),
                        "cam_distance": 2.0,
                    },
                )
                frame = backend.capture_video_frame()
                if args.capture.suffix.lower() != ".ppm":
                    raise ValueError("--capture currently requires a .ppm output path")
                height, width, channels = frame.shape
                if channels != 3:
                    raise RuntimeError(f"expected RGB frame, got shape {frame.shape}")
                with args.capture.open("wb") as output:
                    output.write(f"P6\\n{width} {height}\\n255\\n".encode("ascii"))
                    output.write(np.ascontiguousarray(frame).tobytes())
                print(f"Saved visualization to {args.capture}")

            if not args.no_interactive and args.capture is None:
                backend.init_renderer(
                    spacing=0.8,
                    headless=False,
                    capture=False,
                    camera_kwargs={
                        "cam_tracking": True,
                        "cam_tracking_env_idx": int(rows[0]),
                        "cam_distance": 2.0,
                    },
                )
                print(
                    "Interactive renderer is open. Press r= randomize, "
                    "a= toggle ball, s= switch shape, q/esc= quit."
                )
                while True:
                    backend.render()
                    if backend._render_app is not None:
                        input_events = backend._render_app.input
                        quit_pressed = input_events.is_key_just_pressed("q") or (
                            input_events.is_key_just_pressed("esc")
                        )
                        if quit_pressed:
                            break
                        if input_events.is_key_just_pressed("r"):
                            apply_randomization(backend, rows, geom_ids, seed=None)
                        if input_events.is_key_just_pressed("a"):
                            toggle_ball_activity(backend, rows, geom_ids)
                        if input_events.is_key_just_pressed("s"):
                            toggle_ball_shape(backend, rows, geom_ids)
                    time.sleep(1.0 / 30.0)
        finally:
            backend.close()


if __name__ == "__main__":
    main()
