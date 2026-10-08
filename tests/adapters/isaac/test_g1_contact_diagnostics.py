"""Opt-in IsaacSim G1 mapped-floor contact diagnostics.

The regular mapped worker is deliberately used here: its ``step()`` path owns
PhysX ``ContactSensor`` refresh, while the CUDA-IPC worker currently rejects
contact reporting.  This is evidence collection, not numerical acceptance.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.adapters.isaac.g1_parity_harness import (
    ISAACSIM_CONTACT_SENSOR_FIELDS,
    PROFILER_ENVIRONMENT_VARIABLES,
    deterministic_control_trajectory,
    gpu_compute_process_snapshot,
    gpu_device_snapshot,
    parse_stand_fixture,
    pytest_skip_if_fixture_unavailable,
    require_acceptance_gpu_idle,
    resolve_g1_fixture_paths,
    validate_mapped_robot_source,
    write_json_report,
)

SIM_DT = 0.006666666666666667
CONTROL_SUBSTEPS = 3
CONTROL_STEP_COUNT = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mode() -> str:
    mode = os.environ.get("UNISIM_TEST_ISAACSIM_G1_CONTACT_DIAG_MODE", "diagnostic")
    if mode not in {"diagnostic", "acceptance"}:
        raise RuntimeError(
            "UNISIM_TEST_ISAACSIM_G1_CONTACT_DIAG_MODE must be diagnostic or acceptance"
        )
    return mode


def _case() -> str:
    case = os.environ.get("UNISIM_TEST_ISAACSIM_G1_CONTACT_DIAG_CASE", "zero_size_plane")
    if case not in {"zero_size_plane", "finite_plane", "finite_box"}:
        raise RuntimeError(
            "UNISIM_TEST_ISAACSIM_G1_CONTACT_DIAG_CASE must be zero_size_plane, "
            "finite_plane, or finite_box"
        )
    return case


def _mapped_scene(fixtures: Any) -> Any:
    from unisim.dr.types import ModelSourceDescriptor
    from unisim.entities import EntityInitialState, SceneEntitySpec
    from unisim.scene import SceneCfg

    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                ModelSourceDescriptor(str(fixtures.isaacsim_robot)),
                root_mode="floating",
                initial_state=EntityInitialState(
                    position=(0.0, 0.0, 0.754),
                    quaternion=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
            SceneEntitySpec(
                "floor",
                ModelSourceDescriptor(str(fixtures.isaacsim_floor)),
                kind="rigid",
                root_mode="fixed",
            ),
        ),
        fragment_files=[str(fixtures.isaacsim_contact_sensors)],
        default_keyframe_name="stand",
    )


def _case_fixtures(fixtures: Any, tmp_path: Path, case: str) -> Any:
    """Resolve a reproducible floor control without mutating checked-in sources."""

    if case == "finite_box":
        return fixtures
    source = fixtures.isaacsim_floor.read_text(encoding="utf-8")
    original = '<geom name="floor" type="box" size="0.25 0.25 0.05" pos="0 0 -0.05" mass="1"/>'
    if source.count(original) != 1:
        raise ValueError(f"mapped floor fixture no longer contains the repro geom: {source!r}")
    if case == "zero_size_plane":
        replacement = '<geom name="floor" type="plane" size="0 0 0.05"/>'
    elif case == "finite_plane":
        # MuJoCo ignores mass on a plane geom, while USD import gives the
        # finite plane its default rigid-body mass.  Author the expected fixed
        # body inertia explicitly so the diagnostic control reaches rollout.
        replacement = (
            '<inertial mass="88000" pos="0 0 0" diaginertia="1 1 1"/>'
            '<geom name="floor" type="plane" size="10 10 0.05"/>'
        )
    else:
        raise ValueError(f"unknown IsaacSim contact diagnostic case {case!r}")
    path = tmp_path / f"{case}_floor_entity.xml"
    path.write_text(source.replace(original, replacement), encoding="utf-8")
    return replace(fixtures, isaacsim_floor=path)


def test_production_floor_is_an_explicit_task_local_finite_box() -> None:
    import xml.etree.ElementTree as ET

    fixtures = resolve_g1_fixture_paths()
    pytest_skip_if_fixture_unavailable(fixtures)
    geom = ET.parse(fixtures.isaacsim_floor).find("./worldbody/body/geom")
    assert geom is not None
    assert geom.attrib == {
        "name": "floor",
        "type": "box",
        "size": "0.25 0.25 0.05",
        "pos": "0 0 -0.05",
        "mass": "1",
    }


def test_contact_controls_reproduce_legacy_plane_cases_from_the_task_local_box_floor(
    tmp_path: Path,
) -> None:
    fixtures = resolve_g1_fixture_paths()
    pytest_skip_if_fixture_unavailable(fixtures)
    assert _case_fixtures(fixtures, tmp_path, "finite_box") is fixtures
    zero = _case_fixtures(fixtures, tmp_path, "zero_size_plane").isaacsim_floor
    finite = _case_fixtures(fixtures, tmp_path, "finite_plane").isaacsim_floor
    assert '<geom name="floor" type="plane" size="0 0 0.05"/>' in zero.read_text()
    assert 'size="10 10 0.05"' in finite.read_text()


def _force_snapshot(backend: Any) -> dict[str, np.ndarray]:
    forces = {
        name: np.asarray(backend.get_sensor_data(name), dtype=np.float32).copy()
        for name in ISAACSIM_CONTACT_SENSOR_FIELDS
    }
    if any(values.shape != (backend.num_envs, 3) for values in forces.values()):
        shapes = {name: list(values.shape) for name, values in forces.items()}
        raise ValueError(f"contact force arrays must have shape {(backend.num_envs, 3)}: {shapes}")
    if any(not np.isfinite(values).all() for values in forces.values()):
        raise ValueError("contact force arrays contain NaN or Inf")
    return forces


def _force_report(forces: dict[str, np.ndarray]) -> dict[str, Any]:
    stacked = np.stack([forces[name] for name in ISAACSIM_CONTACT_SENSOR_FIELDS], axis=1)
    pair_norms = np.linalg.norm(stacked, axis=-1)
    summed = stacked.sum(axis=1)
    normal_force = stacked[..., 2]
    return {
        "forces": {name: values.tolist() for name, values in forces.items()},
        "body_pair_norms": pair_norms.tolist(),
        "active_body_pair_count_exact_nonzero_force": np.count_nonzero(
            pair_norms > 0.0, axis=1
        ).tolist(),
        "max_body_pair_norm": np.max(pair_norms, axis=1).tolist(),
        "normal_force_on_source_bodies_world": normal_force.tolist(),
        "summed_normal_force_on_source_bodies_world": normal_force.sum(axis=1).tolist(),
        "summed_force_on_source_bodies_world": summed.tolist(),
        "summed_force_on_source_bodies_world_norm": np.linalg.norm(summed, axis=-1).tolist(),
        "vertical_summed_force_on_source_bodies_world": summed[:, 2].tolist(),
        "force_semantics": {
            "on": "source_body",
            "frame": "world",
            "floor_world_up_normal": [0.0, 0.0, 1.0],
            "contact_predicate": "force_norm > 0.0",
            "aggregation": "body_pair_sum_across_collision_patches",
        },
    }


def _root_report(qpos: np.ndarray, qvel: np.ndarray) -> dict[str, Any]:
    return {
        "position": qpos[:, :3].tolist(),
        "quaternion": qpos[:, 3:7].tolist(),
        "z": qpos[:, 2].tolist(),
        "linear_velocity": qvel[:, :3].tolist(),
        "vz": qvel[:, 2].tolist(),
    }


def _geometry_audit_report(backend: Any) -> dict[str, Any]:
    """Serialize only the diagnostic floor row from the mapped-worker audit."""

    records = backend._native_entity_records
    record = records["floor"]
    if record.get("geometry_audit_schema_version") != 1:
        raise RuntimeError("mapped floor is missing geometry audit schema v1")
    names = record["geom_names"]
    bodies = record["geom_body_names"]
    masks = record["geom_contact_masks"]
    source_types = record["geom_source_types"]
    source_sizes = record["geom_source_sizes"]
    source_poses = record["geom_source_poses"]
    native_rows = record["geom_native_audits"]
    rows: list[dict[str, Any]] = []
    for env_index, assignment in enumerate(record["assignment"]):
        for geom_index, name in enumerate(names[env_index]):
            native = native_rows[env_index][geom_index]
            rows.append(
                {
                    "env_index": env_index,
                    "variant_index": int(assignment),
                    "geom_index": geom_index,
                    "geom_name": name,
                    "body_name": bodies[env_index][geom_index],
                    "source": {
                        "type": source_types[env_index][geom_index],
                        "size": source_sizes[env_index][geom_index],
                        "pose_local": source_poses[env_index][geom_index],
                        "contact_mask": masks[env_index][geom_index],
                    },
                    "body": {
                        "root_mode": "fixed",
                        "native_mass": record["body_mass"][env_index][0],
                    },
                    "native": native,
                }
            )
    materialization = backend._worker_materialization_report or {}

    def floor_entries(kind: str) -> list[dict[str, Any]]:
        entries = materialization.get(kind, {}).get("entries", ())
        return [dict(entry) for entry in entries if entry.get("entity") == "floor"]

    raw_entries = floor_entries("raw_usd_cache")
    role_entries = floor_entries("role_usd_cache")
    runtime_versions = raw_entries[0]["runtime_versions"] if raw_entries else {}
    if not raw_entries or not role_entries or not runtime_versions:
        raise RuntimeError("mapped floor is missing raw/role USD materialization provenance")
    return {
        "schema_version": 1,
        "capture_phase": "mapped_scene_materialized",
        "entity_name": "floor",
        "num_envs": len(names),
        "source_size_semantics": "mujoco_raw_columns_by_geom_type",
        "pose_convention": "x_y_z_qw_qx_qy_qz",
        "transform_convention": "usd_row_major_row_vector_affine",
        "length_units": "meters",
        "materialization": {
            "raw_usd": raw_entries,
            "role_usd": role_entries,
            "runtime_versions": runtime_versions,
        },
        "rows": rows,
    }


def test_geometry_audit_serializes_only_floor_rows_and_materialization_identity() -> None:
    native = {
        "schema_version": 1,
        "path": "/World/envs/env_0/floor/collisions/floor",
        "mesh_all_points_coincident": True,
    }
    backend = SimpleNamespace(
        _native_entity_records={
            "robot": {"geom_names": []},
            "floor": {
                "geometry_audit_schema_version": 1,
                "assignment": [0, 0],
                "body_mass": [[0.0], [0.0]],
                "geom_names": [["floor"], ["floor"]],
                "geom_body_names": [["floor"], ["floor"]],
                "geom_contact_masks": [[[1, 1]], [[1, 1]]],
                "geom_source_types": [["plane"], ["plane"]],
                "geom_source_sizes": [[[0.0, 0.0, 0.05]], [[0.0, 0.0, 0.05]]],
                "geom_source_poses": [
                    [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                ],
                "geom_native_audits": [[native], [native]],
            },
        },
        _worker_materialization_report={
            "raw_usd_cache": {
                "entries": (
                    {
                        "entity": "robot",
                        "identity": "r" * 64,
                        "runtime_versions": {"isaacsim": "robot-only"},
                    },
                    {
                        "entity": "floor",
                        "identity": "f" * 64,
                        "runtime_versions": {"isaacsim": "5.1.0.0"},
                    },
                )
            },
            "role_usd_cache": {
                "entries": ({"entity": "floor", "identity": "a" * 64, "hit": True},)
            },
        },
    )
    report = _geometry_audit_report(backend)
    assert report["schema_version"] == 1
    assert len(report["rows"]) == 2
    assert all(row["geom_name"] == "floor" for row in report["rows"])
    assert report["materialization"]["raw_usd"][0]["identity"] == "f" * 64
    assert report["materialization"]["role_usd"][0]["hit"] is True
    assert report["materialization"]["runtime_versions"] == {"isaacsim": "5.1.0.0"}


def test_g1_contact_fixture_reaches_regular_worker_payload() -> None:
    pytest.importorskip("mujoco")
    from unisim import IsaacSimBackend

    fixtures = resolve_g1_fixture_paths()
    pytest_skip_if_fixture_unavailable(fixtures)
    backend = IsaacSimBackend(
        _mapped_scene(fixtures),
        num_envs=2,
        sim_dt=SIM_DT,
        device_id=0,
        tensor_cuda_ipc=False,
    )
    try:
        records = backend._worker_init_payload()["contact_force_sensors"]
        assert tuple(record["name"] for record in records) == ISAACSIM_CONTACT_SENSOR_FIELDS
        assert all(record["source_entity"] == "robot" for record in records)
        assert all(record["target_entity"] == "floor" for record in records)
        assert all(record["target_body"] == "floor" for record in records)
        assert [record["source_body"] for record in records] == [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]
        assert backend._tensor_cuda_ipc_requested is False
    finally:
        backend.close()


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACSIM_G1_CONTACT_DIAG_NATIVE") != "1",
    reason=(
        "set UNISIM_TEST_ISAACSIM_G1_CONTACT_DIAG_NATIVE=1 for real IsaacSim contact diagnostics"
    ),
)
def test_isaacsim_g1_floor_contact_diagnostic(tmp_path: Path) -> None:
    mode = _mode()
    case = _case()
    output = os.environ.get("UNISIM_TEST_ISAACSIM_G1_CONTACT_DIAG_OUTPUT")
    if output is None:
        if mode == "acceptance":
            raise RuntimeError(
                "contact acceptance provenance requires UNISIM_TEST_ISAACSIM_G1_CONTACT_DIAG_OUTPUT"
            )
        output = str(tmp_path / f"isaacsim-g1-{case}-contact-diagnostic.json")

    profiler_environment = {
        name: os.environ.get(name, "") for name in PROFILER_ENVIRONMENT_VARIABLES
    }
    if mode == "acceptance" and any(profiler_environment.values()):
        raise RuntimeError("contact acceptance diagnostics must not enable Isaac worker profilers")

    fixtures = resolve_g1_fixture_paths()
    pytest_skip_if_fixture_unavailable(fixtures)
    fixtures = _case_fixtures(fixtures, tmp_path, case)
    stand_qpos, stand_ctrl = parse_stand_fixture(fixtures.canonical_scene, fixtures.isaacsim_robot)
    mapped_topology = validate_mapped_robot_source(
        fixtures.canonical_scene, fixtures.isaacsim_robot
    )
    gpu = {
        "required_idle": mode == "acceptance",
        "compute_processes_before": (
            require_acceptance_gpu_idle(0)
            if mode == "acceptance"
            else gpu_compute_process_snapshot(0)
        ),
        "own_process_pid": os.getpid(),
    }
    output_path = Path(output)
    backend_module = Path(__import__("unisim.backend.isaacsim.backend", fromlist=[""]).__file__)
    scene_worker_module = Path(
        __import__("unisim.backend.isaacsim.scene_worker", fromlist=[""]).__file__
    )
    scene_materialization_module = Path(
        __import__("unisim.backend.subprocess_ipc.scene_materialization", fromlist=[""]).__file__
    )
    raw_usd_cache_module = Path(
        __import__("unisim.backend.isaacsim.raw_usd_cache", fromlist=[""]).__file__
    )
    from unisim import IsaacSimBackend

    backend = IsaacSimBackend(
        _mapped_scene(fixtures),
        num_envs=2,
        sim_dt=SIM_DT,
        device_id=0,
        worker_timeout_s=120.0,
        tensor_cuda_ipc=False,
    )
    contact_sensor_body_pairs = backend._contact_force_sensor_payload()
    report: dict[str, Any] = {
        "schema_version": 4,
        "mode": mode,
        "backend": "isaacsim",
        "worker_path": "regular-mapped",
        "floor_case": case,
        "tensor_cuda_ipc": False,
        "data_plane": "numpy_shared_memory",
        "implementation": {
            "class": type(backend).__name__,
            "sources": {
                "backend": {
                    "path": str(backend_module),
                    "sha256": _sha256(backend_module),
                },
                "scene_worker": {
                    "path": str(scene_worker_module),
                    "sha256": _sha256(scene_worker_module),
                },
                "scene_materialization": {
                    "path": str(scene_materialization_module),
                    "sha256": _sha256(scene_materialization_module),
                },
                "raw_usd_cache": {
                    "path": str(raw_usd_cache_module),
                    "sha256": _sha256(raw_usd_cache_module),
                },
            },
        },
        "diagnostic_harness": {
            "path": str(Path(__file__)),
            "sha256": _sha256(Path(__file__)),
        },
        "num_envs": 2,
        "sim_dt": SIM_DT,
        "control_substeps": CONTROL_SUBSTEPS,
        "control_step_count": CONTROL_STEP_COUNT,
        "fixtures": fixtures.report(),
        "mapped_robot_topology": mapped_topology,
        "stand_qpos": stand_qpos.tolist(),
        "stand_ctrl": stand_ctrl.tolist(),
        "gpu": gpu,
        "gpu_device": gpu_device_snapshot(0),
        "profiler_environment": profiler_environment,
        "contact_sensor_names": list(ISAACSIM_CONTACT_SENSOR_FIELDS),
        "contact_sensor_body_pairs": contact_sensor_body_pairs,
        "assertions": {
            "shape": None,
            "finite": None,
            "nonzero_contact_observed": False,
            "status": "pending",
            "note": (
                "zero contact is valid diagnostic evidence for a mapped-floor failure; "
                "do not relax parity thresholds based on this artifact alone"
            ),
        },
    }
    controls = deterministic_control_trajectory(stand_ctrl, num_envs=2, steps=CONTROL_STEP_COUNT)
    full_qpos = np.repeat(stand_qpos[None, :], 2, axis=0)
    full_qvel = np.zeros((2, stand_qpos.size - 1), dtype=np.float32)

    try:
        backend.set_state(np.arange(2, dtype=np.int64), full_qpos, full_qvel)
        backend.get_geom_contact_masks()
        report["geometry_audit"] = _geometry_audit_report(backend)
        reset_state = backend.get_state(("qpos", "qvel"))
        reset_qpos = np.asarray(reset_state["qpos"], dtype=np.float32).copy()
        reset_qvel = np.asarray(reset_state["qvel"], dtype=np.float32).copy()
        np.testing.assert_allclose(reset_qpos, full_qpos, atol=2e-5, err_msg="stand reset qpos")
        reset_forces = _force_snapshot(backend)
        report["reset"] = {
            "root": _root_report(reset_qpos, reset_qvel),
            "contact": _force_report(reset_forces),
        }

        steps: list[dict[str, Any]] = []
        for index, control in enumerate(controls):
            backend.step(control, nsteps=CONTROL_SUBSTEPS)
            state = backend.get_state(("qpos", "qvel"))
            qpos = np.asarray(state["qpos"], dtype=np.float32).copy()
            qvel = np.asarray(state["qvel"], dtype=np.float32).copy()
            if qpos.shape != full_qpos.shape or qvel.shape != full_qvel.shape:
                raise ValueError(
                    f"step {index} state shapes changed: qpos={qpos.shape}, qvel={qvel.shape}"
                )
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                raise ValueError(f"step {index} state contains NaN or Inf")
            forces = _force_snapshot(backend)
            steps.append(
                {
                    "index": index,
                    "control": control.tolist(),
                    "root": _root_report(qpos, qvel),
                    "contact": _force_report(forces),
                }
            )
        report["control_steps"] = steps
        report["assertions"].update(
            {
                "shape": True,
                "finite": True,
                "nonzero_contact_observed": any(
                    any(active_count)
                    for active_count in (
                        step["contact"]["active_body_pair_count_exact_nonzero_force"]
                        for step in steps
                    )
                ),
                "status": "completed",
            }
        )
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        backend.close()
        write_json_report(output_path, report)
        raise
    else:
        backend.close()

    # Preserve the complete rollout even if a post-run provenance check rejects.
    write_json_report(output_path, report)
    if mode == "acceptance":
        report["gpu"]["compute_processes_after"] = require_acceptance_gpu_idle(
            0, quiesce_timeout_s=10.0, allowed_pids={os.getpid()}
        )
        write_json_report(output_path, report)
