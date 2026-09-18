"""Public factory-to-native scene acceptance; optional vendor runtimes are explicit."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.contract.test_worker_scene_materialization import scene
from unisim import EntityStatePatch, SceneResetRequest, create_backend
from unisim.backend.isaacsim.raw_usd_cache import RawUSDCache, RoleUSDCache
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec


@pytest.mark.parametrize("backend", ["isaacgym", "isaacsim"])
def test_public_entity_factory_native_state_identity_and_reset(tmp_path: Path, backend: str):
    if os.environ.get("UNISIM_TEST_" + backend.upper() + "_SCENE") != "1":
        pytest.skip("set UNISIM_TEST_" + backend.upper() + "_SCENE=1 for vendor acceptance")
    n = 5
    config = scene(tmp_path)
    source_assignment = config.entity_variant.plan.assignment
    assignment = 1 - source_assignment
    config.entity_variant = EntityVariantBinding(
        "object",
        FixedVariantPlan(assignment, tuple(reversed(config.entity_variant.plan.variants))),
    )
    # A keyframe control deliberately differs from its joint position.
    robot_file = Path(config.entity_assets[0].source.model_file)
    robot_file.write_text(
        robot_file.read_text().replace(
            "</mujoco>", '<keyframe><key name="start" qpos=".1" ctrl=".35"/></keyframe></mujoco>'
        )
    )
    config.default_keyframe_name = "start"
    table = tmp_path / "table.xml"
    table.write_text(
        '<mujoco><worldbody><body name="base"><geom name="floor" type="box" '
        'size="2 2 .1" mass="5"/></body></worldbody></mujoco>'
    )
    config.entity_assets = (
        replace(
            config.entity_assets[0], initial_state=EntityInitialState(position=(0.0, 0.0, 0.5))
        ),
        config.entity_assets[1],
        SceneEntitySpec(
            "table",
            ModelSourceDescriptor(str(table)),
            kind="rigid",
            root_mode="fixed",
            initial_state=EntityInitialState(position=(0.0, 0.0, -0.1)),
        ),
        config.entity_assets[2],
    )
    robot_name = "robot-arm" if backend == "isaacsim" else "robot"
    config.entity_assets = (
        replace(config.entity_assets[0], name=robot_name),
        *config.entity_assets[1:],
    )
    options = {"isaacsim_worker_timeout_s": 240.0} if backend == "isaacsim" else {}
    owner = create_backend(backend, config, num_envs=n, sim_dt=0.002, **options)
    try:
        owner.materialize()
        assert owner.get_entity_names() == (robot_name, "object", "table", "target")
        assert owner.num_actuators == 1
        if backend == "isaacsim":
            import mujoco

            layout = owner.get_scene_layout()
            masses = owner.get_body_mass()
            selected = owner.get_body_ipos(env_ids=[n - 1, 0, n - 1])
            assert masses.shape == (n, layout.nbody)
            assert selected.shape == (3, layout.nbody, 3)
            canonical = owner.get_body_ipos()
            canonical_model = mujoco.MjModel.from_xml_path(owner.get_playback_model(0))
            np.testing.assert_allclose(
                canonical, canonical_model.body_ipos, rtol=1e-4, atol=1e-6
            )
            for result_row, env_index in enumerate((n - 1, 0, n - 1)):
                playback = mujoco.MjModel.from_xml_path(owner.get_playback_model(env_index))
                for entity in layout.entities:
                    ids = np.asarray(entity.body_ids)
                    np.testing.assert_allclose(
                        masses[env_index, ids],
                        playback.body_mass[ids],
                        rtol=2e-4,
                        atol=1e-6,
                    )
                    np.testing.assert_allclose(
                        selected[result_row, ids],
                        playback.body_ipos[ids],
                        rtol=1e-4,
                        atol=1e-6,
                    )
        np.testing.assert_allclose(owner.get_state("ctrl")["ctrl"], 0.35, atol=1e-6)
        initial = owner.get_state()
        np.testing.assert_allclose(
            owner.get_entity_state("object")["root_pose"][:, 2], 1.0, atol=1e-5
        )
        assert initial["qpos"].shape == (n, 8) and initial["qvel"].shape == (n, 7)
        if backend == "isaacgym":
            with pytest.raises(NotImplementedError, match="after joint reset"):
                owner.get_body_pos_w(owner.get_body_ids(["robot/tip"]))
        owner.step(np.full((n, 1), 0.6, dtype=np.float32), nsteps=3)
        before = owner.get_state()
        row = n - 1
        root = owner.get_entity_state("object")["root_pose"][[row]]
        root[:, 2] = 2.0
        owner.reset_entities(
            SceneResetRequest((row,), (EntityStatePatch("object", root_pose=root),))
        )
        after = owner.get_state()
        for key in before:
            np.testing.assert_array_equal(after[key][:row], before[key][:row])
        np.testing.assert_allclose(owner.get_state("ctrl")["ctrl"], 0.6, atol=1e-6)
        owner.step(np.full((n, 1), 0.6, dtype=np.float32), nsteps=1)
        assert owner.get_entity_state("object")["root_pose"][row, 2] > 1.99
        owner.reset(np.array([row], dtype=np.int32))
        control = owner.get_state("ctrl")["ctrl"]
        np.testing.assert_allclose(control[row], 0.35, atol=1e-6)
        np.testing.assert_allclose(control[:row], 0.6, atol=1e-6)
        owner.step(control, nsteps=1)
        assert owner.get_entity_state("object")["root_pose"][row, 2] < 1.001
        import mujoco

        playback = mujoco.MjModel.from_xml_path(owner.get_playback_model(row))
        assert playback.body("table/base").id > 0 and playback.body("target/base").id > 0
        np.testing.assert_allclose(
            playback.body("object/base").mass, (3.0, 1.0)[int(assignment[row])]
        )
        (tmp_path / "result.json").write_text(
            json.dumps(
                {
                    "backend": backend,
                    "result": "passed",
                    "assignment": assignment.tolist(),
                    "layout": owner.get_scene_layout().to_dict(),
                    "report": owner.get_import_report().to_dict(),
                },
                indent=2,
            )
        )
    finally:
        owner.close()


def test_isaacsim_native_staged_body_wrench_lifecycle(tmp_path: Path):
    if os.environ.get("UNISIM_TEST_ISAACSIM_SCENE") != "1":
        pytest.skip("set UNISIM_TEST_ISAACSIM_SCENE=1 for vendor acceptance")

    num_envs = 3
    assignment = np.arange(num_envs) % 2
    config = scene(tmp_path)
    config.entity_variant = EntityVariantBinding(
        "object", FixedVariantPlan(assignment, config.entity_variant.plan.variants)
    )
    owner = create_backend(
        "isaacsim", config, num_envs=num_envs, sim_dt=0.002, isaacsim_worker_timeout_s=240.0
    )
    try:
        owner.materialize()
        owner.reset()
        body_ids = owner.get_body_ids(["object/base"])
        masses = np.asarray([1.0, 3.0, 1.0])
        control = np.zeros((num_envs, owner.num_actuators), dtype=np.float32)

        # Two half submissions must accumulate into one gravity-compensating
        # wrench.  Resetting row zero consumes only that row before the step.
        half_hover = np.zeros((num_envs, 1, 3), dtype=np.float32)
        half_hover[:, 0, 2] = masses * 9.81 / 2.0
        owner.apply_body_force(body_ids, half_hover)
        owner.apply_body_force(body_ids, half_hover)
        owner.reset(np.array([0], dtype=np.int32))
        owner.step(control, nsteps=10)
        # The target is the rigid object's sole root body; entity root velocity is
        # the authoritative mapped-scene readback for this wrench target.
        first_velocity = owner.get_entity_state("object")["root_velocity"]
        assert first_velocity[0, 2] < -0.15
        np.testing.assert_allclose(first_velocity[1:, 2], 0.0, atol=0.005)

        # The wrench was consumed: previously compensated rows now fall freely,
        # while the already-falling control row continues from its velocity.
        owner.step(control, nsteps=10)
        idle_velocity = owner.get_entity_state("object")["root_velocity"]
        assert idle_velocity[0, 2] < -0.35
        np.testing.assert_allclose(idle_velocity[1:, 2], -0.1962, atol=0.035)

        owner.reset()
        force = np.zeros((num_envs, 1, 3), dtype=np.float32)
        force[2, 0, 2] = masses[2] * 9.81
        torque = np.zeros((num_envs, 1, 3), dtype=np.float32)
        torque[0, 0, 2] = 0.2
        torque[2, 0, 2] = -0.2
        owner.apply_body_force(body_ids, force, torque=torque)
        owner.step(control, nsteps=10)
        combined_velocity = owner.get_entity_state("object")["root_velocity"]
        np.testing.assert_allclose(combined_velocity[2, 2], 0.0, atol=0.005)
        np.testing.assert_allclose(combined_velocity[1, 2], -0.1962, atol=0.035)
        assert combined_velocity[0, 5] > 0.5
        assert combined_velocity[2, 5] < -0.5
        np.testing.assert_allclose(combined_velocity[1, 5], 0.0, atol=0.005)

        (tmp_path / "isaacsim-body-wrench.json").write_text(
            json.dumps(
                {
                    "result": "passed",
                    "commit_head": "pending-local-run",
                    "assignment": assignment.tolist(),
                    "tolerances": {
                        "hover_linear_velocity_z": 0.005,
                        "free_fall_linear_velocity_z": 0.035,
                        "idle_angular_velocity_z": 0.005,
                    },
                },
                indent=2,
            )
        )
    finally:
        owner.close()


def test_isaacsim_native_raw_and_role_usd_cache_cold_warm_semantics_and_immutability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    if os.environ.get("UNISIM_TEST_ISAACSIM_SCENE") != "1":
        pytest.skip("set UNISIM_TEST_ISAACSIM_SCENE=1 for vendor acceptance")

    cache_root = tmp_path / "raw-usd-cache"
    role_cache_root = tmp_path / "role-usd-cache"
    monkeypatch.setenv("UNISIM_ISAACSIM_RAW_USD_CACHE", str(cache_root))
    monkeypatch.setenv("UNISIM_ISAACSIM_ROLE_USD_CACHE", str(role_cache_root))
    config = scene(tmp_path)
    config.entity_variant = EntityVariantBinding(
        "object",
        FixedVariantPlan(
            np.arange(3) % 2, config.entity_variant.plan.variants
        ),
    )

    reports: list[dict] = []

    def bind(owner):
        original = owner._bind_scene_metadata

        def capture(value):
            original(value)
            if not reports:
                reports.append(value)

        owner._bind_scene_metadata = capture  # type: ignore[method-assign]

    cold_started = time.perf_counter()
    cold = create_backend(
        "isaacsim", config, num_envs=3, sim_dt=0.002, isaacsim_worker_timeout_s=240.0
    )
    cold_report: dict = {}
    cold_role_report: dict = {}
    try:
        bind(cold)
        cold.materialize()
        cold_init_s = time.perf_counter() - cold_started
        assert len(reports) == 1
        cold_report = reports[0]["raw_usd_cache"]
        cold_role_report = reports[0]["role_usd_cache"]
        assert cold_report["enabled"] is True
        assert cold_report["unique_sources"] > 0
        assert cold_report["conversions"] > 0 and cold_report["hits"] == 0
        assert cold_role_report["enabled"] is True
        assert cold_role_report["bakes"] > 0 and cold_role_report["hits"] == 0
        cold_masses = cold.get_body_mass().copy()
        cold_ipos = cold.get_body_ipos().copy()
        cold_state = {key: value.copy() for key, value in cold.get_state().items()}
        control = np.zeros((3, cold.num_actuators), dtype=np.float32)
        cold.step(control, nsteps=10)
        cold_response = {key: value.copy() for key, value in cold.get_state().items()}
    finally:
        cold.close()

    cache = RawUSDCache(cache_root)
    identities = [entry["identity"] for entry in cold_report["entries"]]
    assert len(identities) == len(set(identities))
    role_cache = RoleUSDCache(role_cache_root)
    role_identities = [entry["identity"] for entry in cold_role_report["entries"]]
    assert len(role_identities) == len(set(role_identities))
    records = [cache.load(identity) for identity in identities]
    assert all(record is not None for record in records)
    role_records = [role_cache.load(identity) for identity in role_identities]
    assert all(record is not None for record in role_records)
    cold_hashes = {
        file.path: file.sha256
        for record in records
        for file in record.files  # type: ignore[union-attr]
    }
    assert cold_hashes
    cold_role_hashes = {
        file.path: file.sha256
        for record in role_records
        for file in record.files  # type: ignore[union-attr]
    }
    assert cold_role_hashes

    reports.clear()
    warm_started = time.perf_counter()
    warm = create_backend(
        "isaacsim", config, num_envs=3, sim_dt=0.002, isaacsim_worker_timeout_s=240.0
    )
    try:
        bind(warm)
        warm.materialize()
        warm_init_s = time.perf_counter() - warm_started
        assert len(reports) == 1
        warm_report = reports[0]["raw_usd_cache"]
        warm_role_report = reports[0]["role_usd_cache"]
        assert warm_report["enabled"] is True
        assert warm_report["unique_sources"] == cold_report["unique_sources"]
        assert warm_report["hits"] == cold_report["conversions"]
        assert warm_report["conversions"] == 0
        assert [entry["identity"] for entry in warm_report["entries"]] == identities
        assert warm_role_report["enabled"] is True
        assert warm_role_report["hits"] == cold_role_report["bakes"]
        assert warm_role_report["bakes"] == 0
        assert [entry["identity"] for entry in warm_role_report["entries"]] == role_identities
        np.testing.assert_allclose(warm.get_body_mass(), cold_masses, rtol=2e-4, atol=1e-6)
        np.testing.assert_allclose(warm.get_body_ipos(), cold_ipos, rtol=1e-4, atol=1e-6)
        for key, value in warm.get_state().items():
            np.testing.assert_allclose(value, cold_state[key], rtol=1e-5, atol=1e-5)
        warm.step(np.zeros((3, warm.num_actuators), dtype=np.float32), nsteps=10)
        for key, value in warm.get_state().items():
            np.testing.assert_allclose(value, cold_response[key], rtol=1e-4, atol=2e-4)
    finally:
        warm.close()

    warm_records = [cache.load(identity) for identity in identities]
    assert all(record is not None for record in warm_records)
    warm_hashes = {
        file.path: file.sha256
        for record in warm_records
        for file in record.files  # type: ignore[union-attr]
    }
    assert warm_hashes == cold_hashes
    warm_role_records = [role_cache.load(identity) for identity in role_identities]
    assert all(record is not None for record in warm_role_records)
    warm_role_hashes = {
        file.path: file.sha256
        for record in warm_role_records
        for file in record.files  # type: ignore[union-attr]
    }
    assert warm_role_hashes == cold_role_hashes
    try:
        commit = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    runtime_versions = warm_records[0].runtime_versions  # type: ignore[union-attr]
    (tmp_path / "isaacsim-usd-caches.json").write_text(
        json.dumps(
            {
                "result": "passed",
                "commit_head": commit,
                "runtime_versions": runtime_versions,
                "cache_bytes": sum(
                    record.size_bytes for record in warm_records if record is not None
                ),
                "cold": {
                    "init_s": cold_init_s,
                    "conversions": cold_report["conversions"],
                    "hits": cold_report["hits"],
                    "materialize_ms": [
                        entry["materialize_ms"] for entry in cold_report["entries"]
                    ],
                },
                "warm": {
                    "init_s": warm_init_s,
                    "conversions": warm_report["conversions"],
                    "hits": warm_report["hits"],
                    "materialize_ms": [
                        entry["materialize_ms"] for entry in warm_report["entries"]
                    ],
                },
                "readback": {
                    "fields": ["body_mass", "body_ipos", "qpos", "qvel"],
                    "mass_tolerance": {"rtol": 2e-4, "atol": 1e-6},
                    "com_tolerance": {"rtol": 1e-4, "atol": 1e-6},
                    "state_response_tolerance": {"rtol": 1e-4, "atol": 2e-4},
                },
                "raw_immutability": "all cached file SHA-256 digests matched after warm hit",
                "role_cache_bytes": sum(
                    record.size_bytes for record in warm_role_records if record is not None
                ),
                "role": {
                    "cold": {
                        "bakes": cold_role_report["bakes"],
                        "hits": cold_role_report["hits"],
                        "materialize_ms": [
                            entry["materialize_ms"] for entry in cold_role_report["entries"]
                        ],
                    },
                    "warm": {
                        "bakes": warm_role_report["bakes"],
                        "hits": warm_role_report["hits"],
                        "materialize_ms": [
                            entry["materialize_ms"] for entry in warm_role_report["entries"]
                        ],
                    },
                    "immutability": "all cached role-file SHA-256 digests matched after warm hit",
                },
                "unverified": "memory usage and 1,200-variant scale are not measured here",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
