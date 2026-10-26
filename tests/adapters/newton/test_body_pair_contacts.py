"""Ordered body-pair contact aggregation on synthetic and real Newton data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim import BodyPairContactData
from unisim.backend.newton.materialization import compute_body_pair_contact_aggregates


def test_body_pair_aggregates_multiple_points_direction_and_world_order() -> None:
    count, position, force = compute_body_pair_contact_aggregates(
        contact_world0=np.array([0, -1, 0, 0, 1, -1]),
        contact_world1=np.array([-1, 0, 0, 1, 1, -1]),
        contact_body0=np.array([10, -1, 99, 20, 20, 20]),
        contact_body1=np.array([-1, 10, 10, 10, 30, 30]),
        contact_position0_w=np.array(
            [
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [100.0, 0.0, 0.0],
            ]
        ),
        contact_position1_w=np.array(
            [
                [3.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [100.0, 0.0, 0.0],
            ]
        ),
        contact_force0_w=np.array(
            [
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [100.0, 0.0, 0.0],
            ]
        ),
        body_a=np.array([10, 20]),
        body_b=np.array([-1, 30]),
    )

    assert count.dtype == np.int32
    assert position.dtype == force.dtype == np.float32
    np.testing.assert_array_equal(count, [2, 1])
    np.testing.assert_allclose(position, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(force, [[-2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])


def test_body_pair_empty_contacts_zero_every_aggregate() -> None:
    empty_i = np.zeros(0, dtype=np.int64)
    empty_v = np.zeros((0, 3), dtype=np.float32)
    count, position, force = compute_body_pair_contact_aggregates(
        empty_i,
        empty_i,
        empty_i,
        empty_i,
        empty_v,
        empty_v,
        empty_v,
        np.array([10]),
        np.array([-1]),
    )
    np.testing.assert_array_equal(count, [0])
    np.testing.assert_array_equal(position, np.zeros((1, 3)))
    np.testing.assert_array_equal(force, np.zeros((1, 3)))


_BODY_PAIR_MODEL = """
<mujoco model="body-pair-contact">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.2" quat="0.9238795 0 0.3826834 0">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
  <sensor>
    <contact name="box_floor" body1="box" body2="world" data="found force pos"/>
    <contact name="floor_box" body1="world" body2="box" data="found force pos"/>
    <contact name="box_floor_late" body1="box" body2="world" data="found force pos"/>
  </sensor>
</mujoco>
"""


def _expected_first_body_positions(backend, name: str) -> np.ndarray:
    from unisim.utils.rotation import np_quat_apply_batched

    count = int(np.asarray(backend._contacts.rigid_contact_count.numpy()).max())
    shape0 = np.asarray(backend._contacts.rigid_contact_shape0.numpy()[:count])
    shape1 = np.asarray(backend._contacts.rigid_contact_shape1.numpy()[:count])
    body0 = backend._shape_body[shape0]
    body1 = backend._shape_body[shape1]
    body_q = np.asarray(backend._state.body_q.numpy(), dtype=np.float32)
    transform0 = body_q[np.maximum(body0, 0)]
    transform1 = body_q[np.maximum(body1, 0)]
    point0 = np.asarray(backend._contacts.rigid_contact_point0.numpy()[:count])
    point1 = np.asarray(backend._contacts.rigid_contact_point1.numpy()[:count])
    position0 = transform0[:, :3] + np_quat_apply_batched(
        transform0[:, (6, 3, 4, 5)], point0
    )
    position1 = transform1[:, :3] + np_quat_apply_batched(
        transform1[:, (6, 3, 4, 5)], point1
    )
    position0[body0 < 0] = point0[body0 < 0]
    position1[body1 < 0] = point1[body1 < 0]

    body_a, body_b = backend._contact_sensor_body_pairs[name]
    world0 = backend._shape_world[shape0]
    world1 = backend._shape_world[shape1]
    expected = np.zeros((backend.num_envs, 3), dtype=np.float32)
    hits = np.zeros((backend.num_envs,), dtype=np.int32)
    for index in range(count):
        world = max(int(world0[index]), int(world1[index]))
        if world < 0 or world >= backend.num_envs:
            continue
        if int(world0[index]) not in (-1, world) or int(world1[index]) not in (-1, world):
            continue
        pair = (int(body0[index]), int(body1[index]))
        if pair == (int(body_a[world]), int(body_b[world])):
            point = position0[index]
        elif pair == (int(body_b[world]), int(body_a[world])):
            point = position1[index]
        else:
            continue
        expected[world] += point
        hits[world] += 1
    return expected / np.maximum(hits, 1)[:, None]


def test_real_cuda_body_pair_tensors_multi_point_direction_and_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from unisim.backend.newton.backend import NewtonBackend
    from unisim.backend.newton.dependencies import (
        NewtonDependencyError,
        load_newton_dependencies,
    )
    from unisim.scene import SceneCfg

    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton body-pair contacts require a CUDA Warp device")
    model_file = tmp_path / "body_pair.xml"
    model_file.write_text(_BODY_PAIR_MODEL, encoding="utf-8")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=2,
        sim_dt=0.005,
        device=str(device),
        capacity_check_steps=1,
    )
    try:
        backend.materialize()
        empty_ctrl = backend.get_state_views(("ctrl",))["ctrl"]
        assert empty_ctrl.shape == (2, 0)
        assert empty_ctrl.dtype == torch.float32
        ctrl = torch.zeros((2, 0), dtype=torch.float32, device=str(device))
        backend.step_tensor(ctrl, nsteps=40)
        box_view = backend.get_sensor_view("box_floor")
        floor_view = backend.get_sensor_view("floor_box")
        assert isinstance(box_view, BodyPairContactData)
        for data in (box_view, floor_view):
            assert data.active.shape == data.count.shape == (2,)
            assert data.position_w.shape == data.force_w.shape == (2, 3)
            assert data.active.dtype == torch.bool
            assert data.count.dtype == torch.int32
            assert data.position_w.dtype == data.force_w.dtype == torch.float32
            assert torch.all(data.active)
            assert torch.all(data.count > 1)
            assert bool(torch.isfinite(data.position_w).all())
            assert bool(torch.isfinite(data.force_w).all())
        assert torch.all(box_view.force_w[:, 2] > 0.0)
        np.testing.assert_allclose(
            floor_view.force_w.cpu().numpy(), -box_view.force_w.cpu().numpy(), rtol=1e-5, atol=1e-5
        )
        box_position = box_view.position_w.cpu().numpy()
        floor_position = floor_view.position_w.cpu().numpy()
        assert not np.allclose(box_position, floor_position, atol=1e-6)
        assert not np.allclose(box_position, (box_position + floor_position) * 0.5, atol=1e-6)

        cold_mappings = (
            backend._tensor_contact_shape_world,
            backend._tensor_contact_shape_body,
            backend._tensor_contact_ordinals,
            *(value for pair in backend._tensor_contact_body_pairs.values() for value in pair),
        )
        mapping_pointers = [value.data_ptr() for value in cold_mappings]
        processed_plans: list[str] = []
        original_refresh = backend._tensor_refresh_body_pair_contact_sensor

        def record_refresh(plan, snapshot) -> None:
            processed_plans.append(str(plan.name))
            original_refresh(plan, snapshot)

        monkeypatch.setattr(backend, "_tensor_refresh_body_pair_contact_sensor", record_refresh)
        decode_calls = 0
        original_update = backend._solver.update_contacts

        def counted_update(contacts) -> None:
            nonlocal decode_calls
            decode_calls += 1
            original_update(contacts)

        def forbidden_hot_path_call(*args, **kwargs) -> None:
            raise AssertionError("unexpected scalar readback in tensor step")

        original_as_tensor = torch.as_tensor

        def checked_as_tensor(value, *args, **kwargs):
            if isinstance(value, np.ndarray):
                raise AssertionError("unexpected host-to-device mapping upload in tensor step")
            return original_as_tensor(value, *args, **kwargs)

        monkeypatch.setattr(backend._solver, "update_contacts", counted_update)
        with monkeypatch.context() as hot_path:
            hot_path.setattr(torch.Tensor, "item", forbidden_hot_path_call)
            hot_path.setattr(torch, "as_tensor", checked_as_tensor)
            backend.step_tensor(ctrl)
            assert decode_calls == 1
            backend.step_tensor(ctrl, nsteps=3)
            assert decode_calls == 2
        assert processed_plans == ["box_floor", "floor_box"] * 2
        assert [value.data_ptr() for value in cold_mappings] == mapping_pointers
        assert backend.get_sensor_view("box_floor") is box_view
        assert backend.get_sensor_view("floor_box") is floor_view

        backend._host_step(np.zeros((2, 0), dtype=np.float32))
        for name, data in (("box_floor", box_view), ("floor_box", floor_view)):
            expected = _expected_first_body_positions(backend, name)
            np.testing.assert_allclose(
                backend._host_read_sensor(name)[:, 4:7], expected, rtol=1e-5, atol=2e-6
            )
            np.testing.assert_allclose(
                data.position_w.cpu().numpy(), expected, rtol=1e-5, atol=2e-6
            )

        snapshot = backend.get_physics_state()
        snapshot[1, 1:4] = np.array([20.0, 0.0, 5.0], dtype=np.float32)
        decode_calls = 0
        backend.set_physics_state(snapshot)
        assert decode_calls == 0
        for data in (box_view, floor_view):
            assert not bool(data.active.any())
            assert not bool(data.count.any())
            assert not bool(data.position_w.any())
            assert not bool(data.force_w.any())
        late_view = backend.get_sensor_view("box_floor_late")
        assert decode_calls == 0
        assert not bool(late_view.active.any())
        assert not bool(late_view.count.any())
        assert not bool(late_view.position_w.any())
        assert not bool(late_view.force_w.any())
        backend.step_tensor(ctrl)
        assert decode_calls == 1
        assert bool(box_view.active[0])
        assert bool(late_view.active[0])
        assert not bool(box_view.active[1])
        assert int(box_view.count[1]) == 0
        np.testing.assert_array_equal(box_view.position_w[1].cpu().numpy(), np.zeros(3))
        np.testing.assert_array_equal(box_view.force_w[1].cpu().numpy(), np.zeros(3))
    finally:
        backend.close()


@pytest.mark.parametrize("reset_mode", ["tensor", "host"])
def test_real_cuda_selected_reset_preserves_unselected_contact_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_mode: str
) -> None:
    torch = pytest.importorskip("torch")
    from unisim.backend.newton.backend import NewtonBackend
    from unisim.backend.newton.dependencies import (
        NewtonDependencyError,
        load_newton_dependencies,
    )
    from unisim.scene import SceneCfg

    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton selected contact reset requires a CUDA Warp device")
    model_file = tmp_path / "selected_contact_reset.xml"
    model_file.write_text(_BODY_PAIR_MODEL, encoding="utf-8")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=2,
        sim_dt=0.005,
        device=str(device),
        capacity_check_steps=1,
    )
    fields = ("active", "count", "position_w", "force_w")
    try:
        backend.materialize()
        ctrl = torch.zeros((2, 0), dtype=torch.float32, device=str(device))
        backend.step_tensor(ctrl, nsteps=40)
        views = {
            name: backend.get_sensor_view(name) for name in ("box_floor", "floor_box")
        }
        backend._host_step(np.zeros((2, 0), dtype=np.float32))
        before = {
            name: {field: getattr(data, field).clone() for field in fields}
            for name, data in views.items()
        }
        host_before = {
            name: backend._host_read_sensor(name).copy() for name in views
        }
        state = backend.get_state_views()
        moved_qpos = state["qpos"][1:2].clone()
        moved_qpos[0, 0] = 20.0
        moved_qpos[0, 2] = 5.0
        selected_qvel = state["qvel"][1:2].contiguous()
        decode_calls = 0
        original_update = backend._solver.update_contacts

        def counted_update(contacts) -> None:
            nonlocal decode_calls
            decode_calls += 1
            original_update(contacts)

        monkeypatch.setattr(backend._solver, "update_contacts", counted_update)
        if reset_mode == "tensor":
            backend.set_state_tensor(
                torch.tensor([1], dtype=torch.int64, device=backend._device),
                moved_qpos,
                selected_qvel,
            )
        else:
            backend._host_set_state(
                np.array([1], dtype=np.intp),
                moved_qpos.cpu().numpy(),
                selected_qvel.cpu().numpy(),
            )
        assert decode_calls == 0
        for name, data in views.items():
            assert backend.get_sensor_view(name) is data
            for field in fields:
                current = getattr(data, field)
                torch.testing.assert_close(current[0], before[name][field][0])
                assert not bool(current[1].any())
            host = backend._host_read_sensor(name)
            np.testing.assert_array_equal(host[0], host_before[name][0])
            np.testing.assert_array_equal(host[1], np.zeros(7, dtype=np.float32))
        assert decode_calls == 0
    finally:
        backend.close()


def test_real_cuda_portable_body_pair_tracks_entity_and_env_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from unisim.entities import (
        EntityInitialState,
        EntityStatePatch,
        SceneEntitySpec,
        SceneResetRequest,
    )

    from .test_multi_entity_feasibility import _backend, _object, _scene

    scene = _scene(
        tmp_path,
        assignment=(1, 1),
        object_position=(0.0, 0.0, 0.1501),
        table_position=(0.0, 0.0, 0.0),
        gravity="0 0 -9.81",
        collide=True,
    )
    scene.entity_assets = (
        *scene.entity_assets[:2],
        SceneEntitySpec(
            "payload",
            _object(
                tmp_path,
                "payload",
                radius=0.15,
                mass=0.3,
                com=(0.0, 0.0, 0.0),
                gravity="0 0 -9.81",
                collide=True,
            ),
            initial_state=EntityInitialState((-0.5, 0.0, 0.1501)),
        ),
        scene.entity_assets[2],
    )
    fragment = tmp_path / "body_pair_sensors.xml"
    fragment.write_text(
        '<mujoco><sensor><contact name="object_table" body1="object/base" '
        'body2="table/base" data="found force pos" num="1"/>'
        '<contact name="payload_table" body1="payload/base" body2="table/base" '
        'data="found force pos" num="1"/></sensor></mujoco>',
        encoding="utf-8",
    )
    scene.fragment_files = [str(fragment)]
    backend = _backend(scene)
    fields = ("active", "count", "position_w", "force_w")
    try:
        backend.model
        ctrl = torch.zeros((2, backend.num_actuators), dtype=torch.float32, device=backend._device)
        backend.step_tensor(ctrl)
        views = {
            name: backend.get_sensor_view(name) for name in ("object_table", "payload_table")
        }
        backend._host_step(np.zeros((2, backend.num_actuators), dtype=np.float32))
        for data in views.values():
            assert torch.all(data.active)
            assert torch.all(data.count > 0)
        before = {
            name: {field: getattr(data, field).clone() for field in fields}
            for name, data in views.items()
        }
        host_before = {
            name: backend._host_read_sensor(name).copy() for name in views
        }
        decode_calls = 0
        original_update = backend._solver.update_contacts

        def counted_update(contacts) -> None:
            nonlocal decode_calls
            decode_calls += 1
            original_update(contacts)

        monkeypatch.setattr(backend._solver, "update_contacts", counted_update)
        backend.reset_entities(
            SceneResetRequest(
                (1,),
                (
                    EntityStatePatch(
                        "object",
                        root_pose=np.array(
                            [[100.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32
                        ),
                    ),
                ),
            )
        )
        assert decode_calls == 0
        object_data = views["object_table"]
        payload_data = views["payload_table"]
        assert backend.get_sensor_view("object_table") is object_data
        assert backend.get_sensor_view("payload_table") is payload_data
        for field in fields:
            object_current = getattr(object_data, field)
            payload_current = getattr(payload_data, field)
            torch.testing.assert_close(object_current[0], before["object_table"][field][0])
            assert not bool(object_current[1].any())
            torch.testing.assert_close(payload_current, before["payload_table"][field])
        object_host = backend._host_read_sensor("object_table")
        payload_host = backend._host_read_sensor("payload_table")
        np.testing.assert_array_equal(object_host[0], host_before["object_table"][0])
        np.testing.assert_array_equal(object_host[1], np.zeros(7, dtype=np.float32))
        np.testing.assert_array_equal(payload_host, host_before["payload_table"])
        assert decode_calls == 0

        backend.step_tensor(ctrl)
        assert decode_calls == 1
        assert bool(object_data.active[0])
        assert not bool(object_data.active[1])
        assert int(object_data.count[1]) == 0
        assert torch.all(payload_data.active)
        np.testing.assert_array_equal(object_data.position_w[1].cpu().numpy(), np.zeros(3))
        np.testing.assert_array_equal(object_data.force_w[1].cpu().numpy(), np.zeros(3))
    finally:
        backend.close()
