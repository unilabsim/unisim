"""Ordered body-pair contact aggregation on synthetic and real Newton data."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_body_pair_vectorized_fanout_matches_per_plan_fanout(device: str) -> None:
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("vectorized body-pair contacts require a CUDA device")
    from unisim.backend.newton.backend import NewtonBackend

    def tensor(*args, **kwargs):
        kwargs.setdefault("device", device)
        return torch.tensor(*args, **kwargs)

    backend = cast(NewtonBackend, object.__new__(NewtonBackend))
    backend._num_envs = 2
    backend._device = device
    plans = tuple(
        SimpleNamespace(name=f"pair_{index}", kind="body_pair_contact") for index in range(8)
    )
    body_a = tensor(
        [[10, 10, 90, 11, 12, 13, 14, 15], [10, 10, 90, 31, 32, 33, 34, 35]],
        dtype=torch.int64,
    )
    body_b = tensor(
        [[-1, 90, 10, 90, 90, 90, 90, 90], [90, 90, 10, 100, 100, 100, 100, 100]],
        dtype=torch.int64,
    )
    backend._tensor_bound_contact_plans = plans
    backend._tensor_contact_body_pairs = {
        str(plan.name): (body_a[:, index].clone(), body_b[:, index].clone())
        for index, plan in enumerate(plans)
    }
    backend._contact_sensor_body_pairs = {
        name: (pair[0].cpu().numpy(), pair[1].cpu().numpy())
        for name, pair in backend._tensor_contact_body_pairs.items()
    }
    backend._tensor_sensor_views = {
        str(plan.name): BodyPairContactData(
            active=torch.zeros(2, dtype=torch.bool, device=device),
            count=torch.zeros(2, dtype=torch.int32, device=device),
            position_w=torch.zeros(2, 3, dtype=torch.float32, device=device),
            force_w=torch.zeros(2, 3, dtype=torch.float32, device=device),
        )
        for plan in plans
    }
    backend._contacts = SimpleNamespace(force=torch.zeros(1, 3, device=device))
    backend._tensor_rebuild_body_pair_contact_index()

    worlds: list[torch.Tensor] = []
    bodies0: list[torch.Tensor] = []
    bodies1: list[torch.Tensor] = []
    positions0: list[torch.Tensor] = []
    positions1: list[torch.Tensor] = []
    forces0: list[torch.Tensor] = []
    for world_index in range(2):
        for plan_index, plan in enumerate(plans):
            declared_a = body_a[world_index, plan_index]
            declared_b = body_b[world_index, plan_index]
            worlds.extend((tensor(world_index), tensor(world_index)))
            bodies0.extend((declared_a.clone(), declared_b.clone()))
            bodies1.extend((declared_b.clone(), declared_a.clone()))
            positions0.extend(
                (
                    tensor([1.0, 0.0, 0.0]),
                    tensor([3.0, 0.0, 0.0]),
                )
            )
            positions1.extend(
                (
                    tensor([5.0, 0.0, 0.0]),
                    tensor([8.0, 0.0, 0.0]),
                )
            )
            forces0.extend(
                (
                    tensor([2.0, 0.0, 0.0]),
                    tensor([4.0, 0.0, 0.0]),
                )
            )
    worlds.extend((tensor(0), tensor(1)))
    bodies0.extend((body_a[0, 0].clone(), body_a[1, 0].clone()))
    bodies1.extend((body_b[0, 0].clone(), body_b[1, 0].clone()))
    positions0.extend((tensor([100.0, 0.0, 0.0]), tensor([100.0, 0.0, 0.0])))
    positions1.extend((tensor([100.0, 0.0, 0.0]), tensor([100.0, 0.0, 0.0])))
    forces0.extend((tensor([100.0, 0.0, 0.0]), tensor([100.0, 0.0, 0.0])))
    valid = torch.ones(len(worlds), dtype=torch.bool, device=device)
    valid[-2:] = False
    snapshot = {
        "world": torch.stack(worlds),
        "safe_world": torch.stack(worlds).clamp(0, 1),
        "body0": torch.stack(bodies0),
        "body1": torch.stack(bodies1),
        "valid": valid,
        "position0": torch.stack(positions0),
        "position1": torch.stack(positions1),
        "force0": torch.stack(forces0),
    }
    for key in ("position0", "position1", "force0"):
        snapshot[key][-2:] = float("nan")

    backend._tensor_refresh_vectorized_body_pair_contacts(snapshot)
    vector = {
        name: (
            view.active.clone(),
            view.count.clone(),
            view.position_w.clone(),
            view.force_w.clone(),
        )
        for name, view in backend._tensor_sensor_views.items()
    }
    for plan in plans:
        backend._tensor_refresh_body_pair_contact_sensor(plan, snapshot)
    scalar = {
        name: (
            view.active.clone(),
            view.count.clone(),
            view.position_w.clone(),
            view.force_w.clone(),
        )
        for name, view in backend._tensor_sensor_views.items()
    }

    for plan_index, plan in enumerate(plans):
        name = str(plan.name)
        values = vector[name]
        assert torch.all(values[0])
        if plan_index == 0:
            expected_count = tensor([2, 6], dtype=torch.int32, device=device)
        elif plan_index in (1, 2):
            expected_count = tensor([4, 6], dtype=torch.int32, device=device)
        else:
            expected_count = torch.full((2,), 2, dtype=torch.int32, device=device)
        torch.testing.assert_close(values[1], expected_count)
        if plan_index == 0:
            torch.testing.assert_close(values[2][0], tensor([4.5, 0.0, 0.0]))
        if plan_index == 1:
            expected_force = tensor([[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        elif plan_index == 2:
            expected_force = tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        else:
            expected_force = tensor([[-2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        torch.testing.assert_close(values[3], expected_force)
        torch.testing.assert_close(values[0], scalar[name][0])
        torch.testing.assert_close(values[1], scalar[name][1])
        torch.testing.assert_close(values[2], scalar[name][2], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(values[3], scalar[name][3], rtol=1e-6, atol=1e-7)


_BODY_PAIR_MODEL = """
<mujoco model="body-pair-contact">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.2" quat="0.9238795 0 0.3826834 0">
      <freejoint/>
      <site name="box_site" pos="0.01 0.02 0.03"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
  <sensor>
    <contact name="box_floor" body1="box" body2="world" data="found force pos"/>
    <contact name="floor_box" body1="world" body2="box" data="found force pos"/>
    <contact name="box_floor_duplicate" body1="box" body2="world" data="found force pos"/>
    <contact name="floor_box_duplicate" body1="world" body2="box" data="found force pos"/>
    <contact name="box_floor_second" body1="box" body2="world" data="found force pos"/>
    <contact name="floor_box_second" body1="world" body2="box" data="found force pos"/>
    <contact name="box_floor_third" body1="box" body2="world" data="found force pos"/>
    <contact name="floor_box_third" body1="world" body2="box" data="found force pos"/>
    <contact name="box_floor_late" body1="box" body2="world" data="found force pos"/>
    <contact name="box_floor_unbound" body1="box" body2="world" data="found force pos"/>
    <contact name="geom_box_floor" geom1="box_geom" geom2="floor" data="found"/>
    <velocimeter site="box_site" name="box_velocimeter"/>
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
    position0 = transform0[:, :3] + np_quat_apply_batched(transform0[:, (6, 3, 4, 5)], point0)
    position1 = transform1[:, :3] + np_quat_apply_batched(transform1[:, (6, 3, 4, 5)], point1)
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


@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("deferred_contact_publication", [False, True])
def test_real_cuda_body_pair_tensors_multi_point_direction_and_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_cuda_graph: bool,
    deferred_contact_publication: bool,
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
        use_cuda_graph=use_cuda_graph,
        deferred_contact_publication=deferred_contact_publication,
    )
    try:
        backend.materialize()
        graph = backend.get_tensor_runtime_diagnostics()["cuda_graph"]
        if use_cuda_graph and not graph.enabled:
            pytest.skip(f"Newton CUDA graph contacts unavailable: {graph.disable_reason}")
        empty_ctrl = backend.get_state_views(("ctrl",))["ctrl"]
        assert empty_ctrl.shape == (2, 0)
        assert empty_ctrl.dtype == torch.float32
        ctrl = torch.zeros((2, 0), dtype=torch.float32, device=str(device))
        backend.step_tensor(ctrl, nsteps=40)
        all_names = tuple(
            name
            for name in (
                "box_floor",
                "floor_box",
                "box_floor_duplicate",
                "floor_box_duplicate",
                "box_floor_second",
                "floor_box_second",
                "box_floor_third",
                "floor_box_third",
                "box_floor_late",
            )
        )
        views = {name: backend.get_sensor_view(name) for name in all_names}
        box_view = views["box_floor"]
        floor_view = views["floor_box"]
        assert isinstance(box_view, BodyPairContactData)
        assert len(backend._tensor_body_pair_contact_plans) == len(all_names)
        exact_view = backend.get_sensor_view("geom_box_floor")
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
            assert decode_calls == (0 if deferred_contact_publication else 1)
            backend.step_tensor(ctrl, nsteps=3)
            assert decode_calls == (0 if deferred_contact_publication else 2)
        assert processed_plans == []  # the vectorized path owns all body-pair plans
        assert [value.data_ptr() for value in cold_mappings] == mapping_pointers
        assert backend.get_sensor_view("box_floor") is box_view
        assert backend.get_sensor_view("floor_box") is floor_view
        assert decode_calls == (1 if deferred_contact_publication else 2)

        vectorized = {
            name: {
                field: getattr(view, field).clone()
                for field in ("active", "count", "position_w", "force_w")
            }
            for name, view in views.items()
        }
        with torch.cuda.stream(backend._tensor_contact_execution_stream):
            scalar_snapshot = backend._tensor_decode_contacts()
            for plan in backend._tensor_body_pair_contact_plans:
                backend._tensor_refresh_body_pair_contact_sensor(plan, scalar_snapshot)
        torch.cuda.current_stream(backend._tensor_device()).wait_stream(
            backend._tensor_contact_execution_stream
        )
        for name, view in views.items():
            for field in ("active", "count", "position_w", "force_w"):
                torch.testing.assert_close(
                    vectorized[name][field],
                    getattr(view, field),
                    rtol=2e-5,
                    atol=2e-6,
                )
        assert bool(exact_view.all())

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
        late_view = backend.get_sensor_view("box_floor_unbound")
        assert decode_calls == 0
        assert not bool(late_view.active.any())
        assert not bool(late_view.count.any())
        assert not bool(late_view.position_w.any())
        assert not bool(late_view.force_w.any())
        backend.step_tensor(ctrl)
        assert decode_calls == (0 if deferred_contact_publication else 1)
        backend.get_sensor_view("box_floor")
        assert decode_calls == 1
        assert bool(box_view.active[0])
        assert bool(late_view.active[0])
        assert bool(views["box_floor_late"].active[0])
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
        views = {name: backend.get_sensor_view(name) for name in ("box_floor", "floor_box")}
        backend._host_step(np.zeros((2, 0), dtype=np.float32))
        before = {
            name: {field: getattr(data, field).clone() for field in fields}
            for name, data in views.items()
        }
        host_before = {name: backend._host_read_sensor(name).copy() for name in views}
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
            with monkeypatch.context() as tensor_reset:

                def forbidden_cpu(self):
                    raise AssertionError("unexpected host transfer in tensor reset")

                tensor_reset.setattr(torch.Tensor, "cpu", forbidden_cpu)
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
            assert name not in backend._tensor_contact_host_dirty
        assert decode_calls == 0
    finally:
        backend.close()


def test_real_cuda_bind_sensor_data_mixes_dirty_contact_and_host_slices(
    tmp_path: Path,
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
        pytest.skip("Newton sensor dirty rows require a CUDA Warp device")
    model_file = tmp_path / "sensor_views.xml"
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
        for name in (
            "box_floor",
            "floor_box",
            "box_floor_duplicate",
            "floor_box_duplicate",
            "box_floor_second",
            "floor_box_second",
            "box_floor_third",
            "floor_box_third",
            "box_floor_late",
        ):
            backend.get_sensor_view(name)
        backend.get_sensor_view("geom_box_floor")
        backend._host_step(np.zeros((2, 0), dtype=np.float32), nsteps=40)
        contiguous = backend.bind_sensor_data(("box_floor", "floor_box"))
        noncontiguous = backend.bind_sensor_data(("box_floor", "box_floor_second"))
        mixed = backend.bind_sensor_data(("box_floor", "geom_box_floor", "box_velocimeter"))
        fast = contiguous.read()
        assert np.shares_memory(fast, backend._sensor_cache)
        contact_before = fast.copy()
        mixed_before = mixed.read().copy()
        assert not backend._tensor_contact_host_dirty

        state = backend.get_state_views()
        moved = state["qpos"].clone()
        moved[1, 0] = 20.0
        moved[1, 2] = 5.0
        backend.set_state_tensor(
            torch.tensor([1], dtype=torch.int64, device=state["qpos"].device),
            moved[1:2].contiguous(),
            state["qvel"][1:2].contiguous(),
        )
        dirty_after_tensor = contiguous.read()
        np.testing.assert_array_equal(dirty_after_tensor[0], contact_before[0])
        np.testing.assert_array_equal(dirty_after_tensor[1], np.zeros(14, dtype=np.float32))
        np.testing.assert_array_equal(contiguous.read(), dirty_after_tensor)

        moved[0, 0] = -20.0
        moved[0, 2] = 5.0
        backend.set_state_tensor(
            torch.tensor([0], dtype=torch.int64, device=state["qpos"].device),
            moved[:1].contiguous(),
            state["qvel"][:1].contiguous(),
        )
        mixed_after = mixed.read()
        np.testing.assert_array_equal(mixed_after[:, :7], np.zeros((2, 7), dtype=np.float32))
        np.testing.assert_array_equal(mixed_after[:, 7], np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(mixed_after[:, 8:], mixed_before[:, 8:])
        assert not {
            "box_floor",
            "geom_box_floor",
            "box_velocimeter",
        } & set(backend._tensor_contact_host_dirty)

        backend.set_state_tensor(
            torch.tensor([0], dtype=torch.int64, device=state["qpos"].device),
            moved[:1].contiguous(),
            state["qvel"][:1].contiguous(),
        )
        split = noncontiguous.read()
        np.testing.assert_array_equal(split[:, :7], np.zeros((2, 7), dtype=np.float32))
        np.testing.assert_array_equal(split[:, 7:], np.zeros((2, 7), dtype=np.float32))
        assert not np.shares_memory(split, backend._sensor_cache)
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
        assignment=(0, 1),
        object_b=_object(
            tmp_path / "variant_b",
            "object_b",
            radius=0.1,
            mass=1.5,
            com=(0.03, 0.0, 0.0),
            gravity="0 0 -9.81",
            collide=True,
        ),
        object_position=(0.0, 0.0, 0.1001),
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
        "<mujoco><sensor>"
        '<contact name="object_table" body1="object/base" '
        'body2="table/base" data="found force pos" num="1"/>'
        '<contact name="table_object" body1="table/base" '
        'body2="object/base" data="found force pos" num="1"/>'
        '<contact name="object_table_duplicate" body1="object/base" '
        'body2="table/base" data="found force pos" num="1"/>'
        '<contact name="table_object_duplicate" body1="table/base" '
        'body2="object/base" data="found force pos" num="1"/>'
        '<contact name="object_table_second" body1="object/base" '
        'body2="table/base" data="found force pos" num="1"/>'
        '<contact name="table_object_second" body1="table/base" '
        'body2="object/base" data="found force pos" num="1"/>'
        '<contact name="payload_table" body1="payload/base" body2="table/base" '
        'data="found force pos" num="1"/>'
        '<contact name="table_payload" body1="table/base" body2="payload/base" '
        'data="found force pos" num="1"/>'
        '<contact name="object_table_exact" geom1="object/object_geom" '
        'geom2="table/table_geom" data="found" num="1"/>'
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    scene.fragment_files = [str(fragment)]
    backend = _backend(scene)
    fields = ("active", "count", "position_w", "force_w")
    try:
        backend.model
        ctrl = torch.zeros((2, backend.num_actuators), dtype=torch.float32, device=backend._device)
        object_names = (
            "object_table",
            "table_object",
            "object_table_duplicate",
            "table_object_duplicate",
            "object_table_second",
            "table_object_second",
        )
        payload_names = ("payload_table", "table_payload")
        views = {name: backend.get_sensor_view(name) for name in (*object_names, *payload_names)}
        exact_view = backend.get_sensor_view("object_table_exact")
        assert len(backend._tensor_body_pair_contact_plans) == len(views)
        backend.step_tensor(ctrl)
        backend._host_step(np.zeros((2, backend.num_actuators), dtype=np.float32))
        for data in views.values():
            assert torch.all(data.active)
            assert torch.all(data.count > 0)
        assert bool(exact_view.all())
        before = {
            name: {field: getattr(data, field).clone() for field in fields}
            for name, data in views.items()
        }
        host_before = {name: backend._host_read_sensor(name).copy() for name in views}
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
        for name in object_names:
            for field in fields:
                current = getattr(views[name], field)
                assert not bool(current[1].any())
                torch.testing.assert_close(current[0], before[name][field][0])
        assert not bool(exact_view[1])
        for name in object_names:
            host = backend._host_read_sensor(name)
            np.testing.assert_array_equal(host[0], host_before[name][0])
            np.testing.assert_array_equal(host[1], np.zeros(7, dtype=np.float32))
        for name in payload_names:
            np.testing.assert_array_equal(backend._host_read_sensor(name), host_before[name])
        np.testing.assert_array_equal(backend._host_read_sensor("object_table_exact")[1], 0.0)
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
        for name in payload_names:
            assert torch.all(getattr(views[name], "active"))
        assert bool(exact_view[0])
        np.testing.assert_array_equal(object_data.position_w[1].cpu().numpy(), np.zeros(3))
        np.testing.assert_array_equal(object_data.force_w[1].cpu().numpy(), np.zeros(3))
    finally:
        backend.close()
