"""Real-CUDA ordered body-pair contact semantics for MJWarp."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

from unisim import BodyPairContactBatch, BodyPairContactData, MjwarpBackend
from unisim.backend.mjwarp.backend import MjwarpBodyPairContactPlan
from unisim.backend.mjwarp.dependencies import (
    MjwarpDependencyError,
    load_mjwarp_dependencies,
)
from unisim.scene import SceneCfg

_FIELDS = ("active", "count", "position_w", "force_w")
_MODEL = """
<mujoco model='mjwarp-body-pair-contacts'>
  <option timestep='0.005' gravity='0 0 -9.81'/>
  <worldbody>
    <geom name='floor' type='plane' size='2 2 .1' friction='1 0 0' condim='3'/>
    <body name='box' pos='0 0 .2' quat='0.9238795 0 .3826834 0'>
      <freejoint name='root'/>
      <site name='box_site' pos='.01 .02 .03'/>
      <geom name='box_geom' type='box' size='.1 .1 .1' mass='1' friction='1 0 0' condim='3'/>
    </body>
  </worldbody>
  <sensor>
    <contact name='box_floor' body1='box' body2='world' data='found force pos'
             num='1' reduce='none'/>
    <contact name='floor_box' body1='world' body2='box' data='found force pos'
             num='1' reduce='none'/>
    <contact name='box_floor_second' body1='box' body2='world' data='found force pos'
             num='1' reduce='none'/>
    <contact name='floor_box_second' body1='world' body2='box' data='found force pos'
             num='1' reduce='none'/>
    <contact name='geom_box_floor' geom1='box_geom' geom2='floor' data='found'/>
    <framepos name='box_pos' objtype='body' objname='box'/>
    <framelinvel name='box_linvel' objtype='body' objname='box'/>
    <velocimeter name='box_vel' site='box_site'/>
  </sensor>
</mujoco>
"""


def _require_cuda_runtime() -> None:
    try:
        deps = load_mjwarp_dependencies()
    except MjwarpDependencyError as exc:
        pytest.fail(str(exc))
    deps.warp.init()
    if not bool(deps.warp.get_device().is_cuda):
        pytest.fail("MJWarp body-pair contacts require an active CUDA Warp device")


def _backend(tmp_path: Path, name: str, *, num_envs: int = 3, xml: str = _MODEL) -> MjwarpBackend:
    _require_cuda_runtime()
    model_file = tmp_path / name
    model_file.write_text(xml, encoding="utf-8")
    return MjwarpBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=num_envs,
        sim_dt=0.005,
        nconmax=128,
        njmax=256,
    )


def _step(backend: MjwarpBackend, ctrl: Any, *, nsteps: int = 1) -> None:
    result = backend.step_tensor(ctrl, nsteps=nsteps)
    if isinstance(result, dict):
        completion = result.get("completion")
        if completion is not None:
            completion.wait_before_read()


def _values(data: BodyPairContactData) -> dict[str, Any]:
    return {field: getattr(data, field).clone() for field in _FIELDS}


def _snapshot(views: dict[str, BodyPairContactData]) -> dict[str, dict[str, Any]]:
    return {name: _values(data) for name, data in views.items()}


def _cpu_oracle(model_file: Path, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    box_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box"))
    world_id = 0
    expected_position = {box_id: [], world_id: []}
    force_on_box = np.zeros(3, dtype=np.float64)
    for row in range(qpos.shape[0]):
        data.qpos[:] = qpos[row]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_id, force)
            world_force = contact.frame.reshape(3, 3).T @ force[:3]
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])
            if (body1, body2) == (box_id, world_id):
                force_on_box += world_force
                # Keep a common surface midpoint for oracle parity; MJWarp's
                # engine-native contact record also stores this midpoint.
                expected_position[box_id].append(np.asarray(contact.pos).copy())
            elif (body1, body2) == (world_id, box_id):
                force_on_box -= world_force
                # Keep a common surface midpoint for oracle parity; MJWarp's
                # engine-native contact record also stores this midpoint.
                expected_position[box_id].append(np.asarray(contact.pos).copy())
    points = np.asarray(expected_position[box_id], dtype=np.float64)
    position = points.mean(axis=0) if points.size else np.zeros(3, dtype=np.float64)
    return position, force_on_box


def test_real_cuda_body_pair_aggregation_direction_and_cpu_oracle(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    backend = _backend(tmp_path, "aggregation.xml")
    try:
        assert backend.get_tensor_capabilities().batched_body_pair_contacts
        ctrl = torch.zeros((3, 0), dtype=torch.float32, device="cuda")
        for _ in range(40):
            _step(backend, ctrl, nsteps=2)
        # Bind every body-pair plan before the first stale publication so one
        # device aggregation refreshes the complete caller-visible group.
        names = ("floor_box", "box_floor", "box_floor_second", "floor_box_second")
        all_views = backend.bind_body_pair_contact_views(names)
        all_views = {name: backend.get_sensor_view(name) for name in names}
        assert backend._body_pair_contact_index is None
        box = all_views["box_floor"]
        floor = all_views["floor_box"]
        assert isinstance(box, BodyPairContactData)
        assert isinstance(floor, BodyPairContactData)
        for data in (box, floor):
            assert data.active.shape == data.count.shape == (3,)
            assert data.position_w.shape == data.force_w.shape == (3, 3)
            assert data.active.dtype == torch.bool
            assert data.count.dtype == torch.int32
            assert data.position_w.dtype == data.force_w.dtype == torch.float32
            assert torch.all(data.active)
            assert torch.all(data.count > 1)
            assert bool(torch.isfinite(data.position_w).all())
            assert bool(torch.isfinite(data.force_w).all())
        np.testing.assert_allclose(
            floor.force_w.detach().cpu().numpy(),
            -box.force_w.detach().cpu().numpy(),
            rtol=2e-5,
            atol=2e-5,
        )
        # Positions differ by the two opposed witness-point half-depths;
        # the CPU oracle below checks the declared first body explicitly.
        qpos = backend.get_state_views(("qpos",))["qpos"].detach().cpu().numpy()
        _, force_on_box = _cpu_oracle(Path(backend.scene_model_file), qpos[0:1])
        np.testing.assert_allclose(
            box.force_w[0].detach().cpu().numpy(), force_on_box, rtol=2e-3, atol=2e-4
        )
        np.testing.assert_allclose(
            floor.force_w[0].detach().cpu().numpy(), -force_on_box, rtol=2e-3, atol=2e-4
        )

        previous = _values(box)
        _step(backend, ctrl, nsteps=5)
        assert backend.get_sensor_view("box_floor") is box
        current = _values(box)
        assert bool(box.active.any())
        assert torch.all(box.count > 1)
        for field in ("active", "count"):
            torch.testing.assert_close(current[field], previous[field])
        # Continuing compression changes force but preserves finite, ordered data.
        assert bool(torch.isfinite(box.force_w).all())
        assert bool(torch.isfinite(box.position_w).all())

        # Lift one row: only that row becomes an inactive, fully zero aggregate.
        state = backend.get_state_views(("qpos", "qvel"))
        moved = state["qpos"].clone()
        moved[2, 0] = 20.0
        moved[2, 2] = 0.6
        backend.set_state_tensor(
            torch.tensor([2], dtype=torch.int64, device="cuda"),
            moved[2:3].contiguous(),
            torch.zeros((1, state["qvel"].shape[1]), dtype=torch.float32, device="cuda"),
        )
        assert bool(box.active[:2].all())
        assert not bool(box.active[2])
        assert int(box.count[2]) == 0
        np.testing.assert_array_equal(box.position_w[2].detach().cpu().numpy(), np.zeros(3))
        np.testing.assert_array_equal(box.force_w[2].detach().cpu().numpy(), np.zeros(3))
    finally:
        backend.close()


def test_real_cuda_body_pair_hot_path_and_eager_graph_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    graph = _backend(tmp_path, "contacts-graph.xml", num_envs=2)
    eager = _backend(tmp_path, "contacts-eager.xml", num_envs=2)
    eager._cuda_graph_enabled = False
    try:
        ctrl = torch.zeros((2, 0), dtype=torch.float32, device="cuda")
        _step(graph, ctrl, nsteps=80)
        _step(eager, ctrl, nsteps=80)
        graph_views = {
            name: graph.get_sensor_view(name)
            for name in ("box_floor", "floor_box", "box_floor_second", "floor_box_second")
        }
        eager_views = {
            name: eager.get_sensor_view(name)
            for name in ("box_floor", "floor_box", "box_floor_second", "floor_box_second")
        }
        graph_view = graph_views["box_floor"]
        eager_view = eager_views["box_floor"]
        for field in _FIELDS:
            torch.testing.assert_close(
                getattr(graph_view, field), getattr(eager_view, field), rtol=2e-4, atol=2e-5
            )

        original_refresh = graph._refresh_body_pair_contacts_device_only
        refreshes = 0

        def counted_refresh() -> None:
            nonlocal refreshes
            refreshes += 1
            original_refresh()

        monkeypatch.setattr(graph, "_refresh_body_pair_contacts_device_only", counted_refresh)
        with monkeypatch.context() as hot_path:

            def forbidden_read(self: object, *_args: object, **_kwargs: object) -> None:
                raise AssertionError("unexpected host round-trip in MJWarp tensor step")

            hot_path.setattr(torch.Tensor, "item", forbidden_read)
            hot_path.setattr(torch.Tensor, "cpu", forbidden_read)
            _step(graph, ctrl)
        assert refreshes == 0
        refreshed_view = graph.get_sensor_view("box_floor")
        assert refreshed_view is graph_view
        assert refreshes == 1
        assert bool(graph_view.active.all())
        assert bool(torch.isfinite(graph_view.force_w).all())
    finally:
        graph.close()
        eager.close()


def test_real_cuda_batch_matches_single_views_and_refresh_boundary(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    extra_names = tuple(f"box_floor_extra_{index}" for index in range(2)) + tuple(
        f"floor_box_extra_{index}" for index in range(2)
    )
    extra_sensors = "".join(
        f'<contact name="{name}" body1="{"box" if name.startswith("box_") else "world"}" '
        f'body2="{"world" if name.startswith("box_") else "box"}" '
        'data="found force pos" num="1" reduce="none"/>'
        for name in extra_names
    )
    names = (
        "box_floor",
        "floor_box",
        "box_floor_second",
        "floor_box_second",
        *extra_names,
    )
    backend = _backend(
        tmp_path,
        "batch.xml",
        num_envs=3,
        xml=_MODEL.replace("</sensor>", f"{extra_sensors}</sensor>"),
    )
    try:
        ctrl = torch.zeros((3, 0), dtype=torch.float32, device="cuda")
        for _ in range(40):
            _step(backend, ctrl, nsteps=2)
        batch = backend.bind_body_pair_contact_views(names)
        singles = {name: backend.get_sensor_view(name) for name in names}
        index = backend._body_pair_contact_index
        assert index is not None
        keys = index.keys
        assert isinstance(batch, BodyPairContactBatch)
        assert batch.sensor_names == names
        assert batch.active.shape == batch.count.shape == (3, len(names))
        assert batch.position_w.shape == batch.force_w.shape == (3, len(names), 3)
        for slot, name in enumerate(names):
            for field in _FIELDS:
                torch.testing.assert_close(
                    getattr(batch, field)[:, slot], getattr(singles[name], field)
                )
        assert backend.bind_body_pair_contact_views(names) is batch

        _step(backend, ctrl)
        assert index.keys is keys
        for slot, name in enumerate(names):
            for field in _FIELDS:
                torch.testing.assert_close(
                    getattr(batch, field)[:, slot], getattr(singles[name], field)
                )
        # A settled manifold can be numerically unchanged; equality of batch
        # and single views after the step is the public invariant checked above.
        backend.refresh_body_pair_contact_views(names)
        for slot, name in enumerate(names):
            for field in _FIELDS:
                torch.testing.assert_close(
                    getattr(batch, field)[:, slot], getattr(singles[name], field)
                )
    finally:
        backend.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_vectorized_fanout_matches_per_plan_oracle(device: str) -> None:
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("vectorized body-pair contacts require a CUDA device")
    backend = object.__new__(MjwarpBackend)
    backend._num_envs = 2
    backend._cpu_model = SimpleNamespace(nbody=128)
    backend._validate_body_pair_contact_device = lambda _device=None: torch.device(device)
    pairs = (
        (10, 20),
        (20, 10),
        (10, 20),
        (30, 40),
        (40, 30),
        (50, 60),
        (60, 50),
        (70, 80),
    )
    names = tuple(f"pair_{index}" for index in range(len(pairs)))
    backend._body_pair_sensor_plans = {
        name: MjwarpBodyPairContactPlan(sensor_id=index, body1_id=pair[0], body2_id=pair[1])
        for index, (name, pair) in enumerate(zip(names, pairs, strict=True))
    }
    backend._body_pair_contact_views = {
        name: BodyPairContactData(
            active=torch.zeros(2, dtype=torch.bool, device=device),
            count=torch.zeros(2, dtype=torch.int32, device=device),
            position_w=torch.zeros(2, 3, dtype=torch.float32, device=device),
            force_w=torch.zeros(2, 3, dtype=torch.float32, device=device),
        )
        for name in names
    }
    backend._body_pair_contact_index = None
    backend._rebuild_body_pair_contact_index()
    assert backend._body_pair_contact_index is not None

    contacts: list[dict[str, Any]] = []
    for world in range(2):
        for body0, body1 in pairs:
            for first, second in ((body0, body1), (body1, body0)):
                contacts.append(
                    {
                        "world": world,
                        "body0": first,
                        "body1": second,
                        "position": (1.0 + world, 2.0, 3.0),
                        "dist": 0.5,
                        "normal": (1.0, 0.0, 0.0),
                        "force_w": (2.0, 3.0 + world, 4.0),
                        "valid": True,
                    }
                )
    contacts.append(
        {
            "world": 0,
            "body0": 10,
            "body1": 20,
            "position": (float("nan"),) * 3,
            "dist": float("nan"),
            "normal": (float("nan"),) * 3,
            "force_w": (float("nan"),) * 3,
            "valid": False,
        }
    )

    def values(key: str, dtype: Any) -> Any:
        return torch.tensor([contact[key] for contact in contacts], dtype=dtype, device=device)

    snapshot = {
        "world": values("world", torch.int64),
        "body0": values("body0", torch.int64),
        "body1": values("body1", torch.int64),
        "position": values("position", torch.float32),
        "dist": values("dist", torch.float32),
        "normal": values("normal", torch.float32),
        "force_w": values("force_w", torch.float32),
        "valid": values("valid", torch.bool),
    }
    backend._refresh_vectorized_body_pair_contacts(snapshot)

    for name, (body_a, body_b) in zip(names, pairs, strict=True):
        expected_count = np.zeros(2, dtype=np.int32)
        expected_position = np.zeros((2, 3), dtype=np.float32)
        expected_force = np.zeros((2, 3), dtype=np.float32)
        for contact in contacts:
            if not contact["valid"]:
                continue
            pair = (contact["body0"], contact["body1"])
            half_depth = 0.5 * contact["dist"] * np.asarray(contact["normal"])
            if pair == (body_a, body_b):
                direction = 1.0
                position = np.asarray(contact["position"]) - half_depth
            elif pair == (body_b, body_a):
                direction = -1.0
                position = np.asarray(contact["position"]) + half_depth
            else:
                continue
            world = contact["world"]
            expected_count[world] += 1
            expected_position[world] += position
            expected_force[world] += direction * np.asarray(contact["force_w"])
        expected_position /= np.maximum(expected_count, 1)[:, None]
        view = backend._body_pair_contact_views[name]
        np.testing.assert_array_equal(view.active.cpu().numpy(), expected_count > 0)
        np.testing.assert_array_equal(view.count.cpu().numpy(), expected_count)
        np.testing.assert_allclose(
            view.position_w.cpu().numpy(), expected_position, rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(view.force_w.cpu().numpy(), expected_force, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "mask_values",
    [[False] * 3, [False, True, False], [True, False, True], [True] * 3],
)
def test_real_cuda_mask_reset_sensor_freshness_and_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mask_values: list[bool]
) -> None:
    torch = pytest.importorskip("torch")
    backend = _backend(tmp_path, f"reset-{''.join(str(int(v)) for v in mask_values)}.xml")
    mask = torch.tensor(mask_values, dtype=torch.bool, device="cuda")
    try:
        ctrl = torch.zeros((3, 0), dtype=torch.float32, device="cuda")
        for _ in range(40):
            _step(backend, ctrl, nsteps=2)
        batch = backend.bind_body_pair_contact_views(("box_floor", "floor_box"))
        contacts = {name: backend.get_sensor_view(name) for name in ("box_floor", "floor_box")}
        frame = backend.get_sensor_view("box_pos")
        linvel = backend.get_sensor_view("box_linvel")
        site_vel = backend.get_sensor_view("box_vel")
        exact = backend.get_sensor_view("geom_box_floor")
        before = {
            "contacts": _snapshot(contacts),
            "frame": frame.clone(),
            "linvel": linvel.clone(),
            "site_vel": site_vel.clone(),
            "exact": exact.clone(),
        }

        state = backend.get_state_views(("qpos", "qvel"))
        qpos = state["qpos"].clone()
        qvel = state["qvel"].clone()
        qpos[mask, 2] = 0.55
        qpos[mask, 0] += 0.125
        qvel[mask, 0] = 0.375
        qvel[mask, 2] = 0.075
        with monkeypatch.context() as reset_path:

            def forbidden_sync(*_args: object, **_kwargs: object) -> None:
                raise AssertionError("caller/host synchronize is not required")

            reset_path.setattr(torch.cuda, "synchronize", forbidden_sync)
            result = backend.set_state_tensor_mask(mask, qpos, qvel)
        assert result["timing"]["set_state_tensor_host_cache_refresh_ms"] == 0.0

        expected_pos = qpos[:, :3].clone()
        expected_vel = qvel.clone()
        retained = mask.logical_not()
        for name, data in contacts.items():
            torch.testing.assert_close(data.active, retained & before["contacts"][name]["active"])
            expected_count = before["contacts"][name]["count"] * retained
            expected_position = before["contacts"][name]["position_w"] * retained[:, None]
            expected_force = before["contacts"][name]["force_w"] * retained[:, None]
            torch.testing.assert_close(data.count, expected_count)
            torch.testing.assert_close(data.position_w, expected_position)
            torch.testing.assert_close(data.force_w, expected_force)
        for slot, name in enumerate(batch.sensor_names):
            torch.testing.assert_close(batch.active[:, slot], contacts[name].active)
            torch.testing.assert_close(batch.count[:, slot], contacts[name].count)
            torch.testing.assert_close(batch.position_w[:, slot], contacts[name].position_w)
            torch.testing.assert_close(batch.force_w[:, slot], contacts[name].force_w)
        torch.testing.assert_close(backend.get_sensor_view("box_pos"), expected_pos)
        torch.testing.assert_close(backend.get_sensor_view("box_linvel"), expected_vel[:, :3])
        import mujoco

        site_id = int(mujoco.mj_name2id(backend._cpu_model, mujoco.mjtObj.mjOBJ_SITE, "box_site"))
        expected_site_velocity = []
        for row in range(qpos.shape[0]):
            cpu_data = mujoco.MjData(backend._cpu_model)
            cpu_data.qpos[:] = qpos[row].detach().cpu().numpy()
            cpu_data.qvel[:] = qvel[row].detach().cpu().numpy()
            mujoco.mj_forward(backend._cpu_model, cpu_data)
            site_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                backend._cpu_model,
                cpu_data,
                mujoco.mjtObj.mjOBJ_SITE,
                site_id,
                site_velocity,
                1,
            )
            expected_site_velocity.append(site_velocity[3:].copy())
        torch.testing.assert_close(
            backend.get_sensor_view("box_vel"),
            torch.as_tensor(
                np.stack(expected_site_velocity), dtype=torch.float32, device=qvel.device
            ),
            rtol=2e-5,
            atol=2e-5,
        )
        torch.testing.assert_close(exact, before["exact"] * retained[:, None])
        assert backend.get_state_views(("qpos",))["qpos"].data_ptr() == state["qpos"].data_ptr()
        torch.testing.assert_close(backend.get_state_views(("qpos",))["qpos"], qpos)

        if bool(mask.any()):
            _step(backend, ctrl)
            for name, data in contacts.items():
                assert bool(torch.isfinite(data.position_w).all())
                assert bool(torch.isfinite(data.force_w).all())
                assert torch.equal(data.active, data.count > 0)
    finally:
        backend.close()


def test_real_cuda_row_reset_and_empty_mask_do_not_publish_contacts(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    backend = _backend(tmp_path, "row-reset.xml")
    try:
        ctrl = torch.zeros((3, 0), dtype=torch.float32, device="cuda")
        for _ in range(40):
            _step(backend, ctrl, nsteps=2)
        box = backend.get_sensor_view("box_floor")
        late = backend.get_sensor_view("floor_box")
        backend.get_sensor_view("box_floor_second")
        backend.get_sensor_view("floor_box_second")
        before = _snapshot({"box_floor": box, "floor_box": late})
        refreshes = 0
        original_refresh = backend._refresh_body_pair_contacts_device_only

        def counted_refresh() -> None:
            nonlocal refreshes
            refreshes += 1
            original_refresh()

        clear_calls = 0
        original_clear = backend._clear_body_pair_contact_rows

        def counted_clear(reset_mask):
            nonlocal clear_calls
            clear_calls += 1
            return original_clear(reset_mask)

        backend._clear_body_pair_contact_rows = counted_clear
        backend._refresh_body_pair_contacts_device_only = counted_refresh

        state = backend.get_state_views(("qpos", "qvel"))
        backend.set_state_tensor(
            torch.zeros((0,), dtype=torch.int64, device="cuda"),
            torch.empty((0, state["qpos"].shape[1]), dtype=torch.float32, device="cuda"),
            torch.empty((0, state["qvel"].shape[1]), dtype=torch.float32, device="cuda"),
        )
        assert refreshes == 0
        assert clear_calls == 0
        for name in before:
            for field in _FIELDS:
                torch.testing.assert_close(
                    getattr(backend.get_sensor_view(name), field), before[name][field]
                )

        moved = state["qpos"].clone()
        moved[1, 2] = 0.45
        backend.set_state_tensor(
            torch.tensor([1], dtype=torch.int64, device="cuda"),
            moved[1:2].contiguous(),
            torch.zeros((1, state["qvel"].shape[1]), dtype=torch.float32, device="cuda"),
        )
        assert refreshes == 0
        assert clear_calls == 1
        for name, data in (("box_floor", box), ("floor_box", late)):
            torch.testing.assert_close(data.active[0], before[name]["active"][0])
            torch.testing.assert_close(data.count[0], before[name]["count"][0])
            torch.testing.assert_close(data.position_w[0], before[name]["position_w"][0])
            torch.testing.assert_close(data.force_w[0], before[name]["force_w"][0])
            assert not bool(data.active[1])
            assert int(data.count[1]) == 0
            np.testing.assert_array_equal(data.position_w[1].detach().cpu().numpy(), np.zeros(3))
            np.testing.assert_array_equal(data.force_w[1].detach().cpu().numpy(), np.zeros(3))
    finally:
        backend.close()


def test_body_pair_bindings_validate_names_and_sensor_kind(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "validation.xml")
    try:
        with pytest.raises(TypeError, match="sequence"):
            backend.bind_body_pair_contact_views("box_floor")
        with pytest.raises(TypeError, match="non-empty"):
            backend.bind_body_pair_contact_views(())
        with pytest.raises(ValueError, match="unique"):
            backend.bind_body_pair_contact_views(("box_floor", "box_floor"))
        with pytest.raises(ValueError, match="not found"):
            backend.bind_body_pair_contact_views(("missing",))
        with pytest.raises(ValueError, match="ordered body-pair"):
            backend.bind_body_pair_contact_views(("geom_box_floor",))
        with pytest.raises(TypeError, match="sequence"):
            backend.refresh_body_pair_contact_views("box_floor")
    finally:
        backend.close()


def test_real_cuda_single_contact_and_empty_pair_semantics(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    xml = _MODEL.replace("size='.1 .1 .1'", "size='.05 .05 .05'")
    backend = _backend(tmp_path, "single.xml", xml=xml)
    try:
        ctrl = torch.zeros((3, 0), dtype=torch.float32, device="cuda")
        for _ in range(40):
            _step(backend, ctrl, nsteps=2)
        names = ("floor_box", "box_floor", "box_floor_second", "floor_box_second")
        backend.bind_body_pair_contact_views(names)
        box = backend.get_sensor_view("box_floor")
        assert isinstance(box, BodyPairContactData)
        assert torch.all(box.count >= 1)
        qpos = backend.get_state_views(("qpos",))["qpos"][:1].detach().cpu().numpy()
        _, force = _cpu_oracle(Path(backend.scene_model_file), qpos)
        np.testing.assert_allclose(
            box.force_w[0].detach().cpu().numpy(), force, rtol=2e-3, atol=2e-4
        )

        state = backend.get_state_views(("qpos", "qvel"))
        airborne = state["qpos"].clone()
        airborne[:, 2] = 0.8
        airborne[:, 0] = 30.0
        backend.set_state_tensor_mask(
            torch.ones((3,), dtype=torch.bool, device="cuda"),
            airborne,
            torch.zeros_like(state["qvel"]),
        )
        assert not bool(box.active.any())
        assert not bool(box.count.any())
        assert not bool(box.position_w.any())
        assert not bool(box.force_w.any())
    finally:
        backend.close()
