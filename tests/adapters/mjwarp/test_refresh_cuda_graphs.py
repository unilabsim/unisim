"""Real-CUDA acceptance for the MJWarp final-state and derived-constant refresh graphs.

Graph replay is checked against the per-kernel launches on identical inputs,
against independent CPU MuJoCo references, and inside an external Torch
capture.  The dispatch logic that needs no GPU is covered by
``test_refresh_graph_dispatch.py``.
"""

# ruff: noqa: E402

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
warp = pytest.importorskip("warp")
torch = pytest.importorskip("torch")

from unisim import MjwarpBackend
from unisim.dr.types import TensorResetRandomizationPayload
from unisim.scene import SceneCfg

_FINAL_STATE_KERNELS = ("kinematics", "com_pos", "com_vel", "sensor_pos", "sensor_vel")
_REFRESH_VARIANTS = ((2, False), (2, True), (1, False), (1, True))
_DR_FIELDS = {2: ("body_mass", "body_ipos", "body_iquat"), 1: ("body_inertia", "dof_armature")}

MODEL = """<mujoco model='unisim-test-mjwarp-refresh-graphs'>
  <option timestep='0.01' gravity='0 0 -9.81'/>
  <worldbody>
    <body name='base' pos='0 0 1'>
      <freejoint name='root'/>
      <geom name='base_geom' type='box' size='0.08 0.06 0.05' mass='1.2'/>
      <site name='base_site'/>
      <body name='arm' pos='0.06 0 0'>
        <joint name='hinge' type='hinge' axis='0 1 0' damping='0.3' armature='0.01'/>
        <geom name='arm_geom' type='capsule' fromto='0 0 0 0.2 0 0' size='0.03' mass='0.3'/>
        <site name='arm_site' pos='0.2 0 0'/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <framepos name='base_pos' objtype='body' objname='base'/>
    <framelinvel name='base_linvel' objtype='body' objname='base'/>
    <framequat name='arm_quat' objtype='body' objname='arm'/>
    <framelinvel name='arm_site_linvel' objtype='site' objname='arm_site'/>
    <subtreecom name='base_com' body='base'/>
    <accelerometer name='arm_acc' site='arm_site'/>
  </sensor>
  <actuator><motor name='hinge_motor' joint='hinge' ctrlrange='-5 5'/></actuator>
</mujoco>"""
TRACKED_BODIES = ("base", "arm")
AUTHORED_SENSORS = ("base_pos", "base_linvel", "arm_quat", "arm_site_linvel", "base_com")
NUM_ENVS = 4
# Model fields the derived-constant refresh recomputes from the DR fields.
_ORACLE_FIELDS = ("body_subtreemass", "dof_invweight0", "body_invweight0", "actuator_acc0")
# Everything the restore-less pass leaves computed at qpos0 when no consumer
# has read yet.
_CONSTANT_FIELDS = (*_ORACLE_FIELDS, "geom_rbound", "geom_aabb")
_QPOS0_DATA_FIELDS = ("qpos", "qvel", "xpos", "xquat", "subtree_com", "actuator_moment")


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _model_file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.xml"
    path.write_text(MODEL)
    return path


def _cuda_backend(tmp_path: Path, name: str) -> tuple[MjwarpBackend, Path]:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp refresh graph tests require an active CUDA Warp device")
    model_file = _model_file(tmp_path, name)
    backend = MjwarpBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=NUM_ENVS,
        sim_dt=0.01,
        base_name="base",
        add_body_sensors=True,
        tracked_body_names=TRACKED_BODIES,
    )
    return backend, model_file


def _require_graphs(backend: MjwarpBackend) -> None:
    if not backend._cuda_graph_enabled:
        reason = backend._cuda_graph_disable_reason or "unknown reason"
        pytest.skip(f"test requires mjwarp CUDA graphs; graphs disabled: {reason}")


def _record_routes(backend: MjwarpBackend, monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Record every backend graph launch and every eager kernel the refreshes launch."""
    recorded: dict[str, list] = {"launch": [], "kernel": []}
    original = backend._mujoco_warp
    capture_launch = backend._warp.capture_launch

    class RecordingKernels:
        def __getattr__(self, name: str):
            return getattr(original, name)

    proxy = RecordingKernels()

    def forwarding(name: str):
        def launch(*args: object, **kwargs: object) -> None:
            recorded["kernel"].append((name, kwargs))
            getattr(original, name)(*args, **kwargs)

        return launch

    # Instance attributes, so the proxy itself is never passed as an operand.
    for name in (*_FINAL_STATE_KERNELS, "set_const"):
        setattr(proxy, name, forwarding(name))
    monkeypatch.setattr(backend, "_mujoco_warp", proxy)
    monkeypatch.setattr(
        backend._warp,
        "capture_launch",
        lambda graph, *args, **kwargs: (
            recorded["launch"].append(graph),
            capture_launch(graph, *args, **kwargs),
        )[1],
    )
    return recorded


def _launched_variants(backend: MjwarpBackend, graph_launches: list) -> list[tuple[int, bool]]:
    """Refresh variants whose backend-owned graph was handed to ``capture_launch``."""
    return [key for key, graph in backend._set_const_graphs.items() if graph in graph_launches]


def _set_const_calls(recorded: dict[str, list]) -> list[dict]:
    """Keyword arguments of every eager ``set_const`` launch."""
    return [kwargs for name, kwargs in recorded["kernel"] if name == "set_const"]


def _read_public(backend: MjwarpBackend) -> dict[str, Any]:
    """Clone every tensor view a training step consumes after physics."""
    views = backend.get_tracked_body_views()
    result = {
        "track_pos_w": views.pos_w,
        "track_quat_w": views.quat_w,
        "track_linvel_w": views.lin_vel_w,
        "track_angvel_w": views.ang_vel_w,
        "sensordata": backend.get_state_views(("qpos", "qvel", "sensordata"))["sensordata"],
        **{name: backend.get_sensor_view(name) for name in AUTHORED_SENSORS},
    }
    return {name: value.detach().clone() for name, value in result.items()}


def _assert_public_views_match_cpu_forward(backend: MjwarpBackend, reads: dict[str, Any]) -> None:
    """Tracked bodies and authored sensors agree with an independent CPU forward.

    The CPU reference carries each world's randomized mass/inertia rows, so it
    stays valid after a randomizing reset.
    """
    import copy

    mujoco = backend._mujoco
    model = copy.copy(backend._cpu_model)
    body_ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) for name in TRACKED_BODIES
    ]
    tables = {
        name: getattr(backend._device_model, name).numpy()
        for name in ("body_mass", "body_ipos", "body_iquat", "body_inertia", "dof_armature")
    }
    for row in range(NUM_ENVS):
        for name, table in tables.items():
            getattr(model, name)[:] = table[row] if table.shape[0] == NUM_ENVS else table[0]
        data = mujoco.MjData(model)
        mujoco.mj_setConst(model, data)
        data.qpos[:] = backend._device_data.qpos.numpy()[row]
        data.qvel[:] = backend._device_data.qvel.numpy()[row]
        mujoco.mj_forward(model, data)
        for slot, body_id in enumerate(body_ids):
            np.testing.assert_allclose(
                reads["track_pos_w"][row, slot].cpu().numpy(), data.xpos[body_id], atol=1e-5
            )
        expected = torch.as_tensor(
            data.sensordata.copy(), dtype=torch.float32, device=reads["sensordata"].device
        )
        for name in AUTHORED_SENSORS:
            address, dimension = backend._sensor_slots[name]
            torch.testing.assert_close(
                reads[name][row], expected[address : address + dimension], rtol=2e-5, atol=2e-5
            )


def _dr_payload(backend: MjwarpBackend, rows: int, level: int, *, device: str) -> Any:
    """A payload whose tables differ per row and from the model defaults."""

    model = backend._cpu_model
    scale = np.arange(1, rows + 1, dtype=np.float32) * 0.05

    def scaled(values: np.ndarray, offset: float) -> Any:
        table = np.broadcast_to(values, (rows, *values.shape))
        table = table * (1.0 + scale.reshape(rows, *([1] * values.ndim))) + offset
        return torch.as_tensor(np.ascontiguousarray(table, dtype=np.float32), device=device)

    if level == 2:
        return TensorResetRandomizationPayload(
            body_mass=scaled(model.body_mass, 0.0),
            body_ipos=scaled(model.body_ipos, 0.01),
            body_iquat=torch.as_tensor(
                np.ascontiguousarray(
                    np.broadcast_to(model.body_iquat, (rows, backend._nbody, 4)),
                    dtype=np.float32,
                ),
                device=device,
            ),
        )
    return TensorResetRandomizationPayload(
        body_inertia=scaled(model.body_inertia, 0.0),
        dof_armature=scaled(model.dof_armature, 0.01),
    )


def _reset_tensors(backend: MjwarpBackend) -> tuple[Any, Any, Any, Any]:
    """Selected rows, their mask, and a full-width state that differs from qpos0."""
    rows = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    mask = torch.zeros((NUM_ENVS,), dtype=torch.bool, device="cuda")
    mask[rows] = True
    qpos = backend.get_state_views(("qpos",))["qpos"].clone()
    qvel = backend.get_state_views(("qvel",))["qvel"].clone()
    qpos[:, 0] += torch.linspace(-0.2, 0.3, NUM_ENVS, device="cuda")
    qvel[:, 0] = 0.4
    return rows, mask, qpos, qvel


def _dr_reset(
    backend: MjwarpBackend, reset_kind: str, level: int, tensors: tuple[Any, Any, Any, Any]
) -> None:
    rows, mask, qpos, qvel = tensors
    if reset_kind == "selected":
        payload = _dr_payload(backend, rows.shape[0], level, device="cuda")
        backend.set_state_tensor(rows, qpos[rows], qvel[rows], payload)
    else:
        payload = _dr_payload(backend, NUM_ENVS, level, device="cuda")
        backend.set_state_tensor_mask(mask, qpos, qvel, payload)


def _constant_snapshot(backend: MjwarpBackend) -> dict[str, np.ndarray]:
    backend._warp.synchronize_device()
    return {
        **{
            name: getattr(backend._device_model, name).numpy().copy()
            for name in _CONSTANT_FIELDS
        },
        **{
            name: getattr(backend._device_data, name).numpy().copy()
            for name in _QPOS0_DATA_FIELDS
        },
    }


def _assert_constants_match_cpu_set_const(
    backend: MjwarpBackend, model_file: Path, row: int, level: int
) -> None:
    """Recomputed constants agree with CPU ``mj_setConst`` for one world."""
    mujoco = backend._mujoco
    model = mujoco.MjModel.from_xml_path(str(model_file))
    for name in _DR_FIELDS[level]:
        getattr(model, name)[:] = getattr(backend._device_model, name).numpy()[row]
    mujoco.mj_setConst(model, mujoco.MjData(model))
    for name in _ORACLE_FIELDS:
        np.testing.assert_allclose(
            getattr(backend._device_model, name).numpy()[row],
            getattr(model, name),
            rtol=1e-4,
            atol=1e-5,
        )


# --------------------------------------------------------------------------- #
# Graph ownership and numerical parity                                        #
# --------------------------------------------------------------------------- #


def test_backend_owns_one_graph_per_refresh_variant(tmp_path: Path) -> None:
    backend, _ = _cuda_backend(tmp_path, "owned")
    try:
        _require_graphs(backend)
        assert backend._refresh_graph is not None
        assert set(backend._set_const_graphs) == set(_REFRESH_VARIANTS)
        handles = [
            backend._refresh_graph,
            *backend._set_const_graphs.values(),
            backend._step_graph,
            backend._forward_graph,
            backend._reset_graph,
        ]
        assert len({id(handle) for handle in handles}) == len(handles)
    finally:
        backend.close()


def test_final_state_refresh_graph_matches_eager_kernels_and_cpu_forward(
    tmp_path: Path,
) -> None:
    backend, _ = _cuda_backend(tmp_path, "refresh-parity")
    try:
        _require_graphs(backend)
        ctrl = torch.linspace(-1.0, 1.0, NUM_ENVS, device="cuda").reshape(-1, 1)
        backend.step_tensor(ctrl, nsteps=3)["completion"].wait_before_read()

        graphed = _read_public(backend)
        _assert_public_views_match_cpu_forward(backend, graphed)

        # The same committed qpos/qvel through the five eager kernels.
        backend._cuda_graph_enabled = False
        eager = _read_public(backend)
        backend._cuda_graph_enabled = True
        for name in graphed:
            torch.testing.assert_close(graphed[name], eager[name], rtol=0, atol=0)
    finally:
        backend.close()


def test_refresh_graph_read_paths_match_eager_backend(tmp_path: Path) -> None:
    graphed, _ = _cuda_backend(tmp_path, "reads-graph")
    eager, _ = _cuda_backend(tmp_path, "reads-eager")
    try:
        _require_graphs(graphed)
        eager._cuda_graph_enabled = False
        triggers = (
            lambda backend: backend.get_tracked_body_views(),
            lambda backend: backend.get_sensor_view("arm_site_linvel"),
            lambda backend: backend.get_state_views(("sensordata",)),
            lambda backend: backend._sync_host_cache(),
        )
        for step in range(2 * len(triggers)):
            for backend in (graphed, eager):
                ctrl = torch.full((NUM_ENVS, 1), 0.2 * (step % 3) - 0.2, device="cuda")
                backend.step_tensor(ctrl, nsteps=2)["completion"].wait_before_read()
                # Rotate through every read that publishes derived views: the
                # tracked views, a named sensor, the raw sensor block, and the
                # host-cache publication.
                triggers[step % len(triggers)](backend)
            torch.cuda.synchronize()
            expected, actual = _read_public(eager), _read_public(graphed)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=2e-6, atol=2e-5)
            graphed._sync_host_cache()
            eager._sync_host_cache()
            np.testing.assert_allclose(graphed._sensor_cache, eager._sensor_cache, atol=2e-5)
            np.testing.assert_allclose(graphed._qpos_cache, eager._qpos_cache, atol=2e-6)
    finally:
        graphed.close()
        eager.close()


@pytest.mark.parametrize("reset_kind", ["selected", "mask"])
@pytest.mark.parametrize("level", [2, 1], ids=["level-2", "level-1"])
def test_device_dr_reset_replays_the_matching_constant_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_kind: str, level: int
) -> None:
    backend, model_file = _cuda_backend(tmp_path, f"dr-{reset_kind}-{level}")
    try:
        _require_graphs(backend)
        tensors = _reset_tensors(backend)
        recorded = _record_routes(backend, monkeypatch)

        _dr_reset(backend, reset_kind, level, tensors)

        groups = backend._mujoco_warp.SetConstGroup
        assert _launched_variants(backend, recorded["launch"]) == [(level, False)]
        assert _set_const_calls(recorded) == []
        # No consumer has read yet, so Data derived state still holds the
        # qpos0-computed values the restore-less pass leaves behind.
        replayed = _constant_snapshot(backend)
        _assert_constants_match_cpu_set_const(backend, model_file, 3, level)

        # The same model rows through the eager pass reproduce the graph bit for bit.
        backend._cuda_graph_enabled = False
        backend._refresh_derived_constants(level, restore=False)
        eager = _constant_snapshot(backend)
        backend._cuda_graph_enabled = True
        expected_groups = groups.ALL if level == 2 else groups.KIN0 | groups.INERT0
        assert _set_const_calls(recorded) == [{"groups": expected_groups, "restore": False}]
        for name in replayed:
            np.testing.assert_array_equal(replayed[name], eager[name])

        # The first tensor read republishes the live state, so no qpos0-computed
        # value is observable through the public views.
        ctrl = torch.zeros((NUM_ENVS, 1), device="cuda")
        backend.step_tensor(ctrl, nsteps=1)["completion"].wait_before_read()
        _assert_public_views_match_cpu_forward(backend, _read_public(backend))
        assert _launched_variants(backend, recorded["launch"]) == [(level, False)]
    finally:
        backend.close()


@pytest.mark.parametrize("reset_kind", ["selected", "mask"])
@pytest.mark.parametrize("level", [2, 1], ids=["level-2", "level-1"])
def test_device_dr_reset_trajectory_matches_eager_backend(
    tmp_path: Path, reset_kind: str, level: int
) -> None:
    graphed, _ = _cuda_backend(tmp_path, f"dr-traj-graph-{reset_kind}-{level}")
    eager, _ = _cuda_backend(tmp_path, f"dr-traj-eager-{reset_kind}-{level}")
    try:
        _require_graphs(graphed)
        eager._cuda_graph_enabled = False
        for step in range(3):
            for backend in (graphed, eager):
                _dr_reset(backend, reset_kind, level, _reset_tensors(backend))
                ctrl = torch.full((NUM_ENVS, 1), 0.3 * step - 0.3, device="cuda")
                backend.step_tensor(ctrl, nsteps=2)["completion"].wait_before_read()
            torch.cuda.synchronize()
            expected, actual = _read_public(eager), _read_public(graphed)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=2e-6, atol=2e-5)
    finally:
        graphed.close()
        eager.close()


@pytest.mark.parametrize("level", [2, 1], ids=["level-2", "level-1"])
@pytest.mark.parametrize("graphs", [True, False], ids=["graph", "eager"])
def test_host_dr_reset_replays_the_restoring_constant_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, level: int, graphs: bool
) -> None:
    backend, _ = _cuda_backend(tmp_path, f"host-dr-{level}")
    try:
        _require_graphs(backend)
        if not graphs:
            backend._cuda_graph_enabled = False
        recorded = _record_routes(backend, monkeypatch)
        rows, _, qpos, qvel = _reset_tensors(backend)

        backend.set_state_tensor(
            rows, qpos[rows], qvel[rows], _dr_payload(backend, rows.shape[0], level, device="cpu")
        )

        groups = backend._mujoco_warp.SetConstGroup
        if graphs:
            assert _launched_variants(backend, recorded["launch"]) == [(level, True)]
            assert _set_const_calls(recorded) == []
        else:
            expected = groups.ALL if level == 2 else groups.KIN0 | groups.INERT0
            assert _launched_variants(backend, recorded["launch"]) == []
            assert _set_const_calls(recorded) == [{"groups": expected, "restore": True}]
    finally:
        backend.close()


@pytest.mark.parametrize("level", [2, 1], ids=["level-2", "level-1"])
def test_restoring_constant_graph_matches_eager_pass_bitwise(tmp_path: Path, level: int) -> None:
    backend, _ = _cuda_backend(tmp_path, f"restore-parity-{level}")
    try:
        _require_graphs(backend)
        _dr_reset(backend, "mask", level, _reset_tensors(backend))

        backend._refresh_derived_constants(level, restore=True)
        graphed = _constant_snapshot(backend)
        backend._cuda_graph_enabled = False
        backend._refresh_derived_constants(level, restore=True)
        eager = _constant_snapshot(backend)
        backend._cuda_graph_enabled = True
        for name in graphed:
            np.testing.assert_array_equal(graphed[name], eager[name])
    finally:
        backend.close()


@pytest.mark.parametrize("level", [2, 1], ids=["level-2", "level-1"])
def test_restoring_constant_graph_leaves_derived_state_at_the_live_qpos(
    tmp_path: Path, level: int
) -> None:
    backend, _ = _cuda_backend(tmp_path, f"restore-{level}")
    try:
        _require_graphs(backend)
        _, _, qpos, qvel = _reset_tensors(backend)
        backend._stable_tensor_field_view("qpos").copy_(qpos)
        backend._stable_tensor_field_view("qvel").copy_(qvel)
        assert not np.allclose(qpos.cpu().numpy(), backend._cpu_model.qpos0)

        backend._refresh_derived_constants(level, restore=True)
        restored = {
            name: getattr(backend._device_data, name).numpy().copy()
            for name in ("xpos", "xquat", "subtree_com")
        }

        # The restore passes rebuilt the derived state for qpos, not qpos0.
        backend._cuda_graph_enabled = False
        backend._launch_final_state_refresh()
        for name, value in restored.items():
            np.testing.assert_array_equal(value, getattr(backend._device_data, name).numpy())
    finally:
        backend.close()


# --------------------------------------------------------------------------- #
# External Torch capture                                                      #
# --------------------------------------------------------------------------- #


class _Inputs:
    """Static graph inputs and outputs: refilled before every replay, never reallocated."""

    def __init__(self, backend: MjwarpBackend) -> None:
        state = backend.get_state_views(("qpos", "qvel", "sensordata"))
        views = backend.get_tracked_body_views()
        self.ctrl = torch.zeros((NUM_ENVS, 1), dtype=torch.float32, device="cuda")
        self.mask = torch.zeros((NUM_ENVS,), dtype=torch.bool, device="cuda")
        self.qpos = torch.zeros_like(state["qpos"])
        self.qvel = torch.zeros_like(state["qvel"])
        payload = {}
        for level in (2, 1):
            table = _dr_payload(backend, NUM_ENVS, level, device="cuda")
            payload.update(
                {name: getattr(table, name) for name in table.requested_terms()}
            )
        self.payload = TensorResetRandomizationPayload(**payload)
        self.outputs = {
            "pos_w": torch.zeros_like(views.pos_w),
            "lin_vel_w": torch.zeros_like(views.lin_vel_w),
            "sensordata": torch.zeros_like(state["sensordata"]),
        }

    def load(self, k: int) -> None:
        rows = torch.arange(NUM_ENVS, device="cuda")
        self.ctrl.copy_((torch.sin(rows.float() + 0.37 * k) * 0.8).reshape(-1, 1))
        self.mask.copy_((rows + k) % 2 == 0)
        self.qpos.copy_(torch.zeros_like(self.qpos) + 0.01 * k)
        self.qvel.copy_(torch.zeros_like(self.qvel) - 0.02 * k)


def _cycle(backend: MjwarpBackend, inputs: _Inputs) -> None:
    """One training control step: physics, sensor reads, then a randomizing autoreset."""
    backend.step_tensor(inputs.ctrl, nsteps=2)
    views = backend.get_tracked_body_views()
    sensors = backend.get_state_views(("qpos", "qvel", "sensordata"))
    # A captured read has to land in owner-owned storage.
    inputs.outputs["pos_w"].copy_(views.pos_w)
    inputs.outputs["lin_vel_w"].copy_(views.lin_vel_w)
    inputs.outputs["sensordata"].copy_(sensors["sensordata"])
    backend.set_state_tensor_mask(inputs.mask, inputs.qpos, inputs.qvel, inputs.payload)


def _state(backend: MjwarpBackend) -> dict[str, Any]:
    views = backend.get_state_views(("qpos", "qvel", "time"))
    return {name: value.clone() for name, value in views.items()}


def test_external_capture_records_the_eager_refresh_kernels_without_a_nested_graph_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Torch-owned capture takes the eager kernels of both refreshes.

    UniLab ``autoreset_cuda_graph``-style owners capture the first tensor read
    and the randomizing reset together with physics.  Backend graphs must stay
    out of that capture: a nested Warp graph launch invalidates it.
    """
    backend, _ = _cuda_backend(tmp_path, "external-capture")
    eager, _ = _cuda_backend(tmp_path, "external-eager")
    try:
        _require_graphs(backend)
        eager._cuda_graph_enabled = False
        inputs, eager_inputs = _Inputs(backend), _Inputs(eager)

        # Warm every captured operation eagerly from the owner's steady state.
        for owner, operands in ((backend, inputs), (eager, eager_inputs)):
            operands.load(0)
            _cycle(owner, operands)
        torch.cuda.synchronize()

        recorded = _record_routes(backend, monkeypatch)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with backend.torch_graph_capture_scope() as capture:
                _cycle(backend, inputs)
        assert recorded["launch"] == []
        assert [name for name, _ in recorded["kernel"]] == [
            *_FINAL_STATE_KERNELS,
            *_FINAL_STATE_KERNELS,
            "set_const",
        ]
        assert capture.host_transition_names == (
            "mjwarp.step_tensor",
            "mjwarp.reset_randomization.model_rows",
            "mjwarp.reset_randomization.mirrors",
            "mjwarp.set_state_tensor_mask",
        )
        monkeypatch.undo()

        # Capture records without executing device work, but its host
        # bookkeeping already ran once: it stands in for the first replay's.
        for k in range(1, 9):
            inputs.load(k)
            capture.check_replay_guards()
            graph.replay()
            if k > 1:
                capture.replay_host_transitions()
            backend.flush_tensor_validation()
            eager_inputs.load(k)
            _cycle(eager, eager_inputs)
            torch.cuda.synchronize()
            for name in inputs.outputs:
                torch.testing.assert_close(
                    inputs.outputs[name], eager_inputs.outputs[name], rtol=2e-6, atol=2e-5
                )
            expected, actual = _state(eager), _state(backend)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=2e-6, atol=2e-5)
            np.testing.assert_array_equal(backend._time_cache, eager._time_cache)
    finally:
        backend.close()
        eager.close()


# --------------------------------------------------------------------------- #
# Lifecycle                                                                   #
# --------------------------------------------------------------------------- #


def test_reinitializing_cuda_graphs_retires_the_previous_refresh_graphs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, _ = _cuda_backend(tmp_path, "reinitialize")
    try:
        _require_graphs(backend)
        previous_refresh = backend._refresh_graph
        previous_constants = dict(backend._set_const_graphs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend._initialize_cuda_graphs(backend._warp.get_device())
        assert backend._cuda_graph_enabled
        assert not [item for item in caught if "CUDA graphs disabled" in str(item.message)]
        assert backend._refresh_graph is not previous_refresh
        assert set(backend._set_const_graphs) == set(_REFRESH_VARIANTS)
        assert all(
            backend._set_const_graphs[key] is not previous_constants[key]
            for key in _REFRESH_VARIANTS
        )

        recorded = _record_routes(backend, monkeypatch)
        ctrl = torch.zeros((NUM_ENVS, 1), device="cuda")
        backend.step_tensor(ctrl, nsteps=1)["completion"].wait_before_read()
        backend.get_tracked_body_views()
        for level in (2, 1):
            _dr_reset(backend, "mask", level, _reset_tensors(backend))
        launches = recorded["launch"]
        assert launches.count(backend._refresh_graph) >= 1
        assert _launched_variants(backend, launches) == [(2, False), (1, False)]
        assert previous_refresh not in launches
        assert all(graph not in launches for graph in previous_constants.values())
        assert _set_const_calls(recorded) == []
        # The recaptured graphs still publish correct values.
        ctrl = torch.full((NUM_ENVS, 1), 0.4, device="cuda")
        backend.step_tensor(ctrl, nsteps=2)["completion"].wait_before_read()
        _assert_public_views_match_cpu_forward(backend, _read_public(backend))
    finally:
        backend.close()


@pytest.mark.parametrize("cause", ["ineligible device", "capture failure"])
def test_reinitializing_without_graphs_keeps_every_refresh_eager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    backend, _ = _cuda_backend(tmp_path, f"without-graphs-{cause}")
    try:
        _require_graphs(backend)
        device = backend._warp.get_device()
        if cause == "ineligible device":
            monkeypatch.setattr(
                "unisim.backend.mjwarp.backend._cuda_graph_eligibility",
                lambda warp, device: (False, "forced ineligibility"),
            )
            expected_reason = "forced ineligibility"
        else:
            injected = "injected final-state refresh capture failure"

            def failing_refresh() -> None:
                raise RuntimeError(injected)

            expected_reason = f"capture failed: RuntimeError: {injected}"
        with monkeypatch.context() as patches:
            if cause == "capture failure":
                patches.setattr(backend, "_launch_final_state_refresh", failing_refresh)
            with pytest.warns(RuntimeWarning, match=re.escape(expected_reason)):
                backend._initialize_cuda_graphs(device)

        assert backend._cuda_graph_enabled is False
        assert backend._replay_backend_graphs is False
        assert backend._cuda_graph_disable_reason == expected_reason
        assert backend._step_graph is None
        assert backend._forward_graph is None
        assert backend._reset_graph is None
        assert backend._reset_scratch_reset_graph is None
        assert backend._reset_scratch_forward_graph is None
        assert backend._refresh_graph is None
        assert backend._set_const_graphs == {}

        recorded = _record_routes(backend, monkeypatch)
        ctrl = torch.full((NUM_ENVS, 1), 0.3, device="cuda")
        backend.step_tensor(ctrl, nsteps=2)["completion"].wait_before_read()
        backend.get_tracked_body_views()
        _dr_reset(backend, "mask", 2, _reset_tensors(backend))
        assert recorded["launch"] == []
        assert [name for name, _ in recorded["kernel"]] == [*_FINAL_STATE_KERNELS, "set_const"]
        groups = backend._mujoco_warp.SetConstGroup
        assert _set_const_calls(recorded) == [{"groups": groups.ALL, "restore": False}]
        ctrl = torch.zeros((NUM_ENVS, 1), device="cuda")
        backend.step_tensor(ctrl, nsteps=1)["completion"].wait_before_read()
        _assert_public_views_match_cpu_forward(backend, _read_public(backend))
    finally:
        backend.close()
