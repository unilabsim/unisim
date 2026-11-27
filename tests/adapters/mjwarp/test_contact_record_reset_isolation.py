"""Reset isolation of MJWarp contact-record sensors evaluated by the pos/vel kernels.

MuJoCo-Warp 4.0 evaluates contact sensors that need only the position or
velocity stage, and the geom distance, normal, and from-to sensors, from the
live contact records inside ``sensor_pos`` and ``sensor_vel``.  The backend's
lazy final-state refresh runs those kernels on the first read after a step or a
reset, and a reset leaves the contact records unusable: MuJoCo-Warp's masked
``reset_data`` drops every active contact whenever world 0 is reset.  Between a
reset and the next step or forward the backend therefore keeps completed values
on untouched rows and cleared values on reset rows.
"""

# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
warp = pytest.importorskip("warp")
torch = pytest.importorskip("torch")

from unisim import MjwarpBackend
from unisim.backend.mjwarp.backend import _MjwarpContactRecordGuard
from unisim.scene import SceneCfg

MODEL = """<mujoco model='unisim-test-mjwarp-contact-record-reset'>
  <option timestep='0.005' gravity='0 0 -9.81'/>
  <worldbody>
    <geom name='floor' type='plane' size='2 2 .1'/>
    <body name='box' pos='0 0 .2' quat='0.9238795 0 .3826834 0'>
      <freejoint name='root'/>
      <site name='box_site' pos='.01 .02 .03'/>
      <geom name='box_geom' type='box' size='.1 .1 .1' mass='1'/>
    </body>
  </worldbody>
  <sensor>
    <contact name='box_floor_found' geom1='box_geom' geom2='floor' data='found'/>
    <distance name='box_floor_dist' geom1='box_geom' geom2='floor'/>
    <framepos name='box_pos' objtype='body' objname='box'/>
    <framelinvel name='box_linvel' objtype='body' objname='box'/>
    <accelerometer name='box_acc' site='box_site'/>
  </sensor>
</mujoco>"""
KINEMATIC_ONLY_MODEL = MODEL.replace(
    """    <contact name='box_floor_found' geom1='box_geom' geom2='floor' data='found'/>
    <distance name='box_floor_dist' geom1='box_geom' geom2='floor'/>
""",
    "",
)
FOUND, DISTANCE = "box_floor_found", "box_floor_dist"
GUARDED = (FOUND, DISTANCE)
NUM_ENVS = 4


def _backend(tmp_path: Path, name: str, *, num_envs: int = NUM_ENVS, xml: str = MODEL):
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp contact-record reset tests require an active CUDA Warp device")
    model_file = tmp_path / f"{name}.xml"
    model_file.write_text(xml)
    return MjwarpBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=num_envs,
        sim_dt=0.005,
        nconmax=128,
        njmax=256,
    )


def _require_graphs(backend: MjwarpBackend) -> None:
    if not backend._cuda_graph_enabled:
        reason = backend._cuda_graph_disable_reason or "unknown reason"
        pytest.skip(f"test requires mjwarp CUDA graphs; graphs disabled: {reason}")


def _settle(backend: MjwarpBackend) -> None:
    """Drop every box onto the floor along an edge, so each world has contacts."""
    ctrl = torch.zeros((backend.num_envs, 0), dtype=torch.float32, device="cuda")
    for _ in range(40):
        backend.step_tensor(ctrl, nsteps=2)["completion"].wait_before_read()
    torch.cuda.synchronize()


def _named(backend: MjwarpBackend) -> dict[str, Any]:
    return {name: backend.get_sensor_view(name)[:, 0].clone() for name in GUARDED}


def _read_every_path(backend: MjwarpBackend) -> dict[str, dict[str, Any]]:
    """Read the guarded sensors through the named, raw-block, and host paths."""
    reads = {"named": _named(backend)}
    block = backend.get_state_views(("qpos", "qvel", "sensordata"))["sensordata"]
    reads["block"] = {name: block[:, backend._sensor_slots[name][0]].clone() for name in GUARDED}
    backend._sync_host_cache()
    device = reads["named"][FOUND].device
    reads["host"] = {
        name: torch.as_tensor(
            backend._sensor_cache[:, backend._sensor_slots[name][0]].copy(), device=device
        )
        for name in GUARDED
    }
    return reads


def _lift(backend: MjwarpBackend, selected: Any) -> tuple[Any, Any]:
    """Move ``selected`` rows into the air, away from the floor."""
    state = backend.get_state_views(("qpos", "qvel"))
    qpos, qvel = state["qpos"].clone(), state["qvel"].clone()
    qpos[selected, 2] = 0.55
    qpos[selected, 0] += 0.125
    qvel[selected, 0] = 0.375
    qvel[selected, 2] = 0.075
    return qpos, qvel


def _assert_reset_isolation(
    before: dict[str, Any],
    retained: Any,
    reads: dict[str, dict[str, Any]],
    *,
    distance_recomputed: bool = False,
) -> None:
    """Untouched rows keep completed values and reset rows read as cleared.

    A full forward re-evaluates the penetration depth of untouched rows, so
    there it only has to keep reporting contact.
    """
    for values in reads.values():
        torch.testing.assert_close(values[FOUND], before[FOUND] * retained, rtol=0, atol=0)
        if distance_recomputed:
            assert bool((values[DISTANCE][retained] < 0.0).all())
            assert bool((values[DISTANCE][~retained] == 0.0).all())
        else:
            torch.testing.assert_close(
                values[DISTANCE], before[DISTANCE] * retained, rtol=0, atol=0
            )


def test_guard_covers_exactly_the_contact_record_sensors(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "guard-binding")
    kinematic_only = _backend(tmp_path, "guard-none", xml=KINEMATIC_ONLY_MODEL)
    try:
        # ``box_floor_found`` is a position-stage contact sensor and
        # ``box_floor_dist`` a collision sensor; the accelerometer is solved
        # in the acceleration pass and is not guarded.
        assert backend._contact_record_guard.columns.tolist() == [0, 1]
        assert backend._dynamic_sensor_rows.tolist() == [8, 9, 10]
        assert backend._contact_record_guard.preserved.tolist() == [True] * NUM_ENVS
        assert backend._contact_record_frozen is False
        assert kinematic_only._contact_record_guard is None
    finally:
        backend.close()
        kinematic_only.close()


@pytest.mark.parametrize(
    "mask_values",
    [
        [True, False, False, False],
        [False, True, False, False],
        [True, True, False, True],
        [True] * NUM_ENVS,
    ],
    ids=["world-0", "no-world-0", "world-0-and-3", "all"],
)
def test_mask_reset_keeps_untouched_contact_record_sensors(
    tmp_path: Path, mask_values: list[bool]
) -> None:
    backend = _backend(tmp_path, f"mask-{''.join(str(int(v)) for v in mask_values)}")
    try:
        _settle(backend)
        before = _named(backend)
        assert before[FOUND].tolist() == [2.0] * NUM_ENVS
        assert bool((before[DISTANCE] < 0.0).all())
        mask = torch.tensor(mask_values, dtype=torch.bool, device="cuda")
        qpos, qvel = _lift(backend, mask)

        backend.set_state_tensor_mask(mask, qpos, qvel)

        _assert_reset_isolation(before, ~mask, _read_every_path(backend))
    finally:
        backend.close()


@pytest.mark.parametrize("num_envs", [NUM_ENVS, 1024], ids=["lazy-refresh", "scratch-publication"])
def test_selected_reset_rows_including_world_zero_keep_untouched_contact_record_sensors(
    tmp_path: Path, num_envs: int
) -> None:
    backend = _backend(tmp_path, f"selected-{num_envs}", num_envs=num_envs)
    try:
        _settle(backend)
        before = _named(backend)
        assert before[FOUND].tolist() == [2.0] * num_envs
        rows = torch.tensor([0, 1, num_envs - 1], dtype=torch.int64, device="cuda")
        retained = torch.ones((num_envs,), dtype=torch.bool, device="cuda")
        retained[rows] = False
        qpos, qvel = _lift(backend, rows)

        backend.set_state_tensor(rows, qpos[rows], qvel[rows])

        _assert_reset_isolation(before, retained, _read_every_path(backend))
    finally:
        backend.close()


def test_rows_reset_earlier_stay_cleared_across_later_resets(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "successive")
    try:
        _settle(backend)
        before = _named(backend)
        for row in (1, 0):
            rows = torch.tensor([row], dtype=torch.int64, device="cuda")
            qpos, qvel = _lift(backend, rows)
            backend.set_state_tensor(rows, qpos[rows], qvel[rows])
        retained = torch.tensor([False, False, True, True], device="cuda")

        _assert_reset_isolation(before, retained, _read_every_path(backend))
    finally:
        backend.close()


def test_step_rebuilds_contact_records_and_ends_the_guard(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "step-rearms")
    try:
        _settle(backend)
        before = _named(backend)
        mask = torch.tensor([True, False, False, False], dtype=torch.bool, device="cuda")
        qpos, qvel = _lift(backend, mask)
        backend.set_state_tensor_mask(mask, qpos, qvel)
        assert backend._contact_record_frozen is True
        _assert_reset_isolation(before, ~mask, _read_every_path(backend))

        # Put the reset world back on the floor: the next step rebuilds the
        # contact records, and its sensors describe the new state.
        state = backend.get_state_views(("qpos", "qvel"))
        qpos, qvel = state["qpos"].clone(), state["qvel"].clone()
        qpos[0, 2] = qpos[1, 2]
        qvel[0] = 0.0
        backend.set_state_tensor_mask(mask, qpos, qvel)
        ctrl = torch.zeros((NUM_ENVS, 0), dtype=torch.float32, device="cuda")
        backend.step_tensor(ctrl, nsteps=1)["completion"].wait_before_read()

        assert backend._contact_record_frozen is False
        assert backend._contact_record_guard.preserved.tolist() == [True] * NUM_ENVS
        after = _read_every_path(backend)
        assert after["named"][FOUND][0] > 0
        torch.testing.assert_close(after["named"][FOUND][1:], before[FOUND][1:], rtol=0, atol=0)
        for source in after.values():
            assert bool((source[DISTANCE] < 0.0).all())
    finally:
        backend.close()


def test_full_refresh_mask_reset_needs_no_guard(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "full-refresh")
    try:
        _settle(backend)
        before = _named(backend)
        mask = torch.tensor([True, False, False, False], dtype=torch.bool, device="cuda")
        qpos, qvel = _lift(backend, mask)

        backend.set_state_tensor_mask(mask, qpos, qvel, full_refresh=True)

        # The eager forward rebuilt the contact records of every world.
        assert backend._contact_record_frozen is False
        assert backend._contact_record_guard.preserved.tolist() == [True] * NUM_ENVS
        _assert_reset_isolation(
            before, ~mask, _read_every_path(backend), distance_recomputed=True
        )
    finally:
        backend.close()


class _Cycle:
    """One training step: physics, a selected-row reset to the resting pose, and reads.

    Every reset returns its rows to the pose captured at construction, so the
    rows are on the floor again at the following step.
    """

    def __init__(self, backend: MjwarpBackend, *, reset_first: bool) -> None:
        self.backend = backend
        self.reset_first = reset_first
        state = backend.get_state_views(("qpos", "qvel"))
        self.mask = torch.zeros((NUM_ENVS,), dtype=torch.bool, device="cuda")
        self.qpos = state["qpos"].clone()
        self.qvel = torch.zeros_like(state["qvel"])
        # Captured reads land in this owner-owned storage.
        self.outputs = {
            name: torch.zeros((NUM_ENVS,), dtype=torch.float32, device="cuda") for name in GUARDED
        }

    def load(self, k: int) -> None:
        rows = torch.arange(NUM_ENVS, device="cuda")
        self.mask.copy_((rows + k) % 2 == 0)

    def _reset(self) -> None:
        self.backend.set_state_tensor_mask(self.mask, self.qpos, self.qvel)

    def _read(self) -> None:
        block = self.backend.get_state_views(("qpos", "qvel", "sensordata"))["sensordata"]
        for name in GUARDED:
            self.outputs[name].copy_(block[:, self.backend._sensor_slots[name][0]])

    def __call__(self) -> None:
        ctrl = torch.zeros((NUM_ENVS, 0), dtype=torch.float32, device="cuda")
        if self.reset_first:
            self._reset()
            self.backend.step_tensor(ctrl, nsteps=1)
            self._read()
        else:
            self.backend.step_tensor(ctrl, nsteps=1)
            self._reset()
            self._read()


@pytest.mark.parametrize("reset_first", [False, True], ids=["reset-last", "reset-first"])
def test_external_capture_keeps_contact_record_sensors_in_step(
    tmp_path: Path, reset_first: bool
) -> None:
    """A captured step, reset, and read replays to the eager contact sensors."""
    backend = _backend(tmp_path, f"external-{int(reset_first)}")
    eager = _backend(tmp_path, f"external-eager-{int(reset_first)}")
    try:
        _require_graphs(backend)
        eager._cuda_graph_enabled = False
        _settle(backend)
        _settle(eager)
        before = _named(eager)
        assert before[FOUND].tolist() == [2.0] * NUM_ENVS
        inputs = _Cycle(backend, reset_first=reset_first)
        eager_inputs = _Cycle(eager, reset_first=reset_first)
        for operands in (inputs, eager_inputs):
            operands.load(0)
            operands()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with backend.torch_graph_capture_scope() as capture:
                inputs()
        # Capture records without executing device work; its host bookkeeping
        # already ran once and stands in for the first replay's.
        for k in range(1, 7):
            inputs.load(k)
            eager_inputs.load(k)
            capture.check_replay_guards()
            graph.replay()
            if k > 1:
                capture.replay_host_transitions()
            backend.flush_tensor_validation()
            eager_inputs()
            torch.cuda.synchronize()
            # Untouched worlds keep their contacts through every replay.  With
            # the reset first, its rows are back on the floor before the step
            # and read as in contact again; otherwise they read as cleared.
            expected = before[FOUND] if reset_first else before[FOUND] * ~inputs.mask
            torch.testing.assert_close(inputs.outputs[FOUND], expected, rtol=0, atol=0)
            torch.testing.assert_close(
                inputs.outputs[FOUND], eager_inputs.outputs[FOUND], rtol=0, atol=0
            )
            torch.testing.assert_close(
                inputs.outputs[DISTANCE], eager_inputs.outputs[DISTANCE], rtol=2e-5, atol=1e-7
            )
    finally:
        backend.close()
        eager.close()


def test_guard_mechanics_on_the_host_device() -> None:
    """The stash and restore need no GPU: a corrupting kernel stands in for MuJoCo-Warp."""
    backend = MjwarpBackend.__new__(MjwarpBackend)
    sensordata = torch.tensor([[2.0, -0.25, 0.0]] * 4)
    backend._state_torch_views = {"sensordata": sensordata}
    backend._device_model = object()
    backend._device_data = object()
    backend._external_capture = None
    backend._cuda_graph_enabled = False
    backend._refresh_graph = None
    backend._set_const_graphs = {}
    stash = torch.zeros((4, 2))
    backend._contact_record_guard = _MjwarpContactRecordGuard(
        columns=torch.tensor([0, 1]),
        preserved=torch.ones((4,), dtype=torch.bool),
        stash=stash,
        values=torch.zeros((4, 2)),
        cleared=torch.zeros((4, 2)),
    )
    backend._contact_record_frozen = False

    def noop(*args: object, **kwargs: object) -> None:
        return None

    def sensor_pos(model: object, data: object) -> None:
        sensordata[:, 0].fill_(7.0)

    backend._mujoco_warp = SimpleNamespace(  # type: ignore[misc]
        kinematics=noop, com_pos=noop, com_vel=noop, sensor_pos=sensor_pos, sensor_vel=noop
    )
    backend._warp = SimpleNamespace()  # type: ignore[misc]

    # Outside a reset the refresh result stands.
    backend._launch_final_state_refresh()
    assert sensordata[:, 0].tolist() == [7.0] * 4

    sensordata[:, 0].fill_(2.0)
    backend._mark_contact_record_rows_reset(torch.tensor([True, False, False, True]))
    backend._freeze_contact_record_sensors()
    backend._launch_final_state_refresh()
    assert sensordata.tolist() == [
        [0.0, 0.0, 0.0],
        [2.0, -0.25, 0.0],
        [2.0, -0.25, 0.0],
        [0.0, 0.0, 0.0],
    ]

    # Reset rows accumulate, including when a host reset passes NumPy rows.
    backend._mark_contact_record_rows_reset(np.array([2]))
    backend._launch_final_state_refresh()
    assert sensordata.tolist() == [
        [0.0, 0.0, 0.0],
        [2.0, -0.25, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

    backend._rearm_contact_record_sensors()
    assert backend._contact_record_frozen is False
    assert backend._contact_record_guard.preserved.tolist() == [True] * 4
