"""Behavioral-obligation tests for the mjbatch-backed MuJoCo adapter.

These tests pin the three correctness obligations from unilabsim/unisim#58
plus the native-semantics equivalence gates (unilabsim/unisim#58, plan §8.1):

1. ``xfrc_applied`` is written absolutely before every dispatch and leaves no
   residue after the staged step.
2. Warmstart is zeroed on ``set_state`` and on the velocity-delta path, with
   cross-episode isolation.
3. State layout is per-field bound views, decomposing the ``bind("state")``
   rows by ``mj_stateSize``-derived offsets.

All numeric gates are native semantics (serial ``MjData`` references,
determinism, live-view assertions); there is deliberately no comparison
against the previous executor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mjbatch")

import mujoco

from unisim import MuJoCoBackend, create_backend
from unisim.dr.types import ResetRandomizationPayload
from unisim.scene import SceneCfg

MODEL = """<mujoco model='batch-obligation-test'>
  <option timestep='0.002' iterations='100'/>
  <worldbody>
    <geom name='floor' type='plane' size='5 5 0.1'/>
    <body name='base' pos='0 0 0.25'>
      <freejoint name='root'/>
      <geom name='base_geom' type='box' size='0.05 0.05 0.02'/>
      <site name='base_site' pos='0 0 -0.04'/>
      <body name='link' pos='0 0 -0.06'>
        <joint name='hinge' type='hinge' axis='0 1 0'/>
        <geom name='link_geom' type='capsule' size='0.01 0.05' fromto='0 0 0 0 0 -0.1'/>
        <site name='tip' pos='0 0 -0.1'/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint='hinge' name='hinge_pos' kp='25' ctrlrange='-2 2'/>
  </actuator>
  <sensor>
    <framepos name='base_pos' objtype='body' objname='base'/>
    <frameangvel name='base_angvel' objtype='body' objname='base'/>
    <jointpos name='hinge_pos_s' joint='hinge'/>
    <jointvel name='hinge_vel' joint='hinge'/>
  </sensor>
</mujoco>"""

HFIELD_MODEL = """<mujoco model='batch-hfield-test'>
  <option timestep='0.002'/>
  <asset>
    <hfield name='hf' nrow='9' ncol='9' size='1 1 0.5 0.1'/>
  </asset>
  <worldbody>
    <geom name='terrain' type='hfield' hfield='hf' pos='0.3 -0.2 0.0' euler='0.1 0.2 0.7'/>
    <body name='base' pos='0.1 0.05 0.4'>
      <freejoint name='root'/>
      <geom name='base_geom' type='sphere' size='0.05' mass='1'/>
    </body>
  </worldbody>
  <sensor>
    <framepos name='base_pos' objtype='body' objname='base'/>
  </sensor>
</mujoco>"""


def _write(tmp_path: Path, xml: str) -> str:
    path = tmp_path / "model.xml"
    path.write_text(xml)
    return str(path)


@pytest.fixture
def backend(tmp_path: Path) -> MuJoCoBackend:
    b = MuJoCoBackend(
        SceneCfg(model_file=_write(tmp_path, MODEL)), num_envs=4, sim_dt=0.002, base_name="base"
    )
    b.materialize()
    return b


def _episode_start_state(b: MuJoCoBackend) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.broadcast_to(b.get_default_qpos(), (b.num_envs, b.nq)).copy()
    qvel = np.broadcast_to(b.get_init_qvel(), (b.num_envs, b.nv)).copy()
    return qpos, qvel


# --------------------------------------------------------------------- #
# Obligation 1: xfrc_applied absolute write, no residue                 #
# --------------------------------------------------------------------- #


def test_xfrc_absolute_write_leaves_no_residue(tmp_path: Path) -> None:
    pushed = MuJoCoBackend(
        SceneCfg(model_file=_write(tmp_path, MODEL)), num_envs=2, sim_dt=0.002, base_name="base"
    )
    pushed.materialize()
    reference = MuJoCoBackend(
        SceneCfg(model_file=_write(tmp_path, MODEL)), num_envs=2, sim_dt=0.002, base_name="base"
    )
    reference.materialize()

    base_id = pushed.get_body_ids(["base"])
    wrench = np.zeros((2, 1, 3))
    wrench[0, 0, :] = [10.0, -3.0, 2.0]
    pushed.apply_body_force(base_id, wrench)

    ctrl = np.zeros((2, pushed.num_actuators))
    pushed.step(ctrl)
    reference.step(ctrl)

    # The staged wrench reached env 0's sim channel only, and the staging
    # array is cleared after dispatch.
    xfrc = pushed._xfrc_view.reshape(2, -1)
    column = 6 * int(base_id[0])
    np.testing.assert_allclose(xfrc[0, column : column + 3], wrench[0, 0, :], rtol=1e-12)
    np.testing.assert_allclose(xfrc[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(pushed._pending_xfrc_applied, 0.0, atol=1e-12)
    # The wrench acted for exactly one step: env 0 diverged, env 1 matches the
    # never-pushed reference exactly.
    assert not np.allclose(pushed.get_dof_pos()[0], reference.get_dof_pos()[0])
    np.testing.assert_array_equal(pushed.get_dof_pos()[1], reference.get_dof_pos()[1])

    # An idle step writes zeros everywhere: no residue in the sim channel (the
    # view mirrors the channel via CopyOut).
    pushed.step(ctrl)
    reference.step(ctrl)
    np.testing.assert_allclose(pushed._xfrc_view, 0.0, atol=1e-12)

    # Same-state reset after a push: next states match the never-pushed envs.
    qpos, qvel = _episode_start_state(pushed)
    pushed.set_state(np.arange(2), qpos, qvel)
    reference.set_state(np.arange(2), qpos, qvel)
    pushed.step(ctrl)
    reference.step(ctrl)
    np.testing.assert_array_equal(pushed.get_dof_pos(), reference.get_dof_pos())


def test_xfrc_written_absolutely_on_idle_steps(backend: MuJoCoBackend) -> None:
    ctrl = np.zeros((backend.num_envs, backend.num_actuators))
    backend.step(ctrl)  # idle: nothing staged
    np.testing.assert_allclose(backend._xfrc_view, 0.0, atol=1e-12)
    # Direct view write is overwritten by the next (idle) step.
    backend._xfrc_view[:] = 1.0
    backend.step(ctrl)
    np.testing.assert_allclose(backend._xfrc_view, 0.0, atol=1e-12)


# --------------------------------------------------------------------- #
# Obligation 2: warmstart zeroing                                       #
# --------------------------------------------------------------------- #


def _step_until_warmstart(b: MuJoCoBackend, steps: int = 20) -> None:
    ctrl = np.zeros((b.num_envs, b.num_actuators))
    for _ in range(steps):
        b.step(ctrl)


def test_set_state_zeroes_warmstart(backend: MuJoCoBackend) -> None:
    _step_until_warmstart(backend)
    warm_before = backend._warm_view.copy()
    assert np.any(warm_before != 0.0), "precondition: warmstart is nonzero after stepping"

    ids = np.array([1, 3])
    qpos = np.broadcast_to(backend.get_default_qpos(), (2, backend.nq))
    qvel = np.zeros((2, backend.nv))
    backend.set_state(ids, qpos, qvel)

    np.testing.assert_allclose(backend._warm_view[ids], 0.0, atol=1e-12)
    np.testing.assert_array_equal(backend._warm_view[[0, 2]], warm_before[[0, 2]])
    np.testing.assert_allclose(backend._time_view[ids], 0.0, atol=1e-12)
    # reset ran mj_forward: sensor reads for the reset rows are fresh/finite.
    assert np.isfinite(backend.get_sensor_data("base_pos")).all()


def test_velocity_delta_zeroes_warmstart(backend: MuJoCoBackend) -> None:
    _step_until_warmstart(backend)
    base_id = backend.get_body_ids(["base"])

    warm_before = backend._warm_view.copy()
    delta = np.zeros((backend.num_envs, 1, 3))
    delta[1, 0, 0] = 0.5
    backend._apply_body_velocity_delta(base_id, delta, None)

    # The kicked row's warmstart is zeroed; untouched rows keep their state.
    np.testing.assert_allclose(backend._warm_view[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(backend._warm_view[[0, 2, 3]], warm_before[[0, 2, 3]], rtol=1e-12)
    # The kick itself landed (float32 view rounding tolerated).
    assert abs(backend._qvel_view[1, 0] - 0.5) < 1e-6

    # Whole-population kick: every active row is zeroed.
    backend._apply_body_velocity_delta(base_id, np.ones((backend.num_envs, 1, 3)), None)
    np.testing.assert_allclose(backend._warm_view, 0.0, atol=1e-12)


def test_reset_cross_episode_isolation(backend: MuJoCoBackend) -> None:
    ctrl = np.full((backend.num_envs, backend.num_actuators), 0.3)
    episode_qpos = [backend.get_dof_pos().copy()]
    for _ in range(30):
        backend.step(ctrl)
        episode_qpos.append(backend.get_dof_pos().copy())

    start = backend.get_physics_state()
    qpos = start[:, 1 : 1 + backend.nq]
    qvel = start[:, 1 + backend.nq :]
    backend.set_state(np.arange(backend.num_envs), qpos, qvel)

    np.testing.assert_array_equal(backend.get_dof_pos(), episode_qpos[0])
    for expected in episode_qpos[1:]:
        backend.step(ctrl)
        np.testing.assert_array_equal(backend.get_dof_pos(), expected)


# --------------------------------------------------------------------- #
# Obligation 3: per-field views == bind("state") rows via mj_stateSize  #
# --------------------------------------------------------------------- #


def _independent_layout(model: mujoco.MjModel) -> dict[str, slice]:
    state = mujoco.mjtState
    ordered = (
        ("time", state.mjSTATE_TIME),
        ("qpos", state.mjSTATE_QPOS),
        ("qvel", state.mjSTATE_QVEL),
        ("act", state.mjSTATE_ACT),
        ("history", state.mjSTATE_HISTORY),
        ("qacc_warmstart", state.mjSTATE_WARMSTART),
        ("ctrl", state.mjSTATE_CTRL),
        ("qfrc_applied", state.mjSTATE_QFRC_APPLIED),
        ("xfrc_applied", state.mjSTATE_XFRC_APPLIED),
        ("eq_active", state.mjSTATE_EQ_ACTIVE),
        ("mocap_pos", state.mjSTATE_MOCAP_POS),
        ("mocap_quat", state.mjSTATE_MOCAP_QUAT),
        ("userdata", state.mjSTATE_USERDATA),
        ("plugin_state", state.mjSTATE_PLUGIN),
    )
    slices: dict[str, slice] = {}
    offset = 0
    for name, component in ordered:
        size = int(mujoco.mj_stateSize(model, component))
        slices[name] = slice(offset, offset + size)
        offset += size
    assert offset == int(mujoco.mj_stateSize(model, state.mjSTATE_INTEGRATION))
    return slices


def test_layout_equivalence(tmp_path: Path) -> None:
    # float64 dtype: the equivalence of bind("state") rows and per-field views
    # is exact; float32 rounding is covered by the conformance suite.
    path = _write(tmp_path, MODEL)
    backend = MuJoCoBackend(
        SceneCfg(model_file=path), num_envs=4, sim_dt=0.002, base_name="base", np_dtype=np.float64
    )
    backend.materialize()
    ctrl = np.full((backend.num_envs, backend.num_actuators), 0.1)
    base_id = backend.get_body_ids(["base"])
    force = np.zeros((backend.num_envs, 1, 3))
    force[:, 0, 1] = 0.5
    backend.apply_body_force(base_id, force)
    backend.step(ctrl, nsteps=2)

    rows = backend._state_view
    layout = backend._state_layout
    independent = _independent_layout(backend.model)

    assert rows.shape == (backend.num_envs, layout.nstate)
    for name in ("time", "qpos", "qvel", "act", "ctrl", "xfrc_applied", "qacc_warmstart"):
        assert (layout_slice := getattr(layout, name)) == independent[name], name
        actual = rows[:, layout_slice]
        view = {
            "time": backend._time_view.reshape(-1, 1),
            "qpos": backend._qpos_view,
            "qvel": backend._qvel_view,
            "act": backend._act_view,
            "ctrl": backend._ctrl_view,
            "xfrc_applied": backend._xfrc_view.reshape(backend.num_envs, -1),
            "qacc_warmstart": backend._warm_view,
        }[name]
        np.testing.assert_allclose(actual, view, rtol=1e-12, atol=1e-14, err_msg=name)


def test_get_physics_state_layout(backend: MuJoCoBackend) -> None:
    backend.step(np.full((backend.num_envs, backend.num_actuators), 0.2), nsteps=3)
    snapshot = backend.get_physics_state()
    assert snapshot.shape == (backend.num_envs, 1 + backend.nq + backend.nv)
    assert snapshot.dtype == backend._np_dtype
    np.testing.assert_allclose(snapshot[:, 0], backend._time_view)
    np.testing.assert_array_equal(snapshot[:, 1 : 1 + backend.nq], backend._qpos_view)
    np.testing.assert_array_equal(snapshot[:, 1 + backend.nq :], backend._qvel_view)


# --------------------------------------------------------------------- #
# set_state DR roundtrip via expand/set_const                           #
# --------------------------------------------------------------------- #


def test_set_state_dr_roundtrip(backend: MuJoCoBackend) -> None:
    model = backend.model
    ids = np.array([1, 3])
    others = np.array([0, 2])
    n = len(ids)

    base_mass = backend.get_body_mass()
    base_ipos = backend.get_body_ipos()
    base_friction = backend.get_geom_friction()
    base_armature = backend.get_dof_armature()
    base_kp, base_kd = backend.get_actuator_gains()

    mass_rows = np.broadcast_to(base_mass, (n, model.nbody)).copy()
    mass_rows[:, backend._base_body_id] += np.array([0.7, 1.2])
    payload = ResetRandomizationPayload(
        base_mass_delta=np.array([0.7, 1.2]),
        base_com_offset=np.array([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]]),
        gravity=np.array([[0.0, 0.0, -7.0], [0.0, 0.0, -11.0]]),
        body_iquat=np.tile(np.array([[[1.0, 0.0, 0.0, 0.0]]]), (n, model.nbody, 1)),
        body_inertia=np.broadcast_to(np.array([0.01, 0.01, 0.01]), (n, model.nbody, 3)).copy(),
        geom_friction=np.broadcast_to(base_friction, (n, model.ngeom, 3)).copy(),
        dof_armature=np.broadcast_to(base_armature + 0.005, (n, backend.nv)).copy(),
        kp=np.broadcast_to(base_kp * 1.5, (n, model.nu)).copy(),
        kd=np.broadcast_to(base_kd * 0.5, (n, model.nu)).copy(),
    )

    qpos = np.broadcast_to(backend.get_default_qpos(), (n, backend.nq))
    qvel = np.zeros((n, backend.nv))
    backend.set_state(ids, qpos, qvel, randomization=payload)

    pool = backend._pool
    np.testing.assert_allclose(
        pool.expand("body_mass")[ids], mass_rows, rtol=1e-12
    )
    np.testing.assert_allclose(
        pool.expand("body_mass")[others],
        np.broadcast_to(base_mass, (len(others), model.nbody)),
        rtol=1e-12,
    )
    expected_ipos = np.broadcast_to(base_ipos, (n, model.nbody, 3)).copy()
    expected_ipos[:, backend._base_body_id, :] += payload.base_com_offset
    np.testing.assert_allclose(pool.expand("body_ipos")[ids], expected_ipos, rtol=1e-12)
    np.testing.assert_allclose(pool.expand("gravity")[ids], payload.gravity, rtol=1e-12)
    np.testing.assert_allclose(
        pool.expand("body_iquat")[ids], payload.body_iquat, rtol=1e-12
    )
    np.testing.assert_allclose(
        pool.expand("body_inertia")[ids], payload.body_inertia, rtol=1e-12
    )
    np.testing.assert_allclose(
        pool.expand("geom_friction")[ids], payload.geom_friction, rtol=1e-12
    )
    np.testing.assert_allclose(
        pool.expand("dof_armature")[ids], payload.dof_armature, rtol=1e-12
    )
    # kp/kd map onto the position-actuator gain/bias parameters.
    gain = pool.expand("actuator_gainprm")
    bias = pool.expand("actuator_biasprm")
    np.testing.assert_allclose(gain[ids, :, 0], payload.kp, rtol=1e-12)
    np.testing.assert_allclose(bias[ids, :, 1], -payload.kp, rtol=1e-12)
    np.testing.assert_allclose(bias[ids, :, 2], -payload.kd, rtol=1e-12)
    np.testing.assert_allclose(
        gain[others, :, 0],
        np.broadcast_to(base_kp, (len(others), model.nu)),
        rtol=1e-12,
    )

    # The fields reached physics: a randomized env diverges from a plain-reset
    # sibling started at the same state (set_const ran before the forward).
    plain = MuJoCoBackend(
        SceneCfg(model_file=str(backend.scene_model_file)),
        num_envs=2,
        sim_dt=0.002,
        base_name="base",
    )
    plain.materialize()
    q0 = np.broadcast_to(backend.get_default_qpos(), (2, backend.nq))
    v0 = np.zeros((2, backend.nv))
    plain.set_state(np.arange(2), q0, v0)
    randomized = MuJoCoBackend(
        SceneCfg(model_file=str(backend.scene_model_file)),
        num_envs=2,
        sim_dt=0.002,
        base_name="base",
    )
    randomized.materialize()
    randomized.set_state(
        np.arange(2),
        q0,
        v0,
        randomization=ResetRandomizationPayload(
            gravity=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
        ),
    )
    ctrl = np.zeros((2, backend.num_actuators))
    for _ in range(20):
        randomized.step(ctrl)
        plain.step(ctrl)
    assert not np.allclose(randomized.get_base_pos(), plain.get_base_pos())

    # Persistence: expanded values survive a later plain reset (the old
    # per-env patch semantics).
    backend.set_state(ids, qpos, qvel)
    np.testing.assert_allclose(pool.expand("gravity")[ids], payload.gravity, rtol=1e-12)


def test_set_state_empty_indices_returns_timing(backend: MuJoCoBackend) -> None:
    result = backend.set_state(np.array([], dtype=np.int32), np.zeros((0, backend.nq)),
                               np.zeros((0, backend.nv)))
    assert result["timing"]["set_state_pool_reset_ms"] == 0.0
    assert "set_state_qpos_convert_ms" in result["timing"]
    assert "set_state_state_scatter_ms" in result["timing"]


# --------------------------------------------------------------------- #
# Query-op sensordata clobber guard + serial equivalence                #
# --------------------------------------------------------------------- #


def test_jac_site_matches_serial_reference(backend: MuJoCoBackend) -> None:
    rng = np.random.default_rng(0)
    qpos = np.broadcast_to(backend.get_default_qpos(), (backend.num_envs, backend.nq)).copy()
    qpos[:, :2] += rng.uniform(-0.1, 0.1, (backend.num_envs, 2))
    qvel = rng.uniform(-0.5, 0.5, (backend.num_envs, backend.nv))
    backend.set_state(np.arange(backend.num_envs), qpos, qvel)

    site_id = int(backend.get_site_ids(["tip"])[0])
    dof_indices = np.arange(backend.nv)
    jacp, jacr = backend.get_site_jacobian_w(site_id, dof_indices)
    assert jacp.shape == (backend.num_envs, 3, backend.nv)

    model = backend.model
    for i in range(backend.num_envs):
        data = mujoco.MjData(model)
        data.qpos[:] = backend._qpos_view[i]
        data.qvel[:] = backend._qvel_view[i]
        mujoco.mj_forward(model, data)
        jp, jr = np.zeros((3, backend.nv)), np.zeros((3, backend.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        np.testing.assert_allclose(jacp[i], jp, atol=1e-12)
        np.testing.assert_allclose(jacr[i], jr, atol=1e-12)


def test_jac_site_preserves_sensor_data(backend: MuJoCoBackend) -> None:
    ctrl = np.full((backend.num_envs, backend.num_actuators), 0.4)
    site_id = int(backend.get_site_ids(["base_site"])[0])
    dof_indices = np.arange(backend.nv)
    names = tuple(backend._sensor_views)

    backend.step(ctrl)
    snapshots = []
    for _ in range(2):
        before = {name: backend.get_sensor_data(name).copy() for name in names}
        backend.get_site_jacobian_w(site_id, dof_indices)
        for name in names:
            np.testing.assert_array_equal(backend.get_sensor_data(name), before[name])
        backend.step(ctrl)
        snapshots.append({name: backend.get_sensor_data(name).copy() for name in names})

    # A control run without the interleaved jac calls sees the same sensors.
    control = MuJoCoBackend(
        SceneCfg(model_file=str(backend.scene_model_file)),
        num_envs=backend.num_envs,
        sim_dt=0.002,
        base_name="base",
    )
    control.materialize()
    for _ in range(3):
        control.step(ctrl)
    for name in names:
        np.testing.assert_allclose(
            control.get_sensor_data(name), snapshots[-1][name], rtol=1e-12, atol=1e-14
        )


# --------------------------------------------------------------------- #
# Height scanner: yaw semantics (native serial reference) + guard       #
# --------------------------------------------------------------------- #


def _hfield_reference_height(
    model: mujoco.MjModel,
    qpos_row: np.ndarray,
    geom: int,
    body: int,
    offsets: np.ndarray,
) -> np.ndarray:
    """Serial reference for alignment='yaw', output='height' (world z)."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_row
    mujoco.mj_forward(model, data)
    hfield = model.geom_dataid[geom]
    nrow, ncol = model.hfield_nrow[hfield], model.hfield_ncol[hfield]
    size = model.hfield_size[hfield]
    grid = model.hfield_data[hfield * nrow * ncol : (hfield + 1) * nrow * ncol].reshape(nrow, ncol)
    gpos, gmat = data.geom_xpos[geom], data.geom_xmat[geom].reshape(3, 3)
    bpos = data.xpos[body]
    bmat = data.xmat[body].reshape(3, 3)
    yaw = np.arctan2(bmat[1, 0], bmat[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.zeros(len(offsets))
    for k, (ox, oy) in enumerate(offsets):
        w = bpos + np.array([c * ox - s * oy, s * ox + c * oy, 0.0])
        w[2] = gpos[2]
        lp = gmat.T @ (w - gpos)
        fx = np.clip((lp[0] / size[0] + 1.0) * 0.5 * (ncol - 1), 0, ncol - 1.001)
        fy = np.clip((lp[1] / size[1] + 1.0) * 0.5 * (nrow - 1), 0, nrow - 1.001)
        ix, iy = int(fx), int(fy)
        sx, sy = fx - ix, fy - iy
        h = (
            (1 - sx) * (1 - sy) * grid[iy, ix]
            + sx * (1 - sy) * grid[iy, min(ix + 1, ncol - 1)]
            + (1 - sx) * sy * grid[min(iy + 1, nrow - 1), ix]
            + sx * sy * grid[min(iy + 1, nrow - 1), min(ix + 1, ncol - 1)]
        ) * size[2]
        out[k] = gpos[2] + (gmat @ np.array([lp[0], lp[1], h]))[2]
    return out


def test_scanner_yaw_heights_match_serial_reference(tmp_path: Path) -> None:
    path = _write(tmp_path, HFIELD_MODEL)
    # float64 dtype: scan() output must match the float64 serial reference
    # exactly; float32 rounding is a separate, coarser gate.
    b = MuJoCoBackend(
        SceneCfg(model_file=path), num_envs=4, sim_dt=0.002, base_name="base", np_dtype=np.float64
    )
    geom_id = b.get_geom_id("terrain")
    body_id = int(b.get_body_ids(["base"])[0])
    offsets = np.array([[i * 0.11, j * 0.13] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    scanner = b.create_hfield_scanner(
        hfield_geom_id=geom_id,
        offsets=offsets,
        frame_body_id=body_id,
        alignment="yaw",
        output="height",
    )

    # Pre-materialize (transient pool) at the default state.
    expected0 = [
        _hfield_reference_height(b.model, b.get_default_qpos(), geom_id, body_id, offsets)
    ] * b.num_envs
    np.testing.assert_allclose(scanner.scan(), expected0, rtol=1e-12, atol=1e-14)

    b.materialize()
    rng = np.random.default_rng(5)
    qpos = np.broadcast_to(b.get_default_qpos(), (b.num_envs, b.nq)).copy()
    yaw = rng.uniform(-np.pi, np.pi, b.num_envs)
    qpos[:, 3:7] = np.stack(
        [np.cos(yaw / 2), np.zeros(b.num_envs), np.zeros(b.num_envs), np.sin(yaw / 2)], axis=1
    )
    b.set_state(np.arange(b.num_envs), qpos, np.zeros((b.num_envs, b.nv)))

    expected = np.stack(
        [_hfield_reference_height(b.model, qpos[i], geom_id, body_id, offsets) for i in range(4)]
    )
    np.testing.assert_allclose(scanner.scan(), expected, rtol=1e-12, atol=1e-14)


def test_scanner_preserves_sensor_data(tmp_path: Path) -> None:
    path = _write(tmp_path, HFIELD_MODEL)
    b = MuJoCoBackend(
        SceneCfg(model_file=path), num_envs=4, sim_dt=0.002, base_name="base", np_dtype=np.float64
    )
    b.materialize()
    scanner = b.create_hfield_scanner(
        hfield_geom_id=b.get_geom_id("terrain"),
        offsets=np.zeros((2, 2)),
        frame_body_id=int(b.get_body_ids(["base"])[0]),
        alignment="yaw",
        output="height",
    )
    b.step(np.zeros((b.num_envs, b.num_actuators)))
    names = tuple(b._sensor_views)
    assert names, "hfield model carries no sensors; guard untestable"
    before = {name: b.get_sensor_data(name).copy() for name in names}
    scanner.scan()
    for name in names:
        np.testing.assert_array_equal(b.get_sensor_data(name), before[name])


# --------------------------------------------------------------------- #
# Callback path: per-substep control vs serial MjData reference         #
# --------------------------------------------------------------------- #


def test_callback_path_equivalence_vs_serial(tmp_path: Path) -> None:
    # float64 dtype makes the bound views bit-comparable to a float64 serial
    # reference; the equivalence gate is exact native semantics.
    path = _write(tmp_path, MODEL)
    b = MuJoCoBackend(
        SceneCfg(model_file=path), num_envs=3, sim_dt=0.002, base_name="base", np_dtype=np.float64
    )
    b.materialize()

    seen_qpos: list[np.ndarray] = []
    calls = {"n": 0}

    def hook(backend: MuJoCoBackend, ctrl: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        q = backend.get_dof_pos()
        v = backend.get_dof_vel()
        seen_qpos.append(q.copy())
        return (-2.0 * q - 0.1 * v).astype(ctrl.dtype)

    b.set_pre_step_control(hook)
    # Start away from equilibrium so the hook produces nonzero control.
    qpos0 = np.broadcast_to(b.get_default_qpos(), (b.num_envs, b.nq)).copy()
    qpos0[:, -1] = 0.5  # hinge angle
    b.set_state(np.arange(b.num_envs), qpos0, np.zeros((b.num_envs, b.nv)))
    pre_step_qpos = b.get_dof_pos().copy()
    ctrl = np.zeros((b.num_envs, b.num_actuators))
    nsteps = 4
    b.step(ctrl, nsteps=nsteps)

    assert calls["n"] == nsteps
    # k=0 sees the pre-step state (no cache refresh before the first substep);
    # later calls see the post-substep state.
    np.testing.assert_array_equal(seen_qpos[0], pre_step_qpos)
    assert not np.allclose(seen_qpos[1], seen_qpos[0])

    # Serial reference: same hook recomputed per substep on one MjData per env.
    model = b.model
    for i in range(b.num_envs):
        data = mujoco.MjData(model)
        data.qpos[:] = b.get_default_qpos()
        data.qpos[-1] = 0.5  # hinge angle, matching the batch's reset state
        data.qvel[:] = 0.0
        for _ in range(nsteps):
            q = data.qpos[b._root_qpos_dim :]
            v = data.qvel[b._root_qvel_dim :]
            data.ctrl[:] = -2.0 * q - 0.1 * v
            mujoco.mj_step(model, data)
        np.testing.assert_allclose(b._qpos_view[i], data.qpos, atol=1e-13)
        np.testing.assert_allclose(b._qvel_view[i], data.qvel, atol=1e-13)

    # Sensors are refreshed at the end of the call: an explicit forward brings
    # the batch's one-substep-behind sensordata current for comparison.
    b._pool.forward()
    for i in range(b.num_envs):
        data = mujoco.MjData(model)
        data.qpos[:] = b._qpos_view[i]
        data.qvel[:] = b._qvel_view[i]
        mujoco.mj_forward(model, data)
        for name, view in b._sensor_views.items():
            adr = model.sensor_adr[model.sensor(name).id]
            dim = model.sensor_dim[model.sensor(name).id]
            np.testing.assert_allclose(view[i], data.sensordata[adr : adr + dim], atol=1e-12)


# --------------------------------------------------------------------- #
# Factory surface: warn-and-ignore shims, cpu_ids passthrough           #
# --------------------------------------------------------------------- #


def test_factory_warns_and_ignores_removed_options(tmp_path: Path) -> None:
    path = _write(tmp_path, MODEL)
    scene = SceneCfg(model_file=path)
    with pytest.warns(DeprecationWarning, match="post_step_forward_sensor"):
        backend = create_backend(
            "mujoco",
            scene,
            num_envs=1,
            sim_dt=0.002,
            base_name="base",
            post_step_forward_sensor=True,
            chunk_size=64,
            bench_nsteps=8,
        )
    backend.materialize()
    backend.step(np.zeros((1, backend.num_actuators)))


def test_factory_cpu_ids_passthrough(tmp_path: Path) -> None:
    path = _write(tmp_path, MODEL)
    backend = create_backend(
        "mujoco", SceneCfg(model_file=path), num_envs=1, sim_dt=0.002, base_name="base",
        cpu_ids=[0],
    )
    assert backend._cpu_ids == (0,)
    assert backend._n_threads == 1


def test_ctor_chunk_knobs_warn_and_ignore(tmp_path: Path) -> None:
    path = _write(tmp_path, MODEL)
    with pytest.warns(DeprecationWarning, match="chunk_size"):
        backend = MuJoCoBackend(
            SceneCfg(model_file=path), num_envs=1, sim_dt=0.002, chunk_size=32,
            adaptive_chunk_size=True,
        )
    backend.materialize()
    backend.step(np.zeros((1, backend.num_actuators)))
