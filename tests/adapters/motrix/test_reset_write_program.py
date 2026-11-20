"""Reset-randomization parity for the MotrixSim ``WriteProgram`` batching path.

Upstream ``motrixsim#20`` provides one-call batched overrides through
``SceneModel.compile_write`` write programs.  These tests lock the adapter
behavior on top of that API: selected rows receive the payload, unselected
rows and payload-absent terms keep model defaults, and world-fixed links stay
outside the mass/COM program columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")
pytest.importorskip("mujoco")

import torch  # noqa: E402

from unisim import MotrixBackend  # noqa: E402
from unisim.dr.types import (  # noqa: E402
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    FixedVariantLayout,
    FixedVariantPlan,
    ModelSourceDescriptor,
    TensorResetRandomizationPayload,
)
from unisim.scene import SceneCfg  # noqa: E402

MODEL = """<mujoco model='motrix-reset-write-program'>
  <option timestep='0.002' gravity='0 0 -9.81'/>
  <worldbody>
    <body name='base' pos='0 0 1'>
      <freejoint name='root'/>
      <inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>
      <geom name='visual_geom' type='sphere' size='.10' contype='0' conaffinity='0'
            friction='.70 .006 .0001'/>
      <geom name='base_geom' type='sphere' size='.08'/>
      <body name='link' pos='0 0 .2'>
        <joint name='drive' type='hinge' axis='0 1 0'/>
        <inertial pos='0 0 0' mass='.2' diaginertia='.02 .02 .02'/>
        <geom name='link_geom' type='sphere' size='.03'/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name='drive' joint='drive' kp='10' kv='1'/>
  </actuator>
</mujoco>"""

FIXED_BASE_MODEL = """<mujoco model='motrix-reset-write-fixed-base'>
  <option timestep='0.002' gravity='0 0 -9.81'/>
  <worldbody>
    <body name='pedestal' pos='0 0 .5'>
      <inertial pos='0 0 0' mass='5' diaginertia='.5 .5 .5'/>
      <geom name='pedestal_geom' type='box' size='.1 .1 .1'/>
      <body name='arm' pos='0 0 .3'>
        <joint name='swing' type='hinge' axis='0 1 0'/>
        <inertial pos='0 0 0' mass='.4' diaginertia='.04 .04 .04'/>
        <geom name='arm_geom' type='sphere' size='.05'/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name='swing' joint='swing' kp='8' kv='.8'/>
  </actuator>
</mujoco>"""


@pytest.fixture
def backend(tmp_path: Path) -> MotrixBackend:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL, encoding="utf-8")
    result = MotrixBackend(SceneCfg(str(model_path)), num_envs=4, sim_dt=0.002)
    try:
        yield result
    finally:
        result.close()


def _payload_from_defaults(backend: MotrixBackend) -> dict[str, torch.Tensor]:
    return {
        term: torch.as_tensor(np.array(backend.get_reset_term_default(term)), dtype=torch.float32)
        for term in (
            RESET_TERM_BODY_MASS,
            RESET_TERM_BODY_IPOS,
            RESET_TERM_GEOM_FRICTION,
            RESET_TERM_GRAVITY,
            RESET_TERM_KP,
            RESET_TERM_KD,
        )
    }


def test_reset_write_program_built_and_selected_rows_receive_payload(
    backend: MotrixBackend,
) -> None:
    assert backend._reset_write_program is not None
    fields = backend._reset_write_program.field_names()
    assert {"mass", "com", "friction", "kp", "kd"} <= fields

    defaults = _payload_from_defaults(backend)
    rows = torch.tensor([1, 3], dtype=torch.int64)
    body_mass = defaults[RESET_TERM_BODY_MASS].clone()
    body_ipos = defaults[RESET_TERM_BODY_IPOS].clone()
    friction = defaults[RESET_TERM_GEOM_FRICTION].clone()
    gravity = defaults[RESET_TERM_GRAVITY].clone()
    kp = defaults[RESET_TERM_KP].clone()
    kd = defaults[RESET_TERM_KD].clone()
    link_body = int(backend.get_body_ids(("link",))[0])
    base_geom = backend.get_geom_id("base_geom")
    body_mass[[1, 3], link_body] = torch.tensor([0.9, 1.1])
    body_ipos[[1, 3], link_body] = torch.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
    friction[[1, 3], base_geom] = torch.tensor([[0.9, 0.02, 0.0], [0.7, 0.01, 0.0]])
    gravity[[1, 3]] = torch.tensor([[0.0, 0.0, -7.0], [0.0, 0.0, -8.0]])
    kp[[1, 3]] = torch.tensor([[30.0], [31.0]])
    kd[[1, 3]] = torch.tensor([[3.0], [3.1]])
    payload = TensorResetRandomizationPayload(
        body_mass=body_mass[rows],
        body_ipos=body_ipos[rows],
        geom_friction=friction[rows],
        gravity=gravity[rows],
        kp=kp[rows],
        kd=kd[rows],
    )

    state = backend.get_state_views(device="cpu")
    backend.set_state_tensor(
        rows,
        state["qpos"][rows].contiguous(),
        state["qvel"][rows].contiguous(),
        payload,
    )

    for term, expected in (
        (RESET_TERM_BODY_MASS, body_mass),
        (RESET_TERM_BODY_IPOS, body_ipos),
        (RESET_TERM_GEOM_FRICTION, friction),
        (RESET_TERM_GRAVITY, gravity),
        (RESET_TERM_KP, kp),
        (RESET_TERM_KD, kd),
    ):
        actual = np.asarray(backend.get_reset_term_default(term))
        np.testing.assert_allclose(
            actual[rows.numpy()], expected[rows].numpy(), rtol=2e-6, atol=2e-6
        )
        np.testing.assert_allclose(
            actual[[0, 2]], expected[[0, 2]].numpy(), rtol=2e-6, atol=2e-6
        )


def test_reset_write_program_absent_terms_restore_defaults(backend: MotrixBackend) -> None:
    defaults = _payload_from_defaults(backend)
    state = backend.get_state_views(device="cpu")

    # First reset randomizes mass on row 1 only.
    mass_only = defaults[RESET_TERM_BODY_MASS].clone()
    link_body = int(backend.get_body_ids(("link",))[0])
    mass_only[1, link_body] = 2.5
    row = torch.tensor([1], dtype=torch.int64)
    backend.set_state_tensor(
        row,
        state["qpos"][[1]].contiguous(),
        state["qvel"][[1]].contiguous(),
        TensorResetRandomizationPayload(body_mass=mass_only[[1]]),
    )
    applied = np.asarray(backend.get_reset_term_default(RESET_TERM_BODY_MASS))
    np.testing.assert_allclose(applied[1, link_body], 2.5, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        applied[2, link_body], defaults[RESET_TERM_BODY_MASS][2, link_body], rtol=0.0, atol=1e-7
    )

    # Second reset touches row 2 with a kp-only payload: row 2 must receive
    # default mass (never the stale buffer row from the first reset) while
    # row 1 keeps its randomized mass and default kp.
    kp_only = defaults[RESET_TERM_KP].clone()
    kp_only[2] = torch.tensor([25.0])
    row = torch.tensor([2], dtype=torch.int64)
    backend.set_state_tensor(
        row,
        state["qpos"][[2]].contiguous(),
        state["qvel"][[2]].contiguous(),
        TensorResetRandomizationPayload(kp=kp_only[[2]]),
    )
    applied_mass = np.asarray(backend.get_reset_term_default(RESET_TERM_BODY_MASS))
    np.testing.assert_allclose(
        applied_mass[2, link_body],
        defaults[RESET_TERM_BODY_MASS][2, link_body],
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(applied_mass[1, link_body], 2.5, rtol=0.0, atol=1e-7)
    applied_kp = np.asarray(backend.get_reset_term_default(RESET_TERM_KP))
    np.testing.assert_allclose(applied_kp[2], [25.0], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        applied_kp[1], defaults[RESET_TERM_KP][1].numpy(), rtol=0.0, atol=1e-7
    )


def test_reset_write_program_excludes_world_fixed_links(tmp_path: Path) -> None:
    model_path = tmp_path / "fixed_base.xml"
    model_path.write_text(FIXED_BASE_MODEL, encoding="utf-8")
    fixed = MotrixBackend(SceneCfg(str(model_path)), num_envs=3, sim_dt=0.002, base_name="pedestal")
    try:
        program = fixed._reset_write_program
        assert program is not None
        pedestal_link = int(fixed.get_body_ids(("pedestal",))[0])
        arm_link = int(fixed.get_body_ids(("arm",))[0])
        assert pedestal_link not in set(program.mass_link_ids.tolist())
        assert arm_link in set(program.mass_link_ids.tolist())

        state = fixed.get_state_views(device="cpu")
        default_mass = np.array(fixed.get_reset_term_default(RESET_TERM_BODY_MASS))
        body_mass = torch.as_tensor(default_mass, dtype=torch.float32).clone()
        body_mass[1, arm_link] = 0.8
        row = torch.tensor([1], dtype=torch.int64)
        fixed.set_state_tensor(
            row,
            state["qpos"][[1]].contiguous(),
            state["qvel"][[1]].contiguous(),
            TensorResetRandomizationPayload(body_mass=body_mass[[1]]),
        )
        applied = np.asarray(fixed.get_reset_term_default(RESET_TERM_BODY_MASS))
        np.testing.assert_allclose(applied[1, arm_link], 0.8, rtol=0.0, atol=1e-7)
        # The world-fixed pedestal stays at its model mass on every row.
        np.testing.assert_allclose(
            applied[:, pedestal_link], default_mass[:, pedestal_link], rtol=0.0, atol=0.0
        )
    finally:
        fixed.close()


def test_whole_model_reset_write_program_built_per_runtime(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    assignment = (1, 0, 1, 0)
    sources = []
    for name, mass in (("light", 1.0), ("heavy", 1.5)):
        path = tmp_path / f"{name}.xml"
        path.write_text(
            f"""<mujoco model='whole-write-program'>
      <option timestep='.002' gravity='0 0 -9.81'/>
      <worldbody>
        <body name='arm' pos='0 0 .5'>
          <joint name='swing' type='hinge' axis='0 1 0'/>
          <inertial pos='0 0 0' mass='{mass}' diaginertia='.1 .1 .1'/>
          <geom name='arm_geom' type='sphere' size='.05'/>
        </body>
      </worldbody>
      <actuator>
        <position name='swing' joint='swing' kp='9' kv='.9'/>
      </actuator>
    </mujoco>""",
            encoding="utf-8",
        )
        sources.append(ModelSourceDescriptor(str(path)))
    scene = SceneCfg(
        model_file=sources[0].model_file,
        fixed_variant_plan=FixedVariantPlan(
            np.asarray(assignment, dtype=np.int32),
            tuple(sources),
            layout=FixedVariantLayout.SAME_LAYOUT,
        ),
    )
    backend = MotrixBackend(scene, len(assignment), 0.002, base_name="arm")
    try:
        assert len(backend._whole_model_runtimes) == 2
        for runtime in backend._whole_model_runtimes:
            assert runtime.reset_write_program is not None
            assert {"mass", "com", "friction", "kp", "kd"} <= (
                runtime.reset_write_program.field_names()
            )

        defaults = _payload_from_defaults(backend)
        rows = torch.tensor([3, 1], dtype=torch.int64)
        body_mass = defaults[RESET_TERM_BODY_MASS].clone()
        body_mass[[3, 1]] = torch.tensor([[2.2], [2.6]])
        kp = defaults[RESET_TERM_KP].clone()
        kp[[3, 1]] = torch.tensor([[20.0], [21.0]])
        state = backend.get_state_views(device="cpu")
        backend.set_state_tensor(
            rows,
            state["qpos"][rows].contiguous(),
            state["qvel"][rows].contiguous(),
            TensorResetRandomizationPayload(body_mass=body_mass[rows], kp=kp[rows]),
        )
        applied_mass = np.asarray(backend.get_reset_term_default(RESET_TERM_BODY_MASS))
        np.testing.assert_allclose(applied_mass[rows.numpy()].reshape(-1), [2.2, 2.6], rtol=2e-6)
        applied_kp = np.asarray(backend.get_reset_term_default(RESET_TERM_KP))
        np.testing.assert_allclose(applied_kp[rows.numpy()].reshape(-1), [20.0, 21.0], rtol=2e-6)
        np.testing.assert_allclose(
            applied_kp[[0, 2]], defaults[RESET_TERM_KP][[0, 2]].numpy(), rtol=2e-6, atol=2e-6
        )
    finally:
        backend.close()
