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
from unisim.utils.math import np_matrix_from_quat  # noqa: E402

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


SIZED_MODEL = """<mujoco model='motrix-reset-geom-size'>
  <option timestep='0.002' gravity='0 0 -9.81'/>
  <worldbody>
    <body name='base' pos='0 0 1'>
      <freejoint name='root'/>
      <inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>
      <geom name='sphere_geom' type='sphere' size='.08'/>
      <geom name='box_geom' type='box' size='.05 .06 .07'/>
      <geom name='capsule_geom' type='capsule' size='.03 .1'/>
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


def test_reset_write_program_joint_dof_terms(tmp_path: Path) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL, encoding="utf-8")
    backend = MotrixBackend(SceneCfg(str(model_path)), num_envs=4, sim_dt=0.002)
    try:
        capabilities = backend.get_dr_capabilities()
        assert "dof_armature" in capabilities.supported_reset_terms
        assert "dof_frictionloss" in capabilities.supported_reset_terms
        defaults = {
            term: torch.as_tensor(
                np.array(backend.get_reset_term_default(term)), dtype=torch.float32
            )
            for term in ("dof_armature", "dof_frictionloss")
        }
        # The joint tables use the full native nv layout (free-root dofs first),
        # unlike ``num_dof_vel`` which counts only the scalar joints.
        nv = int(defaults["dof_armature"].shape[1])
        hinge_dof = nv - 1
        armature = defaults["dof_armature"].clone()
        frictionloss = defaults["dof_frictionloss"].clone()
        armature[[1, 3], hinge_dof] = torch.tensor([0.01, 0.02])
        frictionloss[[1, 3], hinge_dof] = torch.tensor([0.05, 0.06])
        rows = torch.tensor([1, 3], dtype=torch.int64)
        state = backend.get_state_views(device="cpu")
        backend.set_state_tensor(
            rows,
            state["qpos"][rows].contiguous(),
            state["qvel"][rows].contiguous(),
            TensorResetRandomizationPayload(
                dof_armature=armature[rows],
                dof_frictionloss=frictionloss[rows],
            ),
        )
        applied_armature = np.asarray(backend.get_reset_term_default("dof_armature"))
        np.testing.assert_allclose(
            applied_armature[[1, 3], hinge_dof], [0.01, 0.02], rtol=2e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            applied_armature[[0, 2], hinge_dof],
            defaults["dof_armature"][[0, 2], hinge_dof].numpy(),
            rtol=0.0,
            atol=0.0,
        )
        applied_frictionloss = np.asarray(backend.get_reset_term_default("dof_frictionloss"))
        np.testing.assert_allclose(
            applied_frictionloss[[1, 3], hinge_dof], [0.05, 0.06], rtol=2e-6, atol=1e-7
        )

        # Free-root DoF columns stay at construction defaults and reject mutation.
        bad = defaults["dof_armature"].clone()
        bad[1, 1] = 0.5
        with pytest.raises(ValueError, match="free-root DoF columns"):
            backend.set_state_tensor(
                rows,
                state["qpos"][rows].contiguous(),
                state["qvel"][rows].contiguous(),
                TensorResetRandomizationPayload(dof_armature=bad[rows]),
            )
        negative = defaults["dof_armature"].clone()
        negative[1, hinge_dof] = -0.1
        with pytest.raises(ValueError, match="non-negative"):
            backend.set_state_tensor(
                rows,
                state["qpos"][rows].contiguous(),
                state["qvel"][rows].contiguous(),
                TensorResetRandomizationPayload(dof_armature=negative[rows]),
            )
    finally:
        backend.close()


def test_reset_write_program_geom_size(tmp_path: Path) -> None:
    model_path = tmp_path / "sized.xml"
    model_path.write_text(SIZED_MODEL, encoding="utf-8")
    backend = MotrixBackend(SceneCfg(str(model_path)), num_envs=4, sim_dt=0.002)
    try:
        capabilities = backend.get_dr_capabilities()
        assert "geom_size" in capabilities.supported_reset_terms
        program = backend._reset_write_program
        assert program is not None
        assert "size" in program.field_names()

        defaults = torch.as_tensor(
            np.array(backend.get_reset_term_default("geom_size")), dtype=torch.float32
        )
        sphere = backend.get_geom_id("sphere_geom")
        box = backend.get_geom_id("box_geom")
        capsule = backend.get_geom_id("capsule_geom")
        requested = defaults.clone()
        requested[[1, 3], sphere, 0] = torch.tensor([0.11, 0.12])
        requested[[1, 3], box, :3] = torch.tensor(
            [[0.06, 0.07, 0.08], [0.09, 0.10, 0.11]]
        )
        requested[[1, 3], capsule, :2] = torch.tensor([[0.04, 0.12], [0.05, 0.13]])
        rows = torch.tensor([1, 3], dtype=torch.int64)
        state = backend.get_state_views(device="cpu")
        backend.set_state_tensor(
            rows,
            state["qpos"][rows].contiguous(),
            state["qvel"][rows].contiguous(),
            TensorResetRandomizationPayload(geom_size=requested[rows]),
        )
        applied = np.asarray(backend.get_reset_term_default("geom_size"))
        np.testing.assert_allclose(
            applied[[1, 3]], requested[[1, 3]].numpy(), rtol=2e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            applied[[0, 2]], defaults[[0, 2]].numpy(), rtol=0.0, atol=0.0
        )

        bad_unused = requested.clone()
        bad_unused[1, sphere, 1] = 0.3
        with pytest.raises(ValueError, match="unused components must stay zero"):
            backend.set_state_tensor(
                rows,
                state["qpos"][rows].contiguous(),
                state["qvel"][rows].contiguous(),
                TensorResetRandomizationPayload(geom_size=bad_unused[rows]),
            )
        bad_zero = requested.clone()
        bad_zero[1, sphere, 0] = 0.0
        with pytest.raises(ValueError, match="required components must be positive"):
            backend.set_state_tensor(
                rows,
                state["qpos"][rows].contiguous(),
                state["qvel"][rows].contiguous(),
                TensorResetRandomizationPayload(geom_size=bad_zero[rows]),
            )
    finally:
        backend.close()


def test_whole_model_reset_write_program_joint_dof_and_geom_size(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    assignment = (1, 0, 1, 0)
    sources = []
    for name, mass in (("light", 1.0), ("heavy", 1.5)):
        path = tmp_path / f"{name}.xml"
        path.write_text(
            f"""<mujoco model='whole-dof-size'>
      <option timestep='.002' gravity='0 0 -9.81'/>
      <worldbody>
        <body name='arm' pos='0 0 .5'>
          <joint name='swing' type='hinge' axis='0 1 0'/>
          <inertial pos='0 0 0' mass='{mass}' diaginertia='.1 .1 .1'/>
          <geom name='arm_geom' type='box' size='.05 .05 .05'/>
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
        capabilities = backend.get_dr_capabilities()
        assert "dof_armature" in capabilities.supported_reset_terms
        assert "geom_size" in capabilities.supported_reset_terms

        nv = backend.num_dof_vel
        swing_dof = nv - 1
        armature = torch.as_tensor(
            np.array(backend.get_reset_term_default("dof_armature")), dtype=torch.float32
        ).clone()
        armature[[3, 1], swing_dof] = torch.tensor([0.03, 0.04])
        geom_size = torch.as_tensor(
            np.array(backend.get_reset_term_default("geom_size")), dtype=torch.float32
        ).clone()
        geom_size[[3, 1], 0, :3] = torch.tensor([[0.08, 0.07, 0.06], [0.09, 0.08, 0.07]])
        rows = torch.tensor([3, 1], dtype=torch.int64)
        state = backend.get_state_views(device="cpu")
        backend.set_state_tensor(
            rows,
            state["qpos"][rows].contiguous(),
            state["qvel"][rows].contiguous(),
            TensorResetRandomizationPayload(
                dof_armature=armature[rows],
                geom_size=geom_size[rows],
            ),
        )
        applied_armature = np.asarray(backend.get_reset_term_default("dof_armature"))
        np.testing.assert_allclose(
            applied_armature[rows.numpy(), swing_dof], [0.03, 0.04], rtol=2e-6, atol=1e-7
        )
        np.testing.assert_allclose(applied_armature[[0, 2], swing_dof], 0.0, rtol=0.0, atol=0.0)
        applied_size = np.asarray(backend.get_reset_term_default("geom_size"))
        np.testing.assert_allclose(
            applied_size[rows.numpy()], geom_size[rows].numpy(), rtol=2e-6, atol=1e-7
        )
    finally:
        backend.close()


def _tensor_from_principal(moments: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    rotation = np_matrix_from_quat(quat_wxyz).astype(np.float64)
    return rotation @ np.diag(moments) @ rotation.T


INERTIA_MODEL = """<mujoco model='motrix-reset-inertia'>
  <option timestep='0.002' gravity='0 0 -9.81'/>
  <worldbody>
    <body name='base' pos='0 0 1'>
      <freejoint name='root'/>
      <inertial pos='0 0 0' mass='1' diaginertia='.1 .2 .3'/>
      <geom name='base_geom' type='sphere' size='.08'/>
      <body name='link' pos='0 0 .2'>
        <joint name='drive' type='hinge' axis='0 1 0'/>
        <inertial pos='0 0 0' mass='.2' diaginertia='.05 .04 .03'/>
        <geom name='link_geom' type='sphere' size='.03'/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name='drive' joint='drive' kp='10' kv='1'/>
  </actuator>
</mujoco>"""


def test_reset_write_program_body_inertia_terms(tmp_path: Path) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text(INERTIA_MODEL, encoding="utf-8")
    backend = MotrixBackend(SceneCfg(str(model_path)), num_envs=4, sim_dt=0.002)
    try:
        capabilities = backend.get_dr_capabilities()
        assert "body_inertia" in capabilities.supported_reset_terms
        assert "body_iquat" in capabilities.supported_reset_terms
        raw_inertia = backend.get_reset_term_default("body_inertia")
        raw_iquat = backend.get_reset_term_default("body_iquat")
        assert not raw_inertia.flags.writeable
        assert not raw_iquat.flags.writeable
        default_inertia = np.array(raw_inertia)
        default_iquat = np.array(raw_iquat)
        assert default_inertia.shape == (4, 2, 3)
        assert default_iquat.shape == (4, 2, 4)
        # Construction diaginertia('.1 .2 .3' / '.05 .04 .03') decomposes to
        # the canonical ascending principal moments.
        np.testing.assert_allclose(
            default_inertia[:, 0], np.tile((0.1, 0.2, 0.3), (4, 1)), rtol=2e-6, atol=0.0
        )
        np.testing.assert_allclose(
            default_inertia[:, 1], np.tile((0.03, 0.04, 0.05), (4, 1)), rtol=2e-6, atol=0.0
        )

        rows = torch.tensor([1, 3], dtype=torch.int64)
        state = backend.get_state_views(device="cpu")
        requested_inertia = torch.as_tensor(default_inertia, dtype=torch.float32).clone()
        requested_iquat = torch.as_tensor(default_iquat, dtype=torch.float32).clone()
        requested_inertia[[1, 3], 0, :] = 0.2
        quarter = float(np.sqrt(0.5))
        requested_iquat[[1, 3], 1, :] = torch.tensor([quarter, 0.0, 0.0, quarter])
        backend.set_state_tensor(
            rows,
            state["qpos"][rows].contiguous(),
            state["qvel"][rows].contiguous(),
            TensorResetRandomizationPayload(
                body_inertia=requested_inertia[rows],
                body_iquat=requested_iquat[rows],
            ),
        )
        live_inertia = np.asarray(backend.get_reset_term_default("body_inertia"))
        live_iquat = np.asarray(backend.get_reset_term_default("body_iquat"))
        np.testing.assert_allclose(live_inertia[[1, 3], 0], 0.2, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(
            live_inertia[[0, 2]], default_inertia[[0, 2]], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(live_iquat[[0, 2]], default_iquat[[0, 2]], rtol=0.0, atol=0.0)
        # The committed tensors reconstruct the requested principal forms.
        for row, body in ((1, 0), (1, 1), (3, 0), (3, 1)):
            got = _tensor_from_principal(live_inertia[row, body], live_iquat[row, body])
            want = _tensor_from_principal(
                requested_inertia[row, body].numpy(), requested_iquat[row, body].numpy()
            )
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

        # An inertia-only write composes with the construction principal axes:
        # the selected-row reset has already restored the slice to defaults, so
        # payload-absent terms keep construction values (matching the
        # write-program default restoration for absent mass/COM fields).
        second = torch.as_tensor(live_inertia, dtype=torch.float32).clone()
        second[1, 0, :] = 0.15
        backend.set_state_tensor(
            rows[:1],
            state["qpos"][rows[:1]].contiguous(),
            state["qvel"][rows[:1]].contiguous(),
            TensorResetRandomizationPayload(body_inertia=second[[1]]),
        )
        live_after = np.asarray(backend.get_reset_term_default("body_iquat"))
        inertia_after = np.asarray(backend.get_reset_term_default("body_inertia"))
        np.testing.assert_allclose(inertia_after[1, 0], 0.15, rtol=2e-6, atol=1e-7)
        after = _tensor_from_principal(inertia_after[1, 1], live_after[1, 1])
        construction = _tensor_from_principal(default_inertia[1, 1], default_iquat[1, 1])
        np.testing.assert_allclose(after, construction, rtol=1e-5, atol=1e-6)

        # Shape and fail-closed validation.
        with pytest.raises(ValueError, match="body_inertia must have shape"):
            backend.set_state_tensor(
                rows,
                state["qpos"][rows].contiguous(),
                state["qvel"][rows].contiguous(),
                TensorResetRandomizationPayload(body_inertia=requested_inertia[rows, :, :2]),
            )
        nonfinite = requested_inertia.clone()
        nonfinite[1, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="body_inertia must contain only finite values"):
            backend.set_state_tensor(
                rows,
                state["qpos"][rows].contiguous(),
                state["qvel"][rows].contiguous(),
                TensorResetRandomizationPayload(body_inertia=nonfinite[rows]),
            )
        negative = requested_inertia.clone()
        negative[1, 0, 0] = -0.1
        with pytest.raises(ValueError, match="body_inertia must contain only positive values"):
            backend.set_state_tensor(
                rows,
                state["qpos"][rows].contiguous(),
                state["qvel"][rows].contiguous(),
                TensorResetRandomizationPayload(body_inertia=negative[rows]),
            )
        triangle = requested_inertia.clone()
        triangle[1, 0, :] = torch.tensor([0.5, 0.1, 0.1])
        with pytest.raises(
            ValueError, match="body_inertia must satisfy the principal-moment triangle"
        ):
            backend.set_state_tensor(
                rows,
                state["qpos"][rows].contiguous(),
                state["qvel"][rows].contiguous(),
                TensorResetRandomizationPayload(body_inertia=triangle[rows]),
            )
        zero_quat = requested_iquat.clone()
        zero_quat[1, 1, :] = 0.0
        with pytest.raises(ValueError, match="body_iquat must contain normalizable"):
            backend.set_state_tensor(
                rows,
                state["qpos"][rows].contiguous(),
                state["qvel"][rows].contiguous(),
                TensorResetRandomizationPayload(body_iquat=zero_quat[rows]),
            )
    finally:
        backend.close()


def test_whole_model_reset_write_program_body_inertia(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    assignment = (1, 0, 1, 0)
    sources = []
    for name, mass in (("light", 1.0), ("heavy", 1.5)):
        path = tmp_path / f"{name}.xml"
        path.write_text(
            f"""<mujoco model='whole-inertia'>
      <option timestep='.002' gravity='0 0 -9.81'/>
      <worldbody>
        <body name='arm' pos='0 0 .5'>
          <joint name='swing' type='hinge' axis='0 1 0'/>
          <inertial pos='0 0 0' mass='{mass}' diaginertia='.1 .2 .3'/>
          <geom name='arm_geom' type='box' size='.05 .05 .05'/>
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
        capabilities = backend.get_dr_capabilities()
        assert "body_inertia" in capabilities.supported_reset_terms
        assert "body_iquat" in capabilities.supported_reset_terms
        default_inertia = np.array(backend.get_reset_term_default("body_inertia"))
        default_iquat = np.array(backend.get_reset_term_default("body_iquat"))
        assert default_inertia.shape == (4, 1, 3)
        assert default_iquat.shape == (4, 1, 4)
        np.testing.assert_allclose(
            default_inertia[:, 0], np.tile((0.1, 0.2, 0.3), (4, 1)), rtol=2e-6, atol=0.0
        )
        np.testing.assert_allclose(
            default_iquat[:, 0], np.tile((1.0, 0.0, 0.0, 0.0), (4, 1)), rtol=0.0, atol=0.0
        )

        rows = torch.tensor([3, 1], dtype=torch.int64)
        state = backend.get_state_views(device="cpu")
        requested_inertia = torch.as_tensor(default_inertia, dtype=torch.float32).clone()
        requested_iquat = torch.as_tensor(default_iquat, dtype=torch.float32).clone()
        requested_inertia[[3, 1], 0, :] = torch.tensor([[0.12, 0.22, 0.32], [0.13, 0.23, 0.33]])
        quarter = float(np.sqrt(0.5))
        requested_iquat[[3, 1], 0, :] = torch.tensor([quarter, 0.0, quarter, 0.0])
        backend.set_state_tensor(
            rows,
            state["qpos"][rows].contiguous(),
            state["qvel"][rows].contiguous(),
            TensorResetRandomizationPayload(
                body_inertia=requested_inertia[rows],
                body_iquat=requested_iquat[rows],
            ),
        )
        live_inertia = np.asarray(backend.get_reset_term_default("body_inertia"))
        live_iquat = np.asarray(backend.get_reset_term_default("body_iquat"))
        np.testing.assert_allclose(
            live_inertia[[3, 1], 0],
            [[0.12, 0.22, 0.32], [0.13, 0.23, 0.33]],
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            live_inertia[[0, 2]], default_inertia[[0, 2]], rtol=0.0, atol=0.0
        )
        for row in (1, 3):
            got = _tensor_from_principal(live_inertia[row, 0], live_iquat[row, 0])
            want = _tensor_from_principal(
                requested_inertia[row, 0].numpy(), requested_iquat[row, 0].numpy()
            )
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    finally:
        backend.close()
