# Mocap poses and geometry reset randomization

Status: accepted implementation of [owner issue #40](https://github.com/unilabsim/unisim/issues/40),
under the approved [Wuji roadmap](https://github.com/unilabsim/wuji_unilab/issues/1).
This owner decision extends the existing backend boundary; it introduces no
second environment lifecycle, tensor protocol, or engine implementation.

## Decision

`SimBackend.bind_mocap_pose(body_name)` resolves one fixed mocap body on the
cold path and returns `BackendMocapPoseBinding` from `unisim.backend.base`.
The immutable binding contains `backend_type`, `body_name`, `num_envs`, a
detached read-only `default_pose` with shape `(7,)`, and backend-owned callbacks.
Its public methods are:

- `read() -> np.ndarray`: detached `(num_envs, 7)` world poses.
- `write(env_ids, poses) -> None`: unique integer environment IDs and floating
  `(len(env_ids), 7)` poses. Layout is world xyz followed by a unit wxyz
  quaternion. An empty selection is a validated no-op.

The adapter applies pose writes, forwards physics to refresh derived state,
and refreshes its sensor/state caches before returning. Generalized positions
and velocities and unselected mocap poses are preserved. Consumers must not
retain native model/data handles or resolve names in reset/step code.
Unknown names, non-mocap bodies, invalid shapes/dtypes/IDs, non-finite values
and non-unit quaternions fail explicitly. The default implementation on other
backends raises `NotImplementedError` at binding; no capability is inferred
from a class name or a private method.

`set_state` retains the existing reset semantics: selected environments reset
their mocap poses to model defaults. A consumer that changes generalized state
and mocap pose commits `set_state` first, then the bound mocap pose writes.
These are ordered operations, not an atomic multi-backend transaction. This
keeps the established `set_state` signature stable and gives the environment
owner responsibility for ordering and prevalidating a composed reset. Existing
bindings remain valid across reset and stepping.

## Reset randomization tables

`ResetRandomizationPayload` adds these dense tables for `R` selected rows:

| Field / capability term | Shape | Meaning |
| --- | --- | --- |
| `geom_size` | `(R, ngeom, 3)` | MuJoCo primitive dimensions |
| `geom_solref` | `(R, ngeom, 2)` | Contact reference parameters |
| `geom_solimp` | `(R, ngeom, 5)` | Contact impedance parameters |
| `dof_damping` | `(R, nv)` | Non-negative joint damping |
| `dof_frictionloss` | `(R, nv)` | Non-negative joint friction loss |

Constants in `unisim.dr.types` are `RESET_TERM_GEOM_SIZE`,
`RESET_TERM_GEOM_SOLREF`, `RESET_TERM_GEOM_SOLIMP`, `RESET_TERM_DOF_DAMPING`
and `RESET_TERM_DOF_FRICTIONLOSS`. `get_geom_sizes`, `get_geom_solref`,
`get_geom_solimp`, `get_dof_damping`, and `get_dof_frictionloss` expose detached
default tables for cold-path consumer binding; they are not per-world current
model views. Model randomization persists across state-only resets.

MJWarp validates the new tables before changing state or uploading model data.
Solref accepts either two positive time-constant/damping-ratio values or two
non-positive direct-format values. Solimp requires impedance endpoints in
`[0,1]`, positive width, midpoint in `(0,1)`, and power at least one. Geometry
size writes require positive primitive radii/extents; unused components may
be zero. All table values must be representable as finite float32 values.

Sphere, capsule, ellipsoid, cylinder and box resizing is supported. The
adapter classifies geometry once, derives `geom_rbound` and `geom_aabb` from
the new dimensions using cached index groups, and uploads all three fields in
place before forwarding. Consumers cannot independently provide inconsistent
bounds. Dense payloads may include unchanged mesh/plane/hfield/SDF columns;
attempting to resize these unsupported types fails. Geometry resizing does
not implicitly change mass or inertia; consumers request those fields
explicitly when desired. Fixed-address per-world expansion precedes CUDA graph
capture, so later writes preserve captured pointers.

## Evidence and limits

`tests/test_reset_capability_contract.py` covers strict binding/default-backend
behavior and payload term filtering. `tests/test_mjwarp_required_capabilities.py`
covers selected-world mocap translation/rotation and reset, geometry/contact
effects, contact-parameter force changes, damping/friction motion effects,
invalid requests, and primitive bounds against official MuJoCo compilation.
Numerical tests require the 3.11 MJWarp extra and CUDA; a skipped runtime test
is not numerical evidence for a support claim.

This change does not add camera/raycast/terrain, post-substep sensor hooks,
arbitrary scene variants, torque perturbation or mesh resizing. Existing native
named sensor bindings remain the mechanism for contact, actuator-force and
site-velocity reads.
