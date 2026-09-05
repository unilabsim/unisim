# Adapter support matrix

| Backend | Import | Install/runtime boundary | Status |
| --- | --- | --- | --- |
| MuJoCo | `unisim.MuJoCoBackend` | `uv sync --extra mujoco` | available |
| Motrix | `unisim.MotrixBackend` | `uv sync --extra motrix` | available |
| Drake | `unisim.DrakeBackend` | `uv sync --extra drake` (`drake-uni`) + native batch extension | available |
| MJWarp | `unisim.MJWarpBackend` | `uv sync --extra mjwarp`, CUDA | available |
| Genesis | `unisim.GenesisBackend` | `uv sync --extra genesis` | available |
| Newton | `unisim.NewtonBackend` | `uv sync --extra newton`, Newton 1.5.1 / MuJoCo-Warp 3.11.0 | available (CUDA) |
| IsaacGym | `unisim.IsaacGymBackend` | `uv sync --extra isaacgym` (empty extra) + dedicated Python 3.8 worker | available |
| IsaacSim | `unisim.IsaacSimBackend` | `uv sync --extra isaacsim` (empty extra) + dedicated IsaacSim/IsaacLab worker | available |

The base wheel imports none of these SDKs. Construction performs cold-path
runtime discovery and raises an adapter-specific, actionable error when the
runtime is unavailable. The matrix is an adapter/API support statement, not a
claim that every host has every vendor SDK or GPU capability.

Newton uses a separate MuJoCo-Warp line from the existing MJWarp extra:
`newton` pins `newton==1.5.1`, `mujoco-warp==3.11.0`, `mujoco==3.11.0`, and
`warp-lang==1.16.0`, while `mjwarp` remains on `mujoco-warp==3.10.0.3`.
These extras are intentionally mutually exclusive in one environment. Run
`uv run scripts/check_newton_runtime.py` after installation for a metadata-only
probe; add `--import` when the native runtime should be imported explicitly.
The adapter's cold-path calibration samples solver counts and raises an explicit
capacity error when `nconmax` or `njmax` is too small; it never accepts silent
constraint truncation.
