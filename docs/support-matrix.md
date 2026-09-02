# Adapter support matrix

| Backend | Import | Install/runtime boundary | Status |
| --- | --- | --- | --- |
| MuJoCo | `unisim.MuJoCoBackend` | `uv sync --extra mujoco` | available |
| Motrix | `unisim.MotrixBackend` | `uv sync --extra motrix` | available |
| Drake | `unisim.DrakeBackend` | `uv sync --extra drake` (`drake-uni`) + native batch extension | available |
| MJWarp | `unisim.MJWarpBackend` | `uv sync --extra mjwarp`, CUDA | available |
| Genesis | `unisim.GenesisBackend` | `uv sync --extra genesis` | available |
| IsaacGym | `unisim.IsaacGymBackend` | dedicated Python 3.8 worker | available |
| IsaacSim | `unisim.IsaacSimBackend` | dedicated IsaacSim/IsaacLab worker | available |

The base wheel imports none of these SDKs. Construction performs cold-path
runtime discovery and raises an adapter-specific, actionable error when the
runtime is unavailable. The matrix is an adapter/API support statement, not a
claim that every host has every vendor SDK or GPU capability.
