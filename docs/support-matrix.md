# Adapter support matrix

| Backend | Import | Install/runtime boundary | Status |
| --- | --- | --- | --- |
| MuJoCo | `unisim.MuJoCoBackend` | `uv sync --extra mujoco` (mjbatch runtime) | available |
| Motrix | `unisim.MotrixBackend` | `uv sync --extra motrix` | available |
| Drake | `unisim.DrakeBackend` | `uv sync --extra drake` (`drake-uni`) + native batch extension | available |
| MJWarp | `unisim.MJWarpBackend` | `uv sync --extra mjwarp`, CUDA | available |
| Genesis | `unisim.GenesisBackend` | `uv sync --extra genesis` | available |
| Newton | `unisim.NewtonBackend` | `uv sync --extra newton`, Newton 1.5.1 / MuJoCo-Warp 3.11.0 | available (CUDA) |
| SuperDex | `unisim.SuperDexBackend` | local checkout `superdex` extra, CPython 3.12 / SuperDex 1.0.0 | experimental CPU; see [profile](superdex.md) |
| IsaacGym | `unisim.IsaacGymBackend` | `uv sync --extra isaacgym` (empty extra) + dedicated Python 3.8 worker | available |
| IsaacSim | `unisim.IsaacSimBackend` | `uv sync --extra isaacsim` (empty extra) + dedicated IsaacSim/IsaacLab worker | available |

The base wheel imports none of these SDKs. Construction performs cold-path
runtime discovery and raises an adapter-specific, actionable error when the
runtime is unavailable. The matrix is an adapter/API support statement, not a
claim that every host has every vendor SDK or GPU capability.

The MuJoCo adapter's native executor is
[mjbatch](https://github.com/unilabsim/mjbatch), a maintained fork of
kevinzakka/mjbatch with prebuilt wheels for Linux x86_64/aarch64 and macOS
(CPython 3.10–3.14t) and an exact `mujoco==3.11.0` pin. Windows and musllinux
are unsupported for the native executor, so the Windows CI job runs the
core/import-boundary subset only. Numerical results before and after the
switch from the previous executor are not guaranteed identical — drift is
characterized by a recorded baseline, not gated. Heterogeneous model variants
are unsupported; field-level domain randomization goes through mjbatch
`expand`/`set_const`. `chunk_size`/`adaptive_chunk_size` are deprecated and
ignored (warn-and-ignore); the chunk scheduler was removed and mjbatch's
work-stealing thread pool is the tuning mechanism.

The MuJoCo-related extras share one version line (MuJoCo 3.11 / MuJoCo-Warp
3.11 / warp-lang 1.16.0) and are jointly installable: `mjwarp` tracks the line
with `mujoco-warp~=3.11.0`, while `newton` keeps exact upstream-coupled pins
(`newton==1.5.1`, `mujoco-warp==3.11.0`, `mujoco==3.11.0`,
`warp-lang==1.16.0`). Run
`uv run scripts/check_newton_runtime.py` after installation for a metadata-only
probe; add `--import` when the native runtime should be imported explicitly.
The adapter's cold-path calibration samples solver counts and raises an explicit
capacity error when `nconmax` or `njmax` is too small; it never accepts silent
constraint truncation.

Newton playback renders natively through `ViewerGL` (`pyglet>=2.1.6,<3`,
`imgui-bundle>=1.92.0`) when installed with the single `newton` extra:
`record` renders offscreen, `interactive` opens the windowed viewer, and
`auto` picks by display availability. If a runtime is incomplete, `record`
falls back to the offline MuJoCo snapshot pipeline and `interactive` fails
closed with an actionable error. Headless offscreen GL needs EGL
(`PYOPENGL_PLATFORM=egl`) or GLX under Wayland.
