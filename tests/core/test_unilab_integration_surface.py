"""Import smoke tests for the adapter-internal surface UniLab consumes.

``AGENTS.md`` ("Scope and compatibility boundaries") declares the modules and
names below stable for UniLab.  Every listed module must stay importable with
no engine SDK installed, so the module check runs in a subprocess whose
meta-path finder fails any attempt to resolve an engine SDK.
"""

from __future__ import annotations

import subprocess
import sys

UNILAB_STABLE_MODULES = (
    "unisim.backend.genesis.dependencies",
    "unisim.backend.isaacsim.dependencies",
    "unisim.backend.isaacgym.dependencies",
    "unisim.backend.superdex.dependencies",
    "unisim.backend.drake.backend",
    "unisim.backend.mujoco.xml",
    "unisim.backend.newton.runtime",
    "unisim.backend.mjwarp.runtime",
    "unisim.backend.playback",
    "unisim.backend.mujoco.playback",
    "unisim.visualization.mujoco_geom",
)

UNILAB_STABLE_SYMBOLS = {
    "unisim.backend.drake.backend": ("ensure_drake_batch_available",),
    "unisim.backend.mujoco.xml": ("materialize_scene_visual_override",),
    "unisim.backend.newton.runtime": (
        "bind_newton_process_device",
        "get_bound_newton_process_device",
    ),
    "unisim.backend.mjwarp.runtime": ("bind_mjwarp_process_device",),
    "unisim.backend.playback": ("write_playback_video",),
    "unisim.backend.mujoco.playback": ("resolve_render_play_model_files",),
    "unisim.visualization.mujoco_geom": (
        "geom_mesh",
    ),
}

ENGINE_SDK_ROOTS = (
    "mujoco",
    "mjbatch",
    "mujoco_warp",
    "motrixsim",
    "pydrake",
    "warp",
    "newton",
    "genesis",
    "superdex",
    "isaacgym",
    "isaacsim",
    "torch",
    "omni",
)


def test_unilab_stable_modules_import_without_engine_sdks() -> None:
    modules = ", ".join(repr(name) for name in UNILAB_STABLE_MODULES)
    blocked = ", ".join(repr(name) for name in ENGINE_SDK_ROOTS)
    code = f"""
import importlib
import importlib.abc
import sys

blocked = {{{blocked}}}

class NoEngineSDKs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in blocked, fullname
        return None

sys.meta_path.insert(0, NoEngineSDKs())
for name in ({modules}):
    importlib.import_module(name)
assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_unilab_stable_symbols_resolve() -> None:
    import importlib

    for module_name, symbols in UNILAB_STABLE_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for symbol in symbols:
            assert callable(getattr(module, symbol)), f"{module_name}.{symbol}"
