"""Static and optional-runtime boundary tests for the Newton adapter."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import unisim
from unisim.backend.newton.backend import NewtonBackend
from unisim.backend.newton.dependencies import (
    NewtonDependencyError,
    load_newton_dependencies,
    newton_dependencies_available,
)
from unisim.conformance import assert_backend_conformance
from unisim.scene import SceneCfg

_MODEL = """
<mujoco model="unisim-newton-test">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.5">
      <joint name="root" type="free"/>
      <geom name="base_geom" type="sphere" size="0.08" mass="1"/>
      <body name="arm" pos="0 0 0.12">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="hinge_motor" joint="hinge" ctrlrange="-1 1"/></actuator>
</mujoco>
"""


def test_newton_getters_do_not_materialize_warp_arrays() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(NewtonBackend)))
    backend = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    getter_nodes = [
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("get_")
    ]
    offenders = [
        getter.name
        for getter in getter_nodes
        for node in ast.walk(getter)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "numpy"
    ]
    assert offenders == []


def test_newton_import_boundary_does_not_load_optional_modules() -> None:
    code = (
        "import sys, unisim; "
        "assert not [name for name in sys.modules if name == 'newton' or "
        "name == 'warp' or name.startswith('mujoco_warp')]"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_newton_manifest_and_lazy_exports() -> None:
    assert unisim.adapter_spec("newton").extra == "newton"
    assert unisim.NewtonBackend is NewtonBackend
    assert unisim.NewtonDependencyError is NewtonDependencyError


def test_newton_dependency_probe_is_fail_closed_when_extra_is_absent() -> None:
    if newton_dependencies_available():
        pytest.skip("Newton optional runtime is installed in this environment")
    with pytest.raises(NewtonDependencyError, match="newton backend requires"):
        unisim.create_backend("newton", scene=object())


def test_newton_conformance_when_cuda_runtime_is_available(tmp_path: Path) -> None:
    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton conformance requires a CUDA Warp device")
    model_file = tmp_path / "newton.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=2,
        sim_dt=0.005,
        device=str(device),
        capacity_check_steps=1,
    )
    assert_backend_conformance(backend)
    backend.close()
