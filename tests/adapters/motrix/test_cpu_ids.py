"""Motrix worker CPU affinity (``cpu_ids``) cold-path contract.

MotrixSim 0.10.1 exposes the process-wide ``motrixsim.init_thread_pool``
initializer; these tests pin the adapter's validation, passthrough, and
default-path behavior without depending on the host's CPU topology. The last
test is the behavioral counterpart: a subprocess steps a ``cpu_ids``
-configured backend and verifies the native worker threads' affinity masks
are confined to the requested CPU block (the process-wide pool can only be
initialized once, so the real pinning runs outside the pytest process).
"""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("motrixsim")

import motrixsim

from unisim import create_backend
from unisim.backend.motrix.backend import _validate_motrix_cpu_ids
from unisim.scene import SceneCfg

MODEL = """<mujoco model='unisim-test'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
  <actuator><motor joint='slide' ctrlrange='-1 1'/></actuator>
</mujoco>"""


def _write_model(tmp_path: Path) -> str:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    return str(model_path)


def _available_ids() -> list[int]:
    import os

    get = getattr(os, "sched_getaffinity", None)
    return sorted(get(0)) if get is not None else [0, 1]


def test_validate_rejects_structurally_invalid_blocks() -> None:
    with pytest.raises(TypeError, match="sequence of integer"):
        _validate_motrix_cpu_ids("0,1")
    with pytest.raises(ValueError, match="non-empty"):
        _validate_motrix_cpu_ids([])
    with pytest.raises(ValueError, match="non-negative integers"):
        _validate_motrix_cpu_ids([0, -1])
    with pytest.raises(ValueError, match="non-negative integers"):
        _validate_motrix_cpu_ids([0, True])
    with pytest.raises(ValueError, match="unique"):
        _validate_motrix_cpu_ids([0, 0])


def test_validate_rejects_unavailable_ids() -> None:
    import os

    if getattr(os, "sched_getaffinity", None) is None:
        pytest.skip("os.sched_getaffinity unavailable on this platform")
    absent = max(os.sched_getaffinity(0)) + 64
    with pytest.raises(ValueError, match="not available to this process"):
        _validate_motrix_cpu_ids([absent])


def test_validate_none_is_empty() -> None:
    assert _validate_motrix_cpu_ids(None) == ()


def test_factory_cpu_ids_passthrough(tmp_path: Path) -> None:
    ids = _available_ids()[:1]
    with mock.patch.object(motrixsim, "init_thread_pool") as init:
        backend = create_backend(
            "motrix",
            SceneCfg(model_file=_write_model(tmp_path)),
            num_envs=1,
            sim_dt=0.01,
            base_name="base",
            cpu_ids=ids,
        )
    try:
        assert backend.cpu_ids == tuple(ids)
        init.assert_called_once_with(core_ids=list(ids))
    finally:
        backend.close()


def test_default_path_leaves_worker_pool_untouched(tmp_path: Path) -> None:
    with mock.patch.object(motrixsim, "init_thread_pool") as init:
        backend = create_backend(
            "motrix",
            SceneCfg(model_file=_write_model(tmp_path)),
            num_envs=1,
            sim_dt=0.01,
            base_name="base",
        )
    try:
        assert backend.cpu_ids is None
        init.assert_not_called()
        backend.step(np.zeros((1, 1)))
    finally:
        backend.close()


def test_already_initialized_pool_degrades_to_warning(tmp_path: Path) -> None:
    ids = _available_ids()[:1]
    with mock.patch.object(
        motrixsim, "init_thread_pool", side_effect=RuntimeError("already initialized")
    ):
        with pytest.warns(UserWarning, match="cpu_ids=.* was not applied"):
            backend = create_backend(
                "motrix",
                SceneCfg(model_file=_write_model(tmp_path)),
                num_envs=1,
                sim_dt=0.01,
                base_name="base",
                cpu_ids=ids,
            )
    try:
        assert backend.cpu_ids == tuple(ids)
    finally:
        backend.close()


def test_worker_threads_run_confined_to_requested_cpus(tmp_path: Path) -> None:
    """Behavioral check: native workers run on the requested CPU set.

    Runs in a subprocess because the MotrixSim pool is process-wide and can
    only be initialized once; the in-process tests above already consume that
    single initialization through the lazy default policy.
    """
    import os
    import subprocess
    import sys

    if sys.platform != "linux" or getattr(os, "sched_getaffinity", None) is None:
        pytest.skip("requires Linux sched_getaffinity")
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("need at least two CPUs available to this process")
    program = """
import os
from pathlib import Path

import numpy as np

from unisim import create_backend
from unisim.scene import SceneCfg

ids = sorted(os.sched_getaffinity(0))[:2]
backend = create_backend(
    "motrix",
    SceneCfg(model_file=%r),
    num_envs=4,
    sim_dt=0.01,
    base_name="base",
    cpu_ids=ids,
)
try:
    backend.step(np.zeros((4, 1), dtype=np.float32))
    pinned = []
    for task in Path("/proc/self/task").iterdir():
        try:
            affinity = set(os.sched_getaffinity(int(task.name)))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue  # thread exited between listing and query
        if len(affinity) == 1:
            pinned.append(affinity)
    # The requested block has two CPUs, so at least two workers exist.
    assert len(pinned) >= 2, f"expected pinned MotrixSim worker threads, got {pinned}"
    assert all(cpus <= set(ids) for cpus in pinned), (
        f"worker affinity outside requested block {ids}: {pinned}"
    )
    assert backend.cpu_ids == tuple(ids)
finally:
    backend.close()
""" % _write_model(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"subprocess affinity check failed:\n{result.stdout}\n{result.stderr}"
    )
