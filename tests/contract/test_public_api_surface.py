from __future__ import annotations

import ast
from pathlib import Path

LEGACY_METHODS = frozenset(
    {
        "step",
        "reset",
        "get_state",
        "set_state",
        "get_base_pos",
        "get_base_quat",
        "get_base_lin_vel",
        "get_base_ang_vel",
        "get_dof_pos",
        "get_dof_vel",
        "get_body_pos_w",
        "get_body_quat_w",
        "get_body_pose_w",
        "get_body_lin_vel_w",
        "get_body_vel_w",
        "get_body_ang_vel_w",
        "get_body_state_w",
        "copy_body_state_w",
        "get_body_pose_w_rows",
        "get_body_lin_vel_w_rows",
        "get_body_ang_vel_w_rows",
        "get_body_pos_b",
        "get_body_quat_b",
        "get_body_lin_vel_b",
        "get_body_ang_vel_b",
        "get_sensor_data",
        "get_sensor_data_rows",
        "get_sensor_data_batch",
    }
)


def test_concrete_backends_do_not_republish_numpy_lifecycle() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "unisim"
    paths = [root / "fake.py", *(root / "backend").glob("*/backend.py")]
    assert paths
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not node.name.endswith("Backend"):
                continue
            methods = {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert not methods & LEGACY_METHODS, f"{path}:{node.name}"


def test_base_contract_has_no_numpy_lifecycle() -> None:
    from unisim.backend.base import SimBackend

    assert not LEGACY_METHODS.intersection(SimBackend.__dict__)


def test_disallowed_prefix_cannot_return_to_runtime_or_tests() -> None:
    repository = Path(__file__).resolve().parents[2]
    forbidden = "_".join(("", "legacy", ""))
    roots = (repository / "src" / "unisim", repository / "tests")
    paths = [path for root in roots for path in root.rglob("*.py")]
    assert paths
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert forbidden not in source, path


def test_public_subprocess_alias_is_removed() -> None:
    import unisim

    assert not hasattr(unisim, "SubprocessBackend")
