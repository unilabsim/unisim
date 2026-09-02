"""Static contracts owned by the MJWarp adapter implementation."""

from __future__ import annotations

import ast
import inspect
import textwrap

from unisim.backend.mjwarp.backend import MjwarpBackend


def test_mjwarp_getters_do_not_materialize_warp_arrays() -> None:
    """Legacy host getters must not trigger an implicit device transfer."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(MjwarpBackend)))
    backend = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    getter_nodes = [
        node
        for node in backend.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("get_")
    ]
    assert getter_nodes

    offenders = [
        getter.name
        for getter in getter_nodes
        for node in ast.walk(getter)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "numpy"
    ]
    assert offenders == []
