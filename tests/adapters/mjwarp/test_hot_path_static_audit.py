"""Static audit for MJWarp's public tensor hot paths."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[3] / "src" / "unisim" / "backend" / "mjwarp" / "backend.py"
)

AUDITED_METHODS = ("step_tensor", "set_state_tensor_mask")

FORBIDDEN_ALLOCATION_CALLS = frozenset({"empty", "zeros", "ones", "cat", "stack"})
FORBIDDEN_HOST_CALLS = frozenset({"cpu", "tolist", "numpy", "item", "nonzero"})
# ``bool(<tensor reduction>)`` is an implicit blocking D2H read, the same cost
# as ``.item()`` but invisible to a call-name scan.
TENSOR_VERDICT_CALLS = frozenset({"all", "any", "isfinite", "isnan", "isinf"})


@dataclass(frozen=True)
class Violation:
    method: str
    line: int
    kind: str
    expression: str


def _methods(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    result: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MjwarpBackend":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in AUDITED_METHODS:
                    result[item.name] = item
    return result


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def _is_blocking_bool(call: ast.Call) -> bool:
    """Return whether ``call`` is ``bool(...)`` over a tensor reduction."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "bool") or not call.args:
        return False
    return any(
        isinstance(node, ast.Call) and _call_name(node) in TENSOR_VERDICT_CALLS
        for node in ast.walk(call.args[0])
    )


def _static_violations(tree: ast.Module, synchronize_lines: set[int]) -> list[Violation]:
    violations: list[Violation] = []
    methods = _methods(tree)
    # These methods are used by every execution mode, not only backend-owned
    # physics graphs; a stream query is still metadata, but it adds host work.
    for name, method in methods.items():
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            called = _call_name(node)
            expression = _unparse(node)
            if called in {"synchronize", "_synchronize"} and node.lineno not in synchronize_lines:
                violations.append(Violation(name, node.lineno, "host-sync", expression))
            if called == "current_stream":
                violations.append(Violation(name, node.lineno, "stream-query", expression))
            if called in FORBIDDEN_HOST_CALLS:
                violations.append(Violation(name, node.lineno, "host-detour", expression))
            if _is_blocking_bool(node):
                violations.append(Violation(name, node.lineno, "host-verdict", expression))
            if called in FORBIDDEN_ALLOCATION_CALLS and not any(
                keyword.arg == "out" for keyword in node.keywords
            ):
                violations.append(Violation(name, node.lineno, "hot-path-allocation", expression))
    return violations


# Completion is represented by the public CompletionToken metadata
# receipt. These hot methods must not synchronize a Torch stream at all.
DECLARED_COMPLETION_BARRIERS: frozenset[int] = frozenset()


def test_mjwarp_audited_hot_path_methods_are_present() -> None:
    methods = _methods(ast.parse(SOURCE.read_text(encoding="utf-8")))
    assert set(methods) == set(AUDITED_METHODS)


def test_mjwarp_hot_path_static_audit_is_zero() -> None:
    violations = _static_violations(
        ast.parse(SOURCE.read_text(encoding="utf-8")), DECLARED_COMPLETION_BARRIERS
    )
    assert not violations, "MJWarp hot-path static audit is nonzero:\n" + "\n".join(
        f"{item.method}:{item.line} {item.kind}: {item.expression}" for item in violations
    )


def test_mjwarp_static_audit_detects_bool_tensor_verdicts() -> None:
    tree = ast.parse(
        "class MjwarpBackend:\n"
        "    def step_tensor(self, ctrl):\n"
        "        dirty = bool(self._tracked_body_names)\n"
        "        if not bool(torch.isfinite(ctrl).all()):\n"
        "            raise ValueError\n"
        "    def set_state_tensor_mask(self, mask):\n"
        "        return bool((mask != 0).any())\n"
    )
    kinds = sorted((item.method, item.kind) for item in _static_violations(tree, set()))
    assert kinds == [
        ("set_state_tensor_mask", "host-verdict"),
        ("step_tensor", "host-verdict"),
    ]


def test_mjwarp_operand_finite_verdict_is_gated_by_backend_option() -> None:
    """The shared hot-path operand helper may block only when finite checks are on."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    helper = next(
        item
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "MjwarpBackend"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "_validate_torch_operand"
    )
    guarded: list[ast.If] = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call) and _is_blocking_bool(call) for call in ast.walk(node.test)
        )
    ]
    assert len(guarded) == 1
    gate_attributes = {
        node.attr for node in ast.walk(guarded[0].test) if isinstance(node, ast.Attribute)
    }
    assert "_validate_tensor_finite" in gate_attributes
    # Deferred validation and external CUDA graph capture must never reach
    # the blocking read; they latch a device verdict instead.
    assert "_tensor_validation_deferred" in gate_attributes
