"""Static AST validator for LLM-generated candidate code.

The contract: candidate modules may define
- `make_model() -> River pipeline` (required)
- `featurize(event: dict, state: dict) -> dict` (optional)

Validation rejects anything that imports outside the allowlist, references
escape hatches, or uses dunder name-access. Strong but not airtight — we
still run the result in a subprocess with RLIMIT and (on Linux) no network.
"""

from __future__ import annotations

import ast

ALLOWED_TOP_LEVEL_IMPORTS = {
    "river",
    "numpy",
    "math",
    "collections",
    "datetime",
    "statistics",
    "itertools",
    "typing",
    "dataclasses",
    "re",
    "functools",
    "__future__",
}

FORBIDDEN_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}


class _Validator(ast.NodeVisitor):
    def __init__(self):
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in ALLOWED_TOP_LEVEL_IMPORTS:
                self.errors.append(f"forbidden import: {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = (node.module or "").split(".")[0]
        if mod not in ALLOWED_TOP_LEVEL_IMPORTS:
            self.errors.append(f"forbidden from-import: {node.module}")

    def visit_Name(self, node: ast.Name):
        if node.id in FORBIDDEN_NAMES:
            self.errors.append(f"forbidden name reference: {node.id}")
        if node.id.startswith("__") and node.id.endswith("__") and node.id != "__name__":
            self.errors.append(f"forbidden dunder name: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.errors.append(f"forbidden dunder attribute: {node.attr}")
        self.generic_visit(node)


def validate(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    v = _Validator()
    v.visit(tree)
    return v.errors
