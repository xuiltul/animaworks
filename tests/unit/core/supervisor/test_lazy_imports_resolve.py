"""Function-local imports are invisible to import-time tests; check them statically.

2026-09-02: task_runner.execute_task_contract imported sentinels that the
supervisor teardown had removed from pending_executor. Unit tests never
executed that function, so every TaskExec child died with ImportError in
production. This test resolves each function-local ``from core... import``
in the supervisor package against the real module.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[4] / "core" / "supervisor"


def _local_import_froms() -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for path in sorted(_PKG.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.ImportFrom) and sub.module and sub.module.startswith("core."):
                        for alias in sub.names:
                            found.append((path.name, sub.module, alias.name))
    return found


@pytest.mark.parametrize(("filename", "module", "name"), _local_import_froms())
def test_function_local_import_resolves(filename: str, module: str, name: str) -> None:
    mod = importlib.import_module(module)
    assert hasattr(mod, name), f"{filename}: `from {module} import {name}` does not resolve"
