"""Ratchet tests for imports that cross the proposed architecture layers.

Run ``python -m tests.unit.test_layering --update`` from the repository root to
regenerate the reviewed baseline after an intentional dependency cleanup.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("core", "cli", "server")
BASELINE_PATH = Path(__file__).with_name("layering_baseline.txt")

# L7 applications must not depend on one another. Core/server imports from
# CLI packages are forbidden and are asserted separately below.
APP_IMPORTS_FORBIDDEN = frozenset({("cli", "server"), ("server", "cli")})
STRICT_DIRECTIONS = frozenset({("core", "cli"), ("core", "server"), ("server", "cli")})
# Deliberately independent of the generated baseline so --update cannot bless
# any core/server -> CLI dependency.
STRICT_DIRECTION_ALLOWLIST: Counter[tuple[str, str]] = Counter()

# Specific rules take precedence over parent rules. Modules not covered by the
# table use DEFAULT_CORE_LAYER (L4), as allowed by the audit plan.
LAYER_RULES: tuple[tuple[str, int], ...] = (
    ("core.runtime", 6),
    ("server.supervisor", 7),
    ("core.migrations", 6),
    ("core.infra.runtime_init", 6),
    ("core.tooling.handler", 5),
    ("core.lifecycle", 5),
    ("core.anima.emotion_tag", 2),  # Pure emotion-tag parser (only core.schemas); used by chat history.
    ("core.anima.workspace_files", 0),  # Pure filename constants and splitter shared by every layer.
    ("core.voice.emotion_style", 4),  # Shared prompt policy used by the lower-level prompt builder.
    ("core.voice", 5),
    ("core.phone", 5),  # Phone channel built on the voice stack (TTS, voice config).
    ("core.mcp", 5),
    ("core.execution.session.session_types", 2),
    ("core.execution.session.session_context", 2),
    ("core.activity", 2),  # Planned package; no current modules expected.
    ("core.trust", 2),  # Planned package; no current modules expected.
    ("core.text", 2),  # Planned package; no current modules expected.
    ("core.llm.guard", 2),  # Planned package; no current modules expected.
    ("core.llm.oneshot", 3),  # Shared one-shot adapter used by memory and execution flows.
    ("core.llm.helper_completion", 3),  # Registry-aware one-shot adapter shared by helper call sites.
    ("core.credentials", 2),  # Planned package; no current modules expected.
    ("core.platform", 0),
    ("core.exceptions", 0),
    ("core.time_utils", 0),
    ("core.paths", 0),
    ("core.i18n", 0),
    ("core.config", 1),
    ("core.enclave", 1),  # Config-level enclave models + guards, used by core.config.
    ("core.auth", 1),
    ("core.schemas", 1),
    ("core.memory", 3),
    ("core.tasks", 3),
    ("core.skills", 3),
    ("core.org", 3),
    ("core.messaging", 3),
    ("core.notification", 3),
    ("core.usage", 3),
    ("core.channels", 3),  # Planned package; no current modules expected.
    ("core.tooling.policy", 3),  # Planned package; no current modules expected.
    ("core.anima", 5),
    ("core.agent", 5),
    ("core.execution.engines", 4),
    ("core.execution", 4),
    ("core.llm", 4),
    ("core.prompt.builder", 4),
    ("core.prompt", 4),
    ("core.integrations", 4),
)
DEFAULT_CORE_LAYER = 4


def _is_module_or_child(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _module_files() -> dict[str, Path]:
    """Return importable module names and their source paths in scan roots."""
    files: dict[str, Path] = {}
    for package in SCAN_ROOTS:
        for path in sorted((ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).with_suffix("")
            parts = list(relative.parts)
            if parts[-1] == "__init__":
                parts.pop()
            files[".".join(parts)] = path
    return files


def _nearest_module(target: str, modules: set[str]) -> str | None:
    parts = target.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in modules:
            return candidate
        parts.pop()
    return None


def _relative_base(module: str, level: int, imported_module: str | None, is_package: bool) -> str:
    parts = module.split(".")
    if not is_package:
        parts.pop()
    if level > 1:
        parts = parts[: len(parts) - (level - 1)]
    if imported_module:
        parts.extend(imported_module.split("."))
    return ".".join(parts)


class _ImportVisitor(ast.NodeVisitor):
    """Collect runtime local-import edges, including imports in function bodies."""

    def __init__(self, module: str, is_package: bool, modules: set[str]) -> None:
        self.module = module
        self.is_package = is_package
        self.modules = modules
        self.edges: list[tuple[str, str]] = []
        self.type_checking_depth = 0
        self.type_checking_names = {"TYPE_CHECKING"}
        self.typing_module_names = {"typing"}

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in {"typing", "typing_extensions"}:
            self.type_checking_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "TYPE_CHECKING"
            )
        self._add_from_imports(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.typing_module_names.update(alias.asname or alias.name for alias in node.names if alias.name == "typing")
        for alias in node.names:
            self._add_target(alias.name)

    def visit_If(self, node: ast.If) -> None:
        condition = self._is_type_checking(node.test)
        negated = (
            isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and self._is_type_checking(node.test.operand)
        )
        if condition:
            self.type_checking_depth += 1
            for statement in node.body:
                self.visit(statement)
            self.type_checking_depth -= 1
            for statement in node.orelse:
                self.visit(statement)
            return
        if negated:
            for statement in node.body:
                self.visit(statement)
            self.type_checking_depth += 1
            for statement in node.orelse:
                self.visit(statement)
            self.type_checking_depth -= 1
            return
        self.generic_visit(node)

    def _is_type_checking(self, expression: ast.expr) -> bool:
        if isinstance(expression, ast.Name):
            return expression.id in self.type_checking_names
        return (
            isinstance(expression, ast.Attribute)
            and expression.attr == "TYPE_CHECKING"
            and isinstance(expression.value, ast.Name)
            and expression.value.id in self.typing_module_names
        )

    def _add_from_imports(self, node: ast.ImportFrom) -> None:
        if node.level:
            base = _relative_base(self.module, node.level, node.module, self.is_package)
        else:
            base = node.module or ""
        for alias in node.names:
            target = f"{base}.{alias.name}" if alias.name != "*" and base else base
            resolved = _nearest_module(target, self.modules)
            if resolved is None and base:
                resolved = _nearest_module(base, self.modules)
            self._add_resolved(resolved)

    def _add_target(self, target: str) -> None:
        self._add_resolved(_nearest_module(target, self.modules))

    def _add_resolved(self, target: str | None) -> None:
        if target and target != self.module and not self.type_checking_depth:
            self.edges.append((self.module, target))


@lru_cache(maxsize=1)
def collect_import_edges() -> list[tuple[str, str]]:
    """Parse all Python sources and return runtime internal-import edges."""
    files = _module_files()
    modules = set(files)
    edges: list[tuple[str, str]] = []
    for module, path in files.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _ImportVisitor(module, path.name == "__init__.py", modules)
        visitor.visit(tree)
        edges.extend(visitor.edges)
    return edges


def layer_for(module: str) -> int:
    """Return a module's layer, applying the most specific table rule."""
    if module.startswith("core.tooling.handler_"):
        return 5
    for prefix, layer in sorted(LAYER_RULES, key=lambda item: len(item[0]), reverse=True):
        if _is_module_or_child(module, prefix):
            return layer
    if module == "cli" or module.startswith("cli.") or module == "server" or module.startswith("server."):
        return 7
    if module.startswith("core.") or module == "core":
        return DEFAULT_CORE_LAYER
    raise ValueError(f"No layer assigned for module outside scan roots: {module}")


def _app_name(module: str) -> str:
    return module.split(".", maxsplit=1)[0]


def find_layer_violations(edges: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
    """Return imports from a lower layer to a higher layer or forbidden app edges."""
    if edges is None:
        edges = collect_import_edges()
    violations = []
    for importer, imported in edges:
        source_app, target_app = _app_name(importer), _app_name(imported)
        forbidden_direction = (source_app, target_app) in APP_IMPORTS_FORBIDDEN or (
            source_app == "core" and target_app in {"cli", "server"}
        )
        if forbidden_direction or layer_for(importer) < layer_for(imported):
            violations.append((importer, imported))
    return violations


def _read_baseline() -> Counter[tuple[str, str]]:
    if not BASELINE_PATH.exists():
        raise AssertionError(
            f"Missing layer baseline: {BASELINE_PATH}; run python -m tests.unit.test_layering --update"
        )
    entries: Counter[tuple[str, str]] = Counter()
    for line_number, raw_line in enumerate(BASELINE_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(" -> ")
        if len(parts) != 2:
            raise AssertionError(f"Invalid baseline entry at {BASELINE_PATH}:{line_number}: {raw_line!r}")
        entries[(parts[0], parts[1])] += 1
    return entries


def _format_edges(edges: Counter[tuple[str, str]]) -> str:
    return "\n".join(
        f"  {source} -> {target}" + (f" (x{count})" if count > 1 else "")
        for (source, target), count in sorted(edges.items())
    )


def update_baseline() -> None:
    """Rewrite the baseline from current code; review the diff before accepting it."""
    violations = sorted(find_layer_violations())
    content = "".join(f"{source} -> {target}\n" for source, target in violations)
    BASELINE_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {len(violations)} layer violations to {BASELINE_PATH}")


def test_layer_violations_match_baseline() -> None:
    expected = _read_baseline()
    actual = Counter(find_layer_violations())
    added = actual - expected
    removed = expected - actual
    messages = []
    if added:
        messages.append(
            "New layer violations (remove the dependency or intentionally update the baseline):\n"
            + _format_edges(added)
        )
    if removed:
        messages.append(
            "Resolved baseline violations; remove these entries from the baseline:\n" + _format_edges(removed)
        )
    assert not messages, "\n\n".join(messages)


def test_strict_app_directions_have_no_new_edges() -> None:
    """Allow only the exact TODO-listed legacy edges in prohibited directions."""
    actual = Counter(
        edge for edge in find_layer_violations() if (_app_name(edge[0]), _app_name(edge[1])) in STRICT_DIRECTIONS
    )
    unexpected = actual - STRICT_DIRECTION_ALLOWLIST
    assert not unexpected, (
        "core -> cli, core -> server, and server -> cli are forbidden; only the exact "
        "TODO-listed legacy edges are temporarily grandfathered.\n"
        f"Unexpected: {_format_edges(unexpected)}"
    )


def _core_to_server_dynamic_imports() -> list[str]:
    """Find dynamic imports from core modules into the server package."""
    violations = []
    for path in sorted((ROOT / "core").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        importlib_modules = set()
        importlib_callables = set()
        builtins_modules = {"builtins"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "importlib" or alias.name.startswith("importlib."):
                        importlib_modules.add(alias.asname or alias.name.split(".", maxsplit=1)[0])
                    elif alias.name == "builtins":
                        builtins_modules.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("importlib"):
                importlib_callables.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name in {"import_module", "find_spec", "resolve_name"}
                )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            is_dynamic_import = isinstance(function, ast.Name) and (
                function.id == "__import__" or function.id in importlib_callables
            )
            if isinstance(function, ast.Attribute):
                owner = function.value
                is_importlib_call = function.attr in {"import_module", "find_spec", "resolve_name"} and (
                    isinstance(owner, ast.Name)
                    and owner.id in importlib_modules
                    or isinstance(owner, ast.Attribute)
                    and owner.attr == "util"
                    and isinstance(owner.value, ast.Name)
                    and owner.value.id in importlib_modules
                )
                is_builtin_call = (
                    function.attr == "__import__" and isinstance(owner, ast.Name) and owner.id in builtins_modules
                )
                is_dynamic_import = is_dynamic_import or is_importlib_call or is_builtin_call
            if not is_dynamic_import:
                continue
            argument = node.args[0]
            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                continue
            module = argument.value.split(":", maxsplit=1)[0].strip()
            if module == "server" or module.startswith("server."):
                relative = path.relative_to(ROOT).as_posix()
                violations.append(f"{relative}:{node.lineno} -> {argument.value}")
    return violations


def test_core_has_no_dynamic_imports_of_server_modules() -> None:
    """Keep importlib and __import__ references within the same layer boundary."""
    violations = _core_to_server_dynamic_imports()
    assert not violations, "core -> server dynamic imports are forbidden:\n" + "\n".join(violations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="regenerate the reviewed layer violation baseline")
    args = parser.parse_args()
    if args.update:
        update_baseline()
    else:
        parser.error("Use --update to regenerate the baseline, or run this module with pytest")


if __name__ == "__main__":
    main()
