"""Detect hardcoded Japanese strings that should use core.i18n.t().

Enforces the project rule:
  プロンプト・UIメッセージのハードコード禁止:
  ユーザーやAnimaに表示する文字列は core/i18n.py の t() で解決する

Uses a ratchet pattern: a per-file baseline of known violations ensures that
new hardcoded Japanese strings cause immediate test failure, while existing
violations can be fixed incrementally.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# ── Constants ──────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_JAPANESE_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]")

_MIN_JAPANESE_CHARS = 2

_SCAN_DIRS = ("core", "cli", "server")

_EXCLUDED_FILES: set[str] = {
    "core/i18n.py",
}

_EXCLUDED_DIRS: set[str] = {
    "archive",
    "__pycache__",
    ".venv",
    "tests",
    "node_modules",
}

_LOGGER_METHODS: set[str] = {
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
    "log",
}

_RE_FUNC_NAMES: set[str] = {
    "compile",
    "match",
    "search",
    "sub",
    "subn",
    "findall",
    "finditer",
    "fullmatch",
    "split",
}

_REGEX_METACHAR_RE = re.compile(
    r"\\[dDwWsSbB]"
    r"|(?:\(\?[:<>=!])"
    r"|(?:\[[^\]]*\])"
    r"|(?:\\[nrt])"
)


# ── Baseline ─────────────────────────────────────────────────
# Per-file known violation counts.  The test fails if a file *exceeds*
# its baseline (new hardcoded string) or is absent from the baseline.
# When you fix violations, lower the count so the ratchet tightens.

KNOWN_VIOLATIONS: dict[str, int] = {
    # command templates with {返信内容} — borderline (platform-specific CLI syntax)
    "core/_anima_inbox.py": 2,
    # MD section names used for parsing (基本情報, 人格, etc.)
    "core/anima_factory.py": 8,
    # label "コマンド:" in audit output
    "core/audit.py": 1,
    # error messages returned to Anima (GlobalOutboundLimitExceeded etc.)
    "core/cascade_limiter.py": 4,
    # deprecation warning message
    "core/config/cli.py": 1,
    # Japanese day-of-week names for cron migration (月曜, 火曜, ...)
    "core/config/migrate.py": 7,
    # model catalog "note" descriptions (最高性能・推奨, etc.)
    "core/config/models.py": 25,
    # cron instruction prompt to Anima
    "core/supervisor/scheduler_manager.py": 1,
    # label "個人ツール"
    "core/tooling/handler_memory.py": 1,
    # tool descriptions — already have ja/en dict structure
    "core/tooling/prompt_db.py": 26,
    # Japanese field names in schema descriptions (上司, 基本情報)
    "core/tooling/schemas.py": 2,
    # user-facing message (バックグラウンドタスク投入)
    "core/tools/__init__.py": 1,
    # tool guide with Japanese content
    "core/tools/_image_schemas.py": 1,
    "core/tools/aws_collector.py": 1,
    "core/tools/github.py": 1,
    # tool schema descriptions (工作機械, 作業ディレクトリ, etc.) — LLM-facing
    "core/tools/machine.py": 7,
    "core/tools/slack.py": 1,
    # mock task data with Japanese titles
    "server/routes/external_tasks.py": 8,
}


# ── AST Helpers ──────────────────────────────────────────────


def _count_japanese(text: str) -> int:
    return len(_JAPANESE_RE.findall(text))


def _annotate_parents(tree: ast.AST) -> None:
    """Set ``_parent`` and ``_in_type_annotation`` on nodes."""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]

    def _mark_annotation(root: ast.AST) -> None:
        for child in ast.walk(root):
            child._in_type_annotation = True  # type: ignore[attr-defined]

    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and node.annotation:
            _mark_annotation(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns:
                _mark_annotation(node.returns)
            for arg_node in ast.walk(node.args):
                if isinstance(arg_node, ast.arg) and arg_node.annotation:
                    _mark_annotation(arg_node.annotation)


def _ancestors(node: ast.AST) -> list[ast.AST]:
    """Collect ancestor chain (nearest parent first)."""
    chain: list[ast.AST] = []
    cur = node
    while hasattr(cur, "_parent"):
        cur = cur._parent  # type: ignore[attr-defined]
        chain.append(cur)
    return chain


def _is_docstring(node: ast.AST, parent_chain: list[ast.AST]) -> bool:
    if not parent_chain:
        return False
    parent = parent_chain[0]
    if not isinstance(parent, ast.Expr):
        return False
    if len(parent_chain) < 2:
        return False
    container = parent_chain[1]
    if not isinstance(
        container,
        (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
    ):
        return False
    for stmt in container.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        return stmt is parent
    return False


def _in_call_to(parent_chain: list[ast.AST], func_names: set[str]) -> bool:
    """Check if a node sits inside a Call whose func.attr is in *func_names*."""
    for anc in parent_chain:
        if isinstance(anc, ast.Call):
            func = anc.func
            if isinstance(func, ast.Attribute) and func.attr in func_names:
                return True
            break
    return False


def _in_regex_context(node: ast.AST, parent_chain: list[ast.AST]) -> bool:
    for anc in parent_chain:
        if isinstance(anc, ast.Call):
            func = anc.func
            if isinstance(func, ast.Attribute) and func.attr in _RE_FUNC_NAMES:
                return True
            if isinstance(func, ast.Name) and func.id in _RE_FUNC_NAMES:
                return True
            break
    return False


def _looks_like_regex(text: str) -> bool:
    return bool(_REGEX_METACHAR_RE.search(text))


def _in_collection_literal(parent_chain: list[ast.AST]) -> bool:
    if not parent_chain:
        return False
    return isinstance(parent_chain[0], (ast.List, ast.Set, ast.Tuple))


def _in_upper_case_assignment(parent_chain: list[ast.AST]) -> bool:
    """Check if the string is in a value assigned to an UPPER_SNAKE_CASE variable."""
    _upper_re = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
    for anc in parent_chain:
        if isinstance(anc, ast.Assign):
            for target in anc.targets:
                if isinstance(target, ast.Name) and _upper_re.match(target.id):
                    return True
                if isinstance(target, ast.Attribute) and _upper_re.match(target.attr):
                    return True
        if isinstance(anc, ast.AnnAssign) and anc.target:
            t = anc.target
            if isinstance(t, ast.Name) and _upper_re.match(t.id):
                return True
            if isinstance(t, ast.Attribute) and _upper_re.match(t.attr):
                return True
    return False


def _in_comparison(parent_chain: list[ast.AST]) -> bool:
    """Check if the string is used in a comparison (== / != / in / not in)."""
    if not parent_chain:
        return False
    return isinstance(parent_chain[0], ast.Compare)


def _is_type_annotation(node: ast.AST) -> bool:
    return getattr(node, "_in_type_annotation", False)


# ── Violation Scanner ────────────────────────────────────────


def _should_skip(text: str, node: ast.AST, parent_chain: list[ast.AST]) -> bool:
    if _is_docstring(node, parent_chain):
        return True
    if _in_call_to(parent_chain, _LOGGER_METHODS):
        return True
    if _in_regex_context(node, parent_chain):
        return True
    if _looks_like_regex(text):
        return True
    jp_count = _count_japanese(text)
    if jp_count <= 5 and _in_collection_literal(parent_chain):
        return True
    if _in_collection_literal(parent_chain) and _in_upper_case_assignment(parent_chain):
        return True
    return bool(_in_comparison(parent_chain))


def _scan_file(filepath: Path) -> list[tuple[int, str]]:
    """Return list of (line, snippet) for violations in one file."""
    source = filepath.read_text(encoding="utf-8")
    if not _JAPANESE_RE.search(source):
        return []
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    _annotate_parents(tree)
    violations: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if _is_type_annotation(node):
            continue

        if isinstance(node, ast.JoinedStr):
            combined_jp = 0
            parts: list[tuple[str, int]] = []
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    cnt = _count_japanese(part.value)
                    combined_jp += cnt
                    if cnt > 0:
                        parts.append((part.value, part.lineno))
            if combined_jp < _MIN_JAPANESE_CHARS:
                continue
            chain = _ancestors(node)
            if _is_docstring(node, chain):
                continue
            if _in_call_to(chain, _LOGGER_METHODS):
                continue
            if _in_regex_context(node, chain):
                continue
            all_text = "".join(p.value for p in node.values if isinstance(p, ast.Constant) and isinstance(p.value, str))
            if _looks_like_regex(all_text):
                continue
            snippet = all_text[:80].replace("\n", "\\n")
            violations.append((node.lineno, snippet))
            continue

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _count_japanese(node.value) < _MIN_JAPANESE_CHARS:
                continue
            if hasattr(node, "_parent") and isinstance(node._parent, ast.JoinedStr):  # type: ignore[attr-defined]
                continue
            chain = _ancestors(node)
            if _should_skip(node.value, node, chain):
                continue
            snippet = node.value[:80].replace("\n", "\\n")
            violations.append((node.lineno, snippet))

    seen: set[tuple[int, str]] = set()
    deduped: list[tuple[int, str]] = []
    for v in violations:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def scan_all_files() -> dict[str, list[tuple[int, str]]]:
    """Scan the project and return ``{relative_path: [(line, snippet), ...]}``."""
    results: dict[str, list[tuple[int, str]]] = {}
    for scan_dir in _SCAN_DIRS:
        base = _PROJECT_ROOT / scan_dir
        if not base.exists():
            continue
        for py_file in sorted(base.rglob("*.py")):
            rel = str(py_file.relative_to(_PROJECT_ROOT))
            if rel in _EXCLUDED_FILES:
                continue
            if any(part in _EXCLUDED_DIRS for part in py_file.parts):
                continue
            file_violations = _scan_file(py_file)
            if file_violations:
                results[rel] = file_violations
    return results


# ── Tests ────────────────────────────────────────────────────


class TestNoHardcodedJapanese:
    """Ensure no new hardcoded Japanese strings bypass i18n.t()."""

    @pytest.fixture(scope="class")
    def all_violations(self) -> dict[str, list[tuple[int, str]]]:
        return scan_all_files()

    def test_no_new_violations(self, all_violations: dict[str, list[tuple[int, str]]]) -> None:
        """Fail if any file has MORE violations than its known baseline."""
        new_violations: list[str] = []
        for filepath, viols in sorted(all_violations.items()):
            baseline = KNOWN_VIOLATIONS.get(filepath, 0)
            if len(viols) > baseline:
                header = f"\n  {filepath}: {len(viols)} violations (baseline {baseline})"
                details = "".join(f"\n    L{line}: {snip}" for line, snip in viols)
                new_violations.append(header + details)

        if new_violations:
            msg = (
                "New hardcoded Japanese strings detected! "
                "Use core.i18n.t() instead.\n"
                "If this is intentional (e.g. NLP data), add to KNOWN_VIOLATIONS "
                "in test_i18n_hardcode.py."
            )
            pytest.fail(msg + "".join(new_violations))

    def test_baseline_not_stale(self, all_violations: dict[str, list[tuple[int, str]]]) -> None:
        """Warn if baseline is higher than actual (= violations were fixed)."""
        stale: list[str] = []
        for filepath, baseline_count in sorted(KNOWN_VIOLATIONS.items()):
            actual = len(all_violations.get(filepath, []))
            if actual < baseline_count:
                stale.append(
                    f"  {filepath}: baseline={baseline_count}, actual={actual} "
                    f"(reduce baseline by {baseline_count - actual})"
                )
        if stale:
            msg = "KNOWN_VIOLATIONS baseline is stale — violations were fixed! Please update the counts:\n"
            pytest.fail(msg + "\n".join(stale))

    def test_no_unexpected_files(self, all_violations: dict[str, list[tuple[int, str]]]) -> None:
        """Fail if a file not in the baseline introduces violations."""
        unexpected: list[str] = []
        for filepath, viols in sorted(all_violations.items()):
            if filepath not in KNOWN_VIOLATIONS:
                details = "".join(f"\n    L{line}: {snip}" for line, snip in viols)
                unexpected.append(f"\n  {filepath} ({len(viols)} violations):{details}")

        if unexpected:
            msg = (
                "Files with hardcoded Japanese not in KNOWN_VIOLATIONS baseline! "
                "Use core.i18n.t() or add to baseline if intentional."
            )
            pytest.fail(msg + "".join(unexpected))


# ── CLI helper ───────────────────────────────────────────────

if __name__ == "__main__":
    results = scan_all_files()
    total = 0
    for fp, viols in sorted(results.items()):
        baseline = KNOWN_VIOLATIONS.get(fp, 0)
        marker = " ← NEW" if fp not in KNOWN_VIOLATIONS else ""
        over = f" ← +{len(viols) - baseline}" if len(viols) > baseline else ""
        under = f" ← fixed {baseline - len(viols)}" if len(viols) < baseline else ""
        print(f"\n{fp} ({len(viols)} violations, baseline {baseline}){marker}{over}{under}")
        for line, snip in viols:
            print(f"  L{line}: {snip}")
        total += len(viols)

    print(f"\n{'=' * 60}")
    print(f"Total: {total} violations across {len(results)} files")
    print(f"Baseline total: {sum(KNOWN_VIOLATIONS.values())}")
    print("\nTo generate baseline dict:")
    print("KNOWN_VIOLATIONS = {")
    for fp in sorted(results):
        print(f'    "{fp}": {len(results[fp])},')
    print("}")
