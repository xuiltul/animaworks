"""Codex ``PreToolUse`` hook: deny shell commands by the rules the other engines enforce.

Codex's native shell tool runs inside ``codex-linux-sandbox`` and never passes
through ``execute_command`` permission checks, so ``permissions.global.json``
and per-anima ``commands.deny`` were unenforced for codex animas.
``codex_sdk`` writes ``$CODEX_HOME/hooks.json`` pointing here; Codex feeds the
tool call on stdin and honours ``permissionDecision: deny``.

Also denies recursive searches over the runtime data tree: on 2026-08-30
dozens of ``grep -R … ~/.animaworks`` (activity_log is >1GB per anima)
saturated disk IO and hang-killed every TaskExec for hours.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

_SEGMENT_SPLIT = re.compile(r"\|(?!\|)|&&|\|\||;|\n|\$\(|`|<\(|\)")
_GREP_FAMILY = {"grep", "egrep", "fgrep"}
_ALWAYS_RECURSIVE = {"rg", "ag", "ack", "find", "fd", "fdfind", "du"}
_GREP_RECURSIVE_FLAGS = {"--recursive", "--dereference-recursive"}
# Directories that are always too big to sweep, wherever they sit.
_HEAVY_DIR_NAMES = {"activity_log", "logs", ".codex_home", "task_results"}

# ponytail: pure heuristic over argv; no shell AST. Good enough for grep/rg/find.


def _is_broad_data_root(path: Path, data_dir: Path) -> bool:
    """True when *path* is the data dir or one of its wide top-level trees."""
    try:
        rel = path.resolve().relative_to(data_dir.resolve())
    except (ValueError, OSError):
        return False
    parts = rel.parts
    # A heavy dir itself (or any dir beneath one) is too big; a single file inside is fine.
    if parts and (parts[-1] in _HEAVY_DIR_NAMES or (any(p in _HEAVY_DIR_NAMES for p in parts) and path.is_dir())):
        return True
    if len(parts) == 0:
        return True
    head = parts[0]
    if head in {"animas", "companies", "shared"} and len(parts) <= 2:
        return True
    if head == "companies" and len(parts) == 3 and parts[2] == "shared":
        return True
    return head == "animas" and len(parts) == 3 and parts[2] == "state"


def _search_targets(argv: list[str]) -> list[str] | None:
    """Return the path operands of a recursive search command, or None if not one."""
    base = Path(argv[0]).name
    operands = [a for a in argv[1:] if not a.startswith("-")]
    if base in _GREP_FAMILY:
        flags = [a for a in argv[1:] if a.startswith("-")]
        recursive = any(
            f in _GREP_RECURSIVE_FLAGS or (not f.startswith("--") and ("r" in f or "R" in f)) for f in flags
        )
        if not recursive:
            return None
        if "-e" in flags or "-f" in flags or "--regexp" in flags:
            paths = operands
        else:
            paths = operands[1:]  # first operand is the pattern
        return paths or ["."]
    if base in {"rg", "ag", "ack"}:
        paths = operands if "-e" in argv or "--regexp" in argv or "--files" in argv else operands[1:]
        return paths or ["."]
    if base in {"find", "fd", "fdfind", "du"}:
        if base == "find":
            paths: list[str] = []
            for a in argv[1:]:
                if a.startswith("-") or a in {"(", "!", ")"}:
                    break
                paths.append(a)
            return paths or ["."]
        return operands[1:] if base in {"fd", "fdfind"} and len(operands) >= 2 else (operands or ["."])
    return None


def check_recursive_search(command: str, cwd: Path, data_dir: Path) -> str | None:
    for segment in (s.strip() for s in _SEGMENT_SPLIT.split(command) if s.strip()):
        try:
            argv = shlex.split(segment)
        except ValueError:
            continue
        # Skip leading env assignments (FOO=bar cmd …).
        while argv and "=" in argv[0] and not argv[0].startswith("-"):
            argv = argv[1:]
        if not argv:
            continue
        if argv[0] == "cd":  # track cwd for the following segments
            cwd = (cwd / Path(argv[1]).expanduser()) if len(argv) > 1 else cwd
            continue
        targets = _search_targets(argv)
        if not targets:
            continue
        for raw in targets:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = cwd / path
            if _is_broad_data_root(path, data_dir):
                return (
                    f"Recursive search over '{raw}' is denied: the runtime data tree is "
                    "multi-GB and sweeping it stalls every running task. Search a specific "
                    "subdirectory or file (e.g. state/task_queue.jsonl, knowledge/, shared/task_results/<file>)."
                )
    return None


def check_deny_lists(command: str, anima_dir: Path, global_permissions: Path | None) -> str | None:
    from core.config.global_permissions import _compile_patterns
    from core.config.schemas import GlobalPermissionsConfig, command_deny_matches, load_permissions

    if global_permissions is not None and global_permissions.is_file():
        config = GlobalPermissionsConfig.model_validate(json.loads(global_permissions.read_text(encoding="utf-8")))
        for pattern, reason in _compile_patterns(config.commands.deny):
            if pattern.search(command):
                return reason
    denied_items = load_permissions(anima_dir).commands.deny
    if not denied_items:
        return None
    for segment in (s.strip() for s in _SEGMENT_SPLIT.split(command) if s.strip()):
        try:
            argv = shlex.split(segment)
        except ValueError:
            continue
        if not argv:
            continue
        for denied in denied_items:
            if command_deny_matches(denied, segment, argv[0]):
                return f"Command '{argv[0]}' is in denied list ('{denied}')"
    return None


def decide(command: str, *, cwd: Path, anima_dir: Path, global_permissions: Path | None) -> str | None:
    """Return a denial reason, or None to allow."""
    data_dir = anima_dir.resolve().parent.parent
    return check_deny_lists(command, anima_dir, global_permissions) or check_recursive_search(command, cwd, data_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anima-dir", required=True, type=Path)
    parser.add_argument("--global-permissions", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        payload = json.load(sys.stdin)
        command = str((payload.get("tool_input") or {}).get("command") or "")
        if not command:
            return 0
        cwd = Path(str(payload.get("cwd") or args.anima_dir))
        reason = decide(command, cwd=cwd, anima_dir=args.anima_dir, global_permissions=args.global_permissions)
    except Exception as exc:  # fail open: a broken hook must not freeze the fleet
        print(f"codex_command_hook error: {exc}", file=sys.stderr)
        return 1
    if reason is None:
        return 0
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
