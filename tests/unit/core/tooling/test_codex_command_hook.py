"""Codex PreToolUse hook: deny lists + recursive-search guard (2026-08-30 IO storm)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.tooling import codex_command_hook as hook


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "natsume"
    (d / "state").mkdir(parents=True)
    (d / "knowledge").mkdir()
    (d / "permissions.json").write_text(
        json.dumps({"commands": {"allow_all": True, "deny": ["re:\\brm\\s+(-\\w*[rR]|--recursive)"]}}),
        encoding="utf-8",
    )
    (tmp_path / "companies" / "fs" / "shared" / "task_results").mkdir(parents=True)
    return d


def _decide(anima_dir: Path, command: str, cwd: Path | None = None, global_permissions: Path | None = None):
    return hook.decide(command, cwd=cwd or anima_dir, anima_dir=anima_dir, global_permissions=global_permissions)


@pytest.mark.parametrize(
    "command",
    [
        "grep -RIl foo {data}",
        "grep -R -n --fixed-strings abc {data}/animas/natsume {data}/shared",
        "grep -r foo .",  # cwd is the anima root
        "rg foo {data}/companies/fs/shared",
        "find {data}/companies/fs/shared -type f -iname '*5119*'",
        "grep -n foo state/task_queue.jsonl && grep -RIl bar activity_log",
        "cd {data}/animas && grep -rn foo natsume",
    ],
)
def test_recursive_search_over_data_tree_is_denied(anima_dir: Path, command: str) -> None:
    data = anima_dir.parent.parent
    reason = _decide(anima_dir, command.format(data=data))
    assert reason and "Recursive search" in reason


@pytest.mark.parametrize(
    "command",
    [
        "grep -n 5127 state/task_queue.jsonl",
        "grep -rn foo knowledge/",
        "rg foo {data}/companies/fs/shared/task_results/pr-5127.md",
        "find knowledge -name '*.md'",
        "gh pr view 5127 --json state",
        "ls -la state",
    ],
)
def test_narrow_commands_are_allowed(anima_dir: Path, command: str) -> None:
    data = anima_dir.parent.parent
    assert _decide(anima_dir, command.format(data=data)) is None


def test_per_anima_and_global_deny_lists_apply(anima_dir: Path, tmp_path: Path) -> None:
    assert "denied list" in (_decide(anima_dir, "rm -r knowledge") or "")
    gp = tmp_path / "permissions.global.json"
    gp.write_text(
        json.dumps({"commands": {"deny": [{"pattern": "\\bmkfs\\b", "reason": "Filesystem creation is blocked"}]}}),
        encoding="utf-8",
    )
    assert _decide(anima_dir, "mkfs.ext4 /dev/sda", global_permissions=gp) == "Filesystem creation is blocked"


def test_main_emits_deny_json_and_fails_open(anima_dir: Path, capsys, monkeypatch) -> None:
    import io

    payload = {"tool_input": {"command": "grep -R x " + str(anima_dir.parent.parent)}, "cwd": str(anima_dir)}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert hook.main(["--anima-dir", str(anima_dir)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert hook.main(["--anima-dir", str(anima_dir)]) == 1
