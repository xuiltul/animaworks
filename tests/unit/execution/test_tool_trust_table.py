from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

from core.trust import (
    TOOL_TRUST_LEVELS,
    read_session_trust,
    record_session_trust,
    resolve_tool_trust,
    trust_state_path,
)


def _make_tool_handler(tmp_path: Path):
    from core.tooling.handler import ToolHandler

    anima_dir = tmp_path / "animas" / "trust-test"
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")
    return ToolHandler(
        anima_dir=anima_dir,
        memory=MagicMock(),
        messenger=None,
        tool_registry=[],
    )


def test_trust_table_covers_all_runtime_tool_names(tmp_path: Path, monkeypatch) -> None:
    """New runtime tools must have an explicit trust decision."""
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path / "data"))

    from core.tooling.policy.schemas import build_unified_tool_list
    from core.tooling.policy.surface import MCP_TOOL_NAMES

    names = set(_make_tool_handler(tmp_path)._dispatch)
    names.update(MCP_TOOL_NAMES)
    for trigger in ("task:x", "heartbeat"):
        names.update(
            tool["name"]
            for tool in build_unified_tool_list(
                include_notification_tools=True,
                include_supervisor_tools=True,
                trigger=trigger,
            )
        )

    assert names - TOOL_TRUST_LEVELS.keys() == set()


def test_resolve_tool_trust_normalizes_and_resolves_dynamic_tools() -> None:
    assert resolve_tool_trust("mcp__aw__read_memory_file") == "trusted"
    assert resolve_tool_trust("Read") == "medium"
    assert resolve_tool_trust("Bash", {"command": "animaworks-tool board read general"}) == "untrusted"
    assert resolve_tool_trust("Bash", {"command": "animaworks-tool board dm-history alice bob"}) == "untrusted"
    assert resolve_tool_trust("Bash", {"command": "animaworks-tool board post general hello"}) == "medium"
    assert resolve_tool_trust("Bash", {"command": "pwd"}) == "medium"
    assert resolve_tool_trust("use_tool", {"tool_name": "slack", "action": "messages"}) == "untrusted"
    assert resolve_tool_trust("use_tool") == "untrusted"
    assert resolve_tool_trust("unknown_tool") == "untrusted"


def test_session_trust_is_minimum_and_isolated(tmp_path: Path) -> None:
    record_session_trust(tmp_path, "session-a", 1)
    record_session_trust(tmp_path, "session-a", 2)
    assert read_session_trust(tmp_path, "session-a") == 1
    assert read_session_trust(tmp_path, "session-b") == 2

    record_session_trust(tmp_path, "session-b", 0)
    assert read_session_trust(tmp_path, "session-a") == 1
    assert read_session_trust(tmp_path, "session-b") == 0


def test_invalid_session_id_is_ignored(tmp_path: Path) -> None:
    record_session_trust(tmp_path, "../escape", 0)
    assert read_session_trust(tmp_path, "../escape") == 2
    assert not (tmp_path / "run").exists()


def test_record_session_trust_removes_files_older_than_24_hours(tmp_path: Path) -> None:
    stale = trust_state_path(tmp_path, "stale-session")
    stale.parent.mkdir(parents=True)
    stale.write_text("0", encoding="utf-8")
    old_time = stale.stat().st_mtime - 25 * 60 * 60
    os.utime(stale, (old_time, old_time))

    record_session_trust(tmp_path, "active-session", 1)

    assert not stale.exists()
    assert read_session_trust(tmp_path, "active-session") == 1
