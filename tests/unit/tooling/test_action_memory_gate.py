from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.exceptions import ConfigError


@dataclass
class FakeRule:
    doc_id: str
    content: str
    score: float = 0.95


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "mei"
    (d / "knowledge").mkdir(parents=True)
    (d / "procedures").mkdir()
    (d / "procedures" / "check.md").write_text("# Check\n\nConfirm before action.\n", encoding="utf-8")
    return d


@pytest.fixture
def handler(anima_dir: Path):
    from core.tooling.handler import ToolHandler

    memory = MagicMock()
    memory.search_memory_text.return_value = []
    with patch("core.config.models.load_config", side_effect=ConfigError("skip subordinate cache")):
        h = ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=["gmail"])
    h._external.dispatch = MagicMock(return_value="draft ok")
    return h


def test_handler_blocks_external_action_until_required_memory_is_read(anima_dir: Path, handler, monkeypatch) -> None:
    from core.memory import action_gate

    rule = FakeRule(
        "rule-1",
        '## [ACTION-RULE] Gmail draft check\ntrigger_tools: gmail_draft\n---\nread_memory_file(path="procedures/check.md")',
    )
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: [rule])

    blocked = handler.handle("gmail_draft", {"to": "a@example.com", "body": "hello"})
    blocked_payload = json.loads(blocked)

    assert blocked_payload["error_type"] == "ActionMemoryGate"
    assert blocked_payload["missing_paths"] == ["procedures/check.md"]
    handler._external.dispatch.assert_not_called()

    read_result = handler.handle("read_memory_file", {"path": "procedures/check.md"})
    assert "Confirm before action" in read_result

    allowed = handler.handle("gmail_draft", {"to": "a@example.com", "body": "hello"})
    assert allowed == "draft ok"
    handler._external.dispatch.assert_called_once()


@pytest.mark.parametrize("tool_name", ["call_human", "send_message", "post_channel", "write_memory_file"])
def test_handler_blocks_core_side_effect_actions_before_dispatch(
    anima_dir: Path,
    handler,
    monkeypatch,
    tool_name: str,
) -> None:
    from core.memory import action_gate

    rule = FakeRule(
        f"rule-{tool_name}",
        (
            f"## [ACTION-RULE] {tool_name} check\n"
            f"trigger_tools: {tool_name}\n"
            "---\n"
            'read_memory_file(path="procedures/check.md")'
        ),
    )
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: [rule])

    result = handler.handle(tool_name, {"text": "hello", "path": "note.md", "content": "hello"})
    payload = json.loads(result)

    assert payload["error_type"] == "ActionMemoryGate"
    assert payload["tool"] == tool_name
    assert payload["missing_paths"] == ["procedures/check.md"]


def test_use_tool_is_gated_by_schema_name_before_dispatch(anima_dir: Path, handler, monkeypatch) -> None:
    from core.memory import action_gate

    rule = FakeRule(
        "rule-2",
        '## [ACTION-RULE] use_tool check\ntrigger_tools: gmail_draft\n---\nread_memory_file(path="procedures/check.md")',
    )
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: [rule])

    result = handler.handle(
        "use_tool",
        {"tool_name": "gmail", "action": "draft", "args": {"to": "a@example.com", "body": "hello"}},
    )
    payload = json.loads(result)

    assert payload["error_type"] == "ActionMemoryGate"
    assert payload["tool"] == "gmail_draft"


def test_handler_gate_fails_open_when_search_fails(anima_dir: Path, handler, monkeypatch) -> None:
    from core.memory import action_gate

    def raise_search(*args, **kwargs):
        raise RuntimeError("search unavailable")

    monkeypatch.setattr(action_gate, "_search_action_rules", raise_search)

    result = handler.handle("gmail_draft", {"to": "a@example.com", "body": "hello"})

    assert result == "draft ok"
