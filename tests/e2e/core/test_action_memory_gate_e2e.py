from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.exceptions import ConfigError


@dataclass
class FakeRule:
    doc_id: str
    content: str
    score: float = 0.95


def test_action_memory_gate_blocks_then_allows_after_read_memory_file(tmp_path: Path, monkeypatch) -> None:
    """Exercise the ToolHandler + memory-read state path end to end."""
    from core.memory import action_gate
    from core.tooling.handler import ToolHandler

    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "procedures").mkdir()
    (anima_dir / "procedures" / "secretary-checklist.md").write_text(
        "# Secretary checklist\n\nCheck duplicates before sending.\n",
        encoding="utf-8",
    )
    memory = MagicMock()
    memory.search_memory_text.return_value = []
    with patch("core.config.models.load_config", side_effect=ConfigError("skip subordinate cache")):
        handler = ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=["gmail"])
    handler._external.dispatch = MagicMock(return_value="draft created")

    rule = FakeRule(
        "rule-e2e",
        (
            "## [ACTION-RULE] Gmail draft duplicate check\n"
            "trigger_tools: gmail_draft\n"
            "---\n"
            'read_memory_file(path="procedures/secretary-checklist.md")'
        ),
    )
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: [rule])

    blocked = json.loads(handler.handle("gmail_draft", {"to": "a@example.com", "body": "hello"}))
    assert blocked["error_type"] == "ActionMemoryGate"
    assert blocked["missing_paths"] == ["procedures/secretary-checklist.md"]

    checklist = handler.handle("read_memory_file", {"path": "procedures/secretary-checklist.md"})
    assert "Check duplicates" in checklist

    assert handler.handle("gmail_draft", {"to": "a@example.com", "body": "hello"}) == "draft created"
