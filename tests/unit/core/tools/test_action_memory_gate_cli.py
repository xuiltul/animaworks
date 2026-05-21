from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class FakeRule:
    doc_id: str
    content: str
    score: float = 0.95


@pytest.mark.parametrize(
    ("argv", "tool_name"),
    [
        (["animaworks-tool", "gmail", "draft", "--to", "a@example.com"], "gmail_draft"),
        (["animaworks-tool", "gmail", "send", "--to", "a@example.com"], "gmail_send"),
        (["animaworks-tool", "chatwork", "send", "--room", "123"], "chatwork_send"),
        (["animaworks-tool", "slack", "send", "--channel", "ops"], "slack_send"),
        (["animaworks-tool", "discord", "send", "--channel", "ops"], "discord_send"),
    ],
)
def test_cli_dispatch_blocks_mapped_action_before_tool_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    tool_name: str,
) -> None:
    from core.memory import action_gate
    from core.tools import cli_dispatch

    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)
    monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))
    monkeypatch.setattr(sys, "argv", argv)
    rule = FakeRule(
        "rule-cli",
        f'## [ACTION-RULE] CLI check\ntrigger_tools: {tool_name}\n---\nread_memory_file(path="procedures/check.md")',
    )
    monkeypatch.setattr(action_gate, "_search_action_rules", lambda *args, **kwargs: [rule])

    with pytest.raises(SystemExit) as exc:
        cli_dispatch()

    assert exc.value.code == 1
    stderr = capsys.readouterr().err
    payload = json.loads(stderr)
    assert payload["error_type"] == "ActionMemoryGate"
    assert payload["tool"] == tool_name
    assert payload["missing_paths"] == ["procedures/check.md"]
