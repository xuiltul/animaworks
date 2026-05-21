"""Tests for Action-Aware Priming in PreToolUse hook."""

from __future__ import annotations

import asyncio
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import patch

import pytest


@dataclass
class FakeRule:
    doc_id: str
    content: str
    score: float = 0.95


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory structure."""
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "test.md").write_text("# test")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return tmp_path


@pytest.fixture
def session_stats() -> dict:
    """Fresh session stats with all required fields."""
    import time

    return {
        "trigger": "chat",
        "start_time": time.monotonic(),
        "tool_call_count": 0,
        "system_prompt_tokens": 5000,
        "user_prompt_tokens": 1000,
        "total_result_bytes": 0,
        "min_trust_seen": 2,
    }


class TestActionAwarePrimingHook:
    """Test the Action-Aware Priming logic within _build_pre_tool_hook."""

    def _build_hook(self, anima_dir: Path, session_stats: dict):
        """Build the pre_tool_hook with mocked retriever."""
        from core.execution._sdk_hooks import _build_pre_tool_hook

        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=8192,
            context_window=200_000,
            session_stats=session_stats,
            superuser=False,
        )
        return hook

    @pytest.mark.asyncio
    async def test_non_output_tool_passes_through(self, anima_dir, session_stats):
        """Tools not in whitelist should not trigger AAP."""
        hook = self._build_hook(anima_dir, session_stats)
        result = await hook(
            {"hook_event_name": "PreToolUse", "tool_name": "search_memory", "tool_input": {}, "tool_use_id": "t1"},
            "t1",
            {"signal": None},
        )
        assert result.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"

    @pytest.mark.asyncio
    async def test_output_tool_with_no_retriever(self, anima_dir, session_stats):
        """If retriever fails to init, should pass through."""
        hook = self._build_hook(anima_dir, session_stats)
        with patch("core.memory.rag.singleton.get_vector_store", return_value=None):
            result = await hook(
                {"hook_event_name": "PreToolUse", "tool_name": "call_human", "tool_input": {"message": "test"}, "tool_use_id": "t1"},
                "t1",
                {"signal": None},
            )
        assert result.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"

    @pytest.mark.asyncio
    async def test_mcp_name_normalizes_and_required_read_denies(self, anima_dir, session_stats, monkeypatch):
        """MCP-prefixed tool names use canonical action-rule names and required-read behavior."""
        from core.memory import action_gate

        seen_tools: list[str] = []

        def fake_search(_anima_dir, tool_name, _query):
            seen_tools.append(tool_name)
            return [
                FakeRule(
                    "rule-required",
                    (
                        "## [ACTION-RULE] send check\n"
                        "trigger_tools: send_message\n"
                        "---\n"
                        'read_memory_file(path="procedures/check.md")'
                    ),
                )
            ]

        monkeypatch.setattr(action_gate, "_search_action_rules", fake_search)
        hook = self._build_hook(anima_dir, session_stats)
        result = await hook(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "mcp__aw__send_message",
                "tool_input": {"content": "test"},
                "tool_use_id": "t1",
            },
            "t1",
            {"signal": None},
        )
        assert result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
        assert seen_tools == ["send_message"]
        assert session_stats["action_gate_denied_count"] == 1

    @pytest.mark.asyncio
    async def test_required_read_allows_after_memory_read(self, anima_dir, session_stats, monkeypatch):
        """Required-read rules keep blocking until the required path is read in the same session."""
        from core.memory import action_gate

        monkeypatch.setattr(
            action_gate,
            "_search_action_rules",
            lambda *args, **kwargs: [
                FakeRule(
                    "rule-required",
                    (
                        "## [ACTION-RULE] notify check\n"
                        "trigger_tools: call_human\n"
                        "---\n"
                        'read_memory_file(path="procedures/check.md")'
                    ),
                )
            ],
        )

        hook = self._build_hook(anima_dir, session_stats)
        blocked = await hook(
            {"hook_event_name": "PreToolUse", "tool_name": "call_human", "tool_input": {"message": "test"}, "tool_use_id": "t2"},
            "t2",
            {"signal": None},
        )
        assert blocked.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"

        action_gate.record_memory_read(anima_dir, "procedures/check.md")
        allowed = await hook(
            {"hook_event_name": "PreToolUse", "tool_name": "call_human", "tool_input": {"message": "test"}, "tool_use_id": "t3"},
            "t3",
            {"signal": None},
        )
        assert allowed.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"

    @pytest.mark.asyncio
    async def test_review_only_rule_blocks_once_then_allows(self, anima_dir, session_stats, monkeypatch):
        """Review-only rules are deduplicated by action-gate state."""
        from core.memory import action_gate

        monkeypatch.setattr(
            action_gate,
            "_search_action_rules",
            lambda *args, **kwargs: [
                FakeRule("chunk_123", "## [ACTION-RULE] review\ntrigger_tools: send_message\n---\nReview.")
            ],
        )

        hook = self._build_hook(anima_dir, session_stats)
        first = await hook(
            {"hook_event_name": "PreToolUse", "tool_name": "send_message", "tool_input": {"content": "test"}, "tool_use_id": "t3"},
            "t3",
            {"signal": None},
        )
        second = await hook(
            {"hook_event_name": "PreToolUse", "tool_name": "send_message", "tool_input": {"content": "test"}, "tool_use_id": "t4"},
            "t4",
            {"signal": None},
        )
        assert first.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
        assert second.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"

    @pytest.mark.asyncio
    async def test_no_global_two_pause_limit(self, anima_dir, session_stats, monkeypatch):
        """Different review-only rules can block more than twice in one session."""
        from core.memory import action_gate

        rules = [
            FakeRule("r1", "## [ACTION-RULE] r1\ntrigger_tools: send_message\n---\nReview 1."),
            FakeRule("r2", "## [ACTION-RULE] r2\ntrigger_tools: send_message\n---\nReview 2."),
            FakeRule("r3", "## [ACTION-RULE] r3\ntrigger_tools: send_message\n---\nReview 3."),
        ]

        def fake_search(*args, **kwargs):
            return [rules.pop(0)]

        monkeypatch.setattr(action_gate, "_search_action_rules", fake_search)
        hook = self._build_hook(anima_dir, session_stats)

        for idx in range(3):
            result = await hook(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "send_message",
                    "tool_input": {"content": f"test {idx}"},
                    "tool_use_id": f"t{idx}",
                },
                f"t{idx}",
                {"signal": None},
            )
            assert result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
        assert session_stats["action_gate_denied_count"] == 3

    @pytest.mark.asyncio
    async def test_session_stats_none_graceful(self, anima_dir):
        """When session_stats is None, hook should not crash."""
        from core.execution._sdk_hooks import _build_pre_tool_hook

        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=8192,
            context_window=200_000,
            session_stats=None,
            superuser=False,
        )
        result = await hook(
            {"hook_event_name": "PreToolUse", "tool_name": "call_human", "tool_input": {}, "tool_use_id": "t1"},
            "t1",
            {"signal": None},
        )
        assert result.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"


class TestActionAwarePrimingInit:
    """Test AAP state initialization."""

    def test_session_stats_initialized(self, anima_dir, session_stats):
        """Action gate fields should be auto-initialized in session_stats."""
        from core.execution._sdk_hooks import _build_pre_tool_hook

        _build_pre_tool_hook(
            anima_dir,
            max_tokens=8192,
            context_window=200_000,
            session_stats=session_stats,
            superuser=False,
        )
        assert session_stats["action_gate_denied_count"] == 0
