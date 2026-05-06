"""Unit tests for PreCompact blocking and compaction_blocked → session chaining."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ── Mock claude_agent_sdk before importing ───
_mock_sdk = MagicMock()
_mock_types = MagicMock()


def _sync_hook_json_output(**kwargs: Any) -> dict[str, Any]:
    """Mimic SyncHookJSONOutput as a plain dict."""
    return dict(kwargs)


_mock_types.SyncHookJSONOutput = _sync_hook_json_output
_mock_types.PreToolUseHookSpecificOutput = dict
_mock_types.HookInput = dict
_mock_types.HookContext = dict

sys.modules.setdefault("claude_agent_sdk", _mock_sdk)
sys.modules.setdefault("claude_agent_sdk.types", _mock_types)

from core.execution._sdk_hooks import (  # noqa: E402
    _build_pre_compact_hook,
    _build_pre_tool_hook,
    _log_compaction_event,
)
from core.prompt.context import CHARS_PER_TOKEN  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory for hook construction."""
    d = tmp_path / "animas" / "test-precompact"
    for sub in ("state", "episodes", "knowledge", "procedures",
                "skills", "shortterm", "activity_log"):
        (d / sub).mkdir(parents=True)
    (d / "identity.md").write_text("# Test", encoding="utf-8")
    (d / "injection.md").write_text("", encoding="utf-8")
    (d / "permissions.md").write_text("", encoding="utf-8")
    return d


def _make_session_stats(**overrides: Any) -> dict[str, Any]:
    """Create default session_stats dict with optional overrides."""
    defaults: dict[str, Any] = {
        "tool_call_count": 0,
        "total_result_bytes": 0,
        "system_prompt_tokens": 1000,
        "user_prompt_tokens": 500,
        "force_chain": False,
    }
    defaults.update(overrides)
    return defaults


# ── PreCompact hook: blocking tests ──────────────────────────


class TestPreCompactBlock:
    """Test that PreCompact hook blocks SDK auto-compact."""

    @pytest.mark.asyncio
    async def test_blocks_auto_compact(self, anima_dir: Path):
        """PreCompact returns decision='block' for auto trigger."""
        stats = _make_session_stats()
        hook = _build_pre_compact_hook(
            anima_dir, session_stats=stats, context_window=200_000
        )
        result = await hook({"trigger": "auto"}, None, {})

        assert result.get("decision") == "block"
        assert "reason" in result
        assert stats.get("compaction_blocked") is True

    @pytest.mark.asyncio
    async def test_sets_compaction_blocked_flag(self, anima_dir: Path):
        """After blocking, compaction_blocked flag is True in session_stats."""
        stats = _make_session_stats()
        hook = _build_pre_compact_hook(
            anima_dir, session_stats=stats, context_window=200_000
        )
        await hook({"trigger": "auto"}, None, {})

        assert stats["compaction_blocked"] is True

    @pytest.mark.asyncio
    async def test_no_session_stats_does_not_crash(self, anima_dir: Path):
        """When session_stats is None, hook blocks but doesn't crash."""
        hook = _build_pre_compact_hook(
            anima_dir, session_stats=None, context_window=200_000
        )
        result = await hook({"trigger": "auto"}, None, {})

        assert result.get("decision") == "block"


class TestPreCompactRecoverySafetyValve:
    """Test the recovery safety valve (95% threshold)."""

    @pytest.mark.asyncio
    async def test_allows_compaction_at_95_percent(self, anima_dir: Path):
        """When context > 95% of window, SDK compaction is allowed through."""
        context_window = 200_000
        stats = _make_session_stats(
            system_prompt_tokens=150_000,
            user_prompt_tokens=50_000,
            total_result_bytes=10_000 * CHARS_PER_TOKEN,
        )
        hook = _build_pre_compact_hook(
            anima_dir, session_stats=stats, context_window=context_window
        )
        result = await hook({"trigger": "auto"}, None, {})

        assert result.get("decision") is None
        assert stats.get("compaction_blocked") is not True

    @pytest.mark.asyncio
    async def test_blocks_below_95_percent(self, anima_dir: Path):
        """When context < 95%, SDK compaction is blocked."""
        context_window = 200_000
        stats = _make_session_stats(
            system_prompt_tokens=50_000,
            user_prompt_tokens=10_000,
            total_result_bytes=0,
        )
        hook = _build_pre_compact_hook(
            anima_dir, session_stats=stats, context_window=context_window
        )
        result = await hook({"trigger": "auto"}, None, {})

        assert result.get("decision") == "block"
        assert stats["compaction_blocked"] is True

    @pytest.mark.asyncio
    async def test_exact_95_percent_boundary_blocks(self, anima_dir: Path):
        """At exactly 95%, compaction is NOT triggered (need strictly greater)."""
        context_window = 200_000
        threshold_tokens = int(context_window * 0.95)
        stats = _make_session_stats(
            system_prompt_tokens=threshold_tokens,
            user_prompt_tokens=0,
            total_result_bytes=0,
        )
        hook = _build_pre_compact_hook(
            anima_dir, session_stats=stats, context_window=context_window
        )
        result = await hook({"trigger": "auto"}, None, {})

        assert result.get("decision") == "block"
        assert stats["compaction_blocked"] is True


# ── PreToolUse hook: compaction_blocked detection ────────────


class TestPreToolUseCompactionBlockedDetection:
    """Test PreToolUse detects compaction_blocked and ends session."""

    @pytest.mark.asyncio
    async def test_detects_compaction_blocked(self, anima_dir: Path):
        """When compaction_blocked is True, returns continue_=False."""
        stats = _make_session_stats(compaction_blocked=True)
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}}
        result = await hook(input_data, "tool-123", {})

        assert result.get("continue_") is False

    @pytest.mark.asyncio
    async def test_sets_force_chain_true(self, anima_dir: Path):
        """After detecting compaction_blocked, force_chain is set to True."""
        stats = _make_session_stats(compaction_blocked=True)
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}}
        await hook(input_data, "tool-123", {})

        assert stats["force_chain"] is True

    @pytest.mark.asyncio
    async def test_consumes_compaction_blocked_flag(self, anima_dir: Path):
        """After detection, compaction_blocked is reset to False."""
        stats = _make_session_stats(compaction_blocked=True)
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}}
        await hook(input_data, "tool-123", {})

        assert stats["compaction_blocked"] is False

    @pytest.mark.asyncio
    async def test_no_compaction_blocked_passes_through(self, anima_dir: Path):
        """Without compaction_blocked, hook proceeds normally."""
        stats = _make_session_stats()
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}}
        result = await hook(input_data, "tool-123", {})

        assert result.get("continue_") is not False
        assert stats["force_chain"] is False

    @pytest.mark.asyncio
    async def test_no_session_stats_passes_through(self, anima_dir: Path):
        """Without session_stats, compaction_blocked check is skipped."""
        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=None,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}}
        result = await hook(input_data, "tool-123", {})

        assert result.get("continue_") is not False


# ── Two-stage integration: PreCompact → PreToolUse ───────────


class TestTwoStageIntegration:
    """Test the full PreCompact block → PreToolUse session end flow."""

    @pytest.mark.asyncio
    async def test_full_flow_block_then_session_end(self, anima_dir: Path):
        """PreCompact blocks → PreToolUse ends session with force_chain."""
        stats = _make_session_stats()

        compact_hook = _build_pre_compact_hook(
            anima_dir, session_stats=stats, context_window=200_000
        )
        compact_result = await compact_hook({"trigger": "auto"}, None, {})
        assert compact_result.get("decision") == "block"
        assert stats["compaction_blocked"] is True

        tool_hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=stats,
        )
        tool_result = await tool_hook(
            {"tool_name": "Bash", "tool_input": {"command": "echo hi"}},
            "tool-456",
            {},
        )
        assert tool_result.get("continue_") is False
        assert stats["force_chain"] is True
        assert stats["compaction_blocked"] is False

    @pytest.mark.asyncio
    async def test_second_tool_call_proceeds_normally(self, anima_dir: Path):
        """After compaction_blocked is consumed, subsequent tools proceed."""
        stats = _make_session_stats(compaction_blocked=True)

        hook = _build_pre_tool_hook(
            anima_dir,
            max_tokens=4096,
            context_window=200_000,
            session_stats=stats,
        )
        input_data = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}}

        result1 = await hook(input_data, "t1", {})
        assert result1.get("continue_") is False
        assert stats["compaction_blocked"] is False

        result2 = await hook(input_data, "t2", {})
        assert result2.get("continue_") is not False


# ── _log_compaction_event ────────────────────────────────────


class TestLogCompactionEvent:
    """Test the _log_compaction_event helper."""

    def test_does_not_raise_on_missing_activity_log(self, anima_dir: Path):
        """Logging should not raise even if ActivityLogger fails."""
        _log_compaction_event(anima_dir, "auto", blocked=True)

    def test_does_not_raise_on_allowed(self, anima_dir: Path):
        """Logging allowed compaction should not raise."""
        _log_compaction_event(anima_dir, "auto", blocked=False)
