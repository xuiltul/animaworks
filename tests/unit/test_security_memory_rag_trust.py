from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for security: memory/RAG trust protection.

Phase 1: activity_log write protection
Phase 2: min_trust_seen tracking across execution engines
Phase 3: knowledge origin propagation
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.tooling.handler_base import (
    _PROTECTED_DIRS,
    _PROTECTED_FILES,
    _is_protected_write,
)

# ── Helpers ───────────────────────────────────────────────────


def _make_handler(tmp_path: Path):
    """Create a ToolHandler with minimal mocked dependencies."""
    from core.tooling.handler import ToolHandler

    anima_dir = tmp_path / "animas" / "test"
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")

    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.search_memory_text.return_value = []
    memory._get_indexer.return_value = None

    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=None,
        tool_registry=[],
    )
    return handler


# ── Phase 1: activity_log write protection ────────────────────


class TestProtectedDirs:
    """_PROTECTED_DIRS constant and _is_protected_write directory checks."""

    def test_protected_dirs_contains_activity_log(self):
        assert "activity_log" in _PROTECTED_DIRS

    def test_protected_files_unchanged(self):
        assert "permissions.md" in _PROTECTED_FILES
        assert "IDENTITY.md" in _PROTECTED_FILES
        assert "BOOTSTRAP.md" in _PROTECTED_FILES
        assert "state/bm25_longterm_index.json" in _PROTECTED_FILES
        assert "state/bm25_longterm_index.dirty" in _PROTECTED_FILES

    def test_is_protected_write_blocks_activity_log_file(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        target = anima_dir / "activity_log" / "2026-03-01.jsonl"
        result = _is_protected_write(anima_dir, target)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "activity_log" in parsed["message"]

    def test_is_protected_write_blocks_nested_activity_log(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        target = anima_dir / "activity_log" / "subdir" / "file.txt"
        result = _is_protected_write(anima_dir, target)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"

    def test_is_protected_write_allows_knowledge(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        target = anima_dir / "knowledge" / "test.md"
        result = _is_protected_write(anima_dir, target)
        assert result is None

    def test_is_protected_write_allows_episodes(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        target = anima_dir / "episodes" / "2026-03-01.md"
        result = _is_protected_write(anima_dir, target)
        assert result is None

    def test_is_protected_write_still_blocks_protected_files(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        for name in _PROTECTED_FILES:
            target = anima_dir / name
            result = _is_protected_write(anima_dir, target)
            assert result is not None

    def test_write_memory_file_rejects_longterm_bm25_index(self, tmp_path):
        handler = _make_handler(tmp_path)
        result = handler.handle(
            "write_memory_file",
            {"path": "state/bm25_longterm_index.json", "content": '{"documents": []}'},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "protected file" in parsed["message"]

    def test_is_protected_write_blocks_path_traversal(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        target = anima_dir / ".." / "other" / "file.txt"
        result = _is_protected_write(anima_dir, target)
        assert result is not None


class TestActivityLogProtectionViaHandler:
    """Integration test: write_memory_file rejects activity_log writes."""

    def test_write_memory_file_rejects_activity_log(self, tmp_path):
        handler = _make_handler(tmp_path)
        result = handler.handle(
            "write_memory_file",
            {"path": "activity_log/2026-03-01.jsonl", "content": '{"fake": true}'},
        )
        parsed = json.loads(result)
        assert parsed["error_type"] == "PermissionDenied"
        assert "activity_log" in parsed["message"]


# ── Phase 2: min_trust_seen tracking ──────────────────────────


class TestMinTrustSeenToolHandler:
    """ToolHandler _min_trust_seen attribute lifecycle."""

    def test_initial_min_trust_seen_is_trusted(self, tmp_path):
        handler = _make_handler(tmp_path)
        assert handler._min_trust_seen == 2

    def test_reset_session_id_resets_min_trust(self, tmp_path):
        handler = _make_handler(tmp_path)
        handler._min_trust_seen = 0
        handler.reset_session_id()
        assert handler._min_trust_seen == 2


class TestMinTrustSeenLiteLLMTools:
    """ToolProcessingMixin updates _min_trust_seen via _execute_tool_call."""

    @pytest.mark.asyncio
    async def test_execute_tool_call_updates_trust_untrusted(self, tmp_path):
        """Calling web_search should set min_trust_seen to 0 (untrusted)."""
        from core.execution.engines.litellm._litellm_tools import ToolProcessingMixin, _ToolCallShim

        mixin = ToolProcessingMixin()
        handler = _make_handler(tmp_path)
        mixin._tool_handler = handler

        tc = _ToolCallShim(
            id="tc1",
            function=_ToolCallShim._Function(name="web_search", arguments="{}"),
        )
        handler._dispatch["web_search"] = lambda args: "search results"
        handler.handle = MagicMock(return_value="search results")

        await mixin._execute_tool_call(tc, {})
        assert handler._min_trust_seen == 0

    @pytest.mark.asyncio
    async def test_execute_tool_call_stays_trusted(self, tmp_path):
        """Calling search_memory should keep min_trust_seen at 2 (trusted)."""
        from core.execution.engines.litellm._litellm_tools import ToolProcessingMixin, _ToolCallShim

        mixin = ToolProcessingMixin()
        handler = _make_handler(tmp_path)
        mixin._tool_handler = handler

        tc = _ToolCallShim(
            id="tc2",
            function=_ToolCallShim._Function(name="search_memory", arguments="{}"),
        )
        handler.handle = MagicMock(return_value="no results")

        await mixin._execute_tool_call(tc, {})
        assert handler._min_trust_seen == 2

    @pytest.mark.asyncio
    async def test_execute_tool_call_medium_trust(self, tmp_path):
        """Calling read_file should set min_trust_seen to 1 (medium)."""
        from core.execution.engines.litellm._litellm_tools import ToolProcessingMixin, _ToolCallShim

        mixin = ToolProcessingMixin()
        handler = _make_handler(tmp_path)
        mixin._tool_handler = handler

        tc = _ToolCallShim(
            id="tc3",
            function=_ToolCallShim._Function(name="read_file", arguments="{}"),
        )
        handler.handle = MagicMock(return_value="file contents")

        await mixin._execute_tool_call(tc, {})
        assert handler._min_trust_seen == 1

    @pytest.mark.asyncio
    async def test_min_trust_seen_takes_minimum(self, tmp_path):
        """After trusted then untrusted, min_trust_seen should be 0."""
        from core.execution.engines.litellm._litellm_tools import ToolProcessingMixin, _ToolCallShim

        mixin = ToolProcessingMixin()
        handler = _make_handler(tmp_path)
        mixin._tool_handler = handler
        handler.handle = MagicMock(return_value="ok")

        tc_trusted = _ToolCallShim(
            id="tc-a",
            function=_ToolCallShim._Function(name="search_memory", arguments="{}"),
        )
        await mixin._execute_tool_call(tc_trusted, {})
        assert handler._min_trust_seen == 2

        tc_untrusted = _ToolCallShim(
            id="tc-b",
            function=_ToolCallShim._Function(name="web_search", arguments="{}"),
        )
        await mixin._execute_tool_call(tc_untrusted, {})
        assert handler._min_trust_seen == 0

        # Subsequent trusted call should NOT raise the minimum back
        tc_trusted2 = _ToolCallShim(
            id="tc-c",
            function=_ToolCallShim._Function(name="search_memory", arguments="{}"),
        )
        await mixin._execute_tool_call(tc_trusted2, {})
        assert handler._min_trust_seen == 0


class TestMinTrustSeenSDKHook:
    """PreToolUse hook tracks min_trust_seen in session_stats."""

    def test_sdk_hook_trust_tracking(self):
        """SDK and MCP tools resolve through the canonical trust table."""
        from core.trust import resolve_tool_trust

        assert resolve_tool_trust("WebSearch") == "untrusted"
        assert resolve_tool_trust("mcp__aw__search_memory") == "trusted"
        assert resolve_tool_trust("Read") == "medium"

    @pytest.mark.asyncio
    async def test_sdk_hook_persists_trust_by_runtime_session(self, tmp_path):
        from core.execution.engines.claude._sdk_hooks import _build_pre_tool_hook
        from core.execution.session.session_context import RuntimeSessionContext, runtime_session_scope
        from core.trust import read_session_trust

        anima_dir = tmp_path / "animas" / "hook-test"
        anima_dir.mkdir(parents=True)
        ctx = RuntimeSessionContext.create(session_type="chat", thread_id="thread", trigger="chat")
        stats = {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": 0,
            "user_prompt_tokens": 0,
            "trigger": "chat",
            "min_trust_seen": 2,
        }
        hook = _build_pre_tool_hook(anima_dir, session_stats=stats)

        with runtime_session_scope(ctx):
            await hook({"tool_name": "WebSearch", "tool_input": {}}, "tool-use", MagicMock())

        assert stats["min_trust_seen"] == 0
        assert read_session_trust(anima_dir, ctx.tool_session_id) == 0
        assert not (anima_dir / "run" / "min_trust_seen").exists()


# ── Phase 3: knowledge origin propagation ─────────────────────


class TestKnowledgeOriginFrontmatter:
    """write_memory_file inserts origin frontmatter for knowledge/ writes."""

    def test_untrusted_session_adds_external_web_origin(self, tmp_path):
        handler = _make_handler(tmp_path)
        handler._min_trust_seen = 0  # untrusted (e.g., web_search was used)

        anima_dir = handler._anima_dir
        knowledge_dir = anima_dir / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        handler.handle(
            "write_memory_file",
            {"path": "knowledge/test-topic.md", "content": "# Test\nSome content"},
        )

        written = (knowledge_dir / "test-topic.md").read_text(encoding="utf-8")
        assert written.startswith("---")
        assert "origin: external_web" in written
        assert "# Test" in written

    def test_mixed_session_adds_mixed_origin(self, tmp_path):
        handler = _make_handler(tmp_path)
        handler._min_trust_seen = 1  # medium (e.g., read_file was used)

        anima_dir = handler._anima_dir
        knowledge_dir = anima_dir / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        handler.handle(
            "write_memory_file",
            {"path": "knowledge/mixed-topic.md", "content": "# Mixed\nContent"},
        )

        written = (knowledge_dir / "mixed-topic.md").read_text(encoding="utf-8")
        assert written.startswith("---")
        assert "origin: mixed" in written

    def test_trusted_session_no_origin_added(self, tmp_path):
        handler = _make_handler(tmp_path)
        handler._min_trust_seen = 2  # trusted (only trusted tools used)

        anima_dir = handler._anima_dir
        knowledge_dir = anima_dir / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        handler.handle(
            "write_memory_file",
            {"path": "knowledge/trusted-topic.md", "content": "# Trusted\nClean data"},
        )

        written = (knowledge_dir / "trusted-topic.md").read_text(encoding="utf-8")
        assert written.startswith("---")
        assert "origin:" not in written.split("---")[1]  # no origin in frontmatter
        assert "# Trusted" in written
        assert "Clean data" in written

    def test_append_mode_does_not_add_frontmatter(self, tmp_path):
        handler = _make_handler(tmp_path)
        handler._min_trust_seen = 0

        anima_dir = handler._anima_dir
        knowledge_dir = anima_dir / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        (knowledge_dir / "existing.md").write_text("old content", encoding="utf-8")

        handler.handle(
            "write_memory_file",
            {
                "path": "knowledge/existing.md",
                "content": "\nnew content",
                "mode": "append",
            },
        )

        written = (knowledge_dir / "existing.md").read_text(encoding="utf-8")
        assert not written.startswith("---\norigin:")

    def test_non_knowledge_unaffected(self, tmp_path):
        handler = _make_handler(tmp_path)
        handler._min_trust_seen = 0

        anima_dir = handler._anima_dir
        episodes_dir = anima_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        handler.handle(
            "write_memory_file",
            {"path": "episodes/2026-03-01.md", "content": "## 10:00 — Test"},
        )

        written = (episodes_dir / "2026-03-01.md").read_text(encoding="utf-8")
        assert not written.startswith("---\norigin:")

    def test_mode_s_file_trust_fallback(self, tmp_path):
        """A session's isolated trust-state file contributes to the write origin."""
        from core.execution.session.session_context import RuntimeSessionContext
        from core.trust import record_session_trust

        handler = _make_handler(tmp_path)
        ctx = RuntimeSessionContext.create(session_type="chat", thread_id="t", trigger="chat")
        handler.bind_runtime_session(ctx)
        handler._min_trust_seen = 2

        anima_dir = handler._anima_dir
        knowledge_dir = anima_dir / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        record_session_trust(anima_dir, ctx.tool_session_id, 0)

        handler.handle(
            "write_memory_file",
            {"path": "knowledge/from-sdk.md", "content": "# SDK Data"},
        )

        written = (knowledge_dir / "from-sdk.md").read_text(encoding="utf-8")
        assert written.startswith("---")
        assert "origin: external_web" in written
