from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for security: memory/RAG trust protection.

Phase 1: activity_log write protection
Phase 2: min_trust_seen tracking across execution engines
Phase 3: knowledge origin propagation + consolidation origin chain
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        assert "identity.md" in _PROTECTED_FILES
        assert "bootstrap.md" in _PROTECTED_FILES

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
        from core.execution._litellm_tools import ToolProcessingMixin, _ToolCallShim

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
        from core.execution._litellm_tools import ToolProcessingMixin, _ToolCallShim

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
        from core.execution._litellm_tools import ToolProcessingMixin, _ToolCallShim

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
        from core.execution._litellm_tools import ToolProcessingMixin, _ToolCallShim

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
        """Verify _SDK_TOOL_TRUST mappings are consistent."""
        from core.execution._sanitize import TOOL_TRUST_LEVELS

        assert TOOL_TRUST_LEVELS.get("web_search") == "untrusted"
        assert TOOL_TRUST_LEVELS.get("search_memory") == "trusted"
        assert TOOL_TRUST_LEVELS.get("read_file") == "medium"


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
        """When _min_trust_seen is 2 (default) but run/min_trust_seen file says 0."""
        handler = _make_handler(tmp_path)
        handler._min_trust_seen = 2

        anima_dir = handler._anima_dir
        knowledge_dir = anima_dir / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        run_dir = anima_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "min_trust_seen").write_text("0", encoding="utf-8")

        handler.handle(
            "write_memory_file",
            {"path": "knowledge/from-sdk.md", "content": "# SDK Data"},
        )

        written = (knowledge_dir / "from-sdk.md").read_text(encoding="utf-8")
        assert written.startswith("---")
        assert "origin: external_web" in written


# ── Phase 3: consolidation origin chain ───────────────────────


class TestConsolidationOriginChain:
    """ConsolidationEngine respects origin during RAG index updates."""

    def test_has_external_origin_detects_external_web(self, tmp_path):
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(tmp_path, "test")
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        (knowledge_dir / "external.md").write_text(
            "---\norigin: external_web\n---\n\n# External Data",
            encoding="utf-8",
        )
        assert engine._has_external_origin_in_files(["external.md"]) is True

    def test_has_external_origin_detects_mixed(self, tmp_path):
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(tmp_path, "test")
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        (knowledge_dir / "mixed.md").write_text(
            "---\norigin: mixed\n---\n\n# Mixed Data",
            encoding="utf-8",
        )
        assert engine._has_external_origin_in_files(["mixed.md"]) is True

    def test_has_external_origin_clean_files_return_false(self, tmp_path):
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(tmp_path, "test")
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        (knowledge_dir / "clean.md").write_text(
            "---\nconfidence: 0.8\n---\n\n# Clean Data",
            encoding="utf-8",
        )
        assert engine._has_external_origin_in_files(["clean.md"]) is False

    def test_has_external_origin_no_frontmatter_returns_false(self, tmp_path):
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(tmp_path, "test")
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        (knowledge_dir / "legacy.md").write_text(
            "# Legacy knowledge without frontmatter",
            encoding="utf-8",
        )
        assert engine._has_external_origin_in_files(["legacy.md"]) is False

    def test_has_external_origin_nonexistent_file(self, tmp_path):
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(tmp_path, "test")
        assert engine._has_external_origin_in_files(["nonexistent.md"]) is False

    def test_update_rag_index_downgrades_with_external_source(self, tmp_path):
        """When source_files contain external origins, origin is downgraded."""
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(tmp_path, "test")
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        (knowledge_dir / "source_ext.md").write_text(
            "---\norigin: external_web\n---\n\nExternal source data",
            encoding="utf-8",
        )
        (knowledge_dir / "output.md").write_text(
            "---\nconfidence: 0.8\n---\n\nConsolidated output",
            encoding="utf-8",
        )

        mock_indexer = MagicMock()
        mock_store = MagicMock()

        with patch("core.memory.rag.MemoryIndexer", return_value=mock_indexer):
            with patch("core.memory.rag.singleton.get_vector_store", return_value=mock_store):
                engine._update_rag_index(
                    ["output.md"],
                    origin="consolidation",
                    source_files=["source_ext.md"],
                )

        if mock_indexer.index_file.called:
            call_kwargs = mock_indexer.index_file.call_args
            assert call_kwargs[1].get("origin") == "consolidation_external"

    def test_update_rag_index_keeps_consolidation_without_external(self, tmp_path):
        """When no external sources, origin stays 'consolidation'."""
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(tmp_path, "test")
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        (knowledge_dir / "clean_src.md").write_text(
            "---\nconfidence: 0.9\n---\n\nClean source",
            encoding="utf-8",
        )
        (knowledge_dir / "output2.md").write_text(
            "Consolidated output",
            encoding="utf-8",
        )

        mock_indexer = MagicMock()
        mock_store = MagicMock()

        with patch("core.memory.rag.MemoryIndexer", return_value=mock_indexer):
            with patch("core.memory.rag.singleton.get_vector_store", return_value=mock_store):
                engine._update_rag_index(
                    ["output2.md"],
                    origin="consolidation",
                    source_files=["clean_src.md"],
                )

        if mock_indexer.index_file.called:
            call_kwargs = mock_indexer.index_file.call_args
            assert call_kwargs[1].get("origin") == "consolidation"

    def test_consolidation_external_is_in_external_origins(self):
        from core.memory.consolidation import ConsolidationEngine

        assert "consolidation_external" in ConsolidationEngine._EXTERNAL_ORIGINS
        assert "external_web" in ConsolidationEngine._EXTERNAL_ORIGINS
        assert "mixed" in ConsolidationEngine._EXTERNAL_ORIGINS
