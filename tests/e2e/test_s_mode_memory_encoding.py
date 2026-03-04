# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for S-mode memory encoding gap fixes.

Validates three fixes that close the memory encoding gap for S-mode
(Claude Agent SDK) Animas:

Fix 1: S mode now saves conversation turns (``if mode != "s":`` guards removed)
Fix 2: compressed_summary is indexed into RAG and searchable via keyword
Fix 3: activity_log is collected as consolidation input
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.memory.conversation import ConversationMemory, ConversationTurn, ToolRecord
from core.memory.consolidation import ConsolidationEngine
from core.schemas import now_jst


# =====================================================================
# Fix 1: S mode append_turn
# =====================================================================


class TestSModeAppendTurn:
    """Verify that S-mode Animas persist conversation turns."""

    def test_s_mode_append_turn_direct(self, data_dir: Path, make_anima) -> None:
        """S-mode conversation memory saves human + assistant turns with tool_records."""
        anima_dir = make_anima(name="test-s", model="claude-sonnet-4-6")
        from core.schemas import ModelConfig

        model_config = ModelConfig(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            max_turns=5,
            credential="anthropic",
        )
        conv = ConversationMemory(anima_dir, model_config)

        # Append human turn
        conv.append_turn("human", "S-mode test message")
        # Append assistant turn with tool records
        conv.append_turn(
            "assistant",
            "Here is my response with tool usage.",
            tool_records=[
                ToolRecord(
                    tool_name="Read",
                    tool_id="t1",
                    input_summary="file.py",
                    result_summary="contents of file.py",
                ),
                ToolRecord(
                    tool_name="Write",
                    tool_id="t2",
                    input_summary="output.py",
                    result_summary="file written successfully",
                    is_error=False,
                ),
            ],
        )
        conv.save()

        # Reload from disk
        conv2 = ConversationMemory(anima_dir, model_config)
        state = conv2.load()

        assert len(state.turns) == 2

        # Verify human turn
        human_turn = state.turns[0]
        assert human_turn.role == "human"
        assert human_turn.content == "S-mode test message"

        # Verify assistant turn
        assistant_turn = state.turns[1]
        assert assistant_turn.role == "assistant"
        assert "tool usage" in assistant_turn.content
        assert len(assistant_turn.tool_records) == 2
        assert assistant_turn.tool_records[0].tool_name == "Read"
        assert assistant_turn.tool_records[0].tool_id == "t1"
        assert assistant_turn.tool_records[1].tool_name == "Write"

    def test_s_mode_no_more_guard(self) -> None:
        """Verify that the ``if mode != "s":`` guards have been removed from anima.py.

        This is a meta-test ensuring the guards that previously prevented
        S-mode from saving conversation turns are no longer present in
        the source code.
        """
        import core.anima as anima_module

        source = inspect.getsource(anima_module)

        # The old guard patterns that should no longer exist
        assert 'if mode != "s":' not in source, (
            "Found 'if mode != \"s\":' guard in core/anima.py -- "
            "this should have been removed to allow S-mode memory encoding"
        )
        assert "if mode != 's':" not in source, (
            "Found \"if mode != 's':\" guard in core/anima.py -- "
            "this should have been removed to allow S-mode memory encoding"
        )

    def test_s_mode_process_message_saves_turns(
        self, data_dir: Path, make_anima,
    ) -> None:
        """Verify process_message code path saves turns for S-mode.

        Inspects the process_message method source to confirm that
        append_turn calls are not guarded by mode checks.
        """
        import core.anima as anima_module

        source = inspect.getsource(anima_module.DigitalAnima.process_message)

        # Must contain append_turn calls (both pre-save and post-save)
        assert "append_turn" in source, (
            "process_message should call append_turn"
        )
        assert "conv_memory.save()" in source, (
            "process_message should call conv_memory.save()"
        )

    def test_s_mode_streaming_saves_turns(self) -> None:
        """Verify process_message_stream code path saves turns for S-mode."""
        import core.anima as anima_module

        source = inspect.getsource(anima_module.DigitalAnima.process_message_stream)

        # Must contain append_turn calls
        assert "append_turn" in source, (
            "process_message_stream should call append_turn"
        )


# =====================================================================
# Fix 2: compressed_summary RAG indexing
# =====================================================================


class TestConversationSummaryChunking:
    """Verify compressed_summary chunking in the MemoryIndexer."""

    def test_chunk_markdown_text_with_sections(self) -> None:
        """Chunking splits compressed_summary by ``### `` headings."""
        chromadb = pytest.importorskip("chromadb")
        pytest.importorskip("sentence_transformers")

        from core.memory.rag.indexer import MemoryIndexer
        from core.memory.rag.store import ChromaVectorStore

        import tempfile

        tmpdir = Path(tempfile.mkdtemp())
        anima_dir = tmpdir / "test-chunk"
        anima_dir.mkdir(parents=True)

        vectordb_dir = tmpdir / "vectordb"
        vectordb_dir.mkdir()
        store = ChromaVectorStore(persist_dir=vectordb_dir)

        indexer = MemoryIndexer(store, "test-chunk", anima_dir)

        summary_text = (
            "Preamble text that describes overall context of the conversation.\n\n"
            "### Session 1: Initial setup\n\n"
            "User asked about configuration. Anima explained config.json structure.\n\n"
            "### Session 2: RAG integration\n\n"
            "Discussed vector search and ChromaDB indexing pipeline.\n\n"
            "### Session 3: Memory consolidation\n\n"
            "Reviewed daily consolidation process and activity log collection."
        )

        source_id = "test-chunk/conversation_summary"
        chunks = indexer._chunk_markdown_text(summary_text, source_id)

        # Should produce 4 chunks: preamble + 3 sections
        assert len(chunks) == 4

        # All chunks must have source: "conversation_gist"
        for chunk in chunks:
            assert chunk.metadata["source"] == "conversation_gist"
            assert chunk.metadata["memory_type"] == "conversation_summary"
            assert chunk.metadata["source_file"] == "state/conversation.json"

        # Verify content
        assert "Preamble" in chunks[0].content
        assert "Session 1" in chunks[1].content
        assert "Session 2" in chunks[2].content
        assert "Session 3" in chunks[3].content

        # Clean up
        import shutil
        shutil.rmtree(tmpdir)

    def test_conversation_summary_empty_skip(self) -> None:
        """index_conversation_summary returns 0 for empty compressed_summary."""
        chromadb = pytest.importorskip("chromadb")
        pytest.importorskip("sentence_transformers")

        from core.memory.rag.indexer import MemoryIndexer
        from core.memory.rag.store import ChromaVectorStore

        import tempfile

        tmpdir = Path(tempfile.mkdtemp())
        anima_dir = tmpdir / "test-empty"
        anima_dir.mkdir(parents=True)

        state_dir = anima_dir / "state"
        state_dir.mkdir(parents=True)

        # Write conversation.json with empty compressed_summary
        conv_data = {
            "anima_name": "test-empty",
            "turns": [],
            "compressed_summary": "",
            "compressed_turn_count": 0,
        }
        (state_dir / "conversation.json").write_text(
            json.dumps(conv_data, ensure_ascii=False), encoding="utf-8",
        )

        vectordb_dir = tmpdir / "vectordb"
        vectordb_dir.mkdir()
        store = ChromaVectorStore(persist_dir=vectordb_dir)
        indexer = MemoryIndexer(store, "test-empty", anima_dir)

        result = indexer.index_conversation_summary(state_dir, "test-empty")
        assert result == 0

        # Clean up
        import shutil
        shutil.rmtree(tmpdir)

    def test_conversation_summary_short_skip(self) -> None:
        """index_conversation_summary skips summaries shorter than 50 chars."""
        chromadb = pytest.importorskip("chromadb")
        pytest.importorskip("sentence_transformers")

        from core.memory.rag.indexer import MemoryIndexer
        from core.memory.rag.store import ChromaVectorStore

        import tempfile

        tmpdir = Path(tempfile.mkdtemp())
        anima_dir = tmpdir / "test-short"
        anima_dir.mkdir(parents=True)

        state_dir = anima_dir / "state"
        state_dir.mkdir(parents=True)

        conv_data = {
            "anima_name": "test-short",
            "turns": [],
            "compressed_summary": "Too short",
            "compressed_turn_count": 0,
        }
        (state_dir / "conversation.json").write_text(
            json.dumps(conv_data, ensure_ascii=False), encoding="utf-8",
        )

        vectordb_dir = tmpdir / "vectordb"
        vectordb_dir.mkdir()
        store = ChromaVectorStore(persist_dir=vectordb_dir)
        indexer = MemoryIndexer(store, "test-short", anima_dir)

        result = indexer.index_conversation_summary(state_dir, "test-short")
        assert result == 0

        # Clean up
        import shutil
        shutil.rmtree(tmpdir)

    def test_chunk_markdown_text_no_headings_fallback(self) -> None:
        """Chunking falls back to single chunk when no ### headings exist."""
        chromadb = pytest.importorskip("chromadb")
        pytest.importorskip("sentence_transformers")

        from core.memory.rag.indexer import MemoryIndexer
        from core.memory.rag.store import ChromaVectorStore

        import tempfile

        tmpdir = Path(tempfile.mkdtemp())
        anima_dir = tmpdir / "test-fallback"
        anima_dir.mkdir(parents=True)

        vectordb_dir = tmpdir / "vectordb"
        vectordb_dir.mkdir()
        store = ChromaVectorStore(persist_dir=vectordb_dir)
        indexer = MemoryIndexer(store, "test-fallback", anima_dir)

        # Plain text without ### headings
        text = (
            "This is a conversation summary without any markdown headings. "
            "It describes various topics discussed over multiple sessions "
            "including deployment procedures and API integration."
        )
        chunks = indexer._chunk_markdown_text(text, "test/summary")

        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "conversation_gist"

        # Clean up
        import shutil
        shutil.rmtree(tmpdir)


class TestConversationSummaryKeywordSearch:
    """Verify keyword search covers compressed_summary."""

    def test_keyword_search_finds_conversation_summary(
        self, data_dir: Path, make_anima,
    ) -> None:
        """search_memory_text returns matches from compressed_summary."""
        anima_dir = make_anima(name="test-kw")

        # Write conversation.json with searchable compressed_summary
        state_dir = anima_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        conv_data = {
            "anima_name": "test-kw",
            "turns": [],
            "compressed_summary": (
                "### Session A\n\n"
                "Discussed deployment strategy.\n\n"
                "### Session B\n\n"
                "Reviewed memory consolidation pipeline."
            ),
            "compressed_turn_count": 10,
        }
        (state_dir / "conversation.json").write_text(
            json.dumps(conv_data, ensure_ascii=False), encoding="utf-8",
        )

        from core.memory.rag_search import RAGMemorySearch

        rag_search = RAGMemorySearch(
            anima_dir,
            common_knowledge_dir=data_dir / "common_knowledge",
            common_skills_dir=data_dir / "common_skills",
        )

        results = rag_search.search_memory_text(
            "consolidation",
            scope="all",
            knowledge_dir=anima_dir / "knowledge",
            episodes_dir=anima_dir / "episodes",
            procedures_dir=anima_dir / "procedures",
            common_knowledge_dir=data_dir / "common_knowledge",
        )

        # Should find a match from conversation_summary
        summary_results = [r for r in results if r[0] == "conversation_summary"]
        assert len(summary_results) > 0
        assert any("consolidation" in r[1].lower() for r in summary_results)

    def test_keyword_search_conversation_summary_scope(
        self, data_dir: Path, make_anima,
    ) -> None:
        """search_memory_text with scope='conversation_summary' searches only summary."""
        anima_dir = make_anima(name="test-scope")

        state_dir = anima_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        conv_data = {
            "anima_name": "test-scope",
            "turns": [],
            "compressed_summary": (
                "### Important Note\n\n"
                "User specified that replies to Hasegawa are not needed."
            ),
            "compressed_turn_count": 5,
        }
        (state_dir / "conversation.json").write_text(
            json.dumps(conv_data, ensure_ascii=False), encoding="utf-8",
        )

        from core.memory.rag_search import RAGMemorySearch

        rag_search = RAGMemorySearch(
            anima_dir,
            common_knowledge_dir=data_dir / "common_knowledge",
            common_skills_dir=data_dir / "common_skills",
        )

        results = rag_search.search_memory_text(
            "Hasegawa",
            scope="conversation_summary",
            knowledge_dir=anima_dir / "knowledge",
            episodes_dir=anima_dir / "episodes",
            procedures_dir=anima_dir / "procedures",
            common_knowledge_dir=data_dir / "common_knowledge",
        )

        assert len(results) > 0
        assert any("Hasegawa" in r[1] for r in results)

    def test_keyword_search_no_conversation_file(
        self, data_dir: Path, make_anima,
    ) -> None:
        """search_memory_text gracefully handles missing conversation.json."""
        anima_dir = make_anima(name="test-noconv")

        # Ensure no conversation.json exists
        conv_path = anima_dir / "state" / "conversation.json"
        if conv_path.exists():
            conv_path.unlink()

        from core.memory.rag_search import RAGMemorySearch

        rag_search = RAGMemorySearch(
            anima_dir,
            common_knowledge_dir=data_dir / "common_knowledge",
            common_skills_dir=data_dir / "common_skills",
        )

        # Should not raise
        results = rag_search.search_memory_text(
            "anything",
            scope="all",
            knowledge_dir=anima_dir / "knowledge",
            episodes_dir=anima_dir / "episodes",
            procedures_dir=anima_dir / "procedures",
            common_knowledge_dir=data_dir / "common_knowledge",
        )

        # No crash, possibly empty results
        assert isinstance(results, list)


# =====================================================================
# Fix 3: activity_log as consolidation input
# =====================================================================


class TestActivityLogConsolidation:
    """Verify activity_log collection for consolidation."""

    def test_collect_activity_entries(
        self, data_dir: Path, make_anima,
    ) -> None:
        """_collect_activity_entries returns formatted activity entries."""
        anima_dir = make_anima(name="test-act")

        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)

        now = now_jst()
        today_str = now.strftime("%Y-%m-%d")
        log_file = log_dir / f"{today_str}.jsonl"

        entries = [
            {
                "ts": (now - timedelta(minutes=30)).isoformat(),
                "type": "message_received",
                "content": "What is the deployment status?",
                "summary": "Deployment status inquiry",
                "from": "human",
            },
            {
                "ts": (now - timedelta(minutes=29)).isoformat(),
                "type": "response_sent",
                "content": "Deployment is running on staging.",
                "summary": "Staging deployment status",
                "to": "human",
            },
            {
                "ts": (now - timedelta(minutes=20)).isoformat(),
                "type": "tool_use",
                "summary": "Executed web_search",
                "tool": "web_search",
            },
        ]

        with log_file.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        engine = ConsolidationEngine(anima_dir, "test-act")
        result = engine._collect_activity_entries(hours=24)

        # Should contain text from our entries
        assert result  # Non-empty
        assert "message_received" in result
        assert "response_sent" in result
        assert "tool_use" in result

    def test_collect_activity_entries_type_filter(
        self, data_dir: Path, make_anima,
    ) -> None:
        """Only response_sent, tool_use, message_received types are collected."""
        anima_dir = make_anima(name="test-filter")

        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)

        now = now_jst()
        today_str = now.strftime("%Y-%m-%d")
        log_file = log_dir / f"{today_str}.jsonl"

        entries = [
            {
                "ts": (now - timedelta(minutes=10)).isoformat(),
                "type": "response_sent",
                "summary": "Answered question",
            },
            {
                "ts": (now - timedelta(minutes=9)).isoformat(),
                "type": "tool_use",
                "summary": "Used web_search",
                "tool": "web_search",
            },
            {
                "ts": (now - timedelta(minutes=8)).isoformat(),
                "type": "message_received",
                "summary": "User asked a question",
                "from": "human",
            },
            {
                "ts": (now - timedelta(minutes=7)).isoformat(),
                "type": "heartbeat_start",
                "summary": "Heartbeat started",
            },
            {
                "ts": (now - timedelta(minutes=6)).isoformat(),
                "type": "heartbeat_end",
                "summary": "Heartbeat ended",
            },
            {
                "ts": (now - timedelta(minutes=5)).isoformat(),
                "type": "channel_post",
                "summary": "Posted to general",
                "channel": "general",
            },
        ]

        with log_file.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        engine = ConsolidationEngine(anima_dir, "test-filter")
        result = engine._collect_activity_entries(hours=24)

        # Should include target types
        assert "response_sent" in result
        assert "tool_use" in result
        assert "message_received" in result

        # Should NOT include filtered types
        assert "heartbeat_start" not in result
        assert "heartbeat_end" not in result
        assert "channel_post" not in result

    def test_collect_activity_entries_budget_limit(
        self, data_dir: Path, make_anima,
    ) -> None:
        """Activity entries are truncated to the 12000-char budget."""
        anima_dir = make_anima(name="test-budget")

        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)

        now = now_jst()
        today_str = now.strftime("%Y-%m-%d")
        log_file = log_dir / f"{today_str}.jsonl"

        # Write many entries that would exceed the budget
        entries = []
        for i in range(500):
            entries.append({
                "ts": (now - timedelta(seconds=500 - i)).isoformat(),
                "type": "response_sent",
                "summary": f"Response #{i}: " + ("A" * 200),
                "to": "human",
            })

        with log_file.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        engine = ConsolidationEngine(anima_dir, "test-budget")
        result = engine._collect_activity_entries(hours=24)

        # Result should be non-empty but within budget
        assert result
        assert len(result) <= 12_000, (
            f"Activity log result should be within 12000 chars, got {len(result)}"
        )

    def test_collect_activity_entries_empty(
        self, data_dir: Path, make_anima,
    ) -> None:
        """Returns empty string when no activity log exists."""
        anima_dir = make_anima(name="test-empty-act")

        # No activity_log directory
        engine = ConsolidationEngine(anima_dir, "test-empty-act")
        result = engine._collect_activity_entries(hours=24)

        assert result == ""

    def test_collect_activity_entries_used_in_consolidation(self) -> None:
        """Verify _collect_episodes_summary calls _collect_activity_entries.

        This meta-test ensures the activity log is wired into the
        consolidation prompt pipeline.
        """
        import core.anima as anima_module

        source = inspect.getsource(
            anima_module.DigitalAnima._collect_episodes_summary,
        )

        assert "_collect_activity_entries" in source, (
            "_collect_episodes_summary should call _collect_activity_entries "
            "to include activity log in consolidation input"
        )

    def test_consolidation_prompt_includes_activity_log(self) -> None:
        """Verify run_consolidation passes activity_log_summary to prompt."""
        import core.anima as anima_module

        source = inspect.getsource(
            anima_module.DigitalAnima.run_consolidation,
        )

        assert "activity_log_summary" in source, (
            "run_consolidation should pass activity_log_summary to the "
            "consolidation prompt template"
        )
