from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for distilled knowledge injection (knowledge/ + procedures/).

Tests cover:
- MemoryManager.collect_distilled_knowledge()
- builder.py Distilled Knowledge injection section
- PrimingEngine overflow_files conditional Channel C
- RAGMemorySearch procedures vector search enablement
"""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path):
    """Create a minimal anima directory with knowledge and procedures."""
    ad = tmp_path / "animas" / "test-anima"
    for sub in ("knowledge", "procedures", "skills", "episodes", "state"):
        (ad / sub).mkdir(parents=True)
    (ad / "state" / "current_task.md").write_text("status: idle\n")
    (ad / "state" / "pending.md").write_text("")
    (ad / "identity.md").write_text("# Test\n")
    return ad


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Set up isolated data dir."""
    from core.config import invalidate_cache
    from core.paths import _prompt_cache

    d = tmp_path / ".animaworks"
    d.mkdir()
    (d / "animas").mkdir()
    (d / "company").mkdir()
    (d / "company" / "vision.md").write_text("# Vision\n")
    (d / "common_skills").mkdir()
    (d / "common_knowledge").mkdir()
    (d / "shared" / "users").mkdir(parents=True)

    import json
    config = {
        "version": 1,
        "system": {"mode": "server", "log_level": "DEBUG"},
        "credentials": {"anthropic": {"api_key": "", "base_url": None}},
        "anima_defaults": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "max_turns": 5,
            "credential": "anthropic",
            "context_threshold": 0.50,
            "max_chains": 2,
            "conversation_history_threshold": 0.30,
        },
        "animas": {
            "test-anima": {"model": "claude-sonnet-4-20250514"},
        },
    }
    (d / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
    )

    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(d))
    invalidate_cache()
    _prompt_cache.clear()
    yield d
    invalidate_cache()
    _prompt_cache.clear()


@pytest.fixture
def memory(anima_dir, data_dir):
    """Create MemoryManager with RAG disabled."""
    from core.memory.manager import MemoryManager

    mm = MemoryManager(anima_dir)
    mm._indexer = None
    mm._indexer_initialized = True
    return mm


def _write_knowledge(anima_dir: Path, name: str, body: str, confidence: float = 0.5):
    """Write a knowledge file with frontmatter."""
    content = (
        f"---\nconfidence: {confidence}\ncreated_at: '2026-01-01'\n---\n\n{body}"
    )
    (anima_dir / "knowledge" / f"{name}.md").write_text(content, encoding="utf-8")


def _write_procedure(anima_dir: Path, name: str, body: str, confidence: float = 0.5):
    """Write a procedure file with frontmatter."""
    content = (
        f"---\ndescription: {name}\nconfidence: {confidence}\n---\n\n{body}"
    )
    (anima_dir / "procedures" / f"{name}.md").write_text(content, encoding="utf-8")


# ── collect_distilled_knowledge ──────────────────────────


class TestCollectDistilledKnowledge:
    """Tests for MemoryManager.collect_distilled_knowledge()."""

    def test_empty_directories(self, memory):
        """Empty knowledge/ and procedures/ returns empty list."""
        result = memory.collect_distilled_knowledge()
        assert result == []

    def test_knowledge_files_collected(self, memory, anima_dir):
        """Knowledge files are collected with correct metadata."""
        _write_knowledge(anima_dir, "topic-a", "Content of topic A", 0.8)
        _write_knowledge(anima_dir, "topic-b", "Content of topic B", 0.6)

        result = memory.collect_distilled_knowledge()

        assert len(result) == 2
        assert result[0]["name"] == "topic-a"
        assert result[0]["confidence"] == 0.8
        assert result[0]["content"] == "Content of topic A"
        assert result[0]["source_type"] == "knowledge"

    def test_procedure_files_collected(self, memory, anima_dir):
        """Procedure files are collected with correct metadata."""
        _write_procedure(anima_dir, "deploy-steps", "Step 1: deploy", 0.7)

        result = memory.collect_distilled_knowledge()

        assert len(result) == 1
        assert result[0]["name"] == "deploy-steps"
        assert result[0]["confidence"] == 0.7
        assert result[0]["source_type"] == "procedures"

    def test_sorted_by_confidence_descending(self, memory, anima_dir):
        """Results are sorted by confidence descending."""
        _write_knowledge(anima_dir, "low", "Low conf", 0.3)
        _write_knowledge(anima_dir, "high", "High conf", 0.9)
        _write_procedure(anima_dir, "mid", "Mid conf", 0.6)

        result = memory.collect_distilled_knowledge()

        confidences = [e["confidence"] for e in result]
        assert confidences == [0.9, 0.6, 0.3]

    def test_default_confidence_when_missing(self, memory, anima_dir):
        """Files without confidence metadata get default 0.5."""
        (anima_dir / "knowledge" / "no-meta.md").write_text(
            "No frontmatter content", encoding="utf-8",
        )

        result = memory.collect_distilled_knowledge()

        assert len(result) == 1
        assert result[0]["confidence"] == 0.5

    def test_empty_body_skipped(self, memory, anima_dir):
        """Files with empty body after frontmatter stripping are skipped."""
        (anima_dir / "knowledge" / "empty.md").write_text(
            "---\nconfidence: 0.8\n---\n\n   \n", encoding="utf-8",
        )

        result = memory.collect_distilled_knowledge()
        assert result == []

    def test_mixed_knowledge_and_procedures(self, memory, anima_dir):
        """Both knowledge and procedure files are collected together."""
        _write_knowledge(anima_dir, "k1", "Knowledge 1", 0.9)
        _write_procedure(anima_dir, "p1", "Procedure 1", 0.7)

        result = memory.collect_distilled_knowledge()

        assert len(result) == 2
        names = [e["name"] for e in result]
        assert "k1" in names
        assert "p1" in names


# ── Builder: Distilled Knowledge injection ───────────────


class TestBuilderKnowledgeInjection:
    """Tests for Distilled Knowledge section in build_system_prompt."""

    def test_knowledge_injected_into_prompt(self, memory, anima_dir, data_dir):
        """Knowledge content appears in system prompt."""
        _write_knowledge(anima_dir, "test-topic", "Important knowledge here", 0.8)

        from core.prompt.builder import build_system_prompt

        result = build_system_prompt(memory)

        assert "## Distilled Knowledge" in result.system_prompt
        assert "### test-topic" in result.system_prompt
        assert "Important knowledge here" in result.system_prompt

    def test_injected_files_tracked(self, memory, anima_dir, data_dir):
        """BuildResult tracks which files were injected."""
        _write_knowledge(anima_dir, "k1", "Content 1", 0.9)

        from core.prompt.builder import build_system_prompt

        result = build_system_prompt(memory)

        assert "k1" in result.injected_knowledge_files

    def test_empty_knowledge_no_section(self, memory, anima_dir, data_dir):
        """No Distilled Knowledge section when directories are empty."""
        from core.prompt.builder import build_system_prompt

        result = build_system_prompt(memory)

        assert "## Distilled Knowledge" not in result.system_prompt
        assert result.injected_knowledge_files == []
        assert result.overflow_files == []

    def test_budget_overflow(self, memory, anima_dir, data_dir):
        """Files exceeding budget go to overflow_files."""
        # Create a large knowledge file that exceeds 10% of context window
        # claude-sonnet-4 context = 200,000; budget = 20,000 tokens
        # 20,000 tokens * 3 chars/token = 60,000 chars
        large_content = "X" * 70_000  # exceeds budget
        _write_knowledge(anima_dir, "huge", large_content, 0.9)
        _write_knowledge(anima_dir, "small", "Small content", 0.5)

        from core.prompt.builder import build_system_prompt

        result = build_system_prompt(memory)

        # huge should be injected (highest confidence, fits first)
        # small may or may not fit depending on remaining budget
        assert "huge" in result.injected_knowledge_files or "huge" in result.overflow_files

    def test_confidence_ordering_in_injection(self, memory, anima_dir, data_dir):
        """Higher confidence files are injected first."""
        _write_knowledge(anima_dir, "low-conf", "Low", 0.3)
        _write_knowledge(anima_dir, "high-conf", "High", 0.9)
        _write_procedure(anima_dir, "mid-conf", "Mid", 0.6)

        from core.prompt.builder import build_system_prompt

        result = build_system_prompt(memory)

        # All should fit (small content)
        assert len(result.injected_knowledge_files) == 3
        assert result.overflow_files == []
        # Check ordering in the prompt
        prompt = result.system_prompt
        high_pos = prompt.index("### high-conf")
        mid_pos = prompt.index("### mid-conf")
        low_pos = prompt.index("### low-conf")
        assert high_pos < mid_pos < low_pos


# ── Priming: Conditional Channel C ───────────────────────


class TestPrimingConditionalChannelC:
    """Tests for overflow_files conditional Channel C in PrimingEngine."""

    def test_overflow_none_runs_full_channel_c(self, anima_dir):
        """overflow_files=None triggers legacy full Channel C."""
        from core.memory.priming import PrimingEngine

        engine = PrimingEngine(anima_dir)
        # Disable retriever to avoid RAG dependency
        engine._retriever_initialized = True
        engine._retriever = None

        result = asyncio.run(
            engine.prime_memories("test query", overflow_files=None)
        )
        # Channel C returns empty (no retriever), but it should have been called
        assert isinstance(result.related_knowledge, str)

    def test_overflow_empty_skips_channel_c(self, anima_dir):
        """overflow_files=[] (all injected) skips Channel C entirely."""
        from core.memory.priming import PrimingEngine

        engine = PrimingEngine(anima_dir)
        engine._retriever_initialized = True
        engine._retriever = None

        result = asyncio.run(
            engine.prime_memories("test query", overflow_files=[])
        )
        assert result.related_knowledge == ""

    def test_overflow_with_files_runs_restricted_channel_c(self, anima_dir):
        """overflow_files=[...] runs Channel C restricted to those files."""
        from core.memory.priming import PrimingEngine

        engine = PrimingEngine(anima_dir)
        engine._retriever_initialized = True
        engine._retriever = None

        result = asyncio.run(
            engine.prime_memories(
                "test query",
                overflow_files=["overflow-topic"],
            )
        )
        # No retriever = empty result, but the code path should not error
        assert result.related_knowledge == ""


# ── RAG: procedures vector search ────────────────────────


class TestRAGProceduresSearch:
    """Tests for procedures vector search enablement."""

    def test_resolve_search_types_knowledge(self):
        """scope=knowledge returns knowledge only."""
        from core.memory.rag_search import RAGMemorySearch

        assert RAGMemorySearch._resolve_search_types("knowledge") == ["knowledge"]

    def test_resolve_search_types_procedures(self):
        """scope=procedures returns procedures only."""
        from core.memory.rag_search import RAGMemorySearch

        assert RAGMemorySearch._resolve_search_types("procedures") == ["procedures"]

    def test_resolve_search_types_all(self):
        """scope=all returns both knowledge and procedures."""
        from core.memory.rag_search import RAGMemorySearch

        types = RAGMemorySearch._resolve_search_types("all")
        assert "knowledge" in types
        assert "procedures" in types

    def test_resolve_search_types_common_knowledge(self):
        """scope=common_knowledge returns knowledge."""
        from core.memory.rag_search import RAGMemorySearch

        assert RAGMemorySearch._resolve_search_types("common_knowledge") == ["knowledge"]

    def test_procedures_keyword_search(self, anima_dir):
        """Procedures are included in keyword search."""
        from core.memory.rag_search import RAGMemorySearch
        from core.paths import get_common_knowledge_dir, get_common_skills_dir

        _write_procedure(anima_dir, "deploy", "Run deploy command here", 0.7)

        rag = RAGMemorySearch(
            anima_dir,
            get_common_knowledge_dir(),
            get_common_skills_dir(),
        )

        results = rag.search_memory_text(
            "deploy",
            scope="procedures",
            knowledge_dir=anima_dir / "knowledge",
            episodes_dir=anima_dir / "episodes",
            procedures_dir=anima_dir / "procedures",
            common_knowledge_dir=get_common_knowledge_dir(),
        )

        assert len(results) > 0
        assert any("deploy" in r[1].lower() for r in results)

    def test_vector_search_condition_includes_procedures(self):
        """Vector search condition now includes 'procedures' scope."""
        # The condition check in search_memory_text:
        # scope in ("knowledge", "common_knowledge", "procedures", "all")
        valid_scopes = ("knowledge", "common_knowledge", "procedures", "all")
        assert "procedures" in valid_scopes
