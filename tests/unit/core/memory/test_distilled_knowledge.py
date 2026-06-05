"""Unit tests for DK cleanup and procedures RAG search behavior."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def anima_dir(tmp_path):
    """Create a minimal anima directory with knowledge and procedures."""
    ad = tmp_path / "animas" / "test-anima"
    for sub in ("knowledge", "procedures", "skills", "episodes", "state"):
        (ad / sub).mkdir(parents=True)
    (ad / "state" / "current_state.md").write_text("status: idle\n")
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
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "max_turns": 5,
            "credential": "anthropic",
            "context_threshold": 0.50,
            "max_chains": 2,
            "conversation_history_threshold": 0.30,
        },
        "animas": {
            "test-anima": {"model": "claude-sonnet-4-6"},
        },
    }
    (d / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
    )

    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(d))
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)
    monkeypatch.delenv("ANIMAWORKS_EMBED_URL", raising=False)

    from core.memory.rag.singleton import _reset_for_testing

    _reset_for_testing()

    invalidate_cache()
    _prompt_cache.clear()
    yield d
    _reset_for_testing()
    invalidate_cache()
    _prompt_cache.clear()


def _write_procedure(anima_dir: Path, name: str, body: str, confidence: float = 0.5):
    """Write a procedure file with frontmatter."""
    content = f"---\ndescription: {name}\nconfidence: {confidence}\n---\n\n{body}"
    (anima_dir / "procedures" / f"{name}.md").write_text(content, encoding="utf-8")


class TestDistilledKnowledgeApisRemoved:
    def test_memory_manager_no_longer_exposes_dk_collectors(self, anima_dir, data_dir):
        from core.memory.manager import MemoryManager

        memory = MemoryManager(anima_dir)

        assert not hasattr(memory, "collect_distilled_knowledge")
        assert not hasattr(memory, "collect_distilled_knowledge_separated")


class TestPrimingAlwaysRunsChannelC:
    @pytest.mark.asyncio
    async def test_prime_memories_always_calls_channel_c(self, anima_dir):
        from core.memory.priming import PrimingEngine

        engine = PrimingEngine(anima_dir)
        with (
            patch.object(
                engine,
                "_channel_c_related_knowledge",
                new_callable=AsyncMock,
                return_value=("related", ""),
            ) as mock_channel_c,
            patch.object(engine, "_channel_a_sender_profile", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_b_recent_activity", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_c0_important_knowledge", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_e_pending_tasks", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_collect_recent_outbound", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_f_episodes", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_collect_pending_human_notifications", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_g_graph_context", new_callable=AsyncMock, return_value=""),
        ):
            await engine.prime_memories("test query")

        mock_channel_c.assert_called_once()
        assert "restrict_to" not in mock_channel_c.call_args.kwargs

    def test_prime_memories_signature_has_no_overflow_files(self) -> None:
        from core.memory.priming import PrimingEngine

        params = inspect.signature(PrimingEngine.prime_memories).parameters
        assert "overflow_files" not in params


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

    def test_procedures_keyword_search(self, anima_dir, data_dir):
        """Procedures are included in keyword search."""
        from core.memory.rag_search import RAGMemorySearch
        from core.paths import get_common_knowledge_dir, get_common_skills_dir

        _write_procedure(anima_dir, "deploy", "Run deploy command here", 0.7)

        rag = RAGMemorySearch(
            anima_dir,
            get_common_knowledge_dir(),
            get_common_skills_dir(),
        )
        rag._indexer_initialized = True
        rag._indexer = None

        results = rag.search_memory_text(
            "deploy",
            scope="procedures",
            knowledge_dir=anima_dir / "knowledge",
            episodes_dir=anima_dir / "episodes",
            procedures_dir=anima_dir / "procedures",
            common_knowledge_dir=get_common_knowledge_dir(),
        )

        assert len(results) > 0
        assert any("deploy" in r["content"].lower() for r in results)
