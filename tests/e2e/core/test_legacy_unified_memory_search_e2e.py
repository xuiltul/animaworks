from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.backend.legacy import LegacyRAGBackend
from core.memory.facts import FactRecord
from core.memory.rag_search import RAGMemorySearch


def _write_fact_store(anima_dir: Path) -> None:
    facts_dir = anima_dir / "facts"
    facts_dir.mkdir(parents=True)
    active = FactRecord(
        text="Alice prefers espresso for morning focus.",
        source_entity="Alice",
        target_entity="espresso",
        edge_type="PREFERS",
        valid_at="2026-06-03T08:00:00+09:00",
        source_episode="episodes/2026-06-03.md",
        source_session_id="session_1",
        confidence=0.95,
    )
    expired = FactRecord(
        text="Alice prefers black tea for morning focus.",
        source_entity="Alice",
        target_entity="black tea",
        edge_type="PREFERS",
        valid_at="2026-01-01T08:00:00+09:00",
        valid_until="2026-02-01T00:00:00+09:00",
        source_episode="episodes/2026-01-01.md",
        source_session_id="session_old",
        confidence=0.9,
    )
    (facts_dir / "2026-06-03.jsonl").write_text(
        active.to_json_line() + "\n" + expired.to_json_line() + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_legacy_unified_search_returns_real_active_facts(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "test"
    common_knowledge = tmp_path / "common_knowledge"
    common_skills = tmp_path / "common_skills"
    for path in (
        anima_dir / "knowledge",
        anima_dir / "episodes",
        anima_dir / "procedures",
        common_knowledge,
        common_skills,
    ):
        path.mkdir(parents=True)
    _write_fact_store(anima_dir)

    rag = RAGMemorySearch(anima_dir, common_knowledge, common_skills)
    results = rag.search_memory_text(
        "Alice espresso",
        scope="facts",
        knowledge_dir=anima_dir / "knowledge",
        episodes_dir=anima_dir / "episodes",
        procedures_dir=anima_dir / "procedures",
        common_knowledge_dir=common_knowledge,
        result_limit=5,
    )

    assert any("espresso" in item["content"] for item in results)
    assert not any("black tea" in item["content"] for item in results)
    assert rag.last_search_meta["abstain"] is False

    backend = LegacyRAGBackend(
        anima_dir,
        common_knowledge_dir=common_knowledge,
        common_skills_dir=common_skills,
    )
    facts = await backend.get_recent_facts("Alice espresso", limit=5)

    assert facts
    assert facts[0].metadata["memory_type"] == "facts"
    assert "espresso" in facts[0].content
