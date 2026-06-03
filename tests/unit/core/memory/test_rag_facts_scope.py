from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.facts import FactRecord, append_fact_records


def _make_indexer(anima_dir: Path):
    from core.memory.rag.indexer import MemoryIndexer

    with patch.object(MemoryIndexer, "_init_embedding_model"):
        return MemoryIndexer(
            MagicMock(),
            anima_name=anima_dir.name,
            anima_dir=anima_dir,
        )


@pytest.mark.unit
def test_rag_resolves_facts_scope() -> None:
    from core.memory.rag_search import RAGMemorySearch

    assert RAGMemorySearch._resolve_search_types("facts") == ["facts"]
    assert "facts" in RAGMemorySearch._resolve_search_types("all")


@pytest.mark.unit
def test_indexer_chunks_facts_jsonl(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    facts_dir = anima_dir / "facts"
    facts_dir.mkdir(parents=True)
    fact = FactRecord(
        text="Alice prefers concise LoCoMo reports.",
        source_entity="Alice",
        target_entity="LoCoMo reports",
        edge_type="PREFERS",
        valid_at="2026-06-03T10:00:00+09:00",
        recorded_at="2026-06-03T10:01:00+09:00",
        source_episode="episodes/2026-06-03.md",
        source_session_id="session-1",
    )
    fact_file = facts_dir / "2026-06-03.jsonl"
    fact_file.write_text(f"{fact.to_json_line()}\n{{invalid}}\n", encoding="utf-8")

    chunks = _make_indexer(anima_dir)._chunk_file(
        fact_file,
        fact_file.read_text(encoding="utf-8"),
        "facts",
        origin="test",
    )

    assert len(chunks) == 1
    assert chunks[0].content == "Alice prefers concise LoCoMo reports."
    assert chunks[0].metadata["memory_type"] == "facts"
    assert chunks[0].metadata["fact_id"] == fact.fact_id
    assert chunks[0].metadata["valid_until"] == ""
    assert chunks[0].metadata["source_episode"] == "episodes/2026-06-03.md"
    assert chunks[0].metadata["origin"] == "test"


@pytest.mark.unit
def test_keyword_fallback_searches_active_facts_only(tmp_path: Path) -> None:
    from core.memory.rag_search import RAGMemorySearch

    anima_dir = tmp_path / "alice"
    for subdir in ("knowledge", "episodes", "procedures", "facts"):
        (anima_dir / subdir).mkdir(parents=True)
    common_knowledge_dir = tmp_path / "common_knowledge"
    common_skills_dir = tmp_path / "common_skills"
    common_knowledge_dir.mkdir()
    common_skills_dir.mkdir()

    append_fact_records(
        anima_dir,
        [
            FactRecord(
                text="Alice prefers LoCoMo benchmark reports.",
                source_entity="Alice",
                target_entity="LoCoMo",
                edge_type="PREFERS",
                recorded_at="2026-06-03T10:00:00+09:00",
            ),
            FactRecord(
                text="Alice used an expired LoCoMo rubric.",
                source_entity="Alice",
                target_entity="LoCoMo",
                edge_type="USED",
                recorded_at="2026-06-03T10:01:00+09:00",
                valid_until="2026-06-03T11:00:00+09:00",
            ),
        ],
    )

    rag = RAGMemorySearch(anima_dir, common_knowledge_dir, common_skills_dir)
    rag._indexer_initialized = True
    rag._indexer = None

    results = rag.search_memory_text(
        "LoCoMo reports",
        scope="facts",
        knowledge_dir=anima_dir / "knowledge",
        episodes_dir=anima_dir / "episodes",
        procedures_dir=anima_dir / "procedures",
        common_knowledge_dir=common_knowledge_dir,
    )

    assert len(results) == 1
    assert results[0]["memory_type"] == "facts"
    assert results[0]["fact_id"]
    assert "expired" not in results[0]["content"].lower()
