from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.backend.legacy import LegacyRAGBackend
from core.memory.fact_invalidation import FactCandidate, ReconcileAction, ReconcileConfig, reconcile_new_fact
from core.memory.facts import FactRecord, append_fact_records, fact_file_for_record, read_fact_records


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_legacy_fact_invalidation_excludes_expired_facts_from_retrieval(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    for subdir in ("knowledge", "episodes", "procedures", "facts"):
        (anima_dir / subdir).mkdir(parents=True)
    common_knowledge_dir = tmp_path / "common_knowledge"
    common_skills_dir = tmp_path / "common_skills"
    common_knowledge_dir.mkdir()
    common_skills_dir.mkdir()

    old_fact = FactRecord(
        text="Alice's LoCoMo score is 70.",
        source_entity="Alice",
        target_entity="LoCoMo score",
        edge_type="HAS_SCORE",
        valid_at="1999-12-31T23:00:00+00:00",
        recorded_at="2000-01-01T00:01:00+00:00",
    )
    append_fact_records(anima_dir, [old_fact])
    old_path = fact_file_for_record(anima_dir, old_fact)

    new_fact = FactRecord(
        text="Alice's LoCoMo score is 85.",
        source_entity="Alice",
        target_entity="LoCoMo score",
        edge_type="HAS_SCORE",
        valid_at="2000-01-01T00:00:00+00:00",
        recorded_at="2000-01-01T00:05:00+00:00",
    )
    result = reconcile_new_fact(
        anima_dir,
        new_fact,
        classifier=lambda *_args: "CONTRADICT",
        candidate_search=lambda *_args: [FactCandidate(record=old_fact, score=0.96, path=old_path)],
        config=ReconcileConfig(enabled=True, threshold=0.82, top_k=5),
    )
    if result.should_append:
        append_fact_records(anima_dir, [new_fact])

    stored = read_fact_records(old_path, include_expired=True)
    assert result.action == ReconcileAction.INVALIDATE_OLD
    assert stored[0].valid_until == "2000-01-01T00:00:00+00:00"
    assert [record.text for record in read_fact_records(old_path)] == [new_fact.text]

    backend = LegacyRAGBackend(
        anima_dir,
        common_knowledge_dir=common_knowledge_dir,
        common_skills_dir=common_skills_dir,
    )
    rag = backend._ensure_rag_search()
    rag._indexer_initialized = True
    rag._indexer = None

    scoped = await backend.retrieve("LoCoMo score", scope="facts", limit=5)
    all_scope = await backend.retrieve("LoCoMo score", scope="all", limit=5)

    assert any("85" in memory.content for memory in scoped)
    assert all("70" not in memory.content for memory in scoped)
    assert any(memory.metadata["memory_type"] == "facts" and "85" in memory.content for memory in all_scope)
    assert all("70" not in memory.content for memory in all_scope)
