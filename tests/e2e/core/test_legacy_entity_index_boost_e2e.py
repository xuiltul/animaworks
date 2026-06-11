from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.entity_index import load_entity_registry, match_query_entities
from core.memory.fact_extraction import extract_and_store_facts
from core.memory.fact_invalidation import ReconcileAction, ReconcileResult
from core.memory.ontology.default import ExtractedEntity, ExtractedFact
from core.memory.retrieval.entity import EntityBoostConfig, apply_entity_boost


class DeterministicExtractor:
    async def extract_entities(self, content: str):
        assert "Caroline" in content
        return [
            ExtractedEntity(name="Caroline", entity_type="Person"),
            ExtractedEntity(name="Becoming Nicole", entity_type="Object"),
        ]

    async def extract_facts(self, content: str, entities, *, reference_time: str | None = None):
        return [
            ExtractedFact(
                source_entity="Caroline",
                target_entity="Becoming Nicole",
                fact="Caroline recommended Becoming Nicole.",
                valid_at=reference_time,
                edge_type="RECOMMENDED",
            )
        ]


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_fact_ingest_updates_entity_registry_and_boosts_metadata_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = tmp_path / "alice"
    for subdir in ("facts", "state", "knowledge", "episodes", "procedures"):
        (anima_dir / subdir).mkdir(parents=True)

    def add_without_reconciliation(anima_dir, fact, **kwargs):
        return ReconcileResult(
            action=ReconcileAction.ADD,
            fact=fact,
            should_append=True,
            reason="test_no_candidates",
        )

    monkeypatch.setattr("core.memory.fact_extraction.reconcile_new_fact", add_without_reconciliation)
    monkeypatch.setattr("core.memory.fact_extraction._index_fact_records", lambda *args, **kwargs: None)

    stored = await extract_and_store_facts(
        anima_dir,
        "Caroline recommended the book Becoming Nicole.",
        source_episode="episodes/2026-06-03.md",
        source_session_id="session-entity",
        reference_time="2026-06-03T10:00:00+09:00",
        extractor=DeterministicExtractor(),
        enabled=True,
    )

    registry = load_entity_registry(anima_dir)
    assert len(stored) == 1
    assert (anima_dir / "facts" / "2026-06-03.jsonl").is_file()
    assert registry["entities"]["caroline"]["mention_count"] == 1
    assert registry["entities"]["becoming nicole"]["source_fact_ids"] == [stored[0].fact_id]
    assert match_query_entities(anima_dir, "What did Caroline recommend?") == {"caroline"}

    boosted = apply_entity_boost(
        "What did Caroline recommend?",
        [
            {"content": "generic answer", "score": 0.5, "entities": ["Unrelated"]},
            {"content": stored[0].text, "score": 0.4, "entities": stored[0].entities},
        ],
        EntityBoostConfig(enabled=True, category=None, query_entities=("caroline",), boost=0.2, max_boost=0.2),
    )

    assert boosted[0]["content"] == "Caroline recommended Becoming Nicole."
    assert boosted[0]["entity_boost"] == 0.2
