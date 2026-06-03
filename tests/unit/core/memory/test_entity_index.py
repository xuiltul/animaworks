from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.memory import entity_index
from core.memory.entity_index import (
    load_entity_registry,
    match_query_entities,
    sync_entity_collection,
    upsert_entities_from_facts,
)
from core.memory.facts import FactRecord, append_fact_records


def _fact(
    text: str,
    *,
    fact_id: str,
    entities: list[str],
) -> FactRecord:
    return FactRecord(
        fact_id=fact_id,
        text=text,
        source_entity=entities[0] if entities else "",
        target_entity=entities[-1] if entities else "",
        edge_type="RELATES_TO",
        recorded_at="2026-06-03T10:00:00+09:00",
        entities=entities,
    )


@pytest.mark.unit
def test_upsert_entities_is_idempotent_and_counts_unique_facts(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    records = [
        _fact("Caroline recommended Becoming Nicole.", fact_id="fact-1", entities=["Caroline", "Becoming Nicole"]),
        _fact("CAROLINE discussed the book.", fact_id="fact-2", entities=["CAROLINE"]),
    ]

    upsert_entities_from_facts(anima_dir, records)
    registry = upsert_entities_from_facts(anima_dir, records)

    caroline = registry["entities"]["caroline"]
    assert caroline["canonical"] == "Caroline"
    assert caroline["mention_count"] == 2
    assert caroline["source_fact_ids"] == ["fact-1", "fact-2"]
    assert caroline["aliases"] == ["caroline"]
    assert registry["entities"]["becoming nicole"]["mention_count"] == 1


@pytest.mark.unit
def test_concurrent_upserts_preserve_registry_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    anima_dir = tmp_path / "alice"
    original_save = entity_index._save_entity_registry_unlocked

    def slow_save(path: Path, registry: dict[str, object]) -> None:
        time.sleep(0.02)
        original_save(path, registry)

    monkeypatch.setattr(entity_index, "_save_entity_registry_unlocked", slow_save)
    records = [
        [_fact("Caroline recommended Becoming Nicole.", fact_id="fact-1", entities=["Caroline"])],
        [_fact("Melanie read Becoming Nicole.", fact_id="fact-2", entities=["Melanie"])],
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(upsert_entities_from_facts, anima_dir, batch) for batch in records]
        for future in futures:
            future.result(timeout=2)

    registry = load_entity_registry(anima_dir)

    assert {"caroline", "melanie"} <= set(registry["entities"])
    assert registry["entities"]["caroline"]["source_fact_ids"] == ["fact-1"]
    assert registry["entities"]["melanie"]["source_fact_ids"] == ["fact-2"]


@pytest.mark.unit
def test_alias_collision_keeps_existing_canonical(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    upsert_entities_from_facts(anima_dir, [_fact("Alice is called Ally.", fact_id="fact-a", entities=["Alice"])])
    path = anima_dir / "state" / "entity_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry["entities"]["alice"]["aliases"].append("ally")
    path.write_text(json.dumps(registry), encoding="utf-8")

    updated = upsert_entities_from_facts(anima_dir, [_fact("Ally joined.", fact_id="fact-b", entities=["Ally"])])

    assert updated["entities"]["alice"]["canonical"] == "Alice"
    assert updated["entities"]["alice"]["source_fact_ids"] == ["fact-a", "fact-b"]
    assert "ally" not in updated["entities"]


@pytest.mark.unit
def test_corrupt_registry_is_archived_and_rebuilt_from_facts(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    append_fact_records(
        anima_dir,
        [_fact("Melanie read Becoming Nicole.", fact_id="fact-1", entities=["Melanie", "Becoming Nicole"])],
    )
    registry_path = anima_dir / "state" / "entity_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text("{bad json", encoding="utf-8")

    registry = load_entity_registry(anima_dir)

    assert "melanie" in registry["entities"]
    assert registry["entities"]["melanie"]["mention_count"] == 1
    assert list(registry_path.parent.glob("entity_registry.json.corrupt.*"))


@pytest.mark.unit
def test_match_query_entities_uses_registry_aliases(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    upsert_entities_from_facts(
        anima_dir,
        [_fact("Caroline suggested a book.", fact_id="fact-1", entities=["Caroline"])],
    )

    assert match_query_entities(anima_dir, "What did Caroline suggest?") == {"caroline"}


@pytest.mark.unit
def test_match_query_entities_uses_entity_collection_for_fuzzy_match(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    upsert_entities_from_facts(
        anima_dir,
        [_fact("Caroline suggested a book.", fact_id="fact-1", entities=["Caroline"])],
    )
    calls: dict[str, object] = {}

    class FakeStore:
        def query(self, collection: str, embedding: list[float], top_k: int = 10):
            calls["query"] = (collection, embedding, top_k)
            return [
                SimpleNamespace(
                    document=SimpleNamespace(metadata={"entity_key": "caroline"}),
                    score=0.72,
                )
            ]

    matches = match_query_entities(
        anima_dir,
        "Who suggested that memoir?",
        vector_store=FakeStore(),
        embedding_fn=lambda texts: [[0.1, 0.2] for _ in texts],
    )

    assert matches == {"caroline"}
    assert calls["query"] == ("alice_entities", [0.1, 0.2], 5)


@pytest.mark.unit
def test_sync_entity_collection_is_best_effort_with_injected_store(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    registry = upsert_entities_from_facts(
        anima_dir,
        [_fact("Caroline suggested a book.", fact_id="fact-1", entities=["Caroline"])],
    )
    calls: dict[str, object] = {}

    class FakeStore:
        def create_collection(self, name: str) -> bool:
            calls["collection"] = name
            return True

        def upsert(self, collection: str, documents: list[object]) -> bool:
            calls["upsert"] = (collection, documents)
            return True

    ok = sync_entity_collection(
        anima_dir,
        registry=registry,
        vector_store=FakeStore(),
        embedding_fn=lambda texts: [[0.1, 0.2] for _ in texts],
    )

    assert ok is True
    assert calls["collection"] == "alice_entities"
    collection, documents = calls["upsert"]
    assert collection == "alice_entities"
    assert documents[0].metadata["canonical"] == "Caroline"


@pytest.mark.unit
def test_sync_entity_collection_can_limit_to_changed_entity_keys(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    registry = upsert_entities_from_facts(
        anima_dir,
        [
            _fact("Caroline suggested a book.", fact_id="fact-1", entities=["Caroline"]),
            _fact("Melanie read a memoir.", fact_id="fact-2", entities=["Melanie"]),
        ],
    )
    docs: list[object] = []

    class FakeStore:
        def create_collection(self, name: str) -> bool:
            return True

        def upsert(self, collection: str, documents: list[object]) -> bool:
            docs.extend(documents)
            return True

    ok = sync_entity_collection(
        anima_dir,
        registry=registry,
        entity_keys={"melanie"},
        vector_store=FakeStore(),
        embedding_fn=lambda texts: [[0.1, 0.2] for _ in texts],
    )

    assert ok is True
    assert [doc.metadata["entity_key"] for doc in docs] == ["melanie"]
