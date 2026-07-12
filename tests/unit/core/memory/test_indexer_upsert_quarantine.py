from __future__ import annotations

import json

from core.config.schemas import RAGConfig
from core.memory.rag.indexer import MemoryIndexer


class _Store:
    def __init__(self, *, transient: bool = False) -> None:
        self.transient = transient

    def is_transient_write_failure(self, _collection: str) -> bool:
        return self.transient


def _indexer(tmp_path, threshold: int = 3) -> MemoryIndexer:
    indexer = MemoryIndexer.__new__(MemoryIndexer)
    indexer.anima_dir = tmp_path
    indexer.anima_name = "anima"
    indexer.collection_prefix = "anima"
    indexer.upsert_quarantine_failure_threshold = threshold
    indexer.upsert_failure_state_path = tmp_path / "state" / "rag_upsert_failures.json"
    indexer.vector_store = _Store()
    return indexer


def test_quarantine_threshold_default() -> None:
    assert RAGConfig().upsert_quarantine_failure_threshold == 3


def test_source_is_non_destructively_skipped_after_consecutive_failures(tmp_path, caplog) -> None:
    indexer = _indexer(tmp_path)
    source = tmp_path / "episodes" / "recovered_20260712.md"
    source.parent.mkdir()
    source.write_text("recovered", encoding="utf-8")
    source_key = "episodes/recovered_20260712.md"

    indexer._record_upsert_failure("anima_episodes", source_key, source)
    indexer._record_upsert_failure("anima_episodes", source_key, source)
    assert source.exists()

    indexer._record_upsert_failure("anima_episodes", source_key, source)

    assert source.read_text(encoding="utf-8") == "recovered"
    assert indexer._is_upsert_quarantined(source_key) is True
    state = json.loads(indexer.upsert_failure_state_path.read_text(encoding="utf-8"))
    assert source_key not in state["failures"]
    assert state["quarantined"] == [
        {
            "source_file": source_key,
            "failure_count": 3,
            "quarantined_at": state["quarantined"][0]["quarantined_at"],
            "reason": "consecutive_upsert_failures",
        }
    ]
    assert "Quarantined RAG source" in caplog.text


def test_success_resets_consecutive_failure_counter(tmp_path) -> None:
    indexer = _indexer(tmp_path, threshold=2)
    source = tmp_path / "episodes" / "recovered.md"
    source.parent.mkdir()
    source.write_text("recovered", encoding="utf-8")
    source_key = "episodes/recovered.md"

    indexer._record_upsert_failure("anima_episodes", source_key, source)
    indexer._clear_upsert_failure(source_key)
    indexer._run_upsert_failures = {}
    indexer._record_upsert_failure("anima_episodes", source_key, source)

    assert source.exists()
    state = json.loads(indexer.upsert_failure_state_path.read_text(encoding="utf-8"))
    assert state["failures"][source_key]["consecutive_failures"] == 1


def test_transient_circuit_failure_is_not_counted(tmp_path) -> None:
    indexer = _indexer(tmp_path)
    indexer.vector_store = _Store(transient=True)
    source = tmp_path / "episodes" / "recovered.md"
    source.parent.mkdir()
    source.write_text("recovered", encoding="utf-8")

    indexer._record_upsert_failure("anima_episodes", "episodes/recovered.md", source)

    assert not indexer.upsert_failure_state_path.exists()
    assert source.exists()


def test_multiple_file_failures_are_treated_as_global_outage(tmp_path) -> None:
    indexer = _indexer(tmp_path, threshold=1)
    indexer._index_directory_active = True
    indexer._run_upsert_failures = {}
    first = tmp_path / "episodes" / "a.md"
    second = tmp_path / "episodes" / "b.md"
    first.parent.mkdir()
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    indexer._record_upsert_failure("anima_episodes", "episodes/a.md", first)
    assert indexer._is_upsert_quarantined("episodes/a.md") is True
    indexer._record_upsert_failure("anima_episodes", "episodes/b.md", second)

    state = json.loads(indexer.upsert_failure_state_path.read_text(encoding="utf-8"))
    assert state == {"failures": {}, "quarantined": []}
    assert first.exists() and second.exists()


def test_separate_direct_index_calls_do_not_share_outage_state(tmp_path) -> None:
    indexer = _indexer(tmp_path, threshold=3)
    first = tmp_path / "episodes" / "a.md"
    second = tmp_path / "episodes" / "b.md"
    first.parent.mkdir()
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    indexer._record_upsert_failure("anima_episodes", "episodes/a.md", first)
    indexer._record_upsert_failure("anima_episodes", "episodes/b.md", second)

    state = json.loads(indexer.upsert_failure_state_path.read_text(encoding="utf-8"))
    assert state["failures"]["episodes/a.md"]["consecutive_failures"] == 1
    assert state["failures"]["episodes/b.md"]["consecutive_failures"] == 1


def test_skip_list_prevents_retry_without_removing_source(tmp_path) -> None:
    indexer = _indexer(tmp_path, threshold=1)
    source = tmp_path / "episodes" / "bad.md"
    source.parent.mkdir()
    source.write_text("bad", encoding="utf-8")
    indexer._record_upsert_failure("anima_episodes", "episodes/bad.md", source)

    assert indexer.index_file(source, "episodes") == 0
    assert source.exists()
