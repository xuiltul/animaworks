from __future__ import annotations

import json

from core.config.schemas import RAGConfig
from core.memory.rag.indexer import MemoryIndexer


def _indexer(tmp_path, threshold: int = 3) -> MemoryIndexer:
    indexer = MemoryIndexer.__new__(MemoryIndexer)
    indexer.anima_dir = tmp_path
    indexer.upsert_quarantine_failure_threshold = threshold
    indexer.upsert_failure_state_path = tmp_path / "state" / "rag_upsert_failures.json"
    return indexer


def test_quarantine_threshold_default() -> None:
    assert RAGConfig().upsert_quarantine_failure_threshold == 3


def test_source_is_quarantined_after_consecutive_failures(tmp_path, caplog) -> None:
    indexer = _indexer(tmp_path)
    source = tmp_path / "episodes" / "recovered_20260712.md"
    source.parent.mkdir()
    source.write_text("recovered", encoding="utf-8")
    source_key = "episodes/recovered_20260712.md"

    indexer._record_upsert_failure(source_key, source)
    indexer._record_upsert_failure(source_key, source)
    assert source.exists()

    indexer._record_upsert_failure(source_key, source)

    quarantined = tmp_path / "quarantine" / "rag_upsert" / source_key
    assert quarantined.read_text(encoding="utf-8") == "recovered"
    assert not source.exists()
    state = json.loads(indexer.upsert_failure_state_path.read_text(encoding="utf-8"))
    assert source_key not in state["failures"]
    assert state["quarantined"] == [
        {
            "source_file": source_key,
            "quarantine_path": f"quarantine/rag_upsert/{source_key}",
            "failure_count": 3,
            "quarantined_at": state["quarantined"][0]["quarantined_at"],
        }
    ]
    assert "Quarantined RAG source" in caplog.text


def test_success_resets_consecutive_failure_counter(tmp_path) -> None:
    indexer = _indexer(tmp_path, threshold=2)
    source = tmp_path / "episodes" / "recovered.md"
    source.parent.mkdir()
    source.write_text("recovered", encoding="utf-8")
    source_key = "episodes/recovered.md"

    indexer._record_upsert_failure(source_key, source)
    indexer._clear_upsert_failure(source_key)
    indexer._record_upsert_failure(source_key, source)

    assert source.exists()
    state = json.loads(indexer.upsert_failure_state_path.read_text(encoding="utf-8"))
    assert state["failures"][source_key]["consecutive_failures"] == 1
