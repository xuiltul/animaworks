from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import CHROMADB_AVAILABLE


@pytest.mark.e2e
def test_temporary_vector_worker_http_store_roundtrip(data_dir: Path) -> None:
    """E2E smoke: parent process uses HttpVectorStore while worker owns Chroma."""
    if not CHROMADB_AVAILABLE:
        pytest.skip("ChromaDB is not installed")

    from core.memory.rag.store import Document
    from core.memory.rag.singleton import get_vector_store
    from core.memory.rag.vector_worker_client import start_temporary_vector_worker

    worker = start_temporary_vector_worker(log_dir=data_dir / "logs")
    try:
        store = get_vector_store("worker_smoke")
        assert store is not None
        assert store.create_collection("worker_smoke_knowledge")
        assert store.upsert(
            "worker_smoke_knowledge",
            [
                Document(
                    id="doc-1",
                    content="worker only",
                    embedding=[0.1, 0.2, 0.3],
                    metadata={"kind": "smoke"},
                )
            ],
        )

        docs = store.get_by_ids("worker_smoke_knowledge", ["doc-1"])
    finally:
        worker.stop()

    assert [doc.id for doc in docs] == ["doc-1"]
    assert docs[0].content == "worker only"
