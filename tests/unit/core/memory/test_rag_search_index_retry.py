from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.memory.rag.indexer import IndexDirectoryResult
from core.memory.rag_search import RAGMemorySearch


@pytest.fixture
def rag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RAGMemorySearch:
    monkeypatch.delenv("ANIMAWORKS_TASK_IPC_PATH", raising=False)
    anima = tmp_path / "animas" / "alice"
    (anima / "knowledge").mkdir(parents=True)
    (anima / "knowledge" / "memo.md").write_text("A memory awaiting indexing. " * 4)
    return RAGMemorySearch(anima, tmp_path / "common_knowledge", tmp_path / "common_skills")


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("embedding service temporarily unavailable"),
        IndexDirectoryResult(files_failed=1, transient_failures=1, files_unprocessed=2),
        IndexDirectoryResult(files_failed=1),
    ],
)
def test_failed_catchup_is_retried_after_cooldown(rag: RAGMemorySearch, failure: object) -> None:
    indexer = MagicMock()
    indexer.index_directory.side_effect = [failure, IndexDirectoryResult(files_indexed=1, chunks_indexed=1)]
    with (
        patch("core.memory.rag.singleton.get_vector_store", return_value=object()),
        patch("core.memory.rag.MemoryIndexer", return_value=indexer) as factory,
        patch.object(rag, "_check_shared_collections"),
        patch("core.memory.rag_search.time.monotonic", return_value=100.0) as clock,
    ):
        assert rag._get_indexer() is indexer
        assert rag._index_retry_at == 130.0
        clock.return_value = 129.0
        assert rag._get_indexer() is indexer  # Old indexed data stays searchable.
        assert indexer.index_directory.call_count == 1
        clock.return_value = 131.0
        assert rag._get_indexer() is indexer
        assert rag._index_retry_at is None
        assert indexer.index_directory.call_count == 2
        clock.return_value = 1000.0
        rag._get_indexer()
        assert factory.call_count == 2


def test_unavailable_store_can_recover_without_process_restart(rag: RAGMemorySearch) -> None:
    indexer = MagicMock()
    indexer.index_directory.return_value = IndexDirectoryResult()
    with (
        patch("core.memory.rag.singleton.get_vector_store", side_effect=[None, object()]) as store,
        patch("core.memory.rag.MemoryIndexer", return_value=indexer),
        patch.object(rag, "_check_shared_collections"),
        patch("core.memory.rag_search.time.monotonic", return_value=100.0) as clock,
    ):
        assert rag._get_indexer() is None
        assert rag._get_indexer() is None
        assert store.call_count == 1
        clock.return_value = 131.0
        assert rag._get_indexer() is indexer
        assert store.call_count == 2


@pytest.mark.parametrize("raises", [True, False])
def test_failed_single_file_write_schedules_catchup(rag: RAGMemorySearch, raises: bool) -> None:
    indexer = MagicMock()
    if raises:
        indexer.index_file.side_effect = RuntimeError("503")
    else:
        indexer._last_index_file_outcome = SimpleNamespace(status="failed")
    with (
        patch.object(rag, "_get_indexer", return_value=indexer),
        patch.object(rag, "_update_longterm_bm25_source") as bm25,
        patch("core.memory.rag_search.time.monotonic", return_value=100.0) as clock,
    ):
        path = rag._anima_dir / "knowledge" / "memo.md"
        rag.index_file(path, "knowledge")
        clock.return_value = 110.0
        rag.index_file(path, "knowledge")
    assert rag._index_retry_at == 130.0  # Repeated writes must not postpone recovery.
    assert bm25.call_count == 2


def test_concurrent_first_access_initializes_only_once(rag: RAGMemorySearch) -> None:
    indexer = MagicMock()
    indexer.index_directory.return_value = IndexDirectoryResult()
    with (
        patch("core.memory.rag.singleton.get_vector_store", return_value=object()),
        patch("core.memory.rag.MemoryIndexer", return_value=indexer) as factory,
        patch.object(rag, "_check_shared_collections"),
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        assert all(result is indexer for result in pool.map(lambda _: rag._get_indexer(), range(8)))
    factory.assert_called_once()
    indexer.index_directory.assert_called_once()


def test_task_runner_store_recovery_never_runs_automatic_indexing(rag: RAGMemorySearch) -> None:
    rag._auto_index_on_access = False
    indexer = MagicMock()
    with (
        patch("core.memory.rag.singleton.get_vector_store", side_effect=[None, object()]),
        patch("core.memory.rag.MemoryIndexer", return_value=indexer),
        patch.object(rag, "_check_shared_collections") as shared,
        patch("core.memory.rag_search.time.monotonic", return_value=100.0) as clock,
    ):
        assert rag._get_indexer() is None
        clock.return_value = 131.0
        assert rag._get_indexer() is indexer
    indexer.index_directory.assert_not_called()
    shared.assert_not_called()


def test_slow_failed_catchup_cools_down_from_completion(rag: RAGMemorySearch) -> None:
    indexer = MagicMock()
    with (
        patch("core.memory.rag.singleton.get_vector_store", return_value=object()),
        patch("core.memory.rag.MemoryIndexer", return_value=indexer),
        patch.object(rag, "_check_shared_collections"),
        patch("core.memory.rag_search.time.monotonic", return_value=100.0) as clock,
    ):

        def fail_slowly(*args):
            rag._schedule_index_retry()
            clock.return_value = 200.0
            raise RuntimeError("slow unavailable service")

        indexer.index_directory.side_effect = fail_slowly
        rag._get_indexer()
        assert rag._index_retry_at == 230.0
        rag._get_indexer()
        indexer.index_directory.assert_called_once()


def test_summary_fail_soft_result_schedules_retry(rag: RAGMemorySearch) -> None:
    (rag._anima_dir / "state").mkdir()
    (rag._anima_dir / "state" / "conversation.json").write_text("{}")
    indexer = MagicMock()
    indexer.index_directory.return_value = IndexDirectoryResult()
    indexer.index_conversation_summary.return_value = 0
    indexer._last_index_file_outcome = SimpleNamespace(status="failed")
    with (
        patch("core.memory.rag.singleton.get_vector_store", return_value=object()),
        patch("core.memory.rag.MemoryIndexer", return_value=indexer),
        patch.object(rag, "_check_shared_collections"),
        patch("core.memory.rag_search.time.monotonic", return_value=100.0),
    ):
        assert rag._get_indexer() is indexer
    assert rag._index_retry_at == 130.0
