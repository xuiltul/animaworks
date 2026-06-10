from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_chroma_vector_store_close_is_idempotent(tmp_path: Path) -> None:
    from core.memory.rag.store import ChromaVectorStore

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.client = MagicMock()
    store.persist_dir = tmp_path
    store._closed = False

    store.close()
    store.close()

    store.client.close.assert_called_once()


def test_chroma_vector_store_startup_quick_check_blocks_corrupt_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.memory.rag.sqlite_health import SQLiteHealthResult
    from core.memory.rag.store import ChromaVectorStore

    fake_chromadb = types.ModuleType("chromadb")
    fake_chromadb.PersistentClient = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)
    monkeypatch.setenv("ANIMAWORKS_ALLOW_DIRECT_CHROMA", "1")

    health = SQLiteHealthResult(
        db_path=tmp_path / "chroma.sqlite3",
        ok=False,
        status="corrupt",
        error="database disk image is malformed",
    )
    repair = MagicMock()
    monkeypatch.setattr("core.memory.rag.sqlite_health.quick_check_chroma_sqlite", lambda _persist_dir: health)
    monkeypatch.setattr("core.memory.rag.sqlite_health.request_repair_for_sqlite_health", repair)

    with pytest.raises(RuntimeError, match="Chroma SQLite database corrupt"):
        ChromaVectorStore(persist_dir=tmp_path, anima_name="sora")

    repair.assert_called_once()
    fake_chromadb.PersistentClient.assert_not_called()  # type: ignore[attr-defined]


def test_chroma_vector_store_configures_sqlite_pragmas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.memory.rag.sqlite_health import SQLiteHealthResult
    from core.memory.rag.store import ChromaVectorStore

    fake_chromadb = types.ModuleType("chromadb")
    fake_chromadb.PersistentClient = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)
    monkeypatch.setenv("ANIMAWORKS_ALLOW_DIRECT_CHROMA", "1")

    monkeypatch.setattr(
        "core.memory.rag.sqlite_health.quick_check_chroma_sqlite",
        lambda persist_dir: SQLiteHealthResult(db_path=persist_dir / "chroma.sqlite3", ok=True, status="ok"),
    )
    configure = MagicMock(return_value=SQLiteHealthResult(db_path=tmp_path / "chroma.sqlite3", ok=True, status="ok"))
    monkeypatch.setattr("core.memory.rag.sqlite_health.configure_chroma_sqlite_pragmas", configure)

    ChromaVectorStore(persist_dir=tmp_path, anima_name="sora")

    assert configure.call_count == 2
    fake_chromadb.PersistentClient.assert_called_once_with(path=str(tmp_path))  # type: ignore[attr-defined]


def test_reset_vector_store_closes_cached_chroma_store() -> None:
    from core.memory.rag import singleton

    singleton._reset_for_testing()
    store = MagicMock()
    singleton._vector_stores["sora"] = store

    singleton.reset_vector_store("sora")

    store.close.assert_called_once()
    assert "sora" not in singleton._vector_stores
