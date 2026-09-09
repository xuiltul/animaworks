from __future__ import annotations

import sqlite3
from contextlib import closing
from unittest.mock import MagicMock

import pytest

from core.memory.rag import sqlite_health


def _assert_closed(connections):
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


@pytest.mark.parametrize("operation", ["bootstrap", "configure", "quick_check"])
@pytest.mark.parametrize("fail", [False, True])
def test_health_connections_close_on_success_and_failure(tmp_path, monkeypatch, operation, fail):
    original_connect = sqlite3.connect
    database = sqlite_health.chroma_sqlite_path(tmp_path)
    if operation != "bootstrap":
        with closing(original_connect(database)) as connection:
            connection.execute("CREATE TABLE fixture (id INTEGER)")
    connections = []

    class FailingConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if fail and sql in {"PRAGMA synchronous=NORMAL", "PRAGMA quick_check"}:
                raise sqlite3.OperationalError("injected statement failure")
            return super().execute(sql, *args, **kwargs)

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs, factory=FailingConnection)
        # Keep a strong reference: success must not depend on destructor/GC.
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite_health.sqlite3, "connect", connect)
    try:
        if operation == "quick_check":
            if fail:
                with pytest.raises(sqlite3.OperationalError, match="injected statement failure"):
                    sqlite_health._run_quick_check(database, 1)
            else:
                assert sqlite_health._run_quick_check(database, 1) == ("ok",)
        else:
            function = (
                sqlite_health.bootstrap_chroma_sqlite_wal
                if operation == "bootstrap"
                else sqlite_health.configure_chroma_sqlite_pragmas
            )
            result = function(tmp_path)
            assert result.ok is not fail
            if fail:
                assert result.status == "corrupt"
                assert result.error == "injected statement failure"
        _assert_closed(connections)
    finally:
        for connection in connections:
            connection.close()


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("native_failure", [False, True])
def test_preflight_connections_close_before_native_client_starts(tmp_path, monkeypatch, existing, native_failure):
    from core.memory.rag.store import ChromaVectorStore

    original_connect = sqlite3.connect
    database = sqlite_health.chroma_sqlite_path(tmp_path)
    if existing:
        with closing(original_connect(database)) as connection:
            connection.execute("CREATE TABLE fixture (id INTEGER)")
    connections = []
    events = []

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    def native_client(**kwargs):
        _assert_closed(connections)
        assert len(connections) == (2 if existing else 1)
        events.append("native_started_after_preflight_closed")
        if native_failure:
            raise RuntimeError("injected native startup failure")
        return MagicMock()

    monkeypatch.setenv("ANIMAWORKS_ALLOW_DIRECT_CHROMA", "1")
    monkeypatch.setattr(sqlite_health.sqlite3, "connect", connect)
    monkeypatch.setattr("chromadb.PersistentClient", native_client)
    try:
        if native_failure:
            with pytest.raises(RuntimeError, match="injected native startup failure"):
                ChromaVectorStore(persist_dir=tmp_path, anima_name="fixture")
        else:
            store = ChromaVectorStore(persist_dir=tmp_path, anima_name="fixture")
            store.close()
        assert events == ["native_started_after_preflight_closed"]
        _assert_closed(connections)
    finally:
        for connection in connections:
            connection.close()
