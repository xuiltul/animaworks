# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RAG auto-repair detection and rebuild orchestration."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from core.memory.rag.repair import (
    RAGRepairService,
    RepairResult,
    classify_corruption_error,
    collection_owner,
)
from core.memory.rag.sqlite_health import SQLiteHealthResult


class _RebuiltStore:
    """Fake vector store standing in for a healthy rebuilt DB.

    The post-rebuild verification calls ``list_collections``; returning a
    non-empty list lets repairs that indexed chunks report success.
    """

    def __init__(self, collections: list[str] | None = None) -> None:
        self._collections = collections if collections is not None else ["rebuilt"]

    def list_collections(self) -> list[str]:
        return list(self._collections)


def test_classifies_today_error_finding_id():
    err = "Error executing plan: Internal error: Error finding id"
    assert classify_corruption_error(err) == "chroma_error_finding_id"


def test_classifies_native_and_sqlite_corruption():
    assert classify_corruption_error("database disk image is malformed") == "sqlite_malformed"
    assert classify_corruption_error(-11) == "native_segfault"
    assert classify_corruption_error("segmentation fault") == "native_segfault"
    assert classify_corruption_error("hnsw index panic: corrupt graph") == "hnsw_corruption"
    assert classify_corruption_error("Failed to get segments for collection") == "chroma_corruption"
    assert classify_corruption_error("no such table: embeddings_queue") == "chroma_corruption"


def test_does_not_classify_operational_noise():
    assert classify_corruption_error("Connection refused") is None
    assert classify_corruption_error("Collection 'foo' not found") is None


def test_does_not_classify_resource_exhaustion_or_transient_io_errors():
    assert classify_corruption_error("hnsw segment reader: Too many open files (os error 24)") is None
    assert classify_corruption_error("unable to open database file") is None
    assert classify_corruption_error("Internal error: error returned from database: (code: 522) disk I/O error") is None


def test_collection_owner_uses_default_anima_for_shared_collection():
    assert collection_owner("shared_common_knowledge", default_anima="sora") == ("sora", True)
    assert collection_owner("mikoto_knowledge") == ("mikoto", False)


def test_record_chroma_error_triggers_after_threshold(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)

    service = RAGRepairService(enabled=True, threshold=2, window_minutes=5, cooldown_minutes=60)
    service.request_repair = MagicMock(return_value=True)  # type: ignore[method-assign]

    assert (
        service.record_chroma_error(
            anima_name="sora",
            collection="sora_knowledge",
            error="Error executing plan: Internal error: Error finding id",
            source="query",
        )
        is False
    )
    assert (
        service.record_chroma_error(
            anima_name="sora",
            collection="sora_knowledge",
            error="Error executing plan: Internal error: Error finding id",
            source="query",
        )
        is True
    )
    service.request_repair.assert_called_once()
    state = json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert len(state["recent_signals"]) == 2


def test_single_shot_corruption_triggers_immediate_repair(data_dir: Path):
    service = RAGRepairService(enabled=True, threshold=2, window_minutes=5, cooldown_minutes=60)
    service.request_repair = MagicMock(return_value=True)  # type: ignore[method-assign]

    assert (
        service.record_chroma_error(
            anima_name="sora",
            collection="sora_knowledge",
            error="database disk image is malformed",
            source="query",
        )
        is True
    )

    service.request_repair.assert_called_once()


def test_chroma_corruption_suppressed_when_sqlite_quick_check_ok(data_dir: Path):
    """A healthy on-disk SQLite refutes a chroma_corruption signal.

    chromadb's process-global cache can be transiently poisoned, making an
    intact DB raise "Failed to get segments". A passing quick_check means the
    store is fine, so we must not escalate to a (destructive) repair.
    """
    service = RAGRepairService(enabled=True, threshold=1, window_minutes=5, cooldown_minutes=60)
    service.request_repair = MagicMock(return_value=True)  # type: ignore[method-assign]
    service._sqlite_quick_check_ok = lambda owner: True  # type: ignore[method-assign,assignment]

    assert (
        service.record_chroma_error(
            anima_name="sora",
            collection="sora_knowledge",
            error="Error getting collection: Failed to get segments",
            source="upsert",
        )
        is False
    )
    service.request_repair.assert_not_called()


def test_chroma_corruption_still_repairs_when_sqlite_check_not_ok(data_dir: Path):
    """Ambiguous/failing quick_check does not suppress a real corruption signal."""
    service = RAGRepairService(enabled=True, threshold=1, window_minutes=5, cooldown_minutes=60)
    service.request_repair = MagicMock(return_value=True)  # type: ignore[method-assign]
    service._sqlite_quick_check_ok = lambda owner: False  # type: ignore[method-assign,assignment]

    assert (
        service.record_chroma_error(
            anima_name="sora",
            collection="sora_knowledge",
            error="Error getting collection: Failed to get segments",
            source="upsert",
        )
        is True
    )
    service.request_repair.assert_called_once()


def test_hnsw_corruption_not_gated_by_sqlite_check(data_dir: Path):
    """Segment-level reasons are not refutable by a SQLite check.

    A healthy quick_check must NOT suppress an hnsw_corruption signal, and the
    gate must not even consult the SQLite check for such reasons.
    """
    service = RAGRepairService(enabled=True, threshold=1, window_minutes=5, cooldown_minutes=60)
    service.request_repair = MagicMock(return_value=True)  # type: ignore[method-assign]
    service._sqlite_quick_check_ok = MagicMock(return_value=True)  # type: ignore[method-assign]

    assert (
        service.record_chroma_error(
            anima_name="sora",
            collection="sora_knowledge",
            error="hnsw index panic: corrupt graph",
            source="query",
        )
        is True
    )
    service.request_repair.assert_called_once()
    service._sqlite_quick_check_ok.assert_not_called()


def test_sqlite_quick_check_ok_true_for_intact_db(data_dir: Path):
    """The gate helper returns True for an intact Chroma SQLite file."""
    import sqlite3

    from core.paths import get_anima_vectordb_dir

    vdb = get_anima_vectordb_dir("sora")
    vdb.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(vdb / "chroma.sqlite3")
    try:
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
    finally:
        conn.close()

    assert RAGRepairService._sqlite_quick_check_ok("sora") is True


def test_sqlite_quick_check_ok_false_when_missing(data_dir: Path):
    """A missing DB is not 'ok' for the gate, so signals are not suppressed."""
    assert RAGRepairService._sqlite_quick_check_ok("sora") is False


def test_record_chroma_error_ignores_noise_and_unknown_owner(data_dir: Path):
    service = RAGRepairService(enabled=True, threshold=2, window_minutes=5, cooldown_minutes=60)

    assert (
        service.record_chroma_error(
            anima_name="sora",
            collection="sora_knowledge",
            error="Connection refused",
            source="query",
        )
        is False
    )
    assert (
        service.record_chroma_error(
            anima_name=None,
            collection="shared_common_knowledge",
            error="database disk image is malformed",
            source="query",
        )
        is False
    )


def test_has_recent_corruption_reads_state_file(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "recent_signals": [
                    {
                        "at": datetime.now(UTC).isoformat(),
                        "collection": "sora_knowledge",
                        "reason": "chroma_error_finding_id",
                        "source": "query",
                        "shared": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert RAGRepairService(enabled=True).has_recent_corruption("sora") is True


def test_has_recent_corruption_ignores_signal_before_success(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    now = datetime.now(UTC)
    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "last_success_at": now.isoformat(),
                "recent_signals": [
                    {
                        "at": (now - timedelta(minutes=1)).isoformat(),
                        "collection": "sora_knowledge",
                        "reason": "chroma_error_finding_id",
                        "source": "query",
                        "shared": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert RAGRepairService(enabled=True).has_recent_corruption("sora") is False


def test_discover_suspect_animas_from_state_and_native_log(data_dir: Path):
    for name in ("rin", "sora"):
        anima_dir = data_dir / "animas" / name
        (anima_dir / "state").mkdir(parents=True)
        (anima_dir / "identity.md").write_text(f"# {name}", encoding="utf-8")
    (data_dir / "animas" / "rin" / "vectordb").mkdir()
    disabled = data_dir / "animas" / "mika"
    disabled.mkdir(parents=True)
    (disabled / "identity.md").write_text("# mika", encoding="utf-8")
    (disabled / "status.json").write_text('{"enabled": false}', encoding="utf-8")

    (data_dir / "animas" / "sora" / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "recent_signals": [
                    {
                        "at": datetime.now(UTC).isoformat(),
                        "collection": "sora_knowledge",
                        "reason": "chroma_error_finding_id",
                        "source": "query",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    log_path = data_dir / "logs" / "server-daemon.log"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(
        f"{datetime.now(UTC).isoformat()} tokio-rt-worker segfault in chromadb_rust_bindings.abi3.so\n",
        encoding="utf-8",
    )

    suspects = RAGRepairService(enabled=True).discover_suspect_animas(
        window_minutes=60,
        log_paths=[log_path],
    )

    assert suspects == ["rin", "sora"]


def test_discover_suspect_animas_includes_quick_check_corruption(data_dir: Path, monkeypatch):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# sora", encoding="utf-8")
    (anima_dir / "vectordb").mkdir()

    calls: list[dict[str, object]] = []

    def fake_quick_check(anima_name: str, **kwargs) -> SQLiteHealthResult:
        calls.append(kwargs)
        return SQLiteHealthResult(
            db_path=data_dir / "animas" / anima_name / "vectordb" / "chroma.sqlite3",
            ok=False,
            status="corrupt",
            error="database disk image is malformed",
        )

    monkeypatch.setattr(
        "core.memory.rag.sqlite_health.check_anima_vectordb_health_via_worker_or_direct",
        fake_quick_check,
    )

    suspects = RAGRepairService(enabled=True).discover_suspect_animas(
        include_logs=False,
        quick_check_timeout_seconds=2.0,
        quick_check_source="startup_quick_check",
    )

    assert suspects == ["sora"]
    assert calls == [
        {
            "timeout_seconds": 2.0,
            "source": "startup_quick_check",
            "record_repair": False,
        }
    ]


def test_discover_suspect_animas_ignores_signals_before_success(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# sora", encoding="utf-8")
    now = datetime.now(UTC)

    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "status": "success",
                "reason": "startup_chroma_crash_preflight",
                "last_success_at": now.isoformat(),
                "recent_signals": [
                    {
                        "at": (now - timedelta(minutes=1)).isoformat(),
                        "collection": "sora_knowledge",
                        "reason": "hnsw_corruption",
                        "source": "upsert",
                        "shared": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suspects = RAGRepairService(enabled=True).discover_suspect_animas(
        window_minutes=60,
        include_logs=False,
        include_quick_check=False,
    )

    assert suspects == []


def test_discover_suspect_animas_keeps_signals_after_success(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# sora", encoding="utf-8")
    now = datetime.now(UTC)

    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "status": "success",
                "reason": "startup_chroma_crash_preflight",
                "last_success_at": (now - timedelta(minutes=1)).isoformat(),
                "recent_signals": [
                    {
                        "at": now.isoformat(),
                        "collection": "sora_knowledge",
                        "reason": "hnsw_corruption",
                        "source": "upsert",
                        "shared": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suspects = RAGRepairService(enabled=True).discover_suspect_animas(
        window_minutes=60,
        include_logs=False,
        include_quick_check=False,
    )

    assert suspects == ["sora"]


def test_discover_suspect_animas_ignores_failed_state_before_success(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# sora", encoding="utf-8")
    now = datetime.now(UTC)

    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "reason": "startup_chroma_crash_preflight",
                "last_success_at": now.isoformat(),
                "updated_at": (now - timedelta(minutes=1)).isoformat(),
                "last_failure_at": (now - timedelta(minutes=1)).isoformat(),
                "last_attempt_at": (now - timedelta(minutes=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    suspects = RAGRepairService(enabled=True).discover_suspect_animas(
        window_minutes=60,
        include_logs=False,
        include_quick_check=False,
    )

    assert suspects == []


def test_discover_suspect_animas_keeps_failed_state_after_success(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# sora", encoding="utf-8")
    now = datetime.now(UTC)

    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "reason": "startup_chroma_crash_preflight",
                "last_success_at": (now - timedelta(minutes=1)).isoformat(),
                "updated_at": now.isoformat(),
                "last_failure_at": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    suspects = RAGRepairService(enabled=True).discover_suspect_animas(
        window_minutes=60,
        include_logs=False,
        include_quick_check=False,
    )

    assert suspects == ["sora"]


def test_discover_suspect_animas_ignores_logs_before_success(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# sora", encoding="utf-8")
    (anima_dir / "vectordb").mkdir()
    now = datetime.now(UTC)

    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "status": "success",
                "last_success_at": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    log_path = data_dir / "logs" / "server-daemon.log"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(
        f"{(now - timedelta(minutes=1)).isoformat()} tokio-rt-worker segfault in chromadb_rust_bindings.abi3.so\n",
        encoding="utf-8",
    )

    suspects = RAGRepairService(enabled=True).discover_suspect_animas(
        window_minutes=60,
        include_quick_check=False,
        log_paths=[log_path],
    )

    assert suspects == []


def test_discover_suspect_animas_keeps_logs_after_success(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# sora", encoding="utf-8")
    (anima_dir / "vectordb").mkdir()
    now = datetime.now(UTC)

    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "status": "success",
                "last_success_at": (now - timedelta(minutes=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    log_path = data_dir / "logs" / "server-daemon.log"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(
        f"{now.isoformat()} tokio-rt-worker segfault in chromadb_rust_bindings.abi3.so\n",
        encoding="utf-8",
    )

    suspects = RAGRepairService(enabled=True).discover_suspect_animas(
        window_minutes=60,
        include_quick_check=False,
        log_paths=[log_path],
    )

    assert suspects == ["sora"]


def test_discover_suspect_animas_ignores_timestampless_logs_after_success(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# sora", encoding="utf-8")
    (anima_dir / "vectordb").mkdir()
    now = datetime.now(UTC)

    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "status": "success",
                "last_success_at": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    log_path = data_dir / "logs" / "server-daemon.log"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(
        "tokio-rt-worker segfault in chromadb_rust_bindings.abi3.so\n",
        encoding="utf-8",
    )

    suspects = RAGRepairService(enabled=True).discover_suspect_animas(
        window_minutes=60,
        include_quick_check=False,
        log_paths=[log_path],
    )

    assert suspects == []


def test_discover_suspect_animas_keeps_timestampless_logs_without_success(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# sora", encoding="utf-8")
    (anima_dir / "vectordb").mkdir()

    log_path = data_dir / "logs" / "server-daemon.log"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(
        "tokio-rt-worker segfault in chromadb_rust_bindings.abi3.so\n",
        encoding="utf-8",
    )

    suspects = RAGRepairService(enabled=True).discover_suspect_animas(
        window_minutes=60,
        include_quick_check=False,
        log_paths=[log_path],
    )

    assert suspects == ["sora"]


def test_discover_suspect_animas_ignores_legacy_unclean_exit_state(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# sora", encoding="utf-8")
    now = datetime.now(UTC)

    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "reason": "startup_unclean_exit_preflight",
                "updated_at": now.isoformat(),
                "last_failure_at": now.isoformat(),
                "recent_signals": [
                    {
                        "at": now.isoformat(),
                        "collection": "sora_knowledge",
                        "reason": "startup_unclean_exit_preflight",
                        "source": "startup_preflight",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suspects = RAGRepairService(enabled=True).discover_suspect_animas(
        window_minutes=60,
        include_logs=False,
        include_quick_check=False,
    )

    assert suspects == []


def test_repair_animas_if_allowed_runs_each_target(data_dir: Path):
    service = RAGRepairService(enabled=True)
    service.repair_anima_if_allowed = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda anima_name, **kwargs: RepairResult(
            status="success",
            anima_name=anima_name,
            reason=kwargs["reason"],
        )
    )

    results = service.repair_animas_if_allowed(
        {"sora", "rin"},
        reason="startup_chroma_crash_preflight",
        source="startup_preflight",
        include_shared=True,
    )

    assert list(results) == ["rin", "sora"]
    assert service.repair_anima_if_allowed.call_count == 2


def test_request_repair_sync_uses_guard(data_dir: Path):
    (data_dir / "animas" / "sora" / "state").mkdir(parents=True)
    service = RAGRepairService(enabled=True)
    service.repair_anima = MagicMock(  # type: ignore[method-assign]
        return_value=RepairResult(status="success", anima_name="sora", reason="test")
    )

    assert service.request_repair("sora", reason="test", source="test", background=False) is True
    service.repair_anima.assert_called_once()


def test_request_repair_disabled_is_blocked(data_dir: Path):
    service = RAGRepairService(enabled=False)

    assert service.request_repair("sora", reason="test", source="test", background=False) is False


def test_request_repair_background_records_request_without_running_repair(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    service = RAGRepairService(enabled=True)
    service.repair_anima = MagicMock()  # type: ignore[method-assign]

    assert service.request_repair("sora", reason="sqlite_malformed", source="query", background=True) is True

    service.repair_anima.assert_not_called()
    state = json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert state["status"] == "requested"
    assert state["stage"] == "detect"
    assert state["reason"] == "sqlite_malformed"
    assert state["source"] == "query"
    assert state["pid"] is None


def test_background_duplicate_request_is_not_started_twice(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    service = RAGRepairService(enabled=True)

    assert service.request_repair("sora", reason="sqlite_malformed", source="query", background=True) is True
    assert service.request_repair("sora", reason="sqlite_malformed", source="query", background=True) is False

    state = json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert state["status"] == "requested"


def test_quarantine_resets_worker_before_local_cache_and_move(data_dir: Path, monkeypatch):
    from core.memory.rag.http_store import HttpVectorStore
    from core.memory.rag.repair_rebuild import quarantine_vectordb

    anima_dir = data_dir / "animas" / "sora"
    vectordb = anima_dir / "vectordb"
    vectordb.mkdir(parents=True)
    (vectordb / "broken.bin").write_text("broken", encoding="utf-8")

    events: list[str] = []
    store = HttpVectorStore("http://worker", anima_name="sora")
    store.reset_store = MagicMock(side_effect=lambda: events.append("worker") or True)  # type: ignore[method-assign]
    local_reset = MagicMock(side_effect=lambda anima_name: events.append("local"))

    monkeypatch.setenv("ANIMAWORKS_VECTOR_URL", "http://worker")
    monkeypatch.setattr("core.memory.rag.singleton.get_vector_store", lambda anima_name=None: store)
    monkeypatch.setattr("core.memory.rag.singleton.reset_vector_store", local_reset)

    archive = quarantine_vectordb("sora")

    # Worker cache is reset before the move (release handles for the OS move) and
    # again after the move (discard any handle a concurrent read pinned while the
    # directory was missing), with the local cache reset each time.
    assert events == ["worker", "local", "worker", "local"]
    assert store.reset_store.call_count == 2
    assert local_reset.call_count == 2
    local_reset.assert_called_with("sora")
    assert archive is not None
    # The vectordb dir is recreated empty so racing reads open a valid (empty) DB
    # rather than a schema-less stub; the corrupt contents live in the archive.
    assert vectordb.exists()
    assert not any(vectordb.iterdir())
    assert (archive / "broken.bin").read_text(encoding="utf-8") == "broken"


class _FakeBuildStore:
    """Fake direct ChromaVectorStore used to drive ``atomic_rebuild_vectordb``."""

    def __init__(self, collections: list[str] | None = None) -> None:
        self._collections = ["rebuilt"] if collections is None else collections
        self.closed = False

    def list_collections(self) -> list[str]:
        return list(self._collections)

    def close(self) -> None:
        self.closed = True


def _patch_atomic_build(monkeypatch, *, chunks_per_dir=2, collections=None, indexer_calls=None):
    """Patch the pieces ``atomic_rebuild_vectordb`` builds with (no real chroma)."""

    class FakeIndexer:
        def __init__(self, *args, **kwargs):
            self.anima_name = kwargs.get("anima_name")

        def index_directory(self, *args, **kwargs):
            if indexer_calls is not None:
                indexer_calls.append((self.anima_name, str(args[1])))
            return chunks_per_dir

        def index_conversation_summary(self, *args, **kwargs):
            return chunks_per_dir

    monkeypatch.setattr("core.memory.rag.MemoryIndexer", FakeIndexer)
    monkeypatch.setattr(
        "core.memory.rag.store.create_chroma_vector_store",
        lambda *a, **k: _FakeBuildStore(collections=collections),
    )
    monkeypatch.setattr("core.memory.rag.repair_rebuild.reset_worker_vector_store", lambda anima_name: True)


def test_repair_rebuilds_swaps_and_archives(data_dir: Path, monkeypatch):
    """Atomic repair builds in staging, swaps in, and archives the old DB."""
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "knowledge" / "topic.md").write_text("# Topic\n\n## A\n\nbody", encoding="utf-8")
    (anima_dir / "state").mkdir()
    (anima_dir / "state" / "conversation.json").write_text(
        '{"compressed_summary": "## Summary\\n\\nhello"}', encoding="utf-8"
    )
    vectordb = anima_dir / "vectordb"
    vectordb.mkdir()
    (vectordb / "broken.bin").write_text("broken", encoding="utf-8")

    _patch_atomic_build(monkeypatch, chunks_per_dir=2)
    monkeypatch.setattr("core.memory.rag.singleton.reset_vector_store", lambda anima_name=None: None)

    service = RAGRepairService(enabled=True, threshold=2, window_minutes=5, cooldown_minutes=60)
    result = service.repair_anima("sora", reason="chroma_error_finding_id", source="test")

    assert result.ok
    assert result.chunks_indexed == 4  # knowledge dir (2) + conversation summary (2)
    # Live DB swapped in; the old corrupt DB archived; no staging dir left behind.
    assert vectordb.exists()
    assert not (vectordb / "broken.bin").exists()
    assert not list(anima_dir.glob("vectordb.staging-*"))
    archive_dirs = list((anima_dir / "archive").glob("vectordb-corrupt-*"))
    assert len(archive_dirs) == 1
    assert (archive_dirs[0] / "broken.bin").read_text(encoding="utf-8") == "broken"

    state = json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert state["status"] == "success"
    assert state["consecutive_failures"] == 0
    assert state["last_chunks_indexed"] == 4


def test_repair_reindexes_shared_collections_when_requested(data_dir: Path, monkeypatch):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    common_knowledge = data_dir / "common_knowledge"
    common_knowledge.mkdir(exist_ok=True)
    (common_knowledge / "ref.md").write_text("# Reference", encoding="utf-8")
    common_skills = data_dir / "common_skills"
    (common_skills / "tool").mkdir(parents=True, exist_ok=True)
    (common_skills / "tool" / "SKILL.md").write_text("# Tool", encoding="utf-8")

    calls: list[tuple[str | None, str]] = []
    _patch_atomic_build(monkeypatch, indexer_calls=calls)
    monkeypatch.setattr("core.memory.rag.singleton.reset_vector_store", lambda anima_name=None: None)

    result = RAGRepairService(enabled=True).repair_anima(
        "sora", reason="chroma_corruption", source="test", include_shared=True
    )

    assert result.ok
    assert ("shared", "common_knowledge") in calls
    assert ("shared", "common_skills") in calls


def test_repair_failure_preserves_live_db_and_records_state(data_dir: Path, monkeypatch):
    """A rebuild that fails must leave the live DB intact and engage cooldown."""
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    vectordb = anima_dir / "vectordb"
    vectordb.mkdir()
    (vectordb / "live.bin").write_text("live-data", encoding="utf-8")

    reset = MagicMock()
    monkeypatch.setattr(
        "core.memory.rag.repair_service.atomic_rebuild_vectordb",
        lambda anima_name, include_shared=False: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr("core.memory.rag.singleton.reset_vector_store", reset)

    result = RAGRepairService(enabled=True).repair_anima("sora", reason="chroma_corruption", source="test")

    assert result.status == "failed"
    assert result.error == "boom"
    reset.assert_called_once_with("sora")
    # The live DB is untouched — a failed atomic rebuild never destroys data.
    assert (vectordb / "live.bin").read_text(encoding="utf-8") == "live-data"
    state = json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["consecutive_failures"] == 1


def test_atomic_rebuild_stub_fails_and_keeps_live_db(data_dir: Path, monkeypatch):
    """A staged DB with no collections (failed upserts) fails before the swap.

    The live DB must remain in place so the false-success re-quarantine loop
    cannot start and no data is lost.
    """
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "knowledge" / "topic.md").write_text("# Topic\n\n## A\n\nbody", encoding="utf-8")
    (anima_dir / "state").mkdir()
    vectordb = anima_dir / "vectordb"
    vectordb.mkdir()
    (vectordb / "live.bin").write_text("live-data", encoding="utf-8")

    # Staged store reports no collections despite indexed chunks -> stub.
    _patch_atomic_build(monkeypatch, chunks_per_dir=3, collections=[])
    monkeypatch.setattr("core.memory.rag.singleton.reset_vector_store", lambda anima_name=None: None)

    service = RAGRepairService(enabled=True, threshold=2, window_minutes=5, cooldown_minutes=60)
    result = service.repair_anima("sora", reason="chroma_corruption", source="test")

    assert result.status == "failed"
    assert (vectordb / "live.bin").read_text(encoding="utf-8") == "live-data"  # live preserved
    assert not list(anima_dir.glob("vectordb.staging-*"))  # staging cleaned up
    state = json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["consecutive_failures"] == 1


def test_record_chroma_error_suppressed_during_active_repair(data_dir: Path):
    """Corruption signals must be ignored while a repair is already in flight.

    Reads during a rebuild see a transiently empty DB; recording those signals
    would re-trigger another repair the instant the current one finishes.
    """
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps({"status": "repairing"}),
        encoding="utf-8",
    )

    service = RAGRepairService(enabled=True, threshold=2, window_minutes=5, cooldown_minutes=60)
    service.request_repair = MagicMock(return_value=True)  # type: ignore[method-assign]

    assert (
        service.record_chroma_error(
            anima_name="sora",
            collection="sora_knowledge",
            error="database disk image is malformed",
            source="query",
        )
        is False
    )
    service.request_repair.assert_not_called()
    state = json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert not state.get("recent_signals")


def test_repair_missing_anima_fails(data_dir: Path):
    result = RAGRepairService(enabled=True).repair_anima(
        "missing",
        reason="chroma_corruption",
        source="test",
    )

    assert result.status == "failed"
    assert result.error == "anima not found"


def test_repair_if_allowed_respects_failed_cooldown(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "last_attempt_at": datetime.now(UTC).isoformat(),
                "last_failure_at": datetime.now(UTC).isoformat(),
                "consecutive_failures": 2,
            }
        ),
        encoding="utf-8",
    )

    service = RAGRepairService(enabled=True, threshold=2, window_minutes=5, cooldown_minutes=60)
    result = service.repair_anima_if_allowed(
        "sora",
        reason="recent_rag_corruption",
        source="test",
    )

    assert result.status == "cooldown"


def test_repair_if_allowed_retries_before_failure_limit(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "last_attempt_at": datetime.now(UTC).isoformat(),
                "last_failure_at": datetime.now(UTC).isoformat(),
                "consecutive_failures": 1,
            }
        ),
        encoding="utf-8",
    )

    service = RAGRepairService(enabled=True, threshold=2, window_minutes=5, cooldown_minutes=60)
    service.repair_anima = MagicMock(  # type: ignore[method-assign]
        return_value=RepairResult(status="success", anima_name="sora", reason="recent_rag_corruption")
    )

    result = service.repair_anima_if_allowed(
        "sora",
        reason="recent_rag_corruption",
        source="test",
    )

    assert result.status == "success"
    service.repair_anima.assert_called_once()
