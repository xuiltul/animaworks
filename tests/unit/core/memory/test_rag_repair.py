# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RAG auto-repair detection and rebuild orchestration."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.memory.rag.repair import (
    RAGRepairService,
    classify_corruption_error,
    collection_owner,
)
from core.memory.rag.repair.detect import SINGLE_SHOT_REASONS
from core.memory.rag.sqlite_health import SQLiteHealthResult


def test_classifies_today_error_finding_id():
    err = "Error executing plan: Internal error: Error finding id"
    assert classify_corruption_error(err) == "chroma_error_finding_id"


def test_classifies_native_and_sqlite_corruption():
    assert classify_corruption_error("database disk image is malformed") == "sqlite_malformed"
    assert classify_corruption_error(-11) == "native_segfault"
    assert classify_corruption_error("segmentation fault") == "native_segfault"
    assert classify_corruption_error("hnsw index panic: corrupt graph") == "hnsw_corruption"
    assert classify_corruption_error("Failed to get segments for collection") == "chroma_transient"
    assert classify_corruption_error("no such table: embeddings_queue") == "chroma_corruption"
    assert classify_corruption_error("store_init_failed: no such table: tenants") == "store_init_failed"


def test_does_not_classify_operational_noise():
    assert classify_corruption_error("Connection refused") is None
    assert classify_corruption_error("Collection 'foo' not found") is None


def test_does_not_classify_resource_exhaustion_as_corruption():
    assert classify_corruption_error("hnsw segment reader: Too many open files (os error 24)") is None
    assert classify_corruption_error("unable to open database file") is None
    assert (
        classify_corruption_error("Internal error: error returned from database: (code: 522) disk I/O error")
        == "chroma_transient"
    )


def test_chroma_transient_is_not_single_shot_and_does_not_start_repair(data_dir: Path):
    service = RAGRepairService(enabled=True, threshold=1, window_minutes=5, cooldown_minutes=60)
    service.request_repair = MagicMock(return_value=True)  # type: ignore[method-assign]
    service._sqlite_quick_check_ok = MagicMock(return_value=True)  # type: ignore[method-assign]

    assert "chroma_transient" not in SINGLE_SHOT_REASONS
    assert (
        service.record_chroma_error(
            anima_name="sora",
            collection="sora_knowledge",
            error="Failed to get segments for collection",
            source="upsert",
        )
        is False
    )
    service.request_repair.assert_not_called()
    service._sqlite_quick_check_ok.assert_not_called()


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


def test_store_init_failed_is_single_shot_and_not_refuted_by_quick_check(data_dir: Path):
    service = RAGRepairService(enabled=True, threshold=99, window_minutes=5, cooldown_minutes=60)
    service.request_repair = MagicMock(return_value=True)  # type: ignore[method-assign]
    service._sqlite_quick_check_ok = MagicMock(return_value=True)  # type: ignore[method-assign]

    assert "store_init_failed" in SINGLE_SHOT_REASONS
    assert (
        service.record_chroma_error(
            anima_name="sora",
            collection="sora_knowledge",
            error="store_init_failed: no such table: tenants",
            source="vector_store_init",
        )
        is True
    )

    service.request_repair.assert_called_once_with(
        "sora",
        reason="store_init_failed",
        collection="sora_knowledge",
        source="vector_store_init",
        include_shared=True,
    )
    service._sqlite_quick_check_ok.assert_not_called()
    state = json.loads((data_dir / "animas" / "sora" / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert state["recent_signals"][-1]["reason"] == "store_init_failed"


def test_store_init_failed_signal_is_throttled_for_ten_minutes(data_dir: Path):
    service = RAGRepairService(enabled=True, threshold=1, window_minutes=5, cooldown_minutes=60)
    service.request_repair = MagicMock(return_value=True)  # type: ignore[method-assign]
    service._has_active_repair_state = MagicMock(return_value=False)  # type: ignore[method-assign]

    first = service.record_chroma_error(
        anima_name="sora",
        collection="sora_knowledge",
        error="store_init_failed: schema missing",
        source="vector_store_init",
    )
    second = service.record_chroma_error(
        anima_name="sora",
        collection="sora_knowledge",
        error="store_init_failed: schema still missing",
        source="vector_store_init",
    )

    assert first is True
    assert second is False
    assert service.request_repair.call_count == 1
    state = json.loads((data_dir / "animas" / "sora" / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert [signal["reason"] for signal in state["recent_signals"]] == ["store_init_failed"]


def test_chroma_corruption_suppressed_when_sqlite_quick_check_ok(data_dir: Path):
    """A healthy on-disk SQLite refutes a chroma_corruption signal.

    chromadb's process-global cache can be transiently poisoned, making an
    intact DB raise sqlite-shaped structural errors. A passing quick_check
    means the store is fine, so we must not escalate to a destructive repair.
    """
    service = RAGRepairService(enabled=True, threshold=1, window_minutes=5, cooldown_minutes=60)
    service.request_repair = MagicMock(return_value=True)  # type: ignore[method-assign]
    service._sqlite_quick_check_ok = lambda owner: True  # type: ignore[method-assign,assignment]

    assert (
        service.record_chroma_error(
            anima_name="sora",
            collection="sora_knowledge",
            error="Error getting collection: no such table: collections",
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
            error="Error getting collection: no such table: collections",
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
        (anima_dir / "IDENTITY.md").write_text(f"# {name}", encoding="utf-8")
    (data_dir / "animas" / "rin" / "vectordb").mkdir()
    disabled = data_dir / "animas" / "mika"
    disabled.mkdir(parents=True)
    (disabled / "IDENTITY.md").write_text("# mika", encoding="utf-8")
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
    (anima_dir / "IDENTITY.md").write_text("# sora", encoding="utf-8")
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
        "core.memory.rag.sqlite_health.check_anima_vectordb_health",
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


def test_discover_suspect_animas_includes_phase3_db_in_quick_check(data_dir: Path, monkeypatch):
    anima_dir = data_dir / "animas" / "sora"
    anima_dir.mkdir(parents=True)
    (anima_dir / "IDENTITY.md").write_text("# sora", encoding="utf-8")
    (anima_dir / "status.json").write_text('{"process_model":"phase3"}', encoding="utf-8")
    quick_check = MagicMock(return_value=MagicMock(corrupt=False))
    monkeypatch.setattr(
        "core.memory.rag.sqlite_health.check_anima_vectordb_health",
        quick_check,
    )

    suspects = RAGRepairService(enabled=True).discover_suspect_animas(include_logs=False)

    assert suspects == []
    quick_check.assert_called_once_with(
        "sora",
        timeout_seconds=10.0,
        source="startup_quick_check",
        record_repair=False,
    )


def test_discover_suspect_animas_ignores_signals_before_success(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "IDENTITY.md").write_text("# sora", encoding="utf-8")
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
    (anima_dir / "IDENTITY.md").write_text("# sora", encoding="utf-8")
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
    (anima_dir / "IDENTITY.md").write_text("# sora", encoding="utf-8")
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
    (anima_dir / "IDENTITY.md").write_text("# sora", encoding="utf-8")
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
    (anima_dir / "IDENTITY.md").write_text("# sora", encoding="utf-8")
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
    (anima_dir / "IDENTITY.md").write_text("# sora", encoding="utf-8")
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
    (anima_dir / "IDENTITY.md").write_text("# sora", encoding="utf-8")
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
    (anima_dir / "IDENTITY.md").write_text("# sora", encoding="utf-8")
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
    (anima_dir / "IDENTITY.md").write_text("# sora", encoding="utf-8")
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


def test_request_repair_disabled_is_blocked(data_dir: Path):
    service = RAGRepairService(enabled=False)

    assert service.request_repair("sora", reason="test", source="test") is False


def test_request_repair_records_request_without_running_repair(data_dir: Path):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    service = RAGRepairService(enabled=True)

    assert service.request_repair("sora", reason="sqlite_malformed", source="query") is True
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

    assert service.request_repair("sora", reason="sqlite_malformed", source="query") is True
    assert service.request_repair("sora", reason="sqlite_malformed", source="query") is False

    state = json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert state["status"] == "requested"


def test_staging_rebuild_rejects_non_staging_direct_chroma_path(tmp_path: Path) -> None:
    from core.memory.rag.repair.rebuild import build_staging_vectordb

    anima_dir = tmp_path / "sora"
    anima_dir.mkdir()

    with pytest.raises(ValueError, match="staging directory"):
        build_staging_vectordb(
            "sora",
            include_shared=False,
            anima_dir=anima_dir,
            staging=anima_dir / "vectordb",
        )


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


def test_store_init_failed_is_persisted_during_active_repair(data_dir: Path):
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
            error="store_init_failed: no such table: tenants",
            source="vector_store_init",
        )
        is False
    )

    service.request_repair.assert_not_called()
    state = json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert state["status"] == "repairing"
    assert state["recent_signals"][-1]["reason"] == "store_init_failed"


def test_chroma_repair_verification_runs_real_query() -> None:
    from core.memory.rag.store import ChromaVectorStore

    collection = MagicMock()
    collection.count.return_value = 1
    collection.get.return_value = {"ids": ["doc-1"], "embeddings": [[0.1, 0.2]]}
    collection.query.return_value = {"ids": [["doc-1"]]}
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.client = MagicMock()
    store.client.list_collections.return_value = [MagicMock(name="knowledge")]
    store.client.list_collections.return_value[0].name = "knowledge"
    store.client.get_collection.return_value = collection

    result = store.verify_rebuilt_data(expected_chunks=1)

    assert result == {"collections": 1, "chunks": 1, "query_results": 1}
    collection.query.assert_called_once_with(query_embeddings=[[0.1, 0.2]], n_results=1)


def test_chroma_repair_verification_rejects_wrong_chunk_count() -> None:
    from core.memory.rag.store import ChromaVectorStore

    collection = MagicMock()
    collection.count.return_value = 0
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.client = MagicMock()
    store.client.list_collections.return_value = [MagicMock(name="knowledge")]
    store.client.list_collections.return_value[0].name = "knowledge"
    store.client.get_collection.return_value = collection

    with pytest.raises(RuntimeError, match="expected 1 chunks.*found 0"):
        store.verify_rebuilt_data(expected_chunks=1)


def test_rag_repair_nonce_env_sets_and_restores(monkeypatch) -> None:
    from core.memory.rag.repair.detect import rag_repair_nonce_env

    monkeypatch.delenv("ANIMAWORKS_RAG_REPAIR_NONCE", raising=False)
    with rag_repair_nonce_env() as nonce:
        assert nonce
        assert os.environ["ANIMAWORKS_RAG_REPAIR_NONCE"] == nonce
    assert "ANIMAWORKS_RAG_REPAIR_NONCE" not in os.environ

    monkeypatch.setenv("ANIMAWORKS_RAG_REPAIR_NONCE", "keep-me")
    with rag_repair_nonce_env() as nonce:
        assert nonce == "keep-me"
    assert os.environ["ANIMAWORKS_RAG_REPAIR_NONCE"] == "keep-me"
