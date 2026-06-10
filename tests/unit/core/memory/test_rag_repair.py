# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RAG auto-repair detection and rebuild orchestration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from core.memory.rag.repair import (
    RAGRepairService,
    RepairResult,
    classify_corruption_error,
    collection_owner,
)
from core.memory.rag.sqlite_health import SQLiteHealthResult


def test_classifies_today_error_finding_id():
    err = "Error executing plan: Internal error: Error finding id"
    assert classify_corruption_error(err) == "chroma_error_finding_id"


def test_classifies_native_and_sqlite_corruption():
    assert classify_corruption_error("database disk image is malformed") == "sqlite_malformed"
    assert classify_corruption_error(-11) == "native_segfault"
    assert classify_corruption_error("hnsw index panic: corrupt graph") == "hnsw_corruption"


def test_does_not_classify_operational_noise():
    assert classify_corruption_error("Connection refused") is None
    assert classify_corruption_error("Collection 'foo' not found") is None


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


def test_repair_quarantines_vectordb_and_reindexes(data_dir: Path, monkeypatch):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "knowledge" / "topic.md").write_text("# Topic\n\n## A\n\nbody", encoding="utf-8")
    (anima_dir / "state").mkdir()
    (anima_dir / "state" / "conversation.json").write_text(
        '{"compressed_summary": "## Summary\\n\\nhello"}',
        encoding="utf-8",
    )
    vectordb = anima_dir / "vectordb"
    vectordb.mkdir()
    (vectordb / "broken.bin").write_text("broken", encoding="utf-8")

    class FakeIndexer:
        def __init__(self, *args, **kwargs):
            pass

        def index_directory(self, *args, **kwargs):
            return 3

        def index_conversation_summary(self, *args, **kwargs):
            return 1

    monkeypatch.setattr("core.memory.rag.MemoryIndexer", FakeIndexer)
    monkeypatch.setenv("ANIMAWORKS_VECTOR_URL", "http://worker")
    monkeypatch.setattr("core.memory.rag.singleton.get_vector_store", lambda anima_name=None: object())

    service = RAGRepairService(enabled=True, threshold=2, window_minutes=5, cooldown_minutes=60)
    result = service.repair_anima(
        "sora",
        reason="chroma_error_finding_id",
        collection="sora_knowledge",
        source="test",
    )

    assert result.ok
    assert result.chunks_indexed == 4
    assert not vectordb.exists()
    archive_dirs = list((anima_dir / "archive").glob("vectordb-corrupt-*"))
    assert len(archive_dirs) == 1
    assert (archive_dirs[0] / "broken.bin").read_text(encoding="utf-8") == "broken"

    state = json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert state["status"] == "success"
    assert state["stage"] == "complete"
    assert state["pid"] is None
    assert state["consecutive_failures"] == 0
    assert state["last_chunks_indexed"] == 4
    assert (anima_dir / "state" / "bm25_longterm_index.json").exists()


def test_repair_reindexes_shared_collections_when_requested(data_dir: Path, monkeypatch):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    common_knowledge = data_dir / "common_knowledge"
    common_knowledge.mkdir(exist_ok=True)
    (common_knowledge / "ref.md").write_text("# Reference", encoding="utf-8")
    common_skills = data_dir / "common_skills"
    common_skills.mkdir(exist_ok=True)
    (common_skills / "tool" / "SKILL.md").parent.mkdir(exist_ok=True)
    (common_skills / "tool" / "SKILL.md").write_text("# Tool", encoding="utf-8")

    calls: list[tuple[str | None, str]] = []

    class FakeIndexer:
        def __init__(self, *args, **kwargs):
            self.anima_name = kwargs.get("anima_name")

        def index_directory(self, *args, **kwargs):
            calls.append((self.anima_name, str(args[1])))
            return 2

        def index_conversation_summary(self, *args, **kwargs):
            return 0

    monkeypatch.setattr("core.memory.rag.MemoryIndexer", FakeIndexer)
    monkeypatch.setenv("ANIMAWORKS_VECTOR_URL", "http://worker")
    monkeypatch.setattr("core.memory.rag.singleton.get_vector_store", lambda anima_name=None: object())

    result = RAGRepairService(enabled=True).repair_anima(
        "sora",
        reason="chroma_corruption",
        source="test",
        include_shared=True,
    )

    assert result.ok
    assert ("shared", "common_knowledge") in calls
    assert ("shared", "common_skills") in calls


def test_repair_failure_records_state_and_resets(data_dir: Path, monkeypatch):
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    reset = MagicMock()

    monkeypatch.setattr("core.memory.rag.repair_service.quarantine_vectordb", lambda anima_name: None)
    monkeypatch.setattr(
        "core.memory.rag.repair_service.full_reindex",
        lambda anima_name, include_shared=False: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr("core.memory.rag.singleton.reset_vector_store", reset)

    result = RAGRepairService(enabled=True).repair_anima(
        "sora",
        reason="chroma_corruption",
        source="test",
    )

    assert result.status == "failed"
    assert result.error == "boom"
    reset.assert_called_once_with("sora")
    state = json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["last_error"] == "boom"


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
