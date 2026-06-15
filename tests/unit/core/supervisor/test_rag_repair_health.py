# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Supervisor integration tests for RAG auto-repair triggers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.supervisor._mgr_health import HealthMixin
from core.supervisor._mgr_rag_repair import RAGRepairMixin


class _SupervisorForTest(HealthMixin, RAGRepairMixin):
    def __init__(self, animas_dir: Path) -> None:
        self.animas_dir = animas_dir
        self._restart_counts: dict[str, int] = {"sora": 2}
        self.events: list[tuple[str, dict]] = []

    async def _broadcast_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


class _RepairServiceForTest:
    def __init__(self, *, recent: bool) -> None:
        self.recent = recent

    def has_recent_corruption(self, anima_name: str) -> bool:
        return self.recent


async def test_repair_before_restart_on_sigsegv(monkeypatch, tmp_path):
    service = _RepairServiceForTest(recent=False)
    monkeypatch.setattr("core.memory.rag.repair.get_repair_service", lambda: service)

    supervisor = _SupervisorForTest(tmp_path)
    repair_calls: list[dict] = []

    async def fake_run_repair(anima_name: str, *, reason: str, include_shared: bool):
        repair_calls.append(
            {
                "anima_name": anima_name,
                "reason": reason,
                "include_shared": include_shared,
            },
        )
        return {"ok": True, "status": "success", "stdout": "", "stderr": ""}

    supervisor._run_rag_repair_cli_process = fake_run_repair  # type: ignore[method-assign]
    handle = SimpleNamespace(stats=SimpleNamespace(exit_code=-11))

    repaired = await supervisor._maybe_repair_rag_before_restart("sora", handle)

    assert repaired is True
    assert repair_calls == [
        {
            "anima_name": "sora",
            "reason": "native_segfault",
            "include_shared": True,
        },
    ]
    assert supervisor._restart_counts["sora"] == 0
    assert supervisor.events[0][0] == "system.rag_repair"


async def test_repair_before_restart_on_recent_rag_corruption(monkeypatch, tmp_path):
    service = _RepairServiceForTest(recent=True)
    monkeypatch.setattr("core.memory.rag.repair.get_repair_service", lambda: service)

    supervisor = _SupervisorForTest(tmp_path)
    repair_calls: list[dict] = []

    async def fake_run_repair(anima_name: str, *, reason: str, include_shared: bool):
        repair_calls.append(
            {
                "anima_name": anima_name,
                "reason": reason,
                "include_shared": include_shared,
            },
        )
        return {"ok": True, "status": "success", "stdout": "", "stderr": ""}

    supervisor._run_rag_repair_cli_process = fake_run_repair  # type: ignore[method-assign]
    handle = SimpleNamespace(stats=SimpleNamespace(exit_code=None))

    repaired = await supervisor._maybe_repair_rag_before_restart("sora", handle)

    assert repaired is True
    assert repair_calls == [
        {
            "anima_name": "sora",
            "reason": "recent_rag_corruption",
            "include_shared": True,
        },
    ]


async def test_recent_corruption_skipped_when_sqlite_quick_check_ok(monkeypatch, tmp_path):
    """A healthy on-disk SQLite refutes the soft recent_rag_corruption evidence.

    The pre-restart repair path must not rebuild an intact DB just because
    transient corruption signals (cache poisoning) were seen recently.
    """
    import sqlite3

    service = _RepairServiceForTest(recent=True)
    monkeypatch.setattr("core.memory.rag.repair.get_repair_service", lambda: service)

    supervisor = _SupervisorForTest(tmp_path)
    repair_calls: list[dict] = []

    async def fake_run_repair(anima_name: str, *, reason: str, include_shared: bool):
        repair_calls.append({"anima_name": anima_name, "reason": reason, "include_shared": include_shared})
        return {"ok": True, "status": "success", "stdout": "", "stderr": ""}

    supervisor._run_rag_repair_cli_process = fake_run_repair  # type: ignore[method-assign]

    # Intact Chroma SQLite at the supervisor-resolved path.
    vdb = tmp_path / "sora" / "vectordb"
    vdb.mkdir(parents=True)
    conn = sqlite3.connect(vdb / "chroma.sqlite3")
    try:
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
    finally:
        conn.close()

    handle = SimpleNamespace(stats=SimpleNamespace(exit_code=143))
    repaired = await supervisor._maybe_repair_rag_before_restart("sora", handle)

    assert repaired is False
    assert repair_calls == []


async def test_native_segfault_not_gated_by_sqlite_check(monkeypatch, tmp_path):
    """A native segfault (hard signal) still repairs even if SQLite looks intact."""
    import sqlite3

    service = _RepairServiceForTest(recent=False)
    monkeypatch.setattr("core.memory.rag.repair.get_repair_service", lambda: service)

    supervisor = _SupervisorForTest(tmp_path)
    repair_calls: list[dict] = []

    async def fake_run_repair(anima_name: str, *, reason: str, include_shared: bool):
        repair_calls.append({"anima_name": anima_name, "reason": reason, "include_shared": include_shared})
        return {"ok": True, "status": "success", "stdout": "", "stderr": ""}

    supervisor._run_rag_repair_cli_process = fake_run_repair  # type: ignore[method-assign]

    vdb = tmp_path / "sora" / "vectordb"
    vdb.mkdir(parents=True)
    conn = sqlite3.connect(vdb / "chroma.sqlite3")
    try:
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
    finally:
        conn.close()

    handle = SimpleNamespace(stats=SimpleNamespace(exit_code=-11))
    repaired = await supervisor._maybe_repair_rag_before_restart("sora", handle)

    assert repaired is True
    assert repair_calls == [{"anima_name": "sora", "reason": "native_segfault", "include_shared": True}]


async def test_no_repair_without_exit_or_recent_signal(monkeypatch, tmp_path):
    service = _RepairServiceForTest(recent=False)
    monkeypatch.setattr("core.memory.rag.repair.get_repair_service", lambda: service)

    supervisor = _SupervisorForTest(tmp_path)
    repair_calls: list[dict] = []

    async def fake_run_repair(anima_name: str, *, reason: str, include_shared: bool):
        repair_calls.append(
            {
                "anima_name": anima_name,
                "reason": reason,
                "include_shared": include_shared,
            },
        )
        return {"ok": True, "status": "success", "stdout": "", "stderr": ""}

    supervisor._run_rag_repair_cli_process = fake_run_repair  # type: ignore[method-assign]
    handle = SimpleNamespace(stats=SimpleNamespace(exit_code=1))

    repaired = await supervisor._maybe_repair_rag_before_restart("sora", handle)

    assert repaired is False
    assert repair_calls == []
