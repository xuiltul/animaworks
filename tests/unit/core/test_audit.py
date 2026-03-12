from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for core.audit — organisation-wide audit collection."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.audit import (
    AnimaAuditEntry,
    OrgAuditReport,
    _collect_single_anima,
    _load_entries_for_date,
    collect_org_audit,
)

_DATE = "2026-03-07"


# ── Fixtures ──────────────────────────────────────────────────


def _write_activity_log(anima_dir: Path, date: str, entries: list[dict]) -> None:
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    (log_dir / f"{date}.jsonl").write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture()
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "test-anima"
    d.mkdir()
    (d / "status.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "model": "claude-sonnet-4-6",
                "supervisor": "boss",
                "role": "engineer",
            }
        ),
        encoding="utf-8",
    )
    (d / "state").mkdir(parents=True)
    (d / "activity_log").mkdir()
    return d


@pytest.fixture()
def disabled_anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "disabled-anima"
    d.mkdir()
    (d / "status.json").write_text(
        json.dumps({"enabled": False, "model": "gpt-4.1", "role": "ops"}),
        encoding="utf-8",
    )
    (d / "state").mkdir(parents=True)
    (d / "activity_log").mkdir()
    return d


# ── AnimaAuditEntry ───────────────────────────────────────────


class TestAnimaAuditEntry:
    def test_to_dict(self):
        entry = AnimaAuditEntry(
            name="alice",
            enabled=True,
            model="claude-sonnet-4-6",
            supervisor=None,
            role="engineer",
            total_entries=100,
            type_counts={"heartbeat_start": 10, "message_sent": 5},
            messages_sent=5,
            messages_received=3,
            errors=1,
            tasks_total=10,
            tasks_pending=2,
            tasks_done=7,
            peers_sent={"bob": 3},
            peers_received={"carol": 2},
            first_activity="2026-03-07T09:00:00+09:00",
            last_activity="2026-03-07T18:00:00+09:00",
        )
        d = entry.to_dict()
        assert d["name"] == "alice"
        assert d["total_entries"] == 100
        assert d["type_counts"]["heartbeat_start"] == 10
        assert d["peers_sent"] == {"bob": 3}


class TestOrgAuditReport:
    def test_to_dict(self):
        report = OrgAuditReport(
            date="2026-03-07",
            animas=[],
            total_entries=0,
            active_anima_count=0,
            disabled_anima_count=0,
        )
        d = report.to_dict()
        assert d["date"] == "2026-03-07"
        assert d["animas"] == []
        assert d["total_entries"] == 0


# ── _load_entries_for_date ────────────────────────────────────


class TestLoadEntriesForDate:
    def test_reads_correct_date_file(self, tmp_path: Path):
        log_dir = tmp_path / "activity_log"
        log_dir.mkdir()
        entries = [
            {"ts": "2026-03-07T10:00:00+09:00", "type": "heartbeat_start"},
            {"ts": "2026-03-07T11:00:00+09:00", "type": "message_sent", "to": "bob"},
        ]
        (log_dir / "2026-03-07.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries),
            encoding="utf-8",
        )
        result = _load_entries_for_date(log_dir, "2026-03-07")
        assert len(result) == 2
        assert result[1].get("to_person") == "bob"

    def test_returns_empty_for_missing_file(self, tmp_path: Path):
        result = _load_entries_for_date(tmp_path, "2026-01-01")
        assert result == []


# ── _collect_single_anima ─────────────────────────────────────


class TestCollectSingleAnima:
    def test_reads_status_json(self, anima_dir: Path):
        result = _collect_single_anima(anima_dir, _DATE)
        assert result is not None
        assert result.name == "test-anima"
        assert result.enabled is True
        assert result.model == "claude-sonnet-4-6"
        assert result.supervisor == "boss"
        assert result.role == "engineer"
        assert result.total_entries == 0

    def test_disabled_anima(self, disabled_anima_dir: Path):
        result = _collect_single_anima(disabled_anima_dir, _DATE)
        assert result is not None
        assert result.enabled is False

    def test_counts_entry_types(self, anima_dir: Path):
        _write_activity_log(
            anima_dir,
            _DATE,
            [
                {"ts": "2026-03-07T10:00:00+09:00", "type": "heartbeat_start"},
                {"ts": "2026-03-07T10:01:00+09:00", "type": "heartbeat_start"},
                {"ts": "2026-03-07T10:02:00+09:00", "type": "message_sent", "to": "bob"},
                {"ts": "2026-03-07T10:03:00+09:00", "type": "message_received", "from": "carol"},
                {"ts": "2026-03-07T10:04:00+09:00", "type": "error", "summary": "something broke"},
            ],
        )
        result = _collect_single_anima(anima_dir, _DATE)
        assert result is not None
        assert result.total_entries == 5
        assert result.type_counts["heartbeat_start"] == 2
        assert result.messages_sent == 1
        assert result.messages_received == 1
        assert result.errors == 1
        assert result.peers_sent == {"bob": 1}
        assert result.peers_received == {"carol": 1}

    def test_handles_empty_status_json(self, tmp_path: Path):
        d = tmp_path / "no-status"
        d.mkdir()
        (d / "activity_log").mkdir()
        (d / "state").mkdir()
        (d / "status.json").write_text("{}", encoding="utf-8")
        result = _collect_single_anima(d, _DATE)
        assert result is not None
        assert result.enabled is True
        assert result.model == "unknown"

    def test_task_metrics(self, anima_dir: Path):
        mock_task = MagicMock()
        mock_task.status = "done"
        mock_pending = MagicMock()
        mock_pending.status = "pending"

        def _list_tasks_side_effect(status=None):
            if status == "done":
                return [mock_task]
            return [mock_pending]

        with patch("core.memory.task_queue.TaskQueueManager") as MockTQM:
            MockTQM.return_value.list_tasks.side_effect = _list_tasks_side_effect
            result = _collect_single_anima(anima_dir, _DATE)

        assert result is not None
        assert result.tasks_total == 2
        assert result.tasks_done == 1
        assert result.tasks_pending == 1

    def test_reads_only_target_date(self, anima_dir: Path):
        """Entries from other dates must not appear in the report."""
        _write_activity_log(
            anima_dir,
            "2026-03-06",
            [
                {"ts": "2026-03-06T10:00:00+09:00", "type": "heartbeat_start"},
            ],
        )
        _write_activity_log(
            anima_dir,
            "2026-03-07",
            [
                {"ts": "2026-03-07T10:00:00+09:00", "type": "message_sent", "to": "x"},
            ],
        )
        result = _collect_single_anima(anima_dir, "2026-03-07")
        assert result is not None
        assert result.total_entries == 1
        assert result.type_counts.get("heartbeat_start", 0) == 0
        assert result.messages_sent == 1


# ── collect_org_audit ─────────────────────────────────────────


class TestCollectOrgAudit:
    @pytest.mark.asyncio()
    async def test_empty_animas_dir(self, tmp_path: Path):
        with patch("core.audit.get_animas_dir", return_value=tmp_path / "nonexistent"):
            report = await collect_org_audit("2026-03-07")

        assert report.date == "2026-03-07"
        assert report.animas == []
        assert report.total_entries == 0

    @pytest.mark.asyncio()
    async def test_aggregates_multiple_animas(self, tmp_path: Path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        for name in ("alice", "bob"):
            d = animas_dir / name
            d.mkdir()
            (d / "status.json").write_text(
                json.dumps({"enabled": True, "model": "test"}),
                encoding="utf-8",
            )
            (d / "activity_log").mkdir()
            (d / "state").mkdir()

        def mock_collect(anima_dir, target_date):
            name = anima_dir.name
            if name == "alice":
                return AnimaAuditEntry(
                    name="alice",
                    enabled=True,
                    model="test",
                    supervisor=None,
                    role=None,
                    total_entries=100,
                    type_counts={},
                    messages_sent=10,
                    messages_received=5,
                    errors=2,
                    tasks_total=5,
                    tasks_pending=1,
                    tasks_done=3,
                    peers_sent={},
                    peers_received={},
                    first_activity=None,
                    last_activity=None,
                )
            return AnimaAuditEntry(
                name="bob",
                enabled=True,
                model="test",
                supervisor=None,
                role=None,
                total_entries=50,
                type_counts={},
                messages_sent=3,
                messages_received=2,
                errors=1,
                tasks_total=2,
                tasks_pending=0,
                tasks_done=2,
                peers_sent={},
                peers_received={},
                first_activity=None,
                last_activity=None,
            )

        with (
            patch("core.audit.get_animas_dir", return_value=animas_dir),
            patch("core.audit._collect_single_anima", side_effect=mock_collect),
        ):
            report = await collect_org_audit("2026-03-07")

        assert len(report.animas) == 2
        assert report.total_entries == 150
        assert report.total_messages == 20
        assert report.total_errors == 3
        assert report.total_tasks_done == 5
        assert report.active_anima_count == 2

    @pytest.mark.asyncio()
    async def test_handles_collection_exception(self, tmp_path: Path):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        d = animas_dir / "broken"
        d.mkdir()
        (d / "status.json").write_text("{}", encoding="utf-8")

        def mock_collect(anima_dir, target_date):
            raise RuntimeError("disk error")

        with (
            patch("core.audit.get_animas_dir", return_value=animas_dir),
            patch("core.audit._collect_single_anima", side_effect=mock_collect),
        ):
            report = await collect_org_audit("2026-03-07")

        assert report.animas == []
        assert report.total_entries == 0
