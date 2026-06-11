# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from core.delegation_recovery import (
    bounce_disabled_delegations,
    build_supervision_context,
    detect_dormant_animas,
    record_dormant_offboarding_proposals,
)
from core.time_utils import now_local


def _write_status(
    animas_dir: Path,
    name: str,
    *,
    enabled: bool = True,
    supervisor: str = "",
    last_activity: str = "",
) -> Path:
    anima_dir = animas_dir / name
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)
    status = {"enabled": enabled, "supervisor": supervisor, "role": "general"}
    if last_activity:
        status["last_activity"] = last_activity
    (anima_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (anima_dir / "identity.md").write_text(f"# {name}\n", encoding="utf-8")
    return anima_dir


def _append_task(anima_dir: Path, payload: dict) -> None:
    queue = anima_dir / "state" / "task_queue.jsonl"
    queue.parent.mkdir(parents=True, exist_ok=True)
    with queue.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_queue(anima_dir: Path) -> list[dict]:
    queue = anima_dir / "state" / "task_queue.jsonl"
    return [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]


def test_bounce_disabled_delegations_creates_delegator_task(tmp_path: Path) -> None:
    animas_dir = tmp_path / "animas"
    boss_dir = _write_status(animas_dir, "boss", enabled=True)
    worker_dir = _write_status(animas_dir, "worker", enabled=False, supervisor="boss")
    old_ts = (now_local() - timedelta(days=15)).isoformat()
    _append_task(
        worker_dir,
        {
            "task_id": "delegated-1",
            "ts": old_ts,
            "source": "anima",
            "original_instruction": "Write the report",
            "assignee": "worker",
            "status": "pending",
            "summary": "Write report",
            "deadline": None,
            "relay_chain": ["boss"],
            "updated_at": old_ts,
            "meta": {},
        },
    )

    result = bounce_disabled_delegations(animas_dir, older_than_days=14, now=now_local())

    assert result == [
        {
            "delegatee": "worker",
            "delegator": "boss",
            "delegated_task_id": "delegated-1",
            "bounce_task_id": result[0]["bounce_task_id"],
        }
    ]
    boss_records = _read_queue(boss_dir)
    assert any(record.get("meta", {}).get("kind") == "delegation_bounced" for record in boss_records)
    worker_records = _read_queue(worker_dir)
    assert any(record.get("_event") == "update" and record.get("status") == "blocked" for record in worker_records)
    assert any(record.get("meta", {}).get("bounced_back") is True for record in worker_records)

    repeated = bounce_disabled_delegations(animas_dir, older_than_days=14, now=now_local())

    assert repeated == []


def test_dormant_detection_records_offboarding_proposal(tmp_path: Path) -> None:
    animas_dir = tmp_path / "animas"
    now = now_local()
    boss_dir = _write_status(animas_dir, "boss", enabled=True)
    old_activity = (now - timedelta(days=61)).isoformat()
    _write_status(animas_dir, "idle_worker", enabled=True, supervisor="boss", last_activity=old_activity)

    detected = detect_dormant_animas(animas_dir, dormant_days=60, now=now)
    assert detected[0]["name"] == "idle_worker"

    proposals = record_dormant_offboarding_proposals(animas_dir, dormant_days=60, now=now)
    assert proposals[0]["name"] == "idle_worker"
    records = _read_queue(boss_dir)
    assert any(record.get("meta", {}).get("kind") == "dormant_anima_offboarding" for record in records)


def test_supervision_context_surfaces_disabled_and_dormant_work(tmp_path: Path) -> None:
    animas_dir = tmp_path / "animas"
    _write_status(animas_dir, "boss", enabled=True)
    worker_dir = _write_status(animas_dir, "worker", enabled=False, supervisor="boss")
    old_ts = (now_local() - timedelta(days=15)).isoformat()
    _append_task(
        worker_dir,
        {
            "task_id": "delegated-2",
            "ts": old_ts,
            "source": "anima",
            "original_instruction": "Prepare slides",
            "assignee": "worker",
            "status": "pending",
            "summary": "Prepare slides",
            "deadline": None,
            "relay_chain": ["boss"],
            "updated_at": old_ts,
            "meta": {},
        },
    )

    context = build_supervision_context("boss", animas_dir, bounce_days=14, dormant_days=60)

    assert "Subordinate supervision status" in context
    assert "delegated-2" in context
