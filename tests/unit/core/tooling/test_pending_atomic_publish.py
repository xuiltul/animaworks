from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Atomic publication tests for pending TaskExec JSON producers."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def observe_pending_publish(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Observe the atomic boundary and assert no partial final JSON is visible."""
    published: list[Path] = []
    real_replace = os.replace

    def checked_replace(src: str | Path, dst: str | Path) -> None:
        source = Path(src)
        destination = Path(dst)
        if destination.parent.name == "pending" and destination.suffix == ".json":
            # The watcher scans *.json. Before the atomic replace it must see
            # neither the final path nor the fully-written temporary file.
            assert not destination.exists()
            assert source.suffix == ".tmp"
            assert source not in destination.parent.glob("*.json")
            json.loads(source.read_text(encoding="utf-8"))
            published.append(destination)
        real_replace(src, dst)

    monkeypatch.setattr("core.memory._io.os.replace", checked_replace)
    return published


def test_submit_tasks_atomically_publishes_pending_json(
    tmp_path: Path,
    observe_pending_publish: list[Path],
) -> None:
    from core.tooling.handler_skills import SkillsToolsMixin

    handler = object.__new__(SkillsToolsMixin)
    handler._anima_dir = tmp_path / "animas" / "sakura"
    handler._anima_name = "sakura"
    handler._pending_executor_wake = None

    result = handler._handle_submit_tasks(
        {
            "batch_id": "batch-atomic",
            "tasks": [
                {
                    "task_id": "task-atomic",
                    "title": "Atomic task",
                    "description": "Publish this task atomically",
                }
            ],
        }
    )

    assert json.loads(result)["status"] == "submitted"
    target = handler._anima_dir / "state" / "pending" / "task-atomic.json"
    assert observe_pending_publish == [target]
    assert json.loads(target.read_text(encoding="utf-8"))["task_id"] == "task-atomic"
    assert list(target.parent.glob("*.tmp")) == []


def test_delegate_task_atomically_publishes_pending_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observe_pending_publish: list[Path],
) -> None:
    from core.tooling.handler_delegation import DelegationMixin

    animas_dir = tmp_path / "animas"
    boss_dir = animas_dir / "boss"
    alice_dir = animas_dir / "alice"
    (boss_dir / "state").mkdir(parents=True)
    (alice_dir / "state").mkdir(parents=True)
    (boss_dir / "status.json").write_text("{}", encoding="utf-8")
    (alice_dir / "status.json").write_text("{}", encoding="utf-8")

    handler = object.__new__(DelegationMixin)
    handler._anima_dir = boss_dir
    handler._anima_name = "boss"
    handler._activity = MagicMock()
    handler._messenger = None
    handler._session_origin = "anima"
    handler._session_origin_chain = []
    handler._check_subordinate = lambda _name: None

    monkeypatch.setattr("core.paths.get_animas_dir", lambda: animas_dir)
    monkeypatch.setattr(
        "core.tooling.handler_delegation._record_taskboard_delegation",
        lambda **_kwargs: None,
    )

    handler._handle_delegate_task(
        {
            "name": "alice",
            "instruction": "Implement atomic publishing",
            "summary": "Atomic delegation",
            "deadline": "1d",
        }
    )

    assert len(observe_pending_publish) == 1
    target = observe_pending_publish[0]
    assert target.parent == alice_dir / "state" / "pending"
    assert json.loads(target.read_text(encoding="utf-8"))["source"] == "delegation"
    assert list(target.parent.glob("*.tmp")) == []
