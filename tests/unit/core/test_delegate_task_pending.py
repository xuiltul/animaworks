"""Unit tests for delegate_task writing to state/pending/ for immediate execution.

Covers:
- handler_org._handle_delegate_task writes pending JSON to subordinate's state/pending/
- Pending JSON schema matches PendingTaskExecutor expectations
- PendingTaskExecutor skips cancelled tasks
- _select_subordinate explicit-only behavior
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.memory.task_queue import TaskQueueManager

# ── Fixtures ─────────────────────────────────────────────────


def _make_config(animas: dict[str, dict]):
    """Create a minimal config with anima supervisor structure."""
    from core.config.models import AnimaModelConfig, AnimaWorksConfig

    cfg = AnimaWorksConfig()
    cfg.animas = {name: AnimaModelConfig(**data) for name, data in animas.items()}
    return cfg


@pytest.fixture()
def animas_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas"
    d.mkdir()
    return d


@pytest.fixture()
def handler_with_sub(animas_dir: Path):
    """Create a ToolHandler for 'boss' with subordinate 'alice'."""
    from core.tooling.handler import ToolHandler

    boss_dir = animas_dir / "boss"
    (boss_dir / "state").mkdir(parents=True)

    alice_dir = animas_dir / "alice"
    (alice_dir / "state").mkdir(parents=True)
    (boss_dir / "status.json").write_text("{}", encoding="utf-8")
    (alice_dir / "status.json").write_text("{}", encoding="utf-8")

    memory = MagicMock()
    messenger = MagicMock()
    msg_result = MagicMock()
    msg_result.id = "msg_001"
    msg_result.thread_id = "thread_001"
    messenger.send.return_value = msg_result

    handler = ToolHandler(
        anima_dir=boss_dir,
        memory=memory,
        messenger=messenger,
        tool_registry=[],
    )
    handler._org_context = SimpleNamespace(
        subordinates=["alice"],
        descendants=["alice"],
    )

    cfg = _make_config(
        {
            "boss": {"supervisor": None},
            "alice": {"supervisor": "boss"},
        }
    )

    return handler, animas_dir, cfg


# ── delegate_task writes pending file ────────────────────────


class TestDelegateTaskWritesPending:
    """Verify _handle_delegate_task creates state/pending/{task_id}.json."""

    def test_pending_file_created(self, handler_with_sub):
        handler, animas_dir, cfg = handler_with_sub
        alice_dir = animas_dir / "alice"

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.tooling.handler_delegation.build_outgoing_origin_chain", return_value=["anima"]),
        ):
            result = handler.handle(
                "delegate_task",
                {
                    "name": "alice",
                    "instruction": "Implement the login form",
                    "summary": "Login form implementation",
                },
            )

        pending_dir = alice_dir / "state" / "pending"
        pending_files = list(pending_dir.glob("*.json"))
        assert len(pending_files) == 1, f"Expected 1 pending file, got {len(pending_files)}: {result}"

        task_data = json.loads(pending_files[0].read_text(encoding="utf-8"))
        assert task_data["task_type"] == "llm"
        assert task_data["title"] == "Login form implementation"
        assert task_data["description"] == "Implement the login form"
        assert task_data["submitted_by"] == "boss"
        assert task_data["reply_to"] == "boss"
        assert task_data["source"] == "delegation"
        assert "exclusive_key" not in task_data
        assert "task_id" in task_data
        assert "submitted_at" in task_data

    def test_pending_file_task_id_matches_queue(self, handler_with_sub):
        handler, animas_dir, cfg = handler_with_sub
        alice_dir = animas_dir / "alice"

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.tooling.handler_delegation.build_outgoing_origin_chain", return_value=["anima"]),
        ):
            handler.handle(
                "delegate_task",
                {
                    "name": "alice",
                    "instruction": "Do something",
                    "summary": "Task summary",
                },
            )

        pending_files = list((alice_dir / "state" / "pending").glob("*.json"))
        pending_task = json.loads(pending_files[0].read_text(encoding="utf-8"))

        tqm = TaskQueueManager(alice_dir)
        tasks = tqm.list_tasks()
        assert len(tasks) >= 1
        assert tasks[0].task_id == pending_task["task_id"]

    def test_pending_file_has_required_fields(self, handler_with_sub):
        """All fields required by PendingTaskExecutor are present."""
        handler, animas_dir, cfg = handler_with_sub
        alice_dir = animas_dir / "alice"

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.tooling.handler_delegation.build_outgoing_origin_chain", return_value=["anima"]),
        ):
            handler.handle(
                "delegate_task",
                {
                    "name": "alice",
                    "instruction": "Test instruction",
                    "summary": "Test summary",
                },
            )

        pending_files = list((alice_dir / "state" / "pending").glob("*.json"))
        task_data = json.loads(pending_files[0].read_text(encoding="utf-8"))

        required_fields = [
            "task_type",
            "task_id",
            "title",
            "description",
            "submitted_by",
            "submitted_at",
            "reply_to",
            "source",
        ]
        for field in required_fields:
            assert field in task_data, f"Missing required field: {field}"

    def test_pending_file_includes_acceptance_criteria(self, handler_with_sub):
        """acceptance_criteria from delegate_task args is written into pending JSON."""
        handler, animas_dir, cfg = handler_with_sub
        alice_dir = animas_dir / "alice"
        criteria = ["tests pass", "PR description updated"]

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.tooling.handler_delegation.build_outgoing_origin_chain", return_value=["anima"]),
        ):
            result = handler.handle(
                "delegate_task",
                {
                    "name": "alice",
                    "instruction": "Ship the fix",
                    "summary": "Ship fix",
                    "acceptance_criteria": criteria,
                },
            )

        pending_files = list((alice_dir / "state" / "pending").glob("*.json"))
        assert len(pending_files) == 1, f"Expected 1 pending file: {result}"
        task_data = json.loads(pending_files[0].read_text(encoding="utf-8"))
        assert task_data["acceptance_criteria"] == criteria

    def test_pending_file_defaults_acceptance_criteria_empty(self, handler_with_sub):
        """Omitting acceptance_criteria leaves an empty list (backward compatible)."""
        handler, animas_dir, cfg = handler_with_sub
        alice_dir = animas_dir / "alice"

        with (
            patch("core.paths.get_animas_dir", return_value=animas_dir),
            patch("core.config.models.load_config", return_value=cfg),
            patch("core.tooling.handler_delegation.build_outgoing_origin_chain", return_value=["anima"]),
        ):
            handler.handle(
                "delegate_task",
                {
                    "name": "alice",
                    "instruction": "Do something",
                    "summary": "Task",
                },
            )

        pending_files = list((alice_dir / "state" / "pending").glob("*.json"))
        task_data = json.loads(pending_files[0].read_text(encoding="utf-8"))
        assert task_data["acceptance_criteria"] == []


# ── PendingTaskExecutor cancelled check ─────────────────────


class TestPendingExecutorCancelledCheck:
    """Verify PendingTaskExecutor skips cancelled tasks."""

    async def test_cancelled_task_returns_cancelled(self, tmp_path: Path):
        """A task marked as cancelled in task_queue.jsonl should return '(cancelled)'."""
        anima_dir = tmp_path / "anima"
        (anima_dir / "state" / "pending").mkdir(parents=True)

        tqm = TaskQueueManager(anima_dir)
        entry = tqm.add_task(
            source="anima",
            original_instruction="test task",
            assignee="anima",
            summary="test",
        )
        tqm.update_status(entry.task_id, status="cancelled")

        task_desc = {
            "task_type": "llm",
            "task_id": entry.task_id,
            "title": "test",
            "description": "test task",
            "submitted_by": "boss",
            "submitted_at": "2026-01-01T00:00:00",
            "reply_to": "boss",
            "source": "delegation",
        }
        task_path = tmp_path / "task.json"
        task_path.write_text(json.dumps(task_desc), encoding="utf-8")

        from core.supervisor.pending_executor import PendingTaskExecutor

        executor = PendingTaskExecutor.__new__(PendingTaskExecutor)
        executor._anima = MagicMock()
        executor._anima._anima_dir = anima_dir
        executor._anima_dir = anima_dir
        executor._anima_name = "anima"

        result = await executor._run_llm_task(task_desc, None)
        assert result == "(cancelled)"


# ── _select_subordinate explicit-only ───────────────────────


class TestSelectSubordinateExplicitOnly:
    """Verify _select_subordinate only delegates when name is in description."""

    def _make_config(self, animas: dict[str, dict]):
        cfg = SimpleNamespace()
        cfg.animas = {name: SimpleNamespace(**data) for name, data in animas.items()}
        return cfg

    @pytest.fixture(autouse=True)
    def _mock_sdk(self):
        sys.modules.setdefault("claude_agent_sdk", MagicMock())
        sys.modules.setdefault("claude_agent_sdk.types", MagicMock())

    @patch("core.config.models.load_config")
    @patch("core.paths.get_animas_dir")
    def test_generic_description_returns_none(self, mock_animas_dir, mock_load, tmp_path: Path):
        """Generic task description without subordinate name → self-pending."""
        from core.execution._sdk_hooks import _select_subordinate

        animas_dir = tmp_path / "animas"
        for name in ("boss", "alice", "bob"):
            d = animas_dir / name
            d.mkdir(parents=True)
            (d / "status.json").write_text(
                json.dumps({"enabled": True, "role": "engineer"}),
                encoding="utf-8",
            )

        mock_animas_dir.return_value = animas_dir
        mock_load.return_value = self._make_config(
            {
                "boss": {"supervisor": None},
                "alice": {"supervisor": "boss"},
                "bob": {"supervisor": "boss"},
            }
        )

        result = _select_subordinate(animas_dir / "boss", "fix the authentication bug")
        assert result is None

    @patch("core.config.models.load_config")
    @patch("core.paths.get_animas_dir")
    def test_explicit_name_delegates(self, mock_animas_dir, mock_load, tmp_path: Path):
        """Task description with subordinate name → delegate to that subordinate."""
        from core.execution._sdk_hooks import _select_subordinate

        animas_dir = tmp_path / "animas"
        for name in ("boss", "alice", "bob"):
            d = animas_dir / name
            d.mkdir(parents=True)
            (d / "status.json").write_text(
                json.dumps({"enabled": True}),
                encoding="utf-8",
            )

        mock_animas_dir.return_value = animas_dir
        mock_load.return_value = self._make_config(
            {
                "boss": {"supervisor": None},
                "alice": {"supervisor": "boss"},
                "bob": {"supervisor": "boss"},
            }
        )

        result = _select_subordinate(animas_dir / "boss", "Ask bob to handle the deployment")
        assert result == "bob"

    @patch("core.config.models.load_config")
    @patch("core.paths.get_animas_dir")
    def test_role_match_alone_does_not_delegate(self, mock_animas_dir, mock_load, tmp_path: Path):
        """Role matching alone does NOT trigger delegation (explicit name required)."""
        from core.execution._sdk_hooks import _select_subordinate

        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True, "role": "engineer"}),
            encoding="utf-8",
        )

        boss_dir = animas_dir / "boss"
        boss_dir.mkdir(parents=True)

        mock_animas_dir.return_value = animas_dir
        mock_load.return_value = self._make_config(
            {
                "boss": {"supervisor": None},
                "alice": {"supervisor": "boss"},
            }
        )

        result = _select_subordinate(boss_dir, "need an engineer to fix the bug")
        assert result is None
