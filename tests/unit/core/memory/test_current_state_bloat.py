# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Tests for current_state.md session-boundary archive and reset.

Issue: 20260326_current-state-session-boundary-archive
Issue #143: Session-boundary auto-archive replaces hard-trim and cleanup instructions.

Covers:
- archive_and_reset_state: skip, archive, reset, failure handling
- _update_state_from_summary routes to task_queue.jsonl
- heartbeat prompt no longer injects cleanup instructions
- builder.py _CURRENT_STATE_MAX_CHARS still exists for prompt-side truncation
"""

from unittest.mock import MagicMock, patch

import pytest

from core.memory.conversation import (
    ConversationMemory,
    ParsedSessionSummary,
)
from core.schemas import ModelConfig
from tests.helpers.filesystem import create_anima_dir, create_test_data_dir

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    from core.config import invalidate_cache
    from core.paths import _prompt_cache

    d = create_test_data_dir(tmp_path)
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(d))
    invalidate_cache()
    _prompt_cache.clear()
    yield d
    invalidate_cache()
    _prompt_cache.clear()


@pytest.fixture
def anima_dir(data_dir):
    return create_anima_dir(data_dir, "test-bloat")


@pytest.fixture
def model_config():
    return ModelConfig(
        model="claude-sonnet-4-6",
        fallback_model="claude-sonnet-4-6",
        max_turns=5,
    )


@pytest.fixture
def conv_memory(anima_dir, model_config):
    return ConversationMemory(anima_dir, model_config)


# ── archive_and_reset_state ───────────────────────────────────


class TestArchiveAndResetState:
    """Tests for MemoryManager.archive_and_reset_state()."""

    def test_skip_when_idle(self, anima_dir):
        """No archive when current_state is just 'status: idle'."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(anima_dir)
        mm.update_state("status: idle")
        mm.archive_and_reset_state("new status")
        assert mm.read_current_state().strip() == "status: idle"

    def test_skip_when_empty(self, anima_dir):
        """No archive when current_state is empty."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(anima_dir)
        (anima_dir / "state" / "current_state.md").write_text("", encoding="utf-8")
        mm.archive_and_reset_state("new status")
        assert mm.read_current_state() == "status: idle"

    def test_archive_and_reset(self, anima_dir):
        """Normal archive: content goes to episodes, state resets."""
        from core.memory.manager import MemoryManager
        from core.time_utils import today_local

        mm = MemoryManager(anima_dir)
        mm.update_state("## Working on feature X\nProgress: 50%")
        mm.archive_and_reset_state("Implementing feature X")

        assert mm.read_current_state().strip() == "Implementing feature X"

        episode_path = anima_dir / "episodes" / f"{today_local().isoformat()}.md"
        episode_content = episode_path.read_text(encoding="utf-8")
        assert "Working notes archived" in episode_content
        assert "Working on feature X" in episode_content

    def test_reset_to_idle_when_empty_new_status(self, anima_dir):
        """Falls back to 'status: idle' when new_status is empty."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(anima_dir)
        mm.update_state("some notes")
        mm.archive_and_reset_state("")

        assert mm.read_current_state().strip() == "status: idle"

    def test_state_unchanged_on_episode_failure(self, anima_dir):
        """State is left unchanged if append_episode raises."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(anima_dir)
        original = "## Important notes\nDo not lose this"
        mm.update_state(original)

        with patch.object(mm, "append_episode", side_effect=OSError("disk full")):
            mm.archive_and_reset_state("new status")

        assert mm.read_current_state().strip() == original.strip()

    def test_default_new_status(self, anima_dir):
        """Default new_status is 'status: idle'."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(anima_dir)
        mm.update_state("some work in progress")
        mm.archive_and_reset_state()

        assert mm.read_current_state().strip() == "status: idle"


# ── _update_state_from_summary (Issue #114: task_queue routing) ────


class TestUpdateStateFromSummary:
    """Tests that _update_state_from_summary() routes to task_queue.jsonl."""

    def test_resolved_items_mark_task_done(self, conv_memory, anima_dir):
        """Resolved items update matching task_queue entries to done."""
        from core.memory.manager import MemoryManager
        from core.memory.task_queue import TaskQueueManager

        memory_mgr = MemoryManager(anima_dir)
        tqm = TaskQueueManager(anima_dir)
        tqm.add_task(
            source="anima",
            original_instruction="Fix login bug",
            assignee=anima_dir.name,
            summary="Fix login bug",
        )
        task_id = list(tqm._load_all().keys())[0]

        parsed = ParsedSessionSummary(
            title="test",
            episode_body="test body",
            resolved_items=["Fix login bug"],
            new_tasks=[],
            current_status="",
            has_state_changes=True,
        )

        conv_memory._update_state_from_summary(memory_mgr, parsed)

        task = tqm.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "done"

    def test_new_tasks_not_added_to_queue(self, conv_memory, anima_dir):
        """new_tasks from session summary are NOT registered (auto-detection disabled)."""
        from core.memory.manager import MemoryManager
        from core.memory.task_queue import TaskQueueManager

        memory_mgr = MemoryManager(anima_dir)
        tqm = TaskQueueManager(anima_dir)

        parsed = ParsedSessionSummary(
            title="test",
            episode_body="test body",
            resolved_items=[],
            new_tasks=["Implement feature X", "Review PR #42"],
            current_status="",
            has_state_changes=True,
        )

        conv_memory._update_state_from_summary(memory_mgr, parsed)

        pending = tqm.get_pending()
        assert len(pending) == 0


# ── Heartbeat prompt (cleanup instruction removed) ─────────────


class TestHeartbeatPromptNoCleanup:
    """Heartbeat prompt no longer injects cleanup instructions."""

    @pytest.fixture
    def mock_heartbeat_mixin(self, anima_dir):
        from core._anima_heartbeat import HeartbeatMixin

        mixin = MagicMock(spec=HeartbeatMixin)
        mixin.name = "test-bloat"
        mixin.anima_dir = anima_dir

        memory_mock = MagicMock()
        mixin.memory = memory_mock

        return mixin

    @pytest.mark.asyncio
    async def test_no_cleanup_even_when_large(self, mock_heartbeat_mixin):
        """No cleanup instruction even for a very large current_state."""
        from core._anima_heartbeat import HeartbeatMixin

        big_state = "x" * 10000
        mock_heartbeat_mixin.memory.read_current_state.return_value = big_state
        mock_heartbeat_mixin.memory.read_heartbeat_config.return_value = None
        mock_heartbeat_mixin._build_background_context_parts = MagicMock(return_value=[])

        with patch("core._anima_heartbeat.load_prompt", return_value="heartbeat prompt"):
            parts = await HeartbeatMixin._build_heartbeat_prompt(mock_heartbeat_mixin)

        cleanup_parts = [p for p in parts if "圧縮" in p or "cleanup" in p]
        assert len(cleanup_parts) == 0

    @pytest.mark.asyncio
    async def test_prompt_contains_heartbeat_only(self, mock_heartbeat_mixin):
        """Prompt parts contain only heartbeat header and background context."""
        from core._anima_heartbeat import HeartbeatMixin

        mock_heartbeat_mixin.memory.read_current_state.return_value = "x" * 500
        mock_heartbeat_mixin.memory.read_heartbeat_config.return_value = None
        mock_heartbeat_mixin._build_background_context_parts = MagicMock(return_value=["bg context"])

        with patch("core._anima_heartbeat.load_prompt", return_value="heartbeat prompt"):
            parts = await HeartbeatMixin._build_heartbeat_prompt(mock_heartbeat_mixin)

        assert parts == ["heartbeat prompt", "bg context"]


# ── Builder truncation (existing defense) ─────────────────────


class TestBuilderTruncation:
    """Verify builder.py's existing _CURRENT_STATE_MAX_CHARS defense."""

    def test_constant_exists(self):
        """_CURRENT_STATE_MAX_CHARS is defined and equals 3000."""
        from core.prompt.builder import _CURRENT_STATE_MAX_CHARS

        assert _CURRENT_STATE_MAX_CHARS == 3000
