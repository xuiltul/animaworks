from __future__ import annotations

import pytest


class FakeTaskEntry:
    """Minimal TaskEntry-like object for testing."""

    def __init__(self, task_id: str, meta: dict | None = None):
        self.task_id = task_id
        self.meta = meta or {}
        self.summary = "test summary"
        self.original_instruction = "test instruction"


class TestRegeneratePendingJsonGuard:
    """Test the retry guards added to _regenerate_pending_json."""

    @pytest.fixture
    def handler(self, tmp_path):
        """Create a minimal SkillsToolsMixin-like object with required attrs."""
        from core.tooling.handler_skills import SkillsToolsMixin

        h = object.__new__(SkillsToolsMixin)
        h._anima_dir = tmp_path / "animas" / "test"
        h._anima_name = "test"
        h._pending_executor_wake = None
        (h._anima_dir / "state" / "pending").mkdir(parents=True)
        (h._anima_dir / "state" / "processing").mkdir(parents=True)
        return h

    def test_skip_when_pending_exists(self, handler):
        entry = FakeTaskEntry("task-001", meta={"task_desc": {"title": "t"}})
        pending = handler._anima_dir / "state" / "pending" / "task-001.json"
        pending.write_text("{}")

        handler._regenerate_pending_json(entry)

        assert pending.read_text() == "{}"

    def test_skip_when_processing_exists(self, handler):
        entry = FakeTaskEntry("task-002", meta={"task_desc": {"title": "t"}})
        processing = handler._anima_dir / "state" / "processing" / "task-002.json"
        processing.write_text("{}")

        handler._regenerate_pending_json(entry)
        assert processing.read_text() == "{}"

    def test_skip_when_retry_count_exceeded(self, handler):
        entry = FakeTaskEntry(
            "task-003",
            meta={"retry_count": 3, "task_desc": {"title": "t"}},
        )

        handler._regenerate_pending_json(entry)

        pending = handler._anima_dir / "state" / "pending" / "task-003.json"
        assert not pending.exists()

    def test_increments_retry_count(self, handler):
        entry = FakeTaskEntry(
            "task-004",
            meta={"retry_count": 1, "task_desc": {"title": "t"}},
        )

        handler._regenerate_pending_json(entry)

        assert entry.meta["retry_count"] == 2
        pending = handler._anima_dir / "state" / "pending" / "task-004.json"
        assert pending.exists()

    def test_first_retry_sets_count(self, handler):
        entry = FakeTaskEntry(
            "task-005",
            meta={"task_desc": {"title": "t"}},
        )

        handler._regenerate_pending_json(entry)

        assert entry.meta["retry_count"] == 1
        pending = handler._anima_dir / "state" / "pending" / "task-005.json"
        assert pending.exists()

    def test_max_retry_boundary(self, handler):
        entry_at_2 = FakeTaskEntry(
            "task-006",
            meta={"retry_count": 2, "task_desc": {"title": "t"}},
        )

        handler._regenerate_pending_json(entry_at_2)

        assert entry_at_2.meta["retry_count"] == 3
        pending = handler._anima_dir / "state" / "pending" / "task-006.json"
        assert pending.exists()

        # Clean up for next attempt
        pending.unlink()

        entry_at_3 = FakeTaskEntry(
            "task-006b",
            meta={"retry_count": 3, "task_desc": {"title": "t"}},
        )
        handler._regenerate_pending_json(entry_at_3)
        pending2 = handler._anima_dir / "state" / "pending" / "task-006b.json"
        assert not pending2.exists()
