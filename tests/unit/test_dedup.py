from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.memory.dedup import _NON_CRITICAL_LIMIT, MessageDeduplicator


@dataclass
class FakeMessage:
    """Minimal message object for testing."""

    from_person: str
    content: str
    type: str = "message"
    intent: str = ""


@pytest.fixture
def dedup(tmp_path):
    anima_dir = tmp_path / "animas" / "test"
    (anima_dir / "state").mkdir(parents=True)
    return MessageDeduplicator(anima_dir)


# ── split_critical ──────────────────────────────────────


class TestSplitCritical:
    def test_empty_list(self, dedup):
        critical, non_critical = dedup.split_critical([])
        assert critical == []
        assert non_critical == []

    def test_all_non_critical(self, dedup):
        msgs = [FakeMessage("alice", "hello"), FakeMessage("bob", "hi")]
        critical, non_critical = dedup.split_critical(msgs)
        assert critical == []
        assert len(non_critical) == 2

    def test_all_critical(self, dedup):
        msgs = [
            FakeMessage("alice", "task1", intent="delegation"),
            FakeMessage("bob", "task2", intent="delegation"),
        ]
        critical, non_critical = dedup.split_critical(msgs)
        assert len(critical) == 2
        assert non_critical == []

    def test_mixed(self, dedup):
        msgs = [
            FakeMessage("alice", "task1", intent="delegation"),
            FakeMessage("bob", "hello", intent="report"),
            FakeMessage("carol", "task2", intent="delegation"),
            FakeMessage("dave", "hi"),
        ]
        critical, non_critical = dedup.split_critical(msgs)
        assert len(critical) == 2
        assert all(m.intent == "delegation" for m in critical)
        assert len(non_critical) == 2
        assert all(m.intent != "delegation" for m in non_critical)

    def test_intent_only_no_string_match(self, dedup):
        """Content containing 'delegation' text should NOT make it critical."""
        msgs = [
            FakeMessage("alice", "[タスク委譲] do something", intent="report"),
        ]
        critical, non_critical = dedup.split_critical(msgs)
        assert len(critical) == 0
        assert len(non_critical) == 1


# ── overflow_to_files ──────────────────────────────────


class TestOverflowToFiles:
    def test_under_limit_no_overflow(self, dedup):
        msgs = [FakeMessage("alice", f"msg{i}") for i in range(_NON_CRITICAL_LIMIT)]
        kept, overflow_count = dedup.overflow_to_files(msgs)
        assert len(kept) == _NON_CRITICAL_LIMIT
        assert overflow_count == 0
        assert not dedup._overflow_dir.exists()

    def test_exact_limit_no_overflow(self, dedup):
        msgs = [FakeMessage("alice", f"msg{i}") for i in range(_NON_CRITICAL_LIMIT)]
        kept, overflow_count = dedup.overflow_to_files(msgs)
        assert len(kept) == _NON_CRITICAL_LIMIT
        assert overflow_count == 0

    def test_overflow_creates_files(self, dedup):
        total = _NON_CRITICAL_LIMIT + 5
        msgs = [FakeMessage("alice", f"msg{i}") for i in range(total)]
        kept, overflow_count = dedup.overflow_to_files(msgs)
        assert len(kept) == _NON_CRITICAL_LIMIT
        assert overflow_count == 5
        assert dedup._overflow_dir.exists()
        files = list(dedup._overflow_dir.glob("*.md"))
        assert len(files) == 5

    def test_overflow_file_content(self, dedup):
        msgs = [FakeMessage("alice", f"msg{i}") for i in range(_NON_CRITICAL_LIMIT + 1)]
        dedup.overflow_to_files(msgs)
        files = list(dedup._overflow_dir.glob("*.md"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "from: alice" in content
        assert f"msg{_NON_CRITICAL_LIMIT}" in content
        assert "---" in content

    def test_overflow_preserves_order(self, dedup):
        """First N messages are kept, later ones overflow."""
        total = _NON_CRITICAL_LIMIT + 3
        msgs = [FakeMessage("alice", f"msg{i}") for i in range(total)]
        kept, _ = dedup.overflow_to_files(msgs)
        assert [m.content for m in kept] == [f"msg{i}" for i in range(_NON_CRITICAL_LIMIT)]

    def test_overflow_different_senders(self, dedup):
        msgs = [FakeMessage(f"sender{i}", f"msg{i}") for i in range(_NON_CRITICAL_LIMIT + 3)]
        kept, overflow_count = dedup.overflow_to_files(msgs)
        assert len(kept) == _NON_CRITICAL_LIMIT
        assert overflow_count == 3
        files = list(dedup._overflow_dir.glob("*.md"))
        assert len(files) == 3

    def test_collision_avoidance(self, dedup):
        """When files with same timestamp exist, counter suffix is used."""
        dedup._overflow_dir.mkdir(parents=True, exist_ok=True)
        msgs = [FakeMessage("alice", f"msg{i}") for i in range(_NON_CRITICAL_LIMIT + 3)]
        dedup.overflow_to_files(msgs)
        files = list(dedup._overflow_dir.glob("*.md"))
        assert len(files) == 3

    def test_empty_list(self, dedup):
        kept, overflow_count = dedup.overflow_to_files([])
        assert kept == []
        assert overflow_count == 0

    def test_overflow_file_has_intent(self, dedup):
        msgs = [FakeMessage("alice", "hi", intent="report") for _ in range(_NON_CRITICAL_LIMIT + 1)]
        dedup.overflow_to_files(msgs)
        files = list(dedup._overflow_dir.glob("*.md"))
        content = files[0].read_text(encoding="utf-8")
        assert "intent: report" in content


# ── Integration: split_critical + overflow_to_files ─────


class TestIntegration:
    def test_critical_bypass_with_overflow(self, dedup):
        """Critical messages bypass overflow; non-critical respects limit."""
        critical_msgs = [FakeMessage("boss", f"delegate{i}", intent="delegation") for i in range(15)]
        non_critical_msgs = [
            FakeMessage("worker", f"report{i}", intent="report") for i in range(_NON_CRITICAL_LIMIT + 5)
        ]
        all_msgs = critical_msgs + non_critical_msgs

        critical, non_critical = dedup.split_critical(all_msgs)
        assert len(critical) == 15

        non_critical, overflow_count = dedup.overflow_to_files(non_critical)
        assert len(non_critical) == _NON_CRITICAL_LIMIT
        assert overflow_count == 5

        final = critical + non_critical
        assert len(final) == 15 + _NON_CRITICAL_LIMIT

    def test_no_messages_lost(self, dedup):
        """All messages end up either in kept or overflow files."""
        total = _NON_CRITICAL_LIMIT + 7
        msgs = [FakeMessage("alice", f"msg{i}") for i in range(total)]
        critical, non_critical = dedup.split_critical(msgs)
        kept, overflow_count = dedup.overflow_to_files(non_critical)
        assert len(critical) + len(kept) + overflow_count == total
