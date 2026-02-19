from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from core.memory.dedup import MessageDeduplicator


@dataclass
class FakeMessage:
    """Minimal message object for testing."""
    from_person: str
    content: str
    type: str = "message"


@pytest.fixture
def dedup(tmp_path):
    anima_dir = tmp_path / "animas" / "test"
    (anima_dir / "state").mkdir(parents=True)
    return MessageDeduplicator(anima_dir)


class TestIsResolvedTopic:
    def test_empty_resolutions(self, dedup):
        assert dedup.is_resolved_topic("IAM脆弱性", []) is False

    def test_empty_message(self, dedup):
        assert dedup.is_resolved_topic("", [{"issue": "IAM脆弱性レポート"}]) is False

    def test_matching_topic(self, dedup):
        resolutions = [{"issue": "IAM 脆弱性 レポート resolved"}]
        assert dedup.is_resolved_topic("IAM脆弱性について報告します", resolutions) is True

    def test_non_matching_topic(self, dedup):
        resolutions = [{"issue": "IAM 脆弱性 レポート"}]
        assert dedup.is_resolved_topic("新しいデプロイの話", resolutions) is False

    def test_case_insensitive(self, dedup):
        resolutions = [{"issue": "AWS IAM Policy Review"}]
        assert dedup.is_resolved_topic("aws iam policy について", resolutions) is True


class TestConsolidateMessages:
    def test_below_threshold_no_change(self, dedup):
        msgs = [FakeMessage("alice", "hello"), FakeMessage("bob", "hi")]
        result, suppressed = dedup.consolidate_messages(msgs)
        assert len(result) == 2
        assert len(suppressed) == 0

    def test_consolidates_3_from_same_sender(self, dedup):
        msgs = [
            FakeMessage("alice", "msg1"),
            FakeMessage("alice", "msg2"),
            FakeMessage("alice", "msg3"),
        ]
        result, suppressed = dedup.consolidate_messages(msgs)
        assert len(result) == 1
        assert len(suppressed) == 2
        assert "3件のメッセージを統合" in result[0].content

    def test_mixed_senders(self, dedup):
        msgs = [
            FakeMessage("alice", "a1"),
            FakeMessage("alice", "a2"),
            FakeMessage("alice", "a3"),
            FakeMessage("bob", "b1"),
        ]
        result, suppressed = dedup.consolidate_messages(msgs)
        assert len(result) == 2  # 1 consolidated alice + 1 bob
        assert len(suppressed) == 2


class TestConsolidateDoesNotMutateInput:
    def test_original_message_content_preserved(self, dedup):
        """Consolidation should not mutate the original message objects."""
        msgs = [
            FakeMessage("alice", "original_1"),
            FakeMessage("alice", "original_2"),
            FakeMessage("alice", "original_3"),
        ]
        # Keep references to originals
        originals = [m.content for m in msgs]
        result, _ = dedup.consolidate_messages(msgs)
        # Original messages should be unchanged
        assert msgs[0].content == originals[0]
        assert msgs[1].content == originals[1]
        assert msgs[2].content == originals[2]
        # Consolidated result should be different from original
        assert "統合" in result[0].content


class TestApplyRateLimit:
    def test_below_threshold_no_change(self, dedup):
        msgs = [FakeMessage("alice", f"msg{i}") for i in range(4)]
        accepted, deferred = dedup.apply_rate_limit(msgs)
        assert len(accepted) == 4
        assert len(deferred) == 0

    def test_rate_limits_5_plus(self, dedup):
        msgs = [FakeMessage("alice", f"msg{i}") for i in range(6)]
        accepted, deferred = dedup.apply_rate_limit(msgs)
        assert len(accepted) == 3  # First 3 accepted
        assert len(deferred) == 3  # Rest deferred

    def test_deferred_saved_to_file(self, dedup):
        msgs = [FakeMessage("alice", f"msg{i}") for i in range(6)]
        dedup.apply_rate_limit(msgs)
        assert dedup._deferred_path.exists()

    def test_mixed_senders_rate_limit(self, dedup):
        msgs = (
            [FakeMessage("alice", f"a{i}") for i in range(6)]
            + [FakeMessage("bob", "b1")]
        )
        accepted, deferred = dedup.apply_rate_limit(msgs)
        # alice: 3 accepted, 3 deferred; bob: 1 accepted
        assert len(accepted) == 4
        assert len(deferred) == 3


class TestLoadDeferred:
    def test_no_file(self, dedup):
        assert dedup.load_deferred() == []

    def test_loads_and_deletes(self, dedup):
        msgs = [FakeMessage("alice", f"msg{i}") for i in range(6)]
        dedup.apply_rate_limit(msgs)
        deferred = dedup.load_deferred()
        assert len(deferred) == 3
        assert not dedup._deferred_path.exists()  # Deleted after load


class TestArchiveSuppressed:
    def test_archives_messages(self, dedup):
        msgs = [FakeMessage("alice", "suppressed msg")]
        dedup.archive_suppressed(msgs)
        assert dedup._suppressed_path.exists()
        content = dedup._suppressed_path.read_text()
        assert "suppressed msg" in content
        assert "dedup_suppressed" in content

    def test_empty_list_no_file(self, dedup):
        dedup.archive_suppressed([])
        assert not dedup._suppressed_path.exists()
