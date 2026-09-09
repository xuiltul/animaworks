"""Guards added 2026-08-29: regex deny entries and sticky task cancellation."""

from pathlib import Path

from core.config.schemas import command_deny_matches
from core.memory.task_queue import TaskQueueManager

RM_RULE = r"re:\brm\s+(-\w*[rR]|--recursive)"


def test_regex_deny_covers_every_recursive_rm_spelling() -> None:
    for cmd in ("rm -rf x", "rm -r x", "rm -fr x", "rm -R x", "rm --recursive x", "ssh h 'rm -r ~/tmp/a'"):
        assert command_deny_matches(RM_RULE, cmd, cmd.split()[0]), cmd
    for cmd in ("rm -f x", "rm x", "rmdir x"):
        assert not command_deny_matches(RM_RULE, cmd, cmd.split()[0]), cmd


def test_plain_deny_entry_is_still_a_substring() -> None:
    assert command_deny_matches("gh pr merge", "gh pr merge 1", "gh")
    assert not command_deny_matches("gh pr merge", "gh pr view 1", "gh")


def test_invalid_regex_never_matches() -> None:
    assert not command_deny_matches("re:(", "anything", "anything")


def test_cancelled_task_cannot_be_flipped_to_done(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir()
    m = TaskQueueManager(tmp_path)
    entry = m.add_task(source="anima", original_instruction="x", assignee="a", summary="x")
    m.update_status(entry.task_id, "in_progress")
    m.update_status(entry.task_id, "cancelled")
    assert m.update_status(entry.task_id, "done") is None
    assert m.get_task_by_id(entry.task_id).status == "cancelled"
    # explicit re-queue is the only way out
    assert m.update_status(entry.task_id, "pending") is not None
