from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.task_queue import TaskQueueManager
from core.skills.reference_rewriter import apply_skill_pointer_rewrites


def test_skill_pointer_rewrite_updates_canonical_input_without_republishing(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas/alice"
    anima_dir.mkdir(parents=True)
    queue = TaskQueueManager(anima_dir)
    entry = queue.add_task(
        source="human",
        original_instruction="Use skills/old/SKILL.md",
        assignee="alice",
        summary="Review",
        task_id="one",
        meta={"skills": ["skills/old/SKILL.md"]},
    )
    queue.store.submit("alice", entry, {"task_id": "one", "skills": ["skills/old/SKILL.md"]})
    changes = apply_skill_pointer_rewrites(anima_dir, {"skills/old/SKILL.md": "skills/new/SKILL.md"})
    assert [c.path for c in changes] == ["task_store/one"]
    assert queue.store.get_input("alice", "one")["skills"] == ["skills/new/SKILL.md"]
    assert queue.get_task_by_id("one").original_instruction == "Use skills/old/SKILL.md"
    assert len(queue.store.pending("alice")) == 1
    assert not queue.queue_path.exists()


def test_skill_pointer_rewrite_rejects_active_attempt_before_file_writes(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas/alice"
    anima_dir.mkdir(parents=True)
    queue = TaskQueueManager(anima_dir)
    entry = queue.add_task(
        source="human", original_instruction="Review", assignee="alice", summary="Review", task_id="one"
    )
    queue.store.submit("alice", entry, {"task_id": "one", "skills": ["skills/old/SKILL.md"]})
    assert queue.store.claim("alice", "one", {})
    cron = anima_dir / "cron.md"
    cron.write_text("skills: [skills/old/SKILL.md]\n")
    with pytest.raises(RuntimeError):
        apply_skill_pointer_rewrites(anima_dir, {"skills/old/SKILL.md": "skills/new/SKILL.md"})
    assert cron.read_text() == "skills: [skills/old/SKILL.md]\n"
    assert queue.store.get_input("alice", "one")["skills"] == ["skills/old/SKILL.md"]
