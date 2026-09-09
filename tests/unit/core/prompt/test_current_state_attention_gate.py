from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

from core.prompt.builder import _build_group3
from core.taskboard.store import TaskBoardStore
from core.time_utils import now_local

_SS = {
    "group3_header": "# 6. Current Situation",
    "current_state_header": "## Current State",
}
_FS = {"truncated": "(earlier portion omitted)"}


def _memory(anima_dir: Path, state: str) -> MagicMock:
    memory = MagicMock()
    memory.anima_dir = anima_dir
    memory.read_current_state.return_value = state
    memory.read_resolutions.return_value = []
    return memory


def _current_state_entries(anima_dir: Path, state: str):
    memory = _memory(anima_dir, state)
    return _build_group3(
        anima_dir,
        memory,
        1.0,
        "",
        "",
        "s",
        False,
        True,
        False,
        _SS,
        _FS,
    )


def test_stale_current_state_body_is_not_injected(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "sakura"
    (anima_dir / "state").mkdir(parents=True)
    state = "status: working\nold task body"
    state_path = anima_dir / "state" / "current_state.md"
    state_path.write_text(state, encoding="utf-8")
    old = (now_local() - timedelta(hours=25)).timestamp()
    os.utime(state_path, (old, old))

    entries = _current_state_entries(anima_dir, state)

    assert all(entry.id != "current_state" for entry in entries)


def test_fresh_current_state_removes_suppressed_task_refs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "sakura"
    (anima_dir / "state").mkdir(parents=True)
    state = "status: working\nremove archived abc12345\nkeep this live note"
    (anima_dir / "state" / "current_state.md").write_text(state, encoding="utf-8")
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(anima_name="sakura", task_id="abc12345", visibility="archived")

    entries = _current_state_entries(anima_dir, state)
    current_state = next(entry.content for entry in entries if entry.id == "current_state")

    assert "abc12345" not in current_state
    assert "keep this live note" in current_state


def test_fresh_current_state_removes_shortened_suppressed_task_refs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "sakura"
    (anima_dir / "state").mkdir(parents=True)
    state = "status: working\nremove shortened abcdef12\nkeep this live note"
    (anima_dir / "state" / "current_state.md").write_text(state, encoding="utf-8")
    store = TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    store.upsert_metadata(anima_name="sakura", task_id="abcdef123456", visibility="archived")

    entries = _current_state_entries(anima_dir, state)
    current_state = next(entry.content for entry in entries if entry.id == "current_state")

    assert "abcdef12" not in current_state
    assert "keep this live note" in current_state


def test_current_state_gate_fails_open_when_taskboard_db_is_corrupt(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    anima_dir = data_dir / "animas" / "sakura"
    (anima_dir / "state").mkdir(parents=True)
    shared_dir = data_dir / "shared"
    shared_dir.mkdir()
    (shared_dir / "taskboard.sqlite3").write_text("not sqlite", encoding="utf-8")
    state = "status: working\nkeep current state"
    (anima_dir / "state" / "current_state.md").write_text(state, encoding="utf-8")

    entries = _current_state_entries(anima_dir, state)
    current_state = next(entry.content for entry in entries if entry.id == "current_state")

    assert "keep current state" in current_state
