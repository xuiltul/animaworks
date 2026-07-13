from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side MemoryManager enforcement for per-Anima read-deny roots."""

import json
from pathlib import Path
from unittest.mock import patch

from core.memory.manager import MemoryManager
from core.time_utils import today_local


def _write_permissions(anima_dir: Path, denied: Path | None) -> None:
    config: dict[str, object] = {"version": 1}
    if denied is not None:
        config["file_roots_denied"] = [str(denied)]
    (anima_dir / "permissions.json").write_text(json.dumps(config), encoding="utf-8")


def _memory_manager(anima_dir: Path) -> MemoryManager:
    with patch.object(MemoryManager, "_init_delegates", autospec=True):
        return MemoryManager(anima_dir)


def test_current_state_symlink_to_denied_target_returns_safe_default(tmp_path: Path) -> None:
    anima_dir = tmp_path / "anima"
    state_dir = anima_dir / "state"
    denied_dir = tmp_path / "private"
    state_dir.mkdir(parents=True)
    denied_dir.mkdir()
    secret = denied_dir / "secret-state.md"
    secret.write_text("status: working\nTOP_SECRET", encoding="utf-8")
    (state_dir / "current_state.md").symlink_to(secret)
    _write_permissions(anima_dir, denied_dir)

    memory = _memory_manager(anima_dir)

    assert memory.read_current_state() == "status: idle"


def test_pending_migration_does_not_read_or_remove_denied_symlink(tmp_path: Path) -> None:
    anima_dir = tmp_path / "anima"
    state_dir = anima_dir / "state"
    denied_dir = tmp_path / "private"
    state_dir.mkdir(parents=True)
    denied_dir.mkdir()
    secret = denied_dir / "secret-pending.md"
    secret.write_text("TOP_SECRET_PENDING", encoding="utf-8")
    pending = state_dir / "pending.md"
    pending.symlink_to(secret)
    _write_permissions(anima_dir, denied_dir)

    _memory_manager(anima_dir)

    assert pending.is_symlink()
    assert secret.read_text(encoding="utf-8") == "TOP_SECRET_PENDING"
    assert not (state_dir / "current_state.md").exists()


def test_prompt_related_direct_reads_reject_denied_symlink_targets(tmp_path: Path) -> None:
    anima_dir = tmp_path / "anima"
    denied_dir = tmp_path / "private"
    heartbeat_dir = anima_dir / "shortterm"
    episodes_dir = anima_dir / "episodes"
    knowledge_dir = anima_dir / "knowledge"
    heartbeat_dir.mkdir(parents=True)
    episodes_dir.mkdir()
    knowledge_dir.mkdir()
    denied_dir.mkdir()

    denied_history = denied_dir / "heartbeat_history"
    denied_history.mkdir()
    (denied_history / "history.jsonl").write_text(
        '{"timestamp":"now","action":"leak","summary":"HEARTBEAT_SECRET"}\n',
        encoding="utf-8",
    )
    (heartbeat_dir / "heartbeat_history").symlink_to(denied_history, target_is_directory=True)

    denied_episode = denied_dir / "episode.md"
    denied_episode.write_text("EPISODE_SECRET", encoding="utf-8")
    (episodes_dir / f"{today_local().isoformat()}.md").symlink_to(denied_episode)

    denied_knowledge = denied_dir / "knowledge.md"
    denied_knowledge.write_text("---\ntopic: secret\n---\n\nKNOWLEDGE_SECRET", encoding="utf-8")
    knowledge_link = knowledge_dir / "linked.md"
    knowledge_link.symlink_to(denied_knowledge)
    _write_permissions(anima_dir, denied_dir)

    memory = _memory_manager(anima_dir)

    assert memory.load_recent_heartbeat_summary() == ""
    assert memory.read_recent_episodes() == ""
    assert memory.read_knowledge_content(knowledge_link) == ""
    assert memory.read_knowledge_metadata(knowledge_link) == {}
    assert memory.list_knowledge_files() == []


def test_symlink_reads_remain_compatible_without_denied_roots(tmp_path: Path) -> None:
    anima_dir = tmp_path / "anima"
    state_dir = anima_dir / "state"
    external_dir = tmp_path / "external"
    state_dir.mkdir(parents=True)
    external_dir.mkdir()
    external_state = external_dir / "state.md"
    external_state.write_text("status: working\nallowed", encoding="utf-8")
    (state_dir / "current_state.md").symlink_to(external_state)
    _write_permissions(anima_dir, None)

    memory = _memory_manager(anima_dir)

    assert memory.read_current_state() == "status: working\nallowed"
