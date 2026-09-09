"""Unit tests for delegation DM framework-level handling in Inbox processing.

Tests the separation of delegation DMs from regular inbox messages,
task state checking, and rescue pending file regeneration.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core._anima_inbox import (
    _append_episode_off_loop,
    _check_task_state,
    _extract_task_id,
    _handle_delegation_dms,
    _split_delegation_items,
)
from core.messenger import InboxItem
from core.schemas import Message

# ── Fixtures ──────────────────────────────────────────────


def _make_message(
    *,
    from_person: str = "alice",
    to_person: str = "bob",
    content: str = "hello",
    intent: str = "",
    meta: dict[str, Any] | None = None,
    source: str = "anima",
    origin_chain: list[str] | None = None,
) -> Message:
    return Message(
        from_person=from_person,
        to_person=to_person,
        content=content,
        intent=intent,
        meta=meta or {},
        source=source,
        origin_chain=origin_chain or [],
    )


def _make_inbox_item(msg: Message, tmp_path: Path) -> InboxItem:
    path = tmp_path / f"{msg.id}.json"
    path.write_text(msg.model_dump_json(), encoding="utf-8")
    return InboxItem(msg=msg, path=path)


def _setup_anima_dir(tmp_path: Path) -> Path:
    anima_dir = tmp_path / "animas" / "bob"
    (anima_dir / "state" / "pending" / "processing").mkdir(parents=True)
    (anima_dir / "state" / "pending" / "failed").mkdir(parents=True)
    (anima_dir / "state" / "task_results").mkdir(parents=True)
    return anima_dir


# ── Async episode append ──────────────────────────────────


@pytest.mark.asyncio
async def test_append_episode_off_loop_uses_to_thread() -> None:
    memory = MagicMock()

    with patch("core._anima_inbox.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        await _append_episode_off_loop(memory, "episode", origin="anima")

    mock_to_thread.assert_awaited_once_with(memory.append_episode, "episode", origin="anima")


# ── _extract_task_id ──────────────────────────────────────────


class TestExtractTaskId:
    def test_extracts_from_meta(self) -> None:
        msg = _make_message(meta={"task_id": "abcdef012345"})
        assert _extract_task_id(msg) == "abcdef012345"

    def test_extracts_from_content_ja(self) -> None:
        msg = _make_message(content="[タスク委譲]\nテスト\n\n期限: 2h\nタスクID: abcdef012345")
        assert _extract_task_id(msg) == "abcdef012345"

    def test_extracts_from_content_en(self) -> None:
        msg = _make_message(content="[Task delegation]\nTest\n\nDeadline: 2h\nTask ID: abcdef012345")
        assert _extract_task_id(msg) == "abcdef012345"

    def test_meta_takes_priority_over_content(self) -> None:
        msg = _make_message(
            content="タスクID: 111111111111",
            meta={"task_id": "222222222222"},
        )
        assert _extract_task_id(msg) == "222222222222"

    def test_returns_none_when_not_found(self) -> None:
        msg = _make_message(content="no task id here")
        assert _extract_task_id(msg) is None

    def test_returns_none_for_empty_meta(self) -> None:
        msg = _make_message(meta={})
        assert _extract_task_id(msg) is None


# ── _split_delegation_items ──────────────────────────────────


class TestSplitDelegationItems:
    def test_separates_delegation_with_task_id(self, tmp_path: Path) -> None:
        delegation_msg = _make_message(
            intent="delegation",
            meta={"task_id": "aaa111bbb222"},
            content="[タスク委譲] do something",
        )
        normal_msg = _make_message(intent="report", content="progress report")
        items = [
            _make_inbox_item(delegation_msg, tmp_path),
            _make_inbox_item(normal_msg, tmp_path),
        ]
        delegation, non_delegation = _split_delegation_items(items, [delegation_msg, normal_msg])
        assert len(delegation) == 1
        assert len(non_delegation) == 1
        assert delegation[0].msg.intent == "delegation"
        assert non_delegation[0].msg.intent == "report"

    def test_delegation_without_task_id_stays_in_non_delegation(self, tmp_path: Path) -> None:
        msg = _make_message(intent="delegation", content="no task id")
        items = [_make_inbox_item(msg, tmp_path)]
        delegation, non_delegation = _split_delegation_items(items, [msg])
        assert len(delegation) == 0
        assert len(non_delegation) == 1

    def test_all_delegation_messages(self, tmp_path: Path) -> None:
        msgs = [
            _make_message(intent="delegation", meta={"task_id": "aaa111bbb222"}),
            _make_message(intent="delegation", meta={"task_id": "ccc333ddd444"}),
        ]
        items = [_make_inbox_item(m, tmp_path) for m in msgs]
        delegation, non_delegation = _split_delegation_items(items, msgs)
        assert len(delegation) == 2
        assert len(non_delegation) == 0

    def test_no_delegation_messages(self, tmp_path: Path) -> None:
        msgs = [_make_message(intent="report"), _make_message(intent="question")]
        items = [_make_inbox_item(m, tmp_path) for m in msgs]
        delegation, non_delegation = _split_delegation_items(items, msgs)
        assert len(delegation) == 0
        assert len(non_delegation) == 2


# ── _check_task_state ────────────────────────────────────────


class TestCanonicalDelegation:
    @pytest.mark.parametrize(
        "status,expected",
        [("pending", "pending"), ("in_progress", "processing"), ("done", "completed"), ("cancelled", "terminal")],
    )
    def test_status_comes_from_task_record(self, tmp_path, status, expected):
        from core.memory.task_queue import TaskQueueManager

        directory = _setup_anima_dir(tmp_path)
        queue = TaskQueueManager(directory)
        entry = queue.add_task(source="anima", original_instruction="full instruction", assignee="bob", summary="task")
        queue.update_status(entry.task_id, status)
        assert _check_task_state(directory, entry.task_id) == expected

    def test_result_file_is_not_completion_proof(self, tmp_path):
        directory = _setup_anima_dir(tmp_path)
        (directory / "state" / "task_results" / "abc123.md").write_text("partial output")
        assert _check_task_state(directory, "abc123") == "missing"

    @pytest.mark.asyncio
    async def test_missing_task_dm_is_not_guessed_or_archived(self, tmp_path):
        directory = _setup_anima_dir(tmp_path)
        mixin = SimpleNamespace(
            anima_dir=directory, name="bob", messenger=MagicMock(), memory=MagicMock(), _activity=MagicMock()
        )
        item = _make_inbox_item(_make_message(intent="delegation", meta={"task_id": "abc123def456"}), tmp_path)
        unresolved = await _handle_delegation_dms(mixin, [item])
        assert unresolved == [item]
        mixin.messenger.archive_paths.assert_called_once_with([])
        assert not list((directory / "state" / "pending").glob("*.json"))

    @pytest.mark.asyncio
    async def test_existing_task_dm_is_archived_and_recorded_without_republication(self, tmp_path):
        from core.memory.task_queue import TaskQueueManager

        directory = _setup_anima_dir(tmp_path)
        queue = TaskQueueManager(directory)
        queue.submit({"task_id": "abc123def456", "task_type": "llm", "title": "work", "description": "full work"})
        mixin = SimpleNamespace(
            anima_dir=directory, name="bob", messenger=MagicMock(), memory=MagicMock(), _activity=MagicMock()
        )
        item = _make_inbox_item(_make_message(intent="delegation", meta={"task_id": "abc123def456"}), tmp_path)
        assert await _handle_delegation_dms(mixin, [item]) == []
        mixin.messenger.archive_paths.assert_called_once_with([item])
        mixin.memory.append_episode.assert_called_once()
        assert len(queue.store.pending("bob")) == 1


class TestMessageMeta:
    def test_meta_serialization(self) -> None:
        msg = _make_message(meta={"task_id": "abc123def456", "extra": "data"})
        data = json.loads(msg.model_dump_json())
        assert data["meta"] == {"task_id": "abc123def456", "extra": "data"}

    def test_meta_deserialization(self) -> None:
        raw = {
            "from_person": "alice",
            "to_person": "bob",
            "content": "test",
            "meta": {"task_id": "abc123def456"},
        }
        msg = Message(**raw)
        assert msg.meta == {"task_id": "abc123def456"}

    def test_meta_default_empty(self) -> None:
        msg = _make_message()
        assert msg.meta == {}

    def test_legacy_message_without_meta(self) -> None:
        raw = {
            "from_person": "alice",
            "to_person": "bob",
            "content": "test",
        }
        msg = Message(**raw)
        assert msg.meta == {}
