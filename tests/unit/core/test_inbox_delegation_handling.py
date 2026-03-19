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
from unittest.mock import MagicMock, patch

import pytest

from core._anima_inbox import (
    _check_task_state,
    _extract_task_id,
    _handle_delegation_dms,
    _rescue_regenerate_pending,
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


class TestCheckTaskState:

    def test_completed(self, tmp_path: Path) -> None:
        anima_dir = _setup_anima_dir(tmp_path)
        (anima_dir / "state" / "task_results" / "abc123def456.md").write_text("done")
        assert _check_task_state(anima_dir, "abc123def456") == "completed"

    def test_processing(self, tmp_path: Path) -> None:
        anima_dir = _setup_anima_dir(tmp_path)
        (anima_dir / "state" / "pending" / "processing" / "abc123def456.json").write_text("{}")
        assert _check_task_state(anima_dir, "abc123def456") == "processing"

    def test_pending(self, tmp_path: Path) -> None:
        anima_dir = _setup_anima_dir(tmp_path)
        (anima_dir / "state" / "pending" / "abc123def456.json").write_text("{}")
        assert _check_task_state(anima_dir, "abc123def456") == "pending"

    def test_terminal_done(self, tmp_path: Path) -> None:
        anima_dir = _setup_anima_dir(tmp_path)
        queue_path = anima_dir / "state" / "task_queue.jsonl"
        entry = {
            "task_id": "abc123def456",
            "ts": "2026-03-19T00:00:00",
            "source": "anima",
            "original_instruction": "test",
            "assignee": "bob",
            "status": "done",
            "summary": "done",
            "updated_at": "2026-03-19T00:00:00",
        }
        queue_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        assert _check_task_state(anima_dir, "abc123def456") == "terminal"

    def test_terminal_failed(self, tmp_path: Path) -> None:
        anima_dir = _setup_anima_dir(tmp_path)
        queue_path = anima_dir / "state" / "task_queue.jsonl"
        entry = {
            "task_id": "abc123def456",
            "ts": "2026-03-19T00:00:00",
            "source": "anima",
            "original_instruction": "test",
            "assignee": "bob",
            "status": "failed",
            "summary": "failed",
            "updated_at": "2026-03-19T00:00:00",
        }
        queue_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        assert _check_task_state(anima_dir, "abc123def456") == "terminal"

    def test_missing(self, tmp_path: Path) -> None:
        anima_dir = _setup_anima_dir(tmp_path)
        assert _check_task_state(anima_dir, "abc123def456") == "missing"

    def test_completed_takes_priority_over_pending(self, tmp_path: Path) -> None:
        anima_dir = _setup_anima_dir(tmp_path)
        (anima_dir / "state" / "task_results" / "abc123def456.md").write_text("done")
        (anima_dir / "state" / "pending" / "abc123def456.json").write_text("{}")
        assert _check_task_state(anima_dir, "abc123def456") == "completed"


# ── _rescue_regenerate_pending ──────────────────────────────


class TestRescueRegeneratePending:

    def test_creates_pending_file(self, tmp_path: Path) -> None:
        anima_dir = _setup_anima_dir(tmp_path)
        msg = _make_message(
            from_person="alice",
            content="[タスク委譲]\nCreate issue for X\n\n期限: 2h\nタスクID: abc123def456",
        )
        _rescue_regenerate_pending(anima_dir, "abc123def456", msg)

        pending_file = anima_dir / "state" / "pending" / "abc123def456.json"
        assert pending_file.exists()

        data = json.loads(pending_file.read_text(encoding="utf-8"))
        assert data["task_id"] == "abc123def456"
        assert data["task_type"] == "llm"
        assert data["source"] == "delegation_rescue"
        assert data["submitted_by"] == "alice"
        assert data["reply_to"] == "alice"

    def test_uses_task_queue_instruction_when_available(self, tmp_path: Path) -> None:
        anima_dir = _setup_anima_dir(tmp_path)
        queue_path = anima_dir / "state" / "task_queue.jsonl"
        entry = {
            "task_id": "abc123def456",
            "ts": "2026-03-19T00:00:00",
            "source": "anima",
            "original_instruction": "Original detailed instruction from queue",
            "assignee": "bob",
            "status": "pending",
            "summary": "test",
            "updated_at": "2026-03-19T00:00:00",
        }
        queue_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        msg = _make_message(content="Short DM content")
        _rescue_regenerate_pending(anima_dir, "abc123def456", msg)

        pending_file = anima_dir / "state" / "pending" / "abc123def456.json"
        data = json.loads(pending_file.read_text(encoding="utf-8"))
        assert data["description"] == "Original detailed instruction from queue"


# ── _handle_delegation_dms ──────────────────────────────────


class TestHandleDelegationDms:

    def _make_anima_mixin(self, tmp_path: Path) -> SimpleNamespace:
        anima_dir = _setup_anima_dir(tmp_path)
        shared_dir = tmp_path / "shared"
        inbox_dir = shared_dir / "inbox" / "bob"
        inbox_dir.mkdir(parents=True)
        processed_dir = inbox_dir / "processed"
        processed_dir.mkdir(parents=True)

        messenger = MagicMock()
        messenger.archive_paths = MagicMock(return_value=1)

        memory = MagicMock()
        memory.append_episode = MagicMock()

        activity = MagicMock()
        activity.log = MagicMock()

        return SimpleNamespace(
            name="bob",
            anima_dir=anima_dir,
            messenger=messenger,
            memory=memory,
            _activity=activity,
        )

    def test_archives_when_task_completed(self, tmp_path: Path) -> None:
        mixin = self._make_anima_mixin(tmp_path)
        (mixin.anima_dir / "state" / "task_results" / "abc123def456.md").write_text("done")

        msg = _make_message(intent="delegation", meta={"task_id": "abc123def456"})
        items = [_make_inbox_item(msg, tmp_path)]

        _handle_delegation_dms(mixin, items)

        mixin.messenger.archive_paths.assert_called_once_with(items)
        mixin._activity.log.assert_called_once()
        assert mixin._activity.log.call_args[1]["meta"]["delegation_state"] == "completed"

    def test_archives_when_task_pending(self, tmp_path: Path) -> None:
        mixin = self._make_anima_mixin(tmp_path)
        (mixin.anima_dir / "state" / "pending" / "abc123def456.json").write_text("{}")

        msg = _make_message(intent="delegation", meta={"task_id": "abc123def456"})
        items = [_make_inbox_item(msg, tmp_path)]

        _handle_delegation_dms(mixin, items)

        mixin.messenger.archive_paths.assert_called_once()
        assert not (mixin.anima_dir / "state" / "pending" / "abc123def456_rescue.json").exists()

    def test_rescues_when_task_missing(self, tmp_path: Path) -> None:
        mixin = self._make_anima_mixin(tmp_path)

        msg = _make_message(
            intent="delegation",
            meta={"task_id": "abc123def456"},
            content="[タスク委譲]\nDo something\n\n期限: 2h\nタスクID: abc123def456",
            from_person="alice",
        )
        items = [_make_inbox_item(msg, tmp_path)]

        _handle_delegation_dms(mixin, items)

        pending_file = mixin.anima_dir / "state" / "pending" / "abc123def456.json"
        assert pending_file.exists()
        data = json.loads(pending_file.read_text(encoding="utf-8"))
        assert data["source"] == "delegation_rescue"

        mixin.messenger.archive_paths.assert_called_once()
        assert mixin._activity.log.call_args[1]["meta"]["delegation_state"] == "missing"

    def test_records_episode(self, tmp_path: Path) -> None:
        mixin = self._make_anima_mixin(tmp_path)
        (mixin.anima_dir / "state" / "pending" / "abc123def456.json").write_text("{}")

        msg = _make_message(
            intent="delegation",
            meta={"task_id": "abc123def456"},
            from_person="alice",
        )
        items = [_make_inbox_item(msg, tmp_path)]

        _handle_delegation_dms(mixin, items)

        mixin.memory.append_episode.assert_called_once()

    def test_handles_multiple_delegation_dms(self, tmp_path: Path) -> None:
        mixin = self._make_anima_mixin(tmp_path)
        (mixin.anima_dir / "state" / "pending" / "aaa111bbb222.json").write_text("{}")

        msgs = [
            _make_message(intent="delegation", meta={"task_id": "aaa111bbb222"}),
            _make_message(intent="delegation", meta={"task_id": "ccc333ddd444"}),
        ]
        items = [_make_inbox_item(m, tmp_path) for m in msgs]

        _handle_delegation_dms(mixin, items)

        assert mixin._activity.log.call_count == 2
        rescue_file = mixin.anima_dir / "state" / "pending" / "ccc333ddd444.json"
        assert rescue_file.exists()


# ── Message.meta serialization ──────────────────────────────


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
