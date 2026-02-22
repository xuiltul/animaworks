from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.execution.reminder — SystemReminderQueue."""

import pytest

from core.execution.reminder import SystemReminderQueue


# ── Push and Drain ────────────────────────────────────────────


class TestPushAndDrain:
    """Test basic push/drain operations."""

    def test_drain_empty_returns_none(self) -> None:
        q = SystemReminderQueue()
        assert q.drain_sync() is None

    def test_push_and_drain(self) -> None:
        q = SystemReminderQueue()
        q.push_sync("hello")
        result = q.drain_sync()
        assert result == "hello"

    def test_drain_clears_queue(self) -> None:
        q = SystemReminderQueue()
        q.push_sync("hello")
        q.drain_sync()
        assert q.drain_sync() is None

    def test_multiple_items_joined(self) -> None:
        q = SystemReminderQueue()
        q.push_sync("first")
        q.push_sync("second")
        result = q.drain_sync()
        assert result is not None
        assert "first" in result
        assert "second" in result
        assert "\n\n" in result

    def test_three_items_joined_in_order(self) -> None:
        q = SystemReminderQueue()
        q.push_sync("aaa")
        q.push_sync("bbb")
        q.push_sync("ccc")
        result = q.drain_sync()
        assert result == "aaa\n\nbbb\n\nccc"


# ── Overflow ──────────────────────────────────────────────────


class TestOverflow:
    """Test overflow behavior when queue exceeds max_size."""

    def test_max_size_enforced(self) -> None:
        q = SystemReminderQueue(max_size=3)
        q.push_sync("a")
        q.push_sync("b")
        q.push_sync("c")
        q.push_sync("d")  # should drop "a"
        result = q.drain_sync()
        assert result is not None
        assert "a" not in result
        assert "d" in result

    def test_overflow_keeps_newest(self) -> None:
        q = SystemReminderQueue(max_size=2)
        q.push_sync("old1")
        q.push_sync("old2")
        q.push_sync("new1")
        q.push_sync("new2")
        result = q.drain_sync()
        assert result is not None
        assert "old1" not in result
        assert "old2" not in result
        assert "new1" in result
        assert "new2" in result

    def test_max_size_one(self) -> None:
        q = SystemReminderQueue(max_size=1)
        q.push_sync("first")
        q.push_sync("second")
        result = q.drain_sync()
        assert result == "second"


# ── Format Reminder ───────────────────────────────────────────


class TestFormatReminder:
    """Test formatting helpers."""

    def test_format_reminder_wraps_in_tags(self) -> None:
        result = SystemReminderQueue.format_reminder("test content")
        assert result == "<system-reminder>\ntest content\n</system-reminder>"

    def test_drain_formatted_empty(self) -> None:
        q = SystemReminderQueue()
        assert q.drain_formatted() is None

    def test_drain_formatted_wraps(self) -> None:
        q = SystemReminderQueue()
        q.push_sync("test")
        result = q.drain_formatted()
        assert result is not None
        assert result.startswith("<system-reminder>")
        assert "test" in result
        assert result.endswith("</system-reminder>")

    def test_drain_formatted_clears_queue(self) -> None:
        q = SystemReminderQueue()
        q.push_sync("test")
        q.drain_formatted()
        assert q.drain_formatted() is None


# ── Async Operations ──────────────────────────────────────────


class TestAsyncOperations:
    """Test async push/drain."""

    async def test_async_push_and_drain(self) -> None:
        q = SystemReminderQueue()
        await q.push("async content")
        result = await q.drain()
        assert result == "async content"

    async def test_async_drain_formatted(self) -> None:
        q = SystemReminderQueue()
        await q.push("async")
        result = await q.drain_formatted_async()
        assert result is not None
        assert "<system-reminder>" in result
        assert "async" in result

    async def test_async_drain_empty(self) -> None:
        q = SystemReminderQueue()
        assert await q.drain() is None

    async def test_async_drain_formatted_empty(self) -> None:
        q = SystemReminderQueue()
        assert await q.drain_formatted_async() is None

    async def test_async_multiple_items(self) -> None:
        q = SystemReminderQueue()
        await q.push("item1")
        await q.push("item2")
        result = await q.drain()
        assert result is not None
        assert "item1" in result
        assert "item2" in result

    async def test_async_overflow(self) -> None:
        q = SystemReminderQueue(max_size=2)
        await q.push("old")
        await q.push("mid")
        await q.push("new")
        result = await q.drain()
        assert result is not None
        assert "old" not in result
        assert "new" in result
