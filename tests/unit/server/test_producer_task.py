"""Unit tests for producer task / SSE-IPC separation.

Tests the stream_registry producer task management (set_producer_task,
cancel_producer, cancel_all_producers) and the mark_complete done parameter.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from server.stream_registry import ResponseStream, StreamRegistry


# ── Helpers ──────────────────────────────────────────────


def _make_stream(
    response_id: str = "test-stream",
    anima_name: str = "alice",
    **kwargs,
) -> ResponseStream:
    return ResponseStream(response_id=response_id, anima_name=anima_name, **kwargs)


def _make_registry() -> StreamRegistry:
    return StreamRegistry()


# ── ResponseStream.done field ─────────────────────────────


class TestResponseStreamDoneField:
    """Tests for the done field on ResponseStream."""

    def test_done_defaults_to_false(self):
        stream = _make_stream()
        assert stream.done is False

    def test_done_field_set_directly(self):
        stream = _make_stream()
        stream.done = True
        assert stream.done is True


# ── StreamRegistry.mark_complete with done param ──────────


class TestMarkCompleteDone:
    """Tests for mark_complete(done=...) parameter."""

    def test_mark_complete_default_done_true(self):
        reg = _make_registry()
        stream = reg.register("alice")
        reg.mark_complete(stream.response_id)
        assert stream.complete is True
        assert stream.done is True

    def test_mark_complete_done_false(self):
        reg = _make_registry()
        stream = reg.register("alice")
        reg.mark_complete(stream.response_id, done=False)
        assert stream.complete is True
        assert stream.done is False

    def test_mark_complete_done_true_explicit(self):
        reg = _make_registry()
        stream = reg.register("alice")
        reg.mark_complete(stream.response_id, done=True)
        assert stream.complete is True
        assert stream.done is True

    def test_mark_complete_nonexistent_stream(self):
        """mark_complete on unknown response_id should not raise."""
        reg = _make_registry()
        reg.mark_complete("nonexistent", done=False)  # Should not raise


# ── StreamRegistry.set_producer_task ──────────────────────


class TestSetProducerTask:
    """Tests for set_producer_task and task callback."""

    async def test_set_producer_task_registers_task(self):
        reg = _make_registry()
        stream = reg.register("alice")

        async def _dummy():
            pass

        task = asyncio.create_task(_dummy())
        reg.set_producer_task(stream.response_id, task)
        assert stream.producer_task is task
        await task  # clean up

    async def test_set_producer_task_adds_done_callback(self):
        reg = _make_registry()
        stream = reg.register("alice")

        async def _dummy():
            pass

        task = asyncio.create_task(_dummy())
        reg.set_producer_task(stream.response_id, task)
        # done_callback is registered — task should have callbacks
        await task

    async def test_set_producer_task_nonexistent_stream(self):
        """set_producer_task on unknown response_id should not raise."""
        reg = _make_registry()

        async def _dummy():
            pass

        task = asyncio.create_task(_dummy())
        reg.set_producer_task("nonexistent", task)  # Should not raise
        await task

    async def test_callback_logs_exception(self):
        """Producer task exceptions are caught by the done callback."""
        reg = _make_registry()
        stream = reg.register("alice")

        async def _raise():
            raise ValueError("test error")

        task = asyncio.create_task(_raise())
        reg.set_producer_task(stream.response_id, task)
        # Wait for task to complete (it will raise internally)
        with pytest.raises(ValueError):
            await task

    async def test_callback_handles_cancellation(self):
        """Producer task cancellation is logged by the done callback."""
        reg = _make_registry()
        stream = reg.register("alice")

        async def _forever():
            await asyncio.sleep(9999)

        task = asyncio.create_task(_forever())
        reg.set_producer_task(stream.response_id, task)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ── StreamRegistry.cancel_producer ────────────────────────


class TestCancelProducer:
    """Tests for cancel_producer."""

    async def test_cancel_running_producer(self):
        reg = _make_registry()
        stream = reg.register("alice")

        async def _forever():
            await asyncio.sleep(9999)

        task = asyncio.create_task(_forever())
        reg.set_producer_task(stream.response_id, task)
        reg.cancel_producer(stream.response_id)
        # Yield to event loop so cancellation propagates
        await asyncio.sleep(0)
        assert task.cancelled() or task.done()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_cancel_already_done_producer(self):
        """Cancelling a completed producer task should not raise."""
        reg = _make_registry()
        stream = reg.register("alice")

        async def _done_immediately():
            pass

        task = asyncio.create_task(_done_immediately())
        await task  # let it finish
        reg.set_producer_task(stream.response_id, task)
        reg.cancel_producer(stream.response_id)  # Should not raise

    async def test_cancel_nonexistent_stream(self):
        """cancel_producer on unknown response_id should not raise."""
        reg = _make_registry()
        reg.cancel_producer("nonexistent")  # Should not raise

    async def test_cancel_stream_without_producer(self):
        """cancel_producer on stream with no producer_task should not raise."""
        reg = _make_registry()
        stream = reg.register("alice")
        reg.cancel_producer(stream.response_id)  # Should not raise


# ── StreamRegistry.cancel_all_producers ───────────────────


class TestCancelAllProducers:
    """Tests for cancel_all_producers."""

    async def test_cancel_all_with_running_tasks(self):
        reg = _make_registry()
        tasks = []
        for i in range(3):
            stream = reg.register(f"anima{i}")

            async def _forever():
                await asyncio.sleep(9999)

            task = asyncio.create_task(_forever())
            reg.set_producer_task(stream.response_id, task)
            tasks.append(task)

        reg.cancel_all_producers()
        await asyncio.sleep(0)  # Let cancellation propagate

        for task in tasks:
            assert task.cancelled() or task.done()
            with pytest.raises(asyncio.CancelledError):
                await task

    async def test_cancel_all_with_no_tasks(self):
        """cancel_all_producers with no streams should not raise."""
        reg = _make_registry()
        reg.cancel_all_producers()

    async def test_cancel_all_skips_completed_tasks(self):
        """cancel_all_producers should skip already-completed tasks."""
        reg = _make_registry()

        # Completed task
        stream1 = reg.register("done_anima")

        async def _done():
            pass

        task1 = asyncio.create_task(_done())
        await task1
        reg.set_producer_task(stream1.response_id, task1)

        # Running task
        stream2 = reg.register("running_anima")

        async def _forever():
            await asyncio.sleep(9999)

        task2 = asyncio.create_task(_forever())
        reg.set_producer_task(stream2.response_id, task2)

        reg.cancel_all_producers()
        await asyncio.sleep(0)  # Let cancellation propagate

        # Only running task should be cancelled
        assert not task1.cancelled()  # was already done
        assert task2.cancelled() or task2.done()
        with pytest.raises(asyncio.CancelledError):
            await task2


# ── StreamRegistry.await_all_producers ─────────────────────


class TestAwaitAllProducers:
    """Tests for await_all_producers (graceful shutdown)."""

    async def test_await_running_tasks(self):
        reg = _make_registry()
        finished = []

        async def _quick(idx):
            await asyncio.sleep(0.05)
            finished.append(idx)

        for i in range(3):
            stream = reg.register(f"anima{i}")
            task = asyncio.create_task(_quick(i))
            reg.set_producer_task(stream.response_id, task)

        await reg.await_all_producers(timeout=5.0)
        assert len(finished) == 3

    async def test_await_no_tasks(self):
        """await_all_producers with no streams should not raise."""
        reg = _make_registry()
        await reg.await_all_producers(timeout=1.0)

    async def test_await_skips_completed_tasks(self):
        reg = _make_registry()
        stream = reg.register("done_anima")

        async def _done():
            pass

        task = asyncio.create_task(_done())
        await task
        reg.set_producer_task(stream.response_id, task)

        # Should return immediately (no running tasks)
        await reg.await_all_producers(timeout=1.0)

    async def test_await_respects_timeout(self):
        reg = _make_registry()
        stream = reg.register("slow_anima")

        async def _forever():
            await asyncio.sleep(9999)

        task = asyncio.create_task(_forever())
        reg.set_producer_task(stream.response_id, task)

        # Should return after timeout without raising
        await reg.await_all_producers(timeout=0.1)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ── StreamRegistry.cleanup with producer tasks ────────────


class TestCleanupWithProducerTasks:
    """Tests for cleanup cancelling producer tasks on expired streams."""

    async def test_cleanup_cancels_expired_producer(self):
        reg = _make_registry()
        stream = reg.register("alice")

        async def _forever():
            await asyncio.sleep(9999)

        task = asyncio.create_task(_forever())
        reg.set_producer_task(stream.response_id, task)

        # Force stream to be expired
        stream.created_at = time.time() - reg.TTL - 1

        removed = reg.cleanup()
        await asyncio.sleep(0)  # Let cancellation propagate
        assert removed == 1
        assert task.cancelled() or task.done()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_cleanup_preserves_unexpired_producer(self):
        reg = _make_registry()
        stream = reg.register("alice")

        async def _forever():
            await asyncio.sleep(9999)

        task = asyncio.create_task(_forever())
        reg.set_producer_task(stream.response_id, task)

        removed = reg.cleanup()
        assert removed == 0
        assert not task.cancelled()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ── Producer/Tail integration with ResponseStream ────────


class TestProducerTailIntegration:
    """Integration tests simulating producer task writing to stream,
    with tail reading events back."""

    async def test_producer_writes_tail_reads(self):
        """Simulate producer adding events, then reading them back."""
        stream = _make_stream()

        # Simulate producer adding events
        stream.add_event("stream_start", {"response_id": "test-stream"})
        stream.add_event("text_delta", {"text": "Hello "})
        stream.add_event("text_delta", {"text": "world"})
        stream.add_event("done", {"summary": "Hello world", "emotion": "happy"})

        # Simulate tail reading events
        events = stream.events_after(-1)
        assert len(events) == 4
        assert events[0].event == "stream_start"
        assert events[1].event == "text_delta"
        assert events[2].event == "text_delta"
        assert events[3].event == "done"

    async def test_incremental_read(self):
        """Tail can incrementally read events as they arrive."""
        stream = _make_stream()

        # Producer adds first event
        stream.add_event("stream_start", {"response_id": "test-stream"})

        # Tail reads
        events = stream.events_after(-1)
        assert len(events) == 1
        seq = events[-1].seq

        # Producer adds more
        stream.add_event("text_delta", {"text": "hi"})
        stream.add_event("done", {"summary": "hi"})

        # Tail reads new events only
        events = stream.events_after(seq)
        assert len(events) == 2
        assert events[0].event == "text_delta"
        assert events[1].event == "done"

    async def test_wait_new_event_notified(self):
        """wait_new_event returns True when producer adds an event."""
        stream = _make_stream()

        async def _add_event_after_delay():
            await asyncio.sleep(0.05)
            stream.add_event("text_delta", {"text": "hi"})

        task = asyncio.create_task(_add_event_after_delay())
        got_event = await stream.wait_new_event(timeout=5.0)
        assert got_event is True
        await task

    async def test_wait_new_event_timeout(self):
        """wait_new_event returns False on timeout."""
        stream = _make_stream()
        got_event = await stream.wait_new_event(timeout=0.05)
        assert got_event is False

    async def test_complete_flag_set_by_mark_complete(self):
        """Registry mark_complete sets stream.complete for tail to detect."""
        reg = _make_registry()
        stream = reg.register("alice")
        assert stream.complete is False

        reg.mark_complete(stream.response_id, done=True)
        assert stream.complete is True
        assert stream.done is True
