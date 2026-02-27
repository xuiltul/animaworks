"""Unit tests for multi-thread chat feature.

Tests lock separation, eviction, conversation memory and streaming journal
thread isolation, orphan detection, and ChatRequest thread_id handling.
"""

from __future__ import annotations

import asyncio
import pytest
from pathlib import Path
from core.memory.conversation import ConversationMemory
from core.memory.streaming_journal import StreamingJournal
from core.schemas import ModelConfig


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory structure."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    shortterm_dir = tmp_path / "shortterm"
    shortterm_dir.mkdir()
    return tmp_path


@pytest.fixture
def model_config() -> ModelConfig:
    """Create a minimal ModelConfig for testing."""
    return ModelConfig(model="claude-sonnet-4-6")


class TestConversationMemoryThreadIsolation:
    """Test that conversation memory files are isolated per thread."""

    def test_default_thread_uses_legacy_path(self, anima_dir: Path, model_config: ModelConfig) -> None:
        conv = ConversationMemory(anima_dir, model_config)
        assert conv._state_path == anima_dir / "state" / "conversation.json"
        assert conv.thread_id == "default"

    def test_explicit_default_thread_uses_legacy_path(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        conv = ConversationMemory(anima_dir, model_config, thread_id="default")
        assert conv._state_path == anima_dir / "state" / "conversation.json"

    def test_custom_thread_uses_thread_path(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        conv = ConversationMemory(anima_dir, model_config, thread_id="abc123")
        assert conv._state_path == anima_dir / "state" / "conversations" / "abc123.json"

    def test_different_threads_different_files(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        conv_a = ConversationMemory(anima_dir, model_config, thread_id="thread-a")
        conv_b = ConversationMemory(anima_dir, model_config, thread_id="thread-b")
        assert conv_a._state_path != conv_b._state_path

    def test_thread_history_isolation(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """Verify that appending to thread A does not affect thread B."""
        conv_a = ConversationMemory(anima_dir, model_config, thread_id="thread-a")
        conv_b = ConversationMemory(anima_dir, model_config, thread_id="thread-b")

        conv_a.append_turn("human", "Hello from thread A")
        conv_a.save()

        conv_b.append_turn("human", "Hello from thread B")
        conv_b.save()

        # Reload and verify isolation
        reloaded_a = ConversationMemory(anima_dir, model_config, thread_id="thread-a")
        state_a = reloaded_a.load()
        assert len(state_a.turns) == 1
        assert state_a.turns[0].content == "Hello from thread A"

        reloaded_b = ConversationMemory(anima_dir, model_config, thread_id="thread-b")
        state_b = reloaded_b.load()
        assert len(state_b.turns) == 1
        assert state_b.turns[0].content == "Hello from thread B"

    def test_class_locks_per_thread(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """Verify different threads get different class locks."""
        conv_a = ConversationMemory(anima_dir, model_config, thread_id="t1")
        conv_b = ConversationMemory(anima_dir, model_config, thread_id="t2")
        assert conv_a._finalize_lock is not conv_b._finalize_lock


class TestConversationMemoryBackwardCompat:
    """Test that ConversationMemory without thread_id uses legacy conversation.json."""

    def test_no_thread_id_uses_legacy_path(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """Default thread_id uses existing conversation.json."""
        conv = ConversationMemory(anima_dir, model_config)
        assert conv._state_path == anima_dir / "state" / "conversation.json"
        assert conv.thread_id == "default"


class TestStreamingJournalThreadIsolation:
    """Test that streaming journals are isolated per thread."""

    def test_default_thread_uses_legacy_path(self, anima_dir: Path) -> None:
        journal = StreamingJournal(anima_dir, session_type="chat")
        assert (
            journal._journal_path
            == anima_dir / "shortterm" / "streaming_journal_chat.jsonl"
        )

    def test_custom_thread_uses_thread_path(self, anima_dir: Path) -> None:
        journal = StreamingJournal(
            anima_dir, session_type="chat", thread_id="abc123"
        )
        assert (
            journal._journal_path
            == anima_dir / "shortterm" / "chat" / "abc123" / "streaming_journal.jsonl"
        )

    def test_has_orphan_detects_thread_journal(self, anima_dir: Path) -> None:
        """has_orphan() should detect journals in thread subdirectories."""
        thread_dir = anima_dir / "shortterm" / "chat" / "my-thread"
        thread_dir.mkdir(parents=True)
        (thread_dir / "streaming_journal.jsonl").write_text('{"ev":"start"}')

        assert StreamingJournal.has_orphan(anima_dir, session_type="chat") is True

    def test_list_orphan_thread_ids(self, anima_dir: Path) -> None:
        """list_orphan_thread_ids() should find all orphan threads."""
        # Create default orphan
        default_path = anima_dir / "shortterm" / "streaming_journal_chat.jsonl"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        default_path.write_text('{"ev":"start"}')

        # Create thread orphan
        thread_dir = anima_dir / "shortterm" / "chat" / "thread-1"
        thread_dir.mkdir(parents=True)
        (thread_dir / "streaming_journal.jsonl").write_text('{"ev":"start"}')

        thread_ids = StreamingJournal.list_orphan_thread_ids(anima_dir, "chat")
        assert "default" in thread_ids
        assert "thread-1" in thread_ids

    def test_recover_thread_journal(self, anima_dir: Path) -> None:
        """recover() should read thread-specific journal."""
        thread_dir = anima_dir / "shortterm" / "chat" / "test-thread"
        thread_dir.mkdir(parents=True)
        journal_content = (
            '{"ev":"start","trigger":"message:user","from":"user","session_id":"",'
            '"ts":"2026-01-01T00:00:00"}\n'
        )
        journal_content += (
            '{"ev":"text","t":"Hello world","ts":"2026-01-01T00:00:01"}\n'
        )
        (thread_dir / "streaming_journal.jsonl").write_text(journal_content)

        recovery = StreamingJournal.recover(
            anima_dir, "chat", thread_id="test-thread"
        )
        assert recovery is not None
        assert recovery.recovered_text == "Hello world"
        assert recovery.trigger == "message:user"


class TestAnimaThreadLock:
    """Test DigitalAnima thread lock management."""

    def test_get_thread_lock_returns_same_for_same_id(self) -> None:
        """Same thread_id should return the same lock instance."""
        class MockAnima:
            _MAX_THREAD_LOCKS = 20
            _conversation_locks: dict[str, asyncio.Lock] = {}

            def _get_thread_lock(self, thread_id: str) -> asyncio.Lock:
                if thread_id not in self._conversation_locks:
                    if len(self._conversation_locks) >= self._MAX_THREAD_LOCKS:
                        for k in list(self._conversation_locks):
                            if not self._conversation_locks[k].locked():
                                del self._conversation_locks[k]
                                break
                    self._conversation_locks[thread_id] = asyncio.Lock()
                return self._conversation_locks[thread_id]

        anima = MockAnima()
        lock1 = anima._get_thread_lock("thread-a")
        lock2 = anima._get_thread_lock("thread-a")
        assert lock1 is lock2

    def test_get_thread_lock_returns_different_for_different_id(self) -> None:
        """Different thread_ids should return different lock instances."""
        class MockAnima:
            _MAX_THREAD_LOCKS = 20
            _conversation_locks: dict[str, asyncio.Lock] = {}

            def _get_thread_lock(self, thread_id: str) -> asyncio.Lock:
                if thread_id not in self._conversation_locks:
                    if len(self._conversation_locks) >= self._MAX_THREAD_LOCKS:
                        for k in list(self._conversation_locks):
                            if not self._conversation_locks[k].locked():
                                del self._conversation_locks[k]
                                break
                    self._conversation_locks[thread_id] = asyncio.Lock()
                return self._conversation_locks[thread_id]

        anima = MockAnima()
        lock1 = anima._get_thread_lock("thread-a")
        lock2 = anima._get_thread_lock("thread-b")
        assert lock1 is not lock2

    def test_lock_eviction_when_max_reached(self) -> None:
        """When max locks reached, idle locks should be evicted."""
        class MockAnima:
            _MAX_THREAD_LOCKS = 3
            _conversation_locks: dict[str, asyncio.Lock] = {}

            def _get_thread_lock(self, thread_id: str) -> asyncio.Lock:
                if thread_id not in self._conversation_locks:
                    if len(self._conversation_locks) >= self._MAX_THREAD_LOCKS:
                        for k in list(self._conversation_locks):
                            if not self._conversation_locks[k].locked():
                                del self._conversation_locks[k]
                                break
                    self._conversation_locks[thread_id] = asyncio.Lock()
                return self._conversation_locks[thread_id]

        anima = MockAnima()
        anima._get_thread_lock("t1")
        anima._get_thread_lock("t2")
        anima._get_thread_lock("t3")
        assert len(anima._conversation_locks) == 3

        # Adding t4 should evict one of the existing idle locks
        anima._get_thread_lock("t4")
        assert len(anima._conversation_locks) == 3
        assert "t4" in anima._conversation_locks
        # First inserted (t1) should have been evicted
        assert "t1" not in anima._conversation_locks


class TestChatRequestThreadId:
    """Test that ChatRequest model includes thread_id."""

    def test_default_thread_id(self) -> None:
        from server.routes.chat import ChatRequest

        req = ChatRequest(message="hello")
        assert req.thread_id == "default"

    def test_custom_thread_id(self) -> None:
        from server.routes.chat import ChatRequest

        req = ChatRequest(message="hello", thread_id="my-thread")
        assert req.thread_id == "my-thread"


class TestThreadIdValidation:
    """Test thread_id validation prevents path traversal."""

    def test_valid_thread_ids(self) -> None:
        from core.anima import DigitalAnima

        for tid in ["default", "abc123", "a1b2c3d4", "my-thread", "MY_THREAD", "A" * 36]:
            DigitalAnima._validate_thread_id(tid)  # should not raise

    def test_path_traversal_rejected(self) -> None:
        from core.anima import DigitalAnima

        for tid in ["../etc", "../../passwd", "/tmp/evil", "a/b", "a\\b"]:
            with pytest.raises(ValueError, match="Invalid thread_id"):
                DigitalAnima._validate_thread_id(tid)

    def test_empty_rejected(self) -> None:
        from core.anima import DigitalAnima

        with pytest.raises(ValueError, match="Invalid thread_id"):
            DigitalAnima._validate_thread_id("")

    def test_too_long_rejected(self) -> None:
        from core.anima import DigitalAnima

        with pytest.raises(ValueError, match="Invalid thread_id"):
            DigitalAnima._validate_thread_id("A" * 37)

    def test_special_chars_rejected(self) -> None:
        from core.anima import DigitalAnima

        for tid in ["a.b", "a b", "a@b", "a$b", "a;b"]:
            with pytest.raises(ValueError, match="Invalid thread_id"):
                DigitalAnima._validate_thread_id(tid)


class TestConversationViewThreadFilter:
    """Test that get_conversation_view filters by thread_id."""

    def test_filter_by_thread_id(self, anima_dir: Path) -> None:
        """Entries with different thread_ids should be separated."""
        import json
        from core.memory.activity import ActivityLogger

        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{today}.jsonl"

        entries = [
            {"ts": "2026-02-27T10:00:00+09:00", "type": "message_received", "content": "hello default", "from": "human", "meta": {"thread_id": "default"}},
            {"ts": "2026-02-27T10:00:01+09:00", "type": "response_sent", "content": "hi from default", "meta": {"thread_id": "default"}},
            {"ts": "2026-02-27T10:00:02+09:00", "type": "message_received", "content": "hello thread-a", "from": "human", "meta": {"thread_id": "thread-a"}},
            {"ts": "2026-02-27T10:00:03+09:00", "type": "response_sent", "content": "hi from thread-a", "meta": {"thread_id": "thread-a"}},
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        activity = ActivityLogger(anima_dir)

        default_view = activity.get_conversation_view(thread_id="default")
        thread_a_view = activity.get_conversation_view(thread_id="thread-a")

        default_msgs = []
        for s in default_view["sessions"]:
            default_msgs.extend(s.get("messages", []))

        thread_a_msgs = []
        for s in thread_a_view["sessions"]:
            thread_a_msgs.extend(s.get("messages", []))

        assert any("hello default" in str(m) for m in default_msgs)
        assert not any("hello thread-a" in str(m) for m in default_msgs)
        assert any("hello thread-a" in str(m) for m in thread_a_msgs)
        assert not any("hello default" in str(m) for m in thread_a_msgs)

    def test_no_thread_id_filter_returns_all(self, anima_dir: Path) -> None:
        """Without thread_id filter, all entries should be returned."""
        import json
        from core.memory.activity import ActivityLogger

        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{today}.jsonl"

        entries = [
            {"ts": "2026-02-27T10:00:00+09:00", "type": "message_received", "content": "hello default", "from": "human", "meta": {"thread_id": "default"}},
            {"ts": "2026-02-27T10:00:01+09:00", "type": "message_received", "content": "hello thread-a", "from": "human", "meta": {"thread_id": "thread-a"}},
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        activity = ActivityLogger(anima_dir)
        all_view = activity.get_conversation_view()

        all_msgs = []
        for s in all_view["sessions"]:
            all_msgs.extend(s.get("messages", []))

        assert any("hello default" in str(m) for m in all_msgs)
        assert any("hello thread-a" in str(m) for m in all_msgs)

    def test_entries_without_thread_id_treated_as_default(self, anima_dir: Path) -> None:
        """Entries without meta.thread_id should be treated as 'default'."""
        import json
        from core.memory.activity import ActivityLogger

        log_dir = anima_dir / "activity_log"
        log_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{today}.jsonl"

        entries = [
            {"ts": "2026-02-27T10:00:00+09:00", "type": "message_received", "content": "old entry no thread", "from": "human", "meta": {}},
            {"ts": "2026-02-27T10:00:01+09:00", "type": "message_received", "content": "new entry with thread", "from": "human", "meta": {"thread_id": "abc123"}},
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        activity = ActivityLogger(anima_dir)
        default_view = activity.get_conversation_view(thread_id="default")

        default_msgs = []
        for s in default_view["sessions"]:
            default_msgs.extend(s.get("messages", []))

        assert any("old entry no thread" in str(m) for m in default_msgs)
        assert not any("new entry with thread" in str(m) for m in default_msgs)
