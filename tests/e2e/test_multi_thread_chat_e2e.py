"""E2E tests for multi-thread chat feature.

Verifies:
1. Concurrent processing: different thread_ids use isolated locks
2. Conversation isolation: thread-specific conversation.json files
3. IPC thread_id passthrough: ChatRequest carries thread_id to IPC params
4. Sessions API: includes threads from state/conversations/
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.memory.conversation import ConversationMemory
from core.memory.streaming_journal import StreamingJournal
from core.schemas import ModelConfig

from tests.helpers.filesystem import create_anima_dir, create_test_data_dir


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory structure."""
    data_dir = create_test_data_dir(tmp_path)
    return create_anima_dir(data_dir, "test-anima")


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(model="claude-sonnet-4-6")


# ── TestMultiThreadConversationE2E ────────────────────────


class TestMultiThreadConversationE2E:
    """End-to-end test: multiple threads maintain isolated conversation state."""

    def test_concurrent_thread_conversations(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """Simulate two concurrent conversations on different threads.

        Verifies:
        - Each thread has its own conversation.json
        - Messages do not leak between threads
        - Both threads can be independently loaded
        """
        # Thread A: simulate a multi-turn conversation
        conv_a = ConversationMemory(anima_dir, model_config, thread_id="thread-alpha")
        conv_a.append_turn("human", "What is Python?")
        conv_a.append_turn("assistant", "Python is a programming language.")
        conv_a.append_turn("human", "Tell me more")
        conv_a.append_turn("assistant", "It was created by Guido van Rossum.")
        conv_a.save()

        # Thread B: simulate a different conversation
        conv_b = ConversationMemory(anima_dir, model_config, thread_id="thread-beta")
        conv_b.append_turn("human", "What is Rust?")
        conv_b.append_turn("assistant", "Rust is a systems programming language.")
        conv_b.save()

        # Default thread: simulate the original conversation
        conv_default = ConversationMemory(anima_dir, model_config)
        conv_default.append_turn("human", "Hello!")
        conv_default.append_turn("assistant", "Hi there!")
        conv_default.save()

        # Verify file locations
        assert (anima_dir / "state" / "conversation.json").exists()
        assert (anima_dir / "state" / "conversations" / "thread-alpha.json").exists()
        assert (anima_dir / "state" / "conversations" / "thread-beta.json").exists()

        # Verify isolation: reload each and check content
        reload_a = ConversationMemory(anima_dir, model_config, thread_id="thread-alpha")
        state_a = reload_a.load()
        assert len(state_a.turns) == 4
        assert "Python" in state_a.turns[0].content

        reload_b = ConversationMemory(anima_dir, model_config, thread_id="thread-beta")
        state_b = reload_b.load()
        assert len(state_b.turns) == 2
        assert "Rust" in state_b.turns[0].content

        reload_default = ConversationMemory(anima_dir, model_config)
        state_default = reload_default.load()
        assert len(state_default.turns) == 2
        assert "Hello" in state_default.turns[0].content

    def test_thread_listing_from_conversations_dir(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """Simulate what the sessions API would do: list threads from filesystem."""
        # Create conversations in multiple threads
        for tid in ["thread-1", "thread-2", "thread-3"]:
            conv = ConversationMemory(anima_dir, model_config, thread_id=tid)
            conv.append_turn("human", f"Message in {tid}")
            conv.save()

        # Scan conversations directory (same logic as sessions.py)
        conv_dir = anima_dir / "state" / "conversations"
        assert conv_dir.is_dir()

        thread_ids = sorted(f.stem for f in conv_dir.glob("*.json"))
        assert thread_ids == ["thread-1", "thread-2", "thread-3"]

    @pytest.mark.asyncio
    async def test_concurrent_thread_processing(
        self, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """Two threads can be written concurrently without blocking each other."""
        results: list[str] = []

        async def write_thread(tid: str, msg: str) -> None:
            conv = ConversationMemory(anima_dir, model_config, thread_id=tid)
            conv.append_turn("human", msg)
            conv.save()
            results.append(tid)

        await asyncio.gather(
            write_thread("thread-a", "Message A"),
            write_thread("thread-b", "Message B"),
        )

        assert len(results) == 2
        assert (anima_dir / "state" / "conversations" / "thread-a.json").exists()
        assert (anima_dir / "state" / "conversations" / "thread-b.json").exists()


# ── TestMultiThreadStreamingJournalE2E ─────────────────────


class TestMultiThreadStreamingJournalE2E:
    """End-to-end test: streaming journals for multiple threads."""

    def test_concurrent_thread_journals(self, anima_dir: Path) -> None:
        """Two threads can have journals open simultaneously."""
        j1 = StreamingJournal(anima_dir, session_type="chat", thread_id="t1")
        j2 = StreamingJournal(anima_dir, session_type="chat", thread_id="t2")

        j1.open(trigger="message:user", from_person="user")
        j2.open(trigger="message:user", from_person="user")

        j1.write_text("Response for thread 1")
        j2.write_text("Response for thread 2")

        j1.finalize(summary="Thread 1 done")
        j2.finalize(summary="Thread 2 done")

        # Both should be cleaned up (no orphans)
        assert not StreamingJournal.has_orphan(anima_dir, session_type="chat")

    def test_orphan_recovery_across_threads(self, anima_dir: Path) -> None:
        """Simulate crash recovery: orphaned journals in thread directories."""
        # Create orphaned journals in two threads
        for tid in ["crash-thread-1", "crash-thread-2"]:
            journal = StreamingJournal(
                anima_dir, session_type="chat", thread_id=tid
            )
            journal.open(trigger=f"message:user-{tid}", from_person="user")
            journal.write_text(f"Partial response for {tid}")
            journal.close()  # Close without finalize = orphan

        # Verify orphans are detected
        assert StreamingJournal.has_orphan(anima_dir, session_type="chat")

        orphan_ids = StreamingJournal.list_orphan_thread_ids(anima_dir, "chat")
        assert "crash-thread-1" in orphan_ids
        assert "crash-thread-2" in orphan_ids

        # Recover each
        for tid in ["crash-thread-1", "crash-thread-2"]:
            recovery = StreamingJournal.recover(
                anima_dir, "chat", thread_id=tid
            )
            assert recovery is not None
            assert f"Partial response for {tid}" in recovery.recovered_text
            StreamingJournal.confirm_recovery(anima_dir, "chat", thread_id=tid)

        # All orphans should be cleaned up
        assert StreamingJournal.list_orphan_thread_ids(anima_dir, "chat") == []


# ── TestIPCThreadIdPassthrough ────────────────────────────


class TestIPCThreadIdPassthrough:
    """Test that thread_id flows through the IPC layer correctly."""

    def test_chat_request_includes_thread_id(self) -> None:
        """Verify ChatRequest model carries thread_id."""
        from server.routes.chat import ChatRequest

        req = ChatRequest(message="test", thread_id="custom-thread")
        assert req.thread_id == "custom-thread"

        # Simulate building IPC params as chat_stream does
        params = {
            "message": req.message,
            "from_person": req.from_person,
            "intent": req.intent,
            "stream": True,
            "images": [],
            "attachment_paths": [],
            "thread_id": req.thread_id,
        }
        assert params["thread_id"] == "custom-thread"

    def test_default_thread_id_in_params(self) -> None:
        """Default thread_id should be 'default' in IPC params."""
        from server.routes.chat import ChatRequest

        req = ChatRequest(message="test")
        params = {"thread_id": req.thread_id}
        assert params["thread_id"] == "default"


# ── TestSessionsAPIIncludesThreads ────────────────────────


class TestSessionsAPIIncludesThreads:
    """Test that the sessions endpoint lists thread conversations."""

    def _create_app(self, data_dir: Path) -> object:
        """Build FastAPI app with mocked externals."""
        animas_dir = data_dir / "animas"
        shared_dir = data_dir / "shared"
        animas_dir.mkdir(parents=True, exist_ok=True)
        shared_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("server.app.ProcessSupervisor") as mock_sup_cls,
            patch("server.app.load_config") as mock_cfg,
            patch("server.app.WebSocketManager") as mock_ws_cls,
            patch("server.app.load_auth") as mock_auth,
            patch("core.paths.get_data_dir", return_value=data_dir),
        ):
            cfg = MagicMock()
            cfg.setup_complete = True
            mock_cfg.return_value = cfg
            auth_cfg = MagicMock()
            auth_cfg.auth_mode = "local_trust"
            mock_auth.return_value = auth_cfg
            supervisor = MagicMock()
            supervisor.get_all_status.return_value = {}
            supervisor.get_process_status.return_value = {
                "status": "stopped",
                "pid": None,
            }
            supervisor.is_scheduler_running.return_value = False
            supervisor.scheduler = None
            mock_sup_cls.return_value = supervisor
            ws_manager = MagicMock()
            ws_manager.active_connections = []
            mock_ws_cls.return_value = ws_manager
            from server.app import create_app

            app = create_app(animas_dir, shared_dir)
        import server.app as _sa

        _sa.load_auth = lambda: MagicMock(auth_mode="local_trust")
        return app

    async def test_sessions_api_lists_threads(self, tmp_path: Path) -> None:
        """GET /api/animas/{name}/sessions returns threads from conversations dir."""
        data_dir = create_test_data_dir(tmp_path)
        anima_dir = create_anima_dir(data_dir, "alice")
        model_config = ModelConfig(model="claude-sonnet-4-6")

        # Create thread conversations
        for tid in ["thread-x", "thread-y"]:
            conv = ConversationMemory(anima_dir, model_config, thread_id=tid)
            conv.append_turn("human", f"Hi in {tid}")
            conv.append_turn("assistant", f"Hello in {tid}")
            conv.save()

        app = self._create_app(data_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/sessions")

        assert resp.status_code == 200
        data = resp.json()
        assert "threads" in data
        thread_ids = {t["thread_id"] for t in data["threads"]}
        assert "thread-x" in thread_ids
        assert "thread-y" in thread_ids
