# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for multi-thread chat frontend integration.

Verifies:
1. ChatRequest accepts thread_id and defaults to "default"
2. Sessions API returns threads list from conversation files
3. Frontend JS body construction matches backend Pydantic model expectations
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.memory.conversation import ConversationMemory
from core.schemas import ModelConfig

from tests.helpers.filesystem import create_anima_dir, create_test_data_dir


# ── Paths ────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAT_JS = PROJECT_ROOT / "server" / "static" / "pages" / "chat.js"


def _read(path: Path) -> str:
    """Read a file's text content."""
    return path.read_text(encoding="utf-8")


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory structure."""
    data_dir = create_test_data_dir(tmp_path)
    return create_anima_dir(data_dir, "test-anima")


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(model="claude-sonnet-4-6")


# ── TestChatRequestThreadIdPassthrough ───────────────────────


@pytest.mark.e2e
class TestChatRequestThreadIdPassthrough:
    """Verify ChatRequest Pydantic model accepts thread_id."""

    def test_chat_request_accepts_thread_id(self) -> None:
        """ChatRequest accepts thread_id parameter."""
        from server.routes.chat import ChatRequest

        req = ChatRequest(message="hello", thread_id="custom-thread")
        assert req.thread_id == "custom-thread"

    def test_chat_request_defaults_thread_id_to_default(self) -> None:
        """ChatRequest defaults thread_id to 'default' when omitted."""
        from server.routes.chat import ChatRequest

        req = ChatRequest(message="hello")
        assert req.thread_id == "default"


# ── TestSessionsAPIThreadsList ───────────────────────────────


@pytest.mark.e2e
class TestSessionsAPIThreadsList:
    """Verify sessions API lists threads from conversation files."""

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

    @pytest.mark.asyncio
    async def test_sessions_api_lists_threads(
        self, tmp_path: Path, anima_dir: Path, model_config: ModelConfig
    ) -> None:
        """GET /api/animas/{name}/sessions returns threads from conversations dir."""
        data_dir = anima_dir.parent.parent  # animas/test-anima -> data_dir
        for tid in ["thread-a", "thread-b"]:
            conv = ConversationMemory(anima_dir, model_config, thread_id=tid)
            conv.append_turn("human", f"Hi in {tid}")
            conv.append_turn("assistant", f"Hello in {tid}")
            conv.save()

        app = self._create_app(data_dir)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/test-anima/sessions")

        assert resp.status_code == 200
        data = resp.json()
        assert "threads" in data
        thread_ids = {t["thread_id"] for t in data["threads"]}
        assert "thread-a" in thread_ids
        assert "thread-b" in thread_ids


# ── TestFrontendBackendIntegration ───────────────────────────


@pytest.mark.e2e
class TestFrontendBackendIntegration:
    """Verify frontend sends thread_id in format backend can parse."""

    def test_chat_js_body_matches_chat_request_schema(self) -> None:
        """Frontend chat.js body construction matches ChatRequest expectations."""
        from server.routes.chat import ChatRequest

        js = _read(CHAT_JS)

        # Frontend must include thread_id in body JSON
        assert "thread_id" in js
        assert "thread_id: tid" in js or "thread_id: threadId" in js

        # Verify ChatRequest can parse the expected shape
        body = {
            "message": "test message",
            "from_person": "human",
            "thread_id": "default",
        }
        req = ChatRequest(**body)
        assert req.message == "test message"
        assert req.thread_id == "default"

    def test_chat_js_body_json_keys_included(self) -> None:
        """chat.js bodyObj includes message, from_person, thread_id."""
        js = _read(CHAT_JS)
        # Body object construction
        assert "message" in js
        assert "from_person" in js
        assert "thread_id" in js
        assert "JSON.stringify" in js or "bodyObj" in js or "body =" in js
