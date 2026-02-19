# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for conversation history pagination API.

Validates that ``/api/animas/{name}/conversation/full`` supports
offset/limit pagination and returns correct ``raw_turns`` count.

The endpoint slices turns from the end (latest first):

    total = len(state.turns)
    end   = max(0, total - offset)
    start = max(0, end - limit)
    paginated = state.turns[start:end]

The response ``raw_turns`` field always reflects the total number of
(uncompressed) turns so the frontend can determine ``hasMore``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helpers ──────────────────────────────────────────────


def _create_app(tmp_path: Path, anima_names: list[str] | None = None):
    """Build a real FastAPI app with mocked externals."""
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
        patch("server.app.load_auth") as mock_auth,
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

    # Persist auth mock beyond the with-block for request-time middleware
    import server.app as _sa
    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth

    if anima_names is not None:
        app.state.anima_names = anima_names

    return app


def _setup_anima_with_conversation(
    tmp_path: Path, name: str, turn_count: int,
) -> None:
    """Create an anima directory with pre-populated conversation turns."""
    anima_dir = tmp_path / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text(f"# {name}", encoding="utf-8")
    state_dir = anima_dir / "state"
    state_dir.mkdir(exist_ok=True)
    (anima_dir / "transcripts").mkdir(exist_ok=True)

    turns = []
    for i in range(turn_count):
        turns.append(
            {
                "role": "human" if i % 2 == 0 else "assistant",
                "content": f"Message {i}",
                "timestamp": f"2026-02-17T10:00:{i % 60:02d}",
                "token_estimate": 10,
            }
        )

    conv_data = {
        "anima_name": name,
        "turns": turns,
        "compressed_summary": "",
        "compressed_turn_count": 0,
    }
    (state_dir / "conversation.json").write_text(
        json.dumps(conv_data, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Tests ────────────────────────────────────────────────


class TestConversationPaginationAPI:
    """Test /api/animas/{name}/conversation/full pagination."""

    async def test_default_limit_and_offset(self, tmp_path):
        """Default params return latest 50 turns."""
        _setup_anima_with_conversation(tmp_path, "test-anima", 100)
        app = _create_app(tmp_path)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/api/animas/test-anima/conversation/full"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["raw_turns"] == 100
        assert len(data["turns"]) == 50
        assert data["turns"][0]["content"] == "Message 50"
        assert data["turns"][-1]["content"] == "Message 99"

    async def test_custom_limit(self, tmp_path):
        """Custom limit returns correct number of latest turns."""
        _setup_anima_with_conversation(tmp_path, "test-anima", 100)
        app = _create_app(tmp_path)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/api/animas/test-anima/conversation/full?limit=20"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["turns"]) == 20
        assert data["turns"][0]["content"] == "Message 80"
        assert data["turns"][-1]["content"] == "Message 99"

    async def test_offset_returns_older_turns(self, tmp_path):
        """Offset skips latest turns and returns older ones."""
        _setup_anima_with_conversation(tmp_path, "test-anima", 100)
        app = _create_app(tmp_path)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/api/animas/test-anima/conversation/full?limit=20&offset=20"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["raw_turns"] == 100
        assert len(data["turns"]) == 20
        assert data["turns"][0]["content"] == "Message 60"
        assert data["turns"][-1]["content"] == "Message 79"

    async def test_pages_are_continuous(self, tmp_path):
        """Sequential pages cover all messages without gaps or overlap."""
        _setup_anima_with_conversation(tmp_path, "test-anima", 50)
        app = _create_app(tmp_path)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Page 1: latest 20
            r1 = await client.get(
                "/api/animas/test-anima/conversation/full?limit=20&offset=0"
            )
            page1 = r1.json()["turns"]

            # Page 2: next 20
            r2 = await client.get(
                "/api/animas/test-anima/conversation/full?limit=20&offset=20"
            )
            page2 = r2.json()["turns"]

            # Page 3: remaining
            r3 = await client.get(
                "/api/animas/test-anima/conversation/full?limit=20&offset=40"
            )
            page3 = r3.json()["turns"]

        # Combine in chronological order: oldest first
        all_messages = page3 + page2 + page1
        assert len(all_messages) == 50
        for i, msg in enumerate(all_messages):
            assert msg["content"] == f"Message {i}"

    async def test_offset_beyond_total(self, tmp_path):
        """Offset >= total returns empty turns list."""
        _setup_anima_with_conversation(tmp_path, "test-anima", 10)
        app = _create_app(tmp_path)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/api/animas/test-anima/conversation/full?limit=20&offset=10"
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["raw_turns"] == 10
        assert len(data["turns"]) == 0

    async def test_raw_turns_reflects_total(self, tmp_path):
        """raw_turns always reflects the actual total regardless of pagination."""
        _setup_anima_with_conversation(tmp_path, "test-anima", 75)
        app = _create_app(tmp_path)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.get(
                "/api/animas/test-anima/conversation/full?limit=20&offset=0"
            )
            r2 = await client.get(
                "/api/animas/test-anima/conversation/full?limit=20&offset=60"
            )

        assert r1.json()["raw_turns"] == 75
        assert r2.json()["raw_turns"] == 75

    async def test_small_conversation(self, tmp_path):
        """Conversation smaller than limit returns all turns."""
        _setup_anima_with_conversation(tmp_path, "test-anima", 5)
        app = _create_app(tmp_path)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/api/animas/test-anima/conversation/full?limit=20"
            )

        data = resp.json()
        assert len(data["turns"]) == 5
        assert data["raw_turns"] == 5

    async def test_nonexistent_anima_404(self, tmp_path):
        """Requesting conversation for non-existent anima returns 404."""
        app = _create_app(tmp_path)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/api/animas/nonexistent/conversation/full"
            )

        assert resp.status_code == 404
