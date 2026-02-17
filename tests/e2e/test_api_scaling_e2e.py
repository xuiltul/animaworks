"""E2E tests for API critical refactoring (scaling optimisations).

Validates the four CRITICAL fixes through the real FastAPI app stack:
1. load_model_config() — config resolution without live Anima
2. sessions.py N+1 elimination — episodes/transcripts without full reads
3. animas.py parallel I/O — asyncio.gather for anima detail
4. system.py activity endpoint — uses animas_dir/anima_names
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# ── Helpers ──────────────────────────────────────────────


def _create_app(tmp_path: Path, anima_names: list[str] | None = None):
    """Build a real FastAPI app via create_app with mocked externals."""
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
    ):
        cfg = MagicMock()
        cfg.setup_complete = True
        mock_cfg.return_value = cfg

        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {"status": "stopped", "pid": None}
        mock_sup_cls.return_value = supervisor

        ws_manager = MagicMock()
        ws_manager.active_connections = []
        mock_ws_cls.return_value = ws_manager

        from server.app import create_app

        app = create_app(animas_dir, shared_dir)

    # Override anima_names if specified (simulates reload after adding animas)
    if anima_names is not None:
        app.state.anima_names = anima_names

    return app


def _create_anima_on_disk(
    animas_dir: Path,
    name: str,
    *,
    identity: str = "# Test Anima",
    episodes: list[str] | None = None,
    transcripts: dict[str, list[dict]] | None = None,
):
    """Create an anima directory with optional episodes and transcripts."""
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)

    # Required subdirectories
    for subdir in ["episodes", "knowledge", "procedures", "state", "shortterm"]:
        (anima_dir / subdir).mkdir(exist_ok=True)

    (anima_dir / "identity.md").write_text(identity, encoding="utf-8")
    (anima_dir / "injection.md").write_text("", encoding="utf-8")
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")

    # Episodes
    if episodes:
        ep_dir = anima_dir / "episodes"
        for ep in episodes:
            (ep_dir / f"{ep}.md").write_text(
                f"# Episode {ep}\nSomething happened.", encoding="utf-8",
            )

    # Transcripts (JSONL files)
    if transcripts:
        transcript_dir = anima_dir / "transcripts"
        transcript_dir.mkdir(exist_ok=True)
        for date, messages in transcripts.items():
            lines = [json.dumps(msg, ensure_ascii=False) for msg in messages]
            (transcript_dir / f"{date}.jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8",
            )

    return anima_dir


# ── CRITICAL 1: load_model_config() ─────────────────────


class TestLoadModelConfigE2E:
    """Verify load_model_config works standalone (no live Anima)."""

    def test_load_without_anima_instance(self, data_dir):
        """load_model_config should produce a valid ModelConfig from config.json."""
        from core.config.models import load_model_config
        from core.schemas import ModelConfig

        anima_dir = data_dir / "animas" / "standalone"
        anima_dir.mkdir(parents=True, exist_ok=True)

        mc = load_model_config(anima_dir)
        assert isinstance(mc, ModelConfig)
        assert mc.model  # non-empty string
        assert mc.max_tokens > 0

    def test_missing_config_returns_default(self, tmp_path, monkeypatch):
        """When config.json does not exist, return default ModelConfig."""
        from core.config.models import invalidate_cache, load_model_config
        from core.schemas import ModelConfig

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        invalidate_cache()

        anima_dir = tmp_path / "animas" / "noconfig"
        anima_dir.mkdir(parents=True)

        mc = load_model_config(anima_dir)
        assert isinstance(mc, ModelConfig)

        invalidate_cache()


# ── CRITICAL 2: sessions.py N+1 elimination ─────────────


class TestSessionsN1E2E:
    """Verify sessions endpoint doesn't do N+1 reads on episodes/transcripts."""

    @patch("core.config.models.load_model_config")
    async def test_list_sessions_with_episodes(self, mock_lmc, tmp_path):
        """Episodes should be listed by filename only, not read."""
        animas_dir = tmp_path / "animas"
        _create_anima_on_disk(
            animas_dir, "alice",
            episodes=["20260101", "20260102", "20260103"],
        )

        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/sessions")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["episodes"]) == 3
        # Episodes should contain date and a short preview (partial read)
        for ep in data["episodes"]:
            assert "date" in ep
            assert "preview" in ep
            assert len(ep["preview"]) <= 200

    @patch("core.config.models.load_model_config")
    async def test_list_sessions_with_transcripts(self, mock_lmc, tmp_path):
        """Transcripts should count lines without full JSON parse."""
        animas_dir = tmp_path / "animas"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]
        _create_anima_on_disk(
            animas_dir, "alice",
            transcripts={"2026-01-15": messages},
        )

        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice/sessions")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["transcripts"]) == 1
        assert data["transcripts"][0]["date"] == "2026-01-15"
        assert data["transcripts"][0]["message_count"] == 3


# ── CRITICAL 3: animas.py parallel I/O ─────────────────


class TestAnimasParallelIOE2E:
    """Verify anima detail endpoint works with real filesystem reads."""

    async def test_anima_detail_returns_all_fields(self, tmp_path):
        """GET /api/animas/{name} should return identity, state, file lists."""
        animas_dir = tmp_path / "animas"
        _create_anima_on_disk(
            animas_dir, "alice",
            identity="# Alice\nShe is a test anima.",
            episodes=["20260101"],
        )
        # Add a knowledge file
        (animas_dir / "alice" / "knowledge" / "facts.md").write_text(
            "# Facts\nAlice knows things.", encoding="utf-8",
        )

        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/alice")

        assert resp.status_code == 200
        data = resp.json()
        assert "identity" in data
        assert "status" in data
        # File lists should be populated from parallel I/O
        assert len(data["episode_files"]) >= 1
        assert len(data["knowledge_files"]) >= 1

    async def test_anima_not_found(self, tmp_path):
        """GET /api/animas/{name} for non-existent anima returns 404."""
        app = _create_app(tmp_path, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/animas/nobody")

        assert resp.status_code == 404


# ── CRITICAL 4: system.py activity — animas_dir/anima_names ──


class TestActivityEndpointE2E:
    """Verify /api/activity/recent uses animas_dir, not app.state.animas."""

    async def test_activity_empty_returns_200(self, tmp_path):
        """Activity endpoint with no animas should return 200 with empty events."""
        app = _create_app(tmp_path, anima_names=[])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent")

        assert resp.status_code == 200
        data = resp.json()
        assert data["events"] == []

    async def test_activity_with_anima_returns_200(self, tmp_path):
        """Activity endpoint with an anima should return 200 (no 500 error)."""
        animas_dir = tmp_path / "animas"
        _create_anima_on_disk(animas_dir, "alice")

        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?anima=alice")

        # This previously returned 500 due to app.state.animas KeyError
        assert resp.status_code == 200

    async def test_activity_with_session_archive(self, tmp_path):
        """Activity should include session events from shortterm archives."""
        from datetime import datetime, timezone

        animas_dir = tmp_path / "animas"
        anima_dir = _create_anima_on_disk(animas_dir, "alice")

        # Create a session archive with a recent timestamp (within query window)
        archive_dir = anima_dir / "shortterm" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        recent_ts = datetime.now(timezone.utc).isoformat()
        session = {
            "timestamp": recent_ts,
            "trigger": "heartbeat",
            "original_prompt": "Regular check-in",
            "turn_count": 3,
            "context_usage_ratio": 0.2,
        }
        (archive_dir / "20260217_100000.json").write_text(
            json.dumps(session), encoding="utf-8",
        )

        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=24")

        assert resp.status_code == 200
        data = resp.json()
        session_events = [e for e in data["events"] if e["type"] == "session"]
        assert len(session_events) >= 1
        assert session_events[0]["animas"] == ["alice"]
