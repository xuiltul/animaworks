# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for Dashboard replay feature."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLAY_ENGINE_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "replay-engine.js"
REPLAY_UI_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "replay-ui.js"
ORG_DASHBOARD_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "org-dashboard.js"
APP_WS_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "app-websocket.js"


# ── Helpers ────────────────────────────────────────────────────


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
        supervisor.get_process_status.return_value = {"status": "stopped", "pid": None}
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        mock_sup_cls.return_value = supervisor
        ws_manager = MagicMock()
        ws_manager.active_connections = []
        mock_ws_cls.return_value = ws_manager
        from server.app import create_app

        app = create_app(animas_dir, shared_dir)
    import server.app as _sa

    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth
    if anima_names is not None:
        app.state.anima_names = anima_names
    return app


def _setup_anima(animas_dir: Path, name: str) -> Path:
    """Create a minimal anima directory."""
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "IDENTITY.md").write_text(f"# {name}", encoding="utf-8")
    return anima_dir


def _write_activity(animas_dir: Path, name: str, entries: list[dict]) -> None:
    """Write test activity entries to {animas_dir}/{name}/activity_log/{date}.jsonl."""
    log_dir = animas_dir / name / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    by_date: dict[str, list[dict]] = {}
    for entry in entries:
        date_str = entry["ts"][:10]
        by_date.setdefault(date_str, []).append(entry)
    for date_str, date_entries in by_date.items():
        path = log_dir / f"{date_str}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for e in date_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ── Event Normalization (Python simulation) ──────────────────────


class TestReplayEventNormalization:
    """Python simulation of the replay-engine normalizeEvent logic."""

    def normalize_event(self, raw: dict) -> dict:
        """Mirror replay-engine.js normalizeEvent: animas→anima, timestamp→ts."""
        evt = dict(raw)
        if isinstance(evt.get("animas"), list) and not evt.get("anima"):
            evt["anima"] = evt["animas"][0] if evt["animas"] else None
        if evt.get("timestamp") and not evt.get("ts"):
            evt["ts"] = evt["timestamp"]
        if not evt.get("id"):
            evt["id"] = evt.get("ts") or str(len(str(id(evt))))
        return evt

    def event_time_ms(self, evt: dict) -> int:
        """Get timestamp in ms for an event."""
        ts = evt.get("ts") or evt.get("timestamp")
        if ts:
            return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
        return 0

    def test_animas_to_anima(self):
        raw = {"animas": ["alice", "bob"], "timestamp": "2026-03-14T12:00:00Z"}
        norm = self.normalize_event(raw)
        assert norm["anima"] == "alice", "animas[0] must become anima"

    def test_timestamp_to_ts(self):
        raw = {"timestamp": "2026-03-14T12:00:00Z"}
        norm = self.normalize_event(raw)
        assert norm["ts"] == "2026-03-14T12:00:00Z", "timestamp must become ts"

    def test_id_added_when_missing(self):
        raw = {"ts": "2026-03-14T12:00:00Z"}
        norm = self.normalize_event(raw)
        assert "id" in norm, "id must be added when missing"

    def test_event_time_ms_parses_iso(self):
        evt = {"ts": "2026-03-14T12:00:00Z"}
        ms = self.event_time_ms(evt)
        assert ms > 0, "eventTimeMs must parse ISO timestamp"


# ── Seek State Reconstruction (Python simulation) ───────────────


class TestReplaySeekStateReconstruction:
    """Python simulation of replay-engine seek logic: cardStreams and kpiCounts."""

    MAX_STREAM_ENTRIES = 4
    ONE_HOUR_MS = 60 * 60 * 1000

    def event_time_ms(self, evt: dict) -> int:
        ts = evt.get("ts") or evt.get("timestamp")
        if ts:
            return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
        return 0

    def event_animas(self, evt: dict) -> list[str]:
        if evt.get("anima"):
            return [evt["anima"]]
        if isinstance(evt.get("animas"), list) and evt["animas"]:
            return evt["animas"]
        return []

    def status_from_event_type(self, type_str: str) -> str:
        if not type_str:
            return "idle"
        t = str(type_str).lower()
        if t == "heartbeat_start":
            return "working"
        if t in ("heartbeat_end", "cron_executed", "response_sent", "message_sent"):
            return "idle"
        return "idle"

    def event_to_stream_entry(self, evt: dict) -> dict:
        return {
            "id": evt.get("id", "gen"),
            "type": (evt.get("type") or evt.get("name") or "tool").lower()[:20],
            "text": evt.get("summary", "activity")[:80],
            "status": "done",
            "ts": self.event_time_ms(evt) or 0,
        }

    def seek_rebuild(
        self, events: list[dict], virtual_time_ms: int
    ) -> tuple[dict[str, list], dict[str, str], dict]:
        """Simulate seek: build cardStreams, cardStatus, kpiCounts."""
        card_streams: dict[str, list] = {}
        card_status: dict[str, str] = {}
        hour_before = virtual_time_ms - self.ONE_HOUR_MS
        events_in_last_hour = 0

        for evt in events:
            ts = self.event_time_ms(evt)
            if ts > virtual_time_ms:
                break
            animas = self.event_animas(evt)
            status = self.status_from_event_type(evt.get("type") or evt.get("name"))
            for name in animas:
                if not name:
                    continue
                entries = card_streams.get(name, [])
                entries.append(self.event_to_stream_entry(evt))
                if len(entries) > self.MAX_STREAM_ENTRIES:
                    entries = entries[-self.MAX_STREAM_ENTRIES:]
                card_streams[name] = entries
                card_status[name] = status
            if ts >= hour_before:
                events_in_last_hour += 1

        kpi_counts = {"eventsInLastHour": events_in_last_hour, "activeTasks": 0}
        return card_streams, card_status, kpi_counts

    def test_seek_produces_card_streams(self):
        now = datetime.now(UTC)
        events = [
            {
                "id": "e1",
                "ts": (now - timedelta(minutes=30)).isoformat(),
                "type": "heartbeat_start",
                "anima": "alice",
                "summary": "HB",
            },
            {
                "id": "e2",
                "ts": (now - timedelta(minutes=20)).isoformat(),
                "type": "tool_use",
                "anima": "alice",
                "summary": "slack",
            },
        ]
        virtual_ms = int(now.timestamp() * 1000)
        streams, status, kpi = self.seek_rebuild(events, virtual_ms)
        assert "alice" in streams, "cardStreams must contain anima"
        assert len(streams["alice"]) == 2, "alice must have 2 stream entries"
        assert kpi["eventsInLastHour"] >= 0, "kpiCounts must have eventsInLastHour"

    def test_seek_respects_max_entries(self):
        now = datetime.now(UTC)
        events = [
            {
                "id": f"e{i}",
                "ts": (now - timedelta(minutes=30 - i)).isoformat(),
                "type": "tool_use",
                "anima": "bob",
                "summary": f"tool{i}",
            }
            for i in range(10)
        ]
        virtual_ms = int(now.timestamp() * 1000)
        streams, _, _ = self.seek_rebuild(events, virtual_ms)
        assert len(streams["bob"]) <= self.MAX_STREAM_ENTRIES, (
            "stream must be clipped to MAX_STREAM_ENTRIES"
        )


# ── Replay Narrative Smoke ──────────────────────────────────────


class TestReplayNarrativeSmoke:
    def test_workspace_replay_narrative_source_contract(self):
        assert "_buildNarrativeState" in REPLAY_ENGINE_JS.read_text(encoding="utf-8")
        assert "orgReplayNarrative" in REPLAY_UI_JS.read_text(encoding="utf-8")
        assert "onNarrativeUpdate: _handleNarrativeUpdate" in ORG_DASHBOARD_JS.read_text(
            encoding="utf-8"
        )


# ── API Data Flow ───────────────────────────────────────────────


class TestReplayAPIDataFlow:
    """API can provide events for replay: write to activity_log and fetch."""

    async def test_api_returns_events_for_replay(self, tmp_path: Path) -> None:
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        entries = [
            {
                "ts": (now - timedelta(minutes=30 - i)).isoformat(),
                "type": "heartbeat_start",
                "summary": f"HB {i}",
                "content": "",
                "anima": "alice",
            }
            for i in range(5)
        ]
        _write_activity(animas_dir, "alice", entries)
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=12&limit=5000")
        assert resp.status_code == 200, f"API must return 200, got {resp.status_code}"
        data = resp.json()
        assert "events" in data, "Response must have events array"
        assert len(data["events"]) >= 1, "Must return at least one event for replay"
        evt = data["events"][0]
        assert "ts" in evt or "timestamp" in evt, "Event must have timestamp"
        assert "anima" in evt or "animas" in evt or "type" in evt, (
            "Event must have anima/animas or type for replay"
        )

    async def test_replay_api_pages_all_events_beyond_logger_cap(self, tmp_path: Path) -> None:
        """Replay mode must bypass per-Anima 500 cap and expose all pages."""
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        entries = [
            {
                "ts": (now - timedelta(seconds=1200 - i)).isoformat(),
                "type": "message_sent",
                "summary": f"Replay event {i}",
                "content": "",
                "from": "alice",
                "to": "bob",
            }
            for i in range(1200)
        ]
        _write_activity(animas_dir, "alice", entries)
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)

        seen_ids: set[str] = set()
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.get("/api/activity/recent?hours=12&limit=500&offset=0&replay=true")
            second = await client.get("/api/activity/recent?hours=12&limit=500&offset=500&replay=true")
            third = await client.get("/api/activity/recent?hours=12&limit=500&offset=1000&replay=true")

        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 200
        first_data = first.json()
        second_data = second.json()
        third_data = third.json()

        assert first_data["total"] == 1200
        assert len(first_data["events"]) == 500
        assert first_data["has_more"] is True
        assert len(second_data["events"]) == 500
        assert second_data["has_more"] is True
        assert len(third_data["events"]) == 200
        assert third_data["has_more"] is False

        for page in (first_data, second_data, third_data):
            for event in page["events"]:
                seen_ids.add(event["id"])
        assert len(seen_ids) == 1200

    async def test_non_replay_api_still_caps_limit_to_500(self, tmp_path: Path) -> None:
        """Non-replay callers keep bounded page size even with a large limit."""
        animas_dir = tmp_path / "animas"
        _setup_anima(animas_dir, "alice")
        now = datetime.now(UTC)
        entries = [
            {
                "ts": (now - timedelta(seconds=600 - i)).isoformat(),
                "type": "tool_use",
                "summary": f"Entry {i}",
                "content": "",
            }
            for i in range(600)
        ]
        _write_activity(animas_dir, "alice", entries)
        app = _create_app(tmp_path, anima_names=["alice"])
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/activity/recent?hours=12&limit=9999")

        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 500
        assert len(data["events"]) == 500
        assert data["total"] == 500
        assert data["has_more"] is True

# ── Speed Behavior (JS source analysis) ──────────────────────────


class TestReplaySpeedBehavior:
    """Speed-dependent message line duration in showMessageLine."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_duration_scaling_at_100x(self):
        assert "replaySpeed >= 100" in self.src or "100 ? 200" in self.src, (
            "JS must have 200ms duration at 100x speed"
        )

    def test_duration_scaling_at_50x(self):
        assert "replaySpeed >= 50" in self.src, (
            "JS must have speed-dependent duration (50x → 500ms)"
        )

    def test_duration_logic_exists(self):
        assert "dur" in self.src and ("replaySpeed" in self.src or "MESSAGE_LINE" in self.src), (
            "showMessageLine must contain duration scaling logic"
        )


# ── WS Buffering Integration ────────────────────────────────────


class TestReplayWSBufferingIntegration:
    """isReplayMode import and bufferReplayEvent usage correctly wired."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = APP_WS_JS.read_text(encoding="utf-8")

    def test_isReplayMode_imported(self):
        assert "isReplayMode" in self.src, (
            "app-websocket must import isReplayMode from org-dashboard"
        )

    def test_bufferReplayEvent_used(self):
        assert "bufferReplayEvent" in self.src, (
            "app-websocket must use bufferReplayEvent when in replay mode"
        )

    def test_heartbeat_handler_buffers_during_replay(self):
        assert "anima.heartbeat" in self.src, (
            "anima.heartbeat handler must exist"
        )
        assert "isReplayMode()" in self.src and "bufferReplayEvent" in self.src, (
            "Handlers must check isReplayMode and buffer when true"
        )


# ── No Regression on Existing WS Handlers ────────────────────────


class TestReplayNoRegression:
    """Existing WS handlers still exist: heartbeat, cron, tool_activity."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = APP_WS_JS.read_text(encoding="utf-8")

    def test_heartbeat_handler_exists(self):
        assert "anima.heartbeat" in self.src, (
            "anima.heartbeat handler must still exist"
        )

    def test_cron_handler_exists(self):
        assert "anima.cron" in self.src, (
            "anima.cron handler must still exist"
        )

    def test_tool_activity_handler_exists(self):
        assert "anima.tool_activity" in self.src, (
            "anima.tool_activity handler must still exist"
        )
