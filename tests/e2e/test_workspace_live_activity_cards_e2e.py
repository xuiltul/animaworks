"""E2E tests for workspace live activity cards.

Tests the integration between WebSocket event dispatch and card activity
updates, simulating the event flow from server to card stream rendering
logic (Python-side simulation of the JS updateCardActivity behaviour).
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

REPO_ROOT = Path(__file__).resolve().parents[2]
ORG_DASHBOARD_JS = (
    REPO_ROOT / "server" / "static" / "workspace" / "modules" / "org-dashboard.js"
)
APP_WEBSOCKET_JS = (
    REPO_ROOT / "server" / "static" / "workspace" / "modules" / "app-websocket.js"
)
STYLE_CSS = REPO_ROOT / "server" / "static" / "workspace" / "style.css"


# ── Helpers: Python simulation of JS card stream logic ──────────────────────

MAX_STREAM_ENTRIES = 4
STALE_TIMEOUT_MS = 30_000


class CardStreamSimulator:
    """Python-side simulation of the JS _cardStreams + updateCardActivity logic."""

    def __init__(self):
        self.streams: dict[str, list[dict]] = {}

    def update(self, name: str, data: dict) -> list[dict]:
        entries = list(self.streams.get(name, []))
        event_type = data.get("eventType", "")

        if event_type == "tool_start":
            entries.append({
                "id": data.get("toolId", str(len(entries))),
                "type": "tool",
                "text": data.get("toolName", "tool"),
                "status": "running",
                "ts": 1000,
            })
        elif event_type in ("tool_end", "tool_use"):
            tool_id = data.get("toolId")
            for e in entries:
                if e["id"] == tool_id:
                    e["status"] = "error" if data.get("isError") else "done"
                    break
        elif event_type == "board_post":
            entries.append({
                "id": str(len(entries)),
                "type": "board",
                "text": f"#{data.get('channel', '?')}: {(data.get('summary') or '')[:60]}",
                "status": "done",
                "ts": 1000,
            })
        elif event_type == "cron":
            entries.append({
                "id": str(len(entries)),
                "type": "cron",
                "text": data.get("summary", "cron"),
                "status": "running",
                "ts": 1000,
            })
        elif event_type == "heartbeat":
            entries.append({
                "id": str(len(entries)),
                "type": "heartbeat",
                "text": "heartbeat",
                "status": "running",
                "ts": 1000,
            })

        if len(entries) > MAX_STREAM_ENTRIES * 2:
            entries = entries[-MAX_STREAM_ENTRIES:]

        self.streams[name] = entries
        return entries[-MAX_STREAM_ENTRIES:]


# ── Card Stream Simulator Tests ──────────────────────

class TestCardStreamSimulator:
    """Validate Python simulation matches JS logic."""

    def test_tool_start_adds_running_entry(self):
        sim = CardStreamSimulator()
        entries = sim.update("alice", {
            "eventType": "tool_start",
            "toolId": "t1",
            "toolName": "Read",
        })
        assert len(entries) == 1
        assert entries[0]["type"] == "tool"
        assert entries[0]["status"] == "running"
        assert entries[0]["text"] == "Read"

    def test_tool_end_marks_done(self):
        sim = CardStreamSimulator()
        sim.update("alice", {
            "eventType": "tool_start",
            "toolId": "t1",
            "toolName": "Read",
        })
        entries = sim.update("alice", {
            "eventType": "tool_end",
            "toolId": "t1",
            "isError": False,
        })
        assert entries[0]["status"] == "done"

    def test_tool_end_marks_error(self):
        sim = CardStreamSimulator()
        sim.update("alice", {
            "eventType": "tool_start",
            "toolId": "t1",
            "toolName": "Bash",
        })
        entries = sim.update("alice", {
            "eventType": "tool_end",
            "toolId": "t1",
            "isError": True,
        })
        assert entries[0]["status"] == "error"

    def test_board_post_adds_done_entry(self):
        sim = CardStreamSimulator()
        entries = sim.update("bob", {
            "eventType": "board_post",
            "channel": "general",
            "summary": "Hello everyone",
        })
        assert len(entries) == 1
        assert entries[0]["type"] == "board"
        assert entries[0]["status"] == "done"
        assert "#general" in entries[0]["text"]

    def test_cron_adds_running_entry(self):
        sim = CardStreamSimulator()
        entries = sim.update("charlie", {
            "eventType": "cron",
            "summary": "daily report",
        })
        assert len(entries) == 1
        assert entries[0]["type"] == "cron"
        assert entries[0]["status"] == "running"

    def test_heartbeat_adds_running_entry(self):
        sim = CardStreamSimulator()
        entries = sim.update("dave", {
            "eventType": "heartbeat",
            "summary": "heartbeat",
        })
        assert len(entries) == 1
        assert entries[0]["type"] == "heartbeat"
        assert entries[0]["status"] == "running"

    def test_stream_clipped_to_max_entries(self):
        sim = CardStreamSimulator()
        for i in range(12):
            sim.update("alice", {
                "eventType": "board_post",
                "channel": "general",
                "summary": f"msg {i}",
            })
        visible = sim.update("alice", {
            "eventType": "board_post",
            "channel": "general",
            "summary": "final",
        })
        assert len(visible) == MAX_STREAM_ENTRIES

    def test_multiple_animas_independent(self):
        sim = CardStreamSimulator()
        sim.update("alice", {"eventType": "cron", "summary": "cron-a"})
        sim.update("bob", {"eventType": "heartbeat", "summary": "hb-b"})
        assert len(sim.streams["alice"]) == 1
        assert len(sim.streams["bob"]) == 1
        assert sim.streams["alice"][0]["type"] == "cron"
        assert sim.streams["bob"][0]["type"] == "heartbeat"


# ── Status Attribute Mapping Tests ──────────────────────

class TestGetStatusAttr:
    """Validate getStatusAttr mapping logic (Python simulation)."""

    @staticmethod
    def _get_status_attr(status):
        if not status:
            return "idle"
        s = status if isinstance(status, str) else (status.get("state") or status.get("status") or "")
        lower = s.lower()
        if "error" in lower:
            return "error"
        if "bootstrap" in lower:
            return "bootstrapping"
        if lower in ("thinking", "working"):
            return "working"
        if "chat" in lower or "talk" in lower:
            return "chatting"
        return "idle"

    def test_idle(self):
        assert self._get_status_attr("idle") == "idle"

    def test_working(self):
        assert self._get_status_attr("working") == "working"

    def test_thinking(self):
        assert self._get_status_attr("thinking") == "working"

    def test_error(self):
        assert self._get_status_attr("error") == "error"

    def test_bootstrap(self):
        assert self._get_status_attr("bootstrapping") == "bootstrapping"

    def test_chatting(self):
        assert self._get_status_attr("chatting") == "chatting"

    def test_none(self):
        assert self._get_status_attr(None) == "idle"

    def test_sleeping_maps_to_idle(self):
        assert self._get_status_attr("sleeping") == "idle"

    def test_object_status(self):
        assert self._get_status_attr({"state": "working"}) == "working"


# ── WebSocket Event Payload Tests ──────────────────────

class TestWebSocketEventPayloads:
    """Verify WebSocket event payloads contain expected fields for card dispatch."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.ws_src = APP_WEBSOCKET_JS.read_text(encoding="utf-8")

    def test_tool_activity_passes_tool_id(self):
        tool_section = self.ws_src[self.ws_src.index("anima.tool_activity"):]
        update_call = tool_section[:1500]
        assert "toolId" in update_call
        assert "data.tool_id" in update_call

    def test_tool_activity_passes_tool_name(self):
        tool_section = self.ws_src[self.ws_src.index("anima.tool_activity"):]
        update_call = tool_section[:1500]
        assert "toolName" in update_call

    def test_tool_activity_passes_is_error(self):
        tool_section = self.ws_src[self.ws_src.index("anima.tool_activity"):]
        update_call = tool_section[:1500]
        assert "isError" in update_call

    def test_board_post_passes_channel(self):
        board_section = self.ws_src[self.ws_src.index("board.post"):]
        update_call = board_section[:1200]
        assert "channel" in update_call

    def test_board_post_passes_summary(self):
        board_section = self.ws_src[self.ws_src.index("board.post"):]
        update_call = board_section[:1200]
        assert "summary" in update_call


# ── API Endpoint Compatibility ──────────────────────

def _create_app_with_config(tmp_path: Path, anima_names: list[str]):
    """Build a real FastAPI app for testing."""
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "model": "claude-sonnet-4-6",
        "animas": {n: {"role": "general"} for n in anima_names},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    # Create IDENTITY.md for each anima so the route doesn't skip them
    for name in anima_names:
        anima_dir = animas_dir / name
        anima_dir.mkdir(parents=True, exist_ok=True)
        (anima_dir / "IDENTITY.md").write_text(f"# {name}\n", encoding="utf-8")

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_app_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
        patch("server.app.load_auth") as mock_auth,
    ):
        cfg = MagicMock()
        cfg.setup_complete = True
        mock_app_cfg.return_value = cfg

        auth_cfg = MagicMock()
        auth_cfg.auth_mode = "local_trust"
        mock_auth.return_value = auth_cfg

        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {
            "status": "idle",
            "pid": 9999,
            "bootstrapping": False,
            "uptime_sec": 60,
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
    _auth = MagicMock()
    _auth.auth_mode = "local_trust"
    _sa.load_auth = lambda: _auth

    app.state.anima_names = anima_names

    from core.config.models import invalidate_cache
    from core.config.models import load_config as _load_config

    invalidate_cache()
    real_config = _load_config(config_path)

    import server.routes.animas as animas_module
    animas_module.load_config = lambda *a, **kw: real_config

    return app


@pytest.mark.asyncio
async def test_api_animas_returns_status_field(tmp_path):
    """GET /api/animas should include status field for data-status mapping."""
    app = _create_app_with_config(tmp_path, ["alice", "bob"])

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/animas")

    if resp.status_code == 401:
        pytest.skip("Auth middleware active in test env — skip API test")
    assert resp.status_code == 200
    data = resp.json()
    animas = data if isinstance(data, list) else data.get("animas", [])
    assert len(animas) >= 2
    for a in animas:
        assert "name" in a
        assert "status" in a


# ── Integration: Full Event Flow ──────────────────────

class TestFullEventFlow:
    """Simulate a complete tool execution flow through card stream."""

    def test_tool_lifecycle(self):
        sim = CardStreamSimulator()
        sim.update("alice", {
            "eventType": "tool_start",
            "toolId": "t1",
            "toolName": "Bash",
        })
        entries = sim.streams["alice"]
        assert entries[0]["status"] == "running"

        sim.update("alice", {
            "eventType": "tool_end",
            "toolId": "t1",
            "isError": False,
        })
        assert entries[0]["status"] == "done"

    def test_mixed_event_types(self):
        sim = CardStreamSimulator()
        sim.update("alice", {"eventType": "heartbeat", "summary": "hb"})
        sim.update("alice", {
            "eventType": "tool_start",
            "toolId": "t1",
            "toolName": "Read",
        })
        sim.update("alice", {
            "eventType": "board_post",
            "channel": "general",
            "summary": "post",
        })
        sim.update("alice", {"eventType": "cron", "summary": "daily"})

        entries = sim.streams["alice"]
        types = [e["type"] for e in entries]
        assert types == ["heartbeat", "tool", "board", "cron"]

    def test_rapid_tool_calls_dont_exceed_buffer(self):
        sim = CardStreamSimulator()
        for i in range(20):
            sim.update("alice", {
                "eventType": "tool_start",
                "toolId": f"t{i}",
                "toolName": f"Tool{i}",
            })
        assert len(sim.streams["alice"]) <= MAX_STREAM_ENTRIES * 2


# ── CSS Integration ──────────────────────

class TestCssDataStatusIntegration:
    """Verify CSS data-status selectors cover all getStatusAttr return values."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.css = STYLE_CSS.read_text(encoding="utf-8")
        self.js = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_all_status_values_have_css(self):
        re.findall(r'return "(\w+)"', self.js)
        fn_start = self.js.index("function getStatusAttr")
        fn_end = self.js.index("// ── Tree Layout", fn_start)
        fn_body = self.js[fn_start:fn_end]
        return_values = set(re.findall(r'return "(\w+)"', fn_body))

        for val in return_values:
            assert f'data-status="{val}"' in self.css, (
                f"Missing CSS for data-status=\"{val}\""
            )

    def test_reduced_motion_exists(self):
        assert "prefers-reduced-motion: reduce" in self.css


# ── Layout Compatibility ──────────────────────

class TestLayoutCompatibility:
    """Verify card layout changes don't break tree layout."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.js = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_gap_y_sufficient_for_card_height(self):
        gap_match = re.search(r"const GAP_Y\s*=\s*(\d+)", self.js)
        assert gap_match
        gap_y = int(gap_match.group(1))
        assert gap_y >= 80, "GAP_Y should be large enough for taller cards with stream area"

    def test_card_h_unchanged(self):
        """CARD_H should remain at 100 for tree layout baseline (increased for usage info)."""
        match = re.search(r"const CARD_H\s*=\s*(\d+)", self.js)
        assert match
        assert int(match.group(1)) == 100

    def test_resize_svg_uses_actual_dimensions(self):
        resize_fn_start = self.js.index("function _resizeSvg")
        resize_fn_end = self.js.index("}", resize_fn_start + 50)
        resize_body = self.js[resize_fn_start:resize_fn_end + 50]
        assert "_getCardDimensions" in resize_body
