"""E2E coverage for the activity session replay data/UI contract."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLAY_JS = REPO_ROOT / "server/static/pages/activity/session-replay.js"
GROUP_DETAIL_JS = REPO_ROOT / "server/static/pages/activity/group-detail.js"
ACTIVITY_CSS = REPO_ROOT / "server/static/styles/activity.css"
DOM_NODE_TEST = REPO_ROOT / "tests/e2e/test_activity_session_replay_dom.mjs"


def _create_app(tmp_path: Path):
    animas_dir = tmp_path / "animas"
    shared_dir = tmp_path / "shared"
    (animas_dir / "alice").mkdir(parents=True)
    shared_dir.mkdir(parents=True)

    with (
        patch("server.app.ProcessSupervisor") as supervisor_cls,
        patch("server.app.load_config") as load_config,
        patch("server.app.WebSocketManager") as websocket_cls,
        patch("server.app.load_auth") as load_auth,
    ):
        config = MagicMock()
        config.setup_complete = True
        load_config.return_value = config
        auth = MagicMock()
        auth.auth_mode = "local_trust"
        load_auth.return_value = auth
        supervisor = MagicMock()
        supervisor.get_all_status.return_value = {}
        supervisor.get_process_status.return_value = {"status": "stopped", "pid": None}
        supervisor.is_scheduler_running.return_value = False
        supervisor.scheduler = None
        supervisor_cls.return_value = supervisor
        websocket_cls.return_value = MagicMock(active_connections=[])

        from server.app import create_app

        app = create_app(animas_dir, shared_dir)

    import server.app as server_app

    persistent_auth = MagicMock()
    persistent_auth.auth_mode = "local_trust"
    server_app.load_auth = lambda: persistent_auth
    app.state.anima_names = ["alice"]
    return app, animas_dir


def _append_events(animas_dir: Path, entries: list[dict]) -> None:
    log_dir = animas_dir / "alice" / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        path = log_dir / f"{entry['ts'][:10]}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def test_chat_group_replay_contract_includes_bubbles_tool_and_thinking(tmp_path: Path) -> None:
    app, animas_dir = _create_app(tmp_path)
    now = datetime.now(UTC)
    _append_events(
        animas_dir,
        [
            {
                "ts": (now - timedelta(seconds=4)).isoformat(),
                "type": "message_received",
                "content": "Please **search**",
                "from": "human",
                "meta": {"from_type": "human"},
            },
            {
                "ts": (now - timedelta(seconds=3)).isoformat(),
                "type": "tool_use",
                "tool": "web_search",
                "summary": "{'query': 'docs'}",
                "meta": {"tool_use_id": "tool-1", "args": {"query": "docs"}},
            },
            {
                "ts": (now - timedelta(seconds=2)).isoformat(),
                "type": "tool_result",
                "tool": "web_search",
                "content": "one result",
                "meta": {"tool_use_id": "tool-1", "is_error": False},
            },
            {
                "ts": (now - timedelta(seconds=1)).isoformat(),
                "type": "response_sent",
                "content": "Found the result.",
                "meta": {"thinking_text": "Check the strongest match."},
            },
        ],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        recent = await client.get("/api/activity/recent?grouped=true&anima=alice")
        assert recent.status_code == 200
        group_id = recent.json()["groups"][0]["id"]
        response = await client.get("/api/activity/group", params={"anima": "alice", "id": group_id})

    assert response.status_code == 200
    group = response.json()["group"]
    assert [event["type"] for event in group["events"]] == [
        "message_received",
        "tool_use",
        "response_sent",
    ]
    assert group["events"][1]["tool_result"] == {
        "id": group["events"][1]["tool_result"]["id"],
        "ts": (now - timedelta(seconds=2)).isoformat(),
        "type": "tool_result",
        "content": "one result",
        "is_error": False,
    }
    assert group["events"][2]["meta"]["thinking_text"] == "Check the strongest match."

    replay_source = REPLAY_JS.read_text(encoding="utf-8")
    assert 'evt?.type === "message_received"' in replay_source
    assert 'evt?.type === "response_sent"' in replay_source
    assert 'evt?.type === "tool_use"' in replay_source
    assert "renderSafeMarkdown" in replay_source
    assert "thinking_text" in replay_source
    assert "resultMs - startMs" in replay_source


async def test_open_group_poll_contract_returns_new_event_ids_and_then_stops(tmp_path: Path) -> None:
    app, animas_dir = _create_app(tmp_path)
    now = datetime.now(UTC)
    start = {
        "ts": (now - timedelta(seconds=2)).isoformat(),
        "type": "message_received",
        "content": "Keep working",
        "from": "human",
        "meta": {"from_type": "human"},
    }
    _append_events(animas_dir, [start])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        recent = await client.get("/api/activity/recent?grouped=true&anima=alice")
        group_id = recent.json()["groups"][0]["id"]
        initial_response = await client.get(
            "/api/activity/group", params={"anima": "alice", "id": group_id}
        )
        initial = initial_response.json()["group"]
        assert initial["is_open"] is True

        _append_events(
            animas_dir,
            [
                {
                    "ts": (now - timedelta(seconds=1)).isoformat(),
                    "type": "tool_use",
                    "tool": "read_file",
                    "summary": "notes.md",
                }
            ],
        )
        updated_response = await client.get(
            "/api/activity/group", params={"anima": "alice", "id": group_id}
        )
        updated = updated_response.json()["group"]
        initial_ids = {event["id"] for event in initial["events"]}
        new_ids = {event["id"] for event in updated["events"]} - initial_ids
        assert len(new_ids) == 1
        assert updated["is_open"] is True

        _append_events(
            animas_dir,
            [
                {
                    "ts": now.isoformat(),
                    "type": "response_sent",
                    "content": "Done",
                }
            ],
        )
        finished_response = await client.get(
            "/api/activity/group", params={"anima": "alice", "id": group_id}
        )
        assert finished_response.json()["group"]["is_open"] is False

    replay_source = REPLAY_JS.read_text(encoding="utf-8")
    assert "REPLAY_POLL_INTERVAL_MS = 5000" in replay_source
    assert "eventSignatures" in replay_source
    assert "_isNearBottom" in replay_source
    assert "if (_updateLiveStatus(replay, group)) _schedulePoll(replay)" in replay_source


async def test_missing_group_and_view_toggle_localization_contract(tmp_path: Path) -> None:
    app, _ = _create_app(tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/activity/group",
            params={"anima": "alice", "id": "grp-alice:2026-01-01T00:00:00+00:00:chat"},
        )
    assert response.status_code == 404

    group_detail_source = GROUP_DETAIL_JS.read_text(encoding="utf-8")
    assert "activity.replay_conversation_view" in group_detail_source
    assert "activity.replay_raw_view" in group_detail_source
    assert "renderSessionReplay" in group_detail_source
    assert ".replay-view-tabs" in ACTIVITY_CSS.read_text(encoding="utf-8")

    for language in ("ja", "en", "ko"):
        strings = json.loads(
            (REPO_ROOT / f"server/static/i18n/{language}.json").read_text(encoding="utf-8")
        )
        assert strings["activity.replay_conversation_view"]
        assert strings["activity.replay_raw_view"]
        assert strings["activity.replay_not_found"]


def test_session_replay_dom_behavior_suite() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    result = subprocess.run(
        [node, "--test", str(DOM_NODE_TEST)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "node session replay suite failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
