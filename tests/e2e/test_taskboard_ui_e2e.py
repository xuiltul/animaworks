from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from core.tasks.queue import TaskQueueManager

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))


def _create_app(tmp_path: Path, anima_names: list[str], *, base_path: str = ""):
    animas_dir = tmp_path / "animas"
    shared_dir = tmp_path / "shared"
    for name in anima_names:
        anima_dir = animas_dir / name
        (anima_dir / "state").mkdir(parents=True, exist_ok=True)
        (anima_dir / "IDENTITY.md").write_text(f"# {name}\n", encoding="utf-8")
    (shared_dir / "channels").mkdir(parents=True, exist_ok=True)

    with (
        patch("server.app.ProcessSupervisor") as mock_sup_cls,
        patch("server.app.load_config") as mock_cfg,
        patch("server.app.WebSocketManager") as mock_ws_cls,
        patch("server.app.load_auth") as mock_auth,
    ):
        cfg = MagicMock()
        cfg.setup_complete = True
        cfg.server.base_path = base_path
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

        async def _accept_ws(ws):
            await ws.accept()

        ws_manager = MagicMock()
        ws_manager.active_connections = []
        ws_manager.connect = AsyncMock(side_effect=_accept_ws)
        ws_manager.handle_client_message = AsyncMock()
        mock_ws_cls.return_value = ws_manager

        from server.app import create_app

        app = create_app(animas_dir, shared_dir)

    import server.app as server_app

    local_auth = MagicMock()
    local_auth.auth_mode = "local_trust"
    server_app.load_auth = lambda: local_auth
    return app


def _seed_taskboard(app) -> tuple[str, str]:
    alice_queue = TaskQueueManager(app.state.animas_dir / "alice")
    bob_queue = TaskQueueManager(app.state.animas_dir / "bob")
    todo = alice_queue.add_task(
        source="human",
        original_instruction="prepare UI rollout",
        assignee="alice",
        summary="prepare UI rollout",
        task_id="task-ui",
    )
    running = bob_queue.add_task(
        source="human",
        original_instruction="verify TaskBoard action",
        assignee="bob",
        summary="verify TaskBoard action",
        task_id="task-action",
    )
    bob_queue.update_status(running.task_id, "in_progress")
    return todo.task_id, running.task_id


async def test_taskboard_route_static_assets_and_api_smoke(tmp_path: Path) -> None:
    app = _create_app(tmp_path, ["alice", "bob"])
    _seed_taskboard(app)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        index_resp = await client.get("/")
        page_resp = await client.get("/_v/test/pages/task-board.js")
        utils_resp = await client.get("/_v/test/pages/task-board-utils.js")
        css_resp = await client.get("/_v/test/styles/task-board.css")
        board_resp = await client.get("/api/task-board", params={"include_missing": "true"})

    assert index_resp.status_code == 200
    assert "#/task-board" in index_resp.text
    assert 'data-route="/task-board"' in index_resp.text
    assert "task-board.css" in index_resp.text
    assert "#/board" in index_resp.text

    assert page_resp.status_code == 200
    assert "/api/task-board" in page_resp.text
    assert "REFRESH_MS = 30000" in page_resp.text
    assert "document.visibilityState" in page_resp.text
    assert "escapeAttr" in page_resp.text
    assert "_renderToken" in page_resp.text
    assert "if (!_container || token !== _renderToken) return;" in page_resp.text
    assert 'method: "POST"' in page_resp.text
    assert "reasonRequired: true, confirmRequired: true" in page_resp.text
    assert 'setCustomValidity(t("taskboard.reason_required"))' in page_resp.text
    assert 't("taskboard.confirm_cancel")' in page_resp.text
    assert "canonical_task_id || task.task_id" in page_resp.text
    assert "const hasVisibleTask = (column)" in page_resp.text
    assert "COLUMNS.find(hasVisibleTask)" in page_resp.text
    assert "taskboard.mark_done" not in page_resp.text
    assert "/api/channels" not in page_resp.text
    assert "../pages/board" not in page_resp.text

    assert utils_resp.status_code == 200
    assert 'export const COLUMNS = ["todo", "running", "waiting", "done"]' in utils_resp.text
    assert "export function isCancellable(task)" in utils_resp.text
    static_assets = page_resp.text + utils_resp.text + css_resp.text
    for removed_ui in ("snooze", "tombstone", "PATCH"):
        assert removed_ui not in static_assets
    assert css_resp.status_code == 200
    assert "@media (max-width: 720px)" in css_resp.text
    assert ".taskboard-mobile-tabs" in css_resp.text

    assert board_resp.status_code == 200
    task_ids = {task["task_id"] for task in board_resp.json()["tasks"]}
    assert {"task-ui", "task-action"} <= task_ids


def test_taskboard_mobile_active_column_logic_uses_visible_tasks(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("node not available")

    source = (Path(__file__).parents[2] / "server/static/pages/task-board.js").read_text(encoding="utf-8")
    module_body = source[source.index("const REFRESH_MS") :]
    module_body = module_body.replace("export async function render", "async function render")
    module_body = module_body.replace("export function destroy", "function destroy")
    script = tmp_path / "taskboard-active-column.mjs"
    script.write_text(
        textwrap.dedent(
            f"""
            import assert from "node:assert/strict";
            const COLUMNS = ["todo", "running", "waiting", "done"];
            globalThis.document = {{ visibilityState: "visible" }};
            globalThis.window = {{}};
            {module_body}
            function choose(active, tasks) {{
              _activeColumn = active;
              _tasks = tasks;
              _ensureActiveColumnHasView();
              return _activeColumn;
            }}
            assert.equal(choose("todo", [{{ column: "running" }}]), "running");
            assert.equal(choose("running", [{{ column: "running" }}]), "running");
            assert.equal(choose("not-a-column", [{{ column: "waiting" }}]), "waiting");
            assert.equal(choose("waiting", []), "todo");
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(["node", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


async def test_taskboard_base_path_serves_direct_prefixed_routes(tmp_path: Path) -> None:
    app = _create_app(tmp_path, ["alice", "bob"], base_path="/app")
    _seed_taskboard(app)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        exact_index_resp = await client.get("/app")
        index_resp = await client.get("/app/")
        api_resp = await client.get("/app/api/task-board", params={"include_missing": "true"})
        workspace_resp = await client.get("/app/workspace/")
        workspace_ws_resp = await client.get("/app/workspace/modules/websocket.js")
        non_matching_resp = await client.get("/application")

        version_match = re.search(r"/app/_v/([^/]+)/modules/app\.js", index_resp.text)
        assert version_match is not None
        version = version_match.group(1)
        app_js_resp = await client.get(f"/app/_v/{version}/modules/app.js")
        base_path_js_resp = await client.get(f"/app/_v/{version}/shared/base-path.js")

    assert index_resp.status_code == 200
    assert exact_index_resp.status_code == 200
    assert 'content="/app"' in index_resp.text
    assert "/app/_v/" in index_resp.text

    assert app_js_resp.status_code == 200
    assert base_path_js_resp.status_code == 200
    assert api_resp.status_code == 200
    assert {"task-ui", "task-action"} <= {task["task_id"] for task in api_resp.json()["tasks"]}

    assert workspace_resp.status_code == 200
    assert 'content="/app"' in workspace_resp.text
    assert workspace_ws_resp.status_code == 200
    assert 'import { basePath } from "/shared/base-path.js";' in workspace_ws_resp.text
    assert "${basePath}/ws" in workspace_ws_resp.text

    assert non_matching_resp.status_code == 404


def test_taskboard_base_path_routes_installed_websocket(tmp_path: Path) -> None:
    app = _create_app(tmp_path, ["alice"], base_path="/app")
    client = TestClient(app)
    auth_cfg = MagicMock(auth_mode="local_trust")

    with (
        patch("server.routes.websocket_route.load_auth", return_value=auth_cfg),
        client.websocket_connect("/app/ws") as ws,
    ):
        ws.send_text('{"type":"ping"}')

    app.state.ws_manager.connect.assert_awaited_once()
    app.state.ws_manager.handle_client_message.assert_awaited_once()
    assert app.state.ws_manager.handle_client_message.await_args.args[1] == '{"type":"ping"}'


async def test_taskboard_base_path_strips_websocket_scope() -> None:
    from server.app import BasePathMiddleware

    captured_scope = {}
    sent_messages = []

    async def downstream(scope, receive, send):  # type: ignore[no-untyped-def]
        captured_scope.update(scope)
        await send({"type": "websocket.close", "code": 1000})

    async def receive():  # type: ignore[no-untyped-def]
        return {"type": "websocket.connect"}

    async def send(message):  # type: ignore[no-untyped-def]
        sent_messages.append(message)

    middleware = BasePathMiddleware(downstream, base_path="/app")
    await middleware(
        {"type": "websocket", "path": "/app/ws", "raw_path": b"/app/ws", "root_path": ""},
        receive,
        send,
    )

    assert captured_scope["path"] == "/ws"
    assert captured_scope["raw_path"] == b"/ws"
    assert captured_scope.get("root_path", "") == ""
    assert captured_scope["app_root_path"] == "/app"
    assert sent_messages == [{"type": "websocket.close", "code": 1000}]

    captured_scope.clear()
    await middleware(
        {
            "type": "websocket",
            "path": "/app/ws/voice/alice",
            "raw_path": b"/app/ws/voice/alice",
            "root_path": "",
        },
        receive,
        send,
    )
    assert captured_scope["path"] == "/ws/voice/alice"
    assert captured_scope["raw_path"] == b"/ws/voice/alice"


async def test_taskboard_cancel_changes_canonical_status_without_modifying_board_channels(tmp_path: Path) -> None:
    app = _create_app(tmp_path, ["alice", "bob"])
    alice_task_id, _bob_task_id = _seed_taskboard(app)
    channel_path = app.state.shared_dir / "channels" / "general.jsonl"
    channel_entry = {
        "ts": "2026-05-14T10:00:00+09:00",
        "from": "alice",
        "text": "Board conversation remains independent",
    }
    channel_path.write_text(json.dumps(channel_entry, ensure_ascii=False) + "\n", encoding="utf-8")
    before = channel_path.read_bytes()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        cancel_resp = await client.post(
            f"/api/task-board/alice/{alice_task_id}/cancel",
            json={"reason": "obsolete from dashboard"},
        )
        board_resp = await client.get("/api/task-board", params={"include_archived": "true"})
        channels_resp = await client.get("/api/channels")

    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["result"]["status"] == "cancelled"
    assert board_resp.status_code == 200
    board_rows = board_resp.json()["tasks"]
    assert {row["canonical_task_id"] for row in board_rows if row["queue_status"] == "cancelled"} == {alice_task_id}
    canonical_row = next(row for row in board_rows if row["canonical_task_id"] == alice_task_id)
    assert canonical_row["queue_status"] == "cancelled"
    assert canonical_row["visibility"] == "archived"
    assert channel_path.read_bytes() == before
    assert channels_resp.status_code == 200
    assert channels_resp.json()[0]["name"] == "general"
    assert channels_resp.json()[0]["message_count"] == 1
