from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Endpoint-level authorization tests for the internal API."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.internal_auth import InternalAuth, require_internal_caller


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _hierarchy_config(mode: str = "enforce"):
    animas = {
        "boss": SimpleNamespace(supervisor=None),
        "sub": SimpleNamespace(supervisor="boss"),
        "grandchild": SimpleNamespace(supervisor="sub"),
        "other": SimpleNamespace(supervisor=None),
    }
    return SimpleNamespace(server=SimpleNamespace(internal_api_auth=mode), animas=animas)


@pytest.fixture
def authz_client(tmp_path, monkeypatch):
    """Build an authenticated router against an isolated org and data directory."""
    import server.internal_auth as internal_auth_module

    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "server": {"internal_api_auth": "enforce"},
                "animas": {
                    "boss": {},
                    "sub": {"supervisor": "boss"},
                    "grandchild": {"supervisor": "sub"},
                    "other": {},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(internal_auth_module, "load_config", lambda: _hierarchy_config())
    monkeypatch.setattr(internal_auth_module, "require_internal_caller", require_internal_caller)
    monkeypatch.setattr("core.config.io.load_config", lambda: _hierarchy_config())

    animas_dir = tmp_path / "animas"
    for name in ("boss", "sub", "grandchild", "other", "created"):
        (animas_dir / name / "state").mkdir(parents=True, exist_ok=True)
    (animas_dir / "boss" / "skills" / "newstaff").mkdir(parents=True)
    (animas_dir / "boss" / "skills" / "newstaff" / "SKILL.md").write_text("hire", encoding="utf-8")
    monkeypatch.setattr("core.paths.get_animas_dir", lambda: animas_dir)
    monkeypatch.setattr("core.paths.get_shared_dir", lambda: tmp_path / "shared")

    auth = InternalAuth.generate()
    app = FastAPI()
    app.state.internal_auth = auth
    # Vector endpoints forward to the owning root over supervisor IPC.
    app.state.supervisor = SimpleNamespace(send_request=AsyncMock(return_value={"ok": True, "result": {}}))
    app.state.ws_manager = MagicMock(broadcast=AsyncMock(), broadcast_notification=AsyncMock())

    from server.routes.internal import create_internal_router

    app.include_router(create_internal_router(), prefix="/api")

    # Side-effecting service layers are stubbed; these tests exercise routing and
    # authorization while keeping all writes inside tmp_path.
    monkeypatch.setattr("server.routes.internal.emit", AsyncMock())
    monkeypatch.setattr("core.config.models.read_anima_company_checked", lambda _path: (True, None))
    monkeypatch.setattr("core.org.company.get_company_display_name", lambda _company: "")
    monkeypatch.setattr("core.notification.reply_routing.save_notification_mapping", lambda *a, **k: True)
    monkeypatch.setattr(
        "core.notification.interactive.get_interaction_router",
        lambda: SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(model_dump=lambda **_k: {"id": "x"}))),
    )
    monkeypatch.setattr("core.anima.factory.create_from_md", lambda *_a, **_k: animas_dir / "created")
    monkeypatch.setattr("core.messaging.messenger.Messenger.post_channel", lambda *_a, **_k: None)
    from core.messaging.board import PostResult

    monkeypatch.setattr("core.messaging.board.post_to_board", lambda *_a, **_k: PostResult(committed=True, id="post-1"))
    monkeypatch.setattr("core.tasks.dispatch.publish_tasks", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "core.tasks.dispatch.publish_delegation",
        lambda *_a, **_k: SimpleNamespace(task_id="aabbccddeeff", status="pending", deduplicated=False, via_host=False),
    )
    monkeypatch.setattr(
        "core.org.company.check_company_boundary",
        lambda *_a, **_k: SimpleNamespace(cross_company=False, resolved_via="config", display_name=""),
    )
    monkeypatch.setattr("core.tasks.board.board_actions.run_board_action", lambda **_k: {"ok": True})

    from core.tasks.queue import TaskQueueManager

    task = TaskQueueManager(animas_dir / "boss").add_task(
        source="anima", original_instruction="work", assignee="boss", summary="work"
    )
    client = TestClient(app)
    yield client, auth, task.task_id, app
    client.close()


def _headers(auth: InternalAuth, identity: str) -> dict[str, str]:
    token = auth.operator_token() if identity == "operator" else auth.token_for_anima(identity)
    return {"X-AnimaWorks-Internal-Auth": token}


def _endpoint_cases(task_id: str):
    """Requests and the exact claimed identity field for all named routes."""
    delegate = {
        "delegator": "boss",
        "target": "sub",
        "instruction": "work",
        "summary": "work",
        "sub_task_id": "aabbccddeeff",
        "tracking_task_id": "112233445566",
    }
    return [
        ("get", "/api/internal/company/boundary?from_anima=boss&to_anima=sub", None, "query:from_anima"),
        (
            "post",
            "/api/internal/message-sent",
            {"from_person": "boss", "to_person": "sub", "content": "hello"},
            "from_person",
        ),
        (
            "post",
            "/api/internal/vector/query",
            {"anima_name": "boss", "collection": "x", "embedding": []},
            "anima_name",
        ),
        (
            "post",
            "/api/internal/vector/upsert",
            {"anima_name": "boss", "collection": "x", "documents": []},
            "anima_name",
        ),
        (
            "post",
            "/api/internal/vector/update-metadata",
            {"anima_name": "boss", "collection": "x", "ids": [], "metadatas": []},
            "anima_name",
        ),
        (
            "post",
            "/api/internal/vector/delete-documents",
            {"anima_name": "boss", "collection": "x", "ids": []},
            "anima_name",
        ),
        (
            "post",
            "/api/internal/vector/get-by-metadata",
            {"anima_name": "boss", "collection": "x", "where": {}},
            "anima_name",
        ),
        ("post", "/api/internal/vector/get-by-ids", {"anima_name": "boss", "collection": "x", "ids": []}, "anima_name"),
        ("post", "/api/internal/vector/create-collection", {"anima_name": "boss", "collection": "x"}, "anima_name"),
        ("post", "/api/internal/vector/delete-collection", {"anima_name": "boss", "collection": "x"}, "anima_name"),
        ("post", "/api/internal/vector/list-collections", {"anima_name": "boss"}, "anima_name"),
        ("post", "/api/internal/notification-mapping", {"ts": "1", "channel": "C", "anima_name": "boss"}, "anima_name"),
        ("post", "/api/internal/call-human/confirm", {"anima_name": "boss", "session_id": "s"}, "anima_name"),
        ("post", "/api/internal/interaction/create", {"anima_name": "boss", "options": ["yes"]}, "anima_name"),
        (
            "post",
            "/api/internal/anima/create",
            {"calling_anima": "boss", "name": "created", "character_sheet_content": "name"},
            "calling_anima",
        ),
        (
            "post",
            "/api/internal/send-message",
            {"message": {"from_person": "boss", "to_person": "sub", "content": "hello", "id": "msg-1"}},
            "message.from_person",
        ),
        (
            "post",
            "/api/internal/post-channel",
            {"from_anima": "boss", "channel": "general", "text": "hello"},
            "from_anima",
        ),
        ("post", "/api/internal/notify-web", {"anima": "boss", "subject": "s", "body": "b"}, "anima"),
        ("get", "/api/internal/tasks?anima_name=boss", None, "query:anima_name"),
        ("post", "/api/internal/submit-tasks", {"anima_name": "boss", "tasks": []}, "anima_name"),
        ("post", "/api/internal/delegate-task", delegate, "delegator"),
        ("post", "/api/internal/task-board-action", {"actor": "boss", "action": "claim", "task_id": task_id}, "actor"),
        (
            "post",
            "/api/internal/update-task",
            {"anima_name": "boss", "task_id": task_id, "status": "done"},
            "anima_name",
        ),
    ]


def _replace_claim(method: str, url: str, body: dict | None, claim_field: str, value: str):
    if claim_field.startswith("query:"):
        key = claim_field.split(":", 1)[1]
        before, _, query = url.partition("?")
        params = dict(part.split("=", 1) for part in query.split("&"))
        params[key] = value
        return method, before + "?" + "&".join(f"{key}={item}" for key, item in params.items()), body
    changed = dict(body or {})
    if claim_field == "message.from_person":
        changed["message"] = {**changed["message"], "from_person": value}
    else:
        changed[claim_field] = value
    return method, url, changed


@pytest.mark.parametrize("method,url,body,claim_field", _endpoint_cases("existing-task"))
def test_named_endpoint_identity_matrix(authz_client, method, url, body, claim_field):
    client, auth, task_id, _app = authz_client
    if "existing-task" in str(body):
        body = {**body, "task_id": task_id}

    # The caller may operate as itself; operator can operate as any name.
    own = client.request(method.upper(), url, json=body, headers=_headers(auth, "boss"))
    assert own.status_code < 300, f"{url}: {own.status_code} {own.text}"

    mismatch_value = "other" if url.startswith("/api/internal/tasks?") else "sub"
    mismatch_method, mismatch_url, mismatch_body = _replace_claim(method, url, body, claim_field, mismatch_value)
    denied = client.request(mismatch_method.upper(), mismatch_url, json=mismatch_body, headers=_headers(auth, "boss"))
    assert denied.status_code == 403, f"{url}: {denied.status_code} {denied.text}"

    operator = client.request(method.upper(), url, json=body, headers=_headers(auth, "operator"))
    assert operator.status_code < 300, f"operator {url}: {operator.status_code} {operator.text}"


def test_vector_requires_named_owner(authz_client):
    client, auth, _task_id, app = authz_client
    for path in ("/api/internal/vector/query", "/api/internal/vector/upsert"):
        resp = client.post(
            path,
            json={"anima_name": None, "collection": "shared", "embedding": [], "documents": []},
            headers=_headers(auth, "boss"),
        )
        assert resp.status_code == 422, f"{path}: {resp.status_code} {resp.text}"
    own = client.post(
        "/api/internal/vector/upsert",
        json={"anima_name": "boss", "collection": "x", "documents": []},
        headers=_headers(auth, "boss"),
    )
    assert own.status_code == 200
    app.state.supervisor.send_request.assert_awaited()


def test_delegate_requires_direct_subordinate(authz_client):
    client, auth, _task_id, _app = authz_client
    base = {
        "delegator": "boss",
        "target": "sub",
        "instruction": "work",
        "summary": "work",
        "sub_task_id": "aabbccddeeff",
        "tracking_task_id": "112233445566",
    }
    assert client.post("/api/internal/delegate-task", json=base, headers=_headers(auth, "boss")).status_code == 200
    indirect = {**base, "target": "grandchild", "sub_task_id": "aabbccddeefe"}
    assert client.post("/api/internal/delegate-task", json=indirect, headers=_headers(auth, "boss")).status_code == 403
    upward = {**base, "delegator": "sub", "target": "boss"}
    assert client.post("/api/internal/delegate-task", json=upward, headers=_headers(auth, "sub")).status_code == 403
    assert (
        client.post("/api/internal/delegate-task", json=indirect, headers=_headers(auth, "operator")).status_code == 200
    )


def test_anima_create_requires_newstaff_and_related_supervisor(authz_client, tmp_path, monkeypatch):
    client, auth, _task_id, _app = authz_client
    from core.paths import get_animas_dir

    boss_dir = get_animas_dir() / "boss"
    skill = boss_dir / "skills" / "newstaff" / "SKILL.md"
    payload = {"calling_anima": "boss", "name": "created", "character_sheet_content": "name"}
    skill.unlink()
    denied = client.post("/api/internal/anima/create", json=payload, headers=_headers(auth, "boss"))
    assert denied.status_code == 403
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text("hire", encoding="utf-8")
    unrelated = {**payload, "supervisor": "other"}
    assert client.post("/api/internal/anima/create", json=unrelated, headers=_headers(auth, "boss")).status_code == 403
    assert client.post("/api/internal/anima/create", json=payload, headers=_headers(auth, "boss")).status_code == 200


def test_tasks_can_be_read_by_supervisor_descendants_only(authz_client):
    client, auth, _task_id, _app = authz_client
    assert client.get("/api/internal/tasks?anima_name=sub", headers=_headers(auth, "boss")).status_code == 200
    assert client.get("/api/internal/tasks?anima_name=grandchild", headers=_headers(auth, "boss")).status_code == 200
    assert client.get("/api/internal/tasks?anima_name=boss", headers=_headers(auth, "sub")).status_code == 403


def test_task_board_anima_cannot_claim_human_identity(authz_client):
    client, auth, task_id, _app = authz_client
    response = client.post(
        "/api/internal/task-board-action",
        json={"actor": "human", "action": "claim", "task_id": task_id},
        headers=_headers(auth, "boss"),
    )
    assert response.status_code == 403
    operator = client.post(
        "/api/internal/task-board-action",
        json={"actor": "human", "action": "claim", "task_id": task_id},
        headers=_headers(auth, "operator"),
    )
    assert operator.status_code == 200


def test_log_mode_records_denial_and_allows_request(authz_client, monkeypatch):
    import server.internal_auth as internal_auth_module

    client, auth, _task_id, _app = authz_client
    monkeypatch.setattr(internal_auth_module, "load_config", lambda: _hierarchy_config("log"))
    fake_logger = MagicMock()
    monkeypatch.setattr(internal_auth_module, "logger", fake_logger)
    response = client.post(
        "/api/internal/notification-mapping",
        json={"ts": "1", "channel": "C", "anima_name": "other"},
        headers=_headers(auth, "boss"),
    )
    assert response.status_code == 200
    assert any("internal_api_authz_denied" in str(call) for call in fake_logger.warning.call_args_list)


def test_authz_off_mode_with_no_caller_preserves_legacy_access(authz_client, monkeypatch):
    import server.internal_auth as internal_auth_module

    client, _auth, _task_id, _app = authz_client
    monkeypatch.setattr(internal_auth_module, "load_config", lambda: _hierarchy_config("off"))
    response = client.post(
        "/api/internal/notification-mapping",
        json={"ts": "1", "channel": "C", "anima_name": "other"},
    )
    assert response.status_code == 200


def test_org_hierarchy_helpers_table_driven():
    from core.config.schemas import AnimaModelConfig
    from core.org.hierarchy import descendants_of, is_direct_subordinate

    animas = {
        "boss": AnimaModelConfig(),
        "sub": AnimaModelConfig(supervisor="boss"),
        "grandchild": AnimaModelConfig(supervisor="sub"),
        "other": AnimaModelConfig(),
    }
    assert is_direct_subordinate(animas, "boss", "sub")
    assert not is_direct_subordinate(animas, "boss", "grandchild")
    assert not is_direct_subordinate(animas, "boss", "boss")
    assert not is_direct_subordinate(animas, "boss", "missing")
    assert descendants_of(animas, "boss") == {"sub", "grandchild"}
    assert descendants_of(animas, "other") == set()


def test_has_newstaff_skill_supports_both_layouts(tmp_path):
    from core.anima.skills_check import has_newstaff_skill

    assert not has_newstaff_skill(tmp_path)
    (tmp_path / "skills" / "newstaff").mkdir(parents=True)
    (tmp_path / "skills" / "newstaff" / "SKILL.md").write_text("skill", encoding="utf-8")
    assert has_newstaff_skill(tmp_path)
    (tmp_path / "skills" / "newstaff" / "SKILL.md").unlink()
    (tmp_path / "skills" / "newstaff.md").write_text("skill", encoding="utf-8")
    assert has_newstaff_skill(tmp_path)
