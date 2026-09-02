from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E coverage for signed GitHub webhooks and the single-notification dispatch."""

import hashlib
import hmac
import json
import secrets
import time
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.config.schemas import GitHubWebhookConfig
from core.schemas import Message
from server.github_gateway import GitHubWebhookManager
from server.routes.webhooks import create_webhooks_router

pytestmark = pytest.mark.e2e


def _signature(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("GitHub webhook background dispatch did not finish in time")


def test_signed_pr_webhook_notifies_dispatcher_after_quiet_period(data_dir, monkeypatch):
    """Signed synchronize event reaches rin (dispatcher_anima) after the quiet period."""
    secret = secrets.token_hex(32)
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)

    shared_dir = data_dir / "shared"
    state_file = shared_dir / "pr-review-dispatch-state.json"
    manager = GitHubWebhookManager(
        config=GitHubWebhookConfig(
            enabled=True,
            repos=["example-org/example-repo"],
            dispatcher_anima="rin",
            quiet_seconds=0.05,
        ),
        shared_dir=shared_dir,
        state_file=state_file,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await manager.start()
        app.state.github_gateway_manager = manager
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(create_webhooks_router(), prefix="/api")

    sha = "b7c3e2616089db2f8e6e53d46c18c5e89418ac42"
    payload = {
        "action": "synchronize",
        "number": 42,
        "repository": {
            "id": 1001,
            "full_name": "example-org/example-repo",
        },
        "pull_request": {
            "id": 2002,
            "number": 42,
            "title": "Webhook-driven PR review dispatch",
            "draft": False,
            "html_url": "https://github.com/example-org/example-repo/pull/42",
            "user": {"login": "animaworks-dev-team"},
            "head": {
                "ref": "feat/webhook-dispatch",
                "sha": sha,
            },
        },
        "sender": {"login": "animaworks-dev-team", "type": "User"},
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    inbox_dir = shared_dir / "inbox" / "rin"

    with TestClient(app) as client:
        invalid = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": "sha256=invalid",
            },
        )
        assert invalid.status_code == 401
        assert not inbox_dir.exists()

        accepted = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "e2e-delivery-001",
                "X-Hub-Signature-256": _signature(body, secret),
            },
        )
        assert accepted.status_code == 200
        assert accepted.json() == {"ok": True}

        _wait_until(lambda: len(list(inbox_dir.glob("*.json"))) == 1)

        def _state_notified() -> bool:
            # The inbox file is written inside the flock *before* the state
            # JSON is atomically replaced, so wait for the state flush too.
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                return False
            entry = state.get("prs", {}).get("example-org/example-repo#42", {})
            return entry.get("notified", {}).get("push") == sha

        _wait_until(_state_notified)

        files = list(inbox_dir.glob("*.json"))
        message = Message.model_validate_json(files[0].read_text(encoding="utf-8"))
        assert message.to_person == "rin"
        assert message.source == "system"
        assert message.intent == "report"
        assert "【GitHub通知】" in message.content
        assert "example-org/example-repo#42" in message.content
        assert "delegate_task" in message.content

        state = json.loads(state_file.read_text(encoding="utf-8"))
        entry = state["prs"]["example-org/example-repo#42"]
        assert entry["sha"] == sha
        assert entry["notified"]["push"] == sha

    assert manager._started is False
    assert manager._event_tasks == set()
    assert manager._debounce_tasks == {}


def test_signed_command_comment_notifies_dispatcher_without_creating_tasks(data_dir, monkeypatch):
    """A GitHub comment only produces one notification; the gateway creates no task."""
    secret = secrets.token_hex(32)
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)
    shared_dir = data_dir / "shared"
    manager = GitHubWebhookManager(
        config=GitHubWebhookConfig(
            enabled=True,
            repos=["example-org/example-repo"],
            dispatcher_anima="rin",
            bot_login="example-bot",
        ),
        shared_dir=shared_dir,
        state_file=shared_dir / "pr-review-dispatch-state.json",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await manager.start()
        app.state.github_gateway_manager = manager
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(create_webhooks_router(), prefix="/api")
    payload = {
        "action": "created",
        "repository": {"full_name": "example-org/example-repo"},
        "issue": {"number": 42},
        "comment": {
            "id": 123,
            "user": {"login": "human"},
            "body": "@example-bot resolve this conflict",
            "html_url": "https://github.com/example-org/example-repo/pull/42#issuecomment-123",
        },
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    inbox_dir = shared_dir / "inbox" / "rin"

    with TestClient(app) as client:
        response = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "issue_comment",
                "X-GitHub-Delivery": "e2e-command-001",
                "X-Hub-Signature-256": _signature(body, secret),
            },
        )
        assert response.status_code == 200
        _wait_until(lambda: len(list(inbox_dir.glob("*.json"))) == 1)

        files = list(inbox_dir.glob("*.json"))
        message = Message.model_validate_json(files[0].read_text(encoding="utf-8"))
        assert message.to_person == "rin"
        assert "@example-bot resolve this conflict" in message.content
        assert not (shared_dir / "animas").exists()
