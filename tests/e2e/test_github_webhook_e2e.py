from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E coverage for signed GitHub webhooks and review dispatch."""

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


def test_signed_pr_webhook_dispatches_to_real_inbox(data_dir, monkeypatch):
    """Signed synchronize event reaches sumire after the quiet period."""
    secret = secrets.token_hex(32)
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)

    shared_dir = data_dir / "shared"
    state_file = shared_dir / "pr-review-dispatch-state.json"
    manager = GitHubWebhookManager(
        config=GitHubWebhookConfig(
            enabled=True,
            repos=["FutureSync/AI-Schreiber"],
            reviewer_anima="sumire",
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
            "full_name": "FutureSync/AI-Schreiber",
        },
        "pull_request": {
            "id": 2002,
            "number": 42,
            "title": "Webhook-driven PR review dispatch",
            "draft": False,
            "html_url": "https://github.com/FutureSync/AI-Schreiber/pull/42",
            "head": {
                "ref": "feat/webhook-dispatch",
                "sha": sha,
            },
        },
        "sender": {"login": "animaworks-dev-team", "type": "User"},
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    inbox_dir = shared_dir / "inbox" / "sumire"

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

        files = list(inbox_dir.glob("*.json"))
        message = Message.model_validate_json(files[0].read_text(encoding="utf-8"))
        assert message.to_person == "sumire"
        assert message.source == "system"
        assert message.intent == "report"
        assert "【PR新規コミット検出（push静穏確認済み）】" in message.content
        assert "FutureSync/AI-Schreiber#42" in message.content
        assert sha[:8] in message.content

        state = json.loads(state_file.read_text(encoding="utf-8"))
        entry = state["prs"]["FutureSync/AI-Schreiber#42"]
        assert entry["sha"] == sha
        assert entry["notified_sha"] == sha

    assert manager._started is False
    assert manager._event_tasks == set()
    assert manager._debounce_tasks == {}
