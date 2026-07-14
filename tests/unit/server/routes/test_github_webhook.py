"""Unit tests for the GitHub webhook endpoint."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes import webhooks
from server.routes.webhooks import create_webhooks_router

GITHUB_SECRET = secrets.token_hex(32)


def _signature(body: bytes, *, secret: str = GITHUB_SECRET) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _headers(body: bytes, event: str = "pull_request") -> dict[str, str]:
    return {
        "X-GitHub-Event": event,
        "X-Hub-Signature-256": _signature(body),
    }


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(webhooks, "_github_webhook_secret", lambda: GITHUB_SECRET)
    test_app = FastAPI()
    test_app.include_router(create_webhooks_router(), prefix="/api")
    return test_app


@pytest.fixture
def client(app: FastAPI):
    with TestClient(app) as test_client:
        yield test_client


class TestGitHubSignatureVerification:
    def test_valid_signature_is_accepted(self, client: TestClient) -> None:
        body = json.dumps({"repository": {"full_name": "owner/repo"}}).encode()

        response = client.post("/api/webhooks/github", content=body, headers=_headers(body))

        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_invalid_signature_is_rejected(self, client: TestClient) -> None:
        body = json.dumps({"repository": {"full_name": "owner/repo"}}).encode()
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": "sha256=invalid",
        }

        response = client.post("/api/webhooks/github", content=body, headers=headers)

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid signature"

    def test_missing_signature_is_rejected(self, client: TestClient) -> None:
        body = json.dumps({"repository": {"full_name": "owner/repo"}}).encode()

        response = client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "pull_request"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid signature"

    def test_missing_secret_returns_service_unavailable(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(webhooks, "_github_webhook_secret", lambda: None)
        body = json.dumps({"zen": "Keep it logically awesome."}).encode()

        response = client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "ping"},
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "GitHub webhook not configured"

    def test_invalid_json_is_not_examined_before_signature(self, client: TestClient) -> None:
        body = b"not-json"

        response = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": "sha256=invalid",
            },
        )

        assert response.status_code == 401

    def test_signed_invalid_json_is_rejected_after_signature(self, client: TestClient) -> None:
        body = b"not-json"

        response = client.post("/api/webhooks/github", content=body, headers=_headers(body))

        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid JSON"


class TestGitHubPing:
    def test_ping_requires_a_valid_signature(self, client: TestClient) -> None:
        body = json.dumps({"zen": "Keep it logically awesome."}).encode()

        response = client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "ping"},
        )

        assert response.status_code == 401

    def test_signed_ping_does_not_dispatch(self, app: FastAPI, client: TestClient) -> None:
        manager = MagicMock()
        app.state.github_gateway_manager = manager
        body = json.dumps({"zen": "Keep it logically awesome."}).encode()

        response = client.post(
            "/api/webhooks/github",
            content=body,
            headers=_headers(body, event="ping"),
        )

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        manager.dispatch_event.assert_not_called()


class TestGitHubManagerDispatch:
    def test_valid_event_is_dispatched_to_manager(self, app: FastAPI, client: TestClient) -> None:
        manager = MagicMock()
        app.state.github_gateway_manager = manager
        payload = {
            "action": "synchronize",
            "repository": {"full_name": "owner/repo"},
            "pull_request": {"number": 42},
        }
        body = json.dumps(payload).encode()

        response = client.post(
            "/api/webhooks/github",
            content=body,
            headers=_headers(body, event="pull_request"),
        )

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        manager.dispatch_event.assert_called_once_with("pull_request", payload)

    def test_invalid_signature_never_dispatches(self, app: FastAPI, client: TestClient) -> None:
        manager = MagicMock()
        app.state.github_gateway_manager = manager
        body = json.dumps({"action": "synchronize"}).encode()

        response = client.post(
            "/api/webhooks/github",
            content=body,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": "sha256=invalid",
            },
        )

        assert response.status_code == 401
        manager.dispatch_event.assert_not_called()

    def test_manager_absent_is_accepted(self, client: TestClient) -> None:
        payload = {
            "action": "synchronize",
            "repository": {"full_name": "owner/repo"},
        }
        body = json.dumps(payload).encode()

        response = client.post(
            "/api/webhooks/github",
            content=body,
            headers=_headers(body, event="pull_request"),
        )

        assert response.status_code == 200
        assert response.json() == {"ok": True}
