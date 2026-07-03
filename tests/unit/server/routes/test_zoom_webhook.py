"""Unit tests for server/routes/webhooks.py — Zoom RTMS webhook endpoint."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes.webhooks import create_webhooks_router

# ── Test App Setup ────────────────────────────────────────


@pytest.fixture
def app():
    """Create a minimal FastAPI app with the webhooks router."""
    app = FastAPI()
    api_router = create_webhooks_router()
    app.include_router(api_router, prefix="/api")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ── Helpers ───────────────────────────────────────────────

ZOOM_SECRET = "zoom_secret_token_abcdef123456"


def _zoom_signature(body: bytes, timestamp: str) -> str:
    """Generate a valid Zoom ``x-zm-signature`` for testing."""
    message = f"v0:{timestamp}:{body.decode('utf-8')}"
    return "v0=" + hmac.new(
        ZOOM_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _zoom_headers(body: bytes, *, timestamp: str | None = None) -> dict[str, str]:
    """Generate valid Zoom request headers."""
    ts = timestamp if timestamp is not None else str(int(time.time()))
    return {
        "x-zm-signature": _zoom_signature(body, ts),
        "x-zm-request-timestamp": ts,
    }


# ── URL Validation Challenge ──────────────────────────────


class TestZoomUrlValidation:
    @patch("server.routes.webhooks.get_credential", return_value=ZOOM_SECRET)
    def test_url_validation_returns_encrypted_token(self, mock_cred, client):
        """A *signed* endpoint.url_validation echoes plainToken + HMAC-SHA256 encryptedToken."""
        plain = "abc123plaintoken"
        payload = json.dumps({
            "event": "endpoint.url_validation",
            "payload": {"plainToken": plain},
        }).encode("utf-8")

        resp = client.post("/api/webhooks/zoom", content=payload, headers=_zoom_headers(payload))
        assert resp.status_code == 200

        expected = hmac.new(
            ZOOM_SECRET.encode("utf-8"),
            plain.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        assert resp.json() == {"plainToken": plain, "encryptedToken": expected}

    @patch("server.routes.webhooks.get_credential", return_value=ZOOM_SECRET)
    def test_unsigned_url_validation_rejected(self, mock_cred, client):
        """An unsigned validation challenge is rejected 401 (no signing oracle)."""
        payload = json.dumps({
            "event": "endpoint.url_validation",
            "payload": {"plainToken": "tok"},
        }).encode("utf-8")
        resp = client.post("/api/webhooks/zoom", content=payload)
        assert resp.status_code == 401

    @patch("server.routes.webhooks.get_credential", return_value=ZOOM_SECRET)
    def test_forged_signature_url_validation_rejected(self, mock_cred, client):
        """A validation challenge with a wrong signature is rejected 401."""
        payload = json.dumps({
            "event": "endpoint.url_validation",
            "payload": {"plainToken": "tok"},
        }).encode("utf-8")
        headers = {
            "x-zm-signature": "v0=deadbeef",
            "x-zm-request-timestamp": str(int(time.time())),
        }
        resp = client.post("/api/webhooks/zoom", content=payload, headers=headers)
        assert resp.status_code == 401


# ── Signature Verification ────────────────────────────────


class TestZoomSignatureVerification:
    @patch("server.routes.webhooks.get_credential", return_value=ZOOM_SECRET)
    def test_valid_signature_accepted(self, mock_cred, client):
        """A correctly signed rtms event returns 200 (manager absent → ignored)."""
        body = json.dumps({
            "event": "meeting.rtms_started",
            "payload": {"object": {"meeting_uuid": "u1", "rtms_stream_id": "s1"}},
        }).encode("utf-8")
        resp = client.post("/api/webhooks/zoom", content=body, headers=_zoom_headers(body))
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    @patch("server.routes.webhooks.get_credential", return_value=ZOOM_SECRET)
    def test_invalid_signature_rejected(self, mock_cred, client):
        """A wrong signature is rejected with 401."""
        body = json.dumps({
            "event": "meeting.rtms_started",
            "payload": {"object": {"meeting_uuid": "u1"}},
        }).encode("utf-8")
        headers = {
            "x-zm-signature": "v0=deadbeef",
            "x-zm-request-timestamp": str(int(time.time())),
        }
        resp = client.post("/api/webhooks/zoom", content=body, headers=headers)
        assert resp.status_code == 401

    @patch("server.routes.webhooks.get_credential", return_value=ZOOM_SECRET)
    def test_replay_old_timestamp_rejected(self, mock_cred, client):
        """A request timestamp older than 5 minutes is rejected with 401."""
        old_ts = str(int(time.time()) - 600)  # 10 minutes ago
        body = json.dumps({
            "event": "meeting.rtms_started",
            "payload": {"object": {"meeting_uuid": "u1"}},
        }).encode("utf-8")
        resp = client.post(
            "/api/webhooks/zoom", content=body, headers=_zoom_headers(body, timestamp=old_ts),
        )
        assert resp.status_code == 401


# ── Manager Delegation ────────────────────────────────────


class TestZoomManagerDelegation:
    @patch("server.routes.webhooks.get_credential", return_value=ZOOM_SECRET)
    def test_rtms_started_delegated_to_manager(self, mock_cred, app, client):
        obj = {
            "meeting_uuid": "u-123",
            "meeting_id": "999",
            "rtms_stream_id": "stream-1",
            "server_urls": "wss://sig.example",
            "topic": "Weekly Sync",
        }
        manager = AsyncMock()
        app.state.zoom_gateway_manager = manager

        body = json.dumps({"event": "meeting.rtms_started", "payload": {"object": obj}}).encode("utf-8")
        resp = client.post("/api/webhooks/zoom", content=body, headers=_zoom_headers(body))
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        manager.handle_rtms_started.assert_awaited_once_with(obj)
        manager.handle_rtms_stopped.assert_not_awaited()

    @patch("server.routes.webhooks.get_credential", return_value=ZOOM_SECRET)
    def test_rtms_stopped_delegated_to_manager(self, mock_cred, app, client):
        obj = {"meeting_uuid": "u-123"}
        manager = AsyncMock()
        app.state.zoom_gateway_manager = manager

        body = json.dumps({"event": "meeting.rtms_stopped", "payload": {"object": obj}}).encode("utf-8")
        resp = client.post("/api/webhooks/zoom", content=body, headers=_zoom_headers(body))
        assert resp.status_code == 200
        manager.handle_rtms_stopped.assert_awaited_once_with(obj)
        manager.handle_rtms_started.assert_not_awaited()

    @patch("server.routes.webhooks.get_credential", return_value=ZOOM_SECRET)
    def test_manager_absent_ignores_event(self, mock_cred, client):
        """When no RTMS manager is running the event is accepted and ignored."""
        body = json.dumps({
            "event": "meeting.rtms_started",
            "payload": {"object": {"meeting_uuid": "u1", "rtms_stream_id": "s1"}},
        }).encode("utf-8")
        resp = client.post("/api/webhooks/zoom", content=body, headers=_zoom_headers(body))
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    @patch("server.routes.webhooks.get_credential", return_value=ZOOM_SECRET)
    def test_unknown_event_is_noop(self, mock_cred, app, client):
        """An unrecognised event with a valid signature touches neither handler."""
        manager = AsyncMock()
        app.state.zoom_gateway_manager = manager
        body = json.dumps({
            "event": "meeting.participant_joined",
            "payload": {"object": {"meeting_uuid": "u1"}},
        }).encode("utf-8")
        resp = client.post("/api/webhooks/zoom", content=body, headers=_zoom_headers(body))
        assert resp.status_code == 200
        manager.handle_rtms_started.assert_not_awaited()
        manager.handle_rtms_stopped.assert_not_awaited()
