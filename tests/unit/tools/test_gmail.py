# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for core/tools/gmail.py — Gmail integration.

We mock google API modules before importing core.tools.gmail
since the google packages are optional dependencies.
"""
from __future__ import annotations

import base64
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Mock google modules before importing core.tools.gmail ─────────

_google_mocks: dict[str, MagicMock] = {}

_GOOGLE_MODULES = [
    "google",
    "google.oauth2",
    "google.oauth2.credentials",
    "google.auth",
    "google.auth.transport",
    "google.auth.transport.requests",
    "google_auth_oauthlib",
    "google_auth_oauthlib.flow",
    "googleapiclient",
    "googleapiclient.discovery",
]


@pytest.fixture(autouse=True, scope="module")
def _mock_google_modules():
    """Inject mock google modules into sys.modules for the test session."""
    saved = {}
    for mod_name in _GOOGLE_MODULES:
        saved[mod_name] = sys.modules.get(mod_name)
        mock = MagicMock()
        sys.modules[mod_name] = mock
        _google_mocks[mod_name] = mock

    # Make sure Credentials class is accessible
    creds_mock = MagicMock()
    sys.modules["google.oauth2.credentials"].Credentials = creds_mock

    # Make sure build is accessible
    build_mock = MagicMock()
    sys.modules["googleapiclient.discovery"].build = build_mock

    # Reload the gmail module so it picks up our mocks
    if "core.tools.gmail" in sys.modules:
        importlib.reload(sys.modules["core.tools.gmail"])
    else:
        import core.tools.gmail  # noqa: F401

    yield

    # Restore
    for mod_name in _GOOGLE_MODULES:
        if saved[mod_name] is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = saved[mod_name]
    if "core.tools.gmail" in sys.modules:
        del sys.modules["core.tools.gmail"]


# Now import from core.tools.gmail (after mocks are in place)
# We use a function to get the module to avoid import-time issues

def _get_gmail():
    import core.tools.gmail as gmail_mod
    return gmail_mod


# ── Email dataclass ───────────────────────────────────────────────


class TestEmail:
    def test_creation(self):
        gmail = _get_gmail()
        email = gmail.Email(
            id="m1",
            thread_id="t1",
            from_addr="alice@example.com",
            subject="Test",
            snippet="Hello...",
        )
        assert email.id == "m1"
        assert email.body == ""

    def test_with_body(self):
        gmail = _get_gmail()
        email = gmail.Email(
            id="m1", thread_id="t1", from_addr="a@b.com",
            subject="S", snippet="snip", body="Full body text",
        )
        assert email.body == "Full body text"


# ── DraftResult dataclass ────────────────────────────────────────


class TestDraftResult:
    def test_success(self):
        gmail = _get_gmail()
        r = gmail.DraftResult(draft_id="d1", message_id="m1", success=True)
        assert r.success is True
        assert r.error is None

    def test_failure(self):
        gmail = _get_gmail()
        r = gmail.DraftResult(draft_id="", message_id="", success=False, error="oops")
        assert r.success is False
        assert r.error == "oops"


# ── GmailClient ──────────────────────────────────────────────────


class TestGmailClient:
    @pytest.fixture
    def client(self):
        gmail = _get_gmail()
        with patch.object(gmail.GmailClient, "_get_credentials", return_value=MagicMock()):
            mock_service = MagicMock()
            with patch.object(gmail, "build", return_value=mock_service):
                c = gmail.GmailClient()
                c._service = mock_service
                yield c, mock_service

    def test_get_unread_emails_success(self, client):
        c, mock_service = client
        gmail = _get_gmail()

        list_resp = {"messages": [{"id": "m1"}, {"id": "m2"}]}
        mock_service.users().messages().list().execute.return_value = list_resp

        def make_msg(msg_id, subject, from_addr):
            return {
                "id": msg_id,
                "threadId": f"t-{msg_id}",
                "snippet": f"snippet-{msg_id}",
                "payload": {
                    "headers": [
                        {"name": "From", "value": from_addr},
                        {"name": "Subject", "value": subject},
                    ],
                },
            }

        mock_service.users().messages().get().execute.side_effect = [
            make_msg("m1", "Subject 1", "alice@example.com"),
            make_msg("m2", "Subject 2", "bob@example.com"),
        ]

        emails = c.get_unread_emails(max_results=2)
        assert len(emails) == 2
        assert emails[0].subject == "Subject 1"

    def test_get_unread_emails_empty(self, client):
        c, mock_service = client
        mock_service.users().messages().list().execute.return_value = {"messages": []}
        assert c.get_unread_emails() == []

    def test_get_unread_emails_no_messages_key(self, client):
        c, mock_service = client
        mock_service.users().messages().list().execute.return_value = {}
        assert c.get_unread_emails() == []

    def test_get_unread_emails_api_error(self, client):
        c, mock_service = client
        mock_service.users().messages().list().execute.side_effect = Exception("API error")
        assert c.get_unread_emails() == []

    def test_get_email_body_plain_text(self, client):
        c, mock_service = client
        body_text = "Hello, this is the email body."
        encoded = base64.urlsafe_b64encode(body_text.encode()).decode()
        msg = {
            "payload": {
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": encoded}},
                ],
            },
        }
        mock_service.users().messages().get().execute.return_value = msg
        assert c.get_email_body("m1") == body_text

    def test_get_email_body_inline(self, client):
        c, mock_service = client
        body_text = "Inline body"
        encoded = base64.urlsafe_b64encode(body_text.encode()).decode()
        msg = {"payload": {"body": {"data": encoded}}}
        mock_service.users().messages().get().execute.return_value = msg
        assert c.get_email_body("m1") == body_text

    def test_get_email_body_empty(self, client):
        c, mock_service = client
        msg = {"payload": {"parts": []}}
        mock_service.users().messages().get().execute.return_value = msg
        assert c.get_email_body("m1") == ""

    def test_get_email_body_error(self, client):
        c, mock_service = client
        mock_service.users().messages().get().execute.side_effect = Exception("err")
        assert c.get_email_body("m1") == ""

    def test_create_draft_success(self, client):
        c, mock_service = client
        mock_service.users().drafts().create().execute.return_value = {
            "id": "d1",
            "message": {"id": "m1"},
        }
        result = c.create_draft(to="r@test.com", subject="Test", body="Body")
        assert result.success is True
        assert result.draft_id == "d1"

    def test_create_draft_with_thread(self, client):
        c, mock_service = client
        mock_service.users().drafts().create().execute.return_value = {
            "id": "d2",
            "message": {"id": "m2"},
        }
        result = c.create_draft(
            to="r@test.com", subject="Re: Thread", body="Reply",
            thread_id="t1", in_reply_to="<msg@example.com>",
        )
        assert result.success is True

    def test_create_draft_error(self, client):
        c, mock_service = client
        mock_service.users().drafts().create().execute.side_effect = Exception("Draft error")
        result = c.create_draft(to="a@b.com", subject="S", body="B")
        assert result.success is False
        assert "Draft error" in result.error

    def test_get_attachments(self, client, tmp_path: Path):
        c, mock_service = client
        msg = {
            "payload": {
                "parts": [
                    {"filename": "test.txt", "body": {"attachmentId": "att1"}},
                ],
            },
        }
        mock_service.users().messages().get().execute.return_value = msg
        att_data = base64.urlsafe_b64encode(b"file content").decode()
        mock_service.users().messages().attachments().get().execute.return_value = {
            "data": att_data,
        }
        attachments = c.get_attachments("m1", tmp_path)
        assert len(attachments) == 1
        assert attachments[0][0] == "test.txt"
        assert (tmp_path / "test.txt").read_bytes() == b"file content"

    def test_get_attachments_error(self, client, tmp_path: Path):
        c, mock_service = client
        mock_service.users().messages().get().execute.side_effect = Exception("error")
        assert c.get_attachments("m1", tmp_path) == []

    def test_extract_body_multipart(self, client):
        c, _ = client
        body_text = "Nested body"
        encoded = base64.urlsafe_b64encode(body_text.encode()).decode()
        payload = {
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": encoded}},
                    ],
                    "body": {},
                },
            ],
        }
        assert c._extract_body(payload) == body_text


# ── get_tool_schemas ──────────────────────────────────────────────


class TestGetToolSchemas:
    def test_returns_schemas(self):
        gmail = _get_gmail()
        schemas = gmail.get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 3
        names = {s["name"] for s in schemas}
        assert names == {"gmail_unread", "gmail_read_body", "gmail_draft"}

    def test_gmail_draft_requires_to_subject_body(self):
        gmail = _get_gmail()
        schemas = gmail.get_tool_schemas()
        draft = [s for s in schemas if s["name"] == "gmail_draft"][0]
        assert set(draft["input_schema"]["required"]) == {"to", "subject", "body"}
