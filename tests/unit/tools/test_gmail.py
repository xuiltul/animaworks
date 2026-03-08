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
        assert email.to_addr == ""
        assert email.date == ""
        assert email.label_ids is None

    def test_with_body(self):
        gmail = _get_gmail()
        email = gmail.Email(
            id="m1", thread_id="t1", from_addr="a@b.com",
            subject="S", snippet="snip", body="Full body text",
        )
        assert email.body == "Full body text"

    def test_with_new_fields(self):
        gmail = _get_gmail()
        email = gmail.Email(
            id="m1", thread_id="t1", from_addr="a@b.com",
            subject="S", snippet="snip",
            to_addr="bob@example.com",
            date="Thu, 5 Mar 2026 10:00:00 +0900",
            label_ids=["INBOX", "UNREAD"],
        )
        assert email.to_addr == "bob@example.com"
        assert email.date == "Thu, 5 Mar 2026 10:00:00 +0900"
        assert email.label_ids == ["INBOX", "UNREAD"]

    def test_backward_compatibility(self):
        """Existing code that doesn't pass new fields should still work."""
        gmail = _get_gmail()
        email = gmail.Email(
            id="m1", thread_id="t1", from_addr="a@b.com",
            subject="S", snippet="snip",
        )
        assert email.to_addr == ""
        assert email.date == ""
        assert email.label_ids is None


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


# ── Helper: build mock Gmail API message ─────────────────────────

def _make_msg(
    msg_id: str,
    subject: str,
    from_addr: str,
    to_addr: str = "recipient@example.com",
    date: str = "Thu, 5 Mar 2026 10:00:00 +0900",
    label_ids: list[str] | None = None,
) -> dict:
    """Build a mock Gmail API message response."""
    msg: dict = {
        "id": msg_id,
        "threadId": f"t-{msg_id}",
        "snippet": f"snippet-{msg_id}",
        "payload": {
            "headers": [
                {"name": "From", "value": from_addr},
                {"name": "To", "value": to_addr},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": date},
            ],
        },
    }
    if label_ids is not None:
        msg["labelIds"] = label_ids
    return msg


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

    # ── get_unread_emails ────────────────────────────────────

    def test_get_unread_emails_success(self, client):
        c, mock_service = client

        list_resp = {"messages": [{"id": "m1"}, {"id": "m2"}]}
        mock_service.users().messages().list().execute.return_value = list_resp

        mock_service.users().messages().get().execute.side_effect = [
            _make_msg("m1", "Subject 1", "alice@example.com",
                      label_ids=["INBOX", "UNREAD"]),
            _make_msg("m2", "Subject 2", "bob@example.com",
                      label_ids=["INBOX", "UNREAD"]),
        ]

        emails = c.get_unread_emails(max_results=2)
        assert len(emails) == 2
        assert emails[0].subject == "Subject 1"
        assert emails[0].to_addr == "recipient@example.com"
        assert emails[0].date == "Thu, 5 Mar 2026 10:00:00 +0900"
        assert emails[0].label_ids == ["INBOX", "UNREAD"]

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

    # ── get_inbox_emails ─────────────────────────────────────

    def test_get_inbox_emails_success(self, client):
        c, mock_service = client

        list_resp = {"messages": [{"id": "m1"}, {"id": "m2"}]}
        mock_service.users().messages().list().execute.return_value = list_resp

        mock_service.users().messages().get().execute.side_effect = [
            _make_msg("m1", "Inbox 1", "alice@example.com",
                      label_ids=["INBOX"]),
            _make_msg("m2", "Inbox 2", "bob@example.com",
                      label_ids=["INBOX", "UNREAD"]),
        ]

        emails = c.get_inbox_emails(max_results=2)
        assert len(emails) == 2
        assert emails[0].subject == "Inbox 1"
        assert emails[0].from_addr == "alice@example.com"
        assert emails[0].to_addr == "recipient@example.com"
        assert emails[1].label_ids == ["INBOX", "UNREAD"]

    def test_get_inbox_emails_empty(self, client):
        c, mock_service = client
        mock_service.users().messages().list().execute.return_value = {"messages": []}
        assert c.get_inbox_emails() == []

    def test_get_inbox_emails_no_messages_key(self, client):
        c, mock_service = client
        mock_service.users().messages().list().execute.return_value = {}
        assert c.get_inbox_emails() == []

    def test_get_inbox_emails_api_error(self, client):
        c, mock_service = client
        mock_service.users().messages().list().execute.side_effect = Exception("API error")
        assert c.get_inbox_emails() == []

    # ── get_sent_emails ──────────────────────────────────────

    def test_get_sent_emails_success(self, client):
        c, mock_service = client

        list_resp = {"messages": [{"id": "s1"}]}
        mock_service.users().messages().list().execute.return_value = list_resp

        mock_service.users().messages().get().execute.side_effect = [
            _make_msg("s1", "Sent Subject", "me@example.com",
                      to_addr="bob@example.com",
                      label_ids=["SENT"]),
        ]

        emails = c.get_sent_emails(max_results=5)
        assert len(emails) == 1
        assert emails[0].subject == "Sent Subject"
        assert emails[0].from_addr == "me@example.com"
        assert emails[0].to_addr == "bob@example.com"
        assert emails[0].label_ids == ["SENT"]

    def test_get_sent_emails_empty(self, client):
        c, mock_service = client
        mock_service.users().messages().list().execute.return_value = {"messages": []}
        assert c.get_sent_emails() == []

    def test_get_sent_emails_api_error(self, client):
        c, mock_service = client
        mock_service.users().messages().list().execute.side_effect = Exception("err")
        assert c.get_sent_emails() == []

    # ── search_emails ────────────────────────────────────────

    def test_search_emails_success(self, client):
        c, mock_service = client

        list_resp = {"messages": [{"id": "q1"}, {"id": "q2"}]}
        mock_service.users().messages().list().execute.return_value = list_resp

        mock_service.users().messages().get().execute.side_effect = [
            _make_msg("q1", "Report Q1", "alice@example.com"),
            _make_msg("q2", "Report Q2", "alice@example.com"),
        ]

        emails = c.search_emails(query="from:alice subject:report", max_results=10)
        assert len(emails) == 2
        assert emails[0].subject == "Report Q1"
        assert emails[1].id == "q2"

    def test_search_emails_no_results(self, client):
        c, mock_service = client
        mock_service.users().messages().list().execute.return_value = {"messages": []}
        assert c.search_emails(query="nonexistent") == []

    def test_search_emails_no_messages_key(self, client):
        c, mock_service = client
        mock_service.users().messages().list().execute.return_value = {}
        assert c.search_emails(query="test") == []

    def test_search_emails_api_error(self, client):
        c, mock_service = client
        mock_service.users().messages().list().execute.side_effect = Exception("search error")
        assert c.search_emails(query="test") == []

    # ── _get_email_details ───────────────────────────────────

    def test_get_email_details_extracts_all_headers(self, client):
        c, mock_service = client

        mock_service.users().messages().get().execute.return_value = _make_msg(
            "d1", "Detail Subject", "sender@example.com",
            to_addr="dest@example.com",
            date="Wed, 4 Mar 2026 09:00:00 +0000",
            label_ids=["INBOX", "IMPORTANT"],
        )

        email = c._get_email_details("d1")
        assert email is not None
        assert email.id == "d1"
        assert email.from_addr == "sender@example.com"
        assert email.to_addr == "dest@example.com"
        assert email.subject == "Detail Subject"
        assert email.date == "Wed, 4 Mar 2026 09:00:00 +0000"
        assert email.label_ids == ["INBOX", "IMPORTANT"]

    def test_get_email_details_missing_headers(self, client):
        c, mock_service = client

        msg = {
            "id": "d2",
            "threadId": "t-d2",
            "snippet": "snippet-d2",
            "payload": {
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                ],
            },
        }
        mock_service.users().messages().get().execute.return_value = msg

        email = c._get_email_details("d2")
        assert email is not None
        assert email.to_addr == ""
        assert email.date == ""
        assert email.label_ids is None

    def test_get_email_details_error(self, client):
        c, mock_service = client
        mock_service.users().messages().get().execute.side_effect = Exception("err")
        assert c._get_email_details("bad") is None

    # ── get_email_body ───────────────────────────────────────

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

    # ── create_draft ─────────────────────────────────────────

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

    # ── get_attachments ──────────────────────────────────────

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

    # ── _extract_body ────────────────────────────────────────

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
    def test_returns_empty_list(self):
        """External tool modules' get_tool_schemas() returns empty list."""
        gmail = _get_gmail()
        schemas = gmail.get_tool_schemas()
        assert isinstance(schemas, list)
        assert schemas == []


# ── Email.to_dict ─────────────────────────────────────────────────


class TestEmailToDict:
    def test_basic_conversion(self):
        gmail = _get_gmail()
        email = gmail.Email(
            id="m1", thread_id="t1", from_addr="a@b.com",
            subject="Sub", snippet="snip",
            to_addr="c@d.com", date="Thu, 5 Mar 2026 10:00:00 +0900",
            label_ids=["INBOX"],
        )
        d = email.to_dict()
        assert d["id"] == "m1"
        assert d["thread_id"] == "t1"
        assert d["from"] == "a@b.com"
        assert d["to"] == "c@d.com"
        assert d["subject"] == "Sub"
        assert d["snippet"] == "snip"
        assert d["date"] == "Thu, 5 Mar 2026 10:00:00 +0900"
        assert d["label_ids"] == ["INBOX"]
        assert "body" not in d
        assert "from_addr" not in d
        assert "to_addr" not in d

    def test_no_label_ids(self):
        gmail = _get_gmail()
        email = gmail.Email(
            id="m1", thread_id="t1", from_addr="a@b.com",
            subject="Sub", snippet="snip",
        )
        d = email.to_dict()
        assert "label_ids" not in d

    def test_empty_label_ids(self):
        gmail = _get_gmail()
        email = gmail.Email(
            id="m1", thread_id="t1", from_addr="a@b.com",
            subject="Sub", snippet="snip", label_ids=[],
        )
        d = email.to_dict()
        assert "label_ids" not in d


# ── dispatch ──────────────────────────────────────────────────────


class TestDispatch:
    @pytest.fixture
    def mock_client(self):
        gmail = _get_gmail()
        with patch.object(gmail.GmailClient, "_get_credentials", return_value=MagicMock()):
            mock_service = MagicMock()
            with patch.object(gmail, "build", return_value=mock_service):
                with patch.object(gmail, "GmailClient") as MockCls:
                    instance = MagicMock()
                    MockCls.return_value = instance
                    yield instance

    def test_dispatch_gmail_unread(self, mock_client):
        gmail = _get_gmail()
        mock_client.get_unread_emails.return_value = [
            gmail.Email(id="m1", thread_id="t1", from_addr="a@b.com",
                        subject="S", snippet="snip", to_addr="c@d.com",
                        date="date1", label_ids=["INBOX", "UNREAD"]),
        ]
        result = gmail.dispatch("gmail_unread", {"max_results": 5})
        mock_client.get_unread_emails.assert_called_once_with(max_results=5)
        assert len(result) == 1
        assert result[0]["from"] == "a@b.com"
        assert result[0]["to"] == "c@d.com"

    def test_dispatch_gmail_inbox(self, mock_client):
        gmail = _get_gmail()
        mock_client.get_inbox_emails.return_value = [
            gmail.Email(id="m1", thread_id="t1", from_addr="a@b.com",
                        subject="S", snippet="snip"),
        ]
        result = gmail.dispatch("gmail_inbox", {"max_results": 10})
        mock_client.get_inbox_emails.assert_called_once_with(max_results=10)
        assert len(result) == 1

    def test_dispatch_gmail_sent(self, mock_client):
        gmail = _get_gmail()
        mock_client.get_sent_emails.return_value = [
            gmail.Email(id="s1", thread_id="t1", from_addr="me@b.com",
                        subject="Sent", snippet="snip", to_addr="bob@b.com"),
        ]
        result = gmail.dispatch("gmail_sent", {})
        mock_client.get_sent_emails.assert_called_once_with(max_results=20)
        assert result[0]["to"] == "bob@b.com"

    def test_dispatch_gmail_search(self, mock_client):
        gmail = _get_gmail()
        mock_client.search_emails.return_value = [
            gmail.Email(id="q1", thread_id="t1", from_addr="a@b.com",
                        subject="Found", snippet="snip"),
        ]
        result = gmail.dispatch("gmail_search", {"query": "from:a", "max_results": 3})
        mock_client.search_emails.assert_called_once_with(query="from:a", max_results=3)
        assert len(result) == 1
        assert result[0]["subject"] == "Found"

    def test_dispatch_gmail_read_body(self, mock_client):
        gmail = _get_gmail()
        mock_client.get_email_body.return_value = "body text"
        result = gmail.dispatch("gmail_read_body", {"message_id": "m1"})
        mock_client.get_email_body.assert_called_once_with("m1")
        assert result == "body text"

    def test_dispatch_gmail_draft(self, mock_client):
        gmail = _get_gmail()
        mock_client.create_draft.return_value = gmail.DraftResult(
            draft_id="d1", message_id="m1", success=True,
        )
        result = gmail.dispatch("gmail_draft", {
            "to": "r@t.com", "subject": "S", "body": "B",
        })
        assert result["success"] is True
        assert result["draft_id"] == "d1"

    def test_dispatch_unknown_tool(self):
        gmail = _get_gmail()
        with pytest.raises(ValueError, match="Unknown tool"):
            gmail.dispatch("gmail_nonexistent", {})


# ── EXECUTION_PROFILE ─────────────────────────────────────────────


class TestExecutionProfile:
    def test_has_all_subcommands(self):
        gmail = _get_gmail()
        expected = {"unread", "inbox", "sent", "search", "read", "draft", "send"}
        assert set(gmail.EXECUTION_PROFILE.keys()) == expected

    def test_profile_values(self):
        gmail = _get_gmail()
        for key, profile in gmail.EXECUTION_PROFILE.items():
            assert "expected_seconds" in profile
            assert "background_eligible" in profile
            assert isinstance(profile["expected_seconds"], int)


# ── get_cli_guide ─────────────────────────────────────────────────


class TestGetCliGuide:
    def test_includes_new_subcommands(self):
        gmail = _get_gmail()
        guide = gmail.get_cli_guide()
        assert "gmail inbox" in guide
        assert "gmail sent" in guide
        assert "gmail search" in guide
        assert "gmail unread" in guide
        assert "gmail read" in guide
        assert "gmail draft" in guide


# ── cli_main ──────────────────────────────────────────────────────


class TestCliMain:
    @pytest.fixture
    def mock_gmail_client(self):
        gmail = _get_gmail()
        with patch.object(gmail, "GmailClient") as MockCls:
            instance = MagicMock()
            MockCls.return_value = instance
            yield instance

    def test_cli_inbox(self, mock_gmail_client, capsys):
        gmail = _get_gmail()
        mock_gmail_client.get_inbox_emails.return_value = [
            gmail.Email(id="m1", thread_id="t1", from_addr="a@b.com",
                        subject="Inbox Sub", snippet="snip",
                        to_addr="c@d.com", date="date1"),
        ]
        gmail.cli_main(["inbox", "-n", "5"])
        mock_gmail_client.get_inbox_emails.assert_called_once_with(max_results=5)
        captured = capsys.readouterr()
        assert "Inbox Sub" in captured.out

    def test_cli_sent(self, mock_gmail_client, capsys):
        gmail = _get_gmail()
        mock_gmail_client.get_sent_emails.return_value = [
            gmail.Email(id="s1", thread_id="t1", from_addr="me@b.com",
                        subject="Sent Sub", snippet="snip",
                        to_addr="bob@b.com", date="date1"),
        ]
        gmail.cli_main(["sent", "-n", "3"])
        mock_gmail_client.get_sent_emails.assert_called_once_with(max_results=3)
        captured = capsys.readouterr()
        assert "Sent Sub" in captured.out

    def test_cli_search(self, mock_gmail_client, capsys):
        gmail = _get_gmail()
        mock_gmail_client.search_emails.return_value = [
            gmail.Email(id="q1", thread_id="t1", from_addr="a@b.com",
                        subject="Found", snippet="snip",
                        to_addr="c@d.com", date="date1"),
        ]
        gmail.cli_main(["search", "from:alice", "-n", "10"])
        mock_gmail_client.search_emails.assert_called_once_with(
            query="from:alice", max_results=10,
        )
        captured = capsys.readouterr()
        assert "Found" in captured.out

    def test_cli_inbox_empty(self, mock_gmail_client, capsys):
        gmail = _get_gmail()
        mock_gmail_client.get_inbox_emails.return_value = []
        gmail.cli_main(["inbox"])
        captured = capsys.readouterr()
        assert "No inbox emails" in captured.out

    def test_cli_sent_empty(self, mock_gmail_client, capsys):
        gmail = _get_gmail()
        mock_gmail_client.get_sent_emails.return_value = []
        gmail.cli_main(["sent"])
        captured = capsys.readouterr()
        assert "No sent emails" in captured.out

    def test_cli_search_empty(self, mock_gmail_client, capsys):
        gmail = _get_gmail()
        mock_gmail_client.search_emails.return_value = []
        gmail.cli_main(["search", "nothing"])
        captured = capsys.readouterr()
        assert "No search emails" in captured.out

    def test_cli_no_command(self, mock_gmail_client):
        gmail = _get_gmail()
        with pytest.raises(SystemExit):
            gmail.cli_main([])
