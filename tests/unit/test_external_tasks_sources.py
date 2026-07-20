# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for external task source collectors (mocked I/O only)."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.exceptions import ToolConfigError
from core.external_tasks.collector import CredentialNotFoundError
from core.external_tasks.sources import chatwork as chatwork_src
from core.external_tasks.sources import github as github_src
from core.external_tasks.sources import gmail as gmail_src
from core.external_tasks.sources import slack as slack_src


# ── GitHub ──────────────────────────────────────────────


def _gh_auth_ok(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    if cmd[:3] == ["gh", "auth", "status"]:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    raise AssertionError(f"unexpected gh command: {cmd}")


def test_github_collects_prs_and_issues() -> None:
    pr_payload = [
        {
            "number": 10,
            "title": "Fix login",
            "url": "https://github.com/acme/app/pull/10",
            "createdAt": "2026-07-18T01:00:00Z",
            "updatedAt": "2026-07-19T02:00:00Z",
            "repository": {"name": "app", "nameWithOwner": "acme/app"},
        }
    ]
    issue_payload = [
        {
            "number": 3,
            "title": "Bug report",
            "url": "https://github.com/acme/app/issues/3",
            "createdAt": "2026-07-17T01:00:00Z",
            "updatedAt": "2026-07-18T02:00:00Z",
            "repository": {"name": "app", "nameWithOwner": "acme/app"},
        }
    ]

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[1:3] == ["search", "prs"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(pr_payload), "")
        if cmd[1:3] == ["search", "issues"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(issue_payload), "")
        raise AssertionError(f"unexpected gh command: {cmd}")

    with patch("core.external_tasks.sources.github.subprocess.run", side_effect=fake_run):
        tasks = github_src.collect_github()

    assert len(tasks) == 2
    pr = next(t for t in tasks if t.id.startswith("github-pr-"))
    issue = next(t for t in tasks if t.id.startswith("github-issue-"))

    assert pr.id == "github-pr-acme-app-10"
    assert pr.priority == 90
    assert pr.title == "app #10: Fix login"
    assert pr.source_type == "github"
    assert pr.source_icon == "github"
    assert pr.source_url == "https://github.com/acme/app/pull/10"
    assert pr.created_at == "2026-07-18T01:00:00Z"
    assert pr.last_updated_at == "2026-07-19T02:00:00Z"
    assert pr.status == "open"

    assert issue.id == "github-issue-acme-app-3"
    assert issue.priority == 75
    assert issue.title == "app #3: Bug report"


def test_github_credential_missing_when_gh_not_installed() -> None:
    with patch(
        "core.external_tasks.sources.github.subprocess.run",
        side_effect=FileNotFoundError("gh"),
    ):
        with pytest.raises(CredentialNotFoundError):
            github_src.collect_github()


def test_github_credential_missing_when_unauthenticated() -> None:
    with patch(
        "core.external_tasks.sources.github.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["gh", "auth", "status"]),
    ):
        with pytest.raises(CredentialNotFoundError):
            github_src.collect_github()


def test_github_api_error_propagates() -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "API rate limit exceeded")

    with patch("core.external_tasks.sources.github.subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="gh command failed"):
            github_src.collect_github()


def test_github_id_deterministic() -> None:
    payload = [
        {
            "number": 42,
            "title": "X",
            "url": "https://github.com/o/r/pull/42",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-02T00:00:00Z",
            "repository": {"name": "r", "nameWithOwner": "o/r"},
        }
    ]

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[1:3] == ["search", "prs"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")
        if cmd[1:3] == ["search", "issues"]:
            return subprocess.CompletedProcess(cmd, 0, "[]", "")
        raise AssertionError(cmd)

    with patch("core.external_tasks.sources.github.subprocess.run", side_effect=fake_run):
        a = github_src.collect_github()
        b = github_src.collect_github()
    assert a[0].id == b[0].id == "github-pr-o-r-42"


# ── Slack ───────────────────────────────────────────────


def test_slack_collects_unreplied_mentions() -> None:
    mock_client = MagicMock()
    mock_client.my_user_id = "Ume"
    mock_client.auth_test.return_value = {"user_id": "Ume"}
    mock_client.get_channel_name.return_value = "ops"

    # Use a near-current epoch so the 7-day filter accepts it.
    import time

    ts = f"{time.time() - 3600:.6f}"
    mention = {
        "channel_id": "C1",
        "ts": ts,
        "ts_epoch": float(ts),
        "user_id": "Uother",
        "text": "please check this deploy",
        "channel_name": "ops",
        "thread_ts": "",
    }

    mock_cache = MagicMock()
    mock_cache.find_unreplied.return_value = [mention]

    with (
        patch("core.tools._slack_client.SlackClient", return_value=mock_client),
        patch("core.tools._slack_cache.MessageCache", return_value=mock_cache),
    ):
        tasks = slack_src.collect_slack()

    assert len(tasks) == 1
    t = tasks[0]
    assert t.id == f"slack-C1-{ts}"
    assert t.priority == 80
    assert t.title == "#ops: please check this deploy"
    assert t.source_type == "slack"
    expected_url = f"https://slack.com/archives/C1/p{ts.replace('.', '')}"
    assert t.source_url == expected_url
    mock_cache.close.assert_called_once()
    # Deterministic permalink: no chat.getPermalink / private _call.
    mock_client._call.assert_not_called()


def test_slack_credential_missing() -> None:
    with patch(
        "core.tools._slack_client.SlackClient",
        side_effect=ToolConfigError("missing slack token"),
    ):
        with pytest.raises(CredentialNotFoundError, match="missing slack token"):
            slack_src.collect_slack()


def test_slack_api_error_propagates() -> None:
    mock_client = MagicMock()
    mock_client.auth_test.side_effect = RuntimeError("slack down")

    with (
        patch("core.tools._slack_client.SlackClient", return_value=mock_client),
        patch("core.tools._slack_cache.MessageCache"),
    ):
        with pytest.raises(RuntimeError, match="slack down"):
            slack_src.collect_slack()


def test_slack_is_actionable_mention_filters_old_and_self() -> None:
    import time

    now = time.time()
    cutoff = now - 7 * 86400
    old = {"user_id": "U2", "ts_epoch": now - 10 * 86400, "ts": str(now - 10 * 86400)}
    own = {"user_id": "Ume", "ts_epoch": now - 100, "ts": str(now - 100)}
    ok = {"user_id": "U2", "ts_epoch": now - 100, "ts": str(now - 100)}
    assert slack_src.is_actionable_mention(old, "Ume", cutoff_epoch=cutoff) is False
    assert slack_src.is_actionable_mention(own, "Ume", cutoff_epoch=cutoff) is False
    assert slack_src.is_actionable_mention(ok, "Ume", cutoff_epoch=cutoff) is True


def test_slack_import_error_becomes_credential_not_found() -> None:
    import sys

    # Force ImportError for slack tool modules without breaking other imports.
    blocked = {
        "core.tools._slack_client": None,
        "core.tools._slack_cache": None,
    }
    with patch.dict(sys.modules, blocked):
        with pytest.raises(CredentialNotFoundError, match="Slack dependencies"):
            slack_src.collect_slack()


def test_slack_missing_user_id_raises() -> None:
    mock_client = MagicMock()
    mock_client.my_user_id = ""
    mock_client.auth_test.return_value = {}

    with (
        patch("core.tools._slack_client.SlackClient", return_value=mock_client),
        patch("core.tools._slack_cache.MessageCache"),
    ):
        with pytest.raises(RuntimeError, match="user_id"):
            slack_src.collect_slack()


def test_slack_build_permalink_deterministic() -> None:
    assert (
        slack_src._build_permalink("C01ABC", "1234567890.123456")
        == "https://slack.com/archives/C01ABC/p1234567890123456"
    )
    assert slack_src._build_permalink("", "1.2") is None
    assert slack_src._build_permalink("C1", "not-a-ts") is None


def test_slack_message_to_task_skips_incomplete_and_uses_channel_fallback() -> None:
    client = MagicMock()
    client.get_channel_name.side_effect = RuntimeError("name lookup failed")
    incomplete = {"channel_id": "", "ts": "1.0"}
    assert slack_src._message_to_task(incomplete, client) is None

    msg = {
        "channel_id": "C9",
        "ts": "1700000000.000100",
        "user_id": "U2",
        "text": "   multi\n  line   body   " + ("x" * 100),
        # no channel_name → falls back via get_channel_name (fails) → channel_id
    }
    task = slack_src._message_to_task(msg, client)
    assert task is not None
    assert task.id == "slack-C9-1700000000.000100"
    assert task.title.startswith("#C9:")
    assert len(task.title) <= len("#C9: ") + 80
    assert task.source_url == "https://slack.com/archives/C9/p1700000000000100"


def test_slack_is_actionable_mention_edge_cases() -> None:
    assert slack_src.is_actionable_mention({}, "Ume") is False
    assert slack_src.is_actionable_mention(None, "Ume") is False  # type: ignore[arg-type]
    # Unparseable ts with cutoff → drop
    bad = {"user_id": "U2", "ts": "nope", "ts_epoch": "also-nope"}
    assert slack_src.is_actionable_mention(bad, "Ume", cutoff_epoch=1.0) is False
    # ts_epoch missing but ts parseable
    ok = {"user_id": "U2", "ts": "9999999999.0"}
    assert slack_src.is_actionable_mention(ok, "Ume", cutoff_epoch=1.0) is True


def test_slack_preview_and_ts_helpers() -> None:
    assert slack_src._preview_text("") == "(no text)"
    assert slack_src._preview_text("  a   b  ") == "a b"
    assert len(slack_src._preview_text("x" * 200)) == 80
    # Invalid ts falls back to "now"-ish ISO (contains T)
    iso = slack_src._ts_to_iso("not-a-timestamp")
    assert "T" in iso
    assert slack_src._message_epoch({"ts_epoch": "bad", "ts": "12.5"}) == 12.5
    assert slack_src._message_epoch({"ts": "x"}) is None


# ── Chatwork ────────────────────────────────────────────


def test_chatwork_collects_open_tasks() -> None:
    mock_client = MagicMock()
    mock_client.my_tasks.return_value = [
        {
            "task_id": 99,
            "body": "[info]Please review the contract[/info]",
            "message_id": "m55",
            "limit_time": 1720000000,
            "room": {"room_id": 12345, "name": "Legal"},
        }
    ]
    mock_client.me.return_value = {"account_id": 7}

    mock_cache = MagicMock()
    mock_cache.find_unreplied.return_value = []

    with (
        patch("core.tools._chatwork_client.ChatworkClient", return_value=mock_client),
        patch("core.tools._chatwork_cache.MessageCache", return_value=mock_cache),
    ):
        tasks = chatwork_src.collect_chatwork()

    assert len(tasks) == 1
    t = tasks[0]
    assert t.id == "chatwork-task-99"
    assert t.priority == 85
    assert t.title.startswith("Legal:")
    assert "Please review the contract" in t.title
    assert t.source_url == "https://www.chatwork.com/#!rid12345-m55"
    assert t.source_type == "chatwork"
    assert t.source_icon == "chatwork"
    # limit_time → deterministic ISO
    assert t.last_updated_at.startswith("2024-")


def test_chatwork_no_limit_time_uses_epoch() -> None:
    mock_client = MagicMock()
    mock_client.my_tasks.return_value = [
        {
            "task_id": 1,
            "body": "no deadline",
            "message_id": "",
            "limit_time": 0,
            "room": {"room_id": 9, "name": "R"},
        }
    ]
    mock_client.me.return_value = {"account_id": 7}
    mock_cache = MagicMock()
    mock_cache.find_unreplied.return_value = []

    with (
        patch("core.tools._chatwork_client.ChatworkClient", return_value=mock_client),
        patch("core.tools._chatwork_cache.MessageCache", return_value=mock_cache),
    ):
        tasks = chatwork_src.collect_chatwork()

    assert len(tasks) == 1
    assert tasks[0].last_updated_at == "1970-01-01T00:00:00+00:00"
    assert tasks[0].created_at == "1970-01-01T00:00:00+00:00"


def test_chatwork_collects_unreplied_mentions() -> None:
    import time

    send_time = int(time.time()) - 100
    mock_client = MagicMock()
    mock_client.my_tasks.return_value = []
    mock_client.me.return_value = {"account_id": 7}

    mock_cache = MagicMock()
    mock_cache.find_unreplied.return_value = [
        {
            "room_id": "100",
            "message_id": "200",
            "room_name": "Ops",
            "body": "[To:7] Alice\nNeed your OK",
            "send_time": send_time,
        }
    ]

    with (
        patch("core.tools._chatwork_client.ChatworkClient", return_value=mock_client),
        patch("core.tools._chatwork_cache.MessageCache", return_value=mock_cache),
    ):
        tasks = chatwork_src.collect_chatwork()

    assert len(tasks) == 1
    t = tasks[0]
    assert t.id == "chatwork-msg-100-200"
    assert t.priority == 80
    assert t.title == "Ops: Need your OK"
    assert t.source_url == "https://www.chatwork.com/#!rid100-200"


def test_chatwork_credential_missing() -> None:
    with patch(
        "core.tools._chatwork_client.ChatworkClient",
        side_effect=ToolConfigError("missing chatwork token"),
    ):
        with pytest.raises(CredentialNotFoundError, match="missing chatwork token"):
            chatwork_src.collect_chatwork()


def test_chatwork_api_error_propagates() -> None:
    mock_client = MagicMock()
    mock_client.my_tasks.side_effect = RuntimeError("chatwork 500")

    with patch("core.tools._chatwork_client.ChatworkClient", return_value=mock_client):
        with pytest.raises(RuntimeError, match="chatwork 500"):
            chatwork_src.collect_chatwork()


def test_chatwork_mentions_skipped_when_me_fails() -> None:
    mock_client = MagicMock()
    mock_client.my_tasks.return_value = []
    mock_client.me.side_effect = RuntimeError("me failed")

    with patch("core.tools._chatwork_client.ChatworkClient", return_value=mock_client):
        tasks = chatwork_src.collect_chatwork()

    assert tasks == []


def test_chatwork_mentions_skipped_when_account_id_empty() -> None:
    mock_client = MagicMock()
    mock_client.my_tasks.return_value = []
    mock_client.me.return_value = {}

    with patch("core.tools._chatwork_client.ChatworkClient", return_value=mock_client):
        tasks = chatwork_src.collect_chatwork()

    assert tasks == []


def test_chatwork_mentions_filter_old_and_incomplete_rows() -> None:
    import time

    fresh = int(time.time()) - 100
    stale = int(time.time()) - 60 * 60 * 24 * 30
    mock_client = MagicMock()
    mock_client.my_tasks.return_value = []
    mock_client.me.return_value = {"account_id": 7}

    mock_cache = MagicMock()
    mock_cache.find_unreplied.return_value = [
        {"room_id": "1", "message_id": "10", "room_name": "A", "body": "old", "send_time": stale},
        {"room_id": "", "message_id": "11", "room_name": "B", "body": "no room", "send_time": fresh},
        {"room_id": "2", "message_id": "", "room_name": "C", "body": "no msg id", "send_time": fresh},
        {"room_id": "3", "message_id": "12", "room_name": "D", "body": "keep me", "send_time": fresh},
    ]

    with (
        patch("core.tools._chatwork_client.ChatworkClient", return_value=mock_client),
        patch("core.tools._chatwork_cache.MessageCache", return_value=mock_cache),
    ):
        tasks = chatwork_src.collect_chatwork()

    assert [t.id for t in tasks] == ["chatwork-msg-3-12"]


# ── Gmail ───────────────────────────────────────────────


def _gmail_client_with_token(tmp_path, **attrs: Any) -> MagicMock:
    """GmailClient mock with a real token file so pre-check passes."""
    token = tmp_path / "token.json"
    token.write_text("{}", encoding="utf-8")
    mock_client = MagicMock()
    mock_client.token_path = token
    mock_client.mcp_token_path = tmp_path / "missing-mcp-token.json"
    for k, v in attrs.items():
        setattr(mock_client, k, v)
    return mock_client


def test_gmail_collects_unread(tmp_path) -> None:
    email = SimpleNamespace(
        id="msgABC",
        thread_id="thr1",
        from_addr="Alice <alice@example.com>",
        subject="Estimate review",
        snippet="...",
        date="Sat, 19 Jul 2026 10:00:00 +0000",
    )
    mock_client = _gmail_client_with_token(tmp_path)
    mock_client.search_emails.return_value = [email]

    with patch("core.tools.gmail.GmailClient", return_value=mock_client):
        tasks = gmail_src.collect_gmail()

    assert len(tasks) == 1
    t = tasks[0]
    assert t.id == "gmail-msgABC"
    assert t.priority == 70
    assert t.title == "Alice: Estimate review"
    assert t.source_url == "https://mail.google.com/mail/u/0/#inbox/msgABC"
    assert t.source_type == "gmail"
    assert t.source_icon == "gmail"
    mock_client.search_emails.assert_called_once_with(
        "is:unread in:inbox newer_than:7d",
        max_results=20,
    )


def test_gmail_credential_missing_on_import_error() -> None:
    with patch(
        "core.tools.gmail.GmailClient",
        side_effect=ImportError("google-api packages missing"),
    ):
        with pytest.raises(CredentialNotFoundError, match="google-api"):
            gmail_src.collect_gmail()


def test_gmail_no_token_skips_interactive_oauth(tmp_path) -> None:
    """Without token files, collect_gmail must not call search_emails."""
    mock_client = MagicMock()
    mock_client.token_path = tmp_path / "no-token.json"
    mock_client.mcp_token_path = tmp_path / "no-mcp.json"
    # If search_emails were reached, _get_credentials could run_local_server.
    mock_client.search_emails.side_effect = AssertionError(
        "search_emails must not be called without token"
    )

    with patch("core.tools.gmail.GmailClient", return_value=mock_client):
        with pytest.raises(CredentialNotFoundError, match="token not found"):
            gmail_src.collect_gmail()

    mock_client.search_emails.assert_not_called()


def test_gmail_credential_missing_on_value_error(tmp_path) -> None:
    mock_client = _gmail_client_with_token(tmp_path)
    mock_client.search_emails.side_effect = ValueError("No OAuth credentials found")

    with patch("core.tools.gmail.GmailClient", return_value=mock_client):
        with pytest.raises(CredentialNotFoundError, match="OAuth"):
            gmail_src.collect_gmail()


def test_gmail_api_error_propagates(tmp_path) -> None:
    mock_client = _gmail_client_with_token(tmp_path)
    mock_client.search_emails.side_effect = RuntimeError("quota exceeded")

    with patch("core.tools.gmail.GmailClient", return_value=mock_client):
        with pytest.raises(RuntimeError, match="quota exceeded"):
            gmail_src.collect_gmail()


def test_gmail_id_deterministic(tmp_path) -> None:
    email = SimpleNamespace(
        id="stable-id",
        thread_id="t",
        from_addr="bob@example.com",
        subject="Hi",
        snippet="",
        date="",
    )
    mock_client = _gmail_client_with_token(tmp_path)
    mock_client.search_emails.return_value = [email]

    with patch("core.tools.gmail.GmailClient", return_value=mock_client):
        a = gmail_src.collect_gmail()
        b = gmail_src.collect_gmail()
    assert a[0].id == b[0].id == "gmail-stable-id"
