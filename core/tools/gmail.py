# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""AnimaWorks Gmail tool -- direct Gmail API access.

Provides inbox/sent/unread mail listing, Gmail search, body reading,
draft creation, and email sending.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import sys
from dataclasses import asdict, dataclass
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr
from pathlib import Path
from typing import Any, cast

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
except ImportError:
    raise ImportError("gmail tool requires google-api packages. Install with: pip install animaworks[gmail]") from None

logger = logging.getLogger(__name__)

# ── Gmail query constants ────────────────────────────────

_QUERY_UNREAD = "is:unread in:inbox category:primary"
_QUERY_INBOX = "in:inbox"
_QUERY_SENT = "in:sent"

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "unread": {"expected_seconds": 15, "background_eligible": False},
    "inbox": {"expected_seconds": 15, "background_eligible": False},
    "sent": {"expected_seconds": 15, "background_eligible": False},
    "search": {"expected_seconds": 15, "background_eligible": False},
    "read": {"expected_seconds": 10, "background_eligible": False},
    "draft": {"expected_seconds": 10, "background_eligible": False},
    "send": {"expected_seconds": 15, "background_eligible": False},
}

# Gmail API scopes
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
]

# Default paths
_DEFAULT_CREDENTIALS_DIR = Path.home() / ".animaworks" / "credentials" / "gmail"
_DEFAULT_MCP_TOKEN_PATH = Path.home() / ".mcp-cache" / "workspace-mcp" / "token.json"


@dataclass
class Email:
    """Email message metadata."""

    id: str
    thread_id: str
    from_addr: str
    subject: str
    snippet: str
    body: str = ""
    to_addr: str = ""
    date: str = ""
    label_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict suitable for tool responses.

        Renames ``from_addr`` → ``from`` and ``to_addr`` → ``to`` to
        match natural email field names, and omits the heavyweight
        ``body`` field (callers use ``gmail_read_body`` for that).
        """
        d = asdict(self)
        d["from"] = d.pop("from_addr")
        d["to"] = d.pop("to_addr")
        d.pop("body", None)
        if not d.get("label_ids"):
            d.pop("label_ids", None)
        return d


@dataclass
class DraftResult:
    """Draft creation result."""

    draft_id: str
    message_id: str
    success: bool
    error: str | None = None


@dataclass
class SendResult:
    """Email send result."""

    message_id: str
    thread_id: str
    success: bool
    error: str | None = None


class GmailClient:
    """Gmail API client."""

    def __init__(
        self,
        credentials_path: Path | None = None,
        token_path: Path | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        mcp_token_path: Path | None = None,
    ):
        """
        Args:
            credentials_path: Path to OAuth credentials file.
            token_path: Path to save/load token.
            client_id: OAuth Client ID (used if credentials_path absent).
            client_secret: OAuth Client Secret.
            mcp_token_path: MCP-GSuite token file path (for reusing existing token).
        """
        self.credentials_path = credentials_path or (_DEFAULT_CREDENTIALS_DIR / "credentials.json")
        self.token_path = token_path or (_DEFAULT_CREDENTIALS_DIR / "token.json")
        self.client_id = client_id or os.environ.get("GMAIL_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("GMAIL_CLIENT_SECRET")
        self.mcp_token_path = mcp_token_path or _DEFAULT_MCP_TOKEN_PATH
        self._service = None

    def _load_mcp_token(self) -> Credentials | None:
        """Load MCP-GSuite JSON token and convert to Credentials."""
        if not self.mcp_token_path or not self.mcp_token_path.exists():
            return None

        try:
            with open(self.mcp_token_path) as f:
                token_data = json.load(f)

            creds = Credentials(
                token=token_data.get("access_token"),
                refresh_token=token_data.get("refresh_token"),
                token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=token_data.get("client_id") or self.client_id,
                client_secret=token_data.get("client_secret") or self.client_secret,
            )
            logger.info("Loaded MCP-GSuite token")
            return creds
        except Exception as e:
            logger.warning("MCP token load error: %s", e)
            return None

    def _get_credentials(self) -> Credentials:
        """Obtain valid credentials."""
        creds = None

        # 1. Try existing MCP token first
        if self.mcp_token_path:
            creds = self._load_mcp_token()

        # 2. Load saved token
        if not creds and self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        # Refresh or start new auth flow if needed
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                self.token_path.parent.mkdir(parents=True, exist_ok=True)
                self.token_path.write_text(creds.to_json())
            else:
                if self.credentials_path.exists():
                    flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
                elif self.client_id and self.client_secret:
                    client_config = {
                        "installed": {
                            "client_id": self.client_id,
                            "client_secret": self.client_secret,
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                            "redirect_uris": ["http://localhost"],
                        }
                    }
                    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                else:
                    raise ValueError(
                        "No OAuth credentials found. "
                        "Place credentials.json or set GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET."
                    )

                creds = cast(Credentials, flow.run_local_server(port=0, open_browser=True))
                self.token_path.parent.mkdir(parents=True, exist_ok=True)
                self.token_path.write_text(creds.to_json())

        return creds

    @property
    def service(self):
        """Get Gmail API service instance."""
        if self._service is None:
            creds = self._get_credentials()
            self._service = build("gmail", "v1", credentials=creds)
        return self._service

    # ── Private helpers ──────────────────────────────────

    def _get_email_details(self, message_id: str) -> Email | None:
        """Get email metadata by message ID.

        Uses ``format="metadata"`` to fetch only headers without the
        full MIME body, which is significantly lighter than ``"full"``.
        """
        try:
            message = (
                self.service.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="metadata",
                    metadataHeaders=["From", "To", "Subject", "Date"],
                )
                .execute()
            )

            headers = {h["name"]: h["value"] for h in message["payload"]["headers"]}

            return Email(
                id=message["id"],
                thread_id=message["threadId"],
                from_addr=headers.get("From", ""),
                subject=headers.get("Subject", ""),
                snippet=message.get("snippet", ""),
                to_addr=headers.get("To", ""),
                date=headers.get("Date", ""),
                label_ids=message.get("labelIds"),
            )

        except Exception as e:
            logger.error("Email detail fetch error (%s): %s", message_id, e)
            return None

    def _fetch_emails(
        self,
        query: str,
        max_results: int,
        label: str,
    ) -> list[Email]:
        """Fetch emails matching *query* from the Gmail API.

        Args:
            query: Gmail search query string.
            max_results: Maximum number of messages to return.
            label: Human-readable label used in log messages
                (e.g. ``"unread"``, ``"inbox"``).

        Returns:
            List of :class:`Email` objects.
        """
        logger.info("Fetching %s emails...", label)

        try:
            results = self.service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()

            messages = results.get("messages", [])
            if not messages:
                logger.info("No %s emails", label)
                return []

            emails = [e for msg in messages if (e := self._get_email_details(msg["id"]))]

            logger.info("Fetched %d %s emails", len(emails), label)
            return emails

        except Exception as e:
            logger.error("%s fetch error: %s", label.capitalize(), e)
            return []

    # ── Public listing methods ────────────────────────────

    def get_unread_emails(self, max_results: int = 20) -> list[Email]:
        """Fetch unread emails from Primary category."""
        return self._fetch_emails(_QUERY_UNREAD, max_results, "unread")

    def get_inbox_emails(self, max_results: int = 20) -> list[Email]:
        """Fetch emails from the inbox (read and unread)."""
        return self._fetch_emails(_QUERY_INBOX, max_results, "inbox")

    def get_sent_emails(self, max_results: int = 20) -> list[Email]:
        """Fetch sent emails."""
        return self._fetch_emails(_QUERY_SENT, max_results, "sent")

    def search_emails(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[Email]:
        """Search emails with a Gmail query string.

        Args:
            query: Gmail search query (e.g. ``from:alice subject:report``).
            max_results: Maximum number of emails to retrieve.
        """
        return self._fetch_emails(query, max_results, "search")

    def get_email_body(self, message_id: str) -> str:
        """Get full email body text."""
        try:
            message = self.service.users().messages().get(userId="me", id=message_id, format="full").execute()

            return self._extract_body(message["payload"])

        except Exception as e:
            logger.error("Email body fetch error (%s): %s", message_id, e)
            return ""

    def _extract_body(self, payload: dict) -> str:
        """Extract plain text body from email payload."""
        if "body" in payload and payload["body"].get("data"):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")

        if "parts" in payload:
            for part in payload["parts"]:
                if part["mimeType"] == "text/plain":
                    if part["body"].get("data"):
                        return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                elif part["mimeType"].startswith("multipart/"):
                    body = self._extract_body(part)
                    if body:
                        return body

        return ""

    def _resolve_reply_headers(
        self,
        message_id: str,
    ) -> tuple[str, str, str]:
        """Fetch RFC Message-ID, threadId, and subject from a Gmail message.

        Args:
            message_id: Gmail internal message ID.

        Returns:
            (rfc_message_id, thread_id, subject) tuple.
        """
        msg = (
            self.service.users()
            .messages()
            .get(userId="me", id=message_id, format="metadata", metadataHeaders=["Message-ID", "Subject"])
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        return (
            headers.get("Message-ID", ""),
            msg.get("threadId", ""),
            headers.get("Subject", ""),
        )

    def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        thread_id: str | None = None,
        in_reply_to: str | None = None,
        attachments: list[Path] | None = None,
    ) -> DraftResult:
        """Create a draft email.

        Args:
            to: Recipient address.
            subject: Email subject.
            body: Email body text.
            thread_id: Thread ID (for replies).
            in_reply_to: Gmail message ID being replied to. The RFC
                Message-ID header and threadId are resolved automatically.
            attachments: List of file paths to attach.

        Returns:
            DraftResult with creation outcome.
        """
        try:
            _, email_addr = parseaddr(to)
            recipient = email_addr if email_addr else to

            # Resolve reply threading from Gmail message ID
            rfc_message_id = ""
            if in_reply_to:
                rfc_message_id, resolved_thread_id, orig_subject = self._resolve_reply_headers(in_reply_to)
                if not thread_id:
                    thread_id = resolved_thread_id
                if not subject.lower().startswith("re:"):
                    subject = f"Re: {orig_subject}" if orig_subject else subject

            if attachments:
                message = MIMEMultipart()
                message.attach(MIMEText(body))
                for file_path in attachments:
                    file_path = Path(file_path)
                    if not file_path.exists():
                        raise FileNotFoundError(f"Attachment not found: {file_path}")
                    content_type, _ = mimetypes.guess_type(str(file_path))
                    if content_type is None:
                        content_type = "application/octet-stream"
                    main_type, sub_type = content_type.split("/", 1)
                    with open(file_path, "rb") as f:
                        part = MIMEBase(main_type, sub_type)
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        "attachment",
                        filename=file_path.name,
                    )
                    message.attach(part)
            else:
                message = MIMEText(body)

            message["to"] = recipient
            message["subject"] = subject

            if rfc_message_id:
                message["In-Reply-To"] = rfc_message_id
                message["References"] = rfc_message_id

            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

            draft_body: dict = {"message": {"raw": raw}}
            if thread_id:
                draft_body["message"]["threadId"] = thread_id

            draft = self.service.users().drafts().create(userId="me", body=draft_body).execute()

            attached_names = [Path(p).name for p in attachments] if attachments else []
            logger.info("Draft created: %s (attachments: %s)", draft["id"], attached_names)

            return DraftResult(
                draft_id=draft["id"],
                message_id=draft["message"]["id"],
                success=True,
            )

        except Exception as e:
            logger.error("Draft creation error: %s", e)
            return DraftResult(
                draft_id="",
                message_id="",
                success=False,
                error=str(e),
            )

    def send_message(
        self,
        to: str,
        subject: str,
        body: str,
        thread_id: str | None = None,
        in_reply_to: str | None = None,
        attachments: list[Path] | None = None,
    ) -> SendResult:
        """Send an email immediately.

        Args:
            to: Recipient address.
            subject: Email subject.
            body: Email body text.
            thread_id: Thread ID (for replies).
            in_reply_to: Gmail message ID being replied to. The RFC
                Message-ID header and threadId are resolved automatically.
            attachments: List of file paths to attach.

        Returns:
            SendResult with send outcome.
        """
        try:
            _, email_addr = parseaddr(to)
            recipient = email_addr if email_addr else to

            # Resolve reply threading from Gmail message ID
            rfc_message_id = ""
            if in_reply_to:
                rfc_message_id, resolved_thread_id, orig_subject = self._resolve_reply_headers(in_reply_to)
                if not thread_id:
                    thread_id = resolved_thread_id
                if not subject.lower().startswith("re:"):
                    subject = f"Re: {orig_subject}" if orig_subject else subject

            if attachments:
                message = MIMEMultipart()
                message.attach(MIMEText(body))
                for file_path in attachments:
                    file_path = Path(file_path)
                    if not file_path.exists():
                        raise FileNotFoundError(f"Attachment not found: {file_path}")
                    content_type, _ = mimetypes.guess_type(str(file_path))
                    if content_type is None:
                        content_type = "application/octet-stream"
                    main_type, sub_type = content_type.split("/", 1)
                    with open(file_path, "rb") as f:
                        part = MIMEBase(main_type, sub_type)
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        "attachment",
                        filename=file_path.name,
                    )
                    message.attach(part)
            else:
                message = MIMEText(body)

            message["to"] = recipient
            message["subject"] = subject

            if rfc_message_id:
                message["In-Reply-To"] = rfc_message_id
                message["References"] = rfc_message_id

            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

            send_body: dict = {"raw": raw}
            if thread_id:
                send_body["threadId"] = thread_id

            sent = self.service.users().messages().send(userId="me", body=send_body).execute()

            attached_names = [Path(p).name for p in attachments] if attachments else []
            logger.info("Email sent: %s to %s (attachments: %s)", sent["id"], recipient, attached_names)

            return SendResult(
                message_id=sent["id"],
                thread_id=sent.get("threadId", ""),
                success=True,
            )

        except Exception as e:
            logger.error("Email send error: %s", e)
            return SendResult(
                message_id="",
                thread_id="",
                success=False,
                error=str(e),
            )

    def get_attachments(self, message_id: str, save_dir: Path) -> list[tuple[str, Path]]:
        """Download attachments from an email.

        Args:
            message_id: Email message ID.
            save_dir: Directory to save attachments.

        Returns:
            List of (filename, save_path) tuples.
        """
        try:
            message = self.service.users().messages().get(userId="me", id=message_id, format="full").execute()

            save_dir.mkdir(parents=True, exist_ok=True)
            attachments = []

            parts = message["payload"].get("parts", [])
            for part in parts:
                if part.get("filename") and part["body"].get("attachmentId"):
                    filename = part["filename"]
                    attachment_id = part["body"]["attachmentId"]

                    attachment = (
                        self.service.users()
                        .messages()
                        .attachments()
                        .get(userId="me", messageId=message_id, id=attachment_id)
                        .execute()
                    )

                    data = base64.urlsafe_b64decode(attachment["data"])
                    save_path = save_dir / filename

                    with open(save_path, "wb") as f:
                        f.write(data)

                    attachments.append((filename, save_path))
                    logger.info("Attachment saved: %s", filename)

            return attachments

        except Exception as e:
            logger.error("Attachment fetch error (%s): %s", message_id, e)
            return []


# ---------------------------------------------------------------------------
# Tool schemas for agent integration
# ---------------------------------------------------------------------------


def get_tool_schemas() -> list[dict]:
    """Return JSON schemas for Gmail tools."""
    return []


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def get_cli_guide() -> str:
    """Return CLI usage guide for Gmail tools."""
    return """\
### Gmail
```bash
animaworks-tool gmail unread -n 10
animaworks-tool gmail inbox -n 10
animaworks-tool gmail sent -n 10
animaworks-tool gmail search "from:alice subject:report" -n 10
animaworks-tool gmail read <メッセージID>
animaworks-tool gmail draft --to "宛先" --subject "件名" --body "本文"
animaworks-tool gmail draft --to "宛先" --subject "件名" --body "本文" --attachment /path/to/file.pdf
animaworks-tool gmail send --to "宛先" --subject "件名" --body "本文"
animaworks-tool gmail send --to "宛先" --subject "件名" --body "本文" --attachment /path/to/file.pdf
```
⚠️ **send はメールを即時送信します。取り消しできません。**"""


def _print_emails(emails: list[Email], label: str) -> None:
    """Print a list of emails to stdout in a human-readable format."""
    if not emails:
        print(f"No {label} emails.")
        return
    for em in emails:
        print(f"[{em.id}] {em.from_addr} -> {em.to_addr}  ({em.date})")
        print(f"  Subject: {em.subject}")
        print(f"  Snippet: {em.snippet}")
        print()


def cli_main(argv: list[str] | None = None) -> None:
    """Standalone CLI for Gmail operations."""
    parser = argparse.ArgumentParser(
        prog="animaworks-gmail",
        description="AnimaWorks Gmail tool -- list, read, and draft emails.",
    )
    sub = parser.add_subparsers(dest="command")

    # unread
    p_unread = sub.add_parser("unread", help="List unread emails")
    p_unread.add_argument(
        "-n",
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of emails to retrieve (default: 20)",
    )

    # inbox
    p_inbox = sub.add_parser("inbox", help="List inbox emails (read and unread)")
    p_inbox.add_argument(
        "-n",
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of emails to retrieve (default: 20)",
    )

    # sent
    p_sent = sub.add_parser("sent", help="List sent emails")
    p_sent.add_argument(
        "-n",
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of emails to retrieve (default: 20)",
    )

    # search
    p_search = sub.add_parser("search", help="Search emails with Gmail query")
    p_search.add_argument("query", help="Gmail search query string")
    p_search.add_argument(
        "-n",
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of emails to retrieve (default: 20)",
    )

    # read
    p_read = sub.add_parser("read", help="Read full email body")
    p_read.add_argument("message_id", help="Gmail message ID")

    # draft
    p_draft = sub.add_parser("draft", help="Create a draft email")
    p_draft.add_argument("--to", required=True, help="Recipient address")
    p_draft.add_argument("--subject", required=True, help="Subject line")
    p_draft.add_argument("--body", required=True, help="Body text")
    p_draft.add_argument("--thread-id", default=None, help="Thread ID (for replies)")
    p_draft.add_argument("--in-reply-to", default=None, help="Original message ID")
    p_draft.add_argument("--attachment", action="append", default=[], help="File path to attach (repeatable)")

    # send
    p_send = sub.add_parser("send", help="Send an email immediately")
    p_send.add_argument("--to", required=True, help="Recipient address")
    p_send.add_argument("--subject", required=True, help="Subject line")
    p_send.add_argument("--body", required=True, help="Body text")
    p_send.add_argument("--thread-id", default=None, help="Thread ID (for replies)")
    p_send.add_argument("--in-reply-to", default=None, help="Original message ID")
    p_send.add_argument("--attachment", action="append", default=[], help="File path to attach (repeatable)")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = GmailClient()

    if args.command == "unread":
        _print_emails(client.get_unread_emails(max_results=args.max_results), "unread")

    elif args.command == "inbox":
        _print_emails(client.get_inbox_emails(max_results=args.max_results), "inbox")

    elif args.command == "sent":
        _print_emails(client.get_sent_emails(max_results=args.max_results), "sent")

    elif args.command == "search":
        _print_emails(
            client.search_emails(query=args.query, max_results=args.max_results),
            "search",
        )

    elif args.command == "read":
        body = client.get_email_body(args.message_id)
        if body:
            print(body)
        else:
            print("(empty or could not retrieve body)", file=sys.stderr)
            sys.exit(1)

    elif args.command == "draft":
        attach_paths = [Path(p) for p in args.attachment] if args.attachment else None
        result = client.create_draft(
            to=args.to,
            subject=args.subject,
            body=args.body,
            thread_id=args.thread_id,
            in_reply_to=args.in_reply_to,
            attachments=attach_paths,
        )
        if result.success:
            print(f"Draft created: {result.draft_id}")
        else:
            print(f"Draft creation failed: {result.error}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "send":
        attach_paths = [Path(p) for p in args.attachment] if args.attachment else None
        result = client.send_message(
            to=args.to,
            subject=args.subject,
            body=args.body,
            thread_id=args.thread_id,
            in_reply_to=args.in_reply_to,
            attachments=attach_paths,
        )
        if result.success:
            print(f"Email sent: {result.message_id}")
        else:
            print(f"Send failed: {result.error}", file=sys.stderr)
            sys.exit(1)


# ── Dispatch ──────────────────────────────────────────


def dispatch(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call by schema name."""
    client = GmailClient()

    if name == "gmail_unread":
        emails = client.get_unread_emails(max_results=args.get("max_results", 20))
        return [e.to_dict() for e in emails]
    if name == "gmail_inbox":
        emails = client.get_inbox_emails(max_results=args.get("max_results", 20))
        return [e.to_dict() for e in emails]
    if name == "gmail_sent":
        emails = client.get_sent_emails(max_results=args.get("max_results", 20))
        return [e.to_dict() for e in emails]
    if name == "gmail_search":
        emails = client.search_emails(
            query=args["query"],
            max_results=args.get("max_results", 20),
        )
        return [e.to_dict() for e in emails]
    if name == "gmail_read_body":
        return client.get_email_body(args["message_id"])
    if name == "gmail_draft":
        raw_attachments = args.get("attachments")
        if isinstance(raw_attachments, str):
            raw_attachments = json.loads(raw_attachments)
        attach_paths = [Path(p) for p in raw_attachments] if raw_attachments else None
        result = client.create_draft(
            to=args["to"],
            subject=args["subject"],
            body=args["body"],
            thread_id=args.get("thread_id"),
            in_reply_to=args.get("in_reply_to"),
            attachments=attach_paths,
        )
        return {"success": result.success, "draft_id": result.draft_id, "error": result.error}
    if name == "gmail_send":
        raw_attachments = args.get("attachments")
        if isinstance(raw_attachments, str):
            raw_attachments = json.loads(raw_attachments)
        attach_paths = [Path(p) for p in raw_attachments] if raw_attachments else None
        result = client.send_message(
            to=args["to"],
            subject=args["subject"],
            body=args["body"],
            thread_id=args.get("thread_id"),
            in_reply_to=args.get("in_reply_to"),
            attachments=attach_paths,
        )
        return {
            "success": result.success,
            "message_id": result.message_id,
            "thread_id": result.thread_id,
            "error": result.error,
        }
    raise ValueError(f"Unknown tool: {name}")


if __name__ == "__main__":
    cli_main()
