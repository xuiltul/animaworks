# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""AnimaWorks Gmail tool -- direct Gmail API access.

Provides unread-mail listing, body reading, and draft creation.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.utils import parseaddr
from pathlib import Path
from typing import Any

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
except ImportError:
    raise ImportError(
        "gmail tool requires google-api packages. "
        "Install with: pip install animaworks[gmail]"
    )

logger = logging.getLogger(__name__)

# ── Execution Profile ─────────────────────────────────────

EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "inbox":  {"expected_seconds": 15, "background_eligible": False},
    "read":   {"expected_seconds": 10, "background_eligible": False},
    "send":   {"expected_seconds": 10, "background_eligible": False},
    "search": {"expected_seconds": 15, "background_eligible": False},
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


@dataclass
class DraftResult:
    """Draft creation result."""

    draft_id: str
    message_id: str
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
            with open(self.mcp_token_path, "r") as f:
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
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_path), SCOPES
                    )
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

                creds = flow.run_local_server(port=0, open_browser=True)
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

    def get_unread_emails(self, max_results: int = 20) -> list[Email]:
        """Fetch unread emails from Primary category.

        Args:
            max_results: Maximum number of emails to retrieve.

        Returns:
            List of unread Email objects.
        """
        logger.info("Fetching unread emails...")

        try:
            results = (
                self.service.users()
                .messages()
                .list(
                    userId="me",
                    q="is:unread in:inbox category:primary",
                    maxResults=max_results,
                )
                .execute()
            )

            messages = results.get("messages", [])
            if not messages:
                logger.info("No unread emails")
                return []

            emails = []
            for msg in messages:
                email = self._get_email_details(msg["id"])
                if email:
                    emails.append(email)

            logger.info("Fetched %d unread emails", len(emails))
            return emails

        except Exception as e:
            logger.error("Email fetch error: %s", e)
            return []

    def _get_email_details(self, message_id: str) -> Email | None:
        """Get email details by message ID."""
        try:
            message = (
                self.service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )

            headers = {h["name"]: h["value"] for h in message["payload"]["headers"]}

            return Email(
                id=message["id"],
                thread_id=message["threadId"],
                from_addr=headers.get("From", ""),
                subject=headers.get("Subject", ""),
                snippet=message.get("snippet", ""),
            )

        except Exception as e:
            logger.error("Email detail fetch error (%s): %s", message_id, e)
            return None

    def get_email_body(self, message_id: str) -> str:
        """Get full email body text."""
        try:
            message = (
                self.service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )

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
                        return base64.urlsafe_b64decode(part["body"]["data"]).decode(
                            "utf-8"
                        )
                elif part["mimeType"].startswith("multipart/"):
                    body = self._extract_body(part)
                    if body:
                        return body

        return ""

    def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        thread_id: str | None = None,
        in_reply_to: str | None = None,
    ) -> DraftResult:
        """Create a draft email.

        Args:
            to: Recipient address.
            subject: Email subject.
            body: Email body text.
            thread_id: Thread ID (for replies).
            in_reply_to: Original message ID (for replies).

        Returns:
            DraftResult with creation outcome.
        """
        try:
            _, email_addr = parseaddr(to)
            recipient = email_addr if email_addr else to

            message = MIMEText(body)
            message["to"] = recipient
            message["subject"] = subject

            if in_reply_to:
                message["In-Reply-To"] = in_reply_to
                message["References"] = in_reply_to

            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

            draft_body: dict = {"message": {"raw": raw}}
            if thread_id:
                draft_body["message"]["threadId"] = thread_id

            draft = (
                self.service.users()
                .drafts()
                .create(userId="me", body=draft_body)
                .execute()
            )

            logger.info("Draft created: %s", draft["id"])

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

    def get_attachments(
        self, message_id: str, save_dir: Path
    ) -> list[tuple[str, Path]]:
        """Download attachments from an email.

        Args:
            message_id: Email message ID.
            save_dir: Directory to save attachments.

        Returns:
            List of (filename, save_path) tuples.
        """
        try:
            message = (
                self.service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )

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
    return [
        {
            "name": "gmail_unread",
            "description": "List unread emails from Gmail Primary inbox.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of emails to retrieve (default 20).",
                        "default": 20,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "gmail_read_body",
            "description": "Read the full body text of a specific email by message ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "Gmail message ID.",
                    },
                },
                "required": ["message_id"],
            },
        },
        {
            "name": "gmail_draft",
            "description": "Create a draft email in Gmail.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body text.",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID for replying to existing thread (optional).",
                    },
                    "in_reply_to": {
                        "type": "string",
                        "description": "Message ID being replied to (optional).",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def get_cli_guide() -> str:
    """Return CLI usage guide for Gmail tools."""
    return """\
### Gmail
```bash
animaworks-tool gmail unread -j
animaworks-tool gmail read <メッセージID>
animaworks-tool gmail draft --to "宛先" --subject "件名" --body "本文"
```"""


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
        "-n", "--max-results", type=int, default=20,
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

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = GmailClient()

    if args.command == "unread":
        emails = client.get_unread_emails(max_results=args.max_results)
        if not emails:
            print("No unread emails.")
            return
        for em in emails:
            print(f"[{em.id}] {em.from_addr}")
            print(f"  Subject: {em.subject}")
            print(f"  Snippet: {em.snippet}")
            print()

    elif args.command == "read":
        body = client.get_email_body(args.message_id)
        if body:
            print(body)
        else:
            print("(empty or could not retrieve body)", file=sys.stderr)
            sys.exit(1)

    elif args.command == "draft":
        result = client.create_draft(
            to=args.to,
            subject=args.subject,
            body=args.body,
            thread_id=args.thread_id,
            in_reply_to=args.in_reply_to,
        )
        if result.success:
            print(f"Draft created: {result.draft_id}")
        else:
            print(f"Draft creation failed: {result.error}", file=sys.stderr)
            sys.exit(1)


# ── Dispatch ──────────────────────────────────────────

def dispatch(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call by schema name."""
    if name == "gmail_unread":
        client = GmailClient()
        emails = client.get_unread_emails(max_results=args.get("max_results", 20))
        return [
            {"id": e.id, "from": e.from_addr, "subject": e.subject, "snippet": e.snippet}
            for e in emails
        ]
    if name == "gmail_read_body":
        client = GmailClient()
        return client.get_email_body(args["message_id"])
    if name == "gmail_draft":
        client = GmailClient()
        result = client.create_draft(
            to=args["to"],
            subject=args["subject"],
            body=args["body"],
            thread_id=args.get("thread_id"),
            in_reply_to=args.get("in_reply_to"),
        )
        return {"success": result.success, "draft_id": result.draft_id, "error": result.error}
    raise ValueError(f"Unknown tool: {name}")


if __name__ == "__main__":
    cli_main()