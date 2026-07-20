# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Gmail external tasks collector (unread inbox, last 7 days)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from core.external_tasks.models import ExternalTask

logger = logging.getLogger("animaworks.external_tasks.sources.gmail")

_PRIORITY = 70
_MAX_RESULTS = 20
_QUERY = "is:unread in:inbox newer_than:7d"


def collect_gmail() -> list[ExternalTask]:
    """Collect unread Gmail inbox messages from the last 7 days (max 20).

    Task ids: ``gmail-{message_id}``.

    Non-interactive: requires an existing token file before calling
    ``search_emails`` so background jobs never trigger interactive OAuth.
    """
    # Local import avoids circular import with collector → sources.
    from core.external_tasks.collector import CredentialNotFoundError

    try:
        from core.tools.gmail import GmailClient
    except ImportError as exc:
        raise CredentialNotFoundError(
            f"Gmail dependencies unavailable: {exc}"
        ) from exc

    try:
        client = GmailClient()
    except ImportError as exc:
        raise CredentialNotFoundError(str(exc)) from exc

    # Block interactive OAuth: only proceed when a token file already exists.
    # (core/tools/gmail.py is left unchanged; expired+unrefreshable tokens
    # still surface via RefreshError → CredentialNotFoundError below.)
    token_path = getattr(client, "token_path", None)
    mcp_token_path = getattr(client, "mcp_token_path", None)
    has_token = bool(
        (token_path is not None and Path(token_path).exists())
        or (mcp_token_path is not None and Path(mcp_token_path).exists())
    )
    if not has_token:
        raise CredentialNotFoundError(
            "Gmail token not found (non-interactive; run OAuth once offline)"
        )

    try:
        emails = client.search_emails(_QUERY, max_results=_MAX_RESULTS)
    except ImportError as exc:
        raise CredentialNotFoundError(str(exc)) from exc
    except ValueError as exc:
        # Missing OAuth client config / client secrets path.
        raise CredentialNotFoundError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise CredentialNotFoundError(str(exc)) from exc
    except Exception as exc:
        # google-auth refresh / invalid_grant style failures
        exc_name = type(exc).__name__
        if exc_name in {"RefreshError", "GoogleAuthError", "TransportError"}:
            raise CredentialNotFoundError(str(exc)) from exc
        raise

    return [_email_to_task(em) for em in emails]


def _email_to_task(email: Any) -> ExternalTask:
    message_id = str(getattr(email, "id", "") or "")
    from_addr = str(getattr(email, "from_addr", "") or "")
    subject = str(getattr(email, "subject", "") or "")
    date_raw = str(getattr(email, "date", "") or "")
    iso_ts = _parse_email_date(date_raw)

    sender = _short_sender(from_addr)
    title = f"{sender}: {subject}" if subject else f"{sender}: (no subject)"

    return ExternalTask(
        id=f"gmail-{message_id}",
        title=title,
        status="open",
        source_type="gmail",
        source_icon="gmail",
        source_url=f"https://mail.google.com/mail/u/0/#inbox/{message_id}" if message_id else None,
        created_at=iso_ts,
        last_updated_at=iso_ts,
        priority=_PRIORITY,
    )


def _short_sender(from_addr: str) -> str:
    """Prefer display name, else the address itself."""
    text = (from_addr or "").strip()
    if not text:
        return "unknown"
    if "<" in text:
        name = text.split("<", 1)[0].strip().strip('"')
        if name:
            return name
    return text


def _parse_email_date(date_header: str) -> str:
    if not date_header:
        return datetime.now(UTC).isoformat()
    try:
        dt = parsedate_to_datetime(date_header)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()
    except (TypeError, ValueError, IndexError):
        logger.debug("Unparseable Gmail Date header: %r", date_header)
        return datetime.now(UTC).isoformat()
