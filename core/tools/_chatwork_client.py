# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""HTTP client for the Chatwork v2 API."""

from __future__ import annotations

from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

from core.tools._async_compat import run_sync
from core.tools._base import logger
from core.tools._retry import retry_on_rate_limit

# ── Constants ──────────────────────────────────────────────

JST = timezone(timedelta(hours=9))
BASE_URL = "https://api.chatwork.com/v2"
RATE_LIMIT_RETRY_MAX = 5
RATE_LIMIT_WAIT_DEFAULT = 60

requests = None


def _require_requests():
    global requests
    if requests is None:
        try:
            import requests as _req

            requests = _req
        except ImportError:
            raise ImportError(
                "chatwork tool requires 'requests'. Install with: pip install animaworks[communication]"
            ) from None
    return requests


# ── ChatworkClient ────────────────────────────────────────


class ChatworkClient:
    """HTTP client for the Chatwork v2 API with rate-limit retry."""

    def __init__(self, api_token: str):
        req = _require_requests()
        self.api_token = api_token
        self.session = req.Session()
        self.session.headers.update(
            {
                "X-ChatWorkToken": api_token,
                "Accept": "application/json",
            }
        )

    def _request(self, method: str, path: str, **kwargs) -> dict | list | None:
        """Send an HTTP request with rate-limit retry."""
        url = f"{BASE_URL}{path}"

        class _RateLimitError(Exception):
            """Raised when Chatwork returns HTTP 429."""

            def __init__(self, retry_after: int) -> None:
                self.retry_after = retry_after
                super().__init__(f"Rate limited, retry after {retry_after}s")

        def _do_request() -> dict | list | None:
            resp = self.session.request(method, url, **kwargs)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", RATE_LIMIT_WAIT_DEFAULT))
                raise _RateLimitError(retry_after)
            if resp.status_code == 204:
                return None
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                return None
            return resp.json()

        def _get_retry_after(exc: Exception) -> float | None:
            if isinstance(exc, _RateLimitError):
                return float(exc.retry_after)
            return None

        return retry_on_rate_limit(
            _do_request,
            max_retries=RATE_LIMIT_RETRY_MAX,
            default_wait=RATE_LIMIT_WAIT_DEFAULT,
            get_retry_after=_get_retry_after,
            retry_on=(_RateLimitError,),
        )

    async def _arequest(self, method: str, path: str, **kwargs) -> dict | list | None:
        """Async wrapper around :meth:`_request` using a thread-pool executor."""
        return await run_sync(self._request, method, path, **kwargs)

    def get(self, path: str, params: dict | None = None) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, data: dict | None = None) -> Any:
        return self._request("POST", path, data=data)

    def put(self, path: str, data: dict | None = None) -> Any:
        return self._request("PUT", path, data=data)

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    # --- High-level API methods ---

    def me(self) -> dict:
        return self.get("/me")

    def rooms(self) -> list[dict]:
        return self.get("/rooms")

    def room_members(self, room_id: str) -> list[dict]:
        return self.get(f"/rooms/{room_id}/members")

    def contacts(self) -> list[dict]:
        return self.get("/contacts")

    def get_message(self, room_id: str, message_id: str) -> dict:
        """Get a single message by ID."""
        return self.get(f"/rooms/{room_id}/messages/{message_id}")

    def get_messages(self, room_id: str, force: bool = False) -> list[dict] | None:
        """Get messages. force=True to include already-read messages."""
        return self.get(
            f"/rooms/{room_id}/messages",
            params={"force": 1 if force else 0},
        )

    def delete_message(self, room_id: str, message_id: str) -> dict | None:
        """Delete a message by ID."""
        return self.delete(f"/rooms/{room_id}/messages/{message_id}")

    def post_message(self, room_id: str, body: str) -> dict:
        if len(body) > 10000:
            raise ValueError(f"Message exceeds 10,000 characters ({len(body)} chars)")
        return self.post(f"/rooms/{room_id}/messages", data={"body": body})

    def my_tasks(self, status: str = "open") -> list[dict]:
        """List my tasks. status: open / done"""
        return self.get("/my/tasks", params={"status": status}) or []

    def room_tasks(self, room_id: str, status: str = "open") -> list[dict]:
        """List tasks in a room."""
        return self.get(f"/rooms/{room_id}/tasks", params={"status": status}) or []

    def add_task(
        self,
        room_id: str,
        body: str,
        to_ids: str,
        limit: int = 0,
        limit_type: str = "time",
    ) -> dict:
        return self.post(
            f"/rooms/{room_id}/tasks",
            data={
                "body": body,
                "to_ids": to_ids,
                "limit": limit,
                "limit_type": limit_type,
            },
        )

    def get_files(self, room_id: str, account_id: str | None = None) -> list[dict]:
        """List files uploaded to a room."""
        params: dict = {}
        if account_id:
            params["account_id"] = account_id
        return self.get(f"/rooms/{room_id}/files", params=params) or []

    def get_file(self, room_id: str, file_id: str, create_download_url: bool = True) -> dict:
        """Get file details, optionally with a one-time download URL."""
        params = {"create_download_url": 1 if create_download_url else 0}
        return self.get(f"/rooms/{room_id}/files/{file_id}", params=params)

    def upload_file(self, room_id: str, file_path: Path, message: str | None = None) -> dict:
        """Upload a file (max 5MB) to a room, optionally with a message."""
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        max_size = 5 * 1024 * 1024
        size = file_path.stat().st_size
        if size > max_size:
            raise ValueError(
                f"File too large: {file_path} is {size} bytes (max {max_size} bytes / 5MB)"
            )
        if message is not None and len(message) > 10000:
            raise ValueError(f"Message exceeds 10,000 characters ({len(message)} chars)")
        with open(file_path, "rb") as f:
            return self._request(
                "POST",
                f"/rooms/{room_id}/files",
                files={"file": (file_path.name, f)},
                data={"message": message} if message else None,
            )

    def download_file(self, download_url: str, output_path: Path) -> int:
        """Download a file to *output_path*. Returns the number of bytes written."""
        _require_requests()
        resp = self.session.get(download_url, stream=True)
        resp.raise_for_status()
        total = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                total += len(chunk)
        return total

    def get_room_by_name(self, name: str) -> dict | None:
        """Search for a room by name (exact match preferred, then partial)."""
        rooms = self.rooms()
        # Exact match first
        for r in rooms:
            if r["name"] == name:
                return r
        # Partial match
        matches = [r for r in rooms if name.lower() in r["name"].lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.warning(
                "Multiple rooms matched '%s': %s",
                name,
                ", ".join(f"[{r['room_id']}] {r['name']}" for r in matches),
            )
            return None
        return None

    def resolve_room_id(self, room: str) -> str:
        """Resolve a room name or ID to a numeric room_id string."""
        from core.exceptions import ToolConfigError

        if room.isdigit():
            return room
        r = self.get_room_by_name(room)
        if r is None:
            raise ToolConfigError(f"Room '{room}' not found")
        return str(r["room_id"])
