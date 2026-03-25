# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Discord REST API v10 client with rate-limit retry."""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import quote

import httpx

from core.exceptions import ToolConfigError
from core.tools._base import get_credential
from core.tools._retry import retry_on_rate_limit

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────

API_BASE = "https://discord.com/api/v10"
RATE_LIMIT_RETRY_MAX = 5
DISCORD_MESSAGE_LIMIT = 2000
_DEFAULT_TIMEOUT = 30.0

# Discord channel types (subset)
_CHANNEL_TYPE_TEXT = 0
_CHANNEL_TYPE_ANNOUNCEMENT = 5


# ── Exceptions ─────────────────────────────────────────────


class DiscordAPIError(Exception):
    """Raised when the Discord API returns a non-success HTTP status."""

    def __init__(self, status: int, message: str, code: int | None = None) -> None:
        self.status = status
        self.code = code
        super().__init__(f"Discord API error {status}: {message}")


class _DiscordRateLimitError(Exception):
    """Internal wrapper for 429 responses so :func:`retry_on_rate_limit` can wait."""

    def __init__(self, api_error: DiscordAPIError, retry_after: float) -> None:
        self.api_error = api_error
        self.retry_after = retry_after
        super().__init__(str(api_error))


# ── DiscordClient ──────────────────────────────────────────


class DiscordClient:
    """Synchronous Discord REST API v10 client using ``httpx``."""

    def __init__(self, token: str | None = None) -> None:
        if token is None:
            token = get_credential("discord", "discord", env_var="DISCORD_BOT_TOKEN")
        self._token = token
        self._http: httpx.Client | None = None
        self._channel_name_cache: dict[str, dict[str, str]] = {}

    def _ensure_http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(
                base_url=API_BASE,
                headers={"Authorization": f"Bot {self._token}"},
                timeout=_DEFAULT_TIMEOUT,
            )
        return self._http

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http is not None:
            self._http.close()
            self._http = None

    def _parse_error_response(self, response: httpx.Response) -> tuple[str, int | None]:
        """Extract message and optional numeric code from a Discord error body."""
        code: int | None = None
        message = response.text
        try:
            data = response.json()
            if isinstance(data, dict):
                message = str(data.get("message", message))
                raw_code = data.get("code")
                if isinstance(raw_code, int):
                    code = raw_code
        except (json.JSONDecodeError, TypeError):
            pass
        return message, code

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Perform an HTTP request with 429 rate-limit retry.

        Args:
            method: HTTP verb.
            path: Path relative to ``API_BASE`` (e.g. ``/users/@me``).
            **kwargs: Forwarded to ``httpx`` (``json``, ``params``, etc.).

        Returns:
            Parsed JSON (``dict`` or ``list``), or empty dict when body is empty.

        Raises:
            DiscordAPIError: On non-2xx after retries are exhausted.
        """

        def _get_retry_after(exc: Exception) -> float | None:
            if isinstance(exc, _DiscordRateLimitError):
                return float(exc.retry_after)
            return None

        def _do() -> Any:
            client = self._ensure_http()
            try:
                response = client.request(method, path, **kwargs)
            except httpx.TimeoutException as exc:
                raise DiscordAPIError(0, f"Request timeout: {exc}") from exc
            except httpx.ConnectError as exc:
                raise DiscordAPIError(0, f"Connection failed: {exc}") from exc
            if response.status_code == 429:
                message, code = self._parse_error_response(response)
                retry_after = 1.0
                try:
                    data = response.json()
                    if isinstance(data, dict) and "retry_after" in data:
                        retry_after = float(data["retry_after"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                api_err = DiscordAPIError(429, message, code=code)
                raise _DiscordRateLimitError(api_err, retry_after) from None
            if response.status_code < 200 or response.status_code >= 300:
                message, code = self._parse_error_response(response)
                raise DiscordAPIError(response.status_code, message, code=code)
            if not response.content:
                return {}
            try:
                return response.json()
            except json.JSONDecodeError:
                return {}

        try:
            return retry_on_rate_limit(
                _do,
                max_retries=RATE_LIMIT_RETRY_MAX,
                default_wait=1.0,
                get_retry_after=_get_retry_after,
                retry_on=(_DiscordRateLimitError,),
            )
        except _DiscordRateLimitError as exc:
            raise exc.api_error from None

    def get_bot_user(self) -> dict:
        """GET /users/@me — current bot user."""
        result = self._request("GET", "/users/@me")
        return result if isinstance(result, dict) else {}

    def guilds(self) -> list[dict]:
        """GET /users/@me/guilds — guilds the bot is in."""
        result = self._request("GET", "/users/@me/guilds")
        return result if isinstance(result, list) else []

    def channels(self, guild_id: str) -> list[dict]:
        """GET /guilds/{guild_id}/channels — text and announcement channels only."""
        result = self._request("GET", f"/guilds/{guild_id}/channels")
        if not isinstance(result, list):
            return []
        out: list[dict] = []
        for ch in result:
            t = ch.get("type")
            if t in (_CHANNEL_TYPE_TEXT, _CHANNEL_TYPE_ANNOUNCEMENT):
                out.append(ch)
        return out

    def channel_history(self, channel_id: str, limit: int = 50) -> list[dict]:
        """GET /channels/{channel_id}/messages with ``limit``."""
        result = self._request(
            "GET",
            f"/channels/{channel_id}/messages",
            params={"limit": min(limit, 100)},
        )
        return result if isinstance(result, list) else []

    def send_message(
        self,
        channel_id: str,
        content: str,
        *,
        reply_to: str | None = None,
    ) -> dict:
        """POST /channels/{channel_id}/messages with optional reply reference."""
        body: dict[str, Any] = {"content": content}
        if reply_to:
            body["message_reference"] = {
                "message_id": reply_to,
                "channel_id": channel_id,
            }
        result = self._request("POST", f"/channels/{channel_id}/messages", json=body)
        return result if isinstance(result, dict) else {}

    def edit_message(self, channel_id: str, message_id: str, content: str) -> dict:
        """PATCH /channels/{channel_id}/messages/{message_id}."""
        result = self._request(
            "PATCH",
            f"/channels/{channel_id}/messages/{message_id}",
            json={"content": content},
        )
        return result if isinstance(result, dict) else {}

    def add_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        """PUT /channels/.../reactions/{emoji}/@me (emoji URL-encoded)."""
        encoded = quote(emoji, safe="")
        self._request(
            "PUT",
            f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me",
        )

    def get_message(self, channel_id: str, message_id: str) -> dict:
        """GET a single message."""
        result = self._request("GET", f"/channels/{channel_id}/messages/{message_id}")
        return result if isinstance(result, dict) else {}

    def resolve_channel(self, guild_id: str, name_or_id: str) -> str:
        """Resolve ``#name`` or numeric channel ID to a channel snowflake ID."""
        raw = name_or_id.strip()
        if re.fullmatch(r"\d{17,22}", raw):
            return raw

        key = name_or_id.lstrip("#").strip()
        cache = self._channel_name_cache.get(guild_id)
        if cache is None:
            cache = {}
            for ch in self.channels(guild_id):
                cid = str(ch.get("id", ""))
                name = str(ch.get("name", ""))
                if cid:
                    cache[name.lower()] = cid
            self._channel_name_cache[guild_id] = cache

        if key.lower() in cache:
            return cache[key.lower()]

        partial = [(n, cid) for n, cid in cache.items() if key.lower() in n]
        if len(partial) == 1:
            return partial[0][1]
        if len(partial) > 1:
            names = ", ".join(f"#{n}" for n, _ in partial)
            raise ToolConfigError(f"Multiple Discord channels matched '{name_or_id}': {names}. Use the channel ID.")

        raise ToolConfigError(f"Discord channel '{name_or_id}' not found in guild {guild_id}")
