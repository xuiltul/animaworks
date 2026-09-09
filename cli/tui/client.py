# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import websockets

from cli.tui.sse import SseEvent, aiter_sse

logger = logging.getLogger(__name__)

_MAX_RECONNECT_DELAY = 30.0
_INITIAL_RECONNECT_DELAY = 1.0


class AnimaWorksClientError(Exception):
    """Raised when a client request fails (connection, HTTP, parse error)."""


def _ws_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.startswith("https"):
        return "wss://" + normalized[len("https://") :] + "/ws"
    if normalized.startswith("http"):
        return "ws://" + normalized[len("http://") :] + "/ws"
    return normalized + "/ws"


def parse_ws_message(message: Any) -> tuple[str, dict[str, Any], bool]:
    """Parse a raw websocket message into ``(type, payload, should_pong)``.

    Pure function (no I/O) so it can be unit tested directly.

    - Handles ``str`` / ``bytes`` / already-parsed ``dict`` inputs.
    - Normalises the ``type`` / ``event`` keys into a single ``type``.
    - Returns ``should_pong=True`` for ``ping`` messages.
    """
    if isinstance(message, (str, bytes, bytearray)):
        try:
            if isinstance(message, (bytes, bytearray)):
                raw = message.decode("utf-8")
            else:
                raw = message
            parsed: Any = json.loads(raw)
        except Exception:
            return "_ws_status", {"handled": False}, False
    else:
        parsed = message

    if not isinstance(parsed, dict):
        return "_ws_status", {"handled": False}, False

    event_type = parsed.get("type") or parsed.get("event")
    if not event_type:
        return "_ws_status", {"handled": False}, False

    if event_type == "ping":
        payload = {k: v for k, v in parsed.items() if k not in ("type", "event")}
        return "ping", payload, True

    payload = parsed.get("data")
    if not isinstance(payload, dict):
        payload = {k: v for k, v in parsed.items() if k not in ("type", "event")}
    return event_type, payload, False


class AnimaWorksClient:
    """Thin HTTP / SSE / WebSocket client for the AnimaWorks gateway.

    Does not depend on Textual so it can be unit tested in isolation.
    """

    def __init__(
        self,
        base_url: str,
        *,
        from_person: str = "human",
        timeout: float | None = None,
        transport: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.from_person = from_person
        self.timeout = timeout
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport, cookies=httpx.Cookies())

    async def aclose(self) -> None:
        await self._http.aclose()

    async def list_animas(self) -> list[dict]:
        resp = await self._get(f"{self.base_url}/api/animas")
        return resp

    async def list_available_models(self, *, refresh: bool = False) -> list[dict]:
        """Return the mode+model catalog from the gateway.

        ``refresh`` asks the server to ignore its cache and re-fetch.
        Any non-catalog response yields an empty list; network / HTTP
        errors surface as :class:`AnimaWorksClientError`.
        """
        params = {"refresh": 1} if refresh else None
        resp = await self._get(
            f"{self.base_url}/api/system/available-models",
            params=params,
        )
        if not isinstance(resp, dict):
            return []
        models = resp.get("models")
        if not isinstance(models, list):
            return []
        return models

    async def get_history(
        self,
        anima: str,
        *,
        thread_id: str = "default",
        limit: int = 50,
        before: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"limit": limit, "thread_id": thread_id}
        if before:
            params["before"] = before
        resp = await self._get(
            f"{self.base_url}/api/animas/{anima}/conversation/history",
            params=params,
        )
        return resp

    async def list_threads(self, anima: str) -> dict:
        """List an anima's conversation threads (``default`` plus UUID threads)."""
        resp = await self._get(
            f"{self.base_url}/api/animas/{anima}/sessions",
        )
        return resp

    async def get_active_stream(
        self,
        anima: str,
        *,
        thread_id: str = "default",
    ) -> dict:
        """Query the server for an in-flight / most-recent stream."""
        resp = await self._get(
            f"{self.base_url}/api/animas/{anima}/stream/active",
            params={"thread_id": thread_id},
        )
        return resp

    async def get_stream_progress(
        self,
        anima: str,
        response_id: str,
    ) -> dict | None:
        """Fetch progress for a specific stream (404 → ``None``)."""
        url = f"{self.base_url}/api/animas/{anima}/stream/{response_id}/progress"
        try:
            resp = await self._http.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise AnimaWorksClientError(str(exc)) from exc

    async def auth_me(self) -> dict | None:
        """Return the current user (or ``None`` when not authenticated).

        Authentication may be disabled entirely (200 with no payload) — any
        non-auth response is treated as ``None``.
        """
        try:
            resp = await self._http.get(f"{self.base_url}/api/auth/me")
        except httpx.HTTPError as exc:
            raise AnimaWorksClientError(str(exc)) from exc
        if resp.status_code in (401, 403):
            return None
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {}

    async def login(self, username: str, password: str) -> dict:
        """Authenticate and store the session cookie in the jar."""
        try:
            resp = await self._http.post(
                f"{self.base_url}/api/auth/login",
                json={"username": username, "password": password},
            )
            if resp.status_code == 400:
                body = resp.json()
                raise AnimaWorksClientError(body.get("error") or "Authentication is not enabled")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise AnimaWorksClientError(str(exc)) from exc

    def ws_cookie_header(self) -> dict[str, str] | None:
        """Return a ``Cookie`` header (for the WS handshake) if we have one."""
        for cookie in self._http.cookies.jar:
            if getattr(cookie, "name", None) == "session_token" and getattr(cookie, "value", None):
                return {"Cookie": f"session_token={cookie.value}"}
        return None

    async def chat_stream(
        self,
        anima: str,
        message: str,
        *,
        thread_id: str = "default",
        resume: str | None = None,
        last_event_id: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[SseEvent]:
        payload = {
            "message": message,
            "from_person": self.from_person,
            "intent": "",
            "images": [],
            "thread_id": thread_id,
            "model": model or None,
            "resume": resume,
            "last_event_id": last_event_id,
        }
        try:
            async with self._http.stream(
                "POST",
                f"{self.base_url}/api/animas/{anima}/chat/stream",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for event in aiter_sse(resp):
                    yield event
        except httpx.HTTPError as exc:
            raise AnimaWorksClientError(str(exc)) from exc

    async def interrupt(self, anima: str, *, thread_id: str) -> dict:
        resp = await self._http.post(
            f"{self.base_url}/api/animas/{anima}/interrupt",
            params={"thread_id": thread_id},
        )
        resp.raise_for_status()
        return resp.json()

    async def compact_session(self, anima: str, thread_id: str = "default") -> dict:
        """Request the server to compact a chat thread's context on demand."""
        try:
            resp = await self._http.post(
                f"{self.base_url}/api/animas/{anima}/chat/compact",
                json={"thread_id": thread_id},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise AnimaWorksClientError(str(exc)) from exc

    # ── Skills ───────────────────────────────────────────
    async def list_skills(self, anima: str, thread_id: str = "default") -> dict:
        resp = await self._get(
            f"{self.base_url}/api/animas/{anima}/skills",
            params={"thread_id": thread_id},
        )
        return resp

    async def get_active_skills(self, anima: str, thread_id: str = "default") -> dict:
        resp = await self._get(
            f"{self.base_url}/api/animas/{anima}/skills/active",
            params={"thread_id": thread_id},
        )
        return resp

    async def set_active_skills(
        self,
        anima: str,
        thread_id: str = "default",
        refs: list[str] | None = None,
        confirm_risk: bool = False,
    ) -> dict:
        try:
            resp = await self._http.put(
                f"{self.base_url}/api/animas/{anima}/skills/active",
                json={
                    "thread_id": thread_id,
                    "refs": refs or [],
                    "confirm_risk": confirm_risk,
                },
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise AnimaWorksClientError(str(exc)) from exc

    # ── Board / channels ────────────────────────────────
    async def list_channels(self) -> list[dict]:
        resp = await self._get(f"{self.base_url}/api/channels")
        return resp

    async def read_channel(self, name: str, limit: int = 50) -> dict:
        try:
            resp = await self._http.get(
                f"{self.base_url}/api/channels/{name}",
                params={"limit": limit},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise AnimaWorksClientError(str(exc)) from exc

    async def post_channel(self, name: str, text: str) -> dict:
        try:
            resp = await self._http.post(
                f"{self.base_url}/api/channels/{name}",
                json={"text": text, "from_name": self.from_person},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise AnimaWorksClientError(str(exc)) from exc

    # ── Task board ──────────────────────────────────────
    async def list_tasks(self, assignee: str | None = None) -> dict:
        params = {}
        if assignee:
            params["assignee"] = assignee
        resp = await self._get(f"{self.base_url}/api/task-board", params=params)
        return resp

    # ── Interaction resolve ─────────────────────────────
    async def resolve_interaction(
        self,
        anima: str,
        callback_id: str,
        decision: str,
        comment: str = "",
    ) -> dict:
        try:
            resp = await self._http.post(
                f"{self.base_url}/api/animas/{anima}/interactions/{callback_id}/resolve",
                json={"decision": decision, "comment": comment},
            )
            if resp.status_code == 409:
                raise AnimaWorksClientError("already resolved or expired")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise AnimaWorksClientError(str(exc)) from exc

    async def ws_events(self) -> AsyncIterator[dict]:
        """Yield normalized websocket events, reconnecting with backoff.

        Each yielded dict is ``{"type": <str>, "data": <dict>}``.
        Connection state changes are yielded as ``_ws_status`` events.
        ``ping`` messages are answered with ``pong`` automatically.
        """
        url = _ws_url(self.base_url)
        delay = _INITIAL_RECONNECT_DELAY
        extra_headers = self.ws_cookie_header()
        while True:
            try:
                async with websockets.connect(url, additional_headers=extra_headers) as ws:
                    delay = _INITIAL_RECONNECT_DELAY
                    yield {"type": "_ws_status", "data": {"connected": True}}
                    async for raw in ws:
                        event_type, payload, should_pong = parse_ws_message(raw)
                        if should_pong:
                            try:
                                await ws.send(json.dumps({"type": "pong"}))
                            except Exception:
                                logger.debug("Failed to send pong", exc_info=True)
                        yield {"type": event_type, "data": payload}
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("WebSocket disconnected (%s); reconnecting in %.1fs", exc, delay)
                yield {"type": "_ws_status", "data": {"connected": False}}
                await asyncio.sleep(delay)
                delay = min(delay * 2, _MAX_RECONNECT_DELAY)

    async def _get(self, url: str, params: dict | None = None) -> Any:
        try:
            resp = await self._http.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise AnimaWorksClientError(str(exc)) from exc
