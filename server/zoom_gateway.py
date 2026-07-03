from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Zoom RTMS (Real-Time Media Streams) gateway.

Ingests speaker-labelled meeting transcripts from Zoom via the RTMS raw
WebSocket protocol (no bot participant, no SDK) and injects buffered
chunks into an Anima inbox through :meth:`Messenger.receive_external`.

Lifecycle: a ``meeting.rtms_started`` webhook (handled in
:mod:`server.routes.webhooks`) is delegated to :meth:`handle_rtms_started`,
which opens the signaling WebSocket, completes the HMAC-SHA256 handshake,
opens the media WebSocket subscribed to transcript only, answers
keep-alives, buffers utterances and flushes them on a time/character
threshold.  ``meeting.rtms_stopped`` flushes the tail and injects a
``meeting_ended`` trigger.

The RTMS protocol constants below follow Zoom's official manual-WebSocket
reference (developers.zoom.us/docs/rtms/, ``zoom/rtms`` samples).
"""

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import urllib.parse
from typing import Any

from core.config.models import load_config
from core.config.schemas import ZoomRTMSConfig
from core.i18n import t
from core.messenger import Messenger
from core.paths import get_shared_dir
from core.time_utils import now_local
from core.tools._base import get_credential

logger = logging.getLogger("animaworks.zoom_gateway")

# ── RTMS protocol constants ──────────────────────────────────
# Signaling socket
_MSG_SIGNALING_HANDSHAKE_REQ = 1
_MSG_SIGNALING_HANDSHAKE_RESP = 2
_MSG_CLIENT_READY_ACK = 7
_MSG_STREAM_STATE_UPDATE = 8
_MSG_SESSION_STATE_UPDATE = 9
# Media socket
_MSG_DATA_HANDSHAKE_REQ = 3
_MSG_DATA_HANDSHAKE_RESP = 4
_MSG_MEDIA_DATA_TRANSCRIPT = 17
# Shared
_MSG_KEEP_ALIVE_REQ = 12
_MSG_KEEP_ALIVE_RESP = 13

_MEDIA_TYPE_TRANSCRIPT = 8  # audio=1, video=2, screen=4, transcript=8, chat=16
_TRANSCRIPT_CONTENT_TYPE_TEXT = 5
_STATUS_OK = 0

# ── Tuning ───────────────────────────────────────────────────
_OPEN_TIMEOUT = 15.0  # WebSocket open handshake timeout (s)
_KEEPALIVE_TIMEOUT = 65.0  # RTMS media keep-alive tolerance (s) — no traffic = dead
_MAX_RECONNECT_ATTEMPTS = 5
_RECONNECT_BASE_DELAY = 1.0
_RECONNECT_MAX_DELAY = 30.0
_GAP_NOTICE = "[接続断により一部欠落]"


class _MeetingSession:
    """Per-meeting connection, buffer and chunk state."""

    def __init__(
        self,
        *,
        meeting_uuid: str,
        meeting_id: str,
        topic: str,
        rtms_stream_id: str,
        signaling_url: str,
        target_anima: str,
    ) -> None:
        self.meeting_uuid = meeting_uuid
        self.meeting_id = meeting_id
        self.topic = topic or meeting_id or meeting_uuid
        self.rtms_stream_id = rtms_stream_id
        self.signaling_url = signaling_url
        self.target_anima = target_anima
        self.start_hhmm = now_local().strftime("%H:%M")

        self.buffer: list[str] = []
        self.buffer_chars = 0
        self.chunk_seq = 1  # next chunk number to emit
        self.delivered_chunks = 0
        self.gap_pending = False  # note dropped-audio gap at head of next chunk

        self.ready = False  # media handshake acknowledged, streaming
        self.stopping = False  # graceful stop or fatal — do not reconnect

        self.lock = asyncio.Lock()  # guards buffer/chunk mutation
        self._sig_send_lock = asyncio.Lock()  # serialize concurrent signaling sends
        self.signaling_ws: Any = None
        self.media_ws: Any = None
        self.task: asyncio.Task[None] | None = None
        self.flush_timer: asyncio.Task[None] | None = None


class ZoomRTMSManager:
    """Manages Zoom RTMS transcript ingestion for all mapped meetings.

    One manager instance handles any number of concurrent meetings, keyed
    by ``meeting_uuid``.  Connections are established on demand when a
    ``meeting.rtms_started`` webhook arrives; nothing is held open while
    idle.
    """

    def __init__(self) -> None:
        self._started = False
        self._sessions: dict[str, _MeetingSession] = {}
        self._config: ZoomRTMSConfig = ZoomRTMSConfig()
        self._client_id = ""
        self._client_secret = ""
        self._websockets: Any = None  # lazy-imported websockets module
        self._last_webhook_at: str | None = None  # ISO timestamp of last webhook

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        """Prepare the manager if Zoom RTMS is enabled.

        No sockets are opened here — RTMS streams are webhook-triggered.
        """
        self._load_runtime_config()
        if not self._config.enabled:
            logger.info("Zoom RTMS gateway is disabled")
            return

        try:
            import websockets as _websockets
        except ImportError:
            logger.error("websockets is not installed — run: pip install 'animaworks[zoom]'")
            return

        self._websockets = _websockets

        if not self._client_id or not self._client_secret:
            logger.error(
                "ZOOM_CLIENT_ID / ZOOM_CLIENT_SECRET not configured — "
                "Zoom RTMS handshake will fail until they are set"
            )

        self._started = True
        logger.info("Zoom RTMS gateway started (waiting for rtms_started webhooks)")

    async def stop(self) -> None:
        """Close all active meeting sessions."""
        for session in list(self._sessions.values()):
            await self._close_session(session)
        self._sessions.clear()
        self._started = False
        logger.info("Zoom RTMS gateway stopped")

    def reload(self) -> None:
        """Refresh runtime config (mappings, thresholds, credentials).

        Running sessions read ``chunk_interval_seconds`` /
        ``chunk_max_chars`` fresh on each flush cycle, so new thresholds
        take effect without reconnecting.
        """
        self._load_runtime_config()
        logger.info("Zoom RTMS gateway config reloaded")

    async def health_check(self) -> dict[str, Any]:
        """Return gateway health for the system status endpoint."""
        meetings = [
            {
                "meeting_uuid": s.meeting_uuid,
                "meeting_id": s.meeting_id,
                "anima": s.target_anima,
                "ready": s.ready,
                "delivered_chunks": s.delivered_chunks,
                "buffered_lines": len(s.buffer),
            }
            for s in self._sessions.values()
        ]
        return {
            "status": "running" if self._started else "disabled",
            "credentials_configured": bool(self._client_id and self._client_secret),
            "active_meetings": len(self._sessions),
            "meetings": meetings,
            "last_webhook_at": self._last_webhook_at,
        }

    # ── Webhook entry points ─────────────────────────────────

    async def handle_rtms_started(self, obj: dict[str, Any]) -> None:
        """Begin ingesting a meeting from a ``meeting.rtms_started`` payload."""
        self._last_webhook_at = now_local().isoformat()
        if not self._started:
            logger.debug("Zoom RTMS started webhook ignored — gateway not running")
            return

        meeting_uuid = str(obj.get("meeting_uuid") or "")
        rtms_stream_id = str(obj.get("rtms_stream_id") or "")
        if not meeting_uuid or not rtms_stream_id:
            logger.warning("Zoom RTMS started webhook missing meeting_uuid/rtms_stream_id: %s", obj)
            return

        if meeting_uuid in self._sessions:
            logger.info("Zoom RTMS already streaming meeting %s — ignoring duplicate start", meeting_uuid)
            return

        if not self._client_id or not self._client_secret:
            logger.warning("Zoom RTMS: credentials missing — cannot start meeting %s", meeting_uuid)
            return

        signaling_url = _first_url(obj.get("server_urls"))
        if not signaling_url:
            logger.warning("Zoom RTMS started webhook missing server_urls for meeting %s", meeting_uuid)
            return
        if not _is_allowed_ws_url(signaling_url):
            logger.warning(
                "Zoom RTMS: refusing non-wss signaling URL for meeting %s: %s",
                meeting_uuid,
                signaling_url,
            )
            return

        meeting_id = str(obj.get("meeting_id") or "")
        target = self._resolve_target_anima(meeting_id)
        if not target:
            logger.warning(
                "Zoom RTMS: no target anima for meeting %s (meeting_id=%s) — discarding",
                meeting_uuid,
                meeting_id,
            )
            return

        session = _MeetingSession(
            meeting_uuid=meeting_uuid,
            meeting_id=meeting_id,
            topic=str(obj.get("topic") or ""),
            rtms_stream_id=rtms_stream_id,
            signaling_url=signaling_url,
            target_anima=target,
        )
        self._sessions[meeting_uuid] = session
        session.task = asyncio.create_task(self._run_session(session))
        session.flush_timer = asyncio.create_task(self._flush_timer(session))
        logger.info(
            "Zoom RTMS started: meeting=%s (id=%s) -> %s",
            meeting_uuid,
            meeting_id or "-",
            target,
        )

    async def handle_rtms_stopped(self, obj: dict[str, Any]) -> None:
        """Finalise a meeting: flush the tail, inject the end trigger."""
        self._last_webhook_at = now_local().isoformat()
        meeting_uuid = str(obj.get("meeting_uuid") or "")
        session = self._sessions.get(meeting_uuid)
        if session is None:
            logger.debug("Zoom RTMS stopped webhook for unknown meeting %s — ignoring", meeting_uuid)
            return

        session.stopping = True
        await self._flush(session)

        if session.delivered_chunks > 0:
            header = (
                f"[Zoom会議終了 | 会議: {session.topic} ({session.meeting_id}) | "
                f"全{session.delivered_chunks}チャンク配信済み]"
            )
            body = t("zoom.meeting_ended_body")
            await self._inject(session, f"{header}\n{body}", source_seq="end", intent="meeting_ended")
            logger.info("Zoom RTMS meeting %s ended (%d chunks delivered)", meeting_uuid, session.delivered_chunks)
        else:
            logger.info("Zoom RTMS meeting %s ended with no transcript — skipping meeting_ended", meeting_uuid)

        await self._close_session(session)
        self._sessions.pop(meeting_uuid, None)

    # ── Connection lifecycle ─────────────────────────────────

    async def _run_session(self, session: _MeetingSession) -> None:
        """Own the signaling+media connection with reconnect/backoff."""
        attempt = 0
        while not session.stopping:
            try:
                await self._connect_once(session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Zoom RTMS session %s connection error: %s", session.meeting_uuid, exc)

            if session.stopping:
                break

            # A session that reached the streaming state resets the backoff.
            if session.ready:
                session.ready = False
                attempt = 0

            attempt += 1
            if attempt > _MAX_RECONNECT_ATTEMPTS:
                logger.error(
                    "Zoom RTMS session %s: unrecoverable after %d reconnect attempts",
                    session.meeting_uuid,
                    _MAX_RECONNECT_ATTEMPTS,
                )
                break

            session.gap_pending = True
            delay = min(_RECONNECT_BASE_DELAY * (2 ** (attempt - 1)), _RECONNECT_MAX_DELAY)
            logger.info(
                "Zoom RTMS session %s reconnecting in %.1fs (attempt %d/%d)",
                session.meeting_uuid,
                delay,
                attempt,
                _MAX_RECONNECT_ATTEMPTS,
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise

        # Reached only on terminal failure (signature/handshake rejection or
        # reconnect exhaustion).  A graceful stop or rtms_stopped cancels this
        # task before it gets here, so this branch never races those paths.
        if self._sessions.get(session.meeting_uuid) is session:
            logger.warning(
                "Zoom RTMS session %s terminated without rtms_stopped — "
                "flushing tail and removing from active set",
                session.meeting_uuid,
            )
            with contextlib.suppress(Exception):
                await self._flush(session)  # preserve whatever was captured
            if session.flush_timer and not session.flush_timer.done():
                session.flush_timer.cancel()
            self._sessions.pop(session.meeting_uuid, None)

    async def _connect_once(self, session: _MeetingSession) -> None:
        """One signaling+media connection attempt; returns when it drops."""
        session.ready = False
        connect = self._websockets.connect
        async with connect(session.signaling_url, open_timeout=_OPEN_TIMEOUT) as sig_ws:
            session.signaling_ws = sig_ws
            loop = asyncio.get_running_loop()
            media_url_future: asyncio.Future[str] = loop.create_future()

            sig_loop = asyncio.create_task(self._signaling_loop(session, sig_ws, media_url_future))
            media_loop = asyncio.create_task(self._media_supervisor(session, sig_ws, media_url_future))
            try:
                await asyncio.wait({sig_loop, media_loop}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in (sig_loop, media_loop):
                    if not task.done():
                        task.cancel()
                results = await asyncio.gather(sig_loop, media_loop, return_exceptions=True)
                session.signaling_ws = None
                session.media_ws = None

            for result in results:
                if isinstance(result, asyncio.CancelledError):
                    continue
                if isinstance(result, BaseException):
                    raise result

    async def _signaling_loop(
        self,
        session: _MeetingSession,
        sig_ws: Any,
        media_url_future: asyncio.Future[str],
    ) -> None:
        """Send the signaling handshake and service the control plane."""
        await self._sig_send(session, sig_ws, self._signaling_handshake(session))
        while True:
            raw = await asyncio.wait_for(sig_ws.recv(), timeout=_KEEPALIVE_TIMEOUT)
            msg = _parse(raw)
            if msg is None:
                continue
            mt = msg.get("msg_type")
            if mt == _MSG_SIGNALING_HANDSHAKE_RESP:
                if msg.get("status_code") != _STATUS_OK:
                    logger.error(
                        "Zoom RTMS signaling handshake failed (status=%s) for meeting %s",
                        msg.get("status_code"),
                        session.meeting_uuid,
                    )
                    session.stopping = True  # signature/auth failure — do not retry
                    return
                media_url = _extract_media_url(msg)
                if not media_url:
                    logger.error("Zoom RTMS: no media server URL in handshake response for %s", session.meeting_uuid)
                    session.stopping = True
                    return
                if not _is_allowed_ws_url(media_url):
                    logger.warning(
                        "Zoom RTMS: refusing non-wss media URL for meeting %s: %s",
                        session.meeting_uuid,
                        media_url,
                    )
                    session.stopping = True
                    return
                if not media_url_future.done():
                    media_url_future.set_result(media_url)
            elif mt == _MSG_KEEP_ALIVE_REQ:
                await self._sig_send(
                    session,
                    sig_ws,
                    {"msg_type": _MSG_KEEP_ALIVE_RESP, "timestamp": msg.get("timestamp")},
                )
            elif mt in (_MSG_STREAM_STATE_UPDATE, _MSG_SESSION_STATE_UPDATE):
                logger.debug("Zoom RTMS state update (%s): %s", session.meeting_uuid, msg.get("state"))

    async def _media_supervisor(
        self,
        session: _MeetingSession,
        sig_ws: Any,
        media_url_future: asyncio.Future[str],
    ) -> None:
        """Wait for the media server URL then run the media data plane."""
        media_url = await media_url_future
        connect = self._websockets.connect
        async with connect(media_url, open_timeout=_OPEN_TIMEOUT) as media_ws:
            session.media_ws = media_ws
            await media_ws.send(json.dumps(self._media_handshake(session)))
            while True:
                raw = await asyncio.wait_for(media_ws.recv(), timeout=_KEEPALIVE_TIMEOUT)
                msg = _parse(raw)
                if msg is None:
                    continue
                mt = msg.get("msg_type")
                if mt == _MSG_DATA_HANDSHAKE_RESP:
                    if msg.get("status_code") != _STATUS_OK:
                        logger.error(
                            "Zoom RTMS media handshake failed (status=%s) for meeting %s",
                            msg.get("status_code"),
                            session.meeting_uuid,
                        )
                        return
                    await self._sig_send(
                        session,
                        sig_ws,
                        {"msg_type": _MSG_CLIENT_READY_ACK, "rtms_stream_id": session.rtms_stream_id},
                    )
                    session.ready = True
                    logger.info(
                        "Zoom RTMS streaming transcript: meeting=%s -> %s",
                        session.meeting_uuid,
                        session.target_anima,
                    )
                elif mt == _MSG_KEEP_ALIVE_REQ:
                    await media_ws.send(
                        json.dumps({"msg_type": _MSG_KEEP_ALIVE_RESP, "timestamp": msg.get("timestamp")})
                    )
                elif mt == _MSG_MEDIA_DATA_TRANSCRIPT:
                    speaker, text = _extract_transcript(msg)
                    if text:
                        await self._append(session, speaker, text)

    async def _sig_send(self, session: _MeetingSession, sig_ws: Any, payload: dict[str, Any]) -> None:
        """Send on the signaling socket, serialized against concurrent writers."""
        async with session._sig_send_lock:
            await sig_ws.send(json.dumps(payload))

    def _signaling_handshake(self, session: _MeetingSession) -> dict[str, Any]:
        return {
            "msg_type": _MSG_SIGNALING_HANDSHAKE_REQ,
            "protocol_version": 1,
            "meeting_uuid": session.meeting_uuid,
            "rtms_stream_id": session.rtms_stream_id,
            "signature": self._signature(session),
            "media_type": _MEDIA_TYPE_TRANSCRIPT,
        }

    def _media_handshake(self, session: _MeetingSession) -> dict[str, Any]:
        return {
            "msg_type": _MSG_DATA_HANDSHAKE_REQ,
            "protocol_version": 1,
            "meeting_uuid": session.meeting_uuid,
            "rtms_stream_id": session.rtms_stream_id,
            "signature": self._signature(session),
            "media_type": _MEDIA_TYPE_TRANSCRIPT,
            "payload_encryption": False,
            "media_params": {
                # Language is left to Zoom's meeting transcription setting
                # (enable_lid keeps automatic language identification on).
                "transcript": {
                    "content_type": _TRANSCRIPT_CONTENT_TYPE_TEXT,
                    "enable_lid": True,
                },
            },
        }

    def _signature(self, session: _MeetingSession) -> str:
        """HMAC-SHA256(client_secret, "client_id,meeting_uuid,rtms_stream_id")."""
        message = f"{self._client_id},{session.meeting_uuid},{session.rtms_stream_id}"
        return hmac.new(
            self._client_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    # ── Buffering & injection ────────────────────────────────

    async def _flush_timer(self, session: _MeetingSession) -> None:
        """Flush the buffer every ``chunk_interval_seconds``."""
        try:
            while not session.stopping:
                interval = max(1, self._config.chunk_interval_seconds)
                await asyncio.sleep(interval)
                if session.stopping:
                    break
                await self._flush(session)
        except asyncio.CancelledError:
            pass

    async def _append(self, session: _MeetingSession, speaker: str, text: str) -> None:
        """Append one utterance; flush early if the char threshold is hit."""
        line = f"{speaker}: {text}"
        async with session.lock:
            session.buffer.append(line)
            session.buffer_chars += len(line) + 1
            over = session.buffer_chars >= max(1, self._config.chunk_max_chars)
        if over:
            await self._flush(session)

    async def _flush(self, session: _MeetingSession) -> None:
        """Emit the buffered utterances as one chunk (no-op if empty)."""
        async with session.lock:
            if not session.buffer:
                return
            lines = session.buffer
            seq = session.chunk_seq
            gap = session.gap_pending
            session.buffer = []
            session.buffer_chars = 0
            session.chunk_seq += 1
            session.gap_pending = False

        header = (
            f"[Zoom会議実況 チャンク#{seq} | 会議: {session.topic} ({session.meeting_id}) | "
            f"{session.start_hhmm}〜]"
        )
        body_lines = ([_GAP_NOTICE] + lines) if gap else lines
        content = header + "\n" + "\n".join(body_lines)
        ok = await self._inject(session, content, source_seq=str(seq), intent="meeting_transcript")

        async with session.lock:
            if ok:
                session.delivered_chunks += 1
            else:
                # Injection failed: return the utterances to the front of the
                # buffer so the next flush resends them (with a new seq — seq
                # gaps are acceptable).  Re-arm the gap notice so the retried
                # chunk keeps the marker.
                session.buffer = lines + session.buffer
                session.buffer_chars += sum(len(line) + 1 for line in lines)
                if gap:
                    session.gap_pending = True

    async def _inject(self, session: _MeetingSession, content: str, *, source_seq: str, intent: str) -> bool:
        """Write a chunk to the target Anima inbox via the unified seam.

        Returns ``True`` on success, ``False`` if delivery raised (so callers
        can decide whether to retry).
        """
        messenger = Messenger(get_shared_dir(), session.target_anima)
        try:
            await asyncio.to_thread(
                messenger.receive_external,
                content=content,
                source="zoom",
                source_message_id=f"{session.meeting_uuid}:{source_seq}",
                external_channel_id=session.meeting_id,
                external_thread_ts=f"zoom-{session.meeting_uuid}",
                intent=intent,
            )
        except Exception:
            logger.exception(
                "Zoom RTMS: failed to inject chunk %s for meeting %s",
                source_seq,
                session.meeting_uuid,
            )
            return False
        return True

    # ── Helpers ──────────────────────────────────────────────

    def _load_runtime_config(self) -> None:
        """Refresh the cached Zoom config and credentials."""
        try:
            self._config = load_config().external_messaging.zoom
        except Exception:
            logger.debug("Zoom RTMS: failed to load config", exc_info=True)
            self._config = ZoomRTMSConfig()

        self._client_id = _get_optional_credential("ZOOM_CLIENT_ID")
        self._client_secret = _get_optional_credential("ZOOM_CLIENT_SECRET")

    def _resolve_target_anima(self, meeting_id: str) -> str:
        """Map a meeting to an Anima: mapping → default → none."""
        mapped = self._config.meeting_mapping.get(meeting_id) if meeting_id else None
        return mapped or self._config.default_anima or ""

    async def _close_session(self, session: _MeetingSession) -> None:
        """Cancel tasks and close sockets for a session."""
        session.stopping = True
        for task in (session.task, session.flush_timer):
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        for ws in (session.media_ws, session.signaling_ws):
            if ws is not None:
                with contextlib.suppress(Exception):
                    await ws.close()
        session.signaling_ws = None
        session.media_ws = None


# ── Module-level helpers ─────────────────────────────────────


def _get_optional_credential(env_var: str) -> str:
    """Resolve a credential, returning '' instead of raising if absent."""
    try:
        return get_credential("zoom", "zoom", env_var=env_var)
    except Exception:
        return ""


def _first_url(value: Any) -> str:
    """Zoom sends ``server_urls`` as a string or a list of strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return str(value[0])
    return ""


def _is_allowed_ws_url(url: str) -> bool:
    """SSRF guard for webhook-supplied RTMS URLs.

    Requires a ``wss://`` scheme so an attacker cannot redirect the gateway
    at ``http://`` internal services or plaintext external hosts.  ``ws://``
    is permitted only for loopback hosts to support local development and
    in-process test servers.  This is defense-in-depth layered on top of the
    webhook signature check that authenticates ``server_urls``.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except (ValueError, TypeError):
        return False
    if parsed.scheme == "wss":
        return True
    return parsed.scheme == "ws" and parsed.hostname in ("127.0.0.1", "::1", "localhost")


def _extract_media_url(msg: dict[str, Any]) -> str:
    """Pull the media-server URL from a signaling handshake response."""
    media_server = msg.get("media_server") or {}
    urls = media_server.get("server_urls") or {}
    if isinstance(urls, dict):
        return str(urls.get("all") or urls.get("transcript") or "")
    if isinstance(urls, str):
        return urls
    if isinstance(urls, list) and urls:
        return str(urls[0])
    return ""


def _extract_transcript(msg: dict[str, Any]) -> tuple[str, str]:
    """Return ``(speaker, text)`` from a transcript media message.

    Handles both the flat shape (``user_name`` / ``content`` as text) and
    the nested shape (``content`` as an object with ``data``/``user_name``).
    """
    content = msg.get("content")
    if isinstance(content, dict):
        text = str(content.get("data") or content.get("text") or "").strip()
        speaker = str(content.get("user_name") or msg.get("user_name") or "").strip()
    else:
        text = str(content or "").strip()
        speaker = str(msg.get("user_name") or "").strip()
    return (speaker or t("zoom.unknown_speaker")), text


def _parse(raw: Any) -> dict[str, Any] | None:
    """Decode a WebSocket frame into a JSON object, or ``None``."""
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None
