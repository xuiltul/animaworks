# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Voice chat WebSocket endpoint; per-Anima status.json settings use shared helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.auth.manager import load_auth, run_auth_call, validate_session
from core.config import load_config
from core.config.models import VoiceConfig
from core.voice.session import VoiceSession  # noqa: F401 -- compatibility seam used by route tests
from core.voice.session_factory import build_voice_session
from core.voice.stt import VoiceSTT
from core.voice.transport import VoiceTransport
from core.voice.tts_factory import create_tts_provider
from core.voice.voice_config import load_per_anima_voice as _load_per_anima_voice
from core.voice.voice_config import load_per_anima_voice_front as _load_per_anima_voice_front

try:
    from server.localhost import _is_safe_localhost_request
except ImportError:

    def _is_safe_localhost_request(_ws: object) -> bool:  # type: ignore[misc]
        return False


logger = logging.getLogger(__name__)


class FastAPIWebSocketVoiceTransport:
    """Adapt a FastAPI WebSocket to the voice transport protocol."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    async def send_event(self, event: dict[str, Any]) -> None:
        await self._websocket.send_json(event)

    async def send_audio(self, data: bytes) -> None:
        await self._websocket.send_bytes(data)


# ── Active session tracking ─────────────────────────────────────

_active_sessions: dict[str, WebSocket] = {}

# ── STT singleton ───────────────────────────────────────────────

_stt_instance: VoiceSTT | None = None
_stt_instance_config: tuple[str, str, str, str | None] | None = None
_stt_lock = threading.Lock()


def _get_stt(voice_config: VoiceConfig) -> VoiceSTT:
    """Return the thread-safe STT singleton for the current voice settings."""
    global _stt_instance, _stt_instance_config
    config = (
        voice_config.stt_model,
        voice_config.stt_device,
        voice_config.stt_compute_type,
        voice_config.stt_language,
    )
    with _stt_lock:
        if _stt_instance is None or _stt_instance_config != config:
            _stt_instance = VoiceSTT(
                model_name=voice_config.stt_model,
                device=voice_config.stt_device,
                compute_type=voice_config.stt_compute_type,
                language=voice_config.stt_language,
            )
            _stt_instance_config = config
        return _stt_instance


# ── Helpers ──────────────────────────────────────────────────────


def _speech_end_done(task: asyncio.Task[None]) -> None:
    """Log unhandled exceptions from fire-and-forget speech_end tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.exception("handle_speech_end failed: %s", exc, exc_info=exc)


# ── Router ──────────────────────────────────────────────────────


def create_voice_router() -> APIRouter:
    """Create the voice WebSocket router."""
    router = APIRouter()

    @router.websocket("/ws/voice/{name}")
    async def voice_websocket(ws: WebSocket, name: str) -> None:
        """Voice conversation WebSocket for a specific Anima."""
        await ws.accept()

        # Guard: reject path-traversal attempts
        if "/" in name or ".." in name or not name or name.startswith("."):
            await ws.close(code=4000, reason="Invalid anima name")
            return

        # Auth check (same pattern as websocket_route.py)
        try:
            auth_config = await run_auth_call(load_auth)
            if auth_config.auth_mode != "local_trust":
                if not (auth_config.trust_localhost and _is_safe_localhost_request(ws)):
                    token = ws.cookies.get("session_token")
                    session = await run_auth_call(validate_session, token) if token else None
                    if not session:
                        await ws.close(code=4001, reason="Unauthorized")
                        return
        except TimeoutError:
            await ws.close(code=1013, reason="Try again later")
            return

        supervisor = ws.app.state.supervisor
        animas_dir: Path = ws.app.state.animas_dir

        # 1 Anima = 1 active voice session: close existing if any
        if name in _active_sessions:
            old_ws = _active_sessions.pop(name, None)
            if old_ws and old_ws != ws:
                try:
                    await old_ws.close(code=4000, reason="Replaced by new session")
                except Exception:
                    logger.debug("Best-effort operation failed", exc_info=True)
        _active_sessions[name] = ws

        session = None
        try:
            config = load_config()
            voice_config = config.voice

            # Send loading status while STT loads
            await ws.send_json({"type": "status", "state": "loading"})

            stt = _get_stt(voice_config)
            logger.debug("Resolving per-Anima front voice settings from %s", animas_dir / name / "status.json")
            transport: VoiceTransport = FastAPIWebSocketVoiceTransport(ws)
            session = build_voice_session(
                anima_name=name,
                transport=transport,
                stt=stt,
                supervisor=supervisor,
                animas_dir=animas_dir,
                voice_config=voice_config,
                channel="web",
                human_notification_config=config.human_notification,
                tts_factory=create_tts_provider,
                tts_config_loader=_load_per_anima_voice,
                front_settings_loader=_load_per_anima_voice_front,
            )
            tts_config = session._tts_config
            logger.info(
                "Voice session created: anima=%s provider=%s voice_id=%s speed=%.1f",
                name,
                tts_config.provider,
                tts_config.voice_id,
                tts_config.speed,
            )

            await ws.send_json({"type": "status", "state": "ready"})

            running_tasks: list[asyncio.Task] = []
            # Greet on connect: instant feedback + warms the chat runner path.
            greet_task = asyncio.create_task(session.greet_and_speak())
            greet_task.add_done_callback(_speech_end_done)
            running_tasks.append(greet_task)
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg["type"] == "websocket.receive":
                    if "bytes" in msg and msg["bytes"]:
                        await session.handle_audio_chunk(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        try:
                            data = json.loads(msg["text"])
                            msg_type = data.get("type")
                            if msg_type == "speech_end":
                                task = asyncio.create_task(session.handle_speech_end())
                                task.add_done_callback(_speech_end_done)
                                running_tasks.append(task)
                            elif msg_type == "interrupt":
                                await session.handle_interrupt()
                            elif msg_type == "barge_probe":
                                await session.handle_barge_probe()
                            elif msg_type == "discard_audio":
                                await session.handle_discard_audio()
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON in voice WebSocket: %s", msg["text"][:100])

        except WebSocketDisconnect:
            logger.info("Voice WebSocket disconnected: anima=%s", name)
        except Exception as e:
            logger.exception("Voice WebSocket error anima=%s: %s", name, e)
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                logger.debug("Best-effort operation failed", exc_info=True)
        finally:
            for t in running_tasks:
                if not t.done():
                    t.cancel()
            # Cancel the proactive idle watcher / TTS worker so a closed popup
            # stops polling the front LLM every couple of seconds (C1).
            if session is not None:
                try:
                    await session.close()
                except Exception:
                    logger.debug("Best-effort operation failed", exc_info=True)
            if _active_sessions.get(name) == ws:
                _active_sessions.pop(name, None)

    return router
