# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Voice chat WebSocket endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.auth.manager import load_auth, validate_session
from core.config import load_config
from core.config.models import VoiceConfig
from core.voice.session import VoiceSession
from core.voice.stt import VoiceSTT
from core.voice.tts_base import TTSConfig
from core.voice.tts_factory import create_tts_provider

try:
    from server.localhost import _is_safe_localhost_request
except ImportError:

    def _is_safe_localhost_request(_ws: object) -> bool:  # type: ignore[misc]
        return False


logger = logging.getLogger(__name__)

# ── Active session tracking ─────────────────────────────────────

_active_sessions: dict[str, WebSocket] = {}

# ── STT singleton ───────────────────────────────────────────────

_stt_instance: VoiceSTT | None = None


def _get_stt(voice_config: VoiceConfig) -> VoiceSTT:
    """Lazy-load VoiceSTT singleton."""
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = VoiceSTT(
            model_name=voice_config.stt_model,
            device=voice_config.stt_device,
            compute_type=voice_config.stt_compute_type,
        )
    return _stt_instance


def _load_per_anima_voice(
    animas_dir: Path,
    name: str,
    voice_config: VoiceConfig,
) -> TTSConfig:
    """Load per-anima voice settings from status.json."""
    status_path = animas_dir / name / "status.json"
    if not status_path.is_file():
        return TTSConfig(
            provider=voice_config.default_tts_provider,
            voice_id="",
            speed=1.0,
            pitch=0.0,
        )
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return TTSConfig(
            provider=voice_config.default_tts_provider,
            voice_id="",
            speed=1.0,
            pitch=0.0,
        )
    voice_section = data.get("voice") or {}
    if not isinstance(voice_section, dict):
        voice_section = {}
    provider = voice_section.get("tts_provider", voice_config.default_tts_provider)
    return TTSConfig(
        provider=provider,
        voice_id=voice_section.get("voice_id", ""),
        speed=float(voice_section.get("speed", 1.0)),
        pitch=float(voice_section.get("pitch", 0.0)),
        extra=voice_section.get("extra", {}),
    )


def _load_per_anima_voice_front(
    animas_dir: Path,
    name: str,
    voice_config: VoiceConfig,
) -> tuple[str | None, str | None]:
    """Load per-anima voice front lane settings from status.json's ``voice`` section.

    Returns ``(front_model, front_api_base)`` falling back to the global
    config; either may be ``None`` (→ legacy path).
    """
    front_model = getattr(voice_config, "front_model", None) or None
    front_api_base = getattr(voice_config, "front_api_base", None) or None
    status_path = animas_dir / name / "status.json"
    if not status_path.is_file():
        return front_model, front_api_base
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return front_model, front_api_base
    voice_section = data.get("voice") or {}
    if not isinstance(voice_section, dict):
        voice_section = {}
    if voice_section.get("front_model"):
        front_model = voice_section["front_model"]
    if voice_section.get("front_api_base"):
        front_api_base = voice_section["front_api_base"]
    return front_model, front_api_base


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
        auth_config = load_auth()
        if auth_config.auth_mode != "local_trust":
            if not (auth_config.trust_localhost and _is_safe_localhost_request(ws)):
                token = ws.cookies.get("session_token")
                session = validate_session(token) if token else None
                if not session:
                    await ws.close(code=4001, reason="Unauthorized")
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
                    pass
        _active_sessions[name] = ws

        session = None
        try:
            config = load_config()
            voice_config = config.voice

            # Send loading status while STT loads
            await ws.send_json({"type": "status", "state": "loading"})

            stt = _get_stt(voice_config)
            tts_config = _load_per_anima_voice(animas_dir, name, voice_config)
            tts = create_tts_provider(tts_config.provider, voice_config)
            front_model, front_api_base = _load_per_anima_voice_front(animas_dir, name, voice_config)
            logger.info(
                "Voice session created: anima=%s provider=%s voice_id=%s speed=%.1f",
                name,
                tts_config.provider,
                tts_config.voice_id,
                tts_config.speed,
            )

            session = VoiceSession(
                anima_name=name,
                ws=ws,
                stt=stt,
                tts=tts,
                tts_config=tts_config,
                supervisor=supervisor,
                voice_config=voice_config,
                front_model=front_model,
                front_api_base=front_api_base,
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
                pass
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
                    pass
            if _active_sessions.get(name) == ws:
                _active_sessions.pop(name, None)

    return router
