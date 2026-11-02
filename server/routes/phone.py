from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Signed Twilio webhooks for phone PIN, alert, and Media Streams setup."""

import asyncio
import hmac
import logging
import re
from html import escape
from urllib.parse import parse_qs, urlsplit, urlunsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from core.config.models import load_config
from core.config.schemas import AnimaWorksConfig, PhoneConfig
from core.config.vault import get_vault_manager
from core.phone.alert import PhoneAlert, acknowledge_alert, get_alert, notify_call_status
from core.phone.audio_store import phone_audio_store
from core.phone.session import PhoneSession, phone_sessions
from core.phone.speech import synthesize_speech
from core.phone.stream_tokens import phone_stream_tokens
from core.phone.twilio_client import build_webhook_url, get_twilio_credentials, validate_signature
from core.voice.speech_text import prepare_speech

logger = logging.getLogger(__name__)

_MAX_PIN_FAILURES = 3
# Twilio gives up on a webhook after 15 seconds, and one response can need two prompts.
_PROMPT_TTS_TIMEOUT_SEC = 7.0


def _get_phone_config() -> tuple[AnimaWorksConfig, PhoneConfig]:
    app_config = load_config()
    phone_config = app_config.phone
    if not phone_config.enabled:
        raise HTTPException(status_code=404, detail="Not found")
    return app_config, phone_config


def _form_params(body: bytes) -> dict[str, str | list[str]]:
    parsed = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
    return {key: values if len(values) > 1 else (values[0] if values else "") for key, values in parsed.items()}


def _form_value(params: dict[str, str | list[str]], key: str, default: str = "") -> str:
    value = params.get(key, default)
    if isinstance(value, list):
        return value[-1] if value else default
    return value


def _public_request_url(request: Request, phone_config: PhoneConfig) -> str:
    query = request.scope.get("query_string", b"").decode("ascii", errors="replace")
    return build_webhook_url(phone_config.public_base_url, request.url.path, raw_query=query or None)


async def _verified_request(
    request: Request,
) -> tuple[AnimaWorksConfig, PhoneConfig, dict[str, str | list[str]]]:
    app_config, phone_config = _get_phone_config()
    params = _form_params(await request.body())
    _, auth_token = get_twilio_credentials(phone_config)
    signature = request.headers.get("X-Twilio-Signature", "")
    if not validate_signature(
        auth_token or "",
        _public_request_url(request, phone_config),
        params,
        signature,
    ):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")
    return app_config, phone_config, params


def _xml_attr(name: str, value: object) -> str:
    return f' {name}="{escape(str(value), quote=True)}"'


def _tag(tag_name: str, content: str = "", **attrs: object) -> str:
    attributes = "".join(_xml_attr(key, value) for key, value in attrs.items() if value is not None)
    return f"<{tag_name}{attributes}>{content}</{tag_name}>"


def _twiml(*parts: str) -> Response:
    return Response(f"<Response>{''.join(parts)}</Response>", media_type="application/xml")


def _hangup() -> str:
    return "<Hangup/>"


def _say(app_config: AnimaWorksConfig, text: str) -> str:
    language = "en-US" if app_config.locale == "en" else "ja-JP"
    return _tag("Say", escape(text), language=language)


def _reject() -> str:
    return "<Reject/>"


def _hook_url(phone_config: PhoneConfig, endpoint: str, **params: str | int) -> str:
    return build_webhook_url(
        phone_config.public_base_url,
        f"/api/webhooks/twilio/{endpoint}",
        params=params or None,
    )


def _stream_ws_url(phone_config: PhoneConfig) -> str:
    """Build the public WSS URL Twilio uses to open the bidirectional stream."""
    parsed = urlsplit(phone_config.public_base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("phone.public_base_url must be an absolute HTTP(S) URL")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/api/webhooks/twilio/stream"
    return urlunsplit((scheme, parsed.netloc, path, "", ""))


def _connect_stream(phone_config: PhoneConfig, token: str) -> str:
    parameter = _tag("Parameter", **{"name": "token", "value": token})
    stream = _tag("Stream", parameter, url=_stream_ws_url(phone_config))
    return _tag("Connect", stream)


async def _play_text(
    request: Request,
    app_config: AnimaWorksConfig,
    phone_config: PhoneConfig,
    anima: str,
    text: str,
    *,
    cache_key: str | None = None,
) -> str:
    from core.i18n import t

    link_placeholder = t("phone.link_placeholder", locale=app_config.locale)
    tts_provider = app_config.voice.default_tts_provider
    try:
        from core.paths import get_animas_dir
        from core.voice.voice_config import load_per_anima_voice

        animas_dir = getattr(request.app.state, "animas_dir", None) or get_animas_dir()
        tts_config = load_per_anima_voice(animas_dir, anima, app_config.voice)
        tts_provider = tts_config.provider
        audio = await asyncio.wait_for(
            synthesize_speech(
                anima,
                text,
                animas_dir=animas_dir,
                voice_config=app_config.voice,
                cache_key=cache_key,
                link_placeholder=link_placeholder,
                max_chars=400,
            ),
            timeout=_PROMPT_TTS_TIMEOUT_SEC,
        )
        token = phone_audio_store.put(audio)
        url = build_webhook_url(
            phone_config.public_base_url,
            f"/api/webhooks/twilio/audio/{token}.wav",
        )
        return _tag("Play", escape(url))
    except Exception:
        logger.warning("Phone speech synthesis failed for anima=%s", anima)
        spoken = prepare_speech(text, provider=tts_provider, link_placeholder=link_placeholder, max_chars=400).spoken
        return _say(app_config, spoken)


async def _play_key(
    request: Request,
    app_config: AnimaWorksConfig,
    phone_config: PhoneConfig,
    key: str,
    **kwargs: object,
) -> str:
    from core.i18n import t

    text = t(key, locale=app_config.locale, **kwargs)
    return await _play_text(
        request,
        app_config,
        phone_config,
        phone_config.anima,
        text,
        cache_key=f"{app_config.locale}:{key}:{text}",
    )


_WARM_KEYS = (
    "phone.pin_prompt",
    "phone.pin_invalid",
    "phone.pin_locked",
    "phone.alert_ack",
    "phone.alert_invalid_choice",
)
_warm_task: asyncio.Task[None] | None = None


def _start_warmup(request: Request, app_config: AnimaWorksConfig, phone_config: PhoneConfig) -> None:
    """Warm only the short PIN and alert prompts used by TwiML playback."""
    global _warm_task
    if _warm_task is not None and not _warm_task.done():
        return

    async def _warm() -> None:
        for key in _WARM_KEYS:
            await _play_key(request, app_config, phone_config, key)
        await _play_key(request, app_config, phone_config, "phone.alert_choice", anima=phone_config.anima)

    _warm_task = asyncio.create_task(_warm(), name="phone-warm-prompts")


def _pin_gather(phone_config: PhoneConfig, prompt: str) -> str:
    return _tag(
        "Gather",
        prompt,
        input="dtmf",
        numDigits="4",
        action=_hook_url(phone_config, "pin"),
        actionOnEmptyResult="true",
        method="POST",
        timeout="8",
    )


def _alert_speech(app_config: AnimaWorksConfig, phone_config: PhoneConfig, alert: PhoneAlert) -> str:
    """Play the synthesized alert, or have Twilio read it when synthesis failed."""
    if not alert.audio_token:
        return _say(app_config, alert.spoken_text)
    audio_url = build_webhook_url(
        phone_config.public_base_url,
        f"/api/webhooks/twilio/audio/{alert.audio_token}.wav",
    )
    return _tag("Play", escape(audio_url))


def _alert_gather(phone_config: PhoneConfig, alert_id: str, prompt: str) -> str:
    return _tag(
        "Gather",
        prompt,
        input="dtmf",
        numDigits="1",
        action=_hook_url(phone_config, "voice", alert=alert_id, phase="choice"),
        actionOnEmptyResult="true",
        method="POST",
        timeout="8",
    )


def _pin_from_vault(phone_config: PhoneConfig) -> str | None:
    try:
        return get_vault_manager().get("shared", phone_config.pin_vault_key)
    except Exception:
        logger.warning("Phone PIN could not be read from the shared vault")
        return None


def _session_for_call(params: dict[str, str | list[str]]) -> PhoneSession | None:
    call_sid = _form_value(params, "CallSid")
    return phone_sessions.get(call_sid) if call_sid else None


def _terminal_call_status(status: str) -> bool:
    return status.lower().replace("_", "-") in {"completed", "busy", "failed", "no-answer", "canceled"}


def create_phone_router() -> APIRouter:
    """Create the Twilio phone webhook and Media Streams router."""
    router = APIRouter(prefix="/webhooks/twilio", tags=["phone"])

    @router.post("/voice")
    async def voice(request: Request) -> Response:
        """Handle inbound calls, alert playback, and alert DTMF responses."""
        app_config, phone_config, params = await _verified_request(request)
        call_sid = _form_value(params, "CallSid")
        if not call_sid:
            return _twiml(_hangup())

        _start_warmup(request, app_config, phone_config)
        alert_id = request.query_params.get("alert", "")
        phase = request.query_params.get("phase", "")
        if alert_id:
            alert = get_alert(alert_id)
            if alert is None or alert.anima != phone_config.anima or not (alert.audio_token or alert.spoken_text):
                return _twiml(_hangup())

            session = phone_sessions.get(call_sid)
            if session is None:
                session = phone_sessions.create(
                    call_sid,
                    phone_config.anima,
                    kind="alert",
                    alert_id=alert_id,
                )
            elif session.kind != "alert" or session.alert_id != alert_id:
                return _twiml(_hangup())

            if phase == "choice":
                digit = _form_value(params, "Digits")
                if digit == "1":
                    session.acknowledged = True
                    acknowledge_alert(alert_id)
                    return _twiml(await _play_key(request, app_config, phone_config, "phone.alert_ack"), _hangup())
                if digit == "2":
                    session.acknowledged = True
                    acknowledge_alert(alert_id)
                    pin_prompt = await _play_key(request, app_config, phone_config, "phone.pin_prompt")
                    return _twiml(_pin_gather(phone_config, pin_prompt))
                if not digit:
                    if session.alert_repeat_done:
                        return _twiml(_hangup())
                    session.alert_repeat_done = True
                    return _twiml(_alert_gather(phone_config, alert_id, _alert_speech(app_config, phone_config, alert)))
                invalid_prompt = await _play_key(request, app_config, phone_config, "phone.alert_invalid_choice")
                choice_prompt = await _play_key(
                    request,
                    app_config,
                    phone_config,
                    "phone.alert_choice",
                    anima=phone_config.anima,
                )
                return _twiml(_alert_gather(phone_config, alert_id, invalid_prompt + choice_prompt))

            alert_speech = _alert_speech(app_config, phone_config, alert)
            choice_prompt = await _play_key(
                request,
                app_config,
                phone_config,
                "phone.alert_choice",
                anima=phone_config.anima,
            )
            # Inside Gather, so a key pressed during the announcement is not thrown away.
            return _twiml(
                _alert_gather(phone_config, alert_id, alert_speech + choice_prompt),
                alert_speech,
                _hangup(),
            )

        direction = _form_value(params, "Direction").lower()
        if direction == "inbound":
            caller = _form_value(params, "From").strip()
            if caller not in phone_config.owner_numbers:
                return _twiml(_reject())
            phone_sessions.create(call_sid, phone_config.anima, kind="inbound")
            pin_prompt = await _play_key(request, app_config, phone_config, "phone.pin_prompt")
            return _twiml(_pin_gather(phone_config, pin_prompt))

        return _twiml(_hangup())

    @router.post("/pin")
    async def pin(request: Request) -> Response:
        """Validate the PIN and issue a one-use credential for the phone stream."""
        app_config, phone_config, params = await _verified_request(request)
        session = _session_for_call(params)
        if session is None:
            return _twiml(_hangup())

        supplied_pin = _form_value(params, "Digits")
        stored_pin = _pin_from_vault(phone_config)
        valid_pin = bool(
            stored_pin
            and re.fullmatch(r"[0-9]{4}", stored_pin)
            and re.fullmatch(r"[0-9]{4}", supplied_pin)
            and hmac.compare_digest(stored_pin, supplied_pin)
        )
        if not valid_pin:
            session.pin_failures += 1
            if session.pin_failures >= _MAX_PIN_FAILURES:
                locked = await _play_key(request, app_config, phone_config, "phone.pin_locked")
                return _twiml(locked, _hangup())
            invalid = await _play_key(request, app_config, phone_config, "phone.pin_invalid")
            prompt = await _play_key(request, app_config, phone_config, "phone.pin_prompt")
            return _twiml(invalid, _pin_gather(phone_config, prompt))

        session.authenticated = True
        session.pin_failures = 0
        phone_stream_tokens.revoke_for_call(session.call_sid)
        token = phone_stream_tokens.issue(session.call_sid)
        return _twiml(_connect_stream(phone_config, token))

    @router.post("/status")
    async def status(request: Request) -> Response:
        """Handle Twilio call status callbacks and release terminal sessions."""
        _app_config, phone_config, params = await _verified_request(request)
        call_sid = _form_value(params, "CallSid")
        call_status = _form_value(params, "CallStatus").lower().replace("_", "-")
        if call_sid and _terminal_call_status(call_status):
            session = phone_sessions.get(call_sid)
            alert_id = request.query_params.get("alert", "") or (session.alert_id if session else "")
            if alert_id:
                notify_call_status(alert_id, call_sid, call_status)
            phone_stream_tokens.revoke_for_call(call_sid)
            phone_sessions.discard(call_sid)
        return Response(status_code=204)

    @router.get("/audio/{token}.wav")
    async def audio(token: str) -> Response:
        """Serve an unexpired synthesized WAV by its unguessable token."""
        _get_phone_config()
        audio_bytes = phone_audio_store.get(token)
        if audio_bytes is None:
            raise HTTPException(status_code=404, detail="Audio not found")
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Cache-Control": "private, no-store"},
        )

    from server.routes.phone_stream import register_phone_stream_route

    register_phone_stream_route(router)
    return router


__all__ = ["create_phone_router"]
