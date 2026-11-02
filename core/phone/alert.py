from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Urgent outbound phone alerts with acknowledgement-aware retries."""

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field

from core.config.schemas import PhoneConfig
from core.phone.audio_store import phone_audio_store
from core.phone.twilio_client import build_webhook_url, create_twilio_client
from core.voice.speech_text import prepare_speech

logger = logging.getLogger(__name__)

_TERMINAL_CALL_STATUSES = frozenset({"completed", "busy", "failed", "no-answer", "canceled"})
_ALERT_RETENTION_SEC = 3600
_ALERT_TTS_TIMEOUT_SEC = 30.0


@dataclass(slots=True)
class PhoneAlert:
    alert_id: str
    anima: str
    config: PhoneConfig
    created_at: float = field(default_factory=time.monotonic)
    attempts: int = 0
    acknowledged: bool = False
    audio_token: str = ""
    spoken_text: str = ""
    task: asyncio.Task[None] | None = None
    attempt_event: asyncio.Event | None = None
    attempt_creating: bool = False
    attempt_call_sids: set[str] = field(default_factory=set)
    completed_call_sids: set[str] = field(default_factory=set)
    finished: bool = False


_alerts: dict[str, PhoneAlert] = {}


def _prune_alerts() -> None:
    now = time.monotonic()
    stale = [
        alert_id
        for alert_id, alert in _alerts.items()
        if alert.finished and now - alert.created_at > _ALERT_RETENTION_SEC
    ]
    for alert_id in stale:
        _alerts.pop(alert_id, None)


async def start_alert(
    anima: str,
    subject: str,
    body: str,
    *,
    phone_config: PhoneConfig | None = None,
) -> str:
    """Queue an alert and return its opaque ID without waiting for Twilio."""
    from core.config import load_config

    app_config = load_config()
    config = phone_config or app_config.phone
    if not config.enabled:
        raise ValueError("Phone channel is disabled")
    if anima != config.anima:
        raise ValueError("Phone channel is configured for a different Anima")

    _prune_alerts()
    alert_id = secrets.token_urlsafe(18)
    alert = PhoneAlert(alert_id=alert_id, anima=anima, config=config)
    _alerts[alert_id] = alert
    alert.task = asyncio.create_task(
        _run_alert(alert, subject, body, voice_config=app_config.voice, locale=app_config.locale),
        name=f"phone-alert-{alert_id}",
    )
    return alert_id


async def _run_alert(alert: PhoneAlert, subject: str, body: str, *, voice_config: object, locale: str) -> None:
    from core.i18n import t
    from core.phone.speech import synthesize_speech

    try:
        alert_text = t("phone.alert_intro", locale=locale, anima=alert.anima, subject=subject, body=body)
        link_placeholder = t("phone.link_placeholder", locale=locale)
        alert.spoken_text = prepare_speech(
            alert_text,
            provider=None,
            link_placeholder=link_placeholder,
            max_chars=400,
        ).spoken
        try:
            audio = await asyncio.wait_for(
                synthesize_speech(
                    alert.anima,
                    alert_text,
                    voice_config=voice_config,
                    link_placeholder=link_placeholder,
                    max_chars=400,
                ),
                timeout=_ALERT_TTS_TIMEOUT_SEC,
            )
            alert.audio_token = phone_audio_store.put(audio)
        except Exception:
            # A call read in Twilio's voice is better than one that never rings.
            logger.warning("Phone alert speech failed for anima=%s; Twilio will read the alert", alert.anima)

        max_attempts = alert.config.alert_max_attempts
        for attempt in range(1, max_attempts + 1):
            if alert.acknowledged:
                break
            alert.attempts = attempt
            alert.attempt_call_sids.clear()
            alert.attempt_event = asyncio.Event()
            alert.attempt_creating = True

            try:
                client = create_twilio_client(alert.config)
            except Exception:
                logger.warning("Unable to start phone alert attempt %d for anima=%s", attempt, alert.anima)
                alert.attempt_event.set()
                if attempt < max_attempts and not alert.acknowledged:
                    await asyncio.sleep(alert.config.alert_retry_interval_sec)
                continue

            voice_url = build_webhook_url(
                alert.config.public_base_url,
                "/api/webhooks/twilio/voice",
                params={"alert": alert.alert_id},
            )
            status_url = build_webhook_url(
                alert.config.public_base_url,
                "/api/webhooks/twilio/status",
                params={"alert": alert.alert_id},
            )
            for number in alert.config.owner_numbers:
                if alert.acknowledged:
                    break
                try:
                    result = await client.create_call(
                        to=number,
                        from_=alert.config.from_number,
                        url=voice_url,
                        status_callback=status_url,
                    )
                except Exception:
                    # Do not include the request, credentials, phone number, or
                    # provider response body in logs.
                    logger.warning("Twilio could not create phone alert call for anima=%s", alert.anima)
                    continue
                sid = str(result.get("sid", ""))
                if sid and sid not in alert.completed_call_sids:
                    alert.attempt_call_sids.add(sid)

            alert.attempt_creating = False
            if not alert.attempt_call_sids:
                alert.attempt_event.set()
            if attempt < max_attempts and not alert.acknowledged:
                await alert.attempt_event.wait()
                if not alert.acknowledged:
                    await asyncio.sleep(alert.config.alert_retry_interval_sec)
    except asyncio.CancelledError:
        raise
    except Exception:
        # TTS/provider exception messages can include request details. Keep the
        # log deliberately generic and never expose credentials or PIN values.
        logger.warning("Phone alert preparation failed for anima=%s", alert.anima)
    finally:
        alert.finished = True


def get_alert(alert_id: str) -> PhoneAlert | None:
    """Return an in-memory alert record, pruning old completed records first."""
    _prune_alerts()
    return _alerts.get(alert_id)


def acknowledge_alert(alert_id: str) -> None:
    """Mark an alert acknowledged and wake its retry loop."""
    alert = get_alert(alert_id)
    if alert is None:
        return
    alert.acknowledged = True
    if alert.attempt_event is not None:
        alert.attempt_event.set()


def notify_call_status(alert_id: str, call_sid: str, status: str) -> None:
    """Notify the retry loop that a Twilio call reached a terminal state."""
    if status.lower() not in _TERMINAL_CALL_STATUSES or not call_sid:
        return
    alert = get_alert(alert_id)
    if alert is None:
        return
    alert.completed_call_sids.add(call_sid)
    if call_sid in alert.attempt_call_sids:
        alert.attempt_call_sids.discard(call_sid)
        if not alert.attempt_creating and not alert.attempt_call_sids and alert.attempt_event is not None:
            alert.attempt_event.set()


async def wait_for_alert(alert_id: str) -> None:
    """Wait until an alert's background task exits (mainly useful in tests)."""
    alert = get_alert(alert_id)
    if alert is not None and alert.task is not None:
        await alert.task


async def clear_alerts() -> None:
    """Cancel and clear alert tasks (used by unit-test cleanup)."""
    tasks = [alert.task for alert in _alerts.values() if alert.task is not None and not alert.task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _alerts.clear()


__all__ = [
    "PhoneAlert",
    "acknowledge_alert",
    "clear_alerts",
    "get_alert",
    "notify_call_status",
    "start_alert",
    "wait_for_alert",
]
