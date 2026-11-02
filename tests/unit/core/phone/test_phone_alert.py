from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

from core.config.models import PhoneConfig
from core.phone import alert as phone_alert
from core.phone.audio_store import phone_audio_store


@pytest.fixture(autouse=True)
async def _clear_alert_state():
    await phone_alert.clear_alerts()
    phone_audio_store.clear()
    yield
    await phone_alert.clear_alerts()
    phone_audio_store.clear()


async def _wait_until(predicate, *, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


class _FakeTwilio:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def create_call(self, *, to: str, from_: str, url: str, status_callback: str) -> dict[str, str]:
        sid = f"CA{len(self.calls) + 1}"
        self.calls.append({"sid": sid, "to": to, "from": from_, "url": url, "status_callback": status_callback})
        return {"sid": sid}


def _config(*, attempts: int = 3, interval: float = 0.01, owner_numbers: list[str] | None = None) -> PhoneConfig:
    return PhoneConfig(
        enabled=True,
        anima="aoi",
        public_base_url="https://phone.example.test",
        from_number="+15557654321",
        owner_numbers=owner_numbers or ["+15551234567"],
        alert_max_attempts=attempts,
        alert_retry_interval_sec=interval,
    )


def _patch_alert_dependencies(monkeypatch: pytest.MonkeyPatch, twilio: _FakeTwilio) -> None:
    from core.phone import speech

    async def fake_synthesize(*_args, **_kwargs) -> bytes:
        return b"RIFF-alert-wav"

    monkeypatch.setattr(speech, "synthesize_speech", fake_synthesize)
    monkeypatch.setattr(phone_alert, "create_twilio_client", lambda _config: twilio)


@pytest.mark.asyncio
async def test_unacknowledged_completed_call_retries_and_ack_stops_future_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    twilio = _FakeTwilio()
    _patch_alert_dependencies(monkeypatch, twilio)

    alert_id = await phone_alert.start_alert("aoi", "Disk", "Disk failure", phone_config=_config())
    await _wait_until(lambda: len(twilio.calls) == 1)
    first = twilio.calls[0]
    assert first["url"].startswith("https://phone.example.test/api/webhooks/twilio/voice?alert=")
    assert first["status_callback"].startswith("https://phone.example.test/api/webhooks/twilio/status?alert=")

    phone_alert.notify_call_status(alert_id, first["sid"], "completed")
    await _wait_until(lambda: len(twilio.calls) == 2)
    phone_alert.acknowledge_alert(alert_id)
    await phone_alert.wait_for_alert(alert_id)
    await asyncio.sleep(0.03)

    state = phone_alert.get_alert(alert_id)
    assert state is not None
    assert state.acknowledged is True
    assert state.attempts == 2
    assert len(twilio.calls) == 2
    assert state.audio_token


@pytest.mark.asyncio
async def test_alert_still_rings_when_speech_synthesis_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.phone import speech

    twilio = _FakeTwilio()
    _patch_alert_dependencies(monkeypatch, twilio)

    async def broken_synthesize(*_args, **_kwargs) -> bytes:
        raise RuntimeError("TTS is down")

    monkeypatch.setattr(speech, "synthesize_speech", broken_synthesize)

    alert_id = await phone_alert.start_alert("aoi", "Disk", "Disk failure", phone_config=_config(attempts=1))
    await phone_alert.wait_for_alert(alert_id)

    state = phone_alert.get_alert(alert_id)
    assert state is not None
    assert len(twilio.calls) == 1
    assert state.audio_token == ""
    assert "Disk failure" in state.spoken_text


@pytest.mark.asyncio
async def test_alert_still_rings_when_speech_synthesis_hangs(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.phone import speech

    twilio = _FakeTwilio()
    _patch_alert_dependencies(monkeypatch, twilio)

    async def hanging_synthesize(*_args, **_kwargs) -> bytes:
        await asyncio.Event().wait()
        return b""

    monkeypatch.setattr(speech, "synthesize_speech", hanging_synthesize)
    monkeypatch.setattr(phone_alert, "_ALERT_TTS_TIMEOUT_SEC", 0.01)

    alert_id = await phone_alert.start_alert("aoi", "Disk", "Disk failure", phone_config=_config(attempts=1))
    await phone_alert.wait_for_alert(alert_id)

    assert len(twilio.calls) == 1


@pytest.mark.asyncio
async def test_retry_waits_for_every_owner_call_to_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    twilio = _FakeTwilio()
    _patch_alert_dependencies(monkeypatch, twilio)

    alert_id = await phone_alert.start_alert(
        "aoi",
        "Network",
        "Network failure",
        phone_config=_config(owner_numbers=["+15550000001", "+15550000002"]),
    )
    await _wait_until(lambda: len(twilio.calls) == 2)

    phone_alert.notify_call_status(alert_id, twilio.calls[0]["sid"], "busy")
    await asyncio.sleep(0.02)
    assert len(twilio.calls) == 2

    phone_alert.notify_call_status(alert_id, twilio.calls[1]["sid"], "no-answer")
    await _wait_until(lambda: len(twilio.calls) == 4)
    phone_alert.acknowledge_alert(alert_id)
    await phone_alert.wait_for_alert(alert_id)

    assert len(twilio.calls) == 4


@pytest.mark.asyncio
async def test_alert_stops_after_configured_maximum_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    twilio = _FakeTwilio()
    _patch_alert_dependencies(monkeypatch, twilio)

    alert_id = await phone_alert.start_alert(
        "aoi",
        "Service",
        "Service is unavailable",
        phone_config=_config(attempts=2, interval=0.001),
    )
    await _wait_until(lambda: len(twilio.calls) == 1)
    phone_alert.notify_call_status(alert_id, twilio.calls[0]["sid"], "failed")
    await _wait_until(lambda: len(twilio.calls) == 2)
    phone_alert.notify_call_status(alert_id, twilio.calls[1]["sid"], "completed")
    await phone_alert.wait_for_alert(alert_id)
    await asyncio.sleep(0.01)

    state = phone_alert.get_alert(alert_id)
    assert state is not None and state.finished is True
    assert state.attempts == 2
    assert len(twilio.calls) == 2


@pytest.mark.asyncio
async def test_nonterminal_status_does_not_trigger_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    twilio = _FakeTwilio()
    _patch_alert_dependencies(monkeypatch, twilio)

    alert_id = await phone_alert.start_alert("aoi", "Service", "Alert", phone_config=_config())
    await _wait_until(lambda: len(twilio.calls) == 1)
    phone_alert.notify_call_status(alert_id, twilio.calls[0]["sid"], "ringing")
    await asyncio.sleep(0.02)

    assert len(twilio.calls) == 1
    await phone_alert.clear_alerts()
