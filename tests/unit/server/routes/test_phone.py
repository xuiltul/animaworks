from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import hashlib
import hmac
from types import SimpleNamespace
from urllib.parse import urlencode
from xml.etree import ElementTree

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.routes.phone as phone_routes
from core.config.models import AnimaWorksConfig, PhoneConfig
from core.phone.audio_store import phone_audio_store
from core.phone.session import phone_sessions
from core.phone.stream_tokens import phone_stream_tokens

_TEST_AUTH_TOKEN = "test-twilio-token-not-a-real-secret"
_PUBLIC_BASE_URL = "https://phone.example.test"
_VOICE_PATH = "/api/webhooks/twilio/voice"


@pytest.fixture
def phone_app(monkeypatch: pytest.MonkeyPatch, tmp_path):
    config = AnimaWorksConfig(
        phone=PhoneConfig(
            enabled=True,
            anima="aoi",
            public_base_url=_PUBLIC_BASE_URL,
            owner_numbers=["+15551234567"],
            from_number="+15557654321",
            thread_id="phone",
            turn_timeout_sec=30,
        )
    )
    app = FastAPI()
    app.include_router(phone_routes.create_phone_router(), prefix="/api")
    app.state.animas_dir = tmp_path / "animas"

    monkeypatch.setattr(phone_routes, "load_config", lambda: config)
    monkeypatch.setattr(phone_routes, "get_twilio_credentials", lambda _config: ("AC123", _TEST_AUTH_TOKEN))
    monkeypatch.setattr(
        phone_routes,
        "get_vault_manager",
        lambda: SimpleNamespace(get=lambda _section, _key: "1234"),
    )
    monkeypatch.setattr(phone_routes, "_start_warmup", lambda *_args: None)

    async def fake_speech(_anima: str, _text: str, **_kwargs) -> bytes:
        return b"RIFF-test-wav"

    monkeypatch.setattr(phone_routes, "synthesize_speech", fake_speech)
    phone_sessions.clear()
    phone_audio_store.clear()
    phone_stream_tokens.clear()
    yield app, config
    phone_sessions.clear()
    phone_audio_store.clear()
    phone_stream_tokens.clear()


def _twilio_signature(url: str, form: dict[str, str]) -> str:
    payload = url + "".join(key + form[key] for key in sorted(form))
    digest = hmac.new(_TEST_AUTH_TOKEN.encode(), payload.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def _signed_post(
    client: TestClient,
    path: str,
    form: dict[str, str],
    *,
    query: str = "",
    signature: str | None = None,
) -> object:
    target = f"{path}?{query}" if query else path
    public_url = f"{_PUBLIC_BASE_URL}{target}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    headers["X-Twilio-Signature"] = signature or _twilio_signature(public_url, form)
    return client.post(target, content=urlencode(form), headers=headers)


def _stream_details(twiml: str) -> tuple[str, str]:
    root = ElementTree.fromstring(twiml)
    stream = root.find("./Connect/Stream")
    assert stream is not None
    parameter = stream.find("Parameter")
    assert parameter is not None
    assert parameter.attrib["name"] == "token"
    return stream.attrib["url"], parameter.attrib["value"]


def test_phone_post_requires_twilio_signature(phone_app) -> None:
    app, _config = phone_app
    with TestClient(app) as client:
        missing = client.post(_VOICE_PATH, data={"CallSid": "CA1", "Direction": "inbound"})
        invalid = _signed_post(
            client,
            _VOICE_PATH,
            {"CallSid": "CA1", "Direction": "inbound"},
            signature="not-a-valid-signature",
        )

    assert missing.status_code == 403
    assert invalid.status_code == 403


def test_disabled_phone_endpoints_return_404(phone_app) -> None:
    app, config = phone_app
    config.phone.enabled = False
    with TestClient(app) as client:
        webhook = client.post(_VOICE_PATH, data={"CallSid": "CA1"})
        audio = client.get("/api/webhooks/twilio/audio/unknown.wav")

    assert webhook.status_code == 404
    assert audio.status_code == 404


def test_unregistered_inbound_number_is_rejected(phone_app) -> None:
    app, _config = phone_app
    with TestClient(app) as client:
        response = _signed_post(
            client,
            _VOICE_PATH,
            {"CallSid": "CA-UNREGISTERED", "Direction": "inbound", "From": "+15550000000"},
        )

    assert response.status_code == 200
    assert "<Reject/>" in response.text
    assert phone_sessions.get("CA-UNREGISTERED") is None


def test_registered_caller_pin_returns_wss_connect_stream(phone_app) -> None:
    app, _config = phone_app
    with TestClient(app) as client:
        incoming = _signed_post(
            client,
            _VOICE_PATH,
            {"CallSid": "CA-VOICE-1", "Direction": "inbound", "From": "+15551234567"},
        )
        pin = _signed_post(
            client,
            "/api/webhooks/twilio/pin",
            {"CallSid": "CA-VOICE-1", "Digits": "1234"},
        )

    assert incoming.status_code == 200 and 'input="dtmf"' in incoming.text
    assert pin.status_code == 200
    stream_url, token = _stream_details(pin.text)
    assert stream_url == "wss://phone.example.test/api/webhooks/twilio/stream"
    assert token
    assert phone_stream_tokens.consume(token, "CA-VOICE-1") is True
    assert phone_stream_tokens.consume(token, "CA-VOICE-1") is False
    assert "<Gather" not in pin.text
    assert "phone.greeting" not in pin.text


def test_legacy_turn_and_poll_webhooks_are_removed(phone_app) -> None:
    app, _config = phone_app
    with TestClient(app) as client:
        turn = client.post("/api/webhooks/twilio/turn")
        poll = client.post("/api/webhooks/twilio/poll")

    assert turn.status_code == 404
    assert poll.status_code == 404


def test_pin_failures_hang_up_after_three_attempts(phone_app) -> None:
    app, _config = phone_app
    with TestClient(app) as client:
        _signed_post(
            client,
            _VOICE_PATH,
            {"CallSid": "CA-PIN-FAIL", "Direction": "inbound", "From": "+15551234567"},
        )
        first = _signed_post(client, "/api/webhooks/twilio/pin", {"CallSid": "CA-PIN-FAIL", "Digits": "１２３４"})
        second = _signed_post(client, "/api/webhooks/twilio/pin", {"CallSid": "CA-PIN-FAIL", "Digits": "9999"})
        third = _signed_post(client, "/api/webhooks/twilio/pin", {"CallSid": "CA-PIN-FAIL", "Digits": "1111"})

    assert first.status_code == second.status_code == third.status_code == 200
    assert "<Gather" in first.text and "<Gather" in second.text
    assert "<Hangup/>" in third.text
    assert "1234" not in first.text + second.text + third.text


def test_voice_accepts_twilio_signature_without_the_public_port(phone_app) -> None:
    app, config = phone_app
    config.phone.public_base_url = "https://phone.example.test:8443"
    form = {"CallSid": "CA-PORT", "Direction": "inbound", "From": "+15551234567"}
    without_port = _twilio_signature(f"{_PUBLIC_BASE_URL}{_VOICE_PATH}", form)
    wrong_port = _twilio_signature(f"{_PUBLIC_BASE_URL}:9000{_VOICE_PATH}", form)

    with TestClient(app) as client:
        accepted = _signed_post(client, _VOICE_PATH, form, signature=without_port)
        rejected = _signed_post(client, _VOICE_PATH, form, signature=wrong_port)

    assert accepted.status_code == 200
    assert rejected.status_code == 403


def test_alert_call_plays_audio_and_digit_one_acknowledges(phone_app, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _config = phone_app
    token = phone_audio_store.put(b"RIFF-alert-wav")
    alert = SimpleNamespace(alert_id="alert-1", anima="aoi", audio_token=token, spoken_text="")
    acknowledged: list[str] = []
    monkeypatch.setattr(phone_routes, "get_alert", lambda _alert_id: alert)
    monkeypatch.setattr(phone_routes, "acknowledge_alert", acknowledged.append)

    with TestClient(app) as client:
        started = _signed_post(
            client,
            _VOICE_PATH,
            {"CallSid": "CA-ALERT-1", "Direction": "outbound-api"},
            query="alert=alert-1",
        )
        choice = _signed_post(
            client,
            _VOICE_PATH,
            {"CallSid": "CA-ALERT-1", "Digits": "1"},
            query="alert=alert-1&phase=choice",
        )

    assert started.status_code == 200
    gather = ElementTree.fromstring(started.text).find("Gather")
    assert gather is not None
    assert gather.get("input") == "dtmf" and "phase=choice" in gather.get("action", "")
    plays = gather.findall("Play")
    assert len(plays) == 2
    assert plays[0].text == f"https://phone.example.test/api/webhooks/twilio/audio/{token}.wav"
    assert choice.status_code == 200 and "<Hangup/>" in choice.text
    assert acknowledged == ["alert-1"]


def test_alert_silence_replays_alert_once_and_hangs_up(phone_app, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _config = phone_app
    token = phone_audio_store.put(b"RIFF-alert-wav")
    alert = SimpleNamespace(alert_id="alert-2", anima="aoi", audio_token=token, spoken_text="")
    acknowledged: list[str] = []
    monkeypatch.setattr(phone_routes, "get_alert", lambda _alert_id: alert)
    monkeypatch.setattr(phone_routes, "acknowledge_alert", acknowledged.append)

    with TestClient(app) as client:
        _signed_post(
            client,
            _VOICE_PATH,
            {"CallSid": "CA-ALERT-2", "Direction": "outbound-api"},
            query="alert=alert-2",
        )
        no_input = _signed_post(
            client,
            _VOICE_PATH,
            {"CallSid": "CA-ALERT-2", "Digits": ""},
            query="alert=alert-2&phase=choice",
        )
        still_no_input = _signed_post(
            client,
            _VOICE_PATH,
            {"CallSid": "CA-ALERT-2", "Digits": ""},
            query="alert=alert-2&phase=choice",
        )

    assert no_input.status_code == 200
    replay = ElementTree.fromstring(no_input.text).find("Gather")
    assert replay is not None and len(replay.findall("Play")) == 1
    assert still_no_input.status_code == 200
    assert "<Hangup/>" in still_no_input.text and "<Gather" not in still_no_input.text
    assert acknowledged == []
    assert phone_sessions.get("CA-ALERT-2").acknowledged is False


def test_alert_key_during_replay_acknowledges(phone_app, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _config = phone_app
    token = phone_audio_store.put(b"RIFF-alert-wav")
    alert = SimpleNamespace(alert_id="alert-3", anima="aoi", audio_token=token, spoken_text="")
    acknowledged: list[str] = []
    monkeypatch.setattr(phone_routes, "get_alert", lambda _alert_id: alert)
    monkeypatch.setattr(phone_routes, "acknowledge_alert", acknowledged.append)

    with TestClient(app) as client:
        for digits in ("", "1"):
            _signed_post(
                client,
                _VOICE_PATH,
                {"CallSid": "CA-ALERT-3", "Digits": digits},
                query="alert=alert-3&phase=choice",
            )

    assert acknowledged == ["alert-3"]


def test_alert_without_audio_is_read_by_twilio(phone_app, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _config = phone_app
    alert = SimpleNamespace(alert_id="alert-4", anima="aoi", audio_token="", spoken_text="ディスク障害 <確認>")
    monkeypatch.setattr(phone_routes, "get_alert", lambda _alert_id: alert)

    with TestClient(app) as client:
        started = _signed_post(
            client,
            _VOICE_PATH,
            {"CallSid": "CA-ALERT-4", "Direction": "outbound-api"},
            query="alert=alert-4",
        )
        replay = _signed_post(
            client,
            _VOICE_PATH,
            {"CallSid": "CA-ALERT-4", "Digits": ""},
            query="alert=alert-4&phase=choice",
        )

    gather = ElementTree.fromstring(started.text).find("Gather")
    assert gather is not None
    say = gather.find("Say")
    assert say is not None and say.text == "ディスク障害 <確認>" and say.get("language") == "ja-JP"
    assert ElementTree.fromstring(replay.text).find("Gather/Say") is not None


def test_alert_without_any_speech_hangs_up(phone_app, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _config = phone_app
    alert = SimpleNamespace(alert_id="alert-5", anima="aoi", audio_token="", spoken_text="")
    monkeypatch.setattr(phone_routes, "get_alert", lambda _alert_id: alert)

    with TestClient(app) as client:
        started = _signed_post(
            client,
            _VOICE_PATH,
            {"CallSid": "CA-ALERT-5", "Direction": "outbound-api"},
            query="alert=alert-5",
        )

    assert "<Hangup/>" in started.text and "<Gather" not in started.text


def test_prompts_are_read_by_twilio_when_synthesis_fails_or_hangs(phone_app, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _config = phone_app
    token = phone_audio_store.put(b"RIFF-alert-wav")
    alert = SimpleNamespace(alert_id="alert-6", anima="aoi", audio_token=token, spoken_text="")
    monkeypatch.setattr(phone_routes, "get_alert", lambda _alert_id: alert)
    monkeypatch.setattr(phone_routes, "_PROMPT_TTS_TIMEOUT_SEC", 0.01)

    async def broken_speech(_anima: str, _text: str, **_kwargs) -> bytes:
        raise RuntimeError("TTS is down")

    async def hanging_speech(_anima: str, _text: str, **_kwargs) -> bytes:
        await asyncio.Event().wait()
        return b""

    for number, speech in enumerate((broken_speech, hanging_speech)):
        monkeypatch.setattr(phone_routes, "synthesize_speech", speech)
        with TestClient(app) as client:
            started = _signed_post(
                client,
                _VOICE_PATH,
                {"CallSid": f"CA-ALERT-6-{number}", "Direction": "outbound-api"},
                query="alert=alert-6",
            )

        assert started.status_code == 200
        gather = ElementTree.fromstring(started.text).find("Gather")
        assert gather is not None
        assert gather.find("Play") is not None and gather.find("Say") is not None


def test_status_callback_discards_session_and_revokes_stream_token(phone_app) -> None:
    app, _config = phone_app
    phone_sessions.create("CA-DONE", "aoi")
    token = phone_stream_tokens.issue("CA-DONE")

    with TestClient(app) as client:
        response = _signed_post(
            client,
            "/api/webhooks/twilio/status",
            {"CallSid": "CA-DONE", "CallStatus": "completed"},
        )

    assert response.status_code == 204
    assert phone_sessions.get("CA-DONE") is None
    assert phone_stream_tokens.consume(token, "CA-DONE") is False
