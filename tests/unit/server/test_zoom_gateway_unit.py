"""Unit tests for server/zoom_gateway.py — ZoomRTMSManager buffering & routing.

These tests exercise the transcript buffering, chunk flushing, routing and
end-of-meeting logic of :class:`ZoomRTMSManager` in isolation.  The Zoom
WebSocket transport is never opened; ``Messenger.receive_external`` is
replaced by a capturing fake so injected chunks can be asserted directly.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.config.schemas import ZoomRTMSConfig
from server import zoom_gateway
from server.zoom_gateway import (
    ZoomRTMSManager,
    _extract_media_url,
    _extract_transcript,
    _first_url,
    _is_allowed_ws_url,
    _MeetingSession,
)

# ── Fixtures & helpers ────────────────────────────────────


@pytest.fixture
def injections(monkeypatch, tmp_path):
    """Capture every ``Messenger.receive_external`` call as a dict."""
    calls: list[dict[str, Any]] = []

    class _FakeMessenger:
        def __init__(self, shared_dir: Any, anima: str) -> None:
            self._anima = anima

        def receive_external(self, **kwargs: Any) -> None:
            calls.append({"anima": self._anima, **kwargs})
            return None

    monkeypatch.setattr("server.zoom_gateway.Messenger", _FakeMessenger)
    monkeypatch.setattr("server.zoom_gateway.get_shared_dir", lambda: tmp_path)
    return calls


def _make_session(
    *,
    target: str = "kotoha",
    uuid: str = "mtg-uuid-1",
    meeting_id: str = "123",
    topic: str = "Weekly Sync",
) -> _MeetingSession:
    return _MeetingSession(
        meeting_uuid=uuid,
        meeting_id=meeting_id,
        topic=topic,
        rtms_stream_id="stream-1",
        signaling_url="wss://sig.example",
        target_anima=target,
    )


def _manager(**config_kwargs: Any) -> ZoomRTMSManager:
    mgr = ZoomRTMSManager()
    mgr._config = ZoomRTMSConfig(**config_kwargs)
    return mgr


# ── Chunk flushing: character threshold ───────────────────


class TestCharThresholdFlush:
    async def test_flush_when_char_threshold_reached(self, injections):
        mgr = _manager(chunk_max_chars=20, chunk_interval_seconds=100_000)
        session = _make_session()

        await mgr._append(session, "田中", "短い")  # under threshold, no flush
        assert injections == []

        await mgr._append(session, "佐藤", "こちらはかなり長い発話でしきい値を超えます")
        assert len(injections) == 1

        chunk = injections[0]
        assert "田中: 短い" in chunk["content"]
        assert "佐藤: こちらはかなり長い発話でしきい値を超えます" in chunk["content"]

    async def test_no_flush_below_threshold(self, injections):
        mgr = _manager(chunk_max_chars=100_000, chunk_interval_seconds=100_000)
        session = _make_session()
        await mgr._append(session, "田中", "予算の件です")
        await mgr._append(session, "佐藤", "来週にしましょう")
        assert injections == []


# ── Chunk flushing: interval timer ────────────────────────


class TestIntervalFlush:
    async def test_flush_timer_fires_after_interval(self, injections, monkeypatch):
        """The periodic flush timer emits a chunk once the interval elapses."""
        mgr = _manager(chunk_interval_seconds=300, chunk_max_chars=100_000)
        session = _make_session()
        await mgr._append(session, "田中", "実況テスト")  # buffered, below char threshold
        assert injections == []

        real_sleep = asyncio.sleep
        interval = max(1, mgr._config.chunk_interval_seconds)
        ticks = {"n": 0}

        async def fake_sleep(delay: float) -> None:
            # Intercept only the flush-timer's interval sleeps; let the first
            # tick through (triggering one flush) and stop on the second.
            if delay == interval:
                ticks["n"] += 1
                if ticks["n"] >= 2:
                    session.stopping = True
                await real_sleep(0)
                return
            await real_sleep(delay)

        monkeypatch.setattr(zoom_gateway.asyncio, "sleep", fake_sleep)
        await mgr._flush_timer(session)

        assert len(injections) == 1
        assert "実況テスト" in injections[0]["content"]


# ── Chunk header & injection metadata ─────────────────────


class TestChunkFormatAndMetadata:
    async def test_chunk_header_and_receive_external_kwargs(self, injections):
        mgr = _manager(chunk_max_chars=100_000, chunk_interval_seconds=100_000)
        session = _make_session(target="kotoha", uuid="mtg-uuid-1", meeting_id="999", topic="定例")
        await mgr._append(session, "田中", "予算の件ですが")
        await mgr._flush(session)

        assert len(injections) == 1
        chunk = injections[0]
        assert chunk["anima"] == "kotoha"
        assert chunk["content"].startswith("[Zoom会議実況 チャンク#1 | 会議: 定例 (999) |")
        assert "田中: 予算の件ですが" in chunk["content"]
        assert chunk["source"] == "zoom"
        assert chunk["intent"] == "meeting_transcript"
        assert chunk["source_message_id"] == "mtg-uuid-1:1"
        assert chunk["external_channel_id"] == "999"
        assert chunk["external_thread_ts"] == "zoom-mtg-uuid-1"

    async def test_chunk_seq_increments_across_flushes(self, injections):
        mgr = _manager(chunk_max_chars=1, chunk_interval_seconds=100_000)  # flush every utterance
        session = _make_session(uuid="mtg-uuid-1")
        await mgr._append(session, "田中", "one")
        await mgr._append(session, "佐藤", "two")

        assert len(injections) == 2
        assert injections[0]["source_message_id"] == "mtg-uuid-1:1"
        assert injections[1]["source_message_id"] == "mtg-uuid-1:2"
        assert "チャンク#1" in injections[0]["content"]
        assert "チャンク#2" in injections[1]["content"]

    async def test_gap_notice_prepended_after_reconnect(self, injections):
        mgr = _manager(chunk_max_chars=100_000, chunk_interval_seconds=100_000)
        session = _make_session()
        session.gap_pending = True
        await mgr._append(session, "田中", "再開後の発話")
        await mgr._flush(session)

        content = injections[0]["content"]
        assert "[接続断により一部欠落]" in content
        # Gap notice sits above the utterance, below the header.
        assert content.index("[接続断により一部欠落]") < content.index("田中: 再開後の発話")


# ── Injection failure recovery ────────────────────────────


class TestInjectFailureRecovery:
    async def test_failed_inject_preserves_lines_and_count(self, monkeypatch, tmp_path):
        """If receive_external raises, utterances stay buffered and no chunk is counted."""
        raising = MagicMock(side_effect=RuntimeError("inbox write failed"))

        class _RaisingMessenger:
            def __init__(self, shared_dir: Any, anima: str) -> None:
                pass

            def receive_external(self, **kwargs: Any) -> None:
                raising(**kwargs)

        monkeypatch.setattr("server.zoom_gateway.Messenger", _RaisingMessenger)
        monkeypatch.setattr("server.zoom_gateway.get_shared_dir", lambda: tmp_path)

        mgr = _manager(chunk_max_chars=100_000, chunk_interval_seconds=100_000)
        session = _make_session()
        await mgr._append(session, "田中", "予算の件です")
        await mgr._append(session, "佐藤", "来週にしましょう")

        await mgr._flush(session)

        # Delivery failed: nothing counted, and the utterances were restored.
        assert raising.call_count == 1
        assert session.delivered_chunks == 0
        assert session.buffer == ["田中: 予算の件です", "佐藤: 来週にしましょう"]
        assert session.buffer_chars > 0
        # seq was consumed (gaps are acceptable) — next chunk uses #2.
        assert session.chunk_seq == 2

    async def test_recovered_inject_resends_preserved_lines(self, monkeypatch, tmp_path):
        """After a failed flush, the next successful flush resends the buffered lines."""
        calls: list[dict[str, Any]] = []
        state = {"fail": True}

        class _FlakyMessenger:
            def __init__(self, shared_dir: Any, anima: str) -> None:
                pass

            def receive_external(self, **kwargs: Any) -> None:
                if state["fail"]:
                    raise RuntimeError("transient failure")
                calls.append(kwargs)

        monkeypatch.setattr("server.zoom_gateway.Messenger", _FlakyMessenger)
        monkeypatch.setattr("server.zoom_gateway.get_shared_dir", lambda: tmp_path)

        mgr = _manager(chunk_max_chars=100_000, chunk_interval_seconds=100_000)
        session = _make_session()
        await mgr._append(session, "田中", "重要な発言")
        await mgr._flush(session)  # fails, lines restored
        assert calls == []
        assert session.delivered_chunks == 0

        state["fail"] = False
        await mgr._flush(session)  # succeeds, resends preserved line with a new seq
        assert len(calls) == 1
        assert "田中: 重要な発言" in calls[0]["content"]
        assert session.delivered_chunks == 1
        assert session.buffer == []


# ── Routing ───────────────────────────────────────────────


class TestRouting:
    def test_mapping_hit(self):
        mgr = _manager(meeting_mapping={"123": "kotoha"}, default_anima="sakura")
        assert mgr._resolve_target_anima("123") == "kotoha"

    def test_falls_back_to_default(self):
        mgr = _manager(meeting_mapping={"123": "kotoha"}, default_anima="sakura")
        assert mgr._resolve_target_anima("999") == "sakura"

    def test_no_mapping_no_default_returns_empty(self):
        mgr = _manager()
        assert mgr._resolve_target_anima("999") == ""

    async def test_started_discards_and_warns_when_no_target(self, injections, caplog):
        mgr = _manager()  # no mapping, no default
        mgr._started = True
        mgr._client_id = "cid"
        mgr._client_secret = "csecret"
        obj = {
            "meeting_uuid": "u1",
            "rtms_stream_id": "s1",
            "meeting_id": "123",
            "server_urls": "wss://sig.example",
            "topic": "T",
        }
        with caplog.at_level(logging.WARNING, logger="animaworks.zoom_gateway"):
            await mgr.handle_rtms_started(obj)

        assert mgr._sessions == {}
        assert "no target anima" in caplog.text.lower()

    async def test_started_creates_session_for_default_anima(self, injections, monkeypatch):
        mgr = _manager(default_anima="sakura")
        mgr._started = True
        mgr._client_id = "cid"
        mgr._client_secret = "csecret"

        async def _noop(session: _MeetingSession) -> None:
            return None

        monkeypatch.setattr(mgr, "_run_session", _noop)
        monkeypatch.setattr(mgr, "_flush_timer", _noop)

        obj = {
            "meeting_uuid": "u1",
            "rtms_stream_id": "s1",
            "meeting_id": "999",
            "server_urls": "wss://sig.example",
            "topic": "T",
        }
        await mgr.handle_rtms_started(obj)
        await asyncio.sleep(0)  # let the no-op tasks run

        assert "u1" in mgr._sessions
        assert mgr._sessions["u1"].target_anima == "sakura"
        await mgr.stop()


# ── End-of-meeting handling ───────────────────────────────


class TestMeetingEnd:
    async def test_stopped_flushes_tail_then_injects_end(self, injections):
        mgr = _manager(chunk_max_chars=100_000, chunk_interval_seconds=100_000)
        session = _make_session(uuid="u1", meeting_id="999", topic="定例")
        mgr._sessions["u1"] = session
        await mgr._append(session, "田中", "最後の発話です")  # buffered, not yet flushed

        await mgr.handle_rtms_stopped({"meeting_uuid": "u1"})

        assert len(injections) == 2
        transcript, ended = injections
        assert transcript["intent"] == "meeting_transcript"
        assert transcript["source_message_id"] == "u1:1"
        assert "最後の発話です" in transcript["content"]

        assert ended["intent"] == "meeting_ended"
        assert ended["source_message_id"] == "u1:end"
        assert ended["content"].startswith("[Zoom会議終了 | 会議: 定例 (999) | 全1チャンク配信済み]")
        assert "u1" not in mgr._sessions

    async def test_stopped_with_no_utterance_skips_end(self, injections):
        mgr = _manager(chunk_max_chars=100_000, chunk_interval_seconds=100_000)
        session = _make_session(uuid="u2")
        mgr._sessions["u2"] = session

        await mgr.handle_rtms_stopped({"meeting_uuid": "u2"})

        assert injections == []
        assert "u2" not in mgr._sessions

    async def test_stopped_unknown_meeting_is_noop(self, injections):
        mgr = _manager()
        await mgr.handle_rtms_stopped({"meeting_uuid": "ghost"})
        assert injections == []


# ── Lifecycle & health ────────────────────────────────────


class TestLifecycle:
    @patch("server.zoom_gateway.get_credential", return_value="")
    @patch("server.zoom_gateway.load_config")
    async def test_start_noop_when_disabled(self, mock_config, mock_cred):
        mock_config.return_value = MagicMock(
            external_messaging=MagicMock(zoom=ZoomRTMSConfig(enabled=False)),
        )
        mgr = ZoomRTMSManager()
        await mgr.start()
        assert mgr._started is False

    async def test_health_check_disabled_by_default(self):
        mgr = ZoomRTMSManager()
        health = await mgr.health_check()
        assert health["status"] == "disabled"
        assert health["active_meetings"] == 0
        assert health["meetings"] == []

    async def test_health_check_reports_active_meeting(self):
        mgr = _manager()
        mgr._started = True
        session = _make_session(uuid="u1", meeting_id="999")
        session.ready = True
        session.delivered_chunks = 3
        mgr._sessions["u1"] = session

        health = await mgr.health_check()
        assert health["status"] == "running"
        assert health["active_meetings"] == 1
        assert health["meetings"][0]["meeting_uuid"] == "u1"
        assert health["meetings"][0]["anima"] == "kotoha"
        assert health["meetings"][0]["delivered_chunks"] == 3

    async def test_health_check_reports_credentials_configured(self):
        mgr = _manager()
        assert (await mgr.health_check())["credentials_configured"] is False
        mgr._client_id = "cid"
        mgr._client_secret = "csecret"
        assert (await mgr.health_check())["credentials_configured"] is True


# ── Pure protocol helpers ─────────────────────────────────


class TestProtocolHelpers:
    def test_first_url_from_string(self):
        assert _first_url("wss://a") == "wss://a"

    def test_first_url_from_list(self):
        assert _first_url(["wss://a", "wss://b"]) == "wss://a"

    def test_first_url_from_empty(self):
        assert _first_url(None) == ""
        assert _first_url([]) == ""

    def test_ws_url_scheme_guard(self):
        # wss:// is always allowed; ws:// only for loopback (local/test).
        assert _is_allowed_ws_url("wss://rtms.zoom.us/stream") is True
        assert _is_allowed_ws_url("ws://127.0.0.1:8765") is True
        assert _is_allowed_ws_url("ws://localhost:8765") is True
        # SSRF vectors: plaintext external and non-ws schemes are refused.
        assert _is_allowed_ws_url("ws://internal.corp/admin") is False
        assert _is_allowed_ws_url("http://169.254.169.254/latest") is False
        assert _is_allowed_ws_url("https://example.com") is False
        assert _is_allowed_ws_url("") is False


class TestSsrfGuard:
    async def test_started_rejects_non_wss_signaling_url(self, injections, monkeypatch):
        """A non-wss server_urls is refused: no session is created."""
        mgr = _manager(default_anima="kotoha")
        mgr._started = True
        mgr._client_id = "cid"
        mgr._client_secret = "csecret"

        async def _noop(session: _MeetingSession) -> None:
            return None

        monkeypatch.setattr(mgr, "_run_session", _noop)
        monkeypatch.setattr(mgr, "_flush_timer", _noop)

        obj = {
            "meeting_uuid": "u1",
            "rtms_stream_id": "s1",
            "meeting_id": "999",
            "server_urls": "ws://internal.corp/admin",  # SSRF attempt
            "topic": "T",
        }
        await mgr.handle_rtms_started(obj)
        assert "u1" not in mgr._sessions

    def test_extract_media_url_dict_all(self):
        msg = {"media_server": {"server_urls": {"all": "wss://media"}}}
        assert _extract_media_url(msg) == "wss://media"

    def test_extract_media_url_dict_transcript(self):
        msg = {"media_server": {"server_urls": {"transcript": "wss://t"}}}
        assert _extract_media_url(msg) == "wss://t"

    def test_extract_media_url_missing(self):
        assert _extract_media_url({}) == ""

    def test_extract_transcript_flat(self):
        speaker, text = _extract_transcript({"user_name": "田中", "content": "こんにちは"})
        assert speaker == "田中"
        assert text == "こんにちは"

    def test_extract_transcript_nested(self):
        msg = {"content": {"user_name": "佐藤", "data": "予算の件"}}
        speaker, text = _extract_transcript(msg)
        assert speaker == "佐藤"
        assert text == "予算の件"

    def test_extract_transcript_unknown_speaker(self):
        speaker, text = _extract_transcript({"content": "発話のみ"})
        assert speaker == "不明"
        assert text == "発話のみ"

    def test_signature_is_deterministic_hmac(self):
        mgr = ZoomRTMSManager()
        mgr._client_id = "client-abc"
        mgr._client_secret = "secret-xyz"
        session = _make_session(uuid="u1")
        import hashlib
        import hmac

        expected = hmac.new(
            b"secret-xyz",
            b"client-abc,u1,stream-1",
            hashlib.sha256,
        ).hexdigest()
        assert mgr._signature(session) == expected
