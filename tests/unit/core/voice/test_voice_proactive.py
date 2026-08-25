# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for proactive (silence-triggered) self-speech.

Covers:
  - enabled by default (opt-out); disabled → the idle watcher task is never started
  - silence threshold + all guards clear → one self-turn runs at a fixed delay
  - each fire-guard blocks the self-turn
  - no cap on consecutive self-turns
  - user turn resets the monologue counter
  - proactive turns are not recorded in conversation.json
  - close() cancels the idle watcher task
"""

from __future__ import annotations

import asyncio
import contextlib
import struct
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.schemas import VoiceConfig
from core.voice.front import READ_MEMORY_TOOL
from core.voice.session import VoiceSession, build_proactive_prompt
from core.voice.tts_base import TTSConfig

INITIAL_DELAY = VoiceConfig().proactive_initial_delay_sec
PROACTIVE_PROMPT = build_proactive_prompt(0)


def _audio_frames() -> bytes:
    """PCM16 buffer with enough energy to pass the voice gate (>350ms)."""
    return struct.pack(f"<{8000}h", *([5000] * 8000))


def _make_session(*, proactive: bool = True, ticks: float = 0.0) -> VoiceSession:
    stt = AsyncMock()
    stt.transcribe_buffer_async = AsyncMock(return_value={"raw_text": "テスト", "language": "ja"})
    tts = AsyncMock()
    tts.health_check = AsyncMock(return_value=True)

    async def _synthesize(text: str, config: object) -> AsyncMock:
        yield b""

    tts.synthesize = _synthesize
    ws = AsyncMock()
    voice_config = VoiceConfig(
        stt_refine_enabled=False,
        proactive_enabled=proactive,
    )
    sess = VoiceSession(
        "test",
        ws,
        stt,
        tts,
        TTSConfig(provider="voicevox"),
        MagicMock(),
        voice_config,
        front_model="openai/qwen3.6-35b-a3b" if proactive else None,
        front_api_base="http://x:8000/v1",
    )
    sess._idle_tick_sec = ticks
    return sess


def _lane_stub(
    *,
    fired: asyncio.Event | None = None,
    log: list | None = None,
    calls: list[dict] | None = None,
) -> AsyncMock:
    """A fake front lane whose stream yields one short line."""
    lane = AsyncMock()
    lane.check_health = AsyncMock(return_value=True)
    lane.reset_turn = MagicMock()

    async def _stream(user_text: str, **kwargs):  # type: ignore[no-untyped-def]
        if log is not None:
            log.append(user_text)
        if calls is not None:
            calls.append(kwargs)
        if fired is not None:
            fired.set()
        yield "こんにちは"

    lane.stream = _stream
    return lane


def _ready_to_fire(sess: VoiceSession) -> None:
    """Force the session into a state where a proactive turn is due."""
    sess._last_activity = time.monotonic() - 1000
    sess._proactive_delay = 1.0
    sess._proactive_count = 0
    sess._processing = False
    sess._tts_playing = False
    sess._audio_buffer.clear()
    sess._delegation_jobs = {}


async def _run_ticks(sess: VoiceSession, n: int = 1) -> None:
    """Run the idle watcher loop for ``n`` ticks then clean up."""
    task = asyncio.create_task(sess._idle_watcher_loop(), name="idle-probe")
    await asyncio.sleep(0.02 * n)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ── Loop enabling / disabling ──────────────────────────────────


class TestProactiveEnablement:
    def test_enabled_by_default(self) -> None:
        # Proactive speech is opt-out: the config default must stay True.
        assert VoiceConfig().proactive_enabled is True
        assert INITIAL_DELAY == 10.0

    def test_disabled_does_not_start_loop(self) -> None:
        sess = _make_session(proactive=False)
        sess._ensure_idle_watcher()
        assert sess._idle_watcher is None

    def test_disabled_but_front_model_set_still_no_loop(self) -> None:
        # proactive_enabled is the master switch; front_model alone is not enough.
        sess = _make_session(proactive=False)
        sess._ensure_idle_watcher()
        assert sess._idle_watcher is None

    @pytest.mark.asyncio
    async def test_enabled_starts_loop(self) -> None:
        sess = _make_session(proactive=True)
        sess._ensure_idle_watcher()
        assert sess._idle_watcher is not None
        # clean up the running task
        task = sess._idle_watcher
        sess._idle_watcher = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_enabled_but_no_front_model_does_not_start_loop(self) -> None:
        sess = _make_session(proactive=True)
        sess._front_model = None
        sess._ensure_idle_watcher()
        assert sess._idle_watcher is None


# ── Proactive turn execution ───────────────────────────────────


class TestProactiveTurn:
    @pytest.mark.asyncio
    async def test_turn_runs_once_at_constant_delay(self) -> None:
        sess = _make_session(proactive=True)
        fired = asyncio.Event()
        sess._front_lane = _lane_stub(fired=fired)
        _ready_to_fire(sess)
        sess._ensure_idle_watcher()
        assert sess._idle_watcher is not None
        await asyncio.wait_for(fired.wait(), timeout=2.0)
        # Give the loop a beat to finalize the turn bookkeeping.
        await asyncio.sleep(0.02)
        assert sess._proactive_count == 1
        assert sess._proactive_delay == pytest.approx(1.0)
        # last activity was refreshed so no immediate second fire
        assert time.monotonic() - sess._last_activity < 1.0
        task = sess._idle_watcher
        sess._idle_watcher = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_turn_uses_synthetic_instruction(self) -> None:
        sess = _make_session(proactive=True)
        log: list[str] = []
        calls: list[dict] = []
        fired = asyncio.Event()
        sess._front_lane = _lane_stub(fired=fired, log=log, calls=calls)
        _ready_to_fire(sess)
        sess._ensure_idle_watcher()
        await asyncio.wait_for(fired.wait(), timeout=2.0)
        assert log and PROACTIVE_PROMPT in log
        assert calls[0]["tools"] == [READ_MEMORY_TOOL]
        task = sess._idle_watcher
        sess._idle_watcher = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class TestProactivePrompt:
    def test_prompt_changes_for_first_and_later_turns(self) -> None:
        first = build_proactive_prompt(0)
        later = build_proactive_prompt(1)

        assert "軽い一言" in first
        assert "read_memory" not in first
        assert "read_memory" in later
        assert "独り言モード" in later

    def test_fourth_later_prompt_requests_older_memory(self) -> None:
        fourth = build_proactive_prompt(4)

        assert "read_memory" in fourth
        assert "少し前の記憶" in fourth


# ── Fire-guards ───────────────────────────────────────────────


class TestProactiveGuards:
    @pytest.mark.asyncio
    async def test_processing_blocks(self) -> None:
        sess = _make_session(proactive=True)
        log: list = []
        sess._front_lane = _lane_stub(log=log)
        _ready_to_fire(sess)
        sess._processing = True
        assert sess._should_proactive() is False
        await _run_ticks(sess)
        assert log == []

    @pytest.mark.asyncio
    async def test_tts_playing_blocks(self) -> None:
        sess = _make_session(proactive=True)
        _ready_to_fire(sess)
        sess._tts_playing = True
        assert sess._should_proactive() is False

    @pytest.mark.asyncio
    async def test_pending_client_playback_blocks(self) -> None:
        sess = _make_session()
        _ready_to_fire(sess)
        sess._proactive_lead_sec = 5.0
        sess._playback_end_at = time.monotonic() + 30
        assert sess._should_proactive() is False

    async def test_playback_within_lead_allows(self) -> None:
        sess = _make_session()
        _ready_to_fire(sess)
        sess._proactive_lead_sec = 5.0
        sess._playback_end_at = time.monotonic() + 4
        assert sess._should_proactive() is True

    async def test_note_playback_accumulates_from_now(self) -> None:
        sess = _make_session()
        sess._playback_end_at = 0.0  # long past
        sess._note_playback(3.0)
        sess._note_playback(2.0)
        assert 4.9 < sess._playback_end_at - time.monotonic() <= 5.0

    async def test_interrupt_resets_playback_estimate(self) -> None:
        sess = _make_session()
        sess._playback_end_at = time.monotonic() + 30
        await sess.handle_interrupt()
        assert sess._playback_end_at == 0.0

    async def test_recognized_speech_blocks(self) -> None:
        sess = _make_session(proactive=True)
        _ready_to_fire(sess)
        sess._streamer._committed = "こんにちは"
        assert sess._should_proactive() is False

    @pytest.mark.asyncio
    async def test_buffered_silence_does_not_block(self) -> None:
        """An open hands-free mic always has buffered audio — that alone must
        not veto a self-turn (otherwise proactive speech never fires)."""
        sess = _make_session(proactive=True)
        _ready_to_fire(sess)
        sess._audio_buffer.extend(b"\x00\x01")
        assert sess._should_proactive() is True

    @pytest.mark.asyncio
    async def test_delegation_jobs_block(self) -> None:
        sess = _make_session(proactive=True)
        _ready_to_fire(sess)
        sess._delegation_jobs = {1: asyncio.create_task(asyncio.sleep(30))}
        try:
            assert sess._should_proactive() is False
        finally:
            for t in sess._delegation_jobs.values():
                t.cancel()
            await asyncio.gather(*sess._delegation_jobs.values(), return_exceptions=True)

    @pytest.mark.asyncio
    async def test_silence_not_long_enough_blocks(self) -> None:
        sess = _make_session(proactive=True)
        _ready_to_fire(sess)
        sess._last_activity = time.monotonic()  # just spoke -> not idle
        assert sess._should_proactive() is False


# ── Continuous monologue + reset ─────────────────────────────


class TestProactiveContinuation:
    @pytest.mark.asyncio
    async def test_count_five_still_allows_self_turn(self) -> None:
        sess = _make_session(proactive=True)
        log: list = []
        sess._front_lane = _lane_stub(log=log)
        _ready_to_fire(sess)
        sess._proactive_count = 5
        assert sess._should_proactive() is True

    @pytest.mark.asyncio
    async def test_user_turn_resets_count_and_delay(self) -> None:
        sess = _make_session(proactive=True)
        sess._front_lane = _lane_stub()
        # simulate prior monologues
        sess._proactive_count = 3
        sess._proactive_delay = 400.0
        sess._audio_buffer.extend(_audio_frames())
        await sess.handle_speech_end()
        assert sess._proactive_count == 0
        assert sess._proactive_delay == pytest.approx(INITIAL_DELAY)
        # a fresh, ready session can fire again
        _ready_to_fire(sess)
        assert sess._should_proactive() is True

    @pytest.mark.asyncio
    async def test_audio_chunk_refreshes_activity_only_when_speech(self) -> None:
        sess = _make_session(proactive=True)
        # silence-only chunks (idle hands-free mic) must NOT rewind the timer
        sess._last_activity = time.monotonic() - 1000
        await sess.handle_audio_chunk(b"\x00\x01")
        assert time.monotonic() - sess._last_activity > 900
        # once the streamer has committed recognized speech, they do
        sess._streamer._committed = "こんにちは"
        await sess.handle_audio_chunk(b"\x00\x01")
        assert time.monotonic() - sess._last_activity < 1.0


# ── Conversation recording ───────────────────────────────────


class TestProactiveRecording:
    @pytest.mark.asyncio
    async def test_proactive_does_not_record_conversation(self) -> None:
        sess = _make_session(proactive=True)
        sess._front_lane = _lane_stub()
        mock_conv = MagicMock()
        with patch("core.memory.conversation.ConversationMemory", return_value=mock_conv):
            await sess._run_front_turn(
                sess._front_lane,
                PROACTIVE_PROMPT,
                "human",
                True,
                record_user=False,
                record=False,
            )
        mock_conv.append_turn.assert_not_called()
        mock_conv.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_self_turn_owns_tts_worker(self) -> None:
        # A self-turn (no live TTS worker) must start one so sentences reach
        # TTS instead of being dropped by _enqueue_tts(queue=None), and stop
        # it afterwards.
        sess = _make_session(proactive=True)
        sess._front_lane = _lane_stub()
        enqueued: list[str] = []

        async def _spy_enqueue(sentence: str) -> None:
            enqueued.append(sentence)

        assert sess._tts_queue is None
        with (
            patch("core.memory.conversation.ConversationMemory", return_value=MagicMock()),
            patch.object(sess, "_enqueue_tts", side_effect=_spy_enqueue),
        ):
            ok = await sess._run_front_turn(
                sess._front_lane,
                PROACTIVE_PROMPT,
                "human",
                True,
                record_user=False,
            )
        assert ok is True
        assert enqueued  # sentences actually flowed toward TTS
        assert sess._tts_queue is None  # worker stopped after the turn
        assert sess._tts_playing is False

    @pytest.mark.asyncio
    async def test_proactive_failure_backs_off(self) -> None:
        # A failing self-turn must rewind the idle timer so the loop does not
        # retry the LLM every tick.
        sess = _make_session(proactive=True)
        lane = _lane_stub()

        async def _boom(user_text: str, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("front down")
            yield  # pragma: no cover

        lane.stream = _boom
        sess._front_lane = lane
        _ready_to_fire(sess)
        await _run_ticks(sess, n=2)
        assert sess._proactive_count == 0  # failure does not advance the prompt
        assert time.monotonic() - sess._last_activity < 1.0  # timer rewound

    @pytest.mark.asyncio
    async def test_default_still_records_user_then_assistant(self) -> None:
        # Regression: the normal front-turn path must keep writing both turns.
        sess = _make_session(proactive=True)
        sess._front_lane = _lane_stub()
        mock_conv = MagicMock()
        with patch("core.memory.conversation.ConversationMemory", return_value=mock_conv):
            await sess._run_front_turn(sess._front_lane, "こんにちは", "human", True, record_user=True)
        roles = [c.args[0] for c in mock_conv.append_turn.call_args_list]
        assert roles == ["human", "assistant"]


# ── Concurrency & lifecycle (review M4) ───────────────────


class TestProactiveConcurrency:
    @pytest.mark.asyncio
    async def test_no_double_stream_with_concurrent_user_turn(self) -> None:
        """A user turn arriving mid-proactive must be blocked by the processing
        lock so the shared front lane is never streamed twice at once."""
        sess = _make_session(proactive=True, ticks=0.0)
        started = asyncio.Event()
        release = asyncio.Event()
        stream_calls = 0

        async def _slow(user_text: str, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal stream_calls
            stream_calls += 1
            started.set()
            await release.wait()
            yield "こんにちは"

        lane = AsyncMock()
        lane.check_health = AsyncMock(return_value=True)
        lane.reset_turn = MagicMock()
        lane.stream = _slow
        sess._front_lane = lane
        _ready_to_fire(sess)
        with patch("core.memory.conversation.ConversationMemory", return_value=MagicMock()):
            loop_task = asyncio.create_task(sess._idle_watcher_loop(), name="idle-probe")
            await asyncio.wait_for(started.wait(), timeout=2.0)
            assert sess._processing is True
            # a user turn while the proactive turn is streaming must be ignored
            sess._audio_buffer.extend(_audio_frames())
            await sess.handle_speech_end()
            release.set()
            await asyncio.sleep(0.05)
            loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop_task
        assert stream_calls == 1

    @pytest.mark.asyncio
    async def test_proactive_not_started_until_greet_complete(self) -> None:
        """The idle watcher must not run while a (possibly 90s) greet is in
        flight — it is started only after the greet finishes (C2)."""
        sess = _make_session(proactive=True)
        started = asyncio.Event()
        release = asyncio.Event()

        async def _greet(**kwargs):  # type: ignore[no-untyped-def]
            started.set()
            await release.wait()
            return {"response": "はい", "emotion": "neutral"}

        sess._supervisor.send_request = AsyncMock(side_effect=_greet)
        greet_task = asyncio.create_task(sess.greet_and_speak())
        await started.wait()
        assert sess._idle_watcher is None
        release.set()
        await greet_task
        assert sess._idle_watcher is not None
        task = sess._idle_watcher
        sess._idle_watcher = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_close_prevents_further_firing(self) -> None:
        """After close() the loop must not fire any further proactive turn."""
        sess = _make_session(proactive=True, ticks=0.0)
        log: list = []
        sess._front_lane = _lane_stub(log=log)
        _ready_to_fire(sess)
        await sess.close()
        await _run_ticks(sess, n=2)
        assert log == []

    @pytest.mark.asyncio
    async def test_barge_in_then_next_self_turn_succeeds(self) -> None:
        """An interrupt during a previous turn must not kill every later
        proactive turn — self-turn entry clears _interrupted (H2)."""
        sess = _make_session(proactive=True)
        fired = asyncio.Event()
        sess._front_lane = _lane_stub(fired=fired)
        await sess.handle_interrupt()
        assert sess._interrupted is True
        _ready_to_fire(sess)
        sess._ensure_idle_watcher()
        await asyncio.wait_for(fired.wait(), timeout=2.0)
        await asyncio.sleep(0.02)
        assert sess._proactive_count == 1
        task = sess._idle_watcher
        sess._idle_watcher = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_self_turn_sends_start_and_done_frames(self) -> None:
        """A self-turn must emit leading response_start and a terminal
        response_done so the client subtitle starts a fresh bubble (H1)."""
        sess = _make_session(proactive=True)
        sess._front_lane = _lane_stub()
        with patch("core.memory.conversation.ConversationMemory", return_value=MagicMock()):
            ok = await sess._run_front_turn(
                sess._front_lane,
                PROACTIVE_PROMPT,
                "human",
                True,
                record_user=False,
                tools=[],
                drain_results=False,
            )
        assert ok is True
        types = [c.args[0]["type"] for c in sess._ws.send_json.call_args_list]
        assert types.count("response_start") == 1
        assert "response_done" in types

    @pytest.mark.asyncio
    async def test_barge_in_during_self_turn_still_terminates(self) -> None:
        """A barge-in mid-self-turn must still deliver a terminal
        response_done (H2) so the client never hangs."""
        sess = _make_session(proactive=True)
        started = asyncio.Event()
        entered = asyncio.Event()

        async def _slow(user_text: str, **kwargs):  # type: ignore[no-untyped-def]
            started.set()
            await entered.wait()
            yield "こんにちは"

        lane = AsyncMock()
        lane.check_health = AsyncMock(return_value=True)
        lane.reset_turn = MagicMock()
        lane.stream = _slow
        sess._front_lane = lane

        async def _run() -> bool:
            return await sess._run_front_turn(
                lane,
                PROACTIVE_PROMPT,
                "human",
                True,
                record_user=False,
                tools=[],
                drain_results=False,
            )

        with patch("core.memory.conversation.ConversationMemory", return_value=MagicMock()):
            task = asyncio.create_task(_run())
            await started.wait()
            await sess.handle_interrupt()
            entered.set()
            ok = await task
        assert ok is False
        types = [c.args[0]["type"] for c in sess._ws.send_json.call_args_list]
        assert types.count("response_start") == 1
        assert "response_done" in types


# ── Empty replies / discarded noise ──────────────────────────


class TestEmptyTurnAndDiscard:
    @pytest.mark.asyncio
    async def test_empty_reply_is_not_recorded(self) -> None:
        """A thinking model can burn its budget and stream no text — that must
        not persist an empty turn nor count as a spoken self-turn."""
        sess = _make_session(proactive=True)
        lane = AsyncMock()
        lane.check_health = AsyncMock(return_value=True)
        lane.reset_turn = MagicMock()

        async def _empty(user_text: str, **kwargs):  # type: ignore[no-untyped-def]
            if False:
                yield ""

        lane.stream = _empty
        sess._front_lane = lane
        mock_conv = MagicMock()
        with patch("core.memory.conversation.ConversationMemory", return_value=mock_conv):
            ok = await sess._run_front_turn(
                lane,
                PROACTIVE_PROMPT,
                "human",
                True,
                record_user=False,
            )
        assert ok is False
        assert mock_conv.append_turn.call_args_list == []

    @pytest.mark.asyncio
    async def test_audio_dropped_while_speaking(self) -> None:
        """Mic input arriving while we talk is our own TTS echo — drop it so it
        can never be transcribed back as a user turn."""
        sess = _make_session(proactive=True)
        sess._tts_playing = True
        await sess.handle_audio_chunk(_audio_frames())
        assert not sess._audio_buffer
        # barge-in reopens the mic immediately
        await sess.handle_interrupt()
        assert sess._tts_playing is False
        await sess.handle_audio_chunk(_audio_frames())
        assert sess._audio_buffer

    @pytest.mark.asyncio
    async def test_discard_audio_keeps_turn_alive(self) -> None:
        """A VAD misfire drops buffered noise but must never abort the reply
        the anima is streaming (that is barge-in / handle_interrupt)."""
        sess = _make_session(proactive=True)
        sess._audio_buffer.extend(_audio_frames())
        sess._interrupted = False
        await sess.handle_discard_audio()
        assert not sess._audio_buffer
        assert sess._interrupted is False


# ── Teardown ─────────────────────────────────────────────────


class TestProactiveTeardown:
    @pytest.mark.asyncio
    async def test_close_cancels_idle_watcher(self) -> None:
        sess = _make_session(proactive=True, ticks=0.0)
        sess._ensure_idle_watcher()
        task = sess._idle_watcher
        assert task is not None and not task.done()
        await sess.close()
        assert sess._idle_watcher is None
        assert task.cancelled()


class TestMonologueBlockList:
    def test_block_list_keeps_content_words_only(self) -> None:
        recent = ["🤭💦あっ、また黙っちゃった… LGTのDynamicって用語、気になる？", ""]
        prompt = build_proactive_prompt(2, recent)
        assert "すでに話した話題" in prompt
        assert "LGT" in prompt and "Dynamic" in prompt and "用語" in prompt
        # phrasing (hiragana) and emoji must not be quoted back to the model
        assert "黙っちゃった" not in prompt and "🤭" not in prompt

    def test_no_block_list_when_nothing_said(self) -> None:
        assert "すでに話した話題" not in build_proactive_prompt(1, [])


def test_wav_seconds_parses_duration() -> None:
    import io
    import wave

    from core.voice.session import _wav_seconds

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 12000)  # 0.5s
    assert _wav_seconds(buf.getvalue()) == pytest.approx(0.5)
    assert _wav_seconds(b"not wav") is None
    assert _wav_seconds(b"") is None
