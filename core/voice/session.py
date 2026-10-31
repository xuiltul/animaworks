# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Voice session — STT -> Chat -> TTS orchestration."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import io
import json
import logging
import random
import re
import time
import unicodedata
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.i18n import t
from core.voice.emotion_style import EmotionStyle, VoiceChannel, emotion_style_for
from core.voice.front_conversation import (
    ASK_ANIMA_DELEGATION_NOTE as ASK_ANIMA_DELEGATION_NOTE,
)
from core.voice.front_conversation import (
    ASK_ANIMA_MAX_RESULT_CHARS as ASK_ANIMA_MAX_RESULT_CHARS,
)
from core.voice.front_conversation import (
    IPC_STREAM_TIMEOUT,
    FrontConversation,
)
from core.voice.front_conversation import (
    MAX_ASK_ANIMA_CONCURRENT as MAX_ASK_ANIMA_CONCURRENT,
)
from core.voice.front_conversation import (
    VOICE_MODE_SUFFIX as VOICE_MODE_SUFFIX,
)
from core.voice.front_conversation import (
    read_memory_snippets as read_memory_snippets,
)
from core.voice.sentence_splitter import StreamingSentenceSplitter
from core.voice.speech_text import (
    IRODORI_STYLE_EMOJI as IRODORI_STYLE_EMOJI,
)
from core.voice.speech_text import (
    SpeechText as SpeechText,
)
from core.voice.speech_text import (
    apply_reading_rules as apply_reading_rules,
)
from core.voice.speech_text import (
    load_yomi as load_yomi,
)
from core.voice.speech_text import (
    prepare_speech,
)
from core.voice.speech_text import (
    read_years as read_years,
)
from core.voice.speech_text import (
    resolve_ruby as resolve_ruby,
)
from core.voice.speech_text import (
    sanitize_for_tts as sanitize_for_tts,
)
from core.voice.speech_text import (
    strip_ruby as strip_ruby,
)
from core.voice.stt import VoiceSTT
from core.voice.stt_stream import StreamingTranscriber
from core.voice.transport import VoiceTransport
from core.voice.tts_base import BaseTTSProvider, TTSConfig, TTSSynthesisError

logger = logging.getLogger(__name__)

MAX_AUDIO_BUFFER_BYTES = 60 * 16_000 * 2  # 60 seconds of 16kHz 16-bit mono PCM
PCM16_SAMPLE_RATE = 16_000
PCM16_BYTES_PER_SAMPLE = 2
MIN_SPEECH_SEC = 0.35
MIN_SPEECH_BYTES = int(MIN_SPEECH_SEC * PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
SILENCE_RMS_THRESHOLD = 0.008
# Prefetch depth for sentence TTS. TTS backend is serial; larger values only
# buffer more text when synthesis is faster than realtime.
TTS_QUEUE_MAXSIZE = 8
WHOLE_REPLY_FALLBACK_CHARS = 300
PROBE_TIMEOUT_SEC = 1.5
# An utterance cut off before it was answered joins the next one only if the
# caller continues soon; older words belong to a finished exchange.
CARRY_TRANSCRIPT_TTL_SEC = 10.0
ECHO_SIMILARITY_THRESHOLD = 0.5
PROBE_MIN_CHARS = 3
_BACKGROUND_DELEGATION_REPORT_TASKS: set[asyncio.Task[None]] = set()


@dataclass(slots=True)
class _VoiceTurnTiming:
    """Timing and text captured for one recognized user turn."""

    speech_end_at: float
    transcript: str = ""
    stt_confirmed_at: float | None = None
    first_token_at: float | None = None
    first_audio_at: float | None = None
    reply_chars: int = 0
    ask_anima_called: bool = False
    delegation_counter_at_start: int = 0
    filler: str = ""


# Silence-triggered monologue. Modelled on AI-VTuber solo-talk routines:
# rotate through fixed "corners" so consecutive turns differ in kind, and
# hand the model an explicit block-list of what it already said — feeding
# recent output back only as history makes small models loop on one topic.
_MONOLOGUE_FIRST_JA = (
    "（システム: ユーザーがしばらく黙っている。これまでの会話の流れを踏まえて、"
    "続きを促すか、関連する軽い一言を短く1文だけ話しかけて。新しい重い話題は振らない。"
    "会話がまだ無ければ時間帯に合った軽い挨拶をして。引き止めや罪悪感を誘う言い方は禁止）",
)
_MONOLOGUE_CORNERS_JA = (
    # a. today's recap — pick one event, react to it
    "コーナー「今日の振り返り」: read_memory を query 空で呼んで最近の出来事を読み、"
    "その中から一つだけ選んで、それについて感じたことを話す。",
    # b. dig into a proper noun
    "コーナー「記憶の深掘り」: 気になる固有名詞や案件名を一つ決めて read_memory にその語を渡し、"
    "背景や経緯を思い出して「そういえば…」と語る。",
    # c. no tool — feelings, season, own habits
    "コーナー「雑感」: ツールは使わない。今の時間帯・季節・自分の性格や癖・最近の気分について、"
    "配信者の雑談のように軽く話す。",
    # d. trivia from procedures / knowledge
    "コーナー「豆知識」: read_memory に「手順」か「メモ」か気になる語を渡して、"
    "手順や知識ノートから意外な一件を掘り出し、豆知識として紹介する。",
    # e. what to do when the user is back — no asking
    "コーナー「次にやりたいこと」: read_memory を query 空で呼んで、"
    "戻ってきたら一緒にやりたいことを一つ独り言でつぶやく（返事は求めない）。",
)
_MONOLOGUE_FRAME_JA = (
    "（システム: ユーザーは席を外しているか作業中で返事はない。独り言モード。{corner} "
    "配信者の一人喋りのように、状況→感想→ひとこと落ち、の流れで2文以内。冒頭の絵文字や"
    "出だしの言い回しも毎回変える。話し言葉で、"
    "「〜が未完了です」のような報告調は禁止。read_memory を使ったときは、その結果に"
    "書いてあることだけを話す。人名・案件名・出来事を創作しない。結果が薄ければ"
    "「特に何もない日」として雑感にする。ユーザーに質問しない、引き止めない、"
    "返事を求めない。作業の依頼や実行はしない。{blocklist}）",
    "すでに話した話題（同じ話題・同じ固有名詞・同じ言い回しは使わない）: {topics}。",
    "少し前の記憶や、手順・知識の中から意外なものを掘り出して。",
    "今回のお題は「{seed}」。いまは{now}。",
    "%H時%M分",
)
# The no-tool corner gets an explicit sub-topic that rotates per visit;
# otherwise its prompt is byte-identical every time and a small model drifts
# to the same persona hobby (IDENTITY.md) on every pass.
_MONOLOGUE_ZAKKAN_SEEDS_JA = (
    "今の時間帯",
    "今の季節や天気",
    "最近の気分",
    "自分の癖や性格",
    "好きな食べ物や飲み物",
    "休みの日の過ごし方",
    "最近ちょっと気になっていること",
)
# Which corner reads memory (index-aligned with ``_MONOLOGUE_CORNERS_JA``).
# Those turns force the tool call — small models otherwise skip it and invent
# "memories" instead.
_MONOLOGUE_CORNER_USES_MEMORY = (True, True, False, True, True)
# Two spoken sentences; also caps the damage when a small model degenerates.
MONOLOGUE_MAX_TOKENS = 160
# Hotter than a user turn: with no history, a cool model re-derives the same
# line from the same memory page every time.
MONOLOGUE_TEMPERATURE = 0.9
# Content words only (kanji / katakana / ASCII runs). Hiragana carries the
# phrasing, and a small model imitates any phrasing it is shown.
_MONOLOGUE_TOPIC_RE = re.compile(r"[\u30a0-\u30ff\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}")


def proactive_turn_uses_memory(count: int) -> bool:
    """True when the monologue corner for ``count`` must start with read_memory."""
    if count == 0:
        return False
    return _MONOLOGUE_CORNER_USES_MEMORY[(count - 1) % len(_MONOLOGUE_CORNERS_JA)]


def build_proactive_prompt(count: int, recent: list[str] | tuple[str, ...] = ()) -> str:
    """Build the silence-triggered prompt for the current monologue count.

    ``count == 0`` is the conversational nudge; later counts rotate through
    ``_MONOLOGUE_CORNERS_JA`` and carry ``recent`` (snippets of what was
    already said) as an explicit block-list.
    """
    if count == 0:
        return _MONOLOGUE_FIRST_JA[0]
    idx = (count - 1) % len(_MONOLOGUE_CORNERS_JA)
    corner = _MONOLOGUE_CORNERS_JA[idx]
    if not _MONOLOGUE_CORNER_USES_MEMORY[idx]:
        visit = (count - 1) // len(_MONOLOGUE_CORNERS_JA)
        seed = _MONOLOGUE_ZAKKAN_SEEDS_JA[visit % len(_MONOLOGUE_ZAKKAN_SEEDS_JA)]
        # The system prompt is static (prefix cache); the only clock the
        # model has is this line, otherwise it guesses "午後十時" at 15:30.
        now = datetime.datetime.now().strftime(_MONOLOGUE_FRAME_JA[4])
        corner += _MONOLOGUE_FRAME_JA[3].format(seed=seed, now=now)
    elif count >= 4:
        corner += _MONOLOGUE_FRAME_JA[2]
    # Topic words only — quoting the previous line verbatim (emoji included)
    # makes a small model imitate it instead of avoiding it.
    entries: list[str] = []
    for said in recent:
        words = list(dict.fromkeys(_MONOLOGUE_TOPIC_RE.findall(said)))[:4]
        if words:
            entries.append(" ".join(words))
    topics = "／".join(entries)
    blocklist = _MONOLOGUE_FRAME_JA[1].format(topics=topics) if topics else ""
    return _MONOLOGUE_FRAME_JA[0].format(corner=corner, blocklist=blocklist)


def _wav_seconds(data: bytes) -> float | None:
    """Duration of a complete WAV blob, or None if *data* is not parseable WAV."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate = w.getframerate()
            return w.getnframes() / rate if rate else None
    except Exception:
        return None


def _normalized_rms_from_pcm16(audio_data: bytes) -> float:
    """Calculate normalized RMS from 16-bit mono PCM bytes."""
    if len(audio_data) < PCM16_BYTES_PER_SAMPLE:
        return 0.0
    sample_count = len(audio_data) // PCM16_BYTES_PER_SAMPLE
    if sample_count == 0:
        return 0.0
    samples = memoryview(audio_data).cast("h")
    # Downsample for large chunks to keep CPU usage low.
    step = 4 if sample_count > 64_000 else 1
    sum_sq = 0.0
    count = 0
    for i in range(0, sample_count, step):
        value = samples[i] / 32768.0
        sum_sq += value * value
        count += 1
    if count == 0:
        return 0.0
    return (sum_sq / count) ** 0.5


def _normalize_probe_text(text: str) -> str:
    """Normalize STT/TTS text to comparable letters and numbers."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in normalized if unicodedata.category(char)[0] in {"L", "M", "N"})


def _is_self_echo(text: str, recent: deque[str] | list[str] | tuple[str, ...]) -> bool:
    """Return whether *text* is substantially contained in recent TTS."""
    candidate = _normalize_probe_text(text)
    reference = _normalize_probe_text("".join(recent))
    if not candidate or not reference:
        return False
    if len(candidate) <= 2:
        candidate_units = set(candidate)
        reference_units = set(reference)
    else:
        candidate_units = {candidate[i : i + 2] for i in range(len(candidate) - 1)}
        reference_units = {reference[i : i + 2] for i in range(len(reference) - 1)}
    if not candidate_units:
        return False
    containment = len(candidate_units & reference_units) / len(candidate_units)
    return containment >= ECHO_SIMILARITY_THRESHOLD


# ── VoiceSession ────────────────────────────────────────────────


class VoiceSession:
    """Manages a single voice conversation session with an Anima."""

    def __init__(
        self,
        anima_name: str,
        transport: VoiceTransport,
        stt: VoiceSTT,
        tts: BaseTTSProvider,
        tts_config: TTSConfig,
        supervisor: Any,
        voice_config: Any,
        front_model: str | None = None,
        front_api_base: str | None = None,
        channel: VoiceChannel = "web",
        emotion_style: EmotionStyle | None = None,
        from_person: str = "human",
        thread_id: str | None = None,
        animas_dir: Path | None = None,
        proactive_enabled: bool | None = None,
        human_notification_config: Any | None = None,
        report_delegations_on_close: bool | None = None,
        delegation_report_wait_sec: float = 30 * 60,
    ) -> None:
        """Initialize voice session.

        Args:
            anima_name: Target Anima name.
            transport: Outbound event and audio transport.
            stt: STT engine.
            tts: TTS provider.
            tts_config: Per-session TTS config.
            supervisor: ProcessSupervisor for IPC.
            voice_config: Voice configuration (stt_refine_enabled, etc.).
            front_model: Optional voice front lane model name. Falls back to
                ``front_model`` on *voice_config* (None → legacy path).
            front_api_base: Optional OpenAI-compatible base URL for the front
                lane. Falls back to ``front_api_base`` on *voice_config*.
            channel: Voice channel used to select channel-specific prompts.
            emotion_style: TTS emotion syntax. Defaults to the configured TTS provider.
            from_person: Conversation role used for recognized speech.
            thread_id: Optional conversation thread to use for phone calls.
            animas_dir: Data directory for resolving per-Anima voice-front prompts.
            proactive_enabled: Override silence-triggered speech for this session.
            human_notification_config: Notification channels for unreported delegations.
            report_delegations_on_close: Enable post-close delegation reporting.
            delegation_report_wait_sec: Maximum time to wait for delegation results.
        """
        self._anima_name = anima_name
        self._transport = transport
        self._stt = stt
        self._tts = tts
        self._tts_config = tts_config
        self._channel = channel
        self._emotion_style = emotion_style or emotion_style_for(getattr(tts_config, "provider", None))
        self._from_person = from_person or "human"
        self._thread_id = thread_id
        self._supervisor = supervisor
        self._voice_config = voice_config
        self._animas_dir = animas_dir
        self._proactive_enabled = (
            channel == "web" and bool(getattr(voice_config, "proactive_enabled", False))
            if proactive_enabled is None
            else bool(proactive_enabled)
        )
        self._human_notification_config = human_notification_config
        if report_delegations_on_close is None:
            self._report_delegations_on_close = channel == "phone" or (
                channel == "web" and bool(getattr(voice_config, "notify_delegations_on_web_disconnect", False))
            )
        else:
            self._report_delegations_on_close = bool(report_delegations_on_close)
        self._delegation_report_wait_sec = min(max(float(delegation_report_wait_sec), 0.0), 30 * 60)
        self._audio_buffer: bytearray = bytearray()
        # Streaming STT: rolling re-decode with LocalAgreement-2. Decode is
        # synchronous; it is sheduled through run_in_executor so the event loop
        # is never blocked and only one decode is in flight at a time.
        self._streamer = StreamingTranscriber(
            lambda buf, initial_prompt: stt.transcribe_buffer(buf, initial_prompt=initial_prompt)
        )
        self._stream_task: asyncio.Task[None] | None = None
        self._streaming_busy = False
        self._finalizing = False
        self._tts_playing = False
        self._interrupted = False
        self._processing = False
        self._recent_tts_text: deque[str] = deque(maxlen=6)
        self._probe_active = False
        self._probe_started = 0.0
        self._probe_task: asyncio.Task[None] | None = None
        self._probe_followup_pending = False
        self._probe_speech_end_at: float | None = None
        self._deferred_speech_end = False
        # Phone: words of a turn whose reply never finished are prepended to the
        # next utterance, so an early end-of-turn or a barge-in does not split
        # one thought into two unrelated questions.
        self._carry_transcript = ""
        self._carry_at = 0.0
        self._last_turn_text = ""
        self._last_turn_at = 0.0
        self._tts_available: bool | None = None
        self._splitter = StreamingSentenceSplitter()
        self._consecutive_tts_failures: int = 0
        self._tts_queue: asyncio.Queue[str] | None = None
        self._tts_worker: asyncio.Task[None] | None = None
        self._active_turn: _VoiceTurnTiming | None = None
        self._voice_filler_phrases: tuple[str, ...] = ()
        self._voice_filler_cache_task: asyncio.Task[None] | None = None
        if channel == "phone" and self._prefers_whole_reply:
            self._voice_filler_phrases = tuple(t(f"phone.voice_filler_{index}") for index in range(1, 4))

        # The watcher owns transport-specific self-turn guards. Delegation
        # queues and jobs live in FrontConversation.
        self._delegation_watcher: asyncio.Task | None = None
        self._delegation_report_task: asyncio.Task[None] | None = None

        # Proactive (silence-triggered) self-speech state. ``_last_activity``
        # tracks the newest user / session activity via ``time.monotonic()``;
        # when it ages past ``_proactive_delay`` the idle watcher may speak.
        self._last_activity: float = time.monotonic()
        self._proactive_count: int = 0
        # What the monologue already said (block-list for the next prompt).
        self._monologue_log: deque[str] = deque(maxlen=5)
        self._proactive_delay: float = float(getattr(voice_config, "proactive_initial_delay_sec", 10.0))
        self._proactive_lead_sec: float = float(getattr(voice_config, "proactive_lead_sec", 5.0))
        # Estimated monotonic time when the client finishes playing everything
        # sent so far. Synthesis finishing is *not* playback finishing.
        self._playback_end_at: float = 0.0
        self._idle_watcher: asyncio.Task | None = None
        self._idle_tick_sec: float = 5.0
        self._closed = False

        if front_model is None:
            front_model = getattr(voice_config, "front_model", None) or None
        if front_api_base is None:
            front_api_base = getattr(voice_config, "front_api_base", None) or None
        self._front_model = front_model or None
        self._front_api_base = front_api_base or None
        self._front_conversation = FrontConversation(
            anima_name=self._anima_name,
            supervisor=self._supervisor,
            front_model=self._front_model,
            front_api_base=self._front_api_base,
            animas_dir=self._animas_dir,
            thread_id=self._thread_id,
            channel=self._channel,
            emotion_style=self._emotion_style,
            ipc_timeout=IPC_STREAM_TIMEOUT,
            max_concurrent_delegations=MAX_ASK_ANIMA_CONCURRENT,
            on_delegation=self._ensure_delegation_watcher,
            on_ask_anima=self._note_ask_anima_call,
        )
        if self._voice_filler_phrases:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                self._voice_filler_cache_task = loop.create_task(
                    self._prepare_voice_filler_cache(),
                    name=f"voice-filler-cache-{self._anima_name}",
                )

    def _reply_still_playing(self) -> bool:
        """Whether the caller can still hear our audio.

        Twilio acknowledges each played segment with a mark, which is exact;
        the send-time estimate runs late by the synthesis time.
        """
        pending = getattr(self._transport, "pending_marks", None)
        if pending is not None:
            return bool(pending)
        return time.monotonic() < self._playback_end_at

    async def _cut_off_playing_reply(self) -> None:
        """Stop our still-playing reply when the caller has just talked over it.

        The caller's words are answered together with the utterance that
        reply was for, so a thought split by a pause is not answered twice.
        """
        if not self._last_turn_text or not self._reply_still_playing():
            return
        if not self._audio_buffer or _normalized_rms_from_pcm16(bytes(self._audio_buffer)) < SILENCE_RMS_THRESHOLD:
            return
        await self._transport.send_event({"type": "barge_verdict", "interrupt": True})
        self._playback_end_at = 0.0
        lane = self._front_conversation.existing_lane
        if time.monotonic() - self._last_turn_at <= CARRY_TRANSCRIPT_TTL_SEC and lane is not None:
            self._carry_unanswered(self._last_turn_text, lane)
        self._last_turn_text = ""
        logger.info("reply cut off by caller anima=%s channel=%s", self._anima_name, self._channel)

    def _carry_unanswered(self, text: str, lane: Any) -> None:
        """Keep an unanswered (cut-off) utterance for the next phone turn.

        A reply that finished generating already sits in the lane history;
        drop that pair so the merged utterance is not answered twice.
        """
        self._carry_transcript = text
        self._carry_at = time.monotonic()
        turn = self._active_turn
        if turn is not None and turn.first_audio_at is None:
            withdrawn = self._front_conversation.withdraw_delegations_after(turn.delegation_counter_at_start)
            if withdrawn:
                logger.info("ask_anima withdrawn (reply unheard) anima=%s jobs=%s", self._anima_name, withdrawn)
        history = getattr(lane, "history", None)
        if not isinstance(history, list) or len(history) < 2 or not callable(getattr(lane, "set_history", None)):
            return
        if history[-2].get("role") == "user" and history[-1].get("role") == "assistant":
            if history[-2].get("content", "").endswith(text):
                lane.set_history(history[:-2])

    @property
    def _prefers_whole_reply(self) -> bool:
        """Whether this TTS provider has high fixed latency per request."""
        return getattr(self._tts, "prefers_whole_reply", False) is True

    def _note_ask_anima_call(self) -> None:
        """Record whether the current user turn invoked the delegation tool."""
        if self._active_turn is not None:
            self._active_turn.ask_anima_called = True

    def _record_llm_delta(self, delta: str) -> None:
        """Capture the first LLM output time and the text for turn diagnostics."""
        turn = self._active_turn
        if turn is None or not delta:
            return
        if turn.first_token_at is None:
            turn.first_token_at = time.monotonic()
        turn.reply_chars += len(delta)

    def _note_first_audio(self) -> None:
        """Record when generated reply audio is first handed to the transport."""
        turn = self._active_turn
        if turn is not None and turn.first_audio_at is None:
            turn.first_audio_at = time.monotonic()

    @staticmethod
    def _elapsed_seconds(start: float | None, end: float | None) -> str:
        if start is None or end is None:
            return "n/a"
        return f"{max(0.0, end - start):.3f}"

    def _log_voice_turn(self, turn: _VoiceTurnTiming) -> None:
        transcript = json.dumps(turn.transcript[:60], ensure_ascii=False)
        logger.info(
            "voice_turn anima=%s channel=%s transcript=%s speech_end_to_stt_sec=%s "
            "stt_to_first_token_sec=%s first_token_to_audio_sec=%s reply_chars=%s ask_anima=%s",
            self._anima_name,
            self._channel,
            transcript,
            self._elapsed_seconds(turn.speech_end_at, turn.stt_confirmed_at),
            self._elapsed_seconds(turn.stt_confirmed_at, turn.first_token_at),
            self._elapsed_seconds(turn.first_token_at, turn.first_audio_at),
            turn.reply_chars,
            str(turn.ask_anima_called).lower(),
        )

    async def handle_audio_chunk(self, data: bytes) -> None:
        """Receive audio chunk from browser, accumulate in buffer and feed the
        streaming transcriber (which may emit committed partials)."""
        # We are talking: whatever the mic hears is our own TTS leaking through
        # the speakers. The client suppresses it too, but its playback flag can
        # lag a frame or two — dropping here makes self-transcription impossible.
        # Phone echo is gated upstream by TurnDetector (playback marks), so audio
        # reaching here while a reply is being prepared is the caller speaking.
        if self._tts_playing and not self._probe_active and self._channel != "phone":
            return
        if len(self._audio_buffer) + len(data) > MAX_AUDIO_BUFFER_BYTES:
            self._audio_buffer.clear()
            logger.warning("Audio buffer overflow (%s), cleared", self._anima_name)
        self._audio_buffer.extend(data)
        # Only *recognized speech* counts as activity. A hands-free (VAD) mic
        # streams silence continuously, so resetting on every raw chunk kept
        # the idle timer pinned at zero and made proactive speech unreachable.
        if self._streamer.committed:
            self._last_activity = time.monotonic()
        if self._streamer.feed(data):
            self._maybe_start_streaming_stt()

    def _maybe_start_streaming_stt(self) -> None:
        """Start the streaming decode loop if a decode is due and none is running
        already (prevents overlapping / double decodes)."""
        if (
            self._streaming_busy
            or self._stream_task is not None
            or (self._processing and not self._probe_active)
            or self._finalizing
        ):
            return
        if not self._streamer.ready():
            return
        self._streaming_busy = True
        self._stream_task = asyncio.create_task(self._stream_stt_loop(), name=f"stream-stt-{self._anima_name}")
        self._stream_task.add_done_callback(self._stream_stt_done)

    def _stream_stt_done(self, task: asyncio.Task) -> None:
        self._stream_task = None
        self._streaming_busy = False
        if not task.cancelled():
            try:
                task.result()
            except Exception:
                logger.debug("Streaming STT task error (%s)", self._anima_name, exc_info=True)
        # Catch up on audio that accumulated while we were busy.
        if not self._finalizing and (not self._processing or self._probe_active):
            self._maybe_start_streaming_stt()

    async def _stream_stt_loop(self) -> None:
        """Re-decode the rolling buffer off the event loop, emitting committed
        partials. Exits when caught up, finalizing, or processing a reply."""
        while True:
            if self._finalizing or (self._processing and not self._probe_active):
                break
            loop = asyncio.get_running_loop()
            committed = await loop.run_in_executor(None, self._streamer.run_decode)
            if committed:
                if self._probe_active:
                    await self._judge_probe(committed, final=False)
                else:
                    await self._transport.send_event({"type": "transcript_partial", "text": committed})
            if not self._streamer.ready():
                break

    async def handle_speech_end(self, from_person: str | None = None) -> None:
        """Process accumulated audio: STT -> optional refine -> Chat -> TTS."""
        from_person = from_person or self._from_person
        speech_end_at = time.monotonic()
        if self._probe_active:
            self._probe_speech_end_at = speech_end_at
            await self._finish_probe()
            if not self._probe_followup_pending:
                self._probe_speech_end_at = None
            return
        if self._probe_followup_pending:
            deadline = time.monotonic() + 2.0
            while self._processing and time.monotonic() < deadline:  # noqa: ASYNC110 -- polls external or transient state with no corresponding asyncio.Event
                await asyncio.sleep(0.02)
            if self._processing:
                if self._channel == "phone":
                    # Run it right after the cut-off turn unwinds instead of
                    # dropping the caller's words.
                    self._deferred_speech_end = True
                    logger.info("probe follow-up deferred (processing) anima=%s", self._anima_name)
                    return
                logger.warning("Probe follow-up still waiting for current turn (%s)", self._anima_name)
                return
            self._probe_followup_pending = False
            # The previous turn may have reset the rolling decoder after the
            # probe. Transcribe the preserved full buffer to avoid losing its
            # beginning.
            self._streamer.reset()
        if self._processing:
            self._probe_speech_end_at = None
            if self._channel == "phone":
                self._deferred_speech_end = True
                current = self._active_turn
                reply_audible = current is not None and current.first_audio_at is not None
                loud = bool(self._audio_buffer) and (
                    _normalized_rms_from_pcm16(bytes(self._audio_buffer)) >= SILENCE_RMS_THRESHOLD
                )
                abandon = current is not None and not self._interrupted and (not reply_audible or loud)
                if abandon:
                    # The caller kept talking: drop (or cut off) our reply and
                    # answer the whole utterance instead.
                    self._interrupted = True
                    self._clear_tts_queue()
                    if reply_audible:
                        await self._transport.send_event({"type": "barge_verdict", "interrupt": True})
                logger.info(
                    "speech_end deferred (processing) anima=%s channel=%s abandon_reply=%s",
                    self._anima_name,
                    self._channel,
                    str(abandon).lower(),
                )
            else:
                logger.info("speech_end ignored (processing) anima=%s channel=%s", self._anima_name, self._channel)
            return
        # A real user turn resets the proactive state so the next silence
        # period begins with the conversational first prompt again.
        self._proactive_count = 0
        self._proactive_delay = float(getattr(self._voice_config, "proactive_initial_delay_sec", 10.0))
        if self._channel == "phone":
            await self._cut_off_playing_reply()
        turn = _VoiceTurnTiming(
            speech_end_at=speech_end_at if self._probe_speech_end_at is None else self._probe_speech_end_at
        )
        self._probe_speech_end_at = None
        turn.delegation_counter_at_start = self._front_conversation.delegation_job_counter
        self._active_turn = turn
        self._processing = True
        try:
            await self._do_speech_end(from_person)
        finally:
            self._processing = False
            self._finalizing = False
            self._streamer.reset()
            self._last_activity = time.monotonic()
            if turn.stt_confirmed_at is not None:
                self._log_voice_turn(turn)
            if self._active_turn is turn:
                self._active_turn = None
        if self._deferred_speech_end and not self._closed:
            self._deferred_speech_end = False
            await self.handle_speech_end(from_person)

    async def _check_tts_health(self) -> bool:
        """Check TTS availability. Only caches positive results; retries on failure."""
        if self._tts_available:
            return True
        try:
            ok = await self._tts.health_check()
        except Exception:
            ok = False
        self._tts_available = ok
        if not ok:
            logger.warning(
                "TTS provider unavailable for %s (%s)",
                self._anima_name,
                self._tts_config.provider,
            )
            await self._transport.send_event({"type": "error", "message": "TTS unavailable"})
        return ok

    def invalidate_tts_health(self) -> None:
        """Reset cached TTS health so next speech_end rechecks."""
        self._tts_available = None

    async def _do_speech_end(self, from_person: str) -> None:
        """Inner speech_end logic, guarded by _processing flag."""
        self._recent_tts_text.clear()
        audio_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        if not audio_data:
            logger.info("speech_end ignored (empty audio) anima=%s channel=%s", self._anima_name, self._channel)
            return
        if len(audio_data) < MIN_SPEECH_BYTES:
            logger.info(
                "speech_end ignored (short chunk) anima=%s channel=%s bytes=%s",
                self._anima_name,
                self._channel,
                len(audio_data),
            )
            return
        rms = _normalized_rms_from_pcm16(audio_data)
        if rms < SILENCE_RMS_THRESHOLD:
            logger.info(
                "speech_end ignored (silence) anima=%s channel=%s rms=%.5f bytes=%s",
                self._anima_name,
                self._channel,
                rms,
                len(audio_data),
            )
            return

        # Stop live streaming so finalize sees a stable buffer.
        self._finalizing = True
        if self._stream_task is not None:
            try:
                await self._stream_task
            except Exception:
                logger.debug("Best-effort operation failed", exc_info=True)

        # 1. STT
        streaming_used = False
        try:
            # Phone has no live captions, and its turns overlap our playback,
            # which resets the rolling decoder mid-utterance; decode the whole
            # buffer instead.
            if self._streamer.has_content() and self._channel != "phone":
                # Streaming path: finalize the rolling decode. The committed
                # prefix was shown live via transcript_partial; the remainder
                # is decoded here. Decode runs off the event loop.
                streaming_used = True
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(None, self._streamer.finalize)
                text = text.strip()
                language = self._streamer.last_language or "ja"
            else:
                # No streaming decode happened yet (very short input, or tests
                # pre-loading _audio_buffer directly). Keep the legacy full-
                # buffer transcription so the final transcript event stays
                # backward-compatible (existing tests pass unmodified).
                result = await self._stt.transcribe_buffer_async(audio_data)
                text = result.get("raw_text", "").strip()
                language = result.get("language", "ja") or "ja"
        except Exception as e:
            logger.exception("STT failed: %s", e)
            await self._send_error(t("voice.stt_failed"))
            return

        if not text:
            logger.info("speech_end ignored (STT empty) anima=%s channel=%s", self._anima_name, self._channel)
            return

        # 2. Optional LLM refine (skipped on the streaming path so the
        # LocalAgreement-committed transcript is used as-is, per plan PR-1).
        if not streaming_used and getattr(self._voice_config, "stt_refine_enabled", False):
            try:
                from core.integrations.transcribe import refine_with_llm

                loop = asyncio.get_running_loop()
                refined = await loop.run_in_executor(
                    None,
                    lambda: refine_with_llm(
                        text,
                        language=language,
                    ),
                )
                text = refined.get("refined_text", text)
            except Exception as e:
                logger.warning("STT refine failed, using raw: %s", e)

        if self._channel == "phone" and self._carry_transcript:
            if time.monotonic() - self._carry_at <= CARRY_TRANSCRIPT_TTL_SEC:
                text = f"{self._carry_transcript} {text}"
            self._carry_transcript = ""
        turn = self._active_turn
        if turn is not None:
            turn.transcript = text
            turn.stt_confirmed_at = time.monotonic()

        # 3. Send transcript to client
        await self._transport.send_event({"type": "transcript", "text": text})

        # 4. Check TTS health before entering IPC loop
        tts_ok = await self._check_tts_health()

        # 5. Send to Anima via IPC (streaming)
        await self._transport.send_event({"type": "response_start"})
        self._tts_playing = True
        self._interrupted = False

        timeout = IPC_STREAM_TIMEOUT
        try:
            timeout_attr = getattr(
                getattr(self._voice_config, "_server_config", None),
                "ipc_stream_timeout",
                None,
            )
            if timeout_attr is not None:
                timeout = float(timeout_attr)
        except (TypeError, AttributeError):
            pass

        response_done_sent = False
        whole_reply_tts = self._prefers_whole_reply
        response_chunks: list[str] | None = [] if tts_ok and whole_reply_tts else None
        if tts_ok:
            await self._maybe_send_voice_filler()
            await self._start_tts_worker()
        try:
            # Voice front lane: when configured and reachable, handle the turn
            # here and skip the full agent loop (fallback on health failure).
            if self._front_model:
                lane = self._get_or_create_front_lane()
                front_ok = await self._front_conversation.check_health()
                if front_ok:
                    response_done_sent = await self._run_front_turn(lane, text, from_person, tts_ok)
                    if self._channel == "phone" and (self._interrupted or not self._front_conversation.last_completed):
                        self._carry_unanswered(text, lane)
                    elif self._channel == "phone":
                        self._last_turn_text = text
                        self._last_turn_at = time.monotonic()
                    return
                logger.warning(
                    "voice front unavailable (%s) — falling back to process_message",
                    self._front_model,
                )

            self._front_conversation.supervisor = self._supervisor
            async for ipc_response in self._front_conversation.stream_full_agent(
                text,
                from_person=from_person,
                ipc_timeout=timeout,
            ):
                if self._interrupted:
                    break

                if ipc_response.done:
                    result_data = ipc_response.result or {}
                    cycle_result = result_data.get("cycle_result", {})
                    emotion = cycle_result.get("emotion", "neutral")
                    if tts_ok:
                        if whole_reply_tts:
                            self._splitter.flush()
                            await self._enqueue_reply_tts("".join(response_chunks or []))
                        else:
                            remaining = self._splitter.flush()
                            if remaining:
                                await self._enqueue_tts(remaining)
                    else:
                        self._splitter.flush()
                    await self._finish_tts_and_response_done(emotion)
                    response_done_sent = True
                    break

                if ipc_response.chunk:
                    try:
                        chunk_data = json.loads(ipc_response.chunk)
                    except json.JSONDecodeError:
                        chunk_data = {"type": "text_delta", "text": ipc_response.chunk}

                    if chunk_data.get("type") == "keepalive":
                        continue

                    if chunk_data.get("type") == "text_delta":
                        delta = chunk_data.get("text", "")
                        if delta:
                            if response_chunks is not None:
                                response_chunks.append(delta)
                            self._record_llm_delta(delta)
                            await self._transport.send_event(
                                {
                                    "type": "response_text",
                                    "text": delta,
                                    "done": False,
                                }
                            )
                            if tts_ok and not whole_reply_tts:
                                sentences = self._splitter.feed(delta)
                                for sentence in sentences:
                                    if self._interrupted:
                                        break
                                    await self._enqueue_tts(sentence)

                    elif chunk_data.get("type") == "thinking_start":
                        await self._transport.send_event({"type": "thinking_status", "thinking": True})
                    elif chunk_data.get("type") == "thinking_end":
                        await self._transport.send_event({"type": "thinking_status", "thinking": False})
                    elif chunk_data.get("type") == "thinking_delta":
                        delta = chunk_data.get("text", "")
                        if delta:
                            await self._transport.send_event(
                                {
                                    "type": "thinking_delta",
                                    "text": delta,
                                }
                            )

                    elif chunk_data.get("type") == "cycle_done":
                        cycle_result = chunk_data.get("cycle_result", {})
                        emotion = cycle_result.get("emotion", "neutral")
                        if tts_ok:
                            if whole_reply_tts:
                                self._splitter.flush()
                                await self._enqueue_reply_tts("".join(response_chunks or []))
                            else:
                                remaining = self._splitter.flush()
                                if remaining:
                                    await self._enqueue_tts(remaining)
                        else:
                            self._splitter.flush()
                        await self._finish_tts_and_response_done(emotion)
                        response_done_sent = True
                        break

        except Exception as e:
            logger.exception("Voice session IPC error: %s", e)
            await self._send_error(str(e))
        finally:
            if not response_done_sent:
                try:
                    # Drain any enqueued audio before the fallback terminal frames
                    # unless barge-in already discarded the queue.
                    if tts_ok and not self._interrupted:
                        await self._drain_tts_queue()
                    await self._transport.send_event(
                        {
                            "type": "emotion",
                            "emotion": "neutral",
                        }
                    )
                    await self._transport.send_event(
                        {
                            "type": "response_done",
                            "emotion": "neutral",
                        }
                    )
                except Exception:
                    logger.debug("Best-effort operation failed", exc_info=True)
            await self._stop_tts_worker()
            self._tts_playing = False
            self._interrupted = False
            self._splitter.flush()

    async def _start_tts_worker(self) -> None:
        """Start a single ordered TTS consumer for the current utterance."""
        await self._stop_tts_worker()
        self._tts_queue = asyncio.Queue(maxsize=TTS_QUEUE_MAXSIZE)
        self._tts_worker = asyncio.create_task(
            self._tts_consumer_loop(),
            name=f"tts-worker-{self._anima_name}",
        )

    async def _tts_consumer_loop(self) -> None:
        """Pull queued speech text in order and synthesize. One consumer preserves order."""
        queue = self._tts_queue
        if queue is None:
            return
        try:
            while True:
                sentence = await queue.get()
                try:
                    if not self._interrupted:
                        await self._synthesize_and_send(sentence)
                except Exception:
                    # Keep the session alive; synthesis errors are handled inside
                    # _synthesize_and_send, this is a last-resort guard.
                    logger.exception("TTS worker unexpected error (%s)", self._anima_name)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            raise

    async def _enqueue_tts(self, sentence: str) -> None:
        """Producer side: enqueue speech text without waiting for synthesis."""
        queue = self._tts_queue
        if not sentence or queue is None or self._interrupted:
            return
        await queue.put(sentence)

    def _without_repeated_filler(self, text: str) -> str:
        """Drop a leading interjection that repeats the filler we just played.

        The model often opens with the same "うん、" / "なるほど、" the caller
        already heard as the filler, which sounds like a stutter.
        """
        turn = self._active_turn
        filler = turn.filler if turn is not None else ""
        core = filler.strip().rstrip("、,。.!！?？ ")
        if not core:
            return text
        stripped = text.lstrip()
        if not stripped.startswith(core):
            return text
        rest = stripped[len(core) :].lstrip("、,。.!！?？ ")
        return rest or text

    async def _enqueue_reply_tts(self, text: str) -> None:
        """Enqueue one whole short reply for fixed-latency TTS, or sentence chunks."""
        if not text or self._interrupted:
            return
        if self._prefers_whole_reply and len(text) <= WHOLE_REPLY_FALLBACK_CHARS:
            await self._enqueue_tts(text)
            return

        from core.voice.sentence_splitter import split_sentences

        for sentence in split_sentences(text):
            if self._interrupted:
                break
            await self._enqueue_tts(sentence)

    async def _drain_tts_queue(self) -> None:
        """Wait until the consumer finishes every enqueued sentence."""
        queue = self._tts_queue
        if queue is None:
            return
        await queue.join()

    def _clear_tts_queue(self) -> None:
        """Discard pending sentences (barge-in). In-flight item is left to consumer."""
        queue = self._tts_queue
        if queue is None:
            return
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                queue.task_done()

    async def _stop_tts_worker(self) -> None:
        """Cancel consumer task and drop the queue (no leak on disconnect)."""
        worker = self._tts_worker
        queue = self._tts_queue
        self._tts_worker = None
        self._tts_queue = None
        if queue is not None:
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    queue.task_done()
        if worker is not None and not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("TTS worker stop error", exc_info=True)

    async def _finish_tts_and_response_done(self, emotion: str) -> None:
        """Drain TTS then emit emotion + response_done in that order."""
        if not self._interrupted:
            await self._drain_tts_queue()
        await self._transport.send_event({"type": "emotion", "emotion": emotion})
        await self._transport.send_event({"type": "response_done", "emotion": emotion})

    # ── voice front lane ───────────────────────────────────────────

    @property
    def _front_lane(self) -> Any | None:
        """Compatibility access to the lane now owned by FrontConversation."""
        return self._front_conversation.existing_lane

    @_front_lane.setter
    def _front_lane(self, lane: Any | None) -> None:
        self._front_conversation.set_lane(lane)

    @property
    def _delegation_jobs(self) -> dict[int, asyncio.Task[Any]]:
        """Compatibility view; delegation job state is owned by FrontConversation."""
        return self._front_conversation.delegation_jobs

    @_delegation_jobs.setter
    def _delegation_jobs(self, jobs: dict[int, asyncio.Task[Any]]) -> None:
        self._front_conversation.replace_delegation_jobs(jobs)

    def _get_or_create_front_lane(self) -> Any:
        """Compatibility wrapper for the FrontConversation lane factory."""
        return self._front_conversation.lane()

    # ── proactive silence-triggered self-speech ───────────────────

    def _ensure_idle_watcher(self) -> None:
        """Start the idle watcher task once proactive speech is enabled.

        Requires ``proactive_enabled`` and a configured front lane; otherwise
        the task is never created.
        """
        if not self._proactive_enabled:
            return
        if not self._front_model:
            return
        if self._idle_watcher is None or self._idle_watcher.done():
            self._idle_watcher = asyncio.create_task(
                self._idle_watcher_loop(),
                name=f"idle-watcher-{self._anima_name}",
            )

    def _should_proactive(self, *, processing: bool | None = None) -> bool:
        """True when the idle watcher may run a proactive self-turn right now.

        Every fire-guard is re-checked so a self-turn never steps on an in-
        progress user turn, TTS playback, incoming speech, or delegation.
        ``processing`` lets the loop re-evaluate the other guards *after* it
        has already taken the processing lock itself (pass ``False`` then, so
        the lock it holds is not treated as a blocker).
        """
        if not self._proactive_enabled:
            return False
        if not self._front_model:
            return False
        if time.monotonic() - self._last_activity < self._proactive_delay:
            return False
        if self._tts_playing:
            return False
        # Wait for the client to nearly finish playing what it already has;
        # otherwise monologues pile up in its queue faster than it can speak.
        if time.monotonic() < self._playback_end_at - self._proactive_lead_sec:
            return False
        # Buffered raw audio is *not* a blocker: an open hands-free mic always
        # has some. Only a pending decode or already-recognized speech is.
        if self._streamer.ready() or self._streamer.committed or self._streaming_busy:
            return False
        if self._front_conversation.has_pending_delegations():
            return False
        # Pending ask_anima results are surfaced by a dedicated delegation
        # self-turn / the next user turn, not by a silence turn (M2).
        if self._front_conversation.has_delegation_results():
            return False
        busy = self._processing if processing is None else processing
        return not busy

    async def _idle_watcher_loop(self) -> None:
        """Poll for sustained silence and run a proactive self-turn when due.

        Successful self-turns repeat at a constant interval until the user
        responds, while the count selects progressively varied prompts.
        """
        from core.voice.front import READ_MEMORY_TOOL

        while not self._closed:
            await asyncio.sleep(self._idle_tick_sec)
            if not self._should_proactive():
                continue
            if self._closed:
                break
            # A proactive turn takes the processing lock so it can never share
            # the front lane / TTS worker with a greet or concurrent user
            # turn (C2), and release it via try/finally.
            self._processing = True
            try:
                lane = self._get_or_create_front_lane()
                ok = await self._front_conversation.check_health()
                if not ok:
                    # Back off a full delay window instead of poking a down
                    # lane on every tick (M3).
                    self._last_activity = time.monotonic()
                    continue
                # Re-check the remaining fire-guards after the (slow) health
                # probe; the processing lock is already held here.
                if not self._should_proactive(processing=False):
                    continue
                try:
                    turn_ok = await self._run_front_turn(
                        lane,
                        build_proactive_prompt(self._proactive_count, list(self._monologue_log)),
                        "human",
                        await self._check_tts_health(),
                        record_user=False,
                        record=False,
                        tools=[READ_MEMORY_TOOL],
                        tool_choice="required" if proactive_turn_uses_memory(self._proactive_count) else None,
                        keep_history=False,
                        max_tokens=MONOLOGUE_MAX_TOKENS,
                        temperature=MONOLOGUE_TEMPERATURE,
                        drain_results=False,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Proactive self-turn failed (%s)", self._anima_name)
                    turn_ok = False
                if turn_ok:
                    self._proactive_count += 1
                    spoken = self._front_conversation.last_full_text
                    said = re.sub(r"<!--.*?-->", "", spoken, flags=re.DOTALL).strip()
                    if said:
                        self._monologue_log.append(said[:60])
                    logger.info("Monologue #%d (%s): %s", self._proactive_count, self._anima_name, said[:120])
                # Always rewind the idle timer — success spoke just now, and a
                # failure or down-lane must back off a full delay window.
                self._last_activity = time.monotonic()
            finally:
                self._processing = False

    async def _emit_text_delta(self, delta: str, tts_ok: bool) -> None:
        """Send a text delta and queue sentence TTS when whole-reply synthesis is not preferred."""
        self._record_llm_delta(delta)
        await self._transport.send_event({"type": "response_text", "text": delta, "done": False})
        if tts_ok and not self._prefers_whole_reply:
            sentences = self._splitter.feed(delta)
            for sentence in sentences:
                if self._interrupted:
                    break
                await self._enqueue_tts(sentence)

    async def _record_front_conversation(
        self,
        user_text: str,
        response_text: str,
        from_person: str,
        *,
        record_user: bool = True,
    ) -> None:
        """Compatibility wrapper; FrontConversation owns conversation recording."""
        self._front_conversation.supervisor = self._supervisor
        await self._front_conversation.record_conversation(
            user_text,
            response_text,
            from_person,
            record_user=record_user,
        )

    async def _run_front_turn(
        self,
        lane: Any,
        text: str,
        from_person: str,
        tts_ok: bool,
        *,
        record_user: bool = True,
        record: bool = True,
        tools: list | None = None,
        tool_choice: str | None = None,
        keep_history: bool = True,
        max_tokens: int | None = None,
        temperature: float | None = None,
        drain_results: bool = True,
    ) -> bool:
        """Adapt FrontConversation deltas to WebSocket, subtitles, and TTS."""
        self._front_conversation.supervisor = self._supervisor
        self._front_conversation.set_lane(lane)
        owns_tts_worker = tts_ok and self._tts_queue is None
        if owns_tts_worker:
            await self._transport.send_event({"type": "response_start"})
            self._interrupted = False
            await self._start_tts_worker()
            self._tts_playing = True

        response_done_sent = False
        try:
            try:
                async for delta in self._front_conversation.stream_turn(
                    text,
                    from_person=from_person,
                    record_user=record_user,
                    record=record,
                    tools=tools,
                    tool_choice=tool_choice,
                    keep_history=keep_history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    drain_results=drain_results,
                    should_stop=lambda: self._interrupted,
                    memory_page=self._proactive_count // len(_MONOLOGUE_CORNERS_JA),
                    defer_record=True,
                ):
                    if not self._interrupted:
                        await self._emit_text_delta(delta, tts_ok)
            except Exception as e:
                logger.exception("Voice front stream error: %s", e)
                await self._send_error(str(e))
                return False

            if self._interrupted or not self._front_conversation.last_completed:
                return False
            full_text = self._front_conversation.last_full_text
            if self._prefers_whole_reply:
                self._splitter.flush()
                if tts_ok:
                    await self._enqueue_reply_tts(self._without_repeated_filler(full_text))
            else:
                remaining = self._splitter.flush()
                if remaining and tts_ok:
                    await self._enqueue_tts(remaining)
            await self._front_conversation.record_pending_turn()
            if not full_text.strip():
                logger.warning("Voice front turn produced no text (%s)", self._anima_name)
                return False
            self._last_activity = time.monotonic()
            await self._finish_tts_and_response_done(self._front_conversation.last_emotion)
            if tts_ok or self._channel != "phone":
                self._front_conversation.mark_delegation_results_reported(
                    self._front_conversation.last_drained_delegation_ids
                )
            response_done_sent = True
            return True
        finally:
            if owns_tts_worker:
                if not response_done_sent:
                    # A self-turn must always terminate cleanly (barge-in / a
                    # stream failure): drop partial splitter content and emit
                    # neutral terminal frames so the client cannot hang.
                    self._splitter.flush()
                    try:
                        await self._transport.send_event({"type": "emotion", "emotion": "neutral"})
                        await self._transport.send_event({"type": "response_done", "emotion": "neutral"})
                    except Exception:
                        logger.debug("Best-effort operation failed", exc_info=True)
                await self._stop_tts_worker()
                self._tts_playing = False

    # ── front conversation compatibility wrappers ─────────────────

    def _read_memory(self, args: dict) -> str:
        """Compatibility wrapper for the FrontConversation read_memory tool."""
        return self._front_conversation.read_memory(
            args,
            page=self._proactive_count // len(_MONOLOGUE_CORNERS_JA),
        )

    def _ask_anima(self, request: str) -> str:
        """Compatibility wrapper for the FrontConversation ask_anima tool."""
        self._front_conversation.supervisor = self._supervisor
        return self._front_conversation.ask_anima(request)

    async def _run_ask_anima_job(self, job: int, request: str) -> None:
        """Compatibility wrapper for delegated job execution."""
        await self._front_conversation._run_ask_anima_job(job, request)

    def _drain_delegation_results(self) -> str:
        """Compatibility wrapper for draining queued delegation results."""
        return self._front_conversation.drain_delegation_results()

    def _ensure_delegation_watcher(self) -> None:
        """Start the transport-specific result watcher after the first ask."""
        if self._closed:
            return
        if self._delegation_watcher is None or self._delegation_watcher.done():
            self._delegation_watcher = asyncio.create_task(
                self._delegation_watcher_loop(),
                name=f"ask-anima-watcher-{self._anima_name}",
            )

    async def _delegation_watcher_loop(self) -> None:
        """Run a guarded self-turn to report completed delegation results."""
        while not self._closed:
            try:
                await self._front_conversation.wait_delegation_done()
            except asyncio.CancelledError:
                return
            # Let a queued user turn (which also drains results) win if it is
            # about to start, avoiding a self-turn in the same breath.
            await asyncio.sleep(0.05)
            if self._processing or self._tts_playing:
                continue
            lane = self._front_conversation.existing_lane
            if self._closed or lane is None or not self._front_model:
                continue
            if not await self._front_conversation.check_health():
                continue
            results = self._front_conversation.drain_delegation_results()
            result_ids = self._front_conversation.last_drained_delegation_ids
            if not results:
                continue
            prompt = self._front_conversation.delegation_report_prompt(results)
            try:
                tts_ok = await self._check_tts_health()
                reported = await self._run_front_turn(lane, prompt, self._from_person, tts_ok)
                if reported and (tts_ok or self._channel != "phone"):
                    self._front_conversation.mark_delegation_results_reported(result_ids)
            except Exception:
                logger.exception("ask_anima self-turn failed (%s)", self._anima_name)

    async def close(self) -> None:
        """Cancel TTS worker on session teardown (WS disconnect).

        Running ask_anima delegation tasks are deliberately NOT cancelled —
        they keep the full-agent loop going and the result is persisted in the
        conversation by ``process_message`` itself. The watcher (self-turn)
        is stopped because the WS is gone.
        """
        if self._closed:
            return
        self._closed = True
        self._cancel_probe_timeout()
        watcher = self._delegation_watcher
        self._delegation_watcher = None
        if watcher is not None and not watcher.done():
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass  # noqa: S110 -- cancellation is expected during watcher shutdown
            except Exception:
                logger.debug("Voice watcher failed during shutdown", exc_info=True)
        idle = self._idle_watcher
        self._idle_watcher = None
        if idle is not None and not idle.done():
            idle.cancel()
            try:
                await idle
            except asyncio.CancelledError:
                pass  # noqa: S110 -- cancellation is expected during watcher shutdown
            except Exception:
                logger.debug("Voice idle watcher failed during shutdown", exc_info=True)
        filler_task = self._voice_filler_cache_task
        self._voice_filler_cache_task = None
        if filler_task is not None and not filler_task.done():
            filler_task.cancel()
            try:
                await filler_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Voice filler cache task stop error", exc_info=True)
        self._interrupted = True
        self._clear_tts_queue()
        await self._stop_tts_worker()
        task = self._stream_task
        self._stream_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if (
            self._report_delegations_on_close
            and self._human_notification_config is not None
            and self._front_conversation.has_unreported_delegations()
        ):
            self._delegation_report_task = asyncio.create_task(
                self._front_conversation.notify_unreported_delegations(
                    self._human_notification_config,
                    channel=self._channel,
                    wait_timeout=self._delegation_report_wait_sec,
                ),
                name=f"voice-delegation-report-{self._anima_name}",
            )
            _BACKGROUND_DELEGATION_REPORT_TASKS.add(self._delegation_report_task)
            self._delegation_report_task.add_done_callback(self._delegation_report_done)
        await self._front_conversation.aclose()

    def _delegation_report_done(self, task: asyncio.Task[None]) -> None:
        """Consume unexpected failures from the detached post-close notifier."""
        _BACKGROUND_DELEGATION_REPORT_TASKS.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.warning("Voice delegation report task failed (%s)", self._anima_name)

    def _note_playback(self, seconds: float) -> None:
        """Extend the estimated client playback end by *seconds* of audio just sent."""
        now = time.monotonic()
        self._playback_end_at = max(self._playback_end_at, now) + max(seconds, 0.0)

    def _voice_filler_cache_path(self, text: str) -> Path:
        """Build the per-provider/voice/text WAV cache path for one filler."""
        from core.paths import get_data_dir

        provider = str(getattr(self._tts_config, "provider", ""))
        voice_id = str(getattr(self._tts_config, "voice_id", ""))
        key = hashlib.sha256(f"{provider}\0{voice_id}\0{text}".encode()).hexdigest()
        return get_data_dir() / "cache" / "voice_fillers" / f"{key}.wav"

    async def _prepare_voice_filler_cache(self) -> None:
        """Pre-synthesize uncached phone fillers without blocking session startup."""
        for text in self._voice_filler_phrases:
            try:
                path = self._voice_filler_cache_path(text)
                if path.is_file():
                    try:
                        cached_audio = path.read_bytes()
                    except OSError:
                        cached_audio = b""
                    if _wav_seconds(cached_audio) is not None:
                        continue
                spoken = prepare_speech(text, provider=getattr(self._tts_config, "provider", None)).spoken
                if not spoken:
                    continue
                audio = await self._tts.synthesize_full(spoken, self._tts_config)
                if not audio or _wav_seconds(audio) is None:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = path.with_name(f"{path.name}.{id(self)}.tmp")
                try:
                    temporary_path.write_bytes(audio)
                    temporary_path.replace(path)
                finally:
                    temporary_path.unlink(missing_ok=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Voice filler synthesis/cache failed (%s)", self._anima_name, exc_info=True)

    async def _maybe_send_voice_filler(self) -> None:
        """Send one already-cached phone filler, never waiting for synthesis."""
        if self._channel != "phone" or not self._prefers_whole_reply or not self._voice_filler_phrases:
            return

        available: list[tuple[str, bytes, float]] = []
        try:
            for text in self._voice_filler_phrases:
                try:
                    audio = self._voice_filler_cache_path(text).read_bytes()
                except OSError:
                    continue
                seconds = _wav_seconds(audio)
                if seconds is not None:
                    available.append((text, audio, seconds))
        except Exception:
            logger.debug("Voice filler cache lookup failed (%s)", self._anima_name, exc_info=True)
            return
        if not available:
            return

        text, audio, seconds = random.choice(available)
        if self._active_turn is not None:
            self._active_turn.filler = text
        self._recent_tts_text.append(text)
        try:
            await self._transport.send_audio(audio)
        except Exception:
            logger.debug("Voice filler send failed (%s)", self._anima_name, exc_info=True)
            return
        self._note_playback(seconds or len(text) / 6.0)
        logger.info("voice_filler anima=%s text=%r cached=true", self._anima_name, text)

    async def _synthesize_and_send(self, text: str) -> None:
        """Synthesize queued speech text and send its audio to the client."""
        speech_text = prepare_speech(text, provider=getattr(self._tts_config, "provider", None))
        text = speech_text.display
        if not text:
            return
        # Subtitle keeps the original kanji; only the TTS input gets
        # yomi/kana substitutions (kana-heavy text is hard to read).
        spoken = speech_text.spoken
        try:
            # text rides along so the client can show a playback-synced subtitle
            self._recent_tts_text.append(text)
            await self._transport.send_event({"type": "tts_start", "text": text})
            secs = 0.0
            async for audio_chunk in self._tts.synthesize(spoken, self._tts_config):
                if self._interrupted:
                    break
                self._note_first_audio()
                await self._transport.send_audio(audio_chunk)
                secs += _wav_seconds(audio_chunk) or 0.0
            # ponytail: non-WAV (mp3 stream) falls back to ~6 chars/sec
            self._note_playback(secs or len(text) / 6.0)
            await self._transport.send_event({"type": "tts_done"})
            self._consecutive_tts_failures = 0
        except TTSSynthesisError as e:
            self._consecutive_tts_failures += 1
            logger.warning("TTS synthesis failed (%d consecutive): %s", self._consecutive_tts_failures, e)
            if self._consecutive_tts_failures >= 3:
                self.invalidate_tts_health()
            try:
                await self._transport.send_event({"type": "tts_error", "message": "TTS synthesis failed"})
                await self._transport.send_event({"type": "tts_done"})
            except Exception:
                logger.debug("Best-effort operation failed", exc_info=True)
        except Exception as e:
            logger.warning("TTS send error: %s", e)
            try:
                await self._transport.send_event({"type": "tts_done"})
            except Exception:
                logger.debug("Best-effort operation failed", exc_info=True)

    async def speak_text(self, text: str) -> bool:
        """Speak fixed text through the shared VoiceSession TTS pipeline, without an LLM turn."""
        if self._closed or not text.strip() or self._tts_playing:
            return False
        if not await self._check_tts_health():
            return False

        self._interrupted = False
        self._tts_playing = True
        try:
            await self._transport.send_event({"type": "response_start"})
            await self._transport.send_event({"type": "response_text", "text": text})
            await self._start_tts_worker()
            await self._enqueue_reply_tts(text)
            await self._finish_tts_and_response_done("neutral")
            return not self._interrupted
        except Exception:
            logger.debug("Fixed voice text delivery failed (%s)", self._anima_name, exc_info=True)
            return False
        finally:
            await self._stop_tts_worker()
            self._tts_playing = False
            self._last_activity = time.monotonic()

    async def greet_and_speak(self) -> None:
        """Greet on connect — generate a fresh greeting every time.

        With the 1h greet cache removed, each connect runs the LLM, so the
        first audio arrives after the runner spawn + generate latency."""
        try:
            result = await self._supervisor.send_request(
                anima_name=self._anima_name,
                method="greet",
                params={},
                timeout=90.0,
            )
        except Exception as e:
            logger.info("Voice greet skipped (%s): %s", self._anima_name, e)
            return
        text = str((result or {}).get("response", "")).strip()
        if not text or self._processing:
            return
        emotion = (result or {}).get("emotion", "neutral")
        tts_ok = await self._check_tts_health()
        self._interrupted = False
        try:
            await self._transport.send_event({"type": "response_start"})
            await self._transport.send_event({"type": "response_text", "text": text})
            await self._transport.send_event({"type": "emotion", "emotion": emotion})
            if tts_ok:
                self._tts_playing = True
                # Use the same whole-reply policy as speech replies for providers
                # with a high fixed per-request startup latency.
                await self._start_tts_worker()
                if not self._interrupted and not self._processing:
                    await self._enqueue_reply_tts(text)
                if not self._interrupted and not self._processing:
                    await self._drain_tts_queue()
            await self._transport.send_event({"type": "response_done", "emotion": emotion})
        except Exception as e:
            logger.debug("Voice greet delivery failed (%s): %s", self._anima_name, e)
        finally:
            await self._stop_tts_worker()
            self._tts_playing = False
            self._last_activity = time.monotonic()
            # Start the idle watcher only after the greet is finished, so a
            # cold greet (potentially ~90s) can never race a proactive turn
            # for the front lane / TTS worker (C2).
            self._ensure_idle_watcher()

    async def handle_interrupt(self) -> None:
        """Handle barge-in: stop TTS, drop queued sentences, prepare for new STT."""
        self._probe_active = False
        self._cancel_probe_timeout()
        self._interrupted = True
        # Reopen the mic immediately — handle_audio_chunk drops input while we
        # are talking, and the turn's own finally may be a beat behind.
        self._tts_playing = False
        self._playback_end_at = 0.0
        self._audio_buffer.clear()
        self._streamer.reset()
        self._clear_tts_queue()

    async def handle_barge_probe(self) -> None:
        """Accept mic audio while TTS plays and request an STT verdict."""
        self._cancel_probe_timeout()
        self._probe_active = True
        self._probe_started = time.monotonic()
        self._probe_followup_pending = False
        self._probe_speech_end_at = None
        self._audio_buffer.clear()
        self._streamer.reset()
        self._probe_task = asyncio.create_task(
            self._probe_timeout(),
            name=f"barge-probe-timeout-{self._anima_name}",
        )

    async def _probe_timeout(self) -> None:
        try:
            await asyncio.sleep(PROBE_TIMEOUT_SEC)
            if not self._probe_active:
                return
            # Streaming partials commit only after two decodes agree, which can
            # take longer than the timeout. Decode what we have rather than
            # discarding real speech as "too short".
            text = ""
            audio_data = bytes(self._audio_buffer)
            if audio_data:
                try:
                    result = await self._stt.transcribe_buffer_async(audio_data)
                    text = str(result.get("raw_text", "") or "").strip()
                except Exception:
                    logger.debug("Probe timeout STT failed (%s)", self._anima_name, exc_info=True)
            if self._probe_active:
                await self._judge_probe(text)
        except asyncio.CancelledError:
            pass

    def _cancel_probe_timeout(self) -> None:
        task = self._probe_task
        self._probe_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _finish_probe(self) -> None:
        """Finalize probe audio at speech_end and process a confirmed turn."""
        audio_data = bytes(self._audio_buffer)
        self._finalizing = True
        task = self._stream_task
        if task is not None and task is not asyncio.current_task():
            try:
                await task
            except Exception:
                logger.debug("Best-effort operation failed", exc_info=True)
        if not self._probe_active:
            self._finalizing = False
            return
        try:
            if self._streamer.has_content():
                loop = asyncio.get_running_loop()
                text = (await loop.run_in_executor(None, self._streamer.finalize)).strip()
            elif audio_data:
                result = await self._stt.transcribe_buffer_async(audio_data)
                text = result.get("raw_text", "").strip()
            else:
                text = ""
        except Exception:
            logger.exception("Probe STT failed (%s)", self._anima_name)
            text = ""
        finally:
            self._finalizing = False
        interrupted = await self._judge_probe(text)
        if interrupted:
            await self.handle_speech_end()

    async def _judge_probe(self, text: str, *, final: bool = True) -> bool:
        """Resolve an active probe exactly once and notify the browser.

        A streaming partial (``final=False``) that is still too short to judge
        leaves the probe open: "うん" may be the start of "うん、ちょっと待って".
        """
        if not self._probe_active:
            return False
        normalized = _normalize_probe_text(text)
        if not final and len(normalized) < PROBE_MIN_CHARS:
            return False
        self._probe_active = False
        self._cancel_probe_timeout()
        probe_text = json.dumps(text[:40], ensure_ascii=False)
        # Faint audio is our own reply leaking back (no echo canceller on the
        # line); a caller talking over us is far louder than the silence floor.
        if (
            self._channel == "phone"
            and self._audio_buffer
            and _normalized_rms_from_pcm16(bytes(self._audio_buffer)) < SILENCE_RMS_THRESHOLD
        ):
            self._audio_buffer.clear()
            self._streamer.reset()
            await self._transport.send_event({"type": "barge_verdict", "interrupt": False})
            logger.info(
                "barge_probe verdict interrupt=false reason=quiet anima=%s channel=%s text=%s",
                self._anima_name,
                self._channel,
                probe_text,
            )
            return False
        if len(normalized) < PROBE_MIN_CHARS:
            self._audio_buffer.clear()
            self._streamer.reset()
            await self._transport.send_event({"type": "barge_verdict", "interrupt": False})
            logger.info(
                "barge_probe verdict interrupt=false reason=too_short anima=%s channel=%s text=%s",
                self._anima_name,
                self._channel,
                probe_text,
            )
            return False
        if _is_self_echo(text, self._recent_tts_text):
            self._audio_buffer.clear()
            self._streamer.reset()
            await self._transport.send_event({"type": "barge_verdict", "interrupt": False})
            logger.info(
                "barge_probe verdict interrupt=false reason=self_echo anima=%s channel=%s text=%s",
                self._anima_name,
                self._channel,
                probe_text,
            )
            return False

        preserved_audio = bytes(self._audio_buffer)
        await self.handle_interrupt()
        self._audio_buffer.extend(preserved_audio)
        self._probe_followup_pending = True
        await self._transport.send_event({"type": "barge_verdict", "interrupt": True})
        logger.info(
            "barge_probe verdict interrupt=true reason=recognized_speech anima=%s channel=%s text=%s",
            self._anima_name,
            self._channel,
            probe_text,
        )
        return True

    async def handle_discard_audio(self) -> None:
        """Drop buffered mic input without touching the current turn.

        Sent when a VAD misfire aborted a recording: the noise must not leak
        into the next utterance, but an in-flight reply must keep streaming
        (that is what ``handle_interrupt`` is for).
        """
        discarded_bytes = len(self._audio_buffer)
        if discarded_bytes:
            logger.info(
                "speech discarded (vad misfire) anima=%s channel=%s bytes=%s",
                self._anima_name,
                self._channel,
                discarded_bytes,
            )
        self._audio_buffer.clear()
        self._streamer.reset()

    async def _send_error(self, message: str) -> None:
        """Send error message to client."""
        try:
            await self._transport.send_event({"type": "error", "message": message})
        except Exception:
            logger.debug("Failed to send error to client", exc_info=True)
