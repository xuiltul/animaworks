# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Voice session — STT -> Chat -> TTS orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from core.i18n import t
from core.voice.sentence_splitter import StreamingSentenceSplitter
from core.voice.stt import VoiceSTT
from core.voice.tts_base import BaseTTSProvider, TTSConfig, TTSSynthesisError

logger = logging.getLogger(__name__)

IPC_STREAM_TIMEOUT = 60.0
MAX_AUDIO_BUFFER_BYTES = 60 * 16_000 * 2  # 60 seconds of 16kHz 16-bit mono PCM
PCM16_SAMPLE_RATE = 16_000
PCM16_BYTES_PER_SAMPLE = 2
MIN_SPEECH_SEC = 0.35
MIN_SPEECH_BYTES = int(MIN_SPEECH_SEC * PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
SILENCE_RMS_THRESHOLD = 0.008

VOICE_MODE_SUFFIX = (
    "\n\n[voice-mode: 音声会話です。話し言葉で200文字以内で簡潔に回答してください。"
    "Markdown記法（見出し・太字・リスト・コードブロック等）は使わないでください]"
)

# ── TTS output sanitization ──────────────────────────────────

_RE_HTML_COMMENT = re.compile(r"<!--[\s\S]*?-->")
_RE_MD_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_RE_MD_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_MD_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_RE_MD_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_RE_MD_LIST_BULLET = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)
_RE_MD_LIST_NUMBERED = re.compile(r"^[\s]*\d+\.\s+", re.MULTILINE)
_RE_MD_TABLE_PIPE = re.compile(r"\|")
_RE_MD_HR = re.compile(r"^-{3,}$", re.MULTILINE)


def sanitize_for_tts(text: str) -> str:
    """Strip Markdown formatting and HTML comments for TTS consumption."""
    text = _RE_HTML_COMMENT.sub("", text)
    text = _RE_MD_CODE_BLOCK.sub("", text)
    text = _RE_MD_HEADING.sub("", text)
    text = _RE_MD_BOLD.sub(r"\1", text)
    text = _RE_MD_ITALIC.sub(r"\1", text)
    text = _RE_MD_INLINE_CODE.sub(r"\1", text)
    text = _RE_MD_LINK.sub(r"\1", text)
    text = _RE_MD_LIST_BULLET.sub("", text)
    text = _RE_MD_LIST_NUMBERED.sub("", text)
    text = _RE_MD_TABLE_PIPE.sub("", text)
    text = _RE_MD_HR.sub("", text)
    return text.strip()


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


# ── VoiceSession ────────────────────────────────────────────────


class VoiceSession:
    """Manages a single voice conversation session with an Anima."""

    def __init__(
        self,
        anima_name: str,
        ws: Any,
        stt: VoiceSTT,
        tts: BaseTTSProvider,
        tts_config: TTSConfig,
        supervisor: Any,
        voice_config: Any,
    ) -> None:
        """Initialize voice session.

        Args:
            anima_name: Target Anima name.
            ws: WebSocket (send_json, send_bytes).
            stt: STT engine.
            tts: TTS provider.
            tts_config: Per-session TTS config.
            supervisor: ProcessSupervisor for IPC.
            voice_config: Voice configuration (stt_refine_enabled, etc.).
        """
        self._anima_name = anima_name
        self._ws = ws
        self._stt = stt
        self._tts = tts
        self._tts_config = tts_config
        self._supervisor = supervisor
        self._voice_config = voice_config
        self._audio_buffer: bytearray = bytearray()
        self._tts_playing = False
        self._interrupted = False
        self._processing = False
        self._tts_available: bool | None = None
        self._splitter = StreamingSentenceSplitter()
        self._consecutive_tts_failures: int = 0

    async def handle_audio_chunk(self, data: bytes) -> None:
        """Receive audio chunk from browser, accumulate in buffer."""
        if len(self._audio_buffer) + len(data) > MAX_AUDIO_BUFFER_BYTES:
            self._audio_buffer.clear()
            logger.warning("Audio buffer overflow (%s), cleared", self._anima_name)
        self._audio_buffer.extend(data)

    async def handle_speech_end(self, from_person: str = "human") -> None:
        """Process accumulated audio: STT -> optional refine -> Chat -> TTS."""
        if self._processing:
            logger.debug("speech_end ignored — already processing (%s)", self._anima_name)
            return
        self._processing = True
        try:
            await self._do_speech_end(from_person)
        finally:
            self._processing = False

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
            await self._ws.send_json({"type": "error", "message": "TTS unavailable"})
        return ok

    def invalidate_tts_health(self) -> None:
        """Reset cached TTS health so next speech_end rechecks."""
        self._tts_available = None

    async def _do_speech_end(self, from_person: str) -> None:
        """Inner speech_end logic, guarded by _processing flag."""
        audio_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        if not audio_data:
            return
        if len(audio_data) < MIN_SPEECH_BYTES:
            logger.debug("Ignore short voice chunk: bytes=%s", len(audio_data))
            return
        rms = _normalized_rms_from_pcm16(audio_data)
        if rms < SILENCE_RMS_THRESHOLD:
            logger.debug(
                "Ignore likely silence: rms=%.5f bytes=%s", rms, len(audio_data)
            )
            return

        # 1. STT
        try:
            result = await self._stt.transcribe_buffer_async(audio_data)
        except Exception as e:
            logger.exception("STT failed: %s", e)
            await self._send_error(t("voice.stt_failed"))
            return

        text = result.get("raw_text", "").strip()
        if not text:
            return

        # 2. Optional LLM refine
        if getattr(self._voice_config, "stt_refine_enabled", False):
            try:
                from core.tools.transcribe import refine_with_llm

                loop = asyncio.get_event_loop()
                refined = await loop.run_in_executor(
                    None,
                    lambda: refine_with_llm(
                        text,
                        language=result.get("language", "ja") or "ja",
                    ),
                )
                text = refined.get("refined_text", text)
            except Exception as e:
                logger.warning("STT refine failed, using raw: %s", e)

        # 3. Send transcript to client
        await self._ws.send_json({"type": "transcript", "text": text})

        # 4. Check TTS health before entering IPC loop
        tts_ok = await self._check_tts_health()

        # 5. Send to Anima via IPC (streaming)
        await self._ws.send_json({"type": "response_start"})
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
        try:
            async for ipc_response in self._supervisor.send_request_stream(
                anima_name=self._anima_name,
                method="process_message",
                params={
                    "message": text + VOICE_MODE_SUFFIX,
                    "from_person": from_person,
                    "intent": "",
                    "stream": True,
                    "images": [],
                    "attachment_paths": [],
                },
                timeout=timeout,
            ):
                if self._interrupted:
                    break

                if ipc_response.done:
                    result_data = ipc_response.result or {}
                    cycle_result = result_data.get("cycle_result", {})
                    emotion = cycle_result.get("emotion", "neutral")
                    remaining = self._splitter.flush()
                    if remaining and tts_ok:
                        await self._synthesize_and_send(remaining)
                    await self._ws.send_json({
                        "type": "response_done",
                        "emotion": emotion,
                    })
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
                            await self._ws.send_json({
                                "type": "response_text",
                                "text": delta,
                                "done": False,
                            })
                            if tts_ok:
                                sentences = self._splitter.feed(delta)
                                for sentence in sentences:
                                    if self._interrupted:
                                        break
                                    await self._synthesize_and_send(sentence)

                    elif chunk_data.get("type") == "thinking_start":
                        await self._ws.send_json({"type": "thinking_status", "thinking": True})
                    elif chunk_data.get("type") == "thinking_end":
                        await self._ws.send_json({"type": "thinking_status", "thinking": False})
                    elif chunk_data.get("type") == "thinking_delta":
                        delta = chunk_data.get("text", "")
                        if delta:
                            await self._ws.send_json({
                                "type": "thinking_delta",
                                "text": delta,
                            })

                    elif chunk_data.get("type") == "cycle_done":
                        cycle_result = chunk_data.get("cycle_result", {})
                        emotion = cycle_result.get("emotion", "neutral")
                        remaining = self._splitter.flush()
                        if remaining and tts_ok:
                            await self._synthesize_and_send(remaining)
                        await self._ws.send_json({
                            "type": "response_done",
                            "emotion": emotion,
                        })
                        response_done_sent = True
                        break

        except Exception as e:
            logger.exception("Voice session IPC error: %s", e)
            await self._send_error(str(e))
        finally:
            if not response_done_sent:
                try:
                    await self._ws.send_json({
                        "type": "response_done",
                        "emotion": "neutral",
                    })
                except Exception:
                    pass
            self._tts_playing = False
            self._interrupted = False
            self._splitter.flush()

    async def _synthesize_and_send(self, text: str) -> None:
        """TTS synthesize a sentence and send audio to client."""
        text = sanitize_for_tts(text)
        if not text:
            return
        try:
            await self._ws.send_json({"type": "tts_start"})
            async for audio_chunk in self._tts.synthesize(text, self._tts_config):
                if self._interrupted:
                    break
                await self._ws.send_bytes(audio_chunk)
            await self._ws.send_json({"type": "tts_done"})
            self._consecutive_tts_failures = 0
        except TTSSynthesisError as e:
            self._consecutive_tts_failures += 1
            logger.warning("TTS synthesis failed (%d consecutive): %s", self._consecutive_tts_failures, e)
            if self._consecutive_tts_failures >= 3:
                self.invalidate_tts_health()
            try:
                await self._ws.send_json({"type": "tts_error", "message": "TTS synthesis failed"})
                await self._ws.send_json({"type": "tts_done"})
            except Exception:
                pass
        except Exception as e:
            logger.warning("TTS send error: %s", e)
            try:
                await self._ws.send_json({"type": "tts_done"})
            except Exception:
                pass

    async def handle_interrupt(self) -> None:
        """Handle barge-in: stop TTS, prepare for new STT."""
        self._interrupted = True
        self._audio_buffer.clear()

    async def _send_error(self, message: str) -> None:
        """Send error message to client."""
        try:
            await self._ws.send_json({"type": "error", "message": message})
        except Exception:
            logger.debug("Failed to send error to client", exc_info=True)
