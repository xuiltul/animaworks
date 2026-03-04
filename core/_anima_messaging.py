from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""MessagingMixin -- human chat processing (blocking/streaming), bootstrap, greeting.

Extracted from ``core.anima.DigitalAnima`` as a Mixin.  All ``self``
references are resolved at runtime via MRO when mixed into ``DigitalAnima``.
"""

import json
import logging
import re
import time
from collections.abc import AsyncGenerator
from datetime import date
from typing import Any

from core.time_utils import now_jst

from core.execution._sanitize import ORIGIN_HUMAN, ORIGIN_SYSTEM
from core.memory.conversation import ConversationMemory, ToolRecord
from core.memory.streaming_journal import StreamingJournal
from core.paths import load_prompt
from core.image_artifacts import extract_image_artifacts_from_tool_records
from core.i18n import t
from core.exceptions import (
    ExecutionError,
    LLMAPIError,
    ToolError,
    MemoryIOError,
)
from core.schemas import CycleResult, VALID_EMOTIONS

logger = logging.getLogger("animaworks.anima")


class MessagingMixin:
    """Mixin: human chat processing, bootstrap, and greeting."""

    def _log_human_conversation(
        self,
        content: str,
        from_person: str,
        thread_id: str = "default",
    ) -> None:
        """Append human message to shared conversation log.

        Writes to ``shared/users/{from_person}/conversations/YYYY-MM-DD.jsonl``
        so that any Anima can search what the human discussed across all Animas.
        """
        if from_person in ("system", "") or from_person == self.name:
            return
        try:
            shared_dir = (
                self.anima_dir.parent.parent
                / "shared" / "users" / from_person / "conversations"
            )
            shared_dir.mkdir(parents=True, exist_ok=True)

            today = date.today().isoformat()
            log_file = shared_dir / f"{today}.jsonl"

            record = {
                "ts": now_jst().isoformat(),
                "anima": self.name,
                "content": content,
                "thread_id": thread_id,
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning(
                "[%s] Failed to log human conversation", self.name, exc_info=True,
            )

    async def run_bootstrap(self) -> CycleResult:
        """Run the first-time bootstrap process in the background.

        Acquires the anima lock, sets status to ``"bootstrapping"``, and
        triggers an agent cycle with the bootstrap prompt.  The agent reads
        ``bootstrap.md`` from the anima directory and follows its
        instructions (identity setup, avatar generation, self-introduction,
        etc.).  Upon completion the agent deletes ``bootstrap.md``.
        """
        if not self.needs_bootstrap:
            logger.info("[%s] run_bootstrap SKIPPED: no bootstrap.md", self.name)
            return CycleResult(
                trigger="bootstrap",
                action="skipped",
                summary="Bootstrap not needed",
            )

        logger.info("[%s] run_bootstrap START", self.name)
        from core.tooling.handler import active_session_type
        try:
            async with self._get_thread_lock("default"):
                self._status_slots["conversation:default"] = "bootstrapping"
                self._task_slots["conversation:default"] = "Initial bootstrap"
                _session_token = self.agent._tool_handler.set_active_session_type("chat")
                self.agent._tool_handler.set_session_origin(ORIGIN_SYSTEM)

                conv_memory = ConversationMemory(self.anima_dir, self.model_config)
                prompt = conv_memory.build_chat_prompt(
                    t("anima.bootstrap_prompt"),
                    "system",
                )

                try:
                    result = await self.agent.run_cycle(
                        prompt, trigger="bootstrap"
                    )
                    self._last_activity = now_jst()

                    logger.info(
                        "[%s] run_bootstrap END duration_ms=%d",
                        self.name, result.duration_ms,
                    )
                    return result
                except Exception:
                    logger.exception("[%s] run_bootstrap FAILED", self.name)
                    raise
                finally:
                    active_session_type.reset(_session_token)
                    self._status_slots["conversation:default"] = "idle"
                    self._task_slots["conversation:default"] = ""
        finally:
            self._notify_lock_released()

    async def process_message(
        self,
        content: str,
        from_person: str = "human",
        images: list[dict[str, Any]] | None = None,
        attachment_paths: list[str] | None = None,
        intent: str = "",
        thread_id: str = "default",
        include_cycle_result: bool = False,
    ) -> str | dict[str, Any]:
        self._validate_thread_id(thread_id)
        # Auto-interrupt: if a session is already running on this thread,
        # signal it to wrap up so the new message can be processed.
        lock = self._get_thread_lock(thread_id)
        if lock.locked():
            evt = self._interrupt_events.get(thread_id)
            if evt:
                evt.set()
                logger.info(
                    "[%s] Auto-interrupting running session for new message from=%s",
                    self.name, from_person,
                )
        logger.info(
            "[%s] process_message WAITING from=%s content_len=%d images=%d",
            self.name, from_person, len(content), len(images or []),
        )
        from core.tooling.handler import active_session_type
        try:
            async with lock:
                # Clear interrupt event for OUR session (after lock acquired)
                self._get_interrupt_event(thread_id).clear()
                self.agent.set_interrupt_event(self._get_interrupt_event(thread_id))
                logger.info(
                    "[%s] process_message START (lock acquired) from=%s",
                    self.name, from_person,
                )
                _conv_key = f"conversation:{thread_id}"
                self._status_slots[_conv_key] = "thinking"
                self._task_slots[_conv_key] = f"Responding to {from_person}"
                _session_token = self.agent._tool_handler.set_active_session_type("chat")
                self.agent._tool_handler.set_session_origin(ORIGIN_HUMAN)

                # Build history-aware prompt via conversation memory
                conv_memory = ConversationMemory(self.anima_dir, self.model_config, thread_id=thread_id)
                await conv_memory.compress_if_needed()

                # Determine prompt and history strategy per execution mode
                mode = self.agent.execution_mode
                prior_messages = None
                if mode == "s":
                    prompt = content
                elif mode == "a":
                    prior_messages = conv_memory.build_structured_messages(content)
                    prompt = content
                elif mode == "b":
                    prompt = conv_memory.build_chat_prompt(
                        content, from_person, max_history_chars=2000,
                    )
                else:
                    prompt = conv_memory.build_chat_prompt(content, from_person)

                # Pre-save: persist user input before agent execution
                conv_memory.append_turn(
                    "human", content, attachments=attachment_paths or [],
                )
                conv_memory.save()

                # Transcript: record human message
                conv_memory.write_transcript(
                    "human", content,
                    from_person=from_person,
                    thread_id=thread_id,
                    attachments=attachment_paths or None,
                )

                # Shared conversation log: record human message
                self._log_human_conversation(content, from_person, thread_id)

                # Activity log: message received
                self._activity.log(
                    "message_received",
                    content=content,
                    summary=content[:100],
                    from_person=from_person,
                    channel="chat",
                    meta={"from_type": "human", "thread_id": thread_id},
                    origin=ORIGIN_HUMAN,
                )

                try:
                    result = await self.agent.run_cycle(
                        prompt, trigger=f"message:{from_person}",
                        message_intent=intent,
                        images=images,
                        prior_messages=prior_messages,
                        thread_id=thread_id,
                    )
                    self._last_activity = now_jst()

                    # Record assistant response with tool records
                    tool_records = [
                        ToolRecord.from_dict(r)
                        for r in result.tool_call_records
                    ]
                    conv_memory.append_turn(
                        "assistant", result.summary,
                        tool_records=tool_records,
                    )
                    conv_memory.save()

                    # Transcript: record assistant response
                    conv_memory.write_transcript(
                        "assistant", result.summary,
                        thread_id=thread_id,
                        tool_names=[r.tool_name for r in tool_records if r.tool_name] or None,
                    )

                    # Activity log: response sent (with thinking text if present)
                    response_artifacts = extract_image_artifacts_from_tool_records(
                        result.tool_call_records
                    )
                    resp_meta: dict[str, Any] = {"thread_id": thread_id}
                    if result.thinking_text:
                        resp_meta["thinking_text"] = result.thinking_text
                    if response_artifacts:
                        resp_meta["images"] = response_artifacts
                    self._activity.log("response_sent", content=result.summary, to_person=from_person, channel="chat", meta=resp_meta)

                    logger.info(
                        "[%s] process_message END duration_ms=%d",
                        self.name, result.duration_ms,
                    )
                    result.images = response_artifacts
                    if include_cycle_result:
                        return result.model_dump(mode="json")
                    return result.summary
                except Exception as exc:
                    logger.exception("[%s] process_message FAILED", self.name)
                    # Activity log: error
                    self._activity.log(
                        "error",
                        summary=t("anima.process_message_error", exc=type(exc).__name__),
                        meta={"phase": "process_message", "error": str(exc)[:200], "thread_id": thread_id},
                    )
                    # Save error marker so the failed exchange is visible
                    conv_memory.append_turn(
                        "assistant", t("anima.agent_error")
                    )
                    conv_memory.save()
                    raise
                finally:
                    active_session_type.reset(_session_token)
                    self._status_slots[_conv_key] = "idle"
                    self._task_slots[_conv_key] = ""
        finally:
            self._notify_lock_released()

    async def process_message_stream(
        self,
        content: str,
        from_person: str = "human",
        images: list[dict[str, Any]] | None = None,
        attachment_paths: list[str] | None = None,
        intent: str = "",
        thread_id: str = "default",
    ) -> AsyncGenerator[dict, None]:
        """Streaming version of process_message.

        Yields stream event dicts. The lock is held for the entire duration.
        If bootstrapping is in progress (lock held + needs_bootstrap), yields
        an immediate "initializing" message instead of waiting.
        """
        self._validate_thread_id(thread_id)
        lock = self._get_thread_lock(thread_id)

        # ── Bootstrap guard: return immediately if bootstrap is running ──
        if self.needs_bootstrap and lock.locked():
            logger.info(
                "[%s] process_message_stream REJECTED (bootstrapping) from=%s",
                self.name, from_person,
            )
            yield {
                "type": "bootstrap_busy",
                "message": t("anima.initializing"),
            }
            return

        # Auto-interrupt: if a session is already running on this thread,
        # signal it to wrap up so the new message can be processed.
        if lock.locked():
            evt = self._interrupt_events.get(thread_id)
            if evt:
                evt.set()
                logger.info(
                    "[%s] Auto-interrupting running session for new message from=%s",
                    self.name, from_person,
                )

        logger.info(
            "[%s] process_message_stream WAITING from=%s content_len=%d images=%d",
            self.name, from_person, len(content), len(images or []),
        )
        from core.tooling.handler import active_session_type
        try:
            async with lock:
                # Clear interrupt event for OUR session (after lock acquired)
                self._get_interrupt_event(thread_id).clear()
                self.agent.set_interrupt_event(self._get_interrupt_event(thread_id))
                logger.info(
                    "[%s] process_message_stream START (lock acquired) from=%s",
                    self.name, from_person,
                )
                _conv_key = f"conversation:{thread_id}"
                self._status_slots[_conv_key] = "thinking"
                self._task_slots[_conv_key] = f"Responding to {from_person}"
                _session_token = self.agent._tool_handler.set_active_session_type("chat")
                self.agent._tool_handler.set_session_origin(ORIGIN_HUMAN)

                # Build history-aware prompt via conversation memory
                conv_memory = ConversationMemory(self.anima_dir, self.model_config, thread_id=thread_id)
                if conv_memory.needs_compression():
                    yield {"type": "compression_start"}
                    await conv_memory.compress_if_needed()
                    yield {"type": "compression_end"}

                # Determine prompt and history strategy per execution mode
                mode = self.agent.execution_mode
                prior_messages = None
                if mode == "s":
                    prompt = content
                elif mode == "a":
                    prior_messages = conv_memory.build_structured_messages(content)
                    prompt = content
                elif mode == "b":
                    prompt = conv_memory.build_chat_prompt(
                        content, from_person, max_history_chars=2000,
                    )
                else:
                    prompt = conv_memory.build_chat_prompt(content, from_person)

                # Pre-save: persist user input before agent execution
                conv_memory.append_turn(
                    "human", content, attachments=attachment_paths or [],
                )
                conv_memory.save()

                # Transcript: record human message
                conv_memory.write_transcript(
                    "human", content,
                    from_person=from_person,
                    thread_id=thread_id,
                    attachments=attachment_paths or None,
                )

                # Shared conversation log: record human message
                self._log_human_conversation(content, from_person, thread_id)

                # Activity log: message received
                self._activity.log(
                    "message_received",
                    content=content,
                    summary=content[:100],
                    from_person=from_person,
                    channel="chat",
                    meta={"from_type": "human", "thread_id": thread_id},
                    origin=ORIGIN_HUMAN,
                )

                # Streaming journal: write-ahead log for crash recovery
                journal = StreamingJournal(self.anima_dir, thread_id=thread_id)
                journal.open(
                    trigger=f"message:{from_person}",
                    from_person=from_person,
                )

                partial_response = ""
                cycle_done = False

                try:
                    async for chunk in self.agent.run_cycle_streaming(
                        prompt, trigger=f"message:{from_person}",
                        message_intent=intent,
                        images=images,
                        prior_messages=prior_messages,
                        thread_id=thread_id,
                    ):
                        if chunk.get("type") == "text_delta":
                            delta_text = chunk.get("text", "")
                            partial_response += delta_text
                            journal.write_text(delta_text)

                        if chunk.get("type") == "tool_start":
                            journal.write_tool_start(
                                tool=chunk.get("tool_name", ""),
                                args_summary="",
                            )
                        if chunk.get("type") == "tool_end":
                            journal.write_tool_end(
                                tool=chunk.get("tool_name", ""),
                                result_summary="",
                            )

                        if chunk.get("type") == "cycle_done":
                            cycle_done = True
                            self._last_activity = now_jst()
                            # Record assistant response with tool records
                            cycle_result = chunk.get("cycle_result", {})
                            summary = cycle_result.get("summary", "")
                            response_artifacts = extract_image_artifacts_from_tool_records(
                                cycle_result.get("tool_call_records", [])
                            )
                            if response_artifacts:
                                cycle_result["images"] = response_artifacts
                            tool_records = [
                                ToolRecord.from_dict(r)
                                for r in cycle_result.get("tool_call_records", [])
                            ]
                            conv_memory.append_turn(
                                "assistant", summary,
                                tool_records=tool_records,
                            )
                            conv_memory.save()

                            # Transcript: record assistant response
                            conv_memory.write_transcript(
                                "assistant", summary,
                                thread_id=thread_id,
                                tool_names=[r.tool_name for r in tool_records if r.tool_name] or None,
                            )

                            # Activity log: response sent (with thinking text if present)
                            thinking_text = cycle_result.get("thinking_text", "")
                            resp_meta: dict[str, Any] = {"thread_id": thread_id}
                            if thinking_text:
                                resp_meta["thinking_text"] = thinking_text
                            if response_artifacts:
                                resp_meta["images"] = response_artifacts
                            self._activity.log("response_sent", content=summary, to_person=from_person, channel="chat", meta=resp_meta)

                            # Finalize streaming journal (deletes the file)
                            journal.finalize(summary=summary[:500])

                            # Yield pending notification events before cycle_done
                            for notif in self.agent.drain_notifications():
                                yield {"type": "notification_sent", "data": notif}

                            logger.info(
                                "[%s] process_message_stream END",
                                self.name,
                            )
                        yield chunk
                except Exception as exc:
                    logger.exception("[%s] process_message_stream FAILED", self.name)
                    if isinstance(exc, ToolError):
                        error_code = "TOOL_ERROR"
                    elif isinstance(exc, (LLMAPIError, ExecutionError)):
                        error_code = "LLM_ERROR"
                    elif isinstance(exc, MemoryIOError):
                        error_code = "MEMORY_ERROR"
                    else:
                        error_code = "STREAM_ERROR"
                    # Activity log: error
                    self._activity.log(
                        "error",
                        summary=t("anima.process_stream_error", exc=type(exc).__name__),
                        meta={"phase": "process_message_stream", "error_code": error_code, "error": str(exc)[:200], "thread_id": thread_id},
                    )
                    yield {
                        "type": "error",
                        "code": error_code,
                        "message": "Internal error",
                    }
                finally:
                    if not cycle_done:
                        logger.warning(
                            "[%s] process_message_stream END (cycle_done not received)",
                            self.name,
                        )
                    # Save partial response if cycle_done was never received
                    if not cycle_done:
                        if partial_response:
                            saved_text = partial_response + t("anima.response_interrupted_prefix")
                        else:
                            saved_text = t("anima.response_interrupted")
                        conv_memory.append_turn("assistant", saved_text)
                        conv_memory.save()
                    # Close journal (no-op if already finalized)
                    journal.close()
                    try:
                        active_session_type.reset(_session_token)
                    except ValueError:
                        pass
                    self._status_slots[_conv_key] = "idle"
                    self._task_slots[_conv_key] = ""
        finally:
            self._notify_lock_released()

    async def process_greet(self) -> dict[str, str | bool]:
        """Generate a greeting response when user clicks the character.

        Returns a cached response if called within the cooldown period.

        Returns:
            Dict with keys: response, emotion, cached.
        """
        # Check cooldown
        now = time.time()
        if (
            self._last_greet_at is not None
            and (now - self._last_greet_at) < self._GREET_COOLDOWN
            and self._last_greet_text is not None
        ):
            logger.info(
                "[%s] process_greet CACHED (%.0fs since last)",
                self.name, now - self._last_greet_at,
            )
            return {
                "response": self._last_greet_text,
                "emotion": self._last_greet_emotion,
                "cached": True,
            }

        logger.info("[%s] process_greet START", self.name)
        from core.tooling.handler import active_session_type
        async with self._get_thread_lock("default"):
            prev_status = self._status_slots.get("conversation:default", "idle")
            prev_task = self._task_slots.get("conversation:default", "")
            _session_token = self.agent._tool_handler.set_active_session_type("chat")

            # Build greet prompt with current state (use primary to include background)
            status_text = self.primary_status if self.primary_status != "idle" else t("anima.status_idle")
            task_text = self.primary_task if self.primary_task else t("anima.task_none")
            prompt = load_prompt(
                "greet", status=status_text, current_task=task_text,
            )

            self._status_slots["conversation:default"] = "greeting"
            self._task_slots["conversation:default"] = "Greeting user"

            conv_memory = ConversationMemory(self.anima_dir, self.model_config)

            # Record visit marker (user turn) before greeting
            visit_text = t("anima.visit_desk")
            conv_memory.append_turn("system", visit_text)
            conv_memory.save()

            try:
                result = await self.agent.run_cycle(
                    prompt, trigger="greet:user",
                )
                self._last_activity = now_jst()

                # Extract emotion from response
                _em_pat = re.compile(r'<!--\s*emotion:\s*(\{.*?\})\s*-->', re.DOTALL)
                _em_match = _em_pat.search(result.summary)
                if _em_match:
                    clean_text = _em_pat.sub("", result.summary).rstrip()
                    try:
                        _meta = json.loads(_em_match.group(1))
                        emotion = _meta.get("emotion", "neutral")
                        if emotion not in VALID_EMOTIONS:
                            emotion = "neutral"
                    except (json.JSONDecodeError, AttributeError):
                        emotion = "neutral"
                else:
                    clean_text = result.summary
                    emotion = "neutral"

                # Record assistant turn in conversation memory
                conv_memory.append_turn("assistant", clean_text)
                conv_memory.save()

                # Update greet cache
                self._last_greet_at = time.time()
                self._last_greet_text = clean_text
                self._last_greet_emotion = emotion

                logger.info(
                    "[%s] process_greet END duration_ms=%d",
                    self.name, result.duration_ms,
                )
                return {
                    "response": clean_text,
                    "emotion": emotion,
                    "cached": False,
                }
            except Exception:
                logger.exception("[%s] process_greet FAILED", self.name)
                # Save error marker in conversation memory
                conv_memory.append_turn(
                    "assistant", t("anima.greeting_error")
                )
                conv_memory.save()
                raise
            finally:
                active_session_type.reset(_session_token)
                self._status_slots["conversation:default"] = prev_status
                self._task_slots["conversation:default"] = prev_task
