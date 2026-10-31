from __future__ import annotations

from core.anima._mixin_protocols import _MessagingHost

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""MessagingMixin -- human chat processing (blocking/streaming), bootstrap, greeting.

Extracted from ``core.anima.digital_anima.DigitalAnima`` as a Mixin.  All ``self``
references are resolved at runtime via MRO when mixed into ``DigitalAnima``.
"""

import asyncio
import contextvars
import inspect
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import nullcontext
from typing import Any

from core.anima.chat_attachments import collect_outbound_files
from core.anima.emotion_tag import extract_emotion as _extract_emotion_from_tag
from core.anima.image_artifacts import extract_image_artifacts_from_tool_records, resolve_local_image_paths
from core.anima.response_normalize import normalize_user_facing_response_text
from core.config.model_config import effective_model_key, resolve_effective_model_config, same_effective_model
from core.exceptions import (
    ExecutionError,
    LLMAPIError,
    MemoryIOError,
    ToolError,
)
from core.execution.fallback_activity import (
    log_model_fallback,
    report_capacity_block,
)
from core.execution.session.session_types import resolve_runtime_session_type
from core.i18n import t
from core.llm.guard.error_classifier import (
    FailoverReason,
    classify_llm_error,
    classify_llm_error_message,
)
from core.memory.conversation.memory import ConversationMemory, ToolRecord
from core.memory.conversation.streaming_journal import StreamingJournal
from core.paths import load_prompt
from core.platform.tasks import spawn
from core.schemas import EXTERNAL_PLATFORM_SOURCES, CycleResult, ImageData, ModelConfig
from core.time_utils import now_local, today_local
from core.trust import ORIGIN_HUMAN, ORIGIN_SYSTEM

logger = logging.getLogger("animaworks.anima")


def _queue_live_fact_extraction(owner: Any, trigger: str, session_started_at: Any) -> None:
    scheduler = getattr(owner, "_schedule_live_fact_extraction", None)
    if not callable(scheduler):
        return
    try:
        scheduler(trigger, session_started_at=session_started_at)
    except Exception as exc:  # noqa: BLE001 - extraction must not affect a completed response
        reason = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        logger.warning("[%s] Live fact extraction hook failed: %s", getattr(owner, "name", "unknown"), reason[:240])


async def _persist_conversation(conversation: Any) -> None:
    """Await StateWriter persistence while supporting synchronous test doubles."""
    async_save = getattr(conversation, "asave", None)
    result = async_save() if callable(async_save) else None
    if inspect.isawaitable(result):
        await result
        return
    sync_save = getattr(conversation, "save", None)
    if callable(sync_save):
        await asyncio.to_thread(sync_save)


async def _write_conversation_transcript(conversation: Any, *args: Any, **kwargs: Any) -> None:
    async_write = getattr(conversation, "awrite_transcript", None)
    result = async_write(*args, **kwargs) if callable(async_write) else None
    if inspect.isawaitable(result):
        await result
        return
    sync_write = getattr(conversation, "write_transcript", None)
    if callable(sync_write):
        await asyncio.to_thread(sync_write, *args, **kwargs)


def _record_chat_user_turn(owner: Any) -> None:
    try:
        from core.memory.maintenance.background_review import request_background_review

        request_background_review(owner.anima_dir, "", user_turn=True)
    except Exception:
        logger.warning("Could not queue chat background review for %s", getattr(owner, "name", "unknown"))


def _chat_received_meta(thread_id: str, files: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build message_received metadata, retaining persisted file references."""
    meta: dict[str, Any] = {"from_type": "human", "thread_id": thread_id}
    if files:
        meta["files"] = files
    return meta


def _chat_fallback_reason_from_exception(exc: Exception) -> FailoverReason | None:
    """Return a chat-retry reason for a classified provider exception."""
    reason, hint = classify_llm_error(exc)
    return reason if hint.fallback_ok else None


def _chat_fallback_reason_from_result(result: CycleResult | dict[str, Any]) -> FailoverReason | None:
    """Return a chat-retry reason from a terminal cycle result, if any."""
    data = result.model_dump(mode="json") if isinstance(result, CycleResult) else result
    raw_reason = data.get("reason")
    if isinstance(raw_reason, str):
        try:
            reason = FailoverReason(raw_reason)
        except ValueError:
            reason = None
        if reason is not None:
            _classified, hint = classify_llm_error_message(
                f"{reason.value.replace('_', ' ')} {data.get('summary') or ''}"
            )
            return reason if hint.fallback_ok else None
    # Mode S can surface quota/overload failures as a normal assistant result.
    # Always inspect the final text; the classifier only opts in known,
    # fallback-safe provider failures.
    reason, hint = classify_llm_error_message(str(data.get("summary") or ""))
    if reason is FailoverReason.UNKNOWN and data.get("action") != "error":
        return None
    return reason if hint.fallback_ok else None


def _resolve_chat_model_config(
    owner: Any,
    primary_config: Any,
    *,
    phase: str,
) -> Any:
    """Resolve and record the effective model for one chat attempt."""
    if not isinstance(primary_config, ModelConfig):
        # Some embedding/tests provide a lightweight config double.  Runtime
        # configuration always resolves to ModelConfig.
        return primary_config
    effective_config = resolve_effective_model_config(primary_config)
    log_model_fallback(
        owner._activity,
        primary_config,
        effective_config,
        channel="chat",
        phase=phase,
    )
    return effective_config


def _resolve_voice_model_config(model_config: Any, voice_mode: bool) -> Any:
    """Override thinking effort for this voice message only."""
    if not voice_mode or not isinstance(model_config, ModelConfig):
        return model_config
    effort = model_config.voice_thinking_effort or "low"
    logger.info(
        "Voice mode: thinking_effort %s -> %s (model=%s)",
        model_config.thinking_effort,
        effort,
        model_config.model,
    )
    return model_config.model_copy(update={"thinking_effort": effort})


def _build_chat_background_notification_context(owner: Any) -> str:
    """Drain completed notices synchronously for local compatibility tests."""
    notifications = owner.drain_chat_background_notifications()
    if not notifications:
        return ""
    return load_prompt("fragments/bg_task_notification") + "\n\n" + "\n\n".join(notifications)


async def _build_chat_background_notification_context_async(owner: Any) -> str:
    """Drain completed task notices through the process state writer."""
    async_drain = getattr(owner, "adrain_chat_background_notifications", None)
    if callable(async_drain):
        notifications = await async_drain()
    else:
        notifications = owner.drain_chat_background_notifications()
    if not notifications:
        return ""
    return load_prompt("fragments/bg_task_notification") + "\n\n" + "\n\n".join(notifications)


def _apply_chat_model_override(
    owner: Any,
    base_config: Any,
    requested_model: str,
    *,
    thread_id: str,
) -> Any:
    """Build and log a per-message model override (Cursor-style).

    On unparseable input or unresolvable credential the override is skipped
    (with a warning) and *base_config* is returned unchanged so chat always
    continues with the default model.  ``fallback_models`` is inherited from
    *base_config* via the shared ``build_model_override_config`` so the
    existing rate_guard fallback walk keeps working.
    """
    if not requested_model or not isinstance(base_config, ModelConfig):
        return base_config

    def _dropped(reason: str) -> Any:
        """Record a dropped override where a reader can see it.

        A dropped override is invisible from the outside — the reply simply
        comes back on the old model — so it goes into the activity feed next
        to the successful case rather than only into the anima's log file.
        """
        logger.warning(
            "[%s] Ignoring chat model override %r: %s",
            owner.name,
            requested_model,
            reason,
        )
        try:
            owner._activity.log(
                "model_override_failed",
                summary=(f"Chat model override ignored ({reason}); still on {base_config.model}"),
                channel="chat",
                meta={
                    "requested": requested_model,
                    "reason": reason,
                    "model": base_config.model,
                    "thread_id": thread_id,
                },
                safe=True,
            )
        except Exception:
            logger.debug("[%s] Failed to log model_override_failed activity", owner.name, exc_info=True)
        return base_config

    try:
        from core.config.io import load_config
        from core.config.model_config import resolve_model_selection

        config = load_config()
        override = resolve_model_selection(
            base_config,
            requested_model=requested_model,
            config=config,
            apply_fallback=False,
        ).effective
        owner._activity.log(
            "model_override",
            summary=(f"Chat model override: {base_config.model} -> {override.resolved_mode}:{override.model}"),
            channel="chat",
            meta={
                "requested": requested_model,
                "resolved": f"{override.resolved_mode}:{override.model}",
                "thread_id": thread_id,
            },
            safe=True,
        )
        logger.info(
            "[%s] Chat model override applied: %s -> %s:%s",
            owner.name,
            base_config.model,
            override.resolved_mode,
            override.model,
        )
        return override
    except ValueError as exc:
        return _dropped(str(exc))
    except Exception:
        logger.warning(
            "[%s] Ignoring chat model override for %r",
            owner.name,
            requested_model,
            exc_info=True,
        )
        return _dropped("resolution error")


def _resolve_chat_retry_config(
    owner: Any,
    primary_config: Any,
    active_config: Any,
    reason: FailoverReason | None,
) -> Any | None:
    """Re-evaluate fallback selection and return a different config once."""
    if reason is None or not isinstance(primary_config, ModelConfig):
        return None
    _classified, hint = classify_llm_error_message(reason.value.replace("_", " "))
    report_capacity_block(active_config, reason, hint)
    retry_config = resolve_effective_model_config(primary_config)
    if same_effective_model(retry_config, active_config):
        return None
    log_model_fallback(
        owner._activity,
        primary_config,
        retry_config,
        channel="chat",
        phase="runtime_retry",
    )
    return retry_config


async def _run_chat_stream_with_fallback(
    owner: Any,
    *,
    prompt: str,
    trigger: str,
    message_intent: str,
    images: list[ImageData] | None,
    prior_messages: list[dict[str, Any]] | None,
    thread_id: str,
    primary_config: Any,
    active_config: Any,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream a chat cycle, walking fallbacks on terminal capacity errors."""
    current_config = active_config
    attempted: set[tuple[Any, ...]] = set()

    while True:
        attempted.add(effective_model_key(current_config))
        retry_config = None
        emitted_payload = False
        try:
            async for chunk in owner.agent.run_cycle_streaming(
                prompt,
                trigger=trigger,
                message_intent=message_intent,
                images=images,
                prior_messages=prior_messages,
                thread_id=thread_id,
                model_config_override=current_config,
            ):
                chunk_type = chunk.get("type")
                reason = None
                if chunk_type == "error":
                    try:
                        reason = FailoverReason(str(chunk.get("reason") or ""))
                    except ValueError:
                        reason, _hint = classify_llm_error_message(str(chunk.get("message") or ""))
                elif chunk_type == "cycle_done":
                    raw_result = chunk.get("cycle_result")
                    if isinstance(raw_result, dict):
                        reason = _chat_fallback_reason_from_result(raw_result)

                if not emitted_payload:
                    retry_config = _resolve_chat_retry_config(
                        owner,
                        primary_config,
                        current_config,
                        reason,
                    )
                    if retry_config is not None:
                        retry_key = effective_model_key(retry_config)
                        if retry_key in attempted:
                            retry_config = None
                    if retry_config is not None:
                        break

                if chunk_type in {"text_delta", "tool_start", "tool_end"}:
                    emitted_payload = True
                yield chunk
        except Exception as exc:
            if not emitted_payload:
                retry_config = _resolve_chat_retry_config(
                    owner,
                    primary_config,
                    current_config,
                    _chat_fallback_reason_from_exception(exc),
                )
                if retry_config is not None:
                    retry_key = effective_model_key(retry_config)
                    if retry_key in attempted:
                        retry_config = None
            if retry_config is None:
                raise

        if retry_config is None:
            return
        current_config = retry_config


def _agent_session_context(owner: Any):
    getter = getattr(owner, "_agent_session_context", None)
    if callable(getter):
        return getter("chat")
    lock = getattr(owner, "_agent_session_lock", None)
    if isinstance(lock, asyncio.Lock):
        return lock
    return nullcontext()


def _schedule_chat_idle_compaction(owner: Any, thread_id: str) -> None:
    """Schedule chat compaction without inheriting the completed cycle context."""

    def _fire_compaction() -> None:
        from core.agent.session_compactor import run_idle_compaction

        contextvars.Context().run(
            spawn,
            run_idle_compaction(owner, thread_id),
            name=f"idle-compaction-{owner.name}-{thread_id}",
        )

    owner._session_compactor.schedule(owner.name, thread_id, _fire_compaction)


async def _inject_chat_message(
    owner: Any,
    content: str,
    *,
    from_person: str,
    thread_id: str,
    attachment_paths: list[str] | None = None,
    files: list[dict[str, Any]] | None = None,
) -> bool:
    """Persist and inject one user turn into the active Mode S stream."""
    conversation = owner._active_chat_conversations.get(thread_id)
    if conversation is None or not owner.agent.supports_message_injection:
        return False

    state = conversation.load()
    conversation.append_turn("human", content, attachments=attachment_paths or [])
    injected_turn = state.turns[-1]
    await _persist_conversation(conversation)
    try:
        injected = await owner.agent.inject_message(content)
    except Exception:
        state.turns[:] = [turn for turn in state.turns if turn is not injected_turn]
        await _persist_conversation(conversation)
        raise
    if not injected:
        state.turns[:] = [turn for turn in state.turns if turn is not injected_turn]
        await _persist_conversation(conversation)
        return False

    await _write_conversation_transcript(
        conversation,
        "human",
        content,
        from_person=from_person,
        thread_id=thread_id,
        attachments=attachment_paths or None,
    )
    owner._log_human_conversation(content, from_person, thread_id)
    await owner._activity.alog(
        "message_received",
        content=content,
        summary=content[:100],
        from_person=from_person,
        channel="chat",
        meta={**_chat_received_meta(thread_id, files), "steer": True},
        origin=ORIGIN_HUMAN,
    )
    _record_chat_user_turn(owner)
    return True


def _chat_cycle_isolated(
    cycle_result: CycleResult | dict[str, Any],
    *,
    expected_trigger: str,
    thread_id: str,
) -> tuple[bool, dict[str, Any]]:
    data = cycle_result.model_dump(mode="json") if isinstance(cycle_result, CycleResult) else cycle_result
    actual_trigger = str(data.get("trigger") or "")
    actual_session_type = str(data.get("session_type") or resolve_runtime_session_type(actual_trigger))
    actual_thread_id = str(data.get("thread_id") or thread_id)
    meta = {
        "expected_trigger": expected_trigger,
        "actual_trigger": actual_trigger,
        "actual_session_type": actual_session_type,
        "expected_thread_id": thread_id,
        "actual_thread_id": actual_thread_id,
        "request_id": str(data.get("request_id") or ""),
        "tool_session_id": str(data.get("tool_session_id") or ""),
    }
    return (
        actual_trigger == expected_trigger and actual_session_type == "chat" and actual_thread_id == thread_id,
        meta,
    )


def _build_meeting_context(
    *,
    thread_id: str,
    room_id: str,
    participants: list[str] | None,
) -> dict[str, Any]:
    if not room_id and thread_id.startswith("meeting-"):
        room_id = thread_id.removeprefix("meeting-")
    try:
        from core.paths import get_shared_dir

        meetings_dir = str(get_shared_dir() / "meetings")
    except Exception:
        meetings_dir = ""
    return {
        "room_id": room_id,
        "meetings_dir": meetings_dir,
        "participants": [str(p) for p in (participants or []) if str(p)],
        "redirects": [],
    }


def _collect_meeting_redirects() -> list[dict[str, str]]:
    from core.tooling.handler_base import meeting_context

    ctx = meeting_context.get()
    if not isinstance(ctx, dict):
        return []
    room_id = str(ctx.get("room_id") or "")
    raw_redirects = ctx.get("redirects", [])
    if not isinstance(raw_redirects, list):
        return []
    redirects: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw_redirects:
        if not isinstance(item, dict):
            continue
        to_name = str(item.get("to") or "")
        content = str(item.get("content") or "")
        if not to_name or not content:
            continue
        key = (to_name.lower(), content.strip(), str(item.get("from") or "").lower())
        if key in seen:
            continue
        seen.add(key)
        redirects.append(
            {
                "room_id": str(item.get("room_id") or room_id),
                "redirect_id": str(item.get("redirect_id") or ""),
                "from": str(item.get("from") or ""),
                "to": to_name,
                "content": content,
                "intent": str(item.get("intent") or ""),
                "ts": str(item.get("ts") or ""),
            }
        )
    return redirects


class MessagingMixin:
    """Mixin: human chat processing, bootstrap, and greeting."""

    async def inject_message(
        self: _MessagingHost,
        content: str,
        *,
        from_person: str = "human",
        thread_id: str = "default",
        attachment_paths: list[str] | None = None,
        files: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Inject and persist a user turn in the active streaming chat."""
        self._validate_thread_id(thread_id)
        file_kwargs = {"files": files} if files else {}
        return await _inject_chat_message(
            self,
            content,
            from_person=from_person,
            thread_id=thread_id,
            attachment_paths=attachment_paths,
            **file_kwargs,
        )

    async def _sync_interactive_bootstrap_state(self: _MessagingHost) -> None:
        """Persist completed/repair state after chat-driven bootstrap changes."""
        try:
            from core.anima.bootstrap_state import get_bootstrap_status
            from core.platform.state_writer import get_state_writer

            status = get_bootstrap_status(self.anima_dir)
            if status.get("state") in {"completed", "needs_repair"}:
                payload = {k: v for k, v in status.items() if not k.startswith("needs_")}
                await get_state_writer(self.anima_dir).write_bootstrap_state(payload)
        except Exception:
            logger.debug("[%s] Failed to sync interactive bootstrap state", self.name, exc_info=True)

    def _log_human_conversation(
        self: _MessagingHost,
        content: str,
        from_person: str,
        thread_id: str = "default",
    ) -> None:
        """Append human message to shared conversation log.

        Writes to ``shared/users/{from_person}/conversations/YYYY-MM-DD.jsonl``
        so that any Anima can search what the human discussed across all Animas.

        Skips system/self senders and known anima names (exact match only) so
        anima-to-anima traffic never creates directories under ``shared/users/``.
        """
        if from_person in ("system", "") or from_person == self.name:
            return
        try:
            from core.anima.roster import is_anima_name

            if is_anima_name(from_person):
                logger.info(
                    "[%s] Skipping shared/users conversation log for anima sender '%s'",
                    self.name,
                    from_person,
                )
                return
        except Exception:
            logger.debug(
                "[%s] Anima roster check failed for '%s'; proceeding with log",
                self.name,
                from_person,
                exc_info=True,
            )
        try:
            shared_dir = self.anima_dir.parent.parent / "shared" / "users" / from_person / "conversations"
            shared_dir.mkdir(parents=True, exist_ok=True)

            today = today_local().isoformat()
            log_file = shared_dir / f"{today}.jsonl"

            record = {
                "ts": now_local().isoformat(),
                "anima": self.name,
                "content": content,
                "thread_id": thread_id,
            }
            from core.platform.atomic_io import append_jsonl_locked

            append_jsonl_locked(log_file, record)
        except Exception:
            logger.warning(
                "[%s] Failed to log human conversation",
                self.name,
                exc_info=True,
            )

    def _resolve_chat_external_recipient(
        self: _MessagingHost,
        from_person: str,
        source: str = "",
    ):
        """Resolve a human chat recipient to an external DM target when configured."""
        if not from_person or from_person in {"human", "user", "system", self.name}:
            return None
        if source and source in EXTERNAL_PLATFORM_SOURCES:
            return None
        try:
            from core.config.models import load_config
            from core.messaging.outbound import resolve_recipient
            from core.paths import get_animas_dir

            config = load_config()
            if not config.external_messaging.chat_dm_redirect:
                return None
            animas_dir = get_animas_dir()
            known_animas = {d.name for d in animas_dir.iterdir() if d.is_dir()} if animas_dir.exists() else set()
            resolved = resolve_recipient(from_person, known_animas, config.external_messaging)
        except Exception:
            logger.debug(
                "[%s] chat external recipient resolution skipped for %s",
                self.name,
                from_person,
                exc_info=True,
            )
            return None
        if resolved is None or resolved.is_internal:
            return None
        return resolved

    @staticmethod
    def _delivery_via_label(via: str) -> str:
        if via == "slack":
            return t("anima.delivery_via_slack")
        if via == "chatwork":
            return t("anima.delivery_via_chatwork")
        return t("anima.delivery_via_external")

    def _send_chat_reply_via_resolved(
        self: _MessagingHost,
        resolved: Any,
        *,
        to_person: str,
        content: str,
    ) -> tuple[str, dict[str, Any]]:
        """Deliver a chat reply via external DM and return a chat-safe notice."""
        from core.messaging.outbound import send_external

        raw = send_external(
            resolved,
            content,
            sender_name=self.name,
            anima_name=self.name,
        )
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {}
        via = str(parsed.get("channel") or getattr(resolved, "channel", "") or "external").lower()
        via_label = self._delivery_via_label(via)
        if parsed.get("status") == "sent":
            return (
                t("anima.chat_reply_sent_via_external", to=to_person, via=via_label),
                {
                    "delivery_via": via,
                    "delivery_target": to_person,
                    "delivery_status": "sent",
                    "display_only": True,
                },
            )
        logger.warning(
            "[%s] External chat reply delivery failed to=%s via=%s raw=%r",
            self.name,
            to_person,
            via,
            raw[:300],
        )
        return (
            t("anima.chat_reply_delivery_failed", to=to_person, via=via_label),
            {
                "delivery_via": via,
                "delivery_target": to_person,
                "delivery_status": "failed",
                "display_only": True,
            },
        )

    async def run_bootstrap(self: _MessagingHost) -> CycleResult:
        """Run the first-time bootstrap process in the background.

        Acquires the anima lock, sets status to ``"bootstrapping"``, and
        triggers an agent cycle with the bootstrap prompt.  The agent reads
        ``BOOTSTRAP.md`` from the anima directory and follows its
        instructions (identity setup, avatar generation, self-introduction,
        etc.).  Upon completion the agent deletes ``BOOTSTRAP.md``.
        """
        if not self.needs_bootstrap:
            logger.info("[%s] run_bootstrap SKIPPED: no BOOTSTRAP.md", self.name)
            return CycleResult(
                trigger="bootstrap",
                action="skipped",
                summary="Bootstrap not needed",
            )

        logger.info("[%s] run_bootstrap START", self.name)
        from core.anima.bootstrap_state import finalize_bootstrap_run, mark_bootstrap_failed, mark_bootstrap_running
        from core.tooling.handler import active_session_type

        try:
            async with self._get_thread_lock("default"):
                self._mark_busy_start()
                self._status_slots["conversation:default"] = "bootstrapping"
                self._task_slots["conversation:default"] = "Initial bootstrap"
                await asyncio.to_thread(
                    mark_bootstrap_running,
                    self.anima_dir,
                    mode="character_sheet" if (self.anima_dir / "character_sheet.md").exists() else "interactive",
                )
                _session_token = self.agent._tool_handler.set_active_session_type("chat")

                conv_memory = ConversationMemory(self.anima_dir, self.model_config)
                prompt = conv_memory.build_chat_prompt(
                    t("anima.bootstrap_prompt"),
                    "system",
                )

                try:
                    async with _agent_session_context(self):
                        self._get_interrupt_event("default").clear()
                        self.agent.set_interrupt_event(self._get_interrupt_event("default"))
                        self.agent._tool_handler.set_session_origin(ORIGIN_SYSTEM)
                        result = await self.agent.run_cycle(prompt, trigger="bootstrap")
                    self._last_activity = now_local()
                    bootstrap_status = await asyncio.to_thread(finalize_bootstrap_run, self.anima_dir)
                    if bootstrap_status.get("state") != "completed":
                        logger.warning(
                            "[%s] bootstrap did not pass validation: state=%s errors=%s",
                            self.name,
                            bootstrap_status.get("state"),
                            bootstrap_status.get("validation_errors", []),
                        )

                    logger.info(
                        "[%s] run_bootstrap END duration_ms=%d",
                        self.name,
                        result.duration_ms,
                    )
                    return result
                except Exception:
                    logger.exception("[%s] run_bootstrap FAILED", self.name)
                    await asyncio.to_thread(mark_bootstrap_failed, self.anima_dir, "run_bootstrap_exception")
                    raise
                finally:
                    active_session_type.reset(_session_token)
                    self._status_slots["conversation:default"] = "idle"
                    self._task_slots["conversation:default"] = ""
        finally:
            self._notify_lock_released()

    async def process_message(
        self: _MessagingHost,
        content: str,
        from_person: str = "human",
        images: list[ImageData] | None = None,
        attachment_paths: list[str] | None = None,
        intent: str = "",
        thread_id: str = "default",
        include_cycle_result: bool = False,
        source: str = "",
        meeting_room_id: str = "",
        meeting_participants: list[str] | None = None,
        voice_mode: bool = False,
        model: str | None = None,
        display_message: str | None = None,
        files: list[dict[str, Any]] | None = None,
    ) -> str | dict[str, Any]:
        """Run the streaming chat pipeline and return its completed response."""
        cycle_result: dict[str, Any] | None = None
        attachment_kwargs: dict[str, Any] = {}
        if display_message is not None:
            attachment_kwargs["display_message"] = display_message
        if files:
            attachment_kwargs["files"] = files
        async for chunk in self.process_message_stream(
            content,
            from_person=from_person,
            images=images,
            attachment_paths=attachment_paths,
            intent=intent,
            thread_id=thread_id,
            source=source,
            meeting_room_id=meeting_room_id,
            meeting_participants=meeting_participants,
            voice_mode=voice_mode,
            model=model,
            _sync_compat=True,
            **attachment_kwargs,
        ):
            if chunk.get("type") == "cycle_done":
                raw_result = chunk.get("cycle_result")
                if isinstance(raw_result, dict):
                    cycle_result = raw_result

        if cycle_result is None:
            raise RuntimeError("Chat stream ended without a cycle result")
        if include_cycle_result:
            return cycle_result
        return cycle_result.get("summary", "")

    async def process_message_stream(
        self: _MessagingHost,
        content: str,
        from_person: str = "human",
        images: list[ImageData] | None = None,
        attachment_paths: list[str] | None = None,
        intent: str = "",
        thread_id: str = "default",
        source: str = "",
        meeting_room_id: str = "",
        meeting_participants: list[str] | None = None,
        voice_mode: bool = False,
        model: str | None = None,
        display_message: str | None = None,
        files: list[dict[str, Any]] | None = None,
        *,
        _sync_compat: bool = False,
    ) -> AsyncGenerator[dict, None]:
        """Streaming version of process_message.

        Yields stream event dicts. The lock is held for the entire duration.
        If bootstrapping is in progress (lock held + needs_bootstrap), yields
        an immediate "initializing" message instead of waiting. The private
        ``_sync_compat`` flag preserves the legacy blocking API contract while
        ``process_message`` drains this generator.
        """
        self._validate_thread_id(thread_id)
        history_content = content if display_message is None else display_message
        lock = self._get_thread_lock(thread_id)

        # ── Bootstrap guard: return immediately if bootstrap is running ──
        if not _sync_compat and self.needs_bootstrap and lock.locked():
            logger.info(
                "[%s] process_message_stream REJECTED (bootstrapping) from=%s",
                self.name,
                from_person,
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
                    self.name,
                    from_person,
                )

        # Cancel any pending idle-compaction timer for this thread
        self._session_compactor.cancel(self.name, thread_id)

        operation_name = "process_message" if _sync_compat else "process_message_stream"
        logger.info(
            "[%s] %s WAITING from=%s content_len=%d images=%d",
            self.name,
            operation_name,
            from_person,
            len(content),
            len(images or []),
        )
        from core.tooling.handler import active_session_type

        try:
            async with lock:
                self._mark_busy_start()
                # Clear interrupt event for OUR session (after lock acquired)
                self._get_interrupt_event(thread_id).clear()
                logger.info(
                    "[%s] %s START (lock acquired) from=%s",
                    self.name,
                    operation_name,
                    from_person,
                )
                _conv_key = f"conversation:{thread_id}"
                self._status_slots[_conv_key] = "thinking"
                self._task_slots[_conv_key] = f"Responding to {from_person}"
                bootstrap_before = self.needs_bootstrap
                _session_token = self.agent._tool_handler.set_active_session_type("chat")
                _meeting_token = None
                _meeting_context_token = None
                if source == "meeting":
                    from core.tooling.handler_base import meeting_context, meeting_mode

                    _meeting_token = meeting_mode.set(True)
                    _meeting_context_token = meeting_context.set(
                        _build_meeting_context(
                            thread_id=thread_id,
                            room_id=meeting_room_id,
                            participants=meeting_participants,
                        )
                    )

                # Human-facing chat must use the per-Anima status.json model,
                # independent of any background/helper model in flight.
                primary_model_config = _resolve_voice_model_config(
                    self.memory.read_model_config(),
                    voice_mode,
                )
                base_model_config = _resolve_chat_model_config(
                    self,
                    primary_model_config,
                    phase="preflight",
                )

                # Optional per-message model override (Cursor-style). Skipped
                # (with a warning) when the model can't be resolved, so chat
                # always continues on the default model.  The override also
                # becomes the fallback context so rate_guard re-routing keeps
                # working within the requested model's family.
                if model:
                    base_model_config = _apply_chat_model_override(
                        self,
                        base_model_config,
                        model,
                        thread_id=thread_id,
                    )
                    primary_model_config = base_model_config

                # Drain completed background-task notices at the start of this
                # chat turn. Keep them out of persisted human content and add
                # them only to the prompt sent to the agent.
                bg_notification_context = await _build_chat_background_notification_context_async(self)

                # Build history-aware prompt via conversation memory
                conv_memory = ConversationMemory(self.anima_dir, base_model_config, thread_id=thread_id)
                if conv_memory.needs_compression():
                    yield {"type": "compression_start"}
                    await conv_memory.compress_if_needed()
                    yield {"type": "compression_end"}

                # Determine prompt and history strategy per execution mode
                mode = self.agent._resolve_execution_mode(base_model_config)
                prior_messages = None
                if mode == "s":
                    prompt = content
                elif mode == "a":
                    prior_messages = conv_memory.build_structured_messages(content)
                    prompt = content
                else:
                    prompt = conv_memory.build_chat_prompt(content, from_person)
                if bg_notification_context:
                    prompt = f"{bg_notification_context}\n\n{prompt}"

                # Pre-save: persist user input before agent execution
                conv_memory.append_turn(
                    "human",
                    history_content,
                    attachments=attachment_paths or [],
                )
                await _persist_conversation(conv_memory)

                # Transcript: record human message
                await _write_conversation_transcript(
                    conv_memory,
                    "human",
                    history_content,
                    from_person=from_person,
                    thread_id=thread_id,
                    attachments=attachment_paths or None,
                )

                # Shared conversation log: record human message
                self._log_human_conversation(history_content, from_person, thread_id)

                # Activity log: message received
                received_activity = await self._activity.alog(
                    "message_received",
                    content=history_content,
                    summary=history_content[:100],
                    from_person=from_person,
                    channel="chat",
                    meta=_chat_received_meta(thread_id, files),
                    origin=ORIGIN_HUMAN,
                )
                live_session_started_at = getattr(received_activity, "ts", None)
                _record_chat_user_turn(self)

                if source and source in EXTERNAL_PLATFORM_SOURCES:
                    _ctx = t("anima.platform_context", source=source)
                    prompt = f"{_ctx}\n\n{prompt}"
                external_chat_recipient = self._resolve_chat_external_recipient(from_person, source)

                # Streaming journal: write-ahead log for crash recovery
                journal = StreamingJournal(self.anima_dir, thread_id=thread_id)
                partial_response = ""
                text_segment = ""
                interim_texts: list[str] = []
                cycle_done = False
                agent_session_acquired = False
                self._active_chat_conversations[thread_id] = conv_memory

                try:
                    await asyncio.to_thread(
                        journal.open,
                        trigger=f"message:{from_person}",
                        from_person=from_person,
                    )
                    agent_session_lock = getattr(self, "_agent_session_lock", None)
                    if isinstance(agent_session_lock, asyncio.Lock):
                        await agent_session_lock.acquire()
                        agent_session_acquired = True
                    self.agent.set_interrupt_event(self._get_interrupt_event(thread_id))
                    self.agent._tool_handler.set_session_origin(ORIGIN_HUMAN)
                    async for chunk in _run_chat_stream_with_fallback(
                        self,
                        prompt=prompt,
                        trigger=f"message:{from_person}",
                        message_intent=intent,
                        images=images,
                        prior_messages=prior_messages,
                        thread_id=thread_id,
                        primary_config=primary_model_config,
                        active_config=base_model_config,
                    ):
                        if chunk.get("type") == "text_delta":
                            delta_text = chunk.get("text", "")
                            partial_response += delta_text
                            text_segment += delta_text
                            await asyncio.to_thread(journal.write_text, delta_text)

                        if chunk.get("type") == "tool_start":
                            if text_segment.strip():
                                interim_texts.append(text_segment.strip()[:2000])
                            text_segment = ""
                            await asyncio.to_thread(
                                journal.write_tool_start,
                                tool=chunk.get("tool_name", ""),
                                args_summary="",
                            )
                        if chunk.get("type") == "tool_end":
                            await asyncio.to_thread(
                                journal.write_tool_end,
                                tool=chunk.get("tool_name", ""),
                                result_summary="",
                            )

                        if chunk.get("type") == "cycle_done":
                            cycle_done = True
                            self._last_activity = now_local()
                            # Record assistant response with tool records
                            raw_cycle_result = chunk.get("cycle_result", {})
                            cycle_result = raw_cycle_result if isinstance(raw_cycle_result, dict) else {}
                            chunk["cycle_result"] = cycle_result
                            meeting_redirects = _collect_meeting_redirects()
                            if meeting_redirects:
                                cycle_result["meeting_redirects"] = meeting_redirects
                            guard_ok, guard_meta = _chat_cycle_isolated(
                                cycle_result,
                                expected_trigger=f"message:{from_person}",
                                thread_id=thread_id,
                            )
                            if not guard_ok:
                                logger.error(
                                    "[%s] session guard blocked non-chat stream result from conversation storage: %s",
                                    self.name,
                                    guard_meta,
                                )
                                await self._activity.alog(
                                    "session_guard_violation",
                                    summary="Blocked non-chat stream result from chat conversation storage",
                                    channel="chat",
                                    meta=guard_meta,
                                    safe=True,
                                )
                                cycle_result["summary"] = ""
                                await asyncio.to_thread(journal.finalize, summary="session guard violation")
                                yield chunk
                                continue
                            summary = normalize_user_facing_response_text(cycle_result.get("summary", ""))

                            # Resolve local absolute/file:// image paths → attachments/
                            summary, local_artifacts = resolve_local_image_paths(
                                summary,
                                self.anima_dir,
                            )
                            cycle_result["summary"] = summary
                            if meeting_redirects:
                                for redirect in meeting_redirects:
                                    yield {"type": "meeting_redirect", **redirect}

                            response_artifacts = extract_image_artifacts_from_tool_records(
                                cycle_result.get("tool_call_records", [])
                            )
                            remaining = max(0, 5 - len(response_artifacts))
                            response_artifacts.extend(local_artifacts[:remaining])
                            cycle_result["images"] = response_artifacts
                            response_files = collect_outbound_files(
                                self.anima_dir,
                                str(cycle_result.get("request_id") or ""),
                                thread_id=thread_id,
                                since=live_session_started_at,
                            )
                            if response_files:
                                cycle_result["files"] = response_files
                            display_summary = summary
                            resp_meta: dict[str, Any] = {"thread_id": thread_id}
                            if interim_texts:
                                resp_meta["interim_texts"] = interim_texts[:20]
                            for _field in ("session_type", "request_id", "tool_session_id"):
                                _value = cycle_result.get(_field)
                                if _value:
                                    resp_meta[_field] = _value
                            if external_chat_recipient is not None:
                                display_summary, delivery_meta = self._send_chat_reply_via_resolved(
                                    external_chat_recipient,
                                    to_person=from_person,
                                    content=summary,
                                )
                                resp_meta.update(delivery_meta)
                            cycle_result["summary"] = display_summary
                            tool_records = [ToolRecord.from_dict(r) for r in cycle_result.get("tool_call_records", [])]
                            conv_memory.append_turn(
                                "assistant",
                                display_summary,
                                tool_records=tool_records,
                            )
                            await _persist_conversation(conv_memory)

                            # Transcript: record assistant response
                            await _write_conversation_transcript(
                                conv_memory,
                                "assistant",
                                display_summary,
                                thread_id=thread_id,
                                tool_names=[r.tool_name for r in tool_records if r.tool_name] or None,
                            )

                            # Activity log: response sent (with thinking text if present)
                            thinking_text = cycle_result.get("thinking_text", "")
                            if thinking_text:
                                resp_meta["thinking_text"] = thinking_text
                            if response_artifacts:
                                resp_meta["images"] = response_artifacts
                            if response_files:
                                resp_meta["files"] = response_files
                            await self._activity.alog(
                                "response_sent",
                                content=display_summary,
                                to_person=from_person,
                                channel="chat",
                                summary=display_summary[:200] if display_summary else "",
                                meta=resp_meta,
                            )
                            _queue_live_fact_extraction(self, "chat", live_session_started_at)

                            if bootstrap_before:
                                bootstrap_sync = self._sync_interactive_bootstrap_state()
                                if inspect.isawaitable(bootstrap_sync):
                                    await bootstrap_sync

                            # Finalize streaming journal (deletes the file)
                            await asyncio.to_thread(journal.finalize, summary=display_summary[:500])

                            # The blocking caller leaves notifications queued for its caller to drain.
                            if not _sync_compat:
                                for notif in self.agent.drain_notifications():
                                    yield {"type": "notification_sent", "data": notif}

                            if _sync_compat:
                                logger.info(
                                    "[%s] process_message END duration_ms=%d",
                                    self.name,
                                    int(cycle_result.get("duration_ms") or 0),
                                )
                            else:
                                logger.info("[%s] process_message_stream END", self.name)
                                _schedule_chat_idle_compaction(self, thread_id)
                        if external_chat_recipient is not None and chunk.get("type") == "text_delta":
                            continue
                        yield chunk
                except Exception as exc:
                    if _sync_compat:
                        logger.exception("[%s] process_message FAILED", self.name)
                        await self._activity.alog(
                            "error",
                            summary=t("anima.process_message_error", exc=type(exc).__name__),
                            meta={"phase": "process_message", "error": str(exc)[:200], "thread_id": thread_id},
                            safe=True,
                        )
                        conv_memory.append_turn("assistant", t("anima.agent_error"))
                        await _persist_conversation(conv_memory)
                        raise

                    logger.exception("[%s] process_message_stream FAILED", self.name)
                    if isinstance(exc, ToolError):
                        error_code = "TOOL_ERROR"
                    elif isinstance(exc, (LLMAPIError, ExecutionError)):
                        error_code = "LLM_ERROR"
                    elif isinstance(exc, MemoryIOError):
                        error_code = "MEMORY_ERROR"
                    else:
                        error_code = "STREAM_ERROR"
                    # Activity log: error (safe=True to prevent double-fault)
                    await self._activity.alog(
                        "error",
                        summary=t("anima.process_stream_error", exc=type(exc).__name__),
                        meta={
                            "phase": "process_message_stream",
                            "error_code": error_code,
                            "error": str(exc)[:200],
                            "thread_id": thread_id,
                        },
                        safe=True,
                    )
                    yield {
                        "type": "error",
                        "code": error_code,
                        "message": "Internal error",
                    }
                finally:
                    if self._active_chat_conversations.get(thread_id) is conv_memory:
                        self._active_chat_conversations.pop(thread_id, None)
                    if agent_session_acquired:
                        agent_session_lock.release()
                    if not cycle_done and not _sync_compat:
                        logger.warning(
                            "[%s] process_message_stream END (cycle_done not received)",
                            self.name,
                        )
                    # Save partial response if cycle_done was never received
                    if not cycle_done and not _sync_compat:
                        if external_chat_recipient is not None:
                            saved_text = t("anima.response_interrupted")
                        elif partial_response:
                            saved_text = partial_response + t("anima.response_interrupted_prefix")
                        else:
                            saved_text = t("anima.response_interrupted")
                        conv_memory.append_turn("assistant", saved_text)
                        await _persist_conversation(conv_memory)
                    # Close journal (no-op if already finalized)
                    await asyncio.to_thread(journal.close)
                    if _meeting_context_token is not None:
                        from core.tooling.handler_base import meeting_context

                        meeting_context.reset(_meeting_context_token)
                    if _meeting_token is not None:
                        from core.tooling.handler_base import meeting_mode

                        meeting_mode.reset(_meeting_token)
                    try:
                        active_session_type.reset(_session_token)
                    except ValueError:
                        pass
                    self._status_slots[_conv_key] = "idle"
                    self._task_slots[_conv_key] = ""
                    if _sync_compat:
                        _schedule_chat_idle_compaction(self, thread_id)
        finally:
            self._notify_lock_released()

    async def process_greet(
        self: _MessagingHost, *, mode: str = "visit", user_name: str = "", user_id: str = ""
    ) -> dict[str, str | bool]:
        """Generate a greeting response for a desk visit or first meeting.

        Always runs the LLM; no response caching.  ``cached`` is kept as
        ``False`` for API compatibility.
        First-meeting greetings use a dedicated prompt and are also recorded
        in the activity log.

        Returns:
            Dict with keys: response, emotion, cached.
        """
        is_first_meeting = mode == "first_meeting"
        live_session_started_at = now_local().isoformat()

        logger.info("[%s] process_greet START", self.name)
        from core.tooling.handler import active_session_type

        async with self._get_thread_lock("default"):
            self._mark_busy_start()
            prev_status = self._status_slots.get("conversation:default", "idle")
            prev_task = self._task_slots.get("conversation:default", "")
            _session_token = self.agent._tool_handler.set_active_session_type("chat")

            # Build the prompt with current state (use primary to include background)
            if is_first_meeting:
                prompt = load_prompt(
                    "first_meeting",
                    user_name=user_name or t("anima.first_meeting_default_user"),
                )
            else:
                status_text = self.primary_status if self.primary_status != "idle" else t("anima.status_idle")
                task_text = self.primary_task if self.primary_task else t("anima.task_none")
                prompt = load_prompt(
                    "greet",
                    status=status_text,
                    active_label=task_text,
                )

            self._status_slots["conversation:default"] = "greeting"
            self._task_slots["conversation:default"] = "Greeting user"

            conv_memory = ConversationMemory(self.anima_dir, self.model_config)

            # A first meeting happens once.  If the anima already spoke in this
            # conversation (e.g. the page was reloaded while the first greeting
            # was in flight), return that utterance instead of generating a
            # second introduction.
            if is_first_meeting:
                existing = next(
                    (turn for turn in reversed(list(conv_memory.load().turns)) if turn.role == "assistant"),
                    None,
                )
                if existing is not None:
                    logger.info("[%s] process_greet first_meeting already answered; reusing", self.name)
                    return {"response": existing.content, "emotion": "neutral", "cached": True}

            # Record the event marker before greeting.
            marker = t("anima.first_meeting_marker") if is_first_meeting else t("anima.visit_desk")
            conv_memory.append_turn("system", marker)
            await _persist_conversation(conv_memory)

            try:
                async with _agent_session_context(self):
                    self._get_interrupt_event("default").clear()
                    self.agent.set_interrupt_event(self._get_interrupt_event("default"))
                    self.agent._tool_handler.set_session_origin(ORIGIN_HUMAN)
                    result = await self.agent.run_cycle(
                        prompt,
                        trigger="greet:first_meeting" if is_first_meeting else "greet:user",
                    )
                self._last_activity = now_local()

                # Extract emotion from response (shared lenient parser)
                clean_text, emotion = _extract_emotion_from_tag(result.summary)

                # Record assistant turn in conversation memory
                conv_memory.append_turn("assistant", clean_text)
                await _persist_conversation(conv_memory)

                # The chat UI builds its history from the activity log, so a
                # first-meeting greeting must be recorded there as well or the
                # page would re-request it on every reload.
                if is_first_meeting:
                    await self._activity.alog(
                        "response_sent",
                        content=clean_text,
                        to_person=user_id or user_name,
                        channel="chat",
                        summary=clean_text[:200],
                        meta={
                            "thread_id": "default",
                            "greet_mode": "first_meeting",
                            "request_id": getattr(result, "request_id", "") or "",
                        },
                    )
                    _queue_live_fact_extraction(self, "chat", live_session_started_at)

                logger.info(
                    "[%s] process_greet END duration_ms=%d",
                    self.name,
                    result.duration_ms,
                )
                return {
                    "response": clean_text,
                    "emotion": emotion,
                    "cached": False,
                }
            except Exception:
                logger.exception("[%s] process_greet FAILED", self.name)
                # Save error marker in conversation memory
                conv_memory.append_turn("assistant", t("anima.greeting_error"))
                await _persist_conversation(conv_memory)
                raise
            finally:
                active_session_type.reset(_session_token)
                self._status_slots["conversation:default"] = prev_status
                self._task_slots["conversation:default"] = prev_task
