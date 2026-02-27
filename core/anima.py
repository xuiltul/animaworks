from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


import asyncio
import logging
import re
import threading
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.time_utils import now_iso, now_jst

from core.agent import AgentCore
from core.background import BackgroundTask
from core.memory.activity import ActivityLogger
from core.memory.streaming_journal import StreamingJournal
from core.memory.conversation import ConversationMemory, ToolRecord
from core.memory import MemoryManager
from core.messenger import InboxItem, Messenger
from core.i18n import t
from core.paths import load_prompt
from core.image_artifacts import extract_image_artifacts_from_tool_records
from core.exceptions import (  # noqa: F401
    AnimaWorksError,
    ExecutionError,
    LLMAPIError,
    ToolError,
    MemoryIOError,
)
from core.schemas import CycleResult, AnimaStatus, ModelConfig, VALID_EMOTIONS

logger = logging.getLogger("animaworks.anima")

# Maximum time (seconds) an unreplied message stays in inbox before
# force-archival.  Prevents re-processing storms when replied_to tracking
# fails (e.g. agent replies via board post instead of DM).
# With a typical heartbeat interval of 5 min, a message gets ~2 chances
# to be replied to before force-archival.
_STALE_MESSAGE_TIMEOUT_SEC = 600

# ── Reflection extraction ─────────────────────────────────────

_RE_REFLECTION = re.compile(
    r"\[REFLECTION\]\s*\n?(.*?)\n?\s*\[/REFLECTION\]",
    re.DOTALL,
)

_MIN_REFLECTION_LENGTH = 50

def _extract_reflection(text: str) -> str:
    """Extract [REFLECTION]...[/REFLECTION] block from heartbeat output.

    Returns empty string if not found or content is trivial.
    """
    if not text:
        return ""
    m = _RE_REFLECTION.search(text)
    if m:
        return m.group(1).strip()
    return ""


@dataclass
class InboxResult:
    """Result of inbox message processing."""

    inbox_items: list[InboxItem] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)
    senders: set[str] = field(default_factory=set)
    unread_count: int = 0
    prompt_parts: list[str] = field(default_factory=list)


class DigitalAnima:
    """A Digital Anima: encapsulates identity, memory, agent, and communication.

    1 anima = 1 directory.
    """

    _MAX_THREAD_LOCKS = 20
    _THREAD_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,36}$")

    @staticmethod
    def _validate_thread_id(thread_id: str) -> None:
        """Validate thread_id to prevent path traversal attacks."""
        if not DigitalAnima._THREAD_ID_PATTERN.match(thread_id):
            raise ValueError(
                f"Invalid thread_id: {thread_id!r}. "
                "Must be 1-36 alphanumeric, underscore, or hyphen characters."
            )

    def __init__(self, anima_dir: Path, shared_dir: Path) -> None:
        self.anima_dir = anima_dir
        self.name = anima_dir.name
        self._activity = ActivityLogger(anima_dir)

        self.memory = MemoryManager(anima_dir)
        self.model_config = self.memory.read_model_config()
        self.messenger = Messenger(shared_dir, self.name)
        self._interrupt_event = asyncio.Event()
        self.agent = AgentCore(
            anima_dir, self.memory, self.model_config, self.messenger
        )
        self.agent.set_interrupt_event(self._interrupt_event)

        # 3-lock structure: conversation (human chat) / inbox (Anima-to-Anima MSG) / background (HB/cron/TaskExec)
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._inbox_lock = asyncio.Lock()
        self._background_lock = asyncio.Lock()
        self._state_file_lock = threading.Lock()  # protects current_task.md / pending.md
        self.agent._tool_handler.set_state_file_lock(self._state_file_lock)
        self._status_slots: dict[str, str] = {"conversation": "idle", "inbox": "idle", "background": "idle"}
        self._task_slots: dict[str, str] = {"conversation": "", "inbox": "", "background": ""}
        self._last_heartbeat: datetime | None = None
        self._last_activity: datetime | None = None
        self._on_lock_released: Callable[[], None] | None = None
        self._pending_executor: Any | None = None  # set by runner after PendingTaskExecutor init

        # Greet cache (1-hour cooldown)
        self._last_greet_at: float | None = None
        self._last_greet_text: str | None = None
        self._last_greet_emotion: str = "neutral"
        self._GREET_COOLDOWN = 3600  # seconds

        # Wire background task completion callback
        self._ws_broadcast: Callable[[dict], Any] | None = None
        if self.agent.background_manager:
            self.agent.background_manager.on_complete = self._on_background_task_complete

        logger.info("DigitalAnima '%s' initialized from %s", self.name, anima_dir)

    def _get_thread_lock(self, thread_id: str) -> asyncio.Lock:
        """Get or create a per-thread conversation lock.

        Implements LRU eviction when max locks reached. Locked (in-use)
        locks are never evicted.
        """
        if thread_id not in self._conversation_locks:
            if len(self._conversation_locks) >= self._MAX_THREAD_LOCKS:
                # Evict oldest idle lock
                for k in list(self._conversation_locks):
                    if not self._conversation_locks[k].locked():
                        del self._conversation_locks[k]
                        break
            self._conversation_locks[thread_id] = asyncio.Lock()
        return self._conversation_locks[thread_id]

    def set_on_message_sent(
        self, fn: Callable[[str, str, str], None],
    ) -> None:
        """Inject a callback fired after this anima sends a message."""
        self.agent.set_on_message_sent(fn)

    def set_on_schedule_changed(
        self, fn: Callable[[str], Any] | None,
    ) -> None:
        """Inject a callback fired when heartbeat.md or cron.md is modified."""
        self.agent.set_on_schedule_changed(fn)

    def drain_notifications(self) -> list[dict[str, Any]]:
        """Return and clear pending notification events."""
        return self.agent.drain_notifications()

    def drain_background_notifications(self) -> list[str]:
        """Read and remove all pending background task notifications.

        Returns list of notification texts for inclusion in heartbeat context.
        """
        notif_dir = self.agent.anima_dir / "state" / "background_notifications"
        if not notif_dir.is_dir():
            return []

        notifications: list[str] = []
        for path in sorted(notif_dir.glob("*.md")):
            try:
                notifications.append(path.read_text(encoding="utf-8"))
                path.unlink()
            except Exception:
                logger.warning("Failed to read notification: %s", path.name)

        return notifications

    async def interrupt(self) -> dict[str, Any]:
        """Interrupt the current LLM session without killing the process."""
        logger.info("Interrupt requested for anima '%s'", self.name)
        self._interrupt_event.set()
        return {"status": "interrupted", "name": self.name}

    def reload_config(self) -> dict[str, Any]:
        """Hot-reload ModelConfig from status.json without process restart."""
        old = self.model_config
        new = self.memory.read_model_config()
        self.model_config = new
        self.agent.update_model_config(new)
        changes = [
            k for k in ModelConfig.model_fields
            if getattr(old, k) != getattr(new, k)
        ]
        logger.info("reload_config: model=%s, changes=%s", new.model, changes)
        return {"status": "ok", "model": new.model, "changes": changes}

    def set_on_lock_released(self, fn: Callable[[], Any]) -> None:
        """Inject a callback invoked when the anima's lock is released."""
        self._on_lock_released = fn

    def set_ws_broadcast(self, fn: Callable[[dict], Any]) -> None:
        """Inject a WebSocket broadcast function for background task notifications."""
        self._ws_broadcast = fn

    async def _on_background_task_complete(self, task: BackgroundTask) -> None:
        """Callback invoked when a background tool call completes."""
        logger.info(
            "[%s] Background task completed: id=%s tool=%s status=%s",
            self.name, task.task_id, task.tool_name, task.status.value,
        )

        # Broadcast via WebSocket
        if self._ws_broadcast:
            try:
                await self._ws_broadcast({
                    "type": "background_task.done",
                    "data": {
                        "task_id": task.task_id,
                        "anima": self.name,
                        "tool_name": task.tool_name,
                        "status": task.status.value,
                        "result_summary": task.summary(),
                    },
                })
            except Exception:
                logger.exception(
                    "[%s] WebSocket broadcast failed for bg task %s",
                    self.name, task.task_id,
                )

        # Notify human via configured channels
        if self.agent.has_human_notifier:
            try:
                notifier = self.agent.human_notifier
                if notifier:
                    await notifier.notify(
                        subject=t("anima.bg_task_done", tool=task.tool_name),
                        body=task.summary(),
                        priority="normal",
                        anima_name=self.name,
                    )
            except Exception:
                logger.exception(
                    "[%s] Human notification failed for bg task %s",
                    self.name, task.task_id,
                )

        # Send inbox notification so next heartbeat picks up the result
        try:
            summary = task.summary()
            subject = t("anima.bg_task_done", tool=task.tool_name)
            if task.status.value == "failed":
                subject = t("anima.bg_task_failed", tool=task.tool_name)

            # Write a notification file to the anima's own inbox-like location
            # so the next heartbeat can process it
            notif_dir = self.agent.anima_dir / "state" / "background_notifications"
            notif_dir.mkdir(parents=True, exist_ok=True)
            notif_path = notif_dir / f"{task.task_id}.md"
            notif_content = (
                f"# {subject}\n\n"
                f"{t('anima.bg_notif_task_id', task_id=task.task_id)}\n"
                f"{t('anima.bg_notif_tool', tool=task.tool_name)}\n"
                f"{t('anima.bg_notif_status', status=task.status.value)}\n"
                f"{t('anima.bg_notif_result', summary=summary)}\n"
            )
            notif_path.write_text(notif_content, encoding="utf-8")
            logger.info(
                "[%s] Background task notification written: %s",
                self.name, notif_path.name,
            )
        except Exception:
            logger.exception(
                "[%s] Failed to write bg task notification for %s",
                self.name, task.task_id,
            )

    @property
    def background_tasks(self) -> list[dict[str, Any]]:
        """Return a list of all background tasks as dicts."""
        mgr = self.agent.background_manager
        if not mgr:
            return []
        return [t.to_dict() for t in mgr.list_tasks()]

    def _notify_lock_released(self) -> None:
        if self._on_lock_released:
            try:
                self._on_lock_released()
            except Exception:
                logger.exception("[%s] on_lock_released callback failed", self.name)

    # ── Heartbeat history ────────────────────────────────────

    _HEARTBEAT_HISTORY_N = 3

    def _load_heartbeat_history(self) -> str:
        """Load last N heartbeat history entries from unified activity log.

        Falls back to legacy ``shortterm/heartbeat_history/`` when the
        activity log is empty (migration period).
        """
        try:
            entries = self._activity.recent(
                days=2,
                types=["heartbeat_end"],
                limit=self._HEARTBEAT_HISTORY_N,
            )
            if entries:
                lines: list[str] = []
                for e in entries:
                    ts_short = e.ts[11:19] if len(e.ts) >= 19 else e.ts
                    summary = e.summary or e.content
                    lines.append(f"- {ts_short}: {summary}")
                return "\n".join(lines)

            # Legacy fallback: read from shortterm/heartbeat_history/
            legacy = self.memory.load_recent_heartbeat_summary(
                limit=self._HEARTBEAT_HISTORY_N,
            )
            if legacy:
                return legacy
            return ""
        except Exception:
            logger.exception("[%s] Failed to load heartbeat history", self.name)
            return ""

    # ── Heartbeat reflections ─────────────────────────────────

    _RECENT_REFLECTIONS_N = 3

    def _load_recent_reflections(self) -> str:
        """Load recent heartbeat reflections from unified activity log."""
        try:
            entries = self._activity.recent(
                days=3,
                types=["heartbeat_reflection"],
                limit=self._RECENT_REFLECTIONS_N,
            )
            if not entries:
                return ""
            lines: list[str] = []
            for e in entries:
                ts_short = e.ts[11:19] if len(e.ts) >= 19 else e.ts
                content = e.content or e.summary
                lines.append(f"- {ts_short}: {content[:300]}")
            return "\n".join(lines)
        except Exception:
            logger.debug(
                "[%s] Failed to load recent reflections",
                self.name, exc_info=True,
            )
            return ""

    @property
    def needs_bootstrap(self) -> bool:
        """True if this anima has not completed the first-run bootstrap."""
        return (self.anima_dir / "bootstrap.md").exists()

    @property
    def primary_status(self) -> str:
        """Primary status: conversation > inbox > background."""
        conv = self._status_slots.get("conversation", "idle")
        if conv != "idle":
            return conv
        inbox = self._status_slots.get("inbox", "idle")
        if inbox != "idle":
            return inbox
        return self._status_slots.get("background", "idle")

    @property
    def primary_task(self) -> str:
        """Primary task: conversation > inbox > background."""
        conv = self._task_slots.get("conversation", "")
        if conv:
            return conv
        inbox = self._task_slots.get("inbox", "")
        if inbox:
            return inbox
        return self._task_slots.get("background", "")

    @property
    def status(self) -> AnimaStatus:
        return AnimaStatus(
            name=self.name,
            status=self.primary_status,
            current_task=self.primary_task,
            last_heartbeat=self._last_heartbeat,
            last_activity=self._last_activity,
            pending_messages=self.messenger.unread_count(),
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
                self._status_slots["conversation"] = "bootstrapping"
                self._task_slots["conversation"] = "Initial bootstrap"
                _session_token = self.agent._tool_handler.set_active_session_type("chat")

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
                    self._status_slots["conversation"] = "idle"
                    self._task_slots["conversation"] = ""
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
        self._interrupt_event.clear()
        logger.info(
            "[%s] process_message WAITING from=%s content_len=%d images=%d",
            self.name, from_person, len(content), len(images or []),
        )
        from core.tooling.handler import active_session_type
        try:
            async with self._get_thread_lock(thread_id):
                logger.info(
                    "[%s] process_message START (lock acquired) from=%s",
                    self.name, from_person,
                )
                self._status_slots["conversation"] = "thinking"
                self._task_slots["conversation"] = f"Responding to {from_person}"
                _session_token = self.agent._tool_handler.set_active_session_type("chat")

                # Build history-aware prompt via conversation memory
                conv_memory = ConversationMemory(self.anima_dir, self.model_config, thread_id=thread_id)
                await conv_memory.compress_if_needed()

                # Determine prompt and history strategy per execution mode
                mode = self.agent.execution_mode
                prior_messages = None
                if mode == "s":
                    # S mode: SDK manages conversation history internally,
                    # but we still save turns for downstream memory processes.
                    prompt = content
                elif mode == "a":
                    # A mode: AnimaWorks manages history via structured messages
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

                # Activity log: message received
                self._activity.log("message_received", content=content, summary=content[:100], from_person=from_person, channel="chat", meta={"from_type": "human", "thread_id": thread_id})

                try:
                    result = await self.agent.run_cycle(
                        prompt, trigger=f"message:{from_person}",
                        message_intent=intent,
                        images=images,
                        prior_messages=prior_messages,
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
                    self._status_slots["conversation"] = "idle"
                    self._task_slots["conversation"] = ""
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
        self._interrupt_event.clear()
        # ── Bootstrap guard: return immediately if bootstrap is running ──
        if self.needs_bootstrap and self._get_thread_lock(thread_id).locked():
            logger.info(
                "[%s] process_message_stream REJECTED (bootstrapping) from=%s",
                self.name, from_person,
            )
            yield {
                "type": "bootstrap_busy",
                "message": t("anima.initializing"),
            }
            return

        logger.info(
            "[%s] process_message_stream WAITING from=%s content_len=%d images=%d",
            self.name, from_person, len(content), len(images or []),
        )
        from core.tooling.handler import active_session_type
        try:
            async with self._get_thread_lock(thread_id):
                logger.info(
                    "[%s] process_message_stream START (lock acquired) from=%s",
                    self.name, from_person,
                )
                self._status_slots["conversation"] = "thinking"
                self._task_slots["conversation"] = f"Responding to {from_person}"
                _session_token = self.agent._tool_handler.set_active_session_type("chat")

                # Build history-aware prompt via conversation memory
                conv_memory = ConversationMemory(self.anima_dir, self.model_config, thread_id=thread_id)
                await conv_memory.compress_if_needed()

                # Determine prompt and history strategy per execution mode
                mode = self.agent.execution_mode
                prior_messages = None
                if mode == "s":
                    # S mode: SDK manages conversation history internally,
                    # but we still save turns for downstream memory processes.
                    prompt = content
                elif mode == "a":
                    # A mode: AnimaWorks manages history via structured messages
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

                # Activity log: message received
                self._activity.log("message_received", content=content, summary=content[:100], from_person=from_person, channel="chat", meta={"from_type": "human", "thread_id": thread_id})

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
                    self._status_slots["conversation"] = "idle"
                    self._task_slots["conversation"] = ""
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
            prev_status = self._status_slots.get("conversation", "idle")
            prev_task = self._task_slots.get("conversation", "")
            _session_token = self.agent._tool_handler.set_active_session_type("chat")

            # Build greet prompt with current state (use primary to include background)
            status_text = self.primary_status if self.primary_status != "idle" else t("anima.status_idle")
            task_text = self.primary_task if self.primary_task else t("anima.task_none")
            prompt = load_prompt(
                "greet", status=status_text, current_task=task_text,
            )

            self._status_slots["conversation"] = "greeting"
            self._task_slots["conversation"] = "Greeting user"

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

                # Extract emotion from response (inline to avoid circular import)
                import re as _re
                _em_pat = _re.compile(r'<!--\s*emotion:\s*(\{.*?\})\s*-->', _re.DOTALL)
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
                self._status_slots["conversation"] = prev_status
                self._task_slots["conversation"] = prev_task

    # ── Inbox MSG Immediate Processing ────────────────────────

    async def process_inbox_message(
        self,
        cascade_suppressed_senders: set[str] | None = None,
    ) -> CycleResult:
        """Process Anima-to-Anima messages immediately under _inbox_lock.

        Separated from heartbeat to provide instant response to inter-Anima
        messages without triggering the full heartbeat observation cycle.
        """
        self._interrupt_event.clear()
        logger.info("[%s] process_inbox_message START", self.name)
        try:
            async with self._inbox_lock:
                self._status_slots["inbox"] = "processing"

                self._activity.log("inbox_processing_start", summary=t("anima.inbox_start"))

                inbox_result: InboxResult | None = None
                try:
                    inbox_result = await self._process_inbox_messages(
                        cascade_suppressed_senders,
                    )

                    if inbox_result.unread_count == 0:
                        logger.info("[%s] process_inbox_message: no messages", self.name)
                        return CycleResult(
                            trigger="inbox",
                            action="idle",
                            summary="No unread messages",
                        )

                    senders_str = ", ".join(inbox_result.senders)
                    trigger = f"inbox:{senders_str}"

                    task_delegation_rules = load_prompt("task_delegation_rules")
                    messages_text = "\n\n".join(inbox_result.prompt_parts)
                    prompt = load_prompt(
                        "inbox_message",
                        messages=messages_text,
                        task_delegation_rules=task_delegation_rules,
                    )

                    # Suppress board fanout when replying to board_mention
                    has_board_mention = any(
                        item.msg.type == "board_mention"
                        for item in inbox_result.inbox_items
                    )
                    from core.tooling.handler import suppress_board_fanout, active_session_type
                    _fanout_token = suppress_board_fanout.set(True) if has_board_mention else None
                    _session_token = self.agent._tool_handler.set_active_session_type("inbox")

                    self.agent.reset_reply_tracking(session_type="inbox")
                    self.agent.reset_posted_channels(session_type="inbox")

                    journal = StreamingJournal(self.anima_dir, session_type="inbox")
                    journal.open(trigger=trigger, from_person=senders_str)

                    accumulated_text = ""
                    result: CycleResult | None = None

                    try:
                        async for chunk in self.agent.run_cycle_streaming(
                            prompt, trigger=trigger,
                        ):
                            if chunk.get("type") == "text_delta":
                                accumulated_text += chunk.get("text", "")
                                journal.write_text(chunk.get("text", ""))

                            if chunk.get("type") == "cycle_done":
                                cycle_result = chunk.get("cycle_result", {})
                                result = CycleResult(
                                    trigger=trigger,
                                    action=cycle_result.get("action", "responded"),
                                    summary=cycle_result.get("summary", ""),
                                    duration_ms=cycle_result.get("duration_ms", 0),
                                    context_usage_ratio=cycle_result.get(
                                        "context_usage_ratio", 0.0
                                    ),
                                    session_chained=cycle_result.get(
                                        "session_chained", False
                                    ),
                                    total_turns=cycle_result.get("total_turns", 0),
                                )
                                journal.finalize(summary=result.summary[:500])

                        if result is None:
                            result = CycleResult(
                                trigger=trigger,
                                action="responded",
                                summary=accumulated_text[:500] or "(no result)",
                            )
                    finally:
                        journal.close()
                        if _fanout_token is not None:
                            suppress_board_fanout.reset(_fanout_token)
                        active_session_type.reset(_session_token)

                    self._last_activity = now_jst()

                    # Archive processed messages
                    await self._archive_processed_messages(
                        inbox_result.inbox_items,
                        inbox_result.senders,
                        self.agent.replied_to,
                    )

                    self._activity.log(
                        "inbox_processing_end",
                        summary=result.summary[:200],
                        meta={"senders": list(inbox_result.senders), "count": inbox_result.unread_count},
                    )

                    logger.info(
                        "[%s] process_inbox_message END duration_ms=%d unread=%d",
                        self.name, result.duration_ms, inbox_result.unread_count,
                    )
                    return result

                except Exception as exc:
                    logger.exception("[%s] process_inbox_message FAILED", self.name)
                    # Archive on crash to prevent re-processing storms
                    if inbox_result is not None and inbox_result.inbox_items:
                        try:
                            self.messenger.archive_paths(inbox_result.inbox_items)
                        except Exception:
                            logger.warning(
                                "[%s] Failed to crash-archive inbox messages",
                                self.name, exc_info=True,
                            )
                    self._activity.log(
                        "error",
                        summary=t("anima.inbox_error", exc=type(exc).__name__),
                        meta={"phase": "process_inbox_message", "error": str(exc)[:200]},
                    )
                    raise
                finally:
                    self._status_slots["inbox"] = "idle"
                    self._task_slots["inbox"] = ""
        finally:
            self._notify_lock_released()

    # ── Heartbeat private methods ──────────────────────────

    def _build_prior_messages(
        self, prompt_text: str,
    ) -> list[dict[str, Any]] | None:
        """Build prior_messages for A mode, None for S/B."""
        mode = self.agent.execution_mode
        if mode != "a":
            return None
        conv = ConversationMemory(self.anima_dir, self.model_config)
        return conv.build_structured_messages(prompt_text)

    def _build_background_context_parts(self) -> list[str]:
        """Build shared context parts for background-auto sessions (heartbeat/cron).

        Collects: recovery note, background task notifications, heartbeat
        history, reflections, dialogue context, subordinate check.
        """
        parts: list[str] = []

        # ── Recovery note from previous failed heartbeat ──
        recovery_note_path = self.anima_dir / "state" / "recovery_note.md"
        if recovery_note_path.exists():
            try:
                recovery_content = recovery_note_path.read_text(encoding="utf-8")
                parts.append(
                    load_prompt("fragments/recovery_note_header") + "\n\n" + recovery_content
                )
                recovery_note_path.unlink(missing_ok=True)
                logger.info("[%s] Recovery note loaded and removed", self.name)
            except Exception:
                logger.debug("[%s] Failed to read recovery note", self.name, exc_info=True)

        # Inject pending background task notifications
        bg_notifications = self.drain_background_notifications()
        if bg_notifications:
            notif_text = "\n\n".join(bg_notifications)
            parts.append(
                load_prompt("fragments/bg_task_notification") + "\n\n" + notif_text
            )

        # Inject recent heartbeat history for continuity
        history_text = self._load_heartbeat_history()
        if history_text:
            parts.append(load_prompt(
                "heartbeat_history", history=history_text,
            ))

        # Inject recent reflections for cognitive continuity
        reflection_text = self._load_recent_reflections()
        if reflection_text:
            parts.append(
                load_prompt("fragments/recent_reflections") + "\n\n" + reflection_text
            )

        # Inject recent dialogue context for cross-session continuity
        try:
            conv_mem = ConversationMemory(self.anima_dir, self.model_config)
            state = conv_mem.load()
            recent_turns = state.turns[-5:] if state.turns else []
            if recent_turns:
                conv_lines = []
                for turn in recent_turns:
                    snippet = turn.content[:200]
                    conv_lines.append(f"- [{turn.role}] {snippet}")
                conv_summary = "\n".join(conv_lines)
                parts.append(
                    t("agent.recent_dialogue_header") + "\n\n"
                    + t("agent.recent_dialogue_intro")
                    + "\n"
                    + t("agent.recent_dialogue_consider") + "\n\n"
                    + conv_summary
                )
        except Exception:
            logger.debug("[%s] Failed to load dialogue context", self.name, exc_info=True)

        # ── Subordinate management check for animas with subordinates ──
        try:
            from core.config.models import load_config
            from core.paths import get_animas_dir
            _cfg = load_config()
            _subordinates = [
                _name for _name, _pcfg in _cfg.animas.items()
                if _pcfg.supervisor == self.name
            ]
            if _subordinates:
                parts.append(load_prompt(
                    "heartbeat_subordinate_check",
                    subordinates=", ".join(_subordinates),
                    animas_dir=str(get_animas_dir()),
                ))
        except Exception:
            logger.debug(
                "[%s] Failed to inject delegation check", self.name,
                exc_info=True,
            )

        return parts

    async def _build_heartbeat_prompt(self) -> list[str]:
        """Build heartbeat prompt parts.

        Heartbeat-specific header + shared background context.
        """
        hb_config = self.memory.read_heartbeat_config()
        checklist = hb_config or load_prompt("heartbeat_default_checklist")
        task_delegation_rules = load_prompt("task_delegation_rules")
        parts = [load_prompt("heartbeat", checklist=checklist, task_delegation_rules=task_delegation_rules)]

        parts.extend(self._build_background_context_parts())

        return parts

    def _build_cron_prompt(
        self, task_name: str, description: str, command_output: str | None = None,
    ) -> str:
        """Build cron task prompt with heartbeat-equivalent context.

        Args:
            task_name: Cron task name from cron.md.
            description: Task description or instruction.
            command_output: Optional stdout from a preceding command-type cron.
        """
        parts: list[str] = []

        # Cron task header
        cron_prompt = load_prompt(
            "cron_task", task_name=task_name, description=description,
        )
        if cron_prompt:
            parts.append(cron_prompt)

        # Inject command output if this is a follow-up to a command cron
        if command_output:
            parts.append(load_prompt("fragments/command_output", output=command_output))

        # Shared background context (same as heartbeat)
        parts.extend(self._build_background_context_parts())

        return "\n\n".join(parts)

    async def _process_inbox_messages(
        self,
        cascade_suppressed_senders: set[str] | None = None,
    ) -> InboxResult:
        """Read, filter, deduplicate, format, and record inbox messages.

        Handles cascade suppression, MessageDeduplicator, retry counter,
        episode recording, and activity logging.
        """
        if not self.messenger.has_unread():
            return InboxResult()

        inbox_items = self.messenger.receive_with_paths()
        messages = [item.msg for item in inbox_items]
        unread_count = len(messages)
        senders: set[str] = {m.from_person for m in messages}

        # ── Filter cascade-suppressed senders ──
        if cascade_suppressed_senders:
            suppressed_items = [
                item for item in inbox_items
                if item.msg.from_person in cascade_suppressed_senders
            ]
            inbox_items = [
                item for item in inbox_items
                if item.msg.from_person not in cascade_suppressed_senders
            ]
            messages = [item.msg for item in inbox_items]
            if suppressed_items:
                logger.info(
                    "[%s] Cascade-suppressed %d messages from %s",
                    self.name, len(suppressed_items),
                    ", ".join(cascade_suppressed_senders & senders),
                )
            senders = {m.from_person for m in messages}
            unread_count = len(messages)

        # ── Message deduplication (Phase 4) ──
        try:
            from core.memory.dedup import MessageDeduplicator
            dedup = MessageDeduplicator(self.anima_dir)

            # Load previously deferred messages and prepend to inbox
            deferred_raw = dedup.load_deferred()
            if deferred_raw:
                from core.schemas import Message as _Msg
                for raw in deferred_raw:
                    try:
                        deferred_msg = _Msg(
                            from_person=raw.get("from", "unknown"),
                            to_person=self.name,
                            content=raw.get("content", ""),
                            type=raw.get("type", "message"),
                        )
                        messages.append(deferred_msg)
                    except Exception:
                        logger.debug("[%s] Skipping invalid deferred message", self.name)
                logger.info("[%s] Restored %d deferred messages", self.name, len(deferred_raw))

            # Apply rate limiting first (before consolidation)
            messages, rate_deferred = dedup.apply_rate_limit(messages)
            if rate_deferred:
                dedup.archive_suppressed(rate_deferred)

            # Consolidate same-sender messages
            messages, consolidated_suppressed = dedup.consolidate_messages(messages)
            if consolidated_suppressed:
                dedup.archive_suppressed(consolidated_suppressed)

            # Suppress resolved topics
            try:
                resolutions = self.memory.read_resolutions(days=7)
            except Exception:
                resolutions = []
            if resolutions:
                filtered = []
                for m in messages:
                    if dedup.is_resolved_topic(m.content, resolutions):
                        dedup.archive_suppressed([m])
                    else:
                        filtered.append(m)
                messages = filtered

            # Update counts after dedup
            unread_count = len(messages)
            senders = {m.from_person for m in messages}
        except Exception:
            logger.debug("[%s] Message dedup failed, using original messages", self.name, exc_info=True)

        logger.info(
            "[%s] Processing %d unread messages in heartbeat (senders: %s)",
            self.name, unread_count, ", ".join(senders),
        )

        # ── Retry counter: track how many times each inbox message is presented ──
        _read_counts_path = self.anima_dir / "state" / "inbox_read_counts.json"
        _read_counts: dict[str, int] = {}
        try:
            if _read_counts_path.exists():
                _read_counts = json.loads(
                    _read_counts_path.read_text(encoding="utf-8")
                )
        except Exception:
            _read_counts = {}

        for item in inbox_items:
            key = item.path.name
            _read_counts[key] = _read_counts.get(key, 0) + 1

        # Prune entries for inbox files that no longer exist
        inbox_dir = self.anima_dir.parent.parent / "shared" / "inbox" / self.name
        _read_counts = {
            k: v for k, v in _read_counts.items()
            if (inbox_dir / k).exists()
        }

        try:
            _read_counts_path.write_text(
                json.dumps(_read_counts, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("[%s] Failed to write inbox_read_counts", self.name, exc_info=True)

        # Format messages with retry annotations
        prompt_parts: list[str] = []
        lines: list[str] = []
        for item in inbox_items:
            m = item.msg
            count = _read_counts.get(item.path.name, 1)
            if count >= 2:
                prefix = t("anima.unread_prefix", from_person=m.from_person, count=count)
            else:
                prefix = f"- {m.from_person}: "
            lines.append(f"{prefix}{m.content[:800]}")
        # Deferred messages (no InboxItem) are appended without counter
        for m in messages:
            if not any(item.msg is m for item in inbox_items):
                lines.append(f"- {m.from_person}: {m.content[:800]}")
        summary = "\n".join(lines)
        prompt_parts.append(load_prompt("unread_messages", summary=summary))

        # Record received message content to episodes so that
        # inter-Anima communications survive in episodic memory.
        _msg_ts = now_jst().strftime("%H:%M")
        _recordable = [m for m in messages if m.type != "ack"]
        if len(_recordable) > 50:
            logger.warning(
                "[%s] DM burst: %d messages, recording first 50",
                self.name, len(_recordable),
            )
        for _m in _recordable[:50]:
            _episode = t(
                "anima.msg_received_episode",
                ts=_msg_ts,
                from_person=_m.from_person,
                content=_m.content[:1000],
            ) + "\n"
            try:
                self.memory.append_episode(_episode)
            except Exception:
                logger.debug(
                    "[%s] Failed to record message episode from %s",
                    self.name, _m.from_person, exc_info=True,
                )

        # Activity log: message received from Anima (full content, summary truncated)
        for _m in _recordable[:50]:
            self._activity.log("message_received", content=_m.content, summary=_m.content[:200], from_person=_m.from_person, to_person=self.name, meta={"from_type": "anima"})

        return InboxResult(
            inbox_items=inbox_items,
            messages=messages,
            senders=senders,
            unread_count=unread_count,
            prompt_parts=prompt_parts,
        )

    async def _execute_heartbeat_cycle(
        self,
        prompt: str,
        inbox_items: list[InboxItem],
        unread_count: int,
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> CycleResult:
        """Write checkpoint, execute agent cycle, record results.

        Args:
            prompt: The heartbeat prompt text.
            inbox_items: Inbox items being processed.
            unread_count: Number of unread messages.
            prior_messages: Structured conversation history for Mode A.

        Returns the CycleResult from the agent execution.
        """
        # ── Heartbeat Checkpoint ──
        checkpoint_path = self.anima_dir / "state" / "heartbeat_checkpoint.json"
        try:
            checkpoint_data = {
                "ts": now_iso(),
                "trigger": "heartbeat",
                "unread_count": unread_count,
            }
            checkpoint_path.write_text(
                json.dumps(checkpoint_data, ensure_ascii=False), encoding="utf-8",
            )
        except Exception:
            logger.debug("[%s] Failed to write heartbeat checkpoint", self.name, exc_info=True)

        # Reset reply tracking before the cycle
        self.agent.reset_reply_tracking(session_type="background")
        self.agent.reset_posted_channels(session_type="background")
        # Clear replied_to persistence file
        _replied_to_path = self.anima_dir / "run" / "replied_to.jsonl"
        if _replied_to_path.exists():
            _replied_to_path.unlink(missing_ok=True)

        accumulated_text = ""
        result: CycleResult | None = None

        # Streaming journal for heartbeat crash recovery
        journal = StreamingJournal(self.anima_dir, session_type="heartbeat")
        journal.open(trigger="heartbeat")

        try:
            async for chunk in self.agent.run_cycle_streaming(
                prompt, trigger="heartbeat",
                prior_messages=prior_messages,
            ):
                # Relay text_delta chunks to waiting user stream
                if chunk.get("type") == "text_delta":
                    accumulated_text += chunk.get("text", "")
                    journal.write_text(chunk.get("text", ""))

                if chunk.get("type") == "cycle_done":
                    cycle_result = chunk.get("cycle_result", {})
                    result = CycleResult(
                        trigger=cycle_result.get("trigger", "heartbeat"),
                        action=cycle_result.get("action", "responded"),
                        summary=cycle_result.get("summary", ""),
                        duration_ms=cycle_result.get("duration_ms", 0),
                        context_usage_ratio=cycle_result.get(
                            "context_usage_ratio", 0.0
                        ),
                        session_chained=cycle_result.get(
                            "session_chained", False
                        ),
                        total_turns=cycle_result.get("total_turns", 0),
                    )
                    journal.finalize(summary=result.summary[:500])

            if result is None:
                result = CycleResult(
                    trigger="heartbeat",
                    action="responded",
                    summary=accumulated_text or "(no result)",
                )

            self._last_activity = now_jst()

            # Activity log: heartbeat end
            self._activity.log("heartbeat_end", summary=result.summary)

            # Session boundary: finalize pending conversation turns
            try:
                conv_mem = ConversationMemory(self.anima_dir, self.model_config)
                await conv_mem.finalize_if_session_ended()
            except Exception:
                logger.debug("[%s] finalize_if_session_ended failed", self.name, exc_info=True)

            # A-3: Record important heartbeat actions to episodes
            if result.summary and "HEARTBEAT_OK" not in result.summary:
                ts = now_jst().strftime("%H:%M")
                episode_entry = t(
                    "anima.heartbeat_episode",
                    ts=ts,
                    summary=result.summary[:500],
                )
                if unread_count > 0:
                    episode_entry += t("anima.heartbeat_msgs_processed", count=unread_count)

                # A-3b: Extract and record reflection from accumulated text
                reflection_text = _extract_reflection(accumulated_text)
                if reflection_text and len(reflection_text) >= _MIN_REFLECTION_LENGTH:
                    episode_entry += (
                        f"\n\n[REFLECTION]\n{reflection_text}\n[/REFLECTION]"
                    )
                    self._activity.log(
                        "heartbeat_reflection",
                        content=reflection_text,
                        summary=reflection_text[:200],
                    )

                try:
                    self.memory.append_episode(episode_entry)
                except Exception:
                    logger.debug("[%s] Failed to record heartbeat episode", self.name, exc_info=True)

            logger.info(
                "[%s] run_heartbeat END duration_ms=%d unread_processed=%d",
                self.name, result.duration_ms, unread_count,
            )
            # Heartbeat completed successfully — remove checkpoint
            try:
                checkpoint_path.unlink(missing_ok=True)
            except Exception:
                logger.debug("[%s] Failed to remove heartbeat checkpoint", self.name, exc_info=True)

            return result
        finally:
            journal.close()

    async def _archive_processed_messages(
        self,
        inbox_items: list[InboxItem],
        senders: set[str],
        replied_to: set[str],
    ) -> None:
        """Archive replied-to messages; force-archive stale unreplied messages.

        Messages from unreplied senders stay in inbox for the next
        heartbeat cycle.
        """
        unreplied = senders - replied_to

        items_to_archive = [
            item for item in inbox_items
            if item.msg.from_person in replied_to
            or item.msg.from_person not in senders  # system msgs
        ]
        items_to_keep = [
            item for item in inbox_items
            if item not in items_to_archive
        ]

        # Safety: force-archive messages that have been sitting
        # in inbox longer than _STALE_MESSAGE_TIMEOUT_SEC to
        # prevent re-processing storms even if replied_to
        # tracking fails.
        if items_to_keep:
            now = time.time()
            stale: list[InboxItem] = []
            for item in items_to_keep:
                try:
                    mtime = item.path.stat().st_mtime
                    if (now - mtime) > _STALE_MESSAGE_TIMEOUT_SEC:
                        stale.append(item)
                except FileNotFoundError:
                    continue  # already archived/deleted
            if stale:
                logger.warning(
                    "[%s] Force-archiving %d stale unreplied "
                    "messages (>%ds old)",
                    self.name, len(stale),
                    _STALE_MESSAGE_TIMEOUT_SEC,
                )
                items_to_archive.extend(stale)
                items_to_keep = [
                    i for i in items_to_keep if i not in stale
                ]

        if unreplied and items_to_keep:
            logger.warning(
                "[%s] Unreplied messages from %s will remain "
                "in inbox for next heartbeat cycle",
                self.name, ", ".join(unreplied),
            )

        total_archived = self.messenger.archive_paths(
            items_to_archive
        )
        logger.info(
            "[%s] Archived %d/%d messages "
            "(kept %d unreplied in inbox)",
            self.name, total_archived, len(inbox_items),
            len(items_to_keep),
        )

    async def _handle_heartbeat_failure(
        self,
        error: Exception,
        inbox_items: list[InboxItem],
        unread_count: int,
    ) -> None:
        """Handle heartbeat failure: crash-archive, log error, save recovery note."""
        logger.exception("[%s] run_heartbeat FAILED", self.name)

        # Archive inbox messages even on crash to prevent
        # re-processing storms on next heartbeat.
        if inbox_items:
            try:
                crash_archived = self.messenger.archive_paths(inbox_items)
                logger.info(
                    "[%s] Crash-archived %d/%d inbox messages",
                    self.name, crash_archived, len(inbox_items),
                )
            except Exception:
                logger.warning(
                    "[%s] Failed to crash-archive inbox messages",
                    self.name, exc_info=True,
                )

        # Activity log: error
        self._activity.log(
            "error",
            summary=t("anima.heartbeat_error", exc=type(error).__name__),
            meta={"phase": "run_heartbeat", "error": str(error)[:200]},
        )

        # ── Save recovery note for next heartbeat ──
        try:
            recovery_path = self.anima_dir / "state" / "recovery_note.md"
            recovery_content = t(
                "anima.recovery_error_info",
                exc_type=type(error).__name__,
                exc_msg=str(error)[:200],
                ts=now_iso(),
                count=unread_count,
            )
            recovery_path.write_text(recovery_content, encoding="utf-8")
            logger.info("[%s] Recovery note saved", self.name)
        except Exception:
            logger.debug("[%s] Failed to save recovery note", self.name, exc_info=True)

        # Clean up orphaned streaming journal in-process so that
        # the next restart does not misreport it as a "crash recovery".
        # The partial heartbeat text is intentionally discarded:
        # heartbeat output is internal monologue, not a user-facing
        # response.  Side effects (tool calls, messages sent) are
        # already persisted independently.
        try:
            if StreamingJournal.has_orphan(self.anima_dir, session_type="heartbeat"):
                StreamingJournal.confirm_recovery(self.anima_dir, session_type="heartbeat")
                logger.info("[%s] Cleaned up orphaned streaming journal", self.name)
        except Exception:
            logger.debug(
                "[%s] Failed to clean up streaming journal",
                self.name, exc_info=True,
            )

    # ── run_heartbeat orchestrator ───────────────────────────

    def _trigger_pending_task_execution(self) -> None:
        """Signal PendingTaskExecutor to check for new tasks.

        Called after heartbeat completion to ensure tasks written
        during planning phase are picked up promptly.
        """
        pending_dir = self.anima_dir / "state" / "pending"
        if not pending_dir.exists():
            return
        task_files = list(pending_dir.glob("*.json"))
        if task_files:
            logger.info(
                "[%s] %d pending tasks found after heartbeat, signaling executor",
                self.name, len(task_files),
            )
            if self._pending_executor is not None:
                self._pending_executor.wake()

    async def run_heartbeat(
        self,
        cascade_suppressed_senders: set[str] | None = None,
    ) -> CycleResult:
        self._interrupt_event.clear()
        logger.info("[%s] run_heartbeat START", self.name)
        try:
            async with self._background_lock:
                self._status_slots["background"] = "checking"
                self._last_heartbeat = now_jst()

                # Activity log: heartbeat start
                self._activity.log("heartbeat_start", summary=t("anima.heartbeat_start"))

                try:
                    # 1. Build prompt parts
                    parts = await self._build_heartbeat_prompt()

                    # 2. Warn if unread messages exist (inbox handled by Path A)
                    if self.messenger.has_unread():
                        logger.warning(
                            "[%s] Unread messages found during heartbeat — "
                            "inbox processing is handled by Path A (process_inbox_message)",
                            self.name,
                        )

                    # 3. Execute agent cycle (plan-only, no inbox)
                    from core.tooling.handler import active_session_type
                    _session_token = self.agent._tool_handler.set_active_session_type("background")
                    heartbeat_text = "\n\n".join(parts)
                    prior_msgs = self._build_prior_messages(heartbeat_text)
                    try:
                        result = await self._execute_heartbeat_cycle(
                            heartbeat_text, [], 0,
                            prior_messages=prior_msgs,
                        )
                    finally:
                        active_session_type.reset(_session_token)

                    return result

                except Exception as exc:
                    await self._handle_heartbeat_failure(exc, [], 0)
                    raise
                finally:
                    self._status_slots["background"] = "idle"
                    self._task_slots["background"] = ""
        finally:
            self._notify_lock_released()
            # Signal pending task execution after heartbeat completes
            self._trigger_pending_task_execution()

    # ── Consolidation helpers ──────────────────────────────────

    def _collect_episodes_summary(self) -> tuple[str, str, str]:
        """Collect recent episodes, resolved events, and activity log as formatted text.

        Returns:
            Tuple of (episodes_summary, resolved_events_summary, activity_log_summary).
            If no episodes are found, returns a placeholder message for episodes
            with empty strings for the other summaries.
        """
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(self.anima_dir, self.name)
        episodes = engine._collect_recent_episodes(hours=24)
        resolved = engine._collect_resolved_events(hours=24)
        activity_log_summary = engine._collect_activity_entries(hours=24)

        # Format episodes
        if episodes:
            episodes_summary = "\n\n".join(
                f"## {e['date']} {e['time']}\n{e['content']}"
                for e in episodes
            )
        else:
            return (t("anima.no_episodes_today"), "", activity_log_summary)

        # Format resolved events
        if resolved:
            resolved_events_summary = "\n".join(
                f"- {r['ts'][:16]}: {r['content']}" for r in resolved
            )
        else:
            resolved_events_summary = ""

        return (episodes_summary, resolved_events_summary, activity_log_summary)

    def count_recent_episodes(self, hours: int = 24) -> int:
        """Count recent episode entries within the given time window.

        Used by lifecycle.py to skip consolidation when no episodes exist.

        Args:
            hours: Number of hours to look back.

        Returns:
            Number of episode entries found.
        """
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(self.anima_dir, self.name)
        episodes = engine._collect_recent_episodes(hours=hours)
        return len(episodes)

    async def run_consolidation(
        self,
        consolidation_type: str = "daily",
        max_turns: int = 30,
    ) -> CycleResult:
        """Run memory consolidation as an Anima-driven task.

        The Anima uses its own tools (search_memory, read_memory_file,
        write_memory_file, archive_memory_file) to organize, consolidate,
        and clean up its memories within a tool-call loop.

        Works with all execution modes: S, A, and B.

        Args:
            consolidation_type: "daily" or "weekly"
            max_turns: Maximum tool-call loop iterations for this task
        """
        logger.info(
            "[%s] run_consolidation START type=%s max_turns=%d",
            self.name, consolidation_type, max_turns,
        )
        from core.tooling.handler import active_session_type
        try:
            async with self._background_lock:
                self._status_slots["background"] = "consolidating"
                self._task_slots["background"] = f"Memory consolidation ({consolidation_type})"
                _session_token = self.agent._tool_handler.set_active_session_type("background")

                try:
                    # Build consolidation prompt
                    if consolidation_type == "daily":
                        episodes_summary, resolved_events_summary, activity_log_summary = (
                            self._collect_episodes_summary()
                        )
                        prompt = load_prompt(
                            "memory/consolidation_instruction",
                            anima_name=self.name,
                            episodes_summary=episodes_summary,
                            resolved_events_summary=resolved_events_summary,
                            activity_log_summary=activity_log_summary or t("anima.no_activity_log"),
                        )
                    else:
                        prompt = load_prompt(
                            "memory/weekly_consolidation_instruction",
                            anima_name=self.name,
                        )

                    # Activity log
                    self._activity.log(
                        "consolidation_start",
                        summary=t("anima.consolidation_start", type=consolidation_type),
                    )

                    result = await self.agent.run_cycle(
                        prompt,
                        trigger=f"consolidation:{consolidation_type}",
                        message_intent="request",
                        max_turns_override=max_turns,
                    )
                    self._last_activity = now_jst()

                    # Activity log: completion
                    self._activity.log(
                        "consolidation_end",
                        summary=t("anima.consolidation_end", type=consolidation_type),
                        content=result.summary[:500] if result.summary else "",
                        meta={
                            "type": consolidation_type,
                            "duration_ms": result.duration_ms,
                        },
                    )

                    logger.info(
                        "[%s] run_consolidation END type=%s duration_ms=%d",
                        self.name, consolidation_type, result.duration_ms,
                    )
                    return result

                except Exception as exc:
                    logger.exception(
                        "[%s] run_consolidation FAILED type=%s",
                        self.name, consolidation_type,
                    )
                    self._activity.log(
                        "error",
                        summary=t("anima.consolidation_error", exc=type(exc).__name__),
                        meta={"phase": "run_consolidation", "error": str(exc)[:200]},
                    )
                    raise
                finally:
                    active_session_type.reset(_session_token)
                    self._status_slots["background"] = "idle"
                    self._task_slots["background"] = ""
        finally:
            self._notify_lock_released()

    async def run_cron_task(
        self,
        task_name: str,
        description: str,
        command_output: str | None = None,
    ) -> CycleResult:
        """Execute a cron LLM task with heartbeat-equivalent context.

        Args:
            task_name: Cron task name from cron.md.
            description: Task description/instruction.
            command_output: Optional stdout from a preceding command cron.
        """
        self._interrupt_event.clear()
        logger.info("[%s] run_cron_task START task=%s", self.name, task_name)
        from core.tooling.handler import active_session_type
        try:
            async with self._background_lock:
                self._status_slots["background"] = "working"
                self._task_slots["background"] = task_name
                _session_token = self.agent._tool_handler.set_active_session_type("background")

                prompt = self._build_cron_prompt(
                    task_name, description, command_output=command_output,
                )

                try:
                    result = await self.agent.run_cycle(
                        prompt, trigger=f"cron:{task_name}"
                    )
                    self._last_activity = now_jst()

                    # Record cron execution result
                    self.memory.append_cron_log(
                        task_name,
                        summary=result.summary,
                        duration_ms=result.duration_ms,
                    )

                    # Activity log: cron executed
                    self._activity.log(
                        "cron_executed",
                        summary=t("anima.cron_task_summary", task=task_name),
                        content=result.summary[:500] if result else "",
                        meta={
                            "task_name": task_name,
                            "duration_ms": result.duration_ms if result else 0,
                        },
                    )

                    logger.info(
                        "[%s] run_cron_task END task=%s duration_ms=%d",
                        self.name, task_name, result.duration_ms,
                    )
                    return result
                except Exception as exc:
                    logger.exception(
                        "[%s] run_cron_task FAILED task=%s", self.name, task_name,
                    )
                    # Activity log: error
                    self._activity.log(
                        "error",
                        summary=t("anima.cron_task_error", exc=type(exc).__name__),
                        meta={"phase": "run_cron_task", "error": str(exc)[:200]},
                    )
                    raise
                finally:
                    active_session_type.reset(_session_token)
                    self._status_slots["background"] = "idle"
                    self._task_slots["background"] = ""
        finally:
            self._notify_lock_released()

    async def run_cron_command(
        self,
        task_name: str,
        *,
        command: str | None = None,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a command-type cron task (bash or internal tool).

        Args:
            task_name: Task identifier for logging
            command: Bash command to execute (mutually exclusive with tool)
            tool: Internal tool name (mutually exclusive with command)
            args: Tool arguments (only used with tool)

        Returns:
            Dictionary with execution results (exit_code, stdout, stderr, duration_ms)
        """
        import time

        logger.info("[%s] run_cron_command START task=%s", self.name, task_name)
        start_ms = time.time_ns() // 1_000_000

        stdout = ""
        stderr = ""
        exit_code = 0

        from core.tooling.handler import active_session_type
        try:
            async with self._background_lock:
                self._status_slots["background"] = "working"
                self._task_slots["background"] = task_name
                _session_token = self.agent._tool_handler.set_active_session_type("background")

                try:
                    if command:
                        # Execute bash command
                        logger.debug("[%s] Executing bash: %s", self.name, command)
                        proc = await asyncio.create_subprocess_shell(
                            command,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        stdout_bytes, stderr_bytes = await proc.communicate()
                        stdout = stdout_bytes.decode("utf-8", errors="replace")
                        stderr = stderr_bytes.decode("utf-8", errors="replace")
                        exit_code = proc.returncode or 0

                    elif tool:
                        # Execute internal tool via ToolHandler
                        logger.debug("[%s] Executing tool: %s", self.name, tool)
                        result = self.agent._tool_handler.handle(tool, args or {})
                        stdout = str(result)
                        exit_code = 0

                    else:
                        stderr = "Neither command nor tool specified"
                        exit_code = 1

                    self._last_activity = now_jst()

                except Exception as exc:
                    stderr = f"{type(exc).__name__}: {exc}"
                    exit_code = 1
                    logger.exception(
                        "[%s] run_cron_command FAILED task=%s", self.name, task_name
                    )
                    # Activity log: error
                    self._activity.log(
                        "error",
                        summary=t("anima.cron_cmd_error", exc=type(exc).__name__),
                        meta={"phase": "run_cron_command", "error": str(exc)[:200]},
                    )
                finally:
                    active_session_type.reset(_session_token)
                    self._status_slots["background"] = "idle"
                    self._task_slots["background"] = ""

            duration_ms = (time.time_ns() // 1_000_000) - start_ms

            # Log to cron_log with command-specific format
            self.memory.append_cron_command_log(
                task_name,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
            )

            # Activity log: cron command executed (intentionally logs even on
            # failure — exit_code captures the error state, unlike run_cron_task
            # which re-raises and never reaches this point on error)
            self._activity.log(
                "cron_executed",
                summary=t("anima.cron_cmd_summary", task=task_name),
                meta={"task_name": task_name, "exit_code": exit_code, "command": command or "", "tool": tool or ""},
            )

            logger.info(
                "[%s] run_cron_command END task=%s exit_code=%d duration_ms=%d",
                self.name,
                task_name,
                exit_code,
                duration_ms,
            )

            return {
                "task": task_name,
                "exit_code": exit_code,
                "stdout": stdout[:1000],  # Preview for response
                "stderr": stderr[:1000],
                "duration_ms": duration_ms,
            }

        finally:
            self._notify_lock_released()
