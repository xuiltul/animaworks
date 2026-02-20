from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


import asyncio
import json
import logging
import re
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
from core.memory.conversation import ConversationMemory
from core.memory import MemoryManager
from core.messenger import InboxItem, Messenger
from core.paths import load_prompt
from core.exceptions import AnimaWorksError  # noqa: F401
from core.schemas import CycleResult, AnimaStatus, VALID_EMOTIONS

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

    def __init__(self, anima_dir: Path, shared_dir: Path) -> None:
        self.anima_dir = anima_dir
        self.name = anima_dir.name
        self._activity = ActivityLogger(anima_dir)

        self.memory = MemoryManager(anima_dir)
        self.model_config = self.memory.read_model_config()
        self.messenger = Messenger(shared_dir, self.name)
        self.agent = AgentCore(
            anima_dir, self.memory, self.model_config, self.messenger
        )

        self._lock = asyncio.Lock()
        self._user_waiting = asyncio.Event()
        # Event NOT set = no user waiting (default state)
        self._status = "idle"
        self._current_task = ""
        self._last_heartbeat: datetime | None = None
        self._last_activity: datetime | None = None
        self._on_lock_released: Callable[[], None] | None = None

        # Heartbeat SSE relay: allow process_message_stream to read
        # heartbeat's streaming chunks while waiting for the lock.
        self._heartbeat_stream_queue: asyncio.Queue | None = None
        self._heartbeat_context: str = ""

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

    def set_on_lock_released(self, fn: Callable[[], None]) -> None:
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
                        subject=f"バックグラウンドタスク完了: {task.tool_name}",
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
            subject = f"バックグラウンドタスク完了: {task.tool_name}"
            if task.status.value == "failed":
                subject = f"バックグラウンドタスク失敗: {task.tool_name}"

            # Write a notification file to the anima's own inbox-like location
            # so the next heartbeat can process it
            notif_dir = self.agent.anima_dir / "state" / "background_notifications"
            notif_dir.mkdir(parents=True, exist_ok=True)
            notif_path = notif_dir / f"{task.task_id}.md"
            notif_content = (
                f"# {subject}\n\n"
                f"- タスクID: {task.task_id}\n"
                f"- ツール: {task.tool_name}\n"
                f"- ステータス: {task.status.value}\n"
                f"- 結果: {summary}\n"
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
    def status(self) -> AnimaStatus:
        return AnimaStatus(
            name=self.name,
            status=self._status,
            current_task=self._current_task,
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
        try:
            async with self._lock:
                self._status = "bootstrapping"
                self._current_task = "Initial bootstrap"

                conv_memory = ConversationMemory(self.anima_dir, self.model_config)
                prompt = conv_memory.build_chat_prompt(
                    "あなたの bootstrap.md ファイルを読み、指示に従ってください。",
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
                    self._status = "idle"
                    self._current_task = ""
        finally:
            self._notify_lock_released()

    async def process_message(
        self,
        content: str,
        from_person: str = "human",
        images: list[dict[str, Any]] | None = None,
        attachment_paths: list[str] | None = None,
    ) -> str:
        logger.info(
            "[%s] process_message WAITING from=%s content_len=%d images=%d",
            self.name, from_person, len(content), len(images or []),
        )
        self._user_waiting.set()
        try:
            async with self._lock:
                logger.info(
                    "[%s] process_message START (lock acquired) from=%s",
                    self.name, from_person,
                )
                self._status = "thinking"
                self._current_task = f"Responding to {from_person}"

                # Build history-aware prompt via conversation memory
                conv_memory = ConversationMemory(self.anima_dir, self.model_config)
                await conv_memory.compress_if_needed()
                prompt = conv_memory.build_chat_prompt(content, from_person)

                # Pre-save: persist user input before agent execution
                conv_memory.append_turn(
                    "human", content, attachments=attachment_paths or [],
                )
                conv_memory.save()

                # Activity log: message received
                self._activity.log("message_received", content=content, summary=content[:100], from_person=from_person, channel="chat")

                try:
                    result = await self.agent.run_cycle(
                        prompt, trigger=f"message:{from_person}",
                        images=images,
                    )
                    self._last_activity = now_jst()

                    # Record assistant response in conversation memory
                    conv_memory.append_turn("assistant", result.summary)
                    conv_memory.save()

                    # Activity log: response sent
                    self._activity.log("response_sent", content=result.summary, to_person=from_person, channel="chat")

                    logger.info(
                        "[%s] process_message END duration_ms=%d",
                        self.name, result.duration_ms,
                    )
                    return result.summary
                except Exception as exc:
                    logger.exception("[%s] process_message FAILED", self.name)
                    # Activity log: error
                    self._activity.log(
                        "error",
                        summary=f"process_messageエラー: {type(exc).__name__}",
                        meta={"phase": "process_message", "error": str(exc)[:200]},
                    )
                    # Save error marker so the failed exchange is visible
                    conv_memory.append_turn(
                        "assistant", "[ERROR: エージェント実行中にエラーが発生しました]"
                    )
                    conv_memory.save()
                    raise
                finally:
                    self._status = "idle"
                    self._current_task = ""
        finally:
            self._user_waiting.clear()
            self._notify_lock_released()

    async def process_message_stream(
        self,
        content: str,
        from_person: str = "human",
        images: list[dict[str, Any]] | None = None,
        attachment_paths: list[str] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Streaming version of process_message.

        Yields stream event dicts. The lock is held for the entire duration.
        If bootstrapping is in progress (lock held + needs_bootstrap), yields
        an immediate "initializing" message instead of waiting.
        """
        # ── Bootstrap guard: return immediately if bootstrap is running ──
        if self.needs_bootstrap and self._lock.locked():
            logger.info(
                "[%s] process_message_stream REJECTED (bootstrapping) from=%s",
                self.name, from_person,
            )
            yield {
                "type": "bootstrap_busy",
                "message": "現在初期化中です。しばらくお待ちください。",
            }
            return

        logger.info(
            "[%s] process_message_stream WAITING from=%s content_len=%d images=%d",
            self.name, from_person, len(content), len(images or []),
        )
        self._user_waiting.set()
        try:
            # ── Heartbeat relay: stream heartbeat output while waiting ──
            if self._lock.locked():
                context_msg = self._heartbeat_context or "処理中です"
                logger.info(
                    "[%s] process_message_stream: lock held, starting heartbeat relay",
                    self.name,
                )
                yield {
                    "type": "heartbeat_relay_start",
                    "message": f"ハートビート処理中（{context_msg}）",
                }

                # Create a queue so run_heartbeat can push chunks to us
                self._heartbeat_stream_queue = asyncio.Queue()
                try:
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                self._heartbeat_stream_queue.get(), timeout=1.0,
                            )
                        except asyncio.TimeoutError:
                            # Check if lock was released while we waited
                            if not self._lock.locked():
                                break
                            continue
                        if chunk is None:
                            # Sentinel: heartbeat finished
                            break
                        yield {
                            "type": "heartbeat_relay",
                            "text": chunk.get("text", ""),
                        }
                finally:
                    self._heartbeat_stream_queue = None

                yield {"type": "heartbeat_relay_done"}
                logger.info(
                    "[%s] process_message_stream: heartbeat relay done",
                    self.name,
                )

            async with self._lock:
                logger.info(
                    "[%s] process_message_stream START (lock acquired) from=%s",
                    self.name, from_person,
                )
                self._status = "thinking"
                self._current_task = f"Responding to {from_person}"

                # Build history-aware prompt via conversation memory
                conv_memory = ConversationMemory(self.anima_dir, self.model_config)
                await conv_memory.compress_if_needed()
                prompt = conv_memory.build_chat_prompt(content, from_person)

                # Pre-save: persist user input before agent execution
                conv_memory.append_turn(
                    "human", content, attachments=attachment_paths or [],
                )
                conv_memory.save()

                # Activity log: message received
                self._activity.log("message_received", content=content, summary=content[:100], from_person=from_person, channel="chat")

                # Streaming journal: write-ahead log for crash recovery
                journal = StreamingJournal(self.anima_dir)
                journal.open(
                    trigger=f"message:{from_person}",
                    from_person=from_person,
                )

                partial_response = ""
                cycle_done = False

                try:
                    async for chunk in self.agent.run_cycle_streaming(
                        prompt, trigger=f"message:{from_person}",
                        images=images,
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
                            # Record assistant response in conversation memory
                            cycle_result = chunk.get("cycle_result", {})
                            summary = cycle_result.get("summary", "")
                            conv_memory.append_turn("assistant", summary)
                            conv_memory.save()

                            # Activity log: response sent
                            self._activity.log("response_sent", content=summary, to_person=from_person, channel="chat")

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
                    # Determine error code from exception type
                    exc_module = type(exc).__module__ or ""
                    exc_name = type(exc).__qualname__
                    if "tool" in exc_module.lower() or "tool" in exc_name.lower():
                        error_code = "TOOL_ERROR"
                    elif any(
                        k in exc_module.lower()
                        for k in ("anthropic", "openai", "litellm", "llm")
                    ):
                        error_code = "LLM_ERROR"
                    else:
                        error_code = "STREAM_ERROR"
                    # Activity log: error
                    self._activity.log(
                        "error",
                        summary=f"process_message_streamエラー: {type(exc).__name__}",
                        meta={"phase": "process_message_stream", "error_code": error_code, "error": str(exc)[:200]},
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
                            saved_text = partial_response + "\n[応答が中断されました]"
                        else:
                            saved_text = "[応答が中断されました]"
                        conv_memory.append_turn("assistant", saved_text)
                        conv_memory.save()
                    # Close journal (no-op if already finalized)
                    journal.close()
                    self._status = "idle"
                    self._current_task = ""
        finally:
            self._user_waiting.clear()
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
        async with self._lock:
            prev_status = self._status
            prev_task = self._current_task

            # Build greet prompt with current state
            status_text = prev_status if prev_status != "idle" else "待機中"
            task_text = prev_task if prev_task else "特になし"
            prompt = load_prompt(
                "greet", status=status_text, current_task=task_text,
            )

            self._status = "greeting"
            self._current_task = "Greeting user"

            conv_memory = ConversationMemory(self.anima_dir, self.model_config)

            # Record visit marker (user turn) before greeting
            visit_text = "[デスクを訪問]"
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
                    "assistant", "[ERROR: 挨拶生成中にエラーが発生しました]"
                )
                conv_memory.save()
                raise
            finally:
                self._status = prev_status
                self._current_task = prev_task

    # ── Heartbeat private methods ──────────────────────────

    async def _build_heartbeat_prompt(self) -> list[str]:
        """Build heartbeat prompt parts.

        Collects: heartbeat checklist, recovery note, background task
        notifications, heartbeat history, dialogue context, delegation check.
        """
        hb_config = self.memory.read_heartbeat_config()
        checklist = hb_config or load_prompt("heartbeat_default_checklist")
        parts = [load_prompt("heartbeat", checklist=checklist)]

        # ── Recovery note from previous failed heartbeat ──
        recovery_note_path = self.anima_dir / "state" / "recovery_note.md"
        if recovery_note_path.exists():
            try:
                recovery_content = recovery_note_path.read_text(encoding="utf-8")
                parts.append(
                    "## ⚠️ 前回のハートビート障害情報\n\n"
                    "前回のハートビートが中断しました。以下の情報を確認し、"
                    "未完了のタスクを優先的に処理してください。\n\n"
                    + recovery_content
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
                "## バックグラウンドタスク完了通知\n\n"
                "以下のバックグラウンドタスクが完了しました。"
                "結果を確認し、必要に応じて対応してください。\n\n"
                + notif_text
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
                "## 直近の振り返り（前回までの気づき）\n\n"
                "以下は過去のハートビートで得た気づきです。"
                "関連があれば今回の判断に活かしてください。\n\n"
                + reflection_text
            )

        # A-1: Inject recent dialogue context for cross-session continuity
        try:
            conv_mem = ConversationMemory(self.anima_dir, self.model_config)
            state = conv_mem.load()
            recent_turns = state.turns[-5:] if state.turns else []
            if recent_turns:
                conv_lines = []
                for t in recent_turns:
                    snippet = t.content[:200]
                    conv_lines.append(f"- [{t.role}] {snippet}")
                conv_summary = "\n".join(conv_lines)
                parts.append(
                    f"## 直近の対話履歴\n\n"
                    f"以下はユーザーとの直近の対話です。"
                    f"進行中のタスクや指示がある場合、この内容を考慮してください。\n\n"
                    f"{conv_summary}"
                )
        except Exception:
            logger.debug("[%s] Failed to load dialogue context for heartbeat", self.name, exc_info=True)

        # NOTE: Task queue is injected via builder.py (system prompt)
        # and priming Channel E. No separate heartbeat injection needed.

        # ── Delegation check for animas with subordinates ──
        try:
            from core.config.models import load_config
            _cfg = load_config()
            _subordinates = [
                _name for _name, _pcfg in _cfg.animas.items()
                if _pcfg.supervisor == self.name
            ]
            if _subordinates:
                parts.append(load_prompt(
                    "heartbeat_delegation_check",
                    subordinates=", ".join(_subordinates),
                ))
        except Exception:
            logger.debug(
                "[%s] Failed to inject delegation check", self.name,
                exc_info=True,
            )

        return parts

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
                prefix = f"- {m.from_person} [⚠️ 未返信{count}回目]: "
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
            _episode = (
                f"## {_msg_ts} {_m.from_person}からのメッセージ受信\n\n"
                f"**送信者**: {_m.from_person}\n"
                f"**内容**:\n{_m.content[:1000]}\n"
            )
            try:
                self.memory.append_episode(_episode)
            except Exception:
                logger.debug(
                    "[%s] Failed to record message episode from %s",
                    self.name, _m.from_person, exc_info=True,
                )

        # Activity log: DM received (full content, summary truncated)
        for _m in _recordable[:50]:
            self._activity.log("dm_received", content=_m.content, summary=_m.content[:200], from_person=_m.from_person, to_person=self.name)

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
    ) -> CycleResult:
        """Write checkpoint, execute agent cycle, record results.

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
        self.agent.reset_reply_tracking()
        # Clear replied_to persistence file
        _replied_to_path = self.anima_dir / "run" / "replied_to.jsonl"
        if _replied_to_path.exists():
            _replied_to_path.unlink(missing_ok=True)

        accumulated_text = ""
        result: CycleResult | None = None

        # Streaming journal for heartbeat crash recovery
        journal = StreamingJournal(self.anima_dir)
        journal.open(trigger="heartbeat")

        try:
            async for chunk in self.agent.run_cycle_streaming(
                prompt, trigger="heartbeat"
            ):
                # Relay text_delta chunks to waiting user stream
                if chunk.get("type") == "text_delta":
                    accumulated_text += chunk.get("text", "")
                    journal.write_text(chunk.get("text", ""))
                    queue = self._heartbeat_stream_queue
                    if queue is not None:
                        await queue.put(chunk)

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

            # Send sentinel to close the relay queue
            queue = self._heartbeat_stream_queue
            if queue is not None:
                await queue.put(None)

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
                episode_entry = (
                    f"## {ts} ハートビート活動\n\n"
                    f"{result.summary[:500]}"
                )
                if unread_count > 0:
                    episode_entry += (
                        f"\n\n（{unread_count}件のメッセージを処理）"
                    )

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
            summary=f"run_heartbeatエラー: {type(error).__name__}",
            meta={"phase": "run_heartbeat", "error": str(error)[:200]},
        )

        # ── Save recovery note for next heartbeat ──
        try:
            recovery_path = self.anima_dir / "state" / "recovery_note.md"
            recovery_content = (
                f"### エラー情報\n\n"
                f"- エラー種別: {type(error).__name__}\n"
                f"- エラー内容: {str(error)[:200]}\n"
                f"- 発生日時: {now_iso()}\n"
                f"- 未処理メッセージ数: {unread_count}\n"
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
            if StreamingJournal.has_orphan(self.anima_dir):
                StreamingJournal.confirm_recovery(self.anima_dir)
                logger.info("[%s] Cleaned up orphaned streaming journal", self.name)
        except Exception:
            logger.debug(
                "[%s] Failed to clean up streaming journal",
                self.name, exc_info=True,
            )

        # Send sentinel to close the relay queue on error
        queue = self._heartbeat_stream_queue
        if queue is not None:
            await queue.put(None)

    # ── run_heartbeat orchestrator ───────────────────────────

    async def run_heartbeat(
        self,
        cascade_suppressed_senders: set[str] | None = None,
    ) -> CycleResult:
        # Defer to user messages: skip heartbeat if a user is waiting for the lock
        if self._user_waiting.is_set():
            logger.info("[%s] run_heartbeat SKIPPED: user message waiting", self.name)
            return CycleResult(
                trigger="heartbeat",
                action="skipped",
                summary="User message priority: heartbeat deferred",
            )

        logger.info("[%s] run_heartbeat START", self.name)
        try:
            async with self._lock:
                self._status = "checking"
                self._last_heartbeat = now_jst()
                inbox_items: list[InboxItem] = []
                unread_count = 0

                # Activity log: heartbeat start
                self._activity.log("heartbeat_start", summary="定期巡回開始")

                try:
                    # 1. Build prompt parts
                    parts = await self._build_heartbeat_prompt()

                    # 2. Process inbox messages
                    inbox_result = await self._process_inbox_messages(
                        cascade_suppressed_senders,
                    )
                    inbox_items = inbox_result.inbox_items
                    unread_count = inbox_result.unread_count
                    parts.extend(inbox_result.prompt_parts)

                    # Set heartbeat context for relay
                    if unread_count > 0:
                        self._heartbeat_context = (
                            f"{', '.join(inbox_result.senders)}からのメッセージを確認中"
                        )
                    else:
                        self._heartbeat_context = "定期巡回中"

                    # 3. Execute agent cycle
                    result = await self._execute_heartbeat_cycle(
                        "\n\n".join(parts),
                        inbox_items,
                        unread_count,
                    )

                    # 4. Archive processed messages
                    if unread_count > 0:
                        await self._archive_processed_messages(
                            inbox_items,
                            inbox_result.senders,
                            self.agent.replied_to,
                        )

                    return result

                except Exception as exc:
                    await self._handle_heartbeat_failure(
                        exc, inbox_items, unread_count,
                    )
                    raise
                finally:
                    self._status = "idle"
                    self._current_task = ""
                    self._heartbeat_context = ""
        finally:
            self._notify_lock_released()

    async def run_cron_task(
        self, task_name: str, description: str
    ) -> CycleResult:
        logger.info("[%s] run_cron_task START task=%s", self.name, task_name)
        try:
            async with self._lock:
                self._status = "working"
                self._current_task = task_name

                prompt = load_prompt(
                    "cron_task", task_name=task_name, description=description
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
                        summary=f"タスク: {task_name}",
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
                        summary=f"run_cron_taskエラー: {type(exc).__name__}",
                        meta={"phase": "run_cron_task", "error": str(exc)[:200]},
                    )
                    raise
                finally:
                    self._status = "idle"
                    self._current_task = ""
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

        try:
            async with self._lock:
                self._status = "working"
                self._current_task = task_name

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
                        summary=f"run_cron_commandエラー: {type(exc).__name__}",
                        meta={"phase": "run_cron_command", "error": str(exc)[:200]},
                    )
                finally:
                    self._status = "idle"
                    self._current_task = ""

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
                summary=f"コマンド: {task_name}",
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
