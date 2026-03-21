from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""InboxMixin -- Anima-to-Anima message (inbox) processing.

Extracted from ``core.anima.DigitalAnima`` as a Mixin.  All ``self``
references are resolved at runtime via MRO when mixed into ``DigitalAnima``.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.execution._sanitize import (
    ORIGIN_ANIMA,
    ORIGIN_EXTERNAL_PLATFORM,
    ORIGIN_HUMAN,
    ORIGIN_UNKNOWN,
    resolve_trust,
)
from core.i18n import t
from core.memory.streaming_journal import StreamingJournal
from core.messenger import InboxItem
from core.paths import load_prompt
from core.schemas import CycleResult
from core.time_utils import now_local

logger = logging.getLogger("animaworks.anima")

_SOURCE_TO_ORIGIN: dict[str, str] = {
    "slack": ORIGIN_EXTERNAL_PLATFORM,
    "chatwork": ORIGIN_EXTERNAL_PLATFORM,
    "human": ORIGIN_HUMAN,
    "anima": ORIGIN_ANIMA,
}

_RE_THREAD_CTX = re.compile(
    r"(\[Thread context[^\]]*\].*?\[/Thread context\]\s*)",
    re.DOTALL,
)
_THREAD_CTX_BUDGET = 300
_MSG_BODY_BUDGET = 2000


def _truncate_with_thread_ctx(
    content: str,
    *,
    ctx_budget: int = _THREAD_CTX_BUDGET,
    body_budget: int = _MSG_BODY_BUDGET,
) -> str:
    """Truncate content while preserving the user message after thread context.

    When ``[Thread context]...[/Thread context]`` markers are present,
    the context portion is truncated to *ctx_budget* characters and the
    actual user message is kept up to *body_budget* characters.  Without
    markers, the whole content is truncated to *body_budget*.
    """
    m = _RE_THREAD_CTX.match(content)
    if not m:
        return content[:body_budget]
    ctx_part = m.group(1)
    body_part = content[m.end() :]
    if len(ctx_part) > ctx_budget:
        ctx_part = ctx_part[:ctx_budget].rstrip() + "...\n"
    return ctx_part + body_part[:body_budget]


def _build_reply_instruction(m: Any) -> str:
    """Build platform-specific reply instruction metadata for external messages.

    Returns a formatted ``[reply_instruction: ...]`` line that the LLM can
    copy-paste to reply via the correct platform, channel, and thread.
    """
    if m.source == "slack":
        mention = f"<@{m.external_user_id}> " if m.external_user_id else ""
        cmd = f"animaworks-tool slack send '{m.external_channel_id}' '{mention}{{返信内容}}'"
        thread_id = m.external_thread_ts or m.source_message_id
        if thread_id:
            cmd += f" --thread {thread_id}"
        return f"  [reply_instruction: {cmd}]"

    if m.source == "chatwork":
        cmd = f"animaworks-tool chatwork send {m.external_channel_id} '{{返信内容}}'"
        return f"  [reply_instruction: {cmd}]"

    return ""


# ── Delegation DM framework-level helpers ────────────────────

_RE_TASK_ID = re.compile(r"(?:タスクID|Task ID):\s*([a-f0-9]{12})")


def _extract_task_id(msg: Any) -> str | None:
    """Extract task_id from Message meta (preferred) or content regex fallback."""
    if hasattr(msg, "meta") and isinstance(msg.meta, dict):
        tid = msg.meta.get("task_id")
        if tid:
            return tid
    m = _RE_TASK_ID.search(getattr(msg, "content", ""))
    return m.group(1) if m else None


def _split_delegation_items(
    inbox_items: list[InboxItem],
    messages: list[Any],
) -> tuple[list[InboxItem], list[InboxItem]]:
    """Split inbox items into delegation and non-delegation groups.

    Delegation items where task_id cannot be extracted are kept in
    non-delegation (safe fallback: let LLM handle them).
    """
    delegation: list[InboxItem] = []
    non_delegation: list[InboxItem] = []
    for item in inbox_items:
        if item.msg.intent == "delegation" and _extract_task_id(item.msg):
            delegation.append(item)
        else:
            non_delegation.append(item)
    return delegation, non_delegation


def _check_task_state(anima_dir: Path, task_id: str) -> str:
    """Check task execution state. Returns one of:
    'completed', 'processing', 'pending', 'terminal', 'missing'.
    """
    results_dir = anima_dir / "state" / "task_results"
    if (results_dir / f"{task_id}.md").exists():
        return "completed"

    pending_dir = anima_dir / "state" / "pending"
    processing_dir = pending_dir / "processing"
    if (processing_dir / f"{task_id}.json").exists():
        return "processing"

    if (pending_dir / f"{task_id}.json").exists():
        return "pending"

    try:
        from core.memory.task_queue import TaskQueueManager

        tqm = TaskQueueManager(anima_dir)
        entry = tqm.get_task_by_id(task_id)
        if entry and entry.status in ("done", "failed", "cancelled"):
            return "terminal"
    except Exception:
        logger.debug("Failed to check task_queue for %s", task_id, exc_info=True)

    return "missing"


def _rescue_regenerate_pending(anima_dir: Path, task_id: str, msg: Any) -> None:
    """Rescue: regenerate pending file from delegation DM content for TaskExec pickup."""
    pending_dir = anima_dir / "state" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    instruction = getattr(msg, "content", "")

    # Try to get original instruction from task_queue
    try:
        from core.memory.task_queue import TaskQueueManager

        tqm = TaskQueueManager(anima_dir)
        entry = tqm.get_task_by_id(task_id)
        if entry and entry.original_instruction:
            instruction = entry.original_instruction
    except Exception:
        logger.debug("Failed to retrieve original instruction for task %s", task_id, exc_info=True)

    task_desc = {
        "task_type": "llm",
        "task_id": task_id,
        "title": instruction[:100],
        "description": instruction,
        "context": "",
        "acceptance_criteria": [],
        "constraints": [],
        "file_paths": [],
        "submitted_by": getattr(msg, "from_person", "unknown"),
        "submitted_at": datetime.now(UTC).isoformat(),
        "reply_to": getattr(msg, "from_person", ""),
        "source": "delegation_rescue",
    }
    path = pending_dir / f"{task_id}.json"
    path.write_text(
        json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Rescue: regenerated pending file for task %s from delegation DM",
        task_id,
    )


def _handle_delegation_dms(anima_mixin: Any, delegation_items: list[InboxItem]) -> None:
    """Handle delegation DMs at framework level without involving LLM.

    Checks task state and either archives (task exists/completed) or
    rescues (regenerates pending file) for each delegation DM.
    """
    anima_dir = anima_mixin.anima_dir

    for item in delegation_items:
        msg = item.msg
        task_id = _extract_task_id(msg)
        if not task_id:
            continue  # should not happen (filtered in _split_delegation_items)

        state = _check_task_state(anima_dir, task_id)

        if state == "missing":
            _rescue_regenerate_pending(anima_dir, task_id, msg)
            logger.info(
                "[%s] Delegation DM rescue: task=%s from=%s (pending file regenerated)",
                anima_mixin.name,
                task_id,
                msg.from_person,
            )
        else:
            logger.info(
                "[%s] Delegation DM handled at framework level: task=%s state=%s from=%s",
                anima_mixin.name,
                task_id,
                state,
                msg.from_person,
            )

        # Record episode for the delegation DM (same as regular inbox messages)
        _msg_ts = now_local().strftime("%H:%M")
        _episode = (
            t(
                "anima.msg_received_episode",
                ts=_msg_ts,
                from_person=msg.from_person,
                content=_truncate_with_thread_ctx(msg.content, body_budget=1000),
            )
            + "\n"
        )
        try:
            anima_mixin.memory.append_episode(_episode, origin=ORIGIN_ANIMA)
        except Exception:
            logger.debug(
                "[%s] Failed to record delegation DM episode from %s",
                anima_mixin.name,
                msg.from_person,
                exc_info=True,
            )

        # Activity log
        anima_mixin._activity.log(
            "message_received",
            content=msg.content,
            summary=f"[delegation DM - framework handled, state={state}] {msg.content[:150]}",
            from_person=msg.from_person,
            to_person=anima_mixin.name,
            meta={"from_type": msg.source, "delegation_state": state, "task_id": task_id},
            origin=ORIGIN_ANIMA,
            origin_chain=msg.origin_chain if msg.origin_chain else [ORIGIN_ANIMA],
        )

    # Archive delegation DMs immediately
    try:
        anima_mixin.messenger.archive_paths(delegation_items)
    except Exception:
        logger.debug(
            "[%s] Failed to archive delegation DMs",
            anima_mixin.name,
            exc_info=True,
        )


@dataclass
class InboxResult:
    """Result of inbox message processing."""

    inbox_items: list[InboxItem] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)
    senders: set[str] = field(default_factory=set)
    unread_count: int = 0
    prompt_parts: list[str] = field(default_factory=list)


class InboxMixin:
    """Mixin: Anima-to-Anima inbox processing, filtering, dedup, archiving."""

    # ── Inbox MSG Immediate Processing ────────────────────────

    async def process_inbox_message(
        self,
        cascade_suppressed_senders: set[str] | None = None,
    ) -> CycleResult:
        """Process Anima-to-Anima messages immediately under _inbox_lock.

        Separated from heartbeat to provide instant response to inter-Anima
        messages without triggering the full heartbeat observation cycle.
        """
        self._get_interrupt_event("_inbox").clear()
        self.agent.set_interrupt_event(self._get_interrupt_event("_inbox"))
        logger.info("[%s] process_inbox_message START", self.name)
        try:
            # Wait for any running cron task to finish before processing inbox.
            # Prevents LLM context confusion when inbox arrives mid-cron.
            if not self._cron_idle.is_set():
                logger.info("[%s] inbox waiting for cron to finish", self.name)
                await self._cron_idle.wait()
            async with self._inbox_lock:
                self._mark_busy_start()
                self._status_slots["inbox"] = "processing"

                self._activity.log("inbox_processing_start", summary=t("anima.inbox_start"))

                inbox_result: InboxResult | None = None
                try:
                    inbox_result = await self._process_inbox_messages(
                        cascade_suppressed_senders,
                    )

                    if inbox_result.unread_count == 0:
                        if inbox_result.inbox_items:
                            await self._archive_processed_messages(
                                inbox_result.inbox_items,
                                inbox_result.senders,
                                set(),
                            )
                        logger.info("[%s] process_inbox_message: no messages", self.name)
                        return CycleResult(
                            trigger="inbox",
                            action="idle",
                            summary="No unread messages",
                        )

                    senders_str = ", ".join(inbox_result.senders)
                    trigger = f"inbox:{senders_str}"

                    messages_text = "\n\n".join(inbox_result.prompt_parts)
                    prompt = load_prompt(
                        "inbox_message",
                        messages=messages_text,
                    )

                    # Suppress board fanout when replying to board_mention
                    has_board_mention = any(item.msg.type == "board_mention" for item in inbox_result.inbox_items)
                    from core.tooling.handler import active_session_type, suppress_board_fanout

                    _fanout_token = suppress_board_fanout.set(True) if has_board_mention else None
                    _session_token = self.agent._tool_handler.set_active_session_type("inbox")

                    # Set session origin from the most untrusted message
                    _batch_origins: list[str] = []
                    _batch_chains: list[str] = []
                    for item in inbox_result.inbox_items:
                        m = item.msg
                        _msg_origin = _SOURCE_TO_ORIGIN.get(m.source, ORIGIN_UNKNOWN)
                        _batch_origins.append(_msg_origin)
                        _batch_chains.extend(m.origin_chain if m.origin_chain else [_msg_origin])
                    _unique_chains = list(dict.fromkeys(_batch_chains))
                    _worst_origin = min(
                        _batch_origins or [ORIGIN_ANIMA],
                        key=lambda o: {"untrusted": 0, "medium": 1, "trusted": 2}.get(resolve_trust(o), 0),
                    )
                    self.agent._tool_handler.set_session_origin(_worst_origin, _unique_chains)

                    self.agent.reset_reply_tracking(session_type="inbox")
                    self.agent.reset_posted_channels(session_type="inbox")
                    self.agent.reset_read_paths()

                    journal = StreamingJournal(self.anima_dir, session_type="inbox")
                    journal.open(trigger=trigger, from_person=senders_str)

                    accumulated_text = ""
                    result: CycleResult | None = None

                    original_config = None
                    bg_config = self._resolve_background_config()
                    if bg_config is not None:
                        original_config = self.agent.model_config
                        self.agent.update_model_config(bg_config)

                    try:
                        async for chunk in self.agent.run_cycle_streaming(
                            prompt,
                            trigger=trigger,
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
                                    context_usage_ratio=cycle_result.get("context_usage_ratio", 0.0),
                                    session_chained=cycle_result.get("session_chained", False),
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
                        if original_config is not None:
                            self.agent.update_model_config(original_config)
                        journal.close()
                        if _fanout_token is not None:
                            suppress_board_fanout.reset(_fanout_token)
                        active_session_type.reset(_session_token)

                    self._last_activity = now_local()

                    # Record inbox response as response_sent so it appears
                    # in the conversation view alongside message_received.
                    if accumulated_text.strip():
                        self._activity.log(
                            "response_sent",
                            content=accumulated_text[:2000],
                            to_person=senders_str,
                            channel="inbox",
                            summary=accumulated_text[:200] if accumulated_text else "",
                            meta={"trigger": "inbox"},
                        )

                    # Archive processed messages — but NOT when the LLM
                    # returned nothing (e.g. SDK empty response due to API
                    # outage / rate limit).  Keeping them lets the next
                    # inbox cycle retry.
                    if accumulated_text.strip() or self.agent.replied_to:
                        await self._archive_processed_messages(
                            inbox_result.inbox_items,
                            inbox_result.senders,
                            self.agent.replied_to,
                        )
                    else:
                        logger.warning(
                            "[%s] Empty LLM response for inbox — messages NOT archived (will retry)",
                            self.name,
                        )

                    self._activity.log(
                        "inbox_processing_end",
                        summary=result.summary[:200],
                        meta={"senders": list(inbox_result.senders), "count": inbox_result.unread_count},
                    )

                    logger.info(
                        "[%s] process_inbox_message END duration_ms=%d unread=%d",
                        self.name,
                        result.duration_ms,
                        inbox_result.unread_count,
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
                                self.name,
                                exc_info=True,
                            )
                    self._activity.log(
                        "error",
                        summary=t("anima.inbox_error", exc=type(exc).__name__),
                        meta={"phase": "process_inbox_message", "error": str(exc)[:200]},
                        safe=True,
                    )
                    raise
                finally:
                    self._status_slots["inbox"] = "idle"
                    self._task_slots["inbox"] = ""
        finally:
            self._notify_lock_released()

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
            suppressed_items = [item for item in inbox_items if item.msg.from_person in cascade_suppressed_senders]
            inbox_items = [item for item in inbox_items if item.msg.from_person not in cascade_suppressed_senders]
            messages = [item.msg for item in inbox_items]
            if suppressed_items:
                logger.info(
                    "[%s] Cascade-suppressed %d messages from %s",
                    self.name,
                    len(suppressed_items),
                    ", ".join(cascade_suppressed_senders & senders),
                )
            senders = {m.from_person for m in messages}
            unread_count = len(messages)

        # ── Message overflow handling ──
        try:
            from core.memory.dedup import MessageDeduplicator

            dedup = MessageDeduplicator(self.anima_dir)

            critical, non_critical = dedup.split_critical(messages)
            non_critical, overflow_count = dedup.overflow_to_files(non_critical)

            messages = critical + non_critical

            if overflow_count:
                logger.info(
                    "[%s] %d messages overflowed to state/overflow_inbox/",
                    self.name,
                    overflow_count,
                )

            unread_count = len(messages)
            senders = {m.from_person for m in messages}
        except Exception:
            logger.debug("[%s] Message dedup failed, using original messages", self.name, exc_info=True)

        logger.info(
            "[%s] Processing %d unread messages in heartbeat (senders: %s)",
            self.name,
            unread_count,
            ", ".join(senders),
        )

        # ── Delegation DM framework-level handling ──
        delegation_items, non_delegation_items = _split_delegation_items(inbox_items, messages)
        if delegation_items:
            _handle_delegation_dms(self, delegation_items)
            inbox_items = non_delegation_items
            messages = [item.msg for item in non_delegation_items]
            unread_count = len(messages)
            senders = {m.from_person for m in messages}

        # ── Retry counter: track how many times each inbox message is presented ──
        _read_counts_path = self.anima_dir / "state" / "inbox_read_counts.json"
        _read_counts: dict[str, int] = {}
        try:
            if _read_counts_path.exists():
                _read_counts = json.loads(_read_counts_path.read_text(encoding="utf-8"))
        except Exception:
            _read_counts = {}

        for item in inbox_items:
            key = item.path.name
            _read_counts[key] = _read_counts.get(key, 0) + 1

        # Prune entries for inbox files that no longer exist
        inbox_dir = self.anima_dir.parent.parent / "shared" / "inbox" / self.name
        _read_counts = {k: v for k, v in _read_counts.items() if (inbox_dir / k).exists()}

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
            line = f"{prefix}{_truncate_with_thread_ctx(m.content)}"
            if m.source in ("slack", "chatwork") and m.external_channel_id:
                reply_instr = _build_reply_instruction(m)
                if reply_instr:
                    line += f"\n{reply_instr}"
            lines.append(line)
        # Deferred messages (no InboxItem) are appended without counter
        for m in messages:
            if not any(item.msg is m for item in inbox_items):
                line = f"- {m.from_person}: {_truncate_with_thread_ctx(m.content)}"
                if m.source in ("slack", "chatwork") and m.external_channel_id:
                    reply_instr = _build_reply_instruction(m)
                    if reply_instr:
                        line += f"\n{reply_instr}"
                lines.append(line)
        summary = "\n".join(lines)
        prompt_parts.append(load_prompt("unread_messages", summary=summary))

        # Record received message content to episodes so that
        # inter-Anima communications survive in episodic memory.
        _msg_ts = now_local().strftime("%H:%M")
        _recordable = [m for m in messages if m.type != "ack"]
        if len(_recordable) > 50:
            logger.warning(
                "[%s] DM burst: %d messages, recording first 50",
                self.name,
                len(_recordable),
            )
        for _m in _recordable[:50]:
            _episode = (
                t(
                    "anima.msg_received_episode",
                    ts=_msg_ts,
                    from_person=_m.from_person,
                    content=_truncate_with_thread_ctx(_m.content, body_budget=1000),
                )
                + "\n"
            )
            _ep_origin = _SOURCE_TO_ORIGIN.get(_m.source, ORIGIN_UNKNOWN)
            try:
                self.memory.append_episode(_episode, origin=_ep_origin)
            except Exception:
                logger.debug(
                    "[%s] Failed to record message episode from %s",
                    self.name,
                    _m.from_person,
                    exc_info=True,
                )

        # Activity log: message received (full content, summary truncated)
        for _m in _recordable[:50]:
            _msg_origin = _SOURCE_TO_ORIGIN.get(_m.source, ORIGIN_UNKNOWN)
            _msg_origin_chain = _m.origin_chain if _m.origin_chain else [_msg_origin]
            self._activity.log(
                "message_received",
                content=_m.content,
                summary=_m.content[:200],
                from_person=_m.from_person,
                to_person=self.name,
                meta={"from_type": _m.source},
                origin=_msg_origin,
                origin_chain=_msg_origin_chain,
            )

        return InboxResult(
            inbox_items=inbox_items,
            messages=messages,
            senders=senders,
            unread_count=unread_count,
            prompt_parts=prompt_parts,
        )

    async def _archive_processed_messages(
        self,
        inbox_items: list[InboxItem],
        senders: set[str],
        replied_to: set[str],
    ) -> None:
        """Archive all inbox messages after successful LLM cycle.

        Once the LLM has completed processing, all messages are archived
        unconditionally.  The LLM already decided whether to reply,
        delegate, or take no action — re-presenting the same messages
        would only produce duplicate ``message_received`` log entries.
        """
        total_archived = self.messenger.archive_paths(inbox_items)

        unreplied = senders - replied_to
        if unreplied:
            logger.info(
                "[%s] Archived %d messages (unreplied senders: %s — processed, no retry)",
                self.name,
                total_archived,
                ", ".join(unreplied),
            )
        else:
            logger.info(
                "[%s] Archived %d/%d messages",
                self.name,
                total_archived,
                len(inbox_items),
            )
