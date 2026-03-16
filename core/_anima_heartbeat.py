from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""HeartbeatMixin -- heartbeat/cron prompt construction and cycle execution.

Extracted from ``core.anima.DigitalAnima`` as a Mixin.  All ``self``
references are resolved at runtime via MRO when mixed into ``DigitalAnima``.
"""

import json
import logging
import math
import re
import time
from typing import Any

from core.i18n import t
from core.memory.conversation import ConversationMemory
from core.memory.streaming_journal import StreamingJournal
from core.messenger import InboxItem
from core.paths import load_prompt
from core.schemas import CycleResult
from core.time_utils import now_iso, now_local

logger = logging.getLogger("animaworks.anima")


def _calc_effective_max_turns(
    base_max_turns: int,
    activity_level: int,
    hb_max_turns: int | None = None,
) -> int | None:
    """Calculate effective max_turns for heartbeat based on activity level.

    When *hb_max_turns* is provided (from ``config.heartbeat.max_turns``),
    it is used as the base instead of the per-anima chat ``max_turns``.

    Below 100%: linear scale (floor 3). At/above 100%: return None (use base).
    """
    base = hb_max_turns if hb_max_turns is not None else base_max_turns
    if activity_level >= 100:
        if hb_max_turns is not None:
            return hb_max_turns
        return None
    scaled = max(3, math.ceil(base * activity_level / 100))
    return scaled


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


class HeartbeatMixin:
    """Mixin: heartbeat/cron prompt building, cycle execution, failure handling."""

    # ── Background model resolution ──────────────────────────

    def _resolve_background_config(self) -> ModelConfig | None:  # noqa: F821
        """Resolve background model config for heartbeat/cron.

        Resolution order:
          1. status.json background_model (per-anima)
          2. config.heartbeat.default_model (global)
          3. None (use main model)
        """
        from core.config.models import load_config, resolve_execution_mode
        from core.schemas import ModelConfig

        bg_model = self.agent.model_config.background_model
        if not bg_model:
            config = load_config()
            bg_model = config.heartbeat.default_model
        if not bg_model:
            return None
        if bg_model == self.agent.model_config.model:
            return None

        # Recalculate resolved_mode for the background model so that
        # the correct executor type is created (e.g. claude-* → S, codex/* → C).
        # Without this, model_copy carries the main model's resolved_mode,
        # which may be incompatible with the background model name.
        config = load_config()
        bg_resolved_mode = resolve_execution_mode(config, bg_model)

        bg_credential = self.agent.model_config.background_credential
        new_config: ModelConfig = self.agent.model_config.model_copy(
            update={"model": bg_model, "resolved_mode": bg_resolved_mode},
        )
        if bg_credential:
            if bg_credential in config.credentials:
                cred = config.credentials[bg_credential]
                new_config.api_key = cred.api_key or None
                new_config.api_base_url = cred.base_url or None
                new_config.extra_keys = dict(cred.keys) if cred.keys else {}
        return new_config

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
                self.name,
                exc_info=True,
            )
            return ""

    # ── Heartbeat private methods ──────────────────────────

    def _build_prior_messages(
        self,
        prompt_text: str,
    ) -> list[dict[str, Any]] | None:
        """Build prior_messages for A mode, None for S/B."""
        mode = self.agent.execution_mode
        if mode != "a":
            return None
        conv = ConversationMemory(self.anima_dir, self.model_config)
        return conv.build_structured_messages(prompt_text)

    def _build_background_context_parts(self, include_dialogue: bool = True) -> list[str]:
        """Build shared context parts for background-auto sessions (heartbeat/cron).

        Collects: recovery note, background task notifications, heartbeat
        history, reflections, dialogue context, subordinate check.

        Args:
            include_dialogue: If True, inject recent chat dialogue turns.
                Set to False for cron tasks to prevent chat context leaking
                into scheduled task execution.
        """
        parts: list[str] = []

        # ── Recovery note from previous failed heartbeat ──
        recovery_note_path = self.anima_dir / "state" / "recovery_note.md"
        if recovery_note_path.exists():
            try:
                recovery_content = recovery_note_path.read_text(encoding="utf-8")
                parts.append(load_prompt("fragments/recovery_note_header") + "\n\n" + recovery_content)
                recovery_note_path.unlink(missing_ok=True)
                logger.info("[%s] Recovery note loaded and removed", self.name)
            except Exception:
                logger.debug("[%s] Failed to read recovery note", self.name, exc_info=True)

        # Inject pending background task notifications
        bg_notifications = self.drain_background_notifications()
        if bg_notifications:
            notif_text = "\n\n".join(bg_notifications)
            parts.append(load_prompt("fragments/bg_task_notification") + "\n\n" + notif_text)

        # Inject recent heartbeat history for continuity
        history_text = self._load_heartbeat_history()
        if history_text:
            parts.append(
                load_prompt(
                    "heartbeat_history",
                    history=history_text,
                )
            )

        # Inject recent reflections for cognitive continuity
        reflection_text = self._load_recent_reflections()
        if reflection_text:
            parts.append(load_prompt("fragments/recent_reflections") + "\n\n" + reflection_text)

        # Inject recent dialogue context for cross-session continuity
        # Skipped for cron tasks to prevent chat context leaking into scheduled execution
        if include_dialogue:
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
                        t("agent.recent_dialogue_header")
                        + "\n\n"
                        + t("agent.recent_dialogue_intro")
                        + "\n"
                        + t("agent.recent_dialogue_consider")
                        + "\n\n"
                        + conv_summary
                    )
            except Exception:
                logger.debug("[%s] Failed to load dialogue context", self.name, exc_info=True)

        # ── Subordinate management check for animas with subordinates ──
        try:
            from core.config.models import load_config
            from core.paths import get_animas_dir

            _cfg = load_config()
            _subordinates = [_name for _name, _pcfg in _cfg.animas.items() if _pcfg.supervisor == self.name]
            if _subordinates:
                parts.append(
                    load_prompt(
                        "heartbeat_subordinate_check",
                        subordinates=", ".join(_subordinates),
                        animas_dir=str(get_animas_dir()),
                    )
                )
        except Exception:
            logger.debug(
                "[%s] Failed to inject delegation check",
                self.name,
                exc_info=True,
            )

        return parts

    _CURRENT_TASK_CLEANUP_THRESHOLD = 3000

    async def _build_heartbeat_prompt(self) -> list[str]:
        """Build heartbeat prompt parts.

        Heartbeat-specific header + shared background context.
        When current_task.md exceeds the cleanup threshold, a compression
        instruction is prepended so the anima trims it first.
        """
        hb_config = self.memory.read_heartbeat_config()
        checklist = hb_config or load_prompt("heartbeat_default_checklist")
        task_delegation_rules = load_prompt("task_delegation_rules")
        parts = [load_prompt("heartbeat", checklist=checklist, task_delegation_rules=task_delegation_rules)]

        state = self.memory.read_current_state()
        state_len = len(state)
        if state_len > self._CURRENT_TASK_CLEANUP_THRESHOLD:
            parts.append(
                t(
                    "heartbeat.current_task_cleanup_required",
                    current_chars=state_len,
                    max_chars=self._CURRENT_TASK_CLEANUP_THRESHOLD,
                )
            )
            logger.info(
                "[%s] current_task.md exceeds threshold (%d > %d), injecting cleanup instruction",
                self.name,
                state_len,
                self._CURRENT_TASK_CLEANUP_THRESHOLD,
            )

        parts.extend(self._build_background_context_parts())

        return parts

    def _build_cron_prompt(
        self,
        task_name: str,
        description: str,
        command_output: str | None = None,
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
            "cron_task",
            task_name=task_name,
            description=description,
        )
        if cron_prompt:
            parts.append(cron_prompt)

        # Inject command output if this is a follow-up to a command cron
        if command_output:
            parts.append(load_prompt("fragments/command_output", output=command_output))

        # Shared background context (without dialogue — cron tasks must not inherit chat context)
        parts.extend(self._build_background_context_parts(include_dialogue=False))

        return "\n\n".join(parts)

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
                json.dumps(checkpoint_data, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("[%s] Failed to write heartbeat checkpoint", self.name, exc_info=True)

        # Reset reply tracking before the cycle
        self.agent.reset_reply_tracking(session_type="background")
        self.agent.reset_posted_channels(session_type="background")
        self.agent.reset_read_paths()
        # Clear replied_to persistence file
        _replied_to_path = self.anima_dir / "run" / "replied_to.jsonl"
        if _replied_to_path.exists():
            _replied_to_path.unlink(missing_ok=True)

        accumulated_text = ""
        result: CycleResult | None = None

        # Streaming journal for heartbeat crash recovery
        journal = StreamingJournal(self.anima_dir, session_type="heartbeat")
        journal.open(trigger="heartbeat")

        # ── Background model swap ──
        original_config = None
        bg_config = self._resolve_background_config()
        if bg_config is not None:
            original_config = self.agent.model_config
            self.agent.update_model_config(bg_config)

        try:
            from core.config.models import load_config as _load_config_fresh

            _cfg = _load_config_fresh()
            _hb_cfg = _cfg.heartbeat
            effective_max_turns = _calc_effective_max_turns(
                base_max_turns=self.agent.model_config.max_turns,
                activity_level=_cfg.activity_level,
                hb_max_turns=_hb_cfg.max_turns,
            )

            _soft_timeout = _hb_cfg.soft_timeout_seconds
            _hard_timeout = _hb_cfg.hard_timeout_seconds
            _start = time.monotonic()
            _soft_warned = False
            _hard_exceeded = False

            async for chunk in self.agent.run_cycle_streaming(
                prompt,
                trigger="heartbeat",
                prior_messages=prior_messages,
                max_turns_override=effective_max_turns,
            ):
                # ── Timeout checks (Mode A: reminder_queue injection) ──
                _elapsed = time.monotonic() - _start
                if not _soft_warned and _elapsed > _soft_timeout:
                    _soft_warned = True
                    self.agent._executor.reminder_queue.push_sync(t("reminder.hb_time_limit"))
                    logger.info(
                        "[%s] Heartbeat soft timeout reached (%.0fs > %ds)",
                        self.name,
                        _elapsed,
                        _soft_timeout,
                    )
                if _elapsed > _hard_timeout:
                    _hard_exceeded = True
                    logger.warning(
                        "[%s] Heartbeat hard timeout reached (%.0fs > %ds) — breaking",
                        self.name,
                        _elapsed,
                        _hard_timeout,
                    )
                    break

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
                        context_usage_ratio=cycle_result.get("context_usage_ratio", 0.0),
                        session_chained=cycle_result.get("session_chained", False),
                        total_turns=cycle_result.get("total_turns", 0),
                    )
                    journal.finalize(summary=result.summary[:500])

            # ── Hard timeout: write recovery note ──
            if _hard_exceeded:
                try:
                    recovery_path = self.anima_dir / "state" / "recovery_note.md"
                    recovery_path.write_text(
                        t("reminder.hb_hard_timeout_recovery", timeout=_hard_timeout),
                        encoding="utf-8",
                    )
                    logger.info("[%s] Hard timeout recovery note saved", self.name)
                except Exception:
                    logger.debug("[%s] Failed to save hard timeout recovery note", self.name, exc_info=True)

            if result is None:
                result = CycleResult(
                    trigger="heartbeat",
                    action="responded",
                    summary=accumulated_text or "(no result)",
                )

            self._last_activity = now_local()

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
                ts = now_local().strftime("%H:%M")
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
                    episode_entry += f"\n\n[REFLECTION]\n{reflection_text}\n[/REFLECTION]"
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
                self.name,
                result.duration_ms,
                unread_count,
            )
            # Heartbeat completed successfully — remove checkpoint
            try:
                checkpoint_path.unlink(missing_ok=True)
            except Exception:
                logger.debug("[%s] Failed to remove heartbeat checkpoint", self.name, exc_info=True)

            # Compact task queue after heartbeat
            try:
                from core.memory.task_queue import TaskQueueManager

                _tqm = TaskQueueManager(self.anima_dir)
                _removed = _tqm.compact()
                if _removed:
                    logger.info(
                        "[%s] Task queue compacted after heartbeat: removed %d tasks",
                        self.name,
                        _removed,
                    )
            except Exception:
                logger.debug(
                    "[%s] Task queue compaction failed after heartbeat",
                    self.name,
                    exc_info=True,
                )

            return result
        finally:
            if original_config is not None:
                self.agent.update_model_config(original_config)
            journal.close()

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
                    self.name,
                    crash_archived,
                    len(inbox_items),
                )
            except Exception:
                logger.warning(
                    "[%s] Failed to crash-archive inbox messages",
                    self.name,
                    exc_info=True,
                )

        # Activity log: heartbeat failure (single event to avoid double-fault)
        self._activity.log(
            "heartbeat_end",
            summary=f"[ERROR] {type(error).__name__}: {str(error)[:100]}",
            meta={
                "status": "failed",
                "phase": "run_heartbeat",
                "error": str(error)[:200],
            },
            safe=True,
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
        try:
            if StreamingJournal.has_orphan(self.anima_dir, session_type="heartbeat"):
                StreamingJournal.confirm_recovery(self.anima_dir, session_type="heartbeat")
                logger.info("[%s] Cleaned up orphaned streaming journal", self.name)
        except Exception:
            logger.debug(
                "[%s] Failed to clean up streaming journal",
                self.name,
                exc_info=True,
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
                self.name,
                len(task_files),
            )
            if self._pending_executor is not None:
                self._pending_executor.wake()
