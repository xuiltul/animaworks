from __future__ import annotations

from core.anima._mixin_protocols import _HeartbeatHost

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""HeartbeatMixin -- heartbeat/cron prompt construction and cycle execution.

Extracted from ``core.anima.digital_anima.DigitalAnima`` as a Mixin.  All ``self``
references are resolved at runtime via MRO when mixed into ``DigitalAnima``.
"""

import asyncio
import hashlib
import inspect
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from core.execution.fallback_activity import run_with_model_fallback
from core.i18n import t
from core.memory.conversation.memory import ConversationMemory
from core.memory.conversation.streaming_journal import StreamingJournal
from core.messaging.messenger import InboxItem
from core.paths import load_prompt
from core.platform.state_writer import get_state_writer, is_task_runner_process, run_writer_sync
from core.schemas import CycleResult
from core.skills.cron_context import SkillContextRejection, SkillContextWarning
from core.time_utils import ensure_aware, now_iso, now_local

logger = logging.getLogger("animaworks.anima")


# ── Reflection extraction ─────────────────────────────────────

_RE_REFLECTION = re.compile(
    r"\[REFLECTION\]\s*\n?(.*?)\n?\s*\[/REFLECTION\]",
    re.DOTALL,
)

_MIN_REFLECTION_LENGTH = 50

# ── Plan extraction ───────────────────────────────────────────

_RE_PLAN = re.compile(
    r"##\s*Plan[^\n]*\n(.*?)(?=\n##\s|\Z)",
    re.DOTALL,
)

_MAX_PLAN_SUMMARY_CHARS = 500


def _cleanup_orphaned_heartbeat_journal(anima_dir: Path) -> bool:
    """Remove a stale heartbeat journal and report whether one was found."""
    if not StreamingJournal.has_orphan(anima_dir, session_type="heartbeat"):
        return False
    StreamingJournal.confirm_recovery(anima_dir, session_type="heartbeat")
    return True


def _extract_plan_summary(text: str) -> str:
    """Extract ## Plan section from heartbeat output.

    Returns empty string if not found.
    """
    if not text:
        return ""
    m = _RE_PLAN.search(text)
    if m:
        return m.group(1).strip()[:_MAX_PLAN_SUMMARY_CHARS]
    return ""


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


def _build_curator_review_part(anima_dir: Path, name: str) -> str | None:
    """Build the conditional Curator-proposal review fragment for heartbeat.

    Injected only when an unreviewed report with at least one proposal exists.
    Any failure is swallowed so heartbeat construction never blocks. Kept as a
    module-level function (not a mixin method) so it reads real state instead of
    being intercepted by ``MagicMock(spec=...)`` in heartbeat prompt tests.
    """
    try:
        from core.skills.curator import latest_unreviewed_report, summarize_curator_report

        report = latest_unreviewed_report(anima_dir)
        if report is None:
            return None
        count, breakdown, top_items = summarize_curator_report(report)
        return load_prompt(
            "fragments/curator_report_review",
            count=count,
            breakdown=breakdown,
            top_items=top_items,
        )
    except Exception:
        logger.debug("[%s] Failed to build curator review part", name, exc_info=True)
        return None


def _build_stale_task_scoreboard(anima_dir: Path, name: str) -> str | None:
    """Build the oldest-first non-terminal task scoreboard for heartbeat."""
    try:
        from core.tasks.queue import TaskQueueManager
        from core.tasks.waiting import waiting_label

        now = now_local()
        rows: list[tuple[float, str]] = []
        for task in TaskQueueManager(anima_dir, read_only=True).get_all_active():
            try:
                elapsed_seconds = max(
                    0.0,
                    (now - ensure_aware(datetime.fromisoformat(task.updated_at or task.ts))).total_seconds(),
                )
            except (TypeError, ValueError):
                elapsed_seconds = -1.0
            marker = "⚠️ " if elapsed_seconds > 24 * 60 * 60 else ""
            hours = max(0, int(elapsed_seconds // 3600))
            summary = task.summary.replace("\n", " ")[:80]
            label = waiting_label(task.meta.get("waiting_for"))
            row = f"- {marker}{task.task_id[:12]} | {task.status} | {hours}h | {summary}"
            if label:
                row += f" | {label}"
            rows.append((elapsed_seconds, row))

        if not rows:
            return None
        rows.sort(key=lambda row: row[0], reverse=True)
        overflow = len(rows) - 20
        return load_prompt(
            "fragments/stale_task_scoreboard",
            tasks="\n".join(row for _, row in rows[:20]),
            overflow=f"\n- {t('heartbeat.stale_task_overflow', count=overflow)}" if overflow > 0 else "",
        )
    except Exception:
        logger.debug("[%s] Failed to build stale task scoreboard", name, exc_info=True)
        return None


async def _build_cron_rejected_notice_async(anima_dir: Path, name: str) -> str | None:
    """Return a notice once for each distinct rejected-cron list."""
    registration_path = anima_dir / "state" / "cron_registration.json"
    marker_path = anima_dir / "state" / "cron_rejected_notice.sha256"
    try:
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        rejected = registration.get("rejected") if isinstance(registration, dict) else None
        if not isinstance(rejected, list):
            return None
        digest = hashlib.sha256(
            json.dumps(rejected, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if marker_path.is_file() and marker_path.read_text(encoding="utf-8").strip() == digest:
            return None
        notice = None
        if rejected:
            jobs = "\n".join(
                f"- {item.get('name', '(unnamed)')}: {item.get('reason', 'unknown reason')}"
                for item in rejected
                if isinstance(item, dict)
            )
            notice = load_prompt("fragments/cron_rejected_notice", rejected_jobs=jobs)
        await get_state_writer(anima_dir).mark_cron_rejected_notice(digest)
        return notice
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        logger.debug("[%s] Failed to build rejected cron notice", name, exc_info=True)
        return None
    except Exception:
        logger.warning("[%s] Failed to persist rejected cron notice marker", name, exc_info=True)
        return None


def _build_cron_rejected_notice(anima_dir: Path, name: str) -> str | None | Any:
    """Synchronous compatibility wrapper; task-runner callers await it."""
    operation = _build_cron_rejected_notice_async(anima_dir, name)
    if is_task_runner_process():
        return operation
    return run_writer_sync(get_state_writer(anima_dir), operation)


class HeartbeatMixin:
    """Mixin: heartbeat/cron prompt building, cycle execution, failure handling."""

    # ── Background model resolution ──────────────────────────

    def _resolve_background_config(self: _HeartbeatHost, channel: str = "background") -> ModelConfig | None:  # noqa: F821
        """Resolve the background lane through the common route selection."""
        from core.config.model_config import resolve_model_selection
        from core.config.models import load_config
        from core.execution.fallback_activity import log_model_fallback
        from core.schemas import ModelConfig

        main_config = self.agent.model_config
        if not isinstance(main_config, ModelConfig):
            return main_config
        selection = resolve_model_selection(main_config, lane="background", config=load_config())
        activity = getattr(self, "_activity", None)
        if activity is not None:
            log_model_fallback(
                activity,
                selection.primary,
                selection.effective,
                channel=channel,
                phase="preflight",
            )
        return None if selection.effective is main_config else selection.effective

    # ── Heartbeat history ────────────────────────────────────

    _HEARTBEAT_HISTORY_N = 3

    _PLAN_OUTCOME_MAX_CHARS = 120

    def _load_heartbeat_history(self: _HeartbeatHost) -> str:
        """Load last N heartbeat history entries with plan-outcome tracking.

        When ``meta.plan_summary`` is available, the entry is rendered as
        a plan item so the next heartbeat can verify execution status.
        Falls back to legacy ``shortterm/heartbeat_history/``.
        """
        try:
            entries = self._activity.recent(
                days=2,
                types=["heartbeat_end"],
                limit=self._HEARTBEAT_HISTORY_N,
            )
            if entries:
                lines: list[str] = []
                limit = self._PLAN_OUTCOME_MAX_CHARS
                for e in entries:
                    ts_short = e.ts[11:19] if len(e.ts) >= 19 else e.ts
                    plan = (e.meta or {}).get("plan_summary", "")
                    if plan:
                        lines.append(
                            t("heartbeat.history_plan_entry", ts=ts_short, plan=plan[:limit].replace("\n", " "))
                        )
                    else:
                        summary = (e.summary or e.content)[:limit].replace("\n", " ")
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

    def _get_recent_dialogue_max_age_hours(self: _HeartbeatHost) -> int:
        """Read the heartbeat recent-dialogue freshness window from config."""
        try:
            from core.config.models import load_config

            value = load_config().heartbeat.recent_dialogue_max_age_hours
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        except Exception:
            logger.debug("Could not read heartbeat.recent_dialogue_max_age_hours", exc_info=True)
        return 6

    def _dialogue_is_recent(self: _HeartbeatHost, turns: list[Any]) -> bool:
        """True when the last turn is newer than the configured freshness window.

        Missing or unparseable timestamps default to including the dialogue so
        older callers (and tests) keep working. A window of 0 always includes
        the dialogue.
        """
        max_age_hours = self._get_recent_dialogue_max_age_hours()
        if max_age_hours <= 0:
            return True
        if not turns:
            return True
        ts = getattr(turns[-1], "timestamp", "") or ""
        if not ts:
            return True
        try:
            last_at = ensure_aware(datetime.fromisoformat(str(ts)))
        except (TypeError, ValueError):
            return True
        return (now_local() - last_at).total_seconds() <= max_age_hours * 3600

    def _load_recent_reflections(self: _HeartbeatHost) -> str:
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
        self: _HeartbeatHost,
        prompt_text: str,
    ) -> list[dict[str, Any]] | None:
        """Build prior_messages for A mode, None for S/B."""
        mode = self.agent.execution_mode
        if mode != "a":
            return None
        conv = ConversationMemory(self.anima_dir, self.model_config)
        return conv.build_structured_messages(prompt_text)

    def _build_background_context_parts(self: _HeartbeatHost, include_dialogue: bool = True) -> list[str] | Any:
        """Build shared background context, retaining a sync local adapter."""
        if is_task_runner_process():
            return self._build_background_context_parts_async(include_dialogue)
        writer = get_state_writer(self.anima_dir)
        recovery_content = run_writer_sync(writer, writer.consume_recovery_note())
        notifications = self.drain_background_notifications()
        return self._compose_background_context_parts(include_dialogue, recovery_content, notifications)

    async def _build_background_context_parts_async(
        self: _HeartbeatHost,
        include_dialogue: bool = True,
    ) -> list[str]:
        writer = get_state_writer(self.anima_dir)
        recovery_content = await writer.consume_recovery_note()
        notifications = await writer.consume_background_notifications("all")
        return self._compose_background_context_parts(include_dialogue, recovery_content, notifications)

    def _compose_background_context_parts(
        self: _HeartbeatHost,
        include_dialogue: bool,
        recovery_content: str,
        bg_notifications: list[str],
    ) -> list[str]:
        parts: list[str] = []
        if recovery_content:
            parts.append(load_prompt("fragments/recovery_note_header") + "\n\n" + recovery_content)
            logger.info("[%s] Recovery note loaded and removed", self.name)

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
                if recent_turns and self._dialogue_is_recent(recent_turns):
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

    def _get_current_state_max_chars(self: _HeartbeatHost) -> int:
        try:
            from core.config.models import load_config

            return load_config().heartbeat.current_state_max_chars
        except Exception:
            return 0

    def _enforce_state_size_limit(self: _HeartbeatHost) -> None:
        """Hard-trim current_state.md if it exceeds the configured threshold.

        Called after heartbeat completion.  Overflow content is archived
        into today's episode file for traceability.
        Disabled when ``heartbeat.current_state_max_chars`` is 0 (default).
        """
        max_chars = self._get_current_state_max_chars()
        if max_chars <= 0:
            return
        episode_path = None
        with self.memory.state_lock:
            state = self.memory.read_current_state()
            if len(state) <= max_chars:
                return
            trimmed = state[-max_chars:]
            first_nl = trimmed.find("\n")
            if first_nl != -1 and first_nl < max_chars * 0.2:
                trimmed = trimmed[first_nl + 1 :]
            overflow = state[: len(state) - len(trimmed)]
            episode_path = self.memory.append_episode(
                f"## current_state.md overflow archived\n\n{overflow}",
                _defer_index=True,
            )
            if episode_path is None:
                return
            self.memory.update_state(trimmed)
        try:
            self.memory._index_episode_file(episode_path)
        except Exception:
            logger.warning("[%s] Failed to index archived current_state overflow", self.name, exc_info=True)
        logger.info(
            "[%s] current_state.md hard-trimmed: %d → %d chars",
            self.name,
            len(state),
            len(trimmed),
        )

    def _get_current_state_cleanup_chars(self: _HeartbeatHost) -> int:
        try:
            from core.config.models import load_config

            return load_config().heartbeat.current_state_cleanup_chars
        except Exception:
            return 0

    def _build_state_cleanup_instruction(self: _HeartbeatHost) -> str | None:
        """Return a self-cleanup instruction when current_state.md nears the limit."""
        max_chars = self._get_current_state_max_chars()
        if max_chars <= 0:
            return None
        state = self.memory.read_current_state()
        state_len = len(state)
        cleanup_chars = self._get_current_state_cleanup_chars()
        soft_threshold = cleanup_chars if cleanup_chars > 0 else int(max_chars * 0.8)
        if state_len <= soft_threshold:
            return None
        logger.info(
            "[%s] current_state.md nears limit (%d > %d of max %d), injecting cleanup instruction",
            self.name,
            state_len,
            soft_threshold,
            max_chars,
        )
        return t(
            "heartbeat.current_state_cleanup_required",
            current_chars=state_len,
            cleanup_chars=soft_threshold,
            max_chars=max_chars,
            target_chars=(cleanup_chars // 2) if cleanup_chars > 0 else (max_chars // 2),
        )

    def _get_heartbeat_md_max_bytes(self: _HeartbeatHost) -> int:
        try:
            from core.config.models import load_config

            return load_config().heartbeat.heartbeat_md_max_bytes
        except Exception:
            return 0

    async def _archive_heartbeat_md_before_cleanup_async(self: _HeartbeatHost) -> str | None:
        return await get_state_writer(self.anima_dir).archive_heartbeat_md_snapshot()

    def _archive_heartbeat_md_before_cleanup(self: _HeartbeatHost) -> str | None:
        """Synchronous compatibility adapter for local callers and tests."""
        writer = get_state_writer(self.anima_dir)
        return run_writer_sync(writer, writer.archive_heartbeat_md_snapshot())

    async def _build_heartbeat_md_cleanup_instruction_async(self: _HeartbeatHost, hb_config: str) -> str | None:
        max_bytes = self._get_heartbeat_md_max_bytes()
        if max_bytes <= 0 or not hb_config:
            return None
        current_bytes = len(hb_config.encode("utf-8"))
        if current_bytes <= max_bytes:
            return None
        logger.info(
            "[%s] HEARTBEAT.md exceeds limit (%d > %d bytes), injecting compaction instruction",
            self.name,
            current_bytes,
            max_bytes,
        )
        archived_path = await self._archive_heartbeat_md_before_cleanup_async()
        archive_notice = t("heartbeat.heartbeat_md_archive_notice", path=archived_path) if archived_path else ""
        return t(
            "heartbeat.heartbeat_md_cleanup_required",
            current_kb=f"{current_bytes / 1024:.1f}",
            max_kb=f"{max_bytes / 1024:.0f}",
            target_kb=f"{max_bytes / 2048:.0f}",
            archive_notice=archive_notice,
        )

    def _build_heartbeat_md_cleanup_instruction(self: _HeartbeatHost, hb_config: str) -> str | None | Any:
        """Return a compaction instruction, using a sync local adapter for tests."""
        if is_task_runner_process():
            return self._build_heartbeat_md_cleanup_instruction_async(hb_config)
        max_bytes = self._get_heartbeat_md_max_bytes()
        if max_bytes <= 0 or not hb_config:
            return None
        current_bytes = len(hb_config.encode("utf-8"))
        if current_bytes <= max_bytes:
            return None
        logger.info(
            "[%s] HEARTBEAT.md exceeds limit (%d > %d bytes), injecting compaction instruction",
            self.name,
            current_bytes,
            max_bytes,
        )
        archived_path = self._archive_heartbeat_md_before_cleanup()
        archive_notice = t("heartbeat.heartbeat_md_archive_notice", path=archived_path) if archived_path else ""
        return t(
            "heartbeat.heartbeat_md_cleanup_required",
            current_kb=f"{current_bytes / 1024:.1f}",
            max_kb=f"{max_bytes / 1024:.0f}",
            target_kb=f"{max_bytes / 2048:.0f}",
            archive_notice=archive_notice,
        )

    async def _build_heartbeat_prompt(self: _HeartbeatHost) -> list[str]:
        """Build heartbeat prompt parts.

        Heartbeat-specific header + shared background context.
        When current_state.md nears the cleanup threshold, a compression
        instruction is prepended so the anima trims it first.
        """
        hb_config = self.memory.read_heartbeat_config()
        checklist = hb_config or load_prompt("heartbeat_default_checklist")
        parts = [load_prompt("heartbeat", checklist=checklist)]

        hb_cleanup = self._build_heartbeat_md_cleanup_instruction(hb_config)
        if inspect.isawaitable(hb_cleanup):
            hb_cleanup = await hb_cleanup
        if hb_cleanup:
            parts.append(hb_cleanup)

        cleanup = self._build_state_cleanup_instruction()
        if cleanup:
            parts.append(cleanup)

        cron_notice = _build_cron_rejected_notice(self.anima_dir, self.name)
        if inspect.isawaitable(cron_notice):
            cron_notice = await cron_notice
        if cron_notice:
            parts.append(cron_notice)

        background_parts = self._build_background_context_parts()
        if inspect.isawaitable(background_parts):
            background_parts = await background_parts
        parts.extend(background_parts)

        scoreboard = _build_stale_task_scoreboard(self.anima_dir, self.name)
        if scoreboard:
            parts.append(scoreboard)

        curator_part = _build_curator_review_part(self.anima_dir, self.name)
        if curator_part:
            parts.append(curator_part)

        return parts

    def _build_cron_prompt(
        self: _HeartbeatHost,
        task_name: str,
        description: str,
        command_output: str | None = None,
        skills: list[str] | None = None,
        skill_rejections_out: list[SkillContextRejection] | None = None,
        skill_warnings_out: list[SkillContextWarning] | None = None,
    ) -> str | Any:
        """Build a cron prompt using local compatibility or awaited IPC state reads."""
        args = (task_name, description, command_output, skills, skill_rejections_out, skill_warnings_out)
        if is_task_runner_process():
            return self._build_cron_prompt_async(*args)
        background_parts = self._build_background_context_parts(include_dialogue=False)
        return HeartbeatMixin._compose_cron_prompt(
            self,
            task_name,
            description,
            command_output,
            skills,
            skill_rejections_out,
            skill_warnings_out,
            background_context_parts=background_parts,
        )

    async def _build_cron_prompt_async(
        self: _HeartbeatHost,
        task_name: str,
        description: str,
        command_output: str | None = None,
        skills: list[str] | None = None,
        skill_rejections_out: list[SkillContextRejection] | None = None,
        skill_warnings_out: list[SkillContextWarning] | None = None,
    ) -> str:
        background_parts = await self._build_background_context_parts_async(include_dialogue=False)
        return HeartbeatMixin._compose_cron_prompt(
            self,
            task_name,
            description,
            command_output,
            skills,
            skill_rejections_out,
            skill_warnings_out,
            background_context_parts=background_parts,
        )

    def _compose_cron_prompt(
        self: _HeartbeatHost,
        task_name: str,
        description: str,
        command_output: str | None,
        skills: list[str] | None,
        skill_rejections_out: list[SkillContextRejection] | None,
        skill_warnings_out: list[SkillContextWarning] | None,
        *,
        background_context_parts: list[str],
    ) -> str:
        """Build cron task prompt with heartbeat-equivalent context.

        Args:
            task_name: Cron task name from cron.md.
            description: Task description or instruction.
            command_output: Optional stdout from a preceding command-type cron.
            skills: Optional cron skill references from cron.md.
            skill_rejections_out: Optional list populated with rejected skill refs.
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

        # Cron sessions append to current_state.md far more often than
        # heartbeats do, so they need the cleanup nudge as well.
        cleanup = self._build_state_cleanup_instruction()
        if cleanup:
            parts.append(cleanup)

        # Inject command output if this is a follow-up to a command cron
        if command_output:
            parts.append(load_prompt("fragments/command_output", output=command_output))

        if skills:
            from core.skills.cron_context import build_cron_skill_context

            skill_context = build_cron_skill_context(self.anima_dir, skills)
            if skill_rejections_out is not None:
                skill_rejections_out.extend(skill_context.rejections)
            if skill_warnings_out is not None:
                skill_warnings_out.extend(skill_context.warnings)
            rendered = skill_context.render()
            if rendered:
                parts.append(rendered)

        # Shared background context (without dialogue — cron tasks must not inherit chat context)
        parts.extend(background_context_parts)

        return "\n\n".join(parts)

    async def _execute_heartbeat_cycle(
        self: _HeartbeatHost,
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
        agent = self._agent_for_lane("background") if hasattr(self, "_agent_for_lane") else self.agent
        # ── Heartbeat Checkpoint ──
        try:
            checkpoint_data = {
                "ts": now_iso(),
                "trigger": "heartbeat",
                "unread_count": unread_count,
            }
            await get_state_writer(self.anima_dir).write_heartbeat_checkpoint(checkpoint_data)
        except Exception:
            logger.debug("[%s] Failed to write heartbeat checkpoint", self.name, exc_info=True)

        # Reset reply tracking before the cycle
        agent.reset_reply_tracking(session_type="heartbeat")
        agent.reset_posted_channels(session_type="heartbeat")
        agent.reset_read_paths()

        accumulated_text = ""
        result: CycleResult | None = None

        # Streaming journal for heartbeat crash recovery
        journal = StreamingJournal(self.anima_dir, session_type="heartbeat")
        await asyncio.to_thread(journal.open, trigger="heartbeat")
        journal_finalized = False

        # ── Background model selection ──
        original_config = agent.model_config
        bg_config = self._resolve_background_config("heartbeat")
        active_config = bg_config or original_config

        try:
            from core.config.models import load_config as _load_config_fresh

            _cfg = _load_config_fresh()
            _hb_cfg = _cfg.heartbeat
            _soft_timeout = _hb_cfg.soft_timeout_seconds
            _hard_timeout = _hb_cfg.hard_timeout_seconds
            _start = time.monotonic()
            _soft_warned = False
            _hard_exceeded = False

            async def _run(config):  # noqa: ANN001
                nonlocal accumulated_text, _hard_exceeded, _soft_warned
                if config is not agent.model_config:
                    agent.update_model_config(config)
                attempt_text = ""
                attempt_result: CycleResult | None = None
                stream = agent.run_cycle_streaming(
                    prompt,
                    trigger="heartbeat",
                    prior_messages=prior_messages,
                )
                try:
                    async for chunk in stream:
                        # ── Timeout checks (Mode A: reminder_queue injection) ──
                        _elapsed = time.monotonic() - _start
                        if not _soft_warned and _elapsed > _soft_timeout:
                            _soft_warned = True
                            agent._executor.reminder_queue.push_sync(t("reminder.hb_time_limit"))
                            logger.info(
                                "[%s] Heartbeat soft timeout reached (%.0fs > %ds)",
                                self.name,
                                _elapsed,
                                _soft_timeout,
                            )
                        if _hard_timeout and _elapsed > _hard_timeout:
                            _hard_exceeded = True
                            logger.warning(
                                "[%s] Heartbeat hard timeout reached (%.0fs > %ds) — breaking",
                                self.name,
                                _elapsed,
                                _hard_timeout,
                            )
                            break

                        if chunk.get("type") == "text_delta":
                            text = chunk.get("text", "")
                            attempt_text += text
                            await asyncio.to_thread(journal.write_text, text)
                        if chunk.get("type") == "cycle_done":
                            attempt_result = CycleResult.model_validate(
                                {
                                    "trigger": "heartbeat",
                                    "action": "responded",
                                    **chunk.get("cycle_result", {}),
                                }
                            )
                finally:
                    try:
                        await asyncio.wait_for(stream.aclose(), timeout=10)
                    except TimeoutError:
                        logger.warning(
                            "[%s] Timed out closing heartbeat stream after 10 seconds",
                            self.name,
                        )
                    except Exception:
                        logger.warning(
                            "[%s] Failed to close heartbeat stream",
                            self.name,
                            exc_info=True,
                        )
                accumulated_text = attempt_text
                return attempt_result or CycleResult(
                    trigger="heartbeat",
                    action="responded",
                    stop_kind="hard_timeout",
                    summary=attempt_text or "(no result)",
                )

            result = await run_with_model_fallback(
                _run,
                activity=self._activity,
                primary_config=active_config,
                active_config=active_config,
                channel="heartbeat",
            )

            # ── Hard timeout: write recovery note ──
            if _hard_exceeded:
                try:
                    await get_state_writer(self.anima_dir).write_recovery_note(
                        t("reminder.hb_hard_timeout_recovery", timeout=_hard_timeout)
                    )
                    logger.info("[%s] Hard timeout recovery note saved", self.name)
                except Exception:
                    logger.debug("[%s] Failed to save hard timeout recovery note", self.name, exc_info=True)

            if not journal_finalized:
                await asyncio.to_thread(journal.finalize, summary=result.summary[:500])
                journal_finalized = True

            self._last_activity = now_local()

            # Activity log: heartbeat end (with plan summary for plan-outcome tracking)
            _plan_summary = _extract_plan_summary(accumulated_text)
            _hb_meta: dict[str, Any] = {"plan_summary": _plan_summary} if _plan_summary else {}
            if result.action == "error":
                _hb_meta.update({"status": "failed", "reason": result.reason})
            await self._activity.alog("heartbeat_end", summary=result.summary, meta=_hb_meta)

            # Session boundary finalization moved to run_heartbeat()'s finally block,
            # so a hard timeout / cancellation cannot skip it.

            # A-3: Record important heartbeat actions to episodes
            if result.action != "error" and result.summary and "HEARTBEAT_OK" not in result.summary:
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
                    await self._activity.alog(
                        "heartbeat_reflection",
                        content=reflection_text,
                        summary=reflection_text[:200],
                    )

                try:
                    await asyncio.to_thread(self.memory.append_episode, episode_entry)
                except Exception:
                    logger.debug("[%s] Failed to record heartbeat episode", self.name, exc_info=True)

            logger.info(
                "[%s] run_heartbeat END duration_ms=%d unread_processed=%d",
                self.name,
                result.duration_ms,
                unread_count,
            )
            # Heartbeat completed successfully — remove checkpoint
            if result.action != "error":
                try:
                    await get_state_writer(self.anima_dir).clear_heartbeat_checkpoint()
                except Exception:
                    logger.debug("[%s] Failed to remove heartbeat checkpoint", self.name, exc_info=True)

            # Compact task queue after heartbeat
            try:
                from core.tasks.queue import TaskQueueManager

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

            # Keep current_state.md across normal heartbeat boundaries. It is
            # working memory, not a disposable session scratchpad; only trim it
            # when an explicit size limit is configured.
            await asyncio.to_thread(self._enforce_state_size_limit)

            return result
        finally:
            if agent.model_config is not original_config:
                agent.update_model_config(original_config)
            await asyncio.to_thread(journal.close)

    async def _handle_heartbeat_failure(
        self: _HeartbeatHost,
        error: Exception,
        inbox_items: list[InboxItem],
        unread_count: int,
    ) -> None:
        """Keep unread work on failure, log the error, and save recovery state."""
        logger.exception("[%s] run_heartbeat FAILED", self.name)

        # Failed model execution never acknowledges unread requests. Retry
        # cadence remains owned by the scheduler/watcher, not a local loop.

        # Activity log: heartbeat failure (single event to avoid double-fault)
        await self._activity.alog(
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
            recovery_content = t(
                "anima.recovery_error_info",
                exc_type=type(error).__name__,
                exc_msg=str(error)[:200],
                ts=now_iso(),
                count=unread_count,
            )
            await get_state_writer(self.anima_dir).write_recovery_note(recovery_content)
            logger.info("[%s] Recovery note saved", self.name)
        except Exception:
            logger.debug("[%s] Failed to save recovery note", self.name, exc_info=True)

        # Clean up orphaned streaming journal in-process so that
        # the next restart does not misreport it as a "crash recovery".
        try:
            cleaned = await asyncio.to_thread(_cleanup_orphaned_heartbeat_journal, self.anima_dir)
            if cleaned:
                logger.info("[%s] Cleaned up orphaned streaming journal", self.name)
        except Exception:
            logger.debug(
                "[%s] Failed to clean up streaming journal",
                self.name,
                exc_info=True,
            )

    # ── run_heartbeat orchestrator ───────────────────────────

    def _trigger_pending_task_execution(self: _HeartbeatHost) -> None:
        """Signal PendingTaskExecutor to check for new tasks.

        Called after heartbeat completion to ensure tasks written
        during planning phase are picked up promptly.
        """
        from core.tasks.wake import request_wake

        request_wake(self.name)
