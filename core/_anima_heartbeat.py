from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""HeartbeatMixin -- heartbeat/cron prompt construction and cycle execution.

Extracted from ``core.anima.DigitalAnima`` as a Mixin.  All ``self``
references are resolved at runtime via MRO when mixed into ``DigitalAnima``.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from core.execution.fallback_activity import run_with_model_fallback
from core.i18n import t
from core.memory.conversation import ConversationMemory
from core.memory.streaming_journal import StreamingJournal
from core.messenger import InboxItem
from core.paths import load_prompt
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
        from core.memory.task_queue import TaskQueueManager

        now = now_local()
        rows: list[tuple[float, str]] = []
        for task in TaskQueueManager(anima_dir).get_all_active():
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
            rows.append(
                (
                    elapsed_seconds,
                    f"- {marker}{task.task_id[:12]} | {task.status} | {hours}h | {summary}",
                )
            )

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


def _build_cron_rejected_notice(anima_dir: Path, name: str) -> str | None:
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
        try:
            from core.memory._io import atomic_write_text

            marker_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(marker_path, digest + "\n")
        except Exception:
            logger.warning("[%s] Failed to persist rejected cron notice marker", name, exc_info=True)
        return notice
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        logger.debug("[%s] Failed to build rejected cron notice", name, exc_info=True)
        return None


class HeartbeatMixin:
    """Mixin: heartbeat/cron prompt building, cycle execution, failure handling."""

    # ── Background model resolution ──────────────────────────

    def _resolve_background_config(self, channel: str = "background") -> ModelConfig | None:  # noqa: F821
        """Resolve background model config for heartbeat/cron.

        Resolution order:
          1. status.json background_model (per-anima)
          2. config.heartbeat.default_model (global)
          3. main model

        The selected base config then passes through the shared rate-guard
        fallback preflight used by heartbeat, cron, and inbox cycles.
        """
        from core.config.model_config import (
            _FAMILY_CREDENTIAL_MAP,
            _model_family,
            infer_mode_s_auth,
            resolve_effective_model_config,
        )
        from core.config.models import load_config, resolve_execution_mode
        from core.execution.fallback_activity import log_model_fallback
        from core.schemas import ModelConfig

        main_config = self.agent.model_config
        bg_model = main_config.background_model
        bg_effort = main_config.background_thinking_effort
        if not bg_model:
            config = load_config()
            bg_model = config.heartbeat.default_model
        if not bg_model or bg_model == main_config.model:
            # Same model: only thinking_effort may differ for background runs.
            if bg_effort and bg_effort != main_config.thinking_effort:
                base_config = main_config.model_copy(update={"thinking_effort": bg_effort})
            else:
                base_config = main_config
        else:
            # Recalculate resolved_mode for the background model so that
            # the correct executor type is created (e.g. claude-* → S, codex/* → C).
            # Without this, model_copy carries the main model's resolved_mode,
            # which may be incompatible with the background model name.
            config = load_config()
            bg_resolved_mode = resolve_execution_mode(config, bg_model)

            bg_credential = main_config.background_credential
            bg_family = _model_family(bg_model)
            main_family = _model_family(main_config.model)
            if not bg_credential and bg_family != main_family:
                mapped_credential = _FAMILY_CREDENTIAL_MAP.get(bg_family)
                if mapped_credential in config.credentials:
                    bg_credential = mapped_credential

            updates: dict[str, Any] = {
                "model": bg_model,
                "resolved_mode": bg_resolved_mode,
            }
            if bg_effort:
                updates["thinking_effort"] = bg_effort
            if bg_credential:
                if bg_credential in config.credentials:
                    cred = config.credentials[bg_credential]
                    updates.update(
                        {
                            "background_credential": bg_credential,
                            "api_key": cred.api_key or None,
                            "api_key_env": f"{bg_credential.upper()}_API_KEY",
                            "api_base_url": cred.base_url or None,
                            "extra_keys": dict(cred.keys) if cred.keys else {},
                        }
                    )
                    if bg_resolved_mode == "S" and not main_config.mode_s_auth:
                        updates["mode_s_auth"] = infer_mode_s_auth(
                            mode=bg_resolved_mode,
                            credential_name=bg_credential,
                            config=config,
                        )
                    elif bg_resolved_mode != "S":
                        updates["mode_s_auth"] = None
            base_config = main_config.model_copy(update=updates)

        # A few lifecycle unit tests deliberately install a generic MagicMock
        # model config.  Preserve the pre-existing background resolution result
        # for those test doubles; runtime AgentCore configs are ModelConfig.
        if not isinstance(base_config, ModelConfig):
            return base_config

        effective_config = resolve_effective_model_config(base_config)
        activity = getattr(self, "_activity", None)
        if activity is not None:
            log_model_fallback(
                activity,
                base_config,
                effective_config,
                channel=channel,
                phase="preflight",
            )

        # Preserve the existing no-swap signal when neither a background
        # override nor a rate-guard fallback changed the main config.
        if base_config is main_config and effective_config is main_config:
            return None
        return effective_config

    # ── Heartbeat history ────────────────────────────────────

    _HEARTBEAT_HISTORY_N = 3

    _PLAN_OUTCOME_MAX_CHARS = 200

    def _load_heartbeat_history(self) -> str:
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
                        lines.append(t("heartbeat.history_plan_entry", ts=ts_short, plan=plan[:limit]))
                    else:
                        summary = (e.summary or e.content)[:limit]
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

    def _get_current_state_max_chars(self) -> int:
        try:
            from core.config.models import load_config

            return load_config().heartbeat.current_state_max_chars
        except Exception:
            return 0

    def _enforce_state_size_limit(self) -> None:
        """Hard-trim current_state.md if it exceeds the configured threshold.

        Called after heartbeat completion.  Overflow content is archived
        into today's episode file for traceability.
        Disabled when ``heartbeat.current_state_max_chars`` is 0 (default).
        """
        max_chars = self._get_current_state_max_chars()
        if max_chars <= 0:
            return
        state = self.memory.read_current_state()
        if len(state) <= max_chars:
            return
        trimmed = state[-max_chars:]
        first_nl = trimmed.find("\n")
        if first_nl != -1 and first_nl < max_chars * 0.2:
            trimmed = trimmed[first_nl + 1 :]
        overflow = state[: len(state) - len(trimmed)]
        self.memory.append_episode(f"## current_state.md overflow archived\n\n{overflow}")
        self.memory.update_state(trimmed)
        logger.info(
            "[%s] current_state.md hard-trimmed: %d → %d chars",
            self.name,
            len(state),
            len(trimmed),
        )

    def _build_state_cleanup_instruction(self) -> str | None:
        """Return a self-cleanup instruction when current_state.md nears the limit.

        Triggers at 80% of ``heartbeat.current_state_max_chars``: the
        post-session hard trim leaves the file just *under* the limit, so a
        trigger at 100% would never fire again once trimming starts — the
        anima would stay in a machine-trim equilibrium and never see the
        instruction.  Returns ``None`` when trimming is disabled or the
        state is below the soft threshold.
        """
        max_chars = self._get_current_state_max_chars()
        if max_chars <= 0:
            return None
        state = self.memory.read_current_state()
        state_len = len(state)
        soft_threshold = int(max_chars * 0.8)
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
            max_chars=max_chars,
            target_chars=max_chars // 2,
        )

    async def _build_heartbeat_prompt(self) -> list[str]:
        """Build heartbeat prompt parts.

        Heartbeat-specific header + shared background context.
        When current_state.md nears the cleanup threshold, a compression
        instruction is prepended so the anima trims it first.
        """
        hb_config = self.memory.read_heartbeat_config()
        checklist = hb_config or load_prompt("heartbeat_default_checklist")
        parts = [load_prompt("heartbeat", checklist=checklist)]

        cleanup = self._build_state_cleanup_instruction()
        if cleanup:
            parts.append(cleanup)

        cron_notice = _build_cron_rejected_notice(self.anima_dir, self.name)
        if cron_notice:
            parts.append(cron_notice)

        parts.extend(self._build_background_context_parts())

        scoreboard = _build_stale_task_scoreboard(self.anima_dir, self.name)
        if scoreboard:
            parts.append(scoreboard)

        curator_part = _build_curator_review_part(self.anima_dir, self.name)
        if curator_part:
            parts.append(curator_part)

        return parts

    def _build_cron_prompt(
        self,
        task_name: str,
        description: str,
        command_output: str | None = None,
        skills: list[str] | None = None,
        skill_rejections_out: list[SkillContextRejection] | None = None,
        skill_warnings_out: list[SkillContextWarning] | None = None,
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
        agent = self._agent_for_lane("background") if hasattr(self, "_agent_for_lane") else self.agent
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
        agent.reset_reply_tracking(session_type="heartbeat")
        agent.reset_posted_channels(session_type="heartbeat")
        agent.reset_read_paths()

        accumulated_text = ""
        result: CycleResult | None = None

        # Streaming journal for heartbeat crash recovery
        journal = StreamingJournal(self.anima_dir, session_type="heartbeat")
        journal.open(trigger="heartbeat")
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
                            journal.write_text(text)
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
                    recovery_path = self.anima_dir / "state" / "recovery_note.md"
                    recovery_path.write_text(
                        t("reminder.hb_hard_timeout_recovery", timeout=_hard_timeout),
                        encoding="utf-8",
                    )
                    logger.info("[%s] Hard timeout recovery note saved", self.name)
                except Exception:
                    logger.debug("[%s] Failed to save hard timeout recovery note", self.name, exc_info=True)

            if not journal_finalized:
                journal.finalize(summary=result.summary[:500])
                journal_finalized = True

            self._last_activity = now_local()

            # Activity log: heartbeat end (with plan summary for plan-outcome tracking)
            _plan_summary = _extract_plan_summary(accumulated_text)
            _hb_meta: dict[str, Any] = {"plan_summary": _plan_summary} if _plan_summary else {}
            if result.action == "error":
                _hb_meta.update({"status": "failed", "reason": result.reason})
            self._activity.log("heartbeat_end", summary=result.summary, meta=_hb_meta)

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
            if result.action != "error":
                try:
                    checkpoint_path.unlink(missing_ok=True)
                except Exception:
                    logger.debug("[%s] Failed to remove heartbeat checkpoint", self.name, exc_info=True)

            # Sync delegated tasks then compact task queue after heartbeat
            try:
                from core.memory.task_queue import TaskQueueManager
                from core.paths import get_animas_dir

                _tqm = TaskQueueManager(self.anima_dir)
                _synced = _tqm.sync_delegated(get_animas_dir())
                if _synced:
                    logger.info(
                        "[%s] Synced %d delegated tasks from subordinates",
                        self.name,
                        _synced,
                    )
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
            self._enforce_state_size_limit()

            return result
        finally:
            if agent.model_config is not original_config:
                agent.update_model_config(original_config)
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
