from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""LifecycleMixin -- heartbeat orchestration, consolidation, cron execution.

Extracted from ``core.anima.DigitalAnima`` as a Mixin.  All ``self``
references are resolved at runtime via MRO when mixed into ``DigitalAnima``.
"""

import asyncio
import logging
import time
from typing import Any

from core.execution._sanitize import ORIGIN_SYSTEM
from core.i18n import t
from core.paths import load_prompt
from core.schemas import CycleResult
from core.time_utils import now_local

logger = logging.getLogger("animaworks.anima")


class LifecycleMixin:
    """Mixin: heartbeat orchestration, memory consolidation, cron task execution."""

    async def run_heartbeat(
        self,
        cascade_suppressed_senders: set[str] | None = None,
    ) -> CycleResult:
        self._get_interrupt_event("_background").clear()
        self.agent.set_interrupt_event(self._get_interrupt_event("_background"))
        logger.info("[%s] run_heartbeat START", self.name)
        try:
            async with self._background_lock:
                self._mark_busy_start()
                self._status_slots["background"] = "checking"
                self._last_heartbeat = now_local()

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
                    from core.config.models import load_config as _load_cfg
                    from core.tooling.handler import active_session_type

                    _session_token = self.agent._tool_handler.set_active_session_type("background")
                    self.agent._tool_handler.set_session_origin(ORIGIN_SYSTEM)
                    heartbeat_text = "\n\n".join(parts)
                    prior_msgs = self._build_prior_messages(heartbeat_text)
                    _hard_timeout = _load_cfg().heartbeat.hard_timeout_seconds
                    try:
                        result = await asyncio.wait_for(
                            self._execute_heartbeat_cycle(
                                heartbeat_text,
                                [],
                                0,
                                prior_messages=prior_msgs,
                            ),
                            timeout=float(_hard_timeout),
                        )
                    except TimeoutError:
                        logger.warning(
                            "[%s] Heartbeat hard timeout (%ds) — forced termination",
                            self.name,
                            _hard_timeout,
                        )
                        try:
                            recovery_path = self.anima_dir / "state" / "recovery_note.md"
                            recovery_path.write_text(
                                t("reminder.hb_hard_timeout_recovery", timeout=_hard_timeout),
                                encoding="utf-8",
                            )
                        except Exception:
                            logger.debug("[%s] Failed to write timeout recovery note", self.name, exc_info=True)
                        self._activity.log(
                            "heartbeat_end",
                            summary=f"[TIMEOUT] Hard timeout after {_hard_timeout}s",
                            meta={"status": "timeout", "hard_timeout_s": _hard_timeout},
                            safe=True,
                        )
                        return CycleResult(
                            trigger="heartbeat",
                            action="timeout",
                            summary=f"Hard timeout after {_hard_timeout}s",
                            duration_ms=_hard_timeout * 1000,
                        )
                    finally:
                        active_session_type.reset(_session_token)

                    return result

                except Exception as exc:
                    _unread = 0
                    try:
                        _unread = self.messenger.unread_count()
                    except OSError:
                        pass
                    await self._handle_heartbeat_failure(exc, [], _unread)
                    raise
                finally:
                    self._status_slots["background"] = "idle"
                    self._task_slots["background"] = ""
        finally:
            self._notify_lock_released()
            # Signal pending task execution after heartbeat completes
            self._trigger_pending_task_execution()

    # ── Consolidation helpers ──────────────────────────────────

    def _collect_episodes_summary(self) -> tuple[str, str, str, str]:
        """Collect recent episodes, resolved events, activity log, and reflections.

        Returns:
            Tuple of (episodes_summary, resolved_events_summary,
            activity_log_summary, reflections_summary).
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
            episodes_summary = "\n\n".join(f"## {e['date']} {e['time']}\n{e['content']}" for e in episodes)
        else:
            return (t("anima.no_episodes_today"), "", activity_log_summary, "")

        # Extract reflections from episodes
        reflections_summary = engine._extract_reflections_from_episodes(episodes_summary)

        # Format resolved events
        if resolved:
            resolved_events_summary = "\n".join(f"- {r['ts'][:16]}: {r['content']}" for r in resolved)
        else:
            resolved_events_summary = ""

        return (episodes_summary, resolved_events_summary, activity_log_summary, reflections_summary)

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
            self.name,
            consolidation_type,
            max_turns,
        )
        from core.tooling.handler import active_session_type

        try:
            async with self._background_lock:
                self._mark_busy_start()
                self._status_slots["background"] = "consolidating"
                self._task_slots["background"] = f"Memory consolidation ({consolidation_type})"
                _session_token = self.agent._tool_handler.set_active_session_type("background")
                self.agent._tool_handler.set_session_origin(ORIGIN_SYSTEM)

                try:
                    # Build consolidation prompt
                    if consolidation_type == "daily":
                        episodes_summary, resolved_events_summary, activity_log_summary, reflections_summary = (
                            self._collect_episodes_summary()
                        )
                        reflections_section = ""
                        if reflections_summary:
                            reflections_section = (
                                "## "
                                + t("anima.reflections_header")
                                + "\n\n"
                                + t("anima.reflections_intro")
                                + "\n\n"
                                + reflections_summary
                            )
                        prompt = load_prompt(
                            "memory/consolidation_instruction",
                            anima_name=self.name,
                            episodes_summary=episodes_summary,
                            resolved_events_summary=resolved_events_summary,
                            activity_log_summary=activity_log_summary or t("anima.no_activity_log"),
                            reflections_summary=reflections_section,
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
                    self._last_activity = now_local()

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
                        self.name,
                        consolidation_type,
                        result.duration_ms,
                    )
                    return result

                except Exception as exc:
                    logger.exception(
                        "[%s] run_consolidation FAILED type=%s",
                        self.name,
                        consolidation_type,
                    )
                    self._activity.log(
                        "error",
                        summary=t("anima.consolidation_error", exc=type(exc).__name__),
                        meta={"phase": "run_consolidation", "error": str(exc)[:200]},
                        safe=True,
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
        self._get_interrupt_event("_background").clear()
        self.agent.set_interrupt_event(self._get_interrupt_event("_background"))
        logger.info("[%s] run_cron_task START task=%s", self.name, task_name)
        from core.tooling.handler import active_session_type

        try:
            async with self._background_lock:
                self._mark_busy_start()
                self._cron_idle.clear()
                self._status_slots["background"] = "working"
                self._task_slots["background"] = task_name
                _session_token = self.agent._tool_handler.set_active_session_type("background")
                self.agent._tool_handler.set_session_origin(ORIGIN_SYSTEM)

                prompt = self._build_cron_prompt(
                    task_name,
                    description,
                    command_output=command_output,
                )

                # ── Background model swap ──
                original_config = None
                bg_config = self._resolve_background_config()
                if bg_config is not None:
                    original_config = self.agent.model_config
                    self.agent.update_model_config(bg_config)

                try:
                    result = await self.agent.run_cycle(prompt, trigger=f"cron:{task_name}")
                    self._last_activity = now_local()

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
                        self.name,
                        task_name,
                        result.duration_ms,
                    )
                    return result
                except Exception as exc:
                    logger.exception(
                        "[%s] run_cron_task FAILED task=%s",
                        self.name,
                        task_name,
                    )
                    # Activity log: error (safe=True to prevent double-fault)
                    self._activity.log(
                        "error",
                        summary=t("anima.cron_task_error", exc=type(exc).__name__),
                        meta={"phase": "run_cron_task", "error": str(exc)[:200]},
                        safe=True,
                    )
                    raise
                finally:
                    if original_config is not None:
                        self.agent.update_model_config(original_config)
                    active_session_type.reset(_session_token)
                    self._cron_idle.set()
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
        logger.info("[%s] run_cron_command START task=%s", self.name, task_name)
        start_ms = time.time_ns() // 1_000_000

        stdout = ""
        stderr = ""
        exit_code = 0

        from core.tooling.handler import active_session_type

        try:
            async with self._background_lock:
                self._mark_busy_start()
                self._cron_idle.clear()
                self._status_slots["background"] = "working"
                self._task_slots["background"] = task_name
                _session_token = self.agent._tool_handler.set_active_session_type("background")
                self.agent._tool_handler.set_session_origin(ORIGIN_SYSTEM)

                proc = None
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

                    self._last_activity = now_local()

                except Exception as exc:
                    stderr = f"{type(exc).__name__}: {exc}"
                    exit_code = 1
                    logger.exception("[%s] run_cron_command FAILED task=%s", self.name, task_name)
                    # Activity log: error (safe=True to prevent double-fault)
                    self._activity.log(
                        "error",
                        summary=t("anima.cron_cmd_error", exc=type(exc).__name__),
                        meta={"phase": "run_cron_command", "error": str(exc)[:200]},
                        safe=True,
                    )
                finally:
                    if proc is not None and proc.returncode is None:
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                        try:
                            await proc.wait()
                        except ProcessLookupError:
                            pass
                    active_session_type.reset(_session_token)
                    self._cron_idle.set()
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
