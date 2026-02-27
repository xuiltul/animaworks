# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Pending task watcher and executor.

Monitors state/background_tasks/pending/ for tasks submitted via
``animaworks-tool submit`` and dispatches them through
BackgroundTaskManager.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.anima import DigitalAnima

logger = logging.getLogger(__name__)

_PENDING_WATCHER_POLL_INTERVAL = 3.0
_LLM_TASK_TTL_HOURS = 24
_PENDING_TASK_SUBPROCESS_TIMEOUT = 1800


class PendingTaskExecutor:
    """Watch pending/ directory and execute submitted tasks."""

    def __init__(
        self,
        anima: DigitalAnima,
        anima_name: str,
        anima_dir: Path,
        shutdown_event: asyncio.Event,
    ) -> None:
        self._anima = anima
        self._anima_name = anima_name
        self._anima_dir = anima_dir
        self._shutdown_event = shutdown_event
        self._wake_event = asyncio.Event()

    async def watcher_loop(self) -> None:
        """Watch state/background_tasks/pending/ for submitted tasks.

        Tasks submitted via ``animaworks-tool submit`` are picked up here
        and executed through BackgroundTaskManager, outside the Anima lock.
        """
        pending_dir = self._anima_dir / "state" / "background_tasks" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        # Also watch state/pending/ for LLM tasks
        llm_pending_dir = self._anima_dir / "state" / "pending"
        llm_pending_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pending task watcher started for %s", self._anima_name)

        while not self._shutdown_event.is_set():
            try:
                for path in sorted(pending_dir.glob("*.json")):
                    try:
                        task_desc = json.loads(path.read_text(encoding="utf-8"))
                        path.unlink()  # Remove immediately to prevent double execution
                        logger.info(
                            "Picked up pending task: id=%s tool=%s subcmd=%s anima=%s",
                            task_desc.get("task_id", "?"),
                            task_desc.get("tool_name", "?"),
                            task_desc.get("subcommand", ""),
                            self._anima_name,
                        )
                        await self.execute_pending_task(task_desc)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Invalid JSON in pending task file: %s", path.name,
                        )
                        path.unlink(missing_ok=True)
                    except Exception:
                        logger.exception(
                            "Error processing pending task file: %s", path.name,
                        )

                # Scan LLM pending tasks
                for path in sorted(llm_pending_dir.glob("*.json")):
                    try:
                        task_desc = json.loads(path.read_text(encoding="utf-8"))
                        path.unlink()
                        logger.info(
                            "Picked up LLM pending task: id=%s anima=%s",
                            task_desc.get("task_id", "?"),
                            self._anima_name,
                        )
                        await self.execute_pending_task(task_desc)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Invalid JSON in LLM pending task file: %s", path.name,
                        )
                        path.unlink(missing_ok=True)
                    except Exception:
                        logger.exception(
                            "Error processing LLM pending task file: %s", path.name,
                        )

                try:
                    await asyncio.wait_for(
                        self._wake_event.wait(),
                        timeout=_PENDING_WATCHER_POLL_INTERVAL,
                    )
                    self._wake_event.clear()
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(
                    "Error in pending task watcher for %s", self._anima_name,
                )
                await asyncio.sleep(_PENDING_WATCHER_POLL_INTERVAL)

        logger.info("Pending task watcher stopped for %s", self._anima_name)

    def wake(self) -> None:
        """Signal the watcher to check for new tasks immediately."""
        self._wake_event.set()

    async def execute_pending_task(self, task_desc: dict[str, Any]) -> None:
        """Execute a pending task via BackgroundTaskManager or LLM.

        Routes by task_type: 'llm' → _execute_llm_task, else command subprocess.
        """
        task_type = task_desc.get("task_type", "command")

        if task_type == "llm":
            await self._execute_llm_task(task_desc)
            return

        if not self._anima:
            logger.warning("Cannot execute pending task: anima not initialized")
            return

        bg_mgr = self._anima.agent.background_manager
        if not bg_mgr:
            logger.warning(
                "Cannot execute pending task: BackgroundTaskManager not available",
            )
            return

        tool_name = task_desc.get("tool_name", "")
        subcommand = task_desc.get("subcommand", "")
        raw_args = task_desc.get("raw_args", [])
        anima_dir = task_desc.get("anima_dir", str(self._anima_dir))

        # Build tool args dict for ExternalToolDispatcher
        tool_args = {
            "subcommand": subcommand,
            "raw_args": raw_args,
            "anima_dir": anima_dir,
        }

        task_id = task_desc.get("task_id", "")

        logger.info(
            "Submitting pending task to BackgroundTaskManager: "
            "id=%s tool=%s subcmd=%s",
            task_id, tool_name, subcommand,
        )

        def _dispatch_fn(name: str, args: dict[str, Any]) -> str:
            """Execute the tool via CLI subprocess (same as direct execution)."""
            import os
            import subprocess

            cmd = ["animaworks-tool", name]
            subcmd = args.get("subcommand", "")
            if subcmd:
                cmd.append(subcmd)
            cmd.extend(args.get("raw_args", []))
            # Remove subcommand from raw_args if it's already the first element
            if subcmd and args.get("raw_args") and args["raw_args"][0] == subcmd:
                cmd = ["animaworks-tool", name] + args["raw_args"]
            cmd.append("-j")

            env = {
                **os.environ,
                "ANIMAWORKS_ANIMA_DIR": args.get("anima_dir", ""),
            }

            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=_PENDING_TASK_SUBPROCESS_TIMEOUT, env=env,
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() or f"Exit code {result.returncode}"
                raise RuntimeError(f"Tool {name} failed: {error_msg}")
            return result.stdout.strip()

        # Submit to BackgroundTaskManager
        composite_name = f"{tool_name}:{subcommand}" if subcommand else tool_name
        bg_mgr.submit(composite_name, tool_args, _dispatch_fn)

    async def _execute_llm_task(self, task_desc: dict[str, Any]) -> None:
        """Execute an LLM task under _background_lock.

        The task is executed as a minimal-context LLM session using
        the task_exec.md template. Only task description and tools
        are provided — no memory, no org context.
        """
        if self._anima and hasattr(self._anima, "_interrupt_event"):
            self._anima._interrupt_event.clear()
        task_id = task_desc.get("task_id", "unknown")
        title = task_desc.get("title", "Untitled task")
        description = task_desc.get("description", "")
        context = task_desc.get("context", "")
        acceptance_criteria = task_desc.get("acceptance_criteria", [])
        constraints = task_desc.get("constraints", [])
        file_paths = task_desc.get("file_paths", [])
        reply_to = task_desc.get("reply_to")
        submitted_by = task_desc.get("submitted_by", "unknown")
        submitted_at = task_desc.get("submitted_at", "")

        # TTL check
        if submitted_at:
            try:
                sub_dt = datetime.fromisoformat(submitted_at)
                if sub_dt.tzinfo is None:
                    sub_dt = sub_dt.replace(tzinfo=timezone.utc)
                now_utc = datetime.now(timezone.utc)
                age_hours = (now_utc - sub_dt).total_seconds() / 3600
                if age_hours > _LLM_TASK_TTL_HOURS:
                    logger.warning(
                        "[%s] Skipping expired LLM task: %s (age=%.1fh, TTL=%dh)",
                        self._anima_name, task_id, age_hours, _LLM_TASK_TTL_HOURS,
                    )
                    return
            except (ValueError, TypeError):
                logger.debug(
                    "[%s] Could not parse submitted_at for task %s",
                    self._anima_name, task_id,
                )

        logger.info(
            "[%s] Executing LLM task: id=%s title=%s",
            self._anima_name, task_id, title,
        )

        try:
            async with self._anima._background_lock:
                self._anima._status_slots["background"] = "task_exec"
                self._anima._task_slots["background"] = task_id

                from core.paths import load_prompt
                from core.memory.streaming_journal import StreamingJournal
                from core.memory.activity import ActivityLogger

                activity = ActivityLogger(self._anima_dir)
                activity.log(
                    "task_exec_start",
                    summary=f"タスク実行開始: {title}",
                    meta={"task_id": task_id, "submitted_by": submitted_by},
                )

                criteria_text = (
                    "\n".join(f"- {c}" for c in acceptance_criteria)
                    if acceptance_criteria
                    else "(なし)"
                )
                constraints_text = (
                    "\n".join(f"- {c}" for c in constraints)
                    if constraints
                    else "(なし)"
                )
                paths_text = (
                    "\n".join(f"- {p}" for p in file_paths)
                    if file_paths
                    else "(なし)"
                )

                prompt = load_prompt(
                    "task_exec",
                    task_id=task_id,
                    title=title,
                    submitted_by=submitted_by,
                    description=description,
                    context=context or "(なし)",
                    acceptance_criteria=criteria_text,
                    constraints=constraints_text,
                    file_paths=paths_text,
                )

                trigger = f"task:{task_id}"
                journal = StreamingJournal(self._anima_dir, session_type="task_exec")
                journal.open(trigger=trigger)

                self._anima.agent.reset_reply_tracking(session_type="background")
                accumulated_text = ""
                result_summary = ""

                try:
                    async for chunk in self._anima.agent.run_cycle_streaming(
                        prompt, trigger=trigger,
                    ):
                        if chunk.get("type") == "text_delta":
                            accumulated_text += chunk.get("text", "")
                            journal.write_text(chunk.get("text", ""))

                        if chunk.get("type") == "cycle_done":
                            cycle_result = chunk.get("cycle_result", {})
                            result_summary = cycle_result.get(
                                "summary", accumulated_text[:500]
                            )
                            journal.finalize(summary=result_summary[:500])
                finally:
                    journal.close()

                if not result_summary:
                    result_summary = accumulated_text[:500] or "(タスク完了)"

                activity.log(
                    "task_exec_end",
                    summary=f"タスク完了: {title} — {result_summary[:200]}",
                    meta={"task_id": task_id},
                )

                # Send completion notification if reply_to is set
                if reply_to:
                    try:
                        notify_text = load_prompt(
                            "task_complete_notify",
                            task_id=task_id,
                            title=title,
                            result_summary=result_summary[:1000],
                        )
                        self._anima.messenger.send(to=reply_to, content=notify_text)
                        logger.info(
                            "[%s] Task completion notification sent to %s",
                            self._anima_name, reply_to,
                        )
                    except Exception:
                        logger.warning(
                            "[%s] Failed to send task completion notification to %s",
                            self._anima_name, reply_to, exc_info=True,
                        )

                self._anima._status_slots["background"] = "idle"
                self._anima._task_slots["background"] = ""

                logger.info(
                    "[%s] LLM task completed: id=%s",
                    self._anima_name, task_id,
                )

        except Exception:
            logger.exception(
                "[%s] LLM task failed: id=%s",
                self._anima_name, task_id,
            )
            self._anima._status_slots["background"] = "idle"
            self._anima._task_slots["background"] = ""
