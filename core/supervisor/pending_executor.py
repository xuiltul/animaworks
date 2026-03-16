# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Pending task watcher and executor.

Monitors state/background_tasks/pending/ for tasks submitted via
``animaworks-tool submit`` and dispatches them through
BackgroundTaskManager.  Supports DAG-based parallel execution
for batched tasks submitted via ``submit_tasks`` tool.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.exceptions import ToolExecutionError
from core.i18n import t

if TYPE_CHECKING:
    from core.anima import DigitalAnima

logger = logging.getLogger(__name__)

_PENDING_WATCHER_POLL_INTERVAL = 3.0
_LLM_TASK_TTL_HOURS = 24
_PENDING_TASK_SUBPROCESS_TIMEOUT = 1800
_TASK_RESULT_MAX_CHARS = 2000


def _resolve_default_workspace(anima_dir: Path) -> str:
    """Resolve default_workspace from status.json via workspace registry.

    Returns absolute path string, or empty string if not set or resolution fails.
    """
    from core.workspace import resolve_default_workspace

    resolved, _alias = resolve_default_workspace(anima_dir)
    return str(resolved) if resolved else ""


# ── DAG helpers ──────────────────────────────────────────────


def _topological_sort(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tasks in topological order. Raises ValueError on cycles."""
    task_map = {td["task_id"]: td for td in tasks}
    in_degree: dict[str, int] = {tid: 0 for tid in task_map}
    for td in tasks:
        for dep in td.get("depends_on", []):
            if dep in in_degree:
                in_degree[td["task_id"]] += 1

    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    result: list[dict[str, Any]] = []
    while queue:
        tid = queue.pop(0)
        result.append(task_map[tid])
        for td in tasks:
            if tid in td.get("depends_on", []):
                in_degree[td["task_id"]] -= 1
                if in_degree[td["task_id"]] == 0:
                    queue.append(td["task_id"])

    if len(result) != len(tasks):
        raise ValueError("Cycle detected in task dependencies")
    return result


def _deps_satisfied(
    task: dict[str, Any],
    completed: dict[str, str],
    failed: set[str],
) -> bool:
    """Check if all dependencies are either completed or failed."""
    for dep in task.get("depends_on", []):  # noqa: SIM110
        if dep not in completed and dep not in failed:
            return False
    return True


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
        self._batch_tasks: dict[str, list[dict[str, Any]]] = {}

    # ── Semaphore lazy init ──────────────────────────────────

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create the task semaphore from config."""
        if self._anima._task_semaphore is None:
            try:
                from core.config.models import load_config

                config = load_config()
                max_parallel = config.background_task.max_parallel_llm_tasks
            except Exception:
                max_parallel = 3
            self._anima._task_semaphore = asyncio.Semaphore(max_parallel)
        return self._anima._task_semaphore

    # ── Result save / dependency context ─────────────────────

    def _save_task_result(self, task_id: str, summary: str) -> None:
        """Save task result summary to state/task_results/{task_id}.md."""
        results_dir = self._anima_dir / "state" / "task_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"{task_id}.md"
        truncated = summary[:_TASK_RESULT_MAX_CHARS]
        path.write_text(truncated, encoding="utf-8")

    def _build_dependency_context(
        self,
        task_desc: dict[str, Any],
        completed: dict[str, str],
    ) -> str:
        """Build context from completed dependency results."""
        parts: list[str] = []
        for dep_id in task_desc.get("depends_on", []):
            result = completed.get(dep_id, "")
            if result:
                parts.append(t("pending_executor.dep_result_header", dep_id=dep_id) + f"\n{result}")
        return "\n\n".join(parts)

    def _write_failed_result(self, task_id: str, reason: str) -> None:
        """Write a failure marker for a task."""
        self._save_task_result(task_id, f"FAILED: {reason}")

    def _sync_task_queue(
        self,
        task_id: str,
        status: str,
        *,
        summary: str | None = None,
    ) -> None:
        """Sync task status to task_queue.jsonl (Layer 2).

        Silently skips if the task is not registered in task_queue
        (e.g., legacy tasks created before this sync was implemented).
        """
        try:
            from core.memory.task_queue import TaskQueueManager

            manager = TaskQueueManager(self._anima_dir)
            manager.update_status(task_id, status, summary=summary)
        except Exception:
            logger.warning(
                "[%s] Failed to sync task %s status=%s to task_queue",
                self._anima_name,
                task_id,
                status,
                exc_info=True,
            )

    # ── Watcher loop ─────────────────────────────────────────

    @staticmethod
    def _recover_processing(processing_dir: Path, failed_dir: Path) -> None:
        """Move orphaned files from processing/ to failed/ on startup."""
        if not processing_dir.exists():
            return
        for orphan in processing_dir.glob("*.json"):
            try:
                orphan.rename(failed_dir / orphan.name)
                logger.warning("Recovered orphaned processing task: %s", orphan.name)
            except OSError:
                logger.exception("Failed to recover orphaned task: %s", orphan.name)

    async def watcher_loop(self) -> None:
        """Watch state/background_tasks/pending/ for submitted tasks.

        Tasks submitted via ``animaworks-tool submit`` are picked up here
        and executed through BackgroundTaskManager, outside the Anima lock.
        Batch tasks (with ``batch_id``) are grouped and dispatched via the
        DAG scheduler for parallel execution.

        File lifecycle: pending/ → processing/ → success: delete | fail: failed/
        """
        pending_dir = self._anima_dir / "state" / "background_tasks" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        cmd_processing_dir = pending_dir / "processing"
        cmd_processing_dir.mkdir(exist_ok=True)
        cmd_failed_dir = pending_dir / "failed"
        cmd_failed_dir.mkdir(exist_ok=True)

        llm_pending_dir = self._anima_dir / "state" / "pending"
        llm_pending_dir.mkdir(parents=True, exist_ok=True)
        llm_processing_dir = llm_pending_dir / "processing"
        llm_processing_dir.mkdir(exist_ok=True)
        llm_failed_dir = llm_pending_dir / "failed"
        llm_failed_dir.mkdir(exist_ok=True)

        self._recover_processing(cmd_processing_dir, cmd_failed_dir)
        self._recover_processing(llm_processing_dir, llm_failed_dir)

        logger.info("Pending task watcher started for %s", self._anima_name)

        while not self._shutdown_event.is_set():
            try:
                # Process command-type pending tasks
                for path in sorted(pending_dir.glob("*.json")):
                    try:
                        task_desc = json.loads(path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Invalid JSON in pending task file: %s",
                            path.name,
                        )
                        path.unlink(missing_ok=True)
                        continue

                    try:
                        processing_path = cmd_processing_dir / path.name
                        path.rename(processing_path)
                    except OSError:
                        logger.exception(
                            "Failed to move task to processing: %s",
                            path.name,
                        )
                        continue

                    try:
                        logger.info(
                            "Picked up pending task: id=%s tool=%s subcmd=%s anima=%s",
                            task_desc.get("task_id", "?"),
                            task_desc.get("tool_name", "?"),
                            task_desc.get("subcommand", ""),
                            self._anima_name,
                        )
                        await self.execute_pending_task(task_desc)
                        processing_path.unlink(missing_ok=True)
                    except Exception:
                        logger.exception(
                            "Error processing pending task file: %s",
                            path.name,
                        )
                        try:
                            processing_path.rename(cmd_failed_dir / path.name)
                        except OSError:
                            logger.exception(
                                "Failed to move task to failed: %s",
                                path.name,
                            )

                # Scan LLM pending tasks — group batch tasks, execute serial ones
                for path in sorted(llm_pending_dir.glob("*.json")):
                    try:
                        task_desc = json.loads(path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Invalid JSON in LLM pending task file: %s",
                            path.name,
                        )
                        path.unlink(missing_ok=True)
                        continue

                    try:
                        processing_path = llm_processing_dir / path.name
                        path.rename(processing_path)
                    except OSError:
                        logger.exception(
                            "Failed to move LLM task to processing: %s",
                            path.name,
                        )
                        continue

                    try:
                        batch_id = task_desc.get("batch_id")
                        if batch_id:
                            self._batch_tasks.setdefault(batch_id, []).append(task_desc)
                            logger.info(
                                "Queued batch task: id=%s batch=%s anima=%s",
                                task_desc.get("task_id", "?"),
                                batch_id,
                                self._anima_name,
                            )
                        else:
                            task_id = task_desc.get("task_id", "")
                            # Skip if task was cancelled in task_queue
                            try:
                                from core.memory.task_queue import TaskQueueManager

                                entry = TaskQueueManager(self._anima_dir).get_task_by_id(task_id)
                                if entry and entry.status == "cancelled":
                                    logger.info(
                                        "[%s] Skipping cancelled LLM task: id=%s",
                                        self._anima_name,
                                        task_id,
                                    )
                                    try:
                                        processing_path.rename(llm_failed_dir / path.name)
                                    except OSError:
                                        logger.exception(
                                            "Failed to move cancelled task to failed: %s",
                                            path.name,
                                        )
                                    continue
                            except Exception:
                                logger.debug(
                                    "Could not check task_queue for cancellation: %s",
                                    task_id,
                                    exc_info=True,
                                )
                            logger.info(
                                "Picked up LLM pending task: id=%s anima=%s",
                                task_id,
                                self._anima_name,
                            )
                            await self.execute_pending_task(task_desc)
                        processing_path.unlink(missing_ok=True)
                    except Exception:
                        logger.exception(
                            "Error processing LLM pending task file: %s",
                            path.name,
                        )
                        try:
                            processing_path.rename(llm_failed_dir / path.name)
                        except OSError:
                            logger.exception(
                                "Failed to move LLM task to failed: %s",
                                path.name,
                            )

                # Dispatch accumulated batch tasks
                for batch_id, tasks in list(self._batch_tasks.items()):
                    del self._batch_tasks[batch_id]
                    await self._dispatch_batch(batch_id, tasks)

                try:
                    await asyncio.wait_for(
                        self._wake_event.wait(),
                        timeout=_PENDING_WATCHER_POLL_INTERVAL,
                    )
                    self._wake_event.clear()
                except TimeoutError:
                    pass
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(
                    "Error in pending task watcher for %s",
                    self._anima_name,
                )
                await asyncio.sleep(_PENDING_WATCHER_POLL_INTERVAL)

        logger.info("Pending task watcher stopped for %s", self._anima_name)

    # ── DAG batch dispatch ──────────────────────────────────────

    async def _dispatch_batch(
        self,
        batch_id: str,
        tasks: list[dict[str, Any]],
    ) -> None:
        """Dispatch a batch of tasks respecting DAG dependencies.

        Independent parallel tasks run under ``_task_semaphore``.
        Serial tasks (``parallel=false``) and dependency-gated tasks
        run sequentially under ``_background_lock``.
        """
        logger.info(
            "[%s] Dispatching batch %s with %d tasks",
            self._anima_name,
            batch_id,
            len(tasks),
        )

        try:
            order = _topological_sort(tasks)
        except ValueError:
            logger.error(
                "[%s] Cycle detected in batch %s; aborting all tasks",
                self._anima_name,
                batch_id,
            )
            for td in tasks:
                self._write_failed_result(td["task_id"], "cycle_in_batch")
                self._sync_task_queue(td["task_id"], "failed", summary="FAILED: cycle_in_batch")
            return

        completed: dict[str, str] = {}  # task_id -> result_summary
        failed: set[str] = set()
        remaining = list(order)

        while remaining:
            ready = [td for td in remaining if _deps_satisfied(td, completed, failed)]
            if not ready:
                for td in remaining:
                    failed.add(td["task_id"])
                    self._write_failed_result(td["task_id"], "failed_dependency")
                    self._sync_task_queue(td["task_id"], "failed", summary="FAILED: failed_dependency")
                break

            parallel_ready = [td for td in ready if td.get("parallel")]
            serial_ready = [td for td in ready if not td.get("parallel")]

            # Skip parallel tasks whose dependencies have failed (mirror of serial check at 461)
            for td in list(parallel_ready):
                if any(dep in failed for dep in td.get("depends_on", [])):
                    parallel_ready.remove(td)
                    remaining.remove(td)
                    failed.add(td["task_id"])
                    self._write_failed_result(td["task_id"], "failed_dependency")
                    self._sync_task_queue(td["task_id"], "failed", summary="FAILED: failed_dependency")

            # Execute parallel tasks concurrently under semaphore
            if parallel_ready:
                coros = [self._execute_parallel_task(td, completed, batch_id) for td in parallel_ready]
                results = await asyncio.gather(*coros, return_exceptions=True)
                for task, result in zip(parallel_ready, results, strict=False):
                    remaining.remove(task)
                    if isinstance(result, Exception):
                        logger.error(
                            "[%s] Parallel task %s failed: %s",
                            self._anima_name,
                            task["task_id"],
                            result,
                        )
                        failed.add(task["task_id"])
                        self._write_failed_result(task["task_id"], str(result))
                        self._sync_task_queue(
                            task["task_id"],
                            "failed",
                            summary=f"FAILED: {str(result)[:200]}",
                        )
                        reply_to = task.get("reply_to")
                        if isinstance(reply_to, dict):
                            reply_to = reply_to.get("name")
                        elif not isinstance(reply_to, str):
                            reply_to = None
                        if reply_to:
                            try:
                                from core.execution._sanitize import ORIGIN_ANIMA
                                from core.i18n import t

                                notify_text = t(
                                    "pending_executor.task_fail_notify",
                                    task_id=task["task_id"],
                                    title=task.get("description", "unknown"),
                                    error=f"Batch execution failed: {type(result).__name__}: {str(result)[:200]}",
                                )
                                for _attempt in range(2):
                                    try:
                                        self._anima.messenger.send(
                                            to=reply_to,
                                            content=notify_text,
                                            origin_chain=[ORIGIN_ANIMA],
                                        )
                                        break
                                    except Exception:
                                        if _attempt > 0:
                                            logger.error(
                                                "[%s] Batch failure notification failed after retry to %s",
                                                self._anima_name,
                                                reply_to,
                                                exc_info=True,
                                            )
                            except Exception:
                                logger.warning("[%s] Failed to build batch failure notification", self._anima_name)
                    else:
                        completed[task["task_id"]] = result or ""

            # Execute serial tasks sequentially
            for task in serial_ready:
                remaining.remove(task)
                if any(dep in failed for dep in task.get("depends_on", [])):
                    failed.add(task["task_id"])
                    self._write_failed_result(task["task_id"], "failed_dependency")
                    continue
                try:
                    result = await self._execute_serial_batch_task(
                        task,
                        completed,
                        batch_id,
                    )
                    completed[task["task_id"]] = result or ""
                except Exception as exc:
                    logger.error(
                        "[%s] Serial batch task %s failed: %s",
                        self._anima_name,
                        task["task_id"],
                        exc,
                    )
                    failed.add(task["task_id"])
                    self._write_failed_result(task["task_id"], str(exc))
                    self._sync_task_queue(
                        task["task_id"],
                        "failed",
                        summary=f"FAILED: {str(exc)[:200]}",
                    )
                    reply_to = task.get("reply_to")
                    if isinstance(reply_to, dict):
                        reply_to = reply_to.get("name")
                    elif not isinstance(reply_to, str):
                        reply_to = None
                    if reply_to:
                        try:
                            from core.execution._sanitize import ORIGIN_ANIMA
                            from core.i18n import t

                            notify_text = t(
                                "pending_executor.task_fail_notify",
                                task_id=task["task_id"],
                                title=task.get("description", "unknown"),
                                error=f"Batch execution failed: {type(exc).__name__}: {str(exc)[:200]}",
                            )
                            for _attempt in range(2):
                                try:
                                    self._anima.messenger.send(
                                        to=reply_to,
                                        content=notify_text,
                                        origin_chain=[ORIGIN_ANIMA],
                                    )
                                    break
                                except Exception:
                                    if _attempt > 0:
                                        logger.error(
                                            "[%s] Batch failure notification failed after retry to %s",
                                            self._anima_name,
                                            reply_to,
                                            exc_info=True,
                                        )
                        except Exception:
                            logger.warning("[%s] Failed to build batch failure notification", self._anima_name)

        logger.info(
            "[%s] Batch %s complete: %d succeeded, %d failed",
            self._anima_name,
            batch_id,
            len(completed),
            len(failed),
        )

    async def _execute_parallel_task(
        self,
        task_desc: dict[str, Any],
        completed_results: dict[str, str],
        batch_id: str,
    ) -> str:
        """Execute a single parallel task under the semaphore (no _background_lock)."""
        task_id = task_desc.get("task_id", "unknown")
        title = task_desc.get("title", "Untitled")

        async with self._get_semaphore():
            # Register in active parallel tasks
            self._anima._active_parallel_tasks[task_id] = {
                "title": title,
                "description": (task_desc.get("description", ""))[:100],
                "started_at": datetime.now(UTC).isoformat(),
                "batch_id": batch_id,
                "status": "running",
                "depends_on": task_desc.get("depends_on", []),
            }
            try:
                result = await self._run_llm_task(task_desc, completed_results)
                self._save_task_result(task_id, result)
                self._sync_task_queue(task_id, "done", summary=(result or "")[:200])
                return result
            finally:
                self._anima._active_parallel_tasks.pop(task_id, None)

    async def _execute_serial_batch_task(
        self,
        task_desc: dict[str, Any],
        completed_results: dict[str, str],
        batch_id: str,
    ) -> str:
        """Execute a serial batch task under _background_lock."""
        task_id = task_desc.get("task_id", "unknown")
        result = await self._run_llm_task(task_desc, completed_results)
        self._save_task_result(task_id, result)
        self._sync_task_queue(task_id, "done", summary=(result or "")[:200])
        return result

    async def _run_llm_task(
        self,
        task_desc: dict[str, Any],
        completed_results: dict[str, str] | None = None,
    ) -> str:
        """Core LLM task execution logic shared by parallel and serial paths.

        Returns the result summary string.
        """
        if self._anima and hasattr(self._anima, "_get_interrupt_event"):
            self._anima._get_interrupt_event("_background").clear()
            self._anima.agent.set_interrupt_event(
                self._anima._get_interrupt_event("_background"),
            )

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

        # Skip if task was cancelled in task_queue (batch path; single path checks in watcher)
        try:
            from core.memory.task_queue import TaskQueueManager

            entry = TaskQueueManager(self._anima_dir).get_task_by_id(task_id)
            if entry and entry.status == "cancelled":
                logger.info(
                    "[%s] Skipping cancelled LLM task: id=%s",
                    self._anima_name,
                    task_id,
                )
                return "(cancelled)"
        except Exception:
            logger.debug(
                "Could not check task_queue for cancellation: %s",
                task_id,
                exc_info=True,
            )

        # TTL check
        if submitted_at:
            try:
                sub_dt = datetime.fromisoformat(submitted_at)
                if sub_dt.tzinfo is None:
                    sub_dt = sub_dt.replace(tzinfo=UTC)
                now_utc = datetime.now(UTC)
                age_hours = (now_utc - sub_dt).total_seconds() / 3600
                if age_hours > _LLM_TASK_TTL_HOURS:
                    logger.warning(
                        "[%s] Skipping expired LLM task: %s (age=%.1fh, TTL=%dh)",
                        self._anima_name,
                        task_id,
                        age_hours,
                        _LLM_TASK_TTL_HOURS,
                    )
                    return "(expired)"
            except (ValueError, TypeError):
                pass

        # Build dependency context for batch tasks
        dep_context = ""
        if completed_results:
            dep_context = self._build_dependency_context(task_desc, completed_results)

        from core.memory.activity import ActivityLogger
        from core.memory.streaming_journal import StreamingJournal
        from core.paths import load_prompt

        activity = ActivityLogger(self._anima_dir)
        activity.log(
            "task_exec_start",
            summary=t("pending_executor.task_exec_start", title=title),
            meta={"task_id": task_id, "submitted_by": submitted_by},
        )

        _none = t("pending_executor.none_value")
        criteria_text = "\n".join(f"- {c}" for c in acceptance_criteria) if acceptance_criteria else _none
        constraints_text = "\n".join(f"- {c}" for c in constraints) if constraints else _none
        paths_text = "\n".join(f"- {p}" for p in file_paths) if file_paths else _none

        full_context = context or _none
        if dep_context:
            full_context = f"{full_context}\n\n{dep_context}"

        working_directory = task_desc.get("working_directory", "")
        if not working_directory:
            working_directory = _resolve_default_workspace(self._anima_dir)
        prompt = load_prompt(
            "task_exec",
            task_id=task_id,
            title=title,
            submitted_by=submitted_by,
            workspace=working_directory or t("pending_executor.workspace_not_specified"),
            description=description,
            context=full_context,
            acceptance_criteria=criteria_text,
            constraints=constraints_text,
            file_paths=paths_text,
        )

        if working_directory:
            self._anima.agent.set_task_cwd(Path(working_directory))

        if "machine" in description.lower():
            prompt += "\n\n" + t("pending_executor.machine_directive")

        trigger = f"task:{task_id}"
        journal = StreamingJournal(self._anima_dir, session_type="task_exec")
        journal.open(trigger=trigger)

        self._anima.agent.reset_reply_tracking(session_type="background")
        self._anima.agent.reset_read_paths()
        accumulated_text = ""
        result_summary = ""

        try:
            async for chunk in self._anima.agent.run_cycle_streaming(
                prompt,
                trigger=trigger,
            ):
                if chunk.get("type") == "text_delta":
                    accumulated_text += chunk.get("text", "")
                    journal.write_text(chunk.get("text", ""))
                if chunk.get("type") == "cycle_done":
                    cycle_result = chunk.get("cycle_result", {})
                    result_summary = cycle_result.get(
                        "summary",
                        accumulated_text[:500],
                    )
                    journal.finalize(summary=result_summary[:500])
        finally:
            journal.close()
            self._anima.agent.set_task_cwd(None)

        if not result_summary:
            result_summary = accumulated_text[:500] or t("pending_executor.task_completed")

        activity.log(
            "task_exec_end",
            summary=t("pending_executor.task_exec_end", title=title, result=result_summary[:200]),
            meta={"task_id": task_id},
        )

        # Send completion notification
        if reply_to:
            if isinstance(reply_to, dict):
                reply_to = reply_to.get("name")
            elif not isinstance(reply_to, str):
                reply_to = None
        if reply_to:
            try:
                notify_text = load_prompt(
                    "task_complete_notify",
                    task_id=task_id,
                    title=title,
                    result_summary=result_summary[:1000],
                )
                from core.execution._sanitize import ORIGIN_ANIMA

                for _attempt in range(2):
                    try:
                        self._anima.messenger.send(
                            to=reply_to,
                            content=notify_text,
                            origin_chain=[ORIGIN_ANIMA],
                        )
                        break
                    except Exception:
                        if _attempt == 0:
                            logger.warning(
                                "[%s] Task completion notification failed, retrying",
                                self._anima_name,
                            )
                        else:
                            logger.error(
                                "[%s] Task completion notification failed after retry to %s",
                                self._anima_name,
                                reply_to,
                                exc_info=True,
                            )
                            if hasattr(self._anima, "_activity"):
                                self._anima._activity.log(
                                    "error",
                                    content=f"Task completion notification failed: {task_id} → {reply_to}",
                                )
            except Exception:
                logger.warning(
                    "[%s] Failed to build task completion notification",
                    self._anima_name,
                    exc_info=True,
                )

        logger.info("[%s] LLM task completed: id=%s", self._anima_name, task_id)
        return result_summary

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
            "Submitting pending task to BackgroundTaskManager: id=%s tool=%s subcmd=%s",
            task_id,
            tool_name,
            subcommand,
        )

        def _dispatch_fn(name: str, args: dict[str, Any]) -> str:
            """Execute the tool via CLI subprocess (same as direct execution)."""
            import os
            import subprocess

            # name may be composite (e.g. "transcribe:audio"); extract module name
            module_name = name.split(":")[0] if ":" in name else name
            cmd = ["animaworks-tool", module_name]
            subcmd = args.get("subcommand", "")
            if subcmd:
                cmd.append(subcmd)
            cmd.extend(args.get("raw_args", []))
            # Remove subcommand from raw_args if it's already the first element
            if subcmd and args.get("raw_args") and args["raw_args"][0] == subcmd:
                cmd = ["animaworks-tool", module_name] + args["raw_args"]
            cmd.append("-j")

            env = {
                **os.environ,
                "ANIMAWORKS_ANIMA_DIR": args.get("anima_dir", ""),
            }

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_PENDING_TASK_SUBPROCESS_TIMEOUT,
                env=env,
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() or f"Exit code {result.returncode}"
                raise ToolExecutionError(f"Tool {name} failed: {error_msg}")
            return result.stdout.strip()

        # Submit to BackgroundTaskManager
        composite_name = f"{tool_name}:{subcommand}" if subcommand else tool_name
        bg_mgr.submit(composite_name, tool_args, _dispatch_fn)

    async def _execute_llm_task(self, task_desc: dict[str, Any]) -> None:
        """Execute an LLM task under _background_lock.

        The task is executed as a minimal-context LLM session using
        the task_exec.md template.  Delegates to ``_run_llm_task``
        for the actual execution logic.
        """
        task_id = task_desc.get("task_id", "unknown")

        logger.info(
            "[%s] Executing LLM task: id=%s title=%s",
            self._anima_name,
            task_id,
            task_desc.get("title", ""),
        )

        try:
            async with self._anima._background_lock:
                self._anima._mark_busy_start()
                self._anima._status_slots["background"] = "task_exec"
                self._anima._task_slots["background"] = task_id
                try:
                    result = await self._run_llm_task(task_desc)
                    self._sync_task_queue(task_id, "done", summary=(result or "")[:200])
                finally:
                    self._anima._status_slots["background"] = "idle"
                    self._anima._task_slots["background"] = ""
        except Exception as exc:
            logger.exception(
                "[%s] LLM task failed: id=%s",
                self._anima_name,
                task_id,
            )
            self._anima._status_slots["background"] = "idle"
            self._anima._task_slots["background"] = ""
            self._write_failed_result(
                task_id,
                f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            self._sync_task_queue(
                task_id,
                "failed",
                summary=f"FAILED: {type(exc).__name__}: {str(exc)[:200]}",
            )
            reply_to = task_desc.get("reply_to")
            if isinstance(reply_to, dict):
                reply_to = reply_to.get("name")
            elif not isinstance(reply_to, str):
                reply_to = None
            if reply_to:
                try:
                    from core.execution._sanitize import ORIGIN_ANIMA
                    from core.i18n import t

                    notify_text = t(
                        "pending_executor.task_fail_notify",
                        task_id=task_id,
                        title=task_desc.get("description", "unknown"),
                        error=f"{type(exc).__name__}: {str(exc)[:200]}",
                    )
                    for _attempt in range(2):
                        try:
                            self._anima.messenger.send(
                                to=reply_to,
                                content=notify_text,
                                origin_chain=[ORIGIN_ANIMA],
                            )
                            break
                        except Exception:
                            if _attempt == 0:
                                logger.warning(
                                    "[%s] Task failure notification failed, retrying to %s",
                                    self._anima_name,
                                    reply_to,
                                )
                            else:
                                logger.error(
                                    "[%s] Task failure notification failed after retry to %s",
                                    self._anima_name,
                                    reply_to,
                                    exc_info=True,
                                )
                                if hasattr(self._anima, "_activity"):
                                    self._anima._activity.log(
                                        "error",
                                        content=f"Task failure notification failed: {task_id} → {reply_to}",
                                    )
                except Exception:
                    logger.warning(
                        "[%s] Failed to build task failure notification for %s",
                        self._anima_name,
                        reply_to,
                        exc_info=True,
                    )
