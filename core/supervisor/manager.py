"""
Process Supervisor - Manages lifecycle of Person child processes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.supervisor.ipc import IPCResponse
from core.supervisor.process_handle import ProcessHandle, ProcessState

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────

@dataclass
class RestartPolicy:
    """Process restart policy configuration."""
    max_retries: int = 5                   # Maximum restart attempts
    backoff_base_sec: float = 2.0          # Initial backoff delay
    backoff_max_sec: float = 60.0          # Maximum backoff delay
    reset_after_sec: float = 300.0         # Stable runtime to reset counter


@dataclass
class HealthConfig:
    """Health check configuration."""
    ping_interval_sec: float = 10.0        # Ping interval
    ping_timeout_sec: float = 5.0          # Ping timeout
    max_missed_pings: int = 3              # Consecutive misses before hang
    startup_grace_sec: float = 30.0        # Grace period after startup


@dataclass
class ReconciliationConfig:
    """Reconciliation loop configuration."""
    interval_sec: float = 30.0             # Scan interval


# ── Process Supervisor ─────────────────────────────────────────────

class ProcessSupervisor:
    """
    Supervisor for managing Person child processes.

    Responsibilities:
    - Start/stop child processes
    - Health monitoring (ping/pong)
    - Hang detection and recovery (SIGKILL + restart)
    - Auto-restart with exponential backoff
    - Schedule coordination (heartbeat/cron triggers)
    """

    def __init__(
        self,
        persons_dir: Path,
        shared_dir: Path,
        run_dir: Path,
        log_dir: Path | None = None,
        restart_policy: RestartPolicy | None = None,
        health_config: HealthConfig | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
        ws_manager: Any | None = None,
    ):
        self.persons_dir = persons_dir
        self.shared_dir = shared_dir
        self.run_dir = run_dir
        self.log_dir = log_dir
        self.ws_manager = ws_manager

        self.restart_policy = restart_policy or RestartPolicy()
        self.health_config = health_config or HealthConfig()
        self.reconciliation_config = reconciliation_config or ReconciliationConfig()

        self.processes: dict[str, ProcessHandle] = {}
        self._health_check_task: asyncio.Task | None = None
        self._reconciliation_task: asyncio.Task | None = None
        self._shutdown = False
        self.scheduler: AsyncIOScheduler | None = None
        self._scheduler_running: bool = False
        self._restart_counts: dict[str, int] = {}
        self._restarting: set[str] = set()
        self._bootstrapping: set[str] = set()

        # Callbacks for person lifecycle events (set by server/app.py)
        self.on_person_added: Callable[[str], None] | None = None
        self.on_person_removed: Callable[[str], None] | None = None

    def is_scheduler_running(self) -> bool:
        """Return whether the system scheduler is running."""
        return self._scheduler_running

    async def start_all(self, person_names: list[str]) -> None:
        """
        Start all Person processes.

        Args:
            person_names: List of person names to start
        """
        logger.info("Starting %d Person processes", len(person_names))

        # Create socket directory
        socket_dir = self.run_dir / "sockets"
        socket_dir.mkdir(parents=True, exist_ok=True)

        # Start each process
        for person_name in person_names:
            await self.start_person(person_name)

        # Start health check loop
        self._health_check_task = asyncio.create_task(
            self._health_check_loop()
        )

        # Start reconciliation loop
        self._reconciliation_task = asyncio.create_task(
            self._reconciliation_loop()
        )

        # Start system scheduler (daily/weekly consolidation)
        self._start_system_scheduler()

        logger.info("All processes started")

    async def start_person(self, person_name: str) -> None:
        """Start a single Person process.

        After the process is ready, checks if bootstrap is needed and
        launches it as a background task automatically.
        """
        if person_name in self.processes:
            logger.warning("Process already exists: %s", person_name)
            return

        socket_dir = self.run_dir / "sockets"
        socket_dir.mkdir(parents=True, exist_ok=True)
        socket_path = socket_dir / f"{person_name}.sock"

        handle = ProcessHandle(
            person_name=person_name,
            socket_path=socket_path,
            persons_dir=self.persons_dir,
            shared_dir=self.shared_dir,
            log_dir=self.log_dir
        )

        try:
            await handle.start()
            self.processes[person_name] = handle
            logger.info("Person process started: %s (PID %s)", person_name, handle.get_pid())

            # Check if bootstrap is needed and launch in background
            try:
                status = await self.send_request(
                    person_name, "get_status", {}, timeout=10.0,
                )
                if status.get("needs_bootstrap"):
                    logger.info(
                        "Bootstrap needed for %s, launching background task",
                        person_name,
                    )
                    asyncio.create_task(self._run_bootstrap(person_name))
            except Exception as e:
                logger.warning(
                    "Could not check bootstrap status for %s: %s",
                    person_name, e,
                )

        except Exception as e:
            logger.error("Failed to start process %s: %s", person_name, e)
            raise

    def is_bootstrapping(self, person_name: str) -> bool:
        """Check if a person is currently bootstrapping."""
        return person_name in self._bootstrapping

    async def _run_bootstrap(self, person_name: str) -> None:
        """Run bootstrap for a person in the background.

        Sends a ``run_bootstrap`` IPC request with a long timeout (600s)
        and broadcasts progress via WebSocket.
        """
        self._bootstrapping.add(person_name)
        logger.info("Bootstrap started for %s", person_name)

        # Broadcast bootstrap started
        await self._broadcast_event(
            "person.bootstrap",
            {"name": person_name, "status": "started"},
        )

        try:
            handle = self.processes.get(person_name)
            if not handle:
                logger.error("Bootstrap failed: process not found for %s", person_name)
                return

            response = await handle.send_request(
                "run_bootstrap", {}, timeout=600.0,
            )

            if response.error:
                logger.error(
                    "Bootstrap failed for %s: %s",
                    person_name, response.error.get("message", "Unknown error"),
                )
                await self._broadcast_event(
                    "person.bootstrap",
                    {"name": person_name, "status": "failed"},
                )
                return

            result = response.result or {}
            logger.info(
                "Bootstrap completed for %s (duration_ms=%s)",
                person_name, result.get("duration_ms", "?"),
            )
            await self._broadcast_event(
                "person.bootstrap",
                {"name": person_name, "status": "completed"},
            )

        except asyncio.TimeoutError:
            logger.error("Bootstrap timed out for %s (600s)", person_name)
            await self._broadcast_event(
                "person.bootstrap",
                {"name": person_name, "status": "failed"},
            )
        except Exception:
            logger.exception("Bootstrap error for %s", person_name)
            await self._broadcast_event(
                "person.bootstrap",
                {"name": person_name, "status": "failed"},
            )
        finally:
            self._bootstrapping.discard(person_name)

    async def _broadcast_event(
        self, event_type: str, data: dict[str, Any],
    ) -> None:
        """Broadcast a WebSocket event if ws_manager is available."""
        if self.ws_manager:
            await self.ws_manager.broadcast({"type": event_type, "data": data})

    async def stop_person(self, person_name: str) -> None:
        """Stop a single Person process."""
        handle = self.processes.get(person_name)
        if not handle:
            logger.warning("Process not found: %s", person_name)
            return

        await handle.stop(timeout=10.0)
        del self.processes[person_name]
        logger.info("Person process stopped: %s", person_name)

    async def restart_person(self, person_name: str) -> None:
        """Restart a Person process."""
        logger.info("Restarting process: %s", person_name)

        # Stop existing process
        if person_name in self.processes:
            await self.stop_person(person_name)

        # Start new process
        await self.start_person(person_name)

    async def shutdown_all(self) -> None:
        """Shutdown all processes gracefully."""
        logger.info("Shutting down all processes")
        self._shutdown = True

        # Stop system scheduler
        if self.scheduler:
            self.scheduler.shutdown(wait=False)
            self._scheduler_running = False
            logger.info("System scheduler stopped")

        # Stop health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop reconciliation loop
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass

        # Stop all processes
        tasks = [
            self.stop_person(name)
            for name in list(self.processes.keys())
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("All processes shut down")

    async def send_request(
        self,
        person_name: str,
        method: str,
        params: dict[str, Any],
        timeout: float = 60.0
    ) -> dict:
        """
        Send IPC request to a Person process.

        Args:
            person_name: Target person name
            method: Method name
            params: Request parameters
            timeout: Timeout in seconds

        Returns:
            Response result dict

        Raises:
            KeyError: If person not found
            RuntimeError: If process not running
            ValueError: If response contains error
        """
        handle = self.processes.get(person_name)
        if not handle:
            raise KeyError(f"Person not found: {person_name}")

        response = await handle.send_request(method, params, timeout)

        if response.error:
            raise ValueError(
                f"Request failed: {response.error.get('message', 'Unknown error')}"
            )

        return response.result or {}

    async def send_request_stream(
        self,
        person_name: str,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,
    ) -> AsyncIterator[IPCResponse]:
        """
        Send IPC request to a Person process and yield streaming responses.

        Args:
            person_name: Target person name
            method: Method name
            params: Request parameters (should include stream=True)
            timeout: Per-chunk timeout in seconds. Resets on each received
                chunk. If None, resolved from config (default 60s).

        Yields:
            IPCResponse objects (chunks and final result)

        Raises:
            KeyError: If person not found
            RuntimeError: If process not running
        """
        handle = self.processes.get(person_name)
        if not handle:
            raise KeyError(f"Person not found: {person_name}")

        async for response in handle.send_request_stream(
            method, params, timeout
        ):
            if response.error:
                raise ValueError(
                    f"Stream error: {response.error.get('message', 'Unknown error')}"
                )
            yield response

    async def _poll_person_events(self) -> None:
        """Read and broadcast event files from child processes."""
        events_base = self.run_dir / "events"
        if not events_base.exists():
            return

        for person_dir in events_base.iterdir():
            if not person_dir.is_dir():
                continue
            for event_file in sorted(person_dir.glob("*.json")):
                try:
                    data = json.loads(event_file.read_text(encoding="utf-8"))
                    event_type = data.get("event", "")
                    event_data = data.get("data", {})
                    await self._broadcast_event(event_type, event_data)
                    event_file.unlink()
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to process event file %s: %s", event_file, e)
                    # Remove corrupted files
                    try:
                        event_file.unlink()
                    except OSError:
                        pass

    async def _health_check_loop(self) -> None:
        """
        Health check loop.

        Periodically pings all processes and handles failures.
        """
        logger.info("Health check loop started")

        while not self._shutdown:
            try:
                await asyncio.sleep(self.health_config.ping_interval_sec)

                # Poll and broadcast child process events
                await self._poll_person_events()

                # Check all processes in parallel
                checks = [
                    self._check_process_health(person_name, handle)
                    for person_name, handle in list(self.processes.items())
                ]
                await asyncio.gather(*checks, return_exceptions=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check loop: %s", e)

        logger.info("Health check loop stopped")

    async def _check_process_health(
        self,
        person_name: str,
        handle: ProcessHandle
    ) -> None:
        """Check health of a single process."""
        # Skip if currently streaming (IPC lock held, ping would block)
        if handle._streaming:
            logger.debug("Skipping health check for %s (streaming)", person_name)
            return

        # Skip if in startup grace period
        uptime = (datetime.now() - handle.stats.started_at).total_seconds()
        if uptime < self.health_config.startup_grace_sec:
            logger.debug("Skipping health check for %s (startup grace)", person_name)
            return

        # Reset restart counter after stable uptime
        if uptime > self.restart_policy.reset_after_sec:
            if self._restart_counts.get(person_name, 0) > 0:
                self._restart_counts[person_name] = 0
                logger.info(
                    "Restart counter reset for %s (stable for %.0fs)",
                    person_name, uptime
                )

        # Check if process is alive
        if not handle.is_alive():
            logger.error(
                "Process exited unexpectedly: %s (exit_code=%s)",
                person_name, handle.stats.exit_code
            )
            asyncio.create_task(self._handle_process_failure(person_name, handle))
            return

        # Ping process
        success = await handle.ping(timeout=self.health_config.ping_timeout_sec)

        if success:
            # Ping successful
            if handle.stats.missed_pings > 0:
                logger.info("Process recovered: %s", person_name)
            return

        # Ping failed
        logger.warning(
            "Health check failed: %s (missed=%d/%d)",
            person_name, handle.stats.missed_pings, self.health_config.max_missed_pings
        )

        # Check if hang threshold exceeded
        if handle.stats.missed_pings >= self.health_config.max_missed_pings:
            logger.error(
                "Process hang detected: %s (PID %s)",
                person_name, handle.get_pid()
            )
            asyncio.create_task(self._handle_process_hang(person_name, handle))

    async def _handle_process_failure(
        self,
        person_name: str,
        handle: ProcessHandle
    ) -> None:
        """Handle process exit/crash.

        Runs as an independent task so the health-check loop is not blocked
        by backoff sleeps.  A per-person guard prevents duplicate restarts.
        """
        if person_name in self._restarting:
            return
        self._restarting.add(person_name)

        try:
            # Check restart count (supervisor-level, survives handle recreation)
            count = self._restart_counts.get(person_name, 0)
            if count >= self.restart_policy.max_retries:
                logger.error(
                    "Max restart retries exceeded for %s. Manual intervention required.",
                    person_name
                )
                handle.state = ProcessState.FAILED
                return

            # Calculate backoff delay
            backoff = min(
                self.restart_policy.backoff_base_sec * (2 ** count),
                self.restart_policy.backoff_max_sec
            )

            logger.info(
                "Scheduling restart for %s (retry %d/%d, delay=%.1fs)",
                person_name, count + 1, self.restart_policy.max_retries, backoff
            )

            # Wait and restart
            await asyncio.sleep(backoff)

            self._restart_counts[person_name] = count + 1
            await self.restart_person(person_name)

            logger.info(
                "Process restarted: %s (PID %s, retry=%d/%d)",
                person_name, self.processes[person_name].get_pid(),
                count + 1, self.restart_policy.max_retries
            )

        except Exception as e:
            logger.error("Failed to restart %s: %s", person_name, e)
            handle.state = ProcessState.FAILED
        finally:
            self._restarting.discard(person_name)

    async def _handle_process_hang(
        self,
        person_name: str,
        handle: ProcessHandle
    ) -> None:
        """Handle hung process (kill and restart)."""
        logger.warning("Killing hung process: %s", person_name)

        # Kill process
        await handle.kill()

        # Restart
        await self._handle_process_failure(person_name, handle)

    # ── Reconciliation ─────────────────────────────────────────────

    @staticmethod
    def read_person_enabled(person_dir: Path) -> bool:
        """Read the enabled flag from a person's status.json.

        Returns True (enabled) when:
        - status.json does not exist (backward compatibility)
        - status.json exists with ``enabled: true``

        Returns False when status.json exists with ``enabled: false``.
        """
        status_file = person_dir / "status.json"
        if not status_file.exists():
            return True  # Backward compatibility: no file = enabled
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
            return bool(data.get("enabled", True))
        except (json.JSONDecodeError, OSError):
            return True  # Treat unreadable file as enabled

    async def _reconciliation_loop(self) -> None:
        """Periodically reconcile desired state (disk) with actual state (processes)."""
        logger.info("Reconciliation loop started (interval=%.0fs)",
                     self.reconciliation_config.interval_sec)

        while not self._shutdown:
            try:
                await asyncio.sleep(self.reconciliation_config.interval_sec)
                await self._reconcile()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Reconciliation failed")

        logger.info("Reconciliation loop stopped")

    async def _reconcile(self) -> None:
        """Scan persons_dir and sync desired state with actual process state."""
        if not self.persons_dir.exists():
            return

        # Build desired state from disk
        on_disk: dict[str, bool] = {}  # name -> enabled
        for person_dir in sorted(self.persons_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            if not (person_dir / "identity.md").exists():
                continue
            on_disk[person_dir.name] = self.read_person_enabled(person_dir)

        running = set(self.processes.keys())

        # enabled + not running → start
        for name, enabled in on_disk.items():
            if enabled and name not in running:
                logger.info("Reconciliation: starting person %s", name)
                try:
                    await self.start_person(name)
                    if self.on_person_added:
                        self.on_person_added(name)
                except Exception:
                    logger.exception(
                        "Reconciliation: failed to start %s", name
                    )

        # disabled + running → stop
        for name, enabled in on_disk.items():
            if not enabled and name in running:
                logger.info(
                    "Reconciliation: stopping person %s (disabled)", name
                )
                try:
                    await self.stop_person(name)
                    if self.on_person_removed:
                        self.on_person_removed(name)
                except Exception:
                    logger.exception(
                        "Reconciliation: failed to stop %s", name
                    )

        # removed from disk + running → stop
        for name in list(running):
            if name not in on_disk:
                logger.info(
                    "Reconciliation: stopping person %s (removed from disk)",
                    name,
                )
                try:
                    await self.stop_person(name)
                    if self.on_person_removed:
                        self.on_person_removed(name)
                except Exception:
                    logger.exception(
                        "Reconciliation: failed to stop %s", name
                    )

        # Check for missing person assets (fallback generation)
        await self._reconcile_assets()

    async def _reconcile_assets(self) -> None:
        """Check for and generate missing person assets during reconciliation."""
        try:
            from core.asset_reconciler import find_persons_with_missing_assets, reconcile_person_assets

            incomplete = find_persons_with_missing_assets(self.persons_dir)
            if not incomplete:
                return

            logger.info(
                "Asset reconciliation: %d person(s) with missing assets",
                len(incomplete),
            )
            for person_name, _check in incomplete:
                person_dir = self.persons_dir / person_name
                result = await reconcile_person_assets(person_dir)
                if not result.get("skipped"):
                    await self._broadcast_event(
                        "person.assets_updated",
                        {"name": person_name, "source": "reconciliation"},
                    )
        except Exception:
            logger.exception("Asset reconciliation failed")

    # ── System Scheduler ────────────────────────────────────────

    def _start_system_scheduler(self) -> None:
        """Start the system-level scheduler for consolidation crons."""
        try:
            self.scheduler = AsyncIOScheduler(timezone="Asia/Tokyo")
            self._setup_system_crons()
            self.scheduler.start()
            self._scheduler_running = True
            logger.info("System scheduler started")
        except Exception:
            logger.exception("Failed to start system scheduler")
            self.scheduler = None
            self._scheduler_running = False

    def _setup_system_crons(self) -> None:
        """Register system-wide cron jobs for memory consolidation."""
        if not self.scheduler:
            return

        # Load consolidation config
        try:
            from core.config import load_config
            config = load_config()
            consolidation_cfg = getattr(config, "consolidation", None)
        except Exception:
            consolidation_cfg = None

        # Daily consolidation
        daily_enabled = True
        daily_time = "02:00"
        if consolidation_cfg:
            daily_enabled = getattr(consolidation_cfg, "daily_enabled", True)
            daily_time = getattr(consolidation_cfg, "daily_time", "02:00")

        if daily_enabled:
            hour, minute = (int(x) for x in daily_time.split(":"))
            self.scheduler.add_job(
                self._run_daily_consolidation,
                CronTrigger(hour=hour, minute=minute),
                id="system_daily_consolidation",
                name="System: Daily Consolidation",
                replace_existing=True,
            )
            logger.info("System cron: Daily consolidation at %s JST", daily_time)

        # Weekly integration
        weekly_enabled = True
        weekly_time = "sun:03:00"
        if consolidation_cfg:
            weekly_enabled = getattr(consolidation_cfg, "weekly_enabled", True)
            weekly_time = getattr(consolidation_cfg, "weekly_time", "sun:03:00")

        if weekly_enabled:
            parts = weekly_time.split(":")
            day_of_week = parts[0] if len(parts) == 3 else "sun"
            time_parts = parts[-2:]
            hour, minute = int(time_parts[0]), int(time_parts[1])
            self.scheduler.add_job(
                self._run_weekly_integration,
                CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute),
                id="system_weekly_integration",
                name="System: Weekly Integration",
                replace_existing=True,
            )
            logger.info("System cron: Weekly integration on %s at %s:%s JST", day_of_week, time_parts[0], time_parts[1])

    async def _run_daily_consolidation(self) -> None:
        """Run daily consolidation for all persons."""
        logger.info("Starting system-wide daily consolidation")

        try:
            from core.config import load_config
            config = load_config()
            consolidation_cfg = getattr(config, "consolidation", None)
        except Exception:
            consolidation_cfg = None

        model = "anthropic/claude-sonnet-4-20250514"
        min_episodes = 1
        if consolidation_cfg:
            model = getattr(consolidation_cfg, "llm_model", model)
            min_episodes = getattr(consolidation_cfg, "min_episodes_threshold", 1)

        for person_name in list(self.processes.keys()):
            try:
                from core.memory.consolidation import ConsolidationEngine

                person_dir = self.persons_dir / person_name
                engine = ConsolidationEngine(
                    person_dir=person_dir,
                    person_name=person_name,
                )

                result = await engine.daily_consolidate(
                    model=model,
                    min_episodes=min_episodes,
                )

                logger.info("Daily consolidation for %s: %s", person_name, result)

                if not result.get("skipped"):
                    await self._broadcast_event(
                        "system.consolidation",
                        {"person": person_name, "type": "daily", "result": result},
                    )
            except Exception:
                logger.exception("Daily consolidation failed for %s", person_name)

    async def _run_weekly_integration(self) -> None:
        """Run weekly integration for all persons."""
        logger.info("Starting system-wide weekly integration")

        try:
            from core.config import load_config
            config = load_config()
            consolidation_cfg = getattr(config, "consolidation", None)
        except Exception:
            consolidation_cfg = None

        model = "anthropic/claude-sonnet-4-20250514"
        duplicate_threshold = 0.85
        episode_retention_days = 30
        if consolidation_cfg:
            model = getattr(consolidation_cfg, "llm_model", model)
            duplicate_threshold = getattr(consolidation_cfg, "duplicate_threshold", 0.85)
            episode_retention_days = getattr(consolidation_cfg, "episode_retention_days", 30)

        for person_name in list(self.processes.keys()):
            try:
                from core.memory.consolidation import ConsolidationEngine

                person_dir = self.persons_dir / person_name
                engine = ConsolidationEngine(
                    person_dir=person_dir,
                    person_name=person_name,
                )

                result = await engine.weekly_integrate(
                    model=model,
                    duplicate_threshold=duplicate_threshold,
                    episode_retention_days=episode_retention_days,
                )

                logger.info(
                    "Weekly integration for %s: merged=%d compressed=%d",
                    person_name,
                    len(result.get("knowledge_files_merged", [])),
                    result.get("episodes_compressed", 0),
                )

                if not result.get("skipped"):
                    await self._broadcast_event(
                        "system.consolidation",
                        {"person": person_name, "type": "weekly", "result": result},
                    )
            except Exception:
                logger.exception("Weekly integration failed for %s", person_name)

    # ── Status ───────────────────────────────────────────────────

    def get_process_status(self, person_name: str) -> dict:
        """
        Get status of a Person process.

        Returns:
            Status dict with state, PID, uptime, etc.
        """
        handle = self.processes.get(person_name)
        if not handle:
            return {"status": "not_found"}

        uptime = (datetime.now() - handle.stats.started_at).total_seconds()

        return {
            "status": "bootstrapping" if self.is_bootstrapping(person_name) else handle.state.value,
            "pid": handle.get_pid(),
            "uptime_sec": uptime,
            "restart_count": self._restart_counts.get(person_name, 0),
            "missed_pings": handle.stats.missed_pings,
            "bootstrapping": self.is_bootstrapping(person_name),
            "last_ping_at": (
                handle.stats.last_ping_at.isoformat()
                if handle.stats.last_ping_at else None
            ),
        }

    def get_all_status(self) -> dict[str, dict]:
        """Get status of all processes."""
        return {
            name: self.get_process_status(name)
            for name in self.processes
        }
