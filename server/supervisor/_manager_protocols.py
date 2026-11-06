"""Structural host protocols for the compositional mixins."""

from __future__ import annotations

from typing import Any, Protocol


class _HealthMixinHost(Protocol):
    """Structural host members used by HealthMixin."""

    _broadcast_event: Any
    _busy_sidecar_path: Any
    _check_process_health: Any
    _ensure_restart_worker: Any
    _handle_process_failure: Any
    _handle_process_hang: Any
    _health_warmup_reason: Any
    _mark_process_error: Any
    _poll_requested_rag_repairs: Any
    _restart_ctl: Any
    _restart_worker: Any
    _restart_worker_tasks: Any
    _restarting: Any
    _shutdown: Any
    animas_dir: Any
    health_config: Any
    processes: Any
    read_anima_enabled: Any
    start_anima: Any
    stop_anima: Any


class _ReconcileMixinHost(Protocol):
    """Structural host members used by ReconcileMixin."""

    _bootstrapping: Any
    _broadcast_event: Any
    _check_config_freshness: Any
    _scan_desired_state: Any
    _ensure_restart_worker: Any
    _flush_task_notices: Any
    _last_config_hash: Any
    _recently_stopped: Any
    _reconcile: Any
    _restart_ctl: Any
    _restarting: Any
    _shutdown: Any
    _starting: Any
    animas_dir: Any
    on_anima_added: Any
    on_anima_removed: Any
    processes: Any
    read_anima_enabled: Any
    reconciliation_config: Any
    restart_anima: Any
    start_anima: Any
    stop_anima: Any


class _SchedulerMixinHost(Protocol):
    """Structural host members used by SchedulerMixin."""

    _CATCHUP_DELAY_SEC: Any
    _apply_activity_schedule_tick: Any
    _broadcast_event: Any
    _catchup_missed_jobs: Any
    _consolidating: Any
    _get_data_dir: Any
    _get_housekeeping_lock: Any
    _housekeeping_lock_obj: Any
    _iter_consolidation_targets: Any
    _resolve_consolidation_ipc_timeout: Any
    _run_activity_log_rotation: Any
    _run_daily_consolidation: Any
    _run_daily_indexing: Any
    _run_housekeeping: Any
    _run_housekeeping_impl: Any
    _run_project_archive_consolidations: Any
    _run_weekly_integration: Any
    _scheduler_running: Any
    _setup_system_crons: Any
    animas_dir: Any
    processes: Any
    read_anima_enabled: Any
    scheduler: Any
