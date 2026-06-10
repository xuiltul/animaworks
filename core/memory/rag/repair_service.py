from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""RAG corruption repair service.

The persistent ChromaDB directory is derived data.  When Chroma reports
internal consistency errors, the safest recovery is to quarantine the
broken vectordb and rebuild it from source memory files.
"""

import logging
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.memory.rag import repair_state
from core.memory.rag.repair_rebuild import full_reindex, quarantine_vectordb
from core.memory.rag.repair_types import RepairResult
from core.memory.rag.repair_utils import (
    SINGLE_SHOT_REASONS,
    classify_corruption_error,
    collection_owner,
    get_repair_lock_path,
    is_repair_locked,
    iso,
    parse_dt,
    utc_now,
)
from core.platform.locks import acquire_file_lock, release_file_lock

logger = logging.getLogger("animaworks.rag.repair")

_LOG_TAIL_BYTES = 5_000_000
_COLLECTION_PATTERNS = (
    re.compile(r"[Cc]ollection ['\"]([^'\"]+)['\"]"),
    re.compile(r"\bcollection=([A-Za-z0-9_.:-]+)"),
)
_ANIMA_PATTERNS = (
    re.compile(r"\banima(?:_name)?=([A-Za-z0-9_.:-]+)"),
    re.compile(r"\banima(?:_name)? ['\"]([^'\"]+)['\"]"),
)
_TIMESTAMP_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)")
_KNOWN_CORRUPTION_REASONS = {
    "chroma_error_finding_id",
    "sqlite_malformed",
    "chroma_corruption",
    "hnsw_corruption",
    "native_segfault",
    "startup_chroma_crash_preflight",
    "startup_unclean_exit_preflight",
    "vector_worker_crash",
}


class RAGRepairService:
    """Detects and repairs corrupt per-anima RAG vector stores."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        threshold: int | None = None,
        window_minutes: int | None = None,
        cooldown_minutes: int | None = None,
        max_consecutive_failures: int | None = None,
    ) -> None:
        cfg = self._load_repair_config()
        self.enabled = cfg["enabled"] if enabled is None else enabled
        self.threshold = max(1, threshold if threshold is not None else cfg["threshold"])
        self.window = timedelta(minutes=window_minutes if window_minutes is not None else cfg["window_minutes"])
        self.cooldown = timedelta(minutes=cooldown_minutes if cooldown_minutes is not None else cfg["cooldown_minutes"])
        self.max_consecutive_failures = max(
            1,
            max_consecutive_failures if max_consecutive_failures is not None else cfg["max_consecutive_failures"],
        )
        self._signals: dict[str, list[dict[str, Any]]] = {}
        self._active_repairs: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def _load_repair_config() -> dict[str, Any]:
        defaults = {
            "enabled": True,
            "threshold": 2,
            "window_minutes": 5,
            "cooldown_minutes": 60,
            "max_consecutive_failures": 2,
        }
        try:
            from core.config import load_config

            rag = load_config().rag
            defaults["enabled"] = bool(getattr(rag, "repair_enabled", defaults["enabled"]))
            defaults["threshold"] = int(getattr(rag, "repair_error_threshold", defaults["threshold"]))
            defaults["window_minutes"] = int(getattr(rag, "repair_window_minutes", defaults["window_minutes"]))
            defaults["cooldown_minutes"] = int(getattr(rag, "repair_cooldown_minutes", defaults["cooldown_minutes"]))
            defaults["max_consecutive_failures"] = int(
                getattr(rag, "repair_max_consecutive_failures", defaults["max_consecutive_failures"])
            )
        except Exception:
            logger.debug("Using default RAG repair config", exc_info=True)
        return defaults

    def record_chroma_error(
        self,
        *,
        anima_name: str | None,
        collection: str,
        error: BaseException | str | int,
        source: str,
    ) -> bool:
        """Record a Chroma error and start repair if thresholds are met."""
        reason = classify_corruption_error(error)
        if not reason:
            return False

        owner, is_shared = collection_owner(collection, anima_name)
        if owner is None:
            logger.warning(
                "RAG corruption signal could not be mapped to an anima: collection=%s reason=%s",
                collection,
                reason,
            )
            return False

        signal = {
            "at": iso(),
            "collection": collection,
            "reason": reason,
            "source": source,
            "shared": is_shared,
        }
        self._record_signal(owner, signal)

        threshold_met = self._threshold_met(owner, collection, reason)
        if reason in SINGLE_SHOT_REASONS:
            threshold_met = True

        if not threshold_met:
            return False

        return self.request_repair(
            owner,
            reason=reason,
            collection=collection,
            source=source,
            include_shared=True,
            background=True,
        )

    def _record_signal(self, anima_name: str, signal: dict[str, Any]) -> None:
        cutoff = utc_now() - self.window
        with self._lock:
            signals = self._signals.setdefault(anima_name, [])
            signals.append(signal)
            self._signals[anima_name] = [s for s in signals[-50:] if (parse_dt(s.get("at")) or utc_now()) >= cutoff]
        repair_state.append_state_signal(anima_name, signal, self.window)

    def _threshold_met(self, anima_name: str, collection: str, reason: str) -> bool:
        cutoff = utc_now() - self.window
        with self._lock:
            signals = list(self._signals.get(anima_name, []))
        count = 0
        for signal in signals:
            at = parse_dt(signal.get("at"))
            if at is None or at < cutoff:
                continue
            if signal.get("collection") == collection and signal.get("reason") == reason:
                count += 1
        return count >= self.threshold

    def has_recent_corruption(self, anima_name: str, *, include_shared: bool = True) -> bool:
        """Return True when recent signals exist for supervisor correlation."""
        cutoff = utc_now() - self.window
        with self._lock:
            in_memory = list(self._signals.get(anima_name, []))
        if self._contains_recent_signal(in_memory, cutoff, include_shared=include_shared):
            return True
        state = repair_state.read_state(anima_name)
        return self._contains_recent_signal(state.get("recent_signals", []), cutoff, include_shared=include_shared)

    def list_repairable_animas(self, *, animas_dir: Path | None = None) -> list[str]:
        """Return enabled anima names that can safely be considered for repair."""
        if animas_dir is None:
            from core.paths import get_animas_dir

            animas_dir = get_animas_dir()
        if not animas_dir.is_dir():
            return []
        return [
            anima_dir.name
            for anima_dir in sorted(animas_dir.iterdir())
            if anima_dir.is_dir() and (anima_dir / "identity.md").is_file() and self._anima_enabled(anima_dir)
        ]

    def discover_suspect_animas(
        self,
        *,
        window_minutes: int | None = None,
        include_logs: bool = True,
        include_quick_check: bool = True,
        quick_check_timeout_seconds: float = 10.0,
        quick_check_source: str = "startup_quick_check",
        animas_dir: Path | None = None,
        log_paths: list[Path] | None = None,
    ) -> list[str]:
        """Discover animas whose RAG DBs have recent corruption evidence.

        Evidence comes from repair state files and recent server logs.  Native
        ChromaDB segfaults often bypass Python exception handling, so log
        discovery intentionally treats collection-less native crash lines as
        evidence for every existing vectordb.
        """
        if animas_dir is None:
            from core.paths import get_animas_dir

            animas_dir = get_animas_dir()
        cutoff = utc_now() - (timedelta(minutes=window_minutes) if window_minutes is not None else self.window)
        repairable = self.list_repairable_animas(animas_dir=animas_dir)
        repairable_set = set(repairable)
        suspects: set[str] = set()

        for anima_name in repairable:
            state = repair_state.read_state(anima_name, animas_dir=animas_dir)
            if self._state_is_suspect(state, cutoff):
                suspects.add(anima_name)

        if include_quick_check:
            from core.memory.rag.sqlite_health import check_anima_vectordb_health_via_worker_or_direct

            for anima_name in repairable:
                try:
                    health = check_anima_vectordb_health_via_worker_or_direct(
                        anima_name,
                        timeout_seconds=quick_check_timeout_seconds,
                        source=quick_check_source,
                        record_repair=False,
                    )
                except Exception:
                    logger.debug("RAG quick_check failed while discovering suspect DBs: %s", anima_name, exc_info=True)
                    continue
                if health.corrupt:
                    suspects.add(anima_name)

        if include_logs:
            for line in self._iter_recent_corruption_log_lines(
                cutoff=cutoff,
                log_paths=log_paths,
            ):
                owners = self._owners_from_log_line(line, repairable=repairable)
                if owners:
                    suspects.update(owner for owner in owners if owner in repairable_set)
                elif "chromadb_rust_bindings" in line.lower() or "native_segfault" in line.lower():
                    suspects.update(
                        anima_name for anima_name in repairable if (animas_dir / anima_name / "vectordb").exists()
                    )

        return [name for name in repairable if name in suspects]

    def repair_animas_if_allowed(
        self,
        anima_names: list[str] | tuple[str, ...] | set[str],
        *,
        reason: str,
        source: str,
        include_shared: bool = False,
    ) -> dict[str, RepairResult]:
        """Synchronously repair multiple animas while preserving per-anima guards."""
        results: dict[str, RepairResult] = {}
        for anima_name in sorted(dict.fromkeys(anima_names)):
            results[anima_name] = self.repair_anima_if_allowed(
                anima_name,
                reason=reason,
                collection=None,
                source=source,
                include_shared=include_shared,
            )
        return results

    @staticmethod
    def _contains_recent_signal(signals: list[dict[str, Any]], cutoff: datetime, *, include_shared: bool) -> bool:
        for signal in signals:
            at = parse_dt(signal.get("at"))
            if at is None or at < cutoff:
                continue
            if include_shared or not bool(signal.get("shared")):
                return True
        return False

    @staticmethod
    def _anima_enabled(anima_dir: Path) -> bool:
        status_path = anima_dir / "status.json"
        if not status_path.is_file():
            return True
        try:
            import json

            data = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return True
        return bool(data.get("enabled", True)) if isinstance(data, dict) else True

    @staticmethod
    def _state_is_suspect(state: dict[str, Any], cutoff: datetime) -> bool:
        if not state:
            return False
        signals = state.get("recent_signals")
        if isinstance(signals, list):
            for signal in signals:
                at = parse_dt(signal.get("at"))
                if at is not None and at >= cutoff:
                    return True

        reason = state.get("reason") or state.get("last_reason")
        if reason not in _KNOWN_CORRUPTION_REASONS:
            return False
        status = state.get("status")
        if status not in {"requested", "stopping", "repairing", "failed", "cooldown", "locked"}:
            return False
        for key in ("updated_at", "last_failure_at", "last_attempt_at", "requested_at"):
            at = parse_dt(state.get(key))
            if at is not None and at >= cutoff:
                return True
        return False

    @staticmethod
    def _default_log_paths() -> list[Path]:
        from core.paths import get_data_dir

        log_dir = get_data_dir() / "logs"
        return [
            log_dir / "server-daemon.log",
            log_dir / "server-daemon.log.1",
            log_dir / "restart-helper.log",
        ]

    def _iter_recent_corruption_log_lines(
        self,
        *,
        cutoff: datetime,
        log_paths: list[Path] | None,
    ) -> list[str]:
        paths = log_paths if log_paths is not None else self._default_log_paths()
        lines: list[str] = []
        for path in paths:
            if not path.is_file():
                continue
            try:
                with path.open("rb") as fh:
                    try:
                        fh.seek(-_LOG_TAIL_BYTES, os.SEEK_END)
                    except OSError:
                        fh.seek(0)
                    text = fh.read().decode("utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                if not self._log_line_is_recent(line, cutoff):
                    continue
                reason = classify_corruption_error(line)
                if reason or "chromadb_rust_bindings" in line.lower():
                    lines.append(line)
        return lines

    @staticmethod
    def _log_line_is_recent(line: str, cutoff: datetime) -> bool:
        match = _TIMESTAMP_RE.search(line)
        if not match:
            return True
        at = parse_dt(match.group("ts").replace("Z", "+00:00"))
        return at is None or at >= cutoff

    @staticmethod
    def _owners_from_log_line(line: str, *, repairable: list[str]) -> set[str]:
        owners: set[str] = set()
        for pattern in _ANIMA_PATTERNS:
            for match in pattern.finditer(line):
                owners.add(match.group(1))
        for pattern in _COLLECTION_PATTERNS:
            for match in pattern.finditer(line):
                owner, is_shared = collection_owner(match.group(1))
                if owner:
                    owners.add(owner)
                elif is_shared:
                    owners.update(repairable)
        return owners

    def request_repair(
        self,
        anima_name: str,
        *,
        reason: str,
        collection: str | None = None,
        source: str,
        include_shared: bool = False,
        background: bool = True,
    ) -> bool:
        """Request repair for an anima.

        Background requests only mark the anima for supervisor-managed repair.
        Synchronous calls are reserved for CLI/supervisor repair execution.
        """
        if background:
            blocked = self._request_blocked(anima_name, reason=reason)
            if blocked is not None:
                repair_state.write_blocked_state(anima_name, blocked)
                return False
            if self._has_active_repair_state(anima_name):
                logger.info("RAG supervised repair already requested or active: %s", anima_name)
                return False
            repair_state.write_repair_request_state(
                anima_name,
                reason=reason,
                collection=collection,
                source=source,
                include_shared=include_shared,
            )
            logger.warning(
                "RAG supervised repair requested: anima=%s reason=%s collection=%s source=%s",
                anima_name,
                reason,
                collection,
                source,
            )
            return True

        blocked = self._reserve_repair(anima_name, reason=reason)
        if blocked is not None:
            return False

        try:
            self.repair_anima(
                anima_name,
                reason=reason,
                collection=collection,
                source=source,
                include_shared=include_shared,
            )
            return True
        finally:
            with self._lock:
                self._active_repairs.discard(anima_name)

    def repair_anima_if_allowed(
        self,
        anima_name: str,
        *,
        reason: str,
        collection: str | None = None,
        source: str,
        include_shared: bool = False,
    ) -> RepairResult:
        """Synchronously repair an anima while respecting loop-prevention guards."""
        blocked = self._reserve_repair(anima_name, reason=reason)
        if blocked is not None:
            return blocked
        try:
            return self.repair_anima(
                anima_name,
                reason=reason,
                collection=collection,
                source=source,
                include_shared=include_shared,
            )
        finally:
            with self._lock:
                self._active_repairs.discard(anima_name)

    def _reserve_repair(self, anima_name: str, *, reason: str) -> RepairResult | None:
        blocked = self._request_blocked(anima_name, reason=reason)
        if blocked is not None:
            return blocked
        with self._lock:
            if anima_name in self._active_repairs:
                logger.info("RAG repair already active: %s", anima_name)
                return RepairResult(status="active", anima_name=anima_name, reason=reason)
            self._active_repairs.add(anima_name)
        return None

    def _request_blocked(self, anima_name: str, *, reason: str) -> RepairResult | None:
        if not self.enabled:
            logger.info("RAG repair disabled; ignoring repair request for %s", anima_name)
            return RepairResult(status="disabled", anima_name=anima_name, reason=reason)
        if self._cooling_down(anima_name):
            logger.warning("RAG repair request skipped during cooldown: %s reason=%s", anima_name, reason)
            return RepairResult(status="cooldown", anima_name=anima_name, reason=reason)
        if is_repair_locked(anima_name):
            logger.warning("RAG repair request skipped because lock is held: %s", anima_name)
            return RepairResult(status="locked", anima_name=anima_name, reason=reason)
        return None

    def _has_active_repair_state(self, anima_name: str) -> bool:
        state = repair_state.read_state(anima_name)
        return state.get("status") in {
            "requested",
            "stopping",
            "repairing",
        }

    def repair_anima(
        self,
        anima_name: str,
        *,
        reason: str,
        collection: str | None = None,
        source: str,
        include_shared: bool = False,
    ) -> RepairResult:
        """Synchronously quarantine and rebuild one anima's RAG index."""
        from core.paths import get_animas_dir

        anima_dir = get_animas_dir() / anima_name
        if not anima_dir.is_dir():
            return RepairResult(
                status="failed",
                anima_name=anima_name,
                reason=reason,
                error="anima not found",
                stage="validate_anima",
                state_path=str(repair_state.state_path(anima_name)),
            )

        lock_path = get_repair_lock_path(anima_name)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                acquire_file_lock(lock_file, exclusive=True, blocking=False)
            except OSError:
                result = RepairResult(
                    status="locked",
                    anima_name=anima_name,
                    reason=reason,
                    stage="locked",
                    state_path=str(repair_state.state_path(anima_name)),
                )
                repair_state.write_blocked_state(anima_name, result)
                return result

            quarantine_path: Path | None = None
            try:
                repair_state.update_repair_state(
                    anima_name,
                    status="repairing",
                    stage="quarantine",
                    pid=os.getpid(),
                    started_at=iso(),
                    last_attempt_at=iso(),
                    reason=reason,
                    collection=collection,
                    source=source,
                    include_shared=include_shared,
                    last_reason=reason,
                    last_collection=collection,
                    last_source=source,
                    last_error=None,
                )
                quarantine_path = quarantine_vectordb(anima_name)
                repair_state.update_repair_state(
                    anima_name,
                    status="repairing",
                    stage="reindex",
                    pid=os.getpid(),
                    last_quarantine_path=str(quarantine_path) if quarantine_path else None,
                )
                chunks = full_reindex(anima_name, include_shared=include_shared)
                from core.memory.rag.singleton import reset_vector_store

                repair_state.update_repair_state(
                    anima_name,
                    status="repairing",
                    stage="reset_store",
                    pid=os.getpid(),
                    last_chunks_indexed=chunks,
                )
                reset_vector_store(anima_name)
                repair_state.update_repair_state(
                    anima_name,
                    status="success",
                    stage="complete",
                    pid=None,
                    last_success_at=iso(),
                    last_error=None,
                    consecutive_failures=0,
                    last_quarantine_path=str(quarantine_path) if quarantine_path else None,
                    last_chunks_indexed=chunks,
                )
                logger.warning(
                    "RAG repair succeeded: anima=%s reason=%s chunks=%d quarantine=%s",
                    anima_name,
                    reason,
                    chunks,
                    quarantine_path,
                )
                return RepairResult(
                    status="success",
                    anima_name=anima_name,
                    reason=reason,
                    quarantine_path=str(quarantine_path) if quarantine_path else None,
                    chunks_indexed=chunks,
                    stage="complete",
                    state_path=str(repair_state.state_path(anima_name)),
                )
            except Exception as exc:
                from core.memory.rag.singleton import reset_vector_store

                reset_vector_store(anima_name)
                state = repair_state.read_state(anima_name)
                failures = int(state.get("consecutive_failures") or 0) + 1
                repair_state.update_repair_state(
                    anima_name,
                    status="failed",
                    stage="failed",
                    pid=None,
                    last_failure_at=iso(),
                    last_error=str(exc),
                    consecutive_failures=failures,
                    last_quarantine_path=str(quarantine_path) if quarantine_path else None,
                )
                logger.exception("RAG repair failed: anima=%s reason=%s", anima_name, reason)
                return RepairResult(
                    status="failed",
                    anima_name=anima_name,
                    reason=reason,
                    quarantine_path=str(quarantine_path) if quarantine_path else None,
                    error=str(exc),
                    stage="failed",
                    state_path=str(repair_state.state_path(anima_name)),
                )
            finally:
                release_file_lock(lock_file)

    def _cooling_down(self, anima_name: str) -> bool:
        state = repair_state.read_state(anima_name)
        failures = int(state.get("consecutive_failures") or 0)
        if failures >= self.max_consecutive_failures:
            last_failure = parse_dt(state.get("last_failure_at"))
            if last_failure and utc_now() - last_failure < self.cooldown:
                return True
        return False


_service: RAGRepairService | None = None
_service_lock = threading.Lock()


def get_repair_service() -> RAGRepairService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = RAGRepairService()
    return _service


def record_chroma_error(
    *,
    anima_name: str | None,
    collection: str,
    error: BaseException | str | int,
    source: str,
) -> bool:
    return get_repair_service().record_chroma_error(
        anima_name=anima_name,
        collection=collection,
        error=error,
        source=source,
    )


def _reset_for_testing() -> None:
    global _service
    with _service_lock:
        _service = None
