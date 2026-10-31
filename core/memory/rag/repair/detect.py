from __future__ import annotations

from core.platform.env import get_env

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
import secrets
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.anima.workspace_files import IDENTITY_FILE
from core.platform.locks import acquire_file_lock, release_file_lock

from . import state as repair_state
from .state import iso, parse_dt, utc_now
from .types import RepairResult

_RAG_REPAIR_NONCE_ENV = "ANIMAWORKS_RAG_REPAIR_NONCE"
KNOWN_MEMORY_SUFFIXES = (
    "_knowledge",
    "_episodes",
    "_procedures",
    "_skills",
    "_conversation_summary",
    "_entities",
)
SINGLE_SHOT_REASONS = {
    "sqlite_malformed",
    "chroma_corruption",
    "hnsw_corruption",
    "native_segfault",
    "store_init_failed",
}
RESOURCE_EXHAUSTION_SIGNATURES = (
    "too many open files",
    "os error 24",
    "errno 24",
    "emfile",
    "enfile",
    "unable to open database file",
)


@contextmanager
def rag_repair_nonce_env() -> Iterator[str]:
    """Ensure ``ANIMAWORKS_RAG_REPAIR_NONCE`` is set for the duration of a repair.

    Preserves a pre-existing value; generates a fresh token when unset and
    restores the previous environment on exit.
    """
    previous = get_env(_RAG_REPAIR_NONCE_ENV)
    if previous is None:
        os.environ[_RAG_REPAIR_NONCE_ENV] = secrets.token_urlsafe(32)
    try:
        yield os.environ[_RAG_REPAIR_NONCE_ENV]
    finally:
        if previous is None:
            os.environ.pop(_RAG_REPAIR_NONCE_ENV, None)
        else:
            os.environ[_RAG_REPAIR_NONCE_ENV] = previous


def classify_corruption_error(error: BaseException | str | int | None) -> str | None:
    """Return a stable corruption reason for known RAG storage failures.

    Non-corruption operational errors such as connection refused and
    missing collections intentionally return ``None``.
    """
    if error is None:
        return None
    if isinstance(error, int):
        return "native_segfault" if error == -11 else None

    text = str(error)
    lower = text.lower()

    if "store_init_failed" in lower:
        return "store_init_failed"
    if any(signature in lower for signature in RESOURCE_EXHAUSTION_SIGNATURES):
        return None
    if "connection refused" in lower or "connecterror" in lower:
        return None
    if "collection" in lower and "not found" in lower:
        return None

    if "error executing plan" in lower and "error finding id" in lower:
        return "chroma_error_finding_id"
    if "database disk image is malformed" in lower:
        return "sqlite_malformed"
    if "sigsegv" in lower or "segmentation fault" in lower or "segfault" in lower:
        return "native_segfault"
    if "failed to get segments" in lower:
        return "chroma_transient"
    if "no such table" in lower:
        return "chroma_corruption"
    if "disk i/o error" in lower:
        return "chroma_transient"
    if "hnsw" in lower and any(token in lower for token in ("error", "panic", "corrupt", "segmentation")):
        return "hnsw_corruption"
    if any(token in lower for token in ("corrupt", "corruption", "malformed")) and any(
        scope in lower for scope in ("chroma", "sqlite", "hnsw", "database", "index")
    ):
        return "chroma_corruption"
    return None


def collection_owner(collection: str, default_anima: str | None = None) -> tuple[str | None, bool]:
    """Resolve collection owner and whether it is a shared collection."""
    is_shared = collection.startswith("shared_")
    if default_anima:
        return default_anima, is_shared
    if is_shared:
        return None, True
    for suffix in KNOWN_MEMORY_SUFFIXES:
        if collection.endswith(suffix):
            owner = collection[: -len(suffix)]
            return (owner or None), False
    return None, False


def get_repair_lock_path(anima_name: str) -> Path:
    from core.paths import get_animas_dir

    return get_animas_dir() / anima_name / "state" / "rag_repair.lock"


def is_repair_locked(anima_name: str) -> bool:
    """Return True when another process holds this anima's repair lock."""
    lock_path = get_repair_lock_path(anima_name)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            acquire_file_lock(lock_file, exclusive=True, blocking=False)
        except OSError:
            return True
        else:
            release_file_lock(lock_file)
            return False


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
    "store_init_failed",
}

# Corruption reasons whose claim is about the SQLite store itself and can
# therefore be definitively refuted by a passing ``PRAGMA quick_check``. A
# poisoned chromadb process cache makes a healthy on-disk DB raise these even
# though the file is intact (the false-positive that drives the repair churn).
# Segment/process-level reasons (hnsw_corruption, native_segfault) are NOT in
# this set because a SQLite check cannot vouch for hnsw segment files.
_SQLITE_REFUTABLE_REASONS = {"chroma_corruption", "sqlite_malformed"}
_STORE_INIT_FAILED_REASON = "store_init_failed"
_STORE_INIT_FAILED_THROTTLE = timedelta(minutes=10)
_PERSISTED_SIGNAL_WINDOW = timedelta(hours=24)


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
        if reason == "chroma_transient":
            return False

        owner, is_shared = collection_owner(collection, anima_name)
        if owner is None:
            logger.warning(
                "RAG corruption signal could not be mapped to an anima: collection=%s reason=%s",
                collection,
                reason,
            )
            return False

        # Suppress signals while a repair is already in flight for this owner.
        # During a rebuild the DB is transiently empty, so reads/writes surface
        # "no such table" errors; recording them would re-trigger another repair
        # the instant the current one releases its lock — the destructive loop.
        repair_in_flight = is_repair_locked(owner) or self._has_active_repair_state(owner)
        if repair_in_flight and reason != _STORE_INIT_FAILED_REASON:
            logger.debug(
                "Ignoring RAG corruption signal during active repair: owner=%s collection=%s reason=%s",
                owner,
                collection,
                reason,
            )
            return False

        # Cache-poisoning false-positive guard. chromadb's process-global system
        # cache can be transiently poisoned (e.g. by a sibling store's close()),
        # making a perfectly healthy on-disk DB raise sqlite-level corruption
        # errors ("Failed to get segments", "file is not a database", torn-read
        # "database corrupt"). Quarantining and rebuilding a healthy DB on those
        # false signals is what drives the destructive repair churn that
        # saturates the vector worker. If quick_check confirms the SQLite store
        # is intact, treat the error as transient and let the store-level
        # self-heal (cache reset + one retry) recover, without escalating to a
        # repair.
        if reason in _SQLITE_REFUTABLE_REASONS and self._sqlite_quick_check_ok(owner):
            logger.info(
                "Ignoring RAG corruption signal; on-disk SQLite passed quick_check "
                "(transient cache poisoning): owner=%s collection=%s reason=%s error=%s",
                owner,
                collection,
                reason,
                error,
            )
            return False

        if reason == _STORE_INIT_FAILED_REASON and self._store_init_signal_throttled(owner):
            logger.debug(
                "Ignoring throttled vector-store init failure signal: owner=%s collection=%s",
                owner,
                collection,
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

        if repair_in_flight:
            logger.debug(
                "Recorded vector-store init failure during active repair without requesting another: owner=%s",
                owner,
            )
            return False

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
        )

    @staticmethod
    def _sqlite_quick_check_ok(owner: str) -> bool:
        """Return True only if the owner's Chroma SQLite passes quick_check.

        Ambiguous results (busy / timeout / unavailable / missing / corrupt)
        return False so a potentially real corruption signal is not suppressed.
        """
        try:
            from core.memory.rag.sqlite_health import quick_check_chroma_sqlite
            from core.paths import get_anima_vectordb_dir

            result = quick_check_chroma_sqlite(get_anima_vectordb_dir(owner))
        except Exception:
            logger.debug("quick_check gate failed for owner=%s", owner, exc_info=True)
            return False
        return result.status == "ok"

    def _record_signal(self, anima_name: str, signal: dict[str, Any]) -> None:
        cutoff = utc_now() - self.window
        with self._lock:
            signals = self._signals.setdefault(anima_name, [])
            signals.append(signal)
            self._signals[anima_name] = repair_state.prune_recent_signals(signals, cutoff)
        repair_state.append_state_signal(
            anima_name,
            signal,
            max(self.window, _PERSISTED_SIGNAL_WINDOW),
        )

    def _store_init_signal_throttled(self, anima_name: str) -> bool:
        cutoff = utc_now() - _STORE_INIT_FAILED_THROTTLE
        with self._lock:
            in_memory = list(self._signals.get(anima_name, []))
        persisted = repair_state.read_state(anima_name).get("recent_signals", [])
        persisted_signals = persisted if isinstance(persisted, list) else []
        for signal in [*in_memory, *persisted_signals]:
            if not isinstance(signal, dict) or signal.get("reason") != _STORE_INIT_FAILED_REASON:
                continue
            at = parse_dt(signal.get("at"))
            if at is not None and at >= cutoff:
                return True
        return False

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
        state = repair_state.read_state(anima_name)
        last_success = parse_dt(state.get("last_success_at"))
        if self._contains_recent_signal(
            in_memory,
            cutoff,
            include_shared=include_shared,
            last_success=last_success,
        ):
            return True
        return self._contains_recent_signal(
            state.get("recent_signals", []),
            cutoff,
            include_shared=include_shared,
            last_success=last_success,
        )

    def list_repairable_animas(self, *, animas_dir: Path | None = None) -> list[str]:
        """Return all enabled Anima names available for repair discovery."""
        if animas_dir is None:
            from core.paths import get_animas_dir

            animas_dir = get_animas_dir()
        if not animas_dir.is_dir():
            return []
        repairable: list[str] = []
        for anima_dir in sorted(animas_dir.iterdir()):
            if (
                not anima_dir.is_dir()
                or not (anima_dir / IDENTITY_FILE).is_file()
                or not self._anima_enabled(anima_dir)
            ):
                continue
            repairable.append(anima_dir.name)
        return repairable

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
        last_success_by_anima: dict[str, datetime | None] = {}
        suspects: set[str] = set()

        for anima_name in repairable:
            state = repair_state.read_state(anima_name, animas_dir=animas_dir)
            last_success_by_anima[anima_name] = parse_dt(state.get("last_success_at"))
            if self._state_is_suspect(state, cutoff):
                suspects.add(anima_name)

        if include_quick_check:
            from core.memory.rag.sqlite_health import check_anima_vectordb_health

            for anima_name in repairable:
                try:
                    health = check_anima_vectordb_health(
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
                log_at = self._log_line_timestamp(line)
                owners = self._owners_from_log_line(line, repairable=repairable)
                if owners:
                    suspects.update(
                        owner
                        for owner in owners
                        if owner in repairable_set
                        and self._after_last_success_or_unknown(log_at, last_success_by_anima.get(owner))
                    )
                elif "chromadb_rust_bindings" in line.lower() or "native_segfault" in line.lower():
                    suspects.update(
                        anima_name
                        for anima_name in repairable
                        if (animas_dir / anima_name / "vectordb").exists()
                        and self._after_last_success_or_unknown(
                            log_at,
                            last_success_by_anima.get(anima_name),
                        )
                    )

        return [name for name in repairable if name in suspects]

    def _after_last_success(at: datetime, last_success: datetime | None) -> bool:
        return last_success is None or at > last_success

    @staticmethod
    def _after_last_success_or_unknown(at: datetime | None, last_success: datetime | None) -> bool:
        return last_success is None if at is None else RAGRepairService._after_last_success(at, last_success)

    @staticmethod
    def _signal_reason_is_corruption(signal: dict[str, Any]) -> bool:
        reason = signal.get("reason")
        if not isinstance(reason, str) or not reason:
            return True
        return reason in _KNOWN_CORRUPTION_REASONS

    @staticmethod
    def _contains_recent_signal(
        signals: list[dict[str, Any]],
        cutoff: datetime,
        *,
        include_shared: bool,
        last_success: datetime | None = None,
    ) -> bool:
        for signal in signals:
            if not RAGRepairService._signal_reason_is_corruption(signal):
                continue
            at = parse_dt(signal.get("at"))
            if at is None or at < cutoff:
                continue
            if not RAGRepairService._after_last_success(at, last_success):
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
        last_success = parse_dt(state.get("last_success_at"))
        signals = state.get("recent_signals")
        if isinstance(signals, list):
            for signal in signals:
                if not RAGRepairService._signal_reason_is_corruption(signal):
                    continue
                at = parse_dt(signal.get("at"))
                if at is not None and at >= cutoff and RAGRepairService._after_last_success(at, last_success):
                    return True

        reason = state.get("reason") or state.get("last_reason")
        if reason not in _KNOWN_CORRUPTION_REASONS:
            return False
        status = state.get("status")
        if status not in {"requested", "stopping", "repairing", "failed", "cooldown", "locked"}:
            return False
        for key in ("updated_at", "last_failure_at", "last_attempt_at", "requested_at"):
            at = parse_dt(state.get(key))
            if at is not None and at >= cutoff and RAGRepairService._after_last_success(at, last_success):
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
    def _log_line_timestamp(line: str) -> datetime | None:
        match = _TIMESTAMP_RE.search(line)
        if not match:
            return None
        return parse_dt(match.group("ts").replace("Z", "+00:00"))

    @staticmethod
    def _log_line_is_recent(line: str, cutoff: datetime) -> bool:
        at = RAGRepairService._log_line_timestamp(line)
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
    ) -> bool:
        """Write a request for supervisor-managed repair of an anima."""
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
