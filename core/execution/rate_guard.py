# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Cross-process LLM rate guard (fleet-wide circuit breaker).

Records rate-limit / overload state for a provider *family* in a single shared
JSON file so every Anima process checks it before hammering the same shared
credential.  When one Anima hits a 429 the others skip that backend until the
recorded window expires, instead of amplifying the throttle across ~27
processes.

The guard is fail-open by design: any read or write error is swallowed and
treated as "not blocked".  A broken guard must never stop a healthy call — the
same lesson as the shared-knowledge-DB corruption cascade, where a self-heal
mechanism became the outage.

Writes are serialized with an advisory ``flock`` on a sidecar lock file (the
JSON body is swapped via ``os.replace`` so its inode changes and cannot itself
be locked).  If ``flock`` is unavailable or fails, writing proceeds unlocked —
fail-open takes priority over strict mutual exclusion.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Iterator

    from core.config.schemas import LlmRateGuardConfig

logger = logging.getLogger("animaworks.execution.rate_guard")

_STATE_FILENAME = "llm_rate_guard.json"
# Auto-resolved config is refreshed at most this often so the ``enabled: false``
# emergency switch takes effect within a few seconds (not at process restart),
# while avoiding a config lookup on every async-loop query.
_CONFIG_TTL_S = 5.0


def _load_guard_config() -> LlmRateGuardConfig:
    """Load the guard config, falling back to defaults on any failure."""
    from core.config.schemas import LlmRateGuardConfig

    try:
        from core.config import load_config

        return load_config().llm_rate_guard
    except Exception:
        logger.debug("failed to load llm_rate_guard config; using defaults", exc_info=True)
        return LlmRateGuardConfig()


def _current_anima_name() -> str:
    """Best-effort identity for the ``updated_by`` field (observability only)."""
    name = os.environ.get("ANIMAWORKS_ANIMA_NAME")
    if name:
        return name
    anima_dir = os.environ.get("ANIMAWORKS_ANIMA_DIR")
    if anima_dir:
        return Path(anima_dir).name
    return "unknown"


class LlmRateGuard:
    """File-backed, fail-open rate guard keyed by provider family."""

    def __init__(
        self,
        *,
        config: LlmRateGuardConfig | None = None,
        path: Path | None = None,
    ) -> None:
        self._explicit_config = config
        self._path = path
        self._cached_config: LlmRateGuardConfig | None = None
        self._cached_config_at = 0.0
        self._state_cache: dict | None = None
        self._state_cache_key: tuple[int, int] | None = None

    @property
    def config(self) -> LlmRateGuardConfig:
        # An explicit config (tests / callers) is authoritative and never
        # re-resolved.  Otherwise resolve through load_config on a short TTL so
        # runtime config changes (notably ``enabled: false``) are picked up.
        if self._explicit_config is not None:
            return self._explicit_config
        now = time.monotonic()
        if self._cached_config is None or (now - self._cached_config_at) > _CONFIG_TTL_S:
            self._cached_config = _load_guard_config()
            self._cached_config_at = now
        return self._cached_config

    def _resolve_path(self) -> Path:
        if self._path is not None:
            return self._path
        from core.paths import get_shared_dir

        return get_shared_dir() / _STATE_FILENAME

    def _resolve_lock_path(self) -> Path:
        return self._resolve_path().with_suffix(".lock")

    def blocked_remaining(self, provider_family: str) -> float:
        """Return seconds this family stays blocked, or ``0.0`` if free.

        Read-only and cheap: a stat + mtime cache avoids re-parsing the file
        when it has not changed.  Any error is treated as "not blocked".
        """
        if not self.config.enabled:
            return 0.0
        try:
            state = self._read_state_cached()
        except Exception:
            logger.debug("rate guard read failed; failing open", exc_info=True)
            return 0.0

        entry = state.get(provider_family)
        if not isinstance(entry, dict):
            return 0.0
        blocked_until = entry.get("blocked_until")
        if not isinstance(blocked_until, (int, float)):
            return 0.0
        remaining = float(blocked_until) - time.time()
        return remaining if remaining > 0 else 0.0

    def report_block(self, provider_family: str, seconds: float, reason: str) -> None:
        """Record a block for *provider_family* lasting *seconds*.

        ``seconds`` is clamped to ``[0, max_block_seconds]``; a non-positive
        value falls back to ``default_block_seconds``.  The read-modify-write is
        serialized with an advisory lock and the JSON body is replaced
        atomically; any failure is swallowed (fail-open).
        """
        cfg = self.config
        if not cfg.enabled:
            return

        try:
            block_s = float(seconds)
        except (TypeError, ValueError):
            block_s = float(cfg.default_block_seconds)
        if block_s <= 0:
            block_s = float(cfg.default_block_seconds)
        block_s = min(block_s, float(cfg.max_block_seconds))

        try:
            with self._locked():
                try:
                    state = self._read_state()
                except Exception:
                    logger.debug("rate guard read-before-write failed; starting fresh", exc_info=True)
                    state = {}
                state[provider_family] = {
                    "blocked_until": time.time() + block_s,
                    "reason": reason,
                    "updated_by": _current_anima_name(),
                }
                self._write_state(state)
        except Exception:
            logger.debug("rate guard write failed; failing open", exc_info=True)
            return
        logger.info(
            "LLM rate guard: %s blocked for %.0fs (%s)",
            provider_family,
            block_s,
            reason,
        )

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold an exclusive advisory lock for the write critical section.

        Fail-open: if ``flock`` is unavailable or cannot be acquired, the block
        is yielded unlocked (last-writer-wins) rather than dropping the write.
        """
        if fcntl is None:
            yield
            return
        lock_path = self._resolve_lock_path()
        lock_file = None
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_file = open(lock_path, "w")  # noqa: SIM115 - closed in finally
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except Exception:
            logger.debug("rate guard flock failed; proceeding without lock", exc_info=True)
            if lock_file is not None:
                with contextlib.suppress(OSError):
                    lock_file.close()
            yield
            return
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                lock_file.close()

    def _read_state_cached(self) -> dict:
        """Read state via a stat+mtime cache (hot path for blocked_remaining)."""
        path = self._resolve_path()
        try:
            st = os.stat(path)
        except FileNotFoundError:
            self._state_cache = {}
            self._state_cache_key = None
            return {}
        key = (st.st_mtime_ns, st.st_size)
        if self._state_cache is not None and self._state_cache_key == key:
            return self._state_cache
        data = self._read_state()
        self._state_cache = data
        self._state_cache_key = key
        return data

    def _read_state(self) -> dict:
        path = self._resolve_path()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _write_state(self, state: dict) -> None:
        path = self._resolve_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.debug("failed to unlink temp guard file %s", tmp_path, exc_info=True)
            raise


_shared_guard: LlmRateGuard | None = None


def get_rate_guard() -> LlmRateGuard:
    """Return the process-wide shared rate guard (config resolved lazily)."""
    global _shared_guard
    if _shared_guard is None:
        _shared_guard = LlmRateGuard()
    return _shared_guard


__all__ = ["LlmRateGuard", "get_rate_guard"]
