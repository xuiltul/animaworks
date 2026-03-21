from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Global permissions cache — singleton loaded once at server startup.

The on-disk ``permissions.global.json`` is read at startup, compiled into
regex patterns, and cached in memory.  Runtime disk changes are detected
by a periodic integrity check and auto-reverted.
"""

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from threading import Lock

from core.config.schemas import GlobalDenyPattern, GlobalPermissionsConfig

logger = logging.getLogger("animaworks.config.global_permissions")

# ── Helpers ──────────────────────────────────────────────────


def _compile_patterns(
    items: list[GlobalDenyPattern],
) -> list[tuple[re.Pattern[str], str]]:
    """Compile a list of *GlobalDenyPattern* into ``(regex, reason)`` tuples.

    Invalid patterns are logged as warnings and skipped.
    """
    compiled: list[tuple[re.Pattern[str], str]] = []
    for item in items:
        try:
            compiled.append((re.compile(item.pattern), item.reason))
        except re.error as exc:
            logger.warning(
                "Invalid regex in permissions.global.json: %r — %s (skipped)",
                item.pattern,
                exc,
            )
    return compiled


def _build_injection_re(
    items: list[GlobalDenyPattern],
) -> re.Pattern[str] | None:
    """Combine injection patterns into a single compiled regex.

    Each pattern is joined with ``|`` so a single ``.search()`` covers all.
    Returns ``None`` when the list is empty.
    """
    if not items:
        return None
    parts: list[str] = []
    for item in items:
        try:
            re.compile(item.pattern)
            parts.append(item.pattern)
        except re.error as exc:
            logger.warning(
                "Invalid injection regex in permissions.global.json: %r — %s (skipped)",
                item.pattern,
                exc,
            )
    if not parts:
        return None
    return re.compile("|".join(parts))


# ── Singleton Cache ──────────────────────────────────────────


class GlobalPermissionsCache:
    """Thread-safe singleton cache for ``permissions.global.json``.

    Lifecycle
    ---------
    1. ``load(path)`` at server startup — reads file, validates JSON & regex,
       computes SHA-256, optionally prompts for interactive confirmation when
       the hash differs from the previous run.
    2. ``check_integrity()`` every 5 minutes — compares on-disk hash with the
       cached hash; restores from cache on mismatch.
    3. ``injection_re`` / ``blocked_patterns`` — read-only accessors used by
       ToolHandler and Mode S security checks.
    """

    _instance: GlobalPermissionsCache | None = None
    _lock = Lock()

    def __init__(self) -> None:
        self._config: GlobalPermissionsConfig | None = None
        self._compiled_injection: re.Pattern[str] | None = None
        self._compiled_deny: list[tuple[re.Pattern[str], str]] = []
        self._file_path: Path | None = None
        self._cached_content: str = ""
        self._cached_hash: str = ""

    # ── Singleton accessor ───────────────────────────────────

    @classmethod
    def get(cls) -> GlobalPermissionsCache:
        """Return the singleton instance (create on first call)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton — **test-only**."""
        with cls._lock:
            cls._instance = None

    # ── Load ─────────────────────────────────────────────────

    def load(self, path: Path, *, interactive: bool = True) -> None:
        """Load, validate, cache, and optionally check startup hash.

        Args:
            path: Absolute path to ``permissions.global.json``.
            interactive: When *True* (default) and the file hash differs
                from the previous run, prompt via stdin (requires TTY).

        Raises:
            FileNotFoundError: If the file does not exist.
            SystemExit: If the user rejects hash-mismatch confirmation
                or a non-interactive session encounters a mismatch.
            json.JSONDecodeError / ValidationError: On parse failure.
        """
        if not path.is_file():
            raise FileNotFoundError(
                f"permissions.global.json not found at {path}. Run 'animaworks init' to generate it."
            )

        content = path.read_text(encoding="utf-8")
        current_hash = hashlib.sha256(content.encode()).hexdigest()

        # Startup hash comparison
        hash_path = path.parent / "run" / "permissions_global.sha256"
        if interactive and hash_path.is_file():
            previous_hash = hash_path.read_text(encoding="utf-8").strip()
            if previous_hash and previous_hash != current_hash:
                logger.warning(
                    "permissions.global.json was modified since last server run (hash: %s → %s)",
                    previous_hash[:12],
                    current_hash[:12],
                )
                if sys.stdin.isatty():
                    answer = input(
                        "permissions.global.json was modified outside of normal "
                        "server lifecycle. Accept changes? [yes/no]: "
                    )
                    if answer.strip().lower() != "yes":
                        raise SystemExit("Aborted: permissions.global.json changes rejected")
                    logger.info("User accepted modified permissions.global.json")
                else:
                    raise SystemExit(
                        "permissions.global.json was modified and non-interactive "
                        "session cannot confirm. Start server from an interactive terminal."
                    )

        config = GlobalPermissionsConfig.model_validate(json.loads(content))

        with self._lock:
            self._config = config
            self._compiled_injection = _build_injection_re(config.injection_patterns)
            self._compiled_deny = _compile_patterns(config.commands.deny)
            self._file_path = path
            self._cached_content = content
            self._cached_hash = current_hash

        # Persist hash for next startup
        hash_path.parent.mkdir(parents=True, exist_ok=True)
        hash_path.write_text(current_hash, encoding="utf-8")

        logger.info(
            "Global permissions loaded: %d injection patterns, %d deny patterns (hash: %s)",
            len(config.injection_patterns),
            len(config.commands.deny),
            current_hash[:12],
        )

    # ── Integrity ────────────────────────────────────────────

    def check_integrity(self) -> bool:
        """Compare on-disk hash with cached hash.  Restore on mismatch.

        Returns *True* when the file is intact, *False* when it was
        tampered with (and has been restored).
        """
        if self._file_path is None or not self._cached_content:
            return True

        try:
            current = self._file_path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("permissions.global.json missing — restoring from cache")
            self._restore()
            return False

        current_hash = hashlib.sha256(current.encode()).hexdigest()
        if current_hash != self._cached_hash:
            logger.warning(
                "permissions.global.json tampered (expected %s, got %s) — restoring",
                self._cached_hash[:12],
                current_hash[:12],
            )
            self._restore()
            return False
        return True

    def _restore(self) -> None:
        """Overwrite the on-disk file with the cached startup content."""
        if self._file_path is None:
            return
        try:
            self._file_path.write_text(self._cached_content, encoding="utf-8")
            logger.info("permissions.global.json restored from cache")
        except OSError as exc:
            logger.error("Failed to restore permissions.global.json: %s", exc)

    # ── Read-only accessors ──────────────────────────────────

    @property
    def injection_re(self) -> re.Pattern[str] | None:
        """Combined injection-detection regex, or *None* if empty."""
        return self._compiled_injection

    @property
    def blocked_patterns(self) -> list[tuple[re.Pattern[str], str]]:
        """Compiled ``(regex, reason)`` tuples for command deny."""
        return self._compiled_deny

    @property
    def config(self) -> GlobalPermissionsConfig | None:
        """Raw config object (for inspection / tests)."""
        return self._config

    @property
    def loaded(self) -> bool:
        """Whether :meth:`load` has been called successfully."""
        return self._config is not None
