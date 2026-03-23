from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helper functions and mixin for org tool handlers."""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from core.i18n import t
from core.tooling.handler_base import _error_result

if TYPE_CHECKING:
    pass

logger = logging.getLogger("animaworks.tool_handler")


def resolve_anima_name(raw: str) -> str:
    """Resolve a raw name string to the canonical (lowercase) anima name.

    Resolution order:
      1. Strip + lowercase → exact match against config.animas keys
      2. Check ``aliases`` field of each anima (case-insensitive), e.g. Japanese names
      3. Return the stripped value as-is (caller will get AnimaNotFound if invalid)

    This function never raises; callers detect unknown names via config lookups.
    """
    from core.config.models import load_config

    stripped = raw.strip()
    lower = stripped.lower()

    try:
        config = load_config()
    except Exception:
        return lower  # fallback; caller will handle missing key

    # 1. Exact lowercase match
    if lower in config.animas:
        return lower

    # 2. Alias match (supports Japanese, display names, etc.)
    for canonical, cfg in config.animas.items():
        for alias in cfg.aliases:
            if alias.strip().lower() == lower or alias.strip() == stripped:
                return canonical

    return lower  # unknown; return as-is so caller gets AnimaNotFound


class OrgHelpersMixin:
    """Shared helpers for subordinate/descendant checks and org utilities."""

    # Declared for type-checker visibility
    _anima_dir: Path
    _anima_name: str

    def _check_subordinate(self, target_name: str) -> str | None:
        """Verify that *target_name* is a direct subordinate of this anima."""
        from core.config.models import load_config

        target_name = resolve_anima_name(target_name)
        if target_name == self._anima_name:
            return _error_result(
                "PermissionDenied",
                t("handler.self_operation_denied"),
            )

        try:
            config = load_config()
        except Exception as e:
            return _error_result("ConfigError", t("handler.config_load_failed", e=e))

        target_cfg = config.animas.get(target_name)
        if target_cfg is None:
            return _error_result(
                "AnimaNotFound",
                t("handler.anima_not_found", target_name=target_name),
            )

        if target_cfg.supervisor != self._anima_name:
            return _error_result(
                "PermissionDenied",
                t("handler.not_direct_subordinate", target_name=target_name),
                context={"supervisor": target_cfg.supervisor or t("handler.none_value")},
            )

        return None

    def _get_all_descendants(self, root_name: str | None = None) -> list[str]:
        """Get all descendant Anima names recursively via supervisor chain."""
        from core.config.models import load_config

        config = load_config()
        root = root_name or self._anima_name
        descendants: list[str] = []
        visited: set[str] = {root}
        queue = [name for name, cfg in config.animas.items() if cfg.supervisor == root]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            descendants.append(current)
            queue.extend(name for name, cfg in config.animas.items() if cfg.supervisor == current)
        return descendants

    def _get_direct_subordinates(self) -> list[str]:
        """Return names of direct subordinates (supervisor == self)."""
        from core.config.models import load_config

        config = load_config()
        return [name for name, cfg in config.animas.items() if cfg.supervisor == self._anima_name]

    def _check_descendant(self, target_name: str) -> str | None:
        """Verify that target_name is a descendant (any depth) of this anima."""
        target_name = resolve_anima_name(target_name)
        if target_name == self._anima_name:
            return _error_result(
                "PermissionDenied",
                t("handler.self_operation_denied"),
            )
        descendants = self._get_all_descendants()
        if target_name not in descendants:
            return _error_result(
                "PermissionDenied",
                t("handler.not_descendant", target_name=target_name),
            )
        return None

    @staticmethod
    def _read_recent_activity(anima_dir: Path, *, limit: int = 1) -> list:
        """Read recent activity entries from another anima's directory."""
        from core.memory.activity import ActivityLogger

        al = ActivityLogger(anima_dir)
        return al.recent(days=1, limit=limit)

    @staticmethod
    def _parse_since(raw: str | None) -> datetime | None:
        """Parse ``HH:MM`` string into a timezone-aware datetime (today, JST)."""
        if not raw:
            return None
        from datetime import time as _time

        from core.memory.activity import now_local

        now = now_local()
        try:
            parts = raw.strip().split(":")
            t_obj = _time(int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return None
        return datetime.combine(now.date(), t_obj, tzinfo=now.tzinfo)
