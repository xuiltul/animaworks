from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""SkillUsageTracker — append-only JSONL event recording and stats replay."""

import json
import logging
from pathlib import Path

from core.skills.models import SkillUsageEvent, SkillUsageEventType, SkillUsageStats
from core.time_utils import now_iso

logger = logging.getLogger(__name__)

_USAGE_FILE = "skill_usage.jsonl"


class SkillUsageTracker:
    """Records skill usage events and replays them into aggregate stats."""

    def __init__(self, anima_dir: Path) -> None:
        self._anima_dir = anima_dir
        self._usage_path = anima_dir / "state" / _USAGE_FILE
        self._session_views: set[str] = set()

    # ── Recording ──────────────────────────────────────────────

    def record(
        self,
        skill_name: str,
        event_type: SkillUsageEventType,
        *,
        is_common: bool = False,
        notes: str | None = None,
        source_origin: str | None = None,
    ) -> None:
        """Append a usage event to the JSONL file.

        For ``view`` events, debounces within the same session (one record per
        skill per tracker instance lifetime).
        """
        if event_type == SkillUsageEventType.view:
            key = f"{skill_name}:{is_common}"
            if key in self._session_views:
                return
            self._session_views.add(key)

        event = SkillUsageEvent(
            ts=now_iso(),
            skill_name=skill_name,
            event_type=event_type,
            is_common=is_common,
            notes=notes,
            source_origin=source_origin,
        )

        self._usage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._usage_path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

        logger.debug(
            "skill_usage: %s %s (common=%s)",
            event_type.value,
            skill_name,
            is_common,
        )

    # ── Statistics ─────────────────────────────────────────────

    def get_stats(self, skill_name: str) -> SkillUsageStats:
        """Replay JSONL and return aggregate stats for a single skill."""
        all_stats = self.get_all_stats()
        return all_stats.get(skill_name, SkillUsageStats(skill_name=skill_name))

    def get_all_stats(self) -> dict[str, SkillUsageStats]:
        """Replay full JSONL into per-skill aggregated stats."""
        stats: dict[str, SkillUsageStats] = {}

        if not self._usage_path.exists():
            return stats

        for line in self._usage_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                event = SkillUsageEvent.model_validate(data)
            except Exception:
                logger.warning("Skipping malformed skill_usage line: %s", line[:100])
                continue

            name = event.skill_name
            if name not in stats:
                stats[name] = SkillUsageStats(
                    skill_name=name,
                    is_common=event.is_common,
                )

            s = stats[name]
            match event.event_type:
                case SkillUsageEventType.view:
                    s.view_count += 1
                case SkillUsageEventType.use:
                    s.use_count += 1
                case SkillUsageEventType.success:
                    s.success_count += 1
                    s.last_used_at = event.ts
                case SkillUsageEventType.failure:
                    s.failure_count += 1
                    s.last_used_at = event.ts
                case SkillUsageEventType.patch:
                    s.patch_count += 1
                case SkillUsageEventType.create:
                    s.create_count += 1
                    s.created_at = s.created_at or event.ts
                    origin = event.source_origin or "unknown"
                    s.create_origins[origin] = s.create_origins.get(origin, 0) + 1

            if event.event_type in (
                SkillUsageEventType.view,
                SkillUsageEventType.use,
                SkillUsageEventType.success,
                SkillUsageEventType.failure,
            ):
                s.last_used_at = event.ts

        return stats

    def get_stale_candidates(self, days: int = 90) -> list[str]:
        """Return skill names not used within the specified number of days."""
        from datetime import UTC, datetime, timedelta

        cutoff = datetime.now(tz=UTC) - timedelta(days=days)
        cutoff_iso = cutoff.isoformat()

        all_stats = self.get_all_stats()
        stale: list[str] = []

        for name, s in all_stats.items():
            if s.last_used_at is None or s.last_used_at < cutoff_iso:
                stale.append(name)

        return sorted(stale)

    def reset_session_views(self) -> None:
        """Clear the session debounce set (useful for testing or new sessions)."""
        self._session_views.clear()
