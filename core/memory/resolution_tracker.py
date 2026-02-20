from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from datetime import timedelta
from pathlib import Path

from core.time_utils import now_iso, now_jst
from core.paths import get_shared_dir

logger = logging.getLogger("animaworks.memory")

# ── ResolutionTracker ─────────────────────────────────────


class ResolutionTracker:
    """Shared resolution tracking via JSONL append-only log."""

    def append_resolution(self, issue: str, resolver: str) -> None:
        """Append resolution info to shared/resolutions.jsonl."""
        shared_dir = get_shared_dir()
        path = shared_dir / "resolutions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": now_iso(),
            "issue": issue,
            "resolver": resolver,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_resolutions(self, days: int = 7) -> list[dict[str, str]]:
        """Read recent resolutions from shared/resolutions.jsonl."""
        shared_dir = get_shared_dir()
        path = shared_dir / "resolutions.jsonl"
        if not path.exists():
            return []
        cutoff = (now_jst() - timedelta(days=days)).isoformat()
        entries: list[dict[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("ts", "") >= cutoff:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
        return entries
