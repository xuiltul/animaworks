# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Atomic JSON snapshot store for external tasks."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.external_tasks.models import Snapshot
from core.memory._io import atomic_write_text

logger = logging.getLogger("animaworks.external_tasks.store")


class ExternalTaskStore:
    """Load/save :class:`Snapshot` to a single JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Snapshot:
        """Load snapshot; return empty on missing file, corrupt JSON, or schema mismatch."""
        if not self.path.exists():
            return Snapshot()
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return Snapshot.model_validate(data)
        except Exception:
            logger.warning(
                "Failed to load external tasks snapshot from %s; returning empty",
                self.path,
                exc_info=True,
            )
            return Snapshot()

    def save(self, snapshot: Snapshot) -> None:
        """Persist *snapshot* atomically."""
        content = json.dumps(
            snapshot.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        atomic_write_text(self.path, content)
