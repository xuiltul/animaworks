from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Resumable Anima merge lifecycle through tombstone finalization."""

from .finalize import AnimaMergeFinalizeService, FinalizeResult
from .journal import FinalizePhase, MergeJournal, MergePhase
from .service import AnimaMergeError, AnimaMergeService, MergeResult

__all__ = [
    "AnimaMergeFinalizeService",
    "AnimaMergeError",
    "AnimaMergeService",
    "FinalizePhase",
    "FinalizeResult",
    "MergeJournal",
    "MergePhase",
    "MergeResult",
]
