from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""MemoryBackend abstraction layer — unified interface for memory storage."""

from core.memory.backend.base import MemoryBackend, RetrievedMemory
from core.memory.backend.registry import get_backend

__all__ = ["MemoryBackend", "RetrievedMemory", "get_backend"]
