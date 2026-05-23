# Copyright 2026 AnimaWorks contributors — Apache-2.0
"""Cross-encoder reranker — re-export from core.memory.retrieval."""

from __future__ import annotations

from core.memory.retrieval.reranker import CrossEncoderReranker, get_reranker

__all__ = ["CrossEncoderReranker", "get_reranker"]
