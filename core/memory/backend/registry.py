from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend registry — factory function for memory backend instantiation."""

import logging
from pathlib import Path

from core.memory.backend.base import MemoryBackend

logger = logging.getLogger(__name__)

# ── Factory ────────────────────────────────────────────────────────────────


def get_backend(
    backend_type: str,
    anima_dir: Path,
    **kwargs: object,
) -> MemoryBackend:
    """Create and return a :class:`MemoryBackend` instance.

    Args:
        backend_type: ``"legacy"`` (ChromaDB) or ``"neo4j"`` (future).
        anima_dir: Root directory of the Anima whose memory to manage.
        **kwargs: Extra keyword arguments forwarded to the backend constructor.

    Returns:
        A concrete :class:`MemoryBackend` implementation.

    Raises:
        NotImplementedError: If *backend_type* is planned but not yet available.
        ValueError: If *backend_type* is unknown.
    """
    if backend_type == "legacy":
        from core.memory.backend.legacy import LegacyRAGBackend

        return LegacyRAGBackend(anima_dir, **kwargs)

    if backend_type == "neo4j":
        try:
            from core.memory.backend.neo4j_graph import Neo4jGraphBackend
        except ImportError:
            raise ImportError(
                "Neo4j backend requires the neo4j extra. Install with: pip install animaworks[neo4j]"
            ) from None

        return Neo4jGraphBackend(anima_dir, **kwargs)

    raise ValueError(f"Unknown memory backend: {backend_type}")
