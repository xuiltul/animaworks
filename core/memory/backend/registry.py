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


def resolve_backend_type(anima_dir: Path) -> str:
    """Resolve memory backend type: per-anima status.json → global config → 'legacy'.

    Resolution order:
        1. ``status.json`` field ``memory_backend`` (per-anima override)
        2. Global ``config.json`` field ``memory.backend``
        3. Default ``"legacy"``
    """
    import json

    status_path = anima_dir / "status.json"
    if status_path.is_file():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            per_anima = data.get("memory_backend")
            if per_anima:
                return per_anima
        except Exception:
            logger.debug("Failed to read memory_backend from status.json", exc_info=True)

    try:
        from core.config.models import load_config

        cfg = load_config()
        return getattr(getattr(cfg, "memory", None), "backend", "legacy")
    except Exception:
        return "legacy"
