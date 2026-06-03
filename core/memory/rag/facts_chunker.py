from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Chunk legacy atomic facts JSONL files for RAG indexing."""

import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from core.memory.facts import FactRecord
from core.time_utils import ensure_aware

logger = logging.getLogger("animaworks.rag.facts_chunker")


def chunk_facts_jsonl(
    file_path: Path,
    content: str,
    *,
    anima_dir: Path,
    collection_prefix: str,
    make_chunk_id: Callable[[Path, str, int], str],
    chunk_factory: Callable[[str, str, dict[str, Any]], Any],
    origin: str = "",
) -> list[Any]:
    """Chunk a facts JSONL file into one vector document per fact."""
    records: list[tuple[int, FactRecord]] = []
    for line_no, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = FactRecord.from_json_line(line)
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Skipping invalid facts JSONL line %s:%d", file_path, line_no)
            continue
        if record.text:
            records.append((line_no, record))

    total = len(records)
    chunks: list[Any] = []
    stat = file_path.stat()
    try:
        source_file = str(file_path.relative_to(anima_dir))
    except ValueError:
        source_file = str(file_path)

    for chunk_idx, (line_no, record) in enumerate(records):
        metadata: dict[str, str | int | float | bool | list[str]] = {
            "anima": collection_prefix,
            "memory_type": "facts",
            "source_file": source_file,
            "chunk_index": chunk_idx,
            "total_chunks": total,
            "line_no": line_no,
            "fact_id": record.fact_id,
            "source_entity": record.source_entity,
            "target_entity": record.target_entity,
            "edge_type": record.edge_type,
            "raw_edge_type": record.raw_edge_type,
            "valid_at_iso": record.valid_at,
            "recorded_at": record.recorded_at,
            "valid_until": record.valid_until,
            "entities": list(record.entities),
            "source_episode": record.source_episode,
            "source_session_id": record.source_session_id,
            "confidence": record.confidence,
            "importance": "normal",
            "access_count": 0,
            "last_accessed_at": "",
            "activation_level": "normal",
            "low_activation_since": "",
            "created_at": ensure_aware(datetime.fromtimestamp(stat.st_ctime)).isoformat(),
            "updated_at": ensure_aware(datetime.fromtimestamp(stat.st_mtime)).isoformat(),
        }
        valid_at_ts = _parse_timestamp(record.valid_at)
        metadata["valid_at"] = valid_at_ts if valid_at_ts is not None else float(stat.st_mtime)
        if origin:
            metadata["origin"] = origin
        chunks.append(chunk_factory(make_chunk_id(file_path, "facts", chunk_idx), record.text, metadata))

    return chunks


def _parse_timestamp(value: str) -> float | None:
    if not value:
        return None
    try:
        return ensure_aware(datetime.fromisoformat(value.replace("Z", "+00:00"))).timestamp()
    except (TypeError, ValueError):
        return None
