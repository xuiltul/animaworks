from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for RAG corruption detection and repair locking."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.platform.locks import acquire_file_lock, release_file_lock

KNOWN_MEMORY_SUFFIXES = (
    "_knowledge",
    "_episodes",
    "_procedures",
    "_skills",
    "_conversation_summary",
    "_shared_users",
)

SINGLE_SHOT_REASONS = {
    "sqlite_malformed",
    "chroma_corruption",
    "hnsw_corruption",
    "native_segfault",
}

RESOURCE_EXHAUSTION_SIGNATURES = (
    "too many open files",
    "os error 24",
    "errno 24",
    "emfile",
    "enfile",
    "unable to open database file",
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).isoformat()


def classify_corruption_error(error: BaseException | str | int | None) -> str | None:
    """Return a stable corruption reason for known RAG storage failures.

    Non-corruption operational errors such as connection refused and
    missing collections intentionally return ``None``.
    """
    if error is None:
        return None
    if isinstance(error, int):
        return "native_segfault" if error == -11 else None

    text = str(error)
    lower = text.lower()

    if any(signature in lower for signature in RESOURCE_EXHAUSTION_SIGNATURES):
        return None
    if "connection refused" in lower or "connecterror" in lower:
        return None
    if "collection" in lower and "not found" in lower:
        return None

    if "error executing plan" in lower and "error finding id" in lower:
        return "chroma_error_finding_id"
    if "database disk image is malformed" in lower:
        return "sqlite_malformed"
    if "sigsegv" in lower or "segmentation fault" in lower or "segfault" in lower:
        return "native_segfault"
    if "failed to get segments" in lower or "no such table" in lower:
        return "chroma_corruption"
    if "disk i/o error" in lower:
        return None
    if "hnsw" in lower and any(token in lower for token in ("error", "panic", "corrupt", "segmentation")):
        return "hnsw_corruption"
    if any(token in lower for token in ("corrupt", "corruption", "malformed")) and any(
        scope in lower for scope in ("chroma", "sqlite", "hnsw", "database", "index")
    ):
        return "chroma_corruption"
    return None


def collection_owner(collection: str, default_anima: str | None = None) -> tuple[str | None, bool]:
    """Resolve collection owner and whether it is a shared collection."""
    is_shared = collection.startswith("shared_")
    if default_anima:
        return default_anima, is_shared
    if is_shared:
        return None, True
    for suffix in KNOWN_MEMORY_SUFFIXES:
        if collection.endswith(suffix):
            owner = collection[: -len(suffix)]
            return (owner or None), False
    return None, False


def get_repair_lock_path(anima_name: str) -> Path:
    from core.paths import get_animas_dir

    return get_animas_dir() / anima_name / "state" / "rag_repair.lock"


def is_repair_locked(anima_name: str) -> bool:
    """Return True when another process holds this anima's repair lock."""
    lock_path = get_repair_lock_path(anima_name)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            acquire_file_lock(lock_file, exclusive=True, blocking=False)
        except OSError:
            return True
        else:
            release_file_lock(lock_file)
            return False
