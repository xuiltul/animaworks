# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""GWS to Notion deduplication and idempotency module.

Prevents duplicate registration of GWS-extracted development requests
into Notion by checking source_id-based idempotency keys against
the Notion DB and a local cache.

Design: s3-2_dedup_design.md
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("animaworks.tools")


# -- Idempotency key generation (section 2.3) --


def generate_idempotency_key(record: dict[str, Any]) -> str | None:
    """Generate an idempotency key for a GWS record.

    Priority:
        1. source_id -> "sid:<source_id>"
        2. SHA256(request_title + received_at) -> "hash:<16 hex chars>"
        3. None (key generation impossible)
    """
    source_id = (record.get("source_id") or "").strip()
    if source_id:
        return f"sid:{source_id}"

    title = (record.get("request_title") or "").strip()
    date = (record.get("received_at") or "").strip()
    if title and date:
        raw = f"{title}|{date}"
        hash_val = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"hash:{hash_val}"

    return None


# -- Notion duplicate check (section 4.2) --


def check_duplicate_in_notion(
    client: Any,
    database_id: str,
    idempotency_key: str,
) -> bool:
    """Check if a record with the given idempotency key exists in Notion."""
    result = client.query_database(
        database_id=database_id,
        filter={
            "property": "冪等性キー",
            "rich_text": {"equals": idempotency_key},
        },
        page_size=1,
    )
    return len(result.get("results", [])) > 0


# -- Local cache (section 6) --


class DedupCache:
    """Local file-backed cache for idempotency keys."""

    def __init__(self, cache_path: str, max_entries: int = 1000) -> None:
        self._path = cache_path
        self._max_entries = max_entries
        self._entries: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self._entries = data.get("entries", {})
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache file corrupted, initializing empty: %s", exc)
            self._entries = {}

    def save(self) -> None:
        """Persist cache to disk atomically."""
        tmp_path = self._path + ".tmp"
        data = {
            "version": 1,
            "max_entries": self._max_entries,
            "entries": self._entries,
            "last_updated": datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    def contains(self, key: str) -> bool:
        return key in self._entries

    def add(self, key: str, notion_page_id: str) -> None:
        self._entries[key] = {
            "notion_page_id": notion_page_id,
            "registered_at": datetime.now().isoformat(),
        }
        self._evict()

    def _evict(self) -> None:
        if len(self._entries) <= self._max_entries:
            return
        sorted_keys = sorted(
            self._entries,
            key=lambda k: self._entries[k].get("registered_at", ""),
        )
        excess = len(self._entries) - self._max_entries
        for k in sorted_keys[:excess]:
            del self._entries[k]


# -- Lock mechanism (section 5.2) --

_STALE_THRESHOLD = 3600  # 1 hour


def acquire_lock(lock_path: str) -> bool:
    """Acquire an exclusive lock file for pipeline execution."""
    if os.path.exists(lock_path):
        lock_age = time.time() - os.path.getmtime(lock_path)
        if lock_age < _STALE_THRESHOLD:
            return False
        logger.warning("Stale lock detected (age: %.0fs), forcing release", lock_age)

    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(
            {"pid": os.getpid(), "started_at": datetime.now().isoformat()}, f
        )
    return True


def release_lock(lock_path: str) -> None:
    """Release the execution lock."""
    if os.path.exists(lock_path):
        os.remove(lock_path)


# -- Dedup result --


@dataclass
class DedupResult:
    """Result of the deduplication check."""

    new: list[tuple[dict, dict]] = field(default_factory=list)
    duplicates: list[tuple[dict, dict, str]] = field(default_factory=list)
    no_key: list[tuple[dict, dict]] = field(default_factory=list)
    api_errors: list[tuple[dict, dict, str]] = field(default_factory=list)


# -- Main dedup function (section 4.1) --


def deduplicate_records(
    records: list[tuple[dict, dict]],
    client: Any,
    database_id: str,
    cache: DedupCache,
) -> DedupResult:
    """Filter out duplicate records using cache and Notion DB.

    Flow per record:
        1. Generate idempotency key
        2. Check local cache -> if hit, skip
        3. Check Notion DB -> if hit, skip + add to cache
        4. Otherwise, mark as new
    """
    result = DedupResult()

    for rec, props in records:
        key = generate_idempotency_key(rec)

        if key is None:
            logger.warning(
                "Cannot generate idempotency key for record: %s",
                rec.get("request_title", "?")[:50],
            )
            result.new.append((rec, props))
            result.no_key.append((rec, props))
            continue

        if cache.contains(key):
            logger.info("Cache hit (duplicate): %s", key)
            result.duplicates.append((rec, props, key))
            continue

        try:
            is_dup = check_duplicate_in_notion(client, database_id, key)
        except Exception as exc:
            logger.error("Notion API error during dedup check for %s: %s", key, exc)
            result.new.append((rec, props))
            result.api_errors.append((rec, props, str(exc)))
            continue

        if is_dup:
            logger.info("Notion hit (duplicate): %s", key)
            cache.add(key, "notion-existing")
            result.duplicates.append((rec, props, key))
        else:
            result.new.append((rec, props))

    return result
