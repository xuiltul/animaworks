"""Tests for core.tools.gws_dedup — deduplication & idempotency module.

TDD RED phase: T1–T13 from s3-2_dedup_design.md §7.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tools.gws_dedup import (
    DedupCache,
    DedupResult,
    acquire_lock,
    check_duplicate_in_notion,
    deduplicate_records,
    generate_idempotency_key,
    release_lock,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_record(**overrides: object) -> dict:
    """Build a valid GWS record with sensible defaults."""
    base = {
        "request_title": "ダッシュボードの表示速度改善",
        "problem": "読み込みに10秒以上かかる",
        "desired_outcome": "3秒以内に表示",
        "product_area": "ダッシュボード",
        "urgency": "高",
        "evidence_quote": "複数顧客から同一報告あり",
        "confidence": "high",
        "source_id": "SRC-001",
        "received_at": "2026-01-15T10:30:00+09:00",
        "source_type": "メール",
        "source_link": "https://support.example.com/ticket/123",
        "product": "ProductA",
        "type": "FEATURE",
        "customer": "ABC株式会社",
        "text_excerpt": "ダッシュボードが遅い",
        "cluster_key": "dashboard-perf",
        "source_row_url": "https://docs.google.com/spreadsheets/d/xxx/edit#gid=0&range=A15",
    }
    base.update(overrides)
    return base


def _make_notion_props(record: dict) -> dict:
    """Build minimal Notion properties dict for testing."""
    return {
        "Name": {"title": [{"text": {"content": record["request_title"]}}]},
        "Status": {"select": {"name": "新規"}},
        "Label": {"multi_select": [{"name": "AI"}]},
        "AIによる要約": {"rich_text": [{"text": {"content": "test summary"}}]},
    }


def _mock_client_no_duplicates() -> MagicMock:
    """NotionClient mock that always returns no results (no duplicates)."""
    client = MagicMock()
    client.query_database.return_value = {"results": []}
    return client


def _mock_client_with_duplicate(page_id: str = "existing-page-123") -> MagicMock:
    """NotionClient mock that returns one existing page (duplicate found)."""
    client = MagicMock()
    client.query_database.return_value = {"results": [{"id": page_id}]}
    return client


# ── generate_idempotency_key ─────────────────────────────────────────────────


class TestGenerateIdempotencyKey:
    """Key generation logic (§2.3)."""

    def test_source_id_produces_sid_prefix(self):
        rec = _make_record(source_id="TICKET-001")
        key = generate_idempotency_key(rec)
        assert key == "sid:TICKET-001"

    def test_fallback_hash_when_no_source_id(self):
        rec = _make_record(source_id="")
        key = generate_idempotency_key(rec)
        assert key is not None
        assert key.startswith("hash:")
        assert len(key) == 5 + 16  # "hash:" + 16 hex chars

    def test_none_when_no_source_id_no_title(self):
        rec = _make_record(source_id="", request_title="", received_at="")
        key = generate_idempotency_key(rec)
        assert key is None

    def test_deterministic_hash(self):
        rec1 = _make_record(source_id="", request_title="foo", received_at="2026-01-01")
        rec2 = _make_record(source_id="", request_title="foo", received_at="2026-01-01")
        assert generate_idempotency_key(rec1) == generate_idempotency_key(rec2)

    def test_different_records_different_hash(self):
        rec1 = _make_record(source_id="", request_title="foo", received_at="2026-01-01")
        rec2 = _make_record(source_id="", request_title="bar", received_at="2026-01-01")
        assert generate_idempotency_key(rec1) != generate_idempotency_key(rec2)


# ── check_duplicate_in_notion ────────────────────────────────────────────────


class TestCheckDuplicateInNotion:
    """Notion DB duplicate check (§4.2)."""

    def test_no_duplicate_returns_false(self):
        client = _mock_client_no_duplicates()
        assert check_duplicate_in_notion(client, "db-id", "sid:SRC-001") is False

    def test_duplicate_found_returns_true(self):
        client = _mock_client_with_duplicate()
        assert check_duplicate_in_notion(client, "db-id", "sid:SRC-001") is True

    def test_queries_correct_filter(self):
        client = _mock_client_no_duplicates()
        check_duplicate_in_notion(client, "db-id", "sid:SRC-001")
        client.query_database.assert_called_once_with(
            database_id="db-id",
            filter={
                "property": "冪等性キー",
                "rich_text": {"equals": "sid:SRC-001"},
            },
            page_size=1,
        )


# ── DedupCache ───────────────────────────────────────────────────────────────


class TestDedupCache:
    """Local cache for dedup keys (§6)."""

    def test_new_cache_is_empty(self, tmp_path: Path):
        cache = DedupCache(cache_path=str(tmp_path / "cache.json"))
        assert cache.contains("sid:SRC-001") is False

    def test_add_and_contains(self, tmp_path: Path):
        cache = DedupCache(cache_path=str(tmp_path / "cache.json"))
        cache.add("sid:SRC-001", "page-123")
        assert cache.contains("sid:SRC-001") is True

    def test_persist_and_reload(self, tmp_path: Path):
        path = str(tmp_path / "cache.json")
        cache1 = DedupCache(cache_path=path)
        cache1.add("sid:SRC-001", "page-123")
        cache1.save()

        cache2 = DedupCache(cache_path=path)
        assert cache2.contains("sid:SRC-001") is True

    def test_lru_eviction_at_max(self, tmp_path: Path):
        """T11: Cache at max_entries evicts oldest."""
        cache = DedupCache(cache_path=str(tmp_path / "cache.json"), max_entries=5)
        for i in range(6):
            cache.add(f"sid:SRC-{i:03d}", f"page-{i}")
        # First entry should be evicted
        assert cache.contains("sid:SRC-000") is False
        assert cache.contains("sid:SRC-005") is True

    def test_corrupted_file_initializes_empty(self, tmp_path: Path):
        """T7: Corrupted cache file → empty cache, no error."""
        path = tmp_path / "cache.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        cache = DedupCache(cache_path=str(path))
        assert cache.contains("sid:anything") is False


# ── Lock mechanism ───────────────────────────────────────────────────────────


class TestLockMechanism:
    """Concurrency lock (§5.2)."""

    def test_acquire_and_release(self, tmp_path: Path):
        """T9: Lock acquire/release basics."""
        lock_path = str(tmp_path / "test.lock")
        assert acquire_lock(lock_path) is True
        assert acquire_lock(lock_path) is False  # Second acquire fails
        release_lock(lock_path)
        assert acquire_lock(lock_path) is True  # Can acquire again
        release_lock(lock_path)

    def test_stale_lock_force_released(self, tmp_path: Path):
        """T10: Lock older than 1 hour is force-released."""
        lock_path = str(tmp_path / "test.lock")
        # Create a stale lock
        with open(lock_path, "w") as f:
            json.dump({"pid": 99999, "started_at": "2026-01-01T00:00:00"}, f)
        # Set mtime to 2 hours ago
        old_time = time.time() - 7200
        os.utime(lock_path, (old_time, old_time))

        assert acquire_lock(lock_path) is True  # Stale lock is released
        release_lock(lock_path)


# ── deduplicate_records (main function) ──────────────────────────────────────


class TestDeduplicateRecords:
    """Main dedup function (§4.1 flow)."""

    def test_t1_new_record_passes_through(self, tmp_path: Path):
        """T1: New record → registered, cache updated."""
        client = _mock_client_no_duplicates()
        cache = DedupCache(cache_path=str(tmp_path / "cache.json"))
        rec = _make_record(source_id="SRC-NEW")
        props = _make_notion_props(rec)

        result = deduplicate_records(
            records=[(rec, props)],
            client=client,
            database_id="db-id",
            cache=cache,
        )

        assert len(result.new) == 1
        assert len(result.duplicates) == 0

    def test_t2_cache_hit_skips(self, tmp_path: Path):
        """T2: Cache hit → skip, no Notion API call."""
        client = _mock_client_no_duplicates()
        cache = DedupCache(cache_path=str(tmp_path / "cache.json"))
        cache.add("sid:SRC-EXISTING", "page-existing")

        rec = _make_record(source_id="SRC-EXISTING")
        props = _make_notion_props(rec)

        result = deduplicate_records(
            records=[(rec, props)],
            client=client,
            database_id="db-id",
            cache=cache,
        )

        assert len(result.new) == 0
        assert len(result.duplicates) == 1
        client.query_database.assert_not_called()

    def test_t3_notion_hit_skips_and_caches(self, tmp_path: Path):
        """T3: Cache miss, Notion hit → skip, add to cache."""
        client = _mock_client_with_duplicate("page-notion-123")
        cache = DedupCache(cache_path=str(tmp_path / "cache.json"))

        rec = _make_record(source_id="SRC-NOTION")
        props = _make_notion_props(rec)

        result = deduplicate_records(
            records=[(rec, props)],
            client=client,
            database_id="db-id",
            cache=cache,
        )

        assert len(result.new) == 0
        assert len(result.duplicates) == 1
        assert cache.contains("sid:SRC-NOTION") is True

    def test_t4_fallback_hash_key(self, tmp_path: Path):
        """T4: No source_id → fallback hash key used for dedup."""
        client = _mock_client_no_duplicates()
        cache = DedupCache(cache_path=str(tmp_path / "cache.json"))

        rec = _make_record(source_id="")
        # source_id is empty, but V2 in gws_transform would skip this.
        # For dedup, we test the key generation fallback directly.
        rec["request_title"] = "テスト要望"
        rec["received_at"] = "2026-01-15"
        props = _make_notion_props(rec)

        result = deduplicate_records(
            records=[(rec, props)],
            client=client,
            database_id="db-id",
            cache=cache,
        )

        assert len(result.new) == 1
        # Verify hash key was used
        client.query_database.assert_called_once()
        call_filter = client.query_database.call_args[1]["filter"]
        assert call_filter["property"] == "冪等性キー"
        assert call_filter["rich_text"]["equals"].startswith("hash:")

    def test_t5_mixed_batch(self, tmp_path: Path):
        """T5: 3 new + 2 duplicate → 3 new, 2 skipped."""
        cache = DedupCache(cache_path=str(tmp_path / "cache.json"))
        cache.add("sid:DUP-001", "page-d1")
        cache.add("sid:DUP-002", "page-d2")

        client = _mock_client_no_duplicates()

        records = []
        for i in range(3):
            rec = _make_record(source_id=f"NEW-{i}")
            records.append((rec, _make_notion_props(rec)))
        for sid in ("DUP-001", "DUP-002"):
            rec = _make_record(source_id=sid)
            records.append((rec, _make_notion_props(rec)))

        result = deduplicate_records(
            records=records,
            client=client,
            database_id="db-id",
            cache=cache,
        )

        assert len(result.new) == 3
        assert len(result.duplicates) == 2

    def test_t6_key_generation_impossible(self, tmp_path: Path):
        """T6: No source_id, no title → warning, register anyway."""
        client = _mock_client_no_duplicates()
        cache = DedupCache(cache_path=str(tmp_path / "cache.json"))

        rec = _make_record(source_id="", request_title="", received_at="")
        props = _make_notion_props(rec)

        result = deduplicate_records(
            records=[(rec, props)],
            client=client,
            database_id="db-id",
            cache=cache,
        )

        # Cannot generate key → pass through (accept duplicate risk)
        assert len(result.new) == 1
        assert len(result.no_key) == 1

    def test_t8_notion_api_failure_passes_through(self, tmp_path: Path):
        """T8: Notion API error → pass through with warning."""
        client = MagicMock()
        client.query_database.side_effect = Exception("API error 500")
        cache = DedupCache(cache_path=str(tmp_path / "cache.json"))

        rec = _make_record(source_id="SRC-ERR")
        props = _make_notion_props(rec)

        result = deduplicate_records(
            records=[(rec, props)],
            client=client,
            database_id="db-id",
            cache=cache,
        )

        # API failure → register anyway (duplicate risk accepted)
        assert len(result.new) == 1
        assert len(result.api_errors) == 1

    def test_t12_all_cache_hits_no_api_calls(self, tmp_path: Path):
        """T12: All records in cache → zero Notion API calls."""
        cache = DedupCache(cache_path=str(tmp_path / "cache.json"))
        records = []
        for i in range(20):
            sid = f"CACHED-{i:03d}"
            cache.add(f"sid:{sid}", f"page-{i}")
            rec = _make_record(source_id=sid)
            records.append((rec, _make_notion_props(rec)))

        client = _mock_client_no_duplicates()

        result = deduplicate_records(
            records=records,
            client=client,
            database_id="db-id",
            cache=cache,
        )

        assert len(result.new) == 0
        assert len(result.duplicates) == 20
        client.query_database.assert_not_called()

    def test_t13_all_cache_misses_api_calls(self, tmp_path: Path):
        """T13: No cache hits → Notion API called for each."""
        client = _mock_client_no_duplicates()
        cache = DedupCache(cache_path=str(tmp_path / "cache.json"))

        records = []
        for i in range(5):
            rec = _make_record(source_id=f"MISS-{i:03d}")
            records.append((rec, _make_notion_props(rec)))

        result = deduplicate_records(
            records=records,
            client=client,
            database_id="db-id",
            cache=cache,
        )

        assert len(result.new) == 5
        assert client.query_database.call_count == 5
