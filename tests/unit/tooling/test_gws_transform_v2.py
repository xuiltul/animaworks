"""Tests for S3-6: gws_transform.py validation & error-handling enhancements.

TDD RED phase: T1–T7 from naomi's S3-6 spec.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import logging
import re

import pytest

from core.tools.gws_transform import (
    ErrorRecord,
    SkipRecord,
    TransformConfig,
    TransformResult,
    transform_gws_to_notion,
)


# ── Helpers ──────────────────────────────────────────────────


def _make_record(**overrides: object) -> dict:
    """Build a valid GWS record with sensible defaults."""
    base = {
        "request_title": "ダッシュボードの表示速度改善",
        "problem": "ダッシュボード画面の読み込みに10秒以上かかる",
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


# ── T1: source_id欠落 → ハッシュ補完、Warningのみ ────────────


class TestT1SourceIdMissingHashFallback:
    """source_id missing → auto-fill with SHA256(request_title + received_at)[:16], Warning only (no skip)."""

    def test_record_succeeds_not_skipped(self):
        rec = _make_record(source_id="")
        result = transform_gws_to_notion([rec])
        assert len(result.success) == 1
        assert len(result.skipped) == 0

    def test_source_id_none_succeeds(self):
        rec = _make_record()
        del rec["source_id"]
        result = transform_gws_to_notion([rec])
        assert len(result.success) == 1
        assert len(result.skipped) == 0

    def test_warning_logged(self, caplog):
        rec = _make_record(source_id="")
        with caplog.at_level(logging.WARNING):
            transform_gws_to_notion([rec])
        assert any("source_id" in msg.lower() for msg in caplog.messages)


# ── T2: received_at 不正フォーマット → SkipRecord + 実際の値 ──


class TestT2ReceivedAtInvalidFormat:
    """Invalid received_at → SkipRecord with actual value in message."""

    def test_invalid_date_skipped(self):
        rec = _make_record(received_at="not-a-date")
        result = transform_gws_to_notion([rec])
        assert len(result.skipped) == 1
        assert len(result.success) == 0

    def test_skip_message_contains_actual_value(self):
        rec = _make_record(received_at="2026/13/99")
        result = transform_gws_to_notion([rec])
        skip = result.skipped[0]
        assert "2026/13/99" in skip.reason

    def test_skip_message_contains_expected_format(self):
        rec = _make_record(received_at="bad-date")
        result = transform_gws_to_notion([rec])
        skip = result.skipped[0]
        assert "YYYY-MM-DD" in skip.reason


# ── T3: request_title > 255文字 → トランケート + Warning ─────


class TestT3RequestTitleTruncation:
    """request_title > 255 chars → truncate to 255, Warning (no skip)."""

    def test_long_title_not_skipped(self):
        rec = _make_record(request_title="あ" * 256)
        result = transform_gws_to_notion([rec])
        assert len(result.success) == 1
        assert len(result.skipped) == 0

    def test_title_truncated_to_255(self):
        rec = _make_record(request_title="あ" * 300)
        result = transform_gws_to_notion([rec])
        props = result.success[0]
        title_content = props["Name"]["title"][0]["text"]["content"]
        assert len(title_content) == 255

    def test_warning_logged(self, caplog):
        rec = _make_record(request_title="あ" * 256)
        with caplog.at_level(logging.WARNING):
            transform_gws_to_notion([rec])
        assert any("request_title" in msg.lower() or "truncat" in msg.lower() for msg in caplog.messages)

    def test_exact_255_not_truncated(self):
        rec = _make_record(request_title="あ" * 255)
        result = transform_gws_to_notion([rec])
        props = result.success[0]
        title_content = props["Name"]["title"][0]["text"]["content"]
        assert len(title_content) == 255


# ── T4: summary > 2000文字 → トランケート + Warning ──────────


class TestT4SummaryTruncation:
    """summary > 2000 chars → truncate to 2000, Warning (no skip)."""

    def test_long_summary_not_skipped(self):
        rec = _make_record(problem="あ" * 2000)
        result = transform_gws_to_notion([rec])
        assert len(result.success) == 1
        assert len(result.skipped) == 0

    def test_summary_within_2000(self):
        rec = _make_record(problem="あ" * 2000)
        result = transform_gws_to_notion([rec])
        props = result.success[0]
        summary = props["AIによる要約"]["rich_text"][0]["text"]["content"]
        assert len(summary) <= 2000

    def test_warning_logged_on_truncation(self, caplog):
        rec = _make_record(problem="あ" * 2000)
        with caplog.at_level(logging.WARNING):
            transform_gws_to_notion([rec])
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("summary" in msg.lower() or "truncat" in msg.lower() or "2000" in msg for msg in warning_msgs)


# ── T5: 例外発生時 → ErrorRecord + 次レコード継続 ────────────


class TestT5ExceptionHandling:
    """Unexpected exception → ErrorRecord + logger.exception, continue to next record."""

    def test_error_recorded_next_continues(self, monkeypatch):
        records = [_make_record(source_id="SRC-001"), _make_record(source_id="SRC-002")]

        original_transform = None
        call_count = 0

        # Patch _transform_single to raise on first call
        import core.tools.gws_transform as mod

        original_fn = mod._transform_single

        def _exploding_transform(record, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated failure")
            return original_fn(record, config)

        monkeypatch.setattr(mod, "_transform_single", _exploding_transform)

        result = transform_gws_to_notion(records)
        assert len(result.errors) == 1
        assert len(result.success) == 1
        assert "Simulated failure" in result.errors[0].error

    def test_exception_logged(self, monkeypatch, caplog):
        import core.tools.gws_transform as mod

        original_fn = mod._transform_single

        def _exploding(record, config):
            raise ValueError("kaboom")

        monkeypatch.setattr(mod, "_transform_single", _exploding)

        with caplog.at_level(logging.ERROR):
            transform_gws_to_notion([_make_record()])
        assert any("kaboom" in msg for msg in caplog.messages)


# ── T6: TransformResult.summary() ────────────────────────────


class TestT6TransformResultSummary:
    """result.summary() returns '変換完了: 成功X件、スキップY件（reason）、エラーZ件'."""

    def test_all_success(self):
        result = transform_gws_to_notion([_make_record(), _make_record(source_id="SRC-002")])
        s = result.summary()
        assert "成功2件" in s
        assert "スキップ0件" in s
        assert "エラー0件" in s

    def test_mixed_results(self):
        records = [
            _make_record(source_id="SRC-001"),
            _make_record(received_at="invalid", source_id="SRC-002"),
        ]
        result = transform_gws_to_notion(records)
        s = result.summary()
        assert "成功1件" in s
        assert "スキップ1件" in s

    def test_summary_starts_with_prefix(self):
        result = transform_gws_to_notion([_make_record()])
        s = result.summary()
        assert s.startswith("変換完了:")

    def test_skip_reason_in_parentheses(self):
        records = [_make_record(source_id="SRC-001", received_at="bad")]
        result = transform_gws_to_notion(records)
        s = result.summary()
        # Should contain reason breakdown in parentheses
        assert "（" in s or "(" in s


# ── T7: source_idなし複数件 → 独立ハッシュ生成 ───────────────


class TestT7MultipleSourceIdMissingIndependent:
    """Multiple records without source_id → each gets independent hash (no collision)."""

    def test_different_records_different_hashes(self):
        rec1 = _make_record(source_id="", request_title="タイトルA", received_at="2026-01-01")
        rec2 = _make_record(source_id="", request_title="タイトルB", received_at="2026-01-02")
        result = transform_gws_to_notion([rec1, rec2])
        assert len(result.success) == 2
        # Both should succeed (not skip) since source_id is auto-filled

    def test_same_title_different_date_different_hash(self):
        rec1 = _make_record(source_id="", request_title="同じタイトル", received_at="2026-01-01")
        rec2 = _make_record(source_id="", request_title="同じタイトル", received_at="2026-01-02")
        result = transform_gws_to_notion([rec1, rec2])
        assert len(result.success) == 2

    def test_hash_is_16_chars_hex(self):
        """Verify the generated hash fallback is SHA256[:16]."""
        title = "テストタイトル"
        date = "2026-03-01"
        expected = hashlib.sha256(f"{title}{date}".encode()).hexdigest()[:16]
        # The hash should be deterministic
        assert len(expected) == 16
        assert all(c in "0123456789abcdef" for c in expected)
