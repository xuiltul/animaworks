"""Tests for core.tools.gws_transform — GWS→Notion transform module.

TDD RED phase: T1-T10 from s3-1_input_spec.md §7.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json

import pytest

from core.tools.gws_transform import (
    ErrorRecord,
    SkipRecord,
    TransformConfig,
    TransformResult,
    transform_gws_to_notion,
)


# ── Fixtures ─────────────────────────────────────────────────


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


# ── T1: 全フィールド正常 ──────────────────────────────────────


class TestT1FullFieldsNormal:
    """T1: All 17 fields have valid values → 5 properties correctly mapped."""

    def test_success_count(self):
        result = transform_gws_to_notion([_make_record()])
        assert len(result.success) == 1
        assert len(result.skipped) == 0
        assert len(result.errors) == 0

    def test_name_property(self):
        result = transform_gws_to_notion([_make_record()])
        props = result.success[0]
        assert props["Name"]["title"][0]["text"]["content"] == "ダッシュボードの表示速度改善"

    def test_status_property(self):
        result = transform_gws_to_notion([_make_record()])
        props = result.success[0]
        assert props["Status"]["select"]["name"] == "新規"

    def test_label_property(self):
        result = transform_gws_to_notion([_make_record()])
        props = result.success[0]
        assert props["Label"]["multi_select"] == [{"name": "AI"}]

    def test_summary_property_contains_required_sections(self):
        result = transform_gws_to_notion([_make_record()])
        props = result.success[0]
        summary = props["AIによる要約"]["rich_text"][0]["text"]["content"]
        assert "【課題】" in summary
        assert "【期待】" in summary
        assert "【緊急度】高" in summary
        assert "【顧客】ABC株式会社" in summary
        assert "【根拠】複数顧客から同一報告あり" in summary
        assert "【ソース】メール" in summary
        assert "【リンク】https://support.example.com/ticket/123" in summary

    def test_source_url_property(self):
        result = transform_gws_to_notion([_make_record()])
        props = result.success[0]
        assert props["発生元"]["url"] == (
            "https://docs.google.com/spreadsheets/d/xxx/edit#gid=0&range=A15"
        )


# ── T2: 任意フィールド省略 ────────────────────────────────────


class TestT2OptionalFieldsOmitted:
    """T2: Optional fields (urgency, customer, etc.) are empty → summary omits those lines."""

    def test_empty_urgency_omitted_from_summary(self):
        rec = _make_record(urgency="", customer="", evidence_quote="", source_type="", source_link="")
        result = transform_gws_to_notion([rec])
        props = result.success[0]
        summary = props["AIによる要約"]["rich_text"][0]["text"]["content"]
        assert "【緊急度】" not in summary
        assert "【顧客】" not in summary
        assert "【根拠】" not in summary
        assert "【ソース】" not in summary
        assert "【リンク】" not in summary

    def test_required_sections_still_present(self):
        rec = _make_record(urgency="", customer="", evidence_quote="", source_type="", source_link="")
        result = transform_gws_to_notion([rec])
        props = result.success[0]
        summary = props["AIによる要約"]["rich_text"][0]["text"]["content"]
        assert "【課題】" in summary
        assert "【期待】" in summary

    def test_problem_empty_uses_placeholder(self):
        rec = _make_record(problem="", desired_outcome="")
        result = transform_gws_to_notion([rec])
        props = result.success[0]
        summary = props["AIによる要約"]["rich_text"][0]["text"]["content"]
        assert "（未記載）" in summary


# ── T3: 要約2000文字ギリギリ ──────────────────────────────────


class TestT3SummaryTruncation:
    """T3: Long problem/desired_outcome → summary truncated to 2000 chars."""

    def test_summary_truncated_at_2000(self):
        long_text = "あ" * 3000
        rec = _make_record(problem=long_text, desired_outcome=long_text)
        result = transform_gws_to_notion([rec])
        props = result.success[0]
        summary = props["AIによる要約"]["rich_text"][0]["text"]["content"]
        assert len(summary) <= 2000
        assert summary.endswith("...")

    def test_summary_exactly_2000_not_truncated(self):
        """If summary is exactly 2000 chars, no truncation."""
        # Build a record that produces exactly <= 2000 char summary
        rec = _make_record(urgency="", customer="", evidence_quote="", source_type="", source_link="")
        result = transform_gws_to_notion([rec])
        props = result.success[0]
        summary = props["AIによる要約"]["rich_text"][0]["text"]["content"]
        assert len(summary) <= 2000
        assert not summary.endswith("...")


# ── T4: バッチ20件 ────────────────────────────────────────────


class TestT4Batch20:
    """T4: 20 valid records → all 20 converted successfully."""

    def test_all_20_converted(self):
        records = [_make_record(source_id=f"SRC-{i:03d}") for i in range(20)]
        result = transform_gws_to_notion(records)
        assert len(result.success) == 20
        assert len(result.skipped) == 0
        assert len(result.errors) == 0


# ── T5: request_title空 ──────────────────────────────────────


class TestT5EmptyRequestTitle:
    """T5: Empty request_title → skip that record, process others."""

    def test_skips_empty_title(self):
        records = [
            _make_record(source_id="SRC-001"),
            _make_record(request_title="", source_id="SRC-002"),
            _make_record(source_id="SRC-003"),
        ]
        result = transform_gws_to_notion(records)
        assert len(result.success) == 2
        assert len(result.skipped) == 1
        assert result.skipped[0].index == 1
        assert result.skipped[0].source_id == "SRC-002"

    def test_skips_null_title(self):
        records = [_make_record(request_title=None, source_id="SRC-NULL")]
        result = transform_gws_to_notion(records)
        assert len(result.success) == 0
        assert len(result.skipped) == 1


# ── T6: source_id欠落 ────────────────────────────────────────


class TestT6MissingSourceId:
    """T6: source_id field missing → auto-fill with hash, Warning only (S3-6)."""

    def test_missing_source_id_succeeds_with_hash(self):
        rec = _make_record()
        del rec["source_id"]
        result = transform_gws_to_notion([rec])
        assert len(result.success) == 1
        assert len(result.skipped) == 0

    def test_empty_source_id_succeeds_with_hash(self):
        rec = _make_record(source_id="")
        result = transform_gws_to_notion([rec])
        assert len(result.success) == 1
        assert len(result.skipped) == 0


# ── T7: 入力が配列でない ─────────────────────────────────────


class TestT7NonArrayInput:
    """T7: Input is not an array → entire batch errors."""

    def test_dict_input_raises_error(self):
        result = transform_gws_to_notion({"single": "object"})  # type: ignore[arg-type]
        assert len(result.success) == 0
        assert len(result.errors) == 1

    def test_string_input_raises_error(self):
        result = transform_gws_to_notion("not a list")  # type: ignore[arg-type]
        assert len(result.success) == 0
        assert len(result.errors) == 1


# ── T8: 不正日付 ──────────────────────────────────────────────


class TestT8InvalidDate:
    """T8: Invalid received_at → skip that record."""

    def test_skips_invalid_date(self):
        rec = _make_record(received_at="invalid")
        result = transform_gws_to_notion([rec])
        assert len(result.success) == 0
        assert len(result.skipped) == 1
        assert "received_at" in result.skipped[0].reason.lower() or "date" in result.skipped[0].reason.lower()

    def test_accepts_iso8601(self):
        rec = _make_record(received_at="2026-01-15T10:30:00+09:00")
        result = transform_gws_to_notion([rec])
        assert len(result.success) == 1

    def test_accepts_gas_date_format(self):
        """GAS Date.toString() outputs like 'Mon Jan 15 2026 10:30:00 GMT+0900'."""
        rec = _make_record(received_at="2026-01-15")
        result = transform_gws_to_notion([rec])
        assert len(result.success) == 1


# ── T9: 空配列 ────────────────────────────────────────────────


class TestT9EmptyArray:
    """T9: Empty array → empty TransformResult with no errors."""

    def test_empty_input(self):
        result = transform_gws_to_notion([])
        assert len(result.success) == 0
        assert len(result.skipped) == 0
        assert len(result.errors) == 0

    def test_result_type(self):
        result = transform_gws_to_notion([])
        assert isinstance(result, TransformResult)


# ── T10: 混在バッチ ───────────────────────────────────────────


class TestT10MixedBatch:
    """T10: 3 valid + 1 invalid + 1 hash-filled → 4 success + 1 skipped (S3-6)."""

    def test_mixed_batch(self):
        records = [
            _make_record(source_id="SRC-001"),  # valid
            _make_record(request_title="", source_id="SRC-002"),  # V1 fail
            _make_record(source_id="SRC-003"),  # valid
            _make_record(source_id="", request_title="test"),  # S3-6: hash-filled, success
            _make_record(source_id="SRC-005"),  # valid
        ]
        result = transform_gws_to_notion(records)
        assert len(result.success) == 4
        assert len(result.skipped) == 1

    def test_skip_indices_correct(self):
        records = [
            _make_record(source_id="SRC-001"),
            _make_record(request_title="", source_id="SRC-002"),
            _make_record(source_id="SRC-003"),
            _make_record(source_id="", request_title="test"),  # S3-6: succeeds
            _make_record(source_id="SRC-005"),
        ]
        result = transform_gws_to_notion(records)
        skip_indices = [s.index for s in result.skipped]
        assert 1 in skip_indices
        assert 3 not in skip_indices


# ── V3: source_row_url空 → 発生元プロパティ省略 ──────────────


class TestV3EmptySourceRowUrl:
    """V3: Empty source_row_url → 発生元 property omitted, record still succeeds."""

    def test_no_source_url_property(self):
        rec = _make_record(source_row_url="")
        result = transform_gws_to_notion([rec])
        assert len(result.success) == 1
        assert "発生元" not in result.success[0]

    def test_null_source_url_property(self):
        rec = _make_record(source_row_url=None)
        result = transform_gws_to_notion([rec])
        assert len(result.success) == 1
        assert "発生元" not in result.success[0]


# ── V6: 配列要素がobjectでない ────────────────────────────────


class TestV6NonObjectElement:
    """V6: Array element is not an object → skip that element."""

    def test_skips_non_object(self):
        records = [_make_record(source_id="SRC-001"), "not-an-object", 42]
        result = transform_gws_to_notion(records)  # type: ignore[arg-type]
        assert len(result.success) == 1
        assert len(result.skipped) == 2


# ── V9: Status値が「新規」以外 → 強制上書き ──────────────────


class TestV9StatusForced:
    """V9: Status is always forced to '新規'."""

    def test_config_default_status(self):
        config = TransformConfig(default_status="新規")
        result = transform_gws_to_notion([_make_record()], config=config)
        props = result.success[0]
        assert props["Status"]["select"]["name"] == "新規"
