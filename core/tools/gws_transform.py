# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""GWS output → Notion property transform module.

Converts GAS `extractNewDevRequests_()` JSON records into Notion API
property format compatible with the `notion_create` tool (Sprint 2).

Phase 1 maps 5 fields: Name, Status, Label, AIによる要約, 発生元.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("animaworks.tools")

# ── Data classes ─────────────────────────────────────────────


@dataclass
class TransformConfig:
    """Configuration for the GWS→Notion transform."""

    database_id: str = "3c7c44cc-2f4a-4b16-8ebb-afe0c78bb43b"
    default_status: str = "新規"
    default_labels: list[str] = field(default_factory=lambda: ["AI"])
    summary_max_chars: int = 2000


@dataclass
class SkipRecord:
    """A record that was skipped due to validation failure."""

    index: int
    source_id: str
    reason: str


@dataclass
class ErrorRecord:
    """A record that caused an unexpected error."""

    index: int
    source_id: str
    error: str


@dataclass
class TransformResult:
    """Result of a batch transform operation."""

    success: list[dict] = field(default_factory=list)
    skipped: list[SkipRecord] = field(default_factory=list)
    errors: list[ErrorRecord] = field(default_factory=list)


# ── Summary template ─────────────────────────────────────────

_SUMMARY_OPTIONAL_LINES: list[tuple[str, str]] = [
    ("urgency", "【緊急度】"),
    ("customer", "【顧客】"),
    ("evidence_quote", "【根拠】"),
    ("source_type", "【ソース】"),
    ("source_link", "【リンク】"),
]


def _build_summary(record: dict, max_chars: int) -> str:
    """Build the AIによる要約 text from a GWS record.

    Template (§3.2):
        【課題】{problem} / 【期待】{desired_outcome}
        【緊急度】{urgency}       ← omit if empty
        【顧客】{customer}        ← omit if empty
        【根拠】{evidence_quote}  ← omit if empty
        【ソース】{source_type}   ← omit if empty
        【リンク】{source_link}   ← omit if empty
    """
    problem = record.get("problem") or "（未記載）"
    desired_outcome = record.get("desired_outcome") or "（未記載）"

    lines = [f"【課題】{problem} / 【期待】{desired_outcome}"]

    for key, prefix in _SUMMARY_OPTIONAL_LINES:
        value = record.get(key)
        if value:
            lines.append(f"{prefix}{value}")

    text = "\n".join(lines)

    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."

    return text


# ── Date validation ──────────────────────────────────────────


def _validate_date(value: str) -> bool:
    """Check if a date string is parseable."""
    if not value:
        return False

    # Try ISO 8601 formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue

    # Try dateutil as fallback
    try:
        from dateutil.parser import parse as dateutil_parse

        dateutil_parse(value)
        return True
    except Exception:
        pass

    return False


# ── Single record transform ──────────────────────────────────


def _transform_single(
    record: dict,
    config: TransformConfig,
) -> dict[str, Any]:
    """Transform a single GWS record to Notion properties (§3.1)."""
    props: dict[str, Any] = {
        "Name": {
            "title": [{"text": {"content": record["request_title"]}}],
        },
        "Status": {
            "select": {"name": config.default_status},
        },
        "Label": {
            "multi_select": [{"name": label} for label in config.default_labels],
        },
        "AIによる要約": {
            "rich_text": [
                {
                    "text": {
                        "content": _build_summary(record, config.summary_max_chars),
                    },
                },
            ],
        },
    }

    source_row_url = record.get("source_row_url")
    if source_row_url:
        props["発生元"] = {"url": source_row_url}

    return props


# ── Main transform function ──────────────────────────────────


def transform_gws_to_notion(
    records: list[dict],
    config: TransformConfig | None = None,
) -> TransformResult:
    """Transform GWS output records to Notion API property format.

    Args:
        records: GWS output JSON array (§2.1 schema).
        config: Transform config (defaults to Phase 1 settings).

    Returns:
        TransformResult with success/skipped/errors lists.
    """
    if config is None:
        config = TransformConfig()

    result = TransformResult()

    # V5: Input must be a list
    if not isinstance(records, list):
        result.errors.append(
            ErrorRecord(index=-1, source_id="", error="Input is not an array")
        )
        return result

    for i, record in enumerate(records):
        # V6: Element must be a dict
        if not isinstance(record, dict):
            result.skipped.append(
                SkipRecord(
                    index=i,
                    source_id="",
                    reason="Element is not an object",
                )
            )
            continue

        source_id = record.get("source_id", "")

        try:
            # V1: request_title required
            request_title = record.get("request_title")
            if not request_title:
                result.skipped.append(
                    SkipRecord(
                        index=i,
                        source_id=source_id,
                        reason="request_title is empty or missing",
                    )
                )
                logger.warning(
                    "Record %d skipped: request_title empty (source_id=%s)",
                    i,
                    source_id,
                )
                continue

            # V2: source_id required
            if not source_id:
                result.skipped.append(
                    SkipRecord(
                        index=i,
                        source_id="",
                        reason="source_id is empty or missing",
                    )
                )
                logger.warning("Record %d skipped: source_id empty", i)
                continue

            # V4: received_at must be a valid date
            received_at = record.get("received_at", "")
            if not _validate_date(str(received_at)):
                result.skipped.append(
                    SkipRecord(
                        index=i,
                        source_id=source_id,
                        reason="received_at is not a valid date",
                    )
                )
                logger.warning(
                    "Record %d skipped: invalid date '%s' (source_id=%s)",
                    i,
                    received_at,
                    source_id,
                )
                continue

            # Transform
            props = _transform_single(record, config)

            # V8: Payload size check
            payload_bytes = len(json.dumps(props, ensure_ascii=False).encode("utf-8"))
            if payload_bytes > 500_000:
                result.skipped.append(
                    SkipRecord(
                        index=i,
                        source_id=source_id,
                        reason=f"Payload exceeds 500KB ({payload_bytes} bytes)",
                    )
                )
                logger.error(
                    "Record %d skipped: payload %d bytes exceeds 500KB (source_id=%s)",
                    i,
                    payload_bytes,
                    source_id,
                )
                continue

            result.success.append(props)

        except Exception as exc:
            result.errors.append(
                ErrorRecord(index=i, source_id=source_id, error=str(exc))
            )
            logger.error(
                "Record %d error: %s (source_id=%s)", i, exc, source_id
            )

    return result
