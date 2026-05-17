from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for procedure-to-skill promotion."""

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from core.skills.guard import SCANNER_VERSION
from core.skills.models import ScanResult, SkillScanVerdict


def safe_skill_name(raw: Any) -> str:
    name = str(raw or "").strip().lower()
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"[^a-z0-9_.-]+", "-", name).strip("-._")
    if not name:
        raise ValueError("skill_name must contain at least one ASCII letter or digit")
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"Invalid skill name: {raw}")
    return name


def as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def normalise_risk(raw: Any) -> dict[str, bool]:
    if not isinstance(raw, dict):
        raw = {}
    risk = {
        "read_only": bool(raw.get("read_only", False)),
        "destructive": bool(raw.get("destructive", False)),
        "external_send": bool(raw.get("external_send", False)),
        "handles_untrusted_data": bool(raw.get("handles_untrusted_data", False)),
        "open_world": bool(raw.get("open_world", False)),
        "requires_human_approval": bool(raw.get("requires_human_approval", False)),
    }
    if risk["destructive"] or risk["external_send"] or risk["open_world"]:
        risk["requires_human_approval"] = True
    return risk


def runtime_approval_required(risk: dict[str, Any], scan_result: ScanResult) -> bool:
    return bool(
        risk.get("requires_human_approval")
        or risk.get("destructive")
        or risk.get("external_send")
        or risk.get("open_world")
        or scan_result.verdict in {SkillScanVerdict.caution, SkillScanVerdict.warn}
    )


def scan_security_metadata(scan_result: ScanResult) -> dict[str, Any]:
    return {
        "verdict": scan_result.verdict.value,
        "scan_status": "scanned",
        "findings": [f.model_dump(mode="json") for f in scan_result.findings],
        "scanned_at": datetime.now(UTC).isoformat(),
        "scanner_version": SCANNER_VERSION,
    }


def within_days(value: str | None, days: int) -> bool:
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    cutoff = datetime.now(UTC) - timedelta(days=days)
    return parsed >= cutoff
