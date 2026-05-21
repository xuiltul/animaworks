from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime gates for human-driven trusted skill promotion."""

from core.execution._sanitize import ORIGIN_HUMAN

_HUMAN_TRUST_TRIGGERS: frozenset[str] = frozenset({"", "manual", "chat", "greet:user"})


def trust_skill_enabled_for_trigger(trigger: str | None) -> bool:
    normalized = (trigger or "").strip()
    return normalized.startswith("message:") or normalized in _HUMAN_TRUST_TRIGGERS


def trust_skill_enabled_for_context(trigger: str | None, origin: str | None = "") -> bool:
    normalized_origin = (origin or "").strip()
    if normalized_origin:
        return normalized_origin == ORIGIN_HUMAN
    return trust_skill_enabled_for_trigger(trigger)
