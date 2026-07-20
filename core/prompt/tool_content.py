from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""File-backed tool descriptions and usage guides."""

from typing import Any

from core.i18n import t
from core.paths import load_prompt_text


def get_default_guide(key: str, locale: str | None = None) -> str:
    """Return the last-resort guide used when no Markdown template exists."""
    if key == "s_builtin":
        return ""
    fallback_key = f"tool_guide.{key}"
    fallback = t(fallback_key, locale=locale)
    return "" if fallback == fallback_key else fallback


def load_guide(key: str, locale: str | None = None) -> str:
    """Load a tool guide from Markdown on every call."""
    try:
        return load_prompt_text(f"tool_guides/{key}", locale=locale)
    except FileNotFoundError:
        return get_default_guide(key, locale)


def apply_prompt_descriptions(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overlay canonical tool descriptions with locale-specific Markdown."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        try:
            description = load_prompt_text(f"tool_descriptions/{tool['name']}")
        except FileNotFoundError:
            result.append(tool)
            continue
        result.append({**tool, "description": description.strip()})
    return result
