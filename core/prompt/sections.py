from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Section loading helpers for prompt building."""

from core.paths import load_prompt


def _parse_kv_template(raw: str) -> dict[str, str]:
    """Parse ``[key]: value`` lines from a template string.

    Only the first ``]: `` occurrence per line is used as delimiter,
    so values may safely contain ``]: ``.
    """
    result: dict[str, str] = {}
    for line in raw.strip().splitlines():
        if not line.startswith("["):
            continue
        bracket_end = line.find("]")
        if bracket_end < 0:
            continue
        sep = line.find("]: ", bracket_end)
        if sep < 0:
            continue
        key = line[1:bracket_end]
        value = line[sep + 3 :]
        result[key] = value
    return result


def _load_section_strings(locale: str | None = None) -> dict[str, str]:
    """Load section headers and labels from template."""
    try:
        raw = load_prompt("builder/sections", locale=locale)
    except FileNotFoundError:
        return {}
    return _parse_kv_template(raw)


def _load_fallback_strings(locale: str | None = None) -> dict[str, str]:
    """Load fallback/placeholder texts from template."""
    try:
        raw = load_prompt("builder/fallbacks", locale=locale)
    except FileNotFoundError:
        return {}
    return _parse_kv_template(raw)
