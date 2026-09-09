from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Language-aware prompt token estimation and truncation.

Coefficients are calibrated from Claude Code CLI measurements taken on
2026-09-07 across Japanese, ASCII, and mixed AnimaWorks prompts.
"""

import re
from typing import Literal

CJK_TOKENS_PER_CHAR = 1.14
OTHER_TOKENS_PER_CHAR = 0.31

_DEFAULT_CJK_RATIO = 0.4
_OMISSION_MARKER = "..."
_CJK_RE = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef]")


def _cjk_char_count(text: str) -> int:
    return len(_CJK_RE.findall(text))


def estimate_tokens(text: str) -> int:
    """Estimate tokens using calibrated CJK and non-CJK character weights."""
    if not text:
        return 0
    cjk_chars = _cjk_char_count(text)
    other_chars = len(text) - cjk_chars
    return int(cjk_chars * CJK_TOKENS_PER_CHAR + other_chars * OTHER_TOKENS_PER_CHAR)


def _truncate_chars(text: str, max_tokens: int, *, keep: Literal["head", "tail"]) -> str:
    """Find the longest marked character slice that fits ``max_tokens``."""
    low = 0
    high = len(text)
    while low < high:
        size = (low + high + 1) // 2
        sliced = text[:size] if keep == "head" else text[-size:]
        candidate = sliced + _OMISSION_MARKER if keep == "head" else _OMISSION_MARKER + sliced
        if estimate_tokens(candidate) <= max_tokens:
            low = size
        else:
            high = size - 1

    if low == 0:
        return _OMISSION_MARKER if estimate_tokens(_OMISSION_MARKER) <= max_tokens else ""
    sliced = text[:low] if keep == "head" else text[-low:]
    return sliced + _OMISSION_MARKER if keep == "head" else _OMISSION_MARKER + sliced


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    *,
    keep: Literal["head", "tail"] = "head",
) -> str:
    """Truncate text to a token budget, preferring complete retained lines."""
    if keep not in ("head", "tail"):
        raise ValueError("keep must be 'head' or 'tail'")
    if not text or max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    lines = text.splitlines(keepends=True)
    if len(lines) <= 1:
        return _truncate_chars(text, max_tokens, keep=keep)

    if keep == "head":
        retained = ""
        for line in lines[:-1]:
            candidate = retained + line + _OMISSION_MARKER
            if estimate_tokens(candidate) > max_tokens:
                break
            retained += line
        if retained:
            return retained + _OMISSION_MARKER
        return _truncate_chars(lines[0], max_tokens, keep=keep)

    retained = ""
    for line in reversed(lines[1:]):
        candidate = _OMISSION_MARKER + line + retained
        if estimate_tokens(candidate) > max_tokens:
            break
        retained = line + retained
    if retained:
        return _OMISSION_MARKER + retained
    return _truncate_chars(lines[-1], max_tokens, keep=keep)


def tokens_to_chars_hint(max_tokens: int, sample: str = "") -> int:
    """Convert a token budget to a character limit using a sample's CJK ratio."""
    if max_tokens <= 0:
        return 0
    cjk_ratio = _cjk_char_count(sample) / len(sample) if sample else _DEFAULT_CJK_RATIO
    tokens_per_char = cjk_ratio * CJK_TOKENS_PER_CHAR + (1 - cjk_ratio) * OTHER_TOKENS_PER_CHAR
    return int(max_tokens / tokens_per_char)
