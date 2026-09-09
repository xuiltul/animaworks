from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Item-level memory candidates used by the priming pipeline."""

from collections.abc import Iterable
from dataclasses import dataclass

from core.prompt.tokens import estimate_tokens


@dataclass(frozen=True)
class MemoryItem:
    """One indivisible block of priming memory."""

    source: str
    key: str
    text: str
    ref: str = ""
    updated: str = ""
    rank: float = 0.0


class ItemizedMemory(str):
    """Backward-compatible string result carrying its source items."""

    items: tuple[MemoryItem, ...]

    def __new__(cls, text: str, items: Iterable[MemoryItem] = ()) -> ItemizedMemory:
        value = super().__new__(cls, text)
        value.items = tuple(items)
        return value


def render_items(items: Iterable[MemoryItem], header: str) -> str:
    """Render item blocks below an optional channel header."""
    texts = [item.text for item in items if item.text]
    if not texts:
        return ""
    body = "\n".join(texts)
    return f"{header}\n{body}" if header else body


def select_within_budget(items: Iterable[MemoryItem], max_tokens: int) -> list[MemoryItem]:
    """Select whole items by rank and freshness without splitting a block."""
    if max_tokens <= 0:
        return []
    ordered = sorted(items, key=lambda item: (item.rank, item.updated), reverse=True)
    selected: list[MemoryItem] = []
    for item in ordered:
        # A partial memory can lose the context that makes it safe or useful.
        candidate = "\n".join([*(selected_item.text for selected_item in selected), item.text])
        if estimate_tokens(candidate) > max_tokens:
            continue
        selected.append(item)
    return selected
