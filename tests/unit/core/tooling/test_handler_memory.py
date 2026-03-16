"""Unit tests for core/tooling/handler_memory.py — search_memory formatting."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock


from core.tooling.handler_memory import MemoryToolsMixin


# ── Fixtures ─────────────────────────────────────────────


class _FakeHandler(MemoryToolsMixin):
    """Minimal stub satisfying MemoryToolsMixin attribute needs."""

    def __init__(self, search_results: list[tuple[str, str]]) -> None:
        self._memory = MagicMock()
        self._memory.search_memory_text.return_value = search_results
        self._activity = MagicMock()
        self._anima_name = "test"
        self._read_paths: set[str] = set()


# ── Tests ────────────────────────────────────────────────


class TestSearchMemoryCountHeader:
    def test_header_shows_total_and_shown_count(self) -> None:
        results = [(f"file{i}.md", f"line {i}") for i in range(25)]
        handler = _FakeHandler(results)

        output = handler._handle_search_memory({"query": "test", "scope": "all"})

        assert output.startswith("Found 25 results (showing top 10):")
        assert output.count("\n- ") == 10

    def test_header_when_fewer_than_10(self) -> None:
        results = [("a.md", "alpha"), ("b.md", "beta")]
        handler = _FakeHandler(results)

        output = handler._handle_search_memory({"query": "test"})

        assert output.startswith("Found 2 results (showing top 2):")
        assert "alpha" in output
        assert "beta" in output

    def test_no_results_message(self) -> None:
        handler = _FakeHandler([])

        output = handler._handle_search_memory({"query": "missing"})

        assert output == "No results for 'missing'"
