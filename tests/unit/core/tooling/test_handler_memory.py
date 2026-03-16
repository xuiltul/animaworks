"""Unit tests for core/tooling/handler_memory.py — search_memory formatting."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

from core.tooling.handler_memory import MemoryToolsMixin


# ── Fixtures ─────────────────────────────────────────────


def _make_result(source: str, content: str, score: float = 0.9) -> dict:
    return {
        "source_file": source,
        "content": content,
        "score": score,
        "chunk_index": 0,
        "total_chunks": 1,
        "search_method": "vector",
    }


class _FakeHandler(MemoryToolsMixin):
    """Minimal stub satisfying MemoryToolsMixin attribute needs."""

    def __init__(self, search_results: list[dict]) -> None:
        self._memory = MagicMock()
        self._memory.search_memory_text.return_value = search_results
        self._activity = MagicMock()
        self._anima_name = "test"
        self._read_paths: set[str] = set()


# ── Tests ────────────────────────────────────────────────


class TestSearchMemoryCountHeader:
    def test_header_shows_total_and_shown_count(self) -> None:
        results = [_make_result(f"file{i}.md", f"line {i}") for i in range(25)]
        handler = _FakeHandler(results)

        output = handler._handle_search_memory({"query": "test", "scope": "all"})

        assert 'Search results for "test"' in output
        assert "vector" in output
        assert "1-25" in output
        assert "score=" in output
        assert "file0.md" in output
        assert "Use offset=25 to see next page" in output

    def test_header_when_fewer_than_10(self) -> None:
        results = [
            _make_result("a.md", "alpha"),
            _make_result("b.md", "beta"),
        ]
        handler = _FakeHandler(results)

        output = handler._handle_search_memory({"query": "test"})

        assert 'Search results for "test"' in output
        assert "1-2" in output
        assert "alpha" in output
        assert "beta" in output

    def test_no_results_message(self) -> None:
        handler = _FakeHandler([])

        output = handler._handle_search_memory({"query": "missing"})

        assert output == "No results for 'missing'"

    def test_no_more_results_at_offset(self) -> None:
        handler = _FakeHandler([])

        output = handler._handle_search_memory({"query": "x", "offset": 10})

        assert output == "No more results for 'x' at offset=10."
