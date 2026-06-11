"""Unit tests for core/tooling/handler_memory.py — search_memory formatting."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

from core.tooling.handler_memory import MemoryToolsMixin

# ── Fixtures ─────────────────────────────────────────────


def _make_result(source: str, content: str, score: float = 0.9, **overrides) -> dict:
    result = {
        "source_file": source,
        "content": content,
        "score": score,
        "chunk_index": 0,
        "total_chunks": 1,
        "search_method": "vector",
    }
    result.update(overrides)
    return result


class _FakeHandler(MemoryToolsMixin):
    """Minimal stub satisfying MemoryToolsMixin attribute needs."""

    def __init__(self, search_results: list[dict]) -> None:
        self._memory = MagicMock()
        self._memory.search_memory_text.return_value = search_results
        self._memory.last_search_meta = {}
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

    def test_abstain_meta_in_no_results_message(self) -> None:
        handler = _FakeHandler([])
        handler._memory.last_search_meta = {
            "abstain": True,
            "abstain_reason": "low_confidence",
        }

        output = handler._handle_search_memory({"query": "missing"})

        assert "abstain=true" in output
        assert "low_confidence" in output

    def test_no_more_results_at_offset(self) -> None:
        handler = _FakeHandler([])

        output = handler._handle_search_memory({"query": "x", "offset": 10})

        assert output == "No more results for 'x' at offset=10."

    def test_knowledge_metadata_line_shows_updated_and_external_origin(self) -> None:
        handler = _FakeHandler(
            [
                _make_result(
                    "knowledge/vendor.md",
                    "Vendor policy changed.",
                    memory_type="knowledge",
                    updated_at="2026-06-03T10:11:12+09:00",
                    origin="external_web",
                    confidence=0.99,
                )
            ]
        )

        output = handler._handle_search_memory({"query": "vendor"})

        assert "updated: 2026-06-03 | origin: external_web" in output
        assert "confidence" not in output

    def test_internal_knowledge_origin_is_omitted(self) -> None:
        handler = _FakeHandler(
            [
                _make_result(
                    "knowledge/internal.md",
                    "Internal note.",
                    memory_type="knowledge",
                    updated_at="2026-06-03T10:11:12+09:00",
                    origin="consolidation",
                )
            ]
        )

        output = handler._handle_search_memory({"query": "internal"})

        assert "updated: 2026-06-03" in output
        assert "origin: consolidation" not in output

    def test_fact_metadata_line_shows_valid_period_and_recorded_at(self) -> None:
        handler = _FakeHandler(
            [
                _make_result(
                    "facts/2026-06-03.jsonl",
                    "Alice is evaluating LoCoMo.",
                    memory_type="facts",
                    valid_at_iso="2026-06-03T10:00:00+09:00",
                    valid_until="",
                    recorded_at="2026-06-03T10:01:00+09:00",
                    confidence=0.85,
                )
            ]
        )

        output = handler._handle_search_memory({"query": "locomo"})

        assert "valid: 2026-06-03〜present | recorded: 2026-06-03" in output
        assert "confidence" not in output

    def test_metadata_counts_toward_output_limit(self, monkeypatch) -> None:
        monkeypatch.setattr("core.tooling.handler_memory._SEARCH_MAX_TOKENS", 24)
        monkeypatch.setattr("core.tooling.handler_memory._SEARCH_MAX_LINES", 8)
        monkeypatch.setattr("core.tooling.handler_memory._SEARCH_MIN_RESULTS", 1)
        results = [
            _make_result(
                f"knowledge/file{i}.md",
                "x" * 120,
                memory_type="knowledge",
                updated_at="2026-06-03T10:00:00+09:00",
                origin="external_web",
            )
            for i in range(3)
        ]
        handler = _FakeHandler(results)

        output = handler._handle_search_memory({"query": "limit"})

        assert "knowledge/file0.md" in output
        assert "(truncated" in output
