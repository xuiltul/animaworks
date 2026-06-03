from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Channel C pointer output and trust separation."""

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.priming import PrimingEngine


@dataclass
class FakeSearchResult:
    """Mock search result for testing Channel C."""

    doc_id: str = "test-anima/knowledge/test.md#0"
    content: str = ""
    score: float = 0.9
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata.setdefault("source_file", "knowledge/test.md")


class TestChannelCTrustSeparation:
    """_channel_c_related_knowledge separates pointer cues by trust."""

    @pytest.fixture
    def engine(self, tmp_path: Path) -> PrimingEngine:
        anima_dir = tmp_path / "animas" / "test-anima"
        (anima_dir / "knowledge").mkdir(parents=True)
        (anima_dir / "episodes").mkdir(parents=True)
        return PrimingEngine(anima_dir, tmp_path / "shared")

    def _make_retriever(self, results: list):
        mock_retriever = MagicMock()
        mock_retriever.search = MagicMock(return_value=results)
        mock_retriever.record_access = MagicMock()
        return mock_retriever

    def _patch_unified_search(self, results: list, meta: dict | None = None):
        searcher = MagicMock()
        searcher.search_many.return_value = [self._to_unified_row(result) for result in results]
        searcher.last_search_meta = meta or {"abstain": False, "abstain_reason": ""}
        return patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=searcher), searcher

    @staticmethod
    def _to_unified_row(result: FakeSearchResult) -> dict:
        row = {
            "doc_id": result.doc_id,
            "content": result.content,
            "score": result.score,
        }
        row.update(result.metadata)
        return row

    @pytest.mark.asyncio
    async def test_all_trusted_results(self, engine: PrimingEngine) -> None:
        """All results with consolidation origin go to the medium bucket."""
        results = [
            FakeSearchResult(
                content="Trusted knowledge",
                score=0.95,
                metadata={"anima": "test-anima", "origin": "consolidation"},
            ),
        ]
        patcher, _searcher = self._patch_unified_search(results)
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["test"])
        assert 'read_memory_file(path="knowledge/test.md")' in medium
        assert "Trusted knowledge" not in medium
        assert untrusted == ""

    @pytest.mark.asyncio
    async def test_all_untrusted_results(self, engine: PrimingEngine) -> None:
        """All results with external origin go to the untrusted bucket."""
        results = [
            FakeSearchResult(
                content="External data",
                score=0.90,
                metadata={"anima": "test-anima", "origin": "external_platform"},
            ),
        ]
        patcher, _searcher = self._patch_unified_search(results)
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["test"])
        assert medium == ""
        assert 'read_memory_file(path="knowledge/test.md")' in untrusted
        assert "External data" not in untrusted

    @pytest.mark.asyncio
    async def test_mixed_trust_results(self, engine: PrimingEngine) -> None:
        """Mixed origins are split between medium and untrusted buckets."""
        results = [
            FakeSearchResult(
                doc_id="consolidated#0",
                content="Consolidated knowledge",
                score=0.95,
                metadata={"anima": "test-anima", "origin": "consolidation"},
            ),
            FakeSearchResult(
                doc_id="external#0",
                content="External data from Slack",
                score=0.85,
                metadata={"anima": "test-anima", "origin": "external_platform"},
            ),
        ]
        patcher, _searcher = self._patch_unified_search(results)
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["test"])
        assert 'read_memory_file(path="knowledge/test.md")' in medium
        assert 'read_memory_file(path="knowledge/test.md")' in untrusted
        assert "Consolidated knowledge" not in medium
        assert "External data from Slack" not in untrusted

    @pytest.mark.asyncio
    async def test_missing_origin_treated_as_untrusted(self, engine: PrimingEngine) -> None:
        """Results without origin metadata are untrusted."""
        results = [
            FakeSearchResult(
                content="Legacy chunk without origin",
                score=0.90,
                metadata={"anima": "test-anima"},
            ),
        ]
        patcher, _searcher = self._patch_unified_search(results)
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["test"])
        assert medium == ""
        assert 'read_memory_file(path="knowledge/test.md")' in untrusted
        assert "Legacy chunk without origin" not in untrusted

    @pytest.mark.asyncio
    async def test_system_origin_is_trusted(self, engine: PrimingEngine) -> None:
        """System origin goes to the medium bucket."""
        results = [
            FakeSearchResult(
                content="System knowledge",
                score=0.90,
                metadata={"anima": "test-anima", "origin": "system"},
            ),
        ]
        patcher, _searcher = self._patch_unified_search(results)
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["test"])
        assert 'read_memory_file(path="knowledge/test.md")' in medium
        assert "System knowledge" not in medium
        assert untrusted == ""

    @pytest.mark.asyncio
    async def test_human_origin_is_medium(self, engine: PrimingEngine) -> None:
        """Human origin goes to the medium bucket."""
        results = [
            FakeSearchResult(
                content="Human-provided knowledge",
                score=0.90,
                metadata={"anima": "test-anima", "origin": "human"},
            ),
        ]
        patcher, _searcher = self._patch_unified_search(results)
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["test"])
        assert 'read_memory_file(path="knowledge/test.md")' in medium
        assert "Human-provided knowledge" not in medium
        assert untrusted == ""

    @pytest.mark.asyncio
    async def test_no_results_returns_empty_tuple(self, engine: PrimingEngine) -> None:
        """No search results returns an empty tuple."""
        patcher, _searcher = self._patch_unified_search([])
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["test"])
        assert medium == ""
        assert untrusted == ""

    @pytest.mark.asyncio
    async def test_no_keywords_returns_empty_tuple(self, engine: PrimingEngine) -> None:
        medium, untrusted = await engine._channel_c_related_knowledge([])
        assert medium == ""
        assert untrusted == ""

    @pytest.mark.asyncio
    async def test_shared_label_preserved(self, engine: PrimingEngine) -> None:
        """Shared chunks retain [shared] label and common_knowledge pointer."""
        results = [
            FakeSearchResult(
                content="Shared common knowledge",
                score=0.90,
                metadata={
                    "anima": "shared",
                    "origin": "system",
                    "source_file": "shared-test.md",
                },
            ),
        ]
        patcher, _searcher = self._patch_unified_search(results)
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["test"])
        assert "[shared]" in medium
        assert 'read_memory_file(path="common_knowledge/shared-test.md")' in medium
        assert "Shared common knowledge" not in medium

    @pytest.mark.asyncio
    async def test_doc_id_fallback_outputs_pointer(self, engine: PrimingEngine) -> None:
        """When source_file is absent, Channel C derives a pointer from doc_id."""
        results = [
            FakeSearchResult(
                doc_id="test-anima/knowledge/from-docid.md#3",
                content="Doc id fallback body",
                score=0.90,
                metadata={"anima": "test-anima", "origin": "system", "source_file": ""},
            ),
        ]
        patcher, _searcher = self._patch_unified_search(results)
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["test"])
        assert 'read_memory_file(path="knowledge/from-docid.md")' in medium
        assert "Doc id fallback body" not in medium
        assert untrusted == ""

    @pytest.mark.asyncio
    async def test_markdown_section_heading_becomes_summary(self, engine: PrimingEngine) -> None:
        """Section chunks indexed by ## headings still produce useful cue summaries."""
        results = [
            FakeSearchResult(
                content="## Deploy Checklist\n\nRun the deploy verifier before release.",
                score=0.90,
                metadata={
                    "anima": "test-anima",
                    "origin": "system",
                    "source_file": "knowledge/deploy.md",
                },
            ),
        ]
        patcher, _searcher = self._patch_unified_search(results)
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["deploy"])
        assert "Deploy Checklist - Run the deploy verifier before release." in medium
        assert 'read_memory_file(path="knowledge/deploy.md")' in medium
        assert untrusted == ""

    @pytest.mark.asyncio
    async def test_shared_doc_id_fallback_keeps_common_knowledge_prefix(
        self,
        engine: PrimingEngine,
    ) -> None:
        """Shared doc_id fallback keeps common_knowledge/ readable pointers."""
        results = [
            FakeSearchResult(
                doc_id="shared/common_knowledge/from-docid.md#3",
                content="Shared doc id fallback body",
                score=0.90,
                metadata={"anima": "shared", "origin": "system", "source_file": ""},
            ),
        ]
        patcher, _searcher = self._patch_unified_search(results)
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["test"])
        assert 'read_memory_file(path="common_knowledge/from-docid.md")' in medium
        assert "Shared doc id fallback body" not in medium
        assert untrusted == ""

    @pytest.mark.asyncio
    async def test_pathless_results_are_not_recorded_as_accessed(self, engine: PrimingEngine) -> None:
        """Only emitted pointer results are counted as accessed."""
        pathless = FakeSearchResult(
            doc_id="opaque-id",
            content="Pathless body",
            score=0.99,
            metadata={"anima": "test-anima", "origin": "system", "source_file": ""},
        )
        readable = FakeSearchResult(
            doc_id="test-anima/knowledge/readable.md#0",
            content="Readable body",
            score=0.90,
            metadata={"anima": "test-anima", "origin": "system", "source_file": ""},
        )
        patcher, _searcher = self._patch_unified_search([pathless, readable])
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["test"])
        assert "--- Result 1" in medium
        assert "--- Result 2" not in medium
        assert 'read_memory_file(path="knowledge/readable.md")' in medium
        assert untrusted == ""

    @pytest.mark.asyncio
    async def test_quotes_path_and_collapses_summary(self, engine: PrimingEngine) -> None:
        """Pointer fields are rendered as safe one-line cues."""
        results = [
            FakeSearchResult(
                content='# Bad "heading"\nbody should not leak',
                score=0.90,
                metadata={
                    "anima": "test-anima",
                    "origin": "system",
                    "source_file": 'knowledge/weird"name.md',
                },
            ),
        ]
        patcher, _searcher = self._patch_unified_search(results)
        with patcher:
            medium, untrusted = await engine._channel_c_related_knowledge(["test"])
        assert 'read_memory_file(path="knowledge/weird\\"name.md")' in medium
        assert 'Bad "heading" - body should not leak' in medium
        assert "\nbody should not leak" not in medium
        assert untrusted == ""
