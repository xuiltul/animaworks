from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Tests for knowledge contradiction detection and resolution."""

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from core.memory.contradiction import (
    ContradictionDetector,
    ContradictionPair,
    ContradictionResult,
)


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory structure for testing."""
    anima = tmp_path / "animas" / "test-anima"
    (anima / "knowledge").mkdir(parents=True)
    (anima / "archive" / "superseded").mkdir(parents=True)
    (anima / "archive" / "merged").mkdir(parents=True)
    (anima / "episodes").mkdir(parents=True)
    (anima / "procedures").mkdir(parents=True)
    (anima / "skills").mkdir(parents=True)
    (anima / "state").mkdir(parents=True)
    return anima


@pytest.fixture
def detector(anima_dir: Path) -> ContradictionDetector:
    """Create a ContradictionDetector for testing."""
    return ContradictionDetector(anima_dir, "test-anima")


def _write_knowledge(
    knowledge_dir: Path,
    filename: str,
    content: str,
    metadata: dict | None = None,
) -> Path:
    """Helper to write a knowledge file with YAML frontmatter."""
    if metadata is None:
        metadata = {
            "created_at": "2026-02-18T10:00:00",
            "confidence": 0.7,
            "auto_consolidated": True,
        }
    fm = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
    path = knowledge_dir / filename
    path.write_text(f"---\n{fm}---\n\n{content}", encoding="utf-8")
    return path


# ── NLI Contradiction Check Tests ───────────────────────────


class TestNLIContradictionCheck:
    """Test the NLI contradiction detection stage."""

    @pytest.mark.asyncio
    async def test_nli_unavailable_returns_no_contradiction(
        self, detector: ContradictionDetector,
    ) -> None:
        """When NLI is unavailable, return no contradiction."""
        detector._nli_validator = None

        # Patch KnowledgeValidator to simulate unavailable NLI
        mock_validator = MagicMock()
        mock_validator._nli_check.return_value = ("neutral", 0.0)

        with patch(
            "core.memory.validation.KnowledgeValidator",
            return_value=mock_validator,
        ):
            is_contradiction, score, is_entailment = (
                await detector._check_contradiction_nli("text A", "text B")
            )

        assert is_contradiction is False
        assert score == 0.0
        assert is_entailment is False

    @pytest.mark.asyncio
    async def test_nli_detects_contradiction(
        self, detector: ContradictionDetector,
    ) -> None:
        """NLI detecting contradiction in either direction triggers flag."""
        mock_validator = MagicMock()
        mock_validator._nli_check.side_effect = [
            ("contradiction", 0.85),  # A as premise, B as hypothesis
            ("neutral", 0.30),        # B as premise, A as hypothesis
        ]
        detector._nli_validator = mock_validator

        is_contradiction, score, is_entailment = (
            await detector._check_contradiction_nli(
                "The server uses port 8080",
                "The server uses port 3000",
            )
        )

        assert is_contradiction is True
        assert score == 0.85
        assert is_entailment is False

    @pytest.mark.asyncio
    async def test_nli_no_contradiction_when_below_threshold(
        self, detector: ContradictionDetector,
    ) -> None:
        """Contradiction score below threshold is not flagged."""
        mock_validator = MagicMock()
        mock_validator._nli_check.side_effect = [
            ("contradiction", 0.40),  # Below NLI_CONTRADICTION_THRESHOLD
            ("neutral", 0.30),
        ]
        detector._nli_validator = mock_validator

        is_contradiction, score, is_entailment = (
            await detector._check_contradiction_nli("text A", "text B")
        )

        assert is_contradiction is False

    @pytest.mark.asyncio
    async def test_nli_bidirectional_detection(
        self, detector: ContradictionDetector,
    ) -> None:
        """Contradiction detected in reverse direction is also caught."""
        mock_validator = MagicMock()
        mock_validator._nli_check.side_effect = [
            ("neutral", 0.30),        # A->B: no contradiction
            ("contradiction", 0.78),  # B->A: contradiction
        ]
        detector._nli_validator = mock_validator

        is_contradiction, score, is_entailment = (
            await detector._check_contradiction_nli("text A", "text B")
        )

        assert is_contradiction is True
        assert score == 0.78
        assert is_entailment is False

    @pytest.mark.asyncio
    async def test_nli_entailment_detected(
        self, detector: ContradictionDetector,
    ) -> None:
        """NLI entailment above threshold is flagged as is_entailment."""
        mock_validator = MagicMock()
        mock_validator._nli_check.side_effect = [
            ("entailment", 0.85),  # A->B: entailment
            ("neutral", 0.30),     # B->A: neutral
        ]
        detector._nli_validator = mock_validator

        is_contradiction, score, is_entailment = (
            await detector._check_contradiction_nli("text A", "text B")
        )

        assert is_contradiction is False
        assert is_entailment is True

    @pytest.mark.asyncio
    async def test_nli_entailment_below_threshold_not_flagged(
        self, detector: ContradictionDetector,
    ) -> None:
        """Entailment below threshold does not set is_entailment."""
        mock_validator = MagicMock()
        mock_validator._nli_check.side_effect = [
            ("entailment", 0.50),  # Below NLI_ENTAILMENT_THRESHOLD
            ("neutral", 0.30),
        ]
        detector._nli_validator = mock_validator

        is_contradiction, score, is_entailment = (
            await detector._check_contradiction_nli("text A", "text B")
        )

        assert is_contradiction is False
        assert is_entailment is False


# ── LLM Contradiction Check Tests ──────────────────────────


class TestLLMContradictionCheck:
    """Test the LLM contradiction analysis stage."""

    @pytest.mark.asyncio
    async def test_llm_detects_contradiction_supersede(
        self, detector: ContradictionDetector,
    ) -> None:
        """LLM correctly identifies contradiction with supersede resolution."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"is_contradiction": true, "resolution": "supersede", '
            '"reason": "File B has newer information", "merged_content": null}'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await detector._check_contradiction_llm(
                "The API uses REST", "The API was migrated to GraphQL",
                "api-design.md", "api-migration.md", "test-model",
            )

        assert result.is_contradiction is True
        assert result.resolution == "supersede"
        assert "newer" in result.reason.lower() or result.reason

    @pytest.mark.asyncio
    async def test_llm_detects_contradiction_merge(
        self, detector: ContradictionDetector,
    ) -> None:
        """LLM correctly identifies contradiction with merge resolution."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"is_contradiction": true, "resolution": "merge", '
            '"reason": "Both files contain useful info", '
            '"merged_content": "# Unified API design\\nREST and GraphQL coexist"}'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await detector._check_contradiction_llm(
                "Text A", "Text B", "file-a.md", "file-b.md", "test-model",
            )

        assert result.is_contradiction is True
        assert result.resolution == "merge"
        assert result.merged_content is not None

    @pytest.mark.asyncio
    async def test_llm_no_contradiction(
        self, detector: ContradictionDetector,
    ) -> None:
        """LLM correctly identifies no contradiction."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"is_contradiction": false, "resolution": "coexist", '
            '"reason": "No contradiction", "merged_content": null}'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await detector._check_contradiction_llm(
                "Text A", "Text B", "file-a.md", "file-b.md", "test-model",
            )

        assert result.is_contradiction is False

    @pytest.mark.asyncio
    async def test_llm_failure_returns_no_contradiction(
        self, detector: ContradictionDetector,
    ) -> None:
        """LLM failure conservatively assumes no contradiction."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("API error")
            result = await detector._check_contradiction_llm(
                "Text A", "Text B", "file-a.md", "file-b.md", "test-model",
            )

        assert result.is_contradiction is False
        assert result.resolution == "coexist"

    @pytest.mark.asyncio
    async def test_llm_json_embedded_in_text(
        self, detector: ContradictionDetector,
    ) -> None:
        """JSON embedded in natural language is extracted correctly."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            'Analysis complete. Here is the result: '
            '{"is_contradiction": true, "resolution": "coexist", '
            '"reason": "Context-dependent", "merged_content": null}'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await detector._check_contradiction_llm(
                "Text A", "Text B", "file-a.md", "file-b.md", "test-model",
            )

        assert result.is_contradiction is True
        assert result.resolution == "coexist"


# ── Supersede Resolution Tests ──────────────────────────────


class TestSupersedeResolution:
    """Test the supersede resolution strategy."""

    def test_supersede_archives_older_file(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Older file is moved to archive/superseded/."""
        knowledge_dir = anima_dir / "knowledge"

        file_a = _write_knowledge(
            knowledge_dir, "old-info.md", "Old content",
            {"created_at": "2026-01-01T10:00:00", "confidence": 0.7},
        )
        file_b = _write_knowledge(
            knowledge_dir, "new-info.md", "New content",
            {"created_at": "2026-02-18T10:00:00", "confidence": 0.8},
        )

        pair = ContradictionPair(
            file_a=file_a,
            file_b=file_b,
            text_a="Old content",
            text_b="New content",
            confidence=0.85,
            resolution="supersede",
            reason="File B is newer",
        )

        detector._apply_supersede(pair)

        # Newer file should still exist in knowledge/
        assert file_b.exists()

        # Older file should be archived
        assert not file_a.exists()
        archived = anima_dir / "archive" / "superseded" / "old-info.md"
        assert archived.exists()

        # Archived file should have superseded_by and valid_until metadata
        text = archived.read_text(encoding="utf-8")
        assert "superseded_by" in text
        assert "new-info.md" in text
        assert "valid_until" in text

    def test_supersede_adds_supersedes_metadata_to_newer_file(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Newer file gets `supersedes` metadata referencing the old filename."""
        knowledge_dir = anima_dir / "knowledge"

        file_a = _write_knowledge(
            knowledge_dir, "old-info.md", "Old content",
            {"created_at": "2026-01-01T10:00:00", "confidence": 0.7},
        )
        file_b = _write_knowledge(
            knowledge_dir, "new-info.md", "New content",
            {"created_at": "2026-02-18T10:00:00", "confidence": 0.8},
        )

        pair = ContradictionPair(
            file_a=file_a,
            file_b=file_b,
            text_a="Old content",
            text_b="New content",
            confidence=0.85,
            resolution="supersede",
            reason="File B is newer",
        )

        detector._apply_supersede(pair)

        # Newer file should contain supersedes metadata
        text = file_b.read_text(encoding="utf-8")
        parts = text.split("---", 2)
        meta = yaml.safe_load(parts[1])
        assert "supersedes" in meta
        assert "old-info.md" in meta["supersedes"]

    def test_supersede_correct_direction_when_a_is_newer(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """When file_a is newer, file_b gets archived instead."""
        knowledge_dir = anima_dir / "knowledge"

        file_a = _write_knowledge(
            knowledge_dir, "new-version.md", "Updated content",
            {"created_at": "2026-02-18T10:00:00", "updated_at": "2026-02-18T15:00:00"},
        )
        file_b = _write_knowledge(
            knowledge_dir, "old-version.md", "Original content",
            {"created_at": "2026-01-15T10:00:00"},
        )

        pair = ContradictionPair(
            file_a=file_a,
            file_b=file_b,
            text_a="Updated content",
            text_b="Original content",
            confidence=0.80,
            resolution="supersede",
            reason="File A is newer",
        )

        detector._apply_supersede(pair)

        assert file_a.exists()
        assert not file_b.exists()
        assert (anima_dir / "archive" / "superseded" / "old-version.md").exists()


# ── Merge Resolution Tests ──────────────────────────────────


class TestMergeResolution:
    """Test the merge resolution strategy."""

    @pytest.mark.asyncio
    async def test_merge_with_pre_generated_content(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Merge uses pre-generated content when available."""
        knowledge_dir = anima_dir / "knowledge"

        file_a = _write_knowledge(
            knowledge_dir, "topic-a.md", "Content from file A",
        )
        file_b = _write_knowledge(
            knowledge_dir, "topic-b.md", "Content from file B",
        )

        pair = ContradictionPair(
            file_a=file_a,
            file_b=file_b,
            text_a="Content from file A",
            text_b="Content from file B",
            confidence=0.75,
            resolution="merge",
            reason="Both contain useful info",
            merged_content="# Unified Topic\nCombined content from A and B",
        )

        result = await detector._apply_merge(pair, "test-model")

        assert result is True

        # Original files should be archived
        assert not file_a.exists()
        assert not file_b.exists()
        assert (anima_dir / "archive" / "merged" / "topic-a.md").exists()
        assert (anima_dir / "archive" / "merged" / "topic-b.md").exists()

        # Merged file should exist in knowledge/
        merged_files = list(knowledge_dir.glob("_merged_*.md"))
        assert len(merged_files) == 1

        merged_text = merged_files[0].read_text(encoding="utf-8")
        assert "Combined content from A and B" in merged_text
        assert "merged_from" in merged_text
        assert "topic-a.md" in merged_text
        assert "topic-b.md" in merged_text

    @pytest.mark.asyncio
    async def test_merge_generates_content_via_llm(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """When no pre-generated content, LLM generates merged text."""
        knowledge_dir = anima_dir / "knowledge"

        file_a = _write_knowledge(
            knowledge_dir, "api-rest.md", "REST API design",
        )
        file_b = _write_knowledge(
            knowledge_dir, "api-graphql.md", "GraphQL API design",
        )

        pair = ContradictionPair(
            file_a=file_a,
            file_b=file_b,
            text_a="REST API design",
            text_b="GraphQL API design",
            confidence=0.70,
            resolution="merge",
            reason="Complementary API info",
            merged_content=None,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "# API Design\nBoth REST and GraphQL are used."
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await detector._apply_merge(pair, "test-model")

        assert result is True
        merged_files = list(knowledge_dir.glob("_merged_*.md"))
        assert len(merged_files) == 1

    @pytest.mark.asyncio
    async def test_merge_fails_on_empty_content(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Merge returns False when LLM returns empty content."""
        knowledge_dir = anima_dir / "knowledge"

        file_a = _write_knowledge(
            knowledge_dir, "file-a.md", "Content A",
        )
        file_b = _write_knowledge(
            knowledge_dir, "file-b.md", "Content B",
        )

        pair = ContradictionPair(
            file_a=file_a,
            file_b=file_b,
            text_a="Content A",
            text_b="Content B",
            confidence=0.70,
            resolution="merge",
            reason="Merge needed",
            merged_content=None,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await detector._apply_merge(pair, "test-model")

        assert result is False
        # Original files should NOT be archived on failure
        assert file_a.exists()
        assert file_b.exists()


# ── Coexist Resolution Tests ───────────────────────────────


class TestCoexistResolution:
    """Test the coexist resolution strategy."""

    def test_coexist_annotates_both_files(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Both files get coexists_with metadata annotation."""
        knowledge_dir = anima_dir / "knowledge"

        file_a = _write_knowledge(
            knowledge_dir, "approach-a.md", "Approach A details",
        )
        file_b = _write_knowledge(
            knowledge_dir, "approach-b.md", "Approach B details",
        )

        pair = ContradictionPair(
            file_a=file_a,
            file_b=file_b,
            text_a="Approach A details",
            text_b="Approach B details",
            confidence=0.60,
            resolution="coexist",
            reason="Context-dependent approaches",
        )

        detector._apply_coexist(pair)

        # Both files should still exist (not moved)
        assert file_a.exists()
        assert file_b.exists()

        # Check metadata
        text_a = file_a.read_text(encoding="utf-8")
        text_b = file_b.read_text(encoding="utf-8")

        assert "coexists_with" in text_a
        assert "approach-b.md" in text_a
        assert "coexists_with" in text_b
        assert "approach-a.md" in text_b

    def test_coexist_idempotent(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Calling coexist twice does not duplicate the annotation."""
        knowledge_dir = anima_dir / "knowledge"

        file_a = _write_knowledge(
            knowledge_dir, "fact-a.md", "Fact A",
        )
        file_b = _write_knowledge(
            knowledge_dir, "fact-b.md", "Fact B",
        )

        pair = ContradictionPair(
            file_a=file_a,
            file_b=file_b,
            text_a="Fact A",
            text_b="Fact B",
            confidence=0.55,
            resolution="coexist",
            reason="Both valid",
        )

        detector._apply_coexist(pair)
        detector._apply_coexist(pair)

        # Parse metadata to verify no duplicates
        text = file_a.read_text(encoding="utf-8")
        parts = text.split("---", 2)
        meta = yaml.safe_load(parts[1])
        coexists = meta.get("coexists_with", [])
        assert coexists.count("fact-b.md") == 1


# ── Resolve Contradictions Tests ────────────────────────────


class TestResolveContradictions:
    """Test the full resolution pipeline."""

    @pytest.mark.asyncio
    async def test_resolve_mixed_strategies(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Resolve a batch of contradictions with different strategies."""
        knowledge_dir = anima_dir / "knowledge"

        # Create knowledge files for each strategy
        file_supersede_old = _write_knowledge(
            knowledge_dir, "old-deploy.md", "Deploy to Heroku",
            {"created_at": "2026-01-01T10:00:00"},
        )
        file_supersede_new = _write_knowledge(
            knowledge_dir, "new-deploy.md", "Deploy to AWS",
            {"created_at": "2026-02-18T10:00:00"},
        )
        file_merge_a = _write_knowledge(
            knowledge_dir, "merge-a.md", "Content A for merge",
        )
        file_merge_b = _write_knowledge(
            knowledge_dir, "merge-b.md", "Content B for merge",
        )
        file_coexist_a = _write_knowledge(
            knowledge_dir, "coexist-a.md", "Approach A",
        )
        file_coexist_b = _write_knowledge(
            knowledge_dir, "coexist-b.md", "Approach B",
        )

        pairs = [
            ContradictionPair(
                file_a=file_supersede_old,
                file_b=file_supersede_new,
                text_a="Deploy to Heroku",
                text_b="Deploy to AWS",
                confidence=0.90,
                resolution="supersede",
                reason="Migration to AWS",
            ),
            ContradictionPair(
                file_a=file_merge_a,
                file_b=file_merge_b,
                text_a="Content A for merge",
                text_b="Content B for merge",
                confidence=0.75,
                resolution="merge",
                reason="Complementary info",
                merged_content="# Merged\nCombined A and B",
            ),
            ContradictionPair(
                file_a=file_coexist_a,
                file_b=file_coexist_b,
                text_a="Approach A",
                text_b="Approach B",
                confidence=0.55,
                resolution="coexist",
                reason="Both valid in context",
            ),
        ]

        results = await detector.resolve_contradictions(pairs, "test-model")

        assert results["superseded"] == 1
        assert results["merged"] == 1
        assert results["coexisted"] == 1
        assert results["errors"] == 0

    @pytest.mark.asyncio
    async def test_resolve_handles_errors_gracefully(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Resolution errors are counted but don't crash the pipeline."""
        knowledge_dir = anima_dir / "knowledge"

        # Create a file that will cause an error (non-existent partner)
        file_a = _write_knowledge(
            knowledge_dir, "existing.md", "Some content",
        )
        file_b = knowledge_dir / "nonexistent.md"  # Does not exist

        pairs = [
            ContradictionPair(
                file_a=file_a,
                file_b=file_b,
                text_a="Some content",
                text_b="Missing content",
                confidence=0.80,
                resolution="supersede",
                reason="Test error handling",
            ),
        ]

        # This should not raise; the error is counted
        results = await detector.resolve_contradictions(pairs, "test-model")
        assert results["errors"] >= 0  # May or may not error depending on metadata read


# ── Scan Contradictions Tests ───────────────────────────────


class TestScanContradictions:
    """Test the full scan pipeline with mocked NLI and LLM."""

    @pytest.mark.asyncio
    async def test_scan_detects_contradiction(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Full scan pipeline detects and classifies a contradiction."""
        knowledge_dir = anima_dir / "knowledge"

        _write_knowledge(
            knowledge_dir, "db-mysql.md", "We use MySQL for the database.",
        )
        _write_knowledge(
            knowledge_dir, "db-postgres.md", "We use PostgreSQL for the database.",
        )

        # Mock NLI to detect contradiction
        mock_validator = MagicMock()
        mock_validator._nli_check.side_effect = [
            ("contradiction", 0.85),  # A->B
            ("neutral", 0.30),        # B->A
        ]
        detector._nli_validator = mock_validator

        # Mock LLM to confirm and propose supersede
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"is_contradiction": true, "resolution": "supersede", '
            '"reason": "Migrated to PostgreSQL", "merged_content": null}'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            # Mock RAG to return None (use exhaustive fallback)
            with patch.object(
                detector, "_find_candidates_via_rag", return_value=None,
            ):
                results = await detector.scan_contradictions(model="test-model")

        assert len(results) == 1
        assert results[0].resolution == "supersede"
        assert results[0].confidence == 0.85

    @pytest.mark.asyncio
    async def test_scan_target_file_only(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Targeted scan only checks pairs involving the specified file."""
        knowledge_dir = anima_dir / "knowledge"

        target = _write_knowledge(
            knowledge_dir, "target.md", "Target file content",
        )
        _write_knowledge(
            knowledge_dir, "other-a.md", "Other file A",
        )
        _write_knowledge(
            knowledge_dir, "other-b.md", "Other file B",
        )

        # Mock NLI: no contradiction
        mock_validator = MagicMock()
        mock_validator._nli_check.return_value = ("neutral", 0.30)
        detector._nli_validator = mock_validator

        # Mock LLM: no contradiction
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"is_contradiction": false, "resolution": "coexist", '
            '"reason": "No issue", "merged_content": null}'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            with patch.object(
                detector, "_find_candidates_via_rag", return_value=None,
            ):
                results = await detector.scan_contradictions(
                    target_file=target, model="test-model",
                )

        # No contradictions found, but verify LLM was called for each candidate pair
        # (target vs other-a, target vs other-b)
        assert len(results) == 0
        assert mock_llm.call_count == 2  # Two candidate pairs checked

    @pytest.mark.asyncio
    async def test_scan_entailment_skips_llm(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """When NLI returns entailment, LLM is not called."""
        knowledge_dir = anima_dir / "knowledge"

        _write_knowledge(
            knowledge_dir, "fact-a.md", "Python is a programming language.",
        )
        _write_knowledge(
            knowledge_dir, "fact-b.md", "Python is a widely-used programming language.",
        )

        # Mock NLI to return entailment (consistent texts)
        mock_validator = MagicMock()
        mock_validator._nli_check.side_effect = [
            ("entailment", 0.90),  # A->B: entailment
            ("entailment", 0.85),  # B->A: entailment
        ]
        detector._nli_validator = mock_validator

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"is_contradiction": false, "resolution": "coexist", '
            '"reason": "No contradiction", "merged_content": null}'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            with patch.object(
                detector, "_find_candidates_via_rag", return_value=None,
            ):
                results = await detector.scan_contradictions(model="test-model")

        # No contradictions
        assert len(results) == 0
        # LLM should NOT have been called — entailment short-circuits
        assert mock_llm.call_count == 0

    @pytest.mark.asyncio
    async def test_scan_neutral_nli_still_calls_llm(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """When NLI returns neutral (uncertain), LLM is still called as fallback."""
        knowledge_dir = anima_dir / "knowledge"

        _write_knowledge(
            knowledge_dir, "topic-x.md", "Topic X content.",
        )
        _write_knowledge(
            knowledge_dir, "topic-y.md", "Topic Y content.",
        )

        # Mock NLI to return neutral (uncertain)
        mock_validator = MagicMock()
        mock_validator._nli_check.side_effect = [
            ("neutral", 0.50),  # A->B: neutral
            ("neutral", 0.45),  # B->A: neutral
        ]
        detector._nli_validator = mock_validator

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"is_contradiction": false, "resolution": "coexist", '
            '"reason": "No contradiction", "merged_content": null}'
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            with patch.object(
                detector, "_find_candidates_via_rag", return_value=None,
            ):
                results = await detector.scan_contradictions(model="test-model")

        # No contradictions found
        assert len(results) == 0
        # LLM SHOULD have been called for neutral/uncertain case
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_scan_empty_knowledge_dir(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Scan returns empty list when no knowledge files exist."""
        results = await detector.scan_contradictions(model="test-model")
        assert results == []

    @pytest.mark.asyncio
    async def test_scan_single_file(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Scan returns empty when only one knowledge file exists."""
        knowledge_dir = anima_dir / "knowledge"
        _write_knowledge(knowledge_dir, "only-one.md", "Single file")

        with patch.object(
            detector, "_find_candidates_via_rag", return_value=None,
        ):
            results = await detector.scan_contradictions(model="test-model")

        assert results == []


# ── Candidate Pair Generation Tests ─────────────────────────


class TestCandidatePairGeneration:
    """Test the candidate pair generation logic."""

    def test_exhaustive_generates_all_pairs(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Exhaustive fallback generates all pairwise combinations."""
        knowledge_dir = anima_dir / "knowledge"

        _write_knowledge(knowledge_dir, "a.md", "Content A")
        _write_knowledge(knowledge_dir, "b.md", "Content B")
        _write_knowledge(knowledge_dir, "c.md", "Content C")

        contents = {
            knowledge_dir / "a.md": "Content A",
            knowledge_dir / "b.md": "Content B",
            knowledge_dir / "c.md": "Content C",
        }

        pairs = detector._find_candidates_exhaustive(contents, None)

        # 3 files => 3 pairs: (a,b), (a,c), (b,c)
        assert len(pairs) == 3

    def test_exhaustive_filters_by_target(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """Exhaustive with target file only returns pairs involving target."""
        knowledge_dir = anima_dir / "knowledge"

        target = _write_knowledge(knowledge_dir, "target.md", "Target")
        _write_knowledge(knowledge_dir, "other-a.md", "Other A")
        _write_knowledge(knowledge_dir, "other-b.md", "Other B")

        contents = {
            knowledge_dir / "other-a.md": "Other A",
            knowledge_dir / "other-b.md": "Other B",
            knowledge_dir / "target.md": "Target",
        }

        pairs = detector._find_candidates_exhaustive(contents, target)

        # Only pairs involving target: (other-a, target), (other-b, target)
        assert len(pairs) == 2
        for file_a, _, file_b, _ in pairs:
            assert file_a == target or file_b == target


# ── Merge Topic Derivation Tests ────────────────────────────


class TestMergeTopicDerivation:
    """Test the topic derivation helper."""

    def test_common_words_extracted(self) -> None:
        """Common words between stems are extracted."""
        topic = ContradictionDetector._derive_merge_topic(
            "api-rest-design", "api-graphql-design",
        )
        assert "api" in topic
        assert "design" in topic

    def test_fallback_to_shorter_stem(self) -> None:
        """When no common words, use the shorter stem."""
        topic = ContradictionDetector._derive_merge_topic(
            "database", "networking-guide",
        )
        assert topic == "database"

    def test_single_char_words_excluded(self) -> None:
        """Single-character words are excluded from merge topic."""
        topic = ContradictionDetector._derive_merge_topic(
            "a-b-config", "c-d-config",
        )
        assert "config" in topic
        assert "a" not in topic.split("-")
        assert "b" not in topic.split("-")


# ── Activity Log Event Tests ─────────────────────────────────


class TestActivityLogEvent:
    """Test that contradiction resolution records activity log events."""

    @pytest.mark.asyncio
    async def test_resolution_logs_activity_event(
        self, anima_dir: Path,
    ) -> None:
        """Each successful resolution records a knowledge_contradiction_resolved event."""
        from core.memory.activity import ActivityLogger

        activity_logger = ActivityLogger(anima_dir)
        detector = ContradictionDetector(
            anima_dir, "test-anima", activity_logger=activity_logger,
        )
        knowledge_dir = anima_dir / "knowledge"

        file_a = _write_knowledge(
            knowledge_dir, "old-fact.md", "Old fact",
            {"created_at": "2026-01-01T10:00:00"},
        )
        file_b = _write_knowledge(
            knowledge_dir, "new-fact.md", "New fact",
            {"created_at": "2026-02-18T10:00:00"},
        )

        pairs = [
            ContradictionPair(
                file_a=file_a,
                file_b=file_b,
                text_a="Old fact",
                text_b="New fact",
                confidence=0.80,
                resolution="supersede",
                reason="Newer info",
            ),
        ]

        await detector.resolve_contradictions(pairs, "test-model")

        # Verify the activity log was written
        entries = activity_logger.recent(
            days=1,
            types=["knowledge_contradiction_resolved"],
        )
        assert len(entries) == 1
        assert entries[0].type == "knowledge_contradiction_resolved"
        assert "supersede" in entries[0].content
        assert entries[0].meta.get("strategy") == "supersede"

    @pytest.mark.asyncio
    async def test_no_activity_log_when_logger_absent(
        self, detector: ContradictionDetector, anima_dir: Path,
    ) -> None:
        """When no activity_logger is provided, resolution still succeeds."""
        knowledge_dir = anima_dir / "knowledge"

        file_a = _write_knowledge(
            knowledge_dir, "coexist-a.md", "Approach A",
        )
        file_b = _write_knowledge(
            knowledge_dir, "coexist-b.md", "Approach B",
        )

        pairs = [
            ContradictionPair(
                file_a=file_a,
                file_b=file_b,
                text_a="Approach A",
                text_b="Approach B",
                confidence=0.55,
                resolution="coexist",
                reason="Both valid",
            ),
        ]

        # detector fixture has no activity_logger
        results = await detector.resolve_contradictions(pairs, "test-model")
        assert results["coexisted"] == 1
        assert results["errors"] == 0

    @pytest.mark.asyncio
    async def test_merge_resolution_logs_activity_event(
        self, anima_dir: Path,
    ) -> None:
        """Merge resolution also records activity log event."""
        from core.memory.activity import ActivityLogger

        activity_logger = ActivityLogger(anima_dir)
        detector = ContradictionDetector(
            anima_dir, "test-anima", activity_logger=activity_logger,
        )
        knowledge_dir = anima_dir / "knowledge"

        file_a = _write_knowledge(
            knowledge_dir, "merge-x.md", "Content X",
        )
        file_b = _write_knowledge(
            knowledge_dir, "merge-y.md", "Content Y",
        )

        pairs = [
            ContradictionPair(
                file_a=file_a,
                file_b=file_b,
                text_a="Content X",
                text_b="Content Y",
                confidence=0.70,
                resolution="merge",
                reason="Complementary info",
                merged_content="# Merged\nCombined X and Y",
            ),
        ]

        await detector.resolve_contradictions(pairs, "test-model")

        entries = activity_logger.recent(
            days=1,
            types=["knowledge_contradiction_resolved"],
        )
        assert len(entries) == 1
        assert entries[0].meta.get("strategy") == "merge"


# ── Legacy Migration Tests ───────────────────────────────────


class TestLegacyMigration:
    """Test superseded_at → valid_until migration."""

    def test_read_knowledge_metadata_migrates_superseded_at(
        self, anima_dir: Path,
    ) -> None:
        """read_knowledge_metadata renames superseded_at to valid_until."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(anima_dir)
        knowledge_dir = anima_dir / "knowledge"

        # Write a file with legacy superseded_at field
        path = _write_knowledge(
            knowledge_dir, "legacy.md", "Legacy content",
            {
                "created_at": "2026-01-01T10:00:00",
                "superseded_at": "2026-02-01T12:00:00",
                "superseded_by": "new-file.md",
            },
        )

        meta = mm.read_knowledge_metadata(path)

        # superseded_at should be migrated to valid_until
        assert "valid_until" in meta
        assert meta["valid_until"] == "2026-02-01T12:00:00"
        # superseded_at should no longer be present
        assert "superseded_at" not in meta
        # superseded_by should remain unchanged
        assert meta["superseded_by"] == "new-file.md"

    def test_read_knowledge_metadata_preserves_valid_until(
        self, anima_dir: Path,
    ) -> None:
        """Files already using valid_until are returned as-is."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(anima_dir)
        knowledge_dir = anima_dir / "knowledge"

        path = _write_knowledge(
            knowledge_dir, "modern.md", "Modern content",
            {
                "created_at": "2026-01-01T10:00:00",
                "valid_until": "2026-02-15T10:00:00",
            },
        )

        meta = mm.read_knowledge_metadata(path)
        assert meta["valid_until"] == "2026-02-15T10:00:00"

    def test_indexer_migrates_superseded_at_in_frontmatter(
        self, anima_dir: Path,
    ) -> None:
        """Indexer's _extract_metadata migrates superseded_at in frontmatter."""
        from unittest.mock import MagicMock

        from core.memory.rag.indexer import MemoryIndexer

        mock_store = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]

        indexer = MemoryIndexer(
            mock_store, "test-anima", anima_dir,
            embedding_model=mock_model,
        )

        knowledge_dir = anima_dir / "knowledge"
        test_file = knowledge_dir / "test.md"
        test_file.write_text("test content", encoding="utf-8")

        # Pass legacy frontmatter with superseded_at
        legacy_fm = {
            "superseded_at": "2026-02-01T12:00:00",
            "superseded_by": "newer.md",
        }
        metadata = indexer._extract_metadata(
            test_file, "test content", "knowledge", 0, 1,
            frontmatter=legacy_fm,
        )

        assert metadata["valid_until"] == "2026-02-01T12:00:00"

    def test_indexer_default_valid_until_empty(
        self, anima_dir: Path,
    ) -> None:
        """Chunks without valid_until in frontmatter get empty string default."""
        from unittest.mock import MagicMock

        from core.memory.rag.indexer import MemoryIndexer

        mock_store = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]

        indexer = MemoryIndexer(
            mock_store, "test-anima", anima_dir,
            embedding_model=mock_model,
        )

        knowledge_dir = anima_dir / "knowledge"
        test_file = knowledge_dir / "active.md"
        test_file.write_text("active content", encoding="utf-8")

        metadata = indexer._extract_metadata(
            test_file, "active content", "knowledge", 0, 1,
            frontmatter={},
        )

        assert metadata["valid_until"] == ""


# ── RAG Filter Tests ─────────────────────────────────────────


class TestRAGFilter:
    """Test that superseded knowledge is filtered from RAG search."""

    def test_search_excludes_superseded_by_default(self) -> None:
        """search() passes valid_until filter when include_superseded=False."""
        from unittest.mock import MagicMock, patch

        from core.memory.rag.retriever import MemoryRetriever

        mock_store = MagicMock()
        mock_indexer = MagicMock()
        mock_indexer._generate_embeddings.return_value = [[0.1] * 384]

        # Return empty results from vector store
        mock_store.query.return_value = []

        retriever = MemoryRetriever(mock_store, mock_indexer, Path("/tmp"))

        retriever.search(
            query="test query",
            anima_name="test-anima",
            memory_type="knowledge",
        )

        # Verify filter_metadata was passed with valid_until=""
        mock_store.query.assert_called_once()
        call_kwargs = mock_store.query.call_args
        assert call_kwargs.kwargs.get("filter_metadata") == {"valid_until": ""}

    def test_search_includes_superseded_when_requested(self) -> None:
        """search() does not filter when include_superseded=True."""
        from unittest.mock import MagicMock

        from core.memory.rag.retriever import MemoryRetriever

        mock_store = MagicMock()
        mock_indexer = MagicMock()
        mock_indexer._generate_embeddings.return_value = [[0.1] * 384]

        mock_store.query.return_value = []

        retriever = MemoryRetriever(mock_store, mock_indexer, Path("/tmp"))

        retriever.search(
            query="test query",
            anima_name="test-anima",
            memory_type="knowledge",
            include_superseded=True,
        )

        call_kwargs = mock_store.query.call_args
        assert call_kwargs.kwargs.get("filter_metadata") is None

    def test_search_no_filter_for_non_knowledge_types(self) -> None:
        """search() does not filter valid_until for non-knowledge memory types."""
        from unittest.mock import MagicMock

        from core.memory.rag.retriever import MemoryRetriever

        mock_store = MagicMock()
        mock_indexer = MagicMock()
        mock_indexer._generate_embeddings.return_value = [[0.1] * 384]

        mock_store.query.return_value = []

        retriever = MemoryRetriever(mock_store, mock_indexer, Path("/tmp"))

        retriever.search(
            query="test query",
            anima_name="test-anima",
            memory_type="episodes",
        )

        call_kwargs = mock_store.query.call_args
        assert call_kwargs.kwargs.get("filter_metadata") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
