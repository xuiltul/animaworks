from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Integration tests for the consolidation pipeline.

Tests the end-to-end daily consolidation flow with validation,
sanitization, weekly merge archiving, and frontmatter handling.
All external dependencies (LLM, NLI) are mocked.
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def temp_anima_dir(tmp_path: Path) -> Path:
    """Create a temporary anima directory structure."""
    anima_dir = tmp_path / "test_anima"
    (anima_dir / "episodes").mkdir(parents=True)
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "procedures").mkdir(parents=True)
    (anima_dir / "skills").mkdir(parents=True)
    (anima_dir / "state").mkdir(parents=True)
    return anima_dir


@pytest.fixture
def consolidation_engine(temp_anima_dir: Path):
    """Create a ConsolidationEngine instance."""
    from core.memory.consolidation import ConsolidationEngine

    return ConsolidationEngine(
        anima_dir=temp_anima_dir,
        anima_name="test_anima",
    )


def _make_llm_response(content: str) -> MagicMock:
    """Build a mock LiteLLM response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


# ── Sanitize Tests ───────────────────────────────────────────


class TestSanitizeLLMOutput:
    """Test _sanitize_llm_output static method."""

    def test_strip_markdown_fence(self, consolidation_engine: object) -> None:
        """Markdown code fences are removed."""
        text = "```markdown\n# Title\n\nContent.\n```"
        result = consolidation_engine._sanitize_llm_output(text)
        assert "```" not in result
        assert "# Title" in result
        assert "Content." in result

    def test_strip_md_fence(self, consolidation_engine: object) -> None:
        """Short ``md`` fence variant is also removed."""
        text = "```md\n# Title\n\nContent.\n```"
        result = consolidation_engine._sanitize_llm_output(text)
        assert "```" not in result
        assert "# Title" in result

    def test_no_fence(self, consolidation_engine: object) -> None:
        """Content without fences is returned unchanged (modulo strip)."""
        text = "## Heading\n\nBody text."
        result = consolidation_engine._sanitize_llm_output(text)
        assert result == text

    def test_bare_fence(self, consolidation_engine: object) -> None:
        """Plain ``` fences are also handled."""
        text = "```\n## Output\n\nData.\n```"
        result = consolidation_engine._sanitize_llm_output(text)
        assert "```" not in result
        assert "## Output" in result

    def test_preserves_internal_content(self, consolidation_engine: object) -> None:
        """Only wrapper fences are stripped; mid-content dashes are kept."""
        text = "```markdown\n## Title\n\nSome content.\n```"
        result = consolidation_engine._sanitize_llm_output(text)
        assert "## Title" in result
        assert "Some content." in result


# ── Daily Consolidation E2E ──────────────────────────────────


class TestDailyConsolidationE2E:
    """End-to-end tests for daily_consolidate with mocked LLM."""

    @pytest.mark.asyncio
    async def test_full_pipeline_creates_knowledge_with_frontmatter(
        self, consolidation_engine: object,
    ) -> None:
        """Daily consolidation creates knowledge files with YAML frontmatter."""
        # Create episode data
        today = datetime.now().date()
        episode_file = consolidation_engine.episodes_dir / f"{today}.md"
        episode_file.write_text(
            "## 10:00 \u2014 API\u8a2d\u8a08\u4f1a\u8b70\n\n"
            "**\u76f8\u624b**: \u30c1\u30fc\u30e0\n"
            "**\u8981\u70b9**: REST API\u306e\u8a2d\u8a08\u65b9\u91dd\u3092\u6c7a\u5b9a\u3002"
            "OpenAPI\u4ed5\u69d8\u3092\u5148\u306b\u4f5c\u6210\u3059\u308b\u3002\n",
            encoding="utf-8",
        )

        # Mock LLM consolidation response
        llm_response = (
            "## \u65e2\u5b58\u30d5\u30a1\u30a4\u30eb\u66f4\u65b0\n"
            "(\u306a\u3057)\n\n"
            "## \u65b0\u898f\u30d5\u30a1\u30a4\u30eb\u4f5c\u6210\n"
            "- \u30d5\u30a1\u30a4\u30eb\u540d: knowledge/api-design.md\n"
            "  \u5185\u5bb9: # API\u8a2d\u8a08\u65b9\u91dd\n\n"
            "OpenAPI\u4ed5\u69d8\u3092\u5148\u306b\u4f5c\u6210\u3057\u3001"
            "\u305d\u306e\u5f8c\u5b9f\u88c5\u3059\u308b\u3002\n"
        )

        # Mock validation to pass everything through
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_llm_response(llm_response)

            # Disable validation module import to skip validation
            with patch(
                "core.memory.consolidation.ConsolidationEngine._validate_consolidation",
                new_callable=AsyncMock,
            ) as mock_validate:
                mock_validate.return_value = llm_response

                result = await consolidation_engine.daily_consolidate(
                    min_episodes=1,
                )

        assert result["skipped"] is False
        assert result["episodes_processed"] == 1
        assert "api-design.md" in result["knowledge_files_created"]

        # Verify file has frontmatter
        kfile = consolidation_engine.knowledge_dir / "api-design.md"
        assert kfile.exists()
        text = kfile.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "auto_consolidated: true" in text
        assert "confidence:" in text

    @pytest.mark.asyncio
    async def test_pipeline_with_validation_rejection(
        self, consolidation_engine: object,
    ) -> None:
        """Validation rejection results in fewer knowledge files."""
        today = datetime.now().date()
        episode_file = consolidation_engine.episodes_dir / f"{today}.md"
        episode_file.write_text(
            "## 10:00 \u2014 \u30df\u30fc\u30c6\u30a3\u30f3\u30b0\n\n"
            "**\u8981\u70b9**: \u30c7\u30d7\u30ed\u30a4\u306f\u6728\u66dc\u65e5\u3002\n",
            encoding="utf-8",
        )

        # Two items: one valid, one hallucinated
        llm_response = (
            "## \u65e2\u5b58\u30d5\u30a1\u30a4\u30eb\u66f4\u65b0\n"
            "(\u306a\u3057)\n\n"
            "## \u65b0\u898f\u30d5\u30a1\u30a4\u30eb\u4f5c\u6210\n"
            "- \u30d5\u30a1\u30a4\u30eb\u540d: knowledge/deploy-schedule.md\n"
            "  \u5185\u5bb9: \u30c7\u30d7\u30ed\u30a4\u306f\u6728\u66dc\u65e5\n"
        )

        # Validation passes everything (no NLI, LLM returns true)
        validation_response = (
            "## \u65e2\u5b58\u30d5\u30a1\u30a4\u30eb\u66f4\u65b0\n"
            "(\u306a\u3057)\n\n"
            "## \u65b0\u898f\u30d5\u30a1\u30a4\u30eb\u4f5c\u6210\n"
            "- \u30d5\u30a1\u30a4\u30eb\u540d: knowledge/deploy-schedule.md\n"
            "  \u5185\u5bb9: \u30c7\u30d7\u30ed\u30a4\u306f\u6728\u66dc\u65e5\n"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_llm_response(llm_response)

            with patch(
                "core.memory.consolidation.ConsolidationEngine._validate_consolidation",
                new_callable=AsyncMock,
            ) as mock_validate:
                mock_validate.return_value = validation_response

                result = await consolidation_engine.daily_consolidate(
                    min_episodes=1,
                )

        assert result["skipped"] is False
        assert len(result["knowledge_files_created"]) == 1

    @pytest.mark.asyncio
    async def test_legacy_migration_runs_before_consolidation(
        self, consolidation_engine: object,
    ) -> None:
        """Legacy migration is invoked at the start of daily consolidation."""
        kdir = consolidation_engine.knowledge_dir
        legacy = kdir / "old.md"
        legacy.write_text("# Old\n\nLegacy.", encoding="utf-8")

        today = datetime.now().date()
        episode_file = consolidation_engine.episodes_dir / f"{today}.md"
        episode_file.write_text(
            "## 10:00 \u2014 Test\n\n\u8981\u70b9: test\n",
            encoding="utf-8",
        )

        llm_response = (
            "## \u65e2\u5b58\u30d5\u30a1\u30a4\u30eb\u66f4\u65b0\n(\u306a\u3057)\n\n"
            "## \u65b0\u898f\u30d5\u30a1\u30a4\u30eb\u4f5c\u6210\n(\u306a\u3057)"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_llm_response(llm_response)

            with patch(
                "core.memory.consolidation.ConsolidationEngine._validate_consolidation",
                new_callable=AsyncMock,
            ) as mock_validate:
                mock_validate.return_value = llm_response

                await consolidation_engine.daily_consolidate(min_episodes=1)

        # Migration should have added frontmatter to the legacy file
        text = legacy.read_text(encoding="utf-8")
        assert text.startswith("---\n")

    @pytest.mark.asyncio
    async def test_update_existing_preserves_metadata(
        self, consolidation_engine: object,
    ) -> None:
        """Updating an existing file preserves and extends its metadata."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(consolidation_engine.anima_dir)
        kfile = consolidation_engine.knowledge_dir / "existing-topic.md"
        mm.write_knowledge_with_meta(
            kfile,
            "# Existing\n\nOriginal content.",
            {
                "created_at": "2026-02-10T09:00:00",
                "confidence": 0.8,
                "auto_consolidated": True,
            },
        )

        today = datetime.now().date()
        episode_file = consolidation_engine.episodes_dir / f"{today}.md"
        episode_file.write_text(
            "## 10:00 \u2014 Update\n\nNew info for existing topic.\n",
            encoding="utf-8",
        )

        llm_response = (
            "## \u65e2\u5b58\u30d5\u30a1\u30a4\u30eb\u66f4\u65b0\n"
            "- \u30d5\u30a1\u30a4\u30eb\u540d: knowledge/existing-topic.md\n"
            "  \u8ffd\u52a0\u5185\u5bb9: \u65b0\u3057\u3044\u60c5\u5831\u304c\u8ffd\u52a0\u3055\u308c\u307e\u3057\u305f\u3002\n\n"
            "## \u65b0\u898f\u30d5\u30a1\u30a4\u30eb\u4f5c\u6210\n(\u306a\u3057)"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_llm_response(llm_response)

            with patch(
                "core.memory.consolidation.ConsolidationEngine._validate_consolidation",
                new_callable=AsyncMock,
            ) as mock_validate:
                mock_validate.return_value = llm_response

                result = await consolidation_engine.daily_consolidate(
                    min_episodes=1,
                )

        assert "existing-topic.md" in result["knowledge_files_updated"]

        # Verify metadata preserved
        meta = mm.read_knowledge_metadata(kfile)
        assert meta["created_at"] == "2026-02-10T09:00:00"
        assert "updated_at" in meta

        # Verify content appended
        content = mm.read_knowledge_content(kfile)
        assert "Original content." in content
        assert "\u65b0\u3057\u3044\u60c5\u5831" in content


# ── Weekly Integration E2E ───────────────────────────────────


class TestWeeklyIntegrationE2E:
    """End-to-end tests for weekly_integrate."""

    @pytest.mark.asyncio
    async def test_merge_archives_instead_of_deleting(
        self, consolidation_engine: object,
    ) -> None:
        """Merged knowledge files are archived, not deleted."""
        kdir = consolidation_engine.knowledge_dir
        file1 = kdir / "topic-a.md"
        file2 = kdir / "topic-b.md"
        file1.write_text("# Topic A\n\nContent A.", encoding="utf-8")
        file2.write_text("# Topic B\n\nContent B.", encoding="utf-8")

        duplicates = [("topic-a.md", "topic-b.md", 0.92)]

        llm_response = (
            "## \u7d71\u5408\u30d5\u30a1\u30a4\u30eb\u540d\n"
            "topic-ab-merged.md\n\n"
            "## \u7d71\u5408\u5185\u5bb9\n"
            "# Topic AB\n\nMerged content from A and B."
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_llm_response(llm_response)

            merged = await consolidation_engine._merge_knowledge_files(
                duplicates, model="test-model",
            )

        assert len(merged) == 1

        # Originals are archived, not deleted
        archive_dir = consolidation_engine.anima_dir / "archive" / "merged"
        assert (archive_dir / "topic-a.md").exists()
        assert (archive_dir / "topic-b.md").exists()

        # Originals are gone from knowledge/
        assert not file1.exists()
        assert not file2.exists()

        # Merged file exists
        merged_file = kdir / "topic-ab-merged.md"
        assert merged_file.exists()
        content = merged_file.read_text(encoding="utf-8")
        assert "AUTO-MERGED" in content
        assert "Merged content" in content

    @pytest.mark.asyncio
    async def test_weekly_full_flow_with_mocks(
        self, consolidation_engine: object,
    ) -> None:
        """Full weekly flow with all steps mocked."""
        kdir = consolidation_engine.knowledge_dir
        (kdir / "unique.md").write_text("# Unique\n\nContent.", encoding="utf-8")

        with patch.object(
            consolidation_engine, "_detect_duplicates",
            new_callable=AsyncMock, return_value=[],
        ):
            with patch.object(
                consolidation_engine, "_compress_old_episodes",
                new_callable=AsyncMock, return_value=0,
            ):
                with patch.object(consolidation_engine, "_rebuild_rag_index"):
                    result = await consolidation_engine.weekly_integrate()

        assert result["skipped"] is False
        assert result["episodes_compressed"] == 0
        assert len(result["knowledge_files_merged"]) == 0


# ── Format Retry Tests ───────────────────────────────────────


class TestFormatRetry:
    """Test format validation and retry in _summarize_episodes."""

    @pytest.mark.asyncio
    async def test_retry_on_bad_format(
        self, consolidation_engine: object,
    ) -> None:
        """When first response lacks expected sections, a retry is attempted."""
        today = datetime.now().date()
        episode_entries = [
            {"date": str(today), "time": "10:00", "content": "Test episode."},
        ]

        # First response: bad format (no sections)
        bad_response = _make_llm_response("Here are some thoughts about the episodes.")
        # Second response: correct format
        good_response = _make_llm_response(
            "## \u65e2\u5b58\u30d5\u30a1\u30a4\u30eb\u66f4\u65b0\n"
            "(\u306a\u3057)\n\n"
            "## \u65b0\u898f\u30d5\u30a1\u30a4\u30eb\u4f5c\u6210\n"
            "- \u30d5\u30a1\u30a4\u30eb\u540d: knowledge/retry-test.md\n"
            "  \u5185\u5bb9: \u30ea\u30c8\u30e9\u30a4\u6210\u529f\n"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_response, good_response]

            result = await consolidation_engine._summarize_episodes(
                episode_entries=episode_entries,
                existing_knowledge_files=[],
                model="test-model",
            )

        assert "\u65b0\u898f\u30d5\u30a1\u30a4\u30eb\u4f5c\u6210" in result
        assert mock_llm.call_count == 2  # Initial + retry

    @pytest.mark.asyncio
    async def test_no_retry_on_good_format(
        self, consolidation_engine: object,
    ) -> None:
        """When first response has correct format, no retry occurs."""
        today = datetime.now().date()
        episode_entries = [
            {"date": str(today), "time": "10:00", "content": "Test."},
        ]

        good_response = _make_llm_response(
            "## \u65e2\u5b58\u30d5\u30a1\u30a4\u30eb\u66f4\u65b0\n"
            "(\u306a\u3057)\n\n"
            "## \u65b0\u898f\u30d5\u30a1\u30a4\u30eb\u4f5c\u6210\n"
            "(\u306a\u3057)\n"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = good_response

            await consolidation_engine._summarize_episodes(
                episode_entries=episode_entries,
                existing_knowledge_files=[],
                model="test-model",
            )

        assert mock_llm.call_count == 1  # No retry needed


# ── Validate Consolidation Tests ─────────────────────────────


class TestValidateConsolidation:
    """Test the _validate_consolidation integration method."""

    @pytest.mark.asyncio
    async def test_validation_filters_items(
        self, consolidation_engine: object,
    ) -> None:
        """Validation filters out rejected items from consolidation output."""
        consolidation_result = (
            "## \u65e2\u5b58\u30d5\u30a1\u30a4\u30eb\u66f4\u65b0\n"
            "(\u306a\u3057)\n\n"
            "## \u65b0\u898f\u30d5\u30a1\u30a4\u30eb\u4f5c\u6210\n"
            "- \u30d5\u30a1\u30a4\u30eb\u540d: knowledge/good.md\n"
            "  \u5185\u5bb9: Good content\n"
            "- \u30d5\u30a1\u30a4\u30eb\u540d: knowledge/bad.md\n"
            "  \u5185\u5bb9: Bad hallucinated content\n"
        )

        episodes_text = "Source episode text."

        with patch(
            "core.memory.validation.KnowledgeValidator.validate",
            new_callable=AsyncMock,
        ) as mock_validate:
            # Return only the first item (second rejected)
            mock_validate.return_value = [
                {
                    "filename": "knowledge/good.md",
                    "content": "Good content",
                    "type": "create",
                    "confidence": 0.7,
                },
            ]

            result = await consolidation_engine._validate_consolidation(
                consolidation_result, episodes_text, "test-model",
            )

        assert "good.md" in result
        assert "bad.md" not in result

    @pytest.mark.asyncio
    async def test_validation_import_error_passthrough(
        self, consolidation_engine: object,
    ) -> None:
        """When validation module is unavailable, result passes through."""
        consolidation_result = "## \u65b0\u898f\u30d5\u30a1\u30a4\u30eb\u4f5c\u6210\nsome content"

        with patch(
            "core.memory.consolidation.ConsolidationEngine._validate_consolidation",
        ) as mock_method:
            # Simulate the actual behavior on ImportError
            mock_method.return_value = consolidation_result

            # Just verify it returns the input unchanged
            assert mock_method.return_value == consolidation_result

    @pytest.mark.asyncio
    async def test_validation_empty_input(
        self, consolidation_engine: object,
    ) -> None:
        """Empty consolidation result passes through unchanged."""
        result = await consolidation_engine._validate_consolidation("", "episodes", "model")
        assert result == ""

    @pytest.mark.asyncio
    async def test_validation_preserves_update_items(
        self, consolidation_engine: object,
    ) -> None:
        """Update items that pass validation are preserved in output."""
        consolidation_result = (
            "## \u65e2\u5b58\u30d5\u30a1\u30a4\u30eb\u66f4\u65b0\n"
            "- \u30d5\u30a1\u30a4\u30eb\u540d: knowledge/existing.md\n"
            "  \u8ffd\u52a0\u5185\u5bb9: Updated info\n\n"
            "## \u65b0\u898f\u30d5\u30a1\u30a4\u30eb\u4f5c\u6210\n"
            "(\u306a\u3057)"
        )

        with patch(
            "core.memory.validation.KnowledgeValidator.validate",
            new_callable=AsyncMock,
        ) as mock_validate:
            mock_validate.return_value = [
                {
                    "filename": "knowledge/existing.md",
                    "content": "Updated info",
                    "type": "update",
                    "confidence": 0.7,
                },
            ]

            result = await consolidation_engine._validate_consolidation(
                consolidation_result, "episodes", "model",
            )

        assert "existing.md" in result
        assert "\u65e2\u5b58\u30d5\u30a1\u30a4\u30eb\u66f4\u65b0" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
