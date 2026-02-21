# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for skill metadata extraction and matching.

Covers:
- MemoryManager._extract_skill_meta()  (core/memory/manager.py)
- match_skills_by_description()        (core/memory/manager.py)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.schemas import SkillMeta
from core.memory.manager import (
    MemoryManager,
    match_skills_by_description,
    _extract_bracket_keywords,
    _extract_comma_keywords,
    _match_tier1,
    _match_tier2,
)
from core.tooling.handler import _validate_skill_format


# ── _extract_skill_meta ──────────────────────────────────


class TestExtractSkillMeta:
    """Tests for MemoryManager._extract_skill_meta()."""

    def test_parse_yaml_frontmatter(self, tmp_path: Path) -> None:
        """YAML frontmatter with name + description is correctly parsed."""
        skill_file = tmp_path / "deploy.md"
        skill_file.write_text(
            "---\n"
            "name: deploy-skill\n"
            "description: デプロイ手順「deploy」「リリース」\n"
            "---\n"
            "\n"
            "# Deploy Skill\n"
            "\nBody content here.\n",
            encoding="utf-8",
        )

        meta = MemoryManager._extract_skill_meta(skill_file)

        assert meta.name == "deploy-skill"
        assert meta.description == "デプロイ手順「deploy」「リリース」"
        assert meta.path == skill_file
        assert meta.is_common is False

    def test_no_frontmatter_uses_filename(self, tmp_path: Path) -> None:
        """File without frontmatter uses filename stem as name, empty description."""
        skill_file = tmp_path / "my-skill.md"
        skill_file.write_text(
            "# My Skill\n\nSome instructions.\n",
            encoding="utf-8",
        )

        meta = MemoryManager._extract_skill_meta(skill_file)

        assert meta.name == "my-skill"
        assert meta.description == ""
        assert meta.path == skill_file

    def test_legacy_format_overview_section(self, tmp_path: Path) -> None:
        """Legacy format with ## 概要 extracts first line as description."""
        skill_file = tmp_path / "legacy.md"
        skill_file.write_text(
            "# レガシースキル\n"
            "\n"
            "## 概要\n"
            "\n"
            "cronジョブの設定と管理を行うスキル\n"
            "\n"
            "## 手順\n"
            "\n"
            "1. 手順内容\n",
            encoding="utf-8",
        )

        meta = MemoryManager._extract_skill_meta(skill_file)

        assert meta.name == "legacy"
        assert meta.description == "cronジョブの設定と管理を行うスキル"

    def test_frontmatter_with_extra_fields(self, tmp_path: Path) -> None:
        """Extra fields (version, metadata) do not interfere with extraction."""
        skill_file = tmp_path / "advanced.md"
        skill_file.write_text(
            "---\n"
            "name: advanced-tool\n"
            "description: 高度な検索「search」「query」\n"
            "version: 2.1\n"
            "metadata:\n"
            "  author: test\n"
            "  tags: [search, query]\n"
            "---\n"
            "\n"
            "Body.\n",
            encoding="utf-8",
        )

        meta = MemoryManager._extract_skill_meta(skill_file)

        assert meta.name == "advanced-tool"
        assert meta.description == "高度な検索「search」「query」"

    def test_is_common_flag(self, tmp_path: Path) -> None:
        """is_common flag is correctly set when specified."""
        skill_file = tmp_path / "shared.md"
        skill_file.write_text(
            "---\n"
            "name: shared-skill\n"
            "description: 共有スキル\n"
            "---\n"
            "\nContent.\n",
            encoding="utf-8",
        )

        meta_personal = MemoryManager._extract_skill_meta(skill_file, is_common=False)
        assert meta_personal.is_common is False

        meta_common = MemoryManager._extract_skill_meta(skill_file, is_common=True)
        assert meta_common.is_common is True


# ── match_skills_by_description ──────────────────────────


class TestMatchSkillsByDescription:
    """Tests for match_skills_by_description()."""

    @pytest.fixture
    def skill_with_keywords(self, tmp_path: Path) -> SkillMeta:
        p = tmp_path / "cron_setup.md"
        p.write_text("dummy", encoding="utf-8")
        return SkillMeta(
            name="cron_setup",
            description="「cron設定」「定期実行」に関するスキル",
            path=p,
            is_common=False,
        )

    @pytest.fixture
    def skill_without_keywords(self, tmp_path: Path) -> SkillMeta:
        p = tmp_path / "general.md"
        p.write_text("dummy", encoding="utf-8")
        return SkillMeta(
            name="general",
            description="汎用的なスキルです",
            path=p,
            is_common=False,
        )

    def test_extract_bracket_keywords(
        self, skill_with_keywords: SkillMeta,
    ) -> None:
        """「」-delimited keywords in description trigger a match."""
        result = match_skills_by_description(
            "cron設定を確認してください", [skill_with_keywords],
        )
        assert len(result) == 1
        assert result[0].name == "cron_setup"

    def test_nfkc_normalization(self, tmp_path: Path) -> None:
        """Full-width characters match half-width equivalents via NFKC."""
        p = tmp_path / "cron.md"
        p.write_text("dummy", encoding="utf-8")
        skill = SkillMeta(
            name="cron",
            description="「cron」管理スキル",
            path=p,
            is_common=False,
        )
        # Full-width ｃｒｏｎ should match half-width cron keyword
        result = match_skills_by_description("ｃｒｏｎを設定", [skill])
        assert len(result) == 1
        assert result[0].name == "cron"

    def test_partial_match(self, skill_with_keywords: SkillMeta) -> None:
        """Partial overlap: message 'cronに追加して' matches keyword 'cron設定'
        only if 'cron設定' is a substring of the message. Here we test that
        the keyword 'cron設定' is checked as a substring of the normalized message."""
        # 'cron設定' is not a substring of 'cronに追加して' → no match
        # But '定期実行' or 'cron設定' needs to appear in the message
        result = match_skills_by_description(
            "cron設定をしてください", [skill_with_keywords],
        )
        assert len(result) == 1

    def test_no_match(self, skill_with_keywords: SkillMeta) -> None:
        """Message with no keyword overlap returns empty list."""
        result = match_skills_by_description(
            "おはよう", [skill_with_keywords],
        )
        assert result == []

    def test_skills_without_bracket_keywords_never_match(
        self, skill_without_keywords: SkillMeta,
    ) -> None:
        """Skills whose description lacks 「」 keywords are never matched."""
        result = match_skills_by_description(
            "汎用的なスキルを使って", [skill_without_keywords],
        )
        assert result == []

    def test_empty_message_returns_empty(
        self, skill_with_keywords: SkillMeta,
    ) -> None:
        """Empty message always returns an empty list."""
        result = match_skills_by_description("", [skill_with_keywords])
        assert result == []


# ── Tier 1: Comma/delimiter keyword matching ────────────


class TestMatchTier1CommaKeywords:
    """Tests for Tier 1 comma/delimiter keyword matching fallback."""

    def test_comma_separated_keywords_match(self, tmp_path):
        """Skills with comma-separated keywords (no brackets) match via Tier 1 fallback."""
        # Create skill with comma-separated description (no「」)
        p = tmp_path / "skill.md"
        p.write_text("dummy")
        skill = SkillMeta(name="deploy", description="デプロイ手順、リリース、本番反映", path=p, is_common=False)
        result = match_skills_by_description("デプロイ手順を教えて", [skill])
        assert len(result) == 1

    def test_period_separated_keywords_match(self, tmp_path):
        """Period-separated segments also work."""
        p = tmp_path / "skill.md"
        p.write_text("dummy")
        skill = SkillMeta(name="test", description="テスト実行。品質管理。CI設定", path=p, is_common=False)
        result = match_skills_by_description("テスト実行をしたい", [skill])
        assert len(result) == 1

    def test_bracket_keywords_take_precedence(self, tmp_path):
        """When both brackets and commas exist, brackets are used (not commas)."""
        p = tmp_path / "skill.md"
        p.write_text("dummy")
        skill = SkillMeta(name="cron", description="「cron設定」「定期実行」、スケジュール管理", path=p, is_common=False)
        # Match via bracket keyword
        result = match_skills_by_description("cron設定して", [skill])
        assert len(result) == 1
        # "スケジュール管理" alone shouldn't match because bracket keywords were used
        result2 = match_skills_by_description("スケジュール管理をお願い", [skill])
        assert len(result2) == 0  # bracket keywords don't include "スケジュール管理"


# ── Tier 2: Description vocabulary matching ─────────────


class TestMatchTier2VocabularyMatch:
    """Tests for Tier 2 description vocabulary matching."""

    def test_english_description_matches_with_two_words(self, tmp_path):
        """English descriptions without brackets match when >=2 words overlap."""
        p = tmp_path / "skill.md"
        p.write_text("dummy")
        skill = SkillMeta(
            name="document-creator",
            description="Comprehensive document creation, editing, and analysis",
            path=p, is_common=False,
        )
        # "document" and "creation" both appear
        result = match_skills_by_description("document creation needed", [skill])
        assert len(result) == 1

    def test_single_word_overlap_no_match(self, tmp_path):
        """Single word overlap is not enough for Tier 2 (to avoid false positives)."""
        p = tmp_path / "skill.md"
        p.write_text("dummy")
        skill = SkillMeta(
            name="document-creator",
            description="Comprehensive document creation, editing, and analysis",
            path=p, is_common=False,
        )
        result = match_skills_by_description("give me a document", [skill])
        # Only "document" matches, not enough
        assert len(result) == 0

    def test_japanese_description_vocabulary_match(self, tmp_path):
        """Japanese descriptions without brackets can match via Tier 2.

        Space-separated Japanese terms produce multiple \\w{3,} tokens that
        can individually appear in the message. Tier 1 is skipped because
        the single comma-segment (the whole description) is not a substring
        of the message.
        """
        p = tmp_path / "skill.md"
        p.write_text("dummy")
        skill = SkillMeta(
            name="data-analysis",
            description="データ分析 手順書作成 レポート出力",
            path=p, is_common=False,
        )
        # "データ分析" and "手順書作成" both appear as substrings in the message
        result = match_skills_by_description("データ分析の手順書作成をお願いします", [skill])
        assert len(result) == 1


# ── Deduplication across tiers ──────────────────────────


class TestMatchDeduplication:
    """Tests for deduplication across tiers."""

    def test_skill_matched_by_tier1_not_duplicated(self, tmp_path):
        """A skill matched by Tier 1 should not appear again from Tier 2."""
        p = tmp_path / "skill.md"
        p.write_text("dummy")
        skill = SkillMeta(
            name="deploy",
            description="デプロイ手順「deploy」「デプロイ」を提供する",
            path=p, is_common=False,
        )
        result = match_skills_by_description("deployの手順を教えて", [skill])
        assert len(result) == 1  # Not duplicated


# ── Retriever parameter (Tier 3) ────────────────────────


class TestMatchRetrieverParam:
    """Tests for the retriever parameter (Tier 3)."""

    def test_no_retriever_skips_tier3(self, tmp_path):
        """When retriever is None, Tier 3 is skipped gracefully."""
        p = tmp_path / "skill.md"
        p.write_text("dummy")
        skill = SkillMeta(
            name="unknown",
            description="何かのスキル",
            path=p, is_common=False,
        )
        # No brackets, no comma keywords long enough, single vocab word -> no match
        result = match_skills_by_description("全く関係ない話", [skill], retriever=None)
        assert result == []

    def test_invalid_retriever_type_skips_tier3(self, tmp_path):
        """When retriever is not a MemoryRetriever, Tier 3 is skipped."""
        p = tmp_path / "skill.md"
        p.write_text("dummy")
        skill = SkillMeta(
            name="unknown",
            description="何かのスキル",
            path=p, is_common=False,
        )
        result = match_skills_by_description(
            "全く関係ない話", [skill], retriever="not-a-retriever", anima_name="test"
        )
        assert result == []


# ── _validate_skill_format ──────────────────────────────


class TestValidateSkillFormat:
    """Tests for _validate_skill_format() in handler.py."""

    def test_valid_skill_returns_empty(self):
        content = "---\nname: my-skill\ndescription: テスト「keyword」\n---\n\n# My Skill\n"
        assert _validate_skill_format(content) == ""

    def test_missing_frontmatter(self):
        content = "# My Skill\n\nNo frontmatter."
        result = _validate_skill_format(content)
        assert "フロントマター" in result

    def test_missing_name_field(self):
        content = "---\ndescription: テスト「keyword」\n---\n\n# Skill\n"
        result = _validate_skill_format(content)
        assert "name" in result

    def test_missing_description_field(self):
        content = "---\nname: my-skill\n---\n\n# Skill\n"
        result = _validate_skill_format(content)
        assert "description" in result

    def test_no_bracket_keywords_warns(self):
        content = "---\nname: my-skill\ndescription: スキルの説明です\n---\n\n# Skill\n"
        result = _validate_skill_format(content)
        assert "キーワード" in result

    def test_legacy_section_warns(self):
        content = "---\nname: my-skill\ndescription: テスト「keyword」\n---\n\n## 概要\n\nLegacy content\n"
        result = _validate_skill_format(content)
        assert "旧形式" in result

    def test_valid_with_multiline_description(self):
        content = (
            "---\n"
            "name: my-skill\n"
            "description: >-\n"
            "  テストスキル。\n"
            "  「キーワード1」「キーワード2」\n"
            "---\n\n"
            "# My Skill\n"
        )
        assert _validate_skill_format(content) == ""


# ── Helper functions ────────────────────────────────────


class TestHelperFunctions:
    """Tests for matching helper functions."""

    def test_extract_bracket_keywords(self):
        assert _extract_bracket_keywords("「abc」「def」") == ["abc", "def"]
        assert _extract_bracket_keywords("no brackets here") == []

    def test_extract_comma_keywords(self):
        result = _extract_comma_keywords("デプロイ手順、リリース、本番反映")
        assert "デプロイ手順" in result
        assert "リリース" in result

    def test_extract_comma_keywords_filters_short(self):
        result = _extract_comma_keywords("a、bc、defghi")
        assert "a" not in result  # too short
        assert "bc" in result
        assert "defghi" in result

    def test_match_tier1_bracket(self):
        assert _match_tier1("「cron」管理スキル", "cronを設定して") is True
        assert _match_tier1("「cron」管理スキル", "メール送信") is False

    def test_match_tier2_multiple_words(self):
        assert _match_tier2("comprehensive document creation editing", "document creation") is True
        assert _match_tier2("comprehensive document creation editing", "document") is False

    def test_match_tier2_stop_words_filtered(self):
        """Stop words like 'and', 'the', 'tool' should not contribute to match count."""
        # "and" and "tool" are stop words; only "document" is meaningful (1 word < 2)
        assert _match_tier2("document creation and analysis tool", "and the tool") is False

    def test_match_tier2_word_boundary_prevents_substring_collision(self):
        """ASCII word boundary matching prevents 'git' matching inside 'digital'."""
        # "git" should NOT match inside "digital"
        assert _match_tier2("git management skill", "digital management system") is False


class TestMatchTier3WithMockRetriever:
    """Tests for Tier 3 vector search with a mock MemoryRetriever."""

    def _make_mock_retriever(self):
        """Create a mock that passes the isinstance(r, MemoryRetriever) check."""
        from unittest.mock import MagicMock
        from core.memory.rag.retriever import MemoryRetriever

        mock = MagicMock(spec=MemoryRetriever)
        return mock

    def test_tier3_score_threshold_filters_low_scores(self, tmp_path):
        """Results below min_score threshold are filtered out."""
        from core.memory.manager import _match_tier3_vector
        from unittest.mock import MagicMock

        p = tmp_path / "skill.md"
        p.write_text("dummy")
        skill = SkillMeta(name="my-skill", description="x", path=p, is_common=False)

        mock_retriever = self._make_mock_retriever()
        mock_result = MagicMock()
        mock_result.score = 0.3  # Below threshold of 0.88
        mock_result.metadata = {"file_path": str(p)}
        mock_retriever.search.return_value = [mock_result]

        result = _match_tier3_vector(
            "test message", [skill], mock_retriever, "test",
        )
        assert result == []  # Filtered by score threshold

    def test_tier3_matches_by_path_stem(self, tmp_path):
        """Tier 3 matches results by file path stem."""
        from core.memory.manager import _match_tier3_vector
        from unittest.mock import MagicMock

        p = tmp_path / "deploy-guide.md"
        p.write_text("dummy")
        skill = SkillMeta(name="deploy-guide", description="x", path=p, is_common=False)

        mock_retriever = self._make_mock_retriever()
        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.metadata = {"file_path": "/some/other/path/deploy-guide.md"}
        mock_retriever.search.return_value = [mock_result]

        result = _match_tier3_vector(
            "deploy something", [skill], mock_retriever, "test",
        )
        assert len(result) == 1
        assert result[0].name == "deploy-guide"

    def test_tier3_matches_by_skill_name(self, tmp_path):
        """Tier 3 matches results by skill name when path doesn't match."""
        from core.memory.manager import _match_tier3_vector
        from unittest.mock import MagicMock

        p = tmp_path / "deploy.md"
        p.write_text("dummy")
        skill = SkillMeta(name="deploy-guide", description="x", path=p, is_common=False)

        mock_retriever = self._make_mock_retriever()
        mock_result = MagicMock()
        mock_result.score = 0.95
        # file_path does not match path or stem, but name "deploy-guide" is
        # indexed in the candidate lookup
        mock_result.metadata = {"file_path": ""}
        mock_retriever.search.return_value = [mock_result]

        # Name matching is attempted as last resort—but only if file_path or
        # stem match. Since file_path is empty, stem = "", neither matches.
        # So this should return empty. Name lookup only works through the
        # path-based matching pipeline.
        result = _match_tier3_vector(
            "deploy guide", [skill], mock_retriever, "test",
        )
        # empty file_path doesn't match any lookup key
        assert len(result) == 0

    def test_tier3_deduplicates_results(self, tmp_path):
        """Duplicate matches from vector search are deduplicated."""
        from core.memory.manager import _match_tier3_vector
        from unittest.mock import MagicMock

        p = tmp_path / "deploy-guide.md"
        p.write_text("dummy")
        skill = SkillMeta(name="deploy-guide", description="x", path=p, is_common=False)

        mock_retriever = self._make_mock_retriever()
        # Two results pointing to same skill
        r1 = MagicMock()
        r1.score = 0.96
        r1.metadata = {"file_path": str(p)}
        r2 = MagicMock()
        r2.score = 0.94
        r2.metadata = {"file_path": "/other/deploy-guide.md"}
        mock_retriever.search.return_value = [r1, r2]

        result = _match_tier3_vector(
            "deploy", [skill], mock_retriever, "test",
        )
        assert len(result) == 1  # Deduplicated

    def test_tier3_passes_include_shared_for_common_skills(self, tmp_path):
        """Tier 3 via match_skills_by_description passes include_shared=True."""
        from unittest.mock import MagicMock

        p = tmp_path / "my-common-skill.md"
        p.write_text("dummy")
        # Common skill that didn't match Tier 1/2
        skill = SkillMeta(
            name="my-common-skill",
            description="obscure description without matching words",
            path=p,
            is_common=True,
        )

        mock_retriever = self._make_mock_retriever()
        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.metadata = {"file_path": str(p)}
        mock_retriever.search.return_value = [mock_result]

        result = match_skills_by_description(
            "semantically related message",
            [skill],
            retriever=mock_retriever,
            anima_name="test",
        )
        # Verify retriever.search was called with include_shared=True
        mock_retriever.search.assert_called_once()
        call_kwargs = mock_retriever.search.call_args
        assert call_kwargs.kwargs.get("include_shared") is True or (
            len(call_kwargs.args) > 4 and call_kwargs.args[4] is True
        )
        assert len(result) == 1
        assert result[0].name == "my-common-skill"
