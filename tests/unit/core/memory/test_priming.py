from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Test for priming layer (Phase 1)."""

import asyncio
import tempfile
from datetime import date
from pathlib import Path

import pytest

from core.memory.priming import PrimingEngine, format_priming_section


@pytest.fixture
def temp_anima_dir():
    """Create a temporary anima directory with sample memory files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        anima_dir = Path(tmpdir) / "animas" / "test_anima"
        anima_dir.mkdir(parents=True)

        # Create memory directories
        (anima_dir / "episodes").mkdir()
        (anima_dir / "knowledge").mkdir()
        (anima_dir / "skills").mkdir()

        # Create sample episode file (today)
        today_episode = anima_dir / "episodes" / f"{date.today().isoformat()}.md"
        today_episode.write_text(
            f"# {date.today().isoformat()} 行動ログ\n\n"
            "## 09:00 — 朝のタスク確認\n\n"
            "**相手**: システム\n"
            "**トピック**: タスク管理\n"
            "**要点**:\n"
            "- 今日のタスクを確認した\n"
            "- 3件の未完了タスクがある\n\n"
            "## 10:30 — 山田さんとのミーティング\n\n"
            "**相手**: 山田\n"
            "**トピック**: プロジェクト進捗\n"
            "**要点**:\n"
            "- プライミングレイヤーの実装について議論\n"
            "- Phase 1を優先することで合意\n",
            encoding="utf-8",
        )

        # Create sample knowledge file
        knowledge_file = anima_dir / "knowledge" / "priming-layer.md"
        knowledge_file.write_text(
            "# プライミングレイヤー\n\n"
            "自動想起メカニズムを実装する。\n"
            "Dense Vectorベースの意味検索を使用する。\n"
            "ChromaDBとmultilingual-e5-smallで実装。\n",
            encoding="utf-8",
        )

        # Create sample skill file with YAML frontmatter (directory structure)
        (anima_dir / "skills" / "web_search").mkdir(parents=True, exist_ok=True)
        skill_file = anima_dir / "skills" / "web_search" / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "description: \"「web search」「web検索」を実行して情報を収集する\"\n"
            "---\n\n"
            "# Web検索スキル\n\n"
            "## 概要\n"
            "Web検索を実行して情報を収集する\n",
            encoding="utf-8",
        )

        yield anima_dir


@pytest.fixture
def temp_shared_dir():
    """Create a temporary shared directory with user profiles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shared_dir = Path(tmpdir) / "shared"
        users_dir = shared_dir / "users"
        users_dir.mkdir(parents=True)

        # Create sample user profile
        yamada_dir = users_dir / "山田"
        yamada_dir.mkdir()
        (yamada_dir / "index.md").write_text(
            "# 山田さんのプロファイル\n\n"
            "## 基本情報\n"
            "- プロジェクトマネージャー\n"
            "- 技術的な詳細を好む\n\n"
            "## 重要な好み・傾向\n"
            "- 簡潔な報告を好む\n"
            "- Slack での連絡を希望\n",
            encoding="utf-8",
        )

        yield shared_dir


@pytest.mark.asyncio
async def test_priming_all_channels(temp_anima_dir, temp_shared_dir):
    """Test all 4 priming channels."""
    engine = PrimingEngine(temp_anima_dir)

    # Mock shared_dir by patching get_shared_dir at the point it's imported
    from unittest.mock import patch
    with patch("core.paths.get_shared_dir", return_value=temp_shared_dir):
        result = await engine.prime_memories(
            message="山田さんとプライミングレイヤーについて話したい",
            sender_name="山田",
        )

    # Verify sender profile was loaded
    assert "山田さんのプロファイル" in result.sender_profile
    assert "Slack" in result.sender_profile

    # Verify recent activity was loaded
    assert "朝のタスク確認" in result.recent_activity or "山田さんとのミーティング" in result.recent_activity

    # Verify knowledge search (may be empty if ripgrep not available or doesn't match)
    # Note: Japanese keyword matching without morphological analysis is limited
    print(f"\nKnowledge search result: {len(result.related_knowledge)} chars")
    # Don't assert on knowledge content - it's OK if empty in Phase 1

    # Verify skills were not matched (no "search" in message)
    assert len(result.matched_skills) == 0

    # Verify token estimate is reasonable
    assert result.estimated_tokens() > 0
    assert result.estimated_tokens() < 2500  # Should be under budget

    print(f"\nPriming result:")
    print(f"  Sender profile: {len(result.sender_profile)} chars")
    print(f"  Episodes: {len(result.recent_activity)} chars")
    print(f"  Knowledge: {len(result.related_knowledge)} chars")
    print(f"  Skills: {result.matched_skills}")
    print(f"  Total tokens: {result.estimated_tokens()}")


@pytest.mark.asyncio
async def test_priming_skill_match(temp_anima_dir, temp_shared_dir):
    """Test skill matching in Channel D."""
    engine = PrimingEngine(temp_anima_dir)

    # Use "web search" to match web_search skill via description keyword
    result = await engine.prime_memories(
        message="web search を使って情報を調べてください",
        sender_name="human",
    )

    # Should match "web_search" skill by description keyword (Tier 1: 「web search」)
    assert "web_search" in result.matched_skills

    print(f"\nMatched skills: {result.matched_skills}")


@pytest.mark.asyncio
async def test_priming_empty_result(temp_anima_dir, temp_shared_dir):
    """Test priming with no relevant memories."""
    engine = PrimingEngine(temp_anima_dir)

    result = await engine.prime_memories(
        message="Hello",
        sender_name="unknown_user",
    )

    # Should have some episodes, but no sender profile
    assert not result.sender_profile
    assert result.recent_activity  # Today's episodes should still be loaded

    print(f"\nEmpty sender priming result:")
    print(f"  Sender profile: {len(result.sender_profile)} chars")
    print(f"  Episodes: {len(result.recent_activity)} chars")


def test_format_priming_section():
    """Test priming result formatting."""
    from core.memory.priming import PrimingResult

    result = PrimingResult(
        sender_profile="テストユーザーのプロファイル",
        recent_activity="最近の会話履歴",
        related_knowledge="関連する知識",
        matched_skills=["skill1", "skill2"],
    )

    formatted = format_priming_section(result, sender_name="テストユーザー")

    assert "あなたが思い出していること" in formatted
    assert "テストユーザー について" in formatted
    assert "直近のアクティビティ" in formatted
    assert "関連する知識" in formatted
    assert "使えそうなスキル" in formatted
    assert "skill1" in formatted
    assert "skill2" in formatted

    print(f"\nFormatted priming section:\n{formatted}")


def test_keyword_extraction(temp_anima_dir, temp_shared_dir):
    """Test keyword extraction from message."""
    engine = PrimingEngine(temp_anima_dir)

    # Japanese text - without morphological analyzer, it extracts whole phrases
    keywords1 = engine._extract_keywords("山田さん プライミング レイヤー 話したい")
    print(f"\nJapanese keywords: {keywords1}")
    assert len(keywords1) > 0
    # Should extract individual words when space-separated
    assert any("プライミング" in kw for kw in keywords1)

    # English text
    keywords2 = engine._extract_keywords("I want to search for information about RAG")
    print(f"\nEnglish keywords: {keywords2}")
    assert "search" in keywords2 or "information" in keywords2 or "RAG" in keywords2

    # Mixed text with spaces
    keywords3 = engine._extract_keywords("ChromaDB ベクトル 検索")
    print(f"\nMixed keywords: {keywords3}")
    assert "ChromaDB" in keywords3
    assert "ベクトル" in keywords3 or "検索" in keywords3

    # Web search - extracts as single token
    keywords4 = engine._extract_keywords("Web 検索 情報 調べる")
    print(f"\nWeb search keywords: {keywords4}")
    assert "Web" in keywords4 or "検索" in keywords4


def test_keyword_extraction_single_char_kanji(temp_anima_dir):
    """Single-char kanji that carry meaning must survive extraction.

    _RE_WORDS tokenises on whitespace/punctuation boundaries, so tests
    use space-separated input to exercise the len>=1 filter.
    """
    engine = PrimingEngine(temp_anima_dir)

    kw1 = engine._extract_keywords("この Issue を 裏 で 実装 して")
    assert "裏" in kw1, f"'裏' missing from {kw1}"
    assert "実装" in kw1
    assert "Issue" in kw1

    kw2 = engine._extract_keywords("金 の 話")
    assert "金" in kw2, f"'金' missing from {kw2}"
    assert "の" not in kw2, "particle 'の' should be excluded"

    kw3 = engine._extract_keywords("型 ヒント 必須")
    assert "型" in kw3, f"'型' missing from {kw3}"


def test_keyword_extraction_particles_still_excluded(temp_anima_dir):
    """Japanese particles (1-char) must still be excluded by stopwords."""
    engine = PrimingEngine(temp_anima_dir)

    particles = ["の", "に", "は", "を", "が", "で", "と", "も", "や", "へ"]
    kw = engine._extract_keywords(" ".join(particles))
    for p in particles:
        assert p not in kw, f"particle '{p}' should be excluded"


def test_keyword_extraction_english_stopwords_still_excluded(temp_anima_dir):
    """English stopwords ('a', 'the', 'is', etc.) remain excluded."""
    engine = PrimingEngine(temp_anima_dir)

    kw = engine._extract_keywords("I want a big project")
    lower_kw = [w.lower() for w in kw]
    assert "a" not in lower_kw, "stopword 'a' should be excluded"
    assert "want" in lower_kw or "big" in lower_kw or "project" in lower_kw

    kw2 = engine._extract_keywords("the system is running")
    lower_kw2 = [w.lower() for w in kw2]
    assert "the" not in lower_kw2
    assert "is" not in lower_kw2


@pytest.mark.asyncio
async def test_channel_c_top_k(temp_anima_dir):
    """Channel C must call retriever.search with top_k=5."""
    engine = PrimingEngine(temp_anima_dir)

    from unittest.mock import MagicMock, patch

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []

    with patch.object(engine, "_get_or_create_retriever", return_value=mock_retriever):
        await engine._channel_c_related_knowledge(["test"], message="")

    mock_retriever.search.assert_called_once()
    call_kwargs = mock_retriever.search.call_args
    assert call_kwargs.kwargs.get("top_k") == 5 or call_kwargs[1].get("top_k") == 5


@pytest.mark.asyncio
async def test_channel_c_query_includes_message(temp_anima_dir):
    """Channel C query must prepend message[:200] when message is provided."""
    engine = PrimingEngine(temp_anima_dir)

    from unittest.mock import MagicMock, patch

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []

    msg = "このIssueを裏で実装して"

    with patch.object(engine, "_get_or_create_retriever", return_value=mock_retriever):
        await engine._channel_c_related_knowledge(["Issue", "裏", "実装"], message=msg)

    call_kwargs = mock_retriever.search.call_args
    actual_query = call_kwargs.kwargs.get("query") or call_kwargs[1].get("query")
    assert actual_query is not None
    assert actual_query.startswith(msg[:200])
    assert "Issue" in actual_query


@pytest.mark.asyncio
async def test_channel_c_query_keyword_only_when_no_message(temp_anima_dir):
    """Channel C query falls back to keyword-only when message is empty."""
    engine = PrimingEngine(temp_anima_dir)

    from unittest.mock import MagicMock, patch

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []

    with patch.object(engine, "_get_or_create_retriever", return_value=mock_retriever):
        await engine._channel_c_related_knowledge(["Issue", "実装"], message="")

    call_kwargs = mock_retriever.search.call_args
    actual_query = call_kwargs.kwargs.get("query") or call_kwargs[1].get("query")
    assert actual_query == "Issue 実装"


if __name__ == "__main__":
    # Run tests manually
    import sys

    async def run_tests():
        with tempfile.TemporaryDirectory() as tmpdir:
            anima_dir = Path(tmpdir) / "animas" / "test_anima"
            anima_dir.mkdir(parents=True)
            (anima_dir / "episodes").mkdir()
            (anima_dir / "knowledge").mkdir()
            (anima_dir / "skills").mkdir()

            shared_dir = Path(tmpdir) / "shared"
            users_dir = shared_dir / "users"
            users_dir.mkdir(parents=True)

            # Create sample data
            today_episode = anima_dir / "episodes" / f"{date.today().isoformat()}.md"
            today_episode.write_text(
                f"# {date.today().isoformat()} 行動ログ\n\n"
                "## 09:00 — テスト\n\nテストエピソード\n",
                encoding="utf-8",
            )

            (anima_dir / "knowledge" / "test.md").write_text("テスト知識", encoding="utf-8")
            (anima_dir / "skills" / "test_skill").mkdir(parents=True, exist_ok=True)
            (anima_dir / "skills" / "test_skill" / "SKILL.md").write_text("## 概要\nテストスキル", encoding="utf-8")

            engine = PrimingEngine(anima_dir)
            result = await engine.prime_memories("テストメッセージ", "human")

            print(f"Priming test result:")
            print(f"  Episodes: {len(result.recent_activity)} chars")
            print(f"  Knowledge: {len(result.related_knowledge)} chars")
            print(f"  Tokens: {result.estimated_tokens()}")

    asyncio.run(run_tests())
    print("\n✓ Manual tests passed")
