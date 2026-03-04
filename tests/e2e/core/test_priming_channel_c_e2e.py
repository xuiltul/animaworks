from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for Priming Channel C keyword / top_k / query fixes.

Verifies the full prime_memories() path with real directory structures,
ensuring that single-char kanji keywords propagate through to
Channel C and that message context improves retrieval relevance.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.priming import PrimingEngine


@pytest.fixture
def rich_anima_dir(tmp_path: Path) -> Path:
    """Anima directory with knowledge files that exercise the fixes."""
    anima_dir = tmp_path / "test_anima"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "episodes").mkdir(parents=True)
    (anima_dir / "skills").mkdir(parents=True)
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "activity_log").mkdir(parents=True)

    (anima_dir / "knowledge" / "background-implementation-workflow.md").write_text(
        "# 「裏で実装」ワークフロー\n\n"
        "## 手順\n"
        "1. Issue を読む\n"
        "2. worktree を作成\n"
        "3. 裏側で自動実装する\n",
        encoding="utf-8",
    )
    (anima_dir / "knowledge" / "chatwork-policy.md").write_text(
        "# Chatworkポリシー\n\nChatworkでの報告は簡潔に\n",
        encoding="utf-8",
    )
    (anima_dir / "knowledge" / "meeting-protocol.md").write_text(
        "# ミーティングプロトコル\n\n事前準備が重要\n",
        encoding="utf-8",
    )
    (anima_dir / "knowledge" / "type-hints-guide.md").write_text(
        "# 型ヒントガイド\n\n型ヒントは全ファイル必須。\n"
        "Pydanticモデルで型安全を担保する。\n",
        encoding="utf-8",
    )
    (anima_dir / "knowledge" / "money-management.md").write_text(
        "# 金銭管理\n\n予算管理と金の流れの把握が重要。\n",
        encoding="utf-8",
    )

    shared_dir = tmp_path / "shared"
    (shared_dir / "users").mkdir(parents=True)
    return anima_dir


@pytest.mark.asyncio
async def test_single_char_kanji_reaches_channel_c(rich_anima_dir: Path):
    """End-to-end: message text (containing '裏') is prepended to Channel C query."""
    engine = PrimingEngine(rich_anima_dir)

    captured_queries: list[str] = []

    mock_retriever = MagicMock()

    def capture_search(**kwargs):
        captured_queries.append(kwargs.get("query", ""))
        return []

    mock_retriever.search.side_effect = capture_search

    msg = "このIssueを裏で実装して"

    with (
        patch.object(engine, "_get_or_create_retriever", return_value=mock_retriever),
        patch("core.paths.get_shared_dir", return_value=rich_anima_dir.parent / "shared"),
    ):
        await engine.prime_memories(
            message=msg,
            sender_name="human",
        )

    assert len(captured_queries) >= 1, "Channel C should have issued a search"
    query = captured_queries[0]
    assert "裏" in query, f"'裏' not found in query: {query}"
    assert query.startswith(msg), (
        f"Original message not prepended to query: {query}"
    )


@pytest.mark.asyncio
async def test_channel_c_uses_top_k_5(rich_anima_dir: Path):
    """End-to-end: Channel C retriever.search is called with top_k=5."""
    engine = PrimingEngine(rich_anima_dir)

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []

    with (
        patch.object(engine, "_get_or_create_retriever", return_value=mock_retriever),
        patch("core.paths.get_shared_dir", return_value=rich_anima_dir.parent / "shared"),
    ):
        await engine.prime_memories(
            message="型ヒントについて教えて",
            sender_name="human",
        )

    # Channel C (top_k=5) and Channel F (top_k=3) share the retriever,
    # so search may be called more than once. Verify Channel C's call exists.
    calls = mock_retriever.search.call_args_list
    assert any(c.kwargs.get("top_k") == 5 for c in calls), (
        f"Expected a search call with top_k=5 (Channel C), got: {calls}"
    )


@pytest.mark.asyncio
async def test_overflow_path_passes_message(rich_anima_dir: Path):
    """End-to-end: overflow_files path also passes message to Channel C."""
    engine = PrimingEngine(rich_anima_dir)

    captured_queries: list[str] = []

    mock_retriever = MagicMock()

    def capture_search(**kwargs):
        captured_queries.append(kwargs.get("query", ""))
        return []

    mock_retriever.search.side_effect = capture_search

    with (
        patch.object(engine, "_get_or_create_retriever", return_value=mock_retriever),
        patch("core.paths.get_shared_dir", return_value=rich_anima_dir.parent / "shared"),
    ):
        await engine.prime_memories(
            message="金の管理について",
            sender_name="human",
            overflow_files=["chatwork-policy"],
        )

    assert len(captured_queries) >= 1
    query = captured_queries[0]
    assert "金" in query, f"'金' not found in query: {query}"
    assert query.startswith("金の管理について")


@pytest.mark.asyncio
async def test_keyword_extraction_full_pipeline(rich_anima_dir: Path):
    """End-to-end: verify keyword extraction results propagate correctly.

    _RE_WORDS tokenises on whitespace/punctuation. Space-separated input
    exercises the len>=1 filter that lets single-char kanji through.
    """
    engine = PrimingEngine(rich_anima_dir)

    keywords = engine._extract_keywords("この Issue を 裏 で 実装 して")
    assert "裏" in keywords
    assert "Issue" in keywords
    assert "実装" in keywords
    assert "を" not in keywords
    assert "で" not in keywords

    keywords2 = engine._extract_keywords("金 の 話 を しよう")
    assert "金" in keywords2
    assert "の" not in keywords2

    keywords3 = engine._extract_keywords("型 ヒント 必須 です")
    assert "型" in keywords3
