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
from types import SimpleNamespace
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

    mock_searcher = SimpleNamespace(
        last_search_meta={},
        search_many=MagicMock(return_value=[]),
    )

    msg = "この Issue を 裏 で 実装 して"

    with (
        patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=mock_searcher),
        patch("core.paths.get_shared_dir", return_value=rich_anima_dir.parent / "shared"),
    ):
        await engine.prime_memories(
            message=msg,
            sender_name="human",
        )

    mock_searcher.search_many.assert_called_once()
    captured_queries = mock_searcher.search_many.call_args.args[0]
    assert len(captured_queries) >= 2, "Channel C dual query should include message and keyword queries"
    msg_query = captured_queries[0]
    kw_query = captured_queries[1]
    assert msg_query.startswith(msg), (
        f"First query (message-context) should start with original message: {msg_query}"
    )
    assert "裏" in kw_query or "裏" in msg_query, (
        f"'裏' not found in either query: msg={msg_query}, kw={kw_query}"
    )


@pytest.mark.asyncio
async def test_channel_c_uses_top_k_5(rich_anima_dir: Path):
    """End-to-end: Channel C unified search is called with limit=5."""
    engine = PrimingEngine(rich_anima_dir)

    mock_searcher = SimpleNamespace(
        last_search_meta={},
        search_many=MagicMock(return_value=[]),
    )

    with (
        patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=mock_searcher),
        patch("core.paths.get_shared_dir", return_value=rich_anima_dir.parent / "shared"),
    ):
        await engine.prime_memories(
            message="型ヒントについて教えて",
            sender_name="human",
        )

    mock_searcher.search_many.assert_called_once()
    assert mock_searcher.search_many.call_args.kwargs["limit"] == 5
    assert mock_searcher.search_many.call_args.kwargs["scope"] == "common_knowledge"
    assert mock_searcher.search_many.call_args.kwargs["trigger"] == "chat"


@pytest.mark.asyncio
async def test_full_channel_c_path_passes_message(rich_anima_dir: Path):
    """End-to-end: full Channel C path passes message context to search."""
    engine = PrimingEngine(rich_anima_dir)

    mock_searcher = SimpleNamespace(
        last_search_meta={},
        search_many=MagicMock(return_value=[]),
    )

    with (
        patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=mock_searcher),
        patch("core.paths.get_shared_dir", return_value=rich_anima_dir.parent / "shared"),
    ):
        await engine.prime_memories(
            message="金の管理について",
            sender_name="human",
        )

    captured_queries = mock_searcher.search_many.call_args.args[0]
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
    assert "裏" in keywords, f"'裏' missing from {keywords}"
    assert "Issue" in keywords
    assert "実装" in keywords

    keywords2 = engine._extract_keywords("金 の 話 を しよう")
    assert "金" in keywords2, f"'金' missing from {keywords2}"

    keywords3 = engine._extract_keywords("型 ヒント 必須 です")
    assert "型" in keywords3, f"'型' missing from {keywords3}"
