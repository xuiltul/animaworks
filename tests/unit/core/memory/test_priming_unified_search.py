from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.priming.channel_c import channel_c_related_knowledge
from core.memory.priming.channel_f import channel_f_episodes


def _fake_unified_search(results: list[dict] | None = None, *, abstain: bool = False) -> MagicMock:
    searcher = MagicMock()
    searcher.search_many.return_value = results or []
    searcher.last_search_meta = {
        "abstain": abstain,
        "abstain_reason": "low_confidence" if abstain else "",
    }
    return searcher


@pytest.mark.asyncio
async def test_channel_c_uses_unified_search_without_direct_retriever(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "test"
    (anima_dir / "knowledge").mkdir(parents=True)
    direct_retriever = MagicMock()
    direct_retriever.search.side_effect = AssertionError("direct retriever search should not be called")
    searcher = _fake_unified_search(
        [
            {
                "doc_id": "test/knowledge/deploy.md#0",
                "source_file": "knowledge/deploy.md",
                "content": "## Deploy\n\nUse the release checklist.",
                "score": 0.9,
                "origin": "system",
            },
        ]
    )

    with patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=searcher):
        medium, untrusted = await channel_c_related_knowledge(
            anima_dir,
            anima_dir / "knowledge",
            lambda: direct_retriever,
            ["deploy"],
        )

    assert 'read_memory_file(path="knowledge/deploy.md")' in medium
    assert untrusted == ""
    searcher.search_many.assert_called_once()
    direct_retriever.search.assert_not_called()


@pytest.mark.asyncio
async def test_channel_f_uses_unified_search_without_direct_retriever(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "test"
    (anima_dir / "episodes").mkdir(parents=True)
    direct_retriever = MagicMock()
    direct_retriever.search.side_effect = AssertionError("direct retriever search should not be called")
    searcher = _fake_unified_search(
        [
            {
                "doc_id": "test/episodes/2026-03-01.md#0",
                "source_file": "episodes/2026-03-01.md",
                "content": "# Deploy outage\n\nFull body should not be injected.",
                "score": 0.82,
            },
        ]
    )

    with patch("core.memory.priming.channel_f.UnifiedMemorySearch", return_value=searcher):
        output = await channel_f_episodes(
            anima_dir,
            anima_dir / "episodes",
            lambda: direct_retriever,
            ["deploy"],
            message="deploy outage",
        )

    assert 'read_memory_file(path="episodes/2026-03-01.md")' in output
    assert "Full body should not be injected" not in output
    searcher.search_many.assert_called_once()
    direct_retriever.search.assert_not_called()


@pytest.mark.asyncio
async def test_priming_channels_return_empty_when_unified_search_abstains(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "test"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "episodes").mkdir(parents=True)
    knowledge_searcher = _fake_unified_search(abstain=True)
    episode_searcher = _fake_unified_search(abstain=True)

    with patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=knowledge_searcher):
        medium, untrusted = await channel_c_related_knowledge(
            anima_dir,
            anima_dir / "knowledge",
            lambda: MagicMock(),
            ["deploy"],
        )
    with patch("core.memory.priming.channel_f.UnifiedMemorySearch", return_value=episode_searcher):
        episodes = await channel_f_episodes(
            anima_dir,
            anima_dir / "episodes",
            lambda: MagicMock(),
            ["deploy"],
        )

    assert medium == ""
    assert untrusted == ""
    assert episodes == ""
