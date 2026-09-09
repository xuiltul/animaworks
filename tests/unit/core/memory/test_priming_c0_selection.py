from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.memory.priming import channel_c
from core.memory.rag.indexer import MemoryIndexer
from core.memory.rag.store import Document, SearchResult


def _stored_chunk(
    name: str,
    *,
    updated: str,
    always_prime: bool = True,
    content: str | None = None,
) -> SearchResult:
    return SearchResult(
        document=Document(
            id=f"mei/knowledge/{name}.md#0",
            content=content or f"# {name}",
            metadata={
                "anima": "mei",
                "source_file": f"knowledge/{name}.md",
                "always_prime": always_prime,
                "updated_at": updated,
            },
        ),
        score=1.0,
    )


def _search_row(
    name: str,
    *,
    score: float,
    importance: str = "important",
    content: str | None = None,
) -> dict:
    return {
        "doc_id": f"mei/knowledge/{name}.md#0",
        "source_file": f"knowledge/{name}.md",
        "anima": "mei",
        "content": content or f"# {name}",
        "importance": importance,
        "score": score,
    }


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    path = tmp_path / "animas" / "mei"
    (path / "knowledge").mkdir(parents=True)
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("abstain", [False, True])
async def test_c0_and_c_share_search_without_losing_selection_or_abstention(
    anima_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    abstain: bool,
) -> None:
    retriever = MagicMock()
    retriever.vector_store.get_by_metadata.return_value = []
    searcher = MagicMock()
    searcher.last_search_meta = {"abstain": abstain}
    searcher.search_many.return_value = [
        _search_row("important-result", score=0.99),
        _search_row("ordinary-result", score=0.95, importance="normal"),
    ]
    monkeypatch.setattr(channel_c, "_build_unified_searcher", lambda *args: searcher)
    cache = channel_c.KnowledgeSearchCache()
    queries = channel_c.build_queries("current topic", [])

    # Use the real worker threads: both consumers can request the same search
    # concurrently, but retain their own ranking and trust filtering.
    important, related = await asyncio.gather(
        channel_c.channel_c0_important_knowledge(
            anima_dir, anima_dir / "knowledge", lambda: retriever, queries, search_cache=cache
        ),
        channel_c.channel_c_related_knowledge(
            anima_dir, anima_dir / "knowledge", lambda: retriever, [], "current topic", search_cache=cache
        ),
    )

    searcher.search_many.assert_called_once()
    if abstain:
        assert not important
        assert not any(related)
    else:
        assert "important-result" in important
        assert "ordinary-result" not in important
        combined_related = "\n".join(related)
        assert "important-result" in combined_related
        assert "ordinary-result" in combined_related


@pytest.mark.asyncio
async def test_c0_limits_residents_and_relevant_important_chunks(
    anima_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = MagicMock()
    retriever.vector_store.get_by_metadata.side_effect = [
        [
            _stored_chunk("resident-old", updated="2026-01-01"),
            _stored_chunk("resident-1", updated="2026-09-04"),
            _stored_chunk("resident-2", updated="2026-09-05"),
            _stored_chunk("resident-3", updated="2026-09-06"),
        ],
        [],
    ]
    searcher = MagicMock()
    searcher.last_search_meta = {"abstain": False}
    searcher.search_many.return_value = [
        _search_row("related-1", score=0.99),
        _search_row("unrelated-normal", score=0.98, importance="normal"),
        _search_row("related-2", score=0.90),
        _search_row("related-3", score=0.80),
        _search_row("related-4", score=0.70),
    ]

    async def direct_call(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(channel_c.asyncio, "to_thread", direct_call)
    monkeypatch.setattr(channel_c, "_build_unified_searcher", lambda *args: searcher)

    result = await channel_c.channel_c0_important_knowledge(
        anima_dir,
        anima_dir / "knowledge",
        lambda: retriever,
        ["current topic"],
    )

    assert "resident-old" not in result
    assert all(name in result for name in ("resident-1", "resident-2", "resident-3"))
    assert all(name in result for name in ("related-1", "related-2", "related-3"))
    assert "related-4" not in result
    assert "unrelated-normal" not in result
    assert result.count("📌") == 6
    assert all(len(line.splitlines()) == 1 for line in result.splitlines() if line.startswith("📌"))
    retriever.get_important_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_c0_excludes_action_rules(anima_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = MagicMock()
    retriever.vector_store.get_by_metadata.side_effect = [
        [
            _stored_chunk("action-rule-mail", updated="2026-09-07"),
            _stored_chunk(
                "ordinary-file",
                updated="2026-09-06",
                content="# Rule\n[ACTION-RULE]\nDo not prime here.",
            ),
            _stored_chunk("safe-resident", updated="2026-09-05"),
        ],
        [],
    ]
    searcher = MagicMock()
    searcher.last_search_meta = {"abstain": False}
    searcher.search_many.return_value = [
        _search_row("action-rule-send", score=0.99),
        _search_row("safe-related", score=0.90),
    ]

    async def direct_call(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(channel_c.asyncio, "to_thread", direct_call)
    monkeypatch.setattr(channel_c, "_build_unified_searcher", lambda *args: searcher)

    result = await channel_c.channel_c0_important_knowledge(
        anima_dir,
        anima_dir / "knowledge",
        lambda: retriever,
        ["mail"],
    )

    assert "action-rule" not in result
    assert "ordinary-file" not in result
    assert "safe-resident" in result
    assert "safe-related" in result


def test_indexer_copies_always_prime_frontmatter_to_chunk_metadata(anima_dir: Path) -> None:
    path = anima_dir / "knowledge" / "resident.md"
    path.write_text("# Resident", encoding="utf-8")
    indexer = MemoryIndexer.__new__(MemoryIndexer)
    indexer.collection_prefix = "mei"
    indexer.anima_dir = anima_dir

    metadata = indexer._extract_metadata(
        path,
        "# Resident",
        "knowledge",
        0,
        1,
        frontmatter={"always_prime": True},
    )

    assert metadata["always_prime"] is True


@pytest.mark.asyncio
async def test_originless_own_company_knowledge_is_medium(
    anima_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (anima_dir / "status.json").write_text(json.dumps({"company": "acme"}), encoding="utf-8")
    searcher = MagicMock()
    searcher.last_search_meta = {"abstain": False}
    searcher.search_many.return_value = [
        {
            "doc_id": "shared/companies/acme/knowledge/policy.md#0",
            "source_file": "companies/acme/knowledge/policy.md",
            "anima": "shared",
            "content": "# Company policy",
            "score": 0.9,
        },
        {
            "doc_id": "shared/companies/other/knowledge/policy.md#0",
            "source_file": "companies/other/knowledge/policy.md",
            "anima": "shared",
            "content": "# Other company policy",
            "score": 0.8,
        },
        {
            "doc_id": "mei/knowledge/external.md#0",
            "source_file": "knowledge/external.md",
            "anima": "mei",
            "content": "# External feed",
            "origin": "external_platform",
            "score": 0.7,
        },
    ]

    async def direct_call(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(channel_c.asyncio, "to_thread", direct_call)
    monkeypatch.setattr(channel_c, "_build_unified_searcher", lambda *args: searcher)

    medium, untrusted = await channel_c.channel_c_related_knowledge(
        anima_dir,
        anima_dir / "knowledge",
        lambda: MagicMock(),
        ["policy"],
    )

    assert "companies/acme/knowledge/policy.md" in medium
    assert "companies/other/knowledge/policy.md" in untrusted
    assert "knowledge/external.md" in untrusted
