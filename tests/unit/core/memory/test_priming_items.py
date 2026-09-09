from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.memory.activity import ActivityEntry
from core.memory.priming import channel_c, channel_f, outbound
from core.memory.priming.channel_e import _itemize_pending_tasks
from core.memory.priming.consolidate import consolidate_items
from core.memory.priming.constants import (
    _BUDGET_IMPORTANT_KNOWLEDGE,
    _BUDGET_RECENT_ACTIVITY,
    _BUDGET_RELATED_KNOWLEDGE,
)
from core.memory.priming.engine import PrimingEngine
from core.memory.priming.items import ItemizedMemory, MemoryItem, render_items, select_within_budget
from core.memory.priming.result import PrimingResult
from core.memory.rag.store import Document, SearchResult
from core.prompt.tokens import estimate_tokens


def test_select_within_budget_keeps_whole_items_in_priority_order() -> None:
    lower = MemoryItem("recent_activity", "lower", "低" * 40, updated="2026-09-07T10:00:00+09:00", rank=1)
    older = MemoryItem("recent_activity", "older", "古" * 40, updated="2026-09-07T09:00:00+09:00", rank=2)
    newer = MemoryItem("recent_activity", "newer", "新" * 40, updated="2026-09-07T11:00:00+09:00", rank=2)
    budget = estimate_tokens(render_items([newer, older], ""))

    selected = select_within_budget([lower, older, newer], budget)

    assert selected == [newer, older]
    assert estimate_tokens(render_items(selected, "")) <= budget
    assert all(item.text in render_items(selected, "") for item in selected)
    assert lower.text not in render_items(selected, "")


def test_consolidate_drops_old_keys_long_text_matches_and_duplicate_refs() -> None:
    shared = "これは時刻だけが異なる重複本文です。" * 8
    items = {
        "recent_activity": (
            MemoryItem("recent_activity", "same", "old", updated="2026-09-06"),
            MemoryItem("recent_activity", "same", "new", updated="2026-09-07"),
            MemoryItem("recent_activity", "a", f"[10:00] {shared} activity suffix"),
        ),
        "episodes": (MemoryItem("episodes", "b", f"[11:30] prefix {shared}"),),
        "important_knowledge": (
            MemoryItem("important_knowledge", "chunk-1", "old ref", ref="knowledge/rule.md", updated="2026-01"),
            MemoryItem("important_knowledge", "chunk-2", "new ref", ref="knowledge/rule.md", updated="2026-09"),
        ),
    }

    result = consolidate_items(PrimingResult(items=items))

    assert [item.text for item in result.items["recent_activity"]] == ["new", f"[10:00] {shared} activity suffix"]
    assert result.items["episodes"] == ()
    assert [item.text for item in result.items["important_knowledge"]] == ["new ref"]
    assert result.related_knowledge.count("knowledge/rule.md") == 0


def test_consolidate_keeps_later_item_with_substantial_unique_text() -> None:
    common = "共" * 61
    first = MemoryItem("recent_activity", "first", common + "先行情報")
    correction = MemoryItem("episodes", "second", common + "訂正" * 25)

    result = consolidate_items(PrimingResult(items={"recent_activity": (first,), "episodes": (correction,)}))

    assert result.items["recent_activity"] == (first,)
    assert result.items["episodes"] == (correction,)


def test_consolidate_drops_later_item_with_little_unique_text() -> None:
    common = "共" * 61
    first = MemoryItem("recent_activity", "first", common + "先行情報")
    duplicate = MemoryItem("episodes", "second", common + "追" * 10)

    result = consolidate_items(PrimingResult(items={"recent_activity": (first,), "episodes": (duplicate,)}))

    assert result.items["recent_activity"] == (first,)
    assert result.items["episodes"] == ()


def test_consolidate_prefers_c0_then_channel_c_over_same_path_episode() -> None:
    path = "knowledge/release.md"
    episode = MemoryItem("episodes", path, "episode", ref=path, updated="2026-09-08")
    related = MemoryItem("related_knowledge", path, "related", ref=path, updated="2026-09-09")
    important = MemoryItem("important_knowledge", path, "important", ref=path, updated="2026-01-01")

    with_c0 = consolidate_items(
        PrimingResult(
            items={
                "episodes": (episode,),
                "related_knowledge": (related,),
                "important_knowledge": (important,),
            }
        )
    )
    without_c0 = consolidate_items(PrimingResult(items={"episodes": (episode,), "related_knowledge": (related,)}))

    assert with_c0.items["important_knowledge"] == (important,)
    assert with_c0.items["related_knowledge"] == ()
    assert with_c0.items["episodes"] == ()
    assert without_c0.items["related_knowledge"] == (related,)
    assert without_c0.items["episodes"] == ()


def test_sentence_boundary_trim_handles_japanese_and_english() -> None:
    from core.memory.activity import ActivityLogger
    from core.memory.priming.channel_b import _format_entry_at_sentence_boundary

    for prefix, boundary in (("日" * 130, "。"), ("English words " * 11, "!")):
        entry = ActivityEntry(
            ts="2026-09-08T12:00:00+09:00",
            type="message_received",
            content=prefix + boundary + "切り捨て対象" * 30,
            from_person="human",
        )

        rendered = _format_entry_at_sentence_boundary(
            ActivityLogger(Path("/tmp/test-anima")),
            entry,
            content_trim=200,
        )

        before_pointer = rendered.split("\n  ->", 1)[0]
        assert before_pointer.endswith(boundary)
        assert "切り捨て対象" not in before_pointer


def test_sentence_boundary_trim_uses_ellipsis_when_boundary_is_too_early() -> None:
    from core.memory.activity import ActivityLogger
    from core.memory.priming.channel_b import _format_entry_at_sentence_boundary

    content = "短い文。" + "続" * 250
    entry = ActivityEntry(
        ts="2026-09-08T12:00:00+09:00",
        type="message_received",
        content=content,
        from_person="human",
    )

    rendered = _format_entry_at_sentence_boundary(
        ActivityLogger(Path("/tmp/test-anima")),
        entry,
        content_trim=200,
    )

    assert content[:200] + "…" in rendered


def test_pending_tasks_are_split_with_task_ids_as_keys() -> None:
    items = _itemize_pending_tasks(
        "## Active Parallel Tasks\n"
        "- [task-a] First task (running 1m)\n"
        "  first description\n"
        "- [task-b] Second task (running 2m)"
    )

    assert [item.key for item in items] == ["task-a", "task-b"]
    assert "first description" in items[0].text
    assert "Second task" in items[1].text


@pytest.mark.asyncio
async def test_c0_uses_title_for_table_header_and_orders_by_updated(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    knowledge_dir = anima_dir / "knowledge"
    knowledge_dir.mkdir(parents=True)
    docs = [
        Document(
            id="old",
            content="# Very short old title\n\nUseful old summary",
            metadata={
                "source_file": "knowledge/old.md",
                "anima": "mei",
                "importance": "important",
                "updated_at": "2026-01-01T00:00:00+09:00",
            },
        ),
        Document(
            id="new",
            content="# A much longer but newer title\n\n| chatID | 名称 | 理由 |",
            metadata={
                "source_file": "knowledge/new.md",
                "anima": "mei",
                "importance": "important",
                "updated_at": "2026-09-07T00:00:00+09:00",
            },
        ),
    ]
    retriever = MagicMock()
    retriever.get_important_chunks.return_value = [SearchResult(document=doc, score=1.0) for doc in docs]

    async def direct_call(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(channel_c.asyncio, "to_thread", direct_call)
    output = await channel_c.channel_c0_important_knowledge(anima_dir, knowledge_dir, lambda: retriever)

    assert "| chatID | 名称 | 理由 |" not in output
    assert output.index("A much longer but newer title") < output.index("Very short old title")
    assert output.items[0].ref == "knowledge/new.md"


@pytest.mark.asyncio
async def test_channel_f_returns_one_item_per_episode(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    episodes_dir = anima_dir / "episodes"
    episodes_dir.mkdir(parents=True)
    searcher = MagicMock()
    searcher.last_search_meta = {"abstain": False}
    searcher.search_many.return_value = [
        {
            "doc_id": "mei/episodes/2026-09-01.md#0",
            "source_file": "episodes/2026-09-01.md",
            "content": "# Release retrospective",
            "score": 0.75,
        }
    ]

    async def direct_call(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(channel_f.asyncio, "to_thread", direct_call)
    monkeypatch.setattr(channel_f, "build_unified_searcher", lambda *args: searcher)
    monkeypatch.setattr(channel_f.MemoryIndexer, "is_ragignored", lambda path: False)

    output = await channel_f.channel_f_episodes(
        anima_dir,
        episodes_dir,
        lambda: None,
        ["release"],
        message="release",
    )

    assert len(output.items) == 1
    assert output.items[0].key == "episodes/2026-09-01.md"
    assert output.items[0].updated == "2026-09-01"
    assert output.items[0].rank == 0.75


@pytest.mark.asyncio
async def test_recent_outbound_returns_one_item_per_activity(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    anima_dir.mkdir(parents=True)
    entry = ActivityEntry(
        ts="2026-09-07T12:00:00+09:00",
        type="message_sent",
        summary="release report sent",
        from_person="mei",
        to_person="human",
    )

    class FakeActivityLogger:
        def __init__(self, path: Path) -> None:
            assert path == anima_dir

        def recent(self, **kwargs):
            return [entry]

    monkeypatch.setattr("core.memory.activity.ActivityLogger", FakeActivityLogger)
    monkeypatch.setattr(outbound, "now_local", lambda: datetime.fromisoformat(entry.ts))

    output = await outbound.collect_recent_outbound(anima_dir)

    assert len(output.items) == 1
    assert output.items[0].key == f"{entry.ts}||mei"
    assert "release report sent" in output.items[0].text


@pytest.mark.asyncio
async def test_prime_memories_item_budgets_and_cross_channel_dedup(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "episodes").mkdir()
    engine = PrimingEngine(anima_dir)
    repeated_key = "2026-09-07T12:00:00+09:00|chat|human"
    activity_items = tuple(
        MemoryItem(
            "recent_activity",
            repeated_key if index == 0 else f"activity-{index}",
            f"活動{index}:" + "あ" * 190,
            updated=f"2026-09-07T12:{index:02d}:00+09:00",
            rank=float(index),
        )
        for index in range(10)
    )
    knowledge_items = tuple(
        MemoryItem(
            "important_knowledge",
            repeated_key if index == 0 else f"knowledge-{index}",
            f"知識{index}:" + "い" * 190 + f' -> read_memory_file(path="knowledge/{index}.md")',
            ref=f"knowledge/{index}.md",
            updated=f"2026-09-07T11:{index:02d}:00+09:00",
            rank=float(index),
        )
        for index in range(10)
    )

    async def empty(*args, **kwargs):
        return ""

    async def activity(*args, **kwargs):
        return ItemizedMemory(render_items(activity_items, ""), activity_items)

    async def knowledge(*args, **kwargs):
        return ItemizedMemory(
            render_items(knowledge_items, "### [IMPORTANT] Knowledge (summary pointers)"),
            knowledge_items,
        )

    monkeypatch.setattr(engine, "_channel_a_sender_profile", empty)
    monkeypatch.setattr(engine, "_channel_b_recent_activity", activity)
    monkeypatch.setattr(engine, "_channel_c0_important_knowledge", knowledge)
    monkeypatch.setattr(engine, "_channel_c_related_knowledge", lambda *args, **kwargs: empty_pair())
    monkeypatch.setattr(engine, "_channel_e_pending_tasks", empty)
    monkeypatch.setattr(engine, "_collect_recent_outbound", empty)
    monkeypatch.setattr(engine, "_channel_f_episodes", empty)
    monkeypatch.setattr(engine, "_collect_pending_human_notifications", empty)
    monkeypatch.setattr(engine, "_channel_g_graph_context", empty)

    result = await engine.prime_memories("重複を確認", enable_dynamic_budget=False)

    assert estimate_tokens(result.recent_activity) <= _BUDGET_RECENT_ACTIVITY
    assert estimate_tokens(result.related_knowledge) <= _BUDGET_RELATED_KNOWLEDGE
    emitted_keys = [item.key for items in result.items.values() for item in items]
    assert emitted_keys.count(repeated_key) == 1
    assert estimate_tokens(result.related_knowledge) <= _BUDGET_IMPORTANT_KNOWLEDGE


@pytest.mark.asyncio
async def test_prime_memories_related_budget_never_splits_channel_c_item(tmp_path: Path, monkeypatch) -> None:
    anima_dir = tmp_path / "animas" / "mei"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "episodes").mkdir()
    engine = PrimingEngine(anima_dir)
    high = MemoryItem(
        "related_knowledge",
        "knowledge/high.md",
        "📌 " + "甲" * 700 + ' → read_memory_file(path="knowledge/high.md")',
        ref="knowledge/high.md",
        rank=2,
    )
    low = MemoryItem(
        "related_knowledge",
        "knowledge/low.md",
        "📌 " + "乙" * 700 + ' → read_memory_file(path="knowledge/low.md")',
        ref="knowledge/low.md",
        rank=1,
    )
    related_items = (low, high)
    legacy_c0 = '📌 Legacy C0 → read_memory_file(path="knowledge/legacy.md")'

    async def empty(*args, **kwargs):
        return ""

    async def related(*args, **kwargs):
        return ItemizedMemory(render_items(related_items, ""), related_items), ItemizedMemory("")

    async def c0(*args, **kwargs):
        return legacy_c0

    monkeypatch.setattr(engine, "_channel_a_sender_profile", empty)
    monkeypatch.setattr(engine, "_channel_b_recent_activity", empty)
    monkeypatch.setattr(engine, "_channel_c0_important_knowledge", c0)
    monkeypatch.setattr(engine, "_channel_c_related_knowledge", related)
    monkeypatch.setattr(engine, "_channel_e_pending_tasks", empty)
    monkeypatch.setattr(engine, "_collect_recent_outbound", empty)
    monkeypatch.setattr(engine, "_channel_f_episodes", empty)
    monkeypatch.setattr(engine, "_collect_pending_human_notifications", empty)

    result = await engine.prime_memories("知識を確認", enable_dynamic_budget=False)

    assert result.related_knowledge == f"{legacy_c0}\n\n{high.text}"
    assert result.items["related_knowledge"] == (high,)
    assert low.text not in result.related_knowledge
    assert "..." not in result.related_knowledge


async def empty_pair() -> tuple[str, str]:
    return "", ""
