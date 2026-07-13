from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.priming import PrimingEngine
from core.memory.priming.channel_c import channel_c0_important_knowledge, channel_c_related_knowledge
from core.memory.priming.channel_f import channel_f_episodes


def _anima_with_deny(tmp_path: Path, relative_deny: str) -> tuple[Path, Path]:
    anima_dir = tmp_path / "animas" / "test-anima"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "episodes").mkdir()
    denied = anima_dir / relative_deny
    denied.mkdir(parents=True, exist_ok=True)
    (anima_dir / "permissions.json").write_text(
        json.dumps({"version": 1, "file_roots_denied": [str(denied)]}),
        encoding="utf-8",
    )
    return anima_dir, denied


@pytest.mark.asyncio
async def test_channel_c0_filters_denied_cached_chunk_before_formatting(tmp_path: Path) -> None:
    anima_dir, _denied = _anima_with_deny(tmp_path, "knowledge/private")
    retriever = MagicMock()
    retriever.get_important_chunks.return_value = [
        SimpleNamespace(
            document=SimpleNamespace(
                id="test-anima/knowledge/private/secret.md#0",
                content="# DENIED C0 CANARY\nsecret body",
                metadata={"source_file": "knowledge/private/secret.md"},
            )
        ),
        SimpleNamespace(
            document=SimpleNamespace(
                id="test-anima/knowledge/public.md#0",
                content="# Public knowledge\nallowed body",
                metadata={"source_file": "knowledge/public.md"},
            )
        ),
    ]

    result = await channel_c0_important_knowledge(anima_dir, anima_dir / "knowledge", lambda: retriever)

    assert "Public knowledge" in result
    assert "DENIED C0 CANARY" not in result
    assert "knowledge/private" not in result


@pytest.mark.asyncio
async def test_channel_c_filters_denied_and_ambiguous_unified_hits(tmp_path: Path) -> None:
    anima_dir, _denied = _anima_with_deny(tmp_path, "knowledge/private")
    searcher = MagicMock()
    searcher.last_search_meta = {"abstain": False}
    searcher.search_many.return_value = [
        {
            "doc_id": "test-anima/knowledge/private/secret.md#0",
            "source_file": "",
            "content": "DENIED C CANARY",
            "score": 0.99,
            "origin": "system",
        },
        {
            "doc_id": "opaque-vector-id",
            "source_file": "",
            "content": "AMBIGUOUS C CANARY",
            "score": 0.95,
            "origin": "system",
        },
        {
            "doc_id": "test-anima/knowledge/public.md#0",
            "source_file": "knowledge/public.md",
            "content": "# Public pointer",
            "score": 0.8,
            "origin": "system",
        },
    ]

    with patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=searcher):
        medium, untrusted = await channel_c_related_knowledge(
            anima_dir,
            anima_dir / "knowledge",
            lambda: None,
            ["test"],
        )

    assert 'read_memory_file(path="knowledge/public.md")' in medium
    assert "knowledge/private" not in medium
    assert "opaque-vector-id" not in medium
    assert "CANARY" not in medium
    assert untrusted == ""


@pytest.mark.asyncio
async def test_channel_f_filters_denied_unified_episode_hit(tmp_path: Path) -> None:
    anima_dir, _denied = _anima_with_deny(tmp_path, "episodes/private")
    searcher = MagicMock()
    searcher.last_search_meta = {"abstain": False}
    searcher.search_many.return_value = [
        {
            "doc_id": "test-anima/episodes/private/secret.md#0",
            "source_file": "",
            "content": "DENIED F CANARY",
            "score": 0.99,
        },
        {
            "doc_id": "test-anima/episodes/public.md#0",
            "source_file": "episodes/public.md",
            "content": "# Public episode",
            "score": 0.8,
        },
    ]

    with patch("core.memory.priming.channel_f.UnifiedMemorySearch", return_value=searcher):
        result = await channel_f_episodes(
            anima_dir,
            anima_dir / "episodes",
            lambda: None,
            ["test"],
        )

    assert 'read_memory_file(path="episodes/public.md")' in result
    assert "episodes/private" not in result
    assert "DENIED F CANARY" not in result


@pytest.mark.asyncio
async def test_channel_f_filters_denied_neo4j_episode_hit(tmp_path: Path) -> None:
    anima_dir, _denied = _anima_with_deny(tmp_path, "episodes/private")

    class FakeNeo4jBackend:
        def __init__(self) -> None:
            self.recorded: list[object] = []

        async def retrieve(self, *args, **kwargs):
            return [
                SimpleNamespace(
                    source="episode:private",
                    content="DENIED NEO4J CANARY",
                    score=0.99,
                    metadata={"source_file": "episodes/private/secret.md"},
                ),
                SimpleNamespace(
                    source="episode:public",
                    content="# Public graph episode",
                    score=0.8,
                    metadata={"source_file": "episodes/public.md"},
                ),
            ]

        async def record_access(self, memories):
            self.recorded = memories

    backend = FakeNeo4jBackend()
    with patch("core.memory.backend.neo4j_graph.Neo4jGraphBackend", FakeNeo4jBackend):
        result = await channel_f_episodes(
            anima_dir,
            anima_dir / "episodes",
            lambda: None,
            ["test"],
            get_memory_backend=lambda: backend,
        )

    assert 'read_memory_file(path="episodes/public.md")' in result
    assert "episodes/private" not in result
    assert "DENIED NEO4J CANARY" not in result
    assert [memory.source for memory in backend.recorded] == ["episode:public"]


@pytest.mark.asyncio
async def test_current_state_symlink_into_denied_root_is_not_read_for_priming(tmp_path: Path) -> None:
    anima_dir, denied = _anima_with_deny(tmp_path, "private")
    secret = denied / "secret.txt"
    secret.write_text("DENIED STATE CANARY", encoding="utf-8")
    state_dir = anima_dir / "state"
    state_dir.mkdir()
    (state_dir / "current_state.md").symlink_to(secret)
    engine = PrimingEngine(anima_dir)
    engine._extract_keywords = MagicMock(return_value=[])

    with (
        patch.object(engine, "_channel_a_sender_profile", new=AsyncMock(return_value="")),
        patch.object(engine, "_channel_b_recent_activity", new=AsyncMock(return_value="")),
        patch.object(engine, "_channel_c0_important_knowledge", new=AsyncMock(return_value="")),
        patch.object(engine, "_channel_c_related_knowledge", new=AsyncMock(return_value=("", ""))),
        patch.object(engine, "_channel_e_pending_tasks", new=AsyncMock(return_value="")),
        patch.object(engine, "_collect_recent_outbound", new=AsyncMock(return_value="")),
        patch.object(engine, "_channel_f_episodes", new=AsyncMock(return_value="")),
        patch.object(engine, "_collect_pending_human_notifications", new=AsyncMock(return_value="")),
        patch.object(engine, "_channel_g_graph_context", new=AsyncMock(return_value="")),
    ):
        await engine.prime_memories(message="", sender_name="human")

    engine._extract_keywords.assert_called_once_with("")
