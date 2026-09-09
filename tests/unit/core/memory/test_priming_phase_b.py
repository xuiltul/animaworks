from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase B priming fixes: trigger wiring (F15), init-latch TTL (F12), guardrail line (F16)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import core.memory.priming.utils as utils_module
from core.memory.priming.channel_c import channel_c_related_knowledge
from core.memory.priming.channel_f import channel_f_episodes
from core.memory.priming.format import format_priming_section
from core.memory.priming.gate import PrimingGateDecision, PrimingPlan, PrimingRenderMode
from core.memory.priming.result import PrimingResult
from core.memory.priming.utils import RetrieverCache, normalize_trigger


def _fake_unified_search(results: list[dict] | None = None, *, abstain: bool = False) -> MagicMock:
    searcher = MagicMock()
    searcher.search_many.return_value = results or []
    searcher.last_search_meta = {"abstain": abstain, "abstain_reason": ""}
    return searcher


# ── F15: trigger normalization ────────────────────────────────────


class TestNormalizeTrigger:
    def test_known_triggers_pass_through(self) -> None:
        for key in ("chat", "inbox", "heartbeat", "task", "cron", "tool"):
            assert normalize_trigger(key) == key

    def test_inbox_sender_form_normalized(self) -> None:
        assert normalize_trigger("inbox:alice") == "inbox"
        assert normalize_trigger("inbox:alice,bob") == "inbox"

    def test_unknown_trigger_falls_back_to_chat(self) -> None:
        assert normalize_trigger("message:yamada") == "chat"
        assert normalize_trigger("") == "chat"
        assert normalize_trigger("HeartBeat") == "heartbeat"


# ── F15: trigger reaches search_many from channel C / F ───────────


@pytest.mark.asyncio
async def test_channel_c_propagates_heartbeat_trigger(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "test"
    (anima_dir / "knowledge").mkdir(parents=True)
    searcher = _fake_unified_search([])

    with patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=searcher):
        await channel_c_related_knowledge(
            anima_dir,
            anima_dir / "knowledge",
            lambda: MagicMock(),
            ["deploy"],
            trigger="heartbeat",
        )

    assert searcher.search_many.call_args.kwargs["trigger"] == "heartbeat"


@pytest.mark.asyncio
async def test_channel_f_propagates_heartbeat_trigger(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "test"
    (anima_dir / "episodes").mkdir(parents=True)
    searcher = _fake_unified_search([])

    with patch("core.memory.priming.channel_f.UnifiedMemorySearch", return_value=searcher):
        await channel_f_episodes(
            anima_dir,
            anima_dir / "episodes",
            lambda: MagicMock(),
            ["deploy"],
            message="deploy",
            trigger="heartbeat",
        )

    assert searcher.search_many.call_args.kwargs["trigger"] == "heartbeat"


@pytest.mark.asyncio
async def test_channel_c_normalizes_inbox_sender_trigger(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "test"
    (anima_dir / "knowledge").mkdir(parents=True)
    searcher = _fake_unified_search([])

    with patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=searcher):
        await channel_c_related_knowledge(
            anima_dir,
            anima_dir / "knowledge",
            lambda: MagicMock(),
            ["deploy"],
            trigger="inbox:alice",
        )

    assert searcher.search_many.call_args.kwargs["trigger"] == "inbox"


@pytest.mark.asyncio
async def test_inbox_human_activity_is_in_channel_c_and_f_queries(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "test"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "episodes").mkdir(parents=True)
    c_searcher = _fake_unified_search([])
    f_searcher = _fake_unified_search([])
    recent_human = ["human-only inbox context"]

    with patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=c_searcher):
        await channel_c_related_knowledge(
            anima_dir,
            anima_dir / "knowledge",
            lambda: MagicMock(),
            [],
            recent_human_messages=recent_human,
            trigger="inbox",
        )
    with patch("core.memory.priming.channel_f.UnifiedMemorySearch", return_value=f_searcher):
        await channel_f_episodes(
            anima_dir,
            anima_dir / "episodes",
            lambda: MagicMock(),
            [],
            recent_human_messages=recent_human,
            trigger="inbox",
        )

    assert "human-only inbox context" in c_searcher.search_many.call_args.args[0]
    assert "human-only inbox context" in f_searcher.search_many.call_args.args[0]


@pytest.mark.asyncio
async def test_prime_memories_wires_channel_to_trigger(tmp_path: Path) -> None:
    """End-to-end: prime_memories(channel="heartbeat") reaches search_many with heartbeat policy."""
    from core.memory.priming.engine import PrimingEngine

    anima_dir = tmp_path / "animas" / "test"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "episodes").mkdir(parents=True)
    engine = PrimingEngine(anima_dir)
    engine._retriever = MagicMock()
    engine._retriever.indexer = MagicMock()

    c_searcher = _fake_unified_search([])
    f_searcher = _fake_unified_search([])

    with (
        patch("core.memory.priming.channel_c.UnifiedMemorySearch", return_value=c_searcher),
        patch("core.memory.priming.channel_f.UnifiedMemorySearch", return_value=f_searcher),
    ):
        await engine.prime_memories("reflection text", channel="heartbeat")

    assert c_searcher.search_many.call_args.kwargs["trigger"] == "heartbeat"
    assert f_searcher.search_many.call_args.kwargs["trigger"] == "heartbeat"


# ── F12: init-latch TTL retry ─────────────────────────────────────


class TestRetrieverCacheTTL:
    def test_failure_latched_within_ttl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = RetrieverCache()
        knowledge_dir = tmp_path / "knowledge"  # does not exist → failure path

        clock = {"t": 1000.0}
        monkeypatch.setattr(utils_module.time, "monotonic", lambda: clock["t"])

        assert cache.get_or_create(tmp_path, knowledge_dir) is None
        # Within TTL the latch short-circuits even though the dir now exists.
        knowledge_dir.mkdir()
        clock["t"] = 1000.0 + utils_module._INIT_RETRY_TTL_SECONDS - 1
        assert cache.get_or_create(tmp_path, knowledge_dir) is None

    def test_retry_after_ttl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = RetrieverCache()
        knowledge_dir = tmp_path / "knowledge"

        clock = {"t": 1000.0}
        monkeypatch.setattr(utils_module.time, "monotonic", lambda: clock["t"])

        # First attempt fails (missing dir).
        assert cache.get_or_create(tmp_path, knowledge_dir) is None
        knowledge_dir.mkdir()

        # Advance past TTL → a fresh init attempt is made.
        clock["t"] = 1000.0 + utils_module._INIT_RETRY_TTL_SECONDS + 1

        sentinel = object()
        with (
            patch("core.memory.rag.singleton.get_vector_store", return_value=MagicMock()),
            patch("core.memory.rag.indexer.MemoryIndexer", return_value=MagicMock()),
            patch("core.memory.rag.MemoryRetriever", return_value=sentinel),
        ):
            result = cache.get_or_create(tmp_path, knowledge_dir)

        assert result is sentinel


class TestBackendLatchTTL:
    def test_backend_latch_retries_after_ttl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.memory.priming import engine as engine_module
        from core.memory.priming.engine import PrimingEngine

        engine = PrimingEngine(tmp_path)
        clock = {"t": 500.0}
        monkeypatch.setattr(engine_module.time, "monotonic", lambda: clock["t"])

        with patch(
            "core.memory.backend.registry.resolve_backend_type",
            side_effect=RuntimeError("transient"),
        ):
            assert engine._get_memory_backend() is None
            # Second call within TTL: latched, not retried.
            clock["t"] = 500.0 + engine_module._BACKEND_INIT_RETRY_TTL_SECONDS - 1
            assert engine._get_memory_backend() is None
        assert engine._memory_backend_init_failed is True

        # After TTL, a retry succeeds.
        clock["t"] = 500.0 + engine_module._BACKEND_INIT_RETRY_TTL_SECONDS + 1
        mock_backend = MagicMock()
        with (
            patch("core.memory.backend.registry.resolve_backend_type", return_value="neo4j"),
            patch("core.memory.backend.registry.get_backend", return_value=mock_backend),
        ):
            assert engine._get_memory_backend() is mock_backend
        assert engine._memory_backend_init_failed is False


# ── F16: guardrail search-before-action line ──────────────────────


def _plan(require_search: bool) -> PrimingPlan:
    decision = PrimingGateDecision(
        channel="sender_profile",
        visible=True,
        render_mode=PrimingRenderMode.ANCHOR,
        reason="test",
        priority=2,
        require_search_before_action=require_search,
    )
    return PrimingPlan(
        channel_decisions={"sender_profile": decision},
        evidence_mode=False,
        require_search_before_action=require_search,
    )


class TestSearchBeforeActionLine:
    def test_line_emitted_when_required(self) -> None:
        result = PrimingResult(sender_profile="profile", gate_plan=_plan(True))
        formatted = format_priming_section(result)
        assert "search_memory" in formatted

    def test_line_absent_when_not_required(self) -> None:
        result = PrimingResult(sender_profile="profile", gate_plan=_plan(False))
        formatted = format_priming_section(result)
        assert "search_memory" not in formatted
