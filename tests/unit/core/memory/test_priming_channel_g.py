"""Unit tests for Channel G: graph context (community + recent facts).

All tests use mocked MemoryBackend — no real Neo4j or LLM calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from core.memory.backend.base import RetrievedMemory
from core.memory.priming.channel_g import collect_graph_context
from core.memory.priming.constants import _BUDGET_GRAPH_CONTEXT

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_backend(
    communities: list[RetrievedMemory] | None = None,
    facts: list[RetrievedMemory] | None = None,
) -> MagicMock:
    """Create a mock MemoryBackend with configured return values."""
    backend = MagicMock()
    backend.get_community_context = AsyncMock(return_value=communities or [])
    backend.get_recent_facts = AsyncMock(return_value=facts or [])
    return backend


def _mem(content: str, score: float = 1.0) -> RetrievedMemory:
    return RetrievedMemory(content=content, score=score, source="test")


# ── TestCollectGraphContext ────────────────────────────────────────────────


class TestCollectGraphContext:
    """Tests for collect_graph_context()."""

    async def test_returns_both_communities_and_facts(self) -> None:
        backend = _make_backend(
            communities=[_mem("[Team A] Frontend team")],
            facts=[_mem("Alice → React: works with")],
        )

        result = await collect_graph_context(backend, "test query")

        assert "## Communities" in result
        assert "[Team A] Frontend team" in result
        assert "## Recent Facts" in result
        assert "Alice → React: works with" in result

    async def test_communities_only(self) -> None:
        backend = _make_backend(communities=[_mem("[Eng] Engineers")])

        result = await collect_graph_context(backend, "test")

        assert "## Communities" in result
        assert "## Recent Facts" not in result

    async def test_facts_only(self) -> None:
        backend = _make_backend(facts=[_mem("Bob → Python: uses")])

        result = await collect_graph_context(backend, "test")

        assert "## Recent Facts" in result
        assert "## Communities" not in result

    async def test_empty_returns_empty_string(self) -> None:
        backend = _make_backend()

        result = await collect_graph_context(backend, "test")

        assert result == ""

    async def test_backend_exception_returns_empty(self) -> None:
        backend = MagicMock()
        backend.get_community_context = AsyncMock(side_effect=RuntimeError("Neo4j down"))
        backend.get_recent_facts = AsyncMock(side_effect=RuntimeError("Neo4j down"))

        result = await collect_graph_context(backend, "test")

        assert result == ""

    async def test_calls_backend_with_correct_params(self) -> None:
        backend = _make_backend()

        await collect_graph_context(backend, "my query", budget_tokens=300)

        backend.get_community_context.assert_awaited_once_with("my query", limit=3)
        backend.get_recent_facts.assert_awaited_once_with("my query", hours=24, limit=10)

    async def test_truncates_to_budget(self) -> None:
        long_facts = [_mem(f"Fact number {i} with some longer content here") for i in range(50)]
        backend = _make_backend(facts=long_facts)

        result = await collect_graph_context(backend, "test", budget_tokens=50)

        chars_per_token = 4
        assert len(result) <= 50 * chars_per_token + 100  # rough check

    async def test_default_budget_matches_constant(self) -> None:
        assert _BUDGET_GRAPH_CONTEXT == 500


# ── TestPrimingResultGraphContext ──────────────────────────────────────────


class TestPrimingResultGraphContext:
    """Tests for graph_context field in PrimingResult."""

    def test_graph_context_default_empty(self) -> None:
        from core.memory.priming.engine import PrimingResult

        result = PrimingResult()
        assert result.graph_context == ""

    def test_graph_context_included_in_is_empty(self) -> None:
        from core.memory.priming.engine import PrimingResult

        result = PrimingResult(graph_context="some context")
        assert not result.is_empty()

    def test_graph_context_included_in_total_chars(self) -> None:
        from core.memory.priming.engine import PrimingResult

        result = PrimingResult(graph_context="hello")
        assert result.total_chars() == 5


# ── TestFormatPrimingSectionGraphContext ───────────────────────────────────


class TestFormatPrimingSectionGraphContext:
    """Tests for graph_context rendering in format_priming_section."""

    def test_graph_context_rendered_in_output(self) -> None:
        from core.memory.priming.engine import PrimingResult
        from core.memory.priming.format import format_priming_section

        result = PrimingResult(graph_context="## Communities\n- [Team A] Engineers")

        with patch(
            "core.execution._sanitize.wrap_priming", side_effect=lambda tag, content, **kw: f"<{tag}>{content}</{tag}>"
        ):
            output = format_priming_section(result, "human")

        assert "<graph_context>" in output
        assert "Team A" in output

    def test_empty_graph_context_not_rendered(self) -> None:
        from core.memory.priming.engine import PrimingResult
        from core.memory.priming.format import format_priming_section

        result = PrimingResult(sender_profile="some profile")

        with patch(
            "core.execution._sanitize.wrap_priming", side_effect=lambda tag, content, **kw: f"<{tag}>{content}</{tag}>"
        ):
            output = format_priming_section(result, "human")

        assert "graph_context" not in output


# ── TestPrimingEngineChannelG ─────────────────────────────────────────────


class TestPrimingEngineChannelG:
    """Tests for channel G integration in PrimingEngine."""

    async def test_channel_g_called_in_prime_memories(self, tmp_path) -> None:
        from core.memory.priming.engine import PrimingEngine

        engine = PrimingEngine(tmp_path)

        backend = _make_backend(
            facts=[_mem("Alice → Bob: knows")],
        )
        engine._memory_backend = backend

        with (
            patch.object(engine, "_channel_a_sender_profile", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_b_recent_activity", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_c0_important_knowledge", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_c_related_knowledge", new_callable=AsyncMock, return_value=("", "")),
            patch.object(engine, "_channel_e_pending_tasks", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_collect_recent_outbound", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_f_episodes", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_collect_pending_human_notifications", new_callable=AsyncMock, return_value=""),
        ):
            result = await engine.prime_memories("test message")

        assert "Alice → Bob: knows" in result.graph_context

    async def test_channel_g_empty_when_no_backend(self, tmp_path) -> None:
        from core.memory.priming.engine import PrimingEngine

        engine = PrimingEngine(tmp_path)
        engine._memory_backend_init_failed = True

        with (
            patch.object(engine, "_channel_a_sender_profile", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_b_recent_activity", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_c0_important_knowledge", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_c_related_knowledge", new_callable=AsyncMock, return_value=("", "")),
            patch.object(engine, "_channel_e_pending_tasks", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_collect_recent_outbound", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_channel_f_episodes", new_callable=AsyncMock, return_value=""),
            patch.object(engine, "_collect_pending_human_notifications", new_callable=AsyncMock, return_value=""),
        ):
            result = await engine.prime_memories("test message")

        assert result.graph_context == ""

    def test_get_memory_backend_lazy_init(self, tmp_path) -> None:
        from core.memory.priming.engine import PrimingEngine

        engine = PrimingEngine(tmp_path)
        mock_backend = MagicMock()

        with (
            patch("core.memory.backend.registry.get_backend", return_value=mock_backend),
            patch("core.config.models.load_config") as mock_cfg,
        ):
            mock_cfg.return_value = MagicMock(memory=MagicMock(backend="neo4j"))
            result = engine._get_memory_backend()

        assert result is mock_backend

    def test_get_memory_backend_caches_failure(self, tmp_path) -> None:
        from core.memory.priming.engine import PrimingEngine

        engine = PrimingEngine(tmp_path)

        with patch("core.memory.backend.registry.resolve_backend_type", side_effect=RuntimeError("no config")):
            result1 = engine._get_memory_backend()
            result2 = engine._get_memory_backend()

        assert result1 is None
        assert result2 is None
        assert engine._memory_backend_init_failed is True
