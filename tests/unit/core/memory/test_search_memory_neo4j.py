"""Tests for Issue #15 — search_memory Neo4j backend integration.

Covers:
- Neo4j scope routing (_should_use_neo4j)
- Scope mapping (knowledge→fact, episodes→episode, etc.)
- Legacy-only scopes (common_knowledge, skills, activity_log)
- Fallback on Neo4j failure
- Output formatting compatibility
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, patch

import pytest

from core.memory.backend.base import RetrievedMemory

# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.search_memory_text = MagicMock(return_value=[])
    return mem


def _make_handler_mixin(mock_memory, backend_class_name="Neo4jGraphBackend"):
    """Create a minimal MemoryToolsMixin-like object for testing."""
    from core.tooling.handler_memory import MemoryToolsMixin

    class FakeHandler(MemoryToolsMixin):
        _anima_name = "test"
        _anima_dir = "/tmp/test"
        _context_window = 128_000
        _read_paths: set[str] = set()

        def _anima_search_hint(self, query):
            return None

    handler = FakeHandler.__new__(FakeHandler)
    handler._memory = mock_memory
    handler._anima_name = "test"
    handler._anima_dir = "/tmp/test"
    handler._context_window = 128_000
    handler._read_paths = set()

    backend_cls = type(backend_class_name, (), {})
    mock_backend = backend_cls()
    mock_backend.retrieve = AsyncMock(return_value=[])
    mock_backend.close = AsyncMock()
    handler._create_neo4j_backend = MagicMock(return_value=mock_backend)
    type(mock_memory).memory_backend = PropertyMock(return_value=mock_backend)

    return handler, mock_backend


# ── TestShouldUseNeo4j ───────────────────────────────────


class TestShouldUseNeo4j:
    def test_returns_true_for_neo4j_backend(self, mock_memory) -> None:
        handler, _ = _make_handler_mixin(mock_memory, "Neo4jGraphBackend")
        assert handler._should_use_neo4j("knowledge") is True

    def test_returns_false_for_legacy_backend(self, mock_memory) -> None:
        handler, _ = _make_handler_mixin(mock_memory, "LegacyRAGBackend")
        assert handler._should_use_neo4j("knowledge") is False

    def test_returns_false_for_activity_log(self, mock_memory) -> None:
        handler, _ = _make_handler_mixin(mock_memory, "Neo4jGraphBackend")
        assert handler._should_use_neo4j("activity_log") is False

    def test_returns_false_for_common_knowledge(self, mock_memory) -> None:
        handler, _ = _make_handler_mixin(mock_memory, "Neo4jGraphBackend")
        assert handler._should_use_neo4j("common_knowledge") is False

    def test_returns_false_for_skills(self, mock_memory) -> None:
        handler, _ = _make_handler_mixin(mock_memory, "Neo4jGraphBackend")
        assert handler._should_use_neo4j("skills") is False

    def test_returns_false_on_backend_error(self, mock_memory) -> None:
        handler, _ = _make_handler_mixin(mock_memory, "Neo4jGraphBackend")
        type(mock_memory).memory_backend = PropertyMock(side_effect=RuntimeError("fail"))
        assert handler._should_use_neo4j("all") is False


# ── TestNeo4jScopeMap ────────────────────────────────────


class TestNeo4jScopeMap:
    def test_knowledge_maps_to_fact(self) -> None:
        from core.tooling.handler_memory import MemoryToolsMixin

        assert MemoryToolsMixin._NEO4J_SCOPE_MAP["knowledge"] == "fact"

    def test_episodes_maps_to_episode(self) -> None:
        from core.tooling.handler_memory import MemoryToolsMixin

        assert MemoryToolsMixin._NEO4J_SCOPE_MAP["episodes"] == "episode"

    def test_procedures_maps_to_fact(self) -> None:
        from core.tooling.handler_memory import MemoryToolsMixin

        assert MemoryToolsMixin._NEO4J_SCOPE_MAP["procedures"] == "fact"

    def test_all_maps_to_all(self) -> None:
        from core.tooling.handler_memory import MemoryToolsMixin

        assert MemoryToolsMixin._NEO4J_SCOPE_MAP["all"] == "all"


class TestScopePolicy:
    def test_scope_policy_defines_neo4j_and_legacy_boundaries(self) -> None:
        from core.memory.scope_policy import (
            LEGACY_ONLY_SCOPES_FOR_ALL,
            is_legacy_only_scope,
            is_neo4j_backed_scope,
            neo4j_scope_for,
            title_for_legacy_scope,
        )

        assert is_neo4j_backed_scope("knowledge") is True
        assert is_neo4j_backed_scope("all") is True
        assert is_legacy_only_scope("common_knowledge") is True
        assert is_legacy_only_scope("skills") is True
        assert is_legacy_only_scope("facts") is True
        assert is_legacy_only_scope("activity_log") is True
        assert LEGACY_ONLY_SCOPES_FOR_ALL == ("common_knowledge", "skills", "facts", "activity_log")
        assert neo4j_scope_for("episodes") == "episode"
        assert title_for_legacy_scope("common_knowledge") == "Common Knowledge"
        assert title_for_legacy_scope("facts") == "Facts"


# ── TestCreateNeo4jBackend ────────────────────────────────


class TestCreateNeo4jBackend:
    def test_fresh_backend_uses_registry_config(self, tmp_path: Path) -> None:
        from core.tooling.handler_memory import MemoryToolsMixin

        class FakeHandler(MemoryToolsMixin):
            pass

        handler = FakeHandler.__new__(FakeHandler)
        handler._anima_dir = str(tmp_path)

        mock_cfg = MagicMock()
        mock_cfg.memory.neo4j.uri = "bolt://configured:7687"
        mock_cfg.memory.neo4j.user = "configured-user"
        mock_cfg.memory.neo4j.password = "configured-password"
        mock_cfg.memory.neo4j.database = "configured-db"

        with patch("core.config.models.load_config", return_value=mock_cfg):
            backend = handler._create_neo4j_backend()

        assert backend._uri == "bolt://configured:7687"
        assert backend._user == "configured-user"
        assert backend._password == "configured-password"
        assert backend._database == "configured-db"


# ── TestSearchViaNeo4j ───────────────────────────────────


class TestSearchViaNeo4j:
    def test_returns_formatted_results(self, mock_memory) -> None:
        handler, mock_backend = _make_handler_mixin(mock_memory)
        mock_backend.retrieve = AsyncMock(
            return_value=[
                RetrievedMemory(
                    content="Alice → Bob: works together",
                    score=0.95,
                    source="fact:uuid-1",
                    metadata={
                        "valid_at_iso": "2026-06-03T10:00:00+09:00",
                        "valid_until": "",
                        "recorded_at": "2026-06-03T10:01:00+09:00",
                    },
                ),
            ]
        )

        result = handler._search_via_neo4j("Alice", "knowledge", 0)
        assert result is not None
        assert "Alice → Bob" in result
        assert "score=0.95" in result
        assert "valid: 2026-06-03〜present | recorded: 2026-06-03" in result
        assert "graph" in result
        handler._create_neo4j_backend.assert_called_once()
        mock_backend.close.assert_awaited_once()

    def test_returns_empty_string_on_no_results(self, mock_memory) -> None:
        handler, mock_backend = _make_handler_mixin(mock_memory)
        mock_backend.retrieve = AsyncMock(return_value=[])

        result = handler._search_via_neo4j("nothing", "all", 0)
        assert result == ""
        mock_backend.close.assert_awaited_once()

    def test_returns_none_on_failure(self, mock_memory) -> None:
        handler, mock_backend = _make_handler_mixin(mock_memory)
        mock_backend.retrieve = AsyncMock(side_effect=RuntimeError("Neo4j down"))

        result = handler._search_via_neo4j("query", "all", 0)
        assert result is None
        mock_backend.close.assert_awaited_once()

    def test_offset_skips_results(self, mock_memory) -> None:
        handler, mock_backend = _make_handler_mixin(mock_memory)
        mock_backend.retrieve = AsyncMock(
            return_value=[
                RetrievedMemory(content="result1", score=0.9, source="fact:1"),
                RetrievedMemory(content="result2", score=0.8, source="fact:2"),
                RetrievedMemory(content="result3", score=0.7, source="fact:3"),
            ]
        )

        result = handler._search_via_neo4j("query", "all", 1)
        assert result is not None
        assert "result1" not in result
        assert "result2" in result

    def test_uses_scope_mapping_in_retrieve_call(self, mock_memory) -> None:
        handler, mock_backend = _make_handler_mixin(mock_memory)
        mock_backend.retrieve = AsyncMock(
            return_value=[
                RetrievedMemory(content="fact result", score=0.9, source="fact:1"),
            ]
        )

        handler._search_via_neo4j("query", "knowledge", 0)

        mock_backend.retrieve.assert_awaited_once()
        assert mock_backend.retrieve.await_args.kwargs["scope"] == "fact"

    def test_consecutive_searches_use_separate_backends(self, mock_memory) -> None:
        handler, first_backend = _make_handler_mixin(mock_memory)
        second_backend_cls = type("Neo4jGraphBackend", (), {})
        second_backend = second_backend_cls()
        second_backend.retrieve = AsyncMock(
            return_value=[
                RetrievedMemory(content="second", score=0.8, source="episode:2"),
            ]
        )
        second_backend.close = AsyncMock()

        first_backend.retrieve = AsyncMock(
            return_value=[
                RetrievedMemory(content="first", score=0.9, source="episode:1"),
            ]
        )
        handler._create_neo4j_backend = MagicMock(side_effect=[first_backend, second_backend])

        first_result = handler._search_via_neo4j("query", "episodes", 0)
        second_result = handler._search_via_neo4j("query", "episodes", 0)

        assert first_result is not None
        assert "first" in first_result
        assert second_result is not None
        assert "second" in second_result
        assert handler._create_neo4j_backend.call_count == 2
        first_backend.close.assert_awaited_once()
        second_backend.close.assert_awaited_once()


# ── TestHandleSearchMemoryIntegration ────────────────────


class TestHandleSearchMemoryIntegration:
    def test_neo4j_path_used_when_available(self, mock_memory) -> None:
        handler, mock_backend = _make_handler_mixin(mock_memory)
        mock_backend.retrieve = AsyncMock(
            return_value=[
                RetrievedMemory(content="graph result", score=0.9, source="fact:1"),
            ]
        )

        result = handler._handle_search_memory({"query": "test", "scope": "knowledge"})
        assert "graph result" in result
        mock_memory.search_memory_text.assert_not_called()

    def test_legacy_path_for_activity_log(self, mock_memory) -> None:
        handler, _ = _make_handler_mixin(mock_memory)
        mock_memory.search_memory_text.return_value = [
            {"content": "legacy result", "source_file": "log.jsonl", "score": 0.5, "search_method": "bm25"}
        ]

        handler._handle_search_memory({"query": "test", "scope": "activity_log"})
        mock_memory.search_memory_text.assert_called_once()

    def test_fallback_to_legacy_on_neo4j_failure(self, mock_memory) -> None:
        handler, mock_backend = _make_handler_mixin(mock_memory)
        mock_backend.retrieve = AsyncMock(side_effect=RuntimeError("crash"))
        mock_memory.search_memory_text.return_value = [
            {"content": "fallback", "source_file": "kb.md", "score": 0.5, "search_method": "vector"}
        ]

        handler._handle_search_memory({"query": "test", "scope": "knowledge"})
        mock_memory.search_memory_text.assert_called_once()

    def test_legacy_backend_never_routes_to_neo4j(self, mock_memory) -> None:
        handler, _ = _make_handler_mixin(mock_memory, "LegacyRAGBackend")
        mock_memory.search_memory_text.return_value = []

        handler._handle_search_memory({"query": "test", "scope": "all"})
        mock_memory.search_memory_text.assert_called_once()

    def test_all_scope_returns_graph_plus_legacy_only_sections(self, mock_memory) -> None:
        handler, mock_backend = _make_handler_mixin(mock_memory)
        mock_backend.retrieve = AsyncMock(
            return_value=[
                RetrievedMemory(content="graph result", score=0.9, source="fact:1"),
            ]
        )

        def legacy_side_effect(_query, scope="all", **_kwargs):
            return {
                "common_knowledge": [
                    {
                        "content": "common result",
                        "source_file": "common_knowledge/policy.md",
                        "score": 0.7,
                        "search_method": "vector",
                    }
                ],
                "skills": [
                    {
                        "content": "skill result",
                        "source_file": "common_skills/skill/SKILL.md",
                        "score": 0.6,
                        "search_method": "vector",
                    }
                ],
                "facts": [
                    {
                        "content": "fact result",
                        "source_file": "facts/2026-06-03.jsonl",
                        "score": 0.65,
                        "search_method": "keyword_fallback",
                    }
                ],
                "activity_log": [
                    {
                        "content": "activity result",
                        "source_file": "activity/log.jsonl",
                        "score": 0.5,
                        "search_method": "bm25",
                    }
                ],
            }.get(scope, [])

        mock_memory.search_memory_text.side_effect = legacy_side_effect

        result = handler._handle_search_memory({"query": "test", "scope": "all"})

        assert "hybrid, all" in result
        assert "## Graph Memory" in result
        assert "graph result" in result
        assert "## Common Knowledge" in result
        assert "common result" in result
        assert "## Skills" in result
        assert "skill result" in result
        assert "## Facts" in result
        assert "fact result" in result
        assert "## Activity Log" in result
        assert "activity result" in result
        assert mock_backend.retrieve.await_args.kwargs["scope"] == "all"
        mock_memory.search_memory_text.assert_has_calls(
            [
                call("test", scope="common_knowledge", offset=0, context_window=128_000),
                call("test", scope="skills", offset=0, context_window=128_000),
                call("test", scope="facts", offset=0, context_window=128_000),
                call("test", scope="activity_log", offset=0, context_window=128_000),
            ]
        )

    def test_all_scope_offset_applies_after_section_assembly(self, mock_memory) -> None:
        handler, mock_backend = _make_handler_mixin(mock_memory)
        mock_backend.retrieve = AsyncMock(
            return_value=[
                RetrievedMemory(content="graph result", score=0.9, source="fact:1"),
            ]
        )

        def legacy_side_effect(_query, scope="all", **_kwargs):
            if scope != "common_knowledge":
                return []
            return [
                {
                    "content": "common result",
                    "source_file": "common_knowledge/policy.md",
                    "score": 0.7,
                    "search_method": "vector",
                }
            ]

        mock_memory.search_memory_text.side_effect = legacy_side_effect

        result = handler._handle_search_memory({"query": "test", "scope": "all", "offset": 1})

        assert "graph result" not in result
        assert "## Graph Memory" not in result
        assert "## Common Knowledge" in result
        assert "common result" in result
        assert "[2]" in result

    def test_all_scope_falls_back_to_legacy_all_on_neo4j_failure(self, mock_memory) -> None:
        handler, mock_backend = _make_handler_mixin(mock_memory)
        mock_backend.retrieve = AsyncMock(side_effect=RuntimeError("crash"))
        mock_memory.search_memory_text.return_value = [
            {"content": "fallback", "source_file": "kb.md", "score": 0.5, "search_method": "vector"}
        ]

        result = handler._handle_search_memory({"query": "test", "scope": "all"})

        assert "fallback" in result
        mock_memory.search_memory_text.assert_called_once_with(
            "test",
            scope="all",
            offset=0,
            context_window=128_000,
        )
