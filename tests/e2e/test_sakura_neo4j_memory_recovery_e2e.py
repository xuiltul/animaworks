"""E2E coverage for Sakura Neo4j memory recovery behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_llm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_custom_endpoint_entity_extraction_routes_through_openai_provider() -> None:
    """A bare model id with api_base should reach LiteLLM as openai/<model>."""
    from core.memory.extraction.extractor import FactExtractor

    payload = {
        "entities": [
            {"name": "ExampleOrg", "entity_type": "Organization", "summary": "A company"},
        ]
    }
    cfg = MagicMock()
    cfg.consolidation.llm_model = "anthropic/claude-sonnet-4-6"
    cfg.credentials = {}

    extractor = FactExtractor(
        model="deepseek-v4-flash",
        max_retries=1,
        llm_extra={
            "api_base": "http://localhost:4000/v1",
            "api_key": "dummy",
            "timeout": 120,
        },
    )

    with (
        patch("core.memory._llm_utils.ensure_credentials_in_env"),
        patch("core.config.load_config", return_value=cfg),
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
    ):
        mock_acompletion.return_value = _make_llm_response(json.dumps(payload, ensure_ascii=False))
        entities = await extractor.extract_entities("ExampleOrgについて")

    assert [entity.name for entity in entities] == ["ExampleOrg"]
    kwargs = mock_acompletion.call_args.kwargs
    assert kwargs["model"] == "openai/deepseek-v4-flash"
    assert kwargs["api_base"] == "http://localhost:4000/v1"


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_neo4j_backend_all_scope_retrieves_and_formats_mixed_results(tmp_path) -> None:
    """Neo4jGraphBackend.retrieve(scope='all') should use HybridSearch end to end."""
    from core.memory.backend.neo4j_graph import Neo4jGraphBackend

    async def _mock_execute(query, params=None, **kw):
        q = query.strip()
        if "queryRelationships('fact_embedding'" in q:
            return [
                {
                    "uuid": "fact-1",
                    "fact": "ExampleOrg uses Neo4j memory",
                    "source_name": "ExampleOrg",
                    "target_name": "Neo4j",
                    "valid_at": "2026-05-14T00:00:00Z",
                }
            ]
        if "queryNodes('episode_content_embedding'" in q:
            return [
                {
                    "uuid": "episode-1",
                    "content": "Recent Sakura episode about Neo4j",
                    "source": "chat:sakura",
                    "valid_at": "2026-05-14T00:00:00Z",
                }
            ]
        if "queryNodes('entity_name_embedding'" in q and "score >= $min_score" not in q:
            return [
                {
                    "uuid": "entity-1",
                    "name": "Sakura",
                    "summary": "An Anima using Neo4j memory",
                    "entity_type": "Agent",
                }
            ]
        if "score >= $min_score" in q:
            return []
        return []

    async def _passthrough_rerank(query, items, **kw):
        return [{**item, "ce_score": 0.5} for item in items]

    driver = AsyncMock()
    driver.execute_query = AsyncMock(side_effect=_mock_execute)
    backend = Neo4jGraphBackend(tmp_path)

    mock_reranker = AsyncMock()
    mock_reranker.rerank = _passthrough_rerank

    with (
        patch.object(backend, "_ensure_driver", new_callable=AsyncMock, return_value=driver),
        patch.object(backend, "_embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 384]),
        patch("core.memory.graph.reranker.get_reranker", return_value=mock_reranker),
    ):
        memories = await backend.retrieve("Sakura Neo4j", scope="all", limit=10)

    sources = {memory.source for memory in memories}
    assert {"fact:fact-1", "episode:episode-1", "entity:entity-1"} <= sources
    assert any("ExampleOrg -[RELATES_TO]-> Neo4j" in memory.content for memory in memories)
    assert any(memory.content == "Recent Sakura episode about Neo4j" for memory in memories)
    assert any(memory.content == "Sakura: An Anima using Neo4j memory" for memory in memories)


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_per_anima_neo4j_edge_ontology_drives_extraction_e2e(tmp_path) -> None:
    """status.json edge ontology should appear in prompts and canonicalize LLM facts."""
    from core.memory.extraction.extractor import FactExtractor
    from core.memory.ontology.default import ExtractedEntity

    anima_dir = tmp_path / "sakura"
    anima_dir.mkdir()
    (anima_dir / "status.json").write_text(
        json.dumps(
            {
                "neo4j_edge_types": [
                    {"name": "mentors", "description": "Mentorship relationship"},
                ]
            }
        ),
        encoding="utf-8",
    )

    cfg = MagicMock()
    cfg.memory.neo4j_edge_types = [
        {"name": "advises", "description": "Advice relationship"},
    ]
    extractor = FactExtractor(model="test-model", max_retries=1, anima_dir=anima_dir)
    captured_prompts: list[str] = []

    async def _mock_llm(_system: str, user: str) -> str:
        captured_prompts.append(user)
        return json.dumps(
            {
                "facts": [
                    {
                        "source_entity": "Alice",
                        "target_entity": "Bob",
                        "fact": "Alice mentors Bob",
                        "edge_type": "mentors",
                    }
                ]
            }
        )

    with (
        patch("core.config.models.load_config", return_value=cfg),
        patch.object(extractor, "_call_llm", side_effect=_mock_llm),
    ):
        facts = await extractor.extract_facts(
            "Alice mentors Bob",
            [
                ExtractedEntity(name="Alice", entity_type="Person"),
                ExtractedEntity(name="Bob", entity_type="Person"),
            ],
        )

    assert "`ADVISES`" in captured_prompts[0]
    assert "`MENTORS`" in captured_prompts[0]
    assert facts[0].edge_type == "MENTORS"
    assert facts[0].raw_edge_type is None


@pytest.mark.e2e
def test_search_memory_all_scope_combines_graph_and_legacy_only_scopes_e2e(tmp_path) -> None:
    """search_memory(scope='all') should combine graph memory with legacy-only sections."""
    from core.memory.backend.base import RetrievedMemory
    from core.tooling.handler_memory import MemoryToolsMixin

    mock_backend = MagicMock()
    mock_backend.retrieve = AsyncMock(
        return_value=[
            RetrievedMemory(content="graph memory", score=0.9, source="fact:1"),
        ]
    )
    mock_backend.close = AsyncMock()

    class FakeMemory:
        @property
        def memory_backend(self):
            return mock_backend

        def search_memory_text(self, _query, scope="all", **_kwargs):
            if scope == "common_knowledge":
                return [{"content": "shared policy", "source_file": "common_knowledge/policy.md", "score": 0.7}]
            if scope == "skills":
                return [{"content": "shared skill", "source_file": "common_skills/ops/SKILL.md", "score": 0.6}]
            if scope == "activity_log":
                return [{"content": "activity entry", "source_file": "activity/log.jsonl", "score": 0.5}]
            return []

    class FakeHandler(MemoryToolsMixin):
        def _anima_search_hint(self, _query):
            return None

    handler = FakeHandler.__new__(FakeHandler)
    handler._anima_name = "sakura"
    handler._anima_dir = tmp_path
    handler._context_window = 128_000
    handler._memory = FakeMemory()
    handler._read_paths = set()
    handler._create_neo4j_backend = MagicMock(return_value=mock_backend)

    with patch("core.memory.backend.registry.resolve_backend_type", return_value="neo4j"):
        result = handler._handle_search_memory({"query": "policy", "scope": "all"})

    assert "## Graph Memory" in result
    assert "graph memory" in result
    assert "## Common Knowledge" in result
    assert "shared policy" in result
    assert "## Skills" in result
    assert "shared skill" in result
    assert "## Activity Log" in result
    assert "activity entry" in result


@pytest.mark.e2e
def test_realtime_neo4j_ingest_records_user_and_assistant_turn(tmp_path) -> None:
    """Realtime chat ingest should store the full turn body with stable metadata."""
    from core._anima_messaging import MessagingMixin

    class FakeAnima(MessagingMixin):
        pass

    obj = FakeAnima.__new__(FakeAnima)
    obj.name = "sakura"

    mock_backend = MagicMock()
    mock_backend.__class__ = type("Neo4jGraphBackend", (), {})
    mock_backend._group_id = "sakura"
    mock_backend.ingest_text = AsyncMock(return_value=1)

    class FakeMemory:
        def __init__(self, anima_dir):
            self.anima_dir = anima_dir

        @property
        def memory_backend(self):
            return mock_backend

    obj.memory = FakeMemory(tmp_path)

    cfg = MagicMock()
    cfg.memory = MagicMock()
    cfg.memory.neo4j_realtime_ingest = True

    with (
        patch("core.memory.backend.registry.resolve_backend_type", return_value="neo4j"),
        patch("core.config.models.load_config", return_value=cfg),
    ):
        obj._maybe_neo4j_realtime_ingest(
            "mio",
            "What did ExampleOrg decide?",
            "ExampleOrg decided to use Neo4j memory.",
            thread_id="sakura-main",
            request_id="req-e2e",
        )

    mock_backend.ingest_text.assert_awaited_once_with(
        "mio: What did ExampleOrg decide?\nsakura: ExampleOrg decided to use Neo4j memory.",
        source="chat:sakura:sakura-main",
        metadata={
            "stable_key": "chat:sakura:sakura-main:req-e2e",
            "description": "chat turn sakura-main",
            "thread_id": "sakura-main",
            "request_id": "req-e2e",
        },
    )


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_channel_g_neo4j_retrieval_is_query_aware(tmp_path) -> None:
    """Channel G should use query-aware Neo4j communities and recent facts."""
    from core.memory.backend.neo4j_graph import Neo4jGraphBackend
    from core.memory.priming.channel_g import collect_graph_context

    now = datetime.now(tz=UTC)
    driver = AsyncMock()
    driver.execute_query = AsyncMock(
        return_value=[
            {
                "uuid": "community-frontend",
                "name": "Frontend",
                "summary": "React and UI ownership",
                "score": 0.9,
                "created_at": now.isoformat(),
            }
        ]
    )
    backend = Neo4jGraphBackend(tmp_path, group_id="sakura")
    backend._driver = driver
    backend._schema_ensured = True

    mock_search = MagicMock()
    mock_search.search = AsyncMock(
        return_value=[
            {
                "uuid": "fact-recent",
                "fact": "owns the dashboard UI",
                "source_name": "Sakura",
                "target_name": "React",
                "edge_type": "OWNS",
                "created_at": (now - timedelta(hours=1)).isoformat(),
                "rrf_score": 0.8,
            },
            {
                "uuid": "fact-old",
                "fact": "owns legacy DB migration",
                "source_name": "Sakura",
                "target_name": "Postgres",
                "created_at": (now - timedelta(days=3)).isoformat(),
                "rrf_score": 0.95,
            },
        ]
    )

    with (
        patch.object(backend, "_embed_texts", new_callable=AsyncMock, return_value=[[0.1] * 384]),
        patch("core.memory.graph.search.HybridSearch", return_value=mock_search),
    ):
        context = await collect_graph_context(backend, "frontend dashboard")

    assert "## Communities" in context
    assert "[Frontend] React and UI ownership" in context
    assert "## Recent Facts" in context
    assert "Sakura -[OWNS]-> React: owns the dashboard UI" in context
    assert "Postgres" not in context
    community_query, community_params = driver.execute_query.call_args.args
    assert "community_fulltext" in community_query
    assert community_params["query"] == "frontend dashboard"


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_scheduled_community_detection_uses_per_anima_neo4j_override(tmp_path) -> None:
    """Scheduled community detection should follow status.json backend resolution."""
    from core.lifecycle.system_consolidation import SystemConsolidationMixin

    anima_dir = tmp_path / "sakura"
    anima_dir.mkdir()
    (anima_dir / "status.json").write_text('{"memory_backend": "neo4j"}', encoding="utf-8")

    mock_anima = MagicMock()
    mock_anima.name = "sakura"
    mock_anima.memory.anima_dir = anima_dir

    mock_backend = MagicMock()
    mock_backend._group_id = "sakura"
    mock_backend._resolve_background_model.return_value = "test-model"
    mock_backend._resolve_extraction_config.return_value = ("test-model", {}, "")
    mock_backend._resolve_locale.return_value = "ja"
    mock_backend._ensure_driver = AsyncMock()
    mock_backend.close = AsyncMock()

    with (
        patch("core.memory.backend.registry.get_backend", return_value=mock_backend) as mock_get_backend,
        patch("core.memory.graph.community.CommunityDetector") as MockDetector,
    ):
        mock_detector = MagicMock()
        mock_detector.detect_and_store = AsyncMock(return_value=[object()])
        mock_detector.get_community_stats = AsyncMock(return_value={"communities": 1, "memberships": 3})
        MockDetector.return_value = mock_detector

        await SystemConsolidationMixin._detect_communities_if_neo4j(mock_anima)

    mock_get_backend.assert_called_once_with("neo4j", anima_dir)
    mock_detector.detect_and_store.assert_awaited_once()
    mock_detector.get_community_stats.assert_awaited_once()
    mock_backend.close.assert_awaited_once()
