from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.schemas import ConsolidationConfig, PrimingConfig, RAGConfig
from core.memory.consolidation import ConsolidationEngine
from core.memory.priming.engine import PrimingEngine
from core.memory.priming.policy import resolve_priming_policy


def _mock_channels(engine: PrimingEngine) -> dict[str, AsyncMock]:
    channels = {}
    for name in (
        "_channel_a_sender_profile",
        "_channel_b_recent_activity",
        "_channel_c0_important_knowledge",
        "_channel_c_related_knowledge",
        "_channel_e_pending_tasks",
        "_collect_recent_outbound",
        "_channel_f_episodes",
        "_channel_g_graph_context",
        "_collect_pending_human_notifications",
    ):
        channels[name] = AsyncMock(return_value=("", "") if name == "_channel_c_related_knowledge" else "")
        setattr(engine, name, channels[name])
    return channels


@pytest.mark.parametrize("channel,intent", [("heartbeat", ""), ("cron", ""), ("inbox", "report")])
@pytest.mark.asyncio
async def test_compact_simple_event_never_starts_general_search(tmp_path: Path, channel: str, intent: str):
    engine = PrimingEngine(tmp_path)
    channels = _mock_channels(engine)
    channels["_channel_e_pending_tasks"].return_value = "Deadline: tomorrow; approval required"
    channels["_collect_pending_human_notifications"].return_value = "Human decision pending " * 500
    result = await engine.prime_memories(
        "Routine result", channel=channel, intent=intent, profile="compact", max_tokens=200
    )
    for name in (
        "_channel_b_recent_activity",
        "_channel_c_related_knowledge",
        "_channel_f_episodes",
        "_channel_g_graph_context",
    ):
        channels[name].assert_not_called()
    channels["_channel_c0_important_knowledge"].assert_awaited_once_with([], trigger=channel, resident_only=True)
    assert "approval required" in result.pending_tasks
    assert result.pending_human_notifications == "Human decision pending " * 500


@pytest.mark.asyncio
async def test_compact_question_keeps_trusted_and_untrusted_separate(tmp_path: Path):
    engine = PrimingEngine(tmp_path)
    channels = _mock_channels(engine)
    channels["_channel_c_related_knowledge"].return_value = ("trusted pointer", "external pointer")
    result = await engine.prime_memories("Customer policy?", channel="chat", profile="compact")
    channels["_channel_c_related_knowledge"].assert_awaited_once()
    assert "trusted pointer" in result.related_knowledge
    assert result.related_knowledge_untrusted == "external pointer"
    channels["_channel_f_episodes"].assert_not_called()


@pytest.mark.asyncio
async def test_compact_never_truncates_an_itemized_memory_pointer(tmp_path: Path):
    from core.memory.priming.items import ItemizedMemory, MemoryItem

    engine = PrimingEngine(tmp_path)
    channels = _mock_channels(engine)
    long_pointer = "details " * 1000 + ' read_memory_file(path="knowledge/complete-path.md")'
    item = MemoryItem(source="related_knowledge", key="long", text=long_pointer)
    channels["_channel_c_related_knowledge"].return_value = (ItemizedMemory(long_pointer, [item]), "")
    result = await engine.prime_memories("policy?", channel="chat", profile="compact", max_tokens=200)
    assert not result.related_knowledge.strip()


@pytest.mark.asyncio
async def test_compact_cap_applies_to_proportional_heartbeat_budget(tmp_path: Path):
    engine = PrimingEngine(tmp_path, context_window=1_000_000)
    engine._adjust_token_budget = MagicMock(return_value=50_000)
    engine._prime_compact = AsyncMock()
    await engine.prime_memories("", channel="heartbeat", profile="compact", max_tokens=1200, enable_dynamic_budget=True)
    assert engine._prime_compact.call_args.args[4] == 1200


def test_per_anima_profile_overrides_global_without_model_assumptions(tmp_path: Path):
    config = SimpleNamespace(priming=PrimingConfig(profile="compact", dynamic_budget=False, max_tokens=1500))
    (tmp_path / "status.json").write_text(json.dumps({"model": "any/future-engine", "priming_profile": "full"}))
    with patch("core.config.load_config", return_value=config):
        policy = resolve_priming_policy(tmp_path)
    assert (policy.profile, policy.dynamic_budget, policy.max_tokens) == ("full", False, 1500)


def test_defaults_keep_storage_but_disable_automatic_mutation():
    config = ConsolidationConfig()
    assert config.daily_enabled and config.indexing_enabled
    assert not config.knowledge_mutation_enabled
    assert not config.weekly_enabled and not config.monthly_enabled
    assert not config.knowledge_self_correction_enabled
    assert not config.weekly_distillation_enabled and not config.skill_autolearn_enabled
    assert RAGConfig().enabled and RAGConfig().repair_enabled and RAGConfig().vector_worker_enabled
    assert RAGConfig().rerank_enabled and not RAGConfig().facts_extraction_enabled


def test_episode_checkpoint_processes_only_new_inputs_and_preserves_source(tmp_path: Path):
    engine = ConsolidationEngine(tmp_path, "fixture")
    day = date(2026, 9, 7)
    raw = tmp_path / "activity_log" / "2026-09-07.jsonl"
    raw.parent.mkdir()
    raw.write_text("original evidence")
    assert engine.unprocessed_activity_chunks(day, ["a", "b"]) == ["a", "b"]
    engine.write_consolidated_episode(day, "episode for a")
    engine.record_consolidated_chunks(day, ["a"])
    restarted = ConsolidationEngine(tmp_path, "fixture")
    assert restarted.unprocessed_activity_chunks(day, ["a", "b"]) == ["b"]
    assert restarted.unprocessed_activity_chunks(date(2026, 9, 8), ["a"]) == ["a"]
    assert raw.read_text() == "original evidence"


def test_incremental_phase_b_carryover_keeps_earlier_unfinished_input(tmp_path: Path):
    engine = ConsolidationEngine(tmp_path, "fixture")
    day = date(2026, 9, 7)
    for summary in ("first episode", "late episode", "late episode"):
        engine.record_phase_b_carryover(summary, target_date=day, reason="pending", incremental=True)
    assert engine.load_phase_b_carryover()[0]["episodes_summary"] == "first episode\n\nlate episode"


@pytest.mark.asyncio
async def test_daily_default_finishes_after_episode_without_tool_loop(tmp_path: Path):
    from core._anima_lifecycle import LifecycleMixin
    from core.config.models import AnimaWorksConfig

    anima = SimpleNamespace(name="fixture", anima_dir=tmp_path)
    engine = ConsolidationEngine(tmp_path, "fixture")
    engine.collect_activity_chunks = MagicMock(return_value=[])
    engine._collect_recent_episodes = MagicMock(side_effect=AssertionError("No whole-library mutation scan"))
    with patch("core.config.load_config", return_value=AnimaWorksConfig()):
        result = await LifecycleMixin._run_daily_consolidation(anima, engine)
    assert result.action == "skipped"
    engine._collect_recent_episodes.assert_not_called()


def test_curator_proposal_cannot_change_access_or_remove_vectors(tmp_path: Path):
    from core.skills.curator import SkillCurator
    from core.skills.models import SkillLifecycleState

    curator = SkillCurator(tmp_path)
    with patch.object(curator, "_purge_personal_skill_vectors") as purge:
        event = curator.propose_state_change("policy", "archived", reason="review suggestion")
    assert event.event_type == "state_change_proposed"
    assert curator.replay_state().state_for("policy") == SkillLifecycleState.active
    purge.assert_not_called()
    assert curator.generate_report([])["pending_proposals"][0]["skill_name"] == "policy"


@pytest.mark.asyncio
async def test_daily_repeat_has_zero_generation_calls(tmp_path: Path):
    from core._anima_lifecycle import LifecycleMixin
    from core.config.models import AnimaWorksConfig

    anima = SimpleNamespace(name="fixture", anima_dir=tmp_path)
    engine = ConsolidationEngine(tmp_path, "fixture")
    engine.collect_activity_chunks = MagicMock(return_value=["new activity"])
    with (
        patch("core.config.load_config", return_value=AnimaWorksConfig()),
        patch(
            "core.memory._llm_utils.one_shot_completion",
            new_callable=AsyncMock,
            return_value="## 12:00 — Work\nEvidence",
        ) as llm,
    ):
        first = await LifecycleMixin._run_daily_consolidation(anima, engine)
        second = await LifecycleMixin._run_daily_consolidation(anima, engine)
    assert first.action == "completed"
    assert second.action == "skipped"
    assert llm.await_count == 1


def test_explicit_graph_disable_skips_indexer_creation(tmp_path: Path):
    from core.memory.rag_search import RAGMemorySearch

    search = RAGMemorySearch(tmp_path, tmp_path / "common_knowledge", tmp_path / "common_skills")
    with (
        patch.object(search, "_load_rag_pipeline_settings", return_value={"enable_spreading_activation": False}),
        patch.object(search, "_get_indexer") as indexer,
    ):
        assert search._graph_episodes_search("query", 10, tmp_path / "knowledge") == []
    indexer.assert_not_called()
