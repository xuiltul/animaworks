from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.locomo.multihop import LocomoMultiHopOrchestrator, multihop_aliases


def _adapter(**attrs):
    defaults = {
        "_entity_ignored_entities": ("Caroline", "Melanie"),
        "_fact_bm25_corpus": [],
        "_last_abstain": False,
        "_last_abstain_reason": "",
        "_last_fact_count": 0,
        "_last_multihop_meta": {},
    }
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def test_multihop_aliases_default_off_to_avoid_test_set_leakage() -> None:
    aliases = multihop_aliases("What does Melanie do to destress?")

    assert aliases == ()


def test_multihop_aliases_expand_known_attribute_terms_when_enabled() -> None:
    aliases = multihop_aliases("What does Melanie do to destress?", enable_alias_map=True)

    assert "running" in aliases
    assert "pottery" in aliases


def test_multihop_feature_enabled_requires_category_1_fact_index_and_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(_last_fact_count=0)
    orchestrator = LocomoMultiHopOrchestrator(adapter)
    monkeypatch.setenv("LOCOMO_FACT_INDEX", "1")

    assert orchestrator.feature_enabled(1) is False

    adapter._last_fact_count = 4
    assert orchestrator.feature_enabled(1) is True
    assert orchestrator.feature_enabled(4) is False

    monkeypatch.delenv("LOCOMO_FACT_INDEX", raising=False)
    assert orchestrator.feature_enabled(1) is False


def test_multihop_query_specs_are_bounded_and_deterministic() -> None:
    orchestrator = LocomoMultiHopOrchestrator(_adapter())
    question = "What subject did Caroline and Melanie both paint?"
    persons = ["Caroline", "Melanie"]
    aliases = ("paint", "painting", "sunrise")

    first = orchestrator.query_specs(question, persons=persons, aliases=aliases)
    second = orchestrator.query_specs(question, persons=persons, aliases=aliases)

    assert first == second
    assert len(first) <= 6
    assert any(spec["helper"] == "intersection" for spec in first)
    assert any("Caroline" in spec["query"] for spec in first)


def test_multihop_profile_candidates_group_person_facts() -> None:
    adapter = _adapter(
        _fact_bm25_corpus=[
            (
                "Melanie: Running and pottery classes help me destress.",
                {"speaker": "Melanie", "fact_id": "melanie-1"},
            ),
            (
                "Caroline: I prefer chess.",
                {"speaker": "Caroline", "fact_id": "caroline-1"},
            ),
        ],
    )
    orchestrator = LocomoMultiHopOrchestrator(adapter)

    items = orchestrator.profile_candidates(
        "What does Melanie do to destress?",
        persons=["Melanie"],
        aliases=("running", "pottery"),
    )

    assert len(items) == 1
    assert "Running and pottery" in items[0]["content"]
    assert items[0]["metadata"]["locomo_multihop_helper"] == "profile"
    assert items[0]["metadata"]["locomo_multihop_person"] == "Melanie"


def test_multihop_merge_prefers_higher_scored_helper_metadata() -> None:
    orchestrator = LocomoMultiHopOrchestrator(_adapter())

    merged = orchestrator.merge_items(
        [
            {
                "content": "Melanie: I run.",
                "score": 0.1,
                "metadata": {"fact_id": "fact-1", "search_method": "base"},
            }
        ],
        [
            {
                "content": "Melanie: I run.",
                "score": 0.9,
                "metadata": {"fact_id": "fact-1", "locomo_multihop_helper": "alias"},
            }
        ],
        limit=3,
    )

    assert len(merged) == 1
    assert merged[0]["score"] == 0.9
    assert merged[0]["metadata"]["locomo_multihop_helper"] == "alias"


def test_multihop_augment_uses_fact_fallback_when_base_context_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(_last_abstain=True, _last_abstain_reason="low_confidence")
    orchestrator = LocomoMultiHopOrchestrator(adapter)

    def fake_fact_search(query: str, spec: dict, *, top_k: int, include_vectors: bool = False):
        if spec["helper"] != "fact_fallback":
            return []
        assert top_k == 5
        assert include_vectors is True
        return [
            {
                "content": "Caroline: I am a transgender woman.",
                "score": 0.7,
                "metadata": {
                    "fact_id": "caroline-identity",
                    "locomo_multihop_helper": "fact_fallback",
                    "locomo_multihop_query": query,
                },
            }
        ]

    monkeypatch.setattr(orchestrator, "fact_search", fake_fact_search)
    monkeypatch.setattr(orchestrator, "profile_candidates", lambda _question, *, persons, aliases: [])

    items = orchestrator.augment("What is Caroline's identity?", [], top_k=5)

    assert len(items) == 1
    assert items[0]["metadata"]["locomo_multihop_helper"] == "fact_fallback"
    assert adapter._last_abstain is False
    assert adapter._last_multihop_meta["fallback_used"] is True
    assert adapter._last_multihop_meta["helper_hit_counts"] == {"fact_fallback": 1}


def test_multihop_augment_preserves_full_base_context_without_helper_search(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    orchestrator = LocomoMultiHopOrchestrator(adapter)

    def fail_fact_search(_query: str, _spec: dict, *, top_k: int, include_vectors: bool = False):
        raise AssertionError("helper search should not run when base context is already full")

    monkeypatch.setattr(orchestrator, "fact_search", fail_fact_search)
    monkeypatch.setattr(orchestrator, "profile_candidates", lambda _question, *, persons, aliases: [])
    base = [{"content": "Episode hit", "score": 0.9, "metadata": {"memory_type": "episodes"}}]

    items = orchestrator.augment("What is Caroline's identity?", base, top_k=1)

    assert items == base
    assert adapter._last_multihop_meta["enabled"] is True
    assert adapter._last_multihop_meta["helper_hit_counts"] == {}
