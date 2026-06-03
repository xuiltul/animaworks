from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.locomo.adapter import (
    SEARCH_MODES,
    _build_episode_markdown,
    _conversation_speaker_names,
    _episode_stem_for_sample,
    _session_indices,
    load_dataset,
    locomo_entity_boost_enabled,
    locomo_temporal_boost_enabled,
)

# ── load_dataset ──────────


class TestLoadDataset:
    def test_valid_json(self, tmp_path: Path):
        data = [
            {
                "sample_id": "conv-1",
                "conversation": {},
                "qa": [
                    {"question": "Q1", "answer": "A1", "category": 1},
                    {
                        "question": "Q5",
                        "category": 5,
                        "adversarial_answer": "adv_ans",
                    },
                ],
            }
        ]
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_dataset(p)
        assert len(result) == 1
        qa = result[0]["qa"]
        assert qa[0]["answer"] == "A1"
        assert qa[1]["answer"] == "adv_ans"

    def test_cat5_no_adversarial(self, tmp_path: Path):
        data = [
            {
                "sample_id": "conv-2",
                "conversation": {},
                "qa": [{"question": "Q", "category": 5, "answer": "already_set"}],
            }
        ]
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = load_dataset(p)
        assert result[0]["qa"][0]["answer"] == "already_set"

    def test_invalid_toplevel(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text('{"not": "a list"}', encoding="utf-8")
        with pytest.raises(ValueError, match="list"):
            load_dataset(p)


# ── Markdown helpers ──────────


class TestSessionIndices:
    def test_basic(self):
        conv = {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1": [],
            "session_1_date_time": "2023-01-01",
            "session_2": [],
            "session_2_date_time": "2023-01-02",
        }
        assert _session_indices(conv) == [1, 2]

    def test_no_sessions(self):
        assert _session_indices({"speaker_a": "A"}) == []


class TestBuildEpisodeMarkdown:
    def test_single_session(self):
        conv = {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1": [
                {"speaker": "Alice", "text": "Hello"},
                {"speaker": "Bob", "text": "Hi there"},
            ],
            "session_1_date_time": "2023-05-10",
        }
        md = _build_episode_markdown("conv-1", conv)
        assert "## Session 1" in md
        assert "2023-05-10" in md
        assert "**Alice**: Hello" in md
        assert "**Bob**: Hi there" in md

    def test_multi_session(self):
        conv = {
            "speaker_a": "X",
            "speaker_b": "Y",
            "session_1": [{"speaker": "X", "text": "S1"}],
            "session_1_date_time": "2023-01-01",
            "session_2": [{"speaker": "Y", "text": "S2"}],
            "session_2_date_time": "2023-06-01",
        }
        md = _build_episode_markdown("conv-99", conv)
        assert "## Session 1" in md
        assert "## Session 2" in md

    def test_empty_conversation(self):
        md = _build_episode_markdown("conv-0", {"speaker_a": "A", "speaker_b": "B"})
        assert "conv-0" in md


class TestEpisodeStemForSample:
    def test_plain_number(self):
        assert _episode_stem_for_sample("42") == "conv-42"

    def test_already_prefixed(self):
        assert _episode_stem_for_sample("conv-26") == "conv-26"

    def test_empty(self):
        assert _episode_stem_for_sample("") == "conv-unknown"


class TestConversationSpeakerNames:
    def test_speaker_names_are_collected_for_entity_ignore(self):
        assert _conversation_speaker_names({"speaker_a": "Caroline", "speaker_b": "Melanie"}) == (
            "Caroline",
            "Melanie",
        )


# ── Adapter validation ──────────


class TestSearchModes:
    def test_modes_tuple(self):
        assert "vector" in SEARCH_MODES
        assert "vector_graph" in SEARCH_MODES
        assert "scope_all" in SEARCH_MODES

    def test_invalid_mode_raises(self):
        from benchmarks.locomo.adapter import AnimaWorksLoCoMoAdapter

        with pytest.raises(ValueError, match="search_mode"):
            AnimaWorksLoCoMoAdapter(search_mode="invalid")


class TestEventMetadataPropagation:
    def _adapter_without_init(self):
        from benchmarks.locomo.adapter import AnimaWorksLoCoMoAdapter

        return object.__new__(AnimaWorksLoCoMoAdapter)

    def test_pipeline_item_preserves_event_metadata(self):
        adapter = self._adapter_without_init()
        item = {
            "content": "## Session 10\n\nBody",
            "score": 0.7,
            "metadata": {
                "source_file": "conv-26.md",
                "chunk_index": 10,
                "search_method": "vector",
                "valid_at": 1689854160.0,
                "event_time_iso": "2023-07-20T20:56:00+09:00",
                "event_time_text": "8:56 pm on 20 July, 2023",
                "session_index": 10,
                "event_time_parse_error": False,
            },
        }

        pipeline_item = adapter._pipeline_item_from_adapter_hit(item)

        assert pipeline_item["valid_at"] == 1689854160.0
        assert pipeline_item["event_time_iso"] == "2023-07-20T20:56:00+09:00"
        assert pipeline_item["event_time_text"] == "8:56 pm on 20 July, 2023"
        assert pipeline_item["session_index"] == 10
        assert pipeline_item["event_time_parse_error"] is False

    def test_adapter_hit_restores_event_metadata(self):
        adapter = self._adapter_without_init()
        item = {
            "content": "Body",
            "score": 0.9,
            "source_file": "conv-26.md",
            "chunk_index": 2,
            "search_method": "cross_encoder",
            "event_time_iso": "2023-08-17T13:50:00+09:00",
            "session_index": 2,
            "base_score": 0.8,
            "temporal_boost": 0.1,
            "entity_boost": 0.08,
            "entity_overlap": ["book", "suggestion"],
            "query_entities": ["book", "suggestion"],
            "candidate_entities": ["becoming nicole", "book", "suggestion"],
        }

        hit = adapter._adapter_hit_from_pipeline_item(item)

        assert hit["metadata"]["event_time_iso"] == "2023-08-17T13:50:00+09:00"
        assert hit["metadata"]["session_index"] == 2
        assert hit["metadata"]["base_score"] == 0.8
        assert hit["metadata"]["temporal_boost"] == 0.1
        assert hit["metadata"]["entity_boost"] == 0.08
        assert hit["metadata"]["entity_overlap"] == ["book", "suggestion"]

    def test_retrieval_diagnostics_remember_top_event_time(self):
        adapter = self._adapter_without_init()
        adapter._remember_retrieval_diagnostics(
            [
                {"content": "low", "score": 0.1, "metadata": {"event_time_iso": "2023-01-01T00:00:00+09:00"}},
                {"content": "high", "score": 0.9, "metadata": {"event_time_iso": "2023-02-01T00:00:00+09:00"}},
            ],
        )

        assert adapter._last_top_score == 0.9
        assert adapter._last_top_event_time_iso == "2023-02-01T00:00:00+09:00"


class TestTemporalBoostEnv:
    def test_temporal_boost_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LOCOMO_TEMPORAL_BOOST", raising=False)
        assert locomo_temporal_boost_enabled() is False

    def test_temporal_boost_enabled_by_explicit_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LOCOMO_TEMPORAL_BOOST", "1")
        assert locomo_temporal_boost_enabled() is True


class TestEntityBoostEnv:
    def test_entity_boost_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LOCOMO_ENTITY_BOOST", raising=False)
        assert locomo_entity_boost_enabled() is False

    def test_entity_boost_enabled_by_explicit_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LOCOMO_ENTITY_BOOST", "1")
        assert locomo_entity_boost_enabled() is True
