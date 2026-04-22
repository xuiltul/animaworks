from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.locomo.adapter import (
    SEARCH_MODES,
    _build_episode_markdown,
    _episode_stem_for_sample,
    _session_indices,
    load_dataset,
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
