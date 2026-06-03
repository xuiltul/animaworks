from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.locomo.retrieval_diagnostics import (
    _temporary_temporal_boost,
    answer_token_recall,
    parse_args,
    summarize_results,
    write_diagnostics_json,
)


class TestAnswerTokenRecall:
    def test_answer_token_recall_partial(self) -> None:
        recall, all_present = answer_token_recall(
            "Becoming Nicole",
            [{"content": "Caroline recommended Becoming to Melanie."}],
        )

        assert recall == 0.5
        assert all_present == 0.0

    def test_answer_token_recall_all_present(self) -> None:
        recall, all_present = answer_token_recall(
            "Becoming Nicole",
            [{"content": "The book was Becoming Nicole."}],
        )

        assert recall == 1.0
        assert all_present == 1.0

    def test_empty_answer_tokens_return_none(self) -> None:
        recall, all_present = answer_token_recall("", [{"content": "anything"}])

        assert recall is None
        assert all_present is None


class TestSummarizeResults:
    def test_category_5_excluded_from_aggregates(self) -> None:
        summary = summarize_results(
            [
                {
                    "category": 2,
                    "answer_token_recall_at_10": 0.5,
                    "answer_token_recall_at_50": 1.0,
                    "all_answer_tokens_present_at_10": 0.0,
                    "all_answer_tokens_present_at_50": 1.0,
                },
                {
                    "category": 5,
                    "answer_token_recall_at_10": None,
                    "answer_token_recall_at_50": None,
                    "all_answer_tokens_present_at_10": None,
                    "all_answer_tokens_present_at_50": None,
                },
            ],
        )

        assert summary["count"] == 1
        assert summary["excluded_adversarial"] == 1
        assert summary["answer_token_recall_at_10"] == 0.5
        assert summary["answer_token_recall_at_50"] == 1.0
        assert summary["by_category"]["temporal"]["count"] == 1
        assert "adversarial" not in summary["by_category"]


class TestWriteDiagnosticsJson:
    def test_write_json_uses_required_config_shape(self, tmp_path: Path) -> None:
        out = write_diagnostics_json(
            tmp_path,
            mode="scope_all",
            conversations=1,
            top_k=10,
            ceiling_top_k=50,
            temporal_boost=False,
            summary={"answer_token_recall_at_10": 0.5},
            results=[],
            errors=0,
        )

        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["config"] == {
            "mode": "scope_all",
            "conversations": 1,
            "top_k": 10,
            "ceiling_top_k": 50,
            "temporal_boost": False,
        }
        assert payload["summary"]["answer_token_recall_at_10"] == 0.5


class TestTemporalAblationCli:
    def test_parse_temporal_ablation_flag(self) -> None:
        args = parse_args(["--temporal-ablation"])

        assert args.temporal_ablation is True

    def test_temporary_temporal_boost_sets_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCOMO_TEMPORAL_BOOST", raising=False)

        with _temporary_temporal_boost(True):
            assert os.environ["LOCOMO_TEMPORAL_BOOST"] == "1"

        assert "LOCOMO_TEMPORAL_BOOST" not in os.environ
