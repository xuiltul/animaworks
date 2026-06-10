from __future__ import annotations

import argparse
import json

import pytest

from benchmarks.locomo.llm_config import default_answer_model
from benchmarks.locomo.protocol import apply_standard_protocol_args, attach_standard_protocol_summary
from benchmarks.locomo.runner import (
    _REFERENCE_SCORES,
    EMBEDDING_MODEL_DEFAULT,
    _build_arg_parser,
    _print_comparison,
    _print_summary,
    _write_checkpoint_json,
    run_benchmark,
)

# ── CLI parser ──────────


class TestArgParser:
    def test_defaults(self):
        p = _build_arg_parser()
        args = p.parse_args([])
        assert args.mode == "all"
        assert args.conversations == 10
        assert args.top_k == 10
        assert args.judge is False
        assert args.judge_model == "gpt-4o"
        assert args.answer_model == default_answer_model()
        assert args.answer_timeout is None
        assert args.answer_max_retries == 2
        assert args.enable_locomo_alias is False
        assert args.enable_locomo_category_branches is False
        assert args.standard_protocol is False
        assert args.checkpoint_every == 0
        assert args.max_questions == 0
        assert args.verbose is False

    def test_mode_choice(self):
        p = _build_arg_parser()
        args = p.parse_args(["--mode", "vector"])
        assert args.mode == "vector"

    def test_invalid_mode(self):
        p = _build_arg_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--mode", "invalid"])

    def test_judge_flag(self):
        p = _build_arg_parser()
        args = p.parse_args(["--judge"])
        assert args.judge is True

    def test_answer_knobs(self):
        p = _build_arg_parser()
        args = p.parse_args(
            [
                "--answer-timeout",
                "60",
                "--answer-max-retries",
                "0",
                "--checkpoint-every",
                "25",
                "--max-questions",
                "20",
            ],
        )
        assert args.answer_timeout == 60.0
        assert args.answer_max_retries == 0
        assert args.checkpoint_every == 25
        assert args.max_questions == 20

    def test_integrity_flags(self):
        p = _build_arg_parser()
        args = p.parse_args(["--enable-locomo-alias", "--enable-locomo-category-branches", "--standard-protocol"])
        assert args.enable_locomo_alias is True
        assert args.enable_locomo_category_branches is True
        assert args.standard_protocol is True

    def test_standard_protocol_overrides_leakage_flags(self):
        p = _build_arg_parser()
        args = p.parse_args(
            ["--mode", "all", "--enable-locomo-alias", "--enable-locomo-category-branches", "--standard-protocol"]
        )

        apply_standard_protocol_args(args)

        assert args.mode == "scope_all"
        assert args.conversations == 10
        assert args.judge is True
        assert args.exclude_cat5 is True
        assert args.enable_locomo_alias is False
        assert args.enable_locomo_category_branches is False


class TestStandardProtocolSummary:
    def test_requires_scope_all_and_cat5_exclusion(self):
        summary = {
            "overall_f1": 0.5,
            "error_rate": 0.0,
            "cat5_excluded": {"overall_judge": 0.75},
        }
        config = {
            "answer_model": "answer-model",
            "judge_model": "judge-model",
            "judge_enabled": True,
            "conversations": 10,
            "search_mode": "all",
            "exclude_cat5": False,
            "max_questions": 0,
            "leakage_alias_map_enabled": False,
            "category_branches_enabled": False,
        }

        attach_standard_protocol_summary(summary, config)

        protocol = summary["standard_protocol"]
        assert protocol["valid"] is False
        assert protocol["scope_all"] is False
        assert protocol["cat5_excluded"] is False

        config["search_mode"] = "scope_all"
        config["exclude_cat5"] = True
        attach_standard_protocol_summary(summary, config)

        protocol = summary["standard_protocol"]
        assert protocol["valid"] is True
        assert protocol["scope_all"] is True
        assert protocol["cat5_excluded"] is True

        config["max_questions"] = 2
        attach_standard_protocol_summary(summary, config)

        protocol = summary["standard_protocol"]
        assert protocol["valid"] is False
        assert protocol["max_questions"] == 2


# ── Missing data ──────────


class TestRunBenchmarkMissingData:
    def test_missing_dataset_returns_error(self, tmp_path):
        args = argparse.Namespace(
            data=str(tmp_path / "nonexistent.json"),
            output=str(tmp_path / "results"),
            mode="vector",
            conversations=1,
            top_k=5,
            judge=False,
            judge_model="gpt-4o",
            answer_model="gpt-4o-mini",
            verbose=False,
        )
        results, err = run_benchmark(args)
        assert results == {}
        assert err == 1


class TestCheckpointJson:
    def test_write_checkpoint_json_includes_partial_summary(self, tmp_path):
        out = _write_checkpoint_json(
            tmp_path,
            "2026-06-04T00-00-00",
            "scope_all",
            timestamp_iso="2026-06-04T00:00:00",
            config={"answer_timeout": 60, "answer_max_retries": 0},
            results=[
                {
                    "sample_id": "conv-1",
                    "question_index": 0,
                    "category": 1,
                    "question": "Q",
                    "reference": "A",
                    "prediction": "A",
                    "f1": 1.0,
                    "judge_score": None,
                }
            ],
            errors=0,
        )

        assert out.name.endswith("_scope_all.partial.json")
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["checkpoint"] is True
        assert payload["summary"]["overall_f1"] == 1.0
        assert payload["config"]["answer_max_retries"] == 0


# ── Summary printing ──────────


class TestPrintSummary:
    def test_no_crash(self, capsys):
        summary = {
            "overall_f1": 0.5,
            "overall_judge": None,
            "by_category": {
                "multi_hop": {"f1": 0.6, "judge": None, "count": 10},
                "temporal": {"f1": 0.4, "judge": None, "count": 5},
            },
        }
        _print_summary("vector", summary, total_count=15)
        out = capsys.readouterr().out
        assert "vector" in out
        assert "multi_hop" in out

    def test_with_judge(self, capsys):
        summary = {
            "overall_f1": 0.5,
            "overall_judge": 0.8,
            "by_category": {
                "multi_hop": {"f1": 0.6, "judge": 0.9, "count": 10},
            },
        }
        _print_summary("vector_graph", summary, total_count=10)
        out = capsys.readouterr().out
        assert "90.0%" in out


class TestPrintComparison:
    def test_no_crash(self, capsys):
        results = {
            "vector": {
                "summary": {
                    "overall_f1": 0.5,
                    "by_category": {
                        "multi_hop": {"f1": 0.5, "count": 10},
                    },
                },
            },
        }
        _print_comparison(results)
        out = capsys.readouterr().out
        assert "vector" in out

    def test_empty_no_crash(self, capsys):
        _print_comparison({})
        out = capsys.readouterr().out
        assert out == ""


# ── Reference scores ──────────


class TestReferenceScores:
    def test_mem0_present(self):
        assert "Mem0" in _REFERENCE_SCORES
        assert "overall" in _REFERENCE_SCORES["Mem0"]

    def test_zep_present(self):
        assert "Zep" in _REFERENCE_SCORES

    def test_embedding_model_constant(self):
        assert EMBEDDING_MODEL_DEFAULT == "intfloat/multilingual-e5-small"
