from __future__ import annotations

import argparse

import pytest

from benchmarks.locomo.runner import (
    _REFERENCE_SCORES,
    EMBEDDING_MODEL_DEFAULT,
    _build_arg_parser,
    _print_comparison,
    _print_summary,
    run_benchmark,
)

# ── CLI parser ──────────


class TestArgParser:
    def test_defaults(self):
        p = _build_arg_parser()
        args = p.parse_args([])
        assert args.mode == "all"
        assert args.conversations == 10
        assert args.top_k == 5
        assert args.judge is False
        assert args.judge_model == "gpt-4o"
        assert args.answer_model == "gpt-4o-mini"
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
