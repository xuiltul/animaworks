from __future__ import annotations

import json
from datetime import date, timedelta, timezone
from pathlib import Path

import pytest

from core.execution.base import TokenUsage
from core.memory.token_usage import DEFAULT_PRICING, TokenUsageLogger

JST = timezone(timedelta(hours=9))


# ── TokenUsage dataclass ──────────────────────────────────────


class TestTokenUsage:
    def test_defaults(self):
        u = TokenUsage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cache_read_tokens == 0
        assert u.cache_write_tokens == 0

    def test_total_tokens(self):
        u = TokenUsage(input_tokens=100, output_tokens=50)
        assert u.total_tokens == 150

    def test_to_dict(self):
        u = TokenUsage(input_tokens=10, output_tokens=20, cache_read_tokens=5, cache_write_tokens=3)
        d = u.to_dict()
        assert d == {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_tokens": 5,
            "cache_write_tokens": 3,
        }

    def test_merge(self):
        a = TokenUsage(input_tokens=100, output_tokens=50, cache_read_tokens=10, cache_write_tokens=5)
        b = TokenUsage(input_tokens=200, output_tokens=100, cache_read_tokens=20, cache_write_tokens=10)
        a.merge(b)
        assert a.input_tokens == 300
        assert a.output_tokens == 150
        assert a.cache_read_tokens == 30
        assert a.cache_write_tokens == 15

    def test_merge_preserves_other(self):
        a = TokenUsage(input_tokens=100, output_tokens=50)
        b = TokenUsage(input_tokens=200, output_tokens=100)
        a.merge(b)
        assert b.input_tokens == 200
        assert b.output_tokens == 100


# ── TokenUsageLogger ──────────────────────────────────────────


@pytest.fixture
def logger_dir(tmp_path: Path) -> Path:
    anima_dir = tmp_path / "animas" / "test_anima"
    anima_dir.mkdir(parents=True)
    return anima_dir


@pytest.fixture
def tul(logger_dir: Path) -> TokenUsageLogger:
    return TokenUsageLogger(logger_dir)


class TestTokenUsageLoggerLog:
    def test_log_creates_file(self, tul: TokenUsageLogger, logger_dir: Path):
        tul.log(model="claude-sonnet-4-6", trigger="chat", mode="a", input_tokens=1000, output_tokens=500)
        usage_dir = logger_dir / "token_usage"
        assert usage_dir.is_dir()
        files = list(usage_dir.glob("*.jsonl"))
        assert len(files) == 1

        lines = files[0].read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["model"] == "claude-sonnet-4-6"
        assert entry["trigger"] == "chat"
        assert entry["mode"] == "a"
        assert entry["input_tokens"] == 1000
        assert entry["output_tokens"] == 500
        assert entry["total_tokens"] == 1500
        assert "estimated_cost_usd" in entry

    def test_log_appends(self, tul: TokenUsageLogger, logger_dir: Path):
        tul.log(model="claude-sonnet-4-6", trigger="chat", mode="a", input_tokens=100, output_tokens=50)
        tul.log(model="claude-sonnet-4-6", trigger="heartbeat", mode="a", input_tokens=200, output_tokens=100)
        files = list((logger_dir / "token_usage").glob("*.jsonl"))
        lines = files[0].read_text().strip().splitlines()
        assert len(lines) == 2

    def test_log_cache_tokens_optional(self, tul: TokenUsageLogger, logger_dir: Path):
        tul.log(model="claude-opus-4-6", trigger="chat", mode="s",
                input_tokens=1000, output_tokens=500,
                cache_read_tokens=300, cache_write_tokens=100)
        files = list((logger_dir / "token_usage").glob("*.jsonl"))
        entry = json.loads(files[0].read_text().strip())
        assert entry["cache_read_tokens"] == 300
        assert entry["cache_write_tokens"] == 100

    def test_log_no_cache_omits_fields(self, tul: TokenUsageLogger, logger_dir: Path):
        tul.log(model="claude-sonnet-4-6", trigger="chat", mode="a", input_tokens=100, output_tokens=50)
        files = list((logger_dir / "token_usage").glob("*.jsonl"))
        entry = json.loads(files[0].read_text().strip())
        assert "cache_read_tokens" not in entry
        assert "cache_write_tokens" not in entry


class TestEstimateCost:
    def test_claude_sonnet(self, tul: TokenUsageLogger):
        cost = tul.estimate_cost("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(3.0 + 15.0)

    def test_claude_opus_with_cache(self, tul: TokenUsageLogger):
        cost = tul.estimate_cost(
            "claude-opus-4-6",
            input_tokens=1_000_000,
            output_tokens=500_000,
            cache_read_tokens=200_000,
            cache_write_tokens=100_000,
        )
        expected = (
            1_000_000 * 15.0
            + 500_000 * 75.0
            + 200_000 * 1.50
            + 100_000 * 18.75
        ) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_zero(self, tul: TokenUsageLogger):
        cost = tul.estimate_cost("unknown-model-xyz", input_tokens=1_000_000)
        assert cost == 0.0

    def test_provider_prefix_stripped(self, tul: TokenUsageLogger):
        cost = tul.estimate_cost("openai/gpt-4o", input_tokens=1_000_000)
        assert cost > 0.0

    def test_bedrock_prefix_stripped(self, tul: TokenUsageLogger):
        cost = tul.estimate_cost("us.anthropic.claude-sonnet-4-6", input_tokens=1_000_000)
        expected = 1_000_000 * 3.0 / 1_000_000
        assert cost == pytest.approx(expected)

    def test_zero_tokens_returns_zero(self, tul: TokenUsageLogger):
        cost = tul.estimate_cost("claude-sonnet-4-6")
        assert cost == 0.0


class TestResolvePricingLongestMatch:
    def test_gpt4o_mini_not_gpt4o(self, tul: TokenUsageLogger):
        """gpt-4o-mini should match its own pricing, not gpt-4o."""
        gpt4o_cost = tul.estimate_cost("gpt-4o", input_tokens=1_000_000)
        gpt4o_mini_cost = tul.estimate_cost("gpt-4o-mini", input_tokens=1_000_000)
        assert gpt4o_mini_cost < gpt4o_cost

    def test_claude_sonnet_4_not_4_6(self, tul: TokenUsageLogger):
        """claude-sonnet-4 and claude-sonnet-4-6 should resolve separately."""
        cost_4 = tul.estimate_cost("claude-sonnet-4", input_tokens=1_000_000)
        cost_46 = tul.estimate_cost("claude-sonnet-4-6", input_tokens=1_000_000)
        assert cost_4 == pytest.approx(cost_46)

    def test_gpt41_mini_not_gpt41(self, tul: TokenUsageLogger):
        """gpt-4.1-mini should not match gpt-4.1 pricing."""
        gpt41_cost = tul.estimate_cost("gpt-4.1", input_tokens=1_000_000)
        gpt41_mini_cost = tul.estimate_cost("gpt-4.1-mini", input_tokens=1_000_000)
        assert gpt41_mini_cost < gpt41_cost


class TestCustomPricing:
    def test_custom_pricing_override(self, logger_dir: Path):
        pricing_file = logger_dir.parent.parent / "pricing.json"
        custom_pricing = {"test-model": {"input": 10.0, "output": 50.0}}
        pricing_file.write_text(json.dumps(custom_pricing))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("core.memory.token_usage.TokenUsageLogger._load_pricing_table",
                        lambda self: custom_pricing)
            tul = TokenUsageLogger(logger_dir)
            tul._pricing = custom_pricing
            cost = tul.estimate_cost("test-model", input_tokens=1_000_000, output_tokens=500_000)
            expected = (1_000_000 * 10.0 + 500_000 * 50.0) / 1_000_000
            assert cost == pytest.approx(expected)


class TestSummarize:
    def test_empty_returns_zeros(self, tul: TokenUsageLogger):
        summary = tul.summarize(days=7)
        assert summary["total_sessions"] == 0
        assert summary["total_input_tokens"] == 0
        assert summary["total_estimated_cost_usd"] == 0.0

    def test_summarize_aggregates(self, tul: TokenUsageLogger):
        tul.log(model="claude-sonnet-4-6", trigger="chat", mode="a",
                input_tokens=1000, output_tokens=500)
        tul.log(model="claude-sonnet-4-6", trigger="heartbeat", mode="a",
                input_tokens=2000, output_tokens=1000)
        tul.log(model="claude-opus-4-6", trigger="chat", mode="s",
                input_tokens=500, output_tokens=200)

        summary = tul.summarize(days=1)
        assert summary["total_sessions"] == 3
        assert summary["total_input_tokens"] == 3500
        assert summary["total_output_tokens"] == 1700
        assert summary["total_tokens"] == 5200
        assert summary["total_estimated_cost_usd"] > 0

        assert "claude-sonnet-4-6" in summary["by_model"]
        assert "claude-opus-4-6" in summary["by_model"]
        assert summary["by_model"]["claude-sonnet-4-6"]["sessions"] == 2
        assert summary["by_model"]["claude-opus-4-6"]["sessions"] == 1

        assert "chat" in summary["by_trigger"]
        assert "heartbeat" in summary["by_trigger"]
        assert summary["by_trigger"]["chat"]["sessions"] == 2
        assert summary["by_trigger"]["heartbeat"]["sessions"] == 1

    def test_summarize_by_date(self, tul: TokenUsageLogger):
        tul.log(model="claude-sonnet-4-6", trigger="chat", mode="a",
                input_tokens=1000, output_tokens=500)

        summary = tul.summarize(days=1)
        assert len(summary["by_date"]) == 1
        day_key = list(summary["by_date"].keys())[0]
        assert summary["by_date"][day_key]["sessions"] == 1

    def test_read_entries(self, tul: TokenUsageLogger):
        tul.log(model="claude-sonnet-4-6", trigger="chat", mode="a",
                input_tokens=1000, output_tokens=500)
        entries = tul.read_entries(days=1)
        assert len(entries) == 1
        assert entries[0]["model"] == "claude-sonnet-4-6"


# ── _merge_stream_usage helper ────────────────────────────────


class TestMergeStreamUsage:
    def test_merge(self):
        from core._agent_cycle import _merge_stream_usage
        acc = {"input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cache_write_tokens": 0}
        _merge_stream_usage(acc, {"input_tokens": 200, "output_tokens": 100, "cache_read_tokens": 10})
        assert acc["input_tokens"] == 300
        assert acc["output_tokens"] == 150
        assert acc["cache_read_tokens"] == 10

    def test_merge_none(self):
        from core._agent_cycle import _merge_stream_usage
        acc = {"input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cache_write_tokens": 0}
        _merge_stream_usage(acc, None)
        assert acc["input_tokens"] == 100

    def test_merge_empty(self):
        from core._agent_cycle import _merge_stream_usage
        acc = {"input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cache_write_tokens": 0}
        _merge_stream_usage(acc, {})
        assert acc["input_tokens"] == 100
