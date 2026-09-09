"""Tests for core.config.model_discovery.

Covers the pure parse functions, cache behaviour, static fallback, and
fault-isolation between probes.  No real subprocess / network is invoked.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

from core.config.model_discovery import (
    DiscoveredModel,
    _parse_claude_help,
    _parse_codex_models,
    _parse_grok_models,
    discover_models,
    discovered_model_ids,
    invalidate_cache,
)

CODEX_JSON = """{"models": [
  {"slug":"gpt-6-astra","display_name":"GPT-6-Astra","visibility":"list","priority":1},
  {"slug":"gpt-5.6-sol","display_name":"GPT-5.6-Sol","visibility":"list","priority":2},
  {"slug":"gpt-reserve","display_name":"GPT-Reserve","visibility":"hide","priority":0},
  {"slug":"codex-auto-review","display_name":"Auto Review","visibility":"hide","priority":3},
  {"slug":"gpt-5.5","display_name":"GPT-5.5","visibility":"list","priority":5}
]}"""

GROK_TEXT = """You are logged in with grok.com.

Default model: grok-4.6

Available models:
  * grok-4.6 (default)
  - grok-4.5
"""

CLAUDE_HELP = """Usage: claude [options]

Options:
  --model <model>                      Model for the current session. Provide
                                       an alias for the latest model (e.g.
                                       'fable', 'opus', or 'sonnet') or a
                                       model's full name (e.g.
                                       'claude-fable-5').
  --print                             Print response and exit.
"""


class TestParseCodex:
    def test_hide_excluded_and_priority_sorted(self):
        result = _parse_codex_models(CODEX_JSON)
        slugs = [m["slug"] for m in result]
        assert slugs == ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.5"]
        # hide entries must not appear
        assert "gpt-reserve" not in slugs
        assert "codex-auto-review" not in slugs
        # label takes display_name
        assert result[0]["label"] == "GPT-6-Astra"

    def test_invalid_json_returns_empty(self):
        assert _parse_codex_models("not json {") == []


class TestParseGrok:
    def test_extracts_models_without_default_marker(self):
        result = _parse_grok_models(GROK_TEXT)
        assert result == ["grok-4.6", "grok-4.5"]
        assert all("default" not in name for name in result)

    def test_empty_text_returns_empty(self):
        assert _parse_grok_models("") == []


class TestParseClaude:
    def test_extracts_aliases_ignores_full_name(self):
        result = _parse_claude_help(CLAUDE_HELP)
        assert result == ["fable", "opus", "sonnet"]
        assert "claude-fable-5" not in result

    def test_no_model_option_returns_empty(self):
        assert _parse_claude_help("no options") == []


class TestDiscoverModels:
    def _all_empty_probes(self):
        return (
            patch("core.config.model_discovery._probe_codex", return_value=[]),
            patch("core.config.model_discovery._probe_grok", return_value=[]),
            patch("core.config.model_discovery._probe_claude", return_value=[]),
            patch("core.config.model_discovery._probe_openai_compatible", return_value=[]),
            patch("core.config.model_discovery._probe_ollama", return_value=[]),
        )

    def test_static_fallback_when_all_probes_empty(self):
        from contextlib import ExitStack

        invalidate_cache()
        fallback = [DiscoveredModel("a:openai/gpt-4.1", "a", "openai/gpt-4.1", "gpt-4.1", "openai", source="static")]
        with ExitStack() as stack:
            for ctx in self._all_empty_probes():
                stack.enter_context(ctx)
            stack.enter_context(patch("core.config.model_discovery._static_fallback", return_value=fallback))
            models = discover_models(config=object())
        assert [m.id for m in models] == ["a:openai/gpt-4.1"]

    def test_cache_reuses_result(self):
        invalidate_cache()
        stub = [DiscoveredModel("s:fable", "s", "fable", "fable", "Claude", source="claude-cli")]
        with (
            patch(
                "core.config.model_discovery._probe_codex",
                return_value=stub,
            ) as codex,
            patch(
                "core.config.model_discovery._probe_grok",
                return_value=[],
            ),
            patch(
                "core.config.model_discovery._probe_claude",
                return_value=[],
            ),
            patch(
                "core.config.model_discovery._probe_openai_compatible",
                return_value=[],
            ),
            patch(
                "core.config.model_discovery._probe_ollama",
                return_value=[],
            ),
        ):
            discover_models(config=object())
            discover_models(config=object())
            assert codex.call_count == 1

    def test_expired_cache_is_returned_while_background_refresh_starts(self):
        """A model picker must not wait for probes every time the TTL expires."""
        invalidate_cache()
        stub = [DiscoveredModel("s:fable", "s", "fable", "fable", "Claude", source="claude-cli")]
        with (
            patch("core.config.model_discovery._probe_codex", return_value=stub) as codex,
            patch("core.config.model_discovery._probe_grok", return_value=[]),
            patch("core.config.model_discovery._probe_claude", return_value=[]),
            patch("core.config.model_discovery._probe_openai_compatible", return_value=[]),
            patch("core.config.model_discovery._probe_ollama", return_value=[]),
            patch("core.config.model_discovery.time.monotonic", return_value=1.0),
        ):
            discover_models(config=object())

        with (
            patch(
                "core.config.model_discovery.time.monotonic",
                return_value=1.0 + 301.0,
            ),
            patch("core.config.model_discovery.threading.Thread") as thread_cls,
        ):
            models = discover_models(config=object())

        assert [model.id for model in models] == ["s:fable"]
        assert codex.call_count == 1
        thread_cls.assert_called_once()
        thread_cls.return_value.start.assert_called_once()
        invalidate_cache()

    def test_refresh_reruns_probes(self):
        invalidate_cache()
        stub = [
            DiscoveredModel("c:codex/gpt-5.6-sol", "c", "codex/gpt-5.6-sol", "GPT-5.6-Sol", "Codex", source="codex-cli")
        ]
        with (
            patch(
                "core.config.model_discovery._probe_codex",
                return_value=stub,
            ) as codex,
            patch(
                "core.config.model_discovery._probe_grok",
                return_value=[],
            ),
            patch(
                "core.config.model_discovery._probe_claude",
                return_value=[],
            ),
            patch(
                "core.config.model_discovery._probe_openai_compatible",
                return_value=[],
            ),
            patch(
                "core.config.model_discovery._probe_ollama",
                return_value=[],
            ),
        ):
            discover_models(config=object())
            discover_models(refresh=True, config=object())
            assert codex.call_count == 2

    def test_one_probe_exception_ignored(self):
        invalidate_cache()
        stub = [DiscoveredModel("x:grok/grok-4.6", "x", "grok/grok-4.6", "grok-4.6", "Grok", source="grok-cli")]
        with (
            patch(
                "core.config.model_discovery._probe_codex",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "core.config.model_discovery._probe_grok",
                return_value=stub,
            ),
            patch(
                "core.config.model_discovery._probe_claude",
                return_value=[],
            ),
            patch(
                "core.config.model_discovery._probe_openai_compatible",
                return_value=[],
            ),
            patch(
                "core.config.model_discovery._probe_ollama",
                return_value=[],
            ),
        ):
            models = discover_models(config=object())
        assert [m.id for m in models] == ["x:grok/grok-4.6"]

    def test_discovered_model_ids_include_three_forms(self):
        models = [
            DiscoveredModel("c:codex/gpt-5.6-sol", "c", "codex/gpt-5.6-sol", "GPT-5.6-Sol", "Codex", source="codex-cli")
        ]
        with patch("core.config.model_discovery.discover_models", return_value=models) as dm:
            ids = discovered_model_ids()
            assert "c:codex/gpt-5.6-sol" in ids
            assert "codex/gpt-5.6-sol" in ids
            assert "gpt-5.6-sol" in ids
            dm.assert_called_once()
