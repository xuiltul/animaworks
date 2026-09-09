"""Tests for the server-side model catalog warmup."""

from __future__ import annotations

from unittest.mock import patch

from core.config.model_discovery import DiscoveredModel
from server.app import _warm_model_catalog


async def test_warm_model_catalog_primes_discovery_cache() -> None:
    models = [
        DiscoveredModel(
            "c:codex/gpt-5.6-sol",
            "c",
            "codex/gpt-5.6-sol",
            "GPT-5.6-Sol",
            "Codex",
        )
    ]
    with patch("core.config.model_discovery.discover_models", return_value=models) as discover:
        await _warm_model_catalog()

    discover.assert_called_once_with()
