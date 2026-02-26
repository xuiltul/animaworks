"""E2E tests for status.json hot-reload — full DigitalAnima lifecycle.

These tests create a minimal anima directory, initialize a DigitalAnima,
modify status.json on disk, call reload_config(), and verify the new
config is reflected.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────


def _write_status_json(anima_dir: Path, **overrides) -> None:
    """Write (or overwrite) status.json in the anima directory."""
    status_path = anima_dir / "status.json"
    data = {}
    if status_path.exists():
        data = json.loads(status_path.read_text(encoding="utf-8"))
    data.update(overrides)
    status_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── E2E Tests ────────────────────────────────────────────────


@pytest.mark.e2e
class TestReloadConfigE2E:
    """End-to-end tests for DigitalAnima.reload_config()."""

    def test_reload_picks_up_model_change(self, data_dir, make_anima):
        """Modify status.json model field and verify reload applies it."""
        anima_dir = make_anima(
            "reload-test",
            model="claude-sonnet-4-6",
            credential="anthropic",
        )
        shared_dir = data_dir / "shared"

        from core.anima import DigitalAnima

        anima = DigitalAnima(anima_dir=anima_dir, shared_dir=shared_dir)
        assert anima.model_config.model == "claude-sonnet-4-6"

        _write_status_json(anima_dir, model="claude-opus-4-6")

        from core.config import invalidate_cache
        invalidate_cache()

        result = anima.reload_config()

        assert result["status"] == "ok"
        assert result["model"] == "claude-opus-4-6"
        assert "model" in result["changes"]
        assert anima.model_config.model == "claude-opus-4-6"
        assert anima.agent.model_config.model == "claude-opus-4-6"

    def test_reload_picks_up_max_tokens_change(self, data_dir, make_anima):
        """Modify max_tokens and verify reload detects the change."""
        anima_dir = make_anima(
            "reload-tokens",
            model="claude-sonnet-4-6",
            credential="anthropic",
        )
        shared_dir = data_dir / "shared"

        from core.anima import DigitalAnima

        anima = DigitalAnima(anima_dir=anima_dir, shared_dir=shared_dir)
        original_max_tokens = anima.model_config.max_tokens

        _write_status_json(anima_dir, max_tokens=32768)

        from core.config import invalidate_cache
        invalidate_cache()

        result = anima.reload_config()

        assert result["status"] == "ok"
        assert "max_tokens" in result["changes"]
        assert anima.model_config.max_tokens == 32768

    def test_reload_no_change_returns_empty(self, data_dir, make_anima):
        """When status.json is unchanged, changes list should be empty."""
        anima_dir = make_anima(
            "reload-noop",
            model="claude-sonnet-4-6",
            credential="anthropic",
        )
        shared_dir = data_dir / "shared"

        from core.anima import DigitalAnima

        anima = DigitalAnima(anima_dir=anima_dir, shared_dir=shared_dir)

        from core.config import invalidate_cache
        invalidate_cache()

        result = anima.reload_config()

        assert result["status"] == "ok"
        assert result["changes"] == []

    def test_reload_thinking_flag(self, data_dir, make_anima):
        """Enable thinking via status.json reload."""
        anima_dir = make_anima(
            "reload-thinking",
            model="claude-sonnet-4-6",
            credential="anthropic",
        )
        shared_dir = data_dir / "shared"

        from core.anima import DigitalAnima

        anima = DigitalAnima(anima_dir=anima_dir, shared_dir=shared_dir)
        assert anima.model_config.thinking is None

        _write_status_json(anima_dir, thinking=True)

        from core.config import invalidate_cache
        invalidate_cache()

        result = anima.reload_config()

        assert "thinking" in result["changes"]
        assert anima.model_config.thinking is True
