# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for injection.md size governance.

Tests that injection_size_warning is injected when injection.md exceeds
the configured threshold, and omitted when under threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers.filesystem import create_anima_dir, create_test_data_dir

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Create isolated data dir and redirect ANIMAWORKS_DATA_DIR."""
    from core.config import invalidate_cache
    from core.paths import _prompt_cache
    from core.tooling.prompt_db import reset_prompt_store

    d = create_test_data_dir(tmp_path)
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(d))
    invalidate_cache()
    _prompt_cache.clear()
    reset_prompt_store()

    yield d

    invalidate_cache()
    _prompt_cache.clear()
    reset_prompt_store()


def _make_anima_with_injection(data_dir: Path, name: str, injection: str) -> Path:
    """Create anima with given injection content."""
    return create_anima_dir(data_dir, name, injection=injection)


# ── Under threshold: no warning ────────────────────────────────────────────


class TestInjectionUnderThreshold:
    """injection.md under threshold does not inject warning."""

    def test_no_warning_when_under_threshold(self, data_dir):
        """When injection size is under threshold, no injection_size_warning."""
        short_injection = "Short injection content.\n" * 10
        anima_dir = _make_anima_with_injection(data_dir, "test-anima", short_injection)

        from core.memory import MemoryManager
        from core.prompt.builder import build_system_prompt

        memory = MemoryManager(anima_dir)
        result = build_system_prompt(memory, trigger="chat")

        assert "injection_size_warning" not in result.system_prompt


# ── Over threshold: warning injected ────────────────────────────────────────


class TestInjectionOverThreshold:
    """injection.md over threshold injects injection_size_warning section."""

    def test_warning_injected_when_over_threshold(self, data_dir):
        """When injection exceeds threshold, injection_size_warning section is added."""
        long_injection = "x" * 6000
        anima_dir = _make_anima_with_injection(data_dir, "test-anima", long_injection)

        from core.memory import MemoryManager
        from core.prompt.builder import build_system_prompt

        memory = MemoryManager(anima_dir)
        result = build_system_prompt(memory, trigger="chat")

        assert "injection_size_warning" in result.system_prompt
        assert "⚠️" in result.system_prompt
        assert "injection.md" in result.system_prompt

    def test_warning_contains_size_and_threshold(self, data_dir):
        """Warning text includes actual size and threshold values."""
        long_injection = "x" * 6000
        anima_dir = _make_anima_with_injection(data_dir, "test-anima", long_injection)

        from core.memory import MemoryManager
        from core.prompt.builder import build_system_prompt

        memory = MemoryManager(anima_dir)
        result = build_system_prompt(memory, trigger="chat")

        assert "6000" in result.system_prompt
        assert "5000" in result.system_prompt


# ── Custom threshold from config ────────────────────────────────────────────


class TestInjectionCustomThreshold:
    """Custom injection_size_warning_chars from config is respected."""

    def test_custom_threshold_reflected(self, data_dir):
        """When config has custom threshold, it is used for the warning."""
        config_path = data_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["prompt"] = {"injection_size_warning_chars": 3000}
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

        from core.config import invalidate_cache

        invalidate_cache()

        injection_4000 = "x" * 4000
        anima_dir = _make_anima_with_injection(data_dir, "test-anima", injection_4000)

        from core.memory import MemoryManager
        from core.prompt.builder import build_system_prompt

        memory = MemoryManager(anima_dir)
        result = build_system_prompt(memory, trigger="chat")

        assert "4000" in result.system_prompt
        assert "3000" in result.system_prompt
        assert "injection_size_warning" in result.system_prompt
