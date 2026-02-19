# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for hiring_context injection into system prompts.

Verifies that a solo top-level anima sees the hiring context before
behavior_rules, and that the context is omitted when peers exist or
the anima has a supervisor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.memory import MemoryManager
from core.prompt.builder import build_system_prompt


class TestHiringContextE2E:
    """End-to-end: hiring_context placement in system prompt."""

    def _build_prompt_for(self, data_dir: Path, name: str) -> str:
        """Helper: build system prompt for the named anima."""
        anima_dir = data_dir / "animas" / name
        memory = MemoryManager(anima_dir)
        return build_system_prompt(memory)

    def test_solo_anima_sees_hiring_context(self, data_dir: Path, make_anima):
        """Solo top-level anima should see hiring context."""
        make_anima("solo")

        prompt = self._build_prompt_for(data_dir, "solo")

        assert "チーム構成について" in prompt
        assert "唯一の社員" in prompt

    def test_hiring_context_before_behavior_rules(self, data_dir: Path, make_anima):
        """hiring_context must appear before behavior_rules."""
        make_anima("solo")

        prompt = self._build_prompt_for(data_dir, "solo")

        hiring_pos = prompt.index("チーム構成について")
        rules_pos = prompt.index("行動ルール")
        assert hiring_pos < rules_pos

    def test_anima_with_peers_no_hiring_context(self, data_dir: Path, make_anima):
        """Anima with peers should NOT see hiring context."""
        make_anima("alice")
        make_anima("bob")

        prompt = self._build_prompt_for(data_dir, "alice")

        assert "チーム構成について" not in prompt

    def test_anima_with_supervisor_no_hiring_context(
        self, data_dir: Path, make_anima
    ):
        """Anima with a supervisor should NOT see hiring context."""
        make_anima("boss")
        make_anima("worker", supervisor="boss")

        # worker has a supervisor → no hiring context
        prompt = self._build_prompt_for(data_dir, "worker")

        assert "チーム構成について" not in prompt
