# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests verifying hiring_context is no longer injected.

hiring_context was migrated to the newstaff skill.
The system prompt should never contain inline hiring context.
"""

from __future__ import annotations

from pathlib import Path


from core.memory import MemoryManager
from core.prompt.builder import build_system_prompt


class TestHiringContextRemovedE2E:
    """End-to-end: hiring_context must not appear in system prompts."""

    def _build_prompt_for(self, data_dir: Path, name: str) -> str:
        """Helper: build system prompt for the named anima."""
        anima_dir = data_dir / "animas" / name
        memory = MemoryManager(anima_dir)
        return build_system_prompt(memory)

    def test_solo_anima_no_hiring_context(self, data_dir: Path, make_anima):
        """Solo top-level anima should NOT see hiring context (migrated to skill)."""
        make_anima("solo")

        prompt = self._build_prompt_for(data_dir, "solo")

        assert "チーム構成について" not in prompt
        assert "唯一の社員" not in prompt

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

        prompt = self._build_prompt_for(data_dir, "worker")

        assert "チーム構成について" not in prompt
