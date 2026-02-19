# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for organisation context injection into system prompts.

Verifies that a realistic multi-anima hierarchy produces correct
supervisor / subordinate / peer annotations in the built system prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.memory import MemoryManager
from core.prompt.builder import build_system_prompt


class TestOrgContextE2E:
    """End-to-end: build_system_prompt includes org context from config."""

    @pytest.fixture
    def org_team(self, data_dir: Path, make_anima):
        """Set up a realistic 3-tier organisation.

        sakura (top)
        ├── rin (development leader)
        │   ├── alice (frontend)
        │   └── bob (backend)
        └── kotoha (communication leader)
        """
        make_anima("sakura")
        make_anima("rin", supervisor="sakura", speciality="development")
        make_anima("kotoha", supervisor="sakura", speciality="communication")
        make_anima("alice", supervisor="rin", speciality="frontend")
        make_anima("bob", supervisor="rin", speciality="backend")
        return data_dir

    def _build_prompt_for(self, data_dir: Path, name: str) -> str:
        """Helper: build system prompt for the named anima."""
        anima_dir = data_dir / "animas" / name
        memory = MemoryManager(anima_dir)
        return build_system_prompt(memory)

    def test_top_level_sees_subordinates(self, org_team: Path):
        prompt = self._build_prompt_for(org_team, "sakura")

        assert "あなたはトップレベルです" in prompt
        assert "rin (development)" in prompt
        assert "kotoha (communication)" in prompt
        # Top has no peers
        assert "コミュニケーションルール" in prompt

    def test_leader_sees_hierarchy(self, org_team: Path):
        prompt = self._build_prompt_for(org_team, "rin")

        # Supervisor
        assert "sakura" in prompt
        # Subordinates
        assert "alice (frontend)" in prompt
        assert "bob (backend)" in prompt
        # Peer leader
        assert "kotoha (communication)" in prompt

    def test_worker_sees_supervisor_and_peers(self, org_team: Path):
        prompt = self._build_prompt_for(org_team, "alice")

        # Supervisor
        assert "rin (development)" in prompt
        # Peer
        assert "bob (backend)" in prompt
        # Should not list sakura as peer (different supervisor)
        # alice's peers are only those with supervisor=rin

    def test_communication_rules_present(self, org_team: Path):
        prompt = self._build_prompt_for(org_team, "rin")

        assert "コミュニケーションルール" in prompt
        assert "上司への報告" in prompt
        assert "部下への指示" in prompt
        assert "同僚との連携" in prompt
        assert "他部署のメンバー" in prompt

    def test_solo_anima_no_communication_rules(self, data_dir: Path, make_anima):
        """A solo anima should not see communication rules."""
        make_anima("solo")
        prompt = self._build_prompt_for(data_dir, "solo")

        assert "あなたがトップです" in prompt
        assert "コミュニケーションルール" not in prompt

    def test_messaging_section_still_present(self, org_team: Path):
        """Messaging instructions remain alongside org context."""
        prompt = self._build_prompt_for(org_team, "rin")

        assert "メッセージ送信" in prompt
        assert "送信可能な相手" in prompt
