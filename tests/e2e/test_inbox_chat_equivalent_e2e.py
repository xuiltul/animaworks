"""E2E tests for inbox-as-chat-equivalent prompt building.

Verifies that inbox trigger produces a system prompt with the same key
sections as a chat trigger, using the real build_system_prompt pipeline.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.prompt.builder import build_system_prompt

# ── Helpers ────────────────────────────────────────────────────

def _make_mock_memory(tmp_path: Path, *, specialty: str = "", current_state: str = "") -> MagicMock:
    memory = MagicMock()
    memory.anima_dir = tmp_path / "animas" / "test_anima"
    memory.anima_dir.mkdir(parents=True, exist_ok=True)
    (memory.anima_dir / "knowledge").mkdir(exist_ok=True)
    (memory.anima_dir / "episodes").mkdir(exist_ok=True)
    (memory.anima_dir / "skills").mkdir(exist_ok=True)
    (memory.anima_dir / "procedures").mkdir(exist_ok=True)
    memory.read_identity.return_value = "# Identity\nI am test_anima."
    memory.read_injection.return_value = ""
    memory.read_permissions.return_value = ""
    memory.read_specialty_prompt.return_value = specialty
    memory.read_bootstrap.return_value = ""
    memory.read_company_vision.return_value = ""
    memory.read_current_state.return_value = current_state
    memory.read_pending.return_value = ""
    memory.read_resolutions.return_value = []
    memory.read_model_config.return_value = MagicMock(
        model="claude-sonnet-4-20250514", supervisor=None, max_chains=3
    )
    memory.list_knowledge_files.return_value = []
    memory.list_episode_files.return_value = []
    memory.list_procedure_files.return_value = []
    memory.list_skill_metas.return_value = []
    memory.list_common_skill_metas.return_value = []
    memory.list_procedure_metas.return_value = []
    memory.list_shared_users.return_value = []
    memory.common_skills_dir = tmp_path / "common_skills"
    return memory


def _build_with_trigger(tmp_path: Path, trigger: str, **kwargs) -> str:
    """Build system prompt for a given trigger and return the prompt text."""
    memory = _make_mock_memory(tmp_path, **kwargs)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    with (
        patch("core.prompt.builder.get_data_dir", return_value=data_dir),
        patch("core.prompt.builder._discover_other_animas", return_value=[]),
        patch("core.tooling.prompt_db.get_prompt_store", return_value=None),
        patch("core.tooling.prompt_db.get_default_guide", return_value=""),
    ):
        result = build_system_prompt(
            memory,
            execution_mode="a",
            trigger=trigger,
            context_window=200_000,
        )
    return result.system_prompt


# ── E2E Tests ─────────────────────────────────────────────────


@pytest.mark.e2e
class TestInboxChatEquivalentE2E:
    """Inbox prompt must match chat prompt in key sections."""

    def test_inbox_and_chat_both_include_specialty(self, tmp_path):
        """Both inbox and chat should include the specialty prompt."""
        sp = "## Specialty\nExpert in Kubernetes cluster management."

        chat_prompt = _build_with_trigger(tmp_path, trigger="", specialty=sp)
        inbox_prompt = _build_with_trigger(tmp_path, trigger="inbox:alice", specialty=sp)

        assert "Kubernetes cluster management" in chat_prompt
        assert "Kubernetes cluster management" in inbox_prompt

    def test_inbox_and_chat_both_include_full_state(self, tmp_path):
        """Both should include full current_state without 500-char cap."""
        long_state = "Resolved ECS migration issue. " * 30  # ~900 chars

        chat_prompt = _build_with_trigger(tmp_path, trigger="", current_state=long_state)
        inbox_prompt = _build_with_trigger(tmp_path, trigger="inbox:alice", current_state=long_state)

        chat_resolved_count = chat_prompt.count("Resolved ECS")
        inbox_resolved_count = inbox_prompt.count("Resolved ECS")
        assert inbox_resolved_count == chat_resolved_count

    def test_heartbeat_still_excludes_specialty(self, tmp_path):
        """Heartbeat should NOT include specialty (regression check)."""
        sp = "## Specialty\nExpert in Kubernetes."

        hb_prompt = _build_with_trigger(tmp_path, trigger="heartbeat", specialty=sp)
        assert "Kubernetes" not in hb_prompt

    def test_inbox_includes_emotion_section(self, tmp_path):
        """Inbox should include emotion instruction (group 6)."""
        inbox_prompt = _build_with_trigger(tmp_path, trigger="inbox:alice")
        assert "6. メタ設定" in inbox_prompt or "Meta" in inbox_prompt

    def test_inbox_and_chat_prompt_length_parity(self, tmp_path):
        """Inbox prompt should be similar length to chat (not drastically shorter)."""
        sp = "## Specialty\nDevOps expert."

        chat_prompt = _build_with_trigger(tmp_path, trigger="", specialty=sp)
        inbox_prompt = _build_with_trigger(tmp_path, trigger="inbox:alice", specialty=sp)

        ratio = len(inbox_prompt) / len(chat_prompt) if len(chat_prompt) > 0 else 0
        assert ratio > 0.9, f"Inbox prompt is only {ratio:.0%} of chat prompt length"
