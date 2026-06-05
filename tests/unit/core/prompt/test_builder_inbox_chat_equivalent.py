"""Unit tests for inbox prompt isolation in builder.py.

Verifies that inbox keeps communication-capable operational sections without
being treated as chat-continuity context.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.prompt.builder import build_system_prompt

_MOCK_SECTIONS = (
    "[group1_header]: # 1. 動作環境と行動ルール\n"
    "[current_time_label]: **現在時刻**:\n"
    "[group2_header]: # 2. あなた自身\n"
    "[group3_header]: # 3. 現在の状況\n"
    "[current_state_header]: ## 現在の状態\n"
    "[pending_tasks_header]: ## 未完了タスク\n"
    "[group4_header]: # 4. 記憶と能力\n"
    "[group5_header]: # 5. 組織とコミュニケーション\n"
    "[group6_header]: # 6. メタ設定\n"
    "[you_marker]:   ← あなた\n"
    "[common_label]: (共通)\n"
    "[recent_tool_results_header]: ## Recent Tool Results\n"
)

_MOCK_FALLBACKS = (
    "[unset]: (未設定)\n"
    "[none]: (なし)\n"
    "[none_top_level]: (なし — あなたがトップです)\n"
    "[no_other_animas]: (まだ他の社員はいません)\n"
    "[truncated]: （前半省略）\n"
    "[summary]: （要約）\n"
)


def _mock_load_prompt(name: str, **kwargs) -> str:
    if name == "builder/sections":
        return _MOCK_SECTIONS
    if name == "builder/fallbacks":
        return _MOCK_FALLBACKS
    if name == "builder/task_in_progress":
        return f"## ⚠️ 進行中タスク\n\n{kwargs.get('state', '')}"
    return "section"


def _mock_memory(tmp_path: Path, *, specialty: str = "", current_state: str = "") -> MagicMock:
    anima_dir = tmp_path / "animas" / "test"
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "knowledge").mkdir(exist_ok=True)
    (anima_dir / "episodes").mkdir(exist_ok=True)
    (anima_dir / "skills").mkdir(exist_ok=True)
    (anima_dir / "procedures").mkdir(exist_ok=True)

    memory = MagicMock()
    memory.anima_dir = anima_dir
    memory.read_identity.return_value = "# Identity\nI am test."
    memory.read_injection.return_value = ""
    memory.read_current_state.return_value = current_state
    memory.read_pending.return_value = ""
    memory.read_bootstrap.return_value = ""
    memory.read_company_vision.return_value = ""
    memory.read_specialty_prompt.return_value = specialty
    memory.read_permissions.return_value = ""
    memory.list_knowledge_files.return_value = []
    memory.list_episode_files.return_value = []
    memory.list_procedure_files.return_value = []
    memory.list_skill_metas.return_value = []
    memory.list_common_skill_metas.return_value = []
    memory.list_procedure_metas.return_value = []
    memory.list_shared_users.return_value = []
    memory.read_resolutions.return_value = []
    return memory


def _apply_patches(stack: ExitStack, tmp_path: Path) -> None:
    stack.enter_context(patch("core.prompt.builder.get_data_dir", return_value=tmp_path))
    stack.enter_context(patch("core.tooling.prompt_db.get_prompt_store", return_value=None))
    stack.enter_context(patch("core.tooling.prompt_db.get_default_guide", return_value=""))
    stack.enter_context(patch("core.prompt.builder.load_prompt", side_effect=_mock_load_prompt))
    stack.enter_context(patch("core.prompt.builder._discover_other_animas", return_value=[]))


class TestInboxPromptIsolation:
    """Inbox trigger is communication-capable but not chat-equivalent."""

    def test_inbox_includes_specialty(self, tmp_path):
        memory = _mock_memory(tmp_path, specialty="## Specialty\nI specialize in DevOps.")

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="inbox:alice",
                context_window=200_000,
            )
        assert "I specialize in DevOps" in result.system_prompt

    def test_heartbeat_excludes_specialty(self, tmp_path):
        """Heartbeat should still NOT include specialty (unchanged behavior)."""
        memory = _mock_memory(tmp_path, specialty="## Specialty\nI specialize in DevOps.")

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="heartbeat",
                context_window=200_000,
            )
        assert "I specialize in DevOps" not in result.system_prompt

    def test_inbox_no_current_state_500_cap(self, tmp_path):
        """Inbox should use scale-based limit, not the old 500-char hard cap."""
        long_state = "A" * 800
        memory = _mock_memory(tmp_path, current_state=long_state)

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="inbox:alice",
                context_window=200_000,
            )
        assert "AAAA" in result.system_prompt
        acount = result.system_prompt.count("A")
        assert acount > 500

    def test_inbox_excludes_chat_emotion_section(self, tmp_path):
        memory = _mock_memory(tmp_path)

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="inbox:alice",
                context_window=200_000,
            )
        assert '<section name="emotion_instruction">' not in result.system_prompt

    def test_inbox_includes_a_reflection(self, tmp_path):
        memory = _mock_memory(tmp_path)

        reflection_text = "## A-mode reflection\nReflect on past actions."

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            stack.enter_context(
                patch("core.prompt.builder._load_a_reflection", return_value=reflection_text)
            )
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="inbox:alice",
                context_window=200_000,
            )
        assert "Reflect on past actions" in result.system_prompt

    def test_chat_still_works_identically(self, tmp_path):
        """Regression: chat trigger should still include all sections."""
        memory = _mock_memory(tmp_path, specialty="## Specialty\nI specialize in DevOps.")

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="",
                context_window=200_000,
            )
        assert "I specialize in DevOps" in result.system_prompt
        assert '<section name="emotion_instruction">' in result.system_prompt

    def test_inbox_excludes_pending_human_notifications(self, tmp_path):
        memory = _mock_memory(tmp_path)

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="inbox:alice",
                context_window=200_000,
                pending_human_notifications="notify human now",
            )
        assert "notify human now" not in result.system_prompt

    def test_chat_includes_pending_human_notifications(self, tmp_path):
        memory = _mock_memory(tmp_path)

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="message:alice",
                context_window=200_000,
                pending_human_notifications="notify human now",
            )
        assert "notify human now" in result.system_prompt

    def test_task_includes_specialty(self, tmp_path):
        """Task trigger now includes specialty (full-context TaskExec)."""
        memory = _mock_memory(tmp_path, specialty="## Specialty\nI specialize in DevOps.")

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="task:abc123",
                context_window=200_000,
            )
        assert "I specialize in DevOps" in result.system_prompt
