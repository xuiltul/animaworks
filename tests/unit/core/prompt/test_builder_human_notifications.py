"""Unit tests for pending_human_notifications injection in builder.py."""
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
    return "section"


def _mock_memory(tmp_path: Path) -> MagicMock:
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
    memory.read_current_state.return_value = ""
    memory.read_pending.return_value = ""
    memory.read_bootstrap.return_value = ""
    memory.read_company_vision.return_value = ""
    memory.read_specialty_prompt.return_value = ""
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


class TestPendingHumanNotificationsInjection:
    def test_chat_includes_notifications(self, tmp_path):
        memory = _mock_memory(tmp_path)
        notifications = (
            "## Pending Human Notifications (last 24h)\n\n"
            "[2026-03-06T16:05] call_human (via slack):\nVM IP=192.168.1.100"
        )

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="",
                context_window=200_000,
                pending_human_notifications=notifications,
            )
        assert "Pending Human Notifications" in result.system_prompt
        assert "192.168.1.100" in result.system_prompt

    def test_heartbeat_includes_notifications(self, tmp_path):
        memory = _mock_memory(tmp_path)
        notifications = "## Pending Human Notifications (last 24h)\n\nNotification content"

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="heartbeat",
                context_window=200_000,
                pending_human_notifications=notifications,
            )
        assert "Pending Human Notifications" in result.system_prompt

    def test_cron_excludes_notifications(self, tmp_path):
        memory = _mock_memory(tmp_path)
        notifications = "## Pending Human Notifications (last 24h)\n\nShould not appear"

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="cron:daily",
                context_window=200_000,
                pending_human_notifications=notifications,
            )
        assert "Pending Human Notifications" not in result.system_prompt

    def test_inbox_excludes_notifications(self, tmp_path):
        """Inbox is isolated from chat pending-human notification context."""
        memory = _mock_memory(tmp_path)
        notifications = "## Pending Human Notifications (last 24h)\n\nShould not appear for inbox"

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="inbox:someone",
                context_window=200_000,
                pending_human_notifications=notifications,
            )
        assert "Pending Human Notifications" not in result.system_prompt

    def test_task_excludes_notifications(self, tmp_path):
        memory = _mock_memory(tmp_path)
        notifications = "## Pending Human Notifications (last 24h)\n\nShould not appear"

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="task:abc",
                context_window=200_000,
                pending_human_notifications=notifications,
            )
        assert "Pending Human Notifications" not in result.system_prompt

    def test_empty_notifications_no_section(self, tmp_path):
        memory = _mock_memory(tmp_path)

        with ExitStack() as stack:
            _apply_patches(stack, tmp_path)
            result = build_system_prompt(
                memory,
                execution_mode="a",
                trigger="",
                context_window=200_000,
                pending_human_notifications="",
            )
        assert "Pending Human Notifications" not in result.system_prompt
