# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""E2E tests for episode dedup, state auto-update, and resolution propagation.

Verifies the complete flow: fire-and-forget removal, heartbeat integration,
differential finalization, state update, and resolution recording.
"""

import re
from pathlib import Path

import pytest


# ── Fire-and-forget removal verification ─────────────────────

class TestFireAndForgetRemoved:
    """Verify fire-and-forget finalize_session calls are removed from anima.py."""

    def test_no_fire_and_forget_in_process_message(self):
        """process_message() should not contain asyncio.create_task(finalize_session)."""
        anima_path = Path(__file__).resolve().parents[2] / "core" / "anima.py"
        content = anima_path.read_text(encoding="utf-8")

        # Find the process_message method
        in_process_message = False
        in_process_message_stream = False
        fire_and_forget_lines: list[str] = []

        for i, line in enumerate(content.splitlines(), 1):
            if "async def process_message(" in line and "stream" not in line:
                in_process_message = True
                in_process_message_stream = False
            elif "async def process_message_stream(" in line:
                in_process_message = False
                in_process_message_stream = True
            elif re.match(r"    (async )?def ", line) and "process_message" not in line:
                in_process_message = False
                in_process_message_stream = False

            if (in_process_message or in_process_message_stream):
                if "create_task" in line and "finalize_session" in line:
                    fire_and_forget_lines.append(f"Line {i}: {line.strip()}")

        assert fire_and_forget_lines == [], (
            f"Fire-and-forget finalize_session calls found in anima.py:\n"
            + "\n".join(fire_and_forget_lines)
        )

    def test_heartbeat_calls_finalize_if_session_ended(self):
        """run_heartbeat() should call finalize_if_session_ended()."""
        anima_path = Path(__file__).resolve().parents[2] / "core" / "anima.py"
        content = anima_path.read_text(encoding="utf-8")

        assert "finalize_if_session_ended" in content, (
            "finalize_if_session_ended() not found in anima.py"
        )

        # Verify it's in the heartbeat section (between heartbeat_end and episode recording)
        heartbeat_section = content[content.find("heartbeat_end"):]
        episode_section_idx = heartbeat_section.find("Record important heartbeat actions")
        if episode_section_idx > 0:
            between = heartbeat_section[:episode_section_idx]
            assert "finalize_if_session_ended" in between, (
                "finalize_if_session_ended() not in the correct location (between heartbeat_end and episode recording)"
            )


# ── Differential finalization E2E ─────────────────────────────

class TestDifferentialFinalizationE2E:
    """End-to-end test for differential episode recording."""

    @pytest.mark.asyncio
    async def test_finalize_full_flow(self, data_dir):
        """Full finalization: turns → episode → state update → resolution."""
        from tests.helpers.filesystem import create_anima_dir
        from tests.helpers.mocks import make_litellm_response, patch_litellm
        from core.memory.conversation import ConversationMemory, ConversationTurn
        from core.schemas import ModelConfig
        from datetime import date

        anima_dir = create_anima_dir(data_dir, "e2e-dedup")
        model_config = ModelConfig(
            model="claude-sonnet-4-20250514",
            fallback_model="claude-sonnet-4-20250514",
        )
        conv = ConversationMemory(anima_dir, model_config)

        # Setup: 4 turns (above min_turns=3)
        state = conv.load()
        state.turns = [
            ConversationTurn(role="human", content="サーバー障害の件、修正しました"),
            ConversationTurn(role="assistant", content="確認しました。修正ありがとうございます。"),
            ConversationTurn(role="human", content="あと新しいデプロイタスクも追加してください"),
            ConversationTurn(role="assistant", content="承知しました。デプロイタスクを登録します。"),
        ]
        conv.save()

        # Mock LLM: summary with resolved + new task
        summary_resp = make_litellm_response(
            content=(
                "## エピソード要約\n"
                "サーバー障害修正とデプロイタスク\n\n"
                "**相手**: human\n"
                "**トピック**: 障害修正, デプロイ\n"
                "**要点**:\n"
                "- サーバー障害を修正\n"
                "- デプロイタスクを新規作成\n\n"
                "## ステート変更\n"
                "### 解決済み\n"
                "- サーバー障害\n"
                "### 新規タスク\n"
                "- デプロイ作業\n"
                "### 現在の状態\n"
                "idle\n"
            )
        )
        compress_resp = make_litellm_response(content="サーバー障害修正完了、デプロイ予定")

        with patch_litellm(summary_resp, compress_resp):
            result = await conv.finalize_session(min_turns=3)

        assert result is True

        # Verify episode was written
        episode_path = anima_dir / "episodes" / f"{date.today().isoformat()}.md"
        assert episode_path.exists()
        episode_content = episode_path.read_text(encoding="utf-8")
        assert "サーバー障害修正とデプロイタスク" in episode_content

        # Verify state was updated
        state_content = (anima_dir / "state" / "current_task.md").read_text(encoding="utf-8")
        assert "デプロイ作業" in state_content

        # Verify resolution was recorded
        from core.memory.manager import MemoryManager
        mm = MemoryManager(anima_dir)
        resolutions = mm.read_resolutions(days=1)
        assert len(resolutions) >= 1
        assert any("サーバー障害" in r["issue"] for r in resolutions)

        # Verify last_finalized_turn_index updated
        conv._state = None
        loaded = conv.load()
        assert loaded.last_finalized_turn_index == 4

    @pytest.mark.asyncio
    async def test_no_duplicate_episodes_on_double_finalize(self, data_dir):
        """Calling finalize_session twice does not create duplicate episodes."""
        from tests.helpers.filesystem import create_anima_dir
        from tests.helpers.mocks import make_litellm_response, patch_litellm
        from core.memory.conversation import ConversationMemory, ConversationTurn
        from core.schemas import ModelConfig

        anima_dir = create_anima_dir(data_dir, "e2e-nodup")
        model_config = ModelConfig(
            model="claude-sonnet-4-20250514",
            fallback_model="claude-sonnet-4-20250514",
        )
        conv = ConversationMemory(anima_dir, model_config)

        state = conv.load()
        state.turns = [
            ConversationTurn(role="human", content=f"message {i}")
            for i in range(4)
        ]
        conv.save()

        summary_resp = make_litellm_response(
            content="## エピソード要約\n要約テスト\n\n## ステート変更\n### 解決済み\n- なし\n### 新規タスク\n- なし\n### 現在の状態\nidle"
        )
        compress_resp = make_litellm_response(content="圧縮")

        # First finalization
        with patch_litellm(summary_resp, compress_resp):
            r1 = await conv.finalize_session(min_turns=3)
        assert r1 is True

        # Second finalization should be skipped (no new turns)
        r2 = await conv.finalize_session(min_turns=3)
        assert r2 is False


# ── Resolution propagation E2E ────────────────────────────────

class TestResolutionPropagationE2E:
    """End-to-end test for resolution propagation across components."""

    def test_resolution_in_system_prompt(self, data_dir):
        """Resolutions are visible in system prompt."""
        from tests.helpers.filesystem import create_anima_dir
        from core.memory.manager import MemoryManager
        from core.prompt.builder import build_system_prompt

        anima_dir = create_anima_dir(data_dir, "e2e-prompt")
        mm = MemoryManager(anima_dir)

        # Record resolutions
        mm.append_resolution(issue="ネットワーク障害修正", resolver="e2e-prompt")
        mm.append_resolution(issue="DBマイグレーション完了", resolver="e2e-prompt")

        # Build system prompt
        prompt = build_system_prompt(mm)

        assert "解決済み案件" in prompt
        assert "ネットワーク障害修正" in prompt
        assert "DBマイグレーション完了" in prompt
        assert "再調査・再報告は不要" in prompt
