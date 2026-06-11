# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Tests for episode dedup, state auto-update, and resolution propagation.

Issue: 20260218_episode-dedup-state-autoupdate-resolution-propagation
"""

import json
from datetime import timedelta
from unittest.mock import patch

import pytest

from core.memory.conversation import (
    SESSION_GAP_MINUTES,
    ConversationMemory,
    ConversationTurn,
    ParsedSessionSummary,
)
from core.schemas import ModelConfig
from core.time_utils import now_jst, today_local
from tests.helpers.filesystem import create_anima_dir, create_test_data_dir
from tests.helpers.mocks import make_litellm_response, patch_litellm

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Isolated AnimaWorks data directory."""
    from core.config import invalidate_cache
    from core.paths import _prompt_cache

    d = create_test_data_dir(tmp_path)
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(d))
    invalidate_cache()
    _prompt_cache.clear()
    yield d
    invalidate_cache()
    _prompt_cache.clear()


@pytest.fixture
def anima_dir(data_dir):
    """Create test anima directory."""
    return create_anima_dir(data_dir, "test-dedup")


@pytest.fixture
def model_config():
    """Basic model config for ConversationMemory."""
    return ModelConfig(
        model="claude-sonnet-4-6",
        fallback_model="claude-sonnet-4-6",
        max_turns=5,
    )


@pytest.fixture
def conv_memory(anima_dir, model_config):
    """ConversationMemory instance."""
    return ConversationMemory(anima_dir, model_config)


# ── ConversationState tests ──────────────────────────────────


class TestConversationStateIndex:
    """Tests for last_finalized_turn_index in ConversationState."""

    def test_load_migration_default_index(self, conv_memory):
        """Old state files without last_finalized_turn_index get default 0."""
        # Write a legacy state file (without last_finalized_turn_index)
        state_path = conv_memory._state_path
        state_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_data = {
            "anima_name": "test-dedup",
            "turns": [],
            "compressed_summary": "",
            "compressed_turn_count": 0,
        }
        state_path.write_text(json.dumps(legacy_data), encoding="utf-8")

        state = conv_memory.load()
        assert state.last_finalized_turn_index == 0

    def test_save_and_load_preserves_index(self, conv_memory):
        """Saving and loading preserves last_finalized_turn_index."""
        state = conv_memory.load()
        state.last_finalized_turn_index = 5
        conv_memory.save()

        # Force reload
        conv_memory._state = None
        loaded = conv_memory.load()
        assert loaded.last_finalized_turn_index == 5


# ── finalize_session tests ────────────────────────────────────


class TestFinalizeSession:
    """Tests for differential session finalization."""

    @pytest.mark.asyncio
    async def test_finalize_session_skips_few_turns(self, conv_memory):
        """Finalization is skipped when new turns < min_turns."""
        state = conv_memory.load()
        state.turns = [
            ConversationTurn(role="human", content="hi"),
            ConversationTurn(role="assistant", content="hello"),
        ]
        conv_memory.save()

        result = await conv_memory.finalize_session(min_turns=3)
        assert result is False

    @pytest.mark.asyncio
    async def test_finalize_session_incremental(self, conv_memory, anima_dir):
        """Finalization only processes turns since last_finalized_turn_index."""
        state = conv_memory.load()
        # 6 turns total, first 2 already finalized
        state.turns = [
            ConversationTurn(role="human", content="old message 1"),
            ConversationTurn(role="assistant", content="old response 1"),
            ConversationTurn(role="human", content="new message 1"),
            ConversationTurn(role="assistant", content="new response 1"),
            ConversationTurn(role="human", content="new message 2"),
            ConversationTurn(role="assistant", content="new response 2"),
        ]
        state.last_finalized_turn_index = 2
        conv_memory.save()

        # Mock LLM responses:
        # 1. _summarize_session_with_state
        summary_resp = make_litellm_response(
            content="## エピソード要約\nテスト会話\n\n**相手**: human\n"
            "**トピック**: テスト\n**要点**:\n- テスト\n\n"
            "## ステート変更\n### 解決済み\n- なし\n### 新規タスク\n- なし\n### 現在の状態\nidle"
        )
        # 2. _generate_compression_summary
        compress_resp = make_litellm_response(content="圧縮された要約")

        with patch_litellm(summary_resp, compress_resp):
            result = await conv_memory.finalize_session(min_turns=3)

        assert result is True

        # Verify turns cleared and index reset after successful finalization
        conv_memory._state = None
        loaded = conv_memory.load()
        assert loaded.turns == []
        assert loaded.last_finalized_turn_index == 0
        assert loaded.compressed_summary == "圧縮された要約"

        # Verify episode was written
        episode_file = anima_dir / "episodes" / f"{today_local().isoformat()}.md"
        assert episode_file.exists()
        content = episode_file.read_text(encoding="utf-8")
        assert "テスト会話" in content

    @pytest.mark.asyncio
    async def test_finalize_session_preserves_existing_current_state(self, conv_memory, anima_dir):
        """Finalization does not erase rich current_state.md content."""
        from core.memory.manager import MemoryManager

        memory_mgr = MemoryManager(anima_dir)
        original_state = "# current_state\n\n## Goal\nKeep this operating context."
        memory_mgr.update_state(original_state)

        state = conv_memory.load()
        state.turns = [ConversationTurn(role="human", content=f"msg {i}") for i in range(4)]
        conv_memory.save()

        summary_resp = make_litellm_response(
            content=(
                "## エピソード要約\n要約\n\n"
                "## ステート変更\n### 解決済み\n- なし\n"
                "### 新規タスク\n- なし\n### 現在の状態\nidle"
            )
        )
        compress_resp = make_litellm_response(content="圧縮")

        with patch_litellm(summary_resp, compress_resp):
            result = await conv_memory.finalize_session(min_turns=3)

        assert result is True
        assert memory_mgr.read_current_state() == original_state

    @pytest.mark.asyncio
    async def test_finalize_session_initializes_idle_current_state(self, conv_memory, anima_dir):
        """Finalization may write extracted status when current_state.md is idle."""
        from core.memory.manager import MemoryManager

        memory_mgr = MemoryManager(anima_dir)
        memory_mgr.update_state("status: idle")

        state = conv_memory.load()
        state.turns = [ConversationTurn(role="human", content=f"msg {i}") for i in range(4)]
        conv_memory.save()

        summary_resp = make_litellm_response(
            content=(
                "## エピソード要約\n要約\n\n"
                "## ステート変更\n### 解決済み\n- なし\n"
                "### 新規タスク\n- なし\n### 現在の状態\nReviewing outbound lead quality"
            )
        )
        compress_resp = make_litellm_response(content="圧縮")

        with patch_litellm(summary_resp, compress_resp):
            result = await conv_memory.finalize_session(min_turns=3)

        assert result is True
        assert memory_mgr.read_current_state() == "Reviewing outbound lead quality"

    @pytest.mark.asyncio
    async def test_finalize_session_clears_turns_and_resets_index(self, conv_memory):
        """After finalization, turns are cleared and last_finalized_turn_index is 0."""
        state = conv_memory.load()
        state.turns = [ConversationTurn(role="human", content=f"msg {i}") for i in range(4)]
        state.last_finalized_turn_index = 0
        conv_memory.save()

        summary_resp = make_litellm_response(
            content="## エピソード要約\n要約\n\n## ステート変更\n### 解決済み\n- なし\n### 新規タスク\n- なし\n### 現在の状態\nidle"
        )
        compress_resp = make_litellm_response(content="圧縮")

        with patch_litellm(summary_resp, compress_resp):
            await conv_memory.finalize_session(min_turns=3)

        conv_memory._state = None
        loaded = conv_memory.load()
        assert loaded.turns == []
        assert loaded.last_finalized_turn_index == 0
        assert loaded.compressed_summary == "圧縮"

    @pytest.mark.asyncio
    async def test_finalize_session_keeps_turns_on_compression_failure(self, conv_memory):
        """When _generate_compression_summary fails, turns are NOT cleared."""
        state = conv_memory.load()
        state.turns = [ConversationTurn(role="human", content=f"msg {i}") for i in range(4)]
        state.last_finalized_turn_index = 0
        conv_memory.save()

        summary_resp = make_litellm_response(
            content="## エピソード要約\n要約\n\n## ステート変更\n### 解決済み\n- なし\n### 新規タスク\n- なし\n### 現在の状態\nidle"
        )

        with (
            patch_litellm(summary_resp),
            patch(
                "core.memory.conversation_finalize._generate_compression_summary",
                side_effect=RuntimeError("LLM API error"),
            ),
        ):
            result = await conv_memory.finalize_session(min_turns=3)

        assert result is True

        conv_memory._state = None
        loaded = conv_memory.load()
        assert len(loaded.turns) == 4
        assert loaded.last_finalized_turn_index == 4
        assert loaded.compressed_turn_count == 0

    @pytest.mark.asyncio
    async def test_finalize_session_compresses_retained_turns_after_previous_failure(self, conv_memory):
        """A later successful finalization includes retained raw turns before clearing them."""
        state = conv_memory.load()
        state.turns = [
            ConversationTurn(role="human", content="old msg 1"),
            ConversationTurn(role="assistant", content="old response 1"),
            ConversationTurn(role="human", content="old msg 2"),
            ConversationTurn(role="assistant", content="old response 2"),
            ConversationTurn(role="human", content="new msg"),
            ConversationTurn(role="assistant", content="new response"),
            ConversationTurn(role="human", content="new follow-up"),
        ]
        state.last_finalized_turn_index = 4
        conv_memory.save()

        summary_resp = make_litellm_response(
            content="## エピソード要約\n要約\n\n## ステート変更\n### 解決済み\n- なし\n### 新規タスク\n- なし\n### 現在の状態\nidle"
        )
        captured: dict[str, str] = {}

        async def fake_compress(old_summary: str, turn_text: str, turns, model_config=None) -> tuple:
            captured["turn_text"] = turn_text
            return ("全turn圧縮", "llm_primary", "", "")

        with (
            patch_litellm(summary_resp),
            patch("core.memory.conversation_finalize._generate_compression_summary", side_effect=fake_compress),
        ):
            result = await conv_memory.finalize_session(min_turns=3)

        assert result is True
        assert "old msg 1" in captured["turn_text"]
        assert "new follow-up" in captured["turn_text"]

        conv_memory._state = None
        loaded = conv_memory.load()
        assert loaded.turns == []
        assert loaded.last_finalized_turn_index == 0
        assert loaded.compressed_summary == "全turn圧縮"
        assert loaded.compressed_turn_count == 7

    @pytest.mark.asyncio
    async def test_needs_compression_false_after_finalization(self, conv_memory, model_config):
        """After finalization, needs_compression() returns False (no double-counting)."""
        state = conv_memory.load()
        state.turns = [ConversationTurn(role="human", content=f"msg {i}") for i in range(6)]
        state.last_finalized_turn_index = 0
        conv_memory.save()

        summary_resp = make_litellm_response(
            content="## エピソード要約\n要約\n\n## ステート変更\n### 解決済み\n- なし\n### 新規タスク\n- なし\n### 現在の状態\nidle"
        )
        compress_resp = make_litellm_response(content="圧縮された要約テキスト")

        with patch_litellm(summary_resp, compress_resp):
            await conv_memory.finalize_session(min_turns=3)

        conv_memory._state = None
        loaded = conv_memory.load()
        assert loaded.turns == []
        assert not conv_memory.needs_compression()


# ── finalize_if_session_ended tests ───────────────────────────


class TestFinalizeIfSessionEnded:
    """Tests for session gap detection."""

    @pytest.mark.asyncio
    async def test_skips_when_no_turns(self, conv_memory):
        """Returns False when there are no turns."""
        result = await conv_memory.finalize_if_session_ended()
        assert result is False

    @pytest.mark.asyncio
    async def test_skips_when_no_new_turns(self, conv_memory):
        """Returns False when all turns are already finalized."""
        state = conv_memory.load()
        state.turns = [ConversationTurn(role="human", content="hi")]
        state.last_finalized_turn_index = 1
        conv_memory.save()

        result = await conv_memory.finalize_if_session_ended()
        assert result is False

    @pytest.mark.asyncio
    async def test_skips_recent_session(self, conv_memory):
        """Returns False when last turn is recent (within SESSION_GAP_MINUTES)."""
        state = conv_memory.load()
        recent_ts = now_jst().isoformat()
        state.turns = [
            ConversationTurn(role="human", content="hi", timestamp=recent_ts),
            ConversationTurn(role="assistant", content="hello", timestamp=recent_ts),
            ConversationTurn(role="human", content="how are you?", timestamp=recent_ts),
        ]
        state.last_finalized_turn_index = 0
        conv_memory.save()

        result = await conv_memory.finalize_if_session_ended()
        assert result is False

    @pytest.mark.asyncio
    async def test_triggers_on_old_session(self, conv_memory):
        """Returns True when last turn is older than SESSION_GAP_MINUTES."""
        old_ts = (now_jst() - timedelta(minutes=SESSION_GAP_MINUTES + 5)).isoformat()
        state = conv_memory.load()
        state.turns = [
            ConversationTurn(role="human", content="msg 1", timestamp=old_ts),
            ConversationTurn(role="assistant", content="resp 1", timestamp=old_ts),
            ConversationTurn(role="human", content="msg 2", timestamp=old_ts),
        ]
        state.last_finalized_turn_index = 0
        conv_memory.save()

        summary_resp = make_litellm_response(
            content="## エピソード要約\n会話要約\n\n## ステート変更\n### 解決済み\n- なし\n### 新規タスク\n- なし\n### 現在の状態\nidle"
        )
        compress_resp = make_litellm_response(content="圧縮")

        with patch_litellm(summary_resp, compress_resp):
            result = await conv_memory.finalize_if_session_ended()

        assert result is True


# ── ParsedSessionSummary tests ────────────────────────────────


class TestParseSessionSummary:
    """Tests for _parse_session_summary."""

    def test_parse_full(self):
        """Parses a complete summary with all sections."""
        raw = (
            "## エピソード要約\n"
            "デバッグ作業の要約\n\n"
            "**相手**: admin\n"
            "**トピック**: バグ修正\n"
            "**要点**:\n- メモリリーク修正\n\n"
            "## ステート変更\n"
            "### 解決済み\n"
            "- メモリリークバグ\n"
            "- APIタイムアウト\n"
            "### 新規タスク\n"
            "- パフォーマンステスト\n"
            "### 現在の状態\n"
            "idle\n"
        )
        parsed = ConversationMemory._parse_session_summary(raw)
        assert parsed.title == "デバッグ作業の要約"
        assert "admin" in parsed.episode_body
        assert parsed.resolved_items == ["メモリリークバグ", "APIタイムアウト"]
        assert parsed.new_tasks == ["パフォーマンステスト"]
        assert parsed.current_status == "idle"
        assert parsed.has_state_changes is True

    def test_parse_no_state_section(self):
        """Falls back when no state section exists."""
        raw = "## エピソード要約\nシンプルな要約\n\n内容テスト\n"
        parsed = ConversationMemory._parse_session_summary(raw)
        assert parsed.title == "シンプルな要約"
        assert parsed.resolved_items == []
        assert parsed.new_tasks == []
        assert parsed.has_state_changes is False

    def test_parse_resolved_none(self):
        """Items marked as 'なし' are excluded."""
        raw = (
            "## エピソード要約\n要約\n\n"
            "## ステート変更\n"
            "### 解決済み\n- なし\n"
            "### 新規タスク\n- なし\n"
            "### 現在の状態\nidle\n"
        )
        parsed = ConversationMemory._parse_session_summary(raw)
        assert parsed.resolved_items == []
        assert parsed.new_tasks == []

    def test_parse_raw_fallback(self):
        """Plain text without expected sections becomes episode_body."""
        raw = "これは構造化されていないテキストです"
        parsed = ConversationMemory._parse_session_summary(raw)
        assert parsed.episode_body == raw


# ── State auto-update tests (Issue #114: task_queue routing) ────


class TestUpdateState:
    """Tests for _update_state_from_summary — routes to task_queue.jsonl."""

    def test_resolved_items_mark_task_done(self, conv_memory, anima_dir):
        """Resolved items update matching task_queue entries to done."""
        from core.memory.manager import MemoryManager
        from core.memory.task_queue import TaskQueueManager

        mm = MemoryManager(anima_dir)
        tqm = TaskQueueManager(anima_dir)
        tqm.add_task(
            source="anima",
            original_instruction="サーバー障害の修正",
            assignee=anima_dir.name,
            summary="サーバー障害の修正",
        )
        task_id = list(tqm._load_all().keys())[0]

        parsed = ParsedSessionSummary(
            title="test",
            episode_body="",
            resolved_items=["サーバー障害の修正"],
            new_tasks=[],
            current_status="",
            has_state_changes=True,
        )

        conv_memory._update_state_from_summary(mm, parsed)
        task = tqm.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "done"

    def test_new_tasks_not_added_to_queue(self, conv_memory, anima_dir):
        """new_tasks from session summary are NOT registered (auto-detection disabled)."""
        from core.memory.manager import MemoryManager
        from core.memory.task_queue import TaskQueueManager

        mm = MemoryManager(anima_dir)
        mm.update_state("status: idle")

        parsed = ParsedSessionSummary(
            title="test",
            episode_body="",
            resolved_items=[],
            new_tasks=["レポート作成"],
            current_status="",
            has_state_changes=True,
        )

        conv_memory._update_state_from_summary(mm, parsed)
        tqm = TaskQueueManager(anima_dir)
        pending = tqm.get_pending()
        assert len(pending) == 0

    def test_resolved_items_mark_matching_task_done(self, conv_memory, anima_dir):
        """Resolved items update matching task_queue entries to done."""
        from core.memory.manager import MemoryManager
        from core.memory.task_queue import TaskQueueManager

        mm = MemoryManager(anima_dir)
        tqm = TaskQueueManager(anima_dir)
        tqm.add_task(
            source="anima",
            original_instruction="ネットワーク障害の対応",
            assignee=anima_dir.name,
            summary="ネットワーク障害",
        )
        task_id = list(tqm._load_all().keys())[0]

        parsed = ParsedSessionSummary(
            title="test",
            episode_body="",
            resolved_items=["ネットワーク障害"],
            new_tasks=[],
            current_status="",
            has_state_changes=True,
        )

        conv_memory._update_state_from_summary(mm, parsed)
        task = tqm.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "done"

    def test_existing_tasks_unaffected_by_new_tasks(self, conv_memory, anima_dir):
        """Existing tasks in queue are not affected when new_tasks are present."""
        from core.memory.manager import MemoryManager
        from core.memory.task_queue import TaskQueueManager

        mm = MemoryManager(anima_dir)
        tqm = TaskQueueManager(anima_dir)
        tqm.add_task(
            source="anima",
            original_instruction="レポート作成",
            assignee=anima_dir.name,
            summary="レポート作成",
        )

        parsed = ParsedSessionSummary(
            title="test",
            episode_body="",
            resolved_items=[],
            new_tasks=["レポート作成"],
            current_status="",
            has_state_changes=True,
        )

        conv_memory._update_state_from_summary(mm, parsed)
        all_tasks = [t for t in tqm._load_all().values() if "レポート作成" in t.summary]
        assert len(all_tasks) == 1


# ── Resolution recording tests ────────────────────────────────


class TestResolutions:
    """Tests for resolution registry and activity log."""

    def test_append_and_read_resolutions(self, data_dir):
        """append_resolution writes, read_resolutions reads back."""
        anima_dir = create_anima_dir(data_dir, "resolver-test")
        from core.memory.manager import MemoryManager

        mm = MemoryManager(anima_dir)
        mm.append_resolution(issue="サーバーダウン", resolver="resolver-test")
        mm.append_resolution(issue="メール障害", resolver="resolver-test")

        results = mm.read_resolutions(days=7)
        assert len(results) == 2
        assert results[0]["issue"] == "サーバーダウン"
        assert results[1]["issue"] == "メール障害"
        assert results[0]["resolver"] == "resolver-test"

    def test_read_resolutions_filters_by_days(self, data_dir):
        """Old entries outside the days window are excluded."""
        from core.paths import get_shared_dir

        anima_dir = create_anima_dir(data_dir, "filter-test")
        from core.memory.manager import MemoryManager

        mm = MemoryManager(anima_dir)

        # Write an old entry directly
        shared_dir = get_shared_dir()
        path = shared_dir / "resolutions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        old_ts = (now_jst() - timedelta(days=10)).isoformat()
        old_entry = json.dumps({"ts": old_ts, "issue": "古い問題", "resolver": "test"})
        new_entry = json.dumps({"ts": now_jst().isoformat(), "issue": "新しい問題", "resolver": "test"})
        path.write_text(old_entry + "\n" + new_entry + "\n", encoding="utf-8")

        results = mm.read_resolutions(days=7)
        assert len(results) == 1
        assert results[0]["issue"] == "新しい問題"

    def test_record_resolutions_writes_activity(self, conv_memory, anima_dir, data_dir):
        """_record_resolutions logs issue_resolved events."""
        from core.memory.activity import ActivityLogger
        from core.memory.manager import MemoryManager

        mm = MemoryManager(anima_dir)
        conv_memory._record_resolutions(mm, ["バグ修正完了"])

        # Check activity log
        activity = ActivityLogger(anima_dir)
        entries = activity.recent(days=1, limit=10)
        resolved_entries = [e for e in entries if e.type == "issue_resolved"]
        assert len(resolved_entries) == 1
        assert "バグ修正完了" in resolved_entries[0].content

        # Check shared registry
        results = mm.read_resolutions(days=1)
        assert len(results) == 1
        assert results[0]["issue"] == "バグ修正完了"


# ── Activity log type test ────────────────────────────────────


class TestActivityLogType:
    """Tests for issue_resolved event type formatting."""

    def test_issue_resolved_ascii_label(self, anima_dir):
        """issue_resolved event gets RSLV label in priming format."""
        from core.memory.activity import ActivityLogger

        activity = ActivityLogger(anima_dir)
        activity.log("issue_resolved", content="テスト解決", summary="解決済み: テスト")

        entries = activity.recent(days=1, limit=10)
        resolved = [e for e in entries if e.type == "issue_resolved"]
        assert len(resolved) == 1

        # Check formatting includes RSLV
        formatted = activity.format_for_priming(entries, budget_tokens=5000)
        assert "RSLV" in formatted


# ── Builder resolution injection test ─────────────────────────


class TestBuilderResolutionInjection:
    """Tests for resolution injection in system prompt."""

    def test_builder_injects_resolutions(self, data_dir):
        """build_system_prompt includes resolution section when resolutions exist."""
        anima_dir = create_anima_dir(data_dir, "builder-test")
        from core.memory.manager import MemoryManager
        from core.prompt.builder import build_system_prompt

        mm = MemoryManager(anima_dir)
        mm.append_resolution(issue="テスト問題解決", resolver="builder-test")

        result = build_system_prompt(mm)
        assert "解決済み案件" in result.system_prompt
        assert "テスト問題解決" in result.system_prompt

    def test_builder_no_resolutions_when_empty(self, data_dir):
        """build_system_prompt omits resolution section when none exist."""
        anima_dir = create_anima_dir(data_dir, "builder-empty")
        from core.memory.manager import MemoryManager
        from core.prompt.builder import build_system_prompt

        mm = MemoryManager(anima_dir)
        result = build_system_prompt(mm)
        assert "## 解決済み案件（組織横断）" not in result.system_prompt


# ── Consolidation resolved events test ────────────────────────


class TestConsolidationResolved:
    """Tests for resolved events in consolidation."""

    def test_collect_resolved_events(self, anima_dir):
        """_collect_resolved_events returns issue_resolved entries."""
        from core.memory.activity import ActivityLogger
        from core.memory.consolidation import ConsolidationEngine

        activity = ActivityLogger(anima_dir)
        activity.log("issue_resolved", content="問題A解決", summary="解決済み")
        activity.log("response_sent", content="通常応答", summary="応答")

        engine = ConsolidationEngine(anima_dir, "test-dedup")
        events = engine._collect_resolved_events(hours=24)
        assert len(events) == 1
        assert "問題A解決" in events[0]["content"]
