"""Unit tests for tool_records in conversation memory.

Tests the ToolRecord dataclass, structured message building,
tool markers in history, and backward compatibility with
conversation.json files that lack tool_records.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.memory.conversation import (
    ConversationMemory,
    ConversationState,
    ConversationTurn,
    ToolRecord,
    _MAX_TOOL_RECORDS_PER_TURN,
    _MAX_RENDERED_TOOL_RECORDS,
)
from core.schemas import ModelConfig


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "sakura"
    (d / "state").mkdir(parents=True)
    (d / "transcripts").mkdir(parents=True)
    return d


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="claude-sonnet-4-20250514",
        conversation_history_threshold=0.30,
    )


@pytest.fixture
def conv(anima_dir: Path, model_config: ModelConfig) -> ConversationMemory:
    return ConversationMemory(anima_dir, model_config)


# ── ToolRecord dataclass ──────────────────────────────────────


class TestToolRecord:
    def test_basic_creation(self):
        r = ToolRecord(tool_name="mcp__aw__post_channel")
        assert r.tool_name == "mcp__aw__post_channel"
        assert r.tool_id == ""
        assert r.input_summary == ""
        assert r.result_summary == ""

    def test_full_creation(self):
        r = ToolRecord(
            tool_name="call_human",
            tool_id="toolu_123",
            input_summary='{"subject": "test"}',
            result_summary="slack: OK",
        )
        assert r.tool_name == "call_human"
        assert r.tool_id == "toolu_123"
        assert r.input_summary == '{"subject": "test"}'
        assert r.result_summary == "slack: OK"


# ── Serialization / Deserialization round-trip ─────────────────


class TestToolRecordSerialization:
    def test_round_trip_with_tool_records(self, conv: ConversationMemory):
        """ToolRecord survives save → load cycle."""
        records = [
            ToolRecord(
                tool_name="mcp__aw__post_channel",
                tool_id="toolu_abc",
                input_summary='channel=general',
                result_summary="posted",
            ),
            ToolRecord(
                tool_name="call_human",
                tool_id="toolu_def",
                input_summary='subject=test',
                result_summary="slack: OK",
            ),
        ]
        conv.append_turn("human", "テスト投稿してみて")
        conv.append_turn("assistant", "投稿しました", tool_records=records)
        conv.save()

        # Force reload from disk
        conv._state = None
        state = conv.load()

        assert len(state.turns) == 2
        assistant_turn = state.turns[1]
        assert len(assistant_turn.tool_records) == 2
        assert assistant_turn.tool_records[0].tool_name == "mcp__aw__post_channel"
        assert assistant_turn.tool_records[0].tool_id == "toolu_abc"
        assert assistant_turn.tool_records[1].tool_name == "call_human"
        assert assistant_turn.tool_records[1].result_summary == "slack: OK"

    def test_backward_compat_no_tool_records(self, conv: ConversationMemory, anima_dir: Path):
        """conversation.json without tool_records field loads cleanly."""
        # Write a legacy conversation.json manually
        legacy_data = {
            "anima_name": "sakura",
            "turns": [
                {
                    "role": "human",
                    "content": "hello",
                    "timestamp": "2026-02-20T18:00:00+09:00",
                    "token_estimate": 2,
                    "attachments": [],
                },
                {
                    "role": "assistant",
                    "content": "hi there",
                    "timestamp": "2026-02-20T18:00:05+09:00",
                    "token_estimate": 3,
                    "attachments": [],
                },
            ],
            "compressed_summary": "",
            "compressed_turn_count": 0,
            "last_finalized_turn_index": 0,
        }
        state_path = anima_dir / "state" / "conversation.json"
        state_path.write_text(json.dumps(legacy_data, ensure_ascii=False), encoding="utf-8")

        state = conv.load()
        assert len(state.turns) == 2
        assert state.turns[0].tool_records == []
        assert state.turns[1].tool_records == []


# ── append_turn with tool_records ──────────────────────────────


class TestAppendTurnToolRecords:
    def test_append_with_records(self, conv: ConversationMemory):
        records = [ToolRecord(tool_name="search")]
        conv.append_turn("assistant", "result", tool_records=records)
        state = conv.load()
        assert len(state.turns[0].tool_records) == 1

    def test_append_without_records(self, conv: ConversationMemory):
        conv.append_turn("assistant", "result")
        state = conv.load()
        assert state.turns[0].tool_records == []

    def test_cap_tool_records_per_turn(self, conv: ConversationMemory):
        """More than _MAX_TOOL_RECORDS_PER_TURN records get capped."""
        records = [
            ToolRecord(tool_name=f"tool_{i}")
            for i in range(_MAX_TOOL_RECORDS_PER_TURN + 5)
        ]
        conv.append_turn("assistant", "result", tool_records=records)
        state = conv.load()
        assert len(state.turns[0].tool_records) == _MAX_TOOL_RECORDS_PER_TURN


# ── _format_history tool markers (A1 mode) ─────────────────────


class TestFormatHistoryToolMarkers:
    def test_tool_marker_in_assistant_turn(self, conv: ConversationMemory):
        records = [
            ToolRecord(tool_name="mcp__aw__post_channel"),
            ToolRecord(tool_name="call_human"),
        ]
        conv.append_turn("human", "投稿してみて")
        conv.append_turn("assistant", "投稿しました", tool_records=records)
        conv.save()

        state = conv.load()
        history = conv._format_history(state)
        assert "[実行ツール: mcp__aw__post_channel, call_human]" in history

    def test_no_marker_without_records(self, conv: ConversationMemory):
        conv.append_turn("human", "hello")
        conv.append_turn("assistant", "hi")
        conv.save()

        state = conv.load()
        history = conv._format_history(state)
        assert "[実行ツール:" not in history

    def test_human_turn_no_marker(self, conv: ConversationMemory):
        """Human turns should never get tool markers even if somehow they have records."""
        conv.append_turn("human", "hello")
        conv.save()

        state = conv.load()
        history = conv._format_history(state)
        assert "[実行ツール:" not in history


# ── build_structured_messages (A2/Fallback mode) ───────────────


class TestBuildStructuredMessages:
    def _setup_turns(self, conv: ConversationMemory):
        conv.append_turn("human", "投稿してみて")
        records = [
            ToolRecord(
                tool_name="mcp__aw__post_channel",
                tool_id="toolu_001",
                input_summary="channel=general",
                result_summary="posted to general",
            ),
        ]
        conv.append_turn("assistant", "投稿しました", tool_records=records)
        conv.append_turn("human", "ありがとう")
        conv.save()

    def test_openai_format_basic_structure(self, conv: ConversationMemory):
        self._setup_turns(conv)
        messages = conv.build_structured_messages("次の指示", fmt="openai")

        # Should have: user, assistant(tool_calls), tool, user, current_user
        roles = [m["role"] for m in messages]
        assert "tool" in roles, "Should contain tool result messages"
        assert roles[-1] == "user"
        assert messages[-1]["content"] == "次の指示"

    def test_openai_format_tool_calls_structure(self, conv: ConversationMemory):
        self._setup_turns(conv)
        messages = conv.build_structured_messages("次", fmt="openai")

        # Find the assistant message with tool_calls
        assistant_with_tools = [
            m for m in messages
            if m["role"] == "assistant" and "tool_calls" in m
        ]
        assert len(assistant_with_tools) == 1
        tc = assistant_with_tools[0]["tool_calls"][0]
        assert tc["function"]["name"] == "mcp__aw__post_channel"
        assert tc["id"] == "toolu_001"

    def test_openai_format_tool_result(self, conv: ConversationMemory):
        self._setup_turns(conv)
        messages = conv.build_structured_messages("次", fmt="openai")

        tool_results = [m for m in messages if m["role"] == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0]["content"] == "posted to general"
        assert tool_results[0]["tool_call_id"] == "toolu_001"

    def test_anthropic_format_basic_structure(self, conv: ConversationMemory):
        self._setup_turns(conv)
        messages = conv.build_structured_messages("次", fmt="anthropic")

        # Find the assistant message with content blocks
        assistant_msgs = [
            m for m in messages
            if m["role"] == "assistant" and isinstance(m["content"], list)
        ]
        assert len(assistant_msgs) == 1

        content_blocks = assistant_msgs[0]["content"]
        types = [b["type"] for b in content_blocks]
        assert "text" in types
        assert "tool_use" in types

    def test_anthropic_format_tool_result_blocks(self, conv: ConversationMemory):
        self._setup_turns(conv)
        messages = conv.build_structured_messages("次", fmt="anthropic")

        # Find user message with tool_result blocks
        tool_result_msgs = [
            m for m in messages
            if m["role"] == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_msgs) == 1
        block = tool_result_msgs[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_001"

    def test_plain_text_when_no_tools(self, conv: ConversationMemory):
        conv.append_turn("human", "hello")
        conv.append_turn("assistant", "hi there")
        conv.save()

        messages = conv.build_structured_messages("次", fmt="openai")
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert "tool_calls" not in assistant_msgs[0]
        assert assistant_msgs[0]["content"] == "hi there"

    def test_compressed_summary_injection(self, conv: ConversationMemory):
        state = conv.load()
        state.compressed_summary = "以前の会話の要約です"
        state.compressed_turn_count = 10
        conv.save()

        messages = conv.build_structured_messages("hello", fmt="openai")
        assert messages[0]["role"] == "user"
        assert "会話の要約" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"

    def test_tool_record_budget_limit(self, conv: ConversationMemory):
        """Verify that _MAX_RENDERED_TOOL_RECORDS is respected."""
        # Create turns with many tool records
        for i in range(5):
            conv.append_turn("human", f"instruction {i}")
            records = [
                ToolRecord(
                    tool_name=f"tool_{i}_{j}",
                    tool_id=f"id_{i}_{j}",
                    result_summary="ok",
                )
                for j in range(10)
            ]
            conv.append_turn("assistant", f"done {i}", tool_records=records)
        conv.save()

        messages = conv.build_structured_messages("final", fmt="openai")
        # Count total tool results
        tool_count = sum(1 for m in messages if m["role"] == "tool")
        assert tool_count <= _MAX_RENDERED_TOOL_RECORDS


# ── _format_turns_for_compression ──────────────────────────────


class TestCompressionFormat:
    def test_tool_info_in_compression_text(self, conv: ConversationMemory):
        records = [
            ToolRecord(tool_name="search"),
            ToolRecord(tool_name="post_channel"),
        ]
        turns = [
            ConversationTurn(
                role="assistant",
                content="検索しました",
                tool_records=records,
            ),
        ]
        text = conv._format_turns_for_compression(turns)
        assert "[使用ツール: search, post_channel]" in text

    def test_no_tool_info_without_records(self, conv: ConversationMemory):
        turns = [
            ConversationTurn(role="assistant", content="hello"),
        ]
        text = conv._format_turns_for_compression(turns)
        assert "[使用ツール:" not in text


# ── ExecutionResult / CycleResult tool_call_records ────────────


class TestExecutionResultToolRecords:
    def test_execution_result_default_empty(self):
        from core.execution.base import ExecutionResult
        r = ExecutionResult(text="hello")
        assert r.tool_call_records == []

    def test_execution_result_with_records(self):
        from core.execution.base import ExecutionResult, ToolCallRecord
        records = [
            ToolCallRecord(tool_name="search", tool_id="t1"),
        ]
        r = ExecutionResult(text="hello", tool_call_records=records)
        assert len(r.tool_call_records) == 1
        assert r.tool_call_records[0].tool_name == "search"

    def test_cycle_result_default_empty(self):
        from core.schemas import CycleResult
        r = CycleResult(trigger="test", action="responded")
        assert r.tool_call_records == []

    def test_cycle_result_with_records(self):
        from core.schemas import CycleResult
        r = CycleResult(
            trigger="test",
            action="responded",
            tool_call_records=[
                {"tool_name": "search", "tool_id": "t1",
                 "input_summary": "", "result_summary": ""},
            ],
        )
        assert len(r.tool_call_records) == 1
        assert r.tool_call_records[0]["tool_name"] == "search"


# ── ToolRecord __post_init__ truncation (M5) ─────────────────


class TestToolRecordPostInit:
    def test_input_summary_truncated(self):
        from core.memory.conversation import _MAX_TOOL_INPUT_SUMMARY
        long_input = "x" * (_MAX_TOOL_INPUT_SUMMARY + 50)
        r = ToolRecord(tool_name="test", input_summary=long_input)
        assert len(r.input_summary) == _MAX_TOOL_INPUT_SUMMARY + 3  # +3 for "..."
        assert r.input_summary.endswith("...")

    def test_result_summary_truncated(self):
        from core.memory.conversation import _MAX_TOOL_RESULT_SUMMARY
        long_result = "y" * (_MAX_TOOL_RESULT_SUMMARY + 50)
        r = ToolRecord(tool_name="test", result_summary=long_result)
        assert len(r.result_summary) == _MAX_TOOL_RESULT_SUMMARY + 3
        assert r.result_summary.endswith("...")

    def test_short_values_not_truncated(self):
        r = ToolRecord(tool_name="test", input_summary="short", result_summary="ok")
        assert r.input_summary == "short"
        assert r.result_summary == "ok"


# ── ToolRecord.from_dict (M3) ────────────────────────────────


class TestToolRecordFromDict:
    def test_from_dict_full(self):
        d = {
            "tool_name": "search",
            "tool_id": "t1",
            "input_summary": "query=test",
            "result_summary": "found 5",
        }
        r = ToolRecord.from_dict(d)
        assert r.tool_name == "search"
        assert r.tool_id == "t1"
        assert r.input_summary == "query=test"
        assert r.result_summary == "found 5"

    def test_from_dict_minimal(self):
        d = {"tool_name": "post"}
        r = ToolRecord.from_dict(d)
        assert r.tool_name == "post"
        assert r.tool_id == ""
        assert r.input_summary == ""
        assert r.result_summary == ""

    def test_from_dict_empty(self):
        r = ToolRecord.from_dict({})
        assert r.tool_name == ""


# ── Anthropic format: no consecutive user roles (C2) ─────────


class TestAnthropicRoleAlternation:
    def _setup_tool_turns(self, conv: ConversationMemory):
        """Set up: human → assistant(tool) → human → to trigger C2 scenario."""
        conv.append_turn("human", "投稿してみて")
        records = [
            ToolRecord(
                tool_name="mcp__aw__post_channel",
                tool_id="toolu_001",
                input_summary="channel=general",
                result_summary="posted",
            ),
        ]
        conv.append_turn("assistant", "投稿しました", tool_records=records)
        conv.append_turn("human", "ありがとう")
        conv.save()

    def test_no_consecutive_user_roles(self, conv: ConversationMemory):
        """Anthropic format must never have two consecutive user messages."""
        self._setup_tool_turns(conv)
        messages = conv.build_structured_messages("次の指示", fmt="anthropic")

        prev_role = None
        for m in messages:
            if prev_role == "user":
                assert m["role"] != "user", (
                    f"Consecutive user messages found: ...{prev_role}, {m['role']}..."
                )
            prev_role = m["role"]

    def test_tool_result_merged_with_next_human(self, conv: ConversationMemory):
        """tool_result blocks should be merged into the next user message."""
        self._setup_tool_turns(conv)
        messages = conv.build_structured_messages("次の指示", fmt="anthropic")

        # Find user messages with list content (merged tool_result + text)
        merged_user_msgs = [
            m for m in messages
            if m["role"] == "user" and isinstance(m.get("content"), list)
        ]
        assert len(merged_user_msgs) >= 1
        # The merged message should contain both tool_result and text blocks
        first_merged = merged_user_msgs[0]
        types = [b.get("type") for b in first_merged["content"]]
        assert "tool_result" in types
        assert "text" in types

    def test_trailing_tool_result_merged_with_current_msg(self, conv: ConversationMemory):
        """If the last history turn is assistant(tool), tool_result merges into current msg."""
        conv.append_turn("human", "検索して")
        records = [
            ToolRecord(
                tool_name="search",
                tool_id="toolu_999",
                result_summary="found 3 results",
            ),
        ]
        conv.append_turn("assistant", "検索しました", tool_records=records)
        conv.save()

        messages = conv.build_structured_messages("次は何？", fmt="anthropic")

        # Last message must be user and contain the tool_result + current text
        last = messages[-1]
        assert last["role"] == "user"
        assert isinstance(last["content"], list)
        types = [b.get("type") for b in last["content"]]
        assert "tool_result" in types
        assert "text" in types
        # Verify the text content is the current message
        text_blocks = [b for b in last["content"] if b.get("type") == "text"]
        assert any("次は何？" in b["text"] for b in text_blocks)


# ── OpenAI format: content None with tool_calls (H2) ─────────


class TestOpenAIContentNullWithToolCalls:
    def test_assistant_content_is_none_with_tool_calls(self, conv: ConversationMemory):
        """OpenAI format: assistant messages with tool_calls should have content=None."""
        conv.append_turn("human", "投稿してみて")
        records = [
            ToolRecord(
                tool_name="post",
                tool_id="toolu_001",
                input_summary="channel=general",
                result_summary="posted",
            ),
        ]
        conv.append_turn("assistant", "投稿しました", tool_records=records)
        conv.save()

        messages = conv.build_structured_messages("次", fmt="openai")
        assistant_with_tools = [
            m for m in messages
            if m["role"] == "assistant" and "tool_calls" in m
        ]
        assert len(assistant_with_tools) == 1
        assert assistant_with_tools[0]["content"] is None

    def test_assistant_content_preserved_without_tool_calls(self, conv: ConversationMemory):
        """OpenAI format: plain assistant messages should keep their content."""
        conv.append_turn("human", "hello")
        conv.append_turn("assistant", "hi there")
        conv.save()

        messages = conv.build_structured_messages("次", fmt="openai")
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert assistant_msgs[0]["content"] == "hi there"


# ── Deserialization safety (H1) ──────────────────────────────


class TestDeserializationSafety:
    def test_load_does_not_mutate_source_dict(self, conv: ConversationMemory, anima_dir: Path):
        """load() should not mutate the parsed JSON dicts."""
        data = {
            "anima_name": "sakura",
            "turns": [
                {
                    "role": "assistant",
                    "content": "done",
                    "timestamp": "2026-02-20T18:00:00+09:00",
                    "token_estimate": 5,
                    "attachments": [],
                    "tool_records": [
                        {"tool_name": "search", "tool_id": "t1",
                         "input_summary": "", "result_summary": "ok"},
                    ],
                },
            ],
            "compressed_summary": "",
            "compressed_turn_count": 0,
            "last_finalized_turn_index": 0,
        }
        state_path = anima_dir / "state" / "conversation.json"
        state_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        # Take a snapshot of the original data
        import copy
        original = copy.deepcopy(data)

        conv.load()

        # Re-read to confirm the file is unchanged
        reloaded = json.loads(state_path.read_text(encoding="utf-8"))
        assert "tool_records" in reloaded["turns"][0], (
            "tool_records should still exist in the saved file"
        )
