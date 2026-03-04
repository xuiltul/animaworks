# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the S-mode memory encoding gap fix.

Verifies the complete pipelines:
1. S mode conversation -> compression -> compressed_summary
2. compressed_summary -> RAG index -> vector/keyword search
3. activity_log entries -> consolidation engine -> prompt template
4. B mode regression (conversation still works)

These tests exercise multiple components together to ensure the
three fixes (conversation save, RAG conversation_summary indexing,
activity_log consolidation input) integrate correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.memory.activity import ActivityLogger
from core.memory.consolidation import ConsolidationEngine
from core.memory.conversation import ConversationMemory, ConversationTurn
from core.schemas import ModelConfig
from tests.helpers.mocks import (
    make_litellm_response,
    patch_anthropic_compression,
    patch_litellm,
)

pytestmark = pytest.mark.e2e


# ── Test 1: S mode conversation -> compression -> summary ──────────


@pytest.mark.asyncio
class TestSModeConversationToCompressionPipeline:
    """Full pipeline: S mode message -> conversation.json turns ->
    compression -> compressed_summary updated.
    """

    async def test_s_mode_conversation_to_compression_pipeline(
        self, make_anima, data_dir,
    ):
        """S mode (claude-sonnet-*) conversation turns accumulate, trigger
        compression, and produce a compressed_summary in conversation.json.

        Pipeline:
        1. Create anima with S-mode model
        2. Populate conversation.json with 20+ turns (simulating Fix 1)
        3. Trigger compression via ConversationMemory
        4. Verify compressed_summary is populated and turns are reduced
        """
        anima_dir = make_anima(
            name="s-conv-compress",
            model="claude-sonnet-4-6",
            conversation_history_threshold=0.001,  # Very low to trigger compression
        )

        model_config = ModelConfig(
            model="claude-sonnet-4-6",
            conversation_history_threshold=0.001,
        )
        conv_mem = ConversationMemory(anima_dir, model_config)

        # Simulate Fix 1: S mode now saves turns to conversation.json.
        # Pre-populate with realistic conversation turns.
        state = conv_mem.load()
        for i in range(25):
            role = "human" if i % 2 == 0 else "assistant"
            if role == "human":
                content = f"Turn {i}: ユーザーからの質問 — Slack連携の設定方法を教えてください。返信不要の場合は省略してOKです。"
            else:
                content = (
                    f"Turn {i}: Slack連携の設定手順を説明します。"
                    f"config.jsonのexternal_messaging.slackセクションを編集し、"
                    f"Bot tokenを設定してください。" + "詳細は..." * 50
                )
            state.turns.append(
                ConversationTurn(role=role, content=content)
            )
        conv_mem.save()

        # Verify the turns were saved to disk
        conv_path = anima_dir / "state" / "conversation.json"
        assert conv_path.exists(), "conversation.json should exist after save"
        raw = json.loads(conv_path.read_text(encoding="utf-8"))
        assert len(raw["turns"]) == 25, "All 25 turns should be persisted"

        # Verify compression is needed
        assert conv_mem.needs_compression(), (
            "With 25 turns and threshold=0.001, compression should be triggered"
        )

        # Mock the LLM compression call and trigger compression
        summary_text = (
            "### Slack連携設定\n\n"
            "ユーザーがSlack連携の設定方法を質問。config.jsonの"
            "external_messaging.slackセクションの編集手順を説明。"
            "Bot tokenの設定が必要。返信不要の場合は省略可。\n\n"
            "### 追加情報\n\n"
            "25ターンにわたる会話で、設定の詳細を段階的に説明した。"
        )
        with patch_anthropic_compression(summary_text=summary_text):
            compressed = await conv_mem.compress_if_needed()

        assert compressed is True, "Compression should have been performed"

        # Reload from disk to verify persistence
        fresh_conv = ConversationMemory(anima_dir, model_config)
        fresh_state = fresh_conv.load()

        assert fresh_state.compressed_summary, (
            "compressed_summary should be populated after compression"
        )
        assert "Slack連携" in fresh_state.compressed_summary, (
            "Summary should contain key topics from the conversation"
        )
        assert fresh_state.compressed_turn_count > 0, (
            "compressed_turn_count should reflect the number of compressed turns"
        )
        assert len(fresh_state.turns) < 25, (
            "Remaining turns should be fewer than original 25"
        )

    async def test_incremental_compression_preserves_summary(
        self, make_anima, data_dir,
    ):
        """When compression runs again on new turns, the existing
        compressed_summary is passed to the LLM for merging.
        """
        anima_dir = make_anima(
            name="s-incr-compress",
            model="claude-sonnet-4-6",
            conversation_history_threshold=0.001,
        )

        model_config = ModelConfig(
            model="claude-sonnet-4-6",
            conversation_history_threshold=0.001,
        )
        conv_mem = ConversationMemory(anima_dir, model_config)

        # Set up initial compressed state
        state = conv_mem.load()
        state.compressed_summary = "前回の要約: Slack設定について議論。"
        state.compressed_turn_count = 10

        # Add new turns to trigger another compression
        for i in range(20):
            role = "human" if i % 2 == 0 else "assistant"
            state.turns.append(
                ConversationTurn(
                    role=role,
                    content=f"Turn {i}: 追加の会話内容。" + "x" * 300,
                )
            )
        conv_mem.save()

        merged_summary = (
            "前回の要約を含む統合サマリー: "
            "Slack設定と追加の議論を統合しました。"
        )
        with patch_anthropic_compression(summary_text=merged_summary):
            compressed = await conv_mem.compress_if_needed()

        assert compressed is True
        fresh = ConversationMemory(anima_dir, model_config)
        state = fresh.load()
        assert "統合" in state.compressed_summary, (
            "Summary should reflect merged content"
        )
        assert state.compressed_turn_count > 10, (
            "Turn count should include both old and new compressed turns"
        )


# ── Test 2: compressed_summary -> RAG index -> search ──────────────


class TestCompressedSummaryToRAGSearchPipeline:
    """Full pipeline: compressed_summary -> RAG index ->
    vector/keyword search finds it.
    """

    def test_compressed_summary_keyword_search(self, tmp_path, monkeypatch):
        """Keyword search via RAGMemorySearch finds text in compressed_summary.

        Pipeline:
        1. Create conversation.json with compressed_summary
        2. Use RAGMemorySearch.search_memory_text with keyword
        3. Verify matching lines are returned from conversation_summary
        """
        # Set up isolated data directory
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        (data_dir / "models").mkdir()
        (data_dir / "shared" / "users").mkdir(parents=True)
        (data_dir / "common_skills").mkdir()
        (data_dir / "common_knowledge").mkdir()
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

        # Create anima directory with conversation.json
        anima_dir = data_dir / "animas" / "test-rag-summary"
        anima_dir.mkdir(parents=True)
        for sub in ("knowledge", "episodes", "procedures", "skills", "state"):
            (anima_dir / sub).mkdir()

        # Write conversation.json with a realistic compressed_summary
        conv_data = {
            "anima_name": "test-rag-summary",
            "turns": [],
            "compressed_summary": (
                "### Slack連携設定\n\n"
                "ユーザーがSlack連携の設定方法を質問。返信不要の場合は省略可能と説明。\n\n"
                "### デプロイ手順\n\n"
                "AWS環境へのデプロイ手順を議論。terraformの設定を確認した。"
            ),
            "compressed_turn_count": 15,
            "last_finalized_turn_index": 0,
        }
        (anima_dir / "state" / "conversation.json").write_text(
            json.dumps(conv_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        from core.memory.rag_search import RAGMemorySearch

        rag_search = RAGMemorySearch(
            anima_dir=anima_dir,
            common_knowledge_dir=data_dir / "common_knowledge",
            common_skills_dir=data_dir / "common_skills",
        )

        # Search for known text in compressed_summary
        results = rag_search.search_memory_text(
            "返信不要",
            scope="all",
            knowledge_dir=anima_dir / "knowledge",
            episodes_dir=anima_dir / "episodes",
            procedures_dir=anima_dir / "procedures",
            common_knowledge_dir=data_dir / "common_knowledge",
        )

        # Verify keyword search found matching content
        summary_results = [
            (fname, line) for fname, line in results
            if fname == "conversation_summary"
        ]
        assert len(summary_results) > 0, (
            "Keyword search should find '返信不要' in conversation_summary"
        )
        assert any(
            "返信不要" in line for _, line in summary_results
        ), "Matched lines should contain the search term"

    def test_compressed_summary_rag_vector_indexing(self, tmp_path, monkeypatch):
        """RAG vector indexing includes conversation_summary collection.

        Pipeline:
        1. Create conversation.json with compressed_summary
        2. Use MemoryIndexer.index_conversation_summary
        3. Verify chunks are created in {prefix}_conversation_summary collection
        """
        chromadb = pytest.importorskip(
            "chromadb",
            reason="ChromaDB not installed",
        )
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed",
        )

        # Set up isolated data directory
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        (data_dir / "models").mkdir()
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

        anima_dir = data_dir / "animas" / "test-rag-vec"
        anima_dir.mkdir(parents=True)
        for sub in ("knowledge", "episodes", "procedures", "skills", "state", "vectordb"):
            (anima_dir / sub).mkdir()

        # Write conversation.json with compressed_summary
        conv_data = {
            "anima_name": "test-rag-vec",
            "turns": [],
            "compressed_summary": (
                "### プロジェクト管理\n\n"
                "Jiraチケットの管理方法について議論。スプリントプランニングの"
                "ベストプラクティスを共有した。\n\n"
                "### コードレビュー\n\n"
                "プルリクエストのレビュー基準を確認。テストカバレッジ80%以上を"
                "目標として設定した。"
            ),
            "compressed_turn_count": 20,
            "last_finalized_turn_index": 0,
        }
        state_dir = anima_dir / "state"
        (state_dir / "conversation.json").write_text(
            json.dumps(conv_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        from core.memory.rag.indexer import MemoryIndexer
        from core.memory.rag.store import ChromaVectorStore

        vector_store = ChromaVectorStore(persist_dir=anima_dir / "vectordb")
        indexer = MemoryIndexer(vector_store, "test-rag-vec", anima_dir)

        # Index the conversation summary
        indexed_count = indexer.index_conversation_summary(
            state_dir, "test-rag-vec",
        )

        assert indexed_count > 0, (
            "index_conversation_summary should index at least one chunk"
        )

        # Verify the collection exists and contains documents
        collection_name = "test-rag-vec_conversation_summary"
        query_embedding = indexer.embedding_model.encode(
            "プロジェクト管理 Jira スプリント",
            convert_to_numpy=True,
        ).tolist()
        results = vector_store.query(
            collection_name,
            query_embedding,
            top_k=5,
        )

        assert len(results) > 0, (
            "Vector search should find documents in conversation_summary collection"
        )
        # Verify metadata
        for sr in results:
            assert sr.document.metadata.get("memory_type") == "conversation_summary"
            assert sr.document.metadata.get("source") == "conversation_gist"
            assert sr.document.metadata.get("source_file") == "state/conversation.json"


# ── Test 3: activity_log -> consolidation prompt ───────────────────


class TestActivityLogToConsolidationPromptPipeline:
    """Full pipeline: activity_log entries -> consolidation engine collects ->
    prompt template includes them.
    """

    def test_activity_log_to_consolidation_collection(self, tmp_path):
        """Activity log entries are collected by ConsolidationEngine.

        Pipeline:
        1. Create activity_log entries (response_sent, tool_use, message_received)
        2. Use ConsolidationEngine._collect_activity_entries(hours=24)
        3. Verify output contains the test entries
        """
        anima_dir = tmp_path / "animas" / "test-activity-consol"
        anima_dir.mkdir(parents=True)
        (anima_dir / "episodes").mkdir()
        (anima_dir / "knowledge").mkdir()

        # Write activity log entries
        activity = ActivityLogger(anima_dir)
        activity.log(
            "message_received",
            content="Slack連携について教えてください",
            from_person="admin",
            summary="Slack連携の質問",
        )
        activity.log(
            "response_sent",
            content="config.jsonのslackセクションを編集してください。Bot tokenが必要です。",
            to_person="admin",
            summary="Slack設定手順を回答",
        )
        activity.log(
            "tool_use",
            tool="search_memory",
            summary="query=slack連携 設定",
        )

        # Collect via ConsolidationEngine
        engine = ConsolidationEngine(anima_dir, "test-activity-consol")
        result = engine._collect_activity_entries(hours=24)

        assert result, "Activity log collection should return non-empty string"
        assert "message_received" in result, (
            "Collected output should contain message_received entries"
        )
        assert "response_sent" in result, (
            "Collected output should contain response_sent entries"
        )
        assert "tool_use" in result, (
            "Collected output should contain tool_use entries"
        )
        assert "Slack" in result or "slack" in result, (
            "Collected output should contain content from the entries"
        )

    def test_consolidation_prompt_has_activity_log_placeholder(self):
        """The consolidation instruction template contains the
        {activity_log_summary} placeholder.
        """
        from core.paths import load_prompt

        # Load the template with dummy values to verify it renders
        prompt = load_prompt(
            "memory/consolidation_instruction",
            anima_name="test-anima",
            episodes_summary="(テストエピソード)",
            resolved_events_summary="",
            activity_log_summary="[10:00] response_sent: テスト回答",
        )

        assert "アクティビティログ" in prompt, (
            "Template should contain the activity log section header"
        )
        assert "テスト回答" in prompt, (
            "Template should render the activity_log_summary content"
        )

    def test_activity_log_to_consolidation_prompt_render(self, tmp_path):
        """End-to-end: activity log -> collect -> render into prompt template.

        Pipeline:
        1. Write activity log entries
        2. Collect via ConsolidationEngine
        3. Render into consolidation_instruction template
        4. Verify the prompt contains the activity data
        """
        anima_dir = tmp_path / "animas" / "test-prompt-render"
        anima_dir.mkdir(parents=True)
        (anima_dir / "episodes").mkdir()
        (anima_dir / "knowledge").mkdir()

        # Write realistic activity entries
        activity = ActivityLogger(anima_dir)
        activity.log(
            "message_received",
            content="AWSのEC2インスタンスを立ち上げてほしい",
            from_person="manager",
            summary="EC2起動リクエスト",
        )
        activity.log(
            "tool_use",
            tool="aws_collector",
            summary="EC2 describe-instances実行",
        )
        activity.log(
            "response_sent",
            content="EC2インスタンスi-12345を起動しました。",
            to_person="manager",
            summary="EC2起動完了を報告",
        )

        # Collect activity entries
        engine = ConsolidationEngine(anima_dir, "test-prompt-render")
        activity_summary = engine._collect_activity_entries(hours=24)
        assert activity_summary, "Collection should return non-empty string"

        # Render into the full consolidation prompt
        from core.paths import load_prompt

        prompt = load_prompt(
            "memory/consolidation_instruction",
            anima_name="test-prompt-render",
            episodes_summary="(テストエピソード)",
            resolved_events_summary="",
            activity_log_summary=activity_summary,
        )

        assert "test-prompt-render" in prompt, (
            "Prompt should contain the anima name"
        )
        assert "EC2" in prompt or "ec2" in prompt.lower(), (
            "Prompt should contain activity content about EC2"
        )
        assert "aws_collector" in prompt, (
            "Prompt should contain tool usage from activity log"
        )

    def test_empty_activity_log_produces_empty_string(self, tmp_path):
        """When no activity log exists, _collect_activity_entries returns ''."""
        anima_dir = tmp_path / "animas" / "test-empty-activity"
        anima_dir.mkdir(parents=True)
        (anima_dir / "episodes").mkdir()
        (anima_dir / "knowledge").mkdir()

        engine = ConsolidationEngine(anima_dir, "test-empty-activity")
        result = engine._collect_activity_entries(hours=24)

        assert result == "", (
            "Empty activity log should produce empty string"
        )


# ── Test 4: B mode regression ──────────────────────────────────────


@pytest.mark.asyncio
class TestBModeConversationRegression:
    """Regression test: B mode conversation still saves turns correctly."""

    async def test_b_mode_conversation_turns_saved(self, make_digital_anima):
        """B mode (ollama/*) process_message saves conversation turns.

        This verifies no regression: Mode B always saved turns,
        and it should continue to do so after the S-mode fix.
        """
        dp = make_digital_anima(
            name="b-regression",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )

        main_resp = make_litellm_response(
            content="B mode response: 了解しました。タスクを実行します。",
        )

        # Multiple components may call litellm; provide enough main responses
        with patch_litellm(main_resp, main_resp, main_resp, main_resp, main_resp):
            await dp.process_message(
                "テストメッセージ",
                from_person="human",
            )

        # Verify conversation.json has turns
        conv_path = dp.anima_dir / "state" / "conversation.json"
        assert conv_path.exists(), (
            "conversation.json should exist after B mode message"
        )

        data = json.loads(conv_path.read_text(encoding="utf-8"))
        turns = data.get("turns", [])

        assert len(turns) >= 2, (
            "Should have at least 2 turns (human + assistant)"
        )
        assert turns[0]["role"] == "human", "First turn should be human"
        assert turns[1]["role"] == "assistant", "Second turn should be assistant"
        assert "テストメッセージ" in turns[0]["content"], (
            "Human turn should contain the original message"
        )
        assert "B mode response" in turns[1]["content"], (
            "Assistant turn should contain the LLM response"
        )

    async def test_b_mode_multiple_messages_accumulate(self, make_digital_anima):
        """Multiple B mode messages accumulate turns in conversation.json."""
        dp = make_digital_anima(
            name="b-multi-msg",
            model="ollama/gemma3:27b",
            execution_mode="assisted",
        )

        for i in range(3):
            main_resp = make_litellm_response(
                content=f"Response {i}: 回答します。",
            )

            with patch_litellm(main_resp):
                await dp.process_message(
                    f"Message {i}",
                    from_person="human",
                )

        conv_path = dp.anima_dir / "state" / "conversation.json"
        data = json.loads(conv_path.read_text(encoding="utf-8"))
        turns = data.get("turns", [])

        # 3 messages * 2 turns each = 6 turns
        assert len(turns) == 6, (
            f"Expected 6 turns (3 human + 3 assistant), got {len(turns)}"
        )
