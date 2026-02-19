#!/usr/bin/env python3
from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Demo script for RAG dense vector search system.

This script demonstrates:
1. Creating sample memory files
2. Indexing them into ChromaDB
3. Performing dense vector search with temporal decay
"""

import asyncio
import logging
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("demo_rag")


def create_sample_memories(anima_dir: Path) -> None:
    """Create sample memory files for demonstration."""
    logger.info("Creating sample memory files...")

    knowledge_dir = anima_dir / "knowledge"
    episodes_dir = anima_dir / "episodes"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # Sample knowledge files
    (knowledge_dir / "chatwork-policy.md").write_text(
        """# Chatwork対応方針

## 基本方針
Chatwork経由の依頼には迅速かつ丁寧に対応する。

## 緊急対応基準
[IMPORTANT] 山田さんからのメッセージには最優先で対応すること。
特に「至急」「緊急」のキーワードがある場合は即座に反応する。

## 通常対応
その他のメンバーからの依頼は営業時間内に対応する。
レスポンスタイムの目標: 30分以内。
""",
        encoding="utf-8",
    )

    (knowledge_dir / "slack-integration.md").write_text(
        """# Slack連携設定

## 通知チャンネル
- #general: 一般的な連絡事項
- #alerts: システムアラート（重要度: 高）
- #dev: 開発関連の議論

## ボット設定
AnimaWorksボットは以下の機能を提供:
- タスク進捗の自動通知
- デプロイ完了の報告
- エラーアラートの転送
""",
        encoding="utf-8",
    )

    (knowledge_dir / "project-guidelines.md").write_text(
        """# プロジェクト運用指針

## コードレビュー
すべてのPRは最低1名のレビューが必要。
レビュー期限: 提出後24時間以内。

## デプロイ手順
1. ステージング環境でテスト
2. レビュー承認取得
3. 本番環境へデプロイ
4. 動作確認とモニタリング

## 緊急対応
本番環境で問題が発生した場合、直ちに山田さんに連絡する。
""",
        encoding="utf-8",
    )

    # Sample episode
    (episodes_dir / "2026-02-14.md").write_text(
        """# 2026-02-14 行動ログ

## 09:30 — 山田さんから見積もり依頼

Chatworkで新規プロジェクトの見積もり依頼を受信。
要件: Webサイトリニューアル、納期は3月末。

## 11:00 — 要件ヒアリング完了

Zoomで詳細ヒアリングを実施。
主な要望: レスポンシブデザイン、SEO対策、管理画面。

## 14:30 — 見積書作成

工数を積算し、見積書を作成。
総額: 300万円、期間: 2ヶ月。

## 16:00 — 見積書送付

Chatworkで見積書を送付。山田さんから「検討します」との返信。
""",
        encoding="utf-8",
    )

    logger.info("Sample memory files created")


def demo_indexing(anima_dir: Path) -> None:
    """Demonstrate memory indexing."""
    logger.info("=" * 70)
    logger.info("STEP 1: Indexing Memory Files")
    logger.info("=" * 70)

    try:
        from core.memory.rag import MemoryIndexer
        from core.memory.rag.store import ChromaVectorStore
    except ImportError as e:
        logger.error("RAG dependencies not installed: %s", e)
        logger.error("Run: pip install 'animaworks[rag]'")
        return

    # Initialize vector store
    vectordb_dir = anima_dir / "vectordb"
    vectordb_dir.mkdir(parents=True, exist_ok=True)
    vector_store = ChromaVectorStore(persist_dir=vectordb_dir)

    # Initialize indexer
    indexer = MemoryIndexer(vector_store, "demo_anima", anima_dir)

    # Index knowledge files
    logger.info("Indexing knowledge files...")
    knowledge_dir = anima_dir / "knowledge"
    chunks = indexer.index_directory(knowledge_dir, "knowledge")
    logger.info("  Indexed %d chunks from knowledge/", chunks)

    # Index episodes
    logger.info("Indexing episode files...")
    episodes_dir = anima_dir / "episodes"
    chunks = indexer.index_directory(episodes_dir, "episodes")
    logger.info("  Indexed %d chunks from episodes/", chunks)

    logger.info("Indexing complete!")
    return vector_store, indexer


def demo_vector_search(vector_store, indexer, knowledge_dir: Path, query: str) -> None:
    """Demonstrate dense vector search with temporal decay."""
    logger.info("")
    logger.info("Dense Vector Search (semantic similarity + temporal decay):")
    logger.info("  Query: %s", query)

    from core.memory.rag.retriever import MemoryRetriever

    retriever = MemoryRetriever(vector_store, indexer, knowledge_dir)

    results = retriever.search(
        query=query,
        anima_name="demo_anima",
        memory_type="knowledge",
        top_k=3,
    )

    logger.info("  Results:")
    for i, result in enumerate(results, 1):
        logger.info("    %d. %s (score: %.3f)", i, result.doc_id, result.score)
        logger.info("       - Vector: %.3f, Recency: %.3f",
                    result.source_scores.get("vector", 0),
                    result.source_scores.get("recency", 0))


async def demo_priming_integration(anima_dir: Path) -> None:
    """Demonstrate priming layer with vector search."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: Priming Layer Integration")
    logger.info("=" * 70)

    from core.memory.priming import PrimingEngine

    engine = PrimingEngine(anima_dir)

    test_messages = [
        "山田さんからChatworkで至急の依頼が来ました",
        "Slackのアラート設定を確認したい",
        "新規プロジェクトの見積もりについて",
    ]

    for message in test_messages:
        logger.info("")
        logger.info("Message: %s", message)

        result = await engine.prime_memories(message, sender_name="yamada")

        logger.info("  Primed memories:")
        logger.info("    - Sender profile: %d chars", len(result.sender_profile))
        logger.info("    - Recent episodes: %d chars", len(result.recent_episodes))
        logger.info("    - Related knowledge: %d chars", len(result.related_knowledge))
        logger.info("    - Matched skills: %s", result.matched_skills)

        if result.related_knowledge:
            logger.info("  Related knowledge preview:")
            preview = result.related_knowledge[:200].replace("\n", " ")
            logger.info("    %s...", preview)


def main() -> None:
    """Run the RAG demo."""
    logger.info("=" * 70)
    logger.info("AnimaWorks RAG Dense Vector Search Demo")
    logger.info("=" * 70)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        anima_dir = Path(tmpdir) / "demo_anima"
        anima_dir.mkdir()

        # Step 1: Create sample memories
        create_sample_memories(anima_dir)

        # Step 2: Index memories
        components = demo_indexing(anima_dir)
        if not components:
            return

        vector_store, indexer = components
        knowledge_dir = anima_dir / "knowledge"

        # Step 3: Demonstrate vector search
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 2: Dense Vector Search")
        logger.info("=" * 70)

        queries = [
            "山田さんからの緊急依頼への対応方法",
            "Slack通知設定",
            "デプロイ手順",
        ]

        for query in queries:
            logger.info("")
            logger.info("-" * 70)
            logger.info("Query: %s", query)
            logger.info("-" * 70)

            demo_vector_search(vector_store, indexer, knowledge_dir, query)

        # Step 4: Priming integration
        asyncio.run(demo_priming_integration(anima_dir))

        logger.info("")
        logger.info("=" * 70)
        logger.info("Demo complete!")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
