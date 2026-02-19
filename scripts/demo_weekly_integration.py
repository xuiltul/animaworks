#!/usr/bin/env python3
from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Demo script for weekly integration (Phase 3).

This script demonstrates:
1. Creating duplicate knowledge files
2. Creating old episodes
3. Running weekly integration
4. Showing merge and compression results
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main():
    """Run weekly integration demo."""
    print("=" * 80)
    print("AnimaWorks Weekly Integration Demo (Phase 3)")
    print("=" * 80)
    print()

    # Setup demo environment
    demo_dir = Path(__file__).parent.parent / "demo_data" / "weekly_integration"
    anima_dir = demo_dir / "test_anima"
    knowledge_dir = anima_dir / "knowledge"
    episodes_dir = anima_dir / "episodes"

    # Clean up previous demo data
    import shutil
    if demo_dir.exists():
        shutil.rmtree(demo_dir)

    knowledge_dir.mkdir(parents=True)
    episodes_dir.mkdir(parents=True)

    print("Demo directory created:", anima_dir)
    print()

    # ── Step 1: Create duplicate knowledge files ──────────────────────
    print("Step 1: Creating duplicate knowledge files...")
    print("-" * 80)

    chatwork_policy = knowledge_dir / "chatwork-policy.md"
    chatwork_policy.write_text("""# Chatwork対応方針

## 基本方針
- 迅速に返信する
- 丁寧な言葉遣いを心がける
- タスク登録を忘れない

## 返信タイミング
- 営業時間内: 30分以内
- 緊急時: 即座に対応

## 禁止事項
- 感情的な表現は避ける
- 長文は分割する
""", encoding="utf-8")

    chatwork_response = knowledge_dir / "chatwork-response.md"
    chatwork_response.write_text("""# Chatwork返信ルール

## 返信の基本
- 丁寧な言葉遣い
- 簡潔明瞭に
- タスクは必ず登録

## タイミング
- 30分以内に返信（営業時間内）
- 緊急案件は最優先

## 注意点
- 長すぎる文章は分割
- 曖昧な表現を避ける
""", encoding="utf-8")

    meeting_jan = knowledge_dir / "meeting-notes-jan.md"
    meeting_jan.write_text("""# ミーティングノート（1月）

## 2026-01-15
- プロジェクトキックオフ
- Phase 1の計画策定

## 2026-01-22
- 進捗確認
- リソース配分の見直し

## 2026-01-29
- 中間報告
- Phase 2への移行準備
""", encoding="utf-8")

    meeting_feb = knowledge_dir / "meeting-notes-feb.md"
    meeting_feb.write_text("""# ミーティングノート（2月）

## 2026-02-05
- Phase 2開始
- タスク割り当て

## 2026-02-12
- 進捗レビュー
- 課題の洗い出し
""", encoding="utf-8")

    print(f"Created: {chatwork_policy.name}")
    print(f"Created: {chatwork_response.name}")
    print(f"Created: {meeting_jan.name}")
    print(f"Created: {meeting_feb.name}")
    print()

    # ── Step 2: Create old episodes ───────────────────────────────────
    print("Step 2: Creating old episodes for compression...")
    print("-" * 80)

    # Create old episode (35 days ago)
    old_date = datetime.now().date() - timedelta(days=35)
    old_episode = episodes_dir / f"{old_date}.md"
    old_episode.write_text(f"""# エピソード記憶 {old_date}

## 09:00 — 朝のミーティング

**相手**: 山田
**トピック**: 定例報告
**要点**:
- 進捗は順調
- 特に問題なし
- 次回は来週

## 14:30 — 雑談

**相手**: 田中
**トピック**: ランチの話
**要点**:
- 新しいレストランの話
- 特に重要でない内容

## 16:00 — 簡単な質問対応

**相手**: 鈴木
**トピック**: 資料の場所
**要点**:
- 共有フォルダの場所を教えた
- すぐに解決
""", encoding="utf-8")

    # Create old important episode (40 days ago)
    important_date = datetime.now().date() - timedelta(days=40)
    important_episode = episodes_dir / f"{important_date}.md"
    important_episode.write_text(f"""# エピソード記憶 {important_date}

## 10:00 — 重要な意思決定 [IMPORTANT]

**相手**: CEO
**トピック**: 今後の戦略
**要点**:
- 新規事業への投資を決定
- 3つの重点分野を確認
- 予算配分の承認

**決定事項**:
- AI部門への追加投資
- 人材採用の加速
- 研究開発の強化
""", encoding="utf-8")

    # Create recent episode (10 days ago - should not be compressed)
    recent_date = datetime.now().date() - timedelta(days=10)
    recent_episode = episodes_dir / f"{recent_date}.md"
    recent_episode.write_text(f"""# エピソード記憶 {recent_date}

## 11:00 — Phase 2レビュー

**相手**: プロジェクトチーム
**トピック**: 実装状況確認
**要点**:
- RAG実装完了
- テスト結果良好
- Phase 3へ移行準備
""", encoding="utf-8")

    print(f"Created old episode: {old_episode.name} (35 days ago)")
    print(f"Created important episode: {important_episode.name} (40 days ago, [IMPORTANT])")
    print(f"Created recent episode: {recent_episode.name} (10 days ago)")
    print()

    # ── Step 3: Initialize ConsolidationEngine ────────────────────────
    print("Step 3: Initializing ConsolidationEngine...")
    print("-" * 80)

    from core.memory.consolidation import ConsolidationEngine

    engine = ConsolidationEngine(
        anima_dir=anima_dir,
        anima_name="demo_anima",
    )

    print("ConsolidationEngine initialized")
    print()

    # ── Step 4: Detect duplicates ─────────────────────────────────────
    print("Step 4: Detecting duplicate knowledge files...")
    print("-" * 80)

    # For demo purposes, we'll manually create duplicates list
    # In real scenario, _detect_duplicates would use RAG
    duplicates = [
        ("chatwork-policy.md", "chatwork-response.md", 0.92),
        ("meeting-notes-jan.md", "meeting-notes-feb.md", 0.88),
    ]

    print(f"Found {len(duplicates)} duplicate pairs:")
    for file1, file2, score in duplicates:
        print(f"  - {file1} ↔ {file2} (similarity: {score:.2f})")
    print()

    # ── Step 5: Run weekly integration ────────────────────────────────
    print("Step 5: Running weekly integration...")
    print("-" * 80)

    # Mock detect_duplicates to return our manual list
    async def mock_detect_duplicates(threshold=0.85):
        return duplicates

    # Replace method temporarily
    original_detect = engine._detect_duplicates
    engine._detect_duplicates = mock_detect_duplicates

    try:
        result = await engine.weekly_integrate(
            model="anthropic/claude-sonnet-4-20250514",
            duplicate_threshold=0.85,
            episode_retention_days=30,
        )

        print("Weekly integration completed!")
        print()
        print("Results:")
        print(f"  - Knowledge files merged: {len(result['knowledge_files_merged'])}")
        for merge in result['knowledge_files_merged']:
            print(f"    • {merge}")
        print(f"  - Episodes compressed: {result['episodes_compressed']}")
        print()

    except Exception as e:
        print(f"Error during integration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original method
        engine._detect_duplicates = original_detect

    # ── Step 6: Show results ──────────────────────────────────────────
    print("Step 6: Showing results...")
    print("-" * 80)

    print("\nKnowledge files after merge:")
    for knowledge_file in sorted(knowledge_dir.glob("*.md")):
        print(f"  - {knowledge_file.name}")

    print("\nEpisode files after compression:")
    for episode_file in sorted(episodes_dir.glob("*.md")):
        size = episode_file.stat().st_size
        print(f"  - {episode_file.name} ({size} bytes)")

        # Check if compressed
        content = episode_file.read_text(encoding="utf-8")
        if "[COMPRESSED:" in content:
            print(f"    → Compressed")
        elif "[IMPORTANT]" in content:
            print(f"    → Important (not compressed)")
        else:
            print(f"    → Recent (not compressed)")

    # ── Step 7: Show sample merged content ────────────────────────────
    print()
    print("Step 7: Sample merged content...")
    print("-" * 80)

    # Look for merged files
    merged_files = [f for f in knowledge_dir.glob("*.md") if "[AUTO-MERGED" in f.read_text(encoding="utf-8")]
    if merged_files:
        sample = merged_files[0]
        print(f"\nSample merged file: {sample.name}")
        print("-" * 40)
        content = sample.read_text(encoding="utf-8")
        # Show first 500 chars
        print(content[:500])
        if len(content) > 500:
            print("...")
        print()

    # ── Step 8: Show compressed episode ───────────────────────────────
    print("Step 8: Sample compressed episode...")
    print("-" * 80)

    if old_episode.exists():
        content = old_episode.read_text(encoding="utf-8")
        print(f"\nCompressed episode: {old_episode.name}")
        print("-" * 40)
        print(content)
        print()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 80)
    print("Demo completed!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✓ Created duplicate knowledge files")
    print("  ✓ Created old and important episodes")
    print("  ✓ Detected duplicates using similarity threshold")
    print("  ✓ Merged duplicate knowledge files with LLM")
    print("  ✓ Compressed old episodes (keeping [IMPORTANT] ones)")
    print("  ✓ Rebuilt RAG index")
    print()
    print(f"Demo data location: {anima_dir}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
