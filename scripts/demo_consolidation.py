#!/usr/bin/env python3
from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Demo script for memory consolidation.

Creates sample episodes and demonstrates the daily consolidation process.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main():
    """Run consolidation demo."""
    from core.memory.consolidation import ConsolidationEngine
    from core.paths import get_data_dir

    print("=== AnimaWorks Memory Consolidation Demo ===\n")

    # Get data directory
    data_dir = get_data_dir()
    animas_dir = data_dir / "animas"

    if not animas_dir.exists():
        print(f"Error: No animas directory found at {animas_dir}")
        print("Please run 'animaworks init' first.")
        return

    # List available animas
    anima_dirs = [d for d in animas_dir.iterdir() if d.is_dir()]

    if not anima_dirs:
        print("Error: No animas found.")
        print("Please create an anima first with 'animaworks anima create <name>'")
        return

    print("Available animas:")
    for i, anima_dir in enumerate(anima_dirs):
        print(f"  {i + 1}. {anima_dir.name}")

    # Use first anima for demo
    anima_dir = anima_dirs[0]
    anima_name = anima_dir.name

    print(f"\nUsing anima: {anima_name}")

    # Create sample episode if none exist
    episodes_dir = anima_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().date()
    episode_file = episodes_dir / f"{today}.md"

    if not episode_file.exists() or episode_file.stat().st_size < 100:
        print("\nCreating sample episode data...")

        sample_episode = f"""# エピソード記憶 — {today}

## 09:00 — 朝のミーティング

**相手**: 山田
**トピック**: プロジェクト進捗確認
**要点**:
- プライミングレイヤーPhase 2の実装が完了した
- RAGハイブリッド検索が動作している
- 日次固定化の実装を進めている

**決定事項**: Phase 2を今日中に完成させる
**未解決**: Phase 3のスケジュール調整

## 14:30 — 技術調査

**相手**: システム
**トピック**: ChromaDB性能検証
**要点**:
- ベクトル検索のレスポンスタイムは平均50ms
- インデックスサイズは想定内
- メモリ使用量も問題なし

## 16:00 — コードレビュー

**相手**: チームメンバー
**トピック**: consolidation.py実装レビュー
**要点**:
- LLMプロンプトの精度を上げる必要がある
- エラーハンドリングを追加すべき
- ログ出力を改善する

**決定事項**: プロンプトテンプレートを改善する
"""

        episode_file.write_text(sample_episode, encoding="utf-8")
        print(f"Sample episode created: {episode_file}")

    print(f"\nEpisode file: {episode_file}")

    # Create consolidation engine
    print(f"\nInitializing ConsolidationEngine for {anima_name}...")
    engine = ConsolidationEngine(
        anima_dir=anima_dir,
        anima_name=anima_name,
    )

    # Check episode count
    entries = engine._collect_recent_episodes(hours=24)
    print(f"Found {len(entries)} episode entries in the past 24 hours")

    if len(entries) == 0:
        print("\nNo recent episodes found. Please add some episodes first.")
        return

    print("\nEpisode summaries:")
    for i, entry in enumerate(entries[:3], 1):  # Show first 3
        print(f"  {i}. {entry['date']} {entry['time']}")
        # Show first line of content
        first_line = entry['content'].split('\n')[0][:60]
        print(f"     {first_line}...")

    # Ask user if they want to proceed
    print("\n" + "=" * 60)
    response = input("Run daily consolidation? This will call the LLM API. [y/N]: ")

    if response.lower() != 'y':
        print("Consolidation cancelled.")
        return

    # Run consolidation
    print("\nRunning daily consolidation...")
    print("(This will take a few seconds...)")

    try:
        result = await engine.daily_consolidate(
            model="anthropic/claude-sonnet-4-20250514",
            min_episodes=1,
        )

        print("\n=== Consolidation Results ===")
        print(f"Skipped: {result['skipped']}")
        print(f"Episodes processed: {result['episodes_processed']}")
        print(f"Knowledge files created: {len(result['knowledge_files_created'])}")
        print(f"Knowledge files updated: {len(result['knowledge_files_updated'])}")

        if result['knowledge_files_created']:
            print("\nNewly created knowledge files:")
            for filename in result['knowledge_files_created']:
                filepath = anima_dir / "knowledge" / filename
                print(f"\n  {filename}")
                print("  " + "-" * 40)
                if filepath.exists():
                    content = filepath.read_text(encoding="utf-8")
                    # Show first 300 chars
                    preview = content[:300]
                    if len(content) > 300:
                        preview += "..."
                    print("  " + preview.replace("\n", "\n  "))

        if result['knowledge_files_updated']:
            print("\nUpdated knowledge files:")
            for filename in result['knowledge_files_updated']:
                print(f"  - {filename}")

    except Exception as e:
        print(f"\nError during consolidation: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n=== Demo Complete ===")
    print(f"\nYou can view the knowledge files at:")
    print(f"  {anima_dir / 'knowledge'}")


if __name__ == "__main__":
    asyncio.run(main())
