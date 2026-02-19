#!/usr/bin/env python3

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Demo script to show priming layer in action."""

import asyncio
import tempfile
from datetime import date
from pathlib import Path

from core.memory.priming import PrimingEngine, format_priming_section


async def demo_priming():
    """Demonstrate priming layer functionality."""
    print("=" * 70)
    print("AnimaWorks プライミングレイヤー Phase 1 デモ")
    print("=" * 70)
    print()

    # Create temporary anima directory with sample data
    with tempfile.TemporaryDirectory() as tmpdir:
        anima_dir = Path(tmpdir) / "animas" / "sakura"
        anima_dir.mkdir(parents=True)

        # Create memory directories
        (anima_dir / "episodes").mkdir()
        (anima_dir / "knowledge").mkdir()
        (anima_dir / "skills").mkdir()

        # Create shared directory for user profiles
        shared_dir = Path(tmpdir) / "shared"
        users_dir = shared_dir / "users"
        users_dir.mkdir(parents=True)

        # Populate sample data
        print("📝 サンプルデータを作成中...")
        print()

        # Episode
        today_episode = anima_dir / "episodes" / f"{date.today().isoformat()}.md"
        today_episode.write_text(
            f"# {date.today().isoformat()} 行動ログ\n\n"
            "## 09:00 — 朝のタスク確認\n\n"
            "**相手**: システム\n"
            "**トピック**: タスク管理\n"
            "**要点**:\n"
            "- 今日のタスクを確認した\n"
            "- プライミングレイヤーの実装を優先\n\n"
            "## 10:30 — 山田さんとのミーティング\n\n"
            "**相手**: 山田\n"
            "**トピック**: Phase 1 実装計画\n"
            "**要点**:\n"
            "- 4チャネル並列検索について議論\n"
            "- トークン予算は2000トークンで合意\n"
            "- BM25のみで開始することを決定\n\n"
            "## 14:00 — コーディング開始\n\n"
            "**相手**: システム\n"
            "**トピック**: 実装作業\n"
            "**要点**:\n"
            "- PrimingEngine クラスを実装\n"
            "- テストケースを作成\n",
            encoding="utf-8",
        )

        # Knowledge
        knowledge_file = anima_dir / "knowledge" / "priming-layer.md"
        knowledge_file.write_text(
            "# プライミングレイヤー設計\n\n"
            "## 概要\n\n"
            "自動想起メカニズムを実装する。\n"
            "人間の脳科学に基づく設計。\n\n"
            "## 技術仕様\n\n"
            "- BM25 + ベクトル検索のハイブリッド\n"
            "- Phase 1 では BM25 のみ実装\n"
            "- Phase 2 で ChromaDB 導入予定\n\n"
            "## トークン予算\n\n"
            "- 送信者プロファイル: 500トークン\n"
            "- 直近エピソード: 600トークン\n"
            "- 関連知識: 700トークン\n"
            "- スキルマッチ: 200トークン\n",
            encoding="utf-8",
        )

        # Skill
        skill_file = anima_dir / "skills" / "python_coding.md"
        skill_file.write_text(
            "# Python コーディングスキル\n\n"
            "## 概要\n"
            "Python でコードを実装する\n\n"
            "## 手順\n"
            "1. 型ヒントを付ける\n"
            "2. docstring を書く\n"
            "3. テストを書く\n",
            encoding="utf-8",
        )

        # User profile
        yamada_dir = users_dir / "山田"
        yamada_dir.mkdir()
        (yamada_dir / "index.md").write_text(
            "# 山田さんのプロファイル\n\n"
            "## 基本情報\n"
            "- プロジェクトマネージャー\n"
            "- 技術的な詳細を好む\n"
            "- AnimaWorks プロジェクトのリード\n\n"
            "## 重要な好み・傾向\n"
            "- 簡潔で正確な報告を好む\n"
            "- Slack での連絡を希望\n"
            "- 実装の進捗を重視\n\n"
            "## 注意事項\n"
            "- 朝のミーティングは10:00から\n"
            "- 金曜午後は予定を入れない\n",
            encoding="utf-8",
        )

        print("✅ サンプルデータ作成完了")
        print()

        # Initialize priming engine
        engine = PrimingEngine(anima_dir)

        # Mock get_shared_dir to use our temp directory
        from unittest.mock import patch

        # Demo 1: Message from 山田
        print("-" * 70)
        print("デモ 1: 山田さんからのメッセージ")
        print("-" * 70)
        print()
        print("📩 受信メッセージ: 「プライミングレイヤーの実装状況を教えてください」")
        print()

        with patch("core.paths.get_shared_dir", return_value=shared_dir):
            result1 = await engine.prime_memories(
                message="プライミングレイヤーの実装状況を教えてください",
                sender_name="山田",
            )

        print("🧠 プライミング結果:")
        print()
        print(format_priming_section(result1, "山田"))
        print()
        print(f"📊 統計:")
        print(f"  - 送信者プロファイル: {len(result1.sender_profile)} 文字")
        print(f"  - 直近エピソード: {len(result1.recent_episodes)} 文字")
        print(f"  - 関連知識: {len(result1.related_knowledge)} 文字")
        print(f"  - スキルマッチ: {len(result1.matched_skills)} 件")
        print(f"  - 推定トークン数: {result1.estimated_tokens()} トークン")
        print()

        # Demo 2: Coding request
        print("-" * 70)
        print("デモ 2: コーディング依頼")
        print("-" * 70)
        print()
        print("📩 受信メッセージ: 「python でテストコードを書いてください」")
        print()

        result2 = await engine.prime_memories(
            message="python でテストコードを書いてください",
            sender_name="human",
        )

        print("🧠 プライミング結果:")
        print()
        print(format_priming_section(result2, "human"))
        print()
        print(f"📊 統計:")
        print(f"  - 送信者プロファイル: {len(result2.sender_profile)} 文字")
        print(f"  - 直近エピソード: {len(result2.recent_episodes)} 文字")
        print(f"  - 関連知識: {len(result2.related_knowledge)} 文字")
        print(f"  - スキルマッチ: {result2.matched_skills}")
        print(f"  - 推定トークン数: {result2.estimated_tokens()} トークン")
        print()

        # Demo 3: Unknown user
        print("-" * 70)
        print("デモ 3: 未知のユーザー")
        print("-" * 70)
        print()
        print("📩 受信メッセージ: 「Hello」")
        print()

        result3 = await engine.prime_memories(
            message="Hello",
            sender_name="unknown_user",
        )

        print("🧠 プライミング結果:")
        print()
        if result3.is_empty():
            print("(送信者プロファイルなし)")
        else:
            print(format_priming_section(result3, "unknown_user"))
        print()
        print(f"📊 統計:")
        print(f"  - 送信者プロファイル: {len(result3.sender_profile)} 文字")
        print(f"  - 直近エピソード: {len(result3.recent_episodes)} 文字 (常に読み込まれる)")
        print(f"  - 関連知識: {len(result3.related_knowledge)} 文字")
        print(f"  - スキルマッチ: {len(result3.matched_skills)} 件")
        print(f"  - 推定トークン数: {result3.estimated_tokens()} トークン")
        print()

    print("=" * 70)
    print("✅ デモ完了")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_priming())
