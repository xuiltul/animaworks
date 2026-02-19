# Integration Tests for Priming Layer

このディレクトリには、プライミングレイヤーのエンドツーエンド（E2E）統合テストが含まれています。

## ファイル構成

- `test_priming_e2e.py` - プライミングレイヤーの統合テスト（738行、8テスト）

## テストケース

### 1. test_priming_with_real_person_directory

実際のPerson構造でプライミングが正常に動作することを確認。

**検証項目:**
- チャネルA: 送信者プロファイル取得
- チャネルB: 直近エピソード取得
- チャネルC: 関連知識検索
- チャネルD: スキルマッチング

### 2. test_message_to_response_flow

メッセージ受信からプライミング、システムプロンプト注入までのフロー確認。

**検証項目:**
- プライミング実行
- プライミング結果のフォーマット
- システムプロンプトへの注入準備

### 3. test_daily_consolidation_e2e

日次固定化の全フロー検証。

**検証項目:**
- エピソード収集
- LLM呼び出し
- knowledge/への書き込み
- [AUTO-CONSOLIDATED]タグ付与
- RAGインデックス更新（利用可能時）

### 4. test_weekly_integration_e2e

週次統合の全フロー検証。

**検証項目:**
- 重複知識ファイルの検出・マージ
- [AUTO-MERGED]タグ付与
- 古いエピソードの圧縮
- [COMPRESSED]タグ付与
- [IMPORTANT]タグ付きエピソードの保持

### 5. test_priming_performance_under_200ms

プライミングのパフォーマンス目標達成確認。

**検証項目:**
- 100回のプライミング実行
- レイテンシ分布測定（平均、中央値、P95、P99）
- 95パーセンタイル < 200ms の確認

**実測値（2026-02-14）:**
- Mean: 4.67 ms
- Median: 3.89 ms
- P95: 4.89 ms
- P99: 69.20 ms

✅ **目標達成**: P95は4.89msで、200msを大幅に下回る

### 6. test_priming_empty_directories

空のディレクトリでプライミングがエラーなく動作することを確認。

### 7. test_consolidation_skipped_when_no_episodes

エピソードが不足している場合に固定化がスキップされることを確認。

### 8. test_priming_dynamic_budget_adjustment

動的予算調整が正しく動作することを確認。

**検証項目:**
- 挨拶（小予算: 500トークン）
- 質問（中予算: 1500トークン）
- リクエスト（大予算: 3000トークン）
- ハートビート（最小予算: 200トークン）

## 実行方法

### 全統合テストを実行

```bash
pytest tests/integration/test_priming_e2e.py -v
```

### 特定のテストを実行

```bash
pytest tests/integration/test_priming_e2e.py::test_priming_performance_under_200ms -v
```

### パフォーマンス統計を表示

```bash
pytest tests/integration/test_priming_e2e.py::test_priming_performance_under_200ms -v -s
```

### 全プライミング関連テストを実行

```bash
pytest tests/test_priming.py tests/test_consolidation.py tests/integration/test_priming_e2e.py -v
```

## テスト結果

**最終結果（2026-02-14）:**
- ✅ 8/8 テスト合格（100%）
- ✅ 実行時間: 1.49秒
- ✅ パフォーマンス目標達成（P95: 4.89ms < 200ms）

**プライミング関連全テスト:**
- ✅ 27/27 テスト合格（100%）
- ✅ 実行時間: 1.54秒

## 依存関係

### 必須

- pytest
- pytest-asyncio

### オプション

- chromadb（RAG検索テスト用）
- ripgrep（BM25検索用、未インストール時はPythonフォールバック）

## モック戦略

- **LLM呼び出し**: `litellm.acompletion()` をモック化
- **ファイルシステム**: `tmp_path` フィクスチャ使用
- **時刻**: 現在時刻を使用（freezegun不使用）

## カバレッジ

このテストスイートは、以下のコンポーネントをカバーしています:

- `core.memory.priming.PrimingEngine` - 4チャネル並列プライミング
- `core.memory.consolidation.ConsolidationEngine` - 日次固定化・週次統合
- `core.memory.priming.format_priming_section` - プライミング結果フォーマット

## 設計ドキュメント

詳細は以下のドキュメントを参照:

- `docs/design/priming-layer-design.md` - プライミングレイヤー設計
- `docs/testing/priming-layer-test-plan.md` - テスト計画書（Phase 2対応）
