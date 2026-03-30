# 開発フルチーム — チーム概要

## 4ロール構成

| ロール | 責務 | 推奨 `--role` | `speciality` 例 | 詳細 |
|--------|------|--------------|-----------------|------|
| **PdM** | 調査・計画・判断 | `manager` | `pdm` | `development/pdm/` |
| **Engineer** | 実装・実装検証 | `engineer` | `backend`, `fullstack` | `development/engineer/` |
| **Reviewer** | コードレビュー（静的検証） | `engineer` | `code-review` | `development/reviewer/` |
| **Tester** | テスト設計・実行（動的検証） | `engineer` | `testing`, `qa` | `development/tester/` |

1つの Anima に集約すると、コンテキスト肥大化・セルフレビューの盲点・直列実行ボトルネックが発生する。

各ロールディレクトリに `injection.template.md`（injection.md 雛形）、`machine.md`（machine 活用パターン）、`checklist.md`（品質チェックリスト）がある。

> 基本原則の詳細: `team-design/guide.md`

## ハンドオフチェーン

```
PdM → investigation.md/plan.md (approved) → delegate_task
  → Engineer → impl.plan.md → 実装 → 実装検証
    → Reviewer (静的検証) ─┐
    → Tester  (動的検証) ─┤ ← 並行実行可
      └─ 修正要 → Engineer に戻る
      └─ 両者 APPROVE → PdM → call_human → 人間がマージ判断
```

### 引き継ぎドキュメント

| 送信元 → 送信先 | ドキュメント | 条件 |
|----------------|------------|------|
| PdM → Engineer | `plan.md` | `status: approved` |
| Engineer → Reviewer/Tester | 実装差分 + `plan.md` | 実装検証済み |
| Reviewer → Engineer | `review.md` | `status: approved` |
| Tester → Engineer/PdM | `test-report.md` | `status: approved` |

### 運用ルール

- **Worktree**: Engineer が plan.md 受領後に作成（`{task-id}/{概要}`）。machine は `-d /path/to/worktree` で実行。完了後の マージ・削除は Engineer が実施
- **修正サイクル**: Critical → 全体再レビュー / Warning → 差分確認のみ / 3往復解消しない → PdM にエスカレーション
- **machine 失敗時**: `current_state.md` に記録 → 次回 heartbeat で再評価

## スケーリング

| 規模 | 構成 | 備考 |
|------|------|------|
| 小規模 | PdM + Engineer（Reviewer 兼務） | セルフレビューのリスクを許容 |
| 中規模 | 本テンプレート通り4名 | 標準構成 |
| 大規模 | PdM 1 + Engineer 複数 + Reviewer/Tester 各1〜2 | PdM が `delegate_task` でモジュール単位委譲 |
