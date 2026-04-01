# トレーディングフルチーム — チーム概要

## 4ロール構成

| ロール | 責務 | 推奨 `--role` | `speciality` 例 | 詳細 |
|--------|------|--------------|-----------------|------|
| **Strategy Director** | 戦略設計・リスク限度設定・PDCA統括・最終判断 | `manager` | `trading-director` | `trading/director/` |
| **Market Analyst** | 市場分析・シグナル生成・モデル開発 | `researcher` | `market-analyst` | `trading/analyst/` |
| **Trading Engineer** | bot実装・バックテスト・実行基盤・データパイプライン | `engineer` | `trading-engineer` | `trading/engineer/` |
| **Risk Auditor** | 独立P&L検証・運用健全性監査・carry-forward追跡 | `engineer` or `ops` | `risk-auditor` | `trading/auditor/` |

1つの Anima に全工程を集約すると、損益の楽観バイアス（損失の軽視）・運用検証の欠落（bot停止に気づかない）・課題追跡の消失（ウォレット未確認の放置）が発生する。

各ロールディレクトリに `injection.template.md`（injection.md 雛形）、`machine.md`（machine 活用パターン）、`checklist.md`（品質チェックリスト）がある。

> 基本原則の詳細: `team-design/guide.md`

## ハンドオフチェーン

```
Director → strategy-plan.md (approved) + performance-tracker 参照
  → delegate_task
    → Engineer: bot 実装 / バックテスト (machine 活用)
    → Analyst: 市場分析 / シグナル検証 (machine 活用)
      → backtest-report.md + market-analysis.md (reviewed)
        → Auditor (P&L検証 + 運用健全性監査) ← 独立検証
          └─ 問題あり → Director に差し戻し
          └─ APPROVE → Director → performance-tracker 更新 → 上位報告 / call_human
```

### 引き継ぎドキュメント

| 送信元 → 送信先 | ドキュメント | 条件 |
|----------------|------------|------|
| Director → Engineer | `strategy-plan.md` | `status: approved` |
| Director → Analyst | `strategy-plan.md`（分析観点指示） | `status: approved` |
| Engineer → Auditor | `backtest-report.md` | `status: reviewed` |
| Analyst → Director | `market-analysis.md` | `status: reviewed` |
| Auditor → Director | `performance-review.md` + `ops-health-report.md` | `status: approved` |

### 運用ルール

- **修正サイクル**: Critical（ドローダウン閾値超過・bot停止・資産不一致）→ 即時対応 + Auditor 再検証 / Warning → 差分確認のみ / 3往復解消しない → 人間にエスカレーション
- **Performance Tracker**: 戦略バージョンを横断して P&L・勝率・Sharpe・最大DDを追跡する。前回フラグした問題が次回で言及なしに消滅すること（silent drop）は禁止
- **Ops Issue Tracker**: 運用上の課題を carry-forward で追跡する。silent drop 禁止
- **PDCA サイクル**: Plan=Director（戦略設計）、Do=Engineer（実装）+ Analyst（分析）、Check=Auditor（独立検証）、Act=Director（判断・修正指示）
- **machine 失敗時**: `current_state.md` に記録 → 次回 heartbeat で再評価

## スケーリング

| 規模 | 構成 | 備考 |
|------|------|------|
| ソロ | Director が全ロール兼務（checklist で品質担保） | ペーパートレード検証、単一戦略 |
| ペア | Director + Auditor | 中リスク、少数戦略のライブ運用 |
| トリオ | Director + Engineer + Auditor | bot 開発フェーズ（Analyst は Director が兼務） |
| フルチーム | 本テンプレート通り4名 | 複数戦略の本番運用 |

## 開発チーム・法務チームとの対応関係

| 開発チームロール | 法務チームロール | トレーディングチームロール | 対応する理由 |
|----------------|----------------|----------------------|-------------|
| PdM（調査・計画・判断） | Director（分析計画・判断） | Director（戦略設計・PDCA判断） | 「何をやるか」を決定する司令塔 |
| Engineer（実装） | Director + machine | Engineer（bot実装・バックテスト） | コードを書き、動くものを作る |
| Reviewer（静的検証） | Verifier（独立検証） | Auditor（P&L検証 + 運用健全性） | 「実行と検証の分離」の核。最も重要な分離ポイント |
| Tester（動的検証） | Researcher（根拠検証） | Analyst（市場分析・シグナル品質） | 外部データに基づく裏付け・品質確認 |

## Strategy Performance Tracker — 戦略パフォーマンス追跡表

戦略のバージョン更新ごとにこの表を更新する。前回フラグした問題が次回で言及なしに消滅すること（silent drop）を構造的に防止する。

### 追跡ルール

- 戦略のバージョン更新（パラメータ変更含む）ごとに行を追加する
- ドローダウンが閾値 `{max_drawdown_pct}` を超えた場合は即座に Director に報告する
- 「継続監視」以外のステータスが付いた問題は次回レビューで必ず言及する
- silent drop（言及なしでの消滅）は禁止

### テンプレート

```markdown
# 戦略パフォーマンス追跡表: {戦略名}

| # | 期間 | バージョン | P&L | 勝率 | Sharpe | 最大DD | ステータス | 備考 |
|---|------|----------|-----|------|--------|--------|----------|------|
| S-1 | {開始〜終了} | {v1} | {金額} | {%} | {値} | {%} | {評価} | {特記事項} |
| S-2 | {開始〜終了} | {v2} | {金額} | {%} | {値} | {%} | {評価} | {特記事項} |

ステータス凡例:
- 本番稼働: ライブ環境で運用中
- ペーパートレード: 検証中
- パラメータ調整中: 改善サイクル実施中
- 停止: 閾値超過またはエッジ消失により停止
- 廃止: 戦略そのものを破棄
```

## Ops Issue Tracker — 運用課題追跡表

運用上の課題を carry-forward で追跡する。silent drop を構造的に防止する。

### 追跡ルール

- 運用課題（bot停止・API障害・資産不一致等）を検出したらこの表に登録する
- 次回 Heartbeat / レビュー時に全項目のステータスを更新する
- 「解消」以外の項目は次回で必ず言及する
- silent drop（言及なしでの消滅）は禁止

### テンプレート

```markdown
# 運用課題追跡表: {チーム名}

| # | 検出日 | 課題 | 深刻度 | ステータス | 対応者 | 解消日 | 備考 |
|---|--------|------|--------|----------|--------|--------|------|
| O-1 | {日付} | {課題内容} | Critical | {対応状況} | {担当} | {日付} | {特記事項} |
| O-2 | {日付} | {課題内容} | Warning | {対応状況} | {担当} | {日付} | {特記事項} |

ステータス凡例:
- 未対応: 検出済み・未着手
- 対応中: 修正作業進行中
- 解消: 問題が除去された
- 再発: 一度解消したが再発
- 継続監視: 暫定対応済み・根本対応は未了
```
