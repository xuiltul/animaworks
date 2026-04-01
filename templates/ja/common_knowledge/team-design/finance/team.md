# 財務フルチーム — チーム概要

## 4ロール構成

| ロール | 責務 | 推奨 `--role` | `speciality` 例 | 詳細 |
|--------|------|--------------|-----------------|------|
| **Finance Director** | 分析計画・財務判断・数値検証・最終承認 | `manager` | `cfo`, `finance-director` | `finance/director/` |
| **Financial Auditor** | 独立検証・仮定検証・Data Lineage 検証 | `researcher` | `financial-auditor` | `finance/auditor/` |
| **Data Analyst** | ソースデータ抽出・構造化・多段階検証 | `general` | `data-analyst` | `finance/analyst/` |
| **Market Data Collector** | 外部市場データ・ベンチマーク・参照価格の収集 | `general` | `market-data` | `finance/collector/` |

1つの Anima に全工程を集約すると、セルフレビューの盲点（解釈の楽観バイアス）・差異の消失（silent drop）・コンテキスト肥大化が発生する。

各ロールディレクトリに `injection.template.md`（injection.md 雛形）、`machine.md`（machine 活用パターン、該当ロールのみ）、`checklist.md`（品質チェックリスト）がある。

> 基本原則の詳細: `team-design/guide.md`

## ハンドオフチェーン

```
Analyst (ソースデータ抽出) + Collector (外部データ収集) ← 並行実行可
  → Director → analysis-plan.md (approved) → machine で分析実行
    → analysis-report.md (reviewed)
      → Auditor (独立検証)
        └─ 指摘あり → Director に差し戻し
        └─ APPROVE → Director → Variance Tracker 更新 → call_human → 人間が最終確認
```

### 引き継ぎドキュメント

| 送信元 → 送信先 | ドキュメント | 条件 |
|----------------|------------|------|
| Analyst/Collector → Director | ソースデータ + 抽出検証結果 | 検証済み |
| Director → Auditor | `analysis-report.md` + `analysis-plan.md` | `status: reviewed` |
| Auditor → Director | `audit-report.md` | `status: approved` |

### 運用ルール

- **修正サイクル**: Critical → 全体再検証（Auditor に再依頼）/ Warning → 差分確認のみ / 3往復解消しない → 人間にエスカレーション
- **Variance Tracker**: 分析で検出した重要差異を月跨ぎで追跡する。前回フラグした差異が次回レポートで言及なしに消滅すること（silent drop）は禁止
- **Data Lineage Rule**: analysis-report.md 内の全数値はソースデータまで遡れなければならない。推定値には「推定」マーカー必須
- **machine 失敗時**: `current_state.md` に記録 → 次回 heartbeat で再評価

## スケーリング

| 規模 | 構成 | 備考 |
|------|------|------|
| ソロ | Director が全ロール兼務（checklist で品質担保） | 定型月次報告、単一法人分析 |
| ペア | Director + Auditor | 重要な判断を含む分析、複数法人比較 |
| トリオ | Director + Auditor + Analyst（Collector 兼務） | データ量が多い案件 |
| フルチーム | 本テンプレート通り4名 | 連結分析、大型案件、ポートフォリオ評価 |

## 開発チーム・法務チームとの対応関係

| 開発チームロール | 法務チームロール | 財務チームロール | 対応する理由 |
|----------------|----------------|----------------|-------------|
| PdM（調査・計画・判断） | Director（分析計画・判断） | Director（分析計画・判断） | 「何を分析するか」を決定する司令塔 |
| Engineer（実装） | Director + machine | Director + machine | Director が machine で分析を実行。独立 Anima 不要 |
| Reviewer（静的検証） | Verifier（独立検証） | Auditor（独立検証） | 「実行と検証の分離」の核。最も重要な分離ポイント |
| Tester（動的検証） | Researcher（根拠検証） | Collector（外部データ収集） | 外部情報で裏付けを取る |
| — | — | Analyst（データ抽出） | 財務固有。ソースデータの正確な抽出・構造化 |

## Monthly Variance Tracker — 月次差異追跡表

分析で検出した重要差異を月跨ぎで追跡する。前回フラグした差異が次回レポートで言及なしに消滅すること（silent drop）を構造的に防止する。

### 追跡ルール

- 重要差異（閾値を超える変動）を検出したらこの表に登録する
- 次回分析時に全項目のステータスを更新する
- 「解消」以外の項目は次回レポートで必ず言及する
- silent drop（言及なしでの消滅）は禁止

### テンプレート

```markdown
# 月次差異追跡表: {対象名}

| # | 初検出月 | 勘定科目 | 初回差異率 | M月ステータス | M+1月ステータス | M+2月ステータス | 現在の残存リスク |
|---|---------|---------|----------|------------|-------------|-------------|--------------|
| V-1 | {月} | {科目} | {差異率} | {対応状況} | {対応状況} | — | {リスク評価} |
| V-2 | {月} | {科目} | {差異率} | {対応状況} | {対応状況} | — | {リスク評価} |

ステータス凡例:
- 解消: 原因特定済み・対処完了、リスクが除去された
- 継続監視: 原因は判明したが再発リスクあり（追跡期間と判断基準を併記）
- 調査中: 原因未特定
- 悪化: 差異が拡大、または新たなリスクが発生
```
