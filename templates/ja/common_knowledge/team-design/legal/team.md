# 法務フルチーム — チーム概要

## 3ロール構成

| ロール | 責務 | 推奨 `--role` | `speciality` 例 | 詳細 |
|--------|------|--------------|-----------------|------|
| **Legal Director** | 分析計画・契約書スキャン・判断・最終承認 | `manager` | `legal-director` | `legal/director/` |
| **Legal Verifier** | 独立検証・楽観バイアス検出・carry-forward 検証 | `researcher` | `legal-verifier` | `legal/verifier/` |
| **Precedent Researcher** | 法令・判例・業界標準の収集・根拠裏付け | `general` | `legal-researcher` | `legal/researcher/` |

1つの Anima に全工程を集約すると、セルフレビューの盲点（楽観バイアス）・指摘事項の消失（silent drop）・コンテキスト肥大化が発生する。

各ロールディレクトリに `injection.template.md`（injection.md 雛形）、`machine.md`（machine 活用パターン、該当ロールのみ）、`checklist.md`（品質チェックリスト）がある。

> 基本原則の詳細: `team-design/guide.md`

## ハンドオフチェーン

```
Director → analysis-plan.md (approved) + carry-forward tracker 参照
  → machine で契約書全文スキャン → Director が検証
    → audit-report.md (reviewed)
      → Verifier (独立検証) ─┐
      → Researcher (根拠検証) ─┤ ← 並行実行可
        └─ 指摘あり → Director に差し戻し
        └─ 両者 APPROVE → Director → carry-forward tracker 更新 → call_human → 人間が最終確認
```

### 引き継ぎドキュメント

| 送信元 → 送信先 | ドキュメント | 条件 |
|----------------|------------|------|
| Director → Verifier/Researcher | `audit-report.md` + `analysis-plan.md` | `status: reviewed` |
| Verifier → Director | `verification-report.md` | `status: approved` |
| Researcher → Director | `precedent-report.md` | `status: approved` |

### 運用ルール

- **修正サイクル**: Critical → 全体再検証（Verifier・Researcher 両方に再依頼）/ Warning → 差分確認のみ / 3往復解消しない → 人間にエスカレーション
- **carry-forward tracker**: 案件の全版に跨って指摘事項を追跡する。前回監査の指摘が言及なしで消滅すること（silent drop）は禁止
- **machine 失敗時**: `current_state.md` に記録 → 次回 heartbeat で再評価

## スケーリング

| 規模 | 構成 | 備考 |
|------|------|------|
| ソロ | Director が全ロール兼務（checklist で品質担保） | NDA 確認、定型契約レビュー |
| ペア | Director + Verifier | SPA 修正版レビュー、中リスク契約 |
| フルチーム | 本テンプレート通り3名 | SPA 初回監査、M&A DD、高リスク案件 |

## 開発チームとの対応関係

| 開発チームロール | 法務チームロール | 対応する理由 |
|----------------|----------------|-------------|
| PdM（調査・計画・判断） | Director（分析計画・判断） | 「何を分析するか」を決定する司令塔 |
| Engineer（実装） | Director + machine（契約書スキャン） | Director が machine で分析を実行。独立 Anima 不要 |
| Reviewer（静的検証） | Verifier（独立検証） | 「実行と検証の分離」の核。最も重要な分離ポイント |
| Tester（動的検証） | Researcher（根拠検証） | 「業界標準」「判例」等の主張を実データで裏付ける |

## Carry-forward Tracker — 指摘事項追跡表

契約の版が更新されるたびにこの表を更新する。前回監査の全指摘事項を管理し、silent drop を構造的に防止する。

### 追跡ルール

- 前回監査の全指摘事項をこの表で管理する
- 新版受領時、全項目のステータスを更新する
- 「解消」以外の項目は次回レビューで必ず言及する
- silent drop（言及なしでの消滅）は禁止

### テンプレート

```markdown
# 指摘事項追跡表: {案件名}

| # | 初出日 | 項目 | 初回リスク | v1 ステータス | v2 ステータス | v3 ステータス | 現在の残存リスク |
|---|--------|------|----------|------------|------------|------------|--------------|
| C-1 | {日付} | {指摘内容} | Critical | {対応状況} | {対応状況} | — | {現在のリスク} |
| C-2 | {日付} | {指摘内容} | Critical | {対応状況} | {対応状況} | — | {現在のリスク} |
| H-1 | {日付} | {指摘内容} | High | {対応状況} | {対応状況} | — | {現在のリスク} |
| M-1 | {日付} | {指摘内容} | Medium | {対応状況} | {対応状況} | — | {現在のリスク} |

ステータス凡例:
- 未修正: 前回から変更なし
- 解消: 修正され、リスクが除去された
- 部分解消: 修正されたが残存リスクあり（残存リスク欄に詳細）
- 悪化: 修正により新たなリスクが発生、または既存リスクが増大
```
