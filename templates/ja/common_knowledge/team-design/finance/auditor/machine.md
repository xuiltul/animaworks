# Financial Auditor — machine 活用パターン

## 基本ルール

1. **計画書を先に書く** — インラインの短い指示文字列での実行は禁止。計画書ファイルを渡す
2. **出力はドラフト** — machine の出力は必ず自分で検証し、`status: approved` にしてから次工程へ
3. **保存場所**: `state/plans/{YYYY-MM-DD}_{概要}.{type}.md`（`/tmp/` 禁止）
4. **レート制限**: chat 5回/session、heartbeat 2回
5. **machine はインフラにアクセスできない** — 記憶・メッセージ・組織情報は計画書に含めること

---

## 概要

Auditor は **machine に検証スキャンを委託し、そのスキャン結果の正当性を検証する（メタ検証）**。

- 検証観点の設計 → Auditor 自身が判断
- 数値再計算・Variance Tracker 照合・Data Lineage 追跡 → machine に委託
- 検出結果の正当性検証 → Auditor 自身が判断
- 仮定の妥当性検証（Assumption Challenge） → Auditor 自身が判断

machine は数値の再計算・差分検出・Data Lineage の機械的追跡を高速に行えるが、
仮定の妥当性判定や悲観シナリオの構築は Auditor の責務である。

---

## ワークフロー

### Step 1: 検証計画書を作成する（Auditor 自身が書く）

検証の観点・対象・基準を明確にした計画書を作成する。

```bash
write_memory_file(path="state/plans/{date}_{対象名}.audit-plan.md", content="...")
```

作成前に以下の情報を Anima 側で準備する:
- Director の `analysis-report.md` と `analysis-plan.md`
- Variance Tracker（前回との比較用）
- ソースデータ（独立検証に必要な場合）

### Step 2: machine に検証スキャンを投げる

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{audit計画書})" \
  -d /path/to/workspace
```

結果を `state/plans/{date}_{対象名}.audit-report.md` に追記または上書きする（`status: draft`）。

### Step 3: 検証結果をメタ検証する

Auditor が audit-report.md を読み、以下を確認する:

- [ ] 指摘内容は事実に基づいているか（誤検出がないか）
- [ ] Variance Tracker の照合結果に漏れがないか
- [ ] Data Lineage の追跡結果が正確か
- [ ] Director の仮定に対する Assumption Challenge が含まれているか
- [ ] Auditor 自身の観点で追加すべき指摘がないか

Auditor 自身が修正・補足し、`status: approved` に変更する。

### Step 4: フィードバック

approved の audit-report.md を Director に送付する。
指摘がある場合は、具体的な反証データと推奨修正を明記する。

---

## 検証計画書テンプレート（audit-plan.md）

```markdown
# 検証計画書: {検証対象の概要}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: audit-plan

## 検証観点

- [ ] Assumption Challenge: Director の仮定・解釈を反証的視点で検証
- [ ] Variance Tracker 追跡: 前回差異の全件が今回レポートに反映されているか
- [ ] Data Lineage: 全数値がソースデータまで遡れるか
- [ ] 数値正確性: 主要指標の独立再計算
- [ ] 会計恒等式: BS 等式・TB 均衡の成立確認

## 対象

- analysis-report.md: {パス}
- analysis-plan.md: {パス}
- Variance Tracker: {パス}
- ソースデータ: {パス / 格納場所}

## 出力形式（必須）

以下の形式で検証結果を出力すること。**この形式に従わない出力は無効とする。**

- **Critical**: 修正必須の問題（仮定の不備・silent drop・数値不整合）
- **Warning**: 修正推奨の問題（根拠不足・Data Lineage 不完全）
- **Info**: 情報提供・改善提案
```

## 検証報告書テンプレート（audit-report.md）

```markdown
# 検証報告書: {対象名}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: audit-report

## 総合判定

{APPROVE / REQUEST_CHANGES / COMMENT}

## Assumption Challenge 結果

| # | 対象項目 | Director の判断 | Auditor 所見 | 反証データ | 推奨 |
|---|---------|---------------|-------------|----------|------|
| 1 | {項目} | {季節要因} | {懸念内容} | {過去データ} | {修正案} |

## Variance Tracker 漏れ検出

| # | 前回差異 | 前回ステータス | 今回レポートでの扱い | 判定 |
|---|---------|-------------|------------------|------|
| 1 | {差異} | {ステータス} | {言及あり / silent drop} | {OK / NG} |

## Data Lineage 検証

| # | 対象数値 | ソース情報 | 判定 |
|---|---------|----------|------|
| 1 | {数値・指標} | {追跡可 / ソース不明} | {OK / NG} |

## 数値再計算結果

| # | 指標 | Director の値 | Auditor 再計算値 | 差異 | 判定 |
|---|------|-------------|----------------|------|------|
| 1 | {指標} | {値} | {値} | {差異} | {OK / NG} |

## Auditor 所見

{Auditor 自身の分析・追加観察・推奨事項}
```

---

## 制約事項

- 検証計画書（何を観点に検証するか）は MUST: Auditor 自身が書く
- machine の検証結果をそのまま Director に渡してはならない（NEVER） — 必ず Auditor がメタ検証する
- `status: approved` でない audit-report.md を Director にフィードバックしてはならない（NEVER）
- 仮定の妥当性検証（Assumption Challenge）は machine に任せず Auditor 自身が行う（MUST）
