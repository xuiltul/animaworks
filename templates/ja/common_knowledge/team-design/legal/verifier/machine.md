# Legal Verifier — machine 活用パターン

## 基本ルール

1. **計画書を先に書く** — インラインの短い指示文字列での実行は禁止。計画書ファイルを渡す
2. **出力はドラフト** — machine の出力は必ず自分で検証し、`status: approved` にしてから次工程へ
3. **保存場所**: `state/plans/{YYYY-MM-DD}_{概要}.{type}.md`（`/tmp/` 禁止）
4. **レート制限**: chat 5回/session、heartbeat 2回
5. **machine はインフラにアクセスできない** — 記憶・メッセージ・組織情報は計画書に含めること

---

## 概要

Verifier は **machine に検証スキャンを委託し、そのスキャン結果の正当性を検証する（メタ検証）**。

- 検証観点の設計 → Verifier 自身が判断
- 差分検出・carry-forward 照合の実行 → machine に委託
- 検出結果の正当性検証 → Verifier 自身が判断
- 最悪シナリオの構築 → Verifier 自身が判断

machine は条項の差分検出・carry-forward の機械的照合・リスク評価の前回比較等を高速に行えるが、
楽観バイアスの判定や最悪シナリオの構築は Verifier の責務である。

---

## ワークフロー

### Step 1: 検証計画書を作成する（Verifier 自身が書く）

検証の観点・対象・基準を明確にした計画書を作成する。

```bash
write_memory_file(path="state/plans/{date}_{案件名}.verification.md", content="...")
```

作成前に以下の情報を Anima 側で準備する:
- Director の `audit-report.md` と `analysis-plan.md`
- carry-forward tracker（前回版との比較用）
- 契約書本文（修正前後の両バージョン）

### Step 2: machine に検証スキャンを投げる

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{verification計画書})" \
  -d /path/to/workspace
```

結果を `state/plans/{date}_{案件名}.verification.md` に追記または上書きする（`status: draft`）。

### Step 3: 検証結果をメタ検証する

Verifier が verification.md を読み、以下を確認する:

- [ ] 指摘内容は事実に基づいているか（誤検出がないか）
- [ ] carry-forward の照合結果に漏れがないか
- [ ] Director の「受容可」判定に対する最悪シナリオ分析が含まれているか
- [ ] リスク評価の前回比較が正確か
- [ ] Verifier 自身の観点で追加すべき指摘がないか

Verifier 自身が修正・補足し、`status: approved` に変更する。

### Step 4: フィードバック

approved の verification-report.md を Director に送付する。
指摘がある場合は、具体的な反論根拠と推奨修正を明記する。

---

## 検証計画書テンプレート（verification.plan.md）

```markdown
# 検証計画書: {検証対象の概要}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: verification-plan

## 検証観点

- [ ] 楽観バイアス検出: 「受容可」判定の全項目を再評価
- [ ] carry-forward 追跡: 前回指摘の全件が今回レポートに反映されているか
- [ ] リスク評価の前回比較: 低下した評価に十分な根拠があるか
- [ ] 法的正確性: 引用法令・判例の正確性
- [ ] 条項網羅性: 契約書の全条項が分析対象になっているか

## 対象

- audit-report.md: {パス}
- analysis-plan.md: {パス}
- carry-forward tracker: {パス}
- 契約書本文: {パス / 格納場所}

## 差分情報

{修正前後の契約書の主要な変更点。全文ではなく差分に絞る}

## 出力形式（必須）

以下の形式で検証結果を出力すること。**この形式に従わない出力は無効とする。**

- **Critical**: 修正必須の問題（楽観バイアス・silent drop・法的誤り）
- **Warning**: 修正推奨の問題（根拠不足・文言リスク）
- **Info**: 情報提供・改善提案
```

## 検証報告書テンプレート（verification-report.md）

```markdown
# 検証報告書: {案件名}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: verification-report

## 総合判定

{APPROVE / REQUEST_CHANGES / COMMENT}

## 楽観バイアス検出

| # | 対象条項 | Director 判定 | Verifier 所見 | 最悪シナリオ | 推奨 |
|---|---------|-------------|-------------|------------|------|
| 1 | {条項} | {受容可} | {懸念内容} | {最悪の場合} | {修正案} |

## Carry-forward 漏れ検出

| # | 前回指摘 | 前回リスク | 今回レポートでの扱い | 判定 |
|---|---------|----------|------------------|------|
| 1 | {指摘} | {リスク} | {言及あり / silent drop} | {OK / NG} |

## リスク評価の前回比較

| # | 条項 | 前回リスク | 今回リスク | 変更理由の妥当性 |
|---|------|----------|----------|---------------|
| 1 | {条項} | {前回} | {今回} | {妥当 / 根拠不足} |

## 法的正確性

| # | 指摘内容 | 重要度 |
|---|---------|--------|
| 1 | {内容} | {Critical/Warning/Info} |

## Verifier 所見

{Verifier 自身の分析・追加観察・推奨事項}
```

---

## 制約事項

- 検証計画書（何を観点に検証するか）は MUST: Verifier 自身が書く
- machine の検証結果をそのまま Director に渡してはならない（NEVER） — 必ず Verifier がメタ検証する
- `status: approved` でない verification-report.md を Director にフィードバックしてはならない（NEVER）
- 「受容可」判定に対する最悪シナリオ検討は machine に任せず Verifier 自身が行う（MUST）
