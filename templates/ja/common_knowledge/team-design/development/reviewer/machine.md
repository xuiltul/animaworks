# Reviewer — machine 活用パターン

## 基本ルール

1. **計画書を先に書く** — インラインの短い指示文字列での実行は禁止。計画書ファイルを渡す
2. **出力はドラフト** — machine の出力は必ず自分で検証し、`status: approved` にしてから次工程へ
3. **保存場所**: `state/plans/{YYYY-MM-DD}_{概要}.{type}.md`（`/tmp/` 禁止）
4. **レート制限**: chat 5回/session、heartbeat 2回
5. **machine はインフラにアクセスできない** — 記憶・メッセージ・組織情報は計画書に含めること

---

## 概要

Reviewer は **machine にレビューを委託し、そのレビュー結果の正当性を検証する（メタレビュー）**。

- レビュー観点の設計 → Reviewer 自身が判断
- コードレビューの実施 → machine に委託
- レビュー結果の正当性検証 → Reviewer 自身が判断

machine はコードの静的分析・パターン検出・要件充足チェック等を高速に行えるが、
設計判断の妥当性や文脈を踏まえた重要度の判定は Reviewer の責務である。

---

## ワークフロー

### Step 1: レビュー計画書を作成する（Reviewer 自身が書く）

レビューの観点・対象・基準を明確にした計画書を作成する。

```bash
write_memory_file(path="state/plans/{date}_{概要}.review.md", content="...")
```

作成前に以下の情報を Anima 側で準備する:
- `git diff` や PR の差分内容
- 関連する Issue・plan.md の要件
- 対象コードの既存構造（必要に応じて `search_memory` や直接読み取り）

### Step 2: machine にレビューを投げる

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{review計画書})" \
  -d /path/to/worktree
```

結果を `state/plans/{date}_{概要}.review.md` に追記または上書きする（`status: draft`）。

### Step 3: レビュー結果を検証する（メタレビュー）

Reviewer が review.md を読み、以下を確認する:

- [ ] 指摘内容は事実に基づいているか（誤検出がないか）
- [ ] 重要な問題を見落としていないか
- [ ] 重要度の判定（Critical/Warning/Info）が妥当か
- [ ] plan.md の完了条件・制約条件との整合性
- [ ] コーディング規約との整合性
- [ ] Reviewer 自身の観点で追加すべき指摘がないか

Reviewer 自身が修正・補足し、`status: approved` に変更する。

### Step 4: フィードバック

approved の review.md を Engineer に送付する。
修正が必要な場合は、具体的なアクションアイテムを明記する。

---

## レビュー計画書テンプレート（review.plan.md）

```markdown
# レビュー計画書: {レビュー対象の概要}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: review

## レビュー観点

{担当する観点を明記}

- [ ] 要件充足: plan.md の完了条件を満たしているか
- [ ] コード品質: 可読性、保守性、命名規約
- [ ] 安全性: セキュリティ上の問題がないか
- [ ] パフォーマンス: N+1 クエリ、不要なループ等
- [ ] テスト: テストカバレッジ、エッジケース

## 対象

{レビュー対象の差分情報}

- ブランチ / PR: {情報}
- 変更ファイル一覧:
  - {file1}
  - {file2}

## Issue / plan.md 要件

{関連する Issue URL、plan.md の完了条件の要約}

## 差分情報

{git diff の出力、または主要な変更箇所の抜粋。全ファイル内容ではなく変更部分に絞る — machine に渡すコンテキストは「差分 + 周辺10行」程度が最も効果的}

## 確認コマンド

{レビュー時に実行すべきコマンド}

- `git diff {base}..{head}`
- `{テスト実行コマンド}`
- `{lint実行コマンド}`

## 出力形式（必須）

以下の形式でレビュー結果を出力すること。**この形式に従わない出力は無効とする。**

- **Critical**: 修正必須の問題（ファイル名・行番号・修正案を含む）
- **Warning**: 修正推奨の問題
- **Info**: 情報提供・改善提案
```

## レビュー結果テンプレート（review.md）

```markdown
# レビュー結果: {レビュー対象の概要}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: review

## 総合判定

{APPROVE / REQUEST_CHANGES / COMMENT}

## 指摘一覧

### Critical（修正必須）

| # | ファイル | 行 | 指摘内容 | 推奨修正 |
|---|---------|-----|---------|---------|
| 1 | {path} | {line} | {内容} | {修正案} |

### Warning（修正推奨）

| # | ファイル | 行 | 指摘内容 | 推奨修正 |
|---|---------|-----|---------|---------|
| 1 | {path} | {line} | {内容} | {修正案} |

### Info（情報提供）

| # | ファイル | 行 | 指摘内容 |
|---|---------|-----|---------|
| 1 | {path} | {line} | {内容} |

## 要件充足チェック

| 完了条件 | 充足 | 備考 |
|---------|------|------|
| {条件1} | Yes / No | {備考} |
| {条件2} | Yes / No | {備考} |

## 追加コメント

{Reviewer 自身の所見・補足}
```

---

## 制約事項

- レビュー計画書（何を観点にレビューするか）は MUST: Reviewer 自身が書く
- machine のレビュー結果をそのまま Engineer に渡してはならない（NEVER） — 必ず Reviewer が検証する
- `status: approved` でない review.md を Engineer にフィードバックしてはならない（NEVER）
- 差分情報や Issue 要件は Anima 側で取得し、計画書に含める（machine は GitHub API にアクセスできない）
