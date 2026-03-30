# PdM — machine 活用パターン

## 基本ルール

1. **計画書を先に書く** — インラインの短い指示文字列での実行は禁止。計画書ファイルを渡す
2. **出力はドラフト** — machine の出力は必ず自分で検証し、`status: approved` にしてから次工程へ
3. **保存場所**: `state/plans/{YYYY-MM-DD}_{概要}.{type}.md`（`/tmp/` 禁止）
4. **レート制限**: chat 5回/session、heartbeat 2回
5. **machine はインフラにアクセスできない** — 記憶・メッセージ・組織情報は計画書に含めること

---

## 概要

PdM は **調査の手足として machine を使い、計画の判断は自分で行う**。

- 調査・情報収集 → machine に委託
- 分析結果の解釈・優先順位付け・実装方針の決定 → PdM 自身が判断
- 計画書（plan.md）の作成 → PdM 自身が書く

---

## Phase 1: 調査

### Step 1: 調査計画書を作成する（PdM 自身が書く）

machine に何を調べさせるかを明確にした計画書を作成する。

```bash
write_memory_file(path="state/plans/{date}_{概要}.investigation.md", content="...")
```

### Step 2: machine に調査を投げる

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{調査計画書})" \
  -d /path/to/worktree
```

machine は調査結果をテキストとして返す。
結果を `state/plans/{date}_{概要}.investigation.md` として保存する（`status: draft`）。

### Step 3: 調査結果を検証する

PdM が investigation.md を読み、以下を確認する:

- [ ] 調査目的に対する回答が含まれているか
- [ ] 事実と推測が区別されているか
- [ ] 重要な情報の見落としがないか
- [ ] 追加調査が必要な箇所はないか

問題があれば PdM 自身が修正・補足し、`status: approved` に変更する。

## Phase 2: 計画書作成

### Step 4: plan.md を作成する（PdM 自身の判断）

investigation.md の内容をベースに、**PdM 自身が** plan.md を作成する。

plan.md の「実装方針」「優先順位」「制約条件」は PdM の判断の核であり、
machine に書かせてはならない（NEVER）。

### Step 5: 計画書を委譲する

plan.md の `status: approved` を確認し、Engineer に `delegate_task` で渡す。

---

## 調査計画書テンプレート

```markdown
# 調査計画書: {問題・テーマの概要}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: investigation

## 調査目的

{何を明らかにしたいか — 1〜3文で明確に}

## 調査対象

{対象を具体的に絞り込む — 「リポジトリ全体」より「core/memory/*.py の RAG 関連」が効果的}

- {対象1}
- {対象2}

## 調査手順

{machine に実行させる具体的な手順。ステップを明確に分ける}

1. {手順1}
2. {手順2}
3. {手順3}

## スコープ外

{調査しなくてよい範囲を明示する — machine がスコープ外に逸れることを防ぐ}

- {除外対象1}
- {除外対象2}

## 判定基準

{何が見つかったら「問題あり」「対応必要」とするかの基準}

## 期待する出力

{出力のセクション構成・形式を具体的に指定する — 曖昧だと machine が自由に構成して後工程で使いにくくなる}

以下の形式で出力すること:
- 発見事項の一覧（テーブル形式: 項目 / 事実 or 推測 / 影響度）
- 影響範囲の評価
- 推奨アクション
```

## 計画書テンプレート（plan.md）

```markdown
# 計画書: {タスク名}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: plan

## ゴール

{達成すべき目的を1〜3文で明確に記述}

## 背景・調査結果

{investigation.md の要約。なぜこの作業が必要かの根拠}

## 対象ファイル

{変更・影響対象のファイルパスをリスト}

- {ファイル1}
- {ファイル2}

## 現状分析

{関連コードの現状把握。問題の原因やコードの構造}

## 実装方針

{具体的な変更計画をステップで記述 — PdM 自身の判断}

1. {方針1}
2. {方針2}
3. {方針3}

## 制約条件

{コーディング規約、既存APIとの整合性、パフォーマンス要件等}

- {制約1}
- {制約2}

## 完了条件

{客観的に検証可能な基準 — 「良くする」ではなく「X を Y にする」}

- {条件1}
- {条件2}

## テスト方針

{テスト対象・テストケースの大まかな方針}

## リスク

| リスク | 対策 |
|--------|------|
| {リスク1} | {対策1} |
| {リスク2} | {対策2} |
```

---

## 制約事項

- 調査計画書は MUST: PdM 自身が書く
- plan.md の判断セクション（実装方針・優先順位・制約条件）は MUST: PdM 自身が書く
- investigation.md は machine が生成するが、PdM が検証・承認するまでドラフト扱い
- `status: approved` の付いていない plan.md を Engineer に渡してはならない（NEVER）
