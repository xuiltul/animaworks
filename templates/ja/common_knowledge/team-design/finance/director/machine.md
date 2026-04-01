# Finance Director — machine 活用パターン

## 基本ルール

1. **計画書を先に書く** — インラインの短い指示文字列での実行は禁止。計画書ファイルを渡す
2. **出力はドラフト** — machine の出力は必ず自分で検証し、`status: approved` にしてから次工程へ
3. **保存場所**: `state/plans/{YYYY-MM-DD}_{概要}.{type}.md`（`/tmp/` 禁止）
4. **レート制限**: chat 5回/session、heartbeat 2回
5. **machine はインフラにアクセスできない** — 記憶・メッセージ・組織情報は計画書に含めること

---

## 概要

Finance Director は PdM（計画・判断）と Engineer（実行）を兼務する。

- 分析計画（analysis-plan.md）の作成 → Director 自身が書く
- 分析の実行 → machine に委託し、Director が検証
- リスク評価・解釈の確定 → Director 自身が判断
- 検証は2回: machine 分析結果の確認時と、Auditor からのフィードバック統合時

---

## Phase 1: 分析計画（PdM 相当）

### Step 1: Variance Tracker を確認する

前回分析が存在する場合、Variance Tracker を読み込み、全差異のステータスを把握する。

### Step 2: analysis-plan.md を作成する（Director 自身が書く）

分析の目的・対象・観点・前回引き継ぎ事項を明確にした計画書を作成する。

```bash
write_memory_file(path="state/plans/{date}_{対象名}.analysis-plan.md", content="...")
```

**analysis-plan.md の「分析観点」「スコープ」「前回引き継ぎ事項」は Director の判断の核であり、machine に書かせてはならない（NEVER）。**

### Step 3: analysis-plan.md を承認する

自分で作成した analysis-plan.md を確認し、`status: approved` に変更する。

## Phase 2: 分析実行（Engineer 相当）

### Step 4: machine に分析実行を投げる

analysis-plan.md とソースデータを入力として、分析を machine に依頼する。

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{analysis-plan.md})" \
  -d /path/to/workspace
```

結果を `state/plans/{date}_{対象名}.analysis-report.md` として保存する（`status: draft`）。

**分析投入時の注意**:
- ソースデータ（Analyst/Collector から受領したデータ）を計画書に含めること（machine は記憶にアクセスできない）
- Variance Tracker の全差異を含めること（各項目のステータス更新を machine に求める）
- 出力形式を明示すること

### Step 5: analysis-report.md を検証する

Director が analysis-report.md を読み、`director/checklist.md` に沿って検証する:

- [ ] Variance Tracker の全差異がカバーされているか（silent drop なし）
- [ ] 全数値をプログラム的に検証したか（assert 文等で恒等式・整合性を確認）
- [ ] 解釈・仮定に十分な根拠があるか
- [ ] 推定値に「推定」マーカーが付いているか

問題があれば Director 自身が修正し、`status: reviewed` に変更する。

### Step 6: 委譲する

`status: reviewed` の analysis-report.md を Auditor に `delegate_task` で渡す。

## Phase 3: 統合と最終判断

### Step 7: フィードバックを統合する

Auditor（audit-report.md）のフィードバックを受け取り:

- 仮定への指摘を受けた項目を過去データで再検証する
- Data Lineage の指摘を受けた数値のソースを再確認する
- 数値正確性の指摘があれば再計算する
- Variance Tracker を最新ステータスに更新する

### Step 8: 最終報告

統合済みの analysis-report.md を `status: approved` に変更し、`call_human` で人間に最終報告する。

---

## 分析計画書テンプレート（analysis-plan.md）

```markdown
# 財務分析計画書: {対象名}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: analysis-plan

## 分析目的

{何を明らかにするか — 1〜3文}

## 対象期間・対象法人

| 対象 | 期間 | ソースデータ |
|------|------|------------|
| {法人名/対象名} | {期間} | {データファイル} |

## 前回分析からの引き継ぎ事項

| # | 前回差異 | 前回ステータス | 今回確認すべき点 |
|---|---------|-------------|---------------|
| V-1 | {内容} | 継続監視 | {解消したか？悪化していないか？} |
| ... | ... | ... | ... |

（前回分析なしの場合は「初回分析」と明記）

## 分析観点（スコープ）

{Director 自身の判断で設定する}

1. {観点1}
2. {観点2}
3. {観点3}

## スコープ外

- {除外対象}

## 出力形式

- 主要指標サマリー
- 異常値・重要差異一覧
- 勘定科目別分析（該当する場合）
- Variance Tracker ステータス更新
- 推奨アクション

## 期限

{deadline}
```

## 分析報告書テンプレート（analysis-report.md）

```markdown
# 財務分析報告書: {対象名}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: analysis-report
source: state/plans/{元の analysis-plan.md}

## 総合評価

{分析全体のサマリー — 1〜3文}

## 主要指標サマリー

| 指標 | 当期 | 前期 | 変動率 | 評価 |
|------|------|------|--------|------|
| {指標名} | {値} | {値} | {%} | {正常/注意/異常} |

## 異常値・重要差異一覧

| # | 項目 | 変動率 | リスク | 根拠 | 推奨アクション |
|---|------|--------|--------|------|-------------|
| 1 | {項目名} | {%} | Critical | {分析根拠} | {具体的アクション} |
| 2 | {項目名} | {%} | High | {分析根拠} | {具体的アクション} |

## 勘定科目別分析

### {科目名}

- **数値**: {当期値} / {前期値} / 変動 {%}
- **分析**: {変動の原因・背景}
- **仮定**: {仮定がある場合は明記}
- **推奨アクション**: {具体的なアクション}

（各科目について繰り返す）

## Variance Tracker ステータス更新

| # | 前回差異 | 前回ステータス | 今回ステータス | 残存リスク |
|---|---------|-------------|-------------|-----------|
| V-1 | {内容} | 継続監視 | {解消/継続監視/悪化} | {評価} |

## 数値検証結果

{プログラム的検証（assert 文等）の実行結果サマリー}

## 推奨アクション

{優先度順に整理した具体的なアクション}
```

---

## 制約事項

- analysis-plan.md は MUST: Director 自身が書く
- analysis-report の解釈・判断は MUST: Director 自身が確定する（machine の出力はドラフトとして検証する）
- `status: reviewed` の付いていない analysis-report.md を Auditor に渡してはならない（NEVER）
- Variance Tracker に記録された差異を言及なしで消滅させてはならない（NEVER — silent drop 禁止）
- 全数値をプログラム的に検証してから報告する（MUST — LLM の暗算を信用しない）
