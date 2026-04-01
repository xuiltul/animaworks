# Legal Director — machine 活用パターン

## 基本ルール

1. **計画書を先に書く** — インラインの短い指示文字列での実行は禁止。計画書ファイルを渡す
2. **出力はドラフト** — machine の出力は必ず自分で検証し、`status: approved` にしてから次工程へ
3. **保存場所**: `state/plans/{YYYY-MM-DD}_{概要}.{type}.md`（`/tmp/` 禁止）
4. **レート制限**: chat 5回/session、heartbeat 2回
5. **machine はインフラにアクセスできない** — 記憶・メッセージ・組織情報は計画書に含めること

---

## 概要

Legal Director は PdM（計画・判断）と Engineer（実行）を兼務する。

- 分析計画（analysis-plan.md）の作成 → Director 自身が書く
- 契約書スキャンの実行 → machine に委託し、Director が検証
- リスク評価の確定 → Director 自身が判断
- 検証は2回: machine スキャン結果の確認時と、Verifier/Researcher からのフィードバック統合時

---

## Phase 1: 分析計画（PdM 相当）

### Step 1: carry-forward tracker を確認する

前回監査が存在する場合、carry-forward tracker を読み込み、全指摘事項のステータスを把握する。

### Step 2: analysis-plan.md を作成する（Director 自身が書く）

分析の目的・対象・観点・前回引き継ぎ事項を明確にした計画書を作成する。

```bash
write_memory_file(path="state/plans/{date}_{案件名}.analysis-plan.md", content="...")
```

**analysis-plan.md の「分析観点」「スコープ」「前回引き継ぎ事項」は Director の判断の核であり、machine に書かせてはならない（NEVER）。**

### Step 3: analysis-plan.md を承認する

自分で作成した analysis-plan.md を確認し、`status: approved` に変更する。

## Phase 2: 契約書スキャン（Engineer 相当）

### Step 4: machine に契約書スキャンを投げる

analysis-plan.md を入力として、契約書全文の条項分析を machine に依頼する。

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{analysis-plan.md})" \
  -d /path/to/workspace
```

結果を `state/plans/{date}_{案件名}.audit-report.md` として保存する（`status: draft`）。

**スキャン投入時の注意**:
- 契約書全文を計画書に含めること（machine は記憶にアクセスできない）
- carry-forward tracker の全指摘事項を含めること（各項目のステータス更新を machine に求める）
- 出力形式（リスクマトリクス形式）を明示すること

### Step 5: audit-report.md を検証する

Director が audit-report.md を読み、`director/checklist.md` に沿って検証する:

- [ ] carry-forward tracker の全指摘事項がカバーされているか（silent drop なし）
- [ ] 各条項のリスク評価に法的根拠があるか
- [ ] 「受容可」「追加交渉不要」の判定理由が明記されているか
- [ ] 前回より低いリスク評価の項目に変更理由があるか

問題があれば Director 自身が修正し、`status: reviewed` に変更する。

### Step 6: 委譲する

`status: reviewed` の audit-report.md を Verifier と Researcher に `delegate_task` で渡す。

## Phase 3: 統合と最終判断

### Step 7: フィードバックを統合する

Verifier（verification-report.md）と Researcher（precedent-report.md）のフィードバックを受け取り:

- 楽観バイアスの指摘を受けた項目のリスク評価を再検討する
- Researcher の裏付け結果を audit-report.md に反映する
- carry-forward tracker を最新ステータスに更新する

### Step 8: 最終報告

統合済みの audit-report.md を `status: approved` に変更し、`call_human` で人間に最終報告する。

---

## 分析計画書テンプレート（analysis-plan.md）

```markdown
# 法務分析計画書: {案件名}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: analysis-plan

## 分析目的

{何を明らかにするか — 1〜3文}

## 対象文書

| 文書 | バージョン | 受領日 |
|------|----------|--------|
| {文書名} | {版} | {日付} |

## 前回監査からの引き継ぎ事項

| # | 前回指摘 | 前回リスク | 今回確認すべき点 |
|---|---------|----------|---------------|
| C-1 | {内容} | Critical | {修正されたか？残存リスクは？} |
| ... | ... | ... | ... |

（前回監査なしの場合は「初回分析」と明記）

## 分析観点（スコープ）

{Director 自身の判断で設定する}

1. {観点1}
2. {観点2}
3. {観点3}

## スコープ外

- {除外対象}

## 出力形式

- リスクマトリクス（項目 / リスク / 根拠 / 推奨アクション）
- carry-forward tracker のステータス更新
- 交渉優先度の順位付け
- メール案（必要な場合）

## 期限

{deadline}
```

## 監査報告書テンプレート（audit-report.md）

```markdown
# 監査報告書: {案件名}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: audit-report
source: state/plans/{元の analysis-plan.md}

## 総合評価

{契約全体のリスク評価サマリー — 1〜3文}

## リスクマトリクス

| # | 条項 | リスク | 根拠 | 推奨アクション | 交渉優先度 |
|---|------|--------|------|-------------|-----------|
| 1 | {条項名・番号} | Critical | {法的根拠} | {具体的アクション} | 最優先 |
| 2 | {条項名・番号} | High | {法的根拠} | {具体的アクション} | 高 |
| ... | ... | ... | ... | ... | ... |

## 条項別分析

### {条項名}

- **条文**: {引用}
- **リスク評価**: {Critical/High/Medium/Low}
- **分析**: {法的分析}
- **推奨アクション**: {具体的な交渉ポイント}

（各条項について繰り返す）

## Carry-forward ステータス更新

| # | 前回指摘 | 前回リスク | 今回ステータス | 残存リスク |
|---|---------|----------|-------------|-----------|
| C-1 | {内容} | Critical | {解消/部分解消/未修正/悪化} | {評価} |
| ... | ... | ... | ... | ... |

## 署名前確認事項

{署名前に解決すべき事項と署名後でよい事項の区分}

## 追加コメント

{Director 自身の所見・補足}
```

---

## 制約事項

- analysis-plan.md は MUST: Director 自身が書く
- audit-report のリスク評価判断は MUST: Director 自身が確定する（machine の出力はドラフトとして検証する）
- `status: reviewed` の付いていない audit-report.md を Verifier / Researcher に渡してはならない（NEVER）
- carry-forward tracker に記録された指摘を言及なしで消滅させてはならない（NEVER — silent drop 禁止）
