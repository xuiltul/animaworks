# Sales & Marketing Director — machine 活用パターン

## 基本ルール

1. **計画書を先に書く** — インラインの短い指示文字列での実行は禁止。計画書ファイルを渡す
2. **出力はドラフト** — machine の出力は必ず自分で検証し、`status: approved` にしてから次工程へ
3. **保存場所**: `state/plans/{YYYY-MM-DD}_{概要}.{type}.md`（`/tmp/` 禁止）
4. **レート制限**: chat 5回/session、heartbeat 2回
5. **machine はインフラにアクセスできない** — 記憶・メッセージ・組織情報は計画書に含めること

---

## 概要

Sales & Marketing Director は PdM（計画・判断）と Engineer（実行）を兼務する。3つのフェーズで machine を活用する。

- Phase A: Creator の draft-content.md を品質チェック → Director が最終判定
- Phase B: 営業コンテンツ（提案書・バトルカード等）を制作 → Director が検証
- Phase C: Deal Pipeline Tracker のデータを分析 → Director が判断

---

## Phase A: コンテンツ品質チェック

### Step 1: draft-content.md を受け取る

Creator から `draft-content.md`（`status: draft`）を受け取る。

### Step 2: machine に QC 分析を投げる

draft-content.md と Brand Voice ガイド（Director が管理）を入力として、品質分析を machine に依頼する。

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{qc-request.md})" \
  -d /path/to/workspace
```

**QC 分析の観点**:
- Brand Voice 準拠（トーン、禁止表現、用語統一）
- ファネルステージと CTA の整合性
- コンプライアンスリスクの有無
- ターゲットペルソナとの適合性

### Step 3: QC 結果を検証し判断する

Director が machine の QC 結果を読み、`director/checklist.md` に沿って最終判断する:

- 承認 → `status: approved` に変更し、公開 / Campaign Tracker 更新
- 差し戻し → 修正指示を Creator に `send_message` で伝達
- コンプライアンスリスク → `compliance-review.md` を作成し {COMPLIANCE_REVIEWER} にレビュー依頼

## Phase B: 営業コンテンツ制作

### Step 4: 制作指示書を作成する（Director 自身が書く）

制作するコンテンツの目的・対象・構成を明確にした指示書を作成する。

```bash
write_memory_file(path="state/plans/{date}_{概要}.sales-content-plan.md", content="...")
```

**制作指示書の「目的」「ターゲット」「差別化ポイント」は Director の判断の核であり、machine に書かせてはならない（NEVER）。**

### Step 5: machine にコンテンツ制作を投げる

指示書を入力として、営業コンテンツを machine に依頼する。

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{sales-content-plan.md})" \
  -d /path/to/workspace
```

対象コンテンツ例:
- 提案書・デモ資料
- バトルカード・反論対応ガイド
- アウトバウンドメール・フォローアップメール
- ROI 計算テンプレート

### Step 6: 営業コンテンツを検証する

Director が成果物を読み、`director/checklist.md` に沿って検証する:

- [ ] ターゲットに適したカスタマイズがされているか
- [ ] 差別化ポイントが正確か
- [ ] 競合情報が最新か
- [ ] コンプライアンス上の問題がないか

問題があれば Director 自身が修正し、`status: approved` に変更する。

## Phase C: パイプライン分析

### Step 7: Deal Pipeline Tracker を machine で分析する

Deal Pipeline Tracker のデータを入力として、分析を machine に依頼する。

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{pipeline-analysis-request.md})" \
  -d /path/to/workspace
```

**分析観点**:
- 停滞案件の検出（2週間以上ステージ変化なし）
- ステージ別のコンバージョン率
- リードソース別のパフォーマンス

### Step 8: 分析結果に基づいて判断する

Director が machine の分析結果を確認し、アクションを決定する:
- 停滞案件へのフォローアップ指示
- SDR へのアウトバウンド方針調整
- 上位への報告（重要な傾向変化）

---

## コンテンツ企画書テンプレート（content-plan.md）

```markdown
# コンテンツ企画書: {タイトル}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: content-plan
funnel_stage: {TOFU | MOFU | BOFU}

## 目的

{このコンテンツで達成したいこと — 1〜2文}

## ターゲット

{ペルソナ / 業種 / 役職 / 課題}

## キーメッセージ

{伝えるべき核心メッセージ — 1〜3点}

## 構成指示

{章立て / トーン / 文字数目安 / 参考資料 / 制約事項}

## コンプライアンス注意事項

{該当する場合: 薬機法、景品表示法、特定電子メール法 等}

## 期限

{deadline}
```

## CS 引き継ぎテンプレート（cs-handoff.md）

```markdown
# CS 引き継ぎ: {企業名}

status: draft
author: {anima名}
date: {YYYY-MM-DD}
type: cs-handoff
deal_id: {Deal Pipeline Tracker の ID}

## 顧客概要

| 項目 | 内容 |
|------|------|
| 企業名 | {名称} |
| 担当者 | {氏名・役職} |
| 契約内容 | {プラン・期間} |

## 営業プロセスの要約

{商談の経緯、決め手となったポイント}

## 合意事項・要望

{営業過程で約束した内容、カスタマイズ要件}

## 未解決事項

{引き継ぎ時点で残っている懸念}

## コミュニケーション特性

{キーパーソンの性格・好むコミュニケーションスタイル}
```

## コンプライアンスレビューテンプレート（compliance-review.md）

```markdown
# コンプライアンスレビュー: {対象コンテンツ}

status: requested
content_ref: {draft-content.md のパス}
risk_flags: {薬機法 | 景品表示法 | 特定電子メール法 | 個人情報 | other}
requested: {YYYY-MM-DD}

## レビュー対象

{対象コンテンツの要約または全文}

## 懸念事項

{Director が検出したリスクフラグの詳細}

---

## レビュー結果（{COMPLIANCE_REVIEWER} 記入）

- judgment: {APPROVE | CONDITIONAL | REJECT}

### 指摘事項

| # | 箇所 | 指摘内容 | 深刻度 | 推奨修正 |
|---|------|---------|--------|---------|
| 1 | {該当箇所} | {内容} | {Critical / Warning / Info} | {修正案} |

### 総評

{総合的な判断理由}
```

---

## 制約事項

- content-plan.md は MUST: Director 自身が書く
- 営業コンテンツの差別化ポイント・ターゲット選定は MUST: Director 自身が判断する（machine の出力はドラフトとして検証する）
- コンプライアンスリスクのあるコンテンツを {COMPLIANCE_REVIEWER} レビューなしで公開してはならない（NEVER）
- Campaign Pipeline Tracker・Deal Pipeline Tracker の項目を言及なしで消滅させてはならない（NEVER — silent drop 禁止）
