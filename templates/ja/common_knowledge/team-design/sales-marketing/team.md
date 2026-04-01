# 営業・マーケティングフルチーム — チーム概要

## 4ロール構成

| ロール | 責務 | 推奨 `--role` | `speciality` 例 | 詳細 |
|--------|------|--------------|-----------------|------|
| **Sales & Marketing Director** | 戦略策定・営業執行[machine]・パイプライン管理・コンテンツQC・最終承認 | `manager` | `sales-marketing-director` | `sales-marketing/director/` |
| **Marketing Creator** | マーケティングコンテンツ制作[machine]・Brand Voice 準拠 | `writer` | `marketing-creator` | `sales-marketing/creator/` |
| **SDR (Sales Development)** | リード開発・ナーチャリング・エンゲージメント・インバウンド対応 | `general` | `sales-development` | `sales-marketing/sdr/` |
| **Market Researcher** | 市場調査・競合分析・見込み客プロファイリング | `researcher` | `market-researcher` | `sales-marketing/researcher/` |

1つの Anima に全工程を集約すると、コンテンツ品質の自己評価バイアス・リード選別の甘さ・営業とマーケの優先度競合によるコンテキスト汚染が発生する。

各ロールディレクトリに `injection.template.md`（injection.md 雛形）、`machine.md`（machine 活用パターン、該当ロールのみ）、`checklist.md`（品質チェックリスト）がある。

> 基本原則の詳細: `team-design/guide.md`

## 2つの実行モード

### Campaign mode（計画ベース）

```
Director → content-plan.md (approved) → delegate_task
  → Marketing Creator → machine で制作 → draft-content.md (draft)
    → Director → checklist + machine QC → 承認 / 差し戻し / {COMPLIANCE_REVIEWER} コンプライアンスレビュー
```

コンテンツマーケティングの標準フロー。Director が企画し、Creator が制作し、Director が検証する。

### Engagement mode（SDR の自律巡回）

```
SDR → SNS/メール/インバウンド監視 (cron)
  → リード発見 → Director に report + lead-report.md
  → ナーチャリング対象 → machine でドラフト → SDR が検証・送信
  → CS 問い合わせ → CS チームにエスカレーション
```

SDR が cron で自律的にチャネルを巡回し、リードを発見・育成する。

## ハンドオフチェーン

```
Director → content-plan.md (approved)
  → delegate_task
    → Creator: コンテンツ制作 (machine 活用)
    → Researcher: 市場調査 (直接ツール使用)
      → draft-content.md / research-report.md (reviewed)
        → Director → machine QC + checklist → 承認 / 差し戻し
          └─ コンプライアンスリスク → {COMPLIANCE_REVIEWER} にレビュー依頼 (cross-team)
          └─ 承認 → 公開 → Campaign Tracker 更新

SDR → 自律巡回 (cron)
  → lead-report.md → Director → Deal Tracker 更新
  → 契約成立 → cs-handoff.md → CS チーム
```

### 引き継ぎドキュメント

| 送信元 → 送信先 | ドキュメント | 条件 |
|----------------|------------|------|
| Director → Creator | `content-plan.md` | `status: approved` |
| Creator → Director | `draft-content.md` | `status: draft` |
| Director → SDR | アウトバウンド指示 | `delegate_task` |
| SDR → Director | `lead-report.md` | リード発見時 |
| Director → {COMPLIANCE_REVIEWER} | `compliance-review.md` | コンプライアンスリスクフラグ |
| {COMPLIANCE_REVIEWER} → Director | 同ファイルにレビュー結果追記 | `status: reviewed` |
| Director → Researcher | 調査依頼 | `delegate_task` |
| Researcher → Director | `research-report.md` | `status: approved` |
| Director → CS チーム | `cs-handoff.md` | 契約成立時 |

### 運用ルール

- **修正サイクル**: Critical → 全体再制作 / Warning → 差分修正のみ / 3往復解消しない → 人間にエスカレーション
- **Campaign Pipeline Tracker**: コンテンツのステージを追跡する。silent drop 禁止
- **Deal Pipeline Tracker**: 商談のステージを追跡する。silent drop 禁止
- **コンプライアンスエスカレーション**: Creator/SDR 一次フィルタ → Director 二次判定 → {COMPLIANCE_REVIEWER} クロスチームレビュー
- **プロダクトマーケティング**: 新機能情報は上位（COO 等）経由で Director に伝達
- **machine 失敗時**: `current_state.md` に記録 → 次回 heartbeat で再評価

## スケーリング

| 規模 | 構成 | 備考 |
|------|------|------|
| ソロ | Director が全ロール兼務（checklist で品質担保） | SNS 投稿、簡易リサーチ |
| ペア | Director + Creator | コンテンツマーケティング中心 |
| トリオ | Director + Creator + SDR | アウトバウンド営業を含む |
| フルチーム | 本テンプレート通り4名 | フルファネルのマーケ + 営業 |

## 他チームとの対応関係

| 開発チームロール | 法務チームロール | 営業・MKT チームロール | 対応する理由 |
|----------------|----------------|-------------------|-------------|
| PdM（計画・判断） | Director（分析計画・判断） | Director（戦略・営業執行） | 「何をやるか」を決定する司令塔 |
| Engineer（実装） | Director + machine | Director + machine（営業制作） | machine で制作を実行。独立 Anima 不要 |
| Reviewer（静的検証） | Verifier（独立検証） | {COMPLIANCE_REVIEWER}（コンプライアンス） | 独立した観点での検証。クロスチーム |
| Tester（動的検証） | Researcher（根拠検証） | Researcher（市場調査） | 外部データで裏付けを取る |
| — | — | Creator（コンテンツ制作） | マーケ固有。大量コンテンツの制作特化 |
| — | — | SDR（リード開発） | 営業固有。リアルタイム監視・エンゲージメント |

## Campaign Pipeline Tracker — キャンペーン追跡表

コンテンツ/キャンペーンの制作ステージを追跡する。silent drop を構造的に防止する。

### 追跡ルール

- 新しいコンテンツ企画が発生したらこの表に登録する
- 次回 Heartbeat / レビュー時に全項目のステージを更新する
- 停滞（2週間以上ステージ変化なし）は Director に報告する
- silent drop（言及なしでの消滅）は禁止

### テンプレート

```markdown
# キャンペーン追跡表: {チーム名}

| # | 企画名 | タイプ | ファネル | ステージ | 担当 | 開始日 | 期限 | 備考 |
|---|--------|-------|---------|---------|------|--------|------|------|
| CP-1 | {名称} | {blog/email/...} | {TOFU/MOFU/BOFU} | {ステージ} | {Creator/Director} | {日付} | {日付} | {特記} |

ステージ凡例:
- 企画中: content-plan.md 作成中
- リサーチ: Researcher に調査依頼中
- 制作中: Creator が machine で制作中
- QC: Director が品質チェック中
- コンプライアンス: {COMPLIANCE_REVIEWER} レビュー中
- 承認済み: 公開待ち
- 公開済み: 配信完了
- 効果測定: パフォーマンス集計中
```

## Deal Pipeline Tracker — 商談追跡表

個別商談のセールスステージを追跡する。silent drop を構造的に防止する。

### 追跡ルール

- SDR がリードを発見したらこの表に登録する
- 次回 Heartbeat / レビュー時に全項目のステージを更新する
- 停滞（2週間以上ステージ変化なし）は原因分析を行う
- silent drop（言及なしでの消滅）は禁止

### テンプレート

```markdown
# 商談追跡表: {チーム名}

| # | 企業名 | ソース | ステージ | 担当 | 開始日 | 更新日 | 備考 |
|---|--------|-------|---------|------|--------|--------|------|
| D-1 | {名称} | {inbound/outbound/...} | {ステージ} | {SDR/Director} | {日付} | {日付} | {特記} |

ステージ凡例:
- Lead: リード獲得（未選別）
- Qualified: BANT 評価通過
- Discovery: ニーズ深掘り中
- Proposal: 提案書提出済み
- Negotiation: 条件交渉中
- Won: 受注
- Lost: 失注（理由を備考に記録）
- CS Handoff: CS チームに引き継ぎ完了
```
