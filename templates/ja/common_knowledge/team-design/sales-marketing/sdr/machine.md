# SDR — machine 活用パターン

## 基本ルール

1. **計画書を先に書く** — インラインの短い指示文字列での実行は禁止。計画書ファイルを渡す
2. **出力はドラフト** — machine の出力は必ず自分で検証してから送信する
3. **保存場所**: `state/plans/{YYYY-MM-DD}_{概要}.{type}.md`（`/tmp/` 禁止）
4. **レート制限**: chat 5回/session、heartbeat 2回
5. **machine はインフラにアクセスできない** — リード情報・ナーチャリング状況は計画書に含めること

---

## 概要

SDR は2つの場面で machine を活用する:
- リード発見時の初動コンタクトメッセージのドラフト
- ナーチャリング対象へのフォローアップメールのドラフト

---

## Phase 1: リード初動ドラフト

### Step 1: リード情報を整理する

発見したリードの情報を整理する:
- 発見経緯（SNS / インバウンド / イベント等）
- BANT 評価の結果
- リードのプロフィール・関心事項

### Step 2: machine に初動メッセージのドラフトを投げる

リード情報とメッセージ方針を入力として、初動コンタクトメッセージを machine に依頼する。

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{outreach-request.md})" \
  -d /path/to/workspace
```

### Step 3: ドラフトを検証する

SDR が machine の出力を読み、`sdr/checklist.md` に沿ってセルフチェックする:

- [ ] トーンが適切か
- [ ] コンプライアンス上の問題がないか（特定電子メール法のオプトイン確認等）
- [ ] パーソナライズが適切か

問題があれば修正し、検証完了後に送信する。

## Phase 2: ナーチャリングメール

### Step 4: ナーチャリング状況を整理する

対象リードの状況を整理する:
- これまでのやり取り
- BANT 評価の変化
- リードの反応・関心の変化

### Step 5: machine にフォローアップメールのドラフトを投げる

ナーチャリング状況を入力として、フォローアップメールを machine に依頼する。

### Step 6: ドラフトを検証して送信する

SDR が machine の出力を `sdr/checklist.md` に沿って検証し、問題なければ送信する。

---

## リードレポートテンプレート（lead-report.md）

```markdown
# リードレポート: {企業名 or 個人名}

status: {new | qualified | disqualified | nurturing}
source: {inbound | outbound | sns | referral | event | other}
author: {anima名}
discovered: {YYYY-MM-DD}

## BANT 評価

| 項目 | 評価 | 根拠 |
|------|------|------|
| Budget（予算） | {あり / 不明 / なし} | {根拠} |
| Authority（決裁権） | {あり / 不明 / なし} | {根拠} |
| Need（課題・ニーズ） | {明確 / 潜在 / なし} | {根拠} |
| Timeline（導入時期） | {具体的 / 未定 / なし} | {根拠} |

## カスタムフィールド

{導入時にチーム固有の評価項目を追加}

## リード概要

{発見経緯、やり取りの要約、注目ポイント}

## 推奨アクション

{Director への提案: 商談化 / ナーチャリング継続 / 見送り / 追加調査}
```

---

## 制約事項

- machine 出力をそのまま送信してはならない（NEVER — 必ずセルフチェック後に送信）
- コンプライアンス上の懸念があるメッセージは Director に確認してから送信する（MUST）
- オプトアウト手段を含まないメール送信は禁止（NEVER）
