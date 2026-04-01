# Marketing Creator — machine 活用パターン

## 基本ルール

1. **計画書を先に書く** — インラインの短い指示文字列での実行は禁止。計画書ファイルを渡す
2. **出力はドラフト** — machine の出力は必ず自分で検証し、`status: draft` にしてから Director に納品
3. **保存場所**: `state/plans/{YYYY-MM-DD}_{概要}.{type}.md`（`/tmp/` 禁止）
4. **レート制限**: chat 5回/session、heartbeat 2回
5. **machine はインフラにアクセスできない** — 記憶・メッセージ・組織情報は計画書に含めること

---

## 概要

Marketing Creator は `content-plan.md` を受け取り、machine でコンテンツを制作し、セルフチェック後に Director に納品する。

---

## Phase 1: コンテンツ制作

### Step 1: content-plan.md を確認する

Director から受け取った `content-plan.md` の内容を確認する:
- 目的・ターゲット・キーメッセージ
- ファネルステージ（TOFU/MOFU/BOFU）
- 構成指示・トーン・文字数目安
- コンプライアンス注意事項

不明点があれば Director に確認する（推測で制作しない）。

### Step 2: machine にコンテンツ制作を投げる

`content-plan.md` と Brand Voice ガイドを入力として、コンテンツ制作を machine に依頼する。

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{content-request.md})" \
  -d /path/to/workspace
```

**制作指示に含めること**:
- `content-plan.md` の全内容
- Brand Voice ガイド（トーン・禁止表現・用語統一）
- ファネルステージに適した CTA の要件
- 出力形式の指定

### Step 3: draft-content.md を検証する

Creator が machine の出力を読み、`creator/checklist.md` に沿ってセルフチェックする。

問題があれば Creator 自身が修正し、`status: draft` で `draft-content.md` として保存する。

```bash
write_memory_file(path="state/plans/{date}_{タイトル}.draft-content.md", content="...")
```

### Step 4: Director に納品する

`draft-content.md` を `send_message(intent: report)` で Director に報告する。

## Phase 2: 修正対応

### Step 5: 差し戻しに対応する

Director から差し戻しがあった場合、修正指示を入力として machine に修正版の制作を依頼する。

修正版を再度 `creator/checklist.md` でセルフチェックし、`draft-content.md` を更新して Director に再納品する。

---

## ドラフトコンテンツテンプレート（draft-content.md）

```markdown
# コンテンツドラフト: {タイトル}

status: draft
plan_ref: {content-plan.md のパス}
version: {v1 | v2 | ...}
author: {anima名}
date: {YYYY-MM-DD}

## 本文

{コンテンツ本文}

## セルフチェック結果

- [ ] キーメッセージが反映されている
- [ ] ターゲットに適切なトーンである
- [ ] Brand Voice に準拠している
- [ ] コンプライアンス注意事項を遵守している
- [ ] ファネルステージに適した CTA がある

## 修正履歴

| 版 | 日付 | 修正内容 |
|----|------|---------|
| v1 | {日付} | 初稿 |
```

---

## 制約事項

- Brand Voice 準拠を machine 指示に含める（MUST）
- machine 出力をセルフチェックなしで納品してはならない（NEVER）
- `content-plan.md` の指示を逸脱した制作をしてはならない（NEVER）
