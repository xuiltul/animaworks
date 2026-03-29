---
name: gmail-tool
description: >-
  Gmail連携ツール。未読確認・本文取得・下書き作成をOAuth2でGmail API経由で行う。
  Use when: メール受信確認、本文読取、下書き作成、受信トレイ検索、ラベル付きメール操作が必要なとき。
tags: [communication, gmail, email, external]
---

# Gmail ツール

GmailのメールをOAuth2で直接操作する外部ツール。

## 呼び出し方法

**Bash**: `animaworks-tool gmail <サブコマンド> [引数]` で実行

## アクション一覧

### unread — 未読メール一覧
```bash
animaworks-tool gmail unread [-n 20]
```

### read_body — メール本文読み取り
```bash
animaworks-tool gmail read MESSAGE_ID
```

### draft — 下書き作成
```bash
animaworks-tool gmail draft --to ADDR --subject SUBJ --body BODY [--thread-id TID]
```

## CLI使用法

```bash
animaworks-tool gmail unread [-n 20]
animaworks-tool gmail read MESSAGE_ID
animaworks-tool gmail draft --to ADDR --subject SUBJ --body BODY [--thread-id TID]
```

## 注意事項

- 初回使用時にOAuth2認証フローが必要
- credentials.json と token.json が ~/.animaworks/ に配置されること
