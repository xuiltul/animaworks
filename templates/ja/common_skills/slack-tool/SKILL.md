---
name: slack-tool
description: >-
  Slack連携ツール。メッセージ送受信・検索・未返信確認・チャンネル一覧・絵文字リアクションを行う。
  Use when: Slackへ投稿、チャンネル一覧、スレッド返信、未返信確認、リアクション追加が必要なとき。
tags: [communication, slack, external]
---

# Slack ツール

Slackのメッセージ送受信・検索・リアクションを行う外部ツール。

## 呼び出し方法

**Bash**: `animaworks-tool slack <サブコマンド> [引数]` で実行

## アクション一覧

### send — メッセージ送信
```bash
animaworks-tool slack send CHANNEL MESSAGE [--thread TS]
```

### messages — メッセージ取得
```bash
animaworks-tool slack messages CHANNEL [-n 20]
```

### search — メッセージ検索
```bash
animaworks-tool slack search KEYWORD [-c CHANNEL] [-n 50]
```

### unreplied — 未返信メッセージ確認
```bash
animaworks-tool slack unreplied [--json]
```

### channels — チャンネル一覧
```bash
animaworks-tool slack channels
```

### react — 絵文字リアクション
- `emoji`: Slack の絵文字名（コロンなし。例: `thumbsup`, `eyes`, `white_check_mark`）
- `message_ts`: リアクション対象メッセージのタイムスタンプ（`messages` アクションの結果から取得可能）
- **注意**: `react` アクションは CLI 未対応。MCP 経由で使用する。

## CLI使用法

```bash
animaworks-tool slack send CHANNEL MESSAGE [--thread TS]
animaworks-tool slack messages CHANNEL [-n 20]
animaworks-tool slack search KEYWORD [-c CHANNEL] [-n 50]
animaworks-tool slack unreplied [--json]
animaworks-tool slack channels
```

## 注意事項

- Slack Bot Token は credentials に事前設定が必要
- チャンネルは #付きの名前またはチャンネルIDで指定
- リアクションには `reactions:write` スコープが必要
