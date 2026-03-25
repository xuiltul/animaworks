---
name: discord-tool
description: >-
  Discord連携ツール。メッセージ送信・取得・検索・サーバー(ギルド)一覧・チャンネル一覧・リアクション。
  「Discord」「ディスコード」「サーバー」「ギルド」「チャンネル」「リアクション」
tags: [communication, discord, external]
---

# Discord ツール

Discordのメッセージ送受信・検索・サーバー/チャンネル一覧・リアクションを行う外部ツール。

## 呼び出し方法

**Bash**: `animaworks-tool discord <サブコマンド> [引数]` で実行

## アクション一覧

### guilds — サーバー一覧
```bash
animaworks-tool discord guilds
```

### channels — チャンネル一覧
```bash
animaworks-tool discord channels GUILD_ID
```
- `GUILD_ID`: 対象ギルド（サーバー）の Snowflake ID（必須）

### send — メッセージ送信
```bash
animaworks-tool discord send CHANNEL_ID MESSAGE [--reply-to MESSAGE_ID]
```
- `CHANNEL_ID`: 送信先テキストチャンネルの Snowflake ID
- `--reply-to`: 任意。返信先メッセージ ID

### messages — メッセージ取得
```bash
animaworks-tool discord messages CHANNEL_ID [-n 20]
```

### search — メッセージ検索
```bash
animaworks-tool discord search KEYWORD [-c CHANNEL_ID] [-n 50]
```

### react — リアクション（MCP経由のみ、CLI未対応）
- 絵文字リアクションの付与。**MCP 経由のみ**。CLI では利用できない。

## CLI使用法

```bash
animaworks-tool discord guilds
animaworks-tool discord channels GUILD_ID
animaworks-tool discord send CHANNEL_ID MESSAGE [--reply-to MESSAGE_ID]
animaworks-tool discord messages CHANNEL_ID [-n 20]
animaworks-tool discord search KEYWORD [-c CHANNEL_ID] [-n 50]
```

## 注意事項

- Discord Bot Token は credentials に事前設定が必要
- チャンネルIDは Discord の開発者モードで右クリック → 「IDをコピー」で取得
- メッセージは 2000 文字制限あり
- ギルド ID・チャンネル ID・メッセージ ID はいずれも数字文字列（Snowflake ID）
- Bot には必要な権限（Send Messages, Read Message History, Add Reactions, View Channels）が必要
