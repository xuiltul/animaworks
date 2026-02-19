# Slack Socket Mode セットアップガイド

AnimaWorksがSlackメッセージをリアルタイムで受信するための設定手順。

## 概要

Socket ModeはSlackからWebSocket経由でイベントをpush受信する方式。
パブリックURL不要で、NAT内のサーバーでも動作する。

```
[Slack] ←WebSocket→ [SlackSocketModeManager] → Messenger.receive_external() → [Anima inbox]
                              ↑
                     server/app.py lifespan で自動起動
                     config.json external_messaging.slack で制御
```

## 前提条件

- AnimaWorksサーバーが起動していること
- `slack-bolt`, `aiohttp` がインストール済み（`pyproject.toml` に含まれる）

## 1. Slack App設定（Slack管理画面）

https://api.slack.com/apps でアプリを設定する。

### Socket Mode有効化

1. 左メニュー「Socket Mode」を選択
2. 「Enable Socket Mode」をON
3. App-Level Tokenを生成（スコープ: `connections:write`）
4. 生成された `xapp-...` トークンを控える

### Event Subscriptions

1. 左メニュー「Event Subscriptions」を選択
2. 「Enable Events」をON
3. Request URLは不要（Socket Modeのため）
4. 「Subscribe to bot events」に以下を追加:

| イベント | 説明 |
|----------|------|
| `message.channels` | パブリックチャンネルのメッセージ |
| `message.groups` | プライベートチャンネルのメッセージ |
| `message.im` | ダイレクトメッセージ |
| `message.mpim` | グループDM |
| `app_mention` | @メンション |

### OAuth Scopes（Bot Token Scopes）

左メニュー「OAuth & Permissions」で以下を追加:

| スコープ | 用途 |
|----------|------|
| `channels:history` | パブリックチャンネル履歴読み取り |
| `channels:read` | チャンネル一覧取得 |
| `chat:write` | メッセージ送信 |
| `groups:history` | プライベートチャンネル履歴読み取り |
| `groups:read` | プライベートチャンネル一覧取得 |
| `im:history` | DM履歴読み取り |
| `im:read` | DM一覧取得 |
| `im:write` | DMを開く |
| `mpim:history` | グループDM履歴読み取り |
| `mpim:read` | グループDM一覧取得 |
| `users:read` | ユーザー情報取得 |
| `app_mentions:read` | @メンション読み取り |

### App Home

1. 左メニュー「App Home」を選択
2. 「Messages Tab」を有効化
3. 「Allow users to send Slash commands and messages from the messages tab」にチェック

### ワークスペースへのインストール

1. 左メニュー「Install App」
2. 「Install to Workspace」をクリック
3. 認可後に表示される `xoxb-...` トークンを控える

### Botをチャンネルに招待

受信したいチャンネルで `/invite @BotName` を実行。

## 2. クレデンシャル設定（AnimaWorks側）

`~/.animaworks/shared/credentials.json` に以下のキーを設定:

```json
{
  "SLACK_BOT_TOKEN": "xoxb-...",
  "SLACK_APP_TOKEN": "xapp-..."
}
```

| キー | prefix | 用途 |
|------|--------|------|
| `SLACK_BOT_TOKEN` | `xoxb-` | Slack API呼び出し（送信・情報取得） |
| `SLACK_APP_TOKEN` | `xapp-` | Socket Mode WebSocket接続確立 |

環境変数 `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` でも可（credentials.json優先）。

## 3. config.json設定

`~/.animaworks/config.json` に `external_messaging` セクションを追加:

```json
{
  "external_messaging": {
    "slack": {
      "enabled": true,
      "mode": "socket",
      "anima_mapping": {
        "C0ACT663B5L": "sakura"
      }
    }
  }
}
```

### 設定項目

| キー | 型 | デフォルト | 説明 |
|------|----|-----------|------|
| `enabled` | bool | `false` | Slack受信の有効/無効 |
| `mode` | string | `"socket"` | `"socket"`（推奨）または `"webhook"` |
| `anima_mapping` | object | `{}` | SlackチャンネルID → Anima名のマッピング |

### チャンネルIDの確認方法

Slackでチャンネル名を右クリック → 「チャンネル詳細を表示」→ 最下部にチャンネルID（`C`で始まる）。DMは `D` で始まる。

### mode の違い

| mode | 接続方向 | パブリックURL | 用途 |
|------|----------|--------------|------|
| `socket` | サーバー→Slack（WebSocket） | 不要 | NAT内サーバー（推奨） |
| `webhook` | Slack→サーバー（HTTP POST） | 必要 | 公開サーバー |

## 4. サーバー再起動

```bash
animaworks start
```

起動ログに以下が表示されれば成功:

```
INFO  animaworks.slack_socket: Slack Socket Mode connected
```

無効時は:

```
INFO  animaworks.slack_socket: Slack Socket Mode is disabled
```

## メッセージの流れ

1. Slackユーザーがマッピングされたチャンネルにメッセージ送信
2. Slack → WebSocket → `SlackSocketModeManager` がイベント受信
3. `anima_mapping` でチャンネルID → Anima名を解決
4. `Messenger.receive_external()` が `~/.animaworks/shared/inbox/{anima_name}/{msg_id}.json` にメッセージ配置
5. Animaが次のrunサイクル（heartbeat/cron/手動）でinboxを処理

## 関連ファイル

| ファイル | 役割 |
|----------|------|
| `server/slack_socket.py` | SlackSocketModeManager実装 |
| `server/app.py:171-179` | lifespan内での起動/停止 |
| `core/messenger.py:150-175` | `receive_external()` — inbox配置 |
| `core/config/models.py:160-172` | `ExternalMessagingChannelConfig` モデル |
| `server/routes/webhooks.py:64-113` | Webhook方式のエンドポイント（mode=webhookの場合） |
| `core/tools/slack.py` | ポーリング型ツール（send/messages/unreplied — 共存） |

## トラブルシューティング

### 接続できない

- `SLACK_APP_TOKEN` が `xapp-` で始まるか確認
- Slack App設定でSocket Modeが有効か確認
- Event Subscriptionsが有効か確認

### メッセージが届かない

- `anima_mapping` のチャンネルIDが正しいか確認
- Botが対象チャンネルに招待されているか確認
- サーバーログで `"No anima mapping for channel"` が出ていないか確認

### 再接続

- `slack-bolt` の `AsyncSocketModeHandler` は自動再接続をサポート
- WebSocket接続は約1時間で定期リフレッシュされる
- 長期稼働時にレート制限（429）が発生する場合はサーバー再起動で解消

## 制約事項

- Socket ModeアプリはSlack App Directoryに公開不可（社内ツール向け）
- 最大同時WebSocket接続数: 10本/アプリ
- `apps.connections.open` のレート制限: 1回/分
- inboxに配置されたメッセージの処理はAnimaの次のrunサイクルに依存
