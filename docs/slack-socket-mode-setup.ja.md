# Slack Socket Mode セットアップガイド

AnimaWorksがSlackメッセージをリアルタイムで受信するための設定手順。

## 概要

Socket ModeはSlackからWebSocket経由でイベントをpush受信する方式。
パブリックURL不要で、NAT内のサーバーでも動作する。

```
[Slack] ←WebSocket→ [SlackSocketModeManager] → Messenger.receive_external() → [Anima inbox]
                              ↑
                     server/app.py の lifespan 内でバックグラウンド起動
                     config.json external_messaging.slack で制御
```

## 前提条件

- AnimaWorksサーバーが起動していること
- 通信系オプションの依存関係が入っていること。推奨:

  ```bash
  pip install "animaworks[communication]"
  ```

  これにより `slack-sdk`（Web API・Webhook検証）、`slack-bolt`（Socket Mode）、`aiohttp` が揃う。いずれか欠けると Socket Mode または `slack` ツールが動かない。

## 1. Slack App設定（Slack管理画面）

https://api.slack.com/apps でアプリを設定する。

### Socket Mode有効化（mode=socket の場合のみ）

1. 左メニュー「Socket Mode」を選択
2. 「Enable Socket Mode」をON
3. App-Level Tokenを生成（スコープ: `connections:write`）
4. 生成された `xapp-...` トークンを控える

Webhook モード（`mode: "webhook"`）の場合はこの手順は不要。

### Event Subscriptions

1. 左メニュー「Event Subscriptions」を選択
2. 「Enable Events」をON
3. **Socket Mode の場合**: Request URLは不要
4. **Webhook の場合**: Request URL に `https://あなたのサーバー/api/webhooks/slack/events` を設定（署名検証チャレンジが自動で処理される）
5. 「Subscribe to bot events」に以下を追加:

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
| `chat:write` | メッセージ送信・更新（`slack_send` / `slack_channel_post` / `slack_channel_update`） |
| `groups:history` | プライベートチャンネル履歴読み取り |
| `groups:read` | プライベートチャンネル一覧取得 |
| `im:history` | DM履歴読み取り |
| `im:read` | DM一覧取得 |
| `im:write` | DMを開く |
| `mpim:history` | グループDM履歴読み取り |
| `mpim:read` | グループDM一覧取得 |
| `users:read` | ユーザー情報取得 |
| `app_mentions:read` | @メンション読み取り |
| `reactions:write` | リアクション追加（`slack_react`） |

Anima 表示名・アイコン付きで `chat.postMessage` する場合は、ワークスペース設定によって `chat:write.customize` が必要になることがある。

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

クレデンシャルは次の優先順位で解決される: `config.json` → vault → `shared/credentials.json` → 環境変数。

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
| `SLACK_APP_TOKEN` | `xapp-` | Socket Mode WebSocket接続確立（mode=socket の場合のみ） |

環境変数でも可。`config.json` の `credentials` セクションでも設定可能。Webhook モードでは `SLACK_APP_TOKEN` は不要。

**Per-Anima Bot 用**: `SLACK_BOT_TOKEN__{anima_name}` と `SLACK_APP_TOKEN__{anima_name}` で各Anima専用のBotを設定可能。vault または `shared/credentials.json` に追加する。

**Webhook モードで app_id_mapping を使う場合**: Per-Anima signing secret として `SLACK_SIGNING_SECRET__{anima_name}` を設定する。

## 3. config.json設定

`~/.animaworks/config.json` の `external_messaging` で制御する。

### 最小例（Slack Socket のみ）

```json
{
  "external_messaging": {
    "slack": {
      "enabled": true,
      "mode": "socket",
      "anima_mapping": {
        "C0ACT663B5L": "sakura"
      },
      "default_anima": "",
      "app_id_mapping": {}
    }
  }
}
```

### `external_messaging` 全体の補足フィールド

| キー | 型 | デフォルト | 説明 |
|------|----|-----------|------|
| `preferred_channel` | string | `"slack"` | アウトバウンドの優先チャネル（`slack` / `chatwork`）。人間通知の Slack アイコン URL 解決などに使われる |
| `user_aliases` | object | `{}` | 人間エイリアス → 連絡先。Slack 受信時の **intent 判定** に `slack_user_id` が使われる（下記） |

`user_aliases` の例（エイリアス名は任意。`slack_user_id` は Slack のメンバーID `U...`）:

```json
{
  "external_messaging": {
    "preferred_channel": "slack",
    "user_aliases": {
      "taro": { "slack_user_id": "U01234567", "chatwork_room_id": "" }
    },
    "slack": {
      "enabled": true,
      "mode": "socket",
      "anima_mapping": {},
      "default_anima": "sakura",
      "app_id_mapping": {}
    }
  }
}
```

チャンネルメッセージ本文に `<@U01234567>` のように **この ID が含まれる** と、Bot メンションと同様に `intent="question"` として扱われる（即時 inbox 処理の対象になりやすい）。DM は従来どおり常に `question`。

### `slack` サブセクション

| キー | 型 | デフォルト | 説明 |
|------|----|-----------|------|
| `enabled` | bool | `false` | Slack受信の有効/無効 |
| `mode` | string | `"socket"` | `"socket"`（推奨）または `"webhook"` |
| `anima_mapping` | object | `{}` | SlackチャンネルID → Anima名のマッピング（共有Bot用） |
| `default_anima` | string | `""` | チャンネルが anima_mapping にない場合のフォールバックAnima |
| `app_id_mapping` | object | `{}` | Slack API App ID → Anima名（Webhook モードで複数Appを使う場合） |

**共有Botのマッピングは接続を張り直さずに反映される**: `server/slack_socket.py` は受信ごとに `load_config()` し、`anima_mapping` / `default_anima` の変更は次のメッセージから有効（WebSocket は維持）。

### チャンネルIDの確認方法

Slackでチャンネル名を右クリック → 「チャンネル詳細を表示」→ 最下部にチャンネルID（`C`で始まる）。DMは `D` で始まる。

### mode の違い

| mode | 接続方向 | パブリックURL | 用途 |
|------|----------|--------------|------|
| `socket` | サーバー→Slack（WebSocket） | 不要 | NAT内サーバー（推奨） |
| `webhook` | Slack→サーバー（HTTP POST） | 必要 | 公開サーバー |

### Per-Anima Bot（Socket Mode）

各Anima専用のSlack Botを設定できる。`SLACK_BOT_TOKEN__{anima_name}` と `SLACK_APP_TOKEN__{anima_name}` を vault または `shared/credentials.json` に追加すると、そのAnima専用のSocket Mode接続が起動する。Per-Anima Botはチャンネルマッピング不要で、全メッセージがそのAnimaのinboxに届く。

```json
{
  "SLACK_BOT_TOKEN__sakura": "xoxb-...",
  "SLACK_APP_TOKEN__sakura": "xapp-..."
}
```

Per-Anima Bot の一覧は、上記 `SLACK_BOT_TOKEN__*` キーを vault / `credentials.json` から走査して決まる（`server/slack_socket.SlackSocketModeManager._discover_per_anima_bots`）。

Per-Anima Bot と共有Bot（`SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN`）は併用可能。共有Botは `anima_mapping` と `default_anima` でチャンネルベースのルーティングを行う。

### Webhook モードの追加設定

`mode: "webhook"` の場合、以下が必要:

1. **Request URL**: Slack App の Event Subscriptions で `https://あなたのサーバー/api/webhooks/slack/events` を設定
2. **署名検証用トークン**: `SLACK_SIGNING_SECRET` を `shared/credentials.json` または環境変数で設定

```json
{
  "SLACK_BOT_TOKEN": "xoxb-...",
  "SLACK_SIGNING_SECRET": "Slack管理画面の「Signing Secret」"
}
```

| キー | 用途 |
|------|------|
| `SLACK_SIGNING_SECRET` | Webhook リクエストの署名検証（リプレイ攻撃防止） |

Signing Secret は Slack App 管理画面の「Basic Information」→「App Credentials」で確認できる。

**複数Slack App（app_id_mapping）を使う場合**: 各Anima専用のSlack Appを用意し、`app_id_mapping` に `api_app_id → Anima名` を設定。api_app_id は Slack App 管理画面の「Basic Information」で確認できる（`A` で始まるID）。その場合、Per-Anima signing secret `SLACK_SIGNING_SECRET__{anima_name}` を使用する。

## 4. サーバー再起動・Slack接続の更新

初回・大きな設定変更後:

```bash
animaworks start
```

起動ログに以下が表示されれば成功:

```
INFO  animaworks.slack_socket: Shared Slack bot registered (bot_uid=U...)
INFO  animaworks.slack_socket: Slack Socket Mode connected (1 handler(s))
```

Per-Anima Bot を設定している場合は `Per-Anima Slack bot registered: {name} (bot_uid=U...)` も表示される。

**共有トークンが無く Per-Anima のみ**の場合は、共有Botの登録に失敗しても `Shared Slack bot not configured; per-Anima bots only` のあと、Per-Anima 接続だけで動作する。

無効時、または `mode: "webhook"` の場合は:

```
INFO  animaworks.slack_socket: Slack Socket Mode is disabled
```

Webhook モードでは Socket Mode は起動せず、HTTP エンドポイント `/api/webhooks/slack/events` でイベントを受信する。

### 再起動なしで Socket を差し替え（ホットリロード）

Per-Anima の `SLACK_*__name` キーを増減したり credential を更新した後、API から Socket Mode ハンドラだけ再構成できる（認証付きサーバー向け）:

- `POST /api/system/hot-reload/slack` — Slack のみ
- `POST /api/system/hot-reload` — 設定・credential・Slack・Anima まとめて

実装: `server/reload_manager.py` → `SlackSocketModeManager.reload()`（追加・削除の diff、既存接続は維持しつつ新規 Per-Anima を `connect_async`）。

## メッセージの流れ

1. Slackユーザーがマッピングされたチャンネルにメッセージ送信
2. Slack → WebSocket（Socket Mode）または HTTP POST（Webhook）でイベント受信
3. **重複抑制**: 同一メッセージで `message` と `app_mention` の両方が届くことがある。`ts` を短い TTL（約10秒）で記録し、二度目は無視する（`server/slack_socket.py` の `_is_duplicate_ts`）
4. **call_human スレッド返信**: メッセージがスレッド返信かつ `route_thread_reply` でマッピング済みの場合、元の通知を送ったAnimaのinboxにルーティング（`core/notification/reply_routing.py`）
5. **ルーティング解決**:
   - **Socket Mode Per-Anima Bot**: そのBotの全メッセージを対応Animaに直接配送
   - **Socket Mode 共有Bot**: 受信のたびに `anima_mapping.get(channel_id) or default_anima`（設定ホット反映）
   - **Webhook**: `app_id_mapping.get(api_app_id)` でAnimaを取得 → なければ `anima_mapping.get(channel_id) or default_anima`
6. **スレッドコンテキスト**: `thread_ts` がある場合、親メッセージの一行要約と返信数を `[Thread context]` ブロックとして本文先頭に付与（`conversations.replies`、最大10件相当の取得ロジック）
7. **本文整形**: `<@U...>` を表示名に展開し、Slack マークアップを平文化（`core/tools/_slack_markdown.py` の `clean_slack_markup`）
8. **アノテーション**: `[slack:DM]` またはチャンネルで Bot/エイリアスへのメンション有無を示す行を先頭に付与（`_build_slack_annotation`）
9. **intent**: Bot の `<@BOT>` または `user_aliases` に登録した `slack_user_id` へのメンション → `question`。それ以外は DM のみ `question`（`_detect_mention_intent` / `_detect_slack_intent`）
10. `Messenger.receive_external()` が `~/.animaworks/shared/inbox/{anima_name}/{msg_id}.json` にメッセージ配置
11. Animaが次のrunサイクル（heartbeat/cron/手動）でinboxを処理

## Slack ツール（`core/tools/slack.py` との関係）

受信（Socket/Webhook）とは独立に、**Slack Web API** 呼び出しは `core/tools/slack.py` をエントリに、実装は分割モジュール（`_slack_client.py`, `_slack_cache.py`, `_slack_markdown.py`, `_slack_cli.py`）に分かれている。

### スキーマ（`get_tool_schemas()`）と `dispatch()` の役割分担

- **`get_tool_schemas()`** が返すのは **`slack_channel_post`** と **`slack_channel_update`** のみ（いずれもゲート付きアクション。下記 permissions）。
- 次のスキーマ名は **`dispatch(name, args)`** で処理される（`slack_send` など）。エージェントからは主に **`use_tool`** で `tool_name: "slack"` と `action: "send"` 等を組み合わせて `slack_send` 相当を呼ぶ。`ExternalToolDispatcher` はモジュールの `get_tool_schemas()` に載った名前だけをレジストリ経由でマッチするため、`slack_send` 等の直叩きはスキーマ経路では届かない点に注意（`use_tool` はモジュールの `dispatch` を直接呼ぶ）。

| スキーマ名（`dispatch` の `name`） | 概要 |
|-----------------------------------|------|
| `slack_send` | チャンネル名または ID へ投稿。`thread_ts` 可。本文は `md_to_slack_mrkdwn`。`resolve_anima_icon_identity` による表示名・アイコン（`username` / `icon_url`）を付与可能 |
| `slack_messages` | チャンネル履歴取得。取得分を SQLite キャッシュに upsert し、キャッシュから返却 |
| `slack_search` | キャッシュ内キーワード検索（任意でチャンネル絞り込み） |
| `slack_unreplied` | 未返信スレッド検出（キャッシュベース。実行前に `auth_test`） |
| `slack_channels` | 参加チャンネル一覧 |
| `slack_react` | リアクション追加（`channel`, `emoji`, `message_ts`） |
| `slack_channel_post` | チャンネル ID 指定で投稿。本文は `md_to_slack_mrkdwn`。`slack_send` と同様に表示名・アイコンを付与。戻り値に `ts`（`slack_channel_update` 用） |
| `slack_channel_update` | 既存メッセージの非通知更新（`channel_id`, `ts`, `text`）。本文のみ `md_to_slack_mrkdwn`（投稿時のアイコン上書きは行わない） |

### トークン解決

- **`_resolve_slack_token(args)`** → **`SLACK_BOT_TOKEN__{anima名}`** を vault → `shared/credentials.json` の順で検索し、なければ **`SlackClient` の既定**（`get_credential("slack", "slack", env_var="SLACK_BOT_TOKEN")`、すなわち共有トークン）にフォールバック。
- Socket Mode 用の **`SLACK_APP_TOKEN`** とは別物。送信・履歴・検索など Web API 系は **Bot User Token（`xoxb-`）のみ** でよい。

### ゲートと `EXECUTION_PROFILE`

`slack_channel_post` / `slack_channel_update` は `slack.py` の **`EXECUTION_PROFILE`** で `gated: True`。**permissions.md**（または権限設定）に次が無いとブロックされる:

- `slack_channel_post: yes`
- `slack_channel_update: yes`

`all: yes` だけではゲートは自動解除されない（他ツールと同様の action-level 許可が必要）。

### メッセージキャッシュ

既定ディレクトリ: `~/.animaworks/cache/slack/`（`MessageCache` / SQLite）。

### CLI

`core/tools/_slack_cli.py` の **`get_cli_guide()`** にあるサブコマンド: **`channels`**, **`messages`**, **`send`**, **`search`**, **`unreplied`**。いずれも CLI では **`ANIMAWORKS_ANIMA_DIR` があると Per-Anima の `SLACK_BOT_TOKEN__{名}`** を優先。表示名・アイコン付与は **`send` のみ**（`messages` 等はトークン解決のみ）。**`react` / チャンネル ID 固定投稿 / 更新は CLI に無く**、エージェントの `slack_react`・`slack_channel_post`・`slack_channel_update`（および `use_tool` 経由）で利用する。

エントリ: `python -m core.tools.slack` または `animaworks-tool slack`（`--help` で確認）。

## 関連ファイル

| ファイル | 役割 |
|----------|------|
| `server/slack_socket.py` | `SlackSocketModeManager`（Socket Mode、Per-Anima / 共有Bot、デデュープ、スレッドコンテキスト、mention/intent） |
| `server/app.py`（lifespan 付近） | `SlackSocketModeManager` の起動・停止 |
| `server/reload_manager.py` | 設定/credential 反映時の `SlackSocketModeManager.reload()` |
| `server/routes/system.py` | `/api/system/hot-reload*` エンドポイント |
| `server/routes/webhooks.py` | Webhook（`/api/webhooks/slack/events`、署名検証、Webhook 側も上記と同様の整形・intent） |
| `core/messenger.py` | `receive_external()` — inbox配置 |
| `core/tooling/handler.py` | `use_tool` → 外部モジュールの `dispatch` を直呼び（`slack` + `action` で `slack_*` を実行） |
| `core/tooling/dispatch.py` | `ExternalToolDispatcher`（コアツールは `get_tool_schemas()` に載ったスキーマ名のみレジストリマッチ） |
| `core/notification/reply_routing.py` | call_human スレッド返信のAnimaルーティング |
| `core/config/schemas.py` | `UserAliasConfig`, `ExternalMessagingChannelConfig`, `ExternalMessagingConfig` |
| `core/tools/slack.py` | `get_tool_schemas`（channel_post/update のみ）・`dispatch`・`EXECUTION_PROFILE`・再エクスポート |
| `core/tools/_anima_icon_url.py` | `resolve_anima_icon_identity`（`slack_send` / `slack_channel_post` の表示名・アイコン） |
| `core/tools/_slack_client.py` | `SlackClient`（Web API・ページング・429リトライ） |
| `core/tools/_slack_cache.py` | `MessageCache`（SQLite） |

## トラブルシューティング

### 接続できない

- `SLACK_APP_TOKEN` が `xapp-` で始まるか確認
- Slack App設定でSocket Modeが有効か確認
- Event Subscriptionsが有効か確認
- `pip show slack-bolt aiohttp` で Socket 用依存が入っているか確認

### メッセージが届かない

- `anima_mapping` のチャンネルIDが正しいか確認
- `default_anima` を設定している場合、フォールバック先が有効か確認
- Botが対象チャンネルに招待されているか確認
- サーバーログで `"No anima mapping for channel"`（Socket Mode）または `"No anima mapping for Slack channel %s and no default_anima"`（Webhook）が出ていないか確認

### Webhook モードで署名エラー（400 Invalid signature）

- `SLACK_SIGNING_SECRET`（共通）または `SLACK_SIGNING_SECRET__{anima_name}`（app_id_mapping 時）が設定されているか確認
- Slack App 管理画面の「Basic Information」→「App Credentials」の Signing Secret と一致しているか確認
- サーバーログで `"SLACK_SIGNING_SECRET not configured"` が出ていないか確認

### ツールで ImportError（slack-sdk）

- `pip install "animaworks[communication]"` または `pip install slack-sdk`

### 再接続

- `slack-bolt` の `AsyncSocketModeHandler` は自動再接続をサポート
- WebSocket接続は約1時間で定期リフレッシュされる
- 長期稼働時にレート制限（429）が発生する場合はサーバー再起動で解消
- Per-Anima 追加だけなら `POST /api/system/hot-reload/slack` で全体再起動を避けられる

## 制約事項

- Socket ModeアプリはSlack App Directoryに公開不可（社内ツール向け）
- 最大同時WebSocket接続数: 10本/アプリ
- `apps.connections.open` のレート制限: 1回/分
- inboxに配置されたメッセージの処理はAnimaの次のrunサイクルに依存
