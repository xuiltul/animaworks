# AnimaWorks REST API リファレンス

**[English version](api-reference.md)**

> 最終更新: 2026-03-06
> ベースURL: `http://localhost:18500`
> 関連: [spec.ja.md](spec.ja.md), [cli-reference.ja.md](cli-reference.ja.md)

---

## 認証

| 方式 | 説明 |
|------|------|
| セッションクッキー | `POST /api/auth/login` で取得した `session_token` クッキー |
| localhost 信頼 | `trust_localhost: true`（デフォルト）時、localhost からの接続は認証不要 |
| local_trust モード | `auth.json` の `mode: "local_trust"` 時、全リクエスト認証不要 |

認証不要エンドポイント: `/api/auth/login`, `/api/setup/*`, `/api/system/health`, `/api/webhooks/*`

---

## 目次

1. [認証 (Auth)](#1-認証-auth)
2. [Anima 管理](#2-anima-管理)
3. [チャット](#3-チャット)
4. [共有チャネル・DM](#4-共有チャネルdm)
5. [記憶 (Memory)](#5-記憶-memory)
6. [セッション・履歴](#6-セッション履歴)
7. [システム](#7-システム)
8. [アセット](#8-アセット)
9. [設定 (Config)](#9-設定-config)
10. [ログ](#10-ログ)
11. [ユーザー管理](#11-ユーザー管理)
12. [ツールプロンプト](#12-ツールプロンプト)
13. [セットアップ](#13-セットアップ)
14. [Webhook](#14-webhook)
15. [WebSocket](#15-websocket)
16. [内部 API](#16-内部-api)

---

## 1. 認証 (Auth)

### POST `/api/auth/login`

ログインしセッションクッキーを取得する。

**リクエスト:**

```json
{ "username": "string", "password": "string" }
```

**レスポンス:** `200 OK`

```json
{ "username": "admin", "display_name": "管理者", "role": "owner" }
```

Set-Cookie: `session_token=...`

---

### POST `/api/auth/logout`

セッションを破棄する。

**レスポンス:** `200 OK` — `{ "status": "ok" }`

---

### GET `/api/auth/me`

現在のログインユーザー情報を取得する。

**レスポンス:**

```json
{
  "username": "admin",
  "display_name": "管理者",
  "bio": "",
  "role": "owner",
  "auth_mode": "password",
  "has_password": true
}
```

---

## 2. Anima 管理

### GET `/api/animas`

全 Anima の一覧を取得する。

**レスポンス:**

```json
[
  {
    "name": "alice",
    "status": "running",
    "bootstrapping": false,
    "pid": 12345,
    "uptime_sec": 3600,
    "appearance": "anime",
    "supervisor": null,
    "speciality": "engineer",
    "role": "engineer",
    "model": "claude-opus-4-6"
  }
]
```

---

### GET `/api/animas/{name}`

Anima の詳細情報を取得する。

| パスパラメータ | 型 | 説明 |
|---------------|-----|------|
| `name` | string | Anima 名 |

**レスポンス:**

```json
{
  "status": { "enabled": true, "model": "claude-opus-4-6", "...": "..." },
  "identity": "# Alice\n...",
  "injection": "## 役割\n...",
  "state": "現在のタスク内容",
  "pending": "バックログ内容",
  "knowledge_files": ["topic1.md"],
  "episode_files": ["2026-03-05.md"],
  "procedure_files": ["deploy.md"]
}
```

---

### GET `/api/animas/{name}/config`

解決済みのモデル設定を取得する。

**レスポンス:**

```json
{
  "anima": "alice",
  "model": "claude-opus-4-6",
  "execution_mode": "S",
  "config": { "max_turns": 200, "max_chains": 10, "..." : "..." }
}
```

---

### POST `/api/animas/{name}/enable`

Anima を有効化する。

**レスポンス:** `200 OK` — `{ "name": "alice", "enabled": true }`

---

### POST `/api/animas/{name}/disable`

Anima を無効化する（約30秒でプロセス停止）。

**レスポンス:** `200 OK` — `{ "name": "alice", "enabled": false }`

---

### POST `/api/animas/{name}/start`

Anima プロセスを起動する。

**レスポンス:** `200 OK` — `{ "status": "started", "name": "alice" }`

既に起動中の場合: `{ "status": "already_running", "current_status": "..." }`

---

### POST `/api/animas/{name}/stop`

Anima プロセスを停止する。

**レスポンス:** `200 OK` — `{ "status": "stopped", "name": "alice" }`

---

### POST `/api/animas/{name}/restart`

Anima プロセスを再起動する。

**レスポンス:** `200 OK` — `{ "status": "restarted", "name": "alice", "pid": 12346 }`

---

### POST `/api/animas/{name}/interrupt`

実行中の LLM セッションを中断する。

| クエリパラメータ | 型 | デフォルト | 説明 |
|----------------|-----|-----------|------|
| `thread_id` | string | null | 中断対象スレッド ID |

**レスポンス:** `200 OK` — IPC 結果 or `{ "status": "timeout" }`

---

### POST `/api/animas/{name}/reload`

status.json のホットリロードを実行する（プロセス再起動不要）。

**レスポンス:** `200 OK` — IPC 結果

---

### POST `/api/animas/reload-all`

全 Anima の設定をホットリロードする。

**レスポンス:** `200 OK` — `{ "status": "ok", "results": { "alice": "...", "..." : "..." } }`

---

### POST `/api/animas/{name}/trigger`

ハートビートを手動でトリガーする。

**レスポンス:** `200 OK` — IPC 結果 | `504` タイムアウト

---

### GET `/api/animas/{name}/background-tasks`

バックグラウンドタスク（submit 経由の非同期ツール実行）の一覧を取得する。

**レスポンス:** `200 OK` — `{ "tasks": [{ "task_id": "...", "status": "...", "..." : "..." }] }`

---

### GET `/api/animas/{name}/background-tasks/{task_id}`

特定のバックグラウンドタスクの詳細を取得する。

---

### GET `/api/org/chart`

組織チャート（階層構造）を取得する。

| クエリパラメータ | 型 | デフォルト | 説明 |
|----------------|-----|-----------|------|
| `include_disabled` | bool | false | 無効化された Anima も含める |
| `format` | string | "json" | `json` or `text`（テキストツリー） |

**レスポンス (JSON):**

```json
{
  "generated_at": "2026-03-06T10:00:00+09:00",
  "total": 5,
  "tree": { "alice": { "children": { "bob": { "children": {} } } } },
  "flat": [{ "name": "alice", "supervisor": null, "role": "manager" }]
}
```

---

## 3. チャット

### POST `/api/animas/{name}/chat`

非ストリーミングでチャットメッセージを送信する。

**リクエスト:**

```json
{
  "message": "こんにちは",
  "from_person": "human",
  "intent": "",
  "images": [{ "data": "base64...", "media_type": "image/png" }],
  "resume": null,
  "last_event_id": null,
  "thread_id": "default"
}
```

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| `message` | string | 必須 | メッセージ本文 |
| `from_person` | string | 任意 | 送信者名（デフォルト: "human"） |
| `intent` | string | 任意 | インテント（delegation, report, question） |
| `images` | array | 任意 | 画像添付（base64 + media_type） |
| `resume` | string | 任意 | セッション再開用 ID |
| `last_event_id` | string | 任意 | SSE 再接続用イベント ID |
| `thread_id` | string | 任意 | スレッド ID（デフォルト: "default"） |

**レスポンス:**

```json
{ "response": "こんにちは！何かお手伝いできますか？", "anima": "alice", "images": [] }
```

---

### POST `/api/animas/{name}/chat/stream`

SSE（Server-Sent Events）でストリーミングチャットを行う。

**リクエスト:** `POST /api/animas/{name}/chat` と同一

**レスポンス:** `text/event-stream`

```
event: text
data: {"text": "こん", "emotion": "happy"}

event: tool_start
data: {"tool": "search_memory", "input": {"query": "..."}}

event: tool_end
data: {"tool": "search_memory", "result": "..."}

event: tool_detail
data: {"tool": "search_memory", "status": "running", "summary": "..."}

event: thinking
data: {"text": "考え中..."}

event: done
data: {"response_id": "abc123", "emotion": "neutral"}

event: error
data: {"error": "メッセージ"}
```

---

### POST `/api/animas/{name}/greet`

挨拶を生成する（会話を開始せずに応答のみ取得）。

**レスポンス:**

```json
{ "response": "おはようございます！", "emotion": "happy", "cached": false, "anima": "alice" }
```

---

### GET `/api/animas/{name}/stream/active`

現在アクティブなストリームの状態を取得する。

| クエリパラメータ | 型 | デフォルト | 説明 |
|----------------|-----|-----------|------|
| `thread_id` | string | null | スレッド ID |

**レスポンス:**

```json
{
  "active": true,
  "response_id": "abc123",
  "status": "streaming",
  "full_text": "途中までのテキスト...",
  "active_tool": "search_memory",
  "tool_history": [{ "tool": "...", "result": "..." }],
  "last_event_id": "evt_42",
  "event_count": 15,
  "emotion": "neutral"
}
```

---

### GET `/api/animas/{name}/stream/{response_id}/progress`

特定のストリームの進捗を取得する。

---

## 4. 共有チャネル・DM

### GET `/api/channels`

共有チャネルの一覧を取得する。

**レスポンス:**

```json
[
  { "name": "general", "message_count": 42, "last_post_ts": "2026-03-06T09:00:00Z" }
]
```

---

### GET `/api/channels/{name}`

チャネルのメッセージを取得する。

| クエリパラメータ | 型 | デフォルト | 説明 |
|----------------|-----|-----------|------|
| `limit` | int | 50 | 最大取得件数 |
| `offset` | int | 0 | オフセット |

**レスポンス:**

```json
{
  "channel": "general",
  "messages": [{ "from": "alice", "text": "報告です", "ts": "..." }],
  "total": 100,
  "offset": 0,
  "limit": 50,
  "has_more": true
}
```

---

### POST `/api/channels/{name}`

チャネルにメッセージを投稿する。

**リクエスト:**

```json
{ "text": "お知らせです", "from_name": "alice" }
```

**レスポンス:** `200 OK` — `{ "status": "ok", "channel": "general" }`

---

### GET `/api/channels/{name}/mentions/{anima}`

特定 Anima へのメンションを取得する。

| クエリパラメータ | 型 | デフォルト | 説明 |
|----------------|-----|-----------|------|
| `limit` | int | 10 | 最大件数 |

---

### GET `/api/dm`

DM 会話ペアの一覧を取得する。

**レスポンス:**

```json
[
  { "pair": "alice-bob", "participants": ["alice", "bob"], "message_count": 15, "last_message_ts": "..." }
]
```

---

### GET `/api/dm/{pair}`

DM 履歴を取得する（`pair` は `alice-bob` 形式）。

| クエリパラメータ | 型 | デフォルト | 説明 |
|----------------|-----|-----------|------|
| `limit` | int | 50 | 最大件数 |

---

## 5. 記憶 (Memory)

### GET `/api/animas/{name}/episodes`

エピソード記憶の一覧を取得する。

**レスポンス:** `{ "files": ["2026-03-05.md", "2026-03-04.md"] }`

---

### GET `/api/animas/{name}/episodes/{date}`

特定日のエピソード記憶を取得する。

**レスポンス:** `{ "date": "2026-03-05", "content": "# 2026-03-05\n..." }`

---

### GET `/api/animas/{name}/knowledge`

知識記憶の一覧を取得する。

**レスポンス:** `{ "files": ["deploy-procedure.md", "api-design.md"] }`

---

### GET `/api/animas/{name}/knowledge/{topic}`

特定の知識記憶を取得する。

---

### GET `/api/animas/{name}/procedures`

手続き記憶の一覧を取得する。

---

### GET `/api/animas/{name}/procedures/{proc}`

特定の手続き記憶を取得する。

---

### GET `/api/animas/{name}/conversation`

現在の会話状態を取得する。

**レスポンス:**

```json
{
  "anima": "alice",
  "total_turn_count": 24,
  "raw_turns": 24,
  "compressed_turn_count": 5,
  "has_summary": true,
  "summary_preview": "これまでの会話の要約...",
  "total_token_estimate": 15000,
  "turns": [{ "role": "user", "content": "..." }]
}
```

---

### DELETE `/api/animas/{name}/conversation`

会話履歴をクリアする。

**レスポンス:** `{ "status": "cleared", "anima": "alice" }`

---

### POST `/api/animas/{name}/conversation/compress`

会話履歴を圧縮する。

**レスポンス:**

```json
{ "compressed": true, "anima": "alice", "total_turn_count": 5, "total_token_estimate": 3000 }
```

---

### GET `/api/animas/{name}/memory/stats`

記憶の統計情報を取得する。

**レスポンス:**

```json
{
  "anima": "alice",
  "episodes": { "count": 20, "total_bytes": 45000 },
  "knowledge": { "count": 15, "total_bytes": 32000 },
  "procedures": { "count": 5, "total_bytes": 8000 }
}
```

---

## 6. セッション・履歴

### GET `/api/animas/{name}/sessions`

セッション一覧を取得する。

**レスポンス:**

```json
{
  "anima": "alice",
  "active_conversation": { "turn_count": 10, "..." : "..." },
  "threads": ["default", "thread-abc"],
  "archived_sessions": ["session_20260305_1234.json"],
  "episodes": ["2026-03-05.md"],
  "transcripts": ["2026-03-05.md"]
}
```

---

### GET `/api/animas/{name}/conversation/history`

アクティビティログベースの会話履歴を取得する。

| クエリパラメータ | 型 | デフォルト | 説明 |
|----------------|-----|-----------|------|
| `limit` | int | 50 | 最大件数 |
| `before` | string | null | このタイムスタンプ以前を取得 |
| `thread_id` | string | "default" | スレッド ID |
| `strict_thread` | bool | false | スレッド厳密一致 |

---

### GET `/api/animas/{name}/sessions/{session_id}`

アーカイブされたセッションの詳細を取得する。

---

### GET `/api/animas/{name}/transcripts/{date}`

トランスクリプト（会話記録）を取得する。

---

## 7. システム

### GET `/api/system/health`

ヘルスチェック（**認証不要**）。

**レスポンス:** `200 OK` — `{ "status": "ok" }`

---

### GET `/api/system/status`

システム全体の状態を取得する。

**レスポンス:**

```json
{
  "animas": [{ "name": "alice", "status": "running", "..." : "..." }],
  "processes": { "alice": { "pid": 12345, "uptime": 3600 } },
  "scheduler_running": true
}
```

---

### GET `/api/system/connections`

WebSocket・プロセス接続情報を取得する。

---

### GET `/api/system/scheduler`

スケジューラ（heartbeat / cron）の状態を取得する。

**レスポンス:**

```json
{
  "running": true,
  "system_jobs": [{ "id": "...", "next_run": "..." }],
  "anima_jobs": { "alice": [{ "type": "heartbeat", "next_run": "..." }] }
}
```

---

### POST `/api/system/reload`

Anima プロセスの再読み込み（新規追加・変更検出）を実行する。

**レスポンス:** `{ "added": 1, "refreshed": 2, "skipped_busy": 0, "removed": 0, "total": 5 }`

---

### GET `/api/activity/recent`

全 Anima の最近のアクティビティを取得する。

| クエリパラメータ | 型 | デフォルト | 説明 |
|----------------|-----|-----------|------|
| `hours` | int | 48 | 取得期間（時間） |
| `anima` | string | null | Anima 名でフィルタ |
| `offset` | int | 0 | オフセット |
| `limit` | int | 200 | 最大件数 |
| `event_type` | string | null | イベントタイプでフィルタ |
| `grouped` | bool | false | トリガーベースでグルーピング |
| `group_limit` | int | 50 | グループ最大数 |
| `group_offset` | int | 0 | グループオフセット |

---

### GET `/api/system/cost`

トークン使用量とコスト推定を取得する。

| クエリパラメータ | 型 | デフォルト | 説明 |
|----------------|-----|-----------|------|
| `anima` | string | null | 特定 Anima のみ |
| `days` | int | 30 | 集計期間（日） |

---

### POST `/api/system/hot-reload`

全設定（config, credentials, Slack, Anima プロセス）を一括ホットリロードする。

---

### POST `/api/system/hot-reload/slack`

Slack 接続のみリロードする。

---

### POST `/api/system/hot-reload/credentials`

認証情報のみリロードする。

---

### POST `/api/system/hot-reload/animas`

Anima プロセスのみリロードする。

---

### GET/POST `/api/system/log-level`

サーバーのログレベルを取得・設定する。

**POST リクエスト:**

```json
{ "level": "DEBUG", "logger_name": "core.memory" }
```

---

### POST `/api/settings/display-mode`

表示モード（anime / realistic）を変更する。

**リクエスト:** `{ "mode": "anime" }` or `{ "mode": "realistic" }`

---

### POST/GET `/api/system/frontend-logs`

フロントエンドログの送信・閲覧。

---

## 8. アセット

### GET `/api/animas/{name}/assets`

アセットファイルの一覧を取得する。

**レスポンス:** `{ "assets": [{ "name": "fullbody.png", "size": 245000 }] }`

---

### GET `/api/animas/{name}/assets/metadata`

アセットメタデータ（表情・アニメーション・カラー情報含む）を取得する。

---

### GET `/api/animas/{name}/assets/{filename}`

アセットファイルを取得する（ETag / 304 対応）。

---

### GET `/api/animas/{name}/attachments/{filename}`

チャット添付ファイルを取得する。

---

### GET `/api/media/proxy`

外部画像をプロキシ経由で取得する（SSRF 対策済み）。

| クエリパラメータ | 型 | 説明 |
|----------------|-----|------|
| `url` | string | プロキシ対象の画像 URL |

---

### POST `/api/animas/{name}/assets/generate`

キャラクターアセットを生成する。

**リクエスト:**

```json
{
  "prompt": "カスタムプロンプト",
  "negative_prompt": "除外ワード",
  "steps": 28,
  "skip_existing": true,
  "image_style": "anime"
}
```

---

### POST `/api/animas/{name}/assets/generate-expression`

表情アセットをオンデマンド生成する。

**リクエスト:** `{ "expression": "angry", "image_style": "anime" }`

---

### POST `/api/animas/{name}/assets/remake-preview`

リメイクプレビューを生成する（Vibe Transfer）。

**リクエスト:**

```json
{
  "style_from": "bob",
  "vibe_strength": 0.6,
  "vibe_info_extracted": 0.8,
  "prompt": null,
  "seed": null,
  "image_style": "anime",
  "backup_id": null
}
```

---

### POST `/api/animas/{name}/assets/remake-confirm`

リメイクを確定し既存アセットを置換する。

**リクエスト:** `{ "backup_id": "bk_abc123", "image_style": "anime" }`

---

### DELETE `/api/animas/{name}/assets/remake-preview`

リメイクプレビューをキャンセルしバックアップを復元する。

---

## 9. 設定 (Config)

### GET `/api/system/config`

config.json の内容を取得する（API キーはマスク済み）。

---

### GET `/api/system/init-status`

初期化状態を確認する。

**レスポンス:**

```json
{
  "checks": { "config_exists": true, "..." : "..." },
  "config_exists": true,
  "animas_count": 3,
  "api_keys": { "anthropic": true, "openai": false },
  "shared_dir_exists": true,
  "initialized": true
}
```

---

## 10. ログ

### GET `/api/system/logs`

ログファイルの一覧を取得する。

**レスポンス:**

```json
{ "files": [{ "name": "animaworks.log", "path": "...", "size_bytes": 150000, "modified": "..." }] }
```

---

### GET `/api/system/logs/stream`

SSE でログをリアルタイムストリーミングする。

| クエリパラメータ | 型 | デフォルト | 説明 |
|----------------|-----|-----------|------|
| `file` | string | "animaworks.log" | ログファイル名 |

---

### GET `/api/system/logs/{filename}`

ログファイルの内容を取得する。

| クエリパラメータ | 型 | デフォルト | 説明 |
|----------------|-----|-----------|------|
| `offset` | int | 0 | 開始行 |
| `limit` | int | 200 | 最大行数 |

---

### GET `/api/system/logs/file/read`

パス指定でログファイルを読み取る。

| クエリパラメータ | 型 | デフォルト | 説明 |
|----------------|-----|-----------|------|
| `file` | string | 必須 | ファイルパス |
| `offset` | int | 0 | 開始行 |
| `limit` | int | 200 | 最大行数 |

---

## 11. ユーザー管理

### GET `/api/users`

ユーザー一覧を取得する。

**レスポンス:**

```json
[{ "username": "admin", "display_name": "管理者", "bio": "", "role": "owner", "created_at": "..." }]
```

---

### POST `/api/users`

ユーザーを追加する（owner 権限が必要）。

**リクエスト:**

```json
{ "username": "user1", "display_name": "ユーザー1", "password": "pass", "bio": "" }
```

---

### DELETE `/api/users/{username}`

ユーザーを削除する（owner 権限が必要）。

---

### PUT `/api/users/me/password`

自分のパスワードを変更する。

**リクエスト:** `{ "current_password": "old", "new_password": "new" }`

---

## 12. ツールプロンプト

Anima のツール説明・ガイド・システムプロンプトセクションを管理する API。

### GET `/api/tool-prompts/descriptions`

全ツール説明の一覧を取得する。

### GET/PUT `/api/tool-prompts/descriptions/{name}`

個別ツール説明の取得・更新。

### GET `/api/tool-prompts/guides`

全ガイドの一覧を取得する。

### GET/PUT `/api/tool-prompts/guides/{key}`

個別ガイドの取得・更新。

### GET `/api/tool-prompts/sections`

全システムプロンプトセクションの一覧を取得する。

### GET/PUT `/api/tool-prompts/sections/{key}`

個別セクションの取得・更新。

### POST `/api/tool-prompts/preview/schema`

ツールスキーマのプレビューを生成する。

**リクエスト:** `{ "mode": "anthropic" }` — `anthropic`, `litellm`, `text` から選択

### POST `/api/tool-prompts/preview/system-prompt`

特定 Anima のシステムプロンプトをプレビューする。

**リクエスト:** `{ "anima_name": "alice" }`

**レスポンス:**

```json
{
  "anima_name": "alice",
  "execution_mode": "S",
  "system_prompt": "...",
  "token_estimate": 8500,
  "char_count": 34000
}
```

---

## 13. セットアップ

初回セットアップ用 API（**全て認証不要**）。

### GET `/api/setup/environment`

セットアップ環境情報を取得する。

**レスポンス:**

```json
{
  "claude_code_available": true,
  "locale": "ja",
  "providers": ["anthropic", "openai"],
  "available_locales": ["ja", "en"]
}
```

---

### GET `/api/setup/detect-locale`

Accept-Language ヘッダーからロケールを検出する。

---

### POST `/api/setup/validate-key`

API キーの有効性を検証する。

**リクエスト:** `{ "provider": "anthropic", "api_key": "sk-..." }`

---

### POST `/api/setup/complete`

セットアップを完了する。

**リクエスト:**

```json
{
  "locale": "ja",
  "credentials": { "anthropic": { "api_key": "sk-..." } },
  "anima": { "name": "alice", "template": "default" },
  "user": { "username": "admin", "password": "pass" },
  "image_style": "anime"
}
```

---

## 14. Webhook

外部プラットフォームからのイベント受信（**認証不要、署名検証あり**）。

### POST `/api/webhooks/slack/events`

Slack Event API からのイベントを受信する（URL 検証 challenge 対応）。

### POST `/api/webhooks/chatwork`

Chatwork Webhook からのイベントを受信する。

---

## 15. WebSocket

### `ws://HOST:PORT/ws`

メイン WebSocket。ダッシュボード・チャット UI のリアルタイム更新用。

**サーバー→クライアント イベント:**

| イベント | 説明 |
|---------|------|
| `anima_status` | Anima のステータス変更 |
| `chat_response` | チャット応答テキスト |
| `tool_activity` | ツール使用状況 |
| `heartbeat` | 定期的な生存確認（ping） |

**クライアント→サーバー:**

| メッセージ | 説明 |
|-----------|------|
| `{ "type": "pong" }` | heartbeat 応答 |

---

### `ws://HOST:PORT/ws/voice/{name}`

音声チャット WebSocket。

**認証:** 接続後に `{ "type": "auth", "token": "SESSION_TOKEN" }` を送信（localhost 信頼時は不要）。

**クライアント→サーバー:**

| 形式 | 説明 |
|------|------|
| binary | 16kHz mono 16-bit PCM 音声データ |
| `{ "type": "speech_end" }` | 発話終了（STT 実行トリガー） |
| `{ "type": "interrupt" }` | TTS 再生中断（barge-in） |
| `{ "type": "config", ... }` | 設定変更 |

**サーバー→クライアント:**

| タイプ | 形式 | 説明 |
|--------|------|------|
| `status` | JSON | セッション状態変更 |
| `transcript` | JSON | STT 結果テキスト |
| `response_text` | JSON | Anima 応答テキスト（チャンク） |
| `tts_audio` | binary | TTS 音声データ |
| `tts_start` / `tts_done` | JSON | TTS 開始/完了通知 |
| `error` | JSON | エラー通知 |

---

## 16. 内部 API

### POST `/api/internal/message-sent`

CLI から送信されたメッセージの通知（UI 更新用）。

**リクエスト:**

```json
{ "from_person": "alice", "to_person": "bob", "content": "報告です", "message_id": "msg_123" }
```

---

### GET `/api/messages/{message_id}`

保存されたメッセージを取得する。

---

## チャット UI 状態

### GET `/api/chat/ui-state`

チャット UI のペイン・タブ状態を取得する。

### PUT `/api/chat/ui-state`

チャット UI の状態を保存する。

**リクエスト:** `{ "state": { "version": 1, "active_anima": "alice", "..." : "..." } }`
