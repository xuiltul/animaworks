---
name: google-tasks-tool
description: >-
  Google Tasks連携ツール。タスクリストとタスクの一覧・追加・更新をOAuth2で行う。
  Use when: TODO一覧取得、タスク追加、完了更新、タスクリスト切り替えが必要なとき。
tags: [tasks, google, todo, external]
---

# Google Tasks ツール

Google Tasks API でタスクリスト・タスクを操作する外部ツール。

## 呼び出し方法

**Bash**: `animaworks-tool google_tasks <サブコマンド> [引数]` で実行

## アクション一覧

### list_tasklists — タスクリスト一覧
```bash
animaworks-tool google_tasks tasklists [-n 50]
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| max_results | integer | 50 | 最大取得件数 |

### list_tasks — タスク一覧
```bash
animaworks-tool google_tasks list <タスクリストID> [-n 50] [--no-completed]
```

| パラメータ | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| tasklist_id | string | Yes | タスクリスト ID |
| max_results | integer | 50 | 最大取得件数 |
| show_completed | boolean | true | 完了タスクを含めるか |

### insert_task — タスク追加
```bash
animaworks-tool google_tasks add <タスクリストID> "タスク名" [--notes メモ] [--due 日時]
```

| パラメータ | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| tasklist_id | string | Yes | タスクリスト ID |
| title | string | Yes | タスク名 |
| notes | string | No | メモ |
| due | string | No | 期限（RFC 3339） |

### insert_tasklist — タスクリスト作成
```bash
animaworks-tool google_tasks new-list "リスト名"
```

| パラメータ | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| title | string | Yes | リスト名 |

### update_task — タスク更新
指定したタスクのタイトル・メモ・期限・完了状態を更新する（指定した項目のみ更新）。

```bash
animaworks-tool google_tasks update <タスクリストID> <タスクID> [--title タイトル] [--notes メモ] [--due 日時] [--status completed|needsAction]
```

| パラメータ | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| tasklist_id | string | Yes | タスクリスト ID |
| task_id | string | Yes | タスク ID |
| title | string | No | 新しいタイトル |
| notes | string | No | メモ |
| due | string | No | 期限（RFC 3339） |
| status | string | No | `needsAction`（未完了）または `completed`（完了）。title/notes/due/status のいずれか1つ以上を指定すること。 |

### update_tasklist — タスクリスト名の更新
```bash
animaworks-tool google_tasks update-list <タスクリストID> "新しいリスト名"
```

| パラメータ | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| tasklist_id | string | Yes | タスクリスト ID |
| title | string | Yes | 新しいリスト名 |

## 注意事項

- 初回使用時に OAuth2 認証フローが必要
- credentials.json を `~/.animaworks/credentials/google_tasks/` に配置すること（Gmail/Calendar と同じ OAuth クライアントをコピー可）
