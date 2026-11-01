---
name: google-tasks-tool
description: >-
  Google Tasks 集成工具。通过 OAuth2 进行任务列表和任务的列表、添加、更新。
  Use when: 需要获取 TODO 列表、添加任务、更新完成状态、切换任务列表时。
tags: [tasks, google, todo, external]
---


# Google Tasks 工具

使用 Google Tasks API 操作任务列表和任务的外部工具。

## 调用方法

**Bash**: 使用 `animaworks-tool google_tasks <サブコマンド> [引数]` 执行

## 操作列表

### list_tasklists — 任务列表一览
```bash
animaworks-tool google_tasks tasklists [-n 50]
```

| 参数 | 类型 | 默认值 | 说明 |
|-----------|-----|-----------|------|
| max_results | integer | 50 | 最大获取数量 |

### list_tasks — 任务一览
```bash
animaworks-tool google_tasks list <タスクリストID> [-n 50] [--no-completed]
```

| 参数 | 类型 | 必填 | 说明 |
|-----------|-----|------|------|
| tasklist_id | string | Yes | 任务列表 ID |
| max_results | integer | 50 | 最大获取数量 |
| show_completed | boolean | true | 是否包含已完成任务 |

### insert_task — 添加任务
```bash
animaworks-tool google_tasks add <タスクリストID> "タスク名" [--notes メモ] [--due 日時]
```

| 参数 | 类型 | 必填 | 说明 |
|-----------|-----|------|------|
| tasklist_id | string | Yes | 任务列表 ID |
| title | string | Yes | 任务名称 |
| notes | string | No | 备注 |
| due | string | No | 截止日期（RFC 3339） |

### insert_tasklist — 创建任务列表
```bash
animaworks-tool google_tasks new-list "リスト名"
```

| 参数 | 类型 | 必填 | 说明 |
|-----------|-----|------|------|
| title | string | Yes | 列表名称 |

### update_task — 更新任务
更新指定任务的标题、备注、截止日期、完成状态（仅更新指定的项目）。

```bash
animaworks-tool google_tasks update <タスクリストID> <タスクID> [--title タイトル] [--notes メモ] [--due 日時] [--status completed|needsAction]
```

| 参数 | 类型 | 必填 | 说明 |
|-----------|-----|------|------|
| tasklist_id | string | Yes | 任务列表 ID |
| task_id | string | Yes | 任务 ID |
| title | string | No | 新标题 |
| notes | string | No | 备注 |
| due | string | No | 截止日期（RFC 3339） |
| status | string | No | `needsAction`（未完成）或 `completed`（完成）。必须指定 title/notes/due/status 中的至少一个。 |

### update_tasklist — 更新任务列表名称
```bash
animaworks-tool google_tasks update-list <タスクリストID> "新しいリスト名"
```

| 参数 | 类型 | 必填 | 说明 |
|-----------|-----|------|------|
| tasklist_id | string | Yes | 任务列表 ID |
| title | string | Yes | 新列表名称 |

## 注意事项

- 首次使用时需要 OAuth2 认证流程
- 将 credentials.json 放置在 `~/.animaworks/credentials/google_tasks/` 中（可复制与 Gmail/Calendar 相同的 OAuth 客户端）
