---
name: google-tasks-tool
description: >-
  Google Tasks integration for listing task lists and tasks and creating or updating tasks via OAuth2.
  Use when: fetching TODO lists, adding tasks, marking items complete, or switching task lists.
tags: [tasks, google, todo, external]
---

# Google Tasks Tool

External tool for Google Tasks API: list task lists, list tasks, create tasks.

## Invocation via Bash

```bash
animaworks-tool google_tasks <subcommand> [args]
```

## Actions

### list_tasklists — List task lists
```bash
animaworks-tool google_tasks tasklists [-n 50]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_results | integer | 50 | Max results |

### list_tasks — List tasks in a task list
```bash
animaworks-tool google_tasks list <tasklist_id> [-n 50] [--no-completed]
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| tasklist_id | string | Yes | Task list ID |
| max_results | integer | 50 | Max results |
| show_completed | boolean | true | Include completed tasks |

### insert_task — Add a task
```bash
animaworks-tool google_tasks add <tasklist_id> "Task title" [--notes notes] [--due datetime]
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| tasklist_id | string | Yes | Task list ID |
| title | string | Yes | Task title |
| notes | string | No | Notes |
| due | string | No | Due date (RFC 3339) |

### insert_tasklist — Create a task list
```bash
animaworks-tool google_tasks new-list "List name"
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| title | string | Yes | List title |

### update_task — Update a task
Update a task's title, notes, due date, or status (only provided fields are updated).

```bash
animaworks-tool google_tasks update <tasklist_id> <task_id> [--title title] [--notes notes] [--due datetime] [--status completed|needsAction]
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| tasklist_id | string | Yes | Task list ID |
| task_id | string | Yes | Task ID |
| title | string | No | New title |
| notes | string | No | Notes |
| due | string | No | Due date (RFC 3339) |
| status | string | No | `needsAction` (incomplete) or `completed`. At least one of title/notes/due/status required. |

### update_tasklist — Update a task list title
```bash
animaworks-tool google_tasks update-list <tasklist_id> "New list name"
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| tasklist_id | string | Yes | Task list ID |
| title | string | Yes | New list title |

## Notes

- OAuth2 flow required on first use
- Place credentials.json at `~/.animaworks/credentials/google_tasks/` (same OAuth client as Gmail/Calendar can be copied)
