# Working Memory (state/) Technical Reference

Detailed specification of the `state/` directory that manages an Anima's working state.
Includes prompt injection logic, size control, migration, and lock control.

---

## state/ Directory Structure

```
state/
├── current_state.md          # Working memory (free-form Markdown)
├── task_queue.jsonl           # Task registry (append-only JSONL)
├── pending/                   # LLM task execution queue (JSON)
│   ├── {task_id}.json         # Submitted tasks
│   ├── processing/            # In progress (moved by PendingTaskExecutor)
│   └── failed/                # Failed tasks
├── task_results/              # TaskExec completion results
│   └── {task_id}.md           # Result summary (max 2000 chars, 7-day TTL)
├── conversation.json          # Conversation state
├── conversations/             # Per-thread conversation files
├── recovery_note.md           # Crash recovery note
├── heartbeat_checkpoint.json  # Heartbeat checkpoint
└── pending_procedures.json    # Pending procedure tracking
```

---

## current_state.md

### Role

Anima's working memory. Records in free form "what I'm doing right now," "what I observed," and "what blockers exist." It is for situational awareness, not task management.

Official task tracking and management is handled by `task_queue.jsonl` (Layer 2).

### Size Control

| Parameter | Value | Source |
|-----------|-----|-------|
| Display limit | 3000 chars | `_CURRENT_STATE_MAX_CHARS` (builder.py) |
| Disk trim limit | Disabled by default | `heartbeat.current_state_max_chars` (0 = disabled) |
| Inbox limit | 500 chars | `min(_state_max, 500)` in builder.py |

**Session boundaries**:

- Normal heartbeat, cron, and conversation finalization preserve `current_state.md`
- If the session summary contains a current status, it is written only when `current_state.md` is empty/idle
- Stale state with no active visible task may still be archived by TaskBoard housekeeping

**Optional cleanup during Heartbeat**:

1. If `heartbeat.current_state_max_chars` is greater than 0 and `current_state.md` exceeds that value before Heartbeat starts, an instruction to organize and compress is injected into the Heartbeat prompt
2. After Heartbeat or cron completes, `_enforce_state_size_limit()` is executed
3. Content exceeding the configured limit is moved to that day's episode memory (`episodes/{date}.md`) under `## current_state.md overflow archived`
4. The last configured number of characters is retained, adjusted at line breaks (if a line break exists within the first 20%, cut there)

### Prompt Injection

| Trigger | Behavior |
|---------|------|
| `chat` | Full content injected (3000 char limit, scale applied) |
| `inbox` | Limited to max 500 chars |
| `heartbeat` / `cron` | Full content injected (3000 char limit) |
| `task` | **Not injected** (Minimal tier) |

When injecting, if only `status: idle` is present, the section itself is omitted.
Otherwise it is injected with an emphasized header via the `builder/task_in_progress` template.

### Lock Control

`_state_file_lock` (`asyncio.Lock`) in `core/anima.py` prevents concurrent writes to `current_state.md`.

`_is_state_file(path)` returns `True` only for `state/current_state.md`. Writes via `write_memory_file` automatically acquire this lock for that file.

### Path Resolution (Backward Compatibility)

When `state/current_task.md` is specified in `read_memory_file` / `write_memory_file`, it is automatically resolved to `state/current_state.md` (`handler_memory.py`).

---

## pending.md (Deprecated)

`state/pending.md` was merged into `current_state.md` and is automatically deleted.

### Migration (on MemoryManager initialization)

1. If `state/current_task.md` exists and `state/current_state.md` does not → rename
2. If both exist → prefer `current_state.md`, log warning
3. If `state/pending.md` exists and has content → append to `current_state.md` under `## Migrated from pending.md`, then delete
4. If `state/pending.md` is empty → delete

### API

| Method | Behavior |
|---------|------|
| `read_pending()` | Always returns empty string `""`. Logs deprecation warning |
| `update_pending()` | No-op. Logs deprecation warning |

---

## task_queue.jsonl

Task registry. See `common_knowledge/anatomy/task-architecture.md` (Layer 2) for details.

### Entry Schema (TaskEntry)

| Field | Type | Description |
|-----------|-----|------|
| `task_id` | string | Unique ID |
| `ts` | ISO8601 | Creation timestamp |
| `source` | `"human"` / `"anima"` | Task origin |
| `original_instruction` | string | Original instruction text |
| `assignee` | string | Assignee Anima name |
| `status` | string | `pending` / `in_progress` / `done` / `cancelled` / `blocked` / `delegated` / `failed` |
| `summary` | string | One-line summary |
| `deadline` | ISO8601 / null | Deadline |
| `relay_chain` | array | Delegation chain |
| `updated_at` | ISO8601 | Last update timestamp |
| `meta` | object | `executor`, `batch_id`, `task_desc`, `origin`, etc. |

---

## pending/ Directory

LLM task execution queue. See `common_knowledge/anatomy/task-architecture.md` (Layer 1) for details.

### Lifecycle

```
pending/{task_id}.json → processing/{task_id}.json → Success: deleted / Failure: moved to failed/
```

- TTL: 24 hours (`_LLM_TASK_TTL_HOURS`). Tasks exceeding this are skipped
- Polling interval: 3 seconds (`_PENDING_WATCHER_POLL_INTERVAL`)
- Tasks with `cancelled` in `task_queue.jsonl` are automatically skipped → moved to `failed/`

### JSON Schema

| Field | Type | Required | Description |
|-----------|-----|------|------|
| `task_type` | string | Yes | `"llm"` |
| `task_id` | string | Yes | Unique ID |
| `batch_id` | string | No | Batch ID (submit_tasks) |
| `title` | string | Yes | Title |
| `description` | string | Yes | Instruction content |
| `parallel` | boolean | No | Whether parallel execution is allowed |
| `depends_on` | array | No | Preceding task IDs |
| `context` | string | No | Additional context |
| `acceptance_criteria` | array | No | Completion criteria |
| `constraints` | array | No | Constraints |
| `file_paths` | array | No | Related files |
| `workspace` | string | No | Working directory (alias) |
| `submitted_by` | string | Yes | Submitter |
| `submitted_at` | ISO8601 | Yes | Submission timestamp |
| `source` | string | No | `"delegation"`, etc. |

---

## task_results/ Directory

Stores result summaries of tasks completed by TaskExec.

| Parameter | Value |
|-----------|-----|
| File name | `{task_id}.md` |
| Max chars | 2000 (`_TASK_RESULT_MAX_CHARS`) |
| TTL | 7 days (auto-deleted by housekeeping) |

Dependent tasks (`depends_on`) automatically receive this file's content as context.

---

## read_subordinate_state

When a supervisor calls `read_subordinate_state(name="subordinate_name")`, only the subordinate's `state/current_state.md` is read (`pending.md` is not included).
