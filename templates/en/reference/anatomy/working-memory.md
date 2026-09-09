# Working Memory (state/) Technical Reference

Detailed specification of the `state/` directory that manages an Anima's working state.
Includes prompt injection logic, size control, migration, and lock control.

---

## state/ Directory Structure

```
state/
├── current_state.md          # Working memory (free-form Markdown)
├── task_results/              # TaskExec completion results
│   └── {task_id}/{attempt_token}.md
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

Task tracking is owned by the host's canonical TaskStore. Inspect it with `list_tasks`; use task tools, never direct database or queue-file writes.

### Size Control

| Parameter | Value | Source |
|-----------|-----|-------|
| Display limit | 3000 chars | `_CURRENT_STATE_MAX_CHARS` (builder.py) |
| Disk trim limit | 8000 chars (default) | `heartbeat.current_state_max_chars` (0 = disabled) |
| Inbox limit | 500 chars | `min(_state_max, 500)` in builder.py |

**Session boundaries**:

- Normal heartbeat, cron, and conversation finalization preserve `current_state.md`
- If the session summary contains a current status, it is written only when `current_state.md` is empty/idle
- Stale state with no active task may still be archived by TaskBoard housekeeping; hidden active tasks also protect state

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

## Legacy task files

`state/task_queue.jsonl` and `state/pending/` are retained only as migration/export evidence. They are not live queues. An operator must stop the old writers and explicitly import legacy tasks with a backup before activating the canonical runtime. Do not delete, replay, or fabricate these files to resume work.

## Task execution and results

The host atomically stores instructions with the task, claims eligible work, and records every attempt. `in_progress` is host-owned; agents declare `done`, `pending`, or `cancelled` through `update_task`. Use `list_tasks(detail=true)` to inspect dependencies and attention reasons. Pending does not imply retry: after resolving the cause, explicitly resume the same task with `submit_tasks(..., tasks=[{"task_id": "ID", "resume": true}])`.

Accepted results are stored under `state/task_results/{task_id}/{attempt_token}.md` (summaries up to 2,000 characters). Dependents receive the host-selected accepted result; a stale file alone is not evidence of completion. Preserve raw records and do not write results to impersonate a successful attempt.

Long-running command tools remain a separate pipeline: `animaworks-tool submit` writes `state/background_tasks/pending/` and BackgroundTaskManager owns command status and notifications. See `operations/background-tasks.md` and `operations/task-management.md`.

## read_subordinate_state

When a supervisor calls `read_subordinate_state(name="subordinate_name")`, only the subordinate's `state/current_state.md` is read (`pending.md` is not included).
