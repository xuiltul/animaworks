# Task Architecture — 3-Layer Model

AnimaWorks task management consists of three layers.
Higher layers are more strictly managed by the system; lower layers are left to Anima's discretion.

## Overview of the 3 Layers

```
┌─────────────────────────────────────────────────┐
│  Layer 1: Execution Queue                        │  ← Most strict. Machine-processed
│  state/pending/*.json                            │
├─────────────────────────────────────────────────┤
│  Layer 2: Task Registry                          │  ← Structured. Managed via tools
│  state/task_queue.jsonl                          │
├─────────────────────────────────────────────────┤
│  Layer 3: Working Notes                          │  ← Free-form. Self-managed
│  state/current_task.md / state/pending.md        │
└─────────────────────────────────────────────────┘
```

## Layer 1: Execution Queue (state/pending/*.json)

Analogous to a message queue (SQS / RabbitMQ).

| Property | Description |
|----------|-------------|
| Format | JSON (fixed schema) |
| Lifecycle | Enqueue → consume → delete (transient) |
| Managed by | System (PendingTaskExecutor auto-consumes) |
| Writers | `submit_tasks`, `delegate_task`, SDK Task/Agent tool |
| Readers | PendingTaskExecutor (3-second polling) |

Contains the full task description (description, acceptance_criteria, constraints, depends_on, workspace, etc.).
PendingTaskExecutor detects tasks, moves them to `processing/`, executes, and deletes on completion.
Failed tasks are moved to `failed/`.

When a task has a `workspace` field, the resolved absolute path is injected as `working_directory` into the TaskExec prompt. Resolution order: task's `workspace` → `status.json` `default_workspace` → none.

Animas do not directly manipulate this layer. They write indirectly through tools.

## Layer 2: Task Registry (state/task_queue.jsonl)

Analogous to an issue tracker (Jira / GitHub Issues).

| Property | Description |
|----------|-------------|
| Format | Append-only JSONL (TaskEntry schema) |
| Lifecycle | Register → status transitions → compact to archive (persistent) |
| Managed by | Anima (via tools) + System (Priming injection, compact) |
| Writers | `submit_tasks`, `update_task`, `delegate_task` |
| Readers | `format_for_priming`, Heartbeat compact (list via CLI: animaworks-tool task list) |

Holds task summary information (task_id, summary, status, deadline, assignee).
Priming Channel E injects pending/in_progress tasks into the system prompt.
This is the official record of "what needs to be done." Human-origin tasks (source=human) must always be registered here.

## Layer 3: Working Notes (state/current_task.md, state/pending.md)

Analogous to sticky notes / personal notes.

| Property | Description |
|----------|-------------|
| Format | Markdown (free-form) |
| Lifecycle | Anima creates, updates, and deletes freely |
| Managed by | Anima (full discretion) |
| Writers | Anima (direct file operations) |
| Readers | Anima itself, Priming (current_task.md only) |

`current_task.md` is "what I'm doing right now." `pending.md` is "my personal backlog."
Content may overlap with Layer 2. Layer 3 is Anima's thinking space for organizing in their own words.

## Inter-Layer Relationships

### Data Flow

```
Human instruction ─┬─► submit_tasks ───────────────► Layer 2 (task_queue.jsonl)
                   └─► Anima writes current_task.md ► Layer 3

submit_tasks ─┬─► state/pending/*.json ──────► Layer 1 (Execution Queue)
            └─► Register in task_queue.jsonl ► Layer 2 (Task Registry)

delegate_task ─┬─► Subordinate's state/pending/ ► Layer 1
               ├─► Subordinate's task_queue.jsonl ► Layer 2
               └─► Own task_queue.jsonl ──────────► Layer 2 (status=delegated)

PendingTaskExecutor ─┬─► Success → Update task_queue to done
                     └─► Failure → Update task_queue to failed
```

### Synchronization Rules

| Event | Layer 1 | Layer 2 | Layer 3 |
|-------|---------|---------|---------|
| submit_tasks submission | JSON created | Registered as pending | — |
| delegate_task submission | JSON created (subordinate) | Registered in both queues | — |
| TaskExec completion | JSON deleted | Updated to done | — |
| TaskExec failure | Moved to failed/ | Updated to failed | — |
| Anima starts work | — | Updated to in_progress | current_task.md updated |
| Anima completes work | — | Updated to done | Reset to idle |
| After Heartbeat | — | compact runs | — |

### What Each Layer Does NOT Need to Know

- **Layer 1** does not know about Layer 2/3 (PendingTaskExecutor just consumes JSON)
- **Layer 3** does not need to know about Layer 1/2 (Anima's free notes)
- **Layer 2** bridges Layer 1 and Layer 3 as the central tracking layer

## Design Principles

1. **All tasks are registered in Layer 2**: Whether via submit_tasks or delegate_task, an entry exists in task_queue.jsonl
2. **Layer 1 is transient**: Execution queue files are deleted after consumption. Persistent records are Layer 2's responsibility
3. **Layer 2 is the SSoT**: The "official status" of a task is determined by task_queue.jsonl status
4. **Layer 3 is free**: It is Anima's working memory; the system imposes no constraints
5. **PendingTaskExecutor updates Layer 2**: On completion or failure, task_queue.jsonl status is synchronized
