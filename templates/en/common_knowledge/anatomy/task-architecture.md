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
│  Layer 3: Working Memory                         │  ← Free-form. Self-managed
│  state/current_state.md                          │
└─────────────────────────────────────────────────┘
```

## Layer 1: Execution Queue (state/pending/*.json)

Analogous to a message queue (SQS / RabbitMQ).

| Property | Description |
|----------|-------------|
| Format | JSON (fixed schema) |
| Lifecycle | Enqueue → consume → delete (transient) |
| Managed by | System (PendingTaskExecutor auto-consumes) |
| Writers | `submit_tasks`, `delegate_task` |
| Readers | PendingTaskExecutor (3-second polling) |

Contains the full task description (description, acceptance_criteria, constraints, depends_on, workspace, etc.).
PendingTaskExecutor detects tasks, moves them to `processing/`, executes, and deletes on completion.
Failed tasks are moved to `failed/`.

- **workspace**: When a task has a `workspace` field, the resolved absolute path is injected as `working_directory` into the TaskExec prompt.
- **task_results**: Completed task result summaries are saved to `state/task_results/{task_id}.md` (max 2,000 characters). Dependent tasks automatically receive these results as context. 7-day TTL for auto-deletion.

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

## Layer 3: Working Memory (state/current_state.md)

Analogous to sticky notes / personal notes.

| Property | Description |
|----------|-------------|
| Format | Markdown (free-form) |
| Lifecycle | Anima creates and updates freely. Preserved across normal boundaries; optional trim when `heartbeat.current_state_max_chars` is configured |
| Managed by | Anima (full discretion) |
| Writers | Anima (direct file operations) |
| Readers | Anima itself, Priming (current_state.md), supervisor (read_subordinate_state) |

`current_state.md` is working memory that records "what I'm doing right now," "what I observed," and "what blockers exist."
Task tracking and management are handled by Layer 2 (task_queue.jsonl).

> **pending.md is deprecated**: The former `state/pending.md` was merged into `current_state.md` and is auto-deleted. Backlog management is now unified in Layer 2.

## Inter-Layer Relationships

### Data Flow

```
Human instruction ─┬─► submit_tasks ─────────────────► Layer 2 (task_queue.jsonl)
                   └─► Anima records in current_state.md ► Layer 3

submit_tasks ─┬─► state/pending/*.json ──────► Layer 1 (Execution Queue)
            └─► Register in task_queue.jsonl ► Layer 2 (Task Registry)

delegate_task ─┬─► Subordinate's state/pending/ ► Layer 1
               ├─► Subordinate's task_queue.jsonl ► Layer 2
               └─► Own task_queue.jsonl ──────────► Layer 2 (status=delegated)

PendingTaskExecutor ─┬─► Success → Update task_queue to done
                     └─► Failure → Update task_queue to failed

update_task(status="pending") ─► Regenerate Layer 1 JSON from meta.task_desc ► Layer 1 (retry)

Session end ─┬─► Resolved tasks → Update task_queue.jsonl to done
             └─► New tasks detected → Auto-register in task_queue.jsonl
```

### Synchronization Rules

| Event | Layer 1 | Layer 2 | Layer 3 |
|-------|---------|---------|---------|
| submit_tasks submission | JSON created | Registered as pending | — |
| delegate_task submission | JSON created (subordinate) | Registered in both queues | — |
| TaskExec completion | JSON deleted | Updated to done | — |
| TaskExec failure | Moved to failed/ | Updated to failed | — |
| TaskExec retry | JSON regenerated | pending→in_progress | — |
| Session end: resolved | — | Updated to done | — |
| Session end: new tasks | — | Registered as pending | — |
| Anima starts work | — | Updated to in_progress | current_state.md updated |
| Anima completes work | — | Updated to done | Reset to idle |
| After Heartbeat | — | compact runs | — |

### What Each Layer Does NOT Need to Know

- **Layer 1** does not know about Layer 2/3 (PendingTaskExecutor just consumes JSON)
- **Layer 3** does not need to know about Layer 1/2 (Anima's working memory)
- **Layer 2** bridges Layer 1 and Layer 3 as the central tracking layer

## Design Principles

1. **All tasks are registered in Layer 2**: Whether via submit_tasks or delegate_task, an entry exists in task_queue.jsonl
2. **Layer 1 is transient**: Execution queue files are deleted after consumption. Persistent records are Layer 2's responsibility
3. **Layer 2 is the SSoT**: The "official status" of a task is determined by task_queue.jsonl status
4. **Layer 3 is free**: It is Anima's working memory; the system imposes no constraints
5. **PendingTaskExecutor updates Layer 2**: On completion or failure, task_queue.jsonl status is synchronized
