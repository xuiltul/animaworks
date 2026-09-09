# Canonical Task Architecture

## One durable authority

The host-owned TaskStore is the single source of truth for LLM tasks. Task identity, full execution input, dependencies, outcomes, execution attempts, delegation aliases, and durable wakeups are committed together where required. There is no separate file execution queue and supervisor ledger to reconcile.

Agents inspect tasks with `list_tasks` / `task_tracker` and use `submit_tasks`, `delegate_task`, and `update_task` for changes. Do not edit the database or task files directly. `backlog_task` records tracking-only work; it does not claim execution.

## Execution contract

1. A new `submit_tasks` submission atomically publishes the task and its full input. Preserve original instructions, constraints, workspace, and acceptance criteria.
2. The host checks dependencies and claims an eligible task with a unique attempt token. Only the host sets `in_progress`.
3. The agent declares `done`, `pending`, or `cancelled` through `update_task`. A stale attempt cannot complete or overwrite a newer attempt's accepted result.
4. Dependency completion and durable wakeups are host-managed, not contingent on periodic Heartbeat. Cancellation and failed/interrupted attempts retain explicit evidence and attention reasons.
5. An interrupted task is not blindly retried. After checking earlier effects and resolving the cause, explicitly resume the same nonterminal task with `submit_tasks(..., tasks=[{"task_id": "ID", "resume": true}])`. Stored input and history are retained. Duplicate delivery without resume is idempotent.

A supervisor's delegated view is an alias to the subordinate's canonical task. Status changes appear immediately in both views without a second mutable ledger or Heartbeat synchronization. A terminal dependency is not necessarily successful: cancelled work does not unlock dependents as if it were done.

## Working context and evidence

`state/current_state.md` is concise working context, not task authority. It holds observations, plans, and blockers and survives normal session boundaries. Keep durable knowledge and procedures in their own memory scopes.

TaskExec result summaries live under `state/task_results/{task_id}/{attempt_token}.md`; the store selects the accepted result reference. A filename or an old summary alone does not prove completion. Activity logs and original instructions remain evidence.

## Legacy storage and command tasks

Legacy `state/task_queue.jsonl` and `state/pending/` are migration/export evidence only. Preserve them. Migration is an explicit operator action after old writers stop and a backup is taken; arbitrary reads do not import live legacy data.

Long-running command tools are different: `animaworks-tool submit` still uses `state/background_tasks/pending/`, and BackgroundTaskManager stores command status and notifications. Do not remove that file-based pipeline when applying the LLM task contract.

See `reference/operations/task-management.md` for tool examples and `operations/background-tasks.md` for command execution.
