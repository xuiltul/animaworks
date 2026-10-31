<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/architecture/tasks.md -->
<!-- i18n: source-sha256=2eafe0a8d1b890d7718379c0b58fba494ea12cf38ee02587bbcd8c19cd559a58 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: b304b7dc

# Task Management

The source of truth for tasks is the TaskStore located at `shared/taskboard.sqlite3`. `core/tasks/board/` reads and writes canonical tasks, aliases, attempts, and leases, providing the same task information to both the CLI and Web UI. There is no separate display card layer.

## Status and Execution Records

The normal statuses are `pending`, `in_progress`, `delegated`, `done`, and `cancelled`. The actual task on the delegated side is stored in the TaskStore, and an alias row is created on the delegating side, so the same work can be tracked without double counting. The Web Task Board projects tasks into TODO, RUNNING, WAITING, and DONE columns, displaying human-registered tasks first within each column.

When execution starts, a task attempt is created, identifying the executor by a unique token and sequence number. Updates from old attempts are rejected, and results or shutdown reasons remain in the attempt history. A task lease is a mechanism that secures an actor's right to operate on a task for a certain period, and is released upon completion or cancellation. Notifications to resume incomplete executions are persisted in the wake-up outbox and marked as acknowledged after processing.

The TaskStore schema and implementation are in `core/tasks/board/tasks.py`, and the list projection is in `view.py`. `state/task_queue.jsonl` is a file related to explicit ingestion of existing data and is not the runtime source of truth.

## CLI, Delegation, and Background Tasks

`animaworks task board` is used for board listing, `list` and `show` for task information, and `add` for new registration. `claim` and `release` acquire and release leases, and `done`, `cancel`, and `note` are used to record results or reasons. `update` changes status, and `resume` re-submits with saved input. For each argument and other CLIs, see the [CLI reference](../reference/cli.md).

To hand work from one Anima to another, use `delegate_task`. The execution result of the delegate can be traced from the alias. Long-running `animaworks-tool` operations can be started as separate background tasks via `submit`, and their status and results are stored in `state/background_tasks/`. Upon completion, the requester is notified. For the list of arguments, see the [Tool CLI reference](../reference/tool-cli.md).

The external task collector gathers task candidates from integrations such as GitHub, Slack, Chatwork, and Gmail, and provides them as a snapshot along with per-source status. This is separate read-only data from Anima's runtime TaskStore.
