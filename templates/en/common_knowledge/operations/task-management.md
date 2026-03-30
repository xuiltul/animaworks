# Task Management

Operational reference for how Digital Anima receives, tracks, and completes tasks.
Search and refer to this when unsure how to proceed with tasks.

## Basic Task Management Structure

Task state is managed by files under `state/` and the task queue.

| Resource | Role |
|----------|------|
| `state/current_state.md` | Working memory (current state, observations, plans, blockers) |
| `state/pending/` directory | **LLM tasks** (JSON). Written by Heartbeat, `submit_tasks`, the Task tool, and the Agent tool. Executed by TaskExec |
| `state/task_queue.jsonl` | Persistent task queue (append-only JSONL). Tracks requests from humans and Anima |
| `state/task_results/` directory | LLM TaskExec completion summaries (`{task_id}.md`, max 2,000 characters). Auto-injected into dependent tasks. 7-day TTL |
| `state/background_tasks/pending/` | **Long-running CLI tool** wait queue. `animaworks-tool submit …` writes descriptor JSON. Picked up by `PendingTaskExecutor` (see below) |
| `state/background_tasks/{task_id}.json` | **BackgroundTaskManager** (`core/background.py`) persists tool run state. `running` → `completed` / `failed` |
| `state/background_notifications/` | Markdown notifications when long-running tools finish. Read and removed on the next Heartbeat by `drain_background_notifications()` |

`state/current_state.md` MUST always reflect the latest state.
Update it whenever task state changes.

### Three-Path Execution Model

In AnimaWorks, tasks are processed on three independent paths:

| Path | Trigger | Role | Execution scope |
|------|---------|------|-----------------|
| **Inbox** | DM received | Handle and reply to inter-Anima messages | Immediate, lightweight replies only |
| **Heartbeat** | Periodic check-in | Situation assessment and planning (Observe → Plan → Reflect) | Observation and judgment only. Execution is written to `state/pending/` (LLM) |
| **TaskExec** | LLM task appears in `state/pending/` | Run the task as an LLM session | Full execution (including tool use) |

**PendingTaskExecutor** (`core/supervisor/pending_executor.py`) watches both `state/pending/` (LLM) above and, on a **separate route**, `state/background_tasks/pending/` (`animaworks-tool submit`) in the **same watcher loop** (at most about every 3 seconds; `wake()` can also trigger immediately). The latter is handed off outside the conversation lock to **BackgroundTaskManager** (`core/background.py`), which runs only long-running external tools in the background. See `operations/background-tasks.md` for details.

**Vocabulary (do not conflate)**: Status values in `task_queue.jsonl` (`pending` / `in_progress` / `done` / `failed`, etc.) are for **operational task tracking**. The `status` field in `state/background_tasks/{task_id}.json` (`running` / `completed` / `failed`) is for **BackgroundTaskManager** execution state and follows a **different** lifecycle.

Heartbeat does **not** execute. When it finds work that must run, either delegate with `delegate_task` if you have subordinates, or enqueue with `submit_tasks` and hand off to the TaskExec path.

In MCP-integrated modes (S/C/D/G: Claude Agent SDK, Codex CLI, Cursor Agent, Gemini CLI) on the Chat path, the **Task tool** (and Agent tool) applies automatic routing:
- With subordinates → immediately delegated to the subordinate with minimum workload and best role match (same flow as `delegate_task`)
- Without subordinates, or when delegation fails → written to `state/pending/`, and the TaskExec path runs it

### Task Queue (submit_tasks / update_task / List via CLI)

The persistent task queue is recorded in `state/task_queue.jsonl` as append-only JSONL.
Use `submit_tasks` to register tasks and `update_task` to update status. To list tasks, use the CLI: `animaworks-tool task list`.
Tasks registered in the queue appear as a summary in the system prompt Priming section.

#### submit_tasks (Task Registration — Executed by You)

> **IMPORTANT**: Tasks submitted with `submit_tasks` are executed by **your own TaskExec** (they are not sent to subordinates). To assign work to a subordinate, use `delegate_task`.

Use `submit_tasks` to create and register tasks. For a single task, specify one entry in the `tasks` array.

```
submit_tasks(batch_id="human-20260313", tasks=[
  {"task_id": "t1", "title": "Monthly report", "description": "Create the monthly sales report and submit it to aoi", "parallel": true}
])
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `batch_id` | MUST | Unique batch identifier |
| `tasks[].task_id` | MUST | Unique task ID within the batch |
| `tasks[].title` | MUST | Task title (one-line summary) |
| `tasks[].description` | MUST | Original instruction text (when delegating, include verbatim quotes) |
| `tasks[].parallel` | MAY | `true` if the task may run in parallel (`true` recommended for a single task) |
| `tasks[].depends_on` | MAY | Array of predecessor task IDs |
| `tasks[].workspace` | MAY | Working directory. A workspace alias (e.g. `aischreiber`) makes TaskExec run in that directory. If omitted, the Anima default applies |

- When you receive instructions from a human, always register the task with `submit_tasks` (MUST)
- Human-origin tasks (equivalent to `source=human`) must be handled with highest priority (MUST)
- Queued tasks are reviewed on Heartbeat; when you start work, update status to `in_progress` with `update_task`

#### update_task

Updates task status. Use `done` when complete, `cancelled` when stopped, `failed` when execution fails.

```
update_task(task_id="abc123def456", status="in_progress")
update_task(task_id="abc123def456", status="done", summary="Report completed")
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `task_id` | MUST | Task ID (returned when `submit_tasks` was called) |
| `status` | MUST | `pending` / `in_progress` / `done` / `cancelled` / `blocked` / `failed` |
| `summary` | MAY | Summary after the update |

#### Task List (CLI)

List the task queue with `animaworks-tool task list`. You can filter by status.

```
Bash: animaworks-tool task list                    # All
Bash: animaworks-tool task list --status pending   # Not started
Bash: animaworks-tool task list --status in_progress
Bash: animaworks-tool task list --status done
Bash: animaworks-tool task list --status failed
```

#### Task Queue States and Markers

| State | Meaning |
|-------|---------|
| `pending` | Not started |
| `in_progress` | In progress |
| `done` | Completed |
| `cancelled` | Cancelled |
| `blocked` | Blocked |
| `failed` | Failed (e.g. TaskExec execution failed) |
| `delegated` | Delegated (tracking entry after `delegate_task` to a subordinate) |

In Priming, human-origin tasks (`source=human`) get a 🔴 HIGH marker, tasks not updated for 30+ minutes get a ⚠️ STALE marker, and overdue tasks get a 🔴 OVERDUE marker.

Delegated tasks (`delegated`) appear in a dedicated section within Priming Channel E. Live status is fetched from the subordinate's task queue (⏳ in progress / ✅ done / ❌ failed / 🚫 cancelled, etc.) showing up to 5 items. Additionally, `sync_delegated` runs automatically after each Heartbeat, detecting completed or failed tasks in subordinates' queues and auto-updating the supervisor's tracking entries (`done` / `failed`).

## Using current_state.md

`current_state.md` is working memory for your current state, observations, plans, and blockers.
It is not a task list. Track tasks in `task_queue.jsonl`.

- **Size limit**: 3,000 characters. Auto-cleaned on Heartbeat when needed
- **Idle state**: When there is no task, record `status: idle`

### Format

```markdown
status: in-progress
task: Slack integration test
assigned_by: aoi
started: 2026-02-15 10:00
context: |
  Request from aoi: Run a Slack API connectivity test and verify posting to #general works.
  Report results when the test is complete.
blockers: none
```

### Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `status` | MUST | Task state (see transitions below) |
| `task` | MUST | Short task description (one line) |
| `assigned_by` | SHOULD | Who assigned the task. Use `self` for self-initiated work |
| `started` | SHOULD | Start date/time |
| `context` | SHOULD | Task details and background |
| `blockers` | SHOULD | Blockers if any; otherwise `none` |

### Idle State

When there is no task, record:

```markdown
status: idle
```

`idle` is normal: you are waiting for the next task.
If Heartbeat sees `idle`, no action is required (`HEARTBEAT_OK`).

## Task State Transitions

Track tasks in `task_queue.jsonl`; `current_state.md` holds the working context for work in progress.

```
register via submit_tasks → update_task(status="in_progress") → work → update_task(status="done")
                                                                    ↘ blocked → report → other task
```

### Transition Steps

**Start**:
1. Pick a task from `task_queue.jsonl` and update with `update_task(task_id="...", status="in_progress")`
2. Write working context in `current_state.md` with `status: in-progress`

**Complete**:
1. Finish with `update_task(task_id="...", status="done", summary="...")`
2. Reset `current_state.md` to `status: idle`
3. Report results to the requester (MUST if `assigned_by` is someone else)

**Blocked**:
1. Set `current_state.md` `status` to `blocked` and record the specific reason in `blockers` (MUST)
2. Take steps to unblock (see blocked-task flow below)
3. If unblocking will take a while, you MAY start another task from `task_queue.jsonl`

## Managing Multiple Tasks by Priority

When several tasks exist in `task_queue.jsonl`, use this order:

1. **Human-origin first**: Tasks equivalent to `source=human` have highest priority (MUST)
2. **Supervisor tasks next**: Instructions from your supervisor outrank peer tasks at the same level (SHOULD)
3. **By deadline**: Start tasks with nearer deadlines first (SHOULD)
4. **FIFO**: Same priority and deadline → process in receive order (MAY)

### When Interrupting a Task

1. Return the current task to the queue with `update_task(status="pending")`
2. Note progress in `current_state.md`, then switch to the new task’s context

## Handling Blocked Tasks

When a task is blocked, follow these steps.

### Step 1: Identify and Record the Blocker

Record the specific cause in `blockers` in `current_state.md` (MUST).

```markdown
status: blocked
task: AWS S3 bucket configuration
blockers: |
  AWS credentials not configured.
  No aws credential in config.json.
  Need to ask aoi to set them up.
```

### Step 2: Take Action to Unblock

Act according to the cause:

| Cause | Action |
|-------|--------|
| Missing information | Message the requester with questions (SHOULD) |
| Insufficient permissions | Ask your supervisor to grant access (SHOULD) |
| External dependency | Tell the requester you are waiting (SHOULD) |
| Technical issue | Search `knowledge/` and `procedures/`. If nothing fits, report |

### Step 3: Switch to Another Task

If unblocking will take time, you MAY start the next task in `task_queue.jsonl`.
Mark the blocked task with `update_task(status="blocked")` and resume when unblocked.

## Task File Templates

### current_state.md — Idle

```markdown
status: idle
```

### current_state.md — In Progress

```markdown
status: in-progress
task: {task name}
assigned_by: {requester or self}
started: {YYYY-MM-DD HH:MM}
context: |
  {details and background}
blockers: none
```

### current_state.md — Blocked

```markdown
status: blocked
task: {task name}
assigned_by: {requester or self}
started: {YYYY-MM-DD HH:MM}
context: |
  {details and background}
blockers: |
  {specific block reason}
  {actions taken toward resolution}
```

## Recording Task Logs in episodes/

Record task start, completion, blocks, and other state changes in `episodes/` (SHOULD).
Filenames: `YYYY-MM-DD.md` (daily log).

```markdown
## 10:00 Task started: Slack integration test

Per aoi’s instructions, started Slack API connectivity test.
Confirmed slack: yes in permissions.json.

## 11:30 Task completed: Slack integration test

Slack API connectivity test done. Post to #general succeeded.
Reported results to aoi.

[IMPORTANT] Slack API rate limit: at most one message per minute.
Space out messages when sending bursts.
```

Tag important learnings with `[IMPORTANT]` (SHOULD). They are prioritized in later Heartbeat and consolidation.

## Parallel Task Execution (submit_tasks)

The `submit_tasks` tool submits multiple tasks with dependencies in one batch and runs them in parallel where possible.
TaskExec resolves dependencies as a DAG (directed acyclic graph) and runs independent tasks concurrently.

### Usage

```
submit_tasks(batch_id="build-20260301", tasks=[
  {"task_id": "compile", "title": "Compile", "description": "Build sources", "parallel": true},
  {"task_id": "lint", "title": "Lint", "description": "Static analysis", "parallel": true},
  {"task_id": "package", "title": "Package", "description": "Package build artifacts",
   "depends_on": ["compile", "lint"]}
])
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `batch_id` | MUST | Unique batch identifier |
| `tasks[].task_id` | MUST | Unique task ID within the batch |
| `tasks[].title` | MUST | Task title |
| `tasks[].description` | MUST | Work to perform |
| `tasks[].parallel` | MAY | `true` if parallel execution is allowed (default: `false`) |
| `tasks[].depends_on` | MAY | Array of predecessor task IDs |
| `tasks[].workspace` | MAY | Working directory. A workspace alias makes TaskExec run there |
| `tasks[].acceptance_criteria` | MAY | Array of completion criteria |
| `tasks[].constraints` | MAY | Array of constraints |
| `tasks[].file_paths` | MAY | Array of related file paths |

### How It Works

1. `submit_tasks` validates (unique IDs, dependencies exist, cycle detection)
2. Task files are written under `state/pending/` with `batch_id`
3. After `submit_tasks`, TaskExec (`PendingTaskExecutor`) detects tasks immediately (`wake` avoids waiting for polling)
4. TaskExec picks up the batch and orders work via topological sort
5. `parallel: true` tasks with no unmet dependencies run concurrently within the semaphore limit
6. Predecessor outputs are auto-injected into dependent task context
7. If a predecessor fails, dependents are skipped
8. Tasks not run within 24 hours of enqueue are skipped (TTL)

### Concurrency Limit

Max parallel runs are set by `config.json` `background_task.max_parallel_llm_tasks` (default: 3, allowed range 1–10).

### Task Result Storage

Completed summaries go to `state/task_results/{task_id}.md` (max 2,000 characters, 7-day TTL).
Dependents receive those results automatically. If a predecessor fails, dependents are skipped and `FAILED: {reason}` is recorded.
On each completion, the Anima that called `submit_tasks` gets a DM notification.

### When to Use submit_tasks vs delegate_task

| Scenario | Method | Executor |
|----------|--------|-------------|
| Run your own LLM tasks in the background | `submit_tasks` | **You** (TaskExec) |
| Several independent tasks in parallel | `submit_tasks` with `parallel: true` | **You** |
| Dependent task groups you run yourself | `submit_tasks` with `depends_on` | **You** |
| **Assign work to a subordinate** | **`delegate_task`** | **Subordinate** |

**Note**: Do not hand-write JSON under `state/pending/`. Always use the `submit_tasks` tool. It registers both Layer 1 (execution queue) and Layer 2 (task registry) so tracking cannot drift.

## Long-Running External Tools (BackgroundTaskManager / `core/background.py`)

Tool-guide entries marked with ⚠ (image/3D generation, `local_llm`, `run_command`, `machine_run`, etc.) should be submitted with **`animaworks-tool submit …`** so they do not block the chat loop for a long time. This path is **unrelated** to **`submit_tasks` (LLM tasks)**.

| Item | Content |
|------|---------|
| Implementation | **BackgroundTaskManager** in `core/background.py` — `submit` / `submit_async` start background work with `asyncio.create_task`, then after completion `_save_task` followed by awaiting `on_complete` (exceptions inside the callback are logged only) |
| Synchronous tools | `submit` → `run_in_executor` runs synchronous `execute_fn(tool_name, tool_args)` on a thread pool |
| Asynchronous tools | `submit_async` → `await execute_fn(tool_name, tool_args)` on the same event loop (internal / extensions) |
| Task ID | `uuid.uuid4().hex[:12]` (12 hex characters) |
| Persistence | `state/background_tasks/{task_id}.json` (UTF-8 JSON, `indent=2`). **`status` is `running` from the first save**. On success: `completed` and `result`; on exception: `failed` and `error` (both set `completed_at`) |
| Main JSON keys | `task_id`, `anima_name`, `tool_name`, `tool_args`, `status`, `created_at`, `completed_at`, `result`, `error` |
| About `pending` | `TaskStatus` defines `pending`, but `submit` / `submit_async` do not use it (state is `running` immediately after enqueue). On disk you usually see only `running` → `completed` / `failed` |
| Eligibility | `is_eligible(tool_name)` — **only whether the key exists in `_eligible_tools`**. Accepts both schema names (e.g. `generate_3d_model`) and `tool:subcommand` (e.g. `image_gen:3d`) |
| Eligible tool set | **Three-layer merge (later wins)**: (1) `_DEFAULT_ELIGIBLE_TOOLS`, (2) `get_eligible_tools_from_profiles` — each tool’s `EXECUTION_PROFILE` entries with `background_eligible: true` (keys `tool:subcommand`, values `expected_seconds`, default 60 if omitted), (3) `config.json` `background_task.eligible_tools` — map of each entry’s **`threshold_s`** as integers. **Numeric values are not used inside `is_eligible`** (for docs / profile consistency) |
| Layer-1 default keys | `generate_character_assets`, `generate_fullbody`, `generate_bustup`, `generate_icon`, `generate_chibi`, `generate_3d_model`, `generate_rigged_model`, `generate_animations` (all 30), `local_llm` / `run_command` (60), `machine_run` (600) |
| Disable | `background_task.enabled: false` in `config.json` — no manager is created (the submit queue may still be ingested with warnings on the execution side) |
| Cleanup | `cleanup_old_tasks(max_age_hours=24)` — deletes JSON whose `status` is `completed`/`failed` and `completed_at` is **older than 24 hours**. Also removes **`running`** files whose **`created_at` is over 48 hours** old (orphans after crash, etc.) |
| Internal API | `get_task` (in-memory first, else disk), `list_tasks` (merge in-memory and `*.json`, sort by `created_at` desc, optional `status` filter), `active_count` (count of `running`) |
| Operational checks | Tools **`list_background_tasks`** / **`check_background_task`** for list and single-task lookup |

The path from enqueue to completion (wait queue → notification → Heartbeat) is summarized in **`operations/background-tasks.md`**. In the same module, **`rotate_dm_logs`** archives entries in `shared/dm_logs/*.jsonl` older than `max_age_days` (default 7) by appending to `{stem}.{YYYYMMDD}.archive.jsonl` and rewriting the active file to recent entries only. That is unrelated to the operational task queue (`task_queue.jsonl`).

## Task Delegation (delegate_task / Task tool) — Executed by Subordinate

> **IMPORTANT**: `delegate_task` runs on the **subordinate’s TaskExec** (not yours). For **LLM tasks** in the background, use **`submit_tasks`**. For long-running **CLI tools**, use **`animaworks-tool submit`** (previous section and `operations/background-tasks.md`).

Anima with subordinates (supervisors) can delegate with the `delegate_task` tool.
In MCP modes (S/C/D/G) on Chat, the Task tool (and Agent tool) can delegate too. The Task tool has no parameter to name a subordinate; it picks minimum workload and role match automatically.

### How delegate_task Works

1. Task is appended to the subordinate’s task queue (`source="anima"`)
2. Task JSON is written to the subordinate’s `state/pending/` for immediate execution
3. A DM is sent to the subordinate automatically
4. Your queue gets a tracking entry (`status="delegated"`)

### Usage

```
delegate_task(name="dave", instruction="Run the API test and report the results", deadline="2d", summary="API test")
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `name` | MUST | Direct subordinate Anima name |
| `instruction` | MUST | Task instructions |
| `deadline` | MUST | Deadline. Relative `30m` / `2h` / `1d` or ISO8601 |
| `summary` | MAY | One-line summary (defaults to first 100 characters of `instruction`) |
| `workspace` | MAY | Working directory. A workspace alias makes the delegate work in that directory |

### Tracking Delegated Tasks

Use `task_tracker` to check delegated task progress.
It cross-checks the latest status from the subordinate’s `task_queue.jsonl`.

```
task_tracker()                     # Active delegated tasks (default)
task_tracker(status="all")         # All including completed
task_tracker(status="completed")   # Completed only
```

| status | Meaning |
|--------|---------|
| `active` | In progress (anything other than done/cancelled/failed). Default |
| `all` | Everything |
| `completed` | Only done/cancelled/failed |

### Receiving a Delegated Task

1. You get the delegation via DM
2. The task is auto-registered in your queue
3. Read it; if anything is unclear, ask the delegator (SHOULD)
4. When done, report back to the delegator (MUST)
