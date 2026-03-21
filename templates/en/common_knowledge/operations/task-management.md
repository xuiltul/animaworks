# Task Management

Operational reference for how Digital Anima receives, tracks, and completes tasks.
Search and refer to this when unsure how to proceed with tasks.

## Basic Task Management Structure

Task state is managed by files in the `state/` directory and the task queue.

| Resource | Role |
|----------|------|
| `state/current_state.md` | Working memory (current state, observations, plans, blockers) |
| `state/pending/` directory | LLM tasks (JSON format). Written by Heartbeat, submit_tasks, Task tool, and Agent tool. TaskExec path automatically picks them up and runs them |
| `state/task_queue.jsonl` | Persistent task queue (append-only JSONL). Tracks requests from humans and Anima |
| `state/task_results/` directory | TaskExec completion results (max 2,000 chars per `{task_id}.md`). Auto-injected into dependent tasks. 7-day TTL |

`state/current_state.md` MUST always reflect the latest state. Update it whenever the task state changes.

### Three-Path Execution Model

In AnimaWorks, tasks are processed across three independent paths:

| Path | Trigger | Role | Scope |
|------|---------|------|-------|
| **Inbox** | DM received | Process and reply to inter-Anima messages | Immediate, lightweight replies only |
| **Heartbeat** | Scheduled check | Situation assessment and planning (Observe → Plan → Reflect) | Observation and judgment only. Writes execution to `pending/` |
| **TaskExec** | Task appears in `pending/` | Execute LLM tasks | Full execution (including tool use) |

Heartbeat does **not** execute. When it finds a task that needs execution, it either delegates via `delegate_task` (if subordinates are available) or submits it via `submit_tasks` for the TaskExec path.

In MCP-integrated modes (S/C/D/G: Claude Agent SDK, Codex CLI, Cursor Agent, Gemini CLI) Chat path, the **Task tool** (and Agent tool) provides automatic routing:
- With subordinates → immediately delegated to the subordinate with minimum workload and best role match (same flow as delegate_task)
- Without subordinates, or when delegation fails → written to `state/pending/`, and TaskExec path runs it

### Task Queue (submit_tasks / update_task / List via CLI)

The persistent task queue is recorded in `state/task_queue.jsonl` in append-only JSONL format.
Use `submit_tasks` to register tasks and `update_task` to update status. To retrieve the list, use the CLI: `animaworks-tool task list`.
Tasks registered in the queue are displayed in summarized form in the system prompt Priming section.

#### submit_tasks (Task Registration — Executed by YOU)

> **IMPORTANT**: Tasks submitted via `submit_tasks` are executed by **your own TaskExec** — they are NOT sent to subordinates. To assign work to a subordinate, use `delegate_task` instead.

Use `submit_tasks` to create and register tasks. For a single task, specify one item in the tasks array.

```
submit_tasks(batch_id="human-20260313", tasks=[
  {"task_id": "t1", "title": "Monthly report creation", "description": "Create the monthly sales report and submit it to aoi", "parallel": true}
])
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `batch_id` | MUST | Unique batch identifier |
| `tasks[].task_id` | MUST | Unique task ID within the batch |
| `tasks[].title` | MUST | Task title (one-line summary) |
| `tasks[].description` | MUST | Original instruction text (include quoted text when delegating) |
| `tasks[].parallel` | MAY | `true` for parallel execution (recommended `true` for single tasks) |
| `tasks[].depends_on` | MAY | Array of predecessor task IDs |
| `tasks[].workspace` | MAY | Working directory. Workspace alias (e.g., `aischreiber`) makes TaskExec run in that directory. Omitted = Anima default |

- When receiving instructions from a human, always register with `submit_tasks` (MUST)
- Human-origin tasks must be processed with highest priority (MUST)
- Queued tasks are reviewed by Heartbeat; when starting work, update to `in_progress` with `update_task`

#### update_task

Updates task status. Set to `done` when complete, `cancelled` when aborted, `failed` when execution fails.

```
update_task(task_id="abc123def456", status="in_progress")
update_task(task_id="abc123def456", status="done", summary="Report creation completed")
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `task_id` | MUST | Task ID (returned when submit_tasks was called) |
| `status` | MUST | `pending` / `in_progress` / `done` / `cancelled` / `blocked` / `failed` |
| `summary` | MAY | Updated summary |

#### Task List (CLI)

Retrieve the task queue list with `animaworks-tool task list`. Can filter by status.

```
Bash: animaworks-tool task list                    # All tasks
Bash: animaworks-tool task list --status pending   # Pending only
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
| `failed` | Failed (when TaskExec or similar execution fails) |
| `delegated` | Delegated (tracking for tasks delegated to subordinates via delegate_task) |

In Priming display, human-originated tasks (source=human) get the 🔴 HIGH marker, tasks not updated for 30+ minutes get the ⚠️ STALE marker, and overdue tasks get the 🔴 OVERDUE marker.

## Using current_state.md

`current_state.md` is working memory that records your current state, observations, plans, and blockers.
It is not a task list. Task tracking is done in `task_queue.jsonl`.

- **Size limit**: 3,000 characters. Auto-cleaned by Heartbeat when exceeded
- **Idle state**: When there are no tasks, record `status: idle`

### Format

```markdown
status: in-progress
task: Slack integration test
assigned_by: aoi
started: 2026-02-15 10:00
context: |
  Aoi's request: Run Slack API connectivity test and verify that posting to #general works.
  Report results when the test is complete.
blockers: None
```

### Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `status` | MUST | Task state (see state transitions below) |
| `task` | MUST | Concise task description (one line) |
| `assigned_by` | SHOULD | Who assigned the task. Use `self` for self-assigned |
| `started` | SHOULD | Start date/time |
| `context` | SHOULD | Task details and background |
| `blockers` | SHOULD | Blockers if any; otherwise `None` |

### Idle State

When there is no task, record:

```markdown
status: idle
```

`idle` is a normal state meaning you are waiting for the next task.
When Heartbeat sees `idle`, no action is required (`HEARTBEAT_OK`).

## Task State Transitions

Tasks are tracked in `task_queue.jsonl`; `current_state.md` records the working context of the task in progress.

```
submit_tasks registration → update_task(status="in_progress") → work → update_task(status="done")
                                                                  ↘ blocked → report → switch to another task
```

### Transition Steps

**Start**:
1. Select a task from `task_queue.jsonl` and update with `update_task(task_id="...", status="in_progress")`
2. Write the working context in `current_state.md` with `status: in-progress`

**Complete**:
1. Mark the task done with `update_task(task_id="...", status="done", summary="...")`
2. Reset `current_state.md` to `status: idle`
3. Report results to the assigner (MUST if assigned_by is someone else)

**Blocked**:
1. Set `current_state.md` `status` to `blocked` and document the specific reason in `blockers` (MUST)
2. Take action to unblock (see blocked-task flow below)
3. If unblocking will take time, you MAY start another task from `task_queue.jsonl` (MAY)

## Managing Multiple Tasks by Priority

When multiple tasks exist in `task_queue.jsonl`, use these criteria:

1. **Human-origin tasks first**: Tasks with source=human must be processed with highest priority (MUST)
2. **Supervisor tasks next**: Instructions from supervisor take priority over other tasks at the same level (SHOULD)
3. **By deadline**: Start tasks with earlier deadlines first (SHOULD)
4. **FIFO**: Same priority and deadline → process in received order (MAY)

### When Interrupting a Task

When a higher-priority task interrupts:

1. Put the current task back in the queue with `update_task(status="pending")`
2. Memo progress in `current_state.md`, then switch to the new task's context

## Handling Blocked Tasks

When a task is blocked, follow these steps.

### Step 1: Identify and Record the Blocker

Document the specific cause in `blockers` in current_state.md (MUST).

```markdown
status: blocked
task: AWS S3 bucket setup
blockers: |
  AWS credentials not configured.
  config.json has no aws credential.
  Need to request configuration from aoi.
```

### Step 2: Take Action to Unblock

Take action according to the block cause:

| Cause | Action |
|-------|--------|
| Missing information | Send a question message to the assigner (SHOULD) |
| Insufficient permissions | Request permission addition from supervisor (SHOULD) |
| External dependency | Report to assigner that you are waiting (SHOULD) |
| Technical issue | Search knowledge/ and procedures/ for solutions. If none found, report |

### Step 3: Switch to Another Task

If unblocking will take time, you MAY start the next task from `task_queue.jsonl`.
Record the blocked task with `update_task(status="blocked")` and resume when unblocked.

## Task File Templates

### current_state.md — Idle

```markdown
status: idle
```

### current_state.md — In Progress

```markdown
status: in-progress
task: {task_name}
assigned_by: {assigner_name or self}
started: {YYYY-MM-DD HH:MM}
context: |
  {task details and background}
blockers: None
```

### current_state.md — Blocked

```markdown
status: blocked
task: {task_name}
assigned_by: {assigner_name or self}
started: {YYYY-MM-DD HH:MM}
context: |
  {task details and background}
blockers: |
  {specific block reason}
  {actions taken toward resolution}
```

## Recording Task Logs in episodes/

Record task start, completion, block, etc. in episodes/ (SHOULD).
File names: `YYYY-MM-DD.md` (daily log).

```markdown
## 10:00 Task started: Slack integration test

Started Slack API connectivity test per aoi's request.
Confirmed slack: yes in permissions.json.

## 11:30 Task completed: Slack integration test

Slack API connectivity test completed. Post test to #general succeeded.
Results reported to aoi.

[IMPORTANT] Slack API rate limit: max 1 message per minute.
Leave gaps when sending in bursts.
```

Use `[IMPORTANT]` for key learnings (SHOULD). These are prioritized during Heartbeat and memory consolidation.

## Parallel Task Execution (submit_tasks)

The `submit_tasks` tool lets you submit multiple tasks with dependencies as a batch for parallel execution.
TaskExec resolves dependencies as a DAG (directed acyclic graph) and runs independent tasks concurrently.

### Usage

```
submit_tasks(batch_id="build-20260301", tasks=[
  {{"task_id": "compile", "title": "Compile", "description": "Build the source", "parallel": true}},
  {{"task_id": "lint", "title": "Lint", "description": "Static analysis", "parallel": true}},
  {{"task_id": "package", "title": "Package", "description": "Package build artifacts",
   "depends_on": ["compile", "lint"]}}
])
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `batch_id` | MUST | Unique batch identifier |
| `tasks[].task_id` | MUST | Unique task ID within the batch |
| `tasks[].title` | MUST | Task title |
| `tasks[].description` | MUST | Work content |
| `tasks[].parallel` | MAY | `true` for parallel execution (default: `false`) |
| `tasks[].depends_on` | MAY | Array of predecessor task IDs |
| `tasks[].workspace` | MAY | Working directory. Workspace alias makes TaskExec run in that directory |
| `tasks[].acceptance_criteria` | MAY | Array of completion criteria |
| `tasks[].constraints` | MAY | Array of constraints |
| `tasks[].file_paths` | MAY | Array of related file paths |

### How It Works

1. `submit_tasks` validates (unique IDs, valid dependencies, cycle detection)
2. Task files are written to `state/pending/` with `batch_id`
3. After submit_tasks runs, TaskExec (PendingTaskExecutor) detects tasks immediately (wake — no polling wait)
4. TaskExec detects the batch and determines execution order via topological sort
5. `parallel: true` tasks with no pending dependencies run concurrently within semaphore limit
6. Predecessor results are automatically injected into dependent task context
7. If a predecessor fails, dependent tasks are skipped
8. Tasks not executed within 24 hours of submit are skipped (TTL)

### Concurrency Limit

Max parallel tasks is controlled by `config.json` `background_task.max_parallel_llm_tasks` (default: 3, range 1–10).

### Task Result Storage

Completed task result summaries are saved to `state/task_results/{task_id}.md` (max 2,000 characters, 7-day TTL).
Dependent tasks automatically receive these results as context. If a predecessor fails, dependent tasks are skipped and `FAILED: {reason}` is recorded.
When each task completes, a completion notification is sent via DM to the Anima that executed submit_tasks.

### When to Use submit_tasks vs delegate_task

| Scenario | Method | Executor |
|----------|--------|----------|
| Background tasks you execute yourself | `submit_tasks` | **You** |
| Multiple independent tasks you run in parallel | `submit_tasks` with `parallel: true` | **You** |
| Tasks with dependencies you execute | `submit_tasks` with `depends_on` | **You** |
| **Assign work to a subordinate** | **`delegate_task`** | **Subordinate** |

**Important**: Do not manually write JSON files to `state/pending/`. Always submit via the `submit_tasks` tool. `submit_tasks` registers in both Layer 1 (execution queue) and Layer 2 (task registry) simultaneously, preventing task tracking gaps.

## Task Delegation (delegate_task / Task tool) — Executed by Subordinate

> **IMPORTANT**: `delegate_task` causes the **subordinate's TaskExec** to execute the task (not yours). To run tasks yourself in background, use `submit_tasks` instead.

Anima with subordinates (supervisors) can delegate tasks to subordinates using the `delegate_task` tool.
In MCP-integrated modes (S/C/D/G) Chat path, the Task tool (and Agent tool) also supports delegation. The Task tool has no parameter to specify a subordinate; it auto-selects the subordinate with minimum workload and best role match.

### How delegate_task Works

1. Task is added to subordinate's task queue (source="anima")
2. Task JSON is written to subordinate's `state/pending/` for immediate execution
3. DM is automatically sent to subordinate
4. Tracking entry is created in your queue (status="delegated")

### Usage

```
delegate_task(name="dave", instruction="Run API test and report results", deadline="2d", summary="API test")
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `name` | MUST | Name of the direct subordinate Anima to delegate to |
| `instruction` | MUST | Task instruction content |
| `deadline` | MUST | Deadline. Relative format `30m` / `2h` / `1d` or ISO8601 |
| `summary` | MAY | One-line task summary (defaults to first 100 chars of instruction if omitted) |
| `workspace` | MAY | Working directory. Workspace alias makes the delegate work in that directory |

### Tracking Delegated Tasks

Use the `task_tracker` tool to check progress of delegated tasks.
It cross-references the latest status from the subordinate's task_queue.jsonl.

```
task_tracker()                     # Active delegated tasks (default)
task_tracker(status="all")         # All including completed
task_tracker(status="completed")   # Completed only
```

| status | Meaning |
|--------|---------|
| `active` | In progress (other than done/cancelled/failed). Default |
| `all` | All tasks |
| `completed` | Completed only (done/cancelled/failed) |

### Receiving a Delegated Task

1. Receive delegation message via DM
2. Task is automatically registered in your task queue
3. Review the content; if unclear, ask the delegator (SHOULD)
4. Report results to the delegator when done (MUST)
