# Task Management

Operational reference for how Digital Anima receives, tracks, and completes tasks.
Search and refer to this when unsure how to proceed with tasks.

## Basic Task Management Structure

Task state is managed by files in the `state/` directory and the task queue.

| Resource | Role |
|----------|------|
| `state/current_task.md` | Current task being worked on (one at a time) |
| `state/pending.md` | Manual backlog (free-form) |
| `state/pending/` directory | LLM tasks (JSON format). Written by Heartbeat, submit_tasks, Task tool, and Agent tool. TaskExec path automatically picks them up and runs them |
| `state/task_queue.jsonl` | Persistent task queue (append-only JSONL). Tracks requests from humans and Anima |

`state/current_task.md` MUST always reflect the latest state. Update it whenever the task state changes.

### Three-Path Execution Model

In AnimaWorks, tasks are processed across three independent paths:

| Path | Trigger | Role | Scope |
|------|---------|------|-------|
| **Inbox** | DM received | Process and reply to inter-Anima messages | Immediate, lightweight replies only |
| **Heartbeat** | Scheduled check | Situation assessment and planning (Observe → Plan → Reflect) | Observation and judgment only. Writes execution to `pending/` |
| **TaskExec** | Task appears in `pending/` | Execute LLM tasks | Full execution (including tool use) |

Heartbeat does **not** execute. When it finds a task that needs execution, it either delegates via `delegate_task` (if subordinates are available) or submits it via `submit_tasks` for the TaskExec path.

In S-mode (Claude Agent SDK) Chat path, the **Task tool** (and Agent tool) provides automatic routing:
- With subordinates → immediately delegated to the subordinate with minimum workload and best role match (same flow as delegate_task)
- Without subordinates, or when delegation fails → written to `state/pending/`, and TaskExec path runs it

### Task Queue (backlog_task / update_task / list_tasks)

The persistent task queue is recorded in `state/task_queue.jsonl` in append-only JSONL format.
Use `backlog_task` to register tasks, `update_task` to update status, and `list_tasks` to retrieve the list.
Tasks registered in the queue are displayed in summarized form in the system prompt Priming section.

#### backlog_task

```
backlog_task(source="human", original_instruction="Create the monthly sales report and submit it to aoi", assignee="your own name", summary="Monthly report creation", deadline="1d")
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `source` | MUST | `human` (request from human) / `anima` (delegation from Anima) |
| `original_instruction` | MUST | Original instruction text (include quoted text when delegating. Max 10,000 characters) |
| `assignee` | MUST | Assignee name (your own name or delegatee Anima name) |
| `summary` | MUST | One-line task summary (defaults to first 100 chars of original_instruction if empty) |
| `deadline` | MUST | Deadline. Relative format `30m` / `2h` / `1d` or ISO8601 |
| `relay_chain` | MAY | Delegation chain (e.g. `["aoi", "taro"]`) |

- When receiving instructions from a human, always record with `backlog_task` and specify `source="human"` (MUST)
- Tasks with `source: human` must be processed with highest priority (MUST)
- Queued tasks are reviewed by Heartbeat; when starting work, update to `in_progress` with `update_task`

#### update_task

Updates task status. Set to `done` when complete, `cancelled` when aborted, `failed` when execution fails.

```
update_task(task_id="abc123def456", status="in_progress")
update_task(task_id="abc123def456", status="done", summary="Report creation completed")
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `task_id` | MUST | Task ID (returned when backlog_task was called) |
| `status` | MUST | `pending` / `in_progress` / `done` / `cancelled` / `blocked` / `failed` |
| `summary` | MAY | Updated summary |

#### list_tasks

Retrieves the task queue list. Can filter by status.

```
list_tasks()                     # All tasks
list_tasks(status="pending")     # Pending only
list_tasks(status="in_progress") # In progress only
list_tasks(status="done")        # Completed only
list_tasks(status="failed")      # Failed only
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

## Using current_task.md

`current_task.md` records the task you are currently working on.
Record only one task (MUST). Even with multiple parallel tasks, write only the top-priority one here.

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

## Using pending.md

`pending.md` manages the backlog of tasks not yet started.
Order by priority (SHOULD).

### Format

```markdown
# Pending Tasks

## [HIGH] Gmail notification template creation
assigned_by: aoi
received: 2026-02-15 11:00
deadline: 2026-02-16 EOD
notes: |
  Auto-notification email for new clients.
  Create 3 template drafts and submit to aoi.

## [MEDIUM] Knowledge base cleanup
assigned_by: self
received: 2026-02-14 09:00
notes: |
  Add tags to files under knowledge/ to improve search efficiency.

## [LOW] Weekly report format improvement proposal
assigned_by: aoi
received: 2026-02-13 15:00
notes: |
  Review current report format and summarize improvement proposals.
  Not urgent.
```

### Priority Labels

| Label | Meaning | Guideline |
|-------|---------|-----------|
| `[URGENT]` | Urgent | Start immediately, prioritizing over all other tasks (MUST) |
| `[HIGH]` | High | Should start same day |
| `[MEDIUM]` | Medium | Should start this week |
| `[LOW]` | Low | Start when time allows |

Tasks without a specified priority are treated as `[MEDIUM]` (SHOULD).

## Task State Transitions

Tasks transition through the following states. Always update current_task.md when the state changes (MUST).

```
received → in-progress → completed
                      ↘ blocked → in-progress (resume)
                                ↘ cancelled
```

### State Definitions

| State | Meaning | Where to record |
|-------|---------|-----------------|
| `received` | Task received but not started | pending.md |
| `in-progress` | Currently working on it | current_task.md |
| `completed` | Completed | Return to idle + log in episodes/ |
| `blocked` | Blocked by blocker | current_task.md with blockers noted |
| `cancelled` | Cancelled | Return to idle + log in episodes/ |

### Transition Steps

**received → in-progress (start)**:
1. Remove the task from pending.md
2. Write current_task.md with `status: in-progress`
3. Log "task started" in episodes/ (SHOULD)

**in-progress → completed**:
1. Set current_task.md to `status: idle`
2. Log "task completed" and result summary in episodes/ (MUST)
3. Report results to the assigner (MUST if assigned_by is someone else)
4. If there is a next task in pending.md, move the top-priority one to current_task.md

**in-progress → blocked**:
1. Change current_task.md `status` to `blocked`
2. Document the specific block reason in the `blockers` field (MUST)
3. Take action to resolve the block (see blocked-task flow below)
4. If there is a next-priority task in pending.md, consider starting it in parallel (MAY)

## Managing Multiple Tasks by Priority

When multiple tasks exist, use these criteria:

1. **URGENT is top priority**: When an `[URGENT]` task arrives, start it even if it means interrupting the current task (MUST)
2. **Supervisor tasks first**: Instructions from supervisor take priority over other tasks at the same level (SHOULD)
3. **By deadline**: Start tasks with earlier deadlines first (SHOULD)
4. **FIFO**: Same priority and deadline → process in received order (MAY)

### When Interrupting a Task

When a higher-priority task interrupts:

1. Note current progress in current_task.md (MUST)
2. Move the current task back to pending.md (with state and progress notes)
3. Write the new task in current_task.md

Format when moving back to pending.md:

```markdown
## [HIGH] Slack integration test (interrupted)
assigned_by: aoi
received: 2026-02-15 10:00
progress: |
  API connectivity test completed. Interrupted during channel post test.
  Remaining: post test to #general, error handling verification
```

## Handling Blocked Tasks

When a task is blocked, follow these steps.

### Step 1: Identify and Record the Blocker

Document the specific cause in `blockers` in current_task.md (MUST).

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

If unblocking will take time, you MAY start the next task in pending.md.
Move the blocked task to pending.md and keep the block reason.

```markdown
## [HIGH] AWS S3 bucket setup (blocked)
assigned_by: aoi
received: 2026-02-15 10:00
blocked_reason: AWS credentials not configured. Requested from aoi (2026-02-15 11:00)
```

### Step 4: Resume After Unblock

When the block is resolved (e.g. notified by message):
1. Take the task from pending.md
2. Re-evaluate priority and decide whether to move it to current_task.md
3. If starting, change to `status: in-progress` and resume work

## Task File Templates

### current_task.md — Idle

```markdown
status: idle
```

### current_task.md — In Progress

```markdown
status: in-progress
task: {task_name}
assigned_by: {assigner_name or self}
started: {YYYY-MM-DD HH:MM}
context: |
  {task details and background}
blockers: None
```

### current_task.md — Blocked

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

### pending.md — Backlog

```markdown
# Pending Tasks

## [{priority}] {task_name}
assigned_by: {assigner_name}
received: {YYYY-MM-DD HH:MM}
deadline: {YYYY-MM-DD or none}
notes: |
  {task details}
```

## Recording Task Logs in episodes/

Record task start, completion, block, etc. in episodes/ (SHOULD).
File names: `YYYY-MM-DD.md` (daily log).

```markdown
## 10:00 Task started: Slack integration test

Started Slack API connectivity test per aoi's request.
Confirmed slack: yes in permissions.md.

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

Completed task result summaries are saved to `state/task_results/{task_id}.md` (max 2,000 characters).
Dependent tasks automatically receive these results as context. If a predecessor fails, dependent tasks are skipped and `FAILED: {reason}` is recorded.
When each task completes, a completion notification is sent via DM to the Anima that executed submit_tasks.

### When to Use submit_tasks

| Scenario | Method |
|----------|--------|
| Single task | `submit_tasks` (submit with a single-item tasks array) |
| Multiple independent tasks | `submit_tasks` with `parallel: true` |
| Tasks with dependencies | `submit_tasks` with `depends_on` |
| Delegation to subordinates | `delegate_task` (separate mechanism) |

**Important**: Do not manually write JSON files to `state/pending/`. Always submit via the `submit_tasks` tool. `submit_tasks` registers in both Layer 1 (execution queue) and Layer 2 (task registry) simultaneously, preventing task tracking gaps.

## Task Delegation (delegate_task / Task tool)

Anima with subordinates (supervisors) can delegate tasks to subordinates using the `delegate_task` tool.
In S-mode Chat path, the Task tool (and Agent tool) also supports delegation. The Task tool has no parameter to specify a subordinate; it auto-selects the subordinate with minimum workload and best role match.

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
