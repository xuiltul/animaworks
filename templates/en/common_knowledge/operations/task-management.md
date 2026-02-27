# Task Management

Reference for how Digital Anima receives, tracks, and completes tasks.
Use when unsure how to proceed with tasks.

## Task Management Structure

Task state lives in `state/` and the task queue.

| Resource | Role |
|----------|------|
| `state/current_task.md` | Current task (one at a time) |
| `state/pending.md` | Manual backlog (free-form) |
| `state/pending/` directory | LLM tasks written by Heartbeat (JSON). TaskExec path picks them up and runs them |
| Task queue (`add_task`) | Persistent task queue for human/Anima requests |

`state/current_task.md` MUST always reflect the latest state. Update it when the task changes.

### Three-Path Execution Model

AnimaWorks runs tasks across three paths:

| Path | Trigger | Role | Scope |
|------|---------|------|-------|
| **Inbox** | DM received | Handle and reply to Anima messages | Immediate, lightweight replies only |
| **Heartbeat** | Scheduled | Observe and plan (Observe → Plan → Reflect) | Observe and plan only. Write execution to `pending/` |
| **TaskExec** | Task appears in `pending/` | Run LLM tasks | Full execution (including tools) |

Heartbeat does **not** execute. It writes tasks to `state/pending/` as JSON and delegates execution to the TaskExec path.

### Task Queue (add_task tool)

Use `add_task` to add tasks to the persistent queue.
Queued tasks appear summarized in the system prompt Priming section.

```
add_task(title="Report creation", description="Create monthly sales report", priority="high", source="human")
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `title` | MUST | Short task title |
| `description` | SHOULD | Detailed description |
| `priority` | MAY | `urgent` / `high` / `medium` / `low` (default: `medium`) |
| `source` | MAY | `human` or `anima` |

- Tasks with `source: human` are highest priority (MUST)
- Heartbeat reviews queued tasks and picks them up in priority order

## Using current_task.md

`current_task.md` records the task you are currently working on.
Keep only one task here (MUST). If you have several, put only the top-priority one.

### Format

```markdown
status: in-progress
task: Slack integration test
assigned_by: hinata
started: 2026-02-15 10:00
context: |
  Hinata's request: Run Slack API connectivity test and verify posting to #general.
  Report results when done.
blockers: None
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `status` | MUST | Task status (see state transitions below) |
| `task` | MUST | One-line summary |
| `assigned_by` | SHOULD | Who assigned. Use `self` for self-assigned |
| `started` | SHOULD | When started |
| `context` | SHOULD | Details and background |
| `blockers` | SHOULD | Blockers, or "None" |

### Idle State

When there is no task:

```markdown
status: idle
```

`idle` means you are waiting for the next task. Heartbeat seeing `idle` needs no action (`HEARTBEAT_OK`).

## Using pending.md

`pending.md` is the backlog of tasks not yet started.
Order by priority (SHOULD).

### Format

```markdown
# Pending Tasks

## [HIGH] Gmail notification template
assigned_by: hinata
received: 2026-02-15 11:00
deadline: 2026-02-16 EOD
notes: |
  Auto-notify for new clients.
  Create 3 template drafts and submit to hinata.

## [MEDIUM] Knowledge base cleanup
assigned_by: self
received: 2026-02-14 09:00
notes: |
  Add tags to knowledge/ files for better search.

## [LOW] Weekly report format improvement
assigned_by: hinata
received: 2026-02-13 15:00
notes: |
  Review current format and propose improvements.
  Not urgent.
```

### Priority Labels

| Label | Meaning | When to start |
|-------|---------|---------------|
| `[URGENT]` | Critical | Stop everything and start immediately (MUST) |
| `[HIGH]` | High | Start same day |
| `[MEDIUM]` | Medium | Start this week |
| `[LOW]` | Low | When time allows |

Unlabeled tasks are treated as `[MEDIUM]` (SHOULD).

## Task State Transitions

Tasks move through these states. Always update `current_task.md` when the state changes (MUST).

```
received → in-progress → completed
                      ↘ blocked → in-progress (resume)
                                ↘ cancelled
```

### State Definitions

| State | Meaning | Where to record |
|-------|---------|-----------------|
| `received` | Received but not started | pending.md |
| `in-progress` | In progress | current_task.md |
| `completed` | Done | Back to idle + log in episodes/ |
| `blocked` | Blocked by blocker | current_task.md with blockers |
| `cancelled` | Cancelled | Back to idle + log in episodes/ |

### Transition Steps

**received → in-progress (start)**:
1. Remove task from pending.md
2. Write current_task.md with `status: in-progress`
3. Log "task started" in episodes/ (SHOULD)

**in-progress → completed**:
1. Set current_task.md to `status: idle`
2. Log "task completed" and summary in episodes/ (MUST)
3. Report to assignee (MUST if assigned_by is someone else)
4. If there is a next task in pending.md, move the top priority to current_task.md

**in-progress → blocked**:
1. Set current_task.md `status` to `blocked`
2. Fill `blockers` with the reason (MUST)
3. Take action to unblock (see blocked-task flow)
4. If another task in pending.md is unblocked, consider starting it (MAY)

## Managing Multiple Tasks

When several tasks exist:

1. **URGENT first**: Drop current work and start `[URGENT]` if one appears (MUST)
2. **Supervisor tasks first**: Supervisor requests have higher priority (SHOULD)
3. **By deadline**: Earlier deadlines first (SHOULD)
4. **FIFO**: Same priority/deadline → process in received order (MAY)

### When Interrupting a Task

If a higher-priority task interrupts:

1. Note current progress in current_task.md (MUST)
2. Move current task back to pending.md (with progress notes)
3. Write the new task in current_task.md

When moving back to pending.md:

```markdown
## [HIGH] Slack integration test (interrupted)
assigned_by: hinata
received: 2026-02-15 10:00
progress: |
  API test done. Interrupted during channel post test.
  Remaining: post test to #general, error handling check
```

## Handling Blocked Tasks

When a task is blocked:

### Step 1: Identify and Record the Blocker

Document the cause in `blockers` in current_task.md (MUST).

```markdown
status: blocked
task: AWS S3 bucket setup
blockers: |
  AWS credentials not configured.
  config.json has no aws credential.
  Need hinata to configure.
```

### Step 2: Take Action to Unblock

| Cause | Action |
|-------|--------|
| Missing info | Ask assignee for clarification (SHOULD) |
| Insufficient permissions | Ask supervisor for permission (SHOULD) |
| External dependency | Tell assignee you are waiting (SHOULD) |
| Technical issue | Search knowledge/ and procedures/ for solutions. If none, report |

### Step 3: Switch to Another Task

If unblocking will take time, you MAY work on the next task in pending.md.
Move the blocked task to pending.md and keep the blocker reason.

```markdown
## [HIGH] AWS S3 bucket setup (blocked)
assigned_by: hinata
received: 2026-02-15 10:00
blocked_reason: AWS credentials missing. Asked hinata (2026-02-15 11:00)
```

### Step 4: Resume After Unblock

When unblocked (e.g. message confirms):
1. Take the task from pending.md
2. Re-evaluate priority and decide whether to put it in current_task.md
3. If yes, set `status: in-progress` and resume

## Task File Templates

### current_task.md — Idle

```markdown
status: idle
```

### current_task.md — In Progress

```markdown
status: in-progress
task: {task_name}
assigned_by: {assignee or self}
started: {YYYY-MM-DD HH:MM}
context: |
  {task details and background}
blockers: None
```

### current_task.md — Blocked

```markdown
status: blocked
task: {task_name}
assigned_by: {assignee or self}
started: {YYYY-MM-DD HH:MM}
context: |
  {task details and background}
blockers: |
  {blocker description}
  {actions taken to unblock}
```

### pending.md — Backlog

```markdown
# Pending Tasks

## [{priority}] {task_name}
assigned_by: {assignee}
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

Started Slack API connectivity test per hinata's request.
Confirmed slack: yes in permissions.md.

## 11:30 Task completed: Slack integration test

Slack API test done. #general post test succeeded.
Reported to hinata.

[IMPORTANT] Slack API rate limit: max 1 message per minute.
Leave gaps when sending in bursts.
```

Use `[IMPORTANT]` for key learnings (SHOULD). These are prioritized during Heartbeat and consolidation.

## Task Delegation (delegate_task)

Anima with subordinates (supervisors) can delegate tasks with the `delegate_task` tool.

### How delegate_task Works

1. Task is added to subordinate’s task queue (source="anima")
2. DM is sent to subordinate (intent="delegation")
3. Tracking entry is created in your queue (status="delegated")

### Usage

```
delegate_task(name="dave", instruction="Run API test and report results", deadline="2026-02-20", summary="API test")
```

### Tracking Delegated Tasks

Use `task_tracker` to see progress:

```
task_tracker()                    # Active delegated tasks
task_tracker(status="all")        # All including completed
task_tracker(status="completed")  # Completed only
```

### Receiving a Delegated Task

1. Receive delegation DM
2. Task is added to your queue
3. Clarify with delegator if needed (SHOULD)
4. Report results when done (MUST)
