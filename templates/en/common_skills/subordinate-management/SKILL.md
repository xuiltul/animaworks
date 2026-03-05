---
name: subordinate-management
description: >-
  Process management for subordinate Anima: pause, resume, model change, restart,
  task delegation, status confirmation, and auditing.
  "pause", "stop", "resume", "wake", "disable", "enable",
  "change model", "restart", "delegate task", "check subordinate status",
  "pause", "resume", "process management", "stop subordinate", "dashboard", "audit"
---

# Skill: Subordinate Management (Supervisor Tools)

Supervisor tools automatically enabled for Anima that have subordinates. Manages pause, resume, model change, and restart of direct subordinates; status confirmation across all subordinates; and task delegation with progress tracking.

## Available Tools

### Direct Subordinates Only

| Tool | Purpose |
|------|---------|
| `disable_subordinate` | Pause subordinate (status.json `enabled: false` → process stop + prevent auto-resume) |
| `enable_subordinate` | Resume paused subordinate |
| `set_subordinate_model` | Change subordinate's LLM model (updates status.json; requires `restart_subordinate` to take effect) |
| `restart_subordinate` | Restart subordinate process (Reconciliation restarts within ~30 seconds) |
| `delegate_task` | Delegate task to direct subordinate (queue add + DM send + tracking entry on your side) |

### All Subordinates (Including Grandchildren)

| Tool | Purpose |
|------|---------|
| `org_dashboard` | Tree view of process status, last activity, current task, and task count for all subordinates |
| `ping_subordinate` | Liveness check for subordinates (`name` omitted = all at once, specified = single) |
| `read_subordinate_state` | Read subordinate's `current_task.md` and `pending.md` |
| `audit_subordinate` | Comprehensive audit of subordinate's recent activity (summary, tasks, errors, tool usage, communication) |

### Delegated Task Tracking

| Tool | Purpose |
|------|---------|
| `task_tracker` | Track progress of tasks delegated via `delegate_task` from the subordinate's queue (`status`: all / active / completed) |

## Important: disable_subordinate vs send_message

- **disable_subordinate**: Sets status.json to `enabled: false`. Reconciliation does not auto-resume. **Use this one**
- Sending "take a break" via send_message alone does **not** stop the process. Reconciliation will restart even after messaging

## Usage

### Pause and Resume

When pausing multiple subordinates, call `disable_subordinate` for each one:

```
disable_subordinate(name="aoi", reason="Temporary pause due to reduced workload")
disable_subordinate(name="taro", reason="Temporary pause due to reduced workload")
enable_subordinate(name="aoi")
```

### Model Change and Restart

Model changes are saved to status.json, but `restart_subordinate` is required to apply them to the running process:

```
set_subordinate_model(name="aoi", model="claude-sonnet-4-6", reason="Load balancing")
restart_subordinate(name="aoi", reason="Apply model change")
```

### Status Confirmation and Audit

```
org_dashboard()                         # Dashboard for all subordinates
ping_subordinate()                      # Liveness check for all subordinates
ping_subordinate(name="aoi")            # Liveness check for single subordinate
read_subordinate_state(name="aoi")      # Current task and pending task content
audit_subordinate(name="aoi")           # Comprehensive audit of last 1 day
audit_subordinate(name="aoi", days=7)   # Audit last 7 days
```

Also available via CLI (useful for S-mode via Bash):

```bash
animaworks anima audit aoi              # Audit last 1 day
animaworks anima audit aoi --days 7     # Audit last 7 days
```

### Task Delegation

```
delegate_task(name="aoi", instruction="Summarize weekly report", deadline="1d", summary="Weekly report creation")
task_tracker(status="active")      # Check progress of delegated tasks
```

## Permissions

- **Direct subordinates only**: disable, enable, set_subordinate_model, restart_subordinate, delegate_task
- **All subordinates (recursive)**: org_dashboard, ping_subordinate, read_subordinate_state, audit_subordinate
- You cannot pause, resume, change model, or delegate to subordinates of subordinates (grandchildren). Ask their supervisor
- You cannot operate on yourself
