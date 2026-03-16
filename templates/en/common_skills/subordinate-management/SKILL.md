---
name: subordinate-management
description: >-
  Process management for subordinate Anima: pause, resume, model change, background model change, restart,
  task delegation, status confirmation, and auditing.
  "pause", "stop", "resume", "wake", "disable", "enable",
  "change model", "background model", "restart", "delegate task", "check subordinate status",
  "pause", "resume", "process management", "stop subordinate", "dashboard", "audit"
---

# Skill: Subordinate Management (Supervisor Tools)

Supervisor tools automatically enabled for Anima that have subordinates. Most tools work on all descendants (children, grandchildren, great-grandchildren, etc.). `delegate_task` is restricted to direct subordinates only.

## Available Tools

### All Descendants (Children, Grandchildren, and Beyond)

| Tool | Purpose |
|------|---------|
| `disable_subordinate` | Pause descendant (status.json `enabled: false` → process stop + prevent auto-resume) |
| `enable_subordinate` | Resume paused descendant |
| `set_subordinate_model` | Change descendant's main LLM model (updates status.json; requires `restart_subordinate` to take effect) |
| `set_subordinate_background_model` | Change descendant's background model (for heartbeat/cron; updates status.json; requires `restart_subordinate` to take effect; empty string to clear) |
| `restart_subordinate` | Restart descendant process (status.json `restart_requested` flag; Reconciliation restarts within ~30 seconds) |
| `delegate_task` | Delegate task to direct subordinate only (queue add + DM send + tracking entry on your side) |
| `org_dashboard` | Tree view of process status, last activity, current task, and task count for all descendants |
| `ping_subordinate` | Liveness check for descendants (`name` omitted = all at once, specified = single) |
| `read_subordinate_state` | Read descendant's `current_task.md` and `pending.md` |
| `audit_subordinate` | Comprehensive audit of descendant's recent activity (summary, tasks, errors, tool usage, communication) |

### Delegated Task Tracking

| Tool | Purpose |
|------|---------|
| `task_tracker` | Track progress of tasks delegated via `delegate_task` from the subordinate's queue (`status`: all / active / completed; default: active) |

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

To change the background model (for heartbeat/cron):

```
set_subordinate_background_model(name="aoi", model="claude-sonnet-4-6", reason="Reduce heartbeat load")
restart_subordinate(name="aoi", reason="Apply background model change")
```

To clear the background model and revert to the main model:

```
set_subordinate_background_model(name="aoi", model="", reason="Unify to main model")
restart_subordinate(name="aoi")
```

### Status Confirmation and Audit

```
org_dashboard()                         # Dashboard for all subordinates
ping_subordinate()                      # Liveness check for all subordinates
ping_subordinate(name="aoi")            # Liveness check for single subordinate
read_subordinate_state(name="aoi")      # Current task and pending task content
audit_subordinate(name="aoi")           # Comprehensive audit of last 1 day
audit_subordinate(name="aoi", days=7)   # Audit last 7 days (days: 1–30)
audit_subordinate(since="09:00")        # All subordinates since 9:00 today
audit_subordinate(name="aoi", since="13:00")  # aoi since 13:00 today
```

Also available via CLI (useful for S-mode via Bash):

```bash
animaworks anima audit aoi              # Audit last 1 day
animaworks anima audit aoi --days 7     # Audit last 7 days
animaworks anima audit --all --since 09:00  # All animas since 9:00 today
```

### Task Delegation

```
delegate_task(name="aoi", instruction="Summarize weekly report", deadline="1d", summary="Weekly report creation")
# name, instruction, deadline are required. summary is optional (defaults to first 100 chars of instruction)
# Specify workspace to have the delegate work in that workspace (see workspace-manager skill)
task_tracker(status="active")      # Check progress of delegated tasks (status: all / active / completed)
```

For assigning workspaces to subordinates (primary working directory), see the `workspace-manager` skill.

## Permissions

- **All descendants (recursive)**: Status, management, and audit tools work on any descendant
- **Direct subordinates only**: `delegate_task` (task delegation)
- You cannot operate on yourself
