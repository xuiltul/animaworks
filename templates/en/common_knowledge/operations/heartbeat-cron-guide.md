# Heartbeat and Cron Setup and Operation

How to configure and run Heartbeat (periodic checks) and Cron (scheduled tasks).
Use when changing periodic behavior or adding new scheduled tasks.

## What is Heartbeat

Heartbeat is the mechanism for Digital Anima to start periodically and observe, plan, and reflect.
It automates the same behavior as a human checking their inbox and reviewing ongoing work.

### Important: Heartbeat Does Only "Observe and Plan"

Heartbeat is limited to three phases: **Observe → Plan → Reflect**.

- MUST: Only observe, plan, and reflect during Heartbeat
- MUST NOT: Do long-running execution (coding, heavy tool use, etc.) during Heartbeat
- MUST: When execution is needed, write tasks out as LLM tasks under `state/pending/`

The **TaskExec path** picks up and runs written tasks automatically.
TaskExec starts within about 3 seconds after Heartbeat finishes.

### Heartbeat and Chat Run in Parallel

Heartbeat and human chat use different locks, so they can run at the same time.
Messages from humans can be answered during Heartbeat.

### Writing LLM Tasks to pending/

When Heartbeat discovers work to do, place a JSON file under `state/pending/`:

```json
{
  "task_type": "llm",
  "task_id": "unique-id",
  "description": "Run API tests and summarize results",
  "context": "Slack API connectivity test requested by hinata",
  "acceptance_criteria": "Summarize test results for all endpoints in a report",
  "reply_to": {"name": "hinata", "content": "Report completion"},
  "submitted_by": "heartbeat"
}
```

TaskExec detects this file and runs the task in an LLM session.
When done, it notifies the `reply_to` party automatically.

### Heartbeat Triggers

Heartbeat has two trigger types:

| Trigger | Description |
|---------|-------------|
| Scheduled | APScheduler runs on the interval in `heartbeat.md` |
| Message | Starts when unread messages arrive in Inbox (processed as Inbox path) |

Message triggers include safeguards:
- **Cooldown**: No new start within 60 seconds of last message-triggered run
- **Cascade detection**: More than 4 round-trips between two parties in 10 minutes is treated as a loop and throttled

## heartbeat.md Configuration

`heartbeat.md` configures each Anima's active hours and checklist.
The interval is fixed at 30 minutes (system-managed); Anima cannot change it.
Path: `~/.animaworks/animas/{name}/heartbeat.md`

### Format

```markdown
# Heartbeat: {name}

## Active Hours
24 hours (JST)

## Checklist
- Unread messages in Inbox?
- Any blockers in current tasks?
- New files in my workspace?
- If nothing, do nothing (HEARTBEAT_OK)

## Notification Rules
- Notify stakeholders only when urgent
- Do not repeat the same notification within 24 hours
```

### Fields

**Interval**: Fixed at 30 minutes (system). Cannot be changed in `heartbeat.md`.

**Active hours** (SHOULD):
- Use `HH:MM - HH:MM` (e.g. `9:00 - 22:00`)
- Heartbeat does not run outside this range
- Default: 24 hours
- Timezone: Asia/Tokyo

**Checklist** (MUST):
- Items the agent checks when Heartbeat runs
- Bullet list (`- `)
- Passed as-is into the agent prompt
- Can be customized per role

### Custom Checklist Examples

Default:
```markdown
## Checklist
- Unread messages in Inbox?
- Any blockers in current tasks?
- If nothing, do nothing (HEARTBEAT_OK)
```

Developer:
```markdown
## Checklist
- Unread messages in Inbox?
- Any blockers in current tasks?
- New Issues or PRs in monitored GitHub repos?
- CI/CD failure alerts?
- If nothing, do nothing (HEARTBEAT_OK)
```

Communications:
```markdown
## Checklist
- Unread messages in Inbox?
- Unread Slack mentions?
- Unreplied emails?
- Any blockers in current tasks?
- If nothing, do nothing (HEARTBEAT_OK)
```

### Heartbeat Hot Reload

When `heartbeat.md` is updated on disk, Anima reloads the schedule automatically.
No server restart needed. `reload_anima_schedule()` is called and APScheduler jobs are reregistered.

## What is Cron

Cron runs tasks at defined times. Heartbeat is "periodic check"; Cron is "scheduled work".

Examples:
- Daily 9:00: plan the day
- Every Friday 17:00: weekly reflection
- Daily 2:00: run backup script

## cron.md Configuration

Cron tasks are defined in `cron.md` in Markdown + YAML.
Path: `~/.animaworks/animas/{name}/cron.md`

### Format

Each task starts with `## Task Name (schedule)`.
The schedule goes in parentheses (both full- and half-width supported).

```markdown
# Cron: {name}

## Daily Plan (Daily 9:00 JST)
type: llm
Check yesterday's progress from long-term memory and plan today's tasks.
Prioritize based on vision and goals.
Write results to state/current_task.md.

## Weekly Reflection (Every Friday 17:00 JST)
type: llm
Review this week's episodes/ and extract patterns into knowledge/.
```

### CronTask Schema

<!-- AUTO-GENERATED:START cron_fields -->
### CronTask Field Reference (auto-generated)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `PydanticUndefined` | Task name (unique) |
| `schedule` | `str` | `PydanticUndefined` | Cron schedule (e.g. `*/30 * * * *`) |
| `type` | `str` | `"llm"` | Task type: `llm` or `command` |
| `description` | `str` | `""` | LLM type: instruction text |
| `command` | `str | None` | None | Command type: Bash command |
| `tool` | `str | None` | None | Command type: internal tool name |
| `args` | `dict[str, Any] | None` | None | Command type: tool args (JSON) |

<!-- AUTO-GENERATED:END -->

Parsed into `CronTask` model:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | (required) | From `##` heading |
| `schedule` | str | (required) | From parentheses in heading |
| `type` | str | `"llm"` | `"llm"` or `"command"` |
| `description` | str | `""` | LLM instruction |
| `command` | str \| None | `None` | Bash command |
| `tool` | str \| None | `None` | Internal tool name |
| `args` | dict \| None | `None` | YAML args for tool |

## LLM Cron Tasks

`type: llm` tasks are executed by the agent (LLM), which can use tools and make decisions.
Description becomes the prompt.

### Characteristics

- Agent uses tools, searches memory, makes decisions
- Output varies by task
- Uses model API (cost)

### Example

```markdown
## Daily Plan (Daily 9:00 JST)
type: llm
Review yesterday's episodes/ and plan today's tasks.
Prioritize by vision and goals.
Write results to state/current_task.md.
Also check pending.md and adjust priorities if needed.
```

For `description` (text after `type:`), SHOULD include:
- What to check (input)
- How to decide (criteria)
- What to produce (output)

## Command Cron Tasks

`type: command` runs fixed commands or tools without the agent.
Suited for deterministic tasks (backup, notifications, etc.).

### Bash

```markdown
## Backup (Daily 2:00 JST)
type: command
command: /usr/local/bin/backup.sh
```

`command:` holds a single Bash command, executed via shell.

### Internal Tool

```markdown
## Slack Morning Greeting (Weekdays 9:00 JST)
type: command
tool: slack_send
args:
  channel: "#general"
  message: "Good morning! Looking forward to working with you today."
```

`tool:` is the internal tool name; `args:` are YAML (2-space indent).

### When to Use LLM vs Command

| Aspect | LLM | Command |
|--------|-----|---------|
| Needs judgment? | Yes | No |
| API cost? | Yes | No |
| Output | Variable | Deterministic |
| Suited for | Planning, reflection, writing | Backup, notifications, data fetch |
| On error | Agent handles | Log only |

Guidelines:
- "Same thing every time" → Command (SHOULD)
- "Judgment depends on context" → LLM (SHOULD)
- "Command + interpret result" → LLM and instruct command in description

## Schedule Syntax

Schedule syntax for the heading parentheses.
Supports both human-readable and standard cron.

### Human-readable

| Syntax | Meaning | Example |
|--------|---------|---------|
| `Daily HH:MM` | Every day at time | `Daily 9:00 JST` |
| `Weekdays HH:MM` | Mon–Fri at time | `Weekdays 9:00 JST` |
| `Every {weekday} HH:MM` | Weekly | `Every Friday 17:00 JST` |
| `Every other {weekday} HH:MM` | Biweekly | `Every other Monday 10:00 JST` |
| `Nth {weekday} HH:MM` | Nth weekday of month | `2nd Tuesday 10:00 JST` |
| `Monthly N HH:MM` | Monthly on day N | `Monthly 1 9:00 JST` |
| `Last day HH:MM` | Last day of month | `Last day 18:00 JST` |

Weekdays: mon, tue, wed, thu, fri, sat, sun

Timezone labels like `JST` / `UTC` are stripped. Timezone is always `Asia/Tokyo`.

### Standard cron

5-field cron:

```
min hour day month weekday
```

Examples:
- `0 9 * * *` — Daily 9:00
- `0 9 * * 1-5` — Weekdays 9:00
- `*/30 9-17 * * *` — Every 30 min 9–17
- `0 2 1 * *` — 2:00 on 1st of month

## Checking cron_logs

Cron runs are logged and also broadcast via WebSocket as `anima.cron` events.

Logs:
- Server: `animaworks.lifecycle` logger (INFO)
- Web UI: activity feed
- episodes/: LLM tasks may write here (SHOULD)

LLM results are recorded as `CycleResult` with:
- `trigger`: `"cron"`
- `action`: Summary of agent behavior
- `summary`: Result summary
- `duration_ms`: Duration (ms)
- `context_usage_ratio`: Context usage

## Common Cron Examples

### Basic (recommended for all Anima)

```markdown
# Cron: {name}

## Daily Plan (Daily 9:00 JST)
type: llm
Check yesterday's activity in episodes/, review pending.md.
Set today's top priorities and update state/current_task.md.

## Weekly Reflection (Every Friday 17:00 JST)
type: llm
Review this week's episodes/, extract patterns and lessons.
Write important findings to knowledge/.
Consider turning repeated work into procedures/.
```

### External Integration

```markdown
## Slack Daily Report (Weekdays 18:00 JST)
type: command
tool: slack_send
args:
  channel: "#daily-report"
  message: "Today's work is complete. Details will be shared at tomorrow's standup."

## GitHub Issue Check (Weekdays 10:00 JST)
type: llm
Check new Issues and PRs in assigned repos.
Report important ones to supervisor.
```

### Memory Maintenance

```markdown
## Knowledge Review (Monthly 1 10:00 JST)
type: llm
Review all knowledge/ files, tidy outdated or conflicting info.
Consider archiving low-priority items.

## Procedure Update Check (Every other Monday 10:00 JST)
type: llm
Review procedures/ and confirm they match practice.
Update if needed.
```

### Disabling a Task

Wrap the section in HTML comments:

```markdown
<!--
## Paused Task (Daily 15:00 JST)
type: llm
This task is temporarily disabled.
-->
```

Comments are ignored by the parser.

## Cron Hot Reload

Updating `cron.md` reloads the schedule, like `heartbeat.md`.
Anima can edit it itself (self-modify pattern).

Reload steps:
1. Remove existing cron jobs for that Anima
2. Re-parse `cron.md` and register new jobs
3. Log `Reloaded schedule for '{name}'`

When editing `cron.md` yourself:
- Use the heading format `## Task Name (schedule)` (MUST)
- Put `type` at the start of the body (SHOULD)
- Use supported schedule syntax (MUST)
