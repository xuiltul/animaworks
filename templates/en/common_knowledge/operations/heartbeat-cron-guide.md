# Heartbeat and Cron Setup and Operation

Configuration and operation guide for Heartbeat (periodic checks) and Cron (scheduled tasks).
Refer to this when changing periodic behavior or adding new scheduled tasks.

## What is Heartbeat

Heartbeat is the mechanism for Digital Anima to start periodically and observe, plan, and reflect.
It automates the same behavior as a human checking their inbox and reviewing ongoing work.

### Important: Heartbeat Does Only "Observe and Plan"

Heartbeat is limited to three phases: **Observe → Plan → Reflect**.

- MUST: Only observe, plan, and reflect during Heartbeat
- MUST NOT: Do long-running execution (coding, heavy tool use, etc.) during Heartbeat
- MUST: When execution is needed, delegate via `delegate_task` if subordinates are available, or submit tasks via `submit_tasks`

The **TaskExec path** picks up and runs written tasks automatically.
TaskExec starts within 3 seconds after Heartbeat finishes.

### Heartbeat and Chat Run in Parallel

Heartbeat and human chat use different locks, so they can run at the same time.
Messages from humans can be answered immediately even during Heartbeat.

### Submitting Tasks via submit_tasks

When Heartbeat discovers work to do, submit tasks using the `submit_tasks` tool:

```
submit_tasks(batch_id="hb-20260301-api-test", tasks=[
  {"task_id": "api-test", "title": "Run API tests",
   "description": "Run Slack API connectivity tests and summarize results for all endpoints. Report to aoi on completion."}
])
```

`submit_tasks` registers in both Layer 1 (execution queue `state/pending/`) and Layer 2 (task registry `task_queue.jsonl`) simultaneously.
TaskExec detects the task and runs it in an LLM session.

**Important**: Do not manually write JSON files to `state/pending/`. Always submit via the `submit_tasks` tool.

Use `submit_tasks` even for single tasks (tasks array with one item).
For multiple independent tasks, use `parallel: true`; for dependencies, specify `depends_on`.
See task-management for details.

### Heartbeat Trigger Types

Heartbeat has two trigger types:

| Trigger | Description |
|---------|-------------|
| Scheduled Heartbeat | APScheduler runs on the interval in `config.json` `heartbeat.interval_minutes` |
| Message Trigger | Starts immediately when unread messages arrive in Inbox (processed as Inbox path) |

Message triggers include safeguards:
- **Cooldown**: No new start within a configured time after the last message-triggered run completes (`config.json` `heartbeat.msg_heartbeat_cooldown_s`, default 300 seconds)
- **Cascade detection**: When round-trips between two parties exceed a threshold within a time window, it is treated as a loop and throttled (`heartbeat.cascade_window_s` default 30 min, `heartbeat.cascade_threshold` default 3)

## heartbeat.md Configuration

`heartbeat.md` is each Anima's configuration file that defines active hours and checklist items.
The execution interval is configurable in `config.json` `heartbeat.interval_minutes` (1–60 min, default 30). It cannot be changed in `heartbeat.md`.
Each Anima gets a name-based 0–9 minute offset to stagger simultaneous startups.
Path: `~/.animaworks/animas/{name}/heartbeat.md`

### Format

```markdown
# Heartbeat: {name}

## Active Hours
24 hours (server timezone)

## Checklist
- Unread messages in Inbox?
- Any blockers in current tasks?
- New files in my workspace?
- If nothing, do nothing (HEARTBEAT_OK)

## Notification Rules
- Notify stakeholders only when urgent
- Do not repeat the same notification within 24 hours
```

### Configuration Fields

**Execution interval**:
- Set in `config.json` `heartbeat.interval_minutes` (1–60 min, default 30). Cannot be changed in `heartbeat.md`

**Active hours** (SHOULD):
- Use `HH:MM - HH:MM` (e.g. `9:00 - 22:00`)
- Heartbeat does not run outside this range
- Default when unset: 24 hours (all day)
- Timezone: configurable via `config.json` `system.timezone`. Default: auto-detected from system

**Checklist** (MUST):
- Items the agent checks when Heartbeat runs
- Bullet list (`- `)
- Passed as-is into the agent prompt
- Can be customized per Anima role

### Custom Checklist Examples

Default (all Anima):
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

### Execution Model (Cost Optimization)

When `background_model` is configured, Heartbeat / Inbox / Cron run on that model instead of the main model.
Chat (human interaction) and TaskExec (actual work) continue using the main model.

Setup: `animaworks anima set-background-model {name} claude-sonnet-4-6`
See the "Background Model" section in `reference/operations/model-guide.md` for details.

### Heartbeat Internal Behavior

- **Crash recovery**: If the previous Heartbeat failed, error info is saved to `state/recovery_note.md`. It is injected into the prompt on the next run and the file is removed after recovery.
- **Reflection logging**: If Heartbeat output contains a `[REFLECTION]...[/REFLECTION]` block, it is recorded in activity_log as `heartbeat_reflection` and included in subsequent Heartbeat context.
- **Subordinate check**: Anima with subordinates get automatic subordinate status check instructions injected into Heartbeat and Cron prompts.

### Heartbeat Hot Reload

When `heartbeat.md` is updated on disk, `_check_schedule_freshness()` detects the change on the next Heartbeat run and SchedulerManager reloads the schedule automatically.
No server restart needed (MAY skip restart). APScheduler jobs are re-registered.

## Per-anima Heartbeat Interval

### Setting in status.json

Set `heartbeat_interval_minutes` in each Anima's `status.json` to configure individual heartbeat intervals.

```json
{
  "heartbeat_interval_minutes": 60
}
```

- Valid range: 1-1440 minutes (1 day)
- If not set: falls back to `config.json` `heartbeat.interval_minutes` (default 30 min)
- Animas can self-adjust by updating `status.json` via `write_memory_file`

### Recommended Guidelines

| Situation | Recommended Interval | Reason |
|-----------|---------------------|--------|
| Active development project | 15-30 min | Frequent situation awareness needed |
| Normal operations | 30-60 min | Default. Balanced frequency |
| Low load / standby | 60-120 min | Cost saving. Longer intervals when idle |
| Long dormancy / inactive | 120-1440 min | Minimal patrol for situation awareness |

### Relationship with Activity Level

When a global Activity Level (10%-400%) is set, the effective interval is calculated as:

```
effective_interval = base_interval / (activity_level / 100)
```

Example: base 30min, Activity Level 50% → effective 60min
Example: base 30min, Activity Level 200% → effective 15min

- Minimum effective interval is 5 minutes (never goes below 5 regardless of boost)
- Below 100%: max_turns also scales down proportionally (floor of 3 turns)
- At/above 100%: max_turns stays unchanged (only interval shortens)

### Activity Schedule (Time-Based Auto-Switching / Night Mode)

A mechanism that automatically switches Activity Level based on time of day.
Use this when you want to reduce costs during nighttime or weekends, or keep Animas active only during business hours.

#### How It Works

- Configure time-based entries in the `activity_schedule` field of `config.json`
- Every minute, the current time is checked against schedule entries
- When a matching entry's level differs from the current Activity Level, it switches automatically
- All Anima heartbeats are immediately rescheduled when Activity Level changes

#### Configuration Format

Each entry has three fields: `start` (start time), `end` (end time), `level` (Activity Level %):

```json
{
  "activity_schedule": [
    {"start": "09:00", "end": "22:00", "level": 100},
    {"start": "22:00", "end": "06:00", "level": 30}
  ]
}
```

- Times are in `HH:MM` format (24-hour clock)
- **Midnight wrap supported**: specifying `"22:00"` to `"06:00"` (start > end) covers the late-night period
- `level` ranges from 10 to 400
- Maximum 24 entries
- Empty array `[]` disables scheduled mode (reverts to fixed Activity Level)

#### How to Configure

- **Settings UI**: Night mode checkbox + time range and level settings
- **API**: Send the above JSON via `PUT /api/settings/activity-schedule`
- **Direct config edit**: Edit `activity_schedule` in `config.json`, then restart the server

#### Important Notes

- Manually changing Activity Level also updates the matching schedule entry for the current time period
- The schedule is applied immediately on server startup (sets the level matching the current time)
- If no schedule entry matches the current time, the last-set Activity Level is maintained

## What is Cron

Cron is "tasks that run at defined times". Heartbeat is "periodic check"; Cron is "scheduled work".

Examples:
- Daily 9:00: plan the day
- Every Friday 17:00: weekly reflection
- Daily 2:00: run backup script

## cron.md Configuration

Cron tasks are defined in `cron.md` in Markdown + YAML format.
Path: `~/.animaworks/animas/{name}/cron.md`

### Basic Format

Each task starts with a `## Task Name` heading, and the body begins with a `schedule:` directive containing the standard 5-field cron expression.

```markdown
# Cron: {name}

## Daily Plan
schedule: 0 9 * * *
type: llm
Check yesterday's progress from long-term memory and plan today's tasks.
Prioritize based on vision and goals.
Write results to state/current_task.md.

## Weekly Reflection
schedule: 0 17 * * 5
type: llm
Review this week's episodes/ and extract patterns into knowledge/.
```

The legacy format (`## Task Name (Daily 9:00 JST)` with schedule in parentheses) can be converted to the new format with `animaworks migrate-cron`.

### CronTask Schema

Each task is parsed into the following `CronTask` model:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | (required) | Task name. Extracted from `##` heading |
| `schedule` | str | (required) | Standard 5-field cron expression. Extracted from `schedule:` directive |
| `type` | str | `"llm"` | Task type: `"llm"` or `"command"` |
| `description` | str | `""` | LLM instruction text (used with type: llm) |
| `command` | str \| None | `None` | Command type: bash command |
| `tool` | str \| None | `None` | Command type: internal tool name |
| `args` | dict \| None | `None` | Tool arguments (YAML format) |
| `skip_pattern` | str \| None | `None` | Command type: skip follow-up LLM when stdout matches this regex |
| `trigger_heartbeat` | bool | `True` | Command type: if `False`, skip follow-up cron LLM after command output |

## LLM Cron Tasks

`type: llm` tasks are executed by the agent (LLM) with judgment and reasoning.
The description is passed as the prompt to the agent.

### Characteristics

- Agent uses tools, searches memory, makes decisions
- Output varies by task
- Uses model API (cost)

### Example

```markdown
## Daily Plan
schedule: 0 9 * * *
type: llm
Review yesterday's episodes/ and plan today's tasks.
Prioritize by vision and goals.
Write results to state/current_task.md.
Also check pending.md and adjust priorities if needed.
```

For description (body after `type:` line), SHOULD include:
- What to check (input)
- How to decide (criteria)
- What to produce (output)

## Command Cron Tasks

`type: command` runs fixed commands or tools without the agent.
Suited for deterministic tasks (backup, notifications, etc.).

### Bash Command Type

```markdown
## Backup
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

Put the bash command on a single line after `command:`.
The command is executed via shell.

### Internal Tool Type

```markdown
## Slack Morning Greeting
schedule: 0 9 * * 1-5
type: command
tool: slack_send
args:
  channel: "#general"
  message: "Good morning! Looking forward to working with you today."
```

Put the internal tool name after `tool:` and arguments in YAML format after `args:`.
args are parsed as a YAML indented block (2-space indent).

### Command Type Follow-up Control

Command type tasks, when the command exits successfully and has stdout, pass that output to the LLM for follow-up analysis (runs with heartbeat-equivalent context).

- **`trigger_heartbeat: false`** — Skip follow-up LLM (when output analysis is not needed)
- **`skip_pattern: <regex>`** — Skip follow-up when stdout matches this regex

```markdown
## Log Fetch (no output analysis needed)
schedule: 0 8 * * *
type: command
trigger_heartbeat: false
command: /usr/local/bin/fetch-logs.sh

## Health Check (skip analysis when "OK")
schedule: */15 * * * *
type: command
skip_pattern: ^OK$
command: /usr/local/bin/health-check.sh
```

### When to Use LLM vs Command

| Aspect | LLM | Command |
|--------|-----|---------|
| Needs judgment? | Yes | No |
| API cost? | Yes | No |
| Output predictability | Variable | Deterministic |
| Suited for | Planning, reflection, writing | Backup, notifications, data fetch |
| On error | Agent handles autonomously | Log only |

Guidelines:
- "Same thing every time" → Command (SHOULD)
- "Judgment depends on context" → LLM (SHOULD)
- "Command + interpret result" → LLM and instruct command in description

## Schedule Syntax

The `schedule:` directive in cron.md uses the **standard 5-field cron expression**.

### Standard Cron Expression (required)

```
minute hour day month weekday
```

Examples:
- `0 9 * * *` — Daily 9:00
- `0 9 * * 1-5` — Weekdays 9:00
- `*/30 9-17 * * *` — Every 30 min between 9:00–17:00
- `0 2 1 * *` — 2:00 on 1st of month
- `0 17 * * 5` — Every Friday 17:00

Timezone is configurable via `config.json` `system.timezone`. Default: auto-detected from system.

### Migration from Japanese Schedule Notation

Legacy cron.md written in the old format (`## Task Name (Daily 9:00 JST)`) can be converted to standard cron with `animaworks migrate-cron`. Conversion table:

| Japanese notation | Cron example |
|-------------------|--------------|
| `Daily HH:MM` | `0 9 * * *` |
| `Weekdays HH:MM` | `0 9 * * 1-5` |
| `Every {weekday} HH:MM` | `0 17 * * 5` (Friday) |
| `Monthly N HH:MM` | `0 9 1 * *` |
| `Every X minutes` | `*/5 * * * *` |
| `Every X hours` | `0 */2 * * *` |

`Biweekly`, `Last day of month`, `Nth weekday` cannot be auto-converted. Write the cron expression manually.

## Checking cron_logs

Cron task results are recorded in server logs.
They are also broadcast via WebSocket as `anima.cron` events.

How to check logs:
- Server logs: `animaworks.lifecycle` logger at INFO level
- Web UI: shown in the dashboard activity feed
- episodes/: For LLM tasks, the agent may write logs here (SHOULD)

LLM task results are recorded as `CycleResult` with:
- `trigger`: `"cron"`
- `action`: Summary of agent behavior
- `summary`: Result summary text
- `duration_ms`: Execution time (ms)
- `context_usage_ratio`: Context usage ratio

## Common Cron Examples

### Basic Set (recommended for all Anima)

```markdown
# Cron: {name}

## Daily Plan
schedule: 0 9 * * *
type: llm
Check yesterday's activity in episodes/, review pending.md.
Set today's top priorities and update state/current_task.md.

## Weekly Reflection
schedule: 0 17 * * 5
type: llm
Review this week's episodes/, extract patterns and lessons.
Write important findings to knowledge/.
Consider turning repeated work into procedures/.
```

### External Integration Tasks

```markdown
## Slack Daily Report
schedule: 0 18 * * 1-5
type: command
tool: slack_send
args:
  channel: "#daily-report"
  message: "Today's work is complete. Details will be shared at tomorrow's standup."

## GitHub Issue Check
schedule: 0 10 * * 1-5
type: llm
Check new Issues and PRs in assigned repos.
Report important ones to supervisor.
```

### Memory Maintenance

```markdown
## Knowledge Review
schedule: 0 10 1 * *
type: llm
Review all knowledge/ files, tidy outdated or conflicting info.
Consider archiving low-priority items.

## Procedure Update Check
schedule: 0 10 * * 1
type: llm
Review procedures/ and confirm they match practice.
Update if needed.
```

### Commenting Out

Wrap tasks you want to disable in HTML comments:

```markdown
<!--
## Paused Task
schedule: 0 15 * * *
type: llm
This task is temporarily disabled.
-->
```

`## ` headings inside comments are ignored by the parser.

## Cron Hot Reload

Updating `cron.md` reloads the schedule, same as `heartbeat.md`.
Changes are reflected immediately even when Anima edits cron.md itself (self-modify pattern).

Reload behavior:
1. Remove all existing cron jobs for that Anima
2. Re-parse cron.md and register new jobs
3. Log `Schedule reloaded for '{name}'`

When editing cron.md yourself:
- Put the `schedule:` directive immediately after the heading (`## Task Name`) (MUST)
- Use the standard 5-field cron expression for schedule (MUST)
- Put the type line right after schedule (SHOULD)
