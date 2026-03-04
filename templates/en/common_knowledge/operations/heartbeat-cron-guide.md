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
- MUST: When execution is needed, delegate via `delegate_task` if subordinates are available, or write tasks out as LLM tasks under `state/pending/`

The **TaskExec path** picks up and runs written tasks automatically.
TaskExec starts within 3 seconds after Heartbeat finishes.

### Heartbeat and Chat Run in Parallel

Heartbeat and human chat use different locks, so they can run at the same time.
Messages from humans can be answered immediately even during Heartbeat.

### Writing LLM Tasks to pending/

When Heartbeat discovers work to do, place a JSON file under `state/pending/`:

```json
{
  "task_type": "llm",
  "task_id": "unique-id",
  "description": "Run API tests and summarize results",
  "context": "Slack API connectivity test requested by aoi",
  "acceptance_criteria": "Summarize test results for all endpoints in a report",
  "reply_to": {"name": "aoi", "content": "Report completion"},
  "submitted_by": "heartbeat"
}
```

TaskExec detects this file and runs the task in an LLM session.
When done, it notifies the `reply_to` party automatically.

To submit multiple tasks with dependencies as a batch, use the `plan_tasks` tool.
Independent tasks run in parallel; dependent tasks execute after predecessors complete.
See task_delegation_rules for details.

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

### Configuration Fields

**Execution interval**:
- Set in `config.json` `heartbeat.interval_minutes` (1–60 min, default 30). Cannot be changed in `heartbeat.md`

**Active hours** (SHOULD):
- Use `HH:MM - HH:MM` (e.g. `9:00 - 22:00`)
- Heartbeat does not run outside this range
- Default when unset: 24 hours (all day)
- Timezone: Asia/Tokyo (fixed)

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

### Heartbeat Internal Behavior

- **Crash recovery**: If the previous Heartbeat failed, error info is saved to `state/recovery_note.md`. It is injected into the prompt on the next run and the file is removed after recovery.
- **Reflection logging**: If Heartbeat output contains a `[REFLECTION]...[/REFLECTION]` block, it is recorded in activity_log as `heartbeat_reflection` and included in subsequent Heartbeat context.
- **Subordinate check**: Anima with subordinates get automatic subordinate status check instructions injected into Heartbeat and Cron prompts.

### Heartbeat Hot Reload

When `heartbeat.md` is updated on disk, `_check_schedule_freshness()` detects the change on the next Heartbeat run and SchedulerManager reloads the schedule automatically.
No server restart needed (MAY skip restart). APScheduler jobs are re-registered.

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

Timezone is always Asia/Tokyo (fixed).

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
