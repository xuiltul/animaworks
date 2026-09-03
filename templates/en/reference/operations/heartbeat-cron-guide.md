# Scheduled execution: configuration and operations

Guide to configuring and operating Heartbeat (periodic patrol) and Cron (scheduled tasks).
Refer to this when you want to change periodic behavior or add new scheduled tasks.

## What is Heartbeat

Heartbeat is the mechanism by which a Digital Anima starts automatically at intervals to observe and plan.
It automates the same behavior as a human periodically checking their inbox and reviewing work in progress.

### Important: Heartbeat is only "observe and plan"

Heartbeat is limited to three phases: **Observe → Plan → Reflect**.

- MUST: During Heartbeat, only observe, plan, and reflect
- MUST NOT: Run long-running execution tasks during Heartbeat (coding, heavy tool use, etc.)
- MUST: When execution is needed, delegate with `delegate_task` if you have subordinates, or submit work with `submit_tasks`

Written tasks are picked up and executed by the **TaskExec path** (`PendingTaskExecutor`) via polling.
LLM tasks written by `submit_tasks` are watched under `state/pending/` and are collected at intervals of up to about 3 seconds (the same loop also processes CLI-submitted tasks under `state/background_tasks/pending/`).

### Heartbeat and chat run in parallel

Heartbeat and human chat are managed with **separate locks**, so they can run at the same time.
Even while Heartbeat is running, messages from humans can be answered immediately.

### Submitting tasks with submit_tasks

When Heartbeat finds work that should be executed, submit tasks using the `submit_tasks` tool:

```
submit_tasks(batch_id="hb-20260301-api-test", tasks=[
  {"task_id": "api-test", "title": "Run API tests",
   "description": "Run Slack API connectivity tests and summarize results for all endpoints in a report. Report to aoi when done."}
])
```

`submit_tasks` registers in both Layer 1 (execution queue `state/pending/`) and Layer 2 (task registry `task_queue.jsonl`) at the same time.
TaskExec moves the JSON to `processing/` and runs it in an LLM session. On failure, it is moved to `state/pending/failed/`.

**Long-running CLI tools** (`animaworks-tool submit …`) are written on a separate path to `state/background_tasks/pending/` and are executed in the background by `BackgroundTaskManager` (`core/background.py`). See `operations/background-tasks.md` for details.

**Note**: Manually placing JSON under `state/pending/` is discouraged. Submit via the `submit_tasks` tool (validation and queue sync are skipped otherwise).

Use `submit_tasks` even for a single task (one item in the `tasks` array).
Multiple independent tasks run in parallel with `parallel: true`; when there are dependencies, set `depends_on`.
See task-management for details.

### Heartbeat trigger types

There are two kinds of Heartbeat triggers:

| Trigger | Description |
|---------|-------------|
| Scheduled Heartbeat | APScheduler starts runs on the interval from `config.json` `heartbeat.interval_minutes` |
| Message trigger | Starts immediately when unread messages arrive in the Inbox (handled as the Inbox path) |

Message triggers include these safeguards:

- **Cooldown**: Does not restart within a fixed time after the previous message-triggered run completes (`config.json` `heartbeat.msg_heartbeat_cooldown_s`, default 300 seconds)
- **Cascade detection**: If round-trips between two parties exceed a threshold within a window, it is treated as a loop and throttled (`heartbeat.cascade_window_s` default 30 minutes, `heartbeat.cascade_threshold` default 3)
- **Intent filter**: Immediate Heartbeat only when there is a message whose `intent` is in `heartbeat.actionable_intents` (default `report`, `question`). Otherwise (e.g. light ack-style messages), wait until the scheduled Heartbeat
- **Round-trip depth limit**: `heartbeat.depth_window_s` (default 600 seconds) and `heartbeat.max_depth` (default 6) limit excessive short-term back-and-forth for the same pair

## heartbeat.md configuration

`heartbeat.md` is each Anima's configuration file; it defines active hours and checklist items.
The Heartbeat interval can be set in `config.json` `heartbeat.interval_minutes` (1–1440 minutes, default 30). It cannot be changed in `heartbeat.md`.
Each Anima gets a name-based 0–9 minute offset to stagger simultaneous startups.
Path: `~/.animaworks/animas/{name}/heartbeat.md`

### Format

```markdown
# Heartbeat: {name}

## Active hours
24 hours (server-configured timezone)

## Checklist
- Any unread messages in the Inbox?
- Any blockers on in-progress tasks?
- Any new files placed in my workspace?
- If nothing applies, do nothing (HEARTBEAT_OK)

## Notification rules
- Notify stakeholders only when you judge it urgent
- Do not repeat the same notification within 24 hours
```

### Configuration fields

**Interval**:

- Set in `config.json` `heartbeat.interval_minutes` (1–1440 minutes, default 30). Cannot be changed in `heartbeat.md`

**Active hours** (SHOULD):

- Write in `HH:MM - HH:MM` form (e.g. `9:00 - 22:00`)
- Heartbeat does not run outside this window
- Default when unset: 24 hours (all day)
- Timezone: configurable via `config.json` `system.timezone`. When unset, the system timezone is auto-detected

**Checklist** (MUST):

- Items the agent checks when Heartbeat runs
- Use a bullet list (lines starting with `- `)
- Checklist content is passed as-is into the agent prompt
- Customizable: you may add or change items to match the Anima's role

### Custom checklist examples

Default (shared by all Anima):

```markdown
## Checklist
- Any unread messages in the Inbox?
- Any blockers on in-progress tasks?
- If nothing applies, do nothing (HEARTBEAT_OK)
```

Example for a developer:

```markdown
## Checklist
- Any unread messages in the Inbox?
- Any blockers on in-progress tasks?
- Any new Issues or PRs on monitored GitHub repositories?
- Any CI/CD failure alerts?
- If nothing applies, do nothing (HEARTBEAT_OK)
```

Example for a communications role:

```markdown
## Checklist
- Any unread messages in the Inbox?
- Any unread Slack mentions?
- Any emails awaiting reply?
- Any blockers on in-progress tasks?
- If nothing applies, do nothing (HEARTBEAT_OK)
```

### Execution model (cost optimization)

When `background_model` is set, Heartbeat / Inbox / Cron run on that model instead of the main model.
Chat (human conversation) and TaskExec (actual work) keep using the main model.

Setup: `animaworks anima set-background-model {name} claude-sonnet-4-6`
See the "Background model" section in `reference/operations/model-guide.md` for details.

### Heartbeat internals

- **Crash recovery**: If the previous Heartbeat failed, error information is saved to `state/recovery_note.md`. It is injected into the prompt on the next startup and the file is removed after recovery.
- **Reflection logging**: If Heartbeat output contains a `[REFLECTION]...[/REFLECTION]` block, it is recorded in activity_log as `heartbeat_reflection` and included in later Heartbeat context.
- **Subordinate check**: Anima with subordinates get automatic instructions to check subordinate status injected into Heartbeat and Cron prompts.
- **Session time limits** (`heartbeat` in `config.json`): After `soft_timeout_seconds` (default 300 s), a wrap-up reminder is injected; `hard_timeout_seconds` (default 600 s) forces the session to end. Setting `max_turns` overrides per-anima `max_turns` as a Heartbeat-specific turn cap.
- **Idle auto-compaction**: `heartbeat.idle_compaction_minutes` (default 10 minutes)—after this much idle time following end of stream, idle auto-compaction runs (execution-engine setting).
- **Board post spacing**: `heartbeat.channel_post_cooldown_s` (default 300 seconds, 0 = unlimited)—throttles repeated `post_channel` from the same Anima.

### How scheduled Heartbeat is scheduled

When the **effective** interval (after Activity Level and a **minimum 5-minute** rounding) is **60 minutes or less and divides 60 evenly**, jobs are registered on distributed minute slots via APScheduler's `CronTrigger` (combined with the name-based 0–9 minute offset).

Otherwise (e.g. effective 61 minutes, or 43 minutes so 60 is not evenly divided), firing uses **1-minute polling** (`_heartbeat_check`) based on elapsed time since the last run. This avoids issues inherited from older `IntervalTrigger` behavior.

### Background tools and DM logs (`core/background.py`)

`core/background.py` does not own Heartbeat/Cron scheduling itself; it handles **background execution of long-running tool calls** (including JSON state persistence) and **rotation of legacy shared DM logs** (`shared/dm_logs/`). For operational details and the CLI path, also see `operations/background-tasks.md`.

#### BackgroundTaskManager

- **Storage**: `state/background_tasks/{task_id}.json`. `task_id` is the first 12 characters of a UUID (hexadecimal). Each file records `task_id`, `anima_name`, `tool_name`, `tool_args`, `status`, `created_at`, `completed_at`, `result`, `error`.
- **States (`TaskStatus`)**: `pending` / `running` / `completed` / `failed`. For `submit` / `submit_async`, JSON is written as `running` immediately after enqueue and updated to `completed` / `failed` on completion or exception.
- **Execution API**: `submit(tool_name, tool_args, execute_fn)` runs a synchronous callable in a thread pool via `asyncio`'s `run_in_executor`. `submit_async` awaits an async callable directly. Provides `get_task` (memory first, otherwise disk), `list_tasks` (merges JSON on disk, newest by creation time first), and `active_count` (in-memory `running` count).
- **Completion callback**: Async functions passed to `on_complete` run after the task is saved. Exceptions inside the callback still preserve task results and are only logged (typically combined with writes to `state/background_notifications/`, **read and removed on the next Heartbeat** and merged into conversation context).
- **Eligibility `is_eligible(tool_name)`**: If the map contains a key, the tool is background-eligible. Keys accept: (1) **schema name** (e.g. `generate_3d_model`)—e.g. Mode A external tool dispatch; (2) **`tool:subcommand`** (e.g. `image_gen:pipeline`)—entries with `background_eligible: true` in each tool module's `EXECUTION_PROFILE` are registered in this form via `get_eligible_tools_from_profiles()` (e.g. Mode S `submit` path).
- **Building the eligible map `from_profiles()`**: Merges three layers with dict `update`; **later wins**: (1) `_DEFAULT_ELIGIBLE_TOOLS` (code defaults) (2) argument `profiles` (aggregated `EXECUTION_PROFILE`) (3) argument `config_eligible` (usually `name → seconds` expanded from `config.json` `background_task.eligible_tools` `threshold_s`). Values are expected seconds (integer) for profile integration.
- **Code defaults `_DEFAULT_ELIGIBLE_TOOLS` (seconds)**: `generate_character_assets` 30; `generate_fullbody` / `generate_bustup` / `generate_icon` / `generate_chibi` each 30; `generate_3d_model` / `generate_rigged_model` / `generate_animations` each 30; `local_llm` 60; `run_command` 60.
- **Cleanup `cleanup_old_tasks(max_age_hours=24)`**: Deletes JSON where `status` is `completed` / `failed` and `completed_at` is **older than the given hours (default 24)**. Also removes files stuck in `running` with `created_at` **more than 48 hours** ago as crash orphans. Return value is the number of deletions.
- **`result_retention_hours`**: Present on `config.json` `background_task.result_retention_hours`, but **`BackgroundTaskManager.cleanup_old_tasks` does not read it** (default remains method argument `max_age_hours=24`). Callers are expected to pass `max_age_hours` when operational retention should differ.

#### rotate_dm_logs (system Cron)

- **When it runs**: System cron in the lifecycle (`core/lifecycle/system_crons.py`, etc.), **daily at 04:30** (server-configured timezone). A job with the same ID is also registered from `core/supervisor/_mgr_scheduler.py`.
- **Scope**: `shared/dm_logs/*.jsonl` (skips filenames containing `.archive.`).
- **Behavior**: Using local current time from `core.time_utils`, parses each line's JSON `ts` (ISO), appends entries **older than the default 7 days** to `{stem}.{YYYYMMDD}.archive.jsonl`, then removes them from the live file. Lines where `ts` cannot be parsed **remain in the live file** (data loss prevention).
- **Other server scheduled jobs**: Besides per-Anima `cron.md`, the lifecycle registers system cron for memory maintenance, RAG, etc. (e.g. daily consolidation 02:00, daily index 04:00). Times use the configured timezone. DM rotation is at 04:30 as above.

### Heartbeat configuration hot reload

When `heartbeat.md` is updated on disk, `_check_schedule_freshness()` detects the change on the next Heartbeat run and SchedulerManager reloads the schedule automatically.
No server restart is required (MAY skip restart). APScheduler jobs are re-registered.

## Per-Anima Heartbeat interval

### Setting in status.json

You can set an individual Heartbeat interval per Anima with `heartbeat_interval_minutes` in each Anima's `status.json`.

```json
{
  "heartbeat_interval_minutes": 60
}
```

- Allowed range: 1–1440 minutes (one day)
- When unset: falls back to `config.json` `heartbeat.interval_minutes` (default 30 minutes)
- An Anima can self-adjust by updating `status.json` with `write_memory_file`

### Recommended guidelines

| Situation | Recommended interval | Reason |
|-----------|---------------------|--------|
| Active development project | 15–30 min | Frequent situational awareness |
| Normal operations | 30–60 min | Default. Balanced frequency |
| Low load / standby | 60–120 min | Cost saving; longer interval when there is little work |
| Long dormancy / inactive | 120–1440 min | Minimal patrol for awareness |

### Relationship with Activity Level

When a global Activity Level (10%–400%) is set, the effective interval is:

```
effective_interval = base_interval / (Activity Level / 100)
```

Example: base 30 min, Activity Level 50% → effective 60 min  
Example: base 30 min, Activity Level 200% → effective 15 min

- Minimum effective interval is 5 minutes (never below 5, however much you boost)
- At Activity Level 100% or below, `max_turns` also scales down proportionally (floor 3 turns)
- Above 100% Activity Level, `max_turns` is unchanged (only the interval shortens)

### Activity schedule (time-of-day auto-switch / night mode)

A mechanism that switches Activity Level automatically by time of day.
Use it to reduce cost at night or on weekends, or to be active only during business hours.

#### How it works

- Configure time-range entries in `config.json` `activity_schedule`
- Every minute, the current time is checked and Activity Level is set to the level for the matching range
- When Activity Level changes, all Anima Heartbeats are rescheduled immediately

#### Configuration format

Each entry has three fields: `start`, `end`, and `level` (Activity Level %):

```json
{
  "activity_schedule": [
    {"start": "09:00", "end": "22:00", "level": 100},
    {"start": "22:00", "end": "06:00", "level": 30}
  ]
}
```

- Times use `HH:MM` (24-hour)
- **Midnight wrap**: You can use `start` > `end`, e.g. `"22:00"`–`"06:00"`, to cover overnight
- `level` is in the range 10–400
- Up to 24 entries
- Empty array `[]` disables schedule mode (back to a fixed Activity Level)

#### How to configure

- **Settings UI**: Night mode checkbox plus time ranges and levels
- **API**: `PUT /api/settings/activity-schedule` with the JSON above
- **Direct config edit**: Edit `activity_schedule` in `config.json`, then restart the server

#### Caveats

- If you change Activity Level manually, the schedule entry for the current time range is updated in sync
- The schedule applies as soon as the server starts (level matching the time at startup)
- If no range matches the current time, the last configured Activity Level is kept

## What is Cron

Cron means "tasks that run automatically at fixed times". Heartbeat is "periodic patrol"; Cron is "scheduled work".

Examples:

- Every morning at 9:00: plan the day
- Every Friday at 17:00: weekly reflection
- Every day at 2:00: run a backup script

## cron.md configuration

Cron tasks are defined in Markdown + YAML in `cron.md`.
Path: `~/.animaworks/animas/{name}/cron.md`

### Basic format

Each task starts with a `## Task name` heading; the body begins with a `schedule:` directive for the standard 5-field cron expression.

```markdown
# Cron: {name}

## Morning work plan
schedule: 0 9 * * *
type: llm
Review yesterday's progress from long-term memory and plan today's tasks.
Prioritize against vision and goals.
Write results to state/current_state.md.

## Weekly reflection
schedule: 0 17 * * 5
type: llm
Reread this week's episodes/, extract patterns, and merge into knowledge/.
```

The legacy format (`## Task name (Daily 9:00 JST)` with the schedule in parentheses) can be converted to the new format with `animaworks migrate-cron`.

### CronTask schema

Each task is parsed internally into the following `CronTask` model:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | (required) | Task name. Taken from the `##` heading |
| `schedule` | str | (required) | Standard 5-field cron expression. From the `schedule:` directive |
| `type` | str | `"llm"` | Task kind: `"llm"` or `"command"` |
| `description` | str | `""` | LLM instructions (for `type: llm`) |
| `command` | str \| None | `None` | Bash command for command type |
| `tool` | str \| None | `None` | Internal tool name for command type |
| `args` | dict \| None | `None` | Tool arguments (YAML) |
| `skip_pattern` | str \| None | `None` | Command type: skip follow-up LLM when stdout matches this regex |
| `trigger_heartbeat` | bool | `True` | Command type: if `False`, skip follow-up cron LLM after command output |

## LLM-type Cron tasks

`type: llm` tasks are run by the agent (LLM) with judgment and reasoning.
Text in `description` is passed to the agent as the prompt.

### Characteristics

- The agent uses tools, searches memory, and decides
- Output is unstructured (varies by task)
- Model API calls are required (cost)

### Example

```markdown
## Morning work plan
schedule: 0 9 * * *
type: llm
Reread yesterday's episodes/ and plan today's tasks.
Prioritize against vision and goals.
Write results to state/current_state.md.
Also check task_queue.jsonl for pending tasks and revise priorities if needed.
```

The description (body after the `type:` line) SHOULD include:

- What to check (inputs)
- How to decide (criteria)
- What to produce (artifacts)

## Command-type Cron tasks

`type: command` runs a fixed command or tool without going through the agent.
Suited to deterministic work (backups, sending notifications, etc.).

### Bash command type

```markdown
## Run backup
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

Put one line of bash after `command:`.
The command runs through a shell.

### Internal tool type

```markdown
## Slack morning greeting
schedule: 0 9 * * 1-5
type: command
tool: slack_send
args:
  channel: "#general"
  message: "Good morning! Thanks in advance for today."
```

Put the internal tool name after `tool:` and arguments in YAML under `args:`.
`args` is parsed as a YAML indented block (2-space indent).

### Follow-up control for command type

For command-type tasks, when the command exits successfully and has stdout, that output is passed to the LLM for follow-up analysis (heartbeat-equivalent context).

- **`trigger_heartbeat: false`** — Skip follow-up LLM when output analysis is unnecessary
- **`skip_pattern: <regex>`** — Skip follow-up when stdout matches this regex

```markdown
## Fetch logs (no output analysis)
schedule: 0 8 * * *
type: command
trigger_heartbeat: false
command: /usr/local/bin/fetch-logs.sh

## Health check (skip analysis when "OK")
schedule: */15 * * * *
type: command
skip_pattern: ^OK$
command: /usr/local/bin/health-check.sh
```

### Choosing LLM vs command type

| Aspect | LLM type | Command type |
|--------|----------|----------------|
| Needs judgment? | Yes | No |
| API cost | Yes | No |
| Output predictability | Unstructured | Deterministic |
| Good for | Planning, reflection, writing | Backup, notifications, data fetch |
| On error | Agent can recover autonomously | Logged only |

When unsure:

- "Same thing every time" → command type (SHOULD)
- "Judgment changes with context" → LLM type (SHOULD)
- "Run command + interpret result" → LLM type and instruct command execution in `description`

## Schedule syntax

The `schedule:` directive in `cron.md` must use the **standard 5-field cron expression**.

### Standard cron expression (required)

```
minute hour day-of-month month day-of-week
```

Examples:

- `0 9 * * *` — Daily 9:00
- `0 9 * * 1-5` — Weekdays 9:00
- `*/30 9-17 * * *` — Every 30 minutes between 9:00 and 17:00
- `0 2 1 * *` — 2:00 on the 1st of each month
- `0 17 * * 5` — Every Friday 17:00

Timezone: configurable via `config.json` `system.timezone`. When unset, the system timezone is auto-detected.

### Migrating from Japanese schedule text

Legacy `cron.md` in the old format (`## Task name (Daily 9:00 JST)`) can be converted to standard cron with `animaworks migrate-cron`. Mapping:

| Japanese-style text | Cron example |
|---------------------|--------------|
| `Daily HH:MM` | `0 9 * * *` |
| `Weekdays HH:MM` | `0 9 * * 1-5` |
| `Every {weekday} HH:MM` | `0 17 * * 5` (Friday) |
| `Monthly Nth HH:MM` | `0 9 1 * *` |
| `Every X minutes` | `*/5 * * * *` |
| `Every X hours` | `0 */2 * * *` |

`Biweekly`, last day of month, and Nth weekday of month are not auto-converted. Write the cron expression by hand.

## How to check cron logs

Cron run results are written to server logs.
They are also broadcast over WebSocket as `anima.cron` events.

How to check:

- Server logs: `animaworks.lifecycle` logger at INFO
- Web UI: dashboard activity feed
- episodes/: For LLM tasks, the agent SHOULD write logs under episodes/

LLM task results are stored as `CycleResult` with:

- `trigger`: `"cron"`
- `action`: Short summary of agent behavior
- `summary`: Result summary text
- `duration_ms`: Runtime in milliseconds
- `context_usage_ratio`: Context usage ratio

## Common Cron configuration examples

### Basic set (recommended for all Anima)

```markdown
# Cron: {name}

## Morning work plan
schedule: 0 9 * * *
type: llm
Review yesterday's actions from episodes/ and pending tasks in task_queue.jsonl.
Set today's priorities and update state/current_state.md.

## Weekly reflection
schedule: 0 17 * * 5
type: llm
Reread this week's episodes/, extract patterns and lessons.
Write important findings to knowledge/.
If the same work repeats, consider capturing it in procedures/.
```

### External integration tasks

```markdown
## Slack daily wrap-up
schedule: 0 18 * * 1-5
type: command
tool: slack_send
args:
  channel: "#daily-report"
  message: "Finished today's work. Details at tomorrow's standup."

## GitHub Issue check
schedule: 0 10 * * 1-5
type: llm
Check new Issues and PRs in repos you own.
Report important ones to your supervisor.
```

### Memory maintenance

```markdown
## Knowledge inventory
schedule: 0 10 1 * *
type: llm
Review all files under knowledge/, tidy outdated or conflicting content.
Consider archiving low-importance knowledge.

## Procedure refresh check
schedule: 0 10 * * 1
type: llm
Review procedures/ and check they still match real operations.
Update procedures when they change.
```

### Commenting out

Wrap tasks you do not want to run in HTML comments:

```markdown
<!--
## Temporarily paused task
schedule: 0 15 * * *
type: llm
This task is paused for now.
-->
```

`## ` headings inside comments are ignored by the parser.

## Cron configuration hot reload

When `cron.md` is updated, the schedule reloads automatically, like `heartbeat.md`.
If an Anima edits `cron.md` itself, changes apply immediately (self-modify pattern).

On reload:

1. Remove all existing cron jobs for that Anima
2. Re-parse updated `cron.md` and register new jobs
3. Log `Schedule reloaded for '{name}'`

If you edit `cron.md` yourself:

- Put the `schedule:` directive immediately after the heading (`## Task name`) (MUST)
- Use the standard 5-field cron expression (MUST)
- Put the `type` line right after `schedule` (SHOULD)
