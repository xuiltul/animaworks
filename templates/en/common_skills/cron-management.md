---
name: cron-management
description: >-
  Skill for reading and writing cron.md in the correct format.
  Always follow this skill's procedure when adding, changing, or removing Cron tasks.
  Use for: "cron setup", "cron task", "schedule", "scheduled task", "cron.md",
  "add cron", "update cron", "remove cron", "periodic execution", "scheduled".
---

## cron.md Structure

### Overall Structure

```markdown
# Cron: {your_name}

## Task Name 1
schedule: 0 9 * * *
type: llm
Task description...

## Task Name 2
schedule: */5 * * * *
type: command
command: /path/to/script.sh
```

### Rules You MUST Follow

1. **Each task must start with `## Task Name`** (H2 header. Do not use H3 or H1)
2. **`schedule:` line is required** (must start with `schedule:` keyword. Not a `###` header)
3. **Schedule must be standard 5-field cron expression only** (`09:00` or `every Friday 17:00` are invalid)
4. **`type:` line is required** (`llm` or `command`)
5. **Insert blank lines between tasks** (for readability)

### Invalid Formats

```markdown
❌ ### */5 * * * *           ← Do NOT write cron expression as H3 header
❌ ### 09:00                 ← Natural language time is invalid
❌ ### Every Friday 17:00    ← Natural language schedule is invalid
❌ cron: 0 9 * * *           ← Key name must be "schedule:" (not "cron:")
❌ interval: 5m              ← Interval format is invalid
❌ schedule: 0 9 * * * *     ← 6 fields are invalid (5 fields only)
```

### Valid Format

```markdown
✅ schedule: 0 9 * * *       ← "schedule:" + space + 5-field cron expression
✅ schedule: */5 * * * *
✅ schedule: 30 21 * * *
✅ schedule: 0 17 * * 5
```

---

## 5-Field Cron Expression Reference

### Field Structure

```
schedule: minute hour day month weekday
```

| Field | Position | Range | Description |
|-------|----------|-------|-------------|
| minute | 1 | 0-59 | Minute to execute |
| hour | 2 | 0-23 | Hour to execute (24-hour) |
| day | 3 | 1-31 | Day of month |
| month | 4 | 1-12 | Month |
| weekday | 5 | 0-6 | Day of week (0=Mon, 6=Sun) |

**Note**: Weekday is **0=Monday, 6=Sunday** (APScheduler spec. Differs from standard cron where 0=Sunday)

### Special Characters

| Char | Meaning | Example |
|------|---------|---------|
| `*` | All values | `* * * * *` = every minute |
| `*/n` | Every n | `*/5 * * * *` = every 5 minutes |
| `n-m` | Range | `0 9-17 * * *` = every hour 9am–5pm |
| `n,m` | List | `0 9,12,18 * * *` = 9am, noon, 6pm |
| `n-m/s` | Range + step | `0 9-17/2 * * *` = every 2 hours 9am–5pm |

### Common Schedule Examples

#### Daily

| Intent | Cron Expression | Notes |
|--------|-----------------|-------|
| Every day 9:00 AM | `0 9 * * *` | minute=0, hour=9 |
| Every day 9:30 AM | `30 9 * * *` | minute=30, hour=9 |
| Every day noon | `0 12 * * *` | minute=0, hour=12 |
| Every day 6:00 PM | `0 18 * * *` | minute=0, hour=18 |
| Every day 9:30 PM | `30 21 * * *` | minute=30, hour=21 |
| Every day 2:00 AM | `0 2 * * *` | minute=0, hour=2 |

#### Intervals

| Intent | Cron Expression | Notes |
|--------|-----------------|-------|
| Every 5 minutes | `*/5 * * * *` | minute=*/5 (0,5,10,...,55) |
| Every 10 minutes | `*/10 * * * *` | minute=*/10 |
| Every 15 minutes | `*/15 * * * *` | minute=*/15 |
| Every 30 minutes | `*/30 * * * *` | minute=*/30 |
| Every hour | `0 * * * *` | Every hour at :00 |
| Every 2 hours | `0 */2 * * *` | 0:00, 2:00, ..., 22:00 |
| Every 5 min during business hours | `*/5 9-17 * * *` | 9:00–17:55 |
| Every hour during business hours | `0 9-17 * * *` | 9:00–17:00 |

#### Weekday

| Intent | Cron Expression | Notes |
|--------|-----------------|-------|
| Weekdays 9:00 AM | `0 9 * * 0-4` | weekday=0-4 (Mon–Fri) |
| Every Monday 9:00 AM | `0 9 * * 0` | weekday=0 (Mon) |
| Every Friday 5:00 PM | `0 17 * * 4` | weekday=4 (Fri) |
| Every Friday 6:00 PM | `0 18 * * 4` | weekday=4 (Fri) |
| Weekdays every 30 min during business hours | `*/30 9-17 * * 0-4` | Mon–Fri 9:00–17:30 |
| Weekend 10:00 AM | `0 10 * * 5,6` | weekday=5,6 (Sat–Sun) |

#### Monthly

| Intent | Cron Expression | Notes |
|--------|-----------------|-------|
| 1st of month 9:00 AM | `0 9 1 * *` | day=1 |
| 15th of month noon | `0 12 15 * *` | day=15 |
| 28th of month 5:00 PM (approx. month-end) | `0 17 28 * *` | day=28 |
| Quarter start 9:00 AM (Jan, Apr, Jul, Oct) | `0 9 1 1,4,7,10 *` | month=1,4,7,10 |

---

## Task Type Details

### type: llm — LLM Judgment Task

Tasks that require thinking and judgment. After `schedule:` and `type: llm`, write the task content in free form.

```markdown
## Morning Planning
schedule: 0 9 * * *
type: llm
Check yesterday's progress from long-term memory and plan today's tasks.
Determine priorities based on vision and goals.
Write the result to state/current_task.md.
```

- The description is passed directly as the LLM prompt
- Stating the concrete output (what to write) is effective
- Multiple lines are allowed

### type: command — Command Execution Task

Deterministic execution of bash commands or tool calls.

#### Pattern A: bash command

```markdown
## Backup Execution
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

- Write the command in one line under `command:`
- Shell redirection (`>`, `>>`, `|`) is allowed
- Multi-line commands are discouraged (combine into one line or use a script file)

#### Pattern B: Tool invocation

```markdown
## Slack Morning Greeting
schedule: 0 9 * * 0-4
type: command
tool: slack_send
args:
  channel: "#general"
  message: "Good morning!"
```

- `tool:` contains the tool name (schema name from `get_tool_schemas()`)
- `args:` and below are YAML block format with 2-space indent
- Executed via `dispatch(tool_name, args)`

### Option: skip_pattern

Skip LLM analysis session when command stdout matches a pattern.

```markdown
## Chatwork Unreplied Check
schedule: */5 * * * *
type: command
command: chatwork_cli.py unreplied --json
skip_pattern: ^\[\]$
```

- `skip_pattern:` contains a regex
- In the example above, skips when unreplied count is 0 (`[]`)

### Option: trigger_heartbeat

Per-task control of whether to trigger LLM analysis session on command output.

```markdown
## Chatwork Unreplied Check
schedule: */15 * * * *
type: command
command: animaworks-tool chatwork unreplied
skip_pattern: ^\[\]$
trigger_heartbeat: false
```

- `trigger_heartbeat: false` — Skip LLM analysis even when output exists
- `trigger_heartbeat: true` (default) — Analyze and respond in cron LLM session when output exists
- `false`, `no`, `0` suppress LLM analysis. Anything else is treated as true
- Can be used with `skip_pattern`. `trigger_heartbeat: false` is evaluated before `skip_pattern`
- LLM analysis session has Heartbeat-equivalent context (memory, Knowledge, org info)

---

## Choosing Task Type

### Use type: command when
- The command to execute is fully determined
- Parameters are fixed (region, cluster name, profile, etc.)
- Let the cron LLM session handle result interpretation

### Use type: llm when
- Execution content must vary by situation
- Multiple tools need to be combined for investigation
- Human-like judgment or analysis is needed at execution time

### Forbidden Patterns
- Including fixed commands in type: llm → That command should be type: command
- Writing "execute exactly this" but using type: llm → LLM cannot reliably reproduce commands. Use type: command

### Anima is Not a "Command Memorizer"
type: command is like a human saving a script.
Anima's value lies in judging and analyzing results.
Let the framework handle deterministic execution,
and let Anima focus on judgment, analysis, and reporting.

---

## cron.md Operations

### Add New Task

1. Read your `cron.md`
2. Add a new section at the end of the file
3. **Verify format before writing** (see checklist below)
4. Write the file

```markdown
## New Task Name
schedule: <5-field cron expression>
type: llm|command
<description or command/tool lines>
```

### Update Existing Task

1. Read `cron.md`
2. Locate the section (`## Task Name` to before next `##`)
3. Edit the lines (schedule:, type:, description, etc.)
4. Write the file

### Remove Task

1. Read `cron.md`
2. Delete the entire section (`## Task Name` to before next `##`)
3. Write the file

### Temporarily Disable Task

Wrap in HTML comment so the parser skips it:

```markdown
<!--
## Paused Task
schedule: 0 9 * * *
type: llm
This task is temporarily paused.
-->
```

---

## Pre-Write Checklist

**Always** verify the following before updating cron.md:

- [ ] Each task starts with `## Task Name` (not `###` or `#`)
- [ ] `schedule:` line exists (not `###` header or natural language)
- [ ] Schedule is 5-field cron (minute hour day month weekday)
- [ ] Each field value is in valid range (min: 0-59, hour: 0-23, day: 1-31, month: 1-12, weekday: 0-6)
- [ ] `type:` line exists (`llm` or `command`)
- [ ] For command type, `command:` or `tool:` exists
- [ ] For tool type, `args:` indent is correct (2 spaces)
- [ ] Blank lines between tasks

### Validation

After writing, you can verify parsing with:

```bash
python -c "
from core.schedule_parser import parse_cron_md, parse_schedule
import pathlib

content = pathlib.Path('$ANIMAWORKS_ANIMA_DIR/cron.md').read_text()
tasks = parse_cron_md(content)
for t in tasks:
    trigger = parse_schedule(t.schedule)
    status = '✅' if trigger else '❌ parse failed'
    print(f'{status} {t.name}: schedule=\"{t.schedule}\" type={t.type}')
")
```

If all tasks show ✅, it is valid. Fix schedule expressions for any ❌.

---

## Full Example

```markdown
# Cron: example_anima

## Morning Planning
schedule: 0 9 * * *
type: llm
Check yesterday's progress from long-term memory and plan today's tasks.
Determine priorities based on vision and goals.
Write the result to state/current_task.md.

## Chatwork Unreplied Check
schedule: */5 9-18 * * 0-4
type: command
command: /home/main/dev/chatwork-cli/chatwork_cli.py unreplied --json > $ANIMAWORKS_ANIMA_DIR/state/chatwork_unreplied.json
skip_pattern: ^\[\]$
trigger_heartbeat: false

## Slack Morning Greeting
schedule: 0 9 * * 0-4
type: command
tool: slack_send
args:
  channel: "#general"
  message: "Good morning! Looking forward to working with you today."

## Weekly Reflection
schedule: 0 17 * * 4
type: llm
Re-read episodes/ from this week, extract patterns, and integrate into knowledge/.
Add procedures to procedures/ if improvements exist.

## Monthly Report
schedule: 0 10 1 * *
type: llm
Analyze episodes/ and knowledge/ from last month and create a monthly summary report.
Save the report as knowledge/monthly_report_YYYY-MM.md.
```

---

## Common Mistakes and Fixes

| Mistake | Correct | Reason |
|---------|---------|--------|
| `### */5 * * * *` | `schedule: */5 * * * *` | H3 header cannot be used for task separation |
| `### 09:00` | `schedule: 0 9 * * *` | Natural language time cannot be parsed |
| `### Every Friday 17:00` | `schedule: 0 17 * * 4` | Natural language cannot be parsed |
| `schedule: 9:00` | `schedule: 0 9 * * *` | HH:MM format is not 5-field cron |
| `schedule: every 5 minutes` | `schedule: */5 * * * *` | Natural language cannot be parsed |
| `schedule: 0 9 * * 7` | `schedule: 0 9 * * 6` | Weekday 7 is out of range (0-6) |
| `schedule: 0 9 * * SUN` | `schedule: 0 9 * * 6` | Weekday names are invalid (numbers only) |
| `schedule: 0 25 * * *` | `schedule: 0 23 * * *` | Hour range is 0-23 |
| `schedule: 60 * * * *` | `schedule: 0 * * * *` | Minute range is 0-59 |

---

## Notes

- cron.md changes take effect on server restart or next Heartbeat (not immediate)
- Timezone follows server config (usually JST)
- Multiple tasks at the same time run in parallel
- Command-type task failures do not stop the server (logged)
- LLM-type tasks use the current model config from status.json
- **When accessing other Anima's tools/ directory, verify permissions first**
