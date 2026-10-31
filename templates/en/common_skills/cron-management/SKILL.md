---
name: cron-management
description: >-
  cron.mdA skill for reading and writing in the correct format. Provides procedures for adding, updating, and deleting scheduled tasks.
  Use when: cron.mdwhen you need to edit, add cron expressions, add or delete LLM-type or command-type tasks, or maintain scheduled jobs.
---


## Framework-side implementation (for reference)

### cron (Scheduled Tasks)

Parsing of cron is handled by `core/runtime/schedule_parser.py` (`parse_cron_md` / `parse_schedule`), while registration, execution, and reloading are handled by `core/runtime/scheduler_manager.py` (APScheduler, `AsyncIOScheduler(timezone=get_app_timezone())`).

### `core/tasks/background.py` (a separate system from cron)

This module does **not perform cron scheduling**. It handles background execution of long-running tool calls and rotation of DM logs. Do not confuse it with cron.

**`BackgroundTaskManager`**

- **Role**: Executes target tools asynchronously on `asyncio` and saves status to disk.
- **Persistence location**: `state/background_tasks/{task_id}.json` (`pending` / `running` / `completed` / `failed`, result strings, errors, timestamps).
- **Public API (overview)**:
  - `submit(tool_name, tool_args, execute_fn)` — Runs synchronous execution functions in a thread pool.
  - `submit_async(...)` — Asynchronous `execute_fn` version.
  - `get_task` / `list_tasks` / `active_count` — Queries.
  - `cleanup_old_tasks(max_age_hours=24)` — Deletes completed and failed tasks older than **24 hours**. Additionally, deletes JSON files that have remained in `running` state for more than **48 hours** (crash orphans).
- **`from_profiles(anima_dir, ..., profiles, config_eligible)`**: The set of background target tools is merged from the following **3 layers** (later wins).
  1. In-code defaults `_DEFAULT_ELIGIBLE_TOOLS` (a fixed list compatible with Mode A)
  2. Entries from `EXECUTION_PROFILE` to `background_eligible: true` in each tool module (`core.integrations._base.get_eligible_tools_from_profiles`). Keys use the `"{tool}:{subcommand}"` format (consistent with Mode S's `submit`)
  3. `config_eligible` passed from `config.json` and others (explicit override)
- **`is_eligible(tool_name)`**: Matching is possible by either of the following names — schema name (e.g., `generate_3d_model`), profile key (e.g., `image_gen:3d`).
- **Examples included in `_DEFAULT_ELIGIBLE_TOOLS` by default** (values are approximate seconds): `generate_character_assets` / `generate_fullbody` / `generate_bustup` / `generate_icon` / `generate_chibi` / `generate_3d_model` / `generate_rigged_model` / `generate_animations` (30 each), `local_llm` / `run_command` (60 each).
- **Completion hook**: `Callable[[BackgroundTask], Awaitable[None]]` can be set in `on_complete`. The application side can write Markdown notifications to `state/background_notifications/` from here (`background.py` itself does not touch the notification directory).

**`rotate_dm_logs`**

- Truncates entries in `shared/dm_logs/*.jsonl` at `max_age_days` (default **7 days**), appending old lines to `{stem}.{YYYYMMDD}.archive.jsonl` as an archive. This is not a cron task, but it is defined in the same file as background maintenance, so it is listed here for reference.

---

## Structure of cron.md

### Overall configuration

```markdown
# Cron: {自分の名前}

## タスク名1
schedule: 0 9 * * *
type: llm
タスクの説明文...

## タスク名2
schedule: */5 * * * *
type: command
command: /path/to/script.sh
```

### Rules that must be strictly followed

1. **Each task must start with `## タスク名`** (H2 heading. Do not use H3 or H1)
2. **The `schedule:` line is mandatory** (must start with the keyword `schedule:`. Do not make it a `###` heading)
3. **The schedule must be a standard 5-field cron expression only** (`09:00` and `毎週金曜 17:00` are not allowed)
4. **The `type:` line is mandatory** (either `llm` or `command`)
5. **Insert a blank line between tasks** (for readability)

### Incorrect writing style

```markdown
❌ ### */5 * * * *           ← H3見出しにcron式を書いてはいけない
❌ ### 09:00                 ← 自然言語の時刻表記は不可
❌ ### 毎週金曜 17:00         ← 日本語のスケジュール表記は不可
❌ cron: 0 9 * * *           ← キー名は "schedule:" であること（"cron:" ではない）
❌ interval: 5m              ← interval形式は不可
❌ schedule: 0 9 * * * *     ← 6フィールドは不可（5フィールドのみ）
```

### Correct writing style

```markdown
✅ schedule: 0 9 * * *       ← "schedule:" + 半角スペース + 5フィールドcron式
✅ schedule: */5 * * * *
✅ schedule: 30 21 * * *
✅ schedule: 0 17 * * 5
```

---

## 5-field cron expression reference

### Field structure

```
schedule: 分 時 日 月 曜日
```

| Field | Position | Range | Description |
|-----------|------|------|------|
| Minute | 1 | 0-59 | At what minute to execute |
| Hour | 2 | 0-23 | At what hour to execute (24-hour format) |
| Day | 3 | 1-31 | On what day to execute |
| Month | 4 | 1-12 | In what month to execute |
| Weekday | 5 | 0-6 | On what weekday to execute (0=Mon, 6=Sun) |

**Note**: Weekday is **0=Monday, 6=Sunday** (APScheduler specification. This differs from the common cron convention where 0=Sunday)

### Special characters

| Character | Meaning | Example |
|------|------|-----|
| `*` | All values | `* * * * *` = every minute |
| `*/n` | Interval of n | `*/5 * * * *` = every 5 minutes |
| `n-m` | Range | `0 9-17 * * *` = every hour from 9:00 to 17:00 |
| `n,m` | List | `0 9,12,18 * * *` = 9:00, 12:00, 18:00 |
| `n-m/s` | Range + interval | `0 9-17/2 * * *` = every 2 hours from 9:00 to 17:00 |

### Commonly used schedule examples

#### Daily

| What you want to do | cron expression | Explanation |
|-------------|--------|------|
| Every morning at 9:00 | `0 9 * * *` | minute=0, hour=9 |
| Every morning at 9:30 | `30 9 * * *` | minute=30, hour=9 |
| Every day at 12:00 (noon) | `0 12 * * *` | minute=0, hour=12 |
| Every day at 18:00 | `0 18 * * *` | minute=0, hour=18 |
| Every evening at 21:30 | `30 21 * * *` | minute=30, hour=21 |
| Every day at 2:00 AM | `0 2 * * *` | minute=0, hour=2 |

#### Interval-based

| What you want to do | cron expression | Explanation |
|-------------|--------|------|
| Every 5 minutes | `*/5 * * * *` | minute=*/5（0,5,10,...,55） |
| Every 10 minutes | `*/10 * * * *` | minute=*/10（0,10,20,...,50） |
| Every 15 minutes | `*/15 * * * *` | minute=*/15（0,15,30,45） |
| Every 30 minutes | `*/30 * * * *` | minute=*/30（0,30） |
| Every hour | `0 * * * *` | at minute 0 of every hour |
| Every 2 hours | `0 */2 * * *` | at minute 0 of 0:00, 2:00, 4:00, ..., 22:00 |
| Every 5 minutes during business hours only | `*/5 9-17 * * *` | every 5 minutes from 9:00 to 17:55 |
| Every hour during business hours only | `0 9-17 * * *` | every hour on the hour from 9:00 to 17:00 |

#### Weekday-based

| What you want to do | cron expression | Explanation |
|-------------|--------|------|
| Every weekday morning at 9:00 | `0 9 * * 0-4` | weekday=0-4 (Mon-Fri) |
| Every Monday at 9:00 | `0 9 * * 0` | weekday=0 (Mon) |
| Every Friday at 17:00 | `0 17 * * 4` | weekday=4 (Fri) |
| Every Friday at 18:00 | `0 18 * * 4` | weekday=4 (Fri) |
| Every 30 minutes during business hours on weekdays | `*/30 9-17 * * 0-4` | weekdays 9:00-17:30 |
| Every weekend morning at 10:00 | `0 10 * * 5,6` | weekday=5,6 (Sat, Sun) |

#### Monthly

| What you want to do | cron expression | Explanation |
|-------------|--------|------|
| On the 1st of every month at 9:00 | `0 9 1 * *` | day=1 |
| On the 15th of every month at 12:00 | `0 12 15 * *` | day=15 |
| Near the last business day of every month (28th) | `0 17 28 * *` | day=28 (approximation) |
| On the first day of each quarter at 9:00 (Jan, Apr, Jul, Oct) | `0 9 1 1,4,7,10 *` | month=1,4,7,10 |

---

## Task type details

### type: llm — LLM judgment task

Tasks that require thinking and judgment. Write the task content as free text after `schedule:` and `type: llm`.

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_state.md に書き出す。
```

- The description is passed directly as the prompt to the LLM
- Clearly specifying the concrete output (what to write out) is effective
- Multiple lines are allowed
- If you want to include a fenced code block in the body (\`\`\`）が含まれると、パーサーが警告ログを出す（確定コマンドなら `type: command` should be considered)

### type: command — Command execution task

Bash commands or tool calls that execute deterministically.

#### Pattern A: bash command

```markdown
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

- Write the command to execute in a single line in `command:`
- Shell redirection (`>`, `>>`, `|`) is usable
- Multi-line commands are not recommended (combine into one line or use a script file)

#### Pattern B: tool call

```markdown
## Slack朝の通知
schedule: 0 9 * * 0-4
type: command
tool: slack_channel_post
args:
  channel_id: "C0123456789"
  text: "おはようございます！"
```

- Write the tool name in `tool:` (a schema name included in the permission settings of `permissions.json`. For example, for Slack posting, use `slack_channel_post`)
- After `args:`, use a YAML block format with 2-space indentation
- It is executed with `ToolHandler.handle(tool, args)`, and the result string is treated as the equivalent of stdout

### Option: skip_pattern

A **follow-up cron LLM** (an analysis session with heartbeat-equivalent context) that runs only when the command **succeeds** (`exit_code == 0`) via `type: command` and stdout is non-empty is **suppressed** when stdout matches this regular expression.

```markdown
## Chatwork未返信チェック
schedule: */5 * * * *
type: command
command: chatwork_cli.py unreplied --json
skip_pattern: "^\[\]$"
```

- Write a regular expression in `skip_pattern:` (at runtime, `re.search(skip_pattern, stdout)` is used)
- If the regular expression contains YAML special characters such as `[]`, enclose it in quotes (`"..."` or `'...'`). The parser automatically removes the outer quotes
- **At parse time**, an invalid regular expression → a warning is logged and `skip_pattern` is treated as unset (no follow-up suppression)
- **At runtime**, if `re.search` throws an exception (e.g., a residual invalid pattern) → a warning is logged and the follow-up is executed **without suppression**
- In the example above, the follow-up is skipped when the number of unreplied messages is 0 (`[]`)

### Option: trigger_heartbeat

Controls per-task whether to run the follow-up cron LLM when the command succeeds (following the evaluation order of `SchedulerManager._run_cron_task`).

```markdown
## Chatwork未返信チェック
schedule: */15 * * * *
type: command
command: animaworks-tool chatwork unreplied
skip_pattern: "^\[\]$"
trigger_heartbeat: false
```

- **Conditions under which a follow-up is considered** (when all are satisfied): `exit_code == 0` and stdout (after `.strip()`) is non-empty
- **Length of stdout**: The `stdout` that `run_cron_command` returns to the scheduler is truncated to the first **1000 characters**. This preview is also what is passed to the follow-up LLM. For the full log, refer to `state/cron_logs/` (JSONL)
- `trigger_heartbeat: false` — Even if the above conditions are satisfied, the follow-up cron LLM is **not executed** (evaluated before `skip_pattern`)
- `trigger_heartbeat: true` (default) — If the conditions are satisfied, the follow-up is executed (then `skip_pattern` is evaluated next)
- If `false`, `no`, or `0` is specified, it is suppressed. Otherwise, it is treated as true
- The follow-up cron LLM runs with a heartbeat-equivalent prompt filter (background context)
- When `exit_code != 0` or stdout is empty, the follow-up **does not start in the first place** (`skip_pattern` / `trigger_heartbeat` are irrelevant)

---

## Criteria for choosing between types

### Cases where type: command should be used
- The command to execute is fully determined
- Parameters are fixed (region, cluster name, profile, etc.)
- Judgment of the result is left to the cron LLM session

### Cases where type: llm should be used
- The execution content needs to change depending on the situation
- Investigation combining multiple tools is required
- Human-like judgment and analysis are needed at the execution stage

### Prohibited patterns
- Including a code block (a fixed command) in type: llm
  → That command should be type: command
- Writing "execute exactly as follows" but using type: llm
  → The LLM cannot accurately reproduce commands. Use type: command

### Anima is not "a person who memorizes commands"
type: command is the same as a human saving a script.
Anima's value lies in the ability to judge based on results.
Leave deterministic execution to the framework,
and have Anima focus on judgment, analysis, and reporting.

---

## cron.md Parse Notifications (Automatic)

When a parse or registration error is detected during the initial registration of cron.md or at `reload_schedule`, the scheduler generates a `state/background_notifications/cron_health_{タイムスタンプ}.md`. The notification is included in the context of the next heartbeat or cron execution. There is no feature for periodically monitoring the execution status of registered cron jobs.

**Detected issues**:

- A task is defined, but **no valid schedule can be registered** (e.g., all expressions are invalid)
- Outside of comments or code fences, there is a `schedule:` line with leading whitespace (`schedule:` must be written at the beginning of the line)
- The valid configuration text contains `schedule:`, but the parser **returns no tasks at all**

---

## cron.md Operating Procedures

### Target Paths and Editing Descendants

- Read your own `cron.md` with `read_memory_file(path="cron.md")`, and update it with `write_memory_file(path="cron.md", ...)`.
- A supervisor can edit descendants' `cron.md` / `HEARTBEAT.md` using the write memory tool by specifying them as `../{anima_name}/cron.md` (all descendants, including children and grandchildren, are covered).
- `status.json` / `IDENTITY.md` / `SOUL.md` / `permissions.json` are owned by root and cannot be written to using memory or file tools. Changes to `AGENTS.md` by an authorized supervisor and requests to create IDENTITY.md / SOUL.md during bootstrap are forwarded to root for authorization. Do not manipulate files directly.

### Adding a New Task

1. Load your own `cron.md`
2. Append a new section at the end of the file
3. **Verify the format before writing** (see the checklist below)
4. Write the file

```markdown
## 新しいタスク名
schedule: <5フィールドcron式>
type: llm|command
<説明またはcommand/tool行>
```

### Modifying an Existing Task

1. Load `cron.md`
2. Locate the relevant section (from `## タスク名` up to just before the next `##`)
3. Edit the lines to change (`schedule:`, `type:`, description text, etc.)
4. Write the file

### Deleting a Task

1. Load `cron.md`
2. Delete the entire relevant section (from `## タスク名` up to just before the next `##`)
3. Write the file

### Temporarily Disabling a Task

Wrapping it in HTML comments makes the parser skip it:

```markdown
<!--
## 一時停止中のタスク
schedule: 0 9 * * *
type: llm
このタスクは一時的に停止中。
-->
```

---

## Pre-Write Checklist

Before updating cron.md, **always** verify the following:

- [ ] Each task starts with `## タスク名` (not `###` or `#`)
- [ ] A `schedule:` line exists (not a `###` heading or natural language)
- [ ] The schedule is a 5-field cron expression (`分 時 日 月 曜日`)
- [ ] Each field's value is within the valid range (minute: 0-59, hour: 0-23, day: 1-31, month: 1-12, weekday: 0-6)
- [ ] A `type:` line exists (`llm` or `command`)
- [ ] For command type, `command:` or `tool:` exists
- [ ] For tool type, the indentation of `args:` is correct (2 spaces)
- [ ] There is a blank line between tasks
- [ ] `schedule:` is not written inside a code block (this can cause health warnings)

### Validation Method

After writing, you can verify that parsing works correctly with the following command:

```bash
# プロジェクトルートで実行。ANIMAWORKS_ANIMA_DIR 未設定時は ~/.animaworks/animas/default を使用
python -c "
from core.runtime.schedule_parser import parse_cron_md, parse_schedule
import os
from pathlib import Path

cron_path = Path(os.environ.get('ANIMAWORKS_ANIMA_DIR', '~/.animaworks/animas/default')) / 'cron.md'
content = cron_path.expanduser().read_text()
tasks = parse_cron_md(content)
for t in tasks:
    trigger = parse_schedule(t.schedule)
    status = '✅' if trigger else '❌ パース失敗'
    print(f'{status} {t.name}: schedule=\"{t.schedule}\" type={t.type}')
"
```

If all tasks show ✅, everything is normal. If ❌ appears, correct the schedule expression.

---

## Complete Example

```markdown
# Cron: example_anima

## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_state.md に書き出す。

## Chatwork未返信チェック
schedule: */5 9-18 * * 0-4
type: command
command: chatwork-cli unreplied --json > $ANIMAWORKS_ANIMA_DIR/state/chatwork_unreplied.json
skip_pattern: "^\[\]$"
trigger_heartbeat: false

## Slack朝の挨拶
schedule: 0 9 * * 0-4
type: command
tool: slack_channel_post
args:
  channel_id: "C0123456789"
  text: "おはようございます！今日もよろしくお願いします。"

## 週次振り返り
schedule: 0 17 * * 4
type: llm
今週のepisodes/を読み返し、パターンを抽出してknowledge/に統合する。
改善点があれば procedures/ に手順を追記する。

## 月次レポート
schedule: 0 10 1 * *
type: llm
先月のepisodes/とknowledge/を分析し、月次サマリーレポートを作成する。
レポートは knowledge/monthly_report_YYYY-MM.md として保存する。
```

---

## Common Mistakes and Fixes

| Mistake | Correct Form | Reason |
|--------|------------|------|
| `### */5 * * * *` | `schedule: */5 * * * *` | H3 headings cannot be used as task separators |
| `### 09:00` | `schedule: 0 9 * * *` | Natural language times cannot be parsed |
| `### 毎週金曜 17:00` | `schedule: 0 17 * * 4` | Japanese notation cannot be parsed |
| `schedule: 9:00` | `schedule: 0 9 * * *` | HH:MM format is not a 5-field cron expression |
| `schedule: every 5 minutes` | `schedule: */5 * * * *` | English notation cannot be parsed |
| `schedule: 0 9 * * 7` | `schedule: 0 9 * * 6` | Weekday 7 is out of range (0-6) |
| `schedule: 0 9 * * SUN` | `schedule: 0 9 * * 6` | Weekday names cannot be used (numbers only) |
| `schedule: 0 25 * * *` | `schedule: 0 23 * * *` | Hour must be in the range 0-23 |
| `schedule: 60 * * * *` | `schedule: 0 * * * *` | Minute must be in the range 0-59 |
| `skip_pattern: ^\[\]$` (without quotes) | `skip_pattern: "^\[\]$"` | `[]` is interpreted as an empty list in YAML. Wrap it in quotes |

---

## Notes

- **Hot reload**: When `cron.md` / `HEARTBEAT.md` are updated through Anima's `write_memory_file`, etc., the schedule-change callback runs `reload_schedule` and immediately registers them again. Edits to descendant management files should also be made using the write memory tool, as in `../{anima_name}/cron.md`. External direct edits are also detected by the mtime check (`_check_schedule_freshness`) immediately before the next **heartbeat or any cron fires**, but this is not the standard procedure for a supervisor editing descendants.
- **Stale jobs immediately after reload**: When the scheduler is rebuilt after detecting an mtime change, cron firings tied to the **previous definition may be skipped as “stale”** (to intentionally prevent duplicate execution).
- **Time zone**: APScheduler uses `get_app_timezone()`. In `config.json`, set `system.timezone` to an IANA name (e.g., `Asia/Tokyo`). When it is an **empty string**, the OS time zone is detected automatically, and if that fails, it falls back to `Asia/Tokyo`.
- **Concurrency**: **If task names differ**, multiple jobs can run concurrently via `asyncio.create_task` even when they overlap within the same minute. The same task name will not re-enter while running (skipped by `_cron_running`). Each job is `max_instances=1`.
- Even if a command-type task fails, the process does not stop (`cron_executed` is recorded, and `stderr` / exit_code are logged). **The follow-up cron LLM runs only when the command succeeds and has stdout** (as mentioned above).
- Periodic execution of `type: llm` uses the **background model** (such as `background_model` in `status.json`; the main model is used if unset).
- Detailed command output accumulates in `state/cron_logs/` (daily JSONL) and is housekeeping-cleaned after a maximum of 14 days.
- **Check permissions in advance when accessing another Anima's tools/ directory.**
