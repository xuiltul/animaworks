# Scheduled Execution Configuration and Operations

A guide to configuring and operating Heartbeat (periodic patrol) and Cron (scheduled tasks).
Refer to this when changing scheduled execution behavior or adding new scheduled tasks.

## What is Heartbeat

Heartbeat is a mechanism where Digital Anima automatically starts periodically to check the situation and make plans.
It automates the same behavior as a human periodically checking their inbox and reviewing ongoing work.

### Important: Heartbeat is for "checking and planning" only

Heartbeat checks for meaningful changes and determines the necessary next action. No ceremonial retrospective reports are needed.

- MUST: Heartbeat is limited to situation checks and decision-making
- MUST NOT: Do not run long-duration tasks (coding, extensive tool calls, etc.) within Heartbeat
- MUST: If you find a task that needs execution, delegate it to a subordinate via `delegate_task`, or submit it as a task via `submit_tasks`

Submitted tasks are executed by the **TaskExec path**, independently of the periodic Heartbeat.
TaskExec retrieves executable persistent tasks from the authoritative TaskStore. Wake-up notifications and recovery are managed by the host; legacy LLM JSON files are not monitored.

### Heartbeat and Conversation Concurrency

Heartbeat and human conversations are managed under **separate locks**, so they can run simultaneously.
Even while Heartbeat is running, immediate responses to human messages are possible.

### Submitting tasks via submit_tasks

When a task to be executed is discovered in the heartbeat, submit the task using the `submit_tasks` tool:

```
submit_tasks(batch_id="hb-20260301-api-test", tasks=[
  {"task_id": "api-test", "title": "APIテスト実施",
   "description": "Slack API接続テストを実施し、全エンドポイントの結果をレポートにまとめる。完了後 aoi に報告する。"}
])
```

After validation, `submit_tasks` saves the task and its complete execution input in bulk to a single authoritative TaskStore.
The host manages execution rights and attempt history. `in_progress` is for viewing; the agent declares `done` / `pending` / `cancelled` via `update_task`. To resume an interrupted pending task, explicitly specify `resume: true` with the same ID; do not replace it with a different task.

**Long-running CLI tools** (`animaworks-tool submit …`) are registered in the TaskStore as `task_type="command"`, and the PendingTaskExecutor retrieves attempts and runs them in the background. Check completion results via `state/background_tasks/{task_id}.json` and notifications. See `operations/background-tasks.md` for details.

**Note**: Do not edit the storage location directly. The old `state/task_queue.jsonl` and `state/pending/` are kept as audit trails for migration and export; do not use them as active submission targets.

Even for a single task, use `submit_tasks` (one item in the tasks array).
Run multiple independent tasks in parallel with `parallel: true`; if there are dependencies, specify `depends_on`.
See task-management for details.

### Heartbeat trigger types

There are two types of heartbeat triggers:

| Trigger | Description |
|---------|------|
| Scheduled heartbeat | Started periodically by APScheduler according to `heartbeat.interval_minutes` in `config.json` |
| Message trigger | Started immediately when an unread message arrives in the Inbox (processed as an Inbox pass) |

The message trigger is activated by a change notification for the Inbox JSON file. As a safeguard against missed notifications, unread messages are rechecked every 45 seconds.
Only one Inbox process runs at a time; messages that arrive during execution are batched into the next single pass. On a Provider error, wait for the recovery time in `rate_guard` and leave unread messages.
There is no startup filter based on received intent, nor message-originated cooldown or cascade suppression. The behavior rule of not replying to messages that only acknowledge or thank is instructed in the Inbox prompt.

## HEARTBEAT.md Configuration

`HEARTBEAT.md` is each Anima's configuration file, defining activity hours and check items.
The Heartbeat execution interval can be set via `heartbeat.interval_minutes` in `config.json` (1–1440 minutes, default 30). It cannot be changed via `HEARTBEAT.md`.
Each Anima is assigned a name-based offset of 0–9 minutes to distribute simultaneous startups.
File path: `~/.animaworks/animas/{name}/HEARTBEAT.md`

When a supervisor edits `HEARTBEAT.md` of a subordinate Anima, use `read_memory_file` / `write_memory_file` rather than direct file operations, and specify it with a relative path such as `../{anima_name}/HEARTBEAT.md`.

### Format

```markdown
# Heartbeat: {name}

## 活動時間
24時間（サーバー設定タイムゾーン）

## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーが発生していないか
- 自分の作業領域に新しいファイルが置かれていないか
- 何もなければ何もしない（HEARTBEAT_OK）

## 通知ルール
- 緊急と判断した場合のみ関係者に通知
- 同じ内容の通知は24時間以内に繰り返さない
```

### Configuration Fields

**Execution interval**:
- Set via `heartbeat.interval_minutes` in `config.json` (1–1440 minutes, default 30). Cannot be changed via `HEARTBEAT.md`

**Activity hours** (SHOULD):
- Write in `HH:MM - HH:MM` format (e.g., `9:00 - 22:00`)
- Heartbeat does not start outside these hours
- Default when unset: 24 hours (all time slots)
- Timezone can be set via `system.timezone` in `config.json`. When unset, the system timezone is auto-detected

**Checklist** (MUST):
- Items the agent checks at Heartbeat startup
- Written as a bullet list (starting with `- `)
- The checklist content is passed directly to the agent's prompt
- Customizable: items may be added or changed to match the Anima's role

### Checklist Customization Examples

Default (common to all Anima):
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

Example for development:
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーが発生していないか
- 監視対象のGitHubリポジトリに新しいIssueやPRがないか
- CI/CDの失敗アラートがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

Example for communications:
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- Slackの未読メンションがないか
- 返信待ちのメールがないか
- 進行中タスクにブロッカーがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

### Execution Model (Cost Optimization)

Heartbeat / Inbox / Cron are executed with `background_model` when it is configured, instead of the main model.
Chat (interaction with humans) and TaskExec (actual work) continue to use the main model.

Configuration method: `animaworks anima set-background-model {名前} claude-sonnet-5-5`
For details, refer to the "Background Model" section in `reference/operations/model-guide.md`.

### Heartbeat internal behavior

- **Crash recovery**: If the previous heartbeat failed, error information is saved in `state/recovery_note.md`. It is injected into the prompt at the next startup, and the file is deleted after recovery.
- **Reflection record**: If the heartbeat output contains a `[REFLECTION]...[/REFLECTION]` block, it is recorded as `heartbeat_reflection` in the activity_log and included in subsequent heartbeat contexts.
- **Subordinate check**: For Anima instances with subordinates, a subordinate status check instruction is automatically injected into the heartbeat and Cron prompts.
- **Session time limit** (`heartbeat` in `config.json`): After `soft_timeout_seconds` (default 300 seconds) elapses, a wrap-up reminder is injected; the session is forcibly terminated after `hard_timeout_seconds` (default 600 seconds).
- **Idle auto-compaction**: `heartbeat.idle_compaction_minutes` (default 10 minutes) — idle auto-compaction runs after this time has elapsed since stream end (execution engine setting).
- **Board posts**: Limited to once per channel within the same run. There is no restriction on the interval between posts across runs.

### Periodic Heartbeat Scheduling Method

When the effective interval (after Activity Level application, rounded down to 5 minutes) is **60 minutes or less and divides 60 evenly**, it is registered in minute slots via APScheduler's `CronTrigger` (combined with the name-based 0–9 minute offset).

Otherwise (e.g., effective 61 minutes, or an interval like 43 minutes that does not divide 60 evenly), it fires via **1-minute polling** (`_heartbeat_check`) based on elapsed time since the last run. This is an implementation to avoid issues from the legacy `IntervalTrigger`.

### Background Tools and DM Logs (core/tasks/background.py）

`core/tasks/background.py` is not the Heartbeat/Cron schedule itself, but handles **background execution of long-running tool calls** (including JSON state persistence) and **rotation of legacy shared DM logs (`shared/dm_logs/`)**. See `operations/background-tasks.md` for operational details and CLI paths.

#### BackgroundTaskManager

- **Storage location**: `state/background_tasks/{task_id}.json`. `task_id` is the first 12 characters of the UUID (hexadecimal). Each file records `task_id`, `anima_name`, `tool_name`, `tool_args`, `status`, `created_at`, `completed_at`, `result`, and `error`.
- **Status (`TaskStatus`)**: `pending` / `running` / `completed` / `failed`. For `submit` / `submit_async`, JSON is written with `running` immediately after submission, and updated to `completed` / `failed` upon completion or exception.
- **Execution API**: `submit(tool_name, tool_args, execute_fn)` runs synchronous callables in a thread pool via `asyncio`’s `run_in_executor`. `submit_async` directly awaits asynchronous callables. It provides `get_task` (memory first, disk if unavailable), `list_tasks` (also merges JSON on disk, sorted by creation time, newest first), and `active_count` (the number of `running` in memory).
- **Completion callback**: The asynchronous function passed to `on_complete` is called after the task is saved. Even if an exception occurs in the callback, the task result is retained and only logged (typically combined with writing to `state/background_notifications/`, through which it is read and deleted on the **next heartbeat** and incorporated into the conversation context).
- **Candidate determination `is_eligible(tool_name)`**: If a key exists in the map, it is considered a background task candidate. The key accepts both of the following: (1) **schema name** (e.g., `generate_3d_model`)—such as external tool dispatch in Mode A. (2) **`ツール名:サブコマンド`** (e.g., `image_gen:pipeline`)—in `EXECUTION_PROFILE` of each tool module, entries for `background_eligible: true` are registered in this format by `get_eligible_tools_from_profiles()` (such as the `submit` path in Mode S).
- **Building the candidate tool map `from_profiles()`**: Merge the following three layers using dict `update`, with **later entries taking precedence**: (1) `_DEFAULT_ELIGIBLE_TOOLS` (code defaults) (2) argument `profiles` (`EXECUTION_PROFILE` aggregation) (3) argument `config_eligible` (usually `config.json`’s `background_task.eligible_tools`, from which `threshold_s` is expanded into `名前 → 秒`). Values are expected seconds (integers) for profile integration.
- **Code defaults `_DEFAULT_ELIGIBLE_TOOLS` (seconds)**: `generate_character_assets` 30, `generate_fullbody` / `generate_bustup` / `generate_icon` / `generate_chibi` 30 each, `generate_3d_model` / `generate_rigged_model` / `generate_animations` 30 each, `local_llm` 60, `run_command` 60.
- **Cleanup `cleanup_old_tasks(max_age_hours=24)`**: `status` deletes JSON files in `completed` / `failed` whose `completed_at` is **older than the duration specified by the argument (24 hours by default)**. In addition, files that remain `running` for **more than 48 hours** since `created_at` are deleted as crash orphans. The return value is the number of deleted files.
- **Retention period**: The retention period for completed tasks in `cleanup_old_tasks` is specified by the argument `max_age_hours` (24 hours by default). There is no corresponding `config.json` configuration key.

#### rotate_dm_logs (System Cron)

- **Execution timing**: Runs **daily at 04:30** (server-configured timezone) via the lifecycle (e.g., `core/lifecycle/system_crons.py`) system cron. The same job ID is also registered on the `server/supervisor/_mgr_scheduler.py` side.
- **Target**: `shared/dm_logs/*.jsonl` (entries whose filenames contain `.archive.` are skipped).
- **Behavior**: Based on the local current time of `core.time_utils`, parse the `ts` (ISO format) in each line of JSON, append entries older than the **default 7 days** to `{stem}.{YYYYMMDD}.archive.jsonl` as an archive, then remove them from the current file. Lines where `ts` parsing fails are **kept in the current file** (to prevent data loss).
- **Other server scheduled jobs**: In addition to the per-Anima `cron.md`, the lifecycle registers system cron jobs for memory maintenance, RAG, etc. (e.g., daily consolidation at 02:00, daily indexing at 04:00). Times are based on the configured timezone. DM rotation runs at 04:30 as described above.

### Heartbeat configuration hot reload

When HEARTBEAT.md is updated on the file system, `_check_schedule_freshness()` detects the change at the next heartbeat execution, and SchedulerManager automatically reloads the schedule.
No server restart is required (MAY skip restart). APScheduler jobs are re-registered.

## Per-anima Heartbeat interval configuration

### Configuration at status.json

The root administrator can configure Anima-specific `heartbeat_interval_minutes` at root CLI/API. Values are stored in root-owned `status.json` and are not edited from the Anima process.

```json
{
  "heartbeat_interval_minutes": 60
}
```

- Configurable range: 1–1440 minutes (1 day)
- If not set: falls back to `heartbeat.interval_minutes` of `config.json` (default 30 minutes)
- `status.json` is root-owned. Administrators can set individual overrides via `animaworks config set animas.<name>.heartbeat_interval_minutes <minutes>` while the server is stopped. During server startup, the CLI applies them through the root API. The Anima process does not edit this file.

### Recommended guidelines

| Situation | Recommended interval | Reason |
|------|----------|------|
| During an active development project | 15 to 30 minutes | Frequent status awareness needed |
| Normal operations | 30 to 60 minutes | Default. Balanced frequency |
| Low load / standby state | 60 to 120 minutes | Cost savings. Longer if there are no tasks |
| Long-term dormancy / inactive | 120 to 1440 minutes | Minimal polling for status awareness |

### Relationship with Activity Level

When a global Activity Level (10% to 400%) is set, the effective interval is calculated with the following formula:

```
実効間隔 = ベース間隔 / (Activity Level / 100)
```

Example: base 30 minutes, Activity Level 50% → effective 60 minutes
Example: base 30 minutes, Activity Level 200% → effective 15 minutes

- The lower limit of the effective interval is 5 minutes (no matter how much it is boosted, it will not go below 5 minutes)

### Activity Schedule (automatic switching by time period / night mode)

A mechanism that automatically switches the Activity Level according to the time period.
Use this when you want to reduce costs at night or on holidays, or when you want the system to operate actively only during business hours.

#### Mechanism

- Set time period entries in `activity_schedule` of `config.json`
- The current time is checked every minute, and the Activity Level is automatically changed to the level of the matching time period
- When the Activity Level changes, the heartbeats of all Anima are immediately rescheduled

#### Configuration format

Each entry has three fields: `start` (start time), `end` (end time), and `level` (Activity Level %):

```json
{
  "activity_schedule": [
    {"start": "09:00", "end": "22:00", "level": 100},
    {"start": "22:00", "end": "06:00", "level": 30}
  ]
}
```

- Times use `HH:MM` format (24-hour notation)
- **Cross-midnight support**: specifications where start > end, such as `"22:00"` to `"06:00"`, are possible (covers late-night hours)
- `level` must be in the range 10 to 400
- Up to 24 entries
- An empty array `[]` disables schedule mode (returns to a fixed Activity Level)

#### Configuration Method

- **Settings UI**: Night mode checkbox + time range and level settings
- **API**: Send the above JSON to `PUT /api/settings/activity-schedule`
- **Root configuration API**: Use the Settings UI, or send the above JSON to `PUT /api/settings/activity-schedule`. Do not edit `config.json` directly from the Anima process.

#### Notes

- If you manually change the Activity Level, the schedule entry corresponding to the current time period is also updated in conjunction
- The schedule is applied immediately at server startup (the level is set according to the time at startup)
- If no time period matches, the last set Activity Level is maintained

## What are Cron tasks

Cron tasks are "tasks that run automatically at scheduled times." While heartbeats are "periodic patrols," Cron tasks are "scheduled operations."

Examples:
- Create a work plan every morning at 9:00
- Do a weekly review every Friday at 17:00
- Run a backup script every day at 2:00

## Configuration of cron.md

Cron tasks are defined in `cron.md` in Markdown + YAML format.
File path: `~/.animaworks/animas/{name}/cron.md`

When a supervisor edits the `cron.md` of a subordinate Anima, use `read_memory_file` / `write_memory_file` rather than direct file operations, and specify with a relative path such as `../{anima_name}/cron.md`.

### Basic format

Each task starts with a heading of `## タスク名`, and the standard 5-field cron expression is written at the beginning of the body with the `schedule:` directive.

```markdown
# Cron: {name}

## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_state.md に書き出す。

## 週次振り返り
schedule: 0 17 * * 5
type: llm
今週のepisodes/を読み返し、パターンを抽出してknowledge/に統合する。
```

The old format (a format that writes the schedule in parentheses, such as `## タスク名（毎日 9:00 JST）`) is automatically converted to the new format at server startup or with `animaworks migrate`.

### CronTask schema

Each task is internally parsed into the following `CronTask` model:

| Field | Type | Default | Description |
|-----------|------|-----------|------|
| `name` | str | (required) | Task name. Extracted from the `##` heading |
| `schedule` | str | (required) | Standard 5-field cron expression. Extracted from the `schedule:` directive |
| `type` | str | `"llm"` | Task type: `"llm"` or `"command"` |
| `description` | str | `""` | Instruction text for LLM type (used with type: llm) |
| `command` | str \| None | `None` | Bash command for Command type |
| `tool` | str \| None | `None` | Internal tool name for Command type |
| `args` | dict \| None | `None` | Tool arguments (YAML format) |
| `skip_pattern` | str \| None | `None` | Command type: if stdout matches this regular expression, skip the follow-up LLM |
| `trigger_heartbeat` | bool | `True` | Command type: if `False`, skip the follow-up cron LLM after command output |

## LLM-type Cron tasks

`type: llm` are tasks executed by an agent (LLM) with judgment and reasoning.
The instructions written in the description are passed to the agent as a prompt.

### Features

- The agent uses tools, searches memory, and makes judgments
- Results are unstructured (different output for each task)
- Execution requires a model API call (cost is incurred)

### Example description

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
昨日の episodes/ を読み返し、今日のタスクを計画する。
優先順位は理念と目標に照らして判断する。
結果は state/current_state.md に書き出す。
正本タスク一覧（`list_tasks`） の未着手タスクも確認し、必要なら優先度を見直す。
```

The description (the body after the `type:` line) should (SHOULD) include:
- What to check (input)
- How to judge (criteria)
- What to output (deliverable)

## Command-type Cron tasks

`type: command` are tasks that execute fixed commands or tools without agent judgment.
Suitable for deterministic processing (backup, notification sending, etc.).

### Bash command type

```markdown
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

Write the bash command on a single line in `command:`.
The command is executed via the shell.

### Internal tool type

```markdown
## Slack朝の挨拶
schedule: 0 9 * * 1-5
type: command
tool: slack_send
args:
  channel: "#general"
  message: "おはようございます！本日もよろしくお願いします。"
```

Write the internal tool name in `tool:` and the arguments in YAML format in `args:`.
args is parsed as a YAML indentation block (2-space indent).

### Follow-up control for Command type

For Command-type tasks, if the command completes successfully and there is stdout, the output is passed to the LLM for follow-up analysis (executed with a heartbeat-equivalent context).

- **`trigger_heartbeat: false`** — skip the follow-up LLM (when output analysis is not needed)
- **`skip_pattern: <正規表現>`** — skip the follow-up if stdout matches this regular expression

```markdown
## ログ取得（出力分析不要）
schedule: 0 8 * * *
type: command
trigger_heartbeat: false
command: /usr/local/bin/fetch-logs.sh

## 監視チェック（"OK" のときは分析不要）
schedule: */15 * * * *
type: command
skip_pattern: ^OK$
command: /usr/local/bin/health-check.sh
```

### Choosing between LLM type and Command type

| Perspective | LLM type | Command type |
|------|--------|-----------|
| Judgment needed | Yes | No |
| API cost | Yes | No |
| Output predictability | Unstructured | Deterministic |
| Suitable tasks | Planning, reflection, writing | Backup, notification sending, data retrieval |
| Error handling | Agent handles autonomously | Logged only |

Guidelines when in doubt:
- "Just doing the same thing every time" → Command type (SHOULD)
- "Judgment changes depending on the situation" → LLM type (SHOULD)
- "Command execution + interpretation of results" → LLM type with command execution instructed in the description

## Schedule notation

Write a **standard 5-field cron expression** in the `schedule:` directive of cron.md.

### Standard cron expression (required)

```
分 時 日 月 曜日
```

Examples:
- `0 9 * * *` — every day at 9:00
- `0 9 * * 1-5` — weekdays at 9:00
- `*/30 9-17 * * *` — every 30 minutes from 9:00 to 17:00
- `0 2 1 * *` — 1st of every month at 2:00
- `0 17 * * 5` — every Friday at 17:00

The time zone can be set with `system.timezone` of `config.json`. If not set, the system time zone is automatically detected.

### Migration from Japanese schedules

cron.md written in the old format (`## タスク名（毎日 9:00 JST）`) is automatically converted to a standard cron expression at server startup or with `animaworks migrate`. Conversion table:

| Japanese notation | Example cron expression |
|-----------|----------|
| `毎日 HH:MM` | `0 9 * * *` |
| `平日 HH:MM` | `0 9 * * 1-5` |
| `毎週{曜日} HH:MM` | `0 17 * * 5` (Friday) |
| `毎月N日 HH:MM` | `0 9 1 * *` |
| `X分毎` | `*/5 * * * *` |
| `X時間毎` | `0 */2 * * *` |

`隔週`, `毎月最終日`, and `第N曜日` cannot be automatically converted. Write the cron expression manually.

## How to check cron_logs

The execution results of Cron tasks are recorded in the server log.
They are also broadcast as `anima.cron` events via WebSocket.

How to check logs:
- Server log: INFO level of the `animaworks.lifecycle` logger
- Web UI: displayed in the dashboard activity feed
- For episodes/: LLM-type tasks, the agent itself writes logs to episodes/ (SHOULD)

The results of LLM-type tasks are recorded as `CycleResult` and include the following information:
- `trigger`: `"cron"`
- `action`: summary of agent actions
- `summary`: summary text of the result
- `duration_ms`: execution time (milliseconds)
- `context_usage_ratio`: context usage rate

## Common Cron configuration examples

### Basic set (recommended for all Anima)

```markdown
# Cron: {name}

## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
episodes/ から昨日の行動を確認し、正本タスク一覧（`list_tasks`） の未着手タスクを見直す。
今日の優先タスクを決め、state/current_state.md を更新する。

## 週次振り返り
schedule: 0 17 * * 5
type: llm
今週の episodes/ を読み返し、パターンや教訓を抽出する。
重要な知見は knowledge/ に書き出す。
繰り返し行った作業があれば procedures/ に手順化を検討する。
```

### External integration tasks

```markdown
## Slack日報送信
schedule: 0 18 * * 1-5
type: command
tool: slack_send
args:
  channel: "#daily-report"
  message: "本日の業務完了しました。詳細は明日の朝礼で共有します。"

## GitHub Issue 確認
schedule: 0 10 * * 1-5
type: llm
担当リポジトリの新しい Issue と PR を確認する。
重要なものがあれば supervisor に報告する。
```

### Memory maintenance

```markdown
## 知識の棚卸し
schedule: 0 10 1 * *
type: llm
knowledge/ の全ファイルを確認し、古い情報や矛盾する記載を整理する。
重要度の低い知識はアーカイブを検討する。

## 手順書の更新確認
schedule: 0 10 * * 1
type: llm
procedures/ の手順書を確認し、実際の運用と乖離がないか見直す。
変更があれば手順書を更新する。
```

### Comments

Wrap tasks you do not want to run in HTML comments:

```markdown
<!--
## 一時停止中のタスク
schedule: 0 15 * * *
type: llm
このタスクは一時的に停止中。
-->
```

The `## ` heading inside the comment is ignored by the parser.

## Hot Reload of Cron Configuration

When cron.md is updated, the schedule is automatically reloaded, just like HEARTBEAT.md.
If Anima itself rewrites cron.md, the change is reflected immediately (self-modify pattern).
Supervisors can edit the `cron.md` / `HEARTBEAT.md` of their subordinates using the write memory tool. Change requests for `AGENTS.md` from authorized supervisors are also forwarded to root and authorized. For changes to `status.json`, use the corresponding supervisor tool or root CLI/API. Do not write directly to root-owned files via Read / Write / Edit / apply_patch / `Path.write_text` / shell redirection, etc.

Behavior on reload:
1. Delete all existing cron jobs for the relevant Anima
2. Parse the updated cron.md and register new jobs
3. `Schedule reloaded for '{name}'` is output to the log

Notes when updating cron.md yourself:
- Place the `schedule:` directive immediately after the heading (`## タスク名`) (MUST)
- Write the schedule as a standard 5-field cron expression (MUST)
- Place the type line immediately after the schedule (SHOULD)
