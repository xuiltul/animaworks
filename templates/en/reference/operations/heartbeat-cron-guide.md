# Configuring and Operating Scheduled Runs

A guide to configuring and operating Heartbeat (periodic checks) and Cron (scheduled tasks).
Refer to this when you want to change scheduled-run behavior or add a new scheduled task.

## What Is Heartbeat?

Heartbeat is a mechanism that automatically starts Digital Anima at regular intervals to check the situation and make plans.
It automates the same behavior as a person periodically checking their inbox and reviewing work in progress.

### Important: Heartbeat Is for “Checking and Planning” Only

Heartbeat checks for meaningful changes and determines what follow-up is needed. No ritualistic retrospective report is required.

- MUST: Keep Heartbeat focused on checking the situation and making decisions.
- MUST NOT: Perform long-running tasks (such as coding or making a large number of tool calls) during Heartbeat.
- MUST: When you discover a task that needs to be performed, delegate it via `delegate_task` if you have subordinates, or submit it as a task via `submit_tasks`.

Submitted tasks are executed by the **TaskExec path**, independently of periodic Heartbeat.
TaskExec retrieves executable persistent tasks from the canonical TaskStore. The host manages wake-up notifications and recovery; legacy LLM JSON files are not monitored.

### Parallel Operation of Heartbeat and Conversations

Heartbeat and conversations with people are managed using **separate locks**, so they can run at the same time.
Messages from people can be answered immediately, even while Heartbeat is running.

### Submitting Tasks with submit_tasks

If you discover a task that should be performed during Heartbeat, submit it using the `submit_tasks` tool:

```
submit_tasks(batch_id="hb-20260301-api-test", tasks=[
  {"task_id": "api-test", "title": "APIテスト実施",
   "description": "Slack API接続テストを実施し、全エンドポイントの結果をレポートにまとめる。完了後 aoi に報告する。"}
])
```

After validation, `submit_tasks` saves the task and its complete execution input together in a single canonical TaskStore.
The host manages execution-right acquisition and attempt history. `in_progress` is for viewing; agents declare `done` / `pending` / `cancelled` using `update_task`. To explicitly resume an interrupted pending task, specify `resume: true` with the same ID; do not replace it with another task.

**Long-running CLI tools** (`animaworks-tool submit …`) are registered in TaskStore as `task_type="command"`, and PendingTaskExecutor claims and runs their attempts in the background. Check completion results via `state/background_tasks/{task_id}.json` and notifications. See `operations/background-tasks.md` for details.

**Note**: Do not edit the storage location directly. Keep legacy `state/task_queue.jsonl` and `state/pending/` as migration/export artifacts; do not use them as active submission targets.

Even for a single task, use `submit_tasks` (with one item in the tasks array).
Run multiple independent tasks in parallel with `parallel: true`; if there are dependencies, specify `depends_on`.
See task-management for details.

### Heartbeat Trigger Types

Heartbeat has two types of triggers:

| Trigger | Description |
|---------|------|
| Periodic Heartbeat | Starts periodically via APScheduler, according to `config.json`’s `heartbeat.interval_minutes` |
| Message trigger | Starts immediately when an unread message arrives in the Inbox (processed as an Inbox path) |

The message trigger starts when a change notification is issued for an Inbox JSON file. To handle missed notifications, unread messages are checked again every 45 seconds.
Only one Inbox process runs at a time; messages received while it is running are processed together in the next run. On a Provider error, it waits for the recovery interval in `rate_guard` and leaves unread messages in place.
There is no startup filter based on receive intent, nor any message-triggered cooldown or cascade suppression. The Inbox prompt instructs the agent not to reply to messages that only say “understood” or “thanks.”

## Configuring HEARTBEAT.md

`HEARTBEAT.md` is each Anima’s configuration file, defining activity hours and checklist items.
The Heartbeat interval can be configured using `heartbeat.interval_minutes` in `config.json` (1–1440 minutes; default 30). It cannot be changed in `HEARTBEAT.md`.
Each Anima is assigned a name-based offset of 0–9 minutes to spread out simultaneous startups.
File path: `~/.animaworks/animas/{name}/HEARTBEAT.md`

When a supervisor edits a subordinate Anima’s `HEARTBEAT.md`, use `read_memory_file` / `write_memory_file` rather than directly manipulating files, and specify a relative path such as `../{anima_name}/HEARTBEAT.md`.

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

**Run interval**:
- Configure using `heartbeat.interval_minutes` in `config.json` (1–1440 minutes; default 30). Changes via `HEARTBEAT.md` are not allowed.

**Activity hours** (SHOULD):
- Specify in `HH:MM - HH:MM` format (e.g., `9:00 - 22:00`).
- Heartbeat does not start outside these hours.
- Default when unset: 24 hours (all hours).
- The time zone can be configured using `system.timezone` in `config.json`. If unset, the system time zone is detected automatically.

**Checklist** (MUST):
- Items for the agent to check when Heartbeat starts.
- Write as a bullet list (each item beginning with `- `).
- The checklist content is passed directly into the agent’s prompt.
- Customizable: Add or change items to suit Anima’s role.

### Checklist Customization Examples

Default (common to all Anima):
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

Example for a development role:
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーが発生していないか
- 監視対象のGitHubリポジトリに新しいIssueやPRがないか
- CI/CDの失敗アラートがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

Example for a communications role:
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- Slackの未読メンションがないか
- 返信待ちのメールがないか
- 進行中タスクにブロッカーがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

### Execution Model (Cost Optimization)

When `background_model` is configured, Heartbeat / Inbox / Cron run using that model instead of the main model.
Chat (conversations with people) and TaskExec (actual work) continue to use the main model.

Configuration: `animaworks anima set-background-model {名前} claude-sonnet-5-5`
For details, see the “Background Model” section of `reference/operations/model-guide.md`.

### Heartbeat Internals

- **Crash recovery**: If the previous Heartbeat failed, error information is saved in `state/recovery_note.md`. It is injected into the prompt at the next startup, and the file is deleted after recovery.
- **Retrospective log**: If Heartbeat output contains a `[REFLECTION]...[/REFLECTION]` block, it is recorded in activity_log as `heartbeat_reflection` and included in the context of subsequent Heartbeats.
- **Subordinate checks**: For Anima with subordinates, instructions to check their status are automatically injected into Heartbeat and Cron prompts.
- **Session time limits** (`heartbeat` in `config.json`): After `soft_timeout_seconds` (default 300 seconds), a wrap-up reminder is injected; after `hard_timeout_seconds` (default 600 seconds), the session is forcibly ended.
- **Automatic compaction when idle**: `heartbeat.idle_compaction_minutes` (default 10 minutes) — idle auto-compaction runs once this much time has passed since the stream ended (an execution-engine setting).

### Periodic Heartbeat Scheduling Method

When the effective interval (after applying Activity Level and rounding down to a minimum of 5 minutes) is **60 minutes or less and divides evenly into 60**, it is distributed across minute slots using APScheduler’s `CronTrigger` (combined with a name-based offset of 0–9 minutes).

Otherwise (for example, an effective interval of 61 minutes, or an interval such as 43 minutes that does not divide evenly into 60), it uses **polling every minute** (`_heartbeat_check`), firing based on “time elapsed since the previous run.” This implementation avoids issues caused by the legacy `IntervalTrigger`.

### Background Tools (core/tasks/background.py）

`core/tasks/background.py` is responsible not for scheduling Heartbeat/Cron itself, but for **background execution of long-running tool calls** (including JSON persistence of status). See also `operations/background-tasks.md` for operational details and the CLI path.

#### BackgroundTaskManager

- **Storage location**: `state/background_tasks/{task_id}.json`. `task_id` is the first 12 characters of a UUID (hexadecimal). Each file records `task_id`, `anima_name`, `tool_name`, `tool_args`, `status`, `created_at`, `completed_at`, `result`, `error`.
- **Status (`TaskStatus`)**: `pending` / `running` / `completed` / `failed`. For `submit` / `submit_async`, JSON is written as `running` immediately after submission, then updated to `completed` / `failed` on completion or exception.
- **Execution API**: `submit(tool_name, tool_args, execute_fn)` runs a synchronous callable in a thread pool using `run_in_executor` from `asyncio`. `submit_async` directly awaits an asynchronous callable. It provides `get_task` (memory first, disk if unavailable), `list_tasks` (also merges JSON on disk, newest creation time first), and `active_count` (the number of `running` in memory).
- **Completion callback**: An asynchronous function passed to `on_complete` is called after the task is saved. If an exception occurs inside the callback, the task result is retained and only the exception is logged (typically combined with writing to `state/background_notifications/`, which is read and deleted on the **next Heartbeat** and incorporated into the conversation context).
- **Candidate check `is_eligible(tool_name)`**: If a key exists in the map, it is a background candidate. Both of the following key types are accepted: (1) **Schema name** (e.g., `generate_3d_model`) — such as external-tool dispatch in Mode A. (2) **`ツール名:サブコマンド`** (e.g., `image_gen:pipeline`) — an entry from `background_eligible: true` in each tool module’s `EXECUTION_PROFILE` is registered in this form via `get_eligible_tools_from_profiles()` (such as the `submit` path in Mode S).
- **Building the candidate tool map `from_profiles()`**: Merge the following three layers using dict’s `update`, with **later entries overriding earlier ones**: (1) `_DEFAULT_ELIGIBLE_TOOLS` (code defaults); (2) argument `profiles` (aggregate of `EXECUTION_PROFILE`); (3) argument `config_eligible` (usually `名前 → 秒` expanded from `threshold_s` in `background_task.eligible_tools` of `config.json`). Values are expected seconds (integers) used for profile integration.
- **Code defaults `_DEFAULT_ELIGIBLE_TOOLS` (seconds)**: `generate_character_assets` 30; `generate_fullbody` / `generate_bustup` / `generate_icon` / `generate_chibi` 30 each; `generate_3d_model` / `generate_rigged_model` / `generate_animations` 30 each; `local_llm` 60; `run_command` 60.
- **Cleanup `cleanup_old_tasks(max_age_hours=24)`**: `status` deletes JSON files where `completed_at` is **older than the time specified by the argument (default 24 hours)**, using `completed` / `failed`. In addition, files still in `running` for **more than 48 hours** from `created_at` are deleted as crash orphans. Returns the number of deleted files.
- **Retention period**: The completed-task retention period for `cleanup_old_tasks` is specified by argument `max_age_hours` (default 24 hours). There is no corresponding `config.json` configuration key.

- **Other scheduled server jobs**: Separate from Anima-level `cron.md`, the lifecycle registers system Cron jobs for memory maintenance, RAG, and similar tasks (e.g., daily consolidation at 02:00 and daily indexing at 04:00). Times are based on the configured time zone.

### Hot Reloading Heartbeat Configuration

When HEARTBEAT.md is updated on the filesystem, `_check_schedule_freshness()` detects the change during the next Heartbeat run, and SchedulerManager automatically reloads the schedule.
A server restart is not required (MAY skip restart). APScheduler jobs are re-registered.

## Per-anima Heartbeat Interval Configuration

### Configuration in status.json

The root administrator can configure an individual Anima’s `heartbeat_interval_minutes` in the root CLI/API. The value is stored in the root-owned `status.json` and is not edited by the Anima process.

```json
{
  "heartbeat_interval_minutes": 60
}
```

- Configurable range: 1–1440 minutes (one day).
- If unset: Falls back to `heartbeat.interval_minutes` in `config.json` (default 30 minutes).
- `status.json` is root-owned. Administrators can configure individual overrides in `animaworks config set animas.<name>.heartbeat_interval_minutes <minutes>` while the server is stopped. While the server is running, the CLI applies them via the root API. The Anima process does not edit this file.

### Recommended Guidelines

| Situation | Recommended interval | Reason |
|------|----------|------|
| During an active development project | 15–30 minutes | Frequent status checks are needed |
| Normal operations | 30–60 minutes | Default. A balanced frequency |
| Low activity or waiting | 60–120 minutes | Save costs. Use longer intervals when there are no tasks |
| Long-term dormant or inactive | 120–1440 minutes | Check status with minimal polling |

### Relationship to Activity Level

When a global Activity Level (10%–400%) is configured, the effective interval is calculated using the following formula:

```
実効間隔 = ベース間隔 / (Activity Level / 100)
```

Example: Base interval 30 minutes, Activity Level 50% → effective interval 60 minutes
Example: Base interval 30 minutes, Activity Level 200% → effective interval 15 minutes

- The minimum effective interval is 5 minutes (no amount of boosting can reduce it below 5 minutes)

### Activity Schedule (Automatic Time-Based Switching / Night Mode)

A mechanism for automatically switching Activity Level based on the time of day.
Use it to reduce costs at night or on holidays, or to keep activity high only during business hours.

#### How It Works

- Set time-period entries in `config.json`’s `activity_schedule`
- The current time is checked every minute, and Activity Level is automatically changed to the level for the matching time period
- When Activity Level changes, the heartbeats of all Anima are rescheduled immediately

#### Configuration Format

Each entry has three fields: `start` (start time), `end` (end time), and `level` (Activity Level %):

```json
{
  "activity_schedule": [
    {"start": "09:00", "end": "22:00", "level": 100},
    {"start": "22:00", "end": "06:00", "level": 30}
  ]
}
```

- Times use the `HH:MM` format (24-hour clock)
- **Supports crossing midnight**: start > end is allowed, such as `"22:00"` to `"06:00"` (to cover late-night hours)
- `level` must be in the range 10–400
- Up to 24 entries are allowed
- An empty array `[]` disables schedule mode (returns to a fixed Activity Level)

#### Configuration Methods

- **Settings UI**: Night Mode checkbox and time-period/level settings
- **API**: Send the JSON above to `PUT /api/settings/activity-schedule`
- **Root configuration API**: Use the Settings UI or send the JSON above to `PUT /api/settings/activity-schedule`. Do not edit `config.json` directly from an Anima process

#### Notes

- Manually changing Activity Level also updates the schedule entry for the current time period
- The schedule is applied immediately when the server starts (the level for the time at startup is set)
- If no time period matches, the last configured Activity Level is retained

## What Are Cron Tasks?

Cron tasks are tasks that “run automatically at a set time.” While a heartbeat is a “regular check-in,” Cron is “scheduled work.”

Examples:
- Make a work plan every morning at 9:00
- Conduct a weekly review every Friday at 17:00
- Run a backup script every day at 2:00

## Configuring cron.md

Cron tasks are defined in `cron.md` using Markdown + YAML format.
File path: `~/.animaworks/animas/{name}/cron.md`

When a supervisor edits a subordinate Anima’s `cron.md`, use `read_memory_file` / `write_memory_file` rather than direct file operations, and specify a relative path such as `../{anima_name}/cron.md`.

### Basic Format

Each task starts with a `## タスク名` heading, and a standard five-field cron expression is specified at the start of the body using the `schedule:` directive.

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

The old format (with the schedule written in parentheses, such as `## タスク名（毎日 9:00 JST）`) is automatically converted to the new format when the server starts or at `animaworks migrate`.

### CronTask Schema

Each task is internally parsed into the following `CronTask` model:

| Field | Type | Default | Description |
|-----------|------|-----------|------|
| `name` | str | (required) | Task name. Extracted from the `##` heading |
| `schedule` | str | (required) | Standard five-field cron expression. Extracted from the `schedule:` directive |
| `type` | str | `"llm"` | Task type: `"llm"` or `"command"` |
| `description` | str | `""` | LLM-type instruction text (used with type: llm) |
| `command` | str \| None | `None` | Command-type bash command |
| `tool` | str \| None | `None` | Command-type internal tool name |
| `args` | dict \| None | `None` | Tool arguments (YAML format) |
| `skip_pattern` | str \| None | `None` | Command type: skip the follow-up LLM if stdout matches this regular expression |
| `trigger_heartbeat` | bool | `True` | Command type: if `False`, skip the follow-up cron LLM after command output |

## LLM-Type Cron Tasks

`type: llm` are tasks performed by an agent (LLM) involving judgment and reasoning.
The instructions written in the description are passed to the agent as a prompt.

### Features

- The agent uses tools, searches memory, and makes decisions
- Results are variable (output differs for each task)
- Execution requires a model API call (incurs cost)

### Example

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
昨日の episodes/ を読み返し、今日のタスクを計画する。
優先順位は理念と目標に照らして判断する。
結果は state/current_state.md に書き出す。
正本タスク一覧（`list_tasks`） の未着手タスクも確認し、必要なら優先度を見直す。
```

The description (the body following the `type:` line) SHOULD include:
- What to check (inputs)
- How to assess it (criteria)
- What to output (deliverables)

## Command-Type Cron Tasks

`type: command` are tasks that run a fixed command or tool without requiring the agent’s judgment.
They are suited to deterministic operations (backups, sending notifications, etc.).

### Bash Command Type

```markdown
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

Write a one-line bash command in `command:`.
The command is executed through a shell.

### Internal Tool Type

```markdown
## Slack朝の挨拶
schedule: 0 9 * * 1-5
type: command
tool: slack_send
args:
  channel: "#general"
  message: "おはようございます！本日もよろしくお願いします。"
```

Specify the internal tool name in `tool:` and its arguments in YAML format in `args:`.
args is parsed as a YAML indented block (2-space indentation).

### Follow-Up Control for Command-Type Tasks

For Command-type tasks, if the command completes successfully and produces stdout, that output is passed to an LLM for follow-up analysis (run in a heartbeat-equivalent context).

- **`trigger_heartbeat: false`** — Skip the follow-up LLM (when analysis of the output is unnecessary)
- **`skip_pattern: <正規表現>`** — Skip the follow-up if stdout matches this regular expression

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

### Choosing Between LLM and Command Types

| Aspect | LLM type | Command type |
|------|--------|-----------|
| Requires judgment | Yes | No |
| API cost | Yes | No |
| Output predictability | Variable | Deterministic |
| Suitable tasks | Planning, reviews, writing | Backups, sending notifications, retrieving data |
| Handling errors | Agent handles autonomously | Recorded in log only |

Guidelines when unsure:
- “Just do the same thing every time” → Command type (SHOULD)
- “The decision changes depending on the situation” → LLM type (SHOULD)
- “Run a command + interpret the result” → LLM type, with the description instructing it to run the command

## Schedule Syntax

The `schedule:` directive in cron.md must contain a **standard five-field cron expression**.

### Standard Cron Expression (Required)

```
分 時 日 月 曜日
```

Examples:
- `0 9 * * *` — Every day at 9:00
- `0 9 * * 1-5` — Weekdays at 9:00
- `*/30 9-17 * * *` — Every 30 minutes from 9:00 to 17:00
- `0 2 1 * *` — At 2:00 on the first day of every month
- `0 17 * * 5` — Every Friday at 17:00

The time zone can be configured with `system.timezone` in `config.json`. If unset, the system time zone is detected automatically.

### Migrating from Japanese Schedules

cron.md written in the old format (`## タスク名（毎日 9:00 JST）`) are automatically converted to standard cron expressions when the server starts or at `animaworks migrate`. Conversion reference:

| Japanese notation | Example cron expression |
|-----------|----------|
| `毎日 HH:MM` | `0 9 * * *` |
| `平日 HH:MM` | `0 9 * * 1-5` |
| `毎週{曜日} HH:MM` | `0 17 * * 5` (Friday) |
| `毎月N日 HH:MM` | `0 9 1 * *` |
| `X分毎` | `*/5 * * * *` |
| `X時間毎` | `0 */2 * * *` |

`隔週`, `毎月最終日`, and `第N曜日` cannot be automatically converted. Write the cron expression manually.

## How to Check cron_logs

Cron task results are recorded in the server log.
They are also broadcast via WebSocket as `anima.cron` events.

How to check logs:
- Server log: INFO level of the `animaworks.lifecycle` logger
- Web UI: Displayed in the dashboard activity feed
- episodes/: For LLM-type tasks, the agent SHOULD write logs to episodes/

Results of LLM-type tasks are recorded as `CycleResult` and include the following information:
- `trigger`: `"cron"`
- `action`: Summary of agent actions
- `summary`: Summary of results
- `duration_ms`: Execution time (milliseconds)
- `context_usage_ratio`: Context usage

## Common Cron Configuration Examples

### Basic Set (Recommended for All Anima)

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

### External Integration Tasks

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

### Memory Maintenance

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

### Commenting Out

Wrap tasks you do not want to run in HTML comments:

```markdown
<!--
## 一時停止中のタスク
schedule: 0 15 * * *
type: llm
このタスクは一時的に停止中。
-->
```

`## ` headings inside comments are ignored by the parser.

## Hot Reloading Cron Configuration

Updating cron.md automatically reloads the schedule, just as with HEARTBEAT.md.
Changes are applied immediately even when Anima itself modifies cron.md (self-modify pattern).
A supervisor can edit a subordinate’s `cron.md` / `HEARTBEAT.md` using the write memory tool. Requests from an authorized supervisor to change `AGENTS.md` are also forwarded to root for authorization. Use the corresponding supervisor-facing tool or root CLI/API to change `status.json`. Do not write directly to root-owned files using Read / Write / Edit / apply_patch / `Path.write_text` / shell redirection, etc.

What happens on reload:
1. Delete all existing cron jobs for the relevant Anima
2. Parse the updated cron.md and register new jobs
3. `Schedule reloaded for '{name}'` is output to the log

Notes when updating your own cron.md:
- Place the `schedule:` directive immediately after the heading (`## タスク名`) (MUST)
- Specify the schedule using a standard five-field cron expression (MUST)
- Place the type line immediately after the schedule (SHOULD)
