# 定期执行的配置与运维

心跳（定期巡回）与 Cron（定时任务）的配置方法与运维指南。
当需要更改定期执行的行为，或添加新的定时任务时，请参照本文。

## 什么是心跳

心跳是 Digital Anima 定期自动启动、确认状况并进行计划的机制。
它将人类定期检查收件箱、审视进行中工作的行为自动化。

### 重要：心跳仅用于“确认与计划”

Heartbeat 确认有意义的变化，并判断必要的下一步应对。不需要仪式性的回顾报告。

- MUST: Heartbeat 聚焦于状况确认与判断
- MUST NOT: 不在心跳内执行长时间运行任务（编码、大量工具调用等）
- MUST: 发现需要执行的任务时，若有部下则通过 `delegate_task` 委派，或通过 `submit_tasks` 投入任务

已投入的任务由 **TaskExec 路径**独立于定期 Heartbeat 执行。
TaskExec 从正本 TaskStore 获取可执行的持久任务。唤醒通知与恢复由宿主管理，不监控旧的 LLM JSON 文件。

### 心跳与对话的并行运行

心跳与人类对话由 **不同的锁** 管理，因此可以同时运行。
即使心跳正在执行，也能立即响应来自人类的消息。

### 通过 submit_tasks 投入任务

发现心跳中应执行的任务时，使用 `submit_tasks` 工具投入任务：

```
submit_tasks(batch_id="hb-20260301-api-test", tasks=[
  {"task_id": "api-test", "title": "APIテスト実施",
   "description": "Slack API接続テストを実施し、全エンドポイントの結果をレポートにまとめる。完了後 aoi に報告する。"}
])
```

`submit_tasks` 在验证后，将任务与完整的执行输入一并保存到一个正本 TaskStore 中。
执行权的获取与尝试历史由宿主管理。`in_progress` 用于查看，智能体通过 `update_task` 声明 `done` / `pending` / `cancelled`。中断的 pending 任务应对同一 ID 指定 `resume: true` 显式恢复，不要用其他任务替换。

**长时间 CLI 工具**（`animaworks-tool submit …`）作为 `task_type="command"` 注册到 TaskStore，由 PendingTaskExecutor 获取尝试并在后台执行。完成结果通过 `state/background_tasks/{task_id}.json` 与通知确认。详情请参照 `operations/background-tasks.md`。

**注意**：不要直接编辑保存目标。旧的 `state/task_queue.jsonl` 与 `state/pending/` 作为迁移/导出用的证据保存，不要作为运行中的投入目标。

即使是单个任务也使用 `submit_tasks`（tasks 数组 1 条）。
多个独立任务通过 `parallel: true` 并行执行，存在依赖关系时指定 `depends_on`。
详情请参照 task-management。

### 心跳的触发类型

心跳有两种触发类型：

| 触发器 | 说明 |
|---------|------|
| 定期心跳 | 按照 `config.json` 的 `heartbeat.interval_minutes`，由 APScheduler 定期启动 |
| 消息触发 | 当 Inbox 中有未读消息到达时立即启动（作为 Inbox 路径处理） |

消息触发通过 Inbox 的 JSON 文件变更通知启动。为防备通知遗漏，每 45 秒重新确认未读。
同时运行的 Inbox 处理只有 1 个，执行中到达的消息会在下一次统一处理。Provider 错误时等待 `rate_guard` 的恢复时间，保留未读消息。
没有基于接收 intent 的启动过滤，也没有以消息为起点的 cooldown / cascade 抑制。仅对了解、感谢类消息不回复的行为规则在 Inbox 提示词中指示。

## HEARTBEAT.md 的配置

`HEARTBEAT.md` 是各 Anima 的配置文件，定义活动时间与检查项。
心跳的执行间隔可通过 `config.json` 的 `heartbeat.interval_minutes` 配置（1〜1440 分钟，默认 30）。在 `HEARTBEAT.md` 中无法更改。
每个 Anima 会被赋予基于名称的 0〜9 分钟偏移，以分散同时启动。
文件路径：`~/.animaworks/animas/{name}/HEARTBEAT.md`

上司编辑部下 Anima 的 `HEARTBEAT.md` 时，不要直接操作文件，而应使用 `read_memory_file` / `write_memory_file`，并以 `../{anima_name}/HEARTBEAT.md` 这样的相对路径指定。

### 格式

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

### 配置字段

**执行间隔**：
- 通过 `config.json` 的 `heartbeat.interval_minutes` 配置（1〜1440 分钟，默认 30）。无法在 `HEARTBEAT.md` 中更改

**活动时间**（SHOULD）：
- 以 `HH:MM - HH:MM` 格式记载（例：`9:00 - 22:00`）
- 在此时间之外心跳不会启动
- 未设置时的默认值：24 小时（全时段）
- 时区可通过 `config.json` 的 `system.timezone` 配置。未设置时自动检测系统时区

**检查清单**（MUST）：
- 心跳启动时智能体确认的项目
- 以项目列表（以 `- ` 开头）记载
- 检查清单内容会原样传递给智能体的提示词
- 可自定义：可根据 Anima 的角色添加、更改项目

### 检查清单的自定义示例

默认（所有 Anima 通用）：
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

开发负责人的示例：
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- 進行中タスクにブロッカーが発生していないか
- 監視対象のGitHubリポジトリに新しいIssueやPRがないか
- CI/CDの失敗アラートがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

沟通负责人的示例：
```markdown
## チェックリスト
- Inboxに未読メッセージがあるか
- Slackの未読メンションがないか
- 返信待ちのメールがないか
- 進行中タスクにブロッカーがないか
- 何もなければ何もしない（HEARTBEAT_OK）
```

### 执行模型（成本优化）

当配置了 `background_model` 时，Heartbeat / Inbox / Cron 会使用该模型而非主模型执行。
Chat（与人类的对话）与 TaskExec（实际作业）维持主模型。

配置方法：`animaworks anima set-background-model {名前} claude-sonnet-5-5`
详情请参照 `reference/operations/model-guide.md` 的“后台模型”章节。

### 心跳的内部行为

- **崩溃恢复**：上次心跳失败时，错误信息会保存到 `state/recovery_note.md`。下次启动时会注入提示词，恢复后文件被删除。
- **回顾记录**：心跳输出中若有 `[REFLECTION]...[/REFLECTION]` 块，会作为 `heartbeat_reflection` 记录到 activity_log，并包含在后续的心跳上下文中。
- **部下检查**：拥有部下的 Anima，其心跳、Cron 提示词中会自动注入部下状态确认指令。
- **会话时间限制**（`config.json` 的 `heartbeat`）：经过 `soft_timeout_seconds`（默认 300 秒）后注入用于收尾的提醒，在 `hard_timeout_seconds`（默认 600 秒）强制结束会话。
- **空闲时自动压缩**：`heartbeat.idle_compaction_minutes`（默认 10 分钟）— 流结束后经过此时间会触发空闲自动压缩（执行引擎侧的配置）。
- **Board 投稿**：同一 run 中对同一频道最多 1 次。跨 run 的投稿间隔没有限制。

### 定期心跳的调度方式

当有效间隔（应用 Activity Level 后、下限 5 分钟取整）为 **60 分钟以下且能整除 60** 时，会以 APScheduler 的 `CronTrigger` 分散注册到分钟槽（与基于名称的 0〜9 分钟偏移组合）。

其他情况（例：有效 61 分钟，或 43 分钟这样不能整除 60 的间隔）则通过 **每分钟轮询**（`_heartbeat_check`）按“距上次经过的时间”触发。这是为规避旧 `IntervalTrigger` 遗留缺陷的实现。

### 后台工具与 DM 日志（core/tasks/background.py）

`core/tasks/background.py` 并非心跳/Cron 的调度本身，而是负责 **长时间工具调用的后台执行**（包括状态的 JSON 持久化）与 **旧版共享 DM 日志（`shared/dm_logs/`）的轮转**。运维详情、CLI 路径也请参照 `operations/background-tasks.md`。

#### BackgroundTaskManager

- **保存目标**：`state/background_tasks/{task_id}.json`。`task_id` 为 UUID 的前 12 个字符（十六进制）。每个文件记录 `task_id`, `anima_name`, `tool_name`, `tool_args`, `status`, `created_at`, `completed_at`, `result`, `error`。
- **状态 (`TaskStatus`)**：`pending` / `running` / `completed` / `failed`。在 `submit` / `submit_async` 中，投入后立即以 `running` 写入 JSON，完成或异常时更新为 `completed` / `failed`。
- **执行 API**：`submit(tool_name, tool_args, execute_fn)` 将同步 callable 通过 `asyncio` 的 `run_in_executor` 在线程池执行。`submit_async` 直接 await 异步 callable。提供 `get_task`（内存优先，无则磁盘）、`list_tasks`（也合并磁盘上的 JSON，按创建时间从新到旧）、`active_count`（内存上的 `running` 条数）。
- **完成回调**：传给 `on_complete` 的异步函数在任务保存后被调用。即使回调内出现异常，任务结果也会保留，仅记录到日志（典型情况下与写入 `state/background_notifications/` 组合，在**下次心跳**中读取、删除并纳入对话上下文的路径）。
- **候选判定 `is_eligible(tool_name)`**：若映射中存在键则为后台对象。键接受以下两者。(1) **schema 名**（例：`generate_3d_model`）— Mode A 的外部工具分发等。(2) **`ツール名:サブコマンド`**（例：`image_gen:pipeline`）— 各工具模块的 `EXECUTION_PROFILE` 中 `background_eligible: true` 的条目由 `get_eligible_tools_from_profiles()` 以此格式注册（Mode S 的 `submit` 路径等）。
- **候选工具映射的构建 `from_profiles()`**：将以下 3 层以 dict 的 `update` 合并，**后者覆盖前者**。(1) `_DEFAULT_ELIGIBLE_TOOLS`（代码默认）(2) 参数 `profiles`（`EXECUTION_PROFILE` 聚合）(3) 参数 `config_eligible`（通常为从 `config.json` 的 `background_task.eligible_tools` 展开 `threshold_s` 得到的 `名前 → 秒`）。值为用于 profile 联动的预期秒数（整数）。
- **代码默认 `_DEFAULT_ELIGIBLE_TOOLS`（秒）**：`generate_character_assets` 30、`generate_fullbody` / `generate_bustup` / `generate_icon` / `generate_chibi` 各 30、`generate_3d_model` / `generate_rigged_model` / `generate_animations` 各 30、`local_llm` 60、`run_command` 60。
- **清理 `cleanup_old_tasks(max_age_hours=24)`**：`status` 在 `completed` / `failed` 中删除 `completed_at` 为 **早于参数指定时间（默认 24 小时）** 的 JSON。此外，仍为 `running` 且从 `created_at` 起经过 **超过 48 小时** 的文件作为崩溃孤儿删除。返回值为删除件数。
- **保留时间**：`cleanup_old_tasks` 的完成任务保留时间由参数 `max_age_hours`（默认 24 小时）指定。没有对应的 `config.json` 配置键。

#### rotate_dm_logs（系统 Cron）

- **执行时机**：在生命周期（`core/lifecycle/system_crons.py` 等）的系统 Cron 中 **每天 04:30**（服务器配置时区）。`server/supervisor/_mgr_scheduler.py` 侧也会注册同一 ID 的作业。
- **对象**：`shared/dm_logs/*.jsonl`（文件名中包含 `.archive.` 的跳过）。
- **动作**：以 `core.time_utils` 的本地当前时间为基准，解析每行 JSON 的 `ts`（ISO 格式），将早于 **默认 7 天** 的条目追加归档到 `{stem}.{YYYYMMDD}.archive.jsonl`，然后从当前文件中移除。`ts` 解析失败的行 **保留在当前文件中**（防止数据丢失）。
- **其他服务器定时作业**：除按 Anima 的 `cron.md` 之外，生命周期还会注册内存维护、RAG 等系统 Cron（例：每日整合 02:00、每日索引 04:00）。时刻以配置时区为基准。DM 轮转为上述 04:30。

### 心跳配置的热重载

在文件系统上更新 HEARTBEAT.md 后，下次心跳执行时 `_check_schedule_freshness()` 会检测到变更，SchedulerManager 会自动重载调度。
无需重启服务器（MAY skip restart）。APScheduler 的作业会被重新注册。

## 每个 Anima 的 Heartbeat 间隔配置

### 在 status.json 中的配置

root 管理员可以在 root CLI/API 中为各个 Anima 单独配置 `heartbeat_interval_minutes`。值保存在 root 所有的 `status.json` 中，Anima 进程不会编辑它。

```json
{
  "heartbeat_interval_minutes": 60
}
```

- 可配置范围：1〜1440 分钟（1 天）
- 未配置时：回退到 `config.json` 的 `heartbeat.interval_minutes`（默认 30 分钟）
- `status.json` 归 root 所有。管理员可以在服务器停止期间通过 `animaworks config set animas.<name>.heartbeat_interval_minutes <minutes>` 设置单独的覆盖值。服务器启动期间，CLI 会通过 root API 反映。Anima 进程不会编辑此文件。

### 推荐指南

| 状况 | 推荐间隔 | 理由 |
|------|----------|------|
| 活跃的开发项目进行中 | 15〜30 分钟 | 需要频繁掌握状况 |
| 常规业务 | 30〜60 分钟 | 默认值。频率均衡 |
| 低负载・待机状态 | 60〜120 分钟 | 节约成本。没有任务时设长一些 |
| 长期休眠・非活跃 | 120〜1440 分钟 | 以最小限度的巡回掌握状况 |

### 与 Activity Level 的关系

当设置了全局 Activity Level（10%〜400%）时，实际间隔按以下公式计算：

```
実効間隔 = ベース間隔 / (Activity Level / 100)
```

例：基准 30 分钟、Activity Level 50% → 实际 60 分钟
例：基准 30 分钟、Activity Level 200% → 实际 15 分钟

- 实际间隔的下限为 5 分钟（无论怎么提升都不会低于 5 分钟）

### Activity Schedule（按时间段自动切换 / 夜间模式）

根据时间段自动切换 Activity Level 的机制。
用于希望在夜间或休息日降低成本，或希望仅在业务时间段内活跃运行的情况。

#### 机制

- 在 `config.json` 的 `activity_schedule` 中设置时间段条目
- 每分钟检查当前时刻，自动将 Activity Level 更改为对应时间段的级别
- Activity Level 变化后，所有 Anima 的心跳会立即重新调度

#### 配置格式

每个条目包含 `start`（开始时刻）、`end`（结束时刻）、`level`（Activity Level %）三个字段：

```json
{
  "activity_schedule": [
    {"start": "09:00", "end": "22:00", "level": 100},
    {"start": "22:00", "end": "06:00", "level": 30}
  ]
}
```

- 时刻为 `HH:MM` 格式（24 小时制）
- **支持跨日期**：可以指定 `"22:00"` 〜 `"06:00"` 这样 start > end 的情况（覆盖深夜时段）
- `level` 的范围为 10〜400
- 最多 24 个条目
- 用空数组 `[]` 禁用调度模式（恢复为固定 Activity Level）

#### 配置方法

- **Settings UI**：夜间模式复选框 + 时间段・级别设置
- **API**：向 `PUT /api/settings/activity-schedule` 发送上述 JSON
- **root 配置 API**：使用 Settings UI，或向 `PUT /api/settings/activity-schedule` 发送上述 JSON。Anima 进程不要直接编辑 `config.json`

#### 注意事项

- 手动更改 Activity Level 时，当前时间段对应的调度条目也会联动更新
- 调度在服务器启动时也会立即应用（按启动时点的时刻设置为对应级别）
- 当不属于任何时间段时，维持最后设置的 Activity Level

## 什么是 Cron 任务

Cron 是“在指定时间自动执行的任务”。如果说心跳是“定期巡回”，那么 Cron 就是“定时业务”。

例：
- 每天早上 9:00 制定业务计划
- 每周五 17:00 进行周度回顾
- 每天 2:00 执行备份脚本

## cron.md 的配置

Cron 任务以 Markdown + YAML 格式定义在 `cron.md` 中。
文件路径：`~/.animaworks/animas/{name}/cron.md`

当上司编辑下属 Anima 的 `cron.md` 时，不要直接操作文件，而应使用 `read_memory_file` / `write_memory_file`，并以 `../{anima_name}/cron.md` 这样的相对路径指定。

### 基本格式

每个任务以 `## タスク名` 标题开始，在正文开头用 `schedule:` 指令记载标准 5 字段 cron 表达式。

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

旧格式（如 `## タスク名（毎日 9:00 JST）` 那样在括号内写调度的方式）会在服务器启动时或通过 `animaworks migrate` 自动转换为新格式。

### CronTask 的 Schema

每个任务在内部会被解析为以下 `CronTask` 模型：

| 字段 | 类型 | 默认值 | 说明 |
|-----------|------|-----------|------|
| `name` | str | （必填） | 任务名。从 `##` 标题中提取 |
| `schedule` | str | （必填） | 标准 5 字段 cron 表达式。从 `schedule:` 指令中提取 |
| `type` | str | `"llm"` | 任务类型：`"llm"` 或 `"command"` |
| `description` | str | `""` | LLM 型的指示文（在 type: llm 时使用） |
| `command` | str \| None | `None` | Command 型的 bash 命令 |
| `tool` | str \| None | `None` | Command 型的内部工具名 |
| `args` | dict \| None | `None` | tool 的参数（YAML 格式） |
| `skip_pattern` | str \| None | `None` | Command 型：若 stdout 匹配此正则表达式则跳过 follow-up LLM |
| `trigger_heartbeat` | bool | `True` | Command 型：若为 `False` 则跳过命令输出后的 follow-up cron LLM |

## LLM 型 Cron 任务

`type: llm` 是由智能体（LLM）伴随判断・思考执行的任务。
description 中写的指示会作为提示词传递给智能体。

### 特点

- 智能体使用工具、搜索记忆、做出判断
- 结果不定形（每个任务输出不同）
- 执行需要模型的 API 调用（产生成本）

### 记述示例

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
昨日の episodes/ を読み返し、今日のタスクを計画する。
優先順位は理念と目標に照らして判断する。
結果は state/current_state.md に書き出す。
正本タスク一覧（`list_tasks`） の未着手タスクも確認し、必要なら優先度を見直す。
```

description（`type:` 行之后的正文）应（SHOULD）包含以下内容：
- 确认什么（输入）
- 如何判断（标准）
- 输出什么（成果物）

## Command 型 Cron 任务

`type: command` 是不经过智能体判断、执行固定命令或工具的任务。
适合确定性的处理（备份、发送通知等）。

### bash 命令型

```markdown
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

在 `command:` 中以一行记载 bash 命令。
命令通过 shell 执行。

### 内部工具型

```markdown
## Slack朝の挨拶
schedule: 0 9 * * 1-5
type: command
tool: slack_send
args:
  channel: "#general"
  message: "おはようございます！本日もよろしくお願いします。"
```

在 `tool:` 中记载内部工具名，在 `args:` 中以 YAML 格式记载参数。
args 作为 YAML 的缩进块被解析（2 空格缩进）。

### Command 型的 follow-up 控制

Command 型任务在命令正常结束且有 stdout 时，会将其输出传递给 LLM 进行 follow-up 分析（在相当于 heartbeat 的上下文中执行）。

- **`trigger_heartbeat: false`** — 跳过 follow-up LLM（当不需要分析输出时）
- **`skip_pattern: <正規表現>`** — 若 stdout 匹配此正则表达式则跳过 follow-up

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

### LLM 型与 Command 型的使用区分

| 观点 | LLM 型 | Command 型 |
|------|--------|-----------|
| 是否需要判断 | 是 | 否 |
| API 成本 | 有 | 无 |
| 输出的可预测性 | 不定形 | 确定性 |
| 适合的任务 | 计划制定、回顾、文章撰写 | 备份、发送通知、数据获取 |
| 出错时的应对 | 智能体自主处理 | 仅记录到日志 |

犹豫时的指南：
- “只是每次做同样的事”→ Command 型（SHOULD）
- “根据状况判断会变化”→ LLM 型（SHOULD）
- “命令执行 + 结果解读”→ LLM 型，在 description 中指示执行命令

## 调度记法

cron.md 的 `schedule:` 指令中记载 **标准 5 字段 cron 表达式**。

### 标准 cron 表达式（必填）

```
分 時 日 月 曜日
```

例：
- `0 9 * * *` — 每天 9:00
- `0 9 * * 1-5` — 工作日 9:00
- `*/30 9-17 * * *` — 9:00〜17:00 每 30 分钟
- `0 2 1 * *` — 每月 1 日 2:00
- `0 17 * * 5` — 每周五 17:00

时区可在 `config.json` 的 `system.timezone` 中配置。未配置时自动检测系统时区。

### 从日语调度迁移

以旧格式（`## タスク名（毎日 9:00 JST）`）书写的 cron.md 会在服务器启动时或通过 `animaworks migrate` 自动转换为标准 cron 表达式。转换对应表：

| 日语记法 | cron 表达式示例 |
|-----------|----------|
| `毎日 HH:MM` | `0 9 * * *` |
| `平日 HH:MM` | `0 9 * * 1-5` |
| `毎週{曜日} HH:MM` | `0 17 * * 5`（周五） |
| `毎月N日 HH:MM` | `0 9 1 * *` |
| `X分毎` | `*/5 * * * *` |
| `X時間毎` | `0 */2 * * *` |

`隔週`、`毎月最終日`、`第N曜日` 无法自动转换。请手动记载 cron 表达式。

## cron_logs 的确认方法

Cron 任务的执行结果会记录到服务器日志中。
也会通过 WebSocket 作为 `anima.cron` 事件广播。

日志的确认方法：
- 服务器日志：`animaworks.lifecycle` logger 的 INFO 级别
- Web UI：显示在仪表盘的活动 feed 中
- episodes/: LLM 型任务的情况下，智能体自身会写入 episodes/ 日志（SHOULD）

LLM 型任务的结果会作为 `CycleResult` 记录，包含以下信息：
- `trigger`：`"cron"`
- `action`：智能体的行动摘要
- `summary`：结果的摘要文本
- `duration_ms`：执行时间（毫秒）
- `context_usage_ratio`：上下文使用率

## 常见的 Cron 配置示例

### 基本套装（推荐所有 Anima）

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

### 外部集成任务

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

### 记忆维护

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

### 注释掉

不想执行的任务用 HTML 注释包围：

```markdown
<!--
## 一時停止中のタスク
schedule: 0 15 * * *
type: llm
このタスクは一時的に停止中。
-->
```

注释内的 `## ` 标题会被解析器忽略。

## Cron 配置的热重载

更新 cron.md 后，与 HEARTBEAT.md 一样，调度会自动重载。
即使 Anima 自身改写了 cron.md，也会立即反映（self-modify 模式）。
上司可以用 write memory 工具编辑下属的 `cron.md` / `HEARTBEAT.md`。来自被授权上司的 `AGENTS.md` 变更请求也会被转发到 root 并获得授权。更改 `status.json` 时使用对应的面向上司的工具或 root CLI/API。不要通过 Read / Write / Edit / apply_patch / `Path.write_text` / shell 重定向等直接写入 root 所有的文件。

重载时的行为：
1. 删除该 Anima 的所有现有 cron 任务
2. 解析更新后的 cron.md，注册新任务
3. 日志中输出 `Schedule reloaded for '{name}'`

自行更新 cron.md 时的注意事项：
- 在标题（`## タスク名`）之后紧接着放置 `schedule:` 指令（MUST）
- 调度以标准 5 字段 cron 表达式记载（MUST）
- type 行放在 schedule 之后（SHOULD）
