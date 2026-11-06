# 定期运行的配置与运维

心跳（定期巡检）和 Cron（定时任务）的配置方法与运维指南。
需要更改定期运行行为或添加新的定时任务时，请参阅本文。

## 什么是心跳

心跳是 Digital Anima 定期自动启动并检查状况、制定计划的机制。
它将人类定期查看收件箱、回顾进行中的工作的行为自动化。

### 重要：心跳仅用于“检查和计划”

Heartbeat 用于检查有意义的变化，并判断接下来需要采取哪些措施。不需要进行形式化的回顾汇报。

- MUST：Heartbeat 仅限于检查状况和作出判断
- MUST NOT：不要在心跳中执行耗时较长的任务（编写代码、大量调用工具等）
- MUST：发现需要执行的任务时，如果有部下，则通过 `delegate_task` 委派；或者通过 `submit_tasks` 投入任务

已投入的任务由 **TaskExec 路径**独立于定期 Heartbeat 执行。
TaskExec 从权威 TaskStore 获取可执行的持久化任务。唤醒通知和恢复由主机管理，不会监控旧版 LLM JSON 文件。

### 心跳与对话并行运行

心跳和与人类的对话由**不同的锁**管理，因此可以同时运行。
即使正在执行心跳，也可以立即响应人类发送的消息。

### 通过 submit_tasks 投入任务

如果在心跳中发现需要执行的任务，请使用 `submit_tasks` 工具投入任务：

```
submit_tasks(batch_id="hb-20260301-api-test", tasks=[
  {"task_id": "api-test", "title": "APIテスト実施",
   "description": "Slack API接続テストを実施し、全エンドポイントの結果をレポートにまとめる。完了後 aoi に報告する。"}
])
```

`submit_tasks` 验证后，会将任务和完整的执行输入一并保存到同一个权威 TaskStore 中。
执行权的获取和尝试历史由主机管理。`in_progress` 仅供查看；智能体通过 `update_task` 声明 `done` / `pending` / `cancelled`。对于已中断的 pending 任务，请为同一 ID 指定 `resume: true` 以明确恢复，不要用其他任务替换。

**长时间运行的 CLI 工具**（`animaworks-tool submit …`）会作为 `task_type="command"` 注册到 TaskStore，由 PendingTaskExecutor 获取尝试并在后台执行。可通过 `state/background_tasks/{task_id}.json` 和通知查看完成结果。详情请参阅 `operations/background-tasks.md`。

**注意**：不要直接编辑保存位置。旧版 `state/task_queue.jsonl` 和 `state/pending/` 仅作为迁移和导出的记录保存，不要用作运行中的任务投入目标。

即使只有一个任务，也要使用 `submit_tasks`（tasks 数组中包含 1 项）。
多个独立任务使用 `parallel: true` 并行执行；存在依赖关系时指定 `depends_on`。
详情请参阅 task-management。

### 心跳触发类型

心跳有两种触发方式：

| 触发方式 | 说明 |
|---------|------|
| 定期心跳 | 根据 `config.json` 中的 `heartbeat.interval_minutes`，由 APScheduler 定期启动 |
| 消息触发 | Inbox 收到未读消息时立即启动（按 Inbox 路径处理） |

消息触发由 Inbox JSON 文件变更通知启动。为防止遗漏通知，每 45 秒重新检查一次未读消息。
同时只会运行一个 Inbox 处理任务；运行期间收到的消息会合并到下一次处理中。Provider 出错时，会等待 `rate_guard` 指定的恢复时间，并保留未读消息。
没有基于接收 intent 的启动筛选，也没有针对消息触发的 cooldown / cascade 抑制。对于只表示知悉或感谢的消息，不作回复的行为规则应通过 Inbox 提示词指定。

## HEARTBEAT.md 的配置

`HEARTBEAT.md` 是每个 Anima 的配置文件，用于定义活动时间和检查项。
心跳的运行间隔可通过 `config.json` 的 `heartbeat.interval_minutes` 配置（1～1440 分钟，默认 30 分钟）。无法通过 `HEARTBEAT.md` 更改。
每个 Anima 会根据名称分配 0～9 分钟的偏移量，以错开同时启动。
文件路径：`~/.animaworks/animas/{name}/HEARTBEAT.md`

上司要编辑下属 Anima 的 `HEARTBEAT.md` 时，应使用 `read_memory_file` / `write_memory_file`，而非直接操作文件，并使用类似 `../{anima_name}/HEARTBEAT.md` 的相对路径指定。

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

**运行间隔**：
- 通过 `config.json` 的 `heartbeat.interval_minutes` 配置（1～1440 分钟，默认 30 分钟）。无法通过 `HEARTBEAT.md` 更改

**活动时间**（SHOULD）：
- 使用 `HH:MM - HH:MM` 格式填写（例如：`9:00 - 22:00`）
- 在此时间范围之外不会启动心跳
- 未配置时的默认值：24 小时（全天）
- 时区可通过 `config.json` 的 `system.timezone` 配置。未配置时会自动检测系统时区

**检查清单**（MUST）：
- 智能体在心跳启动时检查的项目
- 使用项目列表（以 `- ` 开头）填写
- 检查清单内容会原样传递给智能体的提示词
- 可自定义：可根据 Anima 的职责添加或更改项目

### 自定义检查清单示例

默认值（所有 Anima 通用）：
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

### 运行模型（成本优化）

如果已配置 `background_model`，Heartbeat / Inbox / Cron 将使用该模型运行，而非主模型。
Chat（与人类对话）和 TaskExec（实际工作）仍使用主模型。

配置方法：`animaworks anima set-background-model {名前} claude-sonnet-5-5`
详情请参阅 `reference/operations/model-guide.md` 的“后台模型”章节。

### 心跳的内部工作机制

- **崩溃恢复**：如果上次心跳失败，错误信息会保存到 `state/recovery_note.md`。下次启动时，该信息会注入提示词；恢复后会删除此文件。
- **回顾记录**：如果心跳输出中包含 `[REFLECTION]...[/REFLECTION]` 块，该内容会作为 `heartbeat_reflection` 记录到 activity_log 中，并纳入之后心跳的上下文。
- **部下检查**：对于有部下的 Anima，心跳和 Cron 的提示词会自动注入检查部下状态的指令。
- **会话时间限制**（`config.json` 的 `heartbeat`）：经过 `soft_timeout_seconds`（默认 300 秒）后注入收尾提醒；经过 `hard_timeout_seconds`（默认 600 秒）后强制结束会话。
- **空闲时自动压缩**：`heartbeat.idle_compaction_minutes`（默认 10 分钟）— 流结束后经过这段时间，将运行空闲自动压缩（执行引擎侧的配置）。

### 定期心跳的调度方式

当实际间隔（应用 Activity Level 后、下限按 5 分钟取整）**不超过 60 分钟且能整除 60** 时，会通过 APScheduler 的 `CronTrigger` 分散注册到各分钟时段（并结合基于名称的 0～9 分钟偏移量）。

其他情况（例如实际间隔为 61 分钟，或 43 分钟这类无法整除 60 的间隔）则通过**每分钟轮询**（`_heartbeat_check`），根据“距上次运行的时间”触发。此实现用于规避由旧版 `IntervalTrigger` 引发的问题。

### 后台工具（core/tasks/background.py）

`core/tasks/background.py` 不负责心跳/Cron 的调度本身，而负责**长时间工具调用的后台运行**（包括状态的 JSON 持久化）。运维详情和 CLI 路径也请参阅 `operations/background-tasks.md`。

#### BackgroundTaskManager

- **保存位置**：`state/background_tasks/{task_id}.json`。`task_id` 是 UUID 的前 12 个字符（十六进制）。每个文件记录 `task_id`、`anima_name`、`tool_name`、`tool_args`、`status`、`created_at`、`completed_at`、`result`、`error`。
- **状态（`TaskStatus`）**：`pending` / `running` / `completed` / `failed`。对于 `submit` / `submit_async`，投入后会立即以 `running` 写入 JSON；任务完成或发生异常时，状态会更新为 `completed` / `failed`。
- **执行 API**：`submit(tool_name, tool_args, execute_fn)` 会在线程池中通过 `asyncio` 的 `run_in_executor` 执行同步 callable。`submit_async` 会直接 await 异步 callable。提供 `get_task`（优先从内存获取，没有时从磁盘获取）、`list_tasks`（合并磁盘上的 JSON，按创建时间从新到旧排列）、`active_count`（内存中的 `running` 数量）。
- **完成回调**：传递给 `on_complete` 的异步函数会在任务保存后调用。即使回调中发生异常，任务结果仍会保留，异常仅记入日志（通常会结合写入 `state/background_notifications/`；之后在**下次心跳**中读取、删除并纳入对话上下文）。
- **候选判定 `is_eligible(tool_name)`**：如果映射中存在对应键，则视为后台运行目标。键支持以下两种形式：(1) **模式名称**（例如 `generate_3d_model`）— 如 Mode A 的外部工具分发。(2) **`ツール名:サブコマンド`**（例如 `image_gen:pipeline`）— 各工具模块在 `EXECUTION_PROFILE` 中通过 `get_eligible_tools_from_profiles()` 将 `background_eligible: true` 条目按此形式注册（如 Mode S 的 `submit` 路径）。
- **构建候选工具映射 `from_profiles()`**：通过 dict 的 `update` 合并以下 3 层，**后者优先覆盖前者**。(1) `_DEFAULT_ELIGIBLE_TOOLS`（代码默认值）(2) 参数 `profiles`（`EXECUTION_PROFILE` 汇总）(3) 参数 `config_eligible`（通常是从 `config.json` 的 `background_task.eligible_tools` 展开 `threshold_s` 后得到的 `名前 → 秒`）。值为供配置文件联动使用的预期秒数（整数）。
- **代码默认值 `_DEFAULT_ELIGIBLE_TOOLS`（秒）**：`generate_character_assets` 为 30，`generate_fullbody` / `generate_bustup` / `generate_icon` / `generate_chibi` 各为 30，`generate_3d_model` / `generate_rigged_model` / `generate_animations` 各为 30，`local_llm` 为 60，`run_command` 为 60。
- **清理 `cleanup_old_tasks(max_age_hours=24)`**：`status` 会通过 `completed` / `failed` 删除 `completed_at` 早于**参数指定时间（默认 24 小时）**的 JSON。此外，`running` 状态的文件如果从 `created_at` 起已超过 **48 小时**，也会作为崩溃遗留文件删除。返回值为删除数量。
- **保留时间**：`cleanup_old_tasks` 已完成任务的保留时间由参数 `max_age_hours` 指定（默认 24 小时）。没有对应的 `config.json` 配置键。

- **其他服务器定时任务**：除以 Anima 为单位的 `cron.md` 外，生命周期管理还会注册内存维护、RAG 等系统 Cron（例如，每日汇总 02:00，每日索引 04:00）。时间以配置时区为准。

### 心跳配置热重载

在文件系统中更新 HEARTBEAT.md 后，`_check_schedule_freshness()` 会在下次心跳运行时检测到变更，并由 SchedulerManager 自动重新加载调度。
无需重启服务器（MAY skip restart）。APScheduler 作业会重新注册。

## 每个 Anima 的 Heartbeat 间隔配置

### 在 status.json 中配置

root 管理员可以通过 root CLI/API 为单个 Anima 配置 `heartbeat_interval_minutes`。值会保存到 root 所有的 `status.json` 中，Anima 进程不会编辑该文件。

```json
{
  "heartbeat_interval_minutes": 60
}
```

- 可配置范围：1～1440 分钟（1 天）
- 未配置时：回退到 `config.json` 的 `heartbeat.interval_minutes`（默认 30 分钟）
- `status.json` 归 root 所有。管理员可在服务器停止期间通过 `animaworks config set animas.<name>.heartbeat_interval_minutes <minutes>` 设置单独的覆盖值。服务器运行期间，CLI 会通过 root API 应用设置。Anima 进程不会编辑此文件。

### 推荐指南

| 状况 | 推荐间隔 | 原因 |
|------|----------|------|
| 正在进行活跃的开发项目 | 15〜30分钟 | 需要频繁了解情况 |
| 日常工作 | 30〜60分钟 | 默认值。频率较为均衡 |
| 低负载・待机状态 | 60〜120分钟 | 节省成本。没有任务时可设得更长 |
| 长期休眠・非活跃 | 120〜1440分钟 | 以最低限度的巡查了解情况 |

### 与 Activity Level 的关系

如果已设置全局 Activity Level（10%〜400%），实际间隔按以下公式计算：

```
実効間隔 = ベース間隔 / (Activity Level / 100)
```

示例：基准间隔30分钟，Activity Level 50% → 实际间隔60分钟
示例：基准间隔30分钟，Activity Level 200% → 实际间隔15分钟

- 实际间隔下限为5分钟（无论提升多少，都不会低于5分钟）

### Activity Schedule（按时段自动切换／夜间模式）

根据时段自动切换 Activity Level 的机制。
适用于希望在夜间或休息日控制成本，或只在工作时间活跃运行的情况。

#### 工作原理

- 在 `config.json` 的 `activity_schedule` 中设置时段条目
- 每分钟检查当前时间，并自动将 Activity Level 调整为对应时段的级别
- Activity Level 发生变化时，所有 Anima 的心跳都会立即重新安排

#### 配置格式

每个条目包含3个字段：`start`（开始时间）、`end`（结束时间）、`level`（Activity Level %）：

```json
{
  "activity_schedule": [
    {"start": "09:00", "end": "22:00", "level": 100},
    {"start": "22:00", "end": "06:00", "level": 30}
  ]
}
```

- 时间采用 `HH:MM` 格式（24小时制）
- **支持跨日期**：可以指定 `"22:00"` 〜 `"06:00"` 这样的 start > end 时段（覆盖深夜时段）
- `level` 的范围为10〜400
- 最多可设置24个条目
- 使用空数组 `[]` 可禁用计划模式（恢复为固定 Activity Level）

#### 配置方法

- **Settings UI**：夜间模式复选框 + 时段和级别配置
- **API**：将上述 JSON 发送到 `PUT /api/settings/activity-schedule`
- **root 配置 API**：使用 Settings UI，或将上述 JSON 发送到 `PUT /api/settings/activity-schedule`。不要从 Anima 进程直接编辑 `config.json`

#### 注意事项

- 手动更改 Activity Level 时，当前时段对应的计划条目也会同步更新
- 计划会在服务器启动时立即应用（根据启动时的时间设置对应级别）
- 如果当前时间不属于任何时段，则维持最后设置的 Activity Level

## 什么是 Cron 任务

Cron 是“在指定时间自动执行的任务”。心跳是“定期巡查”，而 Cron 则是“定时工作”。

示例：
- 每天早上9:00制定工作计划
- 每周五17:00进行每周回顾
- 每天2:00运行备份脚本

## 配置 cron.md

Cron 任务以 Markdown + YAML 格式定义在 `cron.md` 中。
文件路径：`~/.animaworks/animas/{name}/cron.md`

上司编辑下属 Anima 的 `cron.md` 时，应使用 `read_memory_file` / `write_memory_file`，而不是直接操作文件，并使用类似 `../{anima_name}/cron.md` 的相对路径指定。

### 基本格式

每个任务以 `## タスク名` 标题开头，并在正文开头使用 `schedule:` 指令写入标准的五字段 cron 表达式。

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

旧格式（例如在 `## タスク名（毎日 9:00 JST）` 中将计划写在括号内的格式）会在服务器启动时或通过 `animaworks migrate` 自动转换为新格式。

### CronTask 的模式

每个任务在内部会被解析为以下 `CronTask` 模型：

| 字段 | 类型 | 默认值 | 说明 |
|-----------|------|-----------|------|
| `name` | str | （必填） | 任务名称。从 `##` 标题中提取 |
| `schedule` | str | （必填） | 标准的五字段 cron 表达式。从 `schedule:` 指令中提取 |
| `type` | str | `"llm"` | 任务类型：`"llm"` 或 `"command"` |
| `description` | str | `""` | LLM 类型的指令文本（用于 type: llm） |
| `command` | str \| None | `None` | Command 类型的 bash 命令 |
| `tool` | str \| None | `None` | Command 类型的内部工具名称 |
| `args` | dict \| None | `None` | tool 参数（YAML 格式） |
| `skip_pattern` | str \| None | `None` | Command 类型：stdout 与此正则表达式匹配时，跳过后续 LLM |
| `trigger_heartbeat` | bool | `True` | Command 类型：如果是 `False`，则在命令输出后跳过后续 cron LLM |

## LLM 类型 Cron 任务

`type: llm` 是由代理（LLM）通过判断和思考来执行的任务。
description 中写明的指令会作为提示词传递给代理。

### 特点

- 代理会使用工具、搜索记忆并作出判断
- 结果不固定（每个任务的输出各不相同）
- 执行时需要调用模型 API（会产生费用）

### 示例

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
昨日の episodes/ を読み返し、今日のタスクを計画する。
優先順位は理念と目標に照らして判断する。
結果は state/current_state.md に書き出す。
正本タスク一覧（`list_tasks`） の未着手タスクも確認し、必要なら優先度を見直す。
```

description（`type:` 行之后的正文）应包含（SHOULD）以下内容：
- 要检查什么（输入）
- 如何判断（标准）
- 要输出什么（产出）

## Command 类型 Cron 任务

`type: command` 是不经过代理判断，直接运行固定命令或工具的任务。
适用于确定性处理（备份、发送通知等）。

### bash 命令类型

```markdown
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

在 `command:` 中用一行写入 bash 命令。
命令会通过 shell 执行。

### 内部工具类型

```markdown
## Slack朝の挨拶
schedule: 0 9 * * 1-5
type: command
tool: slack_send
args:
  channel: "#general"
  message: "おはようございます！本日もよろしくお願いします。"
```

在 `tool:` 中写入内部工具名称，并在 `args:` 中以 YAML 格式写入参数。
args 会作为 YAML 缩进块解析（缩进2个空格）。

### Command 类型的后续处理控制

Command 类型任务在命令正常结束且有 stdout 输出时，会将该输出传递给 LLM 进行后续分析（在相当于 heartbeat 的上下文中执行）。

- **`trigger_heartbeat: false`** — 跳过后续 LLM（无需分析输出时）
- **`skip_pattern: <正規表現>`** — stdout 与此正则表达式匹配时，跳过后续处理

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

### LLM 类型与 Command 类型的选择

| 角度 | LLM 类型 | Command 类型 |
|------|--------|-----------|
| 是否需要判断 | 是 | 否 |
| API 成本 | 有 | 无 |
| 输出的可预测性 | 不固定 | 确定 |
| 适用任务 | 制定计划、回顾、撰写文章 | 备份、发送通知、获取数据 |
| 出错时的处理 | 代理自主处理 | 仅记录到日志 |

拿不准时可参考以下指南：
- “每次只做同一件事” → Command 类型（SHOULD）
- “需要根据情况改变判断” → LLM 类型（SHOULD）
- “运行命令 + 解释结果” → 使用 LLM 类型，并在 description 中指示运行命令

## 计划语法

cron.md 的 `schedule:` 指令中必须写入**标准的五字段 cron 表达式**。

### 标准 cron 表达式（必填）

```
分 時 日 月 曜日
```

示例：
- `0 9 * * *` — 每天9:00
- `0 9 * * 1-5` — 工作日9:00
- `*/30 9-17 * * *` — 9:00〜17:00每30分钟
- `0 2 1 * *` — 每月1日2:00
- `0 17 * * 5` — 每周五17:00

可以通过 `config.json` 的 `system.timezone` 设置时区。未设置时会自动检测系统时区。

### 从日语计划迁移

以旧格式（`## タスク名（毎日 9:00 JST）`）编写的 cron.md 会在服务器启动时或通过 `animaworks migrate` 自动转换为标准 cron 表达式。转换对照表：

| 日语写法 | cron 表达式示例 |
|-----------|----------|
| `毎日 HH:MM` | `0 9 * * *` |
| `平日 HH:MM` | `0 9 * * 1-5` |
| `毎週{曜日} HH:MM` | `0 17 * * 5`（周五） |
| `毎月N日 HH:MM` | `0 9 1 * *` |
| `X分毎` | `*/5 * * * *` |
| `X時間毎` | `0 */2 * * *` |

`隔週`、`毎月最終日`、`第N曜日` 无法自动转换。请手动编写 cron 表达式。

## 如何查看 cron_logs

Cron 任务的执行结果会记录在服务器日志中。
也会通过 WebSocket 以 `anima.cron` 事件的形式广播。

查看日志的方法：
- 服务器日志：`animaworks.lifecycle` 记录器的 INFO 级别
- Web UI：显示在仪表板的活动动态中
- 对于 episodes/: LLM 类型任务，代理自身应在 episodes/ 中写入日志（SHOULD）

LLM 类型任务的结果会记录为 `CycleResult`，包含以下信息：
- `trigger`：`"cron"`
- `action`：代理行为摘要
- `summary`：结果摘要文本
- `duration_ms`：执行时间（毫秒）
- `context_usage_ratio`：上下文使用率

## 常见 Cron 配置示例

### 基本套件（推荐所有 Anima 使用）

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

### 注释掉任务

不希望执行的任务请用 HTML 注释括起来：

```markdown
<!--
## 一時停止中のタスク
schedule: 0 15 * * *
type: llm
このタスクは一時的に停止中。
-->
```

注释中的 `## ` 标题会被解析器忽略。

## Cron 配置热重载

更新 cron.md 后，计划会自动重新加载，与 HEARTBEAT.md 一样。
即使 Anima 自身改写 cron.md，也会立即生效（self-modify 模式）。
上司可以使用 write memory 工具编辑下属的 `cron.md` / `HEARTBEAT.md`。来自获准上司的 `AGENTS.md` 修改请求也会转发给 root 并获得授权。修改 `status.json` 时，请使用面向上司的对应工具或 root CLI/API。不要通过 Read / Write / Edit / apply_patch / `Path.write_text` / shell 重定向等方式直接写入 root 所有的文件。

重载时的行为：
1. 删除对应 Anima 的所有现有 cron 作业
2. 解析更新后的 cron.md 并注册新作业
3. 日志中会输出 `Schedule reloaded for '{name}'`

自行更新 cron.md 时的注意事项：
- 在标题（`## タスク名`）之后紧接着放置 `schedule:` 指令（MUST）
- 计划必须使用标准的五字段 cron 表达式编写（MUST）
- type 行应放在 schedule 之后（SHOULD）
