---
name: cron-management
description: >-
  cron.md以正确格式读写的技能。提供定时任务的添加、更新、删除步骤。
  Use when: cron.md的编辑、cron表达式的添加、LLM型・命令型任务的添加・删除、定时作业的维护需要时。
---


## 框架侧实现（供参考）

### cron（定时任务）

cron 的**解析**由 `core/runtime/schedule_parser.py`（`parse_cron_md` / `parse_schedule`）负责，**注册・执行・重载**由 `core/runtime/scheduler_manager.py`（APScheduler、`AsyncIOScheduler(timezone=get_app_timezone())`）负责。

### `core/tasks/background.py`（与 cron 属于不同系统）

此模块**不负责 cron 调度**，而是负责在后台执行耗时较长的工具调用。请勿与 cron 混淆。

**`BackgroundTaskManager`**

- **职责**：在 `asyncio` 上异步执行目标工具，并将状态保存到磁盘。
- **持久化位置**：`state/background_tasks/{task_id}.json`（`pending` / `running` / `completed` / `failed`，结果字符串、错误和时间戳）。
- **公开 API（概要）**：
  - `submit(tool_name, tool_args, execute_fn)` — 在线程池中运行同步执行函数。
  - `submit_async(...)` — 异步 `execute_fn` 版本。
  - `get_task` / `list_tasks` / `active_count` — 查询。
  - `cleanup_old_tasks(max_age_hours=24)` — 删除完成或失败超过 **24 小时**的任务。此外，也会删除仍处于 `running` 状态超过 **48 小时**的 JSON 文件（崩溃后遗留的孤儿文件）。
- **`from_profiles(anima_dir, ..., profiles, config_eligible)`**：后台目标工具集合通过以下 **3 层合并**（后者优先）：
  1. 代码内默认值 `_DEFAULT_ELIGIBLE_TOOLS`（与 Mode A 兼容的固定列表）
  2. 来自各工具模块的 `EXECUTION_PROFILE` 的 `background_eligible: true` 条目（`core.integrations._base.get_eligible_tools_from_profiles`）。键采用 `"{tool}:{subcommand}"` 格式（与 Mode S 的 `submit` 一致）
  3. 通过 `config.json` 等传入的 `config_eligible`（显式覆盖）
- **`is_eligible(tool_name)`**：可通过以下任一名称进行匹配 — 架构名称（例如 `generate_3d_model`）、配置文件键（例如 `image_gen:3d`）。
- **默认包含在 `_DEFAULT_ELIGIBLE_TOOLS` 中的示例**（数值为秒，仅供参考）：`generate_character_assets` / `generate_fullbody` / `generate_bustup` / `generate_icon` / `generate_chibi` / `generate_3d_model` / `generate_rigged_model` / `generate_animations`（各 30 秒），`local_llm` / `run_command`（各 60 秒）。
- **完成钩子**：可在 `on_complete` 中设置 `Callable[[BackgroundTask], Awaitable[None]]`。应用侧可以在此处编写向 `state/background_notifications/` 发送 Markdown 通知等逻辑（`background.py` 本身不会访问通知目录）。

---

## cron.md的结构

### 整体构成

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

### 必须绝对遵守的规则

1. **每个任务以 `## タスク名` 开头**（H2标题。不使用H3或H1）
2. **`schedule:` 行是必需的**（以 `schedule:` 这个关键词开头。不要做成 `###` 标题）
3. **调度仅使用标准5字段cron表达式**（不可使用 `09:00` 或 `毎週金曜 17:00`）
4. **`type:` 行是必需的**（`llm` 或 `command`）
5. **任务之间插入空行**（为了可读性）

### 不应采用的写法

```markdown
❌ ### */5 * * * *           ← H3見出しにcron式を書いてはいけない
❌ ### 09:00                 ← 自然言語の時刻表記は不可
❌ ### 毎週金曜 17:00         ← 日本語のスケジュール表記は不可
❌ cron: 0 9 * * *           ← キー名は "schedule:" であること（"cron:" ではない）
❌ interval: 5m              ← interval形式は不可
❌ schedule: 0 9 * * * *     ← 6フィールドは不可（5フィールドのみ）
```

### 正确的写法

```markdown
✅ schedule: 0 9 * * *       ← "schedule:" + 半角スペース + 5フィールドcron式
✅ schedule: */5 * * * *
✅ schedule: 30 21 * * *
✅ schedule: 0 17 * * 5
```

---

## 5字段cron表达式参考

### 字段构成

```
schedule: 分 時 日 月 曜日
```

| 字段 | 位置 | 范围 | 说明 |
|-----------|------|------|------|
| 分 | 1 | 0-59 | 在第几分执行 |
| 时 | 2 | 0-23 | 在第几时执行（24小时制） |
| 日 | 3 | 1-31 | 在第几日执行 |
| 月 | 4 | 1-12 | 在第几月执行 |
| 星期 | 5 | 0-6 | 在第几星期执行（0=周一, 6=周日） |

**注意**: 星期为 **0=周一, 6=周日**（APScheduler的规格。与一般cron的0=周日不同）

### 特殊字符

| 字符 | 含义 | 示例 |
|------|------|-----|
| `*` | 所有值 | `* * * * *` = 每分钟 |
| `*/n` | n间隔 | `*/5 * * * *` = 每5分钟 |
| `n-m` | 范围 | `0 9-17 * * *` = 9时〜17时的每个整点 |
| `n,m` | 列表 | `0 9,12,18 * * *` = 9时・12时・18时 |
| `n-m/s` | 范围+间隔 | `0 9-17/2 * * *` = 9时〜17时每2小时 |

### 常用调度示例

#### 每日系

| 想要做的事 | cron表达式 | 说明 |
|-------------|--------|------|
| 每天早上9:00 | `0 9 * * *` | 分=0, 时=9 |
| 每天早上9:30 | `30 9 * * *` | 分=30, 时=9 |
| 每天12:00（正午） | `0 12 * * *` | 分=0, 时=12 |
| 每天18:00 | `0 18 * * *` | 分=0, 时=18 |
| 每晚21:30 | `30 21 * * *` | 分=30, 时=21 |
| 每天深夜2:00 | `0 2 * * *` | 分=0, 时=2 |

#### 间隔系

| 想要做的事 | cron表达式 | 说明 |
|-------------|--------|------|
| 每5分钟 | `*/5 * * * *` | 分=*/5（0,5,10,...,55） |
| 每10分钟 | `*/10 * * * *` | 分=*/10（0,10,20,...,50） |
| 每15分钟 | `*/15 * * * *` | 分=*/15（0,15,30,45） |
| 每30分钟 | `*/30 * * * *` | 分=*/30（0,30） |
| 每1小时 | `0 * * * *` | 每小时0分 |
| 每2小时 | `0 */2 * * *` | 0时,2时,4时,...,22时的0分 |
| 仅业务时间内每5分钟 | `*/5 9-17 * * *` | 9:00〜17:55的5分钟间隔 |
| 仅业务时间内每1小时 | `0 9-17 * * *` | 9:00〜17:00的每个整点 |

#### 星期系

| 想要做的事 | cron表达式 | 说明 |
|-------------|--------|------|
| 工作日的每天早上9:00 | `0 9 * * 0-4` | 星期=0-4（周一〜周五） |
| 每周周一9:00 | `0 9 * * 0` | 星期=0（周一） |
| 每周周五17:00 | `0 17 * * 4` | 星期=4（周五） |
| 每周周五18:00 | `0 18 * * 4` | 星期=4（周五） |
| 工作日的业务时间内每30分钟 | `*/30 9-17 * * 0-4` | 工作日9:00〜17:30 |
| 周末的每天早上10:00 | `0 10 * * 5,6` | 星期=5,6（周六周日） |

#### 月次系

| 想要做的事 | cron表达式 | 说明 |
|-------------|--------|------|
| 每月1日的9:00 | `0 9 1 * *` | 日=1 |
| 每月15日的12:00 | `0 12 15 * *` | 日=15 |
| 每月最后营业日附近（28日） | `0 17 28 * *` | 日=28（近似） |
| 季度首日9:00（1,4,7,10月） | `0 9 1 1,4,7,10 *` | 月=1,4,7,10 |

---

## 任务类型详情

### type: llm — LLM判断任务

需要思考・判断的任务。在 `schedule:` 和 `type: llm` 之后，以自由记述书写任务内容。

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_state.md に書き出す。
```

- 说明文会原样作为提示词传递给LLM
- 明确写出具体输出（写出什么）会更有效
- 可多行
- 正文中包含围栏代码块（\`\`\`）が含まれると、パーサーが警告ログを出す（確定コマンドなら `type: command` 可考虑）

### type: command — 命令执行任务

确定性执行的bash命令或工具调用。

#### 模式A: bash命令

```markdown
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

- 在 `command:` 中以一行书写要执行的命令
- 可使用shell重定向（`>`, `>>`, `|`）
- 多行命令不推荐（合并为一行，或做成脚本文件）

#### 模式B: 工具调用

```markdown
## Slack朝の通知
schedule: 0 9 * * 0-4
type: command
tool: slack_channel_post
args:
  channel_id: "C0123456789"
  text: "おはようございます！"
```

- 在 `tool:` 中写工具名（包含在 `permissions.json` 的许可配置中的schema名。例: Slack 投稿为 `slack_channel_post` 等）
- `args:` 之后为YAML块形式，缩进2个空格
- 在 `ToolHandler.handle(tool, args)` 中执行，结果字符串作为 stdout 等价物处理

### 可选: skip_pattern

当 `type: command` 中命令**成功**（`exit_code == 0`）且标准输出非空时才会运行的**后续 cron LLM**（heartbeat 同等上下文的分析会话），在 stdout 匹配此正则表达式时**抑制**。

```markdown
## Chatwork未返信チェック
schedule: */5 * * * *
type: command
command: chatwork_cli.py unreplied --json
skip_pattern: "^\[\]$"
```

- 在 `skip_pattern:` 中写正则表达式（运行时为 `re.search(skip_pattern, stdout)`）
- 若正则表达式包含 `[]` 等YAML特殊字符，则用引号（`"..."` 或 `'...'`）括起。解析器会自动去除外层引号
- **解析时**无效的正则表达式 → 记录警告日志，`skip_pattern` 视为未设置（不抑制后续）
- **运行时** `re.search` 发生异常（无效模式残留等）→ 记录警告日志，**不抑制**并执行后续
- 在上例中，未回复为0件（`[]`）时跳过后续

### 可选: trigger_heartbeat

按任务单位控制命令成功时是否运行后续 cron LLM（遵循 `SchedulerManager._run_cron_task` 的评估顺序）。

```markdown
## Chatwork未返信チェック
schedule: */15 * * * *
type: command
command: animaworks-tool chatwork unreplied
skip_pattern: "^\[\]$"
trigger_heartbeat: false
```

- **考虑后续的条件**（全部满足时）: `exit_code == 0` 且 stdout（`.strip()` 后）非空
- **stdout 的长度**: `run_cron_command` 返回给调度器的 `stdout` 会被截断为前 **1000 字符**。传递给后续 LLM 的也是此预览。完整日志请参照 `state/cron_logs/`（JSONL）侧
- `trigger_heartbeat: false` — 即使满足上述条件也**不执行**后续 cron LLM（在 `skip_pattern` 之前评估）
- `trigger_heartbeat: true`（默认）— 满足条件则执行后续（接着评估 `skip_pattern`）
- 指定 `false`, `no`, `0` 时抑制。其他情况视为 true
- 后续 cron LLM 以 heartbeat 同等的提示词过滤器（后台用上下文）运行
- `exit_code != 0` 或 stdout 为空时，后续**根本不会启动**（`skip_pattern` / `trigger_heartbeat` 无关）

---

## 类型区分判断标准

### 应使用 type: command 的情况
- 要执行的命令完全确定
- 参数固定（区域、集群名、profile等）
- 结果判断交给cron LLM会话

### 应使用 type: llm 的情况
- 需要根据状况改变执行内容
- 需要组合多个工具进行调查
- 在执行阶段需要人类式的判断・分析

### 禁止模式
- 在 type: llm 中包含代码块（确定命令）
  → 该命令应设为 type: command
- 写着“按此执行”却用 type: llm
  → LLM无法准确再现命令。请使用 type: command

### Anima不是“背诵命令的人”
type: command 与人类保存脚本相同。
Anima的价值在于查看结果并做出判断的能力。
将确定性执行交给框架，
让Anima专注于判断・分析・报告。

---

## cron.md 解析通知（自动）

当检测到 cron.md 的首次注册或 `reload_schedule` 时的解析・注册错误时，调度器会生成 `state/background_notifications/cron_health_{タイムスタンプ}.md`。通知会包含在下一次 heartbeat 或 cron 执行的上下文中。没有定期监控已注册 cron 执行状态的功能。

**检测到的问题**：

- 任务已定义，但**无法注册任何一条有效的调度**（例如所有表达式均无效）
- 在注释或代码围栏之外，存在行首包含空白的 `schedule:` 行（`schedule:` 应写在行首）
- 有效的配置文本中存在 `schedule:`，但解析器**没有返回任何任务**

---

## cron.md操作步骤

### 目标路径与下属编辑

- 自己的 `cron.md` 通过 `read_memory_file(path="cron.md")` 读取，通过 `write_memory_file(path="cron.md", ...)` 更新。
- 上司可以使用 write memory 工具，以 `../{anima_name}/cron.md` 的方式指定并编辑下属的 `cron.md` / `HEARTBEAT.md`（对象为子、孙及以下的所有下属）。
- `status.json` / `IDENTITY.md` / `SOUL.md` / `permissions.json` 归 root 所有，无法通过记忆・文件工具写入。被允许的上司对 `AGENTS.md` 的变更以及 bootstrap 期间的 IDENTITY.md / SOUL.md 创建请求会被转发并授权给 root。不要直接操作文件。

### 新增任务

1. 读取自己的 `cron.md`
2. 在文件末尾添加新的 section
3. **写入前确认格式**（参见下方检查清单）
4. 写入文件

```markdown
## 新しいタスク名
schedule: <5フィールドcron式>
type: llm|command
<説明またはcommand/tool行>
```

### 修改现有任务

1. 读取 `cron.md`
2. 定位对应 section（从 `## タスク名` 到下一个 `##` 之前）
3. 编辑要修改的行（`schedule:`、`type:`、说明文字等）
4. 写入文件

### 删除任务

1. 读取 `cron.md`
2. 删除整个对应 section（从 `## タスク名` 到下一个 `##` 之前）
3. 写入文件

### 临时禁用任务

用 HTML 注释包围后，解析器会跳过：

```markdown
<!--
## 一時停止中のタスク
schedule: 0 9 * * *
type: llm
このタスクは一時的に停止中。
-->
```

---

## 写入前检查清单

更新 cron.md 之前，**务必**确认以下内容：

- [ ] 每个任务是否以 `## タスク名` 开头（而不是 `###` 或 `#`）
- [ ] 是否有 `schedule:` 行（而不是 `###` 标题或自然语言）
- [ ] 调度是否为 5 字段 cron 表达式（`分 時 日 月 曜日`）
- [ ] 各字段的值是否在有效范围内（分：0-59，时：0-23，日：1-31，月：1-12，星期：0-6）
- [ ] 是否有 `type:` 行（`llm` 或 `command`）
- [ ] 如果是 command 型，是否有 `command:` 或 `tool:`
- [ ] 如果是 tool 型，`args:` 的缩进是否正确（2 个空格）
- [ ] 任务之间是否有空行
- [ ] 是否没有将 `schedule:` 写在代码块内（可能导致健康警告）

### 验证方法

写入后，可以用以下命令确认是否能正确解析：

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

如果所有任务都显示 ✅ 则正常。如果出现 ❌，请修正调度表达式。

---

## 完整示例

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

## 常见错误与修正

| 错误 | 正确写法 | 原因 |
|--------|------------|------|
| `### */5 * * * *` | `schedule: */5 * * * *` | H3 标题不能用作任务分隔 |
| `### 09:00` | `schedule: 0 9 * * *` | 自然语言时间无法解析 |
| `### 毎週金曜 17:00` | `schedule: 0 17 * * 4` | 日文表记无法解析 |
| `schedule: 9:00` | `schedule: 0 9 * * *` | HH:MM 格式不是 5 字段 cron 表达式 |
| `schedule: every 5 minutes` | `schedule: */5 * * * *` | 英文表记无法解析 |
| `schedule: 0 9 * * 7` | `schedule: 0 9 * * 6` | 星期 7 超出范围（0-6） |
| `schedule: 0 9 * * SUN` | `schedule: 0 9 * * 6` | 不能使用星期名称（仅限数字） |
| `schedule: 0 25 * * *` | `schedule: 0 23 * * *` | 时应在 0-23 范围内 |
| `schedule: 60 * * * *` | `schedule: 0 * * * *` | 分应在 0-59 范围内 |
| `skip_pattern: ^\[\]$`（无引号） | `skip_pattern: "^\[\]$"` | `[]` 在 YAML 中会被解释为空列表。请用引号包围 |

---

## 注意事项

- **热重载**：通过 Anima 的 `write_memory_file` 等更新 `cron.md` / `HEARTBEAT.md` 时，调度变更回调会触发 `reload_schedule` 并立即重新注册。下属管理文件的编辑也应通过 `../{anima_name}/cron.md` 这样的 write memory 工具进行。外部直接编辑也会在下一次 **heartbeat 或任意 cron 触发之前** 的 mtime 检查（`_check_schedule_freshness`）中被检测到，但不作为上司编辑下属的标准步骤
- **重载后立即出现的旧 job**：当检测到 mtime 变化时调度器会重建，**与之前定义绑定的 cron 触发可能被作为「stale」跳过**（有意防止重复执行）
- **时区**：APScheduler 使用 `get_app_timezone()`。可以在 `config.json` 的 `system.timezone` 中设置 IANA 名称（例如 `Asia/Tokyo`）。当为**空字符串**时，会自动检测 OS 时区，失败时回退到 `Asia/Tokyo`
- **并发执行**：**如果任务名不同**，即使同一分钟内有多个 job 重叠，也可能因 `asyncio.create_task` 而并行执行。同一任务名在执行中不会重入（通过 `_cron_running` 跳过）。每个 job 为 `max_instances=1`
- command 型即使失败也不会停止进程（会记录 `cron_executed`，`stderr` / exit_code 会留在日志中）。**后续 cron LLM 仅在成功且存在 stdout 时**（前述）
- `type: llm` 的定期执行会使用**后台用模型**（`status.json` 的 `background_model` 等，未设置时使用主模型）
- command 的详细输出会累积到 `state/cron_logs/`（每日 JSONL），最多保留 14 天进行清理
- **访问其他 Anima 的 tools/ 目录时，请事先确认权限**
