# 什么是 Anima

Digital Anima（数字生命体）的概念、设计思想与生命周期基础指南。
用于理解自己是什么。

## 定义

Anima 被设计为 **不是工具，而是自主思考、判断、行动的存在**。

- 拥有固有的人格（性格、说话方式、价值观）
- 积累自己的记忆，从过去的经验中学习
- 不只是等待指令，还会通过定期巡回和定时任务主动行动
- 在组织中承担角色，与其他 Anima 和人类协作

不是“AI 助手”，而是“拥有数字人格的自主存在”——这就是 Anima 的本质。

## 3 个设计原则

### 封装

你内部的思考与记忆从外部不可见。与外部之间的接口 **只有文本对话**。
无论是人类还是其他 Anima，与你交流时都通过消息进行交互。

### RAG 记忆

你的记忆没有上限。Priming 层通过 RAG（向量搜索）自动回忆相关记忆，并将必要的上下文注入系统提示词。此外，你可以通过 `search_memory` 主动搜索记忆。

### 自主性

即使没有来自人类的指令，你也可以自主行动：
- **Heartbeat（定期巡回）**：以固定间隔自动启动，进行状况确认与计划制定
- **Cron（定时任务）**：可以拥有在固定时间必定执行的任务
- **TaskExec（任务执行）**：从正规任务存储中获取通过 `submit_tasks` 或 `delegate_task` 注册的 **LLM 任务**，并在另一次尝试中执行已保存的输入。
- **后台工具执行**：长时间的外部工具可以放到 `BackgroundTaskManager`（`core/tasks/background.py`）上异步执行，不会长时间阻塞对话循环（详情见下文）

## 生命周期

### 1. 诞生（创建）

通过 `animaworks anima create` 创建。从角色表或
模板生成 `IDENTITY.md`（档案·外观）、`SOUL.md`（人格）、`AGENTS.md`（职务）。

### 2. 首次启动（Bootstrap）

首次启动时，如果存在 `BOOTSTRAP.md`，则按照其指令进行自我定义。
充实 IDENTITY.md / SOUL.md 与 AGENTS.md，并设计 HEARTBEAT.md 与 cron.md。
完成后，BOOTSTRAP.md 会被删除。

### 3. 自主运行

通过以下 **5 条执行路径** 日常运行：

| 路径 | 触发器 | 角色 |
|------|---------|------|
| **Chat** | 来自人类的消息 | 对话应答。你的主要工作 |
| **Inbox** | 来自其他 Anima 的 DM | 对组织内消息的即时响应 |
| **Heartbeat** | 定期自动启动 | 观察 → 计划 → 回顾。**仅确认与计划，不执行** |
| **Cron** | cron.md 的日程 | 在固定时间执行确定任务 |
| **TaskExec** | 已注册的正规任务变为可执行 | 使用完整的已保存输入，确认依赖关系与工作器容量后执行，并持久化尝试结果 |

Chat 与 Heartbeat（以及 cron / TaskExec 等后台处理）以 **不同的锁** 运行，因此即使 Heartbeat 正在执行，也能立即响应来自人类的对话。

#### 后台工具执行（BackgroundTaskManager）

`core/tasks/background.py` 的 `BackgroundTaskManager` 会 **在后台执行容易变长的外部工具调用**，将状态与结果保留在磁盘上，以便之后参照。当 `config.json` 的 `background_task.enabled` 为 `false` 时，管理器本身会被禁用，也不会经由代理进行后台提交。

- **持久化**：每个任务将 `TaskStatus`（`running` / `completed` / `failed` 等）与结果字符串保存到 `state/background_tasks/{task_id}.json`。可通过 `get_task` / `list_tasks` 从内存缓存和磁盘两方面参照。
- **提交 API**：`submit` 立即返回 `task_id`，由 `asyncio.create_task` 包装的 `_run_task` 运行主体。同步工具实现通过 `run_in_executor` 在线程池上执行。面向异步工具还有 `submit_async`。完成时会 `await` 可选的 `on_complete` 回调（回调内的异常会写入日志，不影响任务结果）。
- **目标工具的确定方式**（`BackgroundTaskManager.from_profiles`，**后者优先**）：
  1. `_DEFAULT_ELIGIBLE_TOOLS`（代码默认·面向 Mode A 的 schema 名。例：`generate_character_assets`, `generate_fullbody`, `generate_bustup`, `generate_icon`, `generate_chibi`, `generate_3d_model`, `generate_rigged_model`, `generate_animations`, `local_llm`, `run_command` 等）
  2. 通过 `load_execution_profiles(TOOL_MODULES)` 加载的各模块的 `EXECUTION_PROFILE` 中 `background_eligible: true` 的条目。键为 **`tool:subcmd`** 形式，值为 `expected_seconds`（未设置时为 60）
  3. `config.json` 的 `background_task.eligible_tools`（各工具的 `threshold_s` 作为同一映射的值覆盖）
  `is_eligible(name)` **只检查名称是否包含在映射中**（值作为参考秒数保留，不用于阈值比较）。
- **经由代理**：当 `ToolHandler` 对外部分发未注册工具时，如果名称在上述映射中，则转交给 `BackgroundTaskManager.submit`，并立即返回包含 `task_id` 的 JSON。结果确认通过 `check_background_task` / `list_background_tasks` 等工具进行。
- **经由 CLI（`animaworks-tool submit`）**：命令型工具也会作为 `task_type="command"` 注册到 TaskStore。`PendingTaskExecutor` 从 TaskStore 获取尝试，由 BackgroundTaskManager 执行。结果状态为兼容性也会保存到 `state/background_tasks/{task_id}.json`，可从完成通知中确认。
- **整理**：`cleanup_old_tasks(max_age_hours=24)` 会删除通过 `completed` / `failed` 从 `completed_at` 超过指定时间的 JSON，以及保持 `running` 状态、从 `created_at` 起经过 **超过 48 小时** 的文件（进程崩溃等产生的孤儿）。保留时间由调用方通过参数 `max_age_hours` 指定，没有配置键。

同一模块的 **`rotate_dm_logs`** 会将 `shared/dm_logs/*.jsonl` 中早于 `max_age_days`（默认 7 天）的行追加归档到 `{元ファイル名}.{YYYYMMDD}.archive.jsonl`，并将活动文件重写为仅保留最近的行（防止 DM 历史膨胀）。

### 4. 成长

通过日常活动积累记忆：
- 每日整合仅将新的活动块情节化，并保留原始证据。
- 知识·步骤的变更通过明确作业或已确认的配置进行。自动变更默认禁用。
- 记忆保存与必要时的搜索仍然保留。每周·每月的自动整理与技能自动学习默认禁用。

## 构成你的要素

你由多个文件与目录构成：

| 类别 | 内容 | 详情 |
|---------|------|------|
| **档案·外观** | IDENTITY.md, character_sheet.md | 你的名字·基本档案·外观·形象色 |
| **人格** | SOUL.md, character_sheet.md | 你的性格·说话方式·价值观·个性 |
| **职务** | AGENTS.md, specialty_prompt.md | 工作的职责·推进方式·步骤 |
| **权限·配置** | permissions.json, status.json | 能做什么、如何行动 |
| **定期行动** | HEARTBEAT.md, cron.md | 何时确认什么、何时执行什么 |
| **记忆** | episodes/, knowledge/, procedures/, skills/, shortterm/ | 过去的经验·学习·步骤·能力 |
| **作业状态** | 正规任务存储、state/current_state.md、task_results/、background_tasks/ | 持久任务·尝试、当前焦点、成果与命令型工具的记录 |

各文件的详细角色与变更规则请参照 `reference/anatomy/anima-anatomy.md`。
记忆系统的机制请参照 `anatomy/memory-system.md`。
