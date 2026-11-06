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

`core/tasks/background.py` 中的 `BackgroundTaskManager` 会**在后台执行通常耗时较长的外部工具调用**，并将状态和结果保存到磁盘，以便之后查询。`config.json` 的 `background_task.enabled` 为 `false` 时，管理器本身会被禁用，代理也不会通过它提交后台任务。

- **持久化**：每个任务都会将 `TaskStatus`（如 `running` / `completed` / `failed`）和结果字符串保存到 `state/background_tasks/{task_id}.json`。可通过 `get_task` / `list_tasks` 从内存缓存或磁盘中查询。
- **提交 API**：`submit` 会立即返回 `task_id`，由封装在 `asyncio.create_task` 中的 `_run_task` 执行实际任务。同步工具实现会通过 `run_in_executor` 在线程池中运行。也支持面向异步工具的 `submit_async`。任务完成时，会调用可选的 `on_complete` 回调 `await`（回调中的异常会记入日志，不会影响任务结果）。
- **目标工具的判定方式**（`BackgroundTaskManager.from_profiles`，**后面的优先级更高**）：
  1. `_DEFAULT_ELIGIBLE_TOOLS`（代码默认值，面向 Mode A 的 schema 名称。例如：`generate_character_assets`、`generate_fullbody`、`generate_bustup`、`generate_icon`、`generate_chibi`、`generate_3d_model`、`generate_rigged_model`、`generate_animations`、`local_llm`、`run_command` 等）
  2. 通过 `load_execution_profiles(TOOL_MODULES)` 加载的各个模块中，`EXECUTION_PROFILE` 里属于 `background_eligible: true` 的条目。键的格式为 **`tool:subcmd`**，值为 `expected_seconds`（未配置时为 60）
  3. `config.json` 的 `background_task.eligible_tools`（各工具的 `threshold_s` 会覆盖同一映射中的值）
  `is_eligible(name)` 只会检查名称是否包含在映射中（值会作为估算秒数保留，但不会用于阈值比较）。
- **通过代理执行**：`ToolHandler` 通过外部调度分发未注册的工具时，如果名称存在于上述映射中，就会将其交给 `BackgroundTaskManager.submit` 处理，并立即返回包含 `task_id` 的 JSON。可通过 `check_background_task` / `list_background_tasks` 等工具查询结果。
- **通过 CLI 执行（`animaworks-tool submit`）**：命令型工具也会作为 `task_type="command"` 注册到 TaskStore。`PendingTaskExecutor` 从 TaskStore 获取尝试记录，再由 BackgroundTaskManager 执行。为保持兼容，结果状态也会保存到 `state/background_tasks/{task_id}.json` 中，可通过完成通知查询。
- **清理**：`cleanup_old_tasks(max_age_hours=24)` 会删除自 `completed` / `failed` 起在 `completed_at` 中存放时间超过指定时长的 JSON 文件，以及仍处于 `running` 状态、且自 `created_at` 起已超过 **48 小时**的文件（例如进程崩溃后遗留的孤立文件）。保留时长由调用方通过参数 `max_age_hours` 指定，没有对应的配置键。


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
