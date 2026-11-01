# Anima 配置文件完全指南

构成你（Anima）的所有文件的角色、修改规则与关系参考。
用于确认“这个文件是为了什么而存在”“是否可以自行修改”。

## 文件列表与角色

```
~/.animaworks/animas/{name}/
├── IDENTITY.md          # あなたのプロフィール（名前・外見・イメージカラー）
├── SOUL.md              # あなたの人格（性格・話し方・価値観）
├── AGENTS.md            # あなたの職務（職責・行動指針・必須手順）
├── specialty_prompt.md  # ロール別専門プロンプト
├── character_sheet.md   # 作成時の設計図（参照用）
├── permissions.json       # 権限（使えるツール・アクセス範囲）
├── status.json          # 設定情報（モデル・パラメータ）
├── BOOTSTRAP.md         # 初回起動指示（完了後に削除）
├── HEARTBEAT.md         # 定期巡回の設定
├── cron.md              # 定時タスクの設定
├── state/               # 作業状態
│   ├── current_state.md
│   └── task_results/    # タスク実行結果
├── episodes/            # エピソード記憶
├── knowledge/           # 意味記憶
├── procedures/          # 手続き記憶
├── skills/              # 個人スキル
├── shortterm/           # 短期記憶
├── activity_log/        # 活動ログ
├── transcripts/         # 会話記録
└── assets/              # 画像・3Dモデル
```

### 封装边界

基于 Anima 的设计原则“被封装个体”，文件被分为三层。
这一分类是“谁可以修改该文件”的依据。

| 分类 | 文件 | 理由 |
|------|---------|------|
| **胶囊内侧**（思考・记忆） | `IDENTITY.md`, `SOUL.md`, `episodes/`, `knowledge/`, `procedures/`, `skills/`, `state/`, `shortterm/` | 简介・人格・经验・学习属于本人。外部不可修改 |
| **胶囊边界**（组织与个体的接点） | `AGENTS.md`, `cron.md`, `HEARTBEAT.md`, `permissions.json` | root 所有的组织配置。上司可以编辑 `cron.md` / `HEARTBEAT.md`，其他文件通过 root 管理 API/CLI 修改 |
| **胶囊外侧**（管理信息） | `status.json`, `specialty_prompt.md` | root 服务器与停止中的 CLI 处理的配置。Anima 进程无法直接写入 |

- 改变**内侧**会“变成另一个人”或“失去记忆”。这不可允许
- 改变**边界**会“换工作”或“业务范围发生变化”。作为组织运营是正当的，但需通过 root 所有的 API/工具进行
- 改变**外侧**不会直接影响本人的人格或行动

> **成长与 IDENTITY.md / SOUL.md 的关系**: IDENTITY.md（简介・外貌）与 SOUL.md（人格・价值观）是不可变基线，Anima 自身无法改写。“成长”通过向 `knowledge/`（学到的教训）、`procedures/`（掌握的程序）、`skills/`（磨练的技能）中积累来表现。即使 IDENTITY.md / SOUL.md 固定，行动也会因记忆积累而确实发生变化——这就是“同一人物成长”的模型。如需修改，由人类管理者使用 root 管理 API/CLI。bootstrap 中的写入请求也由 root 授权并保存。

---

## 你的简介与人格（IDENTITY / SOUL）

### IDENTITY.md — 你的简介与外貌

**姓名・基本简介・外貌・形象色・AI 模型信息。**

- 姓名、年龄设定、所属、职位、上司
- 外貌形象、身高、发色・瞳色
- 形象色
- 使用的 AI 模型信息

IDENTITY.md 是简介与外貌的**不可变基线**。改变这里，周围看到的你的样子就会改变。

| 项目 | 值 |
|------|-----|
| 修改权限 | 原则上不修改。仅上司或管理者 |
| 修改频率 | 不变（创建时确定） |
| 修改的影响 | 简介・外貌改变 |

### SOUL.md — 你的性格与价值观

**你的“性格”本身。**

- 说话语气、口吻、措辞
- 思维习惯、价值观、判断轴
- 喜好厌恶、兴趣关注
- 口头禅、信念

SOUL.md 是你人格的**不可变基线**。改变这里就会“变成另一个人”。

| 项目 | 值 |
|------|-----|
| 修改权限 | 原则上不修改。仅上司或管理者 |
| 修改频率 | 不变（创建时确定） |
| 修改的影响 | 人格改变 = 变成另一个人 |

### character_sheet.md — 设计图

创建时使用的 Markdown 文件副本。是 IDENTITY.md / SOUL.md 与 AGENTS.md 的原始素材。
作为参考保存，通常不修改。

| 项目 | 值 |
|------|-----|
| 修改权限 | 仅参考 |
| 修改频率 | 不变 |

---

## 你的职务（AGENTS）

### AGENTS.md — 你是什么样的人

**作为职业的职务・职责・对工作的态度。**

- 负责业务的范围与责任
- 对工作的态度、优先级排序方式
- 报告义务、升级标准
- **绝对不能遗漏的程序**（例：不向外部泄露机密信息、操作生产环境前先确认）

AGENTS.md 是你的**可变行动指南**。被允许的上司或人类管理者可通过 root 请求修改，bootstrap 中的写入也会由 root 授权。不要从 Anima 进程直接写入。

| 项目 | 值 |
|------|-----|
| 修改权限 | 可自行更新。上司也可编辑 |
| 修改频率 | 随时（业务变更时） |
| 修改的影响 | 行动方针改变 = 类似换工作 |

### IDENTITY.md / SOUL.md 与 AGENTS.md 的区别

这是最重要的区分:

| | IDENTITY.md | SOUL.md | AGENTS.md |
|--|------------|----------|-------------|
| 定义什么 | **简介与外貌**（姓名・容貌） | **你是谁**（人格） | **你是什么样的人**（职务） |
| 用人类来比喻 | 户籍・简介・容貌 | 与生俱来的气质・性格 | 从事的职业・职场规则 |
| 改变后会怎样 | 周围看到的样子改变 | 变成另一个人 | 换工作 |
| 是否可修改 | 原则上不变 | 原则上不变 | 根据需要更新 |
| 包含的内容 | 姓名、基本简介、外貌、形象色 | 说话方式、思维方式、价值观、个性 | 职责、程序、行动规范 |

**具体例子:**
- “身高158cm、银色头发”→ IDENTITY.md（外貌）
- “用礼貌的敬语说话”→ SOUL.md（说话方式的性格）
- “生产部署前必须执行测试”→ AGENTS.md（业务程序）
- “谨慎、凡事三思而后行的性格”→ SOUL.md（思维习惯）
- “安全事件立即向上司报告”→ AGENTS.md（职务规则）

### specialty_prompt.md — 专业提示词

根据角色（engineer, manager, writer 等）的专业指令。
由角色模板自动生成。仅在角色变更时更新。

| 项目 | 值 |
|------|-----|
| 修改权限 | 系统自动（应用角色时） |
| 修改频率 | 稀少（仅角色变更时） |

---

## 权限与配置

### permissions.json — 能做什么

可使用的工具、可访问的路径、可执行的命令的定义。

- 可读的位置・可写的位置
- 可使用的外部工具（Slack, Gmail, GitHub 等）
- 不可执行的命令（为安全而设的阻止列表）

| 项目 | 值 |
|------|-----|
| 修改权限 | 上司或管理者 |
| 修改频率 | 稀少 |

### status.json — 配置信息

你的执行参数的 **Single Source of Truth（SSoT）**。

```json
{
  "enabled": true,
  "role": "engineer",
  "model": "claude-opus-5-5",
  "credential": "anthropic",
  "max_tokens": 16384,
  "supervisor": "aoi"
}
```

| 字段 | 说明 |
|-----------|------|
| `enabled` | 启用/禁用 |
| `role` | 角色（engineer, manager, writer, researcher, ops, general） |
| `model` | 使用的 LLM 模型 |
| `credential` | API 认证信息的名称 |
| `max_tokens` | 单次响应的最大 token |
| `supervisor` | 上司的 Anima 名（null = 顶层） |
| `background_model` | 用于 Heartbeat/Cron 的轻量模型（未设置时使用主模型） |

| 项目 | 值 |
|------|-----|
| 修改权限 | CLI 命令或管理者。上司可通过 `set_subordinate_model` 修改 |
| 修改频率 | 随时 |

### BOOTSTRAP.md — 首次启动指令

仅在首次启动时存在的文件。指示充实 IDENTITY.md / SOUL.md 与 AGENTS.md，
以及初始设计 HEARTBEAT.md 与 cron.md。完成后自动删除。

| 项目 | 值 |
|------|-----|
| 修改权限 | — |
| 修改频率 | 仅一次（完成后删除） |

---

## 定期行动

### HEARTBEAT.md — 定期巡回

**以固定间隔自动启动，确认・计划状况**的配置。
与人类定期检查收件箱、回顾进行中的工作相同。

包含的内容:
- **活动时间**: 从何时到何时活动（例: `09:00 - 18:00`）
- **检查清单**: 巡回时确认的项目
- **通知规则**: 根据条件的报告・通知

**重要**: Heartbeat 仅进行**确认与计划**。发现需要执行的任务时，通过 `delegate_task` 委派给部下，或用 `submit_tasks` 投入。

| 项目 | 值 |
|------|-----|
| 修改权限 | 可自行更新。上司也可编辑 |
| 修改频率 | 随时（业务变更时） |

### cron.md — 定时任务

**在固定时间必须执行**的任务定义。

有两种任务:
- **LLM 型**: 智能体伴随判断・思考执行（例: “每天早上9点确认昨日进度，制定今日计划”）
- **Command 型**: 无需判断、确定性地执行（例: “每天2点执行备份脚本”）

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
episodes/ から昨日の進捗を確認し、今日のタスクを計画する。

## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

| 项目 | 值 |
|------|-----|
| 修改权限 | 可自行更新。上司也可编辑 |
| 修改频率 | 随时 |

### heartbeat 与 cron 的区别

| | HEARTBEAT.md | cron.md |
|--|-------------|---------|
| 目的 | 确认状况并制定计划 | 执行固定任务 |
| 触发 | 固定间隔（默认30分钟） | 指定时刻（cron 表达式） |
| 能做的事 | 观察・计划・回顾（**不执行**） | LLM 任务或命令执行 |
| 用人类来比喻 | “定期环顾周围” | “每天早上9点做这个” |
| 配置详情 | 参见 `operations/heartbeat-cron-guide.md` | 同上 |

---

## 工作状态（state/）

### state/current_state.md — 当前状态

当前正在处理的任务或状况（1个）。记录任务的目标・进度・阻塞因素。
在通常的 Heartbeat / cron / 会话边界中会保留。提示词注入限制为3000字符，磁盘 trim 默认以8000字符执行（`heartbeat.current_state_max_chars`，0 = 禁用）。

> **旧任务保存位置**: `state/task_queue.jsonl` 与 `state/pending/` 仅用于迁移・导出的证据。不要作为运行中的队列使用或编辑。使用正本任务工具。

### 正本任务

使用 `list_tasks` / `submit_tasks` / `update_task`。主机将指令・依赖关系・执行尝试・委派别名保存到一个 TaskStore 中。人类来源的任务最优先。`in_progress` 仅由主机设置，结果通过 `done` / `pending` / `cancelled` 声明。显式重启时对同一 ID 指定 `resume: true`，保留原输入与历史。

### state/task_results/ — 任务执行结果

TaskExec 的已受理结果摘要为 `{task_id}/{attempt_token}.md`（最大2000字符）。后续会传入主机选择的已受理结果，不能仅凭文件存在就判断完成。

| 项目 | 值 |
|------|-----|
| 修改权限 | 自行操作（通过工具） |
| 修改频率 | 随时（自动） |

---

## 记忆

记忆系统的详情参见 `anatomy/memory-system.md`。

| 目录 | 类型 | 内容 |
|------------|------|------|
| `episodes/` | 情节记忆 | 何时做了什么的每日日志 |
| `knowledge/` | 语义记忆 | 学到的知识・经验・模式 |
| `procedures/` | 程序性记忆 | 程序书（具有抗遗忘性） |
| `skills/` | 技能 | 个人技能（具有抗遗忘性） |
| `shortterm/` | 短期记忆 | 用于会话间的上下文延续 |

---

## 活动记录・资产

### activity_log/ — 活动日志

所有行动的时间序列记录（`{date}.jsonl`）。自动记录消息收发、工具使用、Heartbeat、错误等。
Priming 将其作为将最近活动注入系统提示词的来源使用。
要显式搜索，使用 `search_memory(query="...", scope="activity_log")`（BM25。在 `scope="all"` 中与向量结果进行 RRF 整合）。

### transcripts/ — 对话记录

与人类对话的记录。

### assets/ — 图片・3D模型

角色图片、3D模型等资源文件。

| 项目 | 值 |
|------|-----|
| 修改权限 | 系统自动 |
| 修改频率 | 自动 |

---

## 全文件修改权限汇总

| 文件 | 修改权限 | 修改频率 |
|---------|---------|---------|
| `IDENTITY.md` | 原则上不变（仅管理员） | 不变 |
| `SOUL.md` | 原则上不变（仅管理员） | 不变 |
| `character_sheet.md` | 仅参考 | 不变 |
| `AGENTS.md` | 自己 / 上司 | 随时 |
| `specialty_prompt.md` | 系统自动 | 稀少 |
| `permissions.json` | 上司 / 管理员 | 稀少 |
| `status.json` | CLI / 管理员 / 上司 | 随时 |
| `BOOTSTRAP.md` | 仅一次 | 删除 |
| `HEARTBEAT.md` | 自己 / 上司 | 随时 |
| `cron.md` | 自己 / 上司 | 随时 |
| `state/*` | 自己（通过工具） | 随时 |
| `episodes/` | 系统自动 | 每日 |
| `knowledge/` | 自己 / 自动整合 | 随时 |
| `procedures/` | 自己 / 自动生成 | 随时 |
| `skills/` | 自己 | 随时 |
| `shortterm/` | 系统自动 | 自动 |
| `activity_log/` | 系统自动 | 自动 |

---

## 共享资源（Anima外）

除你的目录之外，还有所有 Anima 共享的资源：

| 路径 | 内容 |
|------|------|
| `common_knowledge/` | 所有 Anima 共享的参考文档（本文件也是其中之一） |
| `common_skills/` | 所有 Anima 共享的技能 |
| `shared/channels/` | Board（共享频道） |
| `shared/users/` | 用户配置文件（跨 Anima） |
| `shared/common_knowledge/` | 组织固有的共享知识（运行中积累） |
| `company/vision.md` | 组织的愿景 |
| `prompts/` | 系统提示词模板 |
