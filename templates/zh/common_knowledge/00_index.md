# Common Knowledge — 目录・快速指南

AnimaWorks 所有 Anima 共享的参考文档目录。
遇到困难或不清楚步骤时，请通过此文件定位相应文档，
并在 `read_memory_file(path="common_knowledge/...")` 中查看详情。

> 💡 详细的技术参考（配置文件规范・模型配置・认证配置等）已移至 `reference/`。
> 目录: `reference/00_index.md`

---

## ⭐ 首先阅读这里

首次使用 AnimaWorks，或希望梳理整体概况时，请先阅读以下一个文件。
Heartbeat / Cron / 团队设计 / 记忆 / 成本优化的要点汇总在一页中。

| 文件 | 内容 |
|---------|------|
| **`anatomy/essentials.md`** | **AnimaWorks 精华指南** — 一页俯瞰整体概况・5 条执行路径・Heartbeat vs Cron・团队设计・任务流转方式・记忆系统・成本优化 |

阅读后，各主题的详情请从下方目录中查找。

---

## 遇到困难时的快速指南

### 沟通

| 困扰 | 参考位置 |
|---------|--------|
| 不知道如何发送消息 | `reference/communication/messaging-guide.md` |
| 不知道如何使用 Board（共享频道） | `communication/board-guide.md` |
| 不知道如何发出指令・如何报告 | `reference/communication/instruction-patterns.md` / `reference/communication/reporting-guide.md` |
| 想确认委派・完成报告・升级的必填项 | `communication/message-quality-protocol.md` |
| 想确认 Inbox 启动和消息发送的行为 | `communication/sending-limits.md` |
| 不知道如何通知人类 | `communication/call-human-guide.md` |
| 不知道如何配置 Slack 机器人令牌 | `reference/communication/slack-bot-token-guide.md` ※技术参考 |

### 组织・层级

| 困扰 | 参考位置 |
|---------|--------|
| 不知道组织结构・应该联系谁 | `reference/organization/structure.md` ※技术参考 |
| 想确认角色和责任范围 | `reference/organization/roles.md` |
| 不知道层级间的通信规则 | `organization/hierarchy-rules.md` |

### 任务・运用

| 困扰 | 参考位置 |
|---------|--------|
| 不知道任务管理的方法 | `reference/operations/task-management.md` |
| 想使用任务板（面向人类的仪表盘） | `operations/task-board-guide.md` |
| 不知道心跳或 cron 的配置 | `reference/operations/heartbeat-cron-guide.md` |
| 想在发送・投稿・写入记忆前加入确认规则 | `operations/action-rules-guide.md` |
| 不知道如何执行长时间工具 | `operations/background-tasks.md` |
| 不知道工作区的注册・使用方法 | `operations/workspace-guide.md` |
| worktree・临时文件・缓存应该放在哪里 | `operations/scratch-and-worktrees.md` |
| 想确认新技能的创建方法・元数据 | `common_skills/skill-creator/SKILL.md` |
| 想修改项目配置 | `reference/operations/project-setup.md` ※技术参考 |

### 工具・模型・技术

| 困扰 | 参考位置 |
|---------|--------|
| 不知道工具的使用方法・调用方式 | `reference/operations/tool-usage-overview.md` |
| 不知道模型的选择方法・修改方法 | `reference/operations/model-guide.md` ※技术参考 |
| 想更改 Mode S 的认证方式 | `reference/operations/mode-s-auth-guide.md` ※技术参考 |
| 不知道语音聊天的配置・使用方法 | `reference/operations/voice-chat-guide.md` ※技术参考 |

### 理解自身

| 困扰 | 参考位置 |
|---------|--------|
| 想知道 Anima 是什么 | `anatomy/what-is-anima.md` |
| 想知道自己配置文件的角色 | `reference/anatomy/anima-anatomy.md` ※技术参考 |
| 想知道记忆的机制・种类 | `reference/anatomy/memory-system.md` |

### 故障排查

| 困扰 | 参考位置 |
|---------|--------|
| 工具或命令无法使用 / 出现错误 | `reference/troubleshooting/common-issues.md` |
| 任务被阻塞 / 难以判断 | `reference/troubleshooting/escalation-flowchart.md` |
| Gmail 工具的认证配置不顺利 | `reference/troubleshooting/gmail-credential-setup.md` ※技术参考 |

### 安全

| 困扰 | 参考位置 |
|---------|--------|
| 担心外部数据的可靠性 | `security/prompt-injection-awareness.md` |

### 应用示例

| 困扰 | 参考位置 |
|---------|--------|
| 想知道 AnimaWorks 能做什么 | `reference/usecases/usecase-overview.md` |

**不属于上述情况** → 在 `search_memory(query="キーワード", scope="common_knowledge")` 中搜索

---

## 文档列表

### anatomy/ — Anima 的构成要素

| 文件 | 概要 |
|---------|------|
| ⭐ `essentials.md` | **精华指南** — 一页俯瞰 AnimaWorks 的整体概况（执行路径・Heartbeat vs Cron・团队设计・记忆・成本优化） |
| `what-is-anima.md` | Anima 是什么（概念・设计思想・生命周期・执行路径） |
| `anima-anatomy.md` | → 已移至 `reference/anatomy/anima-anatomy.md`。配置文件完全指南 |
| `memory-system.md` | 记忆系统指南（记忆的种类・Priming・Consolidation・Forgetting・工具的使用区分） |

### organization/ — 组织・结构

| 文件 | 概要 |
|---------|------|
| `structure.md` | → 已移至 `reference/organization/structure.md`。组织结构的机制 |
| `roles.md` | 角色和责任范围（顶层 / 中层管理 / 执行 Anima 的职责） |
| `hierarchy-rules.md` | 层级间的规则（通信路径、监督者工具、紧急时的异常） |

### communication/ — 沟通

| 文件 | 概要 |
|---------|------|
| `messaging-guide.md` | 消息收发指南（send_message 的参数、线程管理、回复方针） |
| `board-guide.md` | Board（共享频道）指南（通过 CLI 读取和发布的方法、发布规则） |
| `instruction-patterns.md` | 发布指令的模式集（如何编写明确的指令、委派模式、进度确认） |
| `reporting-guide.md` | 报告与升级处理的方法（报告时机、格式、紧急 vs 定期） |
| `message-quality-protocol.md` | 消息质量协议（委派 4 项、完成报告 3 项、升级处理 4 项的必查清单） |
| `sending-limits.md` | Inbox 文件唤醒、消息汇总、run 内防重复、避免循环 |
| `call-human-guide.md` | 通知人类指南（call_human 的用法、接收回复、通知频道配置） |
| `slack-bot-token-guide.md` | → 已移至 `reference/communication/slack-bot-token-guide.md`。Slack 机器人令牌配置指南 |

### operations/ — 运用・任务管理

| 文件 | 概要 |
|---------|------|
| `project-setup.md` | → 已移至 `reference/operations/project-setup.md`。项目配置方法 |
| `task-management.md` | 任务管理（current_state.md 的使用方法与任务队列、状态迁移、优先级） |
| `task-board-guide.md` | 任务板（面向人类的仪表盘）的机制与运用方法 |
| `heartbeat-cron-guide.md` | 定期执行的配置与运用（心跳的机制、cron 任务定义、自我更新） |
| `action-rules-guide.md` | 动作规则（`[ACTION-RULE]`、`trigger_tools`、发送前确认、必填 `read_memory_file`） |
| `tool-usage-overview.md` | 工具使用概要（S/C/D/G/A/B 模式别工具体系、内部/外部工具、调用方法） |
| `background-tasks.md` | 后台任务执行指南（submit 的使用方法、判断标准、结果的接收方式） |
| `workspace-guide.md` | 工作区指南（概念・注册・在工具中的使用・故障排查） |
| `scratch-and-worktrees.md` | 工作物的放置位置（scratch・共享 worktree・缓存・自动删除） |
| `model-guide.md` | → 已移至 `reference/operations/model-guide.md`。模型选择・配置指南 |
| `mode-s-auth-guide.md` | → 已移至 `reference/operations/mode-s-auth-guide.md`。Mode S 认证模式配置指南 |
| `voice-chat-guide.md` | → 已移至 `reference/operations/voice-chat-guide.md`。语音聊天指南 |

### security/ — 安全

| 文件 | 概要 |
|---------|------|
| `prompt-injection-awareness.md` | 提示词注入防御指南（信任级别、边界标签、untrusted 数据的处理规则） |

### troubleshooting/ — 故障排查

| 文件 | 概要 |
|---------|------|
| `common-issues.md` | 常见问题与处理方法（消息未送达、发送限制、权限、工具、上下文） |
| `escalation-flowchart.md` | 遇到困难时的判断流程图（问题分类、紧急度判定、升级对象） |
| `gmail-credential-setup.md` | → 已移至 `reference/troubleshooting/gmail-credential-setup.md`。Gmail Tool 认证配置指南 |

### usecases/ — 用例指南

| 文件 | 概要 |
|---------|------|
| `usecase-overview.md` | 用例指南概要（AnimaWorks 能做什么・如何开始・全部主题列表） |
| `usecase-communication.md` | 沟通自动化（聊天・邮件监控、升级、定期联络） |
| `usecase-development.md` | 软件开发支持（代码审查、CI/CD监控、Issue 实现、Bug 调查） |
| `usecase-monitoring.md` | 基础设施・服务监控（存活监控、资源监控、SSL 证书、日志分析） |
| `usecase-secretary.md` | 秘书・事务支持（日程管理、联络协调、日报制作、提醒） |
| `usecase-research.md` | 调查・研究・分析（Web 搜索、竞品分析、市场调查、报告制作） |
| `usecase-knowledge.md` | 知识管理・文档整理（步骤书制作、FAQ 构建、经验教训积累） |
| `usecase-customer-support.md` | 客户支持（初步应对、FAQ 自动回答、升级管理） |

---

## 关键词索引

| 关键词 | 参照位置 |
|-----------|--------|
| 基础, 入门, 全貌, essentials, 入门方法, 概要 | `anatomy/essentials.md` |
| 消息, send_message, 发送, 回复, 线程, inbox | `reference/communication/messaging-guide.md` |
| Board, 频道, `animaworks-tool board read`, `animaworks-tool board post` | `communication/board-guide.md` |
| DM 历史, `animaworks-tool board dm-history`, 过往对话 | `communication/board-guide.md` |
| 指令, 委派, 任务请求, delegation | `reference/communication/instruction-patterns.md` |
| 报告, 每日报告, 摘要, 完成报告, 升级处理 | `reference/communication/reporting-guide.md` |
| 质量协议, 必查项目, 验证依据, 完成条件, 委派检查 | `communication/message-quality-protocol.md` |
| Inbox wake, 消息汇总, 防止重复发送, 回复“收到”, 对话循环 | `communication/sending-limits.md` |
| call_human, 人类通知, 联系人类, 通知频道 | `communication/call-human-guide.md` |
| Slack, 机器人令牌, SLACK_BOT_TOKEN, not_in_channel | `reference/communication/slack-bot-token-guide.md` |
| 组织, supervisor, 上司, 部下, 同事 | `reference/organization/structure.md` |
| 角色, 责任, speciality, 专业 | `reference/organization/roles.md` |
| 层级, 通信路径, org_dashboard, ping_subordinate | `organization/hierarchy-rules.md` |
| delegate_task, 任务委派, task_tracker | `organization/hierarchy-rules.md`, `reference/operations/task-management.md` |
| 任务, current_state, pending, 进度, 优先级 | `reference/operations/task-management.md` |
| 任务队列, submit_tasks, update_task, TaskExec, animaworks-tool task list | `reference/operations/task-management.md` |
| 任务看板, 仪表板, 面向人类 | `operations/task-board-guide.md` |
| 配置, config, status.json, SSoT, reload | `reference/operations/project-setup.md` |
| 心跳, heartbeat, 定期检查 | `reference/operations/heartbeat-cron-guide.md` |
| cron, 日程安排, 定时任务 | `reference/operations/heartbeat-cron-guide.md` |
| 工具, animaworks-tool, MCP, skill | `reference/operations/tool-usage-overview.md` |
| 执行模式, S-mode, C-mode, D-mode, G-mode, A-mode, B-mode | `reference/operations/tool-usage-overview.md` |
| 后台, submit, 长时间运行的工具 | `operations/background-tasks.md` |
| 工作区, workspace, 工作目录, working_directory | `operations/workspace-guide.md` |
| scratch, worktree, 临时文件, 缓存, ANIMAWORKS_SCRATCH_DIR | `operations/scratch-and-worktrees.md` |
| 模型, models.json, credential, set-model, 上下文窗口 | `reference/operations/model-guide.md` |
| background_model, 后台模型, 成本优化 | `reference/operations/model-guide.md` |
| Mode S, 认证, 直接调用 API, Bedrock, Vertex AI, Max plan | `reference/operations/mode-s-auth-guide.md` |
| 音频, voice, STT, TTS, VOICEVOX, ElevenLabs | `reference/operations/voice-chat-guide.md` |
| WebSocket, /ws/voice, barge-in, VAD, PTT | `reference/operations/voice-chat-guide.md` |
| Anima, 自己, 构成, 设计, 生命周期 | `anatomy/what-is-anima.md` |
| IDENTITY.md, SOUL.md, AGENTS.md, 人格, 个人资料, 行为准则, 不变, 可变 | `reference/anatomy/anima-anatomy.md` |
| permissions.json, bootstrap, HEARTBEAT.md, cron.md | `reference/anatomy/anima-anatomy.md` |
| 记忆, memory, episodes, knowledge, procedures | `reference/anatomy/memory-system.md` |
| Priming, RAG, Consolidation, Forgetting, 遗忘 | `reference/anatomy/memory-system.md` |
| consolidation, 2-phase, multipass, 错误追踪 | `reference/anatomy/memory-system.md` |
| search_memory, write_memory_file, 记忆搜索 | `reference/anatomy/memory-system.md` |
| skills, 技能搜索, common_skills, search_memory scope="skills" | `reference/anatomy/memory-system.md`, `reference/operations/tool-usage-overview.md` |
| activity_log, BM25, RRF, 最近日志搜索 | `reference/anatomy/memory-system.md`, `reference/troubleshooting/common-issues.md` |
| 提示词注入, trust, untrusted, 边界标签 | `security/prompt-injection-awareness.md` |
| 错误, 问题, 无法运行, 权限, 屏蔽的命令 | `reference/troubleshooting/common-issues.md` |
| 流程图, 判断, 犹豫, 紧急, 安全 | `reference/troubleshooting/escalation-flowchart.md` |
| Gmail, token.json, OAuth, pickle | `reference/troubleshooting/gmail-credential-setup.md` |
| 层级, tiered, T1, T2, T3, T4 | `reference/troubleshooting/common-issues.md` |
| 用例, 应用示例, 能做什么 | `reference/usecases/usecase-overview.md` |

---

## 使用方法

```
# キーワードで検索
search_memory(query="メッセージ 送信", scope="common_knowledge")

# パスを直接指定
read_memory_file(path="reference/communication/messaging-guide.md")

# 技術リファレンスを参照
read_memory_file(path="reference/anatomy/anima-anatomy.md")

# このファイル自体を参照
read_memory_file(path="common_knowledge/00_index.md")
```
