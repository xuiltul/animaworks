# AnimaWorks 精要指南

[IMPORTANT] 用于在一页内把握 AnimaWorks 全貌的整合指南。
涵盖 Heartbeat / Cron / 团队设计 / 记忆 / 成本优化的要点。
首次阅读时，或想梳理概念之间的关系时，应最先参考本文。
各主题的详情请参阅末尾的链接。

---

## 什么是 AnimaWorks

将 AI 智能体不作为“工具”，而是作为 **自律的人格** 来运用的框架。

- 每个 Anima 都拥有固有的人格、记忆和判断标准
- 即使没有人类指令，也会定期自主行动（Heartbeat / Cron）
- 在组织中承担角色，与其他 Anima 或人类协作
- 从经验中学习，积累记忆并成长

---

## 5 条执行路径 — “何时・如何行动”

Anima 通过以下 5 条路径运行。除 Chat 外，全部自动启动。

| 路径 | 何时运行 | 做什么 | 谁使用 |
|------|---------|---------|---------|
| **Chat** | 人类发送消息时 | 对话应答 | 人类 → Anima |
| **Inbox** | 其他 Anima 发来 DM 时 | 对组织内消息的即时应答 | Anima → Anima |
| **Heartbeat** | 定期自动启动（默认 30 分钟） | 观察 → 计划 → 回顾。**不执行** | 自动 |
| **Cron** | cron.md 的调度（例：每天 9:00） | 在固定时间执行确定任务 | 自动 |
| **TaskExec** | 正规任务可执行且依赖项已完成时 | 用已获取保存输入的 LLM 尝试执行 | 通过 submit_tasks 或 delegate_task 注册 |

Chat 和 Heartbeat 以**不同的锁**运行，因此即使在巡回中也能立即应答人类的对话。

→ 详情: `anatomy/what-is-anima.md`

---

## Heartbeat vs Cron — 两种自律行动

两者都是“在没有人类指令的情况下行动”的机制，但目的有根本性不同。

| 观点 | Heartbeat（定期巡回） | Cron（定时任务） |
|------|---------------------|------------------|
| **比喻** | 定期巡视办公室的保安 | 每天早上 9 点送达的报纸投递 |
| **目的** | 状况确认・计划制定・回顾 | 执行既定业务 |
| **是否执行** | **不执行**。发现任务时只是通过 `submit_tasks` 或 `delegate_task` 投入 | **执行**。LLM 型则思考并执行，Command 型则立即执行 |
| **间隔** | 固定间隔（默认 30 分钟，随 Activity Level 变动） | 用 cron 表达式灵活指定（每天 9:00、每周五 17:00 等） |
| **配置文件** | `HEARTBEAT.md`（检查清单） | `cron.md`（任务定义） |
| **典型示例** | 未读消息确认、阻塞项检测、进度回顾 | 早晨业务计划、周报、备份执行 |

**判断迷茫时**：“只确认并判断” → 添加到 Heartbeat 的检查清单。“在固定时间做某事” → 定义为 Cron 任务。

→ 详情: `operations/heartbeat-cron-guide.md`

---

## Cron 的 2 种类型 — LLM 型 vs Command 型

Cron 任务根据是否需要思考分为两种类型。

| 观点 | LLM 型（`type: llm`） | Command 型（`type: command`） |
|------|---------------------|---------------------------|
| **比喻** | “思考今天应该优先做什么” | “每天早上按下这个按钮” |
| **判断** | 有（输出随状况变化） | 无（每次执行相同内容） |
| **API 成本** | 有（LLM 调用） | 命令本身无。但 **follow-up LLM 可能启动**（见下文） |
| **输出** | 不定形（因任务而异） | 确定性（命令的 stdout） |
| **适合的任务** | 计划制定、回顾、文章撰写、记忆整理 | 备份、通知发送、数据获取、健康检查 |

### Command 型的 follow-up LLM（重要）

Command 型虽然机械地执行命令本身，但**当 stdout 有返回值时，默认会启动 LLM 分析结果**（follow-up）。也就是说，并非完全零成本。

```
コマンド実行 → stdout あり？
  → なし → 終了（LLM 不要）
  → あり → skip_pattern にマッチする？
      → マッチ → 終了（LLM スキップ）
      → マッチしない → LLM が起動して結果を解釈・対処判断
```

控制此 follow-up 的选项：
- **`trigger_heartbeat: false`** — 始终跳过 follow-up LLM（当不需要分析结果时）
- **`skip_pattern: <正規表現>`** — 仅当 stdout 匹配时跳过（正常时忽略，异常时让 LLM 判断）

### 区分使用的判断标准

```
「毎回同じことをするだけ？」
  → はい、結果も見なくてよい → Command型 + trigger_heartbeat: false
  → はい、ただし異常時だけ判断が必要 → Command型 + skip_pattern（正常パターン）
  → いいえ（状況に応じて判断が変わる） → LLM型
  → コマンド実行 + 毎回結果の解釈が必要 → LLM型（description にコマンド実行を指示）
```

### 记述示例

```markdown
## 毎朝の業務計画（LLM型）
schedule: 0 9 * * *
type: llm
episodes/ から昨日の進捗を確認し、今日のタスクを計画する。

## バックアップ実行（Command型・follow-up不要）
schedule: 0 2 * * *
type: command
trigger_heartbeat: false
command: /usr/local/bin/backup.sh

## 監視チェック（Command型・正常時はスキップ、異常時はLLMが判断）
schedule: */15 * * * *
type: command
skip_pattern: ^OK$
command: /usr/local/bin/health-check.sh
```

→ 详情: `operations/heartbeat-cron-guide.md`

---


## 任务的流转方式 — submit_tasks vs delegate_task

将任务付诸执行的方法有两种。

| 观点 | `submit_tasks` | `delegate_task` |
|------|---------------|----------------|
| **谁执行** | **自己**的 TaskExec 路径 | **直属部下** |
| **使用场景** | 想异步执行自己应做的任务 | 想委派给部下 |
| **DAG/并行** | 用 `parallel: true` 并行，用 `depends_on` 依赖 | 逐件委派 |
| **进度追踪** | 参照正规任务存储的 `list_tasks` / TaskBoard | 用 `task_tracker` 追踪 |
| **典型示例** | 自己执行 Heartbeat 中发现的任务 | 上司将工作委派给部下 |

**判断流程**:
```
「このタスクは部下がやるべき？」
  → はい、直属の部下がいる → delegate_task
  → いいえ、自分でやる → submit_tasks
  → 部下がいない → submit_tasks
```

→ 详情: `operations/task-management.md`, `anatomy/task-architecture.md`

---

## 团队设计 — 从单人开始并扩展

### 为什么要分离角色

| 理由 | 说明 |
|------|------|
| 防止上下文污染 | 让一个人承担全部工序会导致上下文膨胀、判断精度下降 |
| 结构性保障质量 | 分离执行者与验证者（排除自我审查的盲点） |
| 并行执行 | 独立角色同时执行可提升吞吐量 |
| 深化专业性 | 角色固有的检查清单・记忆使其比通用智能体质量更高 |

### 扩展的阶段

| 规模 | 构成 | 何时使用 |
|------|------|---------|
| **单人** | 1 个 Anima 兼任全部角色 | 小任务、原型、刚开始 |
| **配对** | PdM + Engineer | 中等规模的定型任务 |
| **完整团队** | PdM + Engineer + Reviewer + Tester | 正式项目 |
| **规模化** | PdM + 多个 Engineer + 多个 Reviewer + Tester | 大规模・多模块 |

### 判断的参考

- 失败成本高 → 增加角色分离
- 变成“实现者本人审查” → 分离 Reviewer
- 可并行作业的模块多 → 增加 Engineer

---

## 记忆 — 区分使用 5 种类型

| 记忆类型 | 目录 | 一句话 | 示例 |
|-----------|------------|--------|-----|
| **情节记忆** | `episodes/` | 何时做了什么 | “3/15 做了 Slack API 调查” |
| **语义记忆** | `knowledge/` | 学到的东西 | “Slack API 的速率限制是 100 次/分” |
| **程序性记忆** | `procedures/` | 如何做 | “Gmail 认证的设置步骤” |
| **技能** | `skills/` | 可执行的步骤书 | 图像生成技能、调查技能 |
| **短期记忆** | `shortterm/` | 最近的上下文 | 当前对话的流程 |

**Priming（自动想起）** 在每次对话或巡回时自动想起相关记忆，并将必要的上下文注入系统提示词。此外还可以通过 `search_memory` 主动搜索。

**Consolidation（记忆整合）** 只将新的活动块情节化，在没有变化的重新执行中不生成。知识的自动变更・周次月度整理・技能自动学习默认禁用。记忆保存和必要时的搜索保留。

→ 详情: `anatomy/memory-system.md`

---

## 成本优化 — background_model 与 Activity Level

### background_model

Heartbeat / Cron 可以使用显式配置的 background_model。Inbox 使用主模型，任务固有的模型指定在该任务中优先。不要仅因便宜就自动选择。

| 区分 | 使用模型 | 对象 |
|------|-----------|------|
| foreground | 主模型，或显式指定的任务固有模型 | Chat、Inbox、TaskExec |
| background | 显式配置的 background_model，未配置则用主模型 | Heartbeat、Cron |

配置: `animaworks anima set-background-model {名前} claude-sonnet-5-5`

### Activity Level

可将整体活动频率调整为 10%〜400%。直接影响 Heartbeat 间隔。

| Activity Level | Heartbeat 间隔（基准 30 分钟时） | 用途 |
|---------------|-------------------------------|------|
| 200% | 15 分钟 | 繁忙期・活跃开发 |
| 100%（默认） | 30 分钟 | 常规运营 |
| 50% | 60 分钟 | 低负载・节约成本 |
| 30% | 100 分钟 | 夜间・休息日 |

**Activity Schedule** 可按时间段自动切换（例：9:00-22:00 为 100%，22:00-6:00 为 30%）。

→ 详情: `reference/operations/model-guide.md`, `operations/heartbeat-cron-guide.md`

---

## 组织基础 — 上司・部下・同僚

Anima 的层级由 `status.json` 的 `supervisor` 字段决定。

| 关系 | 定义 | 沟通 |
|------|------|------------------|
| **上司** | 在 `supervisor` 中指定的 Anima | 进度报告（MUST）、问题升级 |
| **部下** | `supervisor` 为自己的 Anima | 用 `delegate_task` 委派，用 `org_dashboard` 监控 |
| **同僚** | 拥有相同 `supervisor` 的 Anima | 可直接联系 |
| **其他部门** | 不属于上述任一 | 经由自己的上司（原则上禁止直接联系） |
| **人类** | 为 `supervisor: null`（顶层）时 | 用 `call_human` 通知 |

→ 详情: `organization/hierarchy-rules.md`, `organization/roles.md`

---

## 遇到困难时的第一步

| 状况 | 要做的事 |
|------|---------|
| 不知道操作方法 | `search_memory(query="キーワード", scope="common_knowledge")` |
| 任务被阻塞 | 参照 `troubleshooting/escalation-flowchart.md` |
| 工具无法运行 | 参照 `troubleshooting/common-issues.md` |
| 不知道做什么 | 确认 current_state.md 和 `list_tasks`。必要时参照已配置的 Heartbeat 检查清单 |
| 判断迷茫 | 用 `send_message(intent="question")` 向上司咨询 |

→ 全部文档目录: `common_knowledge/00_index.md`
