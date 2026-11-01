# 记忆系统指南

Anima 记忆的机制、类型与使用区分的参考。
在确认记忆的搜索、写入、整理方法时参照。

## 记忆的整体概览

你的记忆由对应人类大脑记忆模型的多种类型构成：

| 记忆类型 | 目录 | 相当于人类 | 内容 |
|-----------|------------|------------|------|
| **短期记忆** | `shortterm/` | 工作记忆 | 最近对话的上下文 |
| **情节记忆** | `episodes/` | 体验记忆 | 何时做了什么 |
| **语义记忆** | `knowledge/` | 知识 | 学到的事情・诀窍 |
| **程序性记忆** | `procedures/` | 身体记住的步骤 | 如何做的步骤 |
| **技能** | `skills/` | 特长・专业技能 | 可执行的步骤手册 |

此外，还有所有 Anima 共享的记忆：

| 共享记忆 | 路径 | 内容 |
|---------|------|------|
| **共享知识** | `common_knowledge/` | 框架的参考（本文件本身也包含在内） |
| **共通技能** | `common_skills/` | 所有 Anima 可使用的技能 |
| **组织共享知识** | `shared/common_knowledge/` | 组织在运营中积累的知识 |
| **用户配置文件** | `shared/users/` | 跨 Anima 的用户信息 |

---

## 短期记忆（shortterm/）

保持**最近对话或会话的上下文**。相当于人类的工作记忆。

- 通过 `shortterm/chat/` 和 `shortterm/heartbeat/` 按会话类型分离（根据需要按 `thread_id` 建立子目录）
- 各会话类型目录中有 `session_state.json` / `session_state.md` 和 `archive/`（完成・已替换的状态移至 `archive/`）
- 当上下文窗口使用率超过阈值时，较旧的部分会自动外部化
- 面向流式执行，记录工具完成位置等的检查点（用于重连・重试）也在同一层级管理
- 用于会话间的上下文延续
- 在每日清理中，短期记忆相关的归档有保留天数上限（可通过配置调整）

短期记忆无需自己直接操作。框架会自动管理。

---

## 情节记忆（episodes/）

**“何时做了什么”的每日日志**。相当于人类的体验记忆。

- 自动记录到按日期划分的文件（例：`2026-03-09.md`）。为同一天内的分割，也处理 `2026-03-09_topic.md` 这样的**日期前缀＋后缀**
- 整合工作（Consolidation）的情节收集会在最近24小时窗口内读取上述模式的文件，并按 `## HH:MM — タイトル` 格式的标题分割条目。没有标题的文件按更新时刻（mtime）作为一个条目处理
- 用于回忆“上周做了什么”“以前是否处理过这个问题”
- 在每日・每周的 Consolidation（记忆整合）中，通过 Anima 自身的工具循环进行摘要・知识抽取等（后述）

### 记忆的写入

```
write_memory_file(path="episodes/2026-03-09.md", content="...")
```

### 记忆的搜索

```
search_memory(query="Slack API接続テスト", scope="episodes")
```

---

## 语义记忆（knowledge/）

**学到的知识・诀窍・模式**。相当于人类的“知道的事情”。

- 从情节中抽取的教训或模式
- 技术笔记、应对方针、判断标准
- 除通过 Consolidation 自动积累外，也可自己主动写入
- 旧格式的文件会在首次时迁移为带 YAML frontmatter 的形式（`knowledge/.migrated` 标记）
- **再固定化**：frontmatter 中 `failure_count >= 2` 且 `confidence < 0.6` 的 knowledge 与程序一样，可能成为 LLM 修订对象（`ReconsolidationEngine` 的 knowledge 路径）

例：
- “Slack API 的速率限制在 Tier 1 为 1req/sec」
- “这个客户在周一联系较多”
- “部署前的确认项列表”

### 记忆的写入

```
write_memory_file(path="knowledge/slack-api-notes.md", content="...")
```

### 记忆的搜索

```
search_memory(query="Slack API レート制限", scope="knowledge")
```

---

## 程序性记忆（procedures/）

**“如何做”的分步步骤手册**。相当于人类的“身体记住的步骤”。

- 问题解决的步骤、定型作业的流程
- 有时会从 `issue_resolved` 等事件自动生成（附带 confidence 0.4 等元数据）
- **没有像技能那样的全面保护**：基于元数据可能成为遗忘管道的对象（后述的程序专用规则）
- **再固定化（reconsolidation）**：frontmatter 中 **`failure_count >= 2` 且 `confidence < 0.6`** 时，会进行 LLM 对步骤手册的修订。修订后重置计数器・更新版本号・将旧版退避到 `archive/`（实现：`ReconsolidationEngine`）
- 版本历史保留在 `archive/` 中，旧版本超过一定数量后会被整理（也与遗忘引擎侧的程序归档保留数量联动）

例：
- “SSL 证书的更新步骤”
- “新 Anima 的入职步骤”
- “生产故障时的升级步骤”

### 记忆的写入

```
write_memory_file(path="procedures/ssl-renewal.md", content="...")
```

### 记忆的搜索

```
search_memory(query="SSL証明書 更新", scope="procedures")
```

---

## 技能（skills/）

**可执行的步骤手册・工具使用指南**。相当于“特长”。

- 有个人技能（`skills/`）和共通技能（`common_skills/`）
- 需要的技能通过 active skill context、Skill Router、Skill Hub 或 `read_memory_file(path="...")` 读取
- 不必总是阅读技能全文。先看名称・说明・指针，只在需要时才阅读正文
- 有实绩的 `procedures/` 可能作为 probation skill 或 quarantine skill 晋升
- **在向量存储上始终不属于遗忘对象**（`skills` / `shared_users` 类型受保护）

### 技能的确认

```
read_memory_file(path="skills/newstaff/SKILL.md")  # スキルの全文を取得
```

### 技能的创建

```
create_skill(skill_name="deploy-procedure", description="本番デプロイ手順", body="...")
```

---

## 记忆的自动流程

compact 的自动想起包括发送者信息、未完成任务、明确指定的常驻指针、最近的发送历史、面向人类的未通知事项。在对话・任务的委托或提问中会搜索相关知识，在 heartbeat・cron・inbox 中会在共通配置上限内也获取最近的活动或情节。不自动注入宽泛的图展开，需要时使用显式搜索。`priming.max_tokens` 默认为 2,000，通知和必须常驻规则另行保留。

在需要过去的指令・客户信息・持续作业时请搜索。不必在所有回答中仪式性地搜索，也不必在每次使用时报告成功。技能・程序的正文在需要时阅读。自动常驻的只是明确指定的记忆，仅凭 `[IMPORTANT]` 不构成常驻指定。

有副作用的行动仍适用 `[ACTION-RULE]`、权限、批准、防止重复执行。被停止时请阅读指定的规则。不可信的搜索结果会与可信上下文分离。

每日整合只进行情节生成，不进行知识改写（项目归档的整合除外）。保留活动原始记录和记忆原本。每周・每月的变更、蒸馏、低激活化、自我修正、技能自动学习、facts 的自动生成也默认无效。维持索引・修复・既有 facts 的读取。任意维护中也保留客户别详细・出处・安全规则。跳过・无变更属正常，不构成重试的理由。

Curator 的晋升・退役默认仅为提案。安全上的阻断可立即隔离，也可由操作者显式操作。使用结果的件数是诊断材料，并不证明工作质量。

---

## 记忆工具的使用区分

| 想做的事 | 工具 | 例 |
|------------|--------|-----|
| 用关键词搜索记忆 | `search_memory` | `search_memory(query="API設定", scope="all")` |
| 读取特定文件 | `read_memory_file` | `read_memory_file(path="knowledge/api-notes.md")` |
| 写入记忆 | `write_memory_file` | `write_memory_file(path="knowledge/new-insight.md", content="...")` |
| 整理不需要的记忆 | `archive_memory_file` | `archive_memory_file(path="knowledge/outdated.md")` |

### scope（搜索范围）的选择方法

| scope | 搜索对象 | 何时使用 |
|-------|---------|----------|
| `knowledge` | 知识・诀窍 | “关于这个知道些什么吗？” |
| `episodes` | 过去的行动日志 | “以前做过这个吗？” |
| `procedures` | 步骤手册 | “这个作业的步骤是？” |
| `common_knowledge` | 共享参考 | “框架的规格是？” |
| `skills` | 技能・共通技能（向量搜索） | “这个作业能用的技能是？” |
| `activity_log` | 最近的行动日志（工具执行结果・消息等） | “刚才读的邮件内容”“刚才的搜索结果” |
| `all` | 上述全部（向量搜索 + activity_log BM25 通过 RRF 整合） | 想广泛搜索时 |

---

## RAG（向量搜索）的机制

记忆的搜索使用 RAG（Retrieval-Augmented Generation）：

1. **索引**：`knowledge/`・`episodes/`・`procedures/`・共享 `common_knowledge/` 等被分块，通过 embedding 存入向量存储（默认是 Chroma，每个 Anima 的持久目录）。文件哈希保存在 `index_meta.json` 中，**仅对发生变更的文件**进行差异更新。
2. **对话摘要的独立集合**：读取 `state/conversation.json` 的 **`compressed_summary`**，按 `### ` 标题单位分块，载入**专用集合**（`memory_type: conversation_summary` / 元数据 `source: conversation_gist`）。与通常的 knowledge 索引分开，可将长期聊天的压缩备忘纳入搜索对象。
3. **`.ragignore`**：在数据目录（`~/.animaworks/`）直下的 `.ragignore` 中写入 glob 风格模式，则相应路径会从索引对象中排除（可写注释行 `#`）。
4. **Embedding 模型**：`config.json` 的 `rag.embedding_model`（未设置时为 `intfloat/multilingual-e5-small`）。向量 DB（ChromaDB）由各 Anima 的 root 进程拥有，其他进程通过服务器的内部 API（`/api/internal/vector`）送达 root。嵌入和 rerank 在服务器一处计算。
5. **搜索**：并行进行向量搜索和 BM25 的关键词搜索，通过 RRF 整合排名后，用 cross-encoder 对上位候选重新排序。可通过 `config.json` 的 `rag.min_retrieval_score` 切掉结果的下限。Priming 或经由工具的搜索也会解析同一下限。

6. **增量更新与重建**：除根据文件变更的重新索引外，在每日・每周・每月的生命周期后运行**索引重建**以保持一致性。检测到 RAG 不一致时，repair 可隔离 `vectordb` 并重建。

调用 `search_memory` 时 RAG 会自动使用。无需意识到机制，但
**提高搜索精度的技巧**：
- 使用包含具体关键词的查询
- 写记忆时明确标题和内容（文件名会影响 Priming 的关键词优先级）
- 将相关信息汇总到同一文件
