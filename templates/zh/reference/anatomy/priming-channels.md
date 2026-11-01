# Priming 频道技术参考

Priming 在 compact 的单一流程中运行。获取发送者、任务、常驻知识、最近的发送、面向人类的通知，并在满足条件时也搜索相关知识。在 heartbeat / inbox / cron 中，按照共同的上限配置，也获取最近的活动和情节。

`PrimingEngine` 展示了所获取的频道与预算的规格。C0（important_knowledge）是 Channel C 的知识管道内的辅助块。

## 频道列表

| 频道 | 来源 | trust |
|---------|--------|-------|
| A: sender_profile | `shared/users/{sender}/index.md` | medium |
| B: recent_activity | `activity_log/` + shared channels | trusted |
| C: related_knowledge | RAG 向量搜索（knowledge + common_knowledge） | medium / untrusted |
| C0: important_knowledge | `[IMPORTANT]` 带标签的块 | medium |
| E: pending_tasks | TaskStore + task results | trusted |
| F: episodes | RAG 向量搜索（episodes/） | medium |

追加注入:

| 项目 | 来源 | trust |
|------|--------|-------|
| Recent outbound | activity_log（最多3条，`channel_post` / `message_sent`） | trusted |
| Pending human notifications | `human_notify` 事件 | trusted |

技能与程序的正文不会在 Priming 中注入。通过 `read_memory_file` 读取系统提示词中技能目录所示的路径（例如: `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`, `procedures/baz.md`）。

---

## Channel A: sender_profile

注入发送者的用户配置文件。

- **来源**: 直接读取 `shared/users/{sender}/index.md`
- **上限**: `min(400, max_tokens // 4)`
- **发送者不明时**: 跳过

---

## Channel B: recent_activity

注入最近的活动时间线。

- **来源**: `activity_log/{date}.jsonl` + 共享频道的最新投稿

**Priming 注入与显式搜索的区别**: Channel B 是 heartbeat / inbox / cron 的 compact 背景回忆，在共同配置的上限内获取。用关键词广泛查找过去行动日志的用途应使用 `search_memory(scope="activity_log")`。注入与工具搜索是不同路径。

### 按触发器过滤

| 触发器 | 被排除的事件类型 |
|---------|----------------------|
| `heartbeat` / `cron` / `inbox` / `task` | `tool_use`, `tool_result`, `heartbeat_start`, `heartbeat_reflection`, `inbox_processing_start`, `inbox_processing_end` |
| 其他 | `tool_use`, `tool_result`, `memory_write`, `cron_executed`, `heartbeat_start`, `heartbeat_end`, `heartbeat_reflection`, `inbox_processing_start`, `inbox_processing_end` |

---

## Channel C: related_knowledge

通过 RAG 向量搜索注入相关知识。

- **搜索方式**: Dual-query（消息上下文 + 仅关键词）
- **搜索对象**: 个人 `knowledge/` + `shared_common_knowledge` 集合
- **最低分数**: `config.json` 的 `rag.min_retrieval_score`（默认 0.3）

### trust 分离

搜索结果根据块的 `origin` 按 trust 级别分离:

| trust | 对象 | 处理 |
|-------|------|------|
| `medium` | 个人 knowledge、common_knowledge | 优先消耗预算 |
| `untrusted` | 来自外部平台（`origin_chain` 中包含 `external_platform`） | 用剩余预算注入。带 `origin=ORIGIN_EXTERNAL_PLATFORM` 标签 |

---

## Channel C0: important_knowledge

注入带 `[IMPORTANT]` 标签的块的摘要指针。

- **对象**: `knowledge/` 内带 `[IMPORTANT]` 标签的块
- **注入形式**: 摘要指针。详情通过 `read_memory_file` 获取
- **用途**: 回忆重要的业务规则与判断标准

---

## Channel E: pending_tasks

注入任务队列的摘要。

- **上限**: `min(500, max_tokens // 3)`
- **来源**: `TaskQueueManager.format_for_priming()`
- **内容**:
  - `pending` / `in_progress` 任务的列表与摘要
  - 人类创建的任务带 🔴 HIGH 标记
  - 超过30分钟未更新的任务带 ⚠️ STALE 标记
  - 委派任务的状态
  - 来自 `task_results/` 的已完成任务结果

---

## Channel F: episodes

通过 RAG 向量搜索注入相关情节。

- **搜索对象**: `episodes/` 集合
- **最低分数**: 与 Channel C 共通（`rag.min_retrieval_score`）

---

## 预算与配置

没有配置文件选择，Priming 始终使用 compact 的获取路径。`priming.max_tokens` 是回忆预算（默认: 2000），`priming.channel_timeout_seconds` 是每个频道的获取超时（默认: 60秒）。`compact_background_recall` 是在 heartbeat / inbox / cron 中共享的一组上限，可通过 `compact_background_recall_enabled` 一次性禁用。

- A（发送者）的上限是 `min(400, max_tokens // 4)`，E（任务）的上限是 `min(500, max_tokens // 3)`。最近的发送最多3条、250 token。
- C（相关知识）在 chat/task 触发器，或存在 question/request/delegation 意图且有消息时获取。
- B（最近的活动）在 heartbeat / inbox / cron 中于配置上限内获取。F（情节）在同一触发器且有消息、相关获取启用时于配置上限内获取。
- 待处理的人类通知与回忆预算分开处理。

---

## Hebbian LTP（长期增强）

在 Priming 中被搜索、显示的块会由 `record_access(kind="retrieved")` 更新轻量的搜索记录。通过 `read_memory_file` 或 outcome 报告的显式使用会作为 `used` 记录，遗忘保护以此显式使用为基准。
