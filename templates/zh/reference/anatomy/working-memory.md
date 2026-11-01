# 工作记忆（state/）技术参考

管理 Anima 工作状态的 `state/` 目录的详细规格。
包含注入提示词的逻辑、大小控制、迁移、锁控制。

---

## state/ 目录结构

```
state/
├── current_state.md          # ワーキングメモリ（自由形式Markdown）
├── task_results/              # TaskExec完了結果
│   └── {task_id}/{attempt_token}.md
├── conversation.json          # 会話状態
├── conversations/             # スレッド別会話ファイル
├── recovery_note.md           # クラッシュ復旧ノート
├── heartbeat_checkpoint.json  # Heartbeatチェックポイント
└── pending_procedures.json    # 保留中の手続き追跡
```

---

## current_state.md

### 角色

Anima 的工作记忆。以自由形式记录“现在正在做什么”“观察到了什么”“有哪些阻塞项”。不是用于任务管理，而是用于情境认知的场所。

任务的追踪由宿主管理的正本 TaskStore 负责。确认使用 `list_tasks`，变更使用任务工具，不要直接编辑 DB 或队列文件。

### 大小控制

| 参数 | 值 | 来源 |
|-----------|-----|-------|
| 显示上限 | 3000 字符 | `_CURRENT_STATE_MAX_CHARS`（builder.py） |
| 磁盘 trim 上限 | 8000 字符（默认） | `heartbeat.current_state_max_chars`（0 = 无效） |
| Inbox 时上限 | 500 字符 | builder.py 内由 `min(_state_max, 500)` |

**会话边界**：

- 通常的 Heartbeat / cron / 对话最终化时保留 `current_state.md`
- 即使会话摘要中包含当前状态，也仅在 `current_state.md` 为空/idle 时写入
- 没有 active 任务的旧 state 可能会被 TaskBoard housekeeping 归档。即使不可见，active 任务也会保护 state

**Heartbeat 时的可选清理**：

1. 如果 `heartbeat.current_state_max_chars` 大于 0，且 Heartbeat 开始前 `current_state.md` 已超过该值，则会将“整理并压缩”的指令注入 Heartbeat 提示词
2. Heartbeat 或 cron 完成后，执行 `_enforce_state_size_limit()`
3. 超过配置上限的部分会作为 `## current_state.md overflow archived` 移动到当天的情节记忆（`episodes/{date}.md`）
4. 保留末尾的配置字符数，并按换行位置调整（如果前 20% 以内有换行，则在该处截断）

### 注入提示词

| 触发器 | 行为 |
|---------|------|
| `chat` | 全文注入（3000 字符上限，应用缩放） |
| `inbox` | 限制为最多 500 字符 |
| `heartbeat` / `cron` | 全文注入（3000 字符上限） |
| `task` | **不注入**（Minimal 层级） |

注入时，如果仅有 `status: idle`，则省略该 section 本身。
否则会使用 `builder/task_in_progress` 模板，以带强调标题的形式注入。

### 锁控制

`core/anima/digital_anima.py` 的 `_state_file_lock`（`asyncio.Lock`）可防止对 `current_state.md` 的并行写入。

`_is_state_file(path)` 仅对 `state/current_state.md` 返回 `True`。通过 `write_memory_file` 写入时，会自动获取针对该文件的锁。

### 路径解析（向后兼容）

当 `read_memory_file` / `write_memory_file` 指定了 `state/current_task.md` 时，会自动解析为 `state/current_state.md`（`handler_memory.py`）。

---

## pending.md（已废弃）

`state/pending.md` 在合并到 `current_state.md` 后会被自动删除。

### 迁移（MemoryManager 初始化时）

1. `state/current_task.md` 存在且 `state/current_state.md` 不存在 → 重命名
2. 两者都存在 → 优先使用 `current_state.md`，并记录警告日志
3. `state/pending.md` 存在且有内容 → 作为 `## Migrated from pending.md` 追加到 `current_state.md` 后删除
4. `state/pending.md` 为空 → 删除

---

## 旧任务文件

`state/task_queue.jsonl` 和 `state/pending/` 仅作为迁移/导出用的证据保留。不是运行中的队列。运维人员应停止旧的写入处理，并在带备份的情况下显式导入后，再启动正本运行时。不要为了恢复而删除、重新投入或捏造文件。

## 任务执行与结果

宿主会一并保存原始指令和任务，获取可执行的工作并记录每次尝试。`in_progress` 由宿主管理。代理在 `update_task` 中声明 `done` / `pending` / `cancelled`。在 `list_tasks(detail=true)` 中确认依赖关系和需要处理的原因。pending 并不意味着重试。原因解决后，通过 `submit_tasks(..., tasks=[{"task_id": "ID", "resume": true}])` 显式重新开始同一任务。

被接受的结果摘要会保存到 `state/task_results/{task_id}/{attempt_token}.md`（最多 2000 字符）。后续会传递宿主选择的已接受结果。不要仅凭旧文件存在就判断为完成。保存原始记录，写入结果，不要伪装成成功的尝试。

长时间命令工具也会作为 `task_type="command"` 注册到 TaskStore。PendingTaskExecutor 获取尝试，BackgroundTaskManager 执行。尝试会记录到 TaskStore，并且为了兼容性会保留 `state/background_tasks/{task_id}.json` 和完成通知。详情见 `operations/background-tasks.md` 和 `operations/task-management.md`。

## read_subordinate_state

当上司调用 `read_subordinate_state(name="部下名")` 时，只会读取部下的 `state/current_state.md`（`pending.md` 不在对象范围内）。
