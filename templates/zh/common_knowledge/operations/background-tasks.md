# 后台任务执行指南

## 概述

部分外部工具（图像生成、3D 模型生成、本地 LLM 推理、语音转写等）
运行需要几分钟到几十分钟。如果直接运行这些工具，运行期间会一直持有锁，
导致消息接收和 heartbeat 停止。

使用 `animaworks-tool submit` 可以在后台运行任务，
而自己则能立即转去处理下一项工作。

`animaworks-tool submit` 会作为 `task_type="command"` 注册到 TaskStore，
Anima 子进程中的 **PendingTaskExecutor**（`core/tasks/pending_executor.py`）会获取执行尝试。
**BackgroundTaskManager**（`core/tasks/background.py`）会在后台运行工具，
执行尝试保存在 TaskStore 中，兼容用途的结果 JSON 则保存在 `state/background_tasks/{task_id}.json` 中。

## 何时使用 submit

### 必须使用 submit 的工具

工具指南（系统提示词）中带有 ⚠ 标记的子命令：

- `image_gen pipeline` / `fullbody` / `bustup` / `icon` / `chibi` / `3d` / `rigging` / `animations`
- `local_llm generate` / `chat`
- `transcribe audio`（子命令名称为 `audio`）

各工具 `EXECUTION_PROFILE` 中属于 `background_eligible: true` 的项目，会通过配置文件登记为后台运行候选项
（例如：`chatwork sync` / `download` 等）。
操作方针仍应优先遵循 **⚠ 标记**。

### 无需 submit 的工具

运行时间较短的工具（通常不到几十秒）：

- `web_search`, `x_search`
- `slack`, `chatwork`, `gmail`（常规操作）
- `github`, `aws_collector`

### 判断标准

- 带 ⚠ 标记 → 必须使用 submit
- 不带 ⚠ 标记 → 直接运行（`animaworks-tool submit` 运行时，如果配置文件将其标为“短时间”，stderr 中可能会出现警告）

## 用法

### 基本语法

```bash
animaworks-tool submit <ツール名> <サブコマンド> [引数...]
```

### 运行示例

```bash
# 3Dモデル生成（Meshy API 等）
animaworks-tool submit image_gen 3d assets/avatar_chibi.png

# キャラクター画像一括生成（全ステップ）
animaworks-tool submit image_gen pipeline "1girl, black hair, ..." --negative "lowres, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR

# ローカルLLM推論（Ollama）
animaworks-tool submit local_llm generate "要約してください: ..."

# 音声文字起こし（Whisper 等）— サブコマンドは audio
animaworks-tool submit transcribe audio "/path/to/audio.wav" --language ja
```

### 返回值

submit 会立即将 JSON 输出到标准输出后退出（`task_id` 为 12 位 hex）：

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "submitted",
  "tool": "image_gen",
  "subcommand": "3d",
  "message": "バックグラウンドタスクを投入しました。完了時にinboxに通知されます。(task_id: a1b2c3d4e5f6)"
}
```

## 接收结果

1. submit 会将包含工具名称、参数和 Anima 的 `task_type="command"` 输入注册到 TaskStore。PendingTaskExecutor 会获取执行尝试，并在会话锁之外请求 BackgroundTaskManager 执行。
2. TaskStore 会记录执行尝试和结果引用。`state/background_tasks/{task_id}.json` 会使用相同的 task_id 记录 `running` → `completed` / `failed`，以兼容旧版查询 API。
3. 完成时，`_on_background_task_complete` 会在 **`state/background_notifications/{task_id}.md`** 中写入 Markdown 通知。
4. 下一次 **heartbeat** 时，`drain_background_notifications()` 会读取并删除相应的 `.md`，并将其注入上下文。
5. 如果 Web UI 的 WebSocket 或 `call_human` 系列的人类通知已启用，完成时也可能同时通过这些渠道通知。

可以使用工具 **`list_background_tasks`** 查看合并内存和磁盘数据后的列表，并使用 **`check_background_task`** 查询指定 `task_id` 的状态（两者分别对应 `BackgroundTaskManager` 的 `list_tasks` / `get_task`）。

## 失败时的处理

- 如果通知中写有“失败”：
  1. 检查错误内容
  2. 确定原因（未配置 API 密钥、超时、参数错误等）
  3. 修正后重新 submit
  4. 如果无法解决，向上司报告

- 进程异常退出时，检查 TaskStore 中执行尝试的所有者，并将已确认所有者已结束的任务记录为 pending（需要确认）。由于可能已经产生副作用，因此不会自动重新运行。即使结果 JSON 仍处于 `running` 状态，也不要将其视为已完成；应先确认实际状态，再明确恢复运行。

## 常见错误

### 直接运行

```bash
# 悪い例: 直接実行 → 長時間ロックされうる
animaworks-tool image_gen 3d assets/avatar_chibi.png -j

# 良い例: submit で非同期実行
animaworks-tool submit image_gen 3d assets/avatar_chibi.png
```

如果已经直接运行，只能等待任务完成。
下次务必使用 submit。

### 省略 transcribe 子命令

```bash
# 悪い例: audio サブコマンドがないと意図したプロファイル判定にならない
animaworks-tool submit transcribe "/path/to/audio.wav"

# 良い例
animaworks-tool submit transcribe audio "/path/to/audio.wav"
```

### submit 后一直等待结果

submit 后立即转去处理下一项工作。
结果会通过面向 heartbeat 的通知文件导入，因此无需轮询或等待。

## 技术机制（参考）

### BackgroundTaskManager（`core/tasks/background.py`）

- **作用**：将长时间运行的工具调用作为 `asyncio` 任务在后台执行，并在完成或失败时 `on_complete`（可选的异步回调）`await`。在构造函数中 `state/background_tasks/` `mkdir(parents=True)`。
- **同步工具**：`submit(tool_name, tool_args, execute_fn, task_id=None)` → 通过 `run_in_executor(None, ...)` 在线程池中执行 `execute_fn(name, args) -> str | None`。如果省略 `task_id`，则会创建它；对于 CLI command task，传入与 TaskStore 相同的 ID。
- **异步工具**：`submit_async`（相同签名下 `execute_fn` 为 `Awaitable[str]`）→ 在事件循环上 `await execute_fn(...)`。
- **调度**：用 `asyncio.create_task(..., name=f"bg-{task_id}")` 包装。完成时，从 `_async_tasks` 中移除相应条目。
- **持久化**：每次更改后，通过 `_save_task` 将 `state/background_tasks/{task_id}.json` 写入 `to_dict()`（`ensure_ascii=False`、`indent=2`）。遇到损坏的 JSON 时，通过 `_load_task` 记录警告日志，然后 `None`。
- **查询**：`get_task` 优先从内存中查询，若不存在则查询磁盘。`list_tasks(status=...)` 与磁盘上的 `*.json` 合并，并按 `created_at` 降序排列。`active_count` 是内存中的 `RUNNING` 数量。
- **`on_complete`**：即使回调中发生异常，任务的完成/失败状态仍会保留；失败只会记录到日志中。
- **具备资格的工具名称**（`is_eligible`）通过合并以下 **3 层**得出（后者优先）。键直接用于字典查找（可能同时存在 Mode A 的架构名称 `generate_3d_model` 和 Mode S 提交用的 `image_gen:3d`）：
  1. 代码内默认值 `_DEFAULT_ELIGIBLE_TOOLS`（值为参考秒数；当前键）：
     `generate_character_assets`、`generate_fullbody`、`generate_bustup`、`generate_icon`、`generate_chibi`、`generate_3d_model`、`generate_rigged_model`、`generate_animations`（各 30 秒）、`local_llm` / `run_command`（各 60 秒）
  2. 通过 `BackgroundTaskManager.from_profiles`，从各模块的 `EXECUTION_PROFILE` 中提取 `background_eligible: true` 的子命令（`core.integrations._base.get_eligible_tools_from_profiles`）。键为 `"{tool_name}:{subcmd}"`，秒数为 `expected_seconds`（未设置时为 60）
  3. `config.json` 中的 `background_task.eligible_tools` — 对每个键，将 `threshold_s` 覆盖为秒数
- **禁用**：在 `config.json` 中将 `background_task.enabled: false` 设为 `BackgroundTaskManager` 时，不会创建其本身。submit 会登记到 TaskStore，但执行尝试会恢复为 pending，并触发需要确认的通知。
- **清理**：`cleanup_old_tasks(max_age_hours=24)` 会删除以下文件：(1) `completed` / `failed` 中 `completed_at` 超过指定时间的 JSON 文件；(2) `running` 仍为该状态，且 `created_at` 已超过 **48 小时**的文件（例如进程崩溃后遗留的孤立文件）。返回值为删除数量。保留时长由调用方通过 `max_age_hours` 指定，没有对应的 `config.json` 配置键。

### 命令型任务（`animaworks-tool submit`）

1. `animaworks-tool submit` 会将包含 `task_type="command"`、工具名称、参数、Anima 和 `task_id` 的输入以原子方式注册到 TaskStore（必须提供 `ANIMAWORKS_ANIMA_DIR`）。
2. TaskStore watcher 使用现有的 claim / attempt 机制获取任务。`BackgroundTaskManager` 也会将原始的 `task_id` 用作结果 JSON。
3. 命令型任务会在内部将 `animaworks-tool … -j` 作为 **子进程**启动。**每次运行的墙钟超时为 1800 秒（30 分钟）**（`pending_executor._PENDING_TASK_SUBPROCESS_TIMEOUT`）。
4. stdout 或错误会保存到 `state/task_results/{task_id}/{attempt_token}.md`，结果引用则记录在 TaskStore 的执行尝试中。状态、参数和执行尝试以 TaskStore 为准。
5. BackgroundTaskManager 会在 `state/background_tasks/{task_id}.json` 中保存 `running` → `completed` / `failed`，`_on_background_task_complete` 会在 `state/background_notifications/{task_id}.md` 中写入通知。heartbeat 会导入该通知。

### 旧命令描述符迁移

新运行时不会监控以前的 `state/background_tasks/pending/*.json`。停止期间运行 `animaworks task-store migrate --anima NAME --backup PATH` 时，会验证未处理的顶层描述符并将其导入 TaskStore。原文件会作为审计证据保留。`processing/` 和 `failed/` 不会自动重新运行或迁移，而是连同警告一起保留；因此，操作人员应在确认副作用后再作判断。

### LLM 型任务（规范任务存储）

1. `submit_tasks` / `delegate_task` 会以原子方式注册完整的指令、上下文、完成条件、约束、模型和依赖关系。
2. watcher 会在同一个 transaction 中获取可运行的任务并创建执行尝试。它会检查依赖项和已配置的工作线程容量；非并行限制仅在同一批次内生效。
3. 如果没有 `done` / `cancelled` 声明就结束执行尝试，任务会恢复为 pending，但不会自动重新运行。应先确认副作用，必要时通过 `submit_tasks(batch_id="resume", tasks=[{"task_id":"ID","resume":true}])` 明确恢复运行。
4. 结果有时可以通过 `state/task_results/{task_id}.md` 查阅，但状态、输入和执行尝试以规范存储为准。完成状态必须通过 `update_task` 声明，不要根据 LLM 的响应或文件是否存在来推断。
5. 需要确认和完成通知会持久化，不依赖定期 heartbeat 投递。不会从 DM 重建执行输入。

请使用任务工具，不要直接编辑 SQLite 或旧版任务文件。这套机制与上述命令型管道不同。

### 文件生命周期

**命令型**（`animaworks-tool submit`）：

```
TaskStore の command 入力 → claim / attempt
  → 成功: done | 失敗・中断: pending（要確認。自動再実行なし）
state/task_results/{task_id}/{attempt_token}.md  # attempt output / error
state/background_tasks/{task_id}.json           # 互換用の running → completed / failed 結果
```

**LLM 型**（`submit_tasks` / `delegate_task`）：

```
保存済み入力 → 実行可能pending → 取得済み試行（in_progress）
  → done/cancelled宣言、またはpending + 永続的な要確認通知
  → 明示resumeで保存済み入力を使う新しい試行
```

命令型和 LLM 型采用不同的输入格式，但两者都通过 TaskStore 的 claim / attempt 机制执行。旧版 LLM 的 JSONL 和 `state/pending/` 描述符用于迁移或导出，并不代表执行启动信号。
