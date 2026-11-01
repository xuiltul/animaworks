# 后台任务执行指南

## 概述

部分外部工具（图像生成、3D 模型生成、本地 LLM 推理、语音转写等）
执行需要数分钟至数十分钟。若直接执行，执行期间会一直持有锁，
导致消息接收和 heartbeat 停止。

通过使用 `animaworks-tool submit`，可以让任务在后台执行，
自身则立即转入下一项工作。

`animaworks-tool submit` 会以 `task_type="command"` 注册到 TaskStore，
由 Anima 子进程内的 **PendingTaskExecutor**（`core/tasks/pending_executor.py`）获取执行尝试。
**BackgroundTaskManager**（`core/tasks/background.py`）在后台执行工具，
执行尝试保存到 TaskStore，兼容用的结果 JSON 保存到 `state/background_tasks/{task_id}.json`。

## 何时使用 submit

### 必须使用 submit 的工具

工具指南（系统提示词）中带有 ⚠ 标记的子命令:

- `image_gen pipeline` / `fullbody` / `bustup` / `icon` / `chibi` / `3d` / `rigging` / `animations`
- `local_llm generate` / `chat`
- `transcribe audio`（子命令名为 `audio`）

各工具的 `EXECUTION_PROFILE` 中为 `background_eligible: true` 的项，会通过 profile
注册为后台执行的候选（例如: `chatwork sync` / `download` 等）。
运用方针仍应优先依据 **⚠ 标记**。

### 无需 submit 的工具

执行时间较短（以不足数十秒为参考）的工具:

- `web_search`, `x_search`
- `slack`, `chatwork`, `gmail`（常规操作）
- `github`, `aws_collector`

### 判断标准

- 有 ⚠ 标记 → 必须 submit
- 无 ⚠ 标记 → 直接执行（`animaworks-tool submit` 执行时，若 profile 上为“短时间”，可能会向 stderr 输出警告）

## 使用方法

### 基本语法

```bash
animaworks-tool submit <ツール名> <サブコマンド> [引数...]
```

### 执行示例

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

submit 会立即将 JSON 输出到标准输出并结束（`task_id` 为 12 位 hex）:

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

1. submit 会将包含工具名、参数、Anima 的 `task_type="command"` 输入注册到 TaskStore。PendingTaskExecutor 获取执行尝试，并在会话锁之外请求 BackgroundTaskManager 执行。
2. TaskStore 记录执行尝试和结果引用。`state/background_tasks/{task_id}.json` 为兼容传统查询 API，以相同 task_id 记录 `running` → `completed` / `failed`。
3. 完成时 `_on_background_task_complete` 会向 **`state/background_notifications/{task_id}.md`** 写入 Markdown 通知。
4. 在下一次 **heartbeat** 中，`drain_background_notifications()` 会读取并删除该 `.md`，并注入到上下文中。
5. 若 Web UI 的 WebSocket 或 `call_human` 系的人类通知已启用，同一完成时机也可能载入到那边。

可通过工具 **`list_background_tasks`** 查看合并内存与磁盘的列表，通过 **`check_background_task`** 参照 `task_id` 指定的状态（二者均相当于 `BackgroundTaskManager` 的 `list_tasks` / `get_task`）。

## 失败时的应对

- 若通知中记载“失败”:
  1. 确认错误内容
  2. 确定原因（API key 未配置、超时、参数错误等）
  3. 修正后再次 submit
  4. 若无法解决，则向上司报告

- 进程异常终止时，确认 TaskStore 的执行尝试所有者，并将所有者终止已确定的任务记录为 pending（需确认）。由于可能已产生副作用，不自动重新执行。即使结果 JSON 仍停留在 `running`，也不要判断为完成，应确认实际状态后再显式恢复。

## 常见错误

### 直接执行

```bash
# 悪い例: 直接実行 → 長時間ロックされうる
animaworks-tool image_gen 3d assets/avatar_chibi.png -j

# 良い例: submit で非同期実行
animaworks-tool submit image_gen 3d assets/avatar_chibi.png
```

若直接执行，只能等待任务完成。
下次起务必使用 submit。

### transcribe 的子命令省略

```bash
# 悪い例: audio サブコマンドがないと意図したプロファイル判定にならない
animaworks-tool submit transcribe "/path/to/audio.wav"

# 良い例
animaworks-tool submit transcribe audio "/path/to/audio.wav"
```

### submit 后持续等待结果

submit 后应立即转入下一项工作。
结果会通过面向 heartbeat 的通知文件引入，因此无需轮询或等待。

## 技术机制（参考）

### BackgroundTaskManager（`core/tasks/background.py`）

- **角色**: 将长时间工具调用作为 `asyncio` 任务在后台执行，并在完成/失败时 `await` `on_complete`（任意异步回调）。在构造函数中 `mkdir(parents=True)` `state/background_tasks/`。
- **同步工具**: `submit(tool_name, tool_args, execute_fn, task_id=None)` → `execute_fn(name, args) -> str | None` 通过 `run_in_executor(None, ...)` 在线程池执行。若省略 `task_id` 则生成，CLI command task 中传入与 TaskStore 相同的 ID。
- **异步工具**: `submit_async`（相同签名，`execute_fn` 为 `Awaitable[str]`）→ 在事件循环上 `await execute_fn(...)`。
- **调度**: 用 `asyncio.create_task(..., name=f"bg-{task_id}")` 包装。完成时从 `_async_tasks` 中移除对应条目。
- **持久化**: 每次变更后通过 `_save_task` 向 `state/background_tasks/{task_id}.json` 写入 `to_dict()`（`ensure_ascii=False`, `indent=2`）。损坏的 JSON 会通过 `_load_task` 记录警告日志并 `None`。
- **查询**: `get_task` 优先内存，否则磁盘。`list_tasks(status=...)` 与磁盘上的 `*.json` 合并，按 `created_at` 降序。`active_count` 为内存中 `RUNNING` 的件数。
- **`on_complete`**: 即使回调内出现异常，任务的完成/失败状态仍会保持，失败仅记录到日志。
- **合格工具名**（`is_eligible`）合并以下 **3 层**（后者覆盖前者）。键按原样进行字典匹配（可能同时存在 Mode A 的 schema 名 `generate_3d_model` 和 Mode S 提交用的 `image_gen:3d`）:
  1. 代码内默认 `_DEFAULT_ELIGIBLE_TOOLS`（值为参考秒数。当前键）:
     `generate_character_assets`, `generate_fullbody`, `generate_bustup`, `generate_icon`, `generate_chibi`, `generate_3d_model`, `generate_rigged_model`, `generate_animations`（各 30）、`local_llm` / `run_command`（各 60）
  2. 通过 `BackgroundTaskManager.from_profiles`，从各模块的 `EXECUTION_PROFILE` 中提取 `background_eligible: true` 的子命令（`core.integrations._base.get_eligible_tools_from_profiles`）。键为 `"{tool_name}:{subcmd}"`，秒数为 `expected_seconds`（未设置时 60）
  3. `config.json` 的 `background_task.eligible_tools` — 对每个键以 `threshold_s` 作为秒数覆盖
- **禁用**: 通过 `config.json` 设为 `background_task.enabled: false` 时，`BackgroundTaskManager` 本身不会被创建。submit 会注册到 TaskStore，但执行尝试会回到 pending，并产生需确认通知。
- **清理**: `cleanup_old_tasks(max_age_hours=24)` 会删除 (1) 通过 `completed` / `failed` 使 `completed_at` 超过指定时间的 JSON，(2) 仍为 `running` 且 `created_at` 已过去 **超过 48 小时** 的文件（进程崩溃等产生的孤儿）。返回值为删除件数。保留时间由调用方通过 `max_age_hours` 指定，没有对应的 `config.json` 配置键。

### 同文件内的其他 API: `rotate_dm_logs`

`core/tasks/background.py` 中，**独立于后台工具执行**，存在 `rotate_dm_logs(shared_dir, max_age_days=7)`。在 `shared/dm_logs/*.jsonl` 中，将条目 `ts` 早于阈值的行追加归档到 `{stem}.{YYYYMMDD}.archive.jsonl`，并从活动文件中移除。实际处理为 `_rotate_dm_logs_sync`（通过 `run_in_executor` 卸载）。

### 命令型任务（`animaworks-tool submit`）

1. `animaworks-tool submit` 会原子地将包含 `task_type="command"`、工具名、参数、Anima、`task_id` 的输入注册到 TaskStore（`ANIMAWORKS_ANIMA_DIR` 必需）。
2. TaskStore watcher 通过现有的 claim / attempt 机制获取任务。`BackgroundTaskManager` 也会将原始 `task_id` 用于结果 JSON。
3. 命令型会在内部将 `animaworks-tool … -j` 作为 **子进程** 启动。**每次的墙钟超时为 1800 秒（30 分钟）**（`pending_executor._PENDING_TASK_SUBPROCESS_TIMEOUT`）。
4. stdout 或错误保存到 `state/task_results/{task_id}/{attempt_token}.md`，并在 TaskStore 的执行尝试中记录结果引用。状态、参数、执行尝试的权威来源为 TaskStore。
5. BackgroundTaskManager 会向 `state/background_tasks/{task_id}.json` 保存 `running` → `completed` / `failed`，`_on_background_task_complete` 会向 `state/background_notifications/{task_id}.md` 写入通知。heartbeat 会引入通知。

### 旧命令描述符的迁移

以前的 `state/background_tasks/pending/*.json` 在新运行时中不会被监控。停止期间执行 `animaworks task-store migrate --anima NAME --backup PATH` 时，会验证未处理的顶层描述符并引入 TaskStore。原文件作为证据保留。`processing/` 和 `failed/` 不会自动重新执行或迁移，而是连同警告一起保留，因此需确认副作用后由运用者判断。

### LLM 型任务（正规任务存储）

1. `submit_tasks` / `delegate_task` 会原子地注册完整的指令、上下文、完成条件、约束、模型、依赖关系。
2. watcher 在同一 transaction 中获取可执行任务并创建执行尝试。确认依赖目标和已配置的 worker 容量，非并行的约束限定在同一批次内。
3. `done` / `cancelled` 在无声明的情况下执行尝试结束时会回到 pending，但不会自动重新执行。确认副作用后，如有必要通过 `submit_tasks(batch_id="resume", tasks=[{"task_id":"ID","resume":true}])` 显式恢复。
4. 结果有时可参照 `state/task_results/{task_id}.md`，但状态、输入、执行尝试的权威来源为正规存储。完成通过 `update_task` 声明，不要根据 LLM 的响应或文件存在来推测。
5. 需确认、完成通知会持久化，不依赖定期 heartbeat 即可送达。不要从 DM 重新构建执行输入。

不要直接编辑 SQLite 或旧任务文件，应使用任务工具。这与上述命令型流水线是不同的机制。

### 文件生命周期

**命令型**（`animaworks-tool submit`）:

```
TaskStore の command 入力 → claim / attempt
  → 成功: done | 失敗・中断: pending（要確認。自動再実行なし）
state/task_results/{task_id}/{attempt_token}.md  # attempt output / error
state/background_tasks/{task_id}.json           # 互換用の running → completed / failed 結果
```

**LLM 型**（`submit_tasks` / `delegate_task`）:

```
保存済み入力 → 実行可能pending → 取得済み試行（in_progress）
  → done/cancelled宣言、またはpending + 永続的な要確認通知
  → 明示resumeで保存済み入力を使う新しい試行
```

命令型和 LLM 型是不同的输入格式，但二者都通过 TaskStore 的 claim / attempt 执行。旧 LLM 的 JSONL、`state/pending/` 描述符用于迁移/export，并非执行开始的信号。
