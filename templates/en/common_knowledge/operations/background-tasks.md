# Background Task Execution Guide

## Overview

Some external tools (image generation, 3D model generation, local LLM inference, speech transcription, etc.)
take several minutes to several tens of minutes to run. Running them directly holds the lock for the entire duration,
stopping message reception and heartbeat.

Using `animaworks-tool submit` lets you run tasks in the background
and move on to your next task immediately.

`animaworks-tool submit` is registered in TaskStore as `task_type="command"`,
and **PendingTaskExecutor** (`core/tasks/pending_executor.py`) in the Anima child process picks up attempts.
**BackgroundTaskManager** (`core/tasks/background.py`) runs the tool in the background,
and attempts are stored in TaskStore, while compatibility result JSON is saved to `state/background_tasks/{task_id}.json`.

## When to Use submit

### Tools That Must Use submit

Subcommands marked ⚠ in the tool guide (system prompt):

- `image_gen pipeline` / `fullbody` / `bustup` / `icon` / `chibi` / `3d` / `rigging` / `animations`
- `local_llm generate` / `chat`
- `transcribe audio` (the subcommand name is `audio`)

Subcommands designated as `background_eligible: true` in each tool's `EXECUTION_PROFILE` are registered as candidates
for background execution via profiles (e.g., `chatwork sync` / `download`, etc.).
Operational policy should continue to prioritize **⚠ marks**.

### Tools That Do Not Require submit

Tools with short runtimes (typically under several tens of seconds):

- `web_search`, `x_search`
- `slack`, `chatwork`, `gmail` (ordinary operations)
- `github`, `aws_collector`

### Decision Criteria

- Marked ⚠ → Always use submit
- Not marked ⚠ → Run directly (when running `animaworks-tool submit`, a warning may be written to stderr if the profile classifies it as "short-running")

## How to Use

### Basic Syntax

```bash
animaworks-tool submit <ツール名> <サブコマンド> [引数...]
```

### Example

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

### Return Value

submit immediately outputs JSON to stdout and exits (`task_id` is 12-digit hex):

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "submitted",
  "tool": "image_gen",
  "subcommand": "3d",
  "message": "バックグラウンドタスクを投入しました。完了時にinboxに通知されます。(task_id: a1b2c3d4e5f6)"
}
```

## Receiving Results

1. submit registers input containing the tool name, arguments, and Anima as `task_type="command"` in TaskStore. PendingTaskExecutor picks up the execution attempt and asks BackgroundTaskManager to run it outside the conversation lock.
2. TaskStore records attempts and result references. `state/background_tasks/{task_id}.json` records `running` → `completed` / `failed` under the same task_id for compatibility with the legacy query API.
3. Upon completion, `_on_background_task_complete` writes a Markdown notification to **`state/background_notifications/{task_id}.md`**.
4. On the next **heartbeat**, `drain_background_notifications()` reads and deletes that `.md`, which is injected into the context.
5. If the Web UI's WebSocket or `call_human`-based human notifications are enabled, the result may also be sent there at the same time as completion.

Use tool **`list_background_tasks`** to view a list merging memory and disk, and **`check_background_task`** to view the status specified by `task_id` (both correspond to `list_tasks` / `get_task` of `BackgroundTaskManager`).

## Handling Failures

- If the notification says "failure":
  1. Check the error details
  2. Identify the cause (missing API key configuration, timeout, incorrect arguments, etc.)
  3. Fix it and submit again
  4. If you cannot resolve it, report it to your supervisor

- On abnormal process termination, check the attempt owner in TaskStore and record tasks whose owner's termination has been confirmed as pending (requires review). Since side effects may already have occurred, do not automatically rerun them. Even if the result JSON remains at `running`, do not treat it as complete; check the actual status and explicitly resume it only after doing so.

## Common Mistakes

### Running Directly

```bash
# 悪い例: 直接実行 → 長時間ロックされうる
animaworks-tool image_gen 3d assets/avatar_chibi.png -j

# 良い例: submit で非同期実行
animaworks-tool submit image_gen 3d assets/avatar_chibi.png
```

If you run it directly, you have no choice but to wait until the task is complete.
Always use submit from next time onward.

### Omitting the transcribe Subcommand

```bash
# 悪い例: audio サブコマンドがないと意図したプロファイル判定にならない
animaworks-tool submit transcribe "/path/to/audio.wav"

# 良い例
animaworks-tool submit transcribe audio "/path/to/audio.wav"
```

### Continuing to Wait for Results After submit

Move on to your next task as soon as you submit.
Results are received through notification files for heartbeat, so polling or waiting is unnecessary.

## Technical Details (Reference)

### BackgroundTaskManager（`core/tasks/background.py`）

- **Role**: Runs long-running tool calls as background `asyncio` tasks and `await` an `on_complete` (optional async callback) on completion or failure. The constructor `mkdir(parents=True)` `state/background_tasks/`.
- **Synchronous tools**: Runs `submit(tool_name, tool_args, execute_fn, task_id=None)` → `execute_fn(name, args) -> str | None` in a thread pool using `run_in_executor(None, ...)`. If `task_id` is omitted, one is generated; for CLI command tasks, pass the same ID as TaskStore.
- **Asynchronous tools**: `submit_async` (with the same signature, and `execute_fn` is `Awaitable[str]`) → `await execute_fn(...)` on the event loop.
- **Scheduling**: Wrapped in `asyncio.create_task(..., name=f"bg-{task_id}")`. On completion, remove the relevant entry from `_async_tasks`.
- **Persistence**: After each change, `_save_task` `state/background_tasks/{task_id}.json` to `to_dict()` (`ensure_ascii=False`, `indent=2`). For corrupted JSON, log a warning with `_load_task` and then `None`.
- **Queries**: `get_task` checks memory first, then disk. `list_tasks(status=...)` merges with `*.json` on disk, sorted by `created_at` in descending order. `active_count` is the number of `RUNNING` in memory.
- **`on_complete`**: Even if an exception occurs in a callback, the task’s completion/failure status is preserved; failures are only logged.
- **Qualified tool names** (`is_eligible`) are merged from the following **3 layers** (later layers take precedence). Keys are looked up directly in the dictionary (both Mode A schema names `generate_3d_model` and Mode S submission names `image_gen:3d` may be present):
  1. In-code defaults `_DEFAULT_ELIGIBLE_TOOLS` (values are approximate seconds; current keys):
     `generate_character_assets`, `generate_fullbody`, `generate_bustup`, `generate_icon`, `generate_chibi`, `generate_3d_model`, `generate_rigged_model`, `generate_animations` (30 each), `local_llm` / `run_command` (60 each)
  2. Via `BackgroundTaskManager.from_profiles`, extract subcommands of `background_eligible: true` from each module’s `EXECUTION_PROFILE` (`core.integrations._base.get_eligible_tools_from_profiles`). Keys are `"{tool_name}:{subcmd}"`, and the duration is `expected_seconds` (60 if unset).
  3. `background_task.eligible_tools` in `config.json` — override each key with `threshold_s` as the duration.
- **Disabling**: When `background_task.enabled: false` is set to `config.json`, `BackgroundTaskManager` itself is not created. The submission is registered in TaskStore, but the execution attempt returns to pending and a notification requiring review is generated.
- **Cleanup**: `cleanup_old_tasks(max_age_hours=24)` deletes (1) JSON files where `completed_at` in `completed` / `failed` exceeds the specified time, and (2) files that remain `running` whose `created_at` is more than **48 hours** in the past (orphans, such as those left by a process crash). The return value is the number of files deleted. The caller specifies the retention period with `max_age_hours`; there is no corresponding `config.json` configuration key.

### Command-Type Tasks (`animaworks-tool submit`)

1. `animaworks-tool submit` atomically registers input containing `task_type="command"`, the tool name, arguments, Anima, and `task_id` in TaskStore (`ANIMAWORKS_ANIMA_DIR` required).
2. The TaskStore watcher claims tasks using the existing claim/attempt mechanism. `BackgroundTaskManager` also uses the original `task_id` for the result JSON.
3. Command-type tasks internally launch `animaworks-tool … -j` as a **subprocess**. **The wall-clock timeout per run is 1800 seconds (30 minutes)** (`pending_executor._PENDING_TASK_SUBPROCESS_TIMEOUT`).
4. stdout or errors are saved to `state/task_results/{task_id}/{attempt_token}.md`, and the result reference is recorded in the TaskStore attempt. TaskStore is authoritative for status, arguments, and attempts.
5. BackgroundTaskManager saves `running` → `completed` / `failed` in `state/background_tasks/{task_id}.json`, and `_on_background_task_complete` writes a notification to `state/background_notifications/{task_id}.md`. heartbeat consumes the notification.

### Migration of Legacy Command Descriptors

The previous `state/background_tasks/pending/*.json` are not monitored by the new runtime. While stopped, running `animaworks task-store migrate --anima NAME --backup PATH` validates unprocessed top-level descriptors and imports them into TaskStore. The original files are retained as evidence. `processing/` and `failed/` are not automatically rerun or migrated; they are left with a warning, so operators should check for side effects before deciding how to proceed.

### LLM-Type Tasks (Canonical Task Store)

1. `submit_tasks` / `delegate_task` atomically register complete instructions, context, completion criteria, constraints, model, and dependencies.
2. The watcher claims runnable tasks and creates attempts in the same transaction. It checks dependencies and configured worker capacity, and limits non-parallel constraints to the same batch.
3. If an attempt ends without declaring `done` / `cancelled`, it returns to pending but is not automatically rerun. Check for side effects and, if necessary, explicitly resume it with `submit_tasks(batch_id="resume", tasks=[{"task_id":"ID","resume":true}])`.
4. Results may be accessible through `state/task_results/{task_id}.md`, but the canonical store is authoritative for status, input, and attempts. Declare completion with `update_task`; do not infer it from an LLM response or the existence of a file.
5. Review-required and completion notifications are persisted and delivered without depending on periodic heartbeats. Do not reconstruct execution input from DMs.

Do not directly edit SQLite or legacy task files; use task tools. This is a separate mechanism from the command-type pipeline above.

### File Lifecycle

**Command-type** (`animaworks-tool submit`):

```
TaskStore の command 入力 → claim / attempt
  → 成功: done | 失敗・中断: pending（要確認。自動再実行なし）
state/task_results/{task_id}/{attempt_token}.md  # attempt output / error
state/background_tasks/{task_id}.json           # 互換用の running → completed / failed 結果
```

**LLM-type** (`submit_tasks` / `delegate_task`):

```
保存済み入力 → 実行可能pending → 取得済み試行（in_progress）
  → done/cancelled宣言、またはpending + 永続的な要確認通知
  → 明示resumeで保存済み入力を使う新しい試行
```

Command-type and LLM-type tasks use different input formats, but both are executed via TaskStore claims/attempts. Legacy LLM JSONL and `state/pending/` descriptors are for migration/export and do not signal the start of execution.
