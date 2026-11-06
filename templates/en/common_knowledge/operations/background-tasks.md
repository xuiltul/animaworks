# Background Task Execution Guide

## Overview

Some external tools (image generation, 3D model generation, local LLM inference, speech transcription, etc.)
can take anywhere from a few minutes to several tens of minutes to complete. If you run these directly, the lock is held for the entire duration,
and message reception and heartbeat processing come to a halt.

By using `animaworks-tool submit`, you can run tasks in the background and
move on to the next task immediately.

`animaworks-tool submit` is registered in the TaskStore as `task_type="command"`,
and the **PendingTaskExecutor** (`core/tasks/pending_executor.py`) inside the Anima child process picks up the attempt.
The **BackgroundTaskManager** (`core/tasks/background.py`) runs the tool in the background,
and the attempt is stored in the TaskStore, while the compatibility result JSON is saved to `state/background_tasks/{task_id}.json`.

## When to use submit

### Tools that must use submit

Subcommands marked with ⚠ in the tool guide (system prompt):

- `image_gen pipeline` / `fullbody` / `bustup` / `icon` / `chibi` / `3d` / `rigging` / `animations`
- `local_llm generate` / `chat`
- `transcribe audio` (subcommand name is `audio`)

Tools whose `EXECUTION_PROFILE` is `background_eligible: true` are registered as background execution candidates
via the profile (e.g., `chatwork sync` / `download`, etc.).
Operational policy should continue to prioritize the **⚠ mark**.

### Tools that do not require submit

Tools with short execution times (roughly under a few tens of seconds):

- `web_search`, `x_search`
- `slack`, `chatwork`, `gmail` (normal operations)
- `github`, `aws_collector`

### Decision criteria

- Has ⚠ mark → must use submit
- No ⚠ mark → run directly (at `animaworks-tool submit` runtime, a warning may appear on stderr if the profile marks it as "short duration")

## Usage

### Basic syntax

```bash
animaworks-tool submit <ツール名> <サブコマンド> [引数...]
```

### Execution example

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

### Return value

submit immediately outputs JSON to stdout and exits (`task_id` is a 12-digit hex):

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "submitted",
  "tool": "image_gen",
  "subcommand": "3d",
  "message": "バックグラウンドタスクを投入しました。完了時にinboxに通知されます。(task_id: a1b2c3d4e5f6)"
}
```

## Receiving results

1. submit registers the input containing the tool name, arguments, and Anima (`task_type="command"`) in the TaskStore. The PendingTaskExecutor picks up the execution attempt and requests execution from the BackgroundTaskManager outside the conversation lock.
2. The TaskStore records the attempt and result reference. `state/background_tasks/{task_id}.json` records `running` → `completed` / `failed` with the same task_id for compatibility with the legacy query API.
3. On completion, `_on_background_task_complete` writes a Markdown notification to **`state/background_notifications/{task_id}.md`**.
4. At the next **heartbeat**, `drain_background_notifications()` reads and deletes the relevant `.md`, and it is injected into the context.
5. If the Web UI's WebSocket or `call_human`-family human notifications are enabled, they may also appear there at the same completion timing.

Use tool **`list_background_tasks`** to view a merged list of memory and disk, and **`check_background_task`** to view the status specified by `task_id` (both correspond to `BackgroundTaskManager`'s `list_tasks` / `get_task`).

## Handling failures

- If the notification says "failure":
  1. Check the error details
  2. Identify the cause (API key not set, timeout, argument error, etc.)
  3. Fix the issue and submit again
  4. If it cannot be resolved, report to the supervisor

- If the process terminates abnormally, check the attempt owner in the TaskStore, and record tasks whose owner has confirmed termination as pending (needs review). Since side effects may have already occurred, do not automatically re-run. Even if the result JSON remains as `running`, do not judge it as complete; confirm the actual status and then explicitly resume.

## Common mistakes

### Running directly

```bash
# 悪い例: 直接実行 → 長時間ロックされうる
animaworks-tool image_gen 3d assets/avatar_chibi.png -j

# 良い例: submit で非同期実行
animaworks-tool submit image_gen 3d assets/avatar_chibi.png
```

If you ran it directly, you have no choice but to wait until the task completes.
Always use submit from next time onward.

### Omitting the transcribe subcommand

```bash
# 悪い例: audio サブコマンドがないと意図したプロファイル判定にならない
animaworks-tool submit transcribe "/path/to/audio.wav"

# 良い例
animaworks-tool submit transcribe audio "/path/to/audio.wav"
```

### Waiting for results after submitting

Once you submit, move on to the next task immediately.
Results are picked up via the notification file for heartbeat, so no polling or waiting is needed.

## Technical details (reference)

### BackgroundTaskManager (`core/tasks/background.py`)

- **Role**: Runs long-running tool calls as `asyncio` tasks in the background, and invokes `on_complete` (arbitrary asynchronous callbacks) on completion or failure via `await`. In the constructor, `state/background_tasks/` is `mkdir(parents=True)`.
- **Synchronous tools**: `submit(tool_name, tool_args, execute_fn, task_id=None)` → executes `execute_fn(name, args) -> str | None` on a thread pool via `run_in_executor(None, ...)`. If `task_id` is omitted, it is generated, and for CLI command tasks, the same ID as the TaskStore is passed.
- **Asynchronous tools**: `submit_async` (with the same signature, where `execute_fn` is `Awaitable[str]`) → runs on the event loop via `await execute_fn(...)`.
- **Scheduling**: Wrapped with `asyncio.create_task(..., name=f"bg-{task_id}")`. On completion, removes the relevant entry from `_async_tasks`.
- **Persistence**: After each change, `_save_task` writes `to_dict()` (`ensure_ascii=False`, `indent=2`) to `state/background_tasks/{task_id}.json`. Corrupted JSON is logged as a warning via `_load_task` and then `None`.
- **Query**: `get_task` prioritizes in-memory, falling back to disk. `list_tasks(status=...)` merges with `*.json` on disk, sorted by `created_at` descending. `active_count` is the count of `RUNNING` in memory.
- **`on_complete`**: Even if an exception occurs inside a callback, the task's completion/failure status is maintained, and the failure is only recorded in the log.
- **Eligible tool names** (`is_eligible`) merge the following **3 layers** (later wins). Keys are matched directly as dictionary lookups (both Mode A schema names `generate_3d_model` and Mode S submission `image_gen:3d` are possible):
  1. In-code defaults `_DEFAULT_ELIGIBLE_TOOLS` (values are approximate seconds; current keys):
     `generate_character_assets`, `generate_fullbody`, `generate_bustup`, `generate_icon`, `generate_chibi`, `generate_3d_model`, `generate_rigged_model`, `generate_animations` (30 each), `local_llm` / `run_command` (60 each)
  2. Via `BackgroundTaskManager.from_profiles`, extract subcommands of `background_eligible: true` from each module's `EXECUTION_PROFILE` (`core.integrations._base.get_eligible_tools_from_profiles`). Keys are `"{tool_name}:{subcmd}"`, seconds are `expected_seconds` (default 60 if unset)
  3. `background_task.eligible_tools` of `config.json` — for each key, overwrite `threshold_s` as the number of seconds
- **Disabling**: Setting `background_task.enabled: false` via `config.json` prevents `BackgroundTaskManager` itself from being created. submit is still registered in the TaskStore, but the execution attempt returns to pending, and a needs-review notification is generated.
- **Cleanup**: `cleanup_old_tasks(max_age_hours=24)` deletes (1) JSON files where `completed_at` has exceeded the specified time via `completed` / `failed`, and (2) files where `created_at` has been in the `running` state for more than **48 hours** (orphans from process crashes, etc.). The return value is the number of deletions. The retention time is specified by the caller via `max_age_hours`, and there is no corresponding `config.json` configuration key.

### Command-type tasks (`animaworks-tool submit`)

1. `animaworks-tool submit` atomically registers input containing `task_type="command"`, the tool name, arguments, Anima, and `task_id` in the TaskStore (`ANIMAWORKS_ANIMA_DIR` required).
2. The TaskStore watcher picks up tasks via the existing claim/attempt mechanism. `BackgroundTaskManager` also uses the original `task_id` for the result JSON.
3. Command-type tasks internally launch `animaworks-tool … -j` as a **subprocess**. **The wall-clock timeout per run is 1800 seconds (30 minutes)** (`pending_executor._PENDING_TASK_SUBPROCESS_TIMEOUT`).
4. stdout or errors are saved to `state/task_results/{task_id}/{attempt_token}.md`, and the result reference is recorded on the TaskStore attempt. The authoritative source for status, arguments, and attempts is the TaskStore.
5. The BackgroundTaskManager saves `running` → `completed` / `failed` to `state/background_tasks/{task_id}.json`, and `_on_background_task_complete` writes a notification to `state/background_notifications/{task_id}.md`. The heartbeat picks up the notification.

### Migration of legacy command descriptors

The previous `state/background_tasks/pending/*.json` is no longer monitored by the new runtime. If `animaworks task-store migrate --anima NAME --backup PATH` is executed during shutdown, unprocessed top-level descriptors are validated and imported into the TaskStore. The original files are kept as evidence. `processing/` and `failed/` are not automatically re-run or migrated; they are left with a warning, so the operator checks for side effects and then decides.

### LLM-type tasks (canonical task store)

1. `submit_tasks` / `delegate_task` atomically register the complete instruction, context, completion conditions, constraints, model, and dependencies.
2. The watcher acquires runnable tasks and creates attempts in the same transaction. It checks dependencies and configured worker capacity, and restricts non-parallel constraints to the same batch.
3. If an attempt finishes without `done` / `cancelled` declarations, it returns to pending but is not automatically re-run. Check for side effects, and if necessary, explicitly resume via `submit_tasks(batch_id="resume", tasks=[{"task_id":"ID","resume":true}])`.
4. Results may reference `state/task_results/{task_id}.md`, but the canonical store is the authoritative source for status, input, and attempts. Completion is declared via `update_task` and is not inferred from LLM responses or file existence.
5. Needs-review and completion notifications are persisted and delivered without depending on the periodic heartbeat. Do not reconstruct execution input from DMs.

Do not edit SQLite or legacy task files directly; use the task tools. This is a separate mechanism from the command-type pipeline described above.

### File lifecycle

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

Command-type and LLM-type use different input formats, but both are executed via the TaskStore's claim/attempt mechanism. The legacy LLM JSONL and `state/pending/` descriptors are for migration and export purposes, not a signal to start execution.
