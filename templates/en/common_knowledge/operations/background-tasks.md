# Background Task Execution Guide

## Overview

Some external tools (image generation, 3D model generation, local LLM inference, audio transcription, etc.)
take several minutes to tens of minutes to run. Running them directly keeps the lock held for the entire run,
which stops message handling and heartbeat.

Using `animaworks-tool submit` runs the task in the background so you can
move on to the next task immediately.

At runtime, **BackgroundTaskManager** in `core/background.py` handles tool execution and
persisting state to `state/background_tasks/{task_id}.json`, while **PendingTaskExecutor**
inside the Anima child process (`core/supervisor/pending_executor.py`) watches the wait queue written by
`animaworks-tool submit` and hands tasks off to `BackgroundTaskManager`.

## When to Use submit

### Tools That Must Use submit

Subcommands marked with ⚠ in the tool guide (system prompt):

- `image_gen pipeline` / `fullbody` / `bustup` / `icon` / `chibi` / `3d` / `rigging` / `animations`
- `local_llm generate` / `chat`
- `transcribe audio` (subcommand name is `audio`)
- Potentially long-running commands (follow the ⚠ marks in the tool guide)

Tools with `background_eligible: true` in each tool’s `EXECUTION_PROFILE` are registered as background-eligible
via the profile (e.g. `chatwork sync` / `download`).
Operational policy should still **prioritize the ⚠ marks**.

### Tools That Don't Need submit

Tools that finish quickly (tens of seconds or less as a rule of thumb):

- `web_search`, `x_search`
- `slack`, `chatwork`, `gmail` (normal operations)
- `github`, `aws_collector`

### How to Decide

- ⚠ present → Always use submit
- No ⚠ → Run directly (when using `animaworks-tool submit` for a profile-classified “short” tool, a warning may appear on stderr)

## Usage

### Basic Syntax

```bash
animaworks-tool submit <tool_name> <subcommand> [args...]
```

### Examples

```bash
# 3D model generation (Meshy API, etc.)
animaworks-tool submit image_gen 3d assets/avatar_chibi.png

# Character image full pipeline (all steps)
animaworks-tool submit image_gen pipeline "1girl, black hair, ..." --negative "lowres, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR

# Local LLM inference (Ollama)
animaworks-tool submit local_llm generate "Please summarize: ..."

# Audio transcription (Whisper, etc.) — subcommand is audio
animaworks-tool submit transcribe audio "/path/to/audio.wav" --language ja
```

### Return Value

submit prints JSON to stdout and exits immediately (`task_id` is 12 hex digits):

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "submitted",
  "tool": "image_gen",
  "subcommand": "3d",
  "message": "Background task submitted. You will be notified in inbox when complete. (task_id: a1b2c3d4e5f6)"
}
```

## Receiving Results

1. After submit, **PendingTaskExecutor** ingests descriptors under `state/background_tasks/pending/` and runs them via
   `BackgroundTaskManager.submit` in the background (outside the Anima conversation lock).
2. From start through completion, state for the same `task_id` is also stored in **`state/background_tasks/{task_id}.json`**.
   When `BackgroundTaskManager.submit` / `submit_async` first writes the file, status is **`running` from the outset** (saved to disk right after a 12-digit UUID `task_id` is assigned). On success it becomes `completed`; on exception, `failed`.
   The data model also has a `pending` enum value, but the manager’s normal submission path does not use it. **Waiting in queue** is separate: it is represented by descriptors under `state/background_tasks/pending/*.json`.
3. On completion, `_on_background_task_complete` writes a Markdown notification to **`state/background_notifications/{task_id}.md`**.
4. On the next **heartbeat**, `drain_background_notifications()` reads and removes that `.md` and injects it into context.
5. If the Web UI WebSocket or human notification via `call_human` is enabled, completion may also surface there at the same time.

Use **`list_background_tasks`** for a merged list of in-memory and on-disk tasks, and **`check_background_task`** to look up status by `task_id` (both map to `BackgroundTaskManager`’s `list_tasks` / `get_task`).

## Handling Failures

- If the notification says the run failed:
  1. Review the error details
  2. Identify the cause (missing API key, timeout, wrong arguments, etc.)
  3. Fix and submit again
  4. If you cannot resolve it, report to your supervisor

- After a **crash or abnormal exit**, if JSON is left under `processing/`, **PendingTaskExecutor** recovers it when the Anima process starts:
  - **Command-type** (`animaworks-tool submit`): `state/background_tasks/pending/processing/*.json` → `state/background_tasks/pending/failed/`
  - **LLM-type** (`submit_tasks` / Heartbeat handoff): `state/pending/processing/*.json` → `state/pending/failed/`
  These use **different directories** from submit in this guide (the latter is the `state/pending/` tree).

## Common Mistakes

### Running the tool directly

```bash
# Bad: direct run → may hold the lock for a long time
animaworks-tool image_gen 3d assets/avatar_chibi.png -j

# Good: async via submit
animaworks-tool submit image_gen 3d assets/avatar_chibi.png
```

If you ran directly, you must wait until the task finishes.
Always use submit from now on.

### Omitting the transcribe subcommand

```bash
# Bad: without the audio subcommand, profile resolution may not match intent
animaworks-tool submit transcribe "/path/to/audio.wav"

# Good
animaworks-tool submit transcribe audio "/path/to/audio.wav"
```

### Waiting for results after submit

Move on to the next task immediately after submit.
Results are ingested via heartbeat-oriented notification files, so polling or blocking waits are unnecessary.

## Technical Mechanism (Reference)

### BackgroundTaskManager (`core/background.py`)

- **Role**: Runs long-running tool calls as `asyncio` background tasks and `await`s the optional async `on_complete` callback on completion or failure. The constructor `mkdir(parents=True)` for `state/background_tasks/`.
- **Sync tools**: `submit(tool_name, tool_args, execute_fn)` → `execute_fn(name, args) -> str | None` runs on a thread pool via `run_in_executor(None, ...)`.
- **Async tools**: `submit_async` (same signature but `execute_fn` returns `Awaitable[str]`) → `await execute_fn(...)` on the event loop.
- **Scheduling**: Wrapped with `asyncio.create_task(..., name=f"bg-{task_id}")`. On completion, the matching entry is removed from `_async_tasks`.
- **Persistence**: After each change, `_save_task` writes `state/background_tasks/{task_id}.json` via `to_dict()` (`ensure_ascii=False`, `indent=2`). Corrupt JSON in `_load_task` is logged and treated as `None`.
- **Queries**: `get_task` prefers in-memory, else disk. `list_tasks(status=...)` merges on-disk `*.json` with memory and sorts by `created_at` descending. `active_count` is the in-memory count of `RUNNING` tasks.
- **`on_complete`**: Even if the callback raises, the task’s completed/failed state is preserved; the failure is logged only.
- **Eligible tool names** (`is_eligible`) merge these **three layers** (later wins). Keys are matched as dictionary keys (you may have both Mode A schema names like `generate_3d_model` and Mode S submit-style `image_gen:3d`):
  1. In-code default `_DEFAULT_ELIGIBLE_TOOLS` (values are guideline seconds; current keys):
     `generate_character_assets`, `generate_fullbody`, `generate_bustup`, `generate_icon`, `generate_chibi`, `generate_3d_model`, `generate_rigged_model`, `generate_animations` (30 each), `local_llm` / `run_command` (60 each)
  2. Via `BackgroundTaskManager.from_profiles`, subcommands with `background_eligible: true` from each module’s `EXECUTION_PROFILE` (`core.tools._base.get_eligible_tools_from_profiles`). Keys are `"{tool_name}:{subcmd}"`; seconds come from `expected_seconds` (default 60 if unset).
  3. `config.json` `background_task.eligible_tools` — each key’s `threshold_s` overrides the duration in seconds.
- **Disable**: `background_task.enabled: false` in `config.json` prevents creating `BackgroundTaskManager` at all (in that case, the submit queue may still be picked up but the executor side logs a warning).
- **Cleanup**: `cleanup_old_tasks(max_age_hours=24)` (1) deletes JSON for `completed` / `failed` with `completed_at` older than **24 hours**, and (2) deletes **`running`** files whose `created_at` is older than **48 hours** (orphans from process crashes, etc.). Return value is the number removed. `config.json` `background_task.result_retention_hours` exists in the schema but is **not read by the current `BackgroundTaskManager` implementation**; callers only pass an arbitrary `max_age_hours` to `cleanup_old_tasks`.

### Other API in the same file: `rotate_dm_logs`

Independently of **background tool execution**, `core/background.py` defines `rotate_dm_logs(shared_dir, max_age_days=7)`. For `shared/dm_logs/*.jsonl`, rows whose entry `ts` is older than the threshold are appended to `{stem}.{YYYYMMDD}.archive.jsonl` and removed from the active file. Actual work is `_rotate_dm_logs_sync`, offloaded with `run_in_executor`.

### Command-type tasks (`animaworks-tool submit`)

1. `animaworks-tool submit` writes a descriptor to `state/background_tasks/pending/{task_id}.json` (`ANIMAWORKS_ANIMA_DIR` required).
2. PendingTaskExecutor’s watcher polls `pending/` at most every **3 seconds** (or immediately via `wake()`).
3. `pending/*.json` → renamed to `pending/processing/*.json`, then `execute_pending_task`.
4. Command-type runs invoke `animaworks-tool … -j` in a **subprocess**. **Wall-clock timeout per run is 1800 seconds (30 minutes)** (`pending_executor._PENDING_TASK_SUBPROCESS_TIMEOUT`). On success, the file under processing is removed; on error, moved to `pending/failed/`.
5. Actual work is delegated to `BackgroundTaskManager.submit(composite_name, tool_args, execute_fn)`. `composite_name` is `tool:subcommand` (e.g. `image_gen:3d`) and is checked against `is_eligible`.
6. On completion, `_on_background_task_complete` writes `state/background_notifications/{task_id}.md`, which heartbeat’s `drain_background_notifications()` reads.

### LLM-type tasks (`state/pending/`)

LLM tasks written by Heartbeat or the `submit_tasks` tool go to a **different directory**, `state/pending/`.

1. `submit_tasks` writes task descriptors to `state/pending/{task_id}.json` (`task_type: "llm"`, `batch_id`, etc.)
2. The watcher monitors `state/pending/` similarly.
3. Tasks with `batch_id` are batched and executed via `_dispatch_batch` according to the DAG.
4. Tasks with `parallel: true` run under a semaphore (`config.json` `background_task.max_parallel_llm_tasks`, default 3).
5. Tasks with `depends_on` run after dependencies complete.
6. Results go to `state/task_results/{task_id}.md` (summaries are length-capped). If `reply_to` is set, completion/failure is notified by DM.
7. Tasks older than 24 hours (TTL) are skipped.

This differs from `animaworks-tool submit` in entry point and directory layout.

### File Lifecycle

**Command-type** (wait queue for `animaworks-tool submit`):

```
state/background_tasks/pending/*.json
  → pending/processing/*.json
  → success: deleted | failure: pending/failed/*.json
```

Also the **task state file** (overall execution):

```
state/background_tasks/{task_id}.json   # running → completed / failed
```

**LLM-type** (`submit_tasks` / Heartbeat):

```
state/pending/*.json
  → pending/processing/*.json
  → success: deleted | failure: pending/failed/*.json
```

On startup, orphaned files left in each respective `processing/` are moved to `failed/` for recovery.
