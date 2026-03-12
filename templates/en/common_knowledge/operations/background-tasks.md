# Background Task Execution Guide

## Overview

Some external tools (image generation, 3D model generation, local LLM inference, audio transcription, etc.)
take several minutes to tens of minutes to run. Running them directly holds the lock during execution,
blocking message reception and heartbeat.

Using `animaworks-tool submit` runs the task in the background so you can
immediately move on to the next task.

## When to Use submit

### Tools That Must Use submit

Subcommands marked with ⚠ in the tool guide (system prompt):

- `image_gen pipeline` / `fullbody` / `bustup` / `chibi` / `3d` / `rigging` / `animations`
- `local_llm generate` / `chat`
- `transcribe`

### Tools That Don't Need submit

Tools that typically finish in under 30 seconds:

- `web_search`, `x_search`
- `slack`, `chatwork`, `gmail` (normal operations)
- `github`, `aws_collector`

### How to Decide

- ⚠ mark present → Always use submit
- No ⚠ mark → Run directly

## Usage

### Basic Syntax

```bash
animaworks-tool submit <tool_name> <subcommand> [args...]
```

### Examples

```bash
# 3D model generation (Meshy API, up to 10 min)
animaworks-tool submit image_gen 3d assets/avatar_chibi.png

# Character image full pipeline (all steps, up to 30 min)
animaworks-tool submit image_gen pipeline "1girl, black hair, ..." --negative "lowres, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR

# Local LLM inference (Ollama, up to 5 min)
animaworks-tool submit local_llm generate "Please summarize: ..."

# Audio transcription (Whisper + Ollama post-processing, up to 2 min)
animaworks-tool submit transcribe "/path/to/audio.wav" --language ja
```

### Return Value

submit returns immediately with the following JSON and exits:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "submitted",
  "tool": "image_gen",
  "subcommand": "3d",
  "message": "Background task submitted. You will be notified in inbox when complete."
}
```

## Receiving Results

1. After submit, the task runs in the background
2. When finished, the result is written to `state/background_notifications/{task_id}.md`
3. The next heartbeat automatically picks up this notification
4. The notification includes success/failure status and a result summary

## Handling Failures

- When the notification indicates "failed":
  1. Check the error details
  2. Identify the cause (missing API key, timeout, incorrect arguments, etc.)
  3. Fix and resubmit
  4. If you cannot resolve, report to your supervisor

- If the process crashes during execution, tasks left in `state/background_tasks/pending/processing/` or `state/pending/processing/` are automatically moved to `pending/failed/` on the next startup for recovery

## Common Mistakes

### Running the tool directly

```bash
# Bad: direct run → holds lock for 10 min
animaworks-tool image_gen 3d assets/avatar_chibi.png -j

# Good: async via submit
animaworks-tool submit image_gen 3d assets/avatar_chibi.png
```

If you ran directly, you must wait for it to finish.
Always use submit next time.

### Waiting for results after submit

Move on to the next task immediately after submit.
Results are delivered automatically, so polling or waiting is unnecessary.

## Technical Mechanism (Reference)

PendingTaskExecutor monitors and executes two types of tasks.

### Command-type tasks (animaworks-tool submit)

1. `animaworks-tool submit` writes a task descriptor to `state/background_tasks/pending/*.json`
2. PendingTaskExecutor's watcher monitors `state/background_tasks/pending/` every 3 seconds (`wake()` can trigger an immediate check)
3. On detection, the task is moved from `pending/*.json` to `pending/processing/` and execution starts
4. `execute_pending_task` enqueues the task in BackgroundTaskManager.submit. It runs as a subprocess outside the Anima lock (30 min timeout)
5. On submit success: the file in processing is deleted. On submit failure: moved to `pending/failed/`
6. On completion, the `_on_background_task_complete` callback writes the notification to `state/background_notifications/{task_id}.md`
7. On the next heartbeat, `drain_background_notifications()` reads the notification and injects it into the context

### LLM-type tasks (state/pending/)

LLM tasks written by Heartbeat or the `submit_tasks` tool are enqueued in a **different directory** `state/pending/`.

1. `submit_tasks` writes task descriptors to `state/pending/{task_id}.json` (with `task_type: "llm"`, `batch_id`, etc.)
2. The watcher monitors `state/pending/` every 3 seconds in the same way
3. Tasks with `batch_id` are accumulated and dispatched via `_dispatch_batch` based on the DAG
4. Tasks with `parallel: true` run concurrently under a semaphore (`config.json` `background_task.max_parallel_llm_tasks`, default 3)
5. Tasks with `depends_on` run after their dependencies complete
6. Results are saved to `state/task_results/{task_id}.md`. Completion/failure notifications are sent via DM to `reply_to`
7. Tasks older than 24 hours (TTL) are skipped

The entry point and directory differ from `animaworks-tool submit` described in this guide.

### File Lifecycle

**Command-type** (animaworks-tool submit):

```
state/background_tasks/pending/*.json
  → pending/processing/*.json
  → success: deleted | failure: pending/failed/*.json
```

**LLM-type** (submit_tasks / Heartbeat):

```
state/pending/*.json
  → pending/processing/*.json
  → success: deleted | failure: pending/failed/*.json
```

On startup, orphaned files left in either `processing/` (e.g., from a crash) are moved to `failed/` for recovery.
