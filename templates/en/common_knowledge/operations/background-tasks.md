# Background Task Guide

## Overview

Some external tools (image generation, 3D model generation, local LLM inference, transcription, etc.)
take minutes to tens of minutes to run. Running them directly holds the lock during execution
and blocks message handling and Heartbeat.

Using `animaworks-tool submit` runs the task in the background so you can
move on to other work immediately.

## When to Use submit

### Tools That Must Use submit

Subcommands marked with ⚠ in the tool guide (system prompt):

- `image_gen pipeline` / `fullbody` / `bustup` / `chibi` / `3d` / `rigging` / `animations`
- `local_llm generate` / `chat`
- `transcribe`

### Tools That Don't Need submit

Tools that typically finish in under 30 seconds:

- `web_search`, `x_search`
- `slack`, `chatwork`, `gmail` (normal use)
- `github`, `aws_collector`

### How to Decide

- ⚠ mark present → Always use submit
- No ⚠ → Run directly

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

# Audio transcription (Whisper + Ollama cleanup, up to 2 min)
animaworks-tool submit transcribe "/path/to/audio.wav" --language ja
```

### Return Value

submit returns immediately with JSON and exits:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "submitted",
  "tool": "image_gen",
  "subcommand": "3d",
  "message": "Task submitted. You will be notified in inbox when complete."
}
```

## Receiving Results

1. After submit, the task runs in the background
2. When finished, the result is written to `state/background_notifications/{task_id}.md`
3. The next Heartbeat automatically picks up this notification
4. The notification includes success/failure and a result summary

## Handling Failures

When the notification says "failed":
1. Check the error details
2. Identify cause (missing API key, timeout, bad args, etc.)
3. Fix and resubmit
4. If you cannot resolve, report to supervisor

## Common Mistakes

### Running the tool directly

```bash
# Bad: direct run → holds lock for 10 min
animaworks-tool image_gen 3d assets/avatar_chibi.png -j

# Good: async via submit
animaworks-tool submit image_gen 3d assets/avatar_chibi.png
```

If you ran directly, you must wait for it to finish.
Use submit next time.

### Waiting for results after submit

Do not poll or wait after submit.
Move on; results are delivered automatically.

## How It Works (Technical)

1. `animaworks-tool submit` writes a task JSON under `state/background_tasks/pending/`
2. Runner's pending watcher polls this dir every 3 seconds
3. On detection, BackgroundTaskManager enqueues (runs outside Anima _lock)
4. BackgroundTaskManager runs it via asyncio.create_task()
5. On completion, result is written under `state/background_notifications/`
6. Next Heartbeat processes the notification
