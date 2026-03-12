## How Task Execution Works

### Task Tool Auto-Routing (S-mode)

When you use the Task tool, the framework automatically routes based on your org structure.

**With subordinates** → Immediately delegated to a subordinate
- Include a subordinate's name in the description to assign them directly
  Example: "Have alice run the API tests"
  Example: "bob handles the code review"
- If no name is given, the least-loaded subordinate with the best role match is auto-selected
- Falls back to state/pending/ if all subordinates are disabled

**Without subordinates** → Submitted as a background task
- Written to state/pending/ and automatically executed by TaskExec in a separate session
- The executor shares your identity, injection, behavior rules, memory guide, and org context
- A task_id is returned. You will receive a DM notification when it completes
- You can check task results in Heartbeat (state/task_results/)

### Choosing the Right Task Tool

| Tool | Purpose | Execution Queue (Layer 1) | Tracking (Layer 2) | When to use |
|------|---------|--------------------------|--------------------|----|
| `submit_tasks` | Submit tasks for execution | Creates in `state/pending/` | Registers in `task_queue.jsonl` | When you find tasks that need execution |
| `delegate_task` | Delegate to subordinates | Creates in subordinate's `state/pending/` | Registers in both `task_queue.jsonl` | When assigning to subordinates |
| `backlog_task` | Tracking registration only | **Not created (not executed)** | Registers in `task_queue.jsonl` | Recording human instructions, tasks for manual pickup |
| Task tool (S-mode) | Auto-routed delegation | Creates at auto-selected target | Registered | Quick delegation from Chat path |

**Important**: `backlog_task` is tracking-only — TaskExec will NOT automatically execute it. Recording human instructions via `backlog_task` is MUST, but if execution is also needed, use `submit_tasks` as well.

From paths without the Task tool (Heartbeat, Inbox, etc.), use `submit_tasks` / `delegate_task` / `backlog_task`.

**[MUST] Do NOT manually create JSON files in `state/pending/`.** Always submit via the `submit_tasks` tool. `submit_tasks` registers in both the execution queue and task registry simultaneously, preventing tracking gaps.

## Task Submission via submit_tasks

`submit_tasks` is the sole means of submitting tasks for execution (except subordinate delegation).
Use `submit_tasks` even for a single task (tasks array with one item).

### About the Executor (TaskExec)

TaskExec runs as a sub-agent. It shares your identity, behavior guidelines, memory directories, and organization info, but **cannot access your conversation history, short-term memory, or Priming results**.

Therefore, including sufficient information in the task's `description` and `context` is critical.

### Description Writing Principles

- **Always include file paths and line numbers**: The executor can search memory, but specifying exact locations ensures it reaches the right files
- **Include current work state**: Copy relevant parts of current_task.md into the `context` field (auto-injected but explicit additions improve accuracy)
- **State the "why"**: Without background and purpose, the executor may make incorrect decisions

### What to Include in description

- **What to do**: Concrete work (e.g., "Convert verify_token() in core/auth/manager.py to async" instead of "do refactoring")
- **Why**: Background and purpose (1–2 sentences)
- **Where to look**: Related file paths and line numbers (also set in `file_paths` field)
- **Completion criteria**: What counts as "done" (also set in `acceptance_criteria` field)
- **Constraints**: Prohibitions, compatibility requirements (also set in `constraints` field)

### Examples

Single task:

```
submit_tasks(batch_id="hb-20260301-api-fix", tasks=[
  {{"task_id": "api-fix", "title": "Convert API auth to async",
   "description": "Convert verify_token() in core/auth/manager.py (L45-60) to async. Blocking synchronous I/O is causing latency in FastAPI async handlers.",
   "context": "current_task.md: Investigating API response delays. verify_token blocks with synchronous I/O",
   "file_paths": ["core/auth/manager.py:45"],
   "acceptance_criteria": ["verify_token is async def", "existing tests pass"],
   "constraints": ["Do not change public API arguments or return values"]}}
])
```

Parallel tasks:

```
submit_tasks(batch_id="deploy-20260301", tasks=[
  {{"task_id": "lint", "title": "Run lint", "description": "Lint all files", "parallel": true}},
  {{"task_id": "test", "title": "Run tests", "description": "Execute unit tests", "parallel": true}},
  {{"task_id": "deploy", "title": "Deploy", "description": "Deploy after lint and test pass",
   "parallel": false, "depends_on": ["lint", "test"]}}
])
```

### Task Object

| Field | Required | Description |
|-------|----------|-------------|
| `task_id` | MUST | Unique task ID within the batch |
| `title` | MUST | Task title |
| `description` | MUST | Concrete work content (follow the writing principles above) |
| `parallel` | MAY | `true` for parallel execution (default: `false`) |
| `depends_on` | MAY | Array of predecessor task IDs |
| `context` | MAY | Background information (include relevant parts of current_task.md) |
| `file_paths` | MAY | Related file paths |
| `acceptance_criteria` | MAY | Completion criteria |
| `constraints` | MAY | Constraints |
| `reply_to` | MAY | Notification target on completion |

### Execution Rules

- Tasks with `parallel: true` and no pending dependencies run concurrently (within semaphore limit)
- Tasks with `depends_on` wait until all predecessors succeed
- Predecessor results are automatically injected into dependent task context
- If a predecessor fails, dependent tasks are skipped
- Cyclic dependencies are rejected at validation

### Forbidden Patterns

- ❌ "Refactor appropriately" (too vague)
- ❌ "Continue from last time" (executor has no conversation history)
- ❌ Instructions without file paths (executor would have to start by exploring)
- ❌ Empty context (executor makes poor decisions without background info)
- ❌ Manually creating JSON in `state/pending/` (always use `submit_tasks`)

### Task Results

Completed task results are saved to `state/task_results/{task_id}.json`.
Predecessor result summaries are automatically injected as context for dependent tasks.
