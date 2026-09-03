## How Task Execution Works

### How to Delegate Tasks

> **Note**: Agent/Task tools (sub-agent spawning) are **disabled**. For task delegation, use `delegate_task`. `submit_tasks` is not shown in normal chat/Heartbeat/Inbox/TaskExec; use it only in explicit background execution workflows.

**With subordinates** → Use `delegate_task` to delegate to a subordinate
- Include a subordinate's name to assign them directly
- If no name is given, the least-loaded subordinate with the best role match is auto-selected

**Without subordinates** → Usually execute directly in this session. Use `submit_tasks` only when an explicit background execution workflow is enabled
- Written to state/pending/ and automatically executed by TaskExec in a separate session
- The executor shares your identity, injection, behavior rules, memory guide, and org context
- A task_id is returned. You will receive a DM notification when it completes
- You can check task results in Heartbeat (state/task_results/)

### Choosing the Right Task Tool

| Tool | Purpose | Execution Queue (Layer 1) | Tracking (Layer 2) | When to use |
|------|---------|--------------------------|--------------------|----|
| `submit_tasks` | Submit tasks for execution and registration | Creates in `state/pending/` | Registers in `task_queue.jsonl` | Explicit background execution workflow, handed to your own TaskExec |
| `delegate_task` | Delegate to subordinates | Creates in subordinate's `state/pending/` | Registers in both `task_queue.jsonl` | When assigning to subordinates |

**Important**: Do not use `submit_tasks` in normal chat after receiving human instructions. Execute directly here, and when follow-up tracking is needed, record it with `update_task`, `state/current_state.md`, or an explicit background execution workflow.

**[MUST] Do NOT manually create JSON files in `state/pending/`.** When an explicit background execution workflow exposes `submit_tasks`, submit through that tool.

## submit_tasks in Explicit Background Execution

Do not use `submit_tasks` in normal sessions. Use it only when the user or a skill explicitly requests background execution and `submit_tasks` is visible in the tool list. Even for one task, submit a tasks array with one item.

### About the Executor (TaskExec)

TaskExec runs as a sub-agent. It shares your identity, behavior guidelines, memory directories, and organization info, but **cannot access your conversation history, short-term memory, or Priming results**.

Therefore, including sufficient information in the task's `description` and `context` is critical.

### Description Writing Principles

- **Always include file paths and line numbers**: The executor can search memory, but specifying exact locations ensures it reaches the right files
- **Include current work state**: Copy relevant parts of current_state.md into the `context` field (auto-injected but explicit additions improve accuracy)
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
   "context": "current_state.md: Investigating API response delays. verify_token blocks with synchronous I/O",
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
| `context` | MAY | Background information (include relevant parts of current_state.md) |
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
- ❌ Trying to use `submit_tasks` in normal chat/Heartbeat/Inbox/TaskExec
- ❌ Manually creating JSON in `state/pending/` (when explicit background execution is enabled, use `submit_tasks`)
- ❌ Instructing writes to another Anima's directory (e.g. their `knowledge/`) — subordinates cannot write there. Use `common_knowledge/` for shared output

### Task Results

Completed task results are saved to `state/task_results/{task_id}.json`.
Predecessor result summaries are automatically injected as context for dependent tasks.

## Tracking Delegated Tasks

Use `task_tracker` to check delegated task progress.
It cross-checks the latest status from the subordinate's `task_queue.jsonl`.

```
task_tracker()                     # Active delegated tasks (default)
task_tracker(status="all")         # All including completed
task_tracker(status="completed")   # Completed only
```

| status | Meaning |
|--------|---------|
| `active` | In progress (anything other than done/cancelled/failed). Default |
| `all` | Everything |
| `completed` | Only done/cancelled/failed |

### Auto-sync (sync_delegated)

Runs automatically after each Heartbeat. Detects the following state changes in subordinate task queues and auto-updates the supervisor's tracking entries (`delegated` status):

- Subordinate `done` or `cancelled` → supervisor entry updated to `done`
- Subordinate `failed` → supervisor entry updated to `failed`
- Archived tasks are also searched (`task_queue_archive.jsonl`)

There is no need to manually call `task_tracker`, but it remains useful for immediate checks between Heartbeats.

## Read before write, read after write (MUST)

- **Read before write**: right before `delegate_task` / `submit_tasks` / `backlog_task`, read `list_tasks(status="delegated")` and `list_tasks(status="in_progress")` (plus pending for your own tasks) and check for an open task on the same PR / Issue / target. If one exists, do not create another: `send_message` the assignee with the existing task_id and the extra instruction.
- **Read after write**: read the returned task_id back with `list_tasks` and confirm the summary is prefixed with `[PR #N]` / `[Issue #N]` / the target name.
- **On duplicates**: keep the newer task and set the older one to `update_task(status="cancelled", summary="merged into <new id>")`. Assignees follow the same rule.
- Write only the target, URL, what to do and the completion target in a delegation. Do not bind the assignee with ledger state words or procedural conditions.
