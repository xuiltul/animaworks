## How Task Execution Works

### Task Tool Auto-Routing (S-mode)

When you use the Task tool, the framework automatically routes based on your org structure.

**With subordinates** → Immediately delegated to a subordinate
- Include a subordinate's name in the description to assign them directly
  Example: "Have alice run the API tests"
  Example: "bob handles the code review"
- If no name is given, the least-loaded subordinate with the best role match is auto-selected
- Falls back to state/pending/ if all subordinates are disabled

**Without subordinates** → Runs immediately as a parallel sub-agent
- The sub-agent shares your identity, tools, and MCP servers
- It cannot access your conversation history or short-term memory

### Task Tool vs delegate_task

| Method | Use case | Subordinate selection |
|--------|----------|----------------------|
| Task tool | Quick delegation (auto-routed) | Include name in description |
| delegate_task | Explicit delegation (with deadline and detailed instructions) | Specify via name parameter |

Use delegate_task for delegation from paths that don't have the Task tool (Heartbeat, Inbox, etc.).

### Writing to state/pending/ (for later self-execution)

When writing tasks to state/pending/, follow these principles.
The executor (TaskExec) runs as a sub-agent. It shares your identity, behavior guidelines, memory directories, and organization info, but **cannot access your conversation history, short-term memory, or Priming results**.

### Context Injection Principles
- **Always include file paths and line numbers**: The executor can search memory, but specifying exact locations ensures it reaches the right files
- **Include current work state**: Copy relevant parts of current_task.md into the `context` field (auto-injected but explicit additions improve accuracy)
- **State the "why"**: Without background and purpose, the executor may make incorrect decisions

### Required Fields
- **What to do**: Concrete work (e.g., "Convert verify_token() in core/auth/manager.py to async" instead of "do refactoring")
- **Why**: Background and purpose (1–2 sentences)
- **Where to look**: Related file paths and line numbers
- **Completion criteria**: What counts as "done"
- **Constraints**: Prohibitions, compatibility requirements

### Task File Format
Create a JSON file in the state/pending/ directory in the following format:

```json
{{
    "task_type": "llm",
    "task_id": "YYYYMMDD-short-description",
    "title": "Task title",
    "submitted_by": "Your Anima name",
    "submitted_at": "ISO8601 current time",
    "description": "Concrete work content",
    "context": "Background information (include relevant parts of current_task.md)",
    "acceptance_criteria": ["Criterion 1", "Criterion 2"],
    "constraints": ["Constraint 1"],
    "file_paths": ["path/to/file.py:line_number"],
    "reply_to": "Sender name or null",
    "priority": 1
}}
```

### Forbidden Patterns
- ❌ "Refactor appropriately" (vague)
- ❌ "Continue from last time" (executor has no conversation history)
- ❌ Instructions without file paths (executor would have to start by exploring)
- ❌ Empty context field (executor makes poor decisions without background info)

## Parallel Task Execution (plan_tasks)

Submit multiple tasks with dependencies as a batch; independent tasks run in parallel.

### plan_tasks Tool

```
plan_tasks(batch_id="deploy-20260301", tasks=[
  {{"task_id": "lint", "title": "Run lint", "description": "Lint all files", "parallel": true}},
  {{"task_id": "test", "title": "Run tests", "description": "Execute unit tests", "parallel": true}},
  {{"task_id": "deploy", "title": "Deploy", "description": "Deploy after lint and test pass",
   "parallel": false, "depends_on": ["lint", "test"]}}
])
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `batch_id` | MUST | Unique batch identifier |
| `tasks` | MUST | Array of task objects (see below) |

#### Task Object

| Field | Required | Description |
|-------|----------|-------------|
| `task_id` | MUST | Unique task ID within the batch |
| `title` | MUST | Task title |
| `description` | MUST | Concrete work content |
| `parallel` | MAY | `true` for parallel execution (default: `false`) |
| `depends_on` | MAY | Array of predecessor task IDs |
| `context` | MAY | Background information |
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

### When to Use

- Single task → write directly to `state/pending/`
- Multiple independent tasks → use `plan_tasks` with `parallel: true`
- Tasks with dependencies → use `plan_tasks` with `depends_on`

### Task Results

Completed task results are saved to `state/task_results/{task_id}.json`.
Predecessor result summaries are automatically injected as context for dependent tasks.
