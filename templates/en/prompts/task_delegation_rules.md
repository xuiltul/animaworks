## Task Delegation Rules

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
