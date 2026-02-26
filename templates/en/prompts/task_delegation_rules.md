## Task Delegation Rules

When writing tasks to state/pending/, describe them **assuming the executor has no memory or context**.
The executor runs as a sub-agent and cannot access your knowledge, memory, or organization information.

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
    "context": "Background information",
    "acceptance_criteria": ["Criterion 1", "Criterion 2"],
    "constraints": ["Constraint 1"],
    "file_paths": ["path/to/file.py:line_number"],
    "reply_to": "Sender name or null",
    "priority": 1
}}
```

### Forbidden Patterns
- ❌ "Refactor appropriately" (vague)
- ❌ "Continue from last time" (executor has no history)
- ❌ Instructions without file paths (executor would have to start by exploring)
