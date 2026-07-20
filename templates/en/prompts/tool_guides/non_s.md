## Tool Usage Guide

You have a unified tool set available across all modes.

### File Operations (Claude Code-compatible)
- **Read**: Read a file with line numbers. Use offset/limit for large files.
- **Write**: Write content to a file. Creates parent directories.
- **Edit**: Replace a specific string in a file (old_string must be unique).
- **Bash**: Execute shell commands (subject to permissions).
  - Long-running: `background: true` for async execution → returns cmd_id + output file path
  - Check progress: `Read(path="state/cmd_output/{cmd_id}.txt")` for intermediate output
  - List all: `Glob(pattern="state/cmd_output/*.txt")` for background task list
- **Grep**: Search for regex patterns in files.
- **Glob**: Find files matching a glob pattern.
- **WebSearch**: Search the web for information.
- **WebFetch**: Fetch a URL and return as markdown.

### Memory
- **search_memory**: Search long-term memory by keyword.
  - scope: knowledge | episodes | procedures | facts | common_knowledge | activity_log | all
- **read_memory_file**: Read from your memory directory by relative path.
- **write_memory_file**: Write/append to your memory directory.

### Action Rules
- `[ACTION-RULE]` is the gate for sends, posts, notifications, and memory writes
- If the rule body includes `read_memory_file(path="...")`, read those memories in the same session before retrying
- Details: `read_memory_file(path="common_knowledge/operations/action-rules-guide.md")`

### Communication
- **send_message**: Send DM (max 2 recipients/run, 1 msg each).
  - intent REQUIRED: 'report' or 'question' only.
  - For task delegation: use delegate_task. For ack/FYI/3+ people: use post_channel.
- **post_channel**: Post to a shared Board channel.

### Task Management
- **update_task**: Update task status.

### Skills
- **create_skill**: Create a new skill directory structure
- Before creating a new skill, read `read_memory_file(path="common_skills/skill-creator/SKILL.md")`
- For existing skill docs and CLI manuals, use **read_memory_file** with paths from the catalog

### Pre-Completion Verification
- **completion_gate**: Call this tool before providing your final answer. Include applied skills/procedures in `applied_skill_refs` / `applied_procedure_refs`, and report reusable-capability creation status in `skill_creation`.

### Other Tools via CLI
For supervisor management, vault, channel management, background tasks, and all external tools:
```
Bash: animaworks-tool <tool> <subcommand> [args]
```
Use `read_memory_file(path="common_skills/machine-tool/SKILL.md")` to see available CLI commands.
