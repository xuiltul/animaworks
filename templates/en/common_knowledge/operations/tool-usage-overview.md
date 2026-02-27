# Tool Usage Overview

Reference for AnimaWorks tool system and usage by execution mode.
Use this to understand the tools you can use and how to call them correctly.

## Execution Mode and Tools

How you call tools and what is available depends on your execution mode.
Your mode is determined automatically from your model name in `status.json`.

| Mode | Target Models | How tools are called |
|------|---------------|----------------------|
| S (SDK) | `claude-*` | MCP tools (`mcp__aw__*`) + Claude Code built-ins + external tools via Bash |
| A (Autonomous) | `openai/*`, `google/*`, `vertex_ai/*`, etc. | LiteLLM function calling (tool name as-is) |
| B (Basic) | Small models like `ollama/*` | JSON text format |

### S-mode (Claude Agent SDK)

Three tool families:

1. **Claude Code built-ins**: Read, Write, Edit, Grep, Glob, Bash, git, etc. For file operations and commands
2. **MCP tools (`mcp__aw__*`)**: AnimaWorks-specific. Call with `mcp__aw__` prefix, e.g. `mcp__aw__send_message`
3. **External tools (`animaworks-tool`)**: Slack, Gmail, GitHub, etc. Run via Bash: `animaworks-tool <tool> <subcmd> [args...]`

### A-mode (LiteLLM)

Two tool families:

1. **Internal tools**: `send_message`, `search_memory`, `add_task`, etc. Call by name using function calling
2. **External tools**: Call `discover_tools(category="...")` to dynamically add tools for that category

### B-mode (Basic)

Only a limited tool set:

- Memory (search_memory, read_memory_file, write_memory_file)
- Communication (send_message, post_channel, read_channel)
- Skill (skill)
- External tools, supervisor tools, and task tools are not available

---

## Tool Categories

### Internal Tools (Always Available)

AnimaWorks internal features for all modes.

| Category | Tool | Description |
|----------|------|-------------|
| Memory | `search_memory` | Keyword search over long-term memory |
| Memory | `read_memory_file` | Read a memory file |
| Memory | `write_memory_file` | Write to a memory file |
| Communication | `send_message` | Send DM |
| Communication | `post_channel` | Post to Board channel |
| Communication | `read_channel` | Read Board channel |
| Communication | `read_dm_history` | Read DM history |
| Skill | `skill` | Fetch full text of skill/procedure |
| Outcome | `report_procedure_outcome` | Report procedure execution result |
| Outcome | `report_knowledge_outcome` | Report usefulness of knowledge |
| Notification | `call_human` | Notify human admin |

In S-mode these tools have the `mcp__aw__` prefix (e.g. `mcp__aw__send_message`).

### Task Tools (A/S-mode only)

| Tool | Description |
|------|-------------|
| `add_task` | Add task to task queue |
| `update_task` | Update task status |
| `list_tasks` | List tasks |

### Supervisor Tools (Anima with subordinates only, A-mode)

Organizational tools for Anima with subordinates. See `organization/hierarchy-rules.md` for details.

| Tool | Description |
|------|-------------|
| `org_dashboard` | Show process status of all subordinates in a tree |
| `ping_subordinate` | Check subordinate liveness (omit `name` for all) |
| `read_subordinate_state` | Read subordinate `current_task.md` and `pending.md` |
| `delegate_task` | Delegate task to subordinate |
| `task_tracker` | Track delegated tasks |
| `disable_subordinate` | Disable subordinate |
| `enable_subordinate` | Re-enable subordinate |
| `set_subordinate_model` | Change subordinate model |
| `restart_subordinate` | Restart subordinate process |

In S-mode only `disable_subordinate` and `enable_subordinate` are available via MCP.

### External Tools (Permission required)

Tools for external services. Only categories allowed in `permissions.md` under `tool_categories` are available.

Main categories: `slack`, `chatwork`, `gmail`, `github`, `aws_collector`, `web_search`, `x_search`, `image_gen`, `local_llm`, `transcribe`

---

## Using External Tools

### S-mode

1. **List categories**: Call `mcp__aw__discover_tools` with no args
2. **Details**: Call with category (e.g. `mcp__aw__discover_tools(category="slack")`)
3. **Run**: Use Bash: `animaworks-tool <tool> <subcmd> [args...]`
4. **Help**: `animaworks-tool <tool> --help` for subcommands and args

For long-running tools (e.g. image generation), use `submit` for async execution:
```bash
animaworks-tool submit <tool> <subcmd> [args...]
```

See `operations/background-tasks.md` for details.

### A-mode

1. **List categories**: Call `discover_tools()` with no args
2. **Enable**: `discover_tools(category="slack")` to enable a category
3. **Call**: Enabled tools are callable via normal function calling

### B-mode

External tools are generally not available. Ask your supervisor if you need them.

---

## Common Questions

### common_knowledge examples use a different mode than mine

Documents use A/B-mode style like `send_message(to="...", content="...")`.
If you are in S-mode, add the `mcp__aw__` prefix (e.g. `mcp__aw__send_message`).

### How to see what tools I have

- **All Anima**: Use `check_permissions` to see your permissions
- **S-mode**: `mcp__aw__discover_tools` for external tool categories
- **A-mode**: `discover_tools()` for external tool categories
- **Details**: `read_memory_file(path="permissions.md")`

### Tool returns an error

→ See "Tools don't work" in `troubleshooting/common-issues.md`
