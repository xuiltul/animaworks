# Tool Usage Overview

Reference for AnimaWorks tool system and usage by execution mode.
Use this to understand the tools you can use and how to call them correctly.

## Execution Mode and Tools

How you call tools and what is available depends on your execution mode.
Your mode is determined automatically from your model name in `status.json`.

| Mode | Target Models | How tools are called |
|------|---------------|----------------------|
| S (SDK) | `claude-*` | MCP tools (`mcp__aw__*`) + Claude Code built-ins + external tools via Bash |
| C (Codex) | `codex/*` | Via Codex CLI. Same tool system as S-mode |
| A (Autonomous) | `openai/*`, `google/*`, `vertex_ai/*`, etc. | LiteLLM function calling (tool name as-is) |
| B (Basic) | Small models like `ollama/*` | JSON text format (`{"tool": "name", "arguments": {...}}`) |

### S-mode (Claude Agent SDK)

Three tool families:

1. **Claude Code built-ins**: Read, Write, Edit, Grep, Glob, Bash, git, etc. For file operations and command execution
2. **MCP tools (`mcp__aw__*`)**: AnimaWorks-specific internal features
   - Memory & communication: `send_message`, `post_channel`, `read_channel`, `manage_channel`, `read_dm_history`
   - Tasks: `backlog_task`, `update_task`, `list_tasks`, `submit_tasks`
   - Notification & permissions: `call_human`, `search_memory`, `check_permissions`
   - Outcome tracking: `report_procedure_outcome`, `report_knowledge_outcome`
   - Skills: `skill`, `create_skill`
   - Supervisor: `disable_subordinate`, `enable_subordinate`, `set_subordinate_model`, `set_subordinate_background_model`, `restart_subordinate`, `org_dashboard`, `ping_subordinate`, `read_subordinate_state`, `delegate_task`, `task_tracker`, `audit_subordinate`
   - Background: `check_background_task`, `list_background_tasks`
   - Credentials: `vault_get`, `vault_store`, `vault_list`
3. **Bash + animaworks-tool**: External tools (chatwork, slack, gmail, web_search, etc.) — use `skill` to check CLI usage, then run with `animaworks-tool <tool> <subcommand>`. Long-running tools use `animaworks-tool submit` for async execution

### C-mode (Codex CLI)

Same tool system as S-mode. Executed via Codex CLI. Falls back to LiteLLM (Mode A) when not installed.

### A-mode (LiteLLM)

Two tool families:

1. **Internal tools**: `send_message`, `search_memory`, `read_file`, `execute_command`, `backlog_task`, `submit_tasks`, etc. Call by name using function calling. `refresh_tools` and `share_tool` can rescan personal and common tools
2. **External tools**: Look up usage via the `skill` tool and execute with `execute_command` running `animaworks-tool <tool> <subcommand>`

### B-mode (Basic)

Tools are invoked in JSON text format. Available tools:

- **Memory**: search_memory, read_memory_file, write_memory_file, archive_memory_file
- **Communication**: send_message, post_channel, read_channel, read_dm_history
- **File & search**: read_file, write_file, edit_file, execute_command, web_fetch, search_code, list_directory
- **Skill**: skill, create_skill
- **Outcome tracking**: report_procedure_outcome, report_knowledge_outcome
- **Tasks**: backlog_task, update_task, list_tasks, submit_tasks
- **Background**: check_background_task, list_background_tasks (when BackgroundTaskManager is configured)
- **Credentials**: vault_get, vault_store, vault_list
- **Notification**: call_human (when notification channels are configured)
- **External tools**: Categories permitted in permissions.md can be called via `use_tool` for structured invocation

refresh_tools, share_tool, and create_anima are not available (A/S-mode only).

---

## Tool Categories

### Internal Tools (Always Available)

AnimaWorks internal features available in all modes (some may be omitted depending on mode).

| Category | Tool | Description |
|----------|------|-------------|
| Memory | `search_memory` | Keyword search over long-term memory |
| Memory | `read_memory_file` | Read a memory file |
| Memory | `write_memory_file` | Write to a memory file |
| Memory | `archive_memory_file` | Move unused memory files to archive/ |
| Communication | `send_message` | Send DM (intent required: report / question). Use delegate_task for task delegation |
| Communication | `post_channel` | Post to Board channel (channel, text) |
| Communication | `read_channel` | Read Board channel |
| Communication | `manage_channel` | Create channel, add/remove members, get channel info |
| Communication | `read_dm_history` | Read DM history |
| Skill | `skill` | Fetch full text of skill/procedure |
| Skill | `create_skill` | Create skill in directory structure (A/B-mode) |
| Outcome tracking | `report_procedure_outcome` | Report procedure execution result |
| Outcome tracking | `report_knowledge_outcome` | Report usefulness of knowledge |
| Notification | `call_human` | Notify human admin |
| Permission check | `check_permissions` | View your permission list (A/B-mode) |
| Credentials | `vault_get`, `vault_store`, `vault_list` | Get, store, list credentials (A/B-mode) |

In S-mode, tools exposed via MCP have the `mcp__aw__` prefix.

### File & Search Tools (A/B-mode)

| Tool | Description |
|------|-------------|
| `read_file` | Read file with line numbers |
| `write_file` | Write to file |
| `edit_file` | Replace text within file |
| `execute_command` | Execute shell command (subject to allowlist in permissions.md) |
| `web_fetch` | Fetch content from URL |
| `search_code` | Search files with regex |
| `list_directory` | List directory with glob filter |

In S-mode, use Claude Code's Read / Write / Edit / Grep / Glob / Bash for equivalent operations.

### Task Management Tools (A/S-mode)

| Tool | Description |
|------|-------------|
| `backlog_task` | Add task to task queue |
| `update_task` | Update task status |
| `list_tasks` | List tasks |
| `submit_tasks` | Submit multiple tasks as DAG for parallel/sequential execution |

backlog_task, update_task, list_tasks, and submit_tasks are also available in B-mode.

### Background Task Tools (A/B/S-mode, when BackgroundTaskManager is configured)

| Tool | Description |
|------|-------------|
| `check_background_task` | Check execution status of specified task_id |
| `list_background_tasks` | List tasks that are running, completed, failed, or pending |

Use these to check status of long-running tools submitted via `animaworks-tool submit`.

### Tool Management Tools (A-mode only)

| Tool | Description |
|------|-------------|
| `refresh_tools` | Rescan personal and common tools |
| `share_tool` | Share personal tool to common_tools/ |

### Supervisor Tools (Anima with subordinates only)

Organizational tools automatically enabled for Anima with subordinates. See `organization/hierarchy-rules.md` for details.

| Tool | Description | S-mode MCP |
|------|-------------|------------|
| `disable_subordinate` | Suspend subordinate | ○ |
| `enable_subordinate` | Resume subordinate | ○ |
| `set_subordinate_model` | Change subordinate model | ○ |
| `set_subordinate_background_model` | Change subordinate background model | ○ |
| `restart_subordinate` | Restart subordinate process | ○ |
| `delegate_task` | Delegate task to subordinate | ○ |
| `org_dashboard` | Show process status of all subordinates in a tree | ○ |
| `ping_subordinate` | Check subordinate liveness | ○ |
| `read_subordinate_state` | Read subordinate current task | ○ |
| `task_tracker` | Track delegated task progress | ○ |
| `audit_subordinate` | Generate activity timeline or statistics summary for subordinates. Omit `name` for batch audit | ○ |

All supervisor tools are available via MCP in S-mode. Audit is also available via CLI: `animaworks anima audit {name} [--all] [--days N] [--mode report|summary]`.

### Admin Tools (Conditional)

| Tool | Description | Condition |
|------|-------------|-----------|
| `create_anima` | Create new Anima from character sheet | When holding skills/newstaff.md (A-mode) |

### External Tools (Permission and enablement required)

Tools for external services. Only categories permitted in the "External tools" section of `permissions.md` are available.

Main categories: `slack`, `chatwork`, `gmail`, `github`, `aws_collector`, `web_search`, `x_search`, `image_gen`, `local_llm`, `transcribe`, `google_calendar`, `notion`

---

## Using External Tools

### S-mode

1. **Check skill**: Use `skill` tool to check CLI usage (e.g. `skill("chatwork-tool")`)
2. **Execute**: Run `animaworks-tool <tool> <subcommand> [args...]` via Bash (e.g. `animaworks-tool chatwork send 123 "message"`)
3. **Long-running tools**: Execute asynchronously with `animaworks-tool submit <tool> <subcommand> [args...]`. Tasks are queued to `state/background_tasks/pending/`; on completion, notifications are written to `state/background_notifications/` and can be checked on the next heartbeat
4. **Help**: `animaworks-tool <tool> --help` for subcommands and arguments

Long-running tools (image generation, local LLM, speech transcription, etc.) must always be executed with `submit`. See `operations/background-tasks.md` for details.

### A-mode

1. **Look up skill**: `skill("slack-tool")` to get CLI usage
2. **Execute**: `execute_command("animaworks-tool slack post ...")` to run via CLI
3. **Long-running tools**: Submit with `animaworks-tool submit`, then use `check_background_task` / `list_background_tasks` to check status

### B-mode

1. **Check permissions**: Use `check_permissions` to see permitted categories
2. **Execute**: Permitted external tools can be called via `use_tool(tool_name="...", action="...", args={...})` for structured invocation
3. **Long-running tools**: Ask your supervisor (A/S-mode) or call them the same way if permitted

---

## Common Questions

### common_knowledge examples use a different mode than mine

Documents use A/B-mode style like `send_message(to="...", content="...", intent="...")`.
If you are in S-mode, add the `mcp__aw__` prefix when reading (e.g. `mcp__aw__send_message`).

### How to see what tools I have

- **A/B-mode**: Use `check_permissions` to see your permissions
- **S-mode**: Tools exposed via MCP have the `mcp__aw__` prefix. Read `permissions.md` for permission details
- **A-mode**: Use `skill` tool to look up external tool CLI usage
- **All modes**: `read_memory_file(path="permissions.md")` to check permitted content (S-mode: read directly with Claude Code's Read)

### Tool returns an error

→ See "Tools don't work" in `troubleshooting/common-issues.md`
