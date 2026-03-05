---
name: animaworks-guide
description: >-
  Complete reference for animaworks commands and CLI.
  Covers server operations (start/stop/restart/status/reset), chat and messaging (chat/send/board),
  Anima management (list/info/create/enable/disable/delete/restart/set-model/set-role/reload/audit),
  model information (models list/info/show), heartbeat/cron, logs/cost,
  task management (task add/update/list), config management (config get/set/list),
  RAG index management (index), asset operations (optimize-assets/remake-assets),
  animaworks-tool external tool execution,
  and background task monitoring (check_background_task/list_background_tasks).
  "command", "usage", "CLI", "animaworks", "start", "stop", "restart", "send",
  "create Anima", "change role", "change model", "status", "info",
  "model list", "model info", "logs", "cost", "task", "config",
  "background task", "task status"
---

# AnimaWorks CLI Complete Reference

All AnimaWorks operations are performed via the `animaworks` command.
This skill is a complete reference for all subcommands, arguments, and examples.

For operational concepts and rules, see `common_knowledge/`:
- Messaging rules → `communication/messaging-guide.md`
- Task management → `operations/task-management.md`
- Tool system → `operations/tool-usage-overview.md`
- Organization structure → `organization/structure.md`
- Model selection and config → `operations/model-guide.md`

---

## Server Operations (Generally Avoid)

```bash
animaworks start                         # Start server (default: 0.0.0.0:18500)
animaworks start --port 8080             # Specify port
animaworks start --foreground            # Foreground mode (for debugging)
animaworks stop                          # Stop server
animaworks restart                       # Restart server
animaworks status                        # Check system state (processes, Anima list)
animaworks reset                         # Delete runtime directory + reinitialize
animaworks reset --restart               # Reset then auto-start server
```

---

## Anima Management (anima subcommand)

### List, Status, and Info

```bash
animaworks anima list                    # List all Anima (name, enabled, model, supervisor)
animaworks anima list --local            # Scan filesystem directly (no API)
animaworks anima status                  # All Anima process states (State, Model, PID, Uptime)
animaworks anima status {name}           # Specific Anima process state
animaworks anima info {name}             # Detailed config (model, role, credential, voice, etc.)
animaworks anima info {name} --json      # JSON output
```

`anima info` output includes:
- Anima name, Enabled, Role, Model, Execution Mode
- Credential, Fallback Model, Max Turns, Max Chains
- Context Threshold, Max Tokens, LLM Timeout
- Thinking settings, Supervisor, Mode S Auth
- Voice settings (tts_provider, voice_id, speed, pitch)

### Create

```bash
# Create from character sheet (MD) — recommended
animaworks anima create --from-md {file} [--role {role}] [--name {name}]

# Create from template
animaworks anima create --template {template_name} [--name {name}]

# Create blank
animaworks anima create --name {name}
```

### Enable, Disable, Delete

```bash
animaworks anima enable {name}           # Enable (return from pause)
animaworks anima disable {name}          # Disable (pause)
animaworks anima delete {name}           # Delete (after ZIP archive)
animaworks anima delete {name} --no-archive  # Delete without archive
animaworks anima delete {name} --force   # Delete without confirmation
animaworks anima restart {name}          # Restart process
animaworks anima audit {name}            # Comprehensive audit of subordinate's recent activity (default: 1 day)
animaworks anima audit {name} --days 7   # Audit last 7 days
```

### Change Model

```bash
animaworks anima set-model {name} {model_name}
animaworks anima set-model {name} {model_name} --credential {credential_name}
animaworks anima set-model --all {model_name}   # Change all Anima at once
```

Requires `anima restart {name}` when server is running.

### Change Role

```bash
# Change role (reapply template + auto restart)
animaworks anima set-role {name} {role}

# Change status.json role field only (do not touch template)
animaworks anima set-role {name} {role} --status-only

# Update files only, no restart
animaworks anima set-role {name} {role} --no-restart
```

Files updated by set-role:
- `status.json` — role, model, max_turns updated from role template defaults
- `specialty_prompt.md` — replaced with role-specific guidelines
- `permissions.md` — replaced with role-specific tool and command permissions

Valid roles: `engineer`, `researcher`, `manager`, `writer`, `ops`, `general`

### Hot Reload

```bash
animaworks anima reload {name}           # Reload model config from status.json (no process restart)
animaworks anima reload --all            # Reload all Anima
```

---

## Model Information (models subcommand)

```bash
animaworks models list                   # List known models (name, mode, context window, note)
animaworks models list --mode S          # Filter by execution mode (S/A/B/C)
animaworks models list --json            # JSON output
animaworks models info {model_name}      # Resolved info (execution mode, context window, threshold, source)
animaworks models show                   # Show current models.json contents
animaworks models show --json            # Raw JSON output
```

Details → `common_knowledge/operations/model-guide.md`

---

## Chat and Messaging

```bash
# Chat with an Anima (human → Anima)
animaworks chat {name} "message"
animaworks chat {name} "message" --from {sender_name}
animaworks chat {name} "message" --local  # Direct execution (no API)

# Send message between Animas
animaworks send {sender} {recipient} "message"
animaworks send {sender} {recipient} "message" --intent delegation
animaworks send {sender} {recipient} "message" --reply-to {message_id}
animaworks send {sender} {recipient} "message" --thread-id {thread_id}

# Manual heartbeat trigger
animaworks heartbeat {name}
animaworks heartbeat {name} --local      # Direct execution (no API)
```

---

## Board (Shared Channels)

```bash
animaworks board read {channel}                         # Read channel messages
animaworks board read {channel} --limit 50              # Specify max messages
animaworks board read {channel} --human-only            # Human messages only
animaworks board post {sender} {channel} "text"         # Post to channel
animaworks board dm-history {self} {peer}               # Get DM history
animaworks board dm-history {self} {peer} --limit 50    # Specify max messages
```

---

## Config Management (config subcommand)

```bash
animaworks config list                   # List all config values
animaworks config list --section system  # Filter by section
animaworks config list --show-secrets    # Show API key values
animaworks config get {key}              # Get specific value (dot notation: system.log_level)
animaworks config get {key} --show-secrets
animaworks config set {key} {value}      # Set a config value
animaworks config export-sections        # Export to template files
animaworks config export-sections --dry-run
```

---

## Log Viewing (logs)

```bash
animaworks logs {name}                   # View specific Anima logs
animaworks logs --all                    # View server + all Anima logs
animaworks logs {name} --lines 100       # Specify line count (default: 50)
animaworks logs {name} --date 20260301   # View logs for specific date
```

---

## Cost Tracking (cost)

```bash
animaworks cost                          # Token usage and cost for all Anima
animaworks cost {name}                   # Specific Anima cost
animaworks cost --today                  # Today only
animaworks cost --days 7                 # Last 7 days (default: 30)
animaworks cost --json                   # JSON output
```

---

## Task Management (task subcommand)

```bash
animaworks task list                     # List tasks
animaworks task list --status pending    # Filter by status (pending/in_progress/done/cancelled/blocked)
animaworks task add --assignee {name} --instruction "task description"
animaworks task add --assignee {name} --instruction "desc" --source human --deadline 2026-03-10T18:00:00
animaworks task update --task-id {id} --status done
animaworks task update --task-id {id} --status done --summary "completion summary"
```

---

## RAG Index Management

```bash
animaworks index                         # Incremental index update for all Anima
animaworks index --anima {name}          # Specific Anima only
animaworks index --full                  # Full reindex of all data
animaworks index --dry-run               # Preview changes only (no execution)
```

---

## Asset Operations

### Optimize Assets

```bash
animaworks optimize-assets                              # Optimize 3D assets for all Anima
animaworks optimize-assets --anima {name}               # Specific Anima only
animaworks optimize-assets --dry-run                    # Preview only
animaworks optimize-assets --simplify                   # Mesh simplification
animaworks optimize-assets --texture-compress           # Texture compression
animaworks optimize-assets --texture-resize 512         # Texture resize
```

### Remake Assets

```bash
animaworks remake-assets {name} --style-from {reference}   # Regenerate assets with style transfer
animaworks remake-assets {name} --style-from {ref} --steps portrait,fullbody
animaworks remake-assets {name} --style-from {ref} --dry-run
animaworks remake-assets {name} --style-from {ref} --no-backup
```

---

## External Tool Execution (animaworks-tool)

Commands for Anima to use external services (Slack, Gmail, GitHub, etc.).

```bash
# Show help
animaworks-tool {tool_name} --help

# Execute
animaworks-tool {tool_name} {subcommand} [args...]

# Background execution (for long-running tools)
animaworks-tool submit {tool_name} {subcommand} [args...]
```

### Examples

```bash
animaworks-tool web_search query "AnimaWorks framework"
animaworks-tool slack send --channel "#general" --text "Good morning"
animaworks-tool github issues --repo owner/repo
animaworks-tool submit image_gen pipeline "1girl, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR
```

For submit details → `common_knowledge/operations/background-tasks.md`

### Background Task Monitoring (Anima Internal Tools)

To check progress of tasks submitted via `submit`, use these internal tools:
- `list_background_tasks` — List running and completed tasks
- `check_background_task(task_id)` — Get status and result for a specific task

These are not CLI commands but MCP tools used by Anima during conversation.

---

## Initialization and Migration

```bash
animaworks init                          # Initialize runtime directory (~/.animaworks/)
animaworks init --force                  # Re-initialize, overwriting existing config
animaworks init --skip-anima             # Skip Anima creation
animaworks migrate-cron                  # Convert cron.md from Japanese format to standard cron
```

---

## Global Options

```bash
animaworks --gateway-url http://host:port {command}   # Specify server URL
animaworks --data-dir /path/to/data {command}         # Specify runtime directory
```
