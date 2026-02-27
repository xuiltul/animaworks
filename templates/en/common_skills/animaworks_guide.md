---
name: animaworks-guide
description: >-
  Quick reference for animaworks commands and CLI.
  Covers server start/stop, chat/send, Anima management (list/create/enable/disable/delete/restart/set-model/set-role),
  manual Heartbeat trigger, RAG index management, and animaworks-tool execution.
  "command", "usage", "CLI", "animaworks", "start", "stop", "restart", "send",
  "create Anima", "change role", "change model", "status", "index"
---

# AnimaWorks CLI Quick Reference

All AnimaWorks operations are performed via the `animaworks` command.
This skill is a reference for command syntax, arguments, and examples.

For operational concepts and rules, see `common_knowledge/`:
- Messaging rules → `communication/messaging-guide.md`
- Task management → `operations/task-management.md`
- Tool system → `operations/tool-usage-overview.md`
- Organization structure → `organization/structure.md`

---

## Server Operations (Generally Avoid)

```bash
animaworks start              # Start server
animaworks stop               # Stop server
animaworks restart            # Restart server
animaworks status             # Check system state (processes, Anima list)
```

---

## Anima Management (anima subcommand)

### List and Status

```bash
animaworks anima list                    # List all Anima (with role)
animaworks anima status                  # All Anima process states
animaworks anima status {name}           # Specific Anima process state
```

### Create

```bash
# Create from character sheet (MD) — recommended
animaworks create-anima --from-md {file} [--role {role}] [--name {name}]

# Create from template
animaworks create-anima --template {template_name} [--name {name}]

# Create blank
animaworks create-anima --name {name}
```

### Enable, Disable, Delete

```bash
animaworks anima enable {name}           # Enable (return from pause)
animaworks anima disable {name}          # Disable (pause)
animaworks anima delete {name}           # Delete (after ZIP archive)
animaworks anima restart {name}          # Restart process
```

### Change Model

```bash
animaworks anima set-model {name} {model_name}
animaworks anima set-model {name} {model_name} --credential {credential_name}
animaworks anima set-model --all {model_name}   # Change all Anima at once
```

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

---

## RAG Index Management

```bash
animaworks index rebuild                 # Rebuild index for all Anima
animaworks index rebuild --name {name}   # Rebuild for specific Anima only
animaworks index status                  # Check index status
```

---

## Initialization

```bash
animaworks init                          # Initialize runtime directory (~/.animaworks/)
animaworks init --force                  # Re-initialize, overwriting existing config
```
