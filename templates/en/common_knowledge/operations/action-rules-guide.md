# Action Rules

## Overview

Action rules are knowledge files that add a pre-action check before side-effect operations such as sending, posting, notifying, or writing memory. Put `[ACTION-RULE]` and `trigger_tools:` in `knowledge/action-rule-*.md`; matching rules are searched before the target tool runs.

## Basic Format

```markdown
## [ACTION-RULE] Rule name
trigger_tools: gmail_draft, gmail_send
keywords: email, draft, duplicate check
---
Before executing, read_memory_file(path="procedures/gmail-draft-check.md").
After completing the check, retry the same tool.
```

| Field | Required | Description |
|-------|----------|-------------|
| `trigger_tools` | Required | Target tool names, comma-separated |
| `keywords` | Optional | Search hints |
| Body | Required | Check text displayed when paused. Required files must be written as `read_memory_file(path="...")` |

## ToolHandler Action Names

- `call_human`
- `send_message`
- `post_channel`
- `write_memory_file`
- `gmail_draft`
- `gmail_send`
- `chatwork_send`
- `slack_send`
- `discord_send`

## CLI Mappings

| CLI | Action-rule name |
|-----|------------------|
| `animaworks-tool gmail draft` | `gmail_draft` |
| `animaworks-tool gmail send` | `gmail_send` |
| `animaworks-tool chatwork send` | `chatwork_send` |
| `animaworks-tool slack send` | `slack_send` |
| `animaworks-tool discord send` | `discord_send` |
| `animaworks-tool call_human` | `call_human` |

`animaworks-tool submit ...` is not an action-rule target. The actual queued subcommand is checked when it runs.

## Gate Behavior

- Rules below score `0.80` do not block.
- Search failure, missing vector store, and no matching rule are fail-open.
- If the body contains `read_memory_file(path="...")`, the gate blocks until all extracted paths have been read in the same action-gate session.
- Review-only rules without required reads block once per `tool:rule` in the same action-gate session.
- There is no global maximum-two-pauses limit.
- When paused, read the displayed rule, perform the required `read_memory_file` or checks, then retry the same operation.

## Examples

```markdown
## [ACTION-RULE] Duplicate check before Gmail draft
trigger_tools: gmail_draft, gmail_send
keywords: Gmail, draft, duplicate, thread
---
Before creating or sending a Gmail draft, read_memory_file(path="procedures/gmail-draft-check.md").
Check existing threads and drafts for duplicates before proceeding.
```

```markdown
## [ACTION-RULE] Verify before customer memory update
trigger_tools: write_memory_file
keywords: customer, client, profile
---
Before updating customer-related `knowledge/`, read the related existing files and verify there is no contradiction.
```

## Location

Create rules under `knowledge/action-rule-{topic}.md`. Before creating a new rule, search `knowledge/` for similar rules and update an existing one when appropriate.
