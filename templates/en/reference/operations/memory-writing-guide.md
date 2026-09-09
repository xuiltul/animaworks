# Memory Destinations and Scheduled Execution

## Choosing Where To Record

| Destination | Use for | Authoring rule |
|-------------|---------|----------------|
| `knowledge/` | Facts, preferences, policies, decisions, lessons, failure records | Search `knowledge/` first and update a related file when one exists |
| `procedures/` | Repeatable step-by-step task execution | Use for procedures that do not need skill-catalog routing |
| `skills/{name}/SKILL.md` | Reusable capabilities, tool workflows, template-backed playbooks, meta-procedures | Create with `create_skill` |
| `knowledge/action-rule-*.md` + `[ACTION-RULE]` | Pre-action checks before send/post/notify/write tools | Include `trigger_tools:`. If a specific memory must be read first, include `read_memory_file(path="...")` in the body |
| `heartbeat.md` | Recurring periodic checks | Read the current file and update the checklist while preserving protected sections |
| `cron.md` | Scheduled work at fixed times | Read the current file and add a valid cron task entry |
| `state/current_state.md` | Session working memory | Store temporary observations, plans, and blockers only. Move durable knowledge or procedures elsewhere |

## Internalizing Work Instructions

You have two scheduled execution mechanisms:

- **Heartbeat (periodic sweep)**: Triggered by the system at fixed 30-minute intervals. Execute the checklist in heartbeat.md. Use for: inbox checks, status verification, and other recurring tasks
- **Cron (scheduled tasks)**: Executed at times specified in cron.md. Two types:
  - `type: llm` — LLM executes with judgment (daily reports, retrospectives, etc.)
  - `type: command` — Deterministic tool/command execution (sending notifications, etc.)

When you receive work instructions:
- "Always check" / "Monitor" → Add checklist items to **heartbeat.md**
- "Every morning do X" / "Every Friday do X" → Add scheduled tasks to **cron.md**

In either case:
- If concrete procedures are involved, also create procedures in `procedures/`
- Report completion to the person who gave the instruction
- If told "this check is no longer needed," remove the corresponding item
