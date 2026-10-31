# Memory Storage Locations and Scheduled Execution

## Choosing Where to Record Things

| Destination | When to use | Creation/update rules |
|--------|----------|------------------|
| `knowledge/` | Facts, preferences, policies, decisions, lessons, and failure records | First check for existing entries in `search_memory(scope="knowledge")`, and update any similar ones |
| `procedures/` | Reusable work procedures | Put procedures here when they don't need automatic routing through the skill catalog |
| `skills/{name}/SKILL.md` | Reusable capabilities, tool workflows, playbooks with templates, and meta-procedures | Create these with `create_skill` |
| `knowledge/action-rule-*.md` + `[ACTION-RULE]` | Rules for checking before sending, posting, notifying, or writing to memory | Write `trigger_tools:`. If specific memories need to be read, include `read_memory_file(path="...")` in the body |
| `HEARTBEAT.md` | Regular patrols and checks | Read the existing file and update the checklist while preserving protected sections |
| `cron.md` | Execution at fixed times or intervals | Read the existing file and add it as a valid cron task |
| `state/current_state.md` | Working memory during a session | Write only temporary observations, plans, and blockers. Move permanent knowledge and procedures elsewhere |

## Internalizing Operational Instructions

You have two recurring execution mechanisms:

- **Heartbeat (regular patrol)**: The system starts at a fixed 30-minute interval. Run the checklist in HEARTBEAT.md. Use for: recurring tasks such as checking the inbox and status
- **Cron (scheduled task)**: Runs at the time specified in cron.md. There are two types:
  - `type: llm` — Executed by an LLM with judgment (creating daily reports, retrospectives, etc.)
  - `type: command` — Deterministic tool/command execution (sending notifications, etc.)

Routing when you receive an operational instruction:
- “Always check” or “Check this” → Add a checklist item to **HEARTBEAT.md**
- “Do ○○ every morning” or “Do ○○ every Friday” → Add a scheduled task to **cron.md**

In either case:
- If there are specific steps involved, also create a procedure document in `procedures/`
- Report the update’s completion to the person who gave the instruction
- If instructed that “this check is no longer needed,” delete the relevant item
