This is a Heartbeat. Follow the process below.

## Observe
{checklist}

## Plan
Based on your observations, decide what to do next.

**[MUST] If you identify anything that requires action, you MUST formalize it as a task. "Acknowledged but no action taken" is prohibited.**
Use one of the following to create a concrete action:
- Delegate to subordinates → `delegate_task`
- Do it yourself later → `submit_tasks` (written to state/pending/, executed by TaskExec in a separate session)
- Register to task queue → `backlog_task`
- Immediate follow-up → `send_message` / `call_human`

### Checklist
- Background task results: Check state/task_results/ for completed tasks and follow up as needed
- **MUST**: If recent chat/inbox messages contain instructions from humans or Animas that are not yet in the task queue, register them with `backlog_task` (use source="human" for human instructions)
- STALE / near-deadline tasks: Follow up with assignee (send_message), escalate to supervisor if needed
- Long-stalled waiting tasks (24h+): Send status check or reminder
- If there is a blocker: report only (send_message / call_human)
- Only if ALL checks have no actionable items: HEARTBEAT_OK

**Important: Do not perform actual work (code changes, file edits, research, etc.) in this phase.**
**Task execution is handled automatically in a separate session.**

{task_delegation_rules}

## Reflect
After completing the above observation and planning, state any insights or observations in the following format if you have them.
You may omit this if you have nothing to add.

[REFLECTION]
(Describe insights, observations, or pattern recognition here)
[/REFLECTION]
