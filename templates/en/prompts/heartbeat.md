This is a Heartbeat. Follow the process below.

## Observe
{checklist}

## Plan
Based on your observations, decide what to do next.

**Message quality check (MUST)**: Before sending delegation/report/escalation, verify required fields in `common_knowledge/communication/message-quality-protocol.md`

**[MUST] If you identify anything that requires action, you MUST formalize it as a task. "Acknowledged but no action taken" is prohibited.**
Use one of the following to create a concrete action:
- Delegate to subordinates → `delegate_task` (**read before write**: first read `list_tasks(status="delegated")` / `list_tasks(status="in_progress")`; if an open task for the same PR / Issue / target exists, do not create a new one — `send_message` the assignee with the existing task_id and the extra instruction. **Read after write**: read the new task back with `list_tasks`)
- Do it yourself → record the next action in `state/current_state.md`; do not start hands-on work during normal Heartbeat
- Immediate follow-up → `send_message` / `call_human`

### Checklist
- Background task results: Check state/task_results/ for completed tasks and follow up as needed
- **MUST**: If recent chat/inbox messages contain unhandled instructions from humans or Animas, concretize them via direct handling, `delegate_task`, `send_message`, `call_human`, or `state/current_state.md`
- STALE / near-deadline tasks: Follow up with assignee (send_message), escalate to supervisor if needed
- Long-stalled waiting tasks (24h+): Send status check or reminder
- If there is a blocker: report only (send_message / call_human)
- Only if ALL checks have no actionable items: HEARTBEAT_OK

**Important: Do not perform actual work (code changes, file edits, research, etc.) in this phase.**
**Task execution is handled automatically in a separate session.**

**Delegation guidelines**: When using `delegate_task`, follow the writing principles and forbidden patterns in `read_memory_file(path="common_knowledge/operations/task-delegation-guide.md")` (MUST). Do not use `submit_tasks` during normal Heartbeat.

## Reflect
After completing the above observation and planning, state any insights or observations in the following format if you have them.
You may omit this if you have nothing to add.

[REFLECTION]
(Describe insights, observations, or pattern recognition here)
[/REFLECTION]
