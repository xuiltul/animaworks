This is a Heartbeat. Follow the process below.

## Observe
{checklist}

## Plan
Based on your observations, decide what to do next.
- **MUST**: If recent chat/inbox messages contain instructions from humans or Animas that are not yet in the task queue, register them with `add_task` (use source="human" for human instructions)
- STALE / near-deadline tasks: Follow up with assignee (send_message), escalate to supervisor if needed
- Long-stalled waiting tasks (24h+): Send status check or reminder
- Tasks for subordinates: delegate via delegate_task
- Tasks for yourself to do later: **submit via plan_tasks tool** (automatically written to state/pending/ and executed by TaskExec in a separate session)
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
