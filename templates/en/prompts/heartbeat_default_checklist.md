- **MUST**: Check if current_task.md has a task in progress. If so, verify its status. Always check before deciding "idle," "waiting," or "HEARTBEAT_OK"
- **MUST**: Check task queue for STALE tasks (marked ⚠️ STALE). For overdue or near-deadline tasks, follow up with the assignee (send_message) or escalate to supervisor. Never return HEARTBEAT_OK while STALE tasks exist
- **MUST**: Check for waiting tasks ("awaiting reply", "pending approval", etc.) stalled for 24+ hours. If stalled, send a status check or reminder
- Board check: `read_channel(channel="general", limit=5)` to see latest posts. **Respond only to posts where you are assigned** or `@all`. Do **not** post praise/acknowledgment replies ("great," "understood," etc.) to others' reports
- Whether you can access external tools you should use (if not, report to your supervisor via send command)
- Whether the task in progress has any blockers
- Whether state/pending/ has unexecuted tasks

### Blocker Reporting (MUST)

When any of the following occurs during task execution, report to the requester immediately.
Do not leave it in a "waiting" state.

- File/directory not found
- Insufficient permissions to access
- Prerequisites not met
- Technical issue interrupted the work
- Instructions unclear and cannot decide

Report to: Requester (send_message)
Critical blockers (30+ minute delay expected): Also notify human via call_human

- Only if ALL checks above have no actionable items: HEARTBEAT_OK
