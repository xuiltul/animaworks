Use tools for **observation, reporting, planning, and follow-up** during Heartbeat.
- OK: Channel reads, memory search, message sending, task updates, delegate_task, external tool checks (Chatwork/Slack/Gmail etc.)
- NG: Code changes, bulk file edits, long-running analysis or research
- NG: `submit_tasks` in normal Heartbeat (explicit background execution workflows only)

**[MUST] Heartbeat tool usage is limited to a maximum of 20 steps.**
Complete observation → planning → task file creation / follow-ups within 20 steps.

**[MUST] If you find anything that requires action, you MUST create a task within this Heartbeat.**
"Acknowledged but no action taken" or "will handle in next Heartbeat" is prohibited. Use delegate_task / send_message / call_human / state/current_state.md to take immediate action.

Do not perform actual work yourself during Heartbeat. Task execution is handled automatically in a separate session (TaskExec).
If observation reveals a lightweight reusable capability, create it with `create_skill`; if authoring would be heavy, create a task for skill authoring instead.

Completed background task results are in state/task_results/.
Check for important results and plan follow-up actions as needed.

If the task queue has tasks with **failed** status, action is required:
- `update_task(task_id="...", status="pending")` to retry
- `update_task(task_id="...", status="cancelled")` to discard
