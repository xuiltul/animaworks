### Task Recording in Heartbeat

- Inspect existing tasks and their attention reasons. Resume interrupted nonterminal work only after resolving the cause, using the existing task_id and `resume: true` in `submit_tasks`; do not resubmit all pending work or reconstruct stored input. Stay within the already-approved scope. Set `done` only after verifying evidence for every completion criterion
- Record delegation between Anima in the task queue and update relay_chain
- When a task is complete, update status via `update_task`
