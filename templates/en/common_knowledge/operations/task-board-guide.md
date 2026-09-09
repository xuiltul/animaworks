# TaskBoard and human-facing reports

## Authority and presentation

TaskBoard projects canonical tasks together with presentation metadata.
Use `list_tasks(detail=true)` / `task_tracker()` for state.
`state/current_state.md` is working context and `state/task_results/` holds attempt results,
not independent execution ledgers. Columns, snooze, and archive do not substitute for task outcomes
or cancellation.

## Avoid duplicate manual tracking

There is no requirement to rewrite `shared/task-board.md` after every delegation, completion, or
heartbeat. Produce a report when a human requests one, using the relevant canonical state and noting
its timestamp and uncertainties. Do not delete or weekly-reset existing reports without authorization.
A report's state must not republish a task.

## External sharing

Post or update Slack only when requested or covered by an existing authorized workflow.
Respect `slack_channel_post` / `slack_channel_update` permissions and approval conditions;
check company boundaries, confidentiality, and previous posts. Do not add automatic posts or alerts.
