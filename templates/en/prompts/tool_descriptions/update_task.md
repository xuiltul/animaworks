Update a task's status. Use status='done' on completion and status='cancelled' on withdrawal. Record interim progress with status='in_progress' and a summary.
[Read before write] Before updating, read the current status / summary with `list_tasks` (another worker may already have set done / cancelled). If two or more tasks exist for the same target, continue the newer one and cancel the older.
[Read after write] After updating, confirm with `list_tasks`. Pushing the fix and replying is completion: write status='done' at that point.
