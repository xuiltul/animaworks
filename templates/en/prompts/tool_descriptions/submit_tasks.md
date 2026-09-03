[IMPORTANT] Tasks submitted with this tool run in your own TaskExec (not sent to subordinates). Use delegate_task for subordinates.
[Read before write (MUST)] Before submitting, read `list_tasks` (pending / in_progress) and check that no task for the same PR / Issue / target already exists. If one exists, do not submit; continue the existing task.
[Read after write (MUST)] After submitting, read it back with `list_tasks` and confirm registration and the summary (prefixed with the target, e.g. `[PR #N]`).
