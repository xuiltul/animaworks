Add a new task to the task queue. Instructions from humans must be recorded with source='human'; delegation between Animas with source='anima'.
[Read before write (MUST)] Before adding, read `list_tasks` and check that no open task exists for the same PR / Issue / target. If one exists, do not add.
[Read after write (MUST)] After adding, read it back with `list_tasks` and confirm the summary (prefixed with the target, e.g. `[PR #N]`).
