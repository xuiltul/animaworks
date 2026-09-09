# Task submission and delegation

## Choose an execution path

Use the exposed task tools, not native Agent/Task subagent spawning.
Finish ordinary chat work directly. Use `backlog_task` for durable tracking only,
`submit_tasks` for your own background execution, and
`delegate_task(name="worker", instruction="Original request and acceptance criteria", summary="Summary")`
for an enabled direct subordinate. Respect tool availability and permissions; do not silently route
work to an unavailable worker or another execution path. Heartbeat is for decisions and submission;
long-running work belongs in TaskExec.

## Preserve the handoff

The executor does not automatically share the conversation history. Include the original request,
purpose, relevant files and known locations, current state, acceptance criteria, approval conditions,
and constraints. Do not invent paths or line numbers. Use `description`, `context`,
`acceptance_criteria`, `constraints`, and `file_paths` as appropriate. Preserve model and registered
workspace overrides. Do not direct a worker to write another Anima's personal directories.

`submit_tasks(batch_id="work", tasks=[{"task_id":"job","title":"Work","description":"Specific request"}])`
publishes the task and execution input atomically. Re-delivery of the same ID is not a retry.
`parallel:true` permits concurrency within the worker limit; `depends_on` waits for both predecessor
completion and the end of its attempt. A cancelled or unfinished dependency needs review, not an
invented success result.

## State, results, and explicit resume

Inspect `list_tasks(detail=true)` and delegated work through `task_tracker()`. A tracking ID is an alias
of the subordinate's canonical task; ledger synchronization and descriptor rescue are unnecessary.
`task_tracker(status="all")` includes all tasks; `status="completed"` selects done/cancelled.
The host owns claims and `in_progress`. Declare evidence-backed `done`, `pending` with a concrete
waiting reason, or `cancelled` for an explicit cancellation.

After an incomplete-attempt notification, check existing effects and results before deciding to
continue. Use `submit_tasks(batch_id="resume-job", tasks=[{"task_id":"job","resume":true}])` to reuse
the saved input. Active, done, and cancelled tasks cannot be resumed this way. Do not create endless
resubmission loops. Result summaries live in `state/task_results/{task_id}/{attempt_token}.md`;
the existence of a file is not proof of task completion.

## Duplicates and reporting

If unfinished work for the same request already exists, send additional context with its ID.
Do not automatically cancel or replace older work based only on a suspected duplicate; check ownership
and execution state. Keep required approvals and independent review. Report to the requester who
needs the result; forwarding the same report through every hierarchy level or maintaining a second
handwritten ledger is not mandatory. See `common_knowledge/anatomy/task-architecture.md`.
