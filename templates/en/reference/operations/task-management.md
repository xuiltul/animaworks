# Task Management

## One authoritative task record

Use `list_tasks(detail=true)` or `animaworks-tool task list` to inspect tasks. The host owns the canonical TaskStore: instructions, dependencies, execution attempts, results, and delegation aliases share one durable record. Agents must not edit the database, fabricate execution claims, or repair task storage by writing files. Legacy `state/task_queue.jsonl` and `state/pending/` are migration/export evidence, not live task inputs; preserve them for operator-led migration.

Normal chat requests may be answered or executed directly. Register tasks only when background execution, parallel work, or durable follow-up is needed. Human-origin work has highest priority; supervisor requests precede peer requests at equal priority. Preserve original instructions, acceptance criteria, constraints, and relevant context when handing work off.

## Choose the execution path

- Inbox handles messages and lightweight replies.
- Heartbeat checks for meaningful changes and decides what needs action. Do not run long coding or bulk tool operations in Heartbeat; use `submit_tasks` for your own TaskExec or `delegate_task` for a direct subordinate.
- TaskExec executes durable submitted tasks with tools. The host claims eligible tasks and manages concurrency, dependencies, cancellation, and attempt recovery independently of periodic Heartbeat.
- Agent/Task sub-agent spawning tools are disabled. Use the task/delegation tools above.

## Submit and inspect

`submit_tasks` runs on **your own** TaskExec, not a subordinate's:

```
submit_tasks(batch_id="report-build", tasks=[
  {"task_id": "collect", "title": "Collect evidence", "description": "Collect the requested evidence with source references.", "parallel": true},
  {"task_id": "report", "title": "Write report", "description": "Use the evidence to write the requested report.", "depends_on": ["collect"]}
])
list_tasks(detail=true)
```

A new task needs `task_id`, `title`, and `description`. Optional fields include `context`, `acceptance_criteria`, `constraints`, `file_paths`, `workspace`, `parallel`, `depends_on`, `reply_to`, and `model`. `workspace` is a registered workspace alias; model routing otherwise follows runtime configuration. Preserve full inputs rather than replacing them with a brief summary.

Submission validates the batch and publishes task plus execution input atomically. Re-delivery of an already submitted task is idempotent; it is not a retry. Dependencies must complete successfully before dependent work runs. Do not infer eligibility from a missing file or a `pending` label alone.

## Declare outcomes and resume deliberately

```
update_task(task_id="TASK_ID", status="done", summary="Verified result", result="Evidence and output locations")
update_task(task_id="TASK_ID", status="pending", summary="Waiting for the specified input")
update_task(task_id="TASK_ID", status="cancelled", summary="Reason this work is no longer needed")
```

Only the host sets `in_progress` when claiming execution. Use it as a read-only status, not an `update_task` command. A task whose attempt ended without a completion declaration may remain pending with an attention reason; pending does not promise an automatic retry. Do not duplicate it under a new ID merely to restart it.

After resolving the reason for interruption, explicitly resume the same nonterminal task:

```
submit_tasks(batch_id="resume-report", tasks=[{"task_id": "TASK_ID", "resume": true}])
```

This reuses stored instructions and preserves history. A live attempt cannot be resumed, and terminal tasks cannot be reopened this way. If a dependency is cancelled or needs attention, inspect the details and ask the requester or cancel work that is no longer applicable; do not invent success.

When blocked, report facts, attempts, missing authority/information, and a concrete next step. Do not repeat the same unsuccessful action. Search relevant knowledge when useful, not as a mandatory ritual. You may work on other authorized tasks while waiting. Report completed delegated work to the requester; avoid duplicate notifications and unnecessary acknowledgements.

## Delegation

```
delegate_task(name="dave", instruction="Run the API test and report verified results", summary="API test")
task_tracker()
```

Delegation creates one task owned by the subordinate and an alias visible to the supervisor. Both views reflect the same status immediately; no separate supervisor ledger or heartbeat synchronization is needed. `task_tracker(status="all")` includes terminal work; `status="completed"` selects done/cancelled. Ask the delegator if instructions are unclear and report the result when complete.

## Working context and results

`state/current_state.md` holds concise observations, context, plans, and blockers, not a duplicate task list or permanent procedures. Use `status: idle` when there is no active context. It is preserved across normal session boundaries; prompt display and disk cleanup have separate limits (see `anatomy/working-memory.md`).

TaskExec stores result summaries under `state/task_results/{task_id}/{attempt_token}.md`. The host links the accepted attempt; dependent tasks receive that result, not an arbitrary stale file. Do not manufacture result files or infer completion from their presence. Raw activity and episodes remain evidence; there is no obligation to manually duplicate every task transition into an episode.

## Long-running command tools remain separate

Use `animaworks-tool submit TOOL ...` for eligible long-running external tools (for example image generation or run_command). This is **not** `submit_tasks`: command descriptors still live in `state/background_tasks/pending/` and BackgroundTaskManager persists `state/background_tasks/{task_id}.json` with `running`, `completed`, or `failed`. Use `list_background_tasks` / `check_background_task` to inspect them. Preserve this file-based command pipeline and its notifications; see `operations/background-tasks.md`.
