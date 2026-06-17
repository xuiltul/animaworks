# Prevent Anima Startup Timeouts from RAG Vector Backpressure

Date: 2026-06-17

## Summary

`sakura` can reach the internal runner-ready point, then fail supervisor startup because immediate inbox processing blocks the child event loop in synchronous memory/RAG vector calls. The supervisor waits for IPC `ping` readiness for 120 seconds, kills the process, retries three times, and eventually reports `sakura` as failed.

This issue hardens startup so an Anima can become supervisor-ready before autonomous inbox/scheduler work begins, and hardens RAG/vector integration so slow vector worker calls do not make IPC readiness unresponsive.

## Evidence

- Runtime status showed `sakura` in `error` with no PID after three startup retries.
- Supervisor log recorded `Failed to start process sakura: Anima 'sakura' not ready within 120.0s`.
- Latest retry sequence on 2026-06-17 JST:
  - 15:18:40: attempt 1 timed out, PID `1724748` killed.
  - 15:21:11: attempt 2 timed out, PID `1762545` killed.
  - 15:23:42: attempt 3 timed out, PID `1807462` killed.
  - 15:23:43: respawn failed after `3/3`.
- `sakura` logs showed each retry reaching `Anima process ready: sakura`, then immediately entering message-triggered inbox processing for one unread `rin` message.
- The inbox path called memory episode append and RAG indexing, then vector worker calls such as `/list-collections`, `/create-collection`, and `/get-by-metadata` timed out or returned 503.
- Direct `sakura` Chroma integrity check returned `ok`; visible failure was vector worker backlog/unavailability rather than current `sakura` sqlite corruption.
- Other Animas had corrupted vector stores in runtime logs, which can amplify global vector worker backlog.
- Commit `ce3df064` introduced per-request vector worker store reset in `core/memory/rag/vector_worker.py`, which increases Chroma lifecycle churn because `reset_vector_store()` closes native store siblings.

## Root Cause

Startup readiness is currently coupled to early autonomous work:

1. `AnimaRunner` sets its ready event, then starts scheduler and inbox watcher work in the same child event loop.
2. The supervisor waits for IPC `ping` to return `ok`.
3. Message-triggered inbox processing can begin before the parent observes readiness.
4. The inbox path writes episodes and synchronously indexes memory through RAG/vector calls.
5. The synchronous HTTP vector client can block the child event loop long enough for IPC readiness polling to time out.
6. The supervisor kills and retries the process even though initialization itself already completed.

## Requirements

### R1. Supervisor-ready handshake gates autonomous work

- Add a startup acknowledgement path from parent supervisor to child runner.
- The child runner must set internal readiness and respond `ok` to IPC readiness checks before starting autonomous work.
- The parent process must send startup ack after `_wait_for_ready()` succeeds and before reporting the process as running.
- The child runner must not start inbox watcher, scheduler, or pending-task watcher until it receives startup ack.
- If startup ack fails, startup must fail explicitly instead of silently starting partially.

### R2. Async inbox processing must not block the event loop on memory/RAG indexing

- In async inbox processing, episode append and its RAG indexing side effects must run off the event loop.
- Slow or failing vector worker calls must not prevent IPC `ping` from being served.
- Existing episode write behavior and logging semantics must remain intact.

### R3. Vector worker store reset must not run after every successful operation

- Successful vector worker actions must keep the cached store alive.
- Unexpected vector worker action exceptions must reset only the affected Anima store.
- Existing self-heal behavior for known Chroma errors must continue to run.

### R4. Tests

- Add unit coverage for the startup ack gate: autonomous watchers do not start before ack and do start after ack.
- Add unit coverage that the parent sends startup ack after readiness succeeds.
- Add unit coverage that async inbox episode append is offloaded from the event loop.
- Add unit coverage that vector worker action success does not reset the store and unexpected exceptions do reset the target store.
- Add an integration-style regression test for startup with immediate inbox work so supervisor readiness is not blocked by slow memory/RAG work.

## Non-Goals

- Do not delete, rewrite, or destructively repair existing runtime vector DBs.
- Do not redesign vector worker process topology.
- Do not change `sakura` model configuration.
- Do not delete unread messages or change mailbox contents.

## Acceptance Criteria

- `sakura`-style startup cannot be failed merely because immediate inbox RAG work is slow after initialization.
- IPC `ping` remains responsive during slow inbox episode indexing.
- Vector worker successful actions no longer trigger `reset_vector_store()` in the worker request wrapper.
- Focused tests for the changed code pass.
- The full implementation is reviewed through the worktree review phase before merge.
