# Code Review: Prevent Anima Startup Timeouts from RAG Vector Backpressure - Approved

Review Date: 2026-06-17
Original Issue: docs/issues/20260617_sakura-startup-rag-backpressure.md
Worktree: /home/main/dev/animaworks-bak-issue-20260617-155431
Status: APPROVED

## Summary

All blocking review checks passed. The implementation is ready for merge.

## Metrics

- Requirement Alignment: PASS
- Diff Coverage: 86% (minimum 80%)
- Code Quality: PASS
- SRP Compliance: PASS for changed responsibilities
- File Sizes: No new file bloat; touched production files were already over 500 lines before this change
- E2E Tests: PASS for targeted E2E smoke coverage
- Regression: PASS for focused and compared checks; unrelated existing/environment failures documented below

## Requirement Alignment

- R1 startup acknowledgement gate: implemented in `core/supervisor/process_handle.py` and `core/supervisor/runner.py`.
- R2 async inbox episode append off-loop: implemented in `core/_anima_inbox.py`.
- R3 vector worker store reset no longer runs after successful actions: implemented in `core/memory/rag/vector_worker.py`.
- R4 tests: unit and integration-style regression coverage added in `tests/unit/test_runner_startup.py`, `tests/unit/test_process_handle_ready.py`, `tests/unit/core/test_inbox_delegation_handling.py`, and `tests/unit/core/memory/test_vector_worker_app.py`.

## Verification

- `uv run ruff check ...`: PASS
- `git diff --check`: PASS
- Focused regression suite: `133 passed, 1 warning`
- Coverage/diff coverage suite: `127 passed, 1 warning`; diff coverage `86%`
- Targeted E2E smoke included:
  - `tests/e2e/test_vector_worker_only_smoke.py`
  - `tests/e2e/test_inbox_chat_equivalent_e2e.py`

## Full Suite Notes

Full `pytest --tb=short -q` was attempted and interrupted at approximately 12% because the repository-wide suite is very large and had already produced unrelated failures.

Isolated comparison of failures:

- `tests/integration/test_health_check.py::test_health_check_loop`: fails on both worktree and main.
- `tests/performance/test_priming_performance.py::test_memory_usage_baseline`: fails on both worktree and main.
- `tests/integration/test_locomo_legacy_smoke.py::test_legacy_scope_all_within_baseline`: fails in the worktree because ignored benchmark data is missing; passes on main where the ignored local dataset exists.

These failures are not caused by the implementation diff.

## Independent Reviews

- Cursor Agent review: launched but produced empty stdout/log; treated as unavailable.
- Codex review pass 1: found missing integration-style regression coverage.
- Codex review pass 2: no issues found after adding the runner IPC/event-loop regression test.

## Residual Risks

- The startup regression test uses mocked `DigitalAnima` and watcher plumbing with a real runner IPC server/client. It validates the event-loop and IPC failure mode directly, but it is not a full subprocess/vector-worker end-to-end test.
- `startup_idle_compress` remains a pre-ack background task. It is intentionally non-blocking and outside the requested inbox/scheduler/pending watcher gate.

No revision required.
