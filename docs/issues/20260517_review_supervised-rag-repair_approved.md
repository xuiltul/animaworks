# Code Review: Supervised RAG Repair - Approved

**Review Date**: 2026-05-17
**Original Issue**: `docs/issues/20260517_supervised-rag-repair.md`
**Worktree**: `/home/main/dev/animaworks-bak-issue-20260517-223039`
**Status**: APPROVED

## Summary

The implementation satisfies the confirmed requirement: live anima/HTTP processes no longer run destructive RAG repair in-process. Corruption detection records a repair request, the supervisor stops only the target anima, runs `repair-rag` out-of-process, then restarts the anima and records operational state.

Cursor Agent review returned APPROVED. It raised maintainability observations; the actionable items were addressed before this report:

- Removed dead `_run_repair_guarded`.
- Consolidated supervisor repair-state writes through `repair_state`.
- Added tests for requested-repair polling and actual CLI timeout kill handling.
- Documented that closed `ChromaVectorStore` instances must be discarded.

Codex subagent review was skipped because this session did not have an explicit user request to spawn subagents.

## Review Checks

| Dimension | Status | Notes |
|-----------|--------|-------|
| Requirement alignment | PASS | Request-only detection, supervised stop/repair/restart, CLI repair, state contract, close hook, and lock regression are implemented. |
| Test coverage | PASS for changed behavior | Full coverage script timed out at its built-in 300s limit. Targeted coverage run passed 34 tests; focused modules include `repair_service` 89%, `repair_state` 90%, `_mgr_rag_repair` 79%. `store.py` is low overall because only `close()` was added to a broad existing class. |
| Code quality | PASS | Repair state and supervisor repair lifecycle are split into focused modules. |
| SRP/file bloat | PASS for this diff | Repo-wide file-size checker fails on many pre-existing oversized files. New/focused changed files are under 500 lines: `repair_rag_cmd.py` 66, `repair_state.py` 125, `_mgr_rag_repair.py` 404, `_mgr_health.py` 384, `repair_service.py` 461. |
| E2E | PASS | `tests/e2e/test_supervised_rag_repair_e2e.py` passes and exercises detection-to-supervisor repair request flow. |
| Regression | PASS | Full test suite passed before final review refinements: 14274 passed, 78 skipped. After refinements, targeted RAG/supervisor/index-lock suites pass. |

## Verification

- `pytest -q --maxfail=1 --tb=short`: 14274 passed, 78 skipped.
- `pytest tests/unit/core/memory/test_rag_repair.py tests/unit/core/memory/test_rag_store_lifecycle.py tests/unit/cli/test_repair_rag_cmd.py tests/unit/core/supervisor/test_rag_repair_health.py tests/unit/core/supervisor/test_supervised_rag_repair.py tests/e2e/test_supervised_rag_repair_e2e.py -q`: 34 passed.
- `pytest tests/unit/cli/test_index_shared.py tests/unit/core/supervisor/test_daily_indexing.py -q`: 19 passed.
- `ruff check core/ cli/ server/ tests/unit/core/supervisor/test_supervised_rag_repair.py`: passed.
- `ruff format --check` on changed files: passed.
- `git diff --check`: passed.

## Residual Risks

- Repo-wide file bloat remains pre-existing technical debt outside this issue.
- Full coverage cannot be measured by the bundled checker because the suite exceeds its 300s timeout.
- `ChromaVectorStore.close()` is an explicit resource cleanup hook; existing store instances should not be reused after close.

## Decision

APPROVED. No revision required before merge.
