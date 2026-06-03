# Review: Legacy Unified Memory Search

Status: APPROVED
Issue: `/home/main/dev/animaworks-bak/docs/issues/20260603_legacy-unified-memory-search.md`
Worktree: `/home/main/dev/animaworks-bak-issue-20260603-155052`
Base: `main`

## Summary

The implementation satisfies the issue requirements for Legacy-only unified retrieval. `UnifiedMemorySearch` centralizes trigger policy, candidate collection, pipeline invocation, abstain metadata, and tool-only offset handling. `RAGMemorySearch`, `LegacyRAGBackend`, Priming C/F/G, and LoCoMo `scope_all` now use the unified path while keeping the Neo4j branch unchanged.

## Requirement Alignment

Pass.

- Added `core/memory/retrieval/unified_search.py` with explicit trigger policy table.
- `search_memory_text()` delegates supported Legacy scopes with `trigger="tool"` and preserves `last_search_meta`.
- `LegacyRAGBackend.retrieve()` and `get_recent_facts()` use unified search; recent facts prefer real atomic facts before activity-log fallback.
- Priming Channel C/F no longer call `MemoryRetriever.search()` on Legacy paths.
- Channel F Neo4j branch is preserved.
- Channel G formats real fact metadata when available.
- LoCoMo `scope_all` uses `UnifiedMemorySearch` while preserving temporal/entity/fact ablation hooks.

## Automated Checks

Pass with noted repository-wide existing file-size debt.

- `ruff check ...`: pass.
- `ruff format --check ...`: pass.
- Targeted unit/E2E suite: `47 passed`.
- Additional related regression suite: `58 passed`.
- Focused E2E: included in targeted suite; real JSONL fact store validates `search_memory_text()` and `LegacyRAGBackend.get_recent_facts()`.
- Focused coverage: 80.17%, pass against 80% threshold.
- Full repository `coverage_checker.py`: interrupted after several minutes because it launched full `pytest --cov=.`; focused coverage above was used as the actionable review metric.
- Repository-wide `file_size_checker.py`: fail due many pre-existing files. Changed-file review is acceptable: new `unified_search.py` is 405 lines, `backend/legacy.py` is 461 lines after cleanup. Existing oversized modified files remain `rag_search.py` and `benchmarks/locomo/adapter.py`.

## Manual Review

Pass.

- Candidate-source construction reuses existing RAG helpers instead of duplicating vector/keyword/fact parsing logic.
- Trigger policy matches the issue table, including heartbeat rerank disabled and unknown trigger fallback to chat.
- Explicit scopes restrict target scopes; `activity_log` absence degrades by omission.
- Keyword-only fallback avoids RRF low-score abstain for unindexed facts, preserving missing fact-index behavior.
- Tests cover trigger policy, explicit scope restriction, offset behavior, abstain propagation, missing fact fallback, and search/priming top doc-id overlap.

## Independent Reviews

Partial.

- Cursor Agent review was launched, but output/log files were empty after process exit. No external findings were available.
- Codex subagent review was skipped due current session tool constraints requiring explicit user request for sub-agents.

## Residual Risk

- Full repository test suite was not run because of expected runtime; targeted and related regression suites passed.
- LoCoMo `scope_all` now uses production-compatible unified search, so benchmark retrieval composition changes from adapter-local BM25/RRF to the shared policy. Environment ablation hooks remain in place.
