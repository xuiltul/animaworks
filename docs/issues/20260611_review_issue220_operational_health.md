# Review: Issue #220 Operational Health Hardening

Date: 2026-06-11
Worktree: `/home/main/dev/animaworks-bak-issue-20260611-170828`
Branch: `issue-20260611-170828`

## Scope

Review of the implementation for parent issue #220 and child issues #221-#231:

- vector worker write-circuit hardening
- consolidation timeout and scheduler ownership cleanup
- stale delegation recovery and TaskBoard delegation persistence
- self-DM and meeting-room routing safeguards
- role taxonomy cleanup
- memory dead-code cleanup
- memory/docs drift updates
- process/log hygiene
- LoCoMo retrieval ablation and fact-extraction rollout

## Review Findings And Resolution

1. **Vector retry-storm mitigation was worker-local only.**
   - Fixed by forwarding worker `Retry-After` headers through `/internal/vector/*`, teaching `HttpVectorStore` to suspend writes per collection after 429 responses, adding vector-worker status to `/api/system/status`, and adding client/status tests.

2. **TaskBoard delegation records were best-effort after legacy queue writes.**
   - Fixed by preallocating task IDs, writing TaskBoard metadata first, aborting delegation if TaskBoard persistence fails, and using the same IDs for compatibility JSONL queue entries.

3. **Disabled-delegation bounce handling could repeat indefinitely.**
   - Fixed by marking bounced disabled tasks in TaskBoard metadata and skipping already-bounced tasks on later recovery passes.

4. **Duplicate lifecycle system scheduler was still present.**
   - Fixed by deleting `core/lifecycle/system_crons.py`, keeping `LifecycleManager` as a deprecated compatibility surface for per-Anima heartbeat/inbox behavior only, and moving system-cron tests to `ProcessSupervisor`.

5. **Docs still advertised deprecated `delegation` message intent.**
   - Fixed in English and Japanese specs; `send_message` now documents only `report/question`, with delegation routed through `delegate_task`.

6. **Conversation finalization compatibility narrowed.**
   - Fixed by accepting legacy optional kwargs on both the facade and helper function.

7. **Follow-up full-suite failures from earlier #231 work.**
   - Fixed a silent `except Exception: pass` in fact reconciliation timeout resolution.
   - Updated embedding centralization E2E expectations for explicit `purpose="document"`.

Cursor review process note: the Cursor review helper process exited without producing output files. A Codex review pass completed and the findings above were addressed locally.

## Verification

Passed:

- `uv run ruff check $(git diff --name-only --diff-filter=ACMRT -- '*.py')`
- `uv run pytest tests/unit/core/test_lifecycle.py tests/unit/core/test_lifecycle_consolidation.py tests/unit/test_consolidation_retry.py tests/unit/core/test_dm_log_rotation.py tests/unit/core/test_intent_filter.py tests/unit/core/test_scheduled_hb_cascade.py tests/unit/core/lifecycle/test_weekly_memory_hygiene.py tests/unit/core/supervisor/test_consolidation_targets.py -q`
- `uv run pytest tests/unit/core/memory/test_vector_worker_app.py tests/unit/core/memory/test_vector_worker_client.py tests/unit/server/test_vector_endpoints.py tests/unit/server/routes/test_system.py tests/unit/test_system_status.py -q`
- `uv run pytest tests/unit/core/test_delegation_recovery.py tests/unit/core/test_delegation_taskboard.py tests/unit/core/memory/test_task_queue.py tests/unit/core/memory/test_conversation.py -q`
- `uv run pytest tests/unit/core/memory/test_embedding_prefix_config.py tests/unit/core/memory/test_fact_extraction_legacy.py tests/unit/core/memory/test_fact_invalidation.py tests/unit/core/memory/test_fact_invalidation_candidate_labels.py tests/unit/test_locomo_adapter.py tests/unit/benchmarks/test_locomo_fact_index.py tests/unit/benchmarks/test_locomo_prompt.py tests/unit/benchmarks/test_locomo_answer_prompt.py -q`
- `uv run pytest tests/e2e/test_custom_exception_hierarchy_e2e.py tests/e2e/test_embed_centralization_e2e.py -q`
- `uv run pytest -m e2e -q`

Full-suite note: `uv run pytest --tb=short -q` was started and stopped after early failures to inspect and fix them directly. The two observed failures were fixed and rerun successfully; the exhaustive 14,795-item suite was not completed end-to-end in this review loop.
