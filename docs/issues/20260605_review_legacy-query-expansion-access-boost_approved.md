# Review: Legacy Query Expansion + Access Boost

Status: APPROVED
Date: 2026-06-05
Worktree: `/home/main/dev/animaworks-bak-issue-20260605-085307`
Branch: `issue-20260605-085307`

## Scope

- C8: `docs/issues/20260522_legacy-query-expansion.md`
- C9: `docs/issues/20260522_legacy-access-boost-rerank.md`

## Findings

No blocking findings.

## Requirement Alignment

Pass.

- Query expansion added in `core/memory/retrieval/query_expansion.py`.
- Relative temporal hints, quoted phrases, and BM25 extra tokens are deterministic and LLM-free.
- `UnifiedMemorySearch` now searches with expanded text and applies widened time-hint filtering.
- LoCoMo adapter stores the latest parseable session timestamp and passes it as `reference_time`.
- Access boost added in `core/memory/retrieval/access_boost.py` and applied after rerank/RRF in `RetrievalPipeline`.
- RAG config and LoCoMo pipeline settings expose `access_boost_enabled`, weight, cap, and half-life.
- RAG search result dicts now propagate `access_count`, `last_accessed_at`, and event-time metadata needed by final ranking/filtering.

## Automated Checks

Pass for changed functionality.

- `python3 -m py_compile core/memory/retrieval/query_expansion.py core/memory/retrieval/access_boost.py core/memory/retrieval/pipeline.py core/memory/retrieval/unified_search.py core/memory/rag_search.py benchmarks/locomo/adapter.py`: pass
- Targeted unit/E2E: 184 passed
- Related existing RAG/access/LoCoMo tests: 59 passed
- Targeted coverage for changed retrieval modules: 87%
- `git diff --check`: pass
- LoCoMo live smoke: pass, 1 conversation / 5 questions, `overall_f1=0.3567`, temporal `0.625`

## Known Non-Blocking Checks

- Full `pytest -q tests/unit` was attempted and reported 28 failures. Representative failures were reproduced unchanged on `main`, including priming mock-count tests, missing `openai_codex`, LoCoMo adapter `_facts_dir` test fixture drift, and provenance `force=False` expectation drift.
- `tests/e2e/core/test_legacy_entity_index_boost_e2e.py::test_fact_ingest_updates_entity_registry_and_boosts_metadata_candidates` fails on both worktree and `main` with the same missing JSONL assertion.
- Repo-wide file-size checker reports many pre-existing oversized files. New files are under 500 lines; `unified_search.py` remains under 500 after moving time filtering helpers into `query_expansion.py`.
- Cursor Agent review launcher produced empty output; no external findings were available.
- Codex subagent review skipped because subagent usage was not explicitly requested.

## Manual Review

Pass.

- Access boost is multiplicative and capped, so low relevance cannot dominate solely from high access count.
- Missing `access_count` is treated as 0; invalid/missing `last_accessed_at` keeps boost recency-neutral rather than crashing.
- Time filtering keeps untimed candidates to avoid degrading non-temporal memories.
- Public search API remains backward-compatible; `reference_time` is optional.
- Config additions are additive with safe defaults and `getattr` fallback for existing configs.

## Decision

APPROVED. No revision cycle required.
