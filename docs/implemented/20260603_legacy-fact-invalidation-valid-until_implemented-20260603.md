---
parent: 20260603_legacy-brain-memory-expansion-epic.md
supersedes_scope:
  - 20260522_legacy-ingest-contradiction-invalidation.md
depends_on:
  - 20260603_legacy-atomic-facts-mvp.md
blocks:
  - 20260603_legacy-unified-memory-search.md
  - 20260603_legacy-entity-aware-graph.md
---

# Legacy Fact Invalidation Valid-Until — facts の矛盾解消と temporal validity を追加する

## Overview

Legacy facts を ADD-only から temporal validity aware な記憶に拡張する。新 fact ingest 時に類似既存 fact を検索し、duplicate は skip、contradiction は旧 fact の `valid_until` を設定、complement は既存 fact を補完する。削除ではなく `valid_until` による失効を使い、再固定化/reconsolidation として扱う。

## Problem / Background

### Current State

- Atomic Facts MVP 後、facts は `facts/{date}.jsonl` に保存され `valid_until` を持つが、ingest時の矛盾解消はない。
- `MemoryRetriever.search()` は knowledge の `valid_until == ""` exact filter を持つ — `core/memory/rag/retriever.py:153`。
- `MemoryIndexer._extract_metadata()` は frontmatterの `valid_until` を metadata に入れる — `core/memory/rag/indexer.py:934`。
- Existing `core/memory/contradiction.py` は knowledge file の supersedeで `valid_until` を設定する — `core/memory/contradiction.py:730`、`core/memory/contradiction.py:760`。
- Existing `FactExtractor` promptには invalidation promptも存在するが、Legacy fact JSONLへの適用はない。

### Root Cause

1. **Fact reconcile path がない** — 新 fact と既存 fact の duplicate/contradiction/complement 判定がない。
2. **JSONL partial update がない** — 旧 fact の `valid_until` を安全に更新する API がない。
3. **Validity filter が exact empty 前提** — `valid_until` が将来日時の場合の as-of handling がない。
4. **Reindex workflow がない** — JSONL fact update後、該当 fact chunk の Chroma metadata/content更新が必要。

### Impact

| Component | Impact | Description |
|-----------|--------|-------------|
| `core/memory/fact_invalidation.py` | Direct | New reconciliation logic。 |
| `core/memory/facts.py` | Direct | Fact row update/rewrite API が必要。 |
| `core/memory/fact_extraction.py` | Direct | Fact append前に reconcile を呼ぶ。 |
| `core/memory/rag/retriever.py` | Direct | facts validity filtering を as-of対応にする。 |
| `core/memory/rag/indexer.py` | Direct | Updated fact metadata reindex。 |
| `core/memory/contradiction.py` | Indirect | Existing LLM call/error-handling patterns are reused; knowledge-file supersede logic is not modified. |

## Decided Approach / 確定方針

### Design Decision

**確定**: `core/memory/fact_invalidation.py` に `reconcile_new_fact(anima_dir, fact, *, as_of_time=None) -> ReconcileResult` を実装する。既存 facts collection から類似 top-5 を取得し、max similarity が threshold 未満なら ADD。threshold 以上なら background LLM で `CONTRADICT | COMPLEMENT | DUPLICATE | ADD` の単一判定を行う。CONTRADICT は旧 fact の `valid_until` を new fact の `valid_at`、なければ `recorded_at` に設定する。物理 delete はしない。

### Reconcile Algorithm

```text
1. Search facts collection for top-5 similar facts in same anima.
2. Filter candidates whose valid_until is active for the new fact's valid_at/recorded_at.
3. If max_similarity < facts_reconcile_similarity_threshold (default 0.82): ADD.
4. Otherwise ask background LLM for one label:
   - DUPLICATE: skip new fact.
   - CONTRADICT: set old.valid_until = new.valid_at or new.recorded_at; append new fact.
   - COMPLEMENT: update old fact text/entities/confidence with additive details; skip appending if merged.
   - ADD: append new fact.
5. Rewrite affected JSONL file atomically and reindex affected fact file(s).
```

### Rejected Alternatives

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| ADD-only forever | Simple | 矛盾 fact が増え adversarial/temporalで悪化 | **Rejected**: reconsolidationが目的 |
| Physical delete old facts | Searchから消える | auditabilityとtemporal reasoningを失う | **Rejected**: `valid_until`で失効する |
| Always trust newest by timestamp without LLM | Fast | complement/duplicate/contradictionを誤判定しやすい | **Rejected**: threshold後だけLLM判定 |
| Full A-Mem neighbor evolution | 高度 | 実装範囲が大きい | **Rejected**: 本Issueでは扱わず、facts invalidation 実装後に別Epicで扱う |
| **Similarity + LLM label + valid_until (Adopted)** | 精度と実装範囲のバランス | LLM依存あり | **Adopted**: failure時ADDで安全側 |

### Key Decisions from Discussion

1. **削除しない** — Reason: brain-mapping上の再固定化/temporal invalidationとして履歴を残す。
2. **LLM failureはADD** — Reason: 誤ったskip/invalidateより記憶を残す方が安全。
3. **旧 fact の valid_until は new valid_at優先** — Reason: factが真になった時点で旧状態は失効する。
4. **DUPLICATE は skip** — Reason: fact storeのノイズとindex肥大を抑える。
5. **COMPLEMENT は同一 fact_id の update** — Reason: 互換性のある補足はfactを分裂させない。

### Changes by Module

| Module | Change Type | Description |
|--------|------------|-------------|
| `core/memory/fact_invalidation.py` | New | `ReconcileAction`, `ReconcileResult`, `reconcile_new_fact()`, LLM classifier。 |
| `core/memory/facts.py` | Modify | fact_id lookup、atomic JSONL rewrite、update_by_fact_id、active/as-of helpers。 |
| `core/memory/fact_extraction.py` | Modify | New fact append前に reconcile を呼び、resultごとに ADD/SKIP/UPDATE/INVALIDATE を実行する。 |
| `core/memory/rag/retriever.py` | Modify | `memory_type == "facts"` validity filtering を追加し、default は active facts only。 |
| `core/memory/rag/indexer.py` | Modify | Fact JSONL rewrite後に該当 file を reindex できるようにする。 |
| `core/config/schemas.py` | Modify | `facts_reconcile_enabled`, `facts_reconcile_similarity_threshold`, `facts_reconcile_top_k` を追加。 |
| `tests/unit/core/memory/test_fact_invalidation.py` | New | ADD/SKIP/INVALIDATE/UPDATE and failure fallback tests。 |

### Edge Cases

| Case | Handling |
|------|----------|
| New fact has no `valid_at` | Use `recorded_at` as invalidation time. |
| Existing fact has future `valid_until` | Treat as active until that timestamp; if contradicted, set earlier valid_until only if new invalidation time is earlier. |
| Existing fact already expired before new fact | Ignore for contradiction; do not update. |
| Multiple contradictory candidates | Invalidate all candidates labeled CONTRADICT above threshold, capped by top_k. |
| LLM returns invalid label | Treat as ADD and log warning. |
| JSONL rewrite fails | Do not append new fact if old invalidation was required and could not be persisted; return error to caller but do not crash session flow. |
| Reindex fails after JSONL update | Log warning; next rebuild must repair. Acceptance includes targeted reindex test. |
| COMPLEMENT target spans multiple files | Update only the selected highest similarity fact in MVP; other candidates remain unchanged. |
| Chroma search unavailable | Skip reconcile and ADD. |

## Implementation Plan

### Phase 1: Fact Store Update API

| # | Task | Target |
|---|------|--------|
| 1-1 | Add fact lookup/update/rewrite helpers | `core/memory/facts.py` |
| 1-2 | Add active/as-of validity helpers | `core/memory/facts.py` |
| 1-3 | Add unit tests for JSONL rewrite and validity | `tests/unit/core/memory/test_fact_invalidation.py` |

**Completion condition**: A fact row can be updated by fact_id with atomic file rewrite and validity helpers return expected active/expired status.

### Phase 2: Reconcile Logic

| # | Task | Target |
|---|------|--------|
| 2-1 | Add `ReconcileAction` and `ReconcileResult` | `core/memory/fact_invalidation.py` |
| 2-2 | Add vector top-k candidate retrieval from facts collection | `core/memory/fact_invalidation.py` |
| 2-3 | Add LLM classifier with strict label parsing | `core/memory/fact_invalidation.py` |
| 2-4 | Add failure fallback behavior | `core/memory/fact_invalidation.py` |

**Completion condition**: Mocked similarity/LLM tests cover ADD, DUPLICATE, CONTRADICT, COMPLEMENT, invalid label, and LLM failure.

### Phase 3: Ingest and Retrieval Integration

| # | Task | Target |
|---|------|--------|
| 3-1 | Call reconcile before fact append | `core/memory/fact_extraction.py` |
| 3-2 | Reindex affected fact files after update | `core/memory/fact_extraction.py`, `core/memory/rag/indexer.py` |
| 3-3 | Add active fact filtering to vector and keyword fact retrieval | `core/memory/rag/retriever.py`, `core/memory/rag_search.py` |
| 3-4 | Add config fields | `core/config/schemas.py` |

**Completion condition**: Contradicted old facts are excluded from `scope=facts` and `scope=all`, while non-expired facts remain retrievable.

## Scope

### In Scope

- Legacy facts JSONL reconciliation.
- Duplicate skip, contradiction invalidation, complement update.
- `valid_until` filtering for facts.
- Config-controlled threshold/top_k.
- Unit tests for reconciliation and retrieval filtering.

### Out of Scope

- Episode/knowledge invalidation — Reason: facts first; existing knowledge contradiction remains separate.
- Entity merge/evolution — Reason: entity registry issue only handles aliases/mentions.
- Graph edge invalidation — Reason: entity-aware graph issue later consumes active facts.
- Neo4j invalidation sync — Reason: Legacy-first scope.
- Full temporal query language — Reason: UnifiedSearch/query expansion can extend later.

## Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| False contradiction invalidates useful fact | Recall loss | Similarity threshold + LLM label; no physical delete; audit via JSONL. |
| LLM failure or malformed output | Ingest instability | Failure => ADD and warning. |
| JSONL rewrite corrupts file | Memory loss | Atomic temp write and tests; backup original on failure. |
| Expired filtering too strict | Valid facts hidden | MVP exact active helper; tests for empty/future/past valid_until. |
| Reindex mismatch | Search returns stale fact | Reindex affected file; add rebuild fallback and tests. |

## Acceptance Criteria

- [ ] `core/memory/fact_invalidation.py` implements ADD/SKIP/INVALIDATE_OLD/UPDATE actions.
- [ ] Similarity below threshold always ADDs without LLM.
- [ ] LLM failure, invalid label, or Chroma failure falls back to ADD.
- [ ] DUPLICATE skips the new fact.
- [ ] CONTRADICT sets old fact `valid_until` to new `valid_at` or `recorded_at` and appends the new fact.
- [ ] COMPLEMENT updates the selected old fact without appending a duplicate.
- [ ] JSONL updates are atomic and covered by tests.
- [ ] Vector and keyword `facts` retrieval exclude expired facts.
- [ ] `scope=all` does not return expired facts.
- [ ] Config defaults are safe and documented in schema.
- [ ] Unit tests cover all reconcile actions and edge cases.

## References

- `docs/issues/20260603_legacy-atomic-facts-mvp.md` — Fact store dependency.
- `docs/issues/20260522_legacy-ingest-contradiction-invalidation.md:1` — Earlier C6 superseded by this revised issue.
- `core/memory/rag/retriever.py:153` — Existing `valid_until` filter for knowledge.
- `core/memory/rag/indexer.py:934` — Existing metadata extraction for `valid_until`.
- `core/memory/contradiction.py:730` — Existing knowledge supersede flow.
- `core/memory/contradiction.py:760` — Existing valid_until assignment.
- `core/memory/extraction/extractor.py:35` — Existing fact extraction pipeline.
