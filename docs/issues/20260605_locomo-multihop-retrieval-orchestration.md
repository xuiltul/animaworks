# LoCoMo Multi-Hop Retrieval Orchestration — category 1の空contextとリスト欠落をfact-firstで潰す

## Overview

Legacy LoCoMo `scope_all` の feature-on retrieval は、直近の regression fix 後に multi_hop recall@10 を `55.58%` から `65.18%` へ改善した。しかし category 1 にはまだ `context_count=0` の質問と、複数要素を取り切れない低 recall 質問が残っている。この Issue では Neo4j や LLM rewriting を使わず、Legacy adapter 内の deterministic retrieval orchestration と diagnostics を拡張して、multi_hop をさらに改善する。

Dependency: `docs/implemented/20260604_legacy-locomo-regression-analysis-and-multihop-retrieval-fix_implemented-20260605.md`

## Problem / Background

### Current State

- Full retrieval diagnostics result: `/tmp/legacy-locomo-feature-on-diagnostics-fixed/2026-06-05T00-34-16_scope_all_retrieval_diagnostics.json`
- feature-on multi_hop recall@10 is `0.6518`, improved from baseline `0.5558`, but still below the desired ceiling.
- feature-on category 1 has three `context_count=0` questions:
  - q4: `What is Caroline's identity?` → `Transgender woman`
  - q7: `What is Caroline's relationship status?` → `Single`
  - q24: `What does Melanie do to destress?` → `Running, pottery`
- Low recall category 1 questions remain for list / profile / shared-subject patterns:
  - q19: kids like `dinosaurs, nature`
  - q23: books read `"Nothing is Impossible", "Charlotte's Web"`
  - q52: pets' names `Oliver, Luna, Bailey`
  - q55: both painted `Sunsets`
  - q56: symbols `Rainbow flag, transgender symbol`
  - q60: instruments `clarinet and violin`
  - q70: transgender-specific events `Poetry reading, conference`
  - q75: number of children `3`

Relevant code:

- `benchmarks/locomo/adapter.py:627` — `retrieve()` dispatches Legacy `scope_all`.
- `benchmarks/locomo/adapter.py:661` — pipeline settings and category gate are resolved before unified search.
- `benchmarks/locomo/adapter.py:669` — feature-on fact scope is included only when fact index has records.
- `benchmarks/locomo/adapter.py:677` — current category 1 retrieval runs one unified query.
- `benchmarks/locomo/adapter.py:754` — fact vector search helper exists but is not used as a category 1 fallback.
- `benchmarks/locomo/adapter.py:786` — fact BM25 search helper exists but is not used as a category 1 fallback.
- `benchmarks/locomo/adapter.py:856` — category 1 entity boost is stricter than category 4.
- `benchmarks/locomo/answer_prompt.py:69` — category confidence gate currently has no multi_hop relaxation.
- `core/memory/retrieval/pipeline.py:102` — confidence gate can remove low-confidence candidates after rerank/RRF.
- `benchmarks/locomo/retrieval_diagnostics.py:147` — diagnostics record context count and top memory type.

### Root Cause

1. **Category 1 still uses a single query** — `benchmarks/locomo/adapter.py:677` runs unified search only with the original question, so list/profile questions do not reliably retrieve all answer-bearing facts.
2. **Fact fallback exists but is not orchestrated** — `benchmarks/locomo/adapter.py:754` and `benchmarks/locomo/adapter.py:786` can search fact vectors / fact BM25, but `retrieve()` does not use them when category 1 returns empty context.
3. **Gate relaxation is not evidence-aware** — `benchmarks/locomo/answer_prompt.py:158` returns configured thresholds without considering category 1 fact evidence; lowering globally would risk adversarial regression.
4. **No entity profile grouping** — `benchmarks/locomo/fact_index.py:53` creates sentence-level facts, but the adapter does not group facts by person and attribute to answer profile/list questions.
5. **No deterministic alias expansion** — questions like `destress`, `identity`, `relationship status`, and `children` do not necessarily share lexical tokens with answer facts.
6. **Intersection/shared-subject questions are not special-cased** — questions with `both`, `and`, or multiple speakers need per-person retrieval and merged candidates.
7. **Diagnostics do not expose which multi_hop helper fired** — `benchmarks/locomo/retrieval_diagnostics.py:147` records generic retrieval fields but not decomposition/profile/alias/intersection provenance.

### Impact

| Component | Impact | Description |
|-----------|--------|-------------|
| `benchmarks/locomo/adapter.py` | Direct | Add category 1 orchestration for fallback, decomposition, entity profile, alias expansion, and intersection patterns. |
| `benchmarks/locomo/answer_prompt.py` | Direct | Add fact-evidence-aware multi_hop gate settings without touching adversarial category 5. |
| `benchmarks/locomo/retrieval_diagnostics.py` | Direct | Surface multi_hop helper metadata and zero-context counters. |
| `benchmarks/locomo/fact_index.py` | Direct | Provide metadata needed for profile and alias grouping. |
| `core/memory/retrieval/entity.py` | Direct | Extend deterministic alias / multi-token entity matching in a category-safe way. |
| `tests/` | Direct | Add unit and integration coverage for all category 1 helper paths. |
| `benchmarks/locomo/neo4j_adapter.py` | No change | This Issue targets Legacy only. |

## Decided Approach / 確定方針

### Design Decision

確定: Legacy `scope_all` の category 1 / multi_hop 専用に deterministic retrieval orchestration を追加する。`LOCOMO_FACT_INDEX=1` feature-on path で、元 query に加えて alias-expanded decomposition queries、person profile candidates、intersection/shared-subject candidates、empty-context fact fallback を統合し、既存 unified search output と RRF/score merge する。LLM rewriting、Neo4j 変更、category 5 gate 変更は行わない。

### Rejected Alternatives

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Neo4j adapter を拡張する | graph traversal can model multi-hop | user direction is Legacy improvement; current best score path is Legacy feature-on | **Rejected**: scope mismatch |
| LLM query decomposition / entity extraction | higher semantic flexibility | non-deterministic, slow, provider-dependent; breaks retrieval-only diagnostics reproducibility | **Rejected**: diagnostics must stay deterministic |
| confidence gate を全カテゴリで緩和する | quick context_count improvement | known adversarial / abstain regression risk | **Rejected**: category 5 protection is mandatory |
| entity boost weight を単純に上げる | small implementation | previous entity boost had limited impact and can amplify noisy overlap | **Rejected**: does not address empty context or list completeness |
| **Category 1 deterministic orchestration (Adopted)** | targeted, testable, uses existing fact index | requires several helper paths and diagnostics | **Adopted**: directly attacks observed failures with bounded blast radius |

### Key Decisions from Discussion

1. **Gate relaxation is category 1 + fact evidence only** — Reason: reduce empty multi_hop contexts without changing adversarial behavior.
2. **Query decomposition is deterministic** — Reason: keep retrieval-only diagnostics reproducible and fast.
3. **Entity profile grouping is adapter-local** — Reason: LoCoMo-specific benchmark behavior should not leak into general memory retrieval.
4. **Alias map is LoCoMo-specific and bounded** — Reason: cover observed vocabulary gaps without introducing broad semantic expansion.
5. **Intersection/shared-subject questions get per-person retrieval** — Reason: `both` / multi-speaker questions often need evidence from more than one entity.
6. **Default feature flags off preserve current behavior** — Reason: feature-on must be validated before changing default smoke behavior.

### Changes by Module

| Module | Change Type | Description |
|--------|------------|-------------|
| `benchmarks/locomo/adapter.py` | Modify | Add `_retrieve_scope_all()` orchestration for category 1 feature-on: original unified search, decomposition queries, profile candidates, alias queries, intersection candidates, and empty-context fact fallback. |
| `benchmarks/locomo/answer_prompt.py` | Modify | Add `multi_hop_fact_gate_settings()` or equivalent helper to lower thresholds only when category 1 has fact evidence. |
| `benchmarks/locomo/retrieval_diagnostics.py` | Modify | Record multi_hop helper metadata: decomposition query count, profile hits, alias hits, intersection hits, fallback used, and zero-context count in summary. |
| `benchmarks/locomo/fact_index.py` | Modify | Preserve speaker/person, normalized entity, and attribute-friendly metadata for profile grouping. |
| `core/memory/retrieval/entity.py` | Modify | Add deterministic alias expansion support that can be called by LoCoMo adapter without changing default extraction behavior. |
| `tests/unit/test_locomo_adapter.py` | Modify | Add tests for gate relaxation, empty-context fact fallback, query decomposition, profile grouping, alias expansion, and intersection retrieval. |
| `tests/unit/benchmarks/test_locomo_retrieval_diagnostics.py` | Modify | Add tests for feature-on multi_hop metadata and zero-context summary. |
| `tests/integration/test_locomo_legacy_smoke.py` | Modify | Ensure feature flags remain explicit and non-live guardrails continue to pass. |
| `benchmarks/locomo/neo4j_adapter.py` | No change | Legacy-only scope. |

### Required Behavior Details

#### Category 1 feature-on orchestration

Target: `benchmarks/locomo/adapter.py`

- Trigger only when all are true:
  - `self._search_mode == "scope_all"`
  - `category == 1`
  - `locomo_fact_index_enabled() is True`
  - `self._last_fact_count > 0`
- Run existing unified search first.
- Build up to 6 additional deterministic queries:
  - original question with speaker names preserved
  - alias-expanded attribute query
  - person + attribute query
  - person + object noun query
  - per-person query for `both` / `and` questions
  - fallback fact query when unified search returns no items
- Merge candidates by stable content/fact id, keeping strongest score and metadata.
- Apply a small source priority only for category 1:
  - facts with alias/profile/intersection provenance can outrank generic episode chunks.
  - episode chunks are not removed; they remain eligible.

#### Alias map

Target: `benchmarks/locomo/adapter.py` or a small new helper under `benchmarks/locomo/`.

Fixed aliases:

| Query signal | Expansion |
|--------------|-----------|
| `destress` | `relax`, `unwind`, `mental health`, `running`, `pottery` |
| `identity` | `transgender`, `woman`, `transgender woman` |
| `relationship status` | `single`, `dating`, `partner`, `relationship` |
| `children`, `kids` | `son`, `daughter`, `children`, `kids`, numeric family facts |
| `symbols` | `rainbow flag`, `transgender symbol`, `necklace`, `symbolizes` |
| `events`, `participated` | `attended`, `joined`, `speech`, `parade`, `support group`, `conference`, `mentoring` |
| `instruments` | `clarinet`, `violin`, `music`, `play` |
| `books`, `read` | `book`, `read`, `Charlotte's Web`, `Nothing is Impossible` |
| `pets` | `pet`, `dog`, `cat`, `Oliver`, `Luna`, `Bailey` |
| `painted`, `both painted` | `paint`, `painting`, `sunset`, `sunrise`, `art` |

#### Diagnostics metadata

Each returned context from category 1 helper paths should carry metadata keys when applicable:

- `locomo_multihop_helper`: one of `decomposition`, `alias`, `profile`, `intersection`, `fact_fallback`
- `locomo_multihop_query`: the helper query string
- `locomo_multihop_aliases`: list of aliases added
- `locomo_multihop_person`: person name when profile/person-specific

Summary should include:

- `multi_hop_zero_context_count`
- `multi_hop_helper_hit_counts`
- `multi_hop_feature_recall_at_10` in feature-on ablation summary when available

### Edge Cases

| Case | Handling |
|------|----------|
| Fact index disabled | Use existing `scope_all` path exactly; no helper metadata appears. |
| Category is not 1 | Use existing path exactly except existing feature-on behavior. |
| Category 5 adversarial | No gate relaxation, no category 1 helpers, no recall inclusion. |
| Decomposition generates no query | Use original query only. |
| Alias expansion produces too many terms | Cap helper queries at 6 and aliases per query at 8. |
| Person names are missing | Fall back to query-derived speaker mentions and original query. |
| Intersection question has only one person's evidence | Keep available candidates as union; do not return empty solely because intersection is incomplete. |
| Empty unified search result | Run fact vector + fact BM25 fallback and return merged fact candidates if any. |
| Helper search errors | Log/debug and keep base unified search result; retrieval should not fail the QA loop. |
| Temporal category regression | Out of scope; category 2 behavior remains unchanged. |

## Implementation Plan

### Phase 1: Category 1 Gate and Empty-Context Fallback

| # | Task | Target |
|---|------|--------|
| 1-1 | Add fact-evidence-aware category 1 gate helper | `benchmarks/locomo/answer_prompt.py` |
| 1-2 | Route `scope_all` category 1 through a dedicated adapter method | `benchmarks/locomo/adapter.py` |
| 1-3 | Add fact vector + fact BM25 fallback when category 1 unified search returns no items | `benchmarks/locomo/adapter.py` |
| 1-4 | Add unit tests for default/off behavior and fallback behavior | `tests/unit/test_locomo_adapter.py` |

**Completion condition**: category 1 empty unified result can return fact candidates when fact index is available, while non-category-1 and feature-off paths are unchanged.

### Phase 2: Query Decomposition and Alias Expansion

| # | Task | Target |
|---|------|--------|
| 2-1 | Add bounded deterministic alias map and expansion helper | `benchmarks/locomo/adapter.py` or new `benchmarks/locomo/multihop.py` |
| 2-2 | Generate category 1 helper queries from person names, attribute nouns, and aliases | `benchmarks/locomo/adapter.py` |
| 2-3 | Merge helper candidates by fact id / content and keep provenance metadata | `benchmarks/locomo/adapter.py` |
| 2-4 | Add unit tests for `identity`, `relationship status`, `destress`, and list-style queries | `tests/unit/test_locomo_adapter.py` |

**Completion condition**: helper query generation is deterministic, bounded, and covered by unit tests.

### Phase 3: Entity Profile and Intersection Candidates

| # | Task | Target |
|---|------|--------|
| 3-1 | Build adapter-local person profile rows from fact BM25 corpus | `benchmarks/locomo/adapter.py` |
| 3-2 | Add profile candidates for person + attribute questions | `benchmarks/locomo/adapter.py` |
| 3-3 | Add per-person retrieval for `both` / multi-speaker shared-subject questions | `benchmarks/locomo/adapter.py` |
| 3-4 | Add unit tests for pets/books/symbols/both-painted cases | `tests/unit/test_locomo_adapter.py` |

**Completion condition**: profile and intersection helpers add candidates with explicit provenance metadata without removing base candidates.

### Phase 4: Diagnostics and Verification

| # | Task | Target |
|---|------|--------|
| 4-1 | Add multi_hop helper metadata to diagnostics rows and summary | `benchmarks/locomo/retrieval_diagnostics.py` |
| 4-2 | Add diagnostics unit tests for helper counts and zero-context summary | `tests/unit/benchmarks/test_locomo_retrieval_diagnostics.py` |
| 4-3 | Run targeted unit tests | tests |
| 4-4 | Run non-live integration smoke | `tests/integration/test_locomo_legacy_smoke.py` |
| 4-5 | Run retrieval diagnostics feature-on smoke for one conversation | `benchmarks.locomo.retrieval_diagnostics` |

**Completion condition**: tests pass and one-conversation feature-on diagnostics show category 1 recall@10 does not regress from `0.6518`.

## Scope

### In Scope

- Legacy LoCoMo `scope_all` category 1 retrieval orchestration.
- Fact-evidence-aware gate relaxation for category 1 only.
- Deterministic query decomposition and alias expansion.
- Adapter-local entity profile candidates.
- Shared-subject / intersection retrieval for multi-person questions.
- Diagnostics metadata for helper provenance and zero-context count.
- Unit tests, non-live integration smoke, retrieval-only diagnostics smoke.

### Out of Scope

- Neo4j adapter changes — Reason: user direction is Legacy improvement.
- LLM query rewriting or LLM entity extraction — Reason: retrieval diagnostics must remain deterministic.
- Production default feature-on enablement — Reason: full smoke validation should remain explicit.
- Temporal category optimization — Reason: previous feature-on diagnostics showed slight temporal regression and needs a separate Issue.
- Judge scoring or answer prompt rewrite — Reason: this Issue targets retrieval, not final answer scoring.

## Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Alias expansion retrieves noisy facts | multi_hop or open_domain recall could drop | Restrict to category 1, cap helper queries, keep diagnostics provenance, and require no multi_hop recall regression. |
| Gate relaxation harms adversarial abstain | Category 5 F1 could regress | Apply only category 1 + fact evidence; category 5 path unchanged and tested. |
| Profile candidates dominate relevant episode chunks | Loss of temporal/detail evidence | Merge with base candidates rather than replacing them; source priority is small and category-limited. |
| Intersection logic returns too little evidence | Both/shared questions remain weak | Use union fallback when full intersection is unavailable. |
| Runtime increases | Diagnostics may slow down | Cap helper queries at 6 and use existing fact vector/BM25 caches. |

## Acceptance Criteria

- [ ] `LOCOMO_FACT_INDEX=1` category 1 `scope_all` retrieval can return fact fallback candidates when base unified search returns no context.
- [ ] `LOCOMO_FACT_INDEX=1` category 1 retrieval generates deterministic bounded helper queries for alias/decomposition cases.
- [ ] Entity profile candidates are available for person profile/list questions and carry `locomo_multihop_helper=profile`.
- [ ] Shared-subject questions can add `locomo_multihop_helper=intersection` candidates.
- [ ] category 5 adversarial gate and abstain behavior are unchanged.
- [ ] Feature flags off preserve existing default retrieval behavior.
- [ ] `python -m benchmarks.locomo.retrieval_diagnostics --mode scope_all --conversations 1 --top-k 10 --ceiling-top-k 10 --feature-on-ablation` outputs multi_hop helper metadata.
- [ ] One-conversation feature-on diagnostics do not regress multi_hop recall@10 from `0.6518`.
- [ ] Targeted unit tests pass.
- [ ] `LOCOMO_LLM_CREDENTIAL=__missing__ uv run pytest tests/integration/test_locomo_legacy_smoke.py -q` passes with live smoke skipped.

## References

- `benchmarks/locomo/adapter.py:627` — Legacy adapter retrieval entrypoint.
- `benchmarks/locomo/adapter.py:669` — fact scope activation when fact index exists.
- `benchmarks/locomo/adapter.py:754` — fact vector search helper.
- `benchmarks/locomo/adapter.py:786` — fact BM25 search helper.
- `benchmarks/locomo/adapter.py:856` — current strict category 1 entity boost config.
- `benchmarks/locomo/answer_prompt.py:69` — category gate thresholds.
- `core/memory/retrieval/pipeline.py:102` — confidence gate application.
- `benchmarks/locomo/retrieval_diagnostics.py:147` — retrieval diagnostics row fields.
- `docs/implemented/20260604_legacy-locomo-regression-analysis-and-multihop-retrieval-fix_implemented-20260605.md` — previous feature-on regression fix.
- `/tmp/legacy-locomo-feature-on-diagnostics-fixed/2026-06-05T00-34-16_scope_all_retrieval_diagnostics.json` — current diagnostic baseline.
