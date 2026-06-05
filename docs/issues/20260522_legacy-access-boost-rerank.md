---
gh_issue:
parent: 20260522_legacy-memory-enhancement-epic.md
depends_on:
  - 20260522_legacy-cross-encoder-rerank.md
blocks: []
---

# C9: Access-count LTP boost — rerank スコアへの Hebbian 融合

## Overview

既存 `record_access` / chunk access_count / time decay を rerank **後**の最終スコアに融合し、頻繁に参照される記憶を優先する（brain-mapping: LTP / ヘブの法則）。

## Problem / Background

### Current State

- `MemoryRetriever.record_access` 存在（`core/memory/rag/retriever.py`）
- time decay は graph PPR 側（`core/memory/rag/graph.py:35`）に部分的
- rerank スコアは access 情報未使用

### Root Cause

検索ランキングと access 統計が未統合。

## Decided Approach / 確定方針

### Design Decision

**確定**: `core/memory/retrieval/access_boost.py` に:

```python
final_score = rerank_score * (1 + access_boost)
access_boost = min(cap, log1p(access_count) * weight) * recency_factor
recency_factor = exp(-age_days / half_life)  # half_life=30 default
```

- `weight` default **0.05**, `cap` default **0.25**（config `rag.access_boost_weight`, `access_boost_cap`）
- access_count は Chroma metadata `access_count`（indexer 更新、retrieve 時 increment）
- rerank 無効時は RRF score に同式適用

### Rejected Alternatives

- **access のみで rerank 代替**: Rejected — semantic 劣化
- **Priming のみ boost**: Rejected — search 統一が目的

### Changes by Module

| Module | Change Type | Description |
|--------|------------|-------------|
| `core/memory/retrieval/access_boost.py` | New | スコア融合 |
| `core/memory/retrieval/pipeline.py` | Modify | rerank 後ステージ |
| `core/memory/rag/indexer.py` | Modify | access_count metadata 初期化 0 |
| `core/memory/rag/retriever.py` | Modify | record_access で metadata 更新 |
| `core/config/schemas.py` | Modify | access boost 設定 |
| `tests/unit/core/memory/test_access_boost.py` | New | 数式検証 |

### Edge Cases

| Case | Handling |
|------|----------|
| access_count 欠落 | 0 扱い |
| 新規 fact（count=0） | rerank のみ、不公平許容 |
| 高 access 低 relevance | rerank 優先（乗算なので低 rerank は勝てない） |

## Scope

### In Scope

- Legacy Chroma metadata
- pipeline 最終段

### Out of Scope

- Neo4j node access stats
- Forgetting 連動 — 既存 forgetting 維持

## Acceptance Criteria

- [ ] retrieve 後 access_count +1 が metadata に反映
- [ ] 同一 relevance で高 access chunk が上位
- [ ] unit test: 数式境界（cap, half_life）
- [ ] overall F1 baseline 維持（±2pp）

## References

- `core/memory/rag/graph.py:35` — PAGERANK_ALPHA 参考
- `docs/brain-mapping.ja.md:86-87` — LTP
