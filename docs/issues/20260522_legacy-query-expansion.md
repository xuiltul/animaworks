---
gh_issue:
parent: 20260522_legacy-memory-enhancement-epic.md
depends_on:
  - 20260522_legacy-cross-encoder-rerank.md
blocks: []
---

# C8: Query expansion（ルールベース）— temporal / keyword 補強

## Overview

検索前にクエリを拡張し、temporal 質問の日付キーワードと固有名詞を BM25 / vector 両方に効かせる。LLM 不使用のルールベースのみ（latency 制約）。

## Problem / Background

### Current State

- LoCoMo temporal: Legacy 60.7%、multi_hop 41.9%
- クエリ "When did Caroline go..." が vector のみに依存

### Root Cause

クエリ側の temporal/entity 信号が weak。

## Decided Approach / 確定方針

### Design Decision

**確定**: `core/memory/retrieval/query_expansion.py` に `expand_query(query: str, *, reference_time: datetime | None) -> ExpandedQuery` を実装。

**ルール**（順次適用）:

1. **相対日付展開**（reference_time 必須、default UTC now）:
   - `yesterday` → `reference_time - 1 day` ISO date
   - `last week` → 7日範囲
   - `last month` → 30日範囲
   - 正規表現リスト固定（`templates` 不要、コード内定数 + i18n エラーなし）

2. **引用符内フレーズ抽出** → `boost_phrases[]`

3. **Wh 語除去後の content tokens** → `bm25_extra[]`

**出力**:

```python
@dataclass
class ExpandedQuery:
    original: str
    search_text: str          # original + expanded date strings + phrases
    time_hint_start: str | None
    time_hint_end: str | None
```

`UnifiedMemorySearch` / `rag_search` は `search_text` で vector/BM25、time_hint で facts/episodes metadata filter（C4 valid_at 範囲、±7日 widen）。

### Rejected Alternatives

- **LLM query rewrite**: Rejected — cost/latency、第2波
- **HyDE**: Rejected — 同上

### Changes by Module

| Module | Change Type | Description |
|--------|------------|-------------|
| `core/memory/retrieval/query_expansion.py` | New | expand_query |
| `core/memory/retrieval/unified_search.py` | Modify | 検索前 expand 呼び出し |
| `tests/unit/core/memory/test_query_expansion.py` | New | 日付パターン |

### Edge Cases

| Case | Handling |
|------|----------|
| reference_time 不明（LoCoMo） | adapter が conversation session_date を渡す |
| 二重展開 | idempotent（既に ISO date あれば skip） |
| 非英語クエリ | 英語 relative のみ第1波 |

## Scope

### In Scope

- 英語 relative temporal（LoCoMo 向け）
- unified search 統合

### Out of Scope

- 日本語相対日付 — 第2波 i18n
- LLM expansion

## Acceptance Criteria

- [ ] "yesterday" + reference 2023-05-08 → time_hint 2023-05-07
- [ ] LoCoMo adapter が session 日付を reference_time に設定
- [ ] temporal F1 ≥ baseline 60.7%（1 conv、C2+C8 組合せ）
- [ ] unit test 15 パターン以上

## References

- LoCoMo answer template — `benchmarks/locomo/adapter.py:35-54`
- Graphiti valid_at — 外部参考
