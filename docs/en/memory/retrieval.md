<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/memory/retrieval.md -->
<!-- i18n: source-sha256=cd9eb191d085ca1974053e1c223707420169fadf846459b8690a8f5b40695a92 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Verified commit: 193a5e72

# Intentional Recall and Search

Anima explicitly searches for missing information using `search_memory` when automatic recall is insufficient, and reads the original file with `read_memory_file` as needed. Tool definitions are in `core/tooling/schemas/memory.py`, and argument handling is in `core/tooling/handler_memory.py`.

## Arguments for `search_memory`

| Argument | Purpose |
|---|---|
| `query` | Natural language search term. Required. |
| `scope` | Select the search target from `knowledge`, `episodes`, `procedures`, `facts`, `common_knowledge`, `skills`, `activity_log`, `code`, `all`. Defaults to `all` when omitted. |
| `offset` | Page position of results. One page contains 10 items, with a maximum offset of 50. |
| `project` | Restrict results to registered project archives. |
| `time_range` | Specify an ISO-format date or time for `after` and `before` to narrow the time range. |

See the [Tool CLI Reference](../reference/tool-cli.md) for argument details and CLI usage. Code search requires `scope="code"` and `project`.

## Search Pipeline

`core/memory/retrieval/unified_search.py` collects candidates based on trigger and scope. Vector search uses semantic similarity, while BM25 uses term matching. Ranked lists from both paths are fused using RRF, then reranked according to configuration and candidate count. The actual main order is as follows:

1. Expand the query and prepare inputs for vector search and term search.
2. Retrieve vector and BM25 candidates. `activity_log` uses BM25 search for activity records.
3. Fuse the ranked lists with RRF using `core/memory/retrieval/pipeline.py`.
4. Apply corrections based on time range and entity, and rerank with a cross-encoder when conditions are met.
5. After reranking, apply time range and entity corrections, then apply access history corrections.
6. The confidence gate assesses result reliability, but candidates below the threshold are not discarded—they are shown as low confidence. If there are no candidates at all, no search results are returned.

Even without an explicit time range, dates or time expressions in the query may be used to correct search candidates. Scope and rerank policies per trigger are described in [Automatic Recall](priming.md).

## Atomic Facts and Entity Index

`facts/` stores `FactRecord` of `core/memory/facts/store.py` as date-based JSONL files. Records include sentences, source/target entities, relation types, effective time and expiration, source episodes, confidence, and more. They are extracted from episodes and can be searched just like normal semantic memory.

`core/memory/facts/entity_index.py` builds `state/entity_registry.json` from facts and uses entity names and aliases for candidate correction during search. For RAG indexing, `core/memory/rag/indexer.py` chunks Markdown and facts, then registers them with metadata into the vector index. The index is derived data with file contents as the source of truth, and periodic processing reflects changes.

## Vector Path

The Chroma persistent store is owned by `MemoryService` of the root process for each Anima. `core/memory/rag/vector_client.py`'s common client and `vector_registry.py` select the connection, and the root itself uses an in-process bridge. The server-side internal vector API forwards to root via IPC, while other processes access it through the HTTP client. Embedding processing is also centralized on the server side.

## RAG Repair

`core/memory/rag/repair/detect.py` requests repair upon signs of corruption. The root `MemoryService` creates a staging index from the original memory file, validates it, and then switches over. If explicit repair is needed during operation, refer to the [CLI reference for `repair-rag`](../reference/cli.md).
