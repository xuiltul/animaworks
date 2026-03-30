# common_knowledge Access Paths

The 5 pathways through which Animas access common_knowledge, and the background RAG indexing mechanism.

---

## Overview of Access Paths

| # | Path | Type | Anima Awareness |
|---|------|------|----------------|
| 1 | System Prompt Hint | Automatic | Anima sees hint and actively accesses |
| 2 | Priming Channel C | Automatic | Relevant knowledge auto-displayed |
| 3 | `search_memory` Tool | Active | Explicit search with scope parameter |
| 4 | `read_memory_file` / `write_memory_file` | Active | Direct access by path |
| 5 | Claude Code Direct File I/O (Mode S) | Active | Direct access via Read/Write tools |

---

## Path 1: System Prompt Hint Injection

`builder.py` injects a **hint text** into Group 4 (Memory & Capabilities) during system prompt construction when `~/.animaworks/common_knowledge/` contains files.

- **Trigger**: Automatic during prompt construction
- **Content**: Hint about common_knowledge existence and usage (file contents not included)
- **Exclusion**: Omitted when `is_task=True` (TaskExec)
- **Anima behavior**: Sees hint, then actively accesses via `search_memory` or `read_memory_file`

## Path 2: Priming Channel C / C0 — RAG Vector Search

`PrimingEngine` automatically performs vector search using message keywords, merging personal knowledge with shared common_knowledge into the system prompt.

- **Channel C budget**: 1200 tokens
- **Channel C0 budget**: 300 tokens (dedicated to overview pointers for chunks tagged with `[IMPORTANT]`)
- **Search target**: `shared_common_knowledge` collection (ChromaDB)
- **Merge method**: Personal knowledge results merged with shared results by score
- **Trust separation**: Channel C results are separated by trust level (medium / untrusted). Chunks from external platforms are treated as untrusted
- **Anima behavior**: Relevant common_knowledge fragments auto-displayed in Priming section

### Note
- The 1200-token constraint means only relevant fragments, not full documents
- Risk of personal knowledge being displaced if common_knowledge document count grows too large
- `[IMPORTANT]` chunks are always injected via Channel C0, making them effective for reliable recall of critical business rules

## Path 3: `search_memory` Tool

When an Anima calls `search_memory(query="...", scope="common_knowledge")`, it performs hybrid keyword + vector search on common_knowledge.

- **Keyword search**: Text scan of .md files in `~/.animaworks/common_knowledge/`
- **Vector search**: Searches `shared_common_knowledge` collection
- **Scope**: `knowledge` / `episodes` / `procedures` / `common_knowledge` / `skills` / `activity_log` / `all`. Use `"common_knowledge"` for targeted search; `"all"` (default) also includes it
- **`scope="all"`**: Merges vector results with **activity_log BM25** hits using **RRF** (reciprocal rank fusion), so broad searches also surface recent unified activity log entries alongside indexed memory chunks

### Examples
```
search_memory(query="message sending", scope="common_knowledge")
search_memory(query="rate limit", scope="all")
```

## Path 4: `read_memory_file` / `write_memory_file`

When an Anima calls `read_memory_file(path="common_knowledge/...")`, the path prefix is detected and resolved to `~/.animaworks/common_knowledge/`.

- **Read**: All Animas can access
- **Write**: All Animas can access (for accumulating shared knowledge)
- **Path traversal defense**: `is_relative_to` check prevents access outside common_knowledge

### Examples
```
read_memory_file(path="common_knowledge/00_index.md")
write_memory_file(path="common_knowledge/operations/new-guide.md", content="...")
```

## Path 5: Claude Code Direct File I/O (Mode S Only)

In Mode S, Claude Code's built-in tools (Read, Write, Grep, Glob, etc.) can directly access `~/.animaworks/common_knowledge/`.

- **Permission**: Allowed as a shared read-only directory in `handler_perms.py`
- **Applicable mode**: Mode S (Agent SDK) only

---

## Background: RAG Index Construction

For common_knowledge to appear in vector search (Paths 2 & 3), it must be indexed in ChromaDB.

### Indexing Timing

1. **Anima startup**: `_ensure_shared_knowledge_indexed()` is called during `MemoryManager` initialization, using SHA-256 hash to detect changes. Re-indexes into `shared_common_knowledge` collection when changed
2. **Daily at 04:00**: `_run_daily_indexing()` incrementally updates all Anima vectorDBs. common_knowledge is also re-indexed at this time

### Chunking Strategy

For `memory_type="common_knowledge"`, uses the same **Markdown heading split** strategy as personal knowledge.

### Collection Name

`shared_common_knowledge` (single collection shared by all Animas)

---

## Difference from reference/

| Item | common_knowledge | reference |
|------|-----------------|-----------|
| RAG Index | Indexed (`shared_common_knowledge`) | **Not indexed** |
| `search_memory` | Searchable via `knowledge`, `episodes`, `procedures`, `common_knowledge`, `skills`, `activity_log`, `all` (`reference/` not indexed) | Not searchable |
| Priming Channel C | Fragments auto-displayed | Not displayed |
| `read_memory_file` | Read/write allowed | **Read-only** |
| Purpose | Everyday practical guides & decision criteria | Detailed technical reference |
| System prompt | Hint injected | Hint injected (separate section) |
