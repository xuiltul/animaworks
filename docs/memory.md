# AnimaWorks Memory System Design Specification

**[日本語版](memory.ja.md)**

> Created: 2026-02-14
> Updated: 2026-03-25
> Related: [vision.md](vision.md), [spec.md](spec.md), [specs/20260214_priming-layer_design.md](specs/20260214_priming-layer_design.md)


---

## Design Philosophy

The AnimaWorks memory system is designed around **human brain memory mechanisms**.

The human brain has distinct systems—working memory, episodic memory, semantic memory, and procedural memory—each processed in different regions. Recall uses two pathways: **automatic recall (priming)** and **intentional recall**. Consolidation follows a three-stage automatic process: immediate encoding, sleep-time consolidation, and long-term integration.

AnimaWorks reproduces these mechanisms faithfully. The agent (LLM) is a “thinking person,” not a “manager of its own brain.” The framework owns memory infrastructure; encoding and consolidation invoke a separate LLM in the background with one-shot calls (independent of the agent’s own LLM session).

---

## Mapping to the Human Memory Model

| Human memory | Brain region | AnimaWorks implementation | Characteristics |
|---|---|---|---|
| **Working memory** | Prefrontal cortex | LLM context window | Capacity-limited. Holds “what is being thought about now.” Spotlight on activated long-term memory |
| **Episodic memory** | Hippocampus → neocortex | `episodes/` | “What happened when.” Stored chronologically as daily logs. Automatically recorded by the framework at conversation end |
| **Semantic memory** | Temporal cortex | `knowledge/` | “What is known.” Lessons, policies, and knowledge abstracted from context. Extracted from episodes in daily consolidation |
| **Procedural memory** | Basal ganglia, cerebellum | `procedures/`, `skills/` | “How to do it.” Work procedures, skills, workflows |
| **Interpersonal memory** | Fusiform gyrus, temporal pole | `shared/users/` | “Who this person is.” User profiles shared across Animas |

### Working memory = context window

Based on Baddeley’s (2000) working memory model.

- **Central executive** = Agent orchestrator. Coordinates attention control and retrieval from long-term memory
- **Episodic buffer** = Context assembly layer. Integrates priming output and conversation history into a unified representation
- **Phonological loop** = Text buffer. Holds recent conversation turns

Following Cowan (2005), working memory is treated as **activated long-term memory**. The context window is not a separate store; it is the part of long-term memory currently under attention.

### Long-term memory = file-based archive

Memories are not truncated into the prompt; they live in a **file-system archive** (archive-based memory). There is no cap on how much can be stored. Only what is needed **now** enters the context.

```
~/.animaworks/animas/{name}/
├── activity_log/    Unified activity log (JSONL timeline of all interactions)
├── episodes/        Episodic memory (daily logs, action records)
├── knowledge/       Semantic memory (learned knowledge, lessons, policies)
├── procedures/      Procedural memory (work procedure documents)
├── skills/          Skill memory (per-Anima skills)
├── shortterm/       Short-term memory (session state, streaming journal; chat/heartbeat split)
└── state/           Persistent working-memory slice (current task, short-term state)
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│          Working memory (prefrontal cortex)          │
│          = LLM context window                        │
│                                                        │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐   │
│  │ Central     │  │ Episodic   │  │ Phonological │   │
│  │ executive   │  │ buffer     │  │ loop         │   │
│  │ = orchestr- │  │ = context  │  │ = text       │   │
│  │   ator      │  │   assembly │  │   buffer     │   │
│  │             │  │   layer    │  │              │   │
│  └──────┬──────┘  └─────┬──────┘  └──────────────┘   │
│         │               │                              │
│    Intentional     Automatic recall                    │
│    search          (priming)                           │
│    (search_memory)                                     │
└─────────┬──────────────┬───────────────────────────────┘
          │              │
    ┌─────┴──────┐  ┌───┴──────────────────┐
    │ Prefrontal │  │  Priming layer         │
    │ cortex     │  │  = automatic recall    │
    │ = intention│  │  Framework runs        │
    │   al search│  │  automatically         │
    │ Agent      │  │                        │
    │ invokes    │  │                        │
    │ tools      │  │                        │
    └─────┬──────┘  └───┬──────────────────┘
          │              │
          │    ┌─────────┴────────────────┐
          │    │  Spreading activation    │
          │    │  Vector similarity +     │
          │    │  temporal decay          │
          │    │  → Related memories      │
          │    │    auto-activated        │
          │    └─────────┬────────────────┘
          │              │
┌─────────┴──────────────┴───────────────────────────────┐
│         Long-term memory (hippocampus + cortex)          │
│                                                          │
│  ┌───────────────────────────────────────────────┐      │
│  │  Unified activity log  activity_log/           │      │
│  │  = JSONL timeline of all interactions          │      │
│  │  Priming “recent activity” source               │      │
│  └───────────────────────────────────────────────┘      │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐    │
│  │ Episodic   │  │ Semantic   │  │ Procedural     │    │
│  │ memory     │  │ memory     │  │ memory         │    │
│  │ episodes/  │  │ knowledge/ │  │ procedures/    │    │
│  │            │  │            │  │ skills/        │    │
│  │ Daily logs │  │ Learned    │  │ Procedures &   │    │
│  │ Action log │  │ knowledge  │  │ skills         │    │
│  │            │  │ Lessons &  │  │ Workflows      │    │
│  │            │  │ policies   │  │                │    │
│  └────────────┘  └────────────┘  └────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │  Shared memory  shared/                        │     │
│  │  users/           Interpersonal (profiles)     │     │
│  │  resolutions.jsonl Resolution registry         │     │
│  │                    (cross-Anima)               │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │  Streaming journal  shortterm/                 │     │
│  │  = WAL (Write-Ahead Log). Crash-resilient      │     │
│  │  Streaming output persisted incrementally      │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  -- Consolidation (Anima-led + framework post) --       │
│                                                          │
│  [Immediate] Session boundary → diff summary → episodes/ │
│         + auto state update + resolution propagation     │
│  [Daily] Midnight cron → Anima.run_consolidation("daily") │
│         (tools: extract knowledge, create procedures,     │
│          resolve contradictions)                          │
│         → Post: Synaptic Downscaling + RAG rebuild       │
│  [Weekly] Weekly cron → Anima.run_consolidation("weekly") │
│         → Post: Neurogenesis reorganization + RAG rebuild│
│  [Monthly] Monthly cron → complete forgetting +          │
│         archive cleanup                                   │
│                                                          │
│  -- Forgetting (synaptic homeostasis) --                 │
│                                                          │
│  [Daily] Synaptic downscaling: knowledge(90d)            │
│         + procedures(180d or low utility) → low-activity   │
│  [Weekly] Neurogenesis reorganization: LLM merge of      │
│         low-activity + similar chunks                   │
│  [Monthly] Complete forgetting: low-activity 90d+      │
│         + access_count≤2 → archive & delete             │
│         Move to archive/forgotten/ + archive/versions/    │
│         cleanup                                           │
│                                                          │
│  * Agent: intentional memorization only                 │
│    (write_memory_file)                                   │
└──────────────────────────────────────────────────────────┘
```

---

## Memory recall: two pathways

Human recall is not one process; it combines **automatic recall** and **intentional recall**. AnimaWorks implements both.

### Automatic recall — priming layer

**Neuroscience basis**: When perceptual input arrives, the auto-associative network in hippocampal CA3 runs pattern completion automatically. Unconscious, fast (250–500 ms), hard to suppress.

**AnimaWorks implementation**: On message receipt, the framework searches related memories and injects them into context **before** the agent runs. From the agent’s view, memory is “already recalled” when the turn starts.

```
Message received → Context extraction → Priming search → Context assembly → Agent run
                   (sender, keywords)   (multi-source     (within token      (memory already
                                         parallel)         budget)            present)
```

Priming lives in `core/memory/priming/` (`engine.py`: `PrimingEngine`). `prime_memories()` runs **nine coroutines in parallel** via `asyncio.gather`: sender **A**, recent activity **B**, **C0: [IMPORTANT] knowledge**, related knowledge **C**, skills/procedures **D**, pending tasks **E**, **Recent Outbound**, episodes **F**, and **pending human notifications**. Class comments that say “6-channel” refer mainly to A–F; C0 and the outbound-related collectors run as additional gathers in parallel. After fetch, each block is `truncate_*`’d per `token_budget` ratios and assembled into the injected string.

**The actual dynamic-budget switch** is the `prime_memories(..., enable_dynamic_budget=...)` argument—not `PrimingConfig.dynamic_budget`. Today `config.json` `priming.dynamic_budget` is schema-only and **not read by the engine**. Normal chat paths pass `enable_dynamic_budget=True` from `core/_agent_priming.py`. When `False`, the overall cap is fixed at `_DEFAULT_MAX_PRIMING_TOKENS` (~2000 token-equivalent).

**Main data sources and default budgets** (`priming/constants.py` allocations × dynamic-budget ratio; when dynamic budget is off, overall cap ~2000 token-equivalent):

| Source | Target | Default allocation (tokens) | Method | Brain analog |
|---|---|---|---|---|
| **A: Sender profile** | shared/users/ | 500 | Exact lookup | Recall the moment you see a face |
| **B: Recent activity** | activity_log/ + shared/channels/ | 1300 (applied as `max(400, 1300×ratio)`) | ActivityLogger + shared channels | Short- to recent-term memory |
| **C0: [IMPORTANT] knowledge** | knowledge / shared_common_knowledge | 500 (summary pointer) | `get_important_chunks()` | Always-on salient memory |
| **C: Related knowledge** | knowledge + shared common_knowledge | 1000 (trusted first, remainder untrusted) | 1–3 vector queries (`build_queries`) + trust split | Semantic association |
| **D: Skill / procedure match** | skills/, procedures/, common_skills/ | 200 | Description-based 3-stage + vector | **Names only** (channel D has **no** heartbeat/cron skip; count is trimmed later by budget) |
| **E: Open tasks** | task_queue + parallel tasks | 500 | TaskQueueManager, etc. | “What must be done” |
| **F: Episodes** | episodes/ | 800 | Vector search (RAG) + optional graph spread | Semantic search over past actions |
| **Recent Outbound** | activity_log/ | Last 2 h, max 3 | `channel_post`, `message_sent` | Self-awareness of outbound behavior |
| **Pending human notifications** | activity_log/ `human_notify` | ~500 tokens max | Last 24 h; `chat` / `heartbeat` / `message:*` only | Unprocessed `call_human` context |

Channel B merges `ActivityLogger.recent(days=2, limit=100)` with `shared/channels/*.jsonl` (membership via `is_channel_member`; latest five per channel plus human posts and @mentions within 24 h). Shared-channel entries are capped at **15** after time sort (`_MAX_CHANNEL_ENTRIES`). If the activity log is empty, it falls back to `episodes/` plus legacy channel reads.

**Noise filtering (channel B)** depends on trigger.

- **heartbeat**, or channel starting with `cron:`: exclude `tool_use`, `tool_result`, `heartbeat_start`, `heartbeat_reflection`, `inbox_processing_start`, `inbox_processing_end` (suppress execution detail, HB self-reference, Inbox lifecycle). **`heartbeat_end` is not excluded.**
- **chat and other foreground**: additionally exclude `memory_write`, `cron_executed`, `heartbeat_end`, `heartbeat_start`, `heartbeat_reflection`, `inbox_*`, prioritizing messaging, errors, and tasks.

**Priority scoring** (`channel_b.prioritize_entries`): take top 50 by score, sort chronologically, then pass to `format_for_priming`.

| Factor | Score | Description |
|---|---|---|
| Own sends / responses | +15.0 | `message_sent`, `response_sent` |
| `message_received` | +15.0 | When `meta.from_type != "anima"` or `origin_chain` contains `"human"` |
| Sender relevance | +10.0 | `from_person` / `to_person` matches current sender |
| Keywords | +3.0 each | Match between body/summary and keywords |
| Recency | variable | `elapsed_seconds / 600` vs. first entry |

**Channel C (related knowledge)**: merge personal collection and `shared_common_knowledge` with `include_shared=True`; drop hits below raw vector similarity `config.json` `rag.min_retrieval_score` (default 0.3). Resolve each chunk’s `origin` with `resolve_trust`; split into **medium** and **untrusted** strings. On injection, trusted text is taken from the head under `_BUDGET_RELATED_KNOWLEDGE`, then untrusted with remaining budget. The C0 “[IMPORTANT] Knowledge (summary pointer)” block is **prepended** to this related text (priority within the same budget pool).

**Budget management**: When `enable_dynamic_budget=True`, `budget.adjust_token_budget()` sets the overall token cap from `classify_message_type()` (short greeting / question / long request / heartbeat; Inbox `intent` can override) and `context_window`; each channel allocation is scaled by `budget_ratio = token_budget / 2000`. On heartbeat, use `max(budget_heartbeat, int(context_window × heartbeat_context_pct))` (overridable in `PrimingConfig`). Channel B is first formatted with `format_for_priming(..., budget_tokens=1300)`, then finally `truncate_tail` to `budget_activity = max(400, int(_BUDGET_RECENT_ACTIVITY × budget_ratio))` character-equivalent.

Channel D matches `skills/`, shared `common_skills/`, and `procedures/` with `match_skills_by_description()` (3-stage). `channel_d` returns at most **five** names; the engine further trims to `matched_skills[: max(1, budget_skills // 50)]`. **Bodies are not returned—skill names only** (details via the `skill` tool). **There is no heartbeat/cron-only skip**; empty messages can still match if `keywords` or state-derived context exist.

Dynamic budget by message type (`PrimingConfig` defaults; overridable in `config.json`):

| Message type | Token budget | Use case |
|---|---|---|
| greeting | 500 | Short greetings, low load |
| question | 2000 | Questions, moderate memory search |
| request | 3000 | Requests/instructions, broad memory search |
| heartbeat | 200 (+ scales with large context) | Periodic patrol; large `context_window` extends up to `heartbeat_context_pct` |

### Intentional recall — `search_memory` tool

**Neuroscience basis**: PFC monitors automatic recall and runs strategic search when insufficient. Conscious and slower.

**AnimaWorks implementation**: Only when priming is insufficient does the agent call `search_memory` / `read_memory_file`.

Typical cases:

- Need exact dates, times, or numbers
- Need details of a specific past exchange
- Following a procedure document
- Unknown topic with no matching memory in context

---

## Memory search via spreading activation

**Neuroscience basis**: Collins & Loftus (1975) spreading activation. Semantic memory is a network of concept nodes linked by associations; activating one node spreads to neighbors (e.g. “doctor” primes “nurse,” “hospital”).

**AnimaWorks implementation**: Dense vector search, temporal decay, importance boost, and graph spreading activation. Default `config.rag.enable_spreading_activation` is **True** (`RAGConfig` in `core/config/schemas.py`). `MemoryRetriever.search(..., enable_spreading_activation=None)` reads config and disables spread only on config load failure. Applicable types: `spreading_memory_types` (default `knowledge`, `episodes`).

Early design used BM25 + vector with RRF; evaluation showed multilingual dense vector search alone beats keyword search, so retrieval is **vector similarity only**.

| Search signal | Method | Brain analog |
|---|---|---|
| **Semantic vector** | Dense similarity (`intfloat/multilingual-e5-small`, 384-d, ChromaDB) | Conceptual neighbors; spreading activation approximated |
| **Temporal decay** | Exponential decay (half-life 30 days) + `WEIGHT_RECENCY` 0.2 | Recency (`updated_at`) |
| **Access frequency** | `min(WEIGHT_FREQUENCY × log1p(access_count), WEIGHT_FREQUENCY × FREQUENCY_LOG_CAP)` | Shared chunks prefer per-Anima key `ac_{anima_name}`; cap prevents score blow-up |
| **Importance** | Metadata `importance == "important"` ([IMPORTANT]) adds `+0.20` | Flat salience boost (amygdala-like) |
| **Graph spread** | Knowledge graph + Personalized PageRank | Multi-hop association; up to `max_graph_hops` (default 2). Explicit `[[link]]` + implicit (similarity ≥ threshold) |

**Other retrieval behavior** (`MemoryRetriever`):

- **superseded exclusion**: For `memory_type=="knowledge"` and `include_superseded=False` (default), exclude chunks with non-empty metadata `valid_until` from vector search.
- **Shared collections**: With `include_shared=True`, merge `shared_common_knowledge` (and `shared_common_skills` for skill search).

After score adjustment on vector hits, optionally append graph-spread results (spread-only hits bypass the `min_score` filter).

```
Adjusted vector score = vector_similarity
                     + WEIGHT_RECENCY × (0.5 ^ (age_days / 30))
                     + min(WEIGHT_FREQUENCY × log1p(access_count), WEIGHT_FREQUENCY × FREQUENCY_LOG_CAP)
                     + (importance=="important" ? WEIGHT_IMPORTANCE : 0)

WEIGHT_RECENCY = 0.2
WEIGHT_FREQUENCY = 0.1
FREQUENCY_LOG_CAP = 3.0   # effective max WEIGHT_FREQUENCY × 3.0
WEIGHT_IMPORTANCE = 0.20

# Neighbor nodes added by graph spread:
spread_contribution = pagerank_score × 0.5
```

### Graph spread implementation flow

> Implementation: `core/memory/rag/graph.py` — `KnowledgeGraph`, `core/memory/rag/retriever.py` — `_apply_spreading_activation()`

From vector hits, run Personalized PageRank on the knowledge graph to activate related memories missed by direct search.

```
Vector search → initial hits
    │
    ▼
Map initial doc_ids to graph nodes
    │
    ▼
Personalized PageRank (alpha=0.85, `PAGERANK_*` in `graph.py`)
    │  personalization weights on seeds
    │  edge "similarity" as weight
    │
    ▼
Top 5 activated neighbors (`max_hops` from `rag.max_graph_hops`, default 2)
    │  exclude initial-result nodes
    │  only nodes with score > 0.001
    │
    ▼
Load content for activated nodes (file or vector store)
    │
    ▼
Append to final results with score × 0.5 (tag `activation: "spreading"`)
```

### Knowledge graph structure

| Element | Description |
|---|---|
| **Nodes** | Each `.md` under `knowledge/` and `episodes/`. Attributes: `path`, `memory_type`, `stem` |
| **Explicit links** | Wikilinks `[[filename]]` / `[[filename|display]]`. `similarity=1.0` |
| **Implicit links** | Per-node embedding, top-5 similar docs; edges if similarity ≥ 0.75. `similarity=score` |

Cached at `{anima_dir}/vectordb/knowledge_graph.json`; incrementally updated when memory files change.

### Main graph spread / RAG settings (`config.json` → `RAGConfig`)

| Parameter | Default | Description |
|---|---|---|
| `enable_spreading_activation` | `true` | Enable/disable graph spread |
| `max_graph_hops` | `2` | Max hops for graph build and spread |
| `implicit_link_threshold` | `0.75` | Implicit-link similarity threshold |
| `spreading_memory_types` | `["knowledge", "episodes"]` | Memory types in the graph and subject to spread |
| `min_retrieval_score` | `0.3` | Minimum raw vector similarity for priming, etc. (`None` disables) |
| `skill_match_min_score` | `0.75` | Vector-stage threshold for skill matching |
| `enabled` | `true` | Master RAG enable flag |
| `embedding_model` | `intfloat/multilingual-e5-small` | Embedding model id |
| `use_gpu` | `false` | Use GPU for embedding inference |
| `enable_file_watcher` | `true` | Watch memory files (incremental index) |
| `graph_cache_enabled` | `true` | Cache knowledge graph JSON |

---

## YAML frontmatter

`knowledge/` and `procedures/` files carry YAML frontmatter for structured metadata. The daily consolidation pipeline assigns it; legacy migration backfills existing files.

### knowledge/ frontmatter

```yaml
---
created_at: "2026-02-18T03:00:00+09:00"
updated_at: "2026-02-18T03:00:00+09:00"
source_episodes: 3
confidence: 0.9
auto_consolidated: true
version: 1
---
```

| Field | Type | Description |
|---|---|---|
| `created_at` | ISO8601 | Created |
| `updated_at` | ISO8601 | Last updated |
| `source_episodes` | int | Count of source episodes |
| `confidence` | float | Confidence (NLI+LLM validation), 0.0–1.0 |
| `auto_consolidated` | bool | Produced by automatic consolidation |
| `version` | int | Version (increment on re-consolidation) |
| `superseded_by` | str | New file that replaced this (contradiction resolution) |
| `supersedes` | str | Old file this one replaced |

### procedures/ frontmatter

```yaml
---
description: Procedure description
confidence: 0.5
success_count: 0
failure_count: 0
version: 1
created_at: "2026-02-18T03:00:00+09:00"
updated_at: "2026-02-18T03:00:00+09:00"
auto_distilled: true
protected: false
---
```

| Field | Type | Description |
|---|---|---|
| `description` | str | Procedure description (used for skill matching) |
| `confidence` | float | `success_count / max(1, success_count + failure_count)` |
| `success_count` | int | Successes |
| `failure_count` | int | Failures |
| `version` | int | Version (increment on re-consolidation) |
| `created_at` | ISO8601 | Created |
| `updated_at` | ISO8601 | Last updated |
| `auto_distilled` | bool | Produced by automatic distillation |
| `protected` | bool | Manual protection from forgetting |

---

## Memory consolidation: three-stage automatic process

The brain consolidates unconsciously. AnimaWorks combines **Anima-led consolidation** and **framework post-processing**.

- **Anima-led**: In `run_consolidation()`, the Anima uses tools (`search_memory`, `read_memory_file`, `write_memory_file`, `archive_memory_file`) to summarize episodes, extract knowledge, resolve contradictions, and create procedures.
- **Framework post-processing**: Synaptic downscaling (metadata), RAG rebuild, and monthly forgetting run automatically.

```
Waking (conversation)                     Sleeping (no conversation)
────────────────                          ────────────────

 Conversation → session boundary           Midnight cron
     │  (10 min idle or heartbeat)            │
     ▼                                      ▼
 [Immediate encoding]                      [Daily consolidation]
 Diff summary → episodes/                  Anima.run_consolidation("daily")
 + auto state update                       (tools: knowledge, procedures,
 + resolution propagation                  contradiction resolution)
 Hippocampal one-shot log                  → Post: Synaptic downscaling
                                           → Post: RAG rebuild
                                                │
                                           Weekly cron
                                                │
                                                ▼
                                           [Weekly integration]
                                           Anima.run_consolidation("weekly")
                                           → Post: Neurogenesis reorganization
                                           → Post: RAG rebuild
                                                │
                                           Monthly cron
                                                │
                                                ▼
                                           [Monthly forgetting]
                                           ForgettingEngine.complete_forgetting()
                                           archive/versions/ cleanup (old procedure versions)
```

### Daily consolidation flow

> Implementation: `core/_anima_lifecycle.py` — `Anima.run_consolidation()`, `core/memory/consolidation.py` — `ConsolidationEngine` (pre/post)
> Schedule: `core/lifecycle/system_crons.py` registers the daily handler from `core/lifecycle/system_consolidation.py` (`ConsolidationConfig.daily_time`, default 02:00 JST)

**1. Preprocessing** (ConsolidationEngine): collect four inputs and inject into `consolidation_instruction`:

| Collected data | Method | Content |
|---|---|---|
| Recent episodes | `_collect_recent_episodes(hours=24)` | `episodes/` from last 24 h |
| Resolved events | `_collect_resolved_events(hours=24)` | `issue_resolved` in activity_log |
| Activity summary | `_collect_activity_entries(hours=24)` | Comms + `tool_result` (~4000 token cap) |
| Reflections | `_extract_reflections_from_episodes()` | `[REFLECTION]...[/REFLECTION]` in episodes |

**2. Anima run**: Follow `consolidation_instruction` with tools (`max_turns=30`):

1. Review today’s episodes and resolved events
2. `search_memory` over existing `knowledge/` and `procedures/`
3. `write_memory_file` to update or create knowledge
4. Record lessons/procedures from resolved events under `procedures/`
5. `archive_memory_file` for duplicates or stale entries

**3. Post-processing**: `ForgettingEngine.synaptic_downscaling()`, `ConsolidationEngine._rebuild_rag_index()`

### Weekly integration flow

> Schedule: weekly handler in `system_consolidation.py` (`ConsolidationConfig.weekly_time`, default `sun:03:00` JST)

**1. Anima run**: `run_consolidation("weekly")` follows `weekly_consolidation_instruction`:

| Task | Content |
|---|---|
| **knowledge merge** | List `knowledge/`, find duplicates with `search_memory`, merge and archive |
| **procedure cleanup** | Update or archive stale/unused procedures |
| **episode compression** | Compress episodes older than 30 days to essentials (skip `[IMPORTANT]`) |
| **contradiction resolution** | Detect conflicting knowledge; archive or merge |

**2. Post-processing**: `ForgettingEngine.neurogenesis_reorganize()` (async), RAG rebuild

### Monthly forgetting pipeline

> Schedule: monthly forgetting in `system_consolidation.py` (`ConsolidationConfig.monthly_time`, default day 1 04:00 JST)

- `ForgettingEngine.complete_forgetting()` (knowledge + episodes + procedures)
- `cleanup_procedure_archives()` — prune old procedure snapshots under `archive/versions/` (keep latest five per procedure stem, `PROCEDURE_ARCHIVE_KEEP_VERSIONS`)

### Models used for consolidation

Daily and weekly consolidation run as **background triggers** (`consolidation:daily`, `consolidation:weekly`). Resolution order:

1. Per-Anima `status.json` `background_model`
2. `config.json` `heartbeat.default_model`
3. Main `model` fallback

A lighter `background_model` (e.g. `claude-sonnet-4-6`) keeps a heavy main model (e.g. `claude-opus-4-6`) for chat while cutting consolidation cost. Weekly `neurogenesis_reorganize` (LLM merge) uses the same resolution.

### Consolidation stages summary

| Stage | Brain process | AnimaWorks implementation | Owner | Frequency |
|---|---|---|---|---|
| **Immediate encoding** | Fast hippocampal encoding | Session boundary (10 min idle or heartbeat) → diff summary → `episodes/` + auto state + resolution propagation | Framework (bg LLM) | At session boundary |
| **Daily consolidation** | NREM slow-wave / spindle / ripple cascade | Midnight cron → `run_consolidation("daily")` → post: synaptic downscaling + RAG rebuild | Anima + framework post | Nightly |
| **Weekly integration** | Cortical long-term integration | Weekly cron → `run_consolidation("weekly")` → post: neurogenesis reorganization + RAG rebuild | Anima + framework post | Weekly |
| **Monthly forgetting** | Sub-threshold synapse loss | Monthly cron → `complete_forgetting()` + archive cleanup | Framework (bg cron) | Monthly |
| **Intentional memorization** | PFC elaborative encoding | `write_memory_file` | Agent | On demand |

The agent’s only write path is **intentional memorization** (`write_memory_file`)—like conscious note-taking. Daily/weekly consolidation is tool-driven by the Anima; downscaling, RAG rebuild, and monthly forgetting are automatic.

### Immediate encoding: session-boundary diff summarization

The old design re-summarized all turns on every reply, duplicating the same conversation N−2 times. The current design tracks `last_finalized_turn_index` and **diff-summarizes only unrecorded turns**.

**Session boundary**: `finalize_session()` runs not on every reply, only when:

- **10-minute idle**: 10 minutes since last turn (`finalize_if_session_ended()`)
- **Heartbeat**: `finalize_if_session_ended()` during periodic patrol

**Integration**: `finalize_session()` also:

1. **Episode log**: Append LLM summary of unrecorded turns to `episodes/`
2. **Auto state update**: Parse resolved items and new tasks from the summary → append to `state/current_state.md`
3. **Resolution propagation**: Log to ActivityLogger (`issue_resolved`) and `shared/resolutions.jsonl`
4. **Turn compression**: Merge recorded turns into `compressed_summary` to cap `conversation.json` growth

### Resolution propagation mechanism

Resolution propagates in three layers to the local Anima and others:

| Layer | Target | Implementation | Propagates to |
|---|---|---|---|
| **1: ActivityLogger** | Local Anima | `issue_resolved` in activity_log | Priming channel B (local recent activity) |
| **2: Resolution registry** | All Animas | `shared/resolutions.jsonl` | builder.py “Resolved issues” (all system prompts) |
| **3: Consolidation injection** | Local Anima | `_collect_resolved_events()` | Daily consolidation prompt (refresh “unresolved” in knowledge/ to “resolved”) |

---

## Knowledge validation: NLI+LLM cascade

> Implementation: `core/memory/validation.py` — `KnowledgeValidator`

Writing raw LLM-extracted knowledge risks hallucinations. A **cascade of NLI and LLM** filters candidates. In Anima-led daily consolidation the Anima writes via tools, so this pipeline is used on other paths (batch, legacy).

### NLI model

- Model: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
- Multilingual zero-shot NLI (includes Japanese)
- GPU if available, else CPU
- If NLI unavailable, LLM-only validation (graceful degradation)

### Cascade flow

```
Knowledge candidate (premise: source episode, hypothesis: extracted knowledge)
    │
    ▼
[NLI]
    ├── entailment ≥ 0.6  → approve confidence=0.9 (skip LLM)
    ├── contradiction ≥ 0.7 → reject (skip LLM)
    └── neutral / below threshold → LLM review
                                  │
                                  ▼
                             [LLM]
                                  ├── approve → write confidence=0.7
                                  └── reject → discard
```

High-confidence NLI skips the LLM for cost/latency; ambiguous neutral cases go to the LLM.

---

## Knowledge contradiction detection and resolution

> Implementation: `core/memory/contradiction.py` — `ContradictionDetector`

Contradictions can appear across `knowledge/` (e.g. “A owns X” vs. “A owns Y”). Anima-led consolidation resolves them per `consolidation_instruction` with tools. `ContradictionDetector` is a utility for automatic NLI+LLM detection/resolution.

### Contradiction detection flow

```
New/updated knowledge file
    │
    ▼
[RAG] Similar knowledge
    │
    ▼
[NLI] Per pair: entailment / contradiction / neutral
    ├── entailment ≥ 0.7  → no contradiction (skip LLM)
    ├── contradiction ≥ 0.7 → contradiction → LLM resolution
    └── neutral / below threshold → LLM deep dive
                                  │
                                  ▼
                             [LLM]
                                  ├── contradiction → choose strategy
                                  └── none → skip
```

### Three resolution strategies

| Strategy | Condition | Action |
|---|---|---|
| **supersede** | New clearly updates old | Old gets `superseded_by` and archive; new records `supersedes` |
| **merge** | Both mergeable | LLM merged text, new file; archive both originals |
| **coexist** | Both true in different contexts | Annotate both with contradiction and conditions |

### When it runs

| When | Target | Description |
|---|---|---|
| **Daily** | Files created/updated that day | Final step of daily consolidation |
| **Weekly** | All `knowledge/` | Catch contradictions missed daily |

---

## Procedural memory lifecycle

> Implementation: `core/memory/distillation.py` — `ProceduralDistiller`, `core/memory/reconsolidation.py` — `ReconsolidationEngine`

Procedural memory (`procedures/`) holds “how to do it” (basal ganglia / cerebellum analog). Semantic memory (`knowledge/`) holds static “what is known”; procedural memory strengthens with execution and feedback.

### Procedure creation

**Anima-led** (inside `run_consolidation()`):

Per `consolidation_instruction`, the Anima creates/updates `procedures/` via `write_memory_file`. Lessons from resolved events land here too.

**ReconsolidationEngine** (alternate path):

`create_procedures_from_resolved()` scans `issue_resolved` and builds procedures with `ProceduralDistiller`. Not invoked on the main daily path; usable from batch jobs.

### 3-stage matching (skill injection)

Priming (channel D) and `builder.py` skill injection match `procedures/` to the message:

| Stage | Method | Description |
|---|---|---|
| **1. Bracket keyword** | `[keyword]` exact match | Match when `[keyword]` in the message appears in frontmatter `description` |
| **2. Lexical match** | Content-word overlap | Rank by overlap of content words (nouns, verbs, etc.) |
| **3. RAG vector** | Dense similarity | Semantic search with sentence-transformers |

Stage 1 wins; stage 3 is fallback—from explicit keywords to fuzzy semantic recall.

### Success/failure tracking

Procedure confidence updates from execution feedback:

| Tracking | Description |
|---|---|
| **`report_procedure_outcome` tool** | Agent reports success/failure explicitly |
| **Framework auto-tracking** | For procedures injected in-session, outcome inferred at session boundary |

```
confidence = success_count / max(1, success_count + failure_count)
```

Initial values (auto-distillation): `confidence: 0.4`, `success_count: 0`, `failure_count: 0`

### Prediction-error-based reconsolidation

> Implementation: `core/memory/reconsolidation.py` — `ReconsolidationEngine`

**Neuroscience basis**: Nader et al. (2000) reconsolidation. Recalled traces destabilize and reconsolidate after new integration. Prediction error triggers reconsolidation.

**AnimaWorks implementation**: Anima-led consolidation follows `consolidation_instruction` (“cross-check existing knowledge,” “archive contradictions”) via tools. `ReconsolidationEngine` supports automatic NLI+LLM reconsolidation on alternate paths. Flow:

```
New episode
    │
    ▼
[RAG] Related knowledge/procedures
    │
    ▼
[NLI] Episode vs. existing memory
    ├── no contradiction → skip
    └── contradiction → LLM analysis
                      │
                      ▼
                 [LLM update decision]
                      ├── update needed → old version to archive/versions/
                      │                   memory update, version++
                      └── no update → skip
```

**Special handling for `procedures/` reconsolidation**:

- Old version saved under `archive/versions/` (`ReconsolidationEngine._archive_version`)
- Increment `version`
- Reset `success_count: 0`, `failure_count: 0`, `confidence: 0.5` (re-validation)
- Update `updated_at`

---

## Active forgetting: synaptic homeostasis

The brain actively forgets as well as remembers. AnimaWorks implements three forgetting stages based on synaptic homeostasis (Tononi & Cirelli, 2003).

```
Waking (conversation)                     Sleeping (no conversation)
────────────────                          ────────────────

 Conversation/search → access_count++      Midnight cron
     │                                      │
     ▼                                      ▼
 [Access log]                              [Daily downscaling]
 Frequently used memories strengthen        knowledge: 90d+ no access + low frequency
 (Hebb / LTP)                               procedures: 180d+ unused + low frequency
                                            or utility<0.3 + failure≥3 → immediate mark
                                            (synaptic homeostasis)
                                                 │
                                            Weekly cron
                                                 │
                                                 ▼
                                            [Neurogenesis reorganization]
                                            LLM merge of low-activity + similar memory
                                                 │
                                            Monthly cron
                                                 │
                                                 ▼
                                            [Complete forgetting]
                                            Low-activity 90d+ + access_count≤2
                                            → archive delete
                                            knowledge + episodes + procedures
                                            archive/versions/ cleanup
```

| Stage | Brain process | AnimaWorks implementation | Frequency |
|---|---|---|---|
| **Daily downscaling** | NREM synaptic downscaling | knowledge: 90d+ no access → low-activity mark. procedures: 180d+ unused or utility<0.3+failure≥3 → low-activity mark | Daily cron |
| **Neurogenesis reorganization** | Dentate gyrus neurogenesis rewiring | LLM merge of similar low-activity pairs | Weekly cron |
| **Complete forgetting** | Sub-threshold synapse loss | Remove vector index for low-activity 90d+ + access_count≤2; archive sources (knowledge + episodes + procedures) | Monthly cron |

### knowledge/ forgetting thresholds

| Condition | Value | Description |
|---|---|---|
| No access period | 90 days | Since last access |
| Access count | < 3 | Low use frequency |

### procedures/ forgetting thresholds

More lenient than knowledge/ (procedural memory is more forgetting-resistant in the brain too):

| Condition | Value | Description |
|---|---|---|
| Unused period | 180 days | Since last use (2× knowledge) |
| Use count | < 3 | Low use frequency |
| Immediate mark | utility < 0.3 AND failure_count ≥ 3 | Repeated failure on low-utility procedures |

### Protection from forgetting

| Target | Protection | Reason |
|---|---|---|
| `skills/` | Always | Anchor for description-based matching; deletion breaks recall |
| `shared/users/` (memory_type: shared_users) | Always | Interpersonal memory |
| `[IMPORTANT]` / `importance: important` | Implementation note | `_is_protected()` begins with `IMPORTANT_SAFETY_NET_DAYS` (365 days) without access, which can lift protection—but `_is_protected_knowledge` / `_is_protected_procedure` return **`True` again when `importance == "important"`**, so **today [IMPORTANT] knowledge/procedures stay excluded from forgetting even after the safety net expires**. Types without that later check (e.g. episodes) still benefit from the safety net. |
| `knowledge/` (success_count ≥ 2) | Conditional | Knowledge validated useful multiple times |
| `procedures/` (version ≥ 3) | Conditional | Mature after 3+ reconsolidations |
| `procedures/` (`protected: true`) | Conditional | Manual frontmatter flag |
| `procedures/` ([IMPORTANT]) | Conditional | Tag-based resistance |

### Monthly archive cleanup

The monthly pipeline tidies `archive/versions/`. Keep only the five latest versions per procedure stem; delete older snapshots.

---

## Unified activity log

> Implementation: `core/memory/activity.py` — `ActivityLogger` (mixins: `PrimingMixin`, `TimelineMixin`, `ConversationMixin`, `RotationMixin`)

Single JSONL timeline for all interactions. Replaces scattered transcript, dm_log, heartbeat_history, etc., as the sole source for Priming channel B. Split across `_activity_models.py`, `_activity_priming.py`, `_activity_timeline.py`, `_activity_conversation.py`, `_activity_rotation.py`.

### Storage location

```
{anima_dir}/activity_log/{date}.jsonl
```

One file per day; append-only JSONL per line.

### JSONL format

```json
{"ts":"2026-02-17T14:30:00","type":"message_received","content":"...","from":"user","channel":"chat"}
{"ts":"2026-02-17T14:30:05","type":"response_sent","content":"...","to":"user","channel":"chat"}
{"ts":"2026-02-17T15:00:00","type":"tool_use","tool":"web_search","summary":"Search executed"}
```

Empty fields omitted. `from`/`to` are sender/recipient names (`from_person`/`to_person` internally), `channel` is channel name, `tool` tool name, `via` notify channel (for `human_notify`), `meta` optional metadata (`from_type`, etc.). `origin` / `origin_chain` trace provenance (`"human"`, `"external_platform"`, etc.).

### Event types

| Event type | ASCII label | Description |
|---|---|---|
| `message_received` | `MSG<` | Inbound (human or Anima; distinguish with `meta.from_type`) |
| `response_sent` | `RESP>` | Outbound reply to human |
| `message_sent` | `MSG>` | DM to another Anima (renamed from `dm_sent`) |
| `channel_post` | `CH.W` | Shared channel post |
| `channel_read` | `CH.R` | Shared channel read |
| `human_notify` | `NTFY` | Human notification (`call_human`) |
| `tool_use` | `TOOL` | External tool use |
| `heartbeat_start` | `HB` | Heartbeat start |
| `heartbeat_end` | `HB` | Heartbeat end |
| `cron_executed` | `CRON` | Cron run |
| `memory_write` | `MEM` | Memory file write |
| `error` | `ERR` | Error |
| `issue_resolved` | `RSLV` | Issue resolved (from auto state update) |
| `task_created` | `TSK+` | Task created |
| `task_updated` | `TSK~` | Task updated |
| `tool_result` | `TRES` | Tool result (consolidation; metadata only, raw content omitted) |
| `inbox_processing_start` / `inbox_processing_end` | — | Inbox start/end (live event delivery) |

Read aliases: `dm_sent` → `message_sent`, `dm_received` → `message_received`.

**Live events**: `tool_use`, `inbox_processing_start`, `inbox_processing_end` are pushed over WebSocket via ProcessSupervisor for real-time UI.

### Priming integration

`ActivityLogger.format_for_priming()` formats entries within the token budget (default 1300 tokens; heartbeat guarantees at least 400).

**ASCII labels**: 2–4 character labels (`MSG<`, `DM>`, `HB`, …) replace emoji (~2–3 tokens each) for stable ~1-token recognition.

**Topic grouping**:

| Group type | Condition | Display |
|---|---|---|
| DM | Same peer, DMs within 30 min | `[HH:MM-HH:MM] DM {peer}: {topic}` + child lines |
| HB | Consecutive heartbeat_start/end | `[HH:MM-HH:MM] HB: {summary}` |
| CRON | Same `task_name` for `cron_executed` | `[HH:MM] CRON {task}: exit={code}` |
| single | Otherwise | `[HH:MM] {LABEL} {content}` |

**Pointers**: If truncated past 200 characters, append `(-> activity_log/{date}.jsonl)`. Groups get `-> activity_log/{date}.jsonl#L{range}` at the end so the LLM can `read_memory_file` for detail.

### Activity log rotation

Configure under `config.json` `activity_log`. `RotationMixin.rotate()` deletes old dated files to cap disk use.

| Parameter | Default | Description |
|---|---|---|
| `rotation_enabled` | true | Enable rotation |
| `rotation_mode` | `"size"` | `"size"` (total size cap), `"time"` (age), `"both"` |
| `max_size_mb` | 1024 | Max total size per Anima (MB) |
| `max_age_days` | 7 | Retention for `time`/`both` |
| `rotation_time` | `"05:00"` | Run time (JST) |

ProcessSupervisor’s scheduler runs `ActivityLogger.rotate_all()` for all Animas at `rotation_time`.

---

## Streaming journal

> Implementation: `core/memory/streaming_journal.py` — `StreamingJournal`

Write-ahead log: while the LLM streams, text chunks are flushed to disk incrementally. Hard crashes (SIGKILL, OOM, etc.) limit loss to about one second of text.

### Storage location

```
{anima_dir}/shortterm/streaming_journal_{session_type}.jsonl
```

Separate files per session type (`chat` / `heartbeat`) because chat and heartbeat use independent locks. With `thread_id`: `shortterm/{session_type}/{thread_id}/streaming_journal.jsonl`. Legacy: `streaming_journal.jsonl` (chat only; migration renames).

### WAL lifecycle

```
Normal:
  open() → write_text() / write_tool_*() → finalize() → delete journal

Crash:
  open() → write_text() / write_tool_*() → <crash> → journal remains
                                                        ↓
  Next boot: recover() → JournalRecovery → delete journal
```

- **open()**: If an orphan journal exists, recover first and persist to episodes; then create a new journal and write `start` (trigger, sender, session id)
- **write_text()**: Append to buffer; flush when thresholds met
- **write_tool_start() / write_tool_end()**: Tool boundaries
- **finalize()**: Write `done`, close, delete file
- **recover()**: Read orphan → return `JournalRecovery`

### Buffer settings

| Parameter | Value | Description |
|---|---|---|
| `_FLUSH_INTERVAL_SEC` | 1.0 s | Minimum flush interval |
| `_FLUSH_SIZE_CHARS` | 500 chars | Flush when buffer reaches this size |

Whichever condition fires first writes a `text` JSONL line and `fsync()`s.

### Recovery

`StreamingJournal.has_orphan(anima_dir, session_type)` checks for orphans; `recover(anima_dir, session_type, thread_id)` returns concatenated text, tool records (start/complete), session info (trigger, sender, start time), and completion flag (`done` present). Corrupt partial lines are skipped. After recovery, `_persist_recovery()` writes `episodes/recovered_{timestamp}.md` and removes the journal.

---

## Design principles

1. **Dual stores are required** — Keep episodic (raw) and semantic (distilled) memory
2. **Dual recall** — Automatic priming and intentional tool recall
3. **Memory infrastructure is the framework’s job** — Priming, RAG, forgetting, logging, etc. The scheduler kicks `run_consolidation`; the **background model** runs the Anima tool loop (**when** and post-processing = framework; read/write decisions = that session’s LLM)
4. **Consolidate daily** — Like nightly NREM: at minimum daily consolidation plus weekly integration
5. **Context is a first-class retrieval dimension** — Rich metadata at write; match current context at read
6. **Working memory limits are intentional** — Context caps are a feature; keep the most relevant material
7. **Active forgetting keeps the system healthy** — Prune low-activity memory to preserve retrieval S/N
8. **Procedural memory strengthens with use** — Confidence from success/failure feedback; repeated success increases forgetting resistance
9. **Detect and resolve contradictions** — NLI+LLM cascade; do not ignore conflicts

---

## `core/memory/` module reference

The memory subsystem is implemented under `core/memory/`.

### Priming (`priming/` package)

| Module | Role |
|---|---|
| `priming/engine.py` | `PrimingEngine`, `PrimingResult` — parallel fetch orchestration |
| `priming/budget.py` | `classify_message_type` / `adjust_token_budget` / `load_config_budgets` (`PrimingConfig` numbers; the `dynamic_budget` field itself is not read by the engine) |
| `priming/constants.py` | Per-channel default budgets, keyword constants |
| `priming/format.py` | `format_priming_section` for prompts |
| `priming/utils.py` | `RetrieverCache`, `build_queries`, `search_and_merge`, keywords, truncate |
| `priming/outbound.py` | Recent Outbound, pending `human_notify` |
| `priming/channel_a.py` … `channel_f.py` | Channels A–F (sender, activity, knowledge, skills, tasks, episodes) |

Public API: `from core.memory.priming import PrimingEngine, PrimingResult, format_priming_section` (re-exported from `core/memory/__init__.py`). Chat path calls `prime_memories` from `core/_agent_priming.py`.

### Conversation memory (split modules)

| Module | Role |
|---|---|
| `conversation.py` | `ConversationMemory` facade (compression, finalize, state updates) |
| `conversation_models.py` | `ConversationTurn`, `ConversationState`, etc. |
| `conversation_compression.py` | Rolling summary / compression |
| `conversation_finalize.py` | Session finalize, episode append, resolution propagation |
| `conversation_prompt.py` | LLM prompt fragments |
| `conversation_state_update.py` | Helpers for `current_state.md`, etc. |

### Other core

| Module | Class / role | Description |
|---|---|---|
| `manager.py` | `MemoryManager` | File memory, skill match, RAG facade |
| `shortterm.py` | `ShortTermMemory` | Session state under `shortterm/{session_type}/` |
| `activity.py` | `ActivityLogger` | Timeline core (split into `_activity_*.py`) |
| `_activity_models.py`, etc. | — | Models / priming formatting / API timeline / conversation view / rotation |
| `consolidation.py` | `ConsolidationEngine` | Consolidation pre/post (RAG rebuild, etc.) |
| `forgetting.py` | `ForgettingEngine` | `synaptic_downscaling` / `neurogenesis_reorganize` / `complete_forgetting` / `cleanup_procedure_archives` |
| `streaming_journal.py` | `StreamingJournal` | WAL for streaming |
| `task_queue.py` | `TaskQueueManager` | Persistent task queue JSONL |
| `distillation.py` | `ProceduralDistiller` | Procedural distillation (auxiliary path) |
| `reconsolidation.py` | `ReconsolidationEngine` | Reconsolidation, issue_resolved→procedure |
| `resolution_tracker.py` | `ResolutionTracker` | `shared/resolutions.jsonl` |
| `cron_logger.py` | `CronLogger` | `state/cron_logs/` |
| `skill_metadata.py` | Functions | Skill match normalization / keywords |
| `validation.py` | `KnowledgeValidator` | NLI+LLM validation |
| `contradiction.py` | `ContradictionDetector` | Contradiction utilities |
| `dedup.py` | — | Message dedup, heartbeat rate limits |
| `housekeeping.py` | `run_housekeeping()` | Daily cleanup of logs, shortterm, etc. |
| `frontmatter.py` | `FrontmatterService` | YAML frontmatter |
| `rag_search.py` | `RAGMemorySearch` | Search / indexer wrapper |
| `audit.py` | `AuditAggregator`, etc. | Supervisor activity + task rollup |
| `token_usage.py` | — | `token_usage/{date}.jsonl` usage and cost; pricing from in-module `DEFAULT_PRICING`, overridable via `~/.animaworks/pricing.json` |
| `config_reader.py` | — | Memory config helpers |
| `_io.py`, `_llm_utils.py` | — | Internal I/O and LLM helpers |

### RAG (`rag/`)

| Module | Role |
|---|---|
| `rag/indexer.py` | `MemoryIndexer` — chunking, embedding, incremental index |
| `rag/retriever.py` | `MemoryRetriever` — vector search, decay, importance, spread, `record_access` |
| `rag/graph.py` | `KnowledgeGraph` — graph build, PageRank, result expansion |
| `rag/store.py` | `ChromaVectorStore`, etc. — Chroma abstraction |
| `rag/http_store.py` | — | HTTP-backed store (when used) |
| `rag/singleton.py` | — | In-process singletons for store + embeddings |
| `rag/watcher.py` | `FileWatcher` | File change → reindex |

---

## Related documents

- [vision.md](vision.md) — Digital Anima philosophy
- [spec.md](spec.md) — Requirements (archive-based memory)
- [features.md](features.md) — Feature list (memory-related history)
- [specs/20260214_priming-layer_design.md](specs/20260214_priming-layer_design.md) — Priming layer plan (RAG, consolidation architecture)
- [specs/20260218_unified-activity-log-implemented-20260218.md](specs/20260218_unified-activity-log-implemented-20260218.md) — Unified activity log
- [specs/20260218_streaming-journal-implemented-20260218.md](specs/20260218_streaming-journal-implemented-20260218.md) — Streaming journal
- [specs/20260218_activity-log-spec-compliance-fixes-implemented-20260218.md](specs/20260218_activity-log-spec-compliance-fixes-implemented-20260218.md) — Activity log spec fixes
- [specs/20260218_priming-format-redesign_implemented-20260218.md](specs/20260218_priming-format-redesign_implemented-20260218.md) — Priming format (ASCII labels, grouping, pointers)
- [specs/20260218_episode-dedup-state-autoupdate-resolution-propagation.md](specs/20260218_episode-dedup-state-autoupdate-resolution-propagation.md) — Episode dedup, auto state, resolution propagation
- [specs/20260218_memory-system-enhancement-checklist-20260218.md](specs/20260218_memory-system-enhancement-checklist-20260218.md) — Memory enhancement checklist
- [specs/20260218_consolidation-validation-pipeline-20260218.md](specs/20260218_consolidation-validation-pipeline-20260218.md) — Daily consolidation validation
- [specs/20260218_knowledge-contradiction-detection-resolution-20260218.md](specs/20260218_knowledge-contradiction-detection-resolution-20260218.md) — Knowledge contradictions
- [specs/20260218_procedural-memory-foundation-20260218.md](specs/20260218_procedural-memory-foundation-20260218.md) — Procedural memory (YAML, 3-stage match)
- [specs/20260218_procedural-memory-auto-distillation-20260218.md](specs/20260218_procedural-memory-auto-distillation-20260218.md) — Auto distillation
- [specs/20260218_procedural-memory-reconsolidation-20260218.md](specs/20260218_procedural-memory-reconsolidation-20260218.md) — Prediction-error reconsolidation
- [specs/20260218_procedural-memory-utility-forgetting-20260218.md](specs/20260218_procedural-memory-utility-forgetting-20260218.md) — Utility-based forgetting
