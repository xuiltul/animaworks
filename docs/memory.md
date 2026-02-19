# AnimaWorks Memory System Design Specification

**[日本語版](memory.ja.md)**

> Created: 2026-02-14
> Updated: 2026-02-18
> Related: [vision.md](vision.md), [spec.md](spec.md), [implemented/20260214_priming-layer_design.md](implemented/20260214_priming-layer_design.md)
> Research: [AI Agent Memory Architecture Survey](research/20260212_AI_Agent記憶アーキテクチャ調査.md) Section 10

---

## Design Philosophy

The AnimaWorks memory system is designed based on **human brain memory mechanisms**.

The human brain has distinct memory systems -- "working memory," "episodic memory," "semantic memory," and "procedural memory" -- each processed in different brain regions. Memory recall operates through two pathways: "automatic recall (priming)" and "intentional recall." Memory consolidation follows a three-stage automatic process: "immediate encoding," "sleep-time consolidation," and "long-term integration."

AnimaWorks faithfully reproduces these mechanisms. The agent (LLM) is a "thinking person," not a "manager of its own brain." The framework handles memory infrastructure management, performing encoding and consolidation by invoking a separate LLM in the background via one-shot calls (independent of the agent's own LLM session).

---

## Mapping to the Human Memory Model

| Human Memory | Brain Region | AnimaWorks Implementation | Characteristics |
|---|---|---|---|
| **Working Memory** | Prefrontal cortex | LLM context window | Capacity-limited. Temporary holding of "what is currently being thought about." A spotlight on activated long-term memory |
| **Episodic Memory** | Hippocampus -> Neocortex | `episodes/` | "What happened when." Stored chronologically as daily logs. Automatically recorded by the framework at conversation end |
| **Semantic Memory** | Temporal lobe cortex | `knowledge/` | "What is known." Lessons, policies, and knowledge decontextualized from episodes. Extracted from episodes during daily consolidation |
| **Procedural Memory** | Basal ganglia, Cerebellum | `procedures/`, `skills/` | "How to do it." Work procedures, skills, workflows |
| **Interpersonal Memory** | Fusiform gyrus, Temporal pole | `shared/users/` | "Who is this person." User profiles shared across all Animas |

### Working Memory = Context Window

Based on Baddeley's (2000) working memory model.

- **Central executive** = Agent orchestrator. Oversees attention control and retrieval from long-term memory
- **Episodic buffer** = Context assembly layer. Integrates priming results and conversation history into a unified representation
- **Phonological loop** = Text buffer. Holds recent conversation turns

Following Cowan's (2005) findings, working memory is conceptualized as "activated long-term memory." The context window is not a separate store but rather the portion of long-term memory currently under attentional focus.

### Long-term Memory = File-based Archive

Memories are not truncated and injected into prompts but stored in a file-system archive (archive-based memory). There is no upper limit on memory volume. Only "what is needed now" enters the context.

```
~/.animaworks/animas/{name}/
├── activity_log/    Unified activity log (JSONL chronological record of all interactions)
├── episodes/        Episodic memory (daily logs, action records)
├── knowledge/       Semantic memory (learned knowledge, lessons, policies)
├── procedures/      Procedural memory (work procedures)
├── skills/          Skill memory (individual skills)
├── shortterm/       Short-term memory (session state, streaming journal)
└── state/           Persistent portion of working memory (current task, short-term state)
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│          Working Memory (Prefrontal Cortex)           │
│          = LLM Context Window                         │
│                                                       │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │Central       │  │Episodic    │  │Phonological   │  │
│  │Executive     │  │Buffer      │  │Loop           │  │
│  │=Orchestrator │  │=Context    │  │=Text          │  │
│  │              │  │ Assembly   │  │ Buffer        │  │
│  └──────┬──────┘  └─────┬──────┘  └──────────────┘  │
│         │               │                             │
│    Intentional      Automatic                         │
│    Search           Recall Results                    │
│    (search_memory)  (Priming)                         │
└─────────┬──────────────┬──────────────────────────────┘
          │              │
    ┌─────┴──────┐  ┌───┴──────────────────┐
    │ Prefrontal  │  │  Priming Layer       │
    │ Cortex      │  │  =Automatic Recall   │
    │ =Intentional│  │  Framework-automated │
    │  Search     │  │                      │
    │ Agent calls │  │                      │
    │ tool        │  │                      │
    └─────┬──────┘  └───┬──────────────────┘
          │              │
          │    ┌─────────┴────────────────┐
          │    │  Spreading Activation     │
          │    │  Vector similarity +      │
          │    │  temporal decay           │
          │    │  -> Auto-activation of    │
          │    │     related memories      │
          │    └─────────┬────────────────┘
          │              │
┌─────────┴──────────────┴───────────────────────────────┐
│                Long-term Memory (Hippocampus + Cortex)   │
│                                                          │
│  ┌───────────────────────────────────────────────┐      │
│  │  Unified Activity Log  activity_log/           │      │
│  │  = JSONL chronological record of all           │      │
│  │    interactions                                │      │
│  │  Source for Priming "Recent Activity" channel  │      │
│  └───────────────────────────────────────────────┘      │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐    │
│  │Episodic    │  │Semantic    │  │Procedural      │    │
│  │Memory      │  │Memory      │  │Memory          │    │
│  │episodes/   │  │knowledge/  │  │procedures/     │    │
│  │            │  │            │  │skills/         │    │
│  │Daily logs  │  │Learned     │  │Procedures &    │    │
│  │Action logs │  │knowledge   │  │skills          │    │
│  │            │  │Lessons &   │  │Workflows       │    │
│  │            │  │policies    │  │                │    │
│  └────────────┘  └────────────┘  └────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │  Shared Memory  shared/                         │     │
│  │  users/           Interpersonal memory          │     │
│  │                   (user profiles)               │     │
│  │  resolutions.jsonl Resolution registry          │     │
│  │                   (cross-organizational)        │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │  Streaming Journal  shortterm/                  │     │
│  │  = WAL (Write-Ahead Log). Crash-resilient      │     │
│  │  Incrementally persists text during streaming   │     │
│  │  output                                        │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  -- Memory Consolidation (Framework-automated) --        │
│                                                          │
│  [Immediate] Session boundary detection -> diff          │
│              summary -> episodes/                        │
│              + auto state update + resolution propagation │
│  [Daily]    Midnight cron -> episodes/ -> procedural     │
│              classification & distillation               │
│              -> NLI+LLM validation -> knowledge/ write   │
│              -> reconsolidation -> contradiction          │
│              detection & resolution                      │
│  [Weekly]   Weekly cron -> knowledge/ merge +            │
│              episodes/ compression                       │
│              + weekly pattern distillation + full         │
│              knowledge contradiction scan                │
│  [Monthly]  Monthly cron -> complete forgetting          │
│              (knowledge + episodes + procedures)         │
│              + archive cleanup                           │
│                                                          │
│  -- Forgetting (Synaptic Homeostasis) --                 │
│                                                          │
│  [Daily]    Synaptic Downscaling: knowledge(90d)        │
│             + procedures(180d or low utility)            │
│             -> low-activity mark                         │
│  [Weekly]   Neurogenesis Reorganization: LLM merge of   │
│             low-activity + similar chunks                │
│  [Monthly]  Complete Forgetting: low-activity 60d+ ->   │
│             archive & delete                            │
│             + archive/procedure_versions/ cleanup        │
│                                                          │
│  * Agent's only write path: intentional memorization     │
│    (write_memory_file)                                   │
└──────────────────────────────────────────────────────────┘
```

---

## Memory Recall: Two Pathways

Human memory recall is not a single process but consists of two stages: **automatic recall** and **intentional recall**. AnimaWorks implements both.

### Automatic Recall -- Priming Layer

**Neuroscience basis**: When perceptual stimuli are received, the auto-associative network in the hippocampal CA3 region automatically performs pattern completion. This is unconscious, fast (250-500ms), and unsuppressible.

**AnimaWorks implementation**: When a message is received, the framework automatically searches for related memories and injects them into the context before the agent starts. From the agent's perspective, relevant memories are "already recalled" when the conversation begins.

```
Message received -> Context extraction -> Priming search -> Context assembly -> Agent execution
                   (sender, keywords)   (4 channels       (within token      (memories already
                                         in parallel)       budget)            present)
```

Four search channels (`core/memory/priming.py`):

| Channel | Target | Budget | Method | Brain Analog |
|---|---|---|---|---|
| **A: Sender Profile** | shared/users/ | 500 tokens | Exact-match lookup | Automatic recall upon seeing a face |
| **B: Recent Activity** | activity_log/ | 1300 tokens | Chronological retrieval from ActivityLogger | Short-to-recent memory. "What happened recently" |
| **C: Related Knowledge** | knowledge/ | 700 tokens | Dense vector similarity search (RAG) | Associative recall via spreading activation |
| **D: Skill/Procedure Match** | skills/, procedures/, common_skills/ | 200 tokens | Description-based matching | List recall of "what I can do" and "how to do it" |

Channel B consolidates the legacy `episodes/` date-filtered retrieval and shared channel reads into a unified retrieval from the `ActivityLogger` unified activity log. When the activity log is empty, it falls back to the legacy format (episodes/ + channels/).

Channel D searches `procedures/` in addition to `skills/`. A 3-stage matching approach (bracket keywords, lexical matching, RAG vector search) enhances procedural memory recall accuracy (see the "Procedural Memory Lifecycle" section for details).

Dynamic budget allocation by message type:

| Message Type | Token Budget | Use Case |
|---|---|---|
| greeting | 500 | Greetings (short text, low load) |
| question | 1500 | Questions (moderate memory search) |
| request | 3000 | Requests/instructions (broad memory search) |
| heartbeat | 200 | Periodic patrol (minimal memory reference) |

### Intentional Recall -- search_memory Tool

**Neuroscience basis**: The prefrontal cortex (PFC) monitors the output of automatic recall and executes strategic searches when it is insufficient. This is conscious and slow.

**AnimaWorks implementation**: Only when the memories injected by priming are insufficient does the agent invoke `search_memory` / `read_memory_file` tools for additional search.

Typical cases requiring intentional recall:
- When specific dates, times, or numbers need to be answered precisely
- When details of a specific past interaction need to be confirmed
- When following a procedure document to perform a task
- When the context contains no relevant memories for an unknown topic

---

## Memory Search via Spreading Activation

**Neuroscience basis**: Collins & Loftus (1975) spreading activation theory. Semantic memory is organized as a network of concept nodes connected by associative links. When a node is activated, activation automatically propagates to adjacent nodes. Activating "doctor" pre-activates "nurse" and "hospital."

**AnimaWorks implementation**: Spreading activation is approximated through a combination of dense vector search and temporal decay.

The initial design planned to integrate BM25 (keyword) and vector search via RRF, but investigation revealed that multilingual dense vector search alone outperforms keyword search in accuracy, so the approach was **unified to vector similarity search only**.

| Search Signal | Method | Brain Analog |
|---|---|---|
| **Semantic vector** | Dense vector similarity search (`intfloat/multilingual-e5-small`, 384 dim, ChromaDB) | Finding conceptual neighbors. Approximation of spreading activation |
| **Temporal decay** | Exponential decay function (half-life 30 days, weight 0.2) | More recent memories are more easily activated (recency effect) |
| **Access frequency** | Logarithmic boost (`log(1 + access_count)`, weight 0.1) | Hebbian learning / LTP. Repeatedly accessed memories are more easily recalled |
| **Graph diffusion** | Knowledge graph + Personalized PageRank (optional) | Multi-hop associative propagation. Explicit links `[[]]` + implicit links (similarity >= 0.75) |

Final score calculation:

```
final_score = vector_similarity_score + (decay_factor × WEIGHT_RECENCY) + (WEIGHT_FREQUENCY × log(1 + access_count))

decay_factor = 0.5 ^ (age_days / 30.0)   # Halves every 30 days
WEIGHT_RECENCY = 0.2                       # Maximum contribution of temporal decay is 0.2
WEIGHT_FREQUENCY = 0.1                     # Access frequency contribution (Hebbian LTP equivalent)

# When graph diffusion is enabled:
final_score += pagerank_score × 0.5
```

---

## YAML Frontmatter

Files in `knowledge/` and `procedures/` are annotated with YAML frontmatter for structured metadata management. Frontmatter is automatically applied by the daily consolidation pipeline and retroactively applied to existing files via legacy migration.

### knowledge/ Frontmatter

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
| `created_at` | ISO8601 | Creation datetime |
| `updated_at` | ISO8601 | Last updated datetime |
| `source_episodes` | int | Number of source episodes extracted from |
| `confidence` | float | Confidence score (NLI+LLM validation result). 0.0-1.0 |
| `auto_consolidated` | bool | Whether generated by automatic consolidation |
| `version` | int | Version number (incremented on each reconsolidation) |
| `superseded_by` | str | New file that superseded this knowledge (on contradiction resolution) |
| `supersedes` | str | Old file that this knowledge superseded (on contradiction resolution) |

### procedures/ Frontmatter

```yaml
---
description: 手順の説明
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
| `confidence` | float | Confidence score. Calculated as success_count / max(1, success_count + failure_count) |
| `success_count` | int | Number of successes |
| `failure_count` | int | Number of failures |
| `version` | int | Version number (incremented on each reconsolidation) |
| `created_at` | ISO8601 | Creation datetime |
| `updated_at` | ISO8601 | Last updated datetime |
| `auto_distilled` | bool | Whether generated by automatic distillation |
| `protected` | bool | Manual protection flag (protection from forgetting) |

---

## Memory Consolidation: Three-Stage Automatic Process

The human brain performs memory consolidation as an unconscious automatic process. AnimaWorks similarly **performs all consolidation automatically on the framework side**. The framework invokes a separate LLM via one-shot calls in the background to execute conversation summarization (immediate encoding) and pattern extraction (daily consolidation). The agent itself is not involved in this process.

```
Waking (during conversation)              Sleeping (non-conversation)
────────────────────────                  ──────────────────────────

 Conversation -> Session boundary          Midnight cron
     │  detection                              │
     │  (10min idle or heartbeat)              │
     ▼                                         ▼
 [Immediate Encoding]                     [Daily Consolidation]
 Diff summary -> episodes/                episodes/ -> procedural
 + auto state update                       classification
 + resolution propagation                  -> procedural distillation
 Hippocampal 1-shot recording               -> procedures/
                                           -> LLM knowledge extraction
                                           -> NLI+LLM validation
                                           -> knowledge/ write
                                           -> RAG index update
                                           -> Synaptic Downscaling
                                           -> reconsolidation ->
                                              contradiction detection
                                              & resolution
                                           NREM sleep consolidation
                                                │
                                           Weekly cron
                                                │
                                                ▼
                                           [Weekly Integration]
                                           knowledge/ dedup & merge
                                           episodes/ compression
                                           Weekly pattern distillation
                                           Full knowledge contradiction
                                             scan
                                           Neocortical long-term
                                             integration
                                                │
                                           Monthly cron
                                                │
                                                ▼
                                           [Monthly Forgetting]
                                           Complete forgetting
                                             (knowledge + episodes
                                              + procedures)
                                           archive/procedure_versions/
                                             cleanup
```

### Daily Consolidation Pipeline (日次固定化)

> Implementation: `core/memory/consolidation.py` -- `daily_consolidate()`

```
daily_consolidate()
├── Legacy migration (auto-convert files without frontmatter)
├── Episode collection
├── Procedural content classification (classify episodes into knowledge/procedural portions)
├── Procedural auto-distillation (auto-extract procedures/ from procedural episodes)
├── LLM knowledge extraction (extract knowledge candidates from semantic episodes)
├── Code fence sanitization
├── NLI+LLM validation (hallucination elimination)
├── knowledge/ write (with YAML frontmatter)
├── RAG index update
├── Synaptic Downscaling (mark low-activity chunks, including procedures)
├── Prediction-error-based reconsolidation (contradiction detection & update between existing memories and new episodes)
└── Contradiction detection & resolution (supersede/merge/coexist decisions among new/updated knowledge)
```

### Weekly Integration Pipeline (週次統合)

- `knowledge/` deduplication and merge (archived to `archive/merged/`)
- `episodes/` compression
- Weekly pattern distillation (detect recurring patterns from `activity_log/` and formalize into `procedures/`)
- Full `knowledge/` contradiction scan (comprehensive detection of contradictions missed by daily runs)

### Monthly Forgetting Pipeline (月次忘却)

- Complete forgetting (`knowledge/` + `episodes/` + `procedures/`)
- `archive/procedure_versions/` cleanup (retain only the 5 most recent versions per procedure file)

### Consolidation Stages Summary

| Stage | Brain Process | AnimaWorks Implementation | Responsible | Frequency |
|---|---|---|---|---|
| **Immediate Encoding** | Hippocampal fast 1-shot encoding | Session boundary detection (10min idle or heartbeat) -> diff summary -> episodes/ auto-recording + auto state update + resolution propagation | Framework (bg LLM call) | At session boundary |
| **Daily Consolidation** (日次固定化) | NREM sleep slow-wave-spindle-ripple cascade | Midnight cron -> procedural classification & distillation -> LLM knowledge extraction -> NLI+LLM validation -> knowledge/ write -> reconsolidation -> contradiction detection & resolution | Framework (bg LLM call) | Every midnight |
| **Weekly Integration** (週次統合) | Neocortical long-term integration, synaptic downscaling | Weekly cron -> knowledge/ merge + episodes/ compression + weekly pattern distillation + full knowledge contradiction scan | Framework (bg LLM call) | Weekly |
| **Monthly Forgetting** (月次忘却) | Sub-threshold synapse elimination | Monthly cron -> complete forgetting (knowledge + episodes + procedures) + archive cleanup | Framework (bg cron) | Monthly |
| **Intentional Memorization** | Prefrontal cortex elaborative encoding | Direct write via write_memory_file | Agent | On demand |

The only write path remaining for the agent is **intentional memorization** (write_memory_file). This corresponds to a human consciously taking notes. All other encoding, consolidation, and integration is performed automatically by the framework.

### Immediate Encoding Details: Session-Boundary-Based Diff Summarization

The previous design re-summarized all turns on every message response, which caused the same conversation's summary to be recorded N-2 times redundantly. The current design tracks the recorded position via `last_finalized_turn_index` and **summarizes only unrecorded turns as a diff**.

**Session boundary**: `finalize_session()` is executed not at message response time but only under the following two conditions:
- **10-minute idle**: 10 minutes elapsed since the last turn (detected by `finalize_if_session_ended()`)
- **Heartbeat arrival**: `finalize_if_session_ended()` is called during periodic patrol

**Integration point**: `finalize_session()` executes the following in addition to diff summarization:
1. **Episode recording**: Append LLM summary of unrecorded turns to `episodes/`
2. **Auto state update**: Auto-parse "resolved items" and "new tasks" from LLM summary and append to `state/current_task.md`
3. **Resolution propagation**: Record resolved items to ActivityLogger (`issue_resolved` event) and `shared/resolutions.jsonl`
4. **Turn compression**: Integrate recorded turns into `compressed_summary` to prevent conversation.json bloat

### Resolution Propagation Mechanism

Resolution information propagates across 3 layers, reflecting changes in both the local Anima and other Animas:

| Layer | Target | Implementation | Propagation Destination |
|---|---|---|---|
| **Layer 1: ActivityLogger** | Local Anima | Record `issue_resolved` event to activity_log | Priming Channel B (local Anima's recent activity) |
| **Layer 2: Resolution Registry** | All Animas | Cross-organizational record in `shared/resolutions.jsonl` | "Resolved Issues" section in builder.py (all Animas' system prompts) |
| **Layer 3: Consolidation Injection** | Local Anima | Collect resolved events via `_collect_resolved_events()` | Injected into daily consolidation prompt (updates "unresolved" entries in knowledge/ to "resolved") |

---

## Knowledge Validation: NLI+LLM Cascade

> Implementation: `core/memory/validation.py` -- `KnowledgeValidator` class

Knowledge candidates extracted by LLM during daily consolidation may contain hallucinations (fabrication of information not present in the original episodes) if written directly. To eliminate this, **cascade verification using an NLI (Natural Language Inference) model and LLM** is performed.

### NLI Model

- Model: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
- Multilingual (including Japanese) zero-shot NLI
- Uses GPU when available, falls back to CPU otherwise
- When the NLI model is unavailable, validation is performed with LLM only (graceful degradation)

### Cascade Flow

```
Knowledge candidate (premise: source episode, hypothesis: extracted knowledge)
    │
    ▼
[NLI Judgment]
    ├── entailment >= 0.6  -> Approved with confidence=0.9 (LLM skipped)
    ├── contradiction >= 0.7 -> Rejected (LLM skipped)
    └── neutral / below threshold -> Proceed to LLM review
                                  │
                                  ▼
                             [LLM Judgment]
                                  ├── Approved -> Write with confidence=0.7
                                  └── Rejected -> Discard
```

When NLI produces a high-confidence judgment, the LLM call is skipped to optimize cost and latency. Only ambiguous cases where NLI returns neutral are escalated to LLM judgment.

---

## Knowledge Contradiction Detection and Resolution

> Implementation: `core/memory/contradiction.py` -- `ContradictionDetector` class

Contradictions can arise among knowledge files accumulated in `knowledge/` (e.g., "Person A is responsible for X" vs. "Person A is responsible for Y"). AnimaWorks detects contradictions using the NLI+LLM cascade and automatically resolves them with three resolution strategies.

### Contradiction Detection Flow

```
Newly created/updated knowledge file
    │
    ▼
[RAG Search] Retrieve similar knowledge
    │
    ▼
[NLI Judgment] Determine entailment/contradiction/neutral for each pair
    ├── entailment >= 0.7  -> No contradiction (LLM skipped, cost-optimized)
    ├── contradiction >= 0.7 -> Contradiction detected -> Proceed to LLM resolution
    └── neutral / below threshold -> Proceed to LLM detailed analysis
                                  │
                                  ▼
                             [LLM Analysis]
                                  ├── Contradiction found -> Determine resolution strategy
                                  └── No contradiction -> Skip
```

### Three Resolution Strategies

| Strategy | Condition | Processing |
|---|---|---|
| **supersede** (Replace) | New information clearly updates old information | Add `superseded_by` to old file and archive it; record `supersedes` in new file |
| **merge** (Integrate) | Both pieces of information can be integrated | LLM generates merged text and creates new file; archive both original files |
| **coexist** (Coexist) | Both are correct depending on context | Annotate both files with the existence and conditions of the contradiction |

### Execution Timing

| Timing | Target | Description |
|---|---|---|
| **Daily** | Files newly created/updated that day | Final step of the daily consolidation pipeline |
| **Weekly** | All `knowledge/` files | Comprehensive scan for contradictions missed by daily runs |

---

## Procedural Memory Lifecycle

> Implementation: `core/memory/distillation.py` -- `ProceduralDistiller`, `core/memory/reconsolidation.py` -- `ReconsolidationEngine`

Procedural memory (`procedures/`) holds "how to do it" knowledge, corresponding to the basal ganglia and cerebellum in the brain. While semantic memory (knowledge/) statically holds "what is known," procedural memory is dynamically reinforced and revised through repeated execution and result feedback.

### Auto-Distillation: Extracting Procedures from Episodes

**Daily distillation** (within `daily_consolidate()`):

1. **Procedural content classification**: Classify episodes as procedural or semantic using 12 regex patterns (e.g., "procedure," "setup method," "command," "workflow")
2. **LLM procedure extraction**: LLM generates structured procedure documents from episodes classified as procedural
3. **RAG deduplication check**: Skip new creation if an existing procedure with similarity >= 0.85 exists (prevents procedure proliferation)
4. **procedures/ write**: Saved with YAML frontmatter. `auto_distilled: true`, `confidence: 0.4`

**Weekly pattern distillation** (within the weekly integration pipeline):

1. Analyze one week's worth of activity from `activity_log/`
2. Detect recurring patterns (e.g., the same procedure executed multiple times)
3. LLM distills patterns into procedure documents and saves to `procedures/`

### 3-Stage Matching (Skill Injection)

For priming (Channel D) and skill injection in `builder.py`, matching against `procedures/` is performed for each message:

| Stage | Method | Description |
|---|---|---|
| **1. Bracket keyword** | `[keyword]` exact match | Matches when `[keyword]` in the message is contained in the frontmatter `description` |
| **2. Lexical match** | Content-word overlap scoring | Ranks by overlap of content words (nouns, verbs, etc.) between message and description |
| **3. RAG vector search** | Dense vector similarity | Semantic similarity search using sentence-transformers |

Stage 1 has highest priority; Stage 3 is the fallback. This enables procedure recall ranging from explicit keyword specification to fuzzy semantic search.

### Success/Failure Tracking

Procedural memory confidence is dynamically updated through execution result feedback:

| Tracking Method | Description |
|---|---|
| **report_procedure_outcome tool** | Agent explicitly reports success/failure via tool call |
| **Framework auto-tracking** | For procedures injected during a session, success/failure is automatically determined at session boundary |

Confidence calculation:

```
confidence = success_count / max(1, success_count + failure_count)
```

Initial values (at auto-distillation): `confidence: 0.4`, `success_count: 0`, `failure_count: 0`

### Prediction-Error-Based Reconsolidation

> Implementation: `core/memory/reconsolidation.py` -- `ReconsolidationEngine`

**Neuroscience basis**: Nader et al. (2000) reconsolidation theory. Recalled memories become destabilized and are reconsolidated after being integrated with new information. Prediction error (the gap between expectation and reality) serves as the trigger for reconsolidation.

**AnimaWorks implementation**: When new episodes contradict existing `knowledge/` or `procedures/`, NLI detects the contradiction and LLM performs analysis and updates.

```
New episode
    │
    ▼
[RAG Search] Retrieve related existing knowledge/procedures
    │
    ▼
[NLI Judgment] Detect contradiction between episode and existing memory
    ├── No contradiction -> Skip
    └── Contradiction found -> LLM analysis
                      │
                      ▼
                 [LLM Update Decision]
                      ├── Update needed -> Save old version to archive/versions/
                      │                    -> Update memory, version++
                      └── No update needed -> Skip
```

**Special handling during procedures/ reconsolidation**:
- Old version saved to `archive/procedure_versions/`
- `version` incremented
- Reset to `success_count: 0`, `failure_count: 0`, `confidence: 0.5` (re-verification needed)
- `updated_at` updated

---

## Active Forgetting: Synaptic Homeostasis

The human brain actively "forgets" as well as "remembers." AnimaWorks implements three stages of active forgetting based on the Synaptic Homeostasis Hypothesis (Tononi & Cirelli, 2003).

```
Waking (during conversation)                Sleeping (non-conversation)
────────────────────────                    ──────────────────────────

 Conversation/Search -> access_count++       Midnight cron
     │                                           │
     ▼                                           ▼
 [Access Recording]                         [Daily Downscaling]
 Frequently used memories are               knowledge: 90d+ unaccessed
 strengthened                               + low frequency
 (Hebbian learning / LTP)                   procedures: 180d+ unused
                                            + low frequency
                                            or utility<0.3 + failure>=3
                                            -> immediate mark
                                            (Synaptic Homeostasis)
                                                 │
                                            Weekly cron
                                                 │
                                                 ▼
                                            [Neurogenesis Reorganization]
                                            LLM merge of low-activity +
                                            similar memories
                                                 │
                                            Monthly cron
                                                 │
                                                 ▼
                                            [Complete Forgetting]
                                            Low-activity 60d+ unaccessed
                                            -> archive & delete
                                            knowledge + episodes +
                                            procedures
                                            archive/procedure_versions/
                                            cleanup
```

| Stage | Brain Process | AnimaWorks Implementation | Frequency |
|---|---|---|---|
| **Daily Downscaling** (日次ダウンスケーリング) | NREM sleep synaptic downscaling | knowledge: 90d+ unaccessed -> low-activity mark. procedures: 180d+ unused or utility<0.3 + failure>=3 -> low-activity mark | Daily cron |
| **Neurogenesis Reorganization** (神経新生的再編) | Memory circuit reorganization via neurogenesis in the hippocampal dentate gyrus | LLM merges similar pairs of low-activity chunks | Weekly cron |
| **Complete Forgetting** (完全忘却) | Sub-threshold synapse elimination | Delete vector index for low-activity 60d+ unaccessed entries; archive source files (knowledge + episodes + procedures) | Monthly cron |

### knowledge/ Forgetting Thresholds

| Condition | Value | Description |
|---|---|---|
| Unaccessed period | 90 days | 90 days since last access |
| Access count | < 3 | Low usage frequency |

### procedures/ Forgetting Thresholds

procedures/ has more lenient thresholds than knowledge/ (procedural memory is more forgetting-resistant in the brain as well):

| Condition | Value | Description |
|---|---|---|
| Unused period | 180 days | 180 days since last use (2x that of knowledge) |
| Usage count | < 3 | Low usage frequency |
| Immediate mark condition | utility < 0.3 AND failure_count >= 3 | Repeatedly failed low-utility procedures are immediately marked as low-activity |

### Forgetting Protection

| Target | Protection Condition | Reason |
|---|---|---|
| `skills/` | Always protected | Origin point for description-based matching. Deleting them severs the recall pathway |
| `shared/users/` | Always protected | Interpersonal memory protection |
| `[IMPORTANT]` tagged | Always protected | Forgetting resistance through elaborative encoding |
| `procedures/` (version >= 3) | Conditionally protected | Mature procedures that have undergone 3+ reconsolidations |
| `procedures/` (protected: true) | Conditionally protected | Manual protection via frontmatter |
| `procedures/` ([IMPORTANT]) | Conditionally protected | Forgetting resistance via tag |

### Monthly Archive Cleanup

The monthly forgetting pipeline organizes old versions accumulated in `archive/procedure_versions/`. Only the 5 most recent versions per procedure file are retained; older versions are deleted.

---

## Unified Activity Log

> Implementation: `core/memory/activity.py` -- `ActivityLogger` class

A unified logging infrastructure that records all interactions in a single JSONL chronological timeline. This consolidates records that were previously scattered across transcript, dm_log, heartbeat_history, etc. into a single source, serving as the sole data source for the Priming Layer's "Recent Activity" channel (Channel B).

### Storage Location

```
{anima_dir}/activity_log/{date}.jsonl
```

One file per date. Written append-only; each line is a single JSON entry.

### JSONL Format

```json
{"ts":"2026-02-17T14:30:00","type":"message_received","content":"...","from":"user","channel":"chat"}
{"ts":"2026-02-17T14:30:05","type":"response_sent","content":"...","to":"user","channel":"chat"}
{"ts":"2026-02-17T15:00:00","type":"tool_use","tool":"web_search","summary":"検索実行"}
```

Empty fields are omitted. `from`/`to` are sender/recipient names, `channel` is the channel name, `tool` is the tool name, and `via` is the notification channel (for human_notify events).

### Event Type Reference

| Event Type | ASCII Label | Description |
|---|---|---|
| `message_received` | `MSG<` | Message received from user |
| `response_sent` | `MSG>` | Response sent by Anima |
| `dm_sent` | `DM>` | Direct message sent to another Anima |
| `dm_received` | `DM<` | Direct message received from another Anima |
| `channel_post` | `CH.W` | Post to shared channel |
| `channel_read` | `CH.R` | Read from shared channel |
| `human_notify` | `NTFY` | Notification to human (via call_human) |
| `tool_use` | `TOOL` | External tool usage |
| `heartbeat_start` | `HB` | Heartbeat start |
| `heartbeat_end` | `HB` | Heartbeat end |
| `cron_executed` | `CRON` | Cron task execution |
| `memory_write` | `MEM` | Write to memory file |
| `error` | `ERR` | Error occurred |
| `issue_resolved` | `RSLV` | Issue resolved (auto-recorded from auto state update) |

### Priming Integration

The `ActivityLogger.format_for_priming()` method formats retrieved entries within the token budget (default 1300 tokens; minimum 400 tokens guaranteed during heartbeat).

**ASCII labeling**: Each event type is displayed with a 2-4 character ASCII label (`MSG<`, `DM>`, `HB`, etc.). The legacy emoji icons (`📨`, `💓`, etc.) consumed 2-3 tokens each, whereas ASCII labels are stably recognized at 1 token.

**Topic grouping**: Related entries are grouped for compact display.

| Group Type | Condition | Display Format |
|---|---|---|
| DM | Same peer, consecutive DMs within 30 min | `[HH:MM-HH:MM] DM {peer}: {topic}` + child lines |
| HB | Consecutive heartbeat_start/end | `[HH:MM-HH:MM] HB: {summary}` |
| CRON | cron_executed with same task_name | `[HH:MM] CRON {task}: exit={code}` |
| single | All others | `[HH:MM] {LABEL} {content}` |

**Pointer references**: When content exceeds 200 characters and is truncated, a source file pointer `(-> activity_log/{date}.jsonl)` is appended. Groups include `-> activity_log/{date}.jsonl#L{range}` at the group end. This allows the LLM to reference the original data via `read_memory_file` when more detail is needed.

---

## Streaming Journal

> Implementation: `core/memory/streaming_journal.py` -- `StreamingJournal` class

A Write-Ahead Log (WAL) that incrementally writes text chunks to disk during LLM streaming response output. Even if a hard process crash occurs (SIGKILL, OOM, etc.), text loss is limited to approximately the last 1 second.

### Storage Location

```
{anima_dir}/shortterm/streaming_journal.jsonl
```

### WAL Lifecycle

```
Normal flow:
  open() -> write_text() / write_tool_*() -> finalize() -> journal file deleted

Abnormal flow (crash):
  open() -> write_text() / write_tool_*() -> <crash> -> journal file remains
                                                         ↓
  Next startup: recover() -> restored as JournalRecovery -> journal file deleted
```

- **open()**: Creates a new journal file and writes the `start` event (trigger, sender, session ID)
- **write_text()**: Appends text fragments to the buffer. Flushes when buffer conditions are met
- **write_tool_start() / write_tool_end()**: Records tool execution start/end
- **finalize()**: Writes the `done` event, closes the file, and deletes it (normal completion)
- **recover()**: Reads orphaned journals and returns a `JournalRecovery` dataclass

### Buffer Configuration

| Parameter | Value | Description |
|---|---|---|
| `_FLUSH_INTERVAL_SEC` | 1.0 sec | Minimum flush interval |
| `_FLUSH_SIZE_CHARS` | 500 chars | Flush when buffer reaches this size |

When either condition is met, the buffer contents are written as a `text` event in a JSONL line and `fsync()`'d.

### Recovery

On next startup, `StreamingJournal.has_orphan()` checks for orphaned journals, and `recover()` restores the following information:

- Recovered text (concatenation of all `text` events)
- Tool call records (with start/completion status)
- Session information (trigger, sender, start time)
- Completion flag (presence/absence of the `done` event)

Broken JSONL lines (partial writes at crash time) are skipped.

---

## Design Principles

1. **Dual stores are essential** -- Both episodic memory (raw records) and semantic memory (distilled knowledge) must be maintained
2. **Recall uses dual pathways** -- Implement both automatic recall (priming) and intentional recall (tool invocation)
3. **Memory infrastructure is the framework's responsibility** -- Encoding, consolidation, and integration are performed automatically by the framework via background LLM one-shot calls. The agent (Anima's primary LLM) does not manage memory infrastructure
4. **Consolidation runs daily** -- The brain's NREM sleep occurs every night. Two-stage daily consolidation + weekly integration is the minimum requirement
5. **Context is a first-class search dimension** -- Rich metadata is attached at memory storage time; during retrieval, priority is given based on relevance to the current context
6. **Working memory capacity limitation is a design feature** -- The context window limit is not a bug but a feature. It selectively retains only the most relevant information
7. **Active forgetting maintains system health** -- Memory does not only accumulate; actively pruning low-activity memories maintains search accuracy (signal-to-noise ratio)
8. **Procedural memory is strengthened through use** -- Procedure confidence is dynamically updated via success/failure feedback. Procedures that have repeatedly succeeded gain greater forgetting resistance
9. **Contradictions are detected and resolved** -- Knowledge contradictions are not left unaddressed; they are automatically detected and resolved via the NLI+LLM cascade

---

## Related Documents

- [vision.md](vision.md) -- Foundational philosophy of Digital Anima
- [spec.md](spec.md) -- Requirements specification (basic design of archive-based memory)
- [features.md](features.md) -- Feature list (includes memory system implementation history)
- [implemented/20260214_priming-layer_design.md](implemented/20260214_priming-layer_design.md) -- Priming layer implementation plan (includes RAG design and consolidation architecture)
- [implemented/20260218_unified-activity-log-implemented-20260218.md](implemented/20260218_unified-activity-log-implemented-20260218.md) -- Unified activity log design document
- [implemented/20260218_streaming-journal-implemented-20260218.md](implemented/20260218_streaming-journal-implemented-20260218.md) -- Streaming journal design document
- [implemented/20260218_activity-log-spec-compliance-fixes-implemented-20260218.md](implemented/20260218_activity-log-spec-compliance-fixes-implemented-20260218.md) -- Activity log spec compliance fixes
- [implemented/20260218_priming-format-redesign_implemented-20260218.md](implemented/20260218_priming-format-redesign_implemented-20260218.md) -- Priming format redesign (ASCII labeling, topic grouping, pointer references)
- [implemented/20260218_episode-dedup-state-autoupdate-resolution-propagation.md](implemented/20260218_episode-dedup-state-autoupdate-resolution-propagation.md) -- Episode deduplication, auto state update, and resolution propagation mechanism
- [implemented/20260218_memory-system-enhancement-checklist-20260218.md](implemented/20260218_memory-system-enhancement-checklist-20260218.md) -- Memory system enhancement checklist
- [implemented/20260218_consolidation-validation-pipeline-20260218.md](implemented/20260218_consolidation-validation-pipeline-20260218.md) -- Daily consolidation validation pipeline
- [implemented/20260218_knowledge-contradiction-detection-resolution-20260218.md](implemented/20260218_knowledge-contradiction-detection-resolution-20260218.md) -- Knowledge contradiction detection and resolution
- [implemented/20260218_procedural-memory-foundation-20260218.md](implemented/20260218_procedural-memory-foundation-20260218.md) -- Procedural memory foundation (YAML frontmatter, 3-stage matching)
- [implemented/20260218_procedural-memory-auto-distillation-20260218.md](implemented/20260218_procedural-memory-auto-distillation-20260218.md) -- Procedural memory auto-distillation
- [implemented/20260218_procedural-memory-reconsolidation-20260218.md](implemented/20260218_procedural-memory-reconsolidation-20260218.md) -- Prediction-error-based reconsolidation
- [implemented/20260218_procedural-memory-utility-forgetting-20260218.md](implemented/20260218_procedural-memory-utility-forgetting-20260218.md) -- Procedural memory utility-based forgetting
- [AI Agent Memory Architecture Survey](research/20260212_AI_Agent記憶アーキテクチャ調査.md) -- Neuroscience survey and prior research review
