# AnimaWorks Brain Mapping — Architecture Mapped to the Human Brain

**[日本語版](brain-mapping.ja.md)**

> Created: 2026-02-19 | Updated: 2026-03-25
> Related: [vision.md](vision.md), [memory.md](memory.md)

---

## Background

The designer of AnimaWorks is a psychiatrist with over 30 years of programming experience. AnimaWorks' memory system, autonomic mechanisms, and execution architecture are **intentionally** mapped to the structure of the human brain, grounded in clinical neuroscience. This is not merely a metaphor — it is an attempt to reuse the brain's information-processing architecture as a design pattern.

In psychiatric practice, one routinely observes dysfunctions of the brain's various subsystems: memory disorders, attention disorders, executive function disorders, and more. Knowing what happens when each subsystem is impaired made it possible to identify the subsystems an AI agent requires and to design a clear separation of their respective roles.

---

## Overall Mapping

### Neocortex — LLM Model

| LLM Function | Brain Region | Description |
|---|---|---|
| Reasoning & decision-making | Prefrontal cortex (PFC) | Executive function. Receives memories injected by priming and makes judgments |
| Language comprehension | Wernicke's area (temporal lobe) | Semantic understanding of input messages |
| Language production | Broca's area (frontal lobe) | Generation of response text |
| Pre-trained knowledge | Crystallized patterns in temporal cortex | World knowledge baked into LLM weights. A separate system from file-based memory — "innate intelligence" |
| Transformer attention | Parietal association cortex + PFC selective attention | Allocation of attention to relevant information within the context |

The LLM in its entirety corresponds to the **neocortex** as a whole. However, in AnimaWorks' design, because the framework handles subcortical functions (memory consolidation, forgetting, arousal maintenance), the role left to the LLM is effectively distilled into the **conscious processing of the prefrontal cortex (PFC)**.

As memory.md states:

> The agent (LLM) is "the one who thinks," not "the administrator of its own brain."

### Duality of Pre-trained Knowledge and File-based Memory

The knowledge baked into the LLM's pre-trained weights and AnimaWorks' file-based memory are **separate systems**. In the human brain as well, patterns crystallized in the cerebral cortex (implicit knowledge / crystallized intelligence) and episodic memory via the hippocampus function as independent systems.

| Type of Knowledge | Human Brain | AnimaWorks |
|---|---|---|
| Innate intelligence | Crystallized intelligence (cortical patterns) | LLM pre-trained weights |
| Experientially acquired knowledge | Fluid intelligence + episodic memory | File-based memory (episodes/, knowledge/, procedures/) |

This distinction aligns with the "imperfect individual" design philosophy described in vision.md. Precisely because pre-trained knowledge alone is insufficient, an experience-based memory system is necessary.

---

### Memory System — Hippocampus, Cerebral Cortex, & Basal Ganglia

| Human Memory | Brain Region | AnimaWorks Implementation | Characteristics |
|---|---|---|---|
| **Working memory** | Prefrontal cortex | LLM context window | Capacity-limited. Temporary holding of "what is currently being thought about" |
| **Episodic memory** | Hippocampus → neocortex | `episodes/` | Chronological record of "when and what happened" |
| **Semantic memory** | Temporal cortex | `knowledge/` | Lessons and knowledge decoupled from context |
| **Procedural memory** | Basal ganglia, cerebellum | `procedures/`, `skills/` | "How to do it." Strengthened through repetition |
| **Person memory** | Fusiform gyrus, temporal pole | `shared/users/` | Automatic recall of "who is this person" |

### Internal Structure of Working Memory — Baddeley's Model

Based on Baddeley (2000):

| Baddeley's Component | Function | AnimaWorks Implementation |
|---|---|---|
| **Central executive** | Attentional control; orchestration of retrieval from long-term memory | Agent orchestrator |
| **Episodic buffer** | Integration of multiple sources into a unified representation | Context assembly layer (priming results + conversation history) |
| **Phonological loop** | Temporary holding of verbal information | Text buffer (recent conversation turns) |

Following Cowan (2005), working memory is understood as a "spotlight on activated long-term memory." The context window is not an independent store, but rather the portion of long-term memory that currently has attention directed toward it.

---

### Memory Recall — Dual Pathways

| Recall Pathway | Brain Process | AnimaWorks Implementation |
|---|---|---|
| **Automatic recall** | Pattern completion by the CA3 auto-associative network of the hippocampus. Unconscious, fast (250–500 ms), unsuppressible | Priming layer (6-channel parallel search) |
| **Deliberate recall** | Strategic search by the prefrontal cortex (PFC). Conscious, slow | `search_memory` / `read_memory_file` tools |

### Spreading Activation — Collins & Loftus (1975)

| Search Signal | Brain Counterpart | AnimaWorks Implementation |
|---|---|---|
| Semantic neighborhood discovery | Spreading activation among concept nodes | Dense vector similarity search (ChromaDB) |
| Prioritization of recent memories | Recency effect | Time-decay function (half-life: 30 days) |
| Strengthening of frequently used memories | Hebb's rule / long-term potentiation (LTP) | Access frequency boost |
| Multi-hop association | Propagation through associative networks | Knowledge graph + Personalized PageRank (implicit-link vector similarity threshold 0.75; procedural distillation RAG duplicate detection at 0.85 is a separate pathway) |

### Priming Channels and Dynamic Budget — Selective Attention

`PrimingEngine` (`core/memory/priming/engine.py`) pulls from multiple sources **in parallel** via `asyncio.gather`. Conceptually this still maps to the **six channels A–F**, but the implementation also includes:

- **C0 ([IMPORTANT] knowledge)** (`channel_c.py`): RAG `get_important_chunks` always fetches `[IMPORTANT]` chunks, placing heading summaries and pointers to `read_memory_file(path=...)` at the top (dedicated slot ~500 tokens). Vector search for related knowledge follows.
- **Recent outbound** (`priming/outbound.py`): Up to 3 `channel_post` / `message_sent` events from the last 2 hours.
- **Pending human notifications** (`collect_pending_human_notifications` in `priming/outbound.py`): `human_notify` from the last 24 hours, formatted to ~500 tokens. Log reads occur when `channel` is `chat` or `heartbeat`, or when `channel.startswith("message:")` (Anima-to-Anima DM priming). In **`build_system_prompt`** (`core/prompt/builder.py`) **Group 3**, `pending_human_notifications` is included **only when `is_chat or is_heartbeat`** (`is_heartbeat` is **strict equality** `trigger == "heartbeat"`). For `inbox:` / `cron:` / `task:` / `consolidation:` etc., `is_chat` is false, so they **do not appear** in the prompt. Normal `message:peer` DMs have `is_chat` true, so they **do appear**. Note: `consolidation:` priming may use `channel="heartbeat"`, so **collection may run**, but it can **fail to inject** when it does not match the `is_heartbeat` check above. For `cron`, the collection side is also empty.

When dynamic budget is enabled, each channel's character budget scales uniformly with `budget_ratio = token_budget / 2000` against the overall cap `token_budget`. Related knowledge merges `related_knowledge` (trusted) and `related_knowledge_untrusted` before truncation, preserving trust separation. Defaults can be overridden via `priming.budget_*` in `config.json`.

| Channel | Function | Brain Counterpart | Base token budget (`constants.py`, before scaling) |
|---|---|---|---|
| A: Sender profile | "Who is talking to me?" | Fusiform face area / temporal pole (person recognition) | 500 |
| B: Recent activity | "What happened recently?" | Hippocampal replay (recent episode reactivation) | 1300 |
| C: Related knowledge | "What do I know about this?" (IMPORTANT prefix + RAG) | Semantic memory retrieval (temporal cortex) | IMPORTANT 500 + related search 1000 |
| D: Skill matching | "Can I handle this?" | Procedural memory activation (basal ganglia) | 200 |
| E: Pending tasks | "What am I supposed to be doing?" | Prospective memory / intention monitoring (rostral PFC) | 500 |
| F: Episodes | "Have I had similar experiences?" | Episodic semantic search (hippocampus–cortex) | 800 |

Channel D returns **names only** for matched skills, common skills, and procedures (`channel_d.py` `channel_d_skill_match`, three-stage matching in `core.memory.manager.match_skills_by_description`, max `_MAX_SKILL_MATCHES` = 5). Full text is fetched on demand via the `skill` tool. Channel B reads `activity_log/` and `shared/channels/`. Noise filtering branches on the set in `channel_b.py`. **Heartbeat** (`channel == "heartbeat"`) excludes `tool_use`, `tool_result`, `heartbeat_start`, `heartbeat_reflection`, `inbox_processing_start`, and `inbox_processing_end`. Docstrings also treat **`channel.startswith("cron:")`** as background, but **`PrimingMixin`** (`core/_agent_priming.py`) passes the fixed string `prime_memories(..., channel="cron")` even during cron execution, so it **does not match** `startswith("cron:")`. As a result, **cron priming applies chat-side `_CHAT_NOISE_TYPES`**, and `memory_write`, `cron_executed`, `heartbeat_end`, etc. are also excluded (if `channel` were passed as `cron:{task}`, `channel_b` could switch to a heartbeat-like filter).

The builder does not read ActivityLogger directly; the priming path (Channel B, recent outbound, pending notifications) is the main activity reader for prompts (hippocampus model).

Channel E (`channel_e.py`, pending tasks plus surrounding state) maps to **prospective memory**. The rostral prefrontal cortex (Brodmann area 10) maintains future intentions at low activation until the right context triggers them. AnimaWorks' task queue surfaces unfinished tasks to the agent's awareness through an analogous mechanism. Beyond `TaskQueueManager.format_for_priming`, the same channel concatenates **active parallel tasks** (`PrimingEngine._get_active_parallel_tasks`, `submit_tasks` DAG), summaries from **`state/overflow_inbox/`**, and excerpts of completed background tasks in **`state/task_results/`** (queue I/O is offloaded with `asyncio.to_thread` for non-blocking behavior).

#### Dynamic Budget Allocation — Attentional Resource Management

When `priming.dynamic_budget = true`, the priming token budget is adjusted dynamically by message type — **selective attention** at the system level:

| Message Type | Budget (default, `PrimingConfig`) | Brain Analogy |
|---|---|---|
| Greeting | 500 | Low attentional load (routine social interaction) |
| Question | 2000 | Moderate-to-high attentional load (retrieval-oriented) |
| Request | 3000 | High attentional load (task-oriented, maximal resource allocation) |
| Heartbeat | max(200, context_window * 5%) | Tonic alertness (minimum arousal maintenance) |

The heartbeat formula `max(budget_heartbeat, int(context_window * heartbeat_context_pct))` gives larger-context models more priming data during autonomous patrol — analogous to tonic firing of the reticular activating system scaling with overall cortical capacity.

`classify_message_type` (`priming/budget.py`) treats the type as **`"heartbeat"` only when `channel == "heartbeat"`**. **Cron priming** uses `channel="cron"`, so it does not qualify; greeting / question / request heuristics apply and one of those **greeting / question / request budgets** is used (the heartbeat percentage scale does **not** apply).

This mirrors Kahneman's (1973) attention-as-resource theory: more cognitive resources for demanding tasks, fewer for routine stimuli, optimizing signal-to-noise within the limited context window.

#### Tiered Prompt and Trigger-Based Filtering

Depending on context window size, `build_system_prompt()` adjusts injected sections across four tiers (T1–T4). At 128k+ all sections; 32k–128k reduced; 16k–32k omits bootstrap/vision/specialty/DK/memory_guide; below 16k also omits permissions/org/messaging/emotion. This implements **selective inclusion under attentional limits**.

Section selection also depends on trigger (`chat` / `inbox` / `heartbeat` / `cron` / `task`, etc.). Heartbeat and cron omit specialty, emotion, and a_reflection; task uses minimal context (identity 3 lines + task description). Controlling what enters "consciousness" per path optimizes cognitive load. Apart from priming body text, conditions for injecting unprocessed `human_notify` summaries as `pending_human_notifications` into Group 3 are the **same as `is_chat or is_heartbeat` above** (see bullets in this section).

#### Unified Activity Log

As the primary data source for Channel B, `ActivityLogger` (`core/memory/activity.py`) appends chronologically to `{anima_dir}/activity_log/{date}.jsonl`. The class is a **facade**; timeline, conversation view, priming formatting, and log rotation are split across mixins such as `_activity_timeline`, `_activity_conversation`, `_activity_priming`, and `_activity_rotation`. Representative types: `message_received` / `message_sent` (resolved with `dm_*` aliases), `response_sent`, `channel_read` / `channel_post`, `human_notify`, `tool_use` / `tool_result`, `heartbeat_start` / `heartbeat_end` / `heartbeat_reflection`, `inbox_processing_start` / `inbox_processing_end`, `cron_executed`, `memory_write`, `error`, `issue_resolved`, `task_created` / `task_updated`, and more. Some tool events are pushed over WebSocket for live UI updates (`tool_use` matching `_LIVE_EVENT_TYPES` / `_VISIBLE_TOOL_NAMES`). Crash resilience for streaming output uses the Write-Ahead Log at `shortterm/streaming_journal_{session_type}.jsonl` (chat vs heartbeat) via `core/memory/streaming_journal.py`. Daily consolidated log cleanup and retention-based deletion are handled by **`core/memory/housekeeping.py`** through the supervisor (prompt logs, short-term storage, `task_results`, etc.).

---

### Memory Consolidation — Sleep and Integration

| AnimaWorks | Brain Process | Description |
|---|---|---|
| **Immediate encoding** (session boundary) | Hippocampal rapid one-shot encoding | At conversation end, a differential summary is recorded in episodes/ |
| **Daily consolidation** (midnight cron) | NREM slow-wave — spindle — ripple cascade | Substantive summarization and extraction are executed by the Anima's tool loop. `ConsolidationEngine` (`core/memory/consolidation.py`) is a module focused on **pre-processing** (episode collection, `issue_resolved` collection) and **post-processing** (RAG index update/rebuild, monthly forgetting invocation, legacy knowledge migration, etc.) |
| **issue_resolved → procedure** | Proceduralization of resolutions | Scans activity_log for `issue_resolved` events; ProceduralDistiller generates procedures (`create_procedures_from_resolved`) |
| **Weekly integration** | Neocortical long-term consolidation | Deduplication and merging of knowledge/, pattern distillation |
| **NLI + LLM validation** | Hippocampal pattern separation | Hallucination control. Consistency checks between episodes and extracted knowledge |
| **Prediction-error-based reconsolidation** (`reconsolidation.py`) | Reconsolidation theory, Nader et al. (2000) | LLM revision of procedures whose failure count exceeds threshold. Versioning and archival |

---

### Forgetting — Synaptic Homeostasis

Based on the synaptic homeostasis hypothesis of Tononi & Cirelli (2003):

| AnimaWorks | Brain Process | Description |
|---|---|---|
| **Daily downscaling** | Synaptic downscaling during NREM sleep | Marking low-activity chunks |
| **Neurogenesis-inspired reorganization** | Memory circuit reorganization via dentate neurogenesis | LLM merge of low-activity + similar chunks |
| **Complete forgetting** (monthly) | Elimination of sub-threshold synapses | Knowledge-like chunks: archive → delete when low activation exceeds **90 days** and `access_count` is below threshold (`FORGETTING_LOW_ACTIVATION_DAYS` / `FORGETTING_MAX_ACCESS_COUNT`). **Procedure** downscaling uses separate thresholds (e.g. **180 days** unused with low use counts) via `PROCEDURE_INACTIVITY_DAYS` etc. Procedure archives keep only **`PROCEDURE_ARCHIVE_KEEP_VERSIONS`** (5 versions) |
| **Forgetting resistance** (procedures, skills, knowledge) | Basal ganglia procedural memory resists forgetting | procedures: `_is_protected_procedure` when `importance == important` / `protected: true` / `version >= 3`. Low utility and high failure feed `_should_downscale_procedure` (`PROCEDURE_*` constants). knowledge: `_is_protected_knowledge` when `importance == important` or `success_count >= 2`. `IMPORTANT_SAFETY_NET_DAYS` (**365 days**) is used in `_is_protected` as the **first-tier** lift for `importance == important`, but **knowledge / procedures still apply type-specific checks that continue to protect `[IMPORTANT]`**. skills / `shared_users` are fully skipped as `PROTECTED_MEMORY_TYPES` |

### Procedural Distillation and Metaplasticity

Beyond the three-stage forgetting cycle, AnimaWorks adds memory subsystems aligned with finer aspects of neural plasticity:

| AnimaWorks | Brain Process | Description |
|---|---|---|
| **Procedural distillation** (`distillation.py`) | Skill consolidation in basal ganglia–cerebellar circuits | LLM sorts episodic memory into knowledge vs procedures. Detects repeated action patterns from activity logs and distills reusable procedure files — analogous to motor sequences automating through basal ganglia loops |
| **Weekly pattern detection** | Metaplasticity (Abraham & Bear, 1996) | Activity-log clustering over 7-day windows finds recurrent behavior. Expresses "learning how to learn" — adapting memory formation, not just content |
| **RAG duplicate detection** (similarity ≥ 0.85) | Hippocampal pattern separation | `RAG_DUPLICATE_THRESHOLD = 0.85` in `distillation.py`. Vector similarity before saving new procedures avoids redundant encoding — like dentate orthogonalization of similar memories |
| **Resolution tracking** (`resolution_tracker.py`) | Organizational long-term memory (transactive memory) | Cross-Anima shared resolution log in `shared/resolutions.jsonl`. Organizational knowledge of "who resolved what" — Wegner (1987) |
| **Persistent task queue** (`task_queue.py`) | Prospective memory / working-memory extension | Append-only JSONL with deadlines and stale detection. Extends working memory past the context window, like an external notepad for the central executive |

The procedural distillation pipeline runs on two timescales:

- **Daily**: LLM classifies episode sections into knowledge / procedures / skip; writes structured procedure files with YAML front matter (confidence scores, success/failure counts)
- **Weekly**: Vector clustering on activity entries detects repeated patterns and distills generalized procedures

This dual-timescale design mirrors skill-acquisition neuroscience: explicit daily classification shifts toward implicit procedural knowledge via weekly pattern distillation — the hippocampus-to-basal-ganglia transition Doyon & Benali (2005) describe.

---

### Arousal & Autonomic Mechanisms

| AnimaWorks | Brain Region | Description |
|---|---|---|
| **Heartbeat** (periodic patrol) | **Reticular activating system (ARAS)** | Maintains arousal. Does not fix the content of consciousness; provides its preconditions. Fires rhythmically — without it, dormancy (coma) |
| **Cron** (scheduled tasks) | Hypothalamic circadian rhythm (SCN) | Time-based periodic triggers. Sleep–wake and daily/weekly/monthly biorhythms |
| **ProcessSupervisor** | Autonomic nervous system | Process life cycle outside awareness: start, monitor, restart each Anima |
| **Unix domain socket IPC** | White-matter tracts | Physical pathways between Anima processes |
| **Messenger** | Synaptic transmission | Message send/receive; text links encapsulated individuals |

#### Heartbeat = Reticular Activating System (ARAS) in Detail

The ascending reticular activating system (ARAS) projects from the brainstem reticular formation through the thalamus to the cortex, sustaining arousal. It maps to AnimaWorks heartbeat as follows:

| ARAS Characteristic | Heartbeat Characteristic |
|---|---|
| Sustains arousal (does not specify conscious content) | Periodically activates the Anima (what to think is left to heartbeat.md) |
| Automatic, rhythmic firing | Runs automatically at the configured interval |
| Failure leads to coma | Without heartbeat, the Anima sleeps unless messaged |
| Precondition for consciousness, not consciousness itself | Precondition for autonomous action, not the judgment itself |
| Arousal varies with sensory input | Wakes immediately on inbound messages (even off heartbeat cadence) |

---

### Organizational Structure — The Social Brain

| AnimaWorks | Brain / Psychology Counterpart | Description |
|---|---|---|
| **Supervisor–subordinate hierarchy** | Neural basis of social hierarchy (PFC–amygdala circuit) | Flow of orders and reports |
| **Encapsulation (internals invisible)** | Theory of mind | Others' internal states are only inferable |
| **Messaging** | Linguistic communication | Text-only links; no shared memory or direct reference |
| **identity.md (personality)** | Personality (stable PFC–limbic patterns) | Immutable baseline for judgment |
| **injection.md (role)** | Social / occupational role | Mutable organizational guidance |

### Execution Modes — Levels of Autonomy

Memory subsystems are mode-agnostic, but cortical (LLM) executor choice changes how autonomous the agent is. Current code distinguishes **six modes** (`resolve_execution_mode()`):

| Mode | Executor | Brain Analogy | Description |
|---|---|---|---|
| **S** (SDK) | Claude Agent SDK | Full cortical function with executive control | Native Claude tools and session continuity |
| **C** (Codex) | Codex CLI | Cortical function close to S | OpenAI Codex-family models via Codex |
| **D** (Cursor Agent) | Cursor Agent CLI | External agent loop | MCP-integrated alternate path |
| **G** (Gemini CLI) | Gemini CLI | External agent loop | stream-json and tool loop |
| **A** (Autonomous) | LiteLLM + tool_use loop | Cortical function via external mediation | Multi-provider tool use managed by the framework |
| **B** (Basic) | One-shot (assisted) | Heavy external scaffolding | Framework handles memory I/O; session chaining largely unsupported |

Wildcard rules in `models.json` (and related config) auto-select mode; `status.json` `execution_mode` can override. S/C/D/G favor tools plus continued sessions; B strongly externalizes working memory — consistent with the memory mapping in this document.

---

## Neuroscientific Rationale for Design Principles

### Why Context Window Limits Are a "Feature"

Human working memory capacity is limited to about 4 ± 1 chunks (Cowan, 2001). This is not a defect but an **evolutionary adaptation that enforces selective attention and preserves judgment quality**. If all memories surfaced at once, relevant information could not be selected and decisions would degrade.

AnimaWorks adopts this as a deliberate design feature: prime only what is needed so the model decides in a clean context.

### Why Forgetting Is Necessary

Sleep-related synaptic downscaling weakens wake-strengthened synapses globally, preserving signal-to-noise ratio. Without forgetting, accumulated memories become noise and retrieval quality falls.

Active forgetting in AnimaWorks mimics this biology and helps keep vector search accurate long term.

### Why Collaboration Among "Imperfect Individuals" Beats One Omniscient Agent

Human organizations work because each member decides with limited view and memory, exchanging imperfect information in their own words (vision.md). That aligns with cognitive load theory (Sweller, 1988) and distributed cognition (Hutchins, 1995).

---

## Summary

AnimaWorks is a design born at the intersection of psychiatric clinical practice and engineering. The brain's information architecture is a **reusable design pattern** independent of its biological substrate (neurons); AnimaWorks is a system that demonstrates that reuse.

By mapping the LLM to the neocortex, the memory system to the hippocampal–cortical complex, heartbeat to the reticular activating system, and forgetting to synaptic homeostasis — integrated end to end — it realizes an entity that thinks, learns, forgets, and collaborates autonomously.
