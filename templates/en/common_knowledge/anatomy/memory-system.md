# Memory System Guide

A reference for how Anima memory works, its types, and how to use each.
Consult it when you need to confirm how to search, write, and organize memories.

## Memory Overview

Your memory is composed of multiple types that correspond to the human brain’s memory model:

| Memory type | Directory | Human analogy | Content |
|-------------|-----------|---------------|---------|
| **Short-term memory** | `shortterm/` | Working memory | Context of recent conversations |
| **Episodic memory** | `episodes/` | Experiential memory | What you did and when |
| **Semantic memory** | `knowledge/` | Knowledge | What you’ve learned, know-how |
| **Procedural memory** | `procedures/` | Procedural / “muscle” memory | Step-by-step how-to |
| **Skills** | `skills/` | Specialties | Executable procedure guides |

Additionally, memories shared across all Animas:

| Shared memory | Path | Content |
|---------------|------|---------|
| **Common knowledge** | `common_knowledge/` | Framework reference (this file included) |
| **Common skills** | `common_skills/` | Skills available to every Anima |
| **Organization shared knowledge** | `shared/common_knowledge/` | Knowledge accumulated during org operations |
| **User profiles** | `shared/users/` | Cross-Anima user information |

---

## Short-Term Memory (`shortterm/`)

**Holds context from recent conversations and sessions.** Corresponds to human working memory.

- Split by session kind: `shortterm/chat/` and `shortterm/heartbeat/` (optional per-`thread_id` subdirectories when needed)
- Each directory has `session_state.json` / `session_state.md` and `archive/`
- When context window usage exceeds the threshold, older portions are automatically externalized
- Used for context continuity across sessions

You do not need to manipulate short-term memory yourself; the framework manages it automatically.

---

## Episodic Memory (`episodes/`)

**Daily logs of “what you did and when.”** Corresponds to human experiential memory.

- Automatically recorded in per-date files (e.g. `2026-03-09.md`)
- Used to recall “what was I doing last week?” or “have I handled this issue before?”
- In daily / weekly **Consolidation** (memory integration), the Anima’s own tool loop performs summarization, knowledge extraction, etc. (see below)

### Writing memories

```
write_memory_file(path="episodes/2026-03-09.md", content="...")
```

### Searching memories

```
search_memory(query="Slack API connection test", scope="episodes")
```

---

## Semantic Memory (`knowledge/`)

**Learned knowledge, know-how, and patterns.** Corresponds to what a human “knows.”

- Lessons and patterns extracted from episodes
- Technical notes, response policies, decision criteria
- Accumulated automatically via Consolidation; you can also write proactively
- Legacy files are migrated to YAML front matter on first run (`knowledge/.migrated` marker)

Examples:

- “Slack API rate limit on Tier 1 is 1 req/sec”
- “This client tends to send many messages on Mondays”
- “Pre-deployment checklist”

### Writing memories

```
write_memory_file(path="knowledge/slack-api-notes.md", content="...")
```

### Searching memories

```
search_memory(query="Slack API rate limit", scope="knowledge")
```

---

## Procedural Memory (`procedures/`)

**Step-by-step “how to do it” guides.** Corresponds to procedures the body “knows by heart.”

- Problem-solving steps, routine workflows
- May be auto-generated from events such as `issue_resolved` (with metadata like confidence 0.4)
- **Not as fully protected as skills**: based on metadata, items can enter the forgetting pipeline (procedure-specific rules below)
- Version history lives under `archive/`; older versions are pruned after a cap

Examples:

- “SSL certificate renewal procedure”
- “New Anima onboarding procedure”
- “Production incident escalation procedure”

### Writing memories

```
write_memory_file(path="procedures/ssl-renewal.md", content="...")
```

### Searching memories

```
search_memory(query="SSL certificate renewal", scope="procedures")
```

---

## Skills (`skills/`)

**Executable procedure guides and tool usage guides.** Corresponds to “specialties.”

- Personal skills (`skills/`) and common skills (`common_skills/`)
- The system prompt skill catalog lists paths such as `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`, and `procedures/baz.md`
- When you need details, fetch the full text with `read_memory_file(path="...")`
- **In the vector store, skills are always outside the forgetting scope** (`skills` / `shared_users` types are protected)

### Inspecting a skill

```
read_memory_file(path="skills/newstaff/SKILL.md")  # Full skill text
```

### Creating a skill

```
create_skill(skill_name="deploy-procedure", description="Production deploy procedure", body="...")
```

---

## Automatic memory processes

### Priming (automatic recall)

Each time a conversation starts, the Priming engine runs **five channels (A, B, C/C0, E, F)** in parallel, searches related memories, and injects them into the system prompt. In implementation, **C0 (important knowledge)** is fetched first and **concatenated with Channel C body text** at injection time (often appearing as one “related knowledge” block).

| Channel | What it searches | Default per-channel budget (token guide) * |
|---------|------------------|-------------------------------------------|
| A: Sender profile | The other party’s user info | 500 |
| B: Recent activity | Unified activity log (timeline) | 1300 |
| C0: Important knowledge | **Summary pointers only** for `[IMPORTANT]`-tagged knowledge | 500 |
| C: Related knowledge | RAG (dense): personal `knowledge` + shared `common_knowledge` | 1000 |
| E: Pending tasks | Task queue summary + in-flight parallel tasks + delegated task status + (if any) overflow inbox file names | 500 |
| F: Episodes | RAG search over `episodes/` | 800 |

Skill and procedure bodies are listed in the system prompt skill catalog (`<available_skills>`, injected into Group 4) and loaded with `read_memory_file`. Semantic skill discovery is also available via `search_memory(scope="skills")`.

\* You can change greeting / question / request / heartbeat caps and `heartbeat_context_pct` under `priming` in `config.json`. When `dynamic_budget` is enabled, overall token limits for message types and heartbeat use expressions such as `max(budget_heartbeat, context_window × heartbeat_context_pct)`, and each channel scales by ratio.

Also injected:

- **Recent outbound history**: `channel_post` / `message_sent` in the last 2 hours (max 3 items)
- **Pending human notifications**: `human_notify` in the last 24 hours (up to ~500 tokens)

**Channel C and trust**: Results are split into **medium** and **untrusted** from chunk `origin`, etc. Untrusted content is trimmed into a separate slice from the remaining budget and handled in the prompt-injection defense context (see `common_knowledge/security/`).

**`[IMPORTANT]` and C0**: `[IMPORTANT]` knowledge appears in C0 as “title + one-line summary + pointer to `read_memory_file`.” Because it is on a path separate from ordinary RAG (C), important rules are harder to miss even when the query does not match. When moving must-have business rules into knowledge, prefix with `[IMPORTANT]`.

Priming runs automatically; no explicit action is required.

### Consolidation (memory integration)

**Production integration work** (episode summarization, extraction into knowledge, consistency checks, etc.) **is executed by the Anima’s own tool loop** (`run_consolidation`). `ConsolidationEngine` mainly handles preprocessing (collecting episodes and resolution events) and postprocessing (RAG rebuild, invoking forgetting).

| Frequency | Flow (summary) |
|-----------|----------------|
| **Daily** | If episode count in the last 24h meets the threshold → `run_consolidation(daily)` → then **Synaptic downscaling** (metadata only) → **RAG index rebuild** |
| **Weekly** | `run_consolidation(weekly)` → **Neurogenesis reorganization** (LLM merge of low-activity similar chunks) → **RAG rebuild** |
| **Monthly** | **Complete forgetting** (delete / archive chunks that stayed low-activity long-term) and procedure archive housekeeping → **RAG rebuild** (no Anima loop) |

Daily runs can be disabled or tuned (episode threshold, `max_turns`) via config. Very short runs may be retried on schedule.

### Forgetting (active forgetting)

If memories accumulate without bound, search quality drops; forgetting is applied in three stages:

| Stage | Frequency | Condition | Action |
|-------|-----------|-----------|--------|
| Synaptic downscaling | Daily | `knowledge` / `episodes`: no access for **90 days** **and** fewer than **3** references → mark `low`. Procedures use different thresholds (e.g. unused **180** days and fewer than **3** total uses, or many failures and low utility) | Record activity `low` and `low_activation_since` |
| Neurogenesis reorganization | Weekly | Activity `low`, not protected, vector similarity **≥ 0.80** between pairs | LLM merge → remove original chunks; merge on-disk sources (moved under `archive/merged/`) |
| Complete forgetting | Monthly | Stays `low` for **> 90 days** and `access_count <= 2` | Remove from vectors; move sources to `archive/forgotten/` |

**Protection rules** (harder to forget):

| Target | Protection |
|--------|------------|
| `skills/`, `shared/users/` (as types) | Always protected (never in forgetting scope) |
| `[IMPORTANT]` (`importance == important`) | Protected in principle. **However**, if there is **no access for 365 days** since last access or update, a safety net lifts protection and normal forgetting applies (weekly integration is assumed to have absorbed content into knowledge) |
| Knowledge: `success_count >= 2` | Protected |
| Procedures: `importance == "important"` / `protected == True` / `version >= 3` | Protected (procedure-specific checks) |

**Procedure-specific rule**: Subject to downscaling if inactive **180** days with fewer than **3** uses, or `failure_count >= 3` and utility below **0.3**.

---

## Choosing memory tools

| Goal | Tool | Example |
|------|------|---------|
| Keyword search | `search_memory` | `search_memory(query="API configuration", scope="all")` |
| Read a file | `read_memory_file` | `read_memory_file(path="knowledge/api-notes.md")` |
| Write memory | `write_memory_file` | `write_memory_file(path="knowledge/new-insight.md", content="...")` |
| Tidy obsolete memory | `archive_memory_file` | `archive_memory_file(path="knowledge/outdated.md")` |

### Choosing `scope`

| scope | Search target | When to use |
|-------|---------------|-------------|
| `knowledge` | Knowledge, know-how | “Do I know anything about this?” |
| `episodes` | Past action logs | “Have I done this before?” |
| `procedures` | Procedure docs | “What are the steps for this task?” |
| `common_knowledge` | Shared reference | “What does the framework spec say?” |
| `skills` | Skills and common skills (vector search) | “Is there a skill for this task?” |
| `activity_log` | Recent action logs (tool results, messages, etc.) | “What was in the email I just read?” “The search results from earlier” |
| `all` | All of the above (vector search + activity_log BM25 fused via RRF) | Broad search across all memory types |

---

## How RAG (vector search) works

Memory search uses RAG (Retrieval-Augmented Generation):

1. **Indexing**: `knowledge/`, `episodes/`, `procedures/`, shared `common_knowledge/`, etc. are chunked, embedded, and stored in a vector store (default Chroma, persistent per-Anima directory). Summaries in `state/conversation.json` may also be indexed.
2. **Embedding model**: `rag.embedding_model` in `config.json` (default `intfloat/multilingual-e5-small`). If `ANIMAWORKS_VECTOR_URL` / `ANIMAWORKS_EMBED_URL` are set, child processes can delegate vector ops and embedding generation via the server.
3. **Search**: The query is embedded; ranking combines similarity with **time decay**, reference frequency, etc. `rag.min_retrieval_score` in `config.json` can floor results.
4. **Graph spreading**: `rag.enable_spreading_activation` (default true) and `rag.spreading_memory_types` control **spreading activation** on the knowledge graph.
5. **Incremental updates**: Re-index changed files and run **full index rebuilds** after daily / weekly / monthly cycles to stay consistent.

RAG is used automatically when you call `search_memory`. You need not think about internals, but **tips for better retrieval**:

- Use concrete keywords in queries
- When writing memories, use clear titles and bodies
- Keep related information in the same file
