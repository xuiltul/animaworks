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
- **Reconsolidation**: knowledge with `failure_count >= 2` and `confidence < 0.6` in front matter can be revised by the LLM through the knowledge path of `ReconsolidationEngine`

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
- **Reconsolidation**: when front matter has `failure_count >= 2` and `confidence < 0.6`, the LLM can revise the procedure. The revised file resets counters, increments version, and archives the old version
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
- Required skills are found through active skill context, the Skill Router, Skill Hub, or `read_memory_file(path="...")`
- You do not need to read every skill body up front. First use names, descriptions, or pointers; read the full text only when details are needed
- Proven `procedures/` may be promoted into probation or quarantine skills
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

The default `compact` profile recalls sender information, pending tasks, explicit resident pointers, recent outbound activity and pending human notifications. Related knowledge is searched for chat/task requests and questions, but not routine heartbeat/cron/report events. Broad activity, episode and graph expansion are available with the opt-in `full` profile or explicit search. `priming.max_tokens` defaults to 2,000; notifications and mandatory resident rules are preserved independently. Per-Anima `status.json: priming_profile` overrides the global profile without changing model routing.

Search when past instructions, customer facts or unfinished work are needed; no ritual search or success report is required for every response. Skills and procedure bodies are read on demand. Only explicitly resident knowledge is included automatically; `[IMPORTANT]` does not by itself mean always resident.

Before side effects, applicable `[ACTION-RULE]` checks, permission boundaries, approval and duplicate-action prevention still apply. Read indicated rules if an action is stopped. Untrusted search results remain separate from trusted context.

Daily consolidation extracts episodes from unprocessed activity chunks and checkpoints successful inputs. Raw activity and memory originals are retained. Knowledge rewriting is a separate, default-off phase (`consolidation.knowledge_mutation_enabled`). Weekly/monthly mutation, distillation, downscaling, self-correction, automatic skill learning and fact extraction are default off; indexing, repair and reading existing facts remain available. Optional maintenance must preserve entity detail, provenance and safety rules. A skipped/no-change run is normal, not a reason to retry.

Curator promotions/retirements are proposals by default; security blocking can still quarantine a skill immediately. Operator-controlled explicit changes remain possible. Outcome counts are diagnostic evidence, not proof of task quality.

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
2. **Embedding model**: `rag.embedding_model` in `config.json` (default `intfloat/multilingual-e5-small`). Child processes normally use the vector worker to delegate vector ops and embedding generation.
3. **Search**: The query is embedded; ranking combines similarity with **time decay**, reference frequency, etc. `rag.min_retrieval_score` in `config.json` can floor results.
4. **Graph spreading**: `rag.enable_spreading_activation` (default true) and `rag.spreading_memory_types` control **spreading activation** on the knowledge graph.
5. **Incremental updates**: Re-index changed files and run **full index rebuilds** after daily / weekly / monthly cycles to stay consistent.
6. **Repair**: If ChromaDB or vector search becomes inconsistent, RAG repair can quarantine `vectordb` and rebuild the index from memory files.

RAG is used automatically when you call `search_memory`. You need not think about internals, but **tips for better retrieval**:

- Use concrete keywords in queries
- When writing memories, use clear titles and bodies
- Keep related information in the same file
