# Memory System Guide

A reference for how Anima memory works, its types, and how to use each.
Refer to this when you need to understand memory search, writing, and organization.

## Memory Overview

Your memory is composed of multiple types that correspond to the human brain's memory model:

| Memory Type | Directory | Human Analogy | Content |
|-------------|-----------|---------------|---------|
| **Short-term memory** | `shortterm/` | Working memory | Context of recent conversations |
| **Episodic memory** | `episodes/` | Experiential memory | What you did and when |
| **Semantic memory** | `knowledge/` | Knowledge | What you've learned, know-how |
| **Procedural memory** | `procedures/` | Muscle memory | Step-by-step procedures |
| **Skills** | `skills/` | Specialties | Executable procedure guides |

Additionally, there are memories shared across all Animas:

| Shared Memory | Path | Content |
|---------------|------|---------|
| **Common knowledge** | `common_knowledge/` | Framework reference docs (including this file) |
| **Common skills** | `common_skills/` | Skills available to all Animas |
| **Organization knowledge** | `shared/common_knowledge/` | Knowledge accumulated by the org during operation |
| **User profiles** | `shared/users/` | Cross-Anima user information |

---

## Short-Term Memory (shortterm/)

**Holds context from recent conversations and sessions.** Corresponds to human working memory.

- Separated into Chat (`shortterm/chat/`) and Heartbeat (`shortterm/heartbeat/`)
- When context window usage exceeds the threshold, older portions are automatically externalized
- Used for context continuity between sessions

You don't need to operate short-term memory directly. The framework manages it automatically.

---

## Episodic Memory (episodes/)

**Daily logs of "what you did and when."** Corresponds to human experiential memory.

- Automatically recorded in date-based files (e.g., `2026-03-09.md`)
- Used for recalling "what was I doing last week?" or "have I handled this issue before?"
- Through Consolidation, patterns and lessons are refined into `knowledge/`

### Writing Memories

```
write_memory_file(path="episodes/2026-03-09.md", content="...")
```

### Searching Memories

```
search_memory(query="Slack API connection test", scope="episodes")
```

---

## Semantic Memory (knowledge/)

**Learned knowledge, know-how, and patterns.** Corresponds to what a human "knows."

- Lessons and patterns extracted from episodes
- Technical notes, response policies, decision criteria
- Automatically accumulated through daily Consolidation; you can also write proactively

Examples:
- "Slack API rate limit is 1 req/sec on Tier 1"
- "This client tends to send many messages on Mondays"
- "Pre-deployment checklist items"

### Writing Memories

```
write_memory_file(path="knowledge/slack-api-notes.md", content="...")
```

### Searching Memories

```
search_memory(query="Slack API rate limit", scope="knowledge")
```

---

## Procedural Memory (procedures/)

**Step-by-step procedure guides for "how to do it."** Corresponds to human "muscle memory."

- Problem-solving procedures, routine workflow steps
- Can be auto-generated from `issue_resolved` events (confidence 0.4)
- **Forgetting-resistant**: Important procedures are protected from the forgetting process

Examples:
- "SSL certificate renewal procedure"
- "New Anima onboarding procedure"
- "Production incident escalation procedure"

### Writing Memories

```
write_memory_file(path="procedures/ssl-renewal.md", content="...")
```

### Searching Memories

```
search_memory(query="SSL certificate renewal", scope="procedures")
```

---

## Skills (skills/)

**Executable procedure guides and tool usage guides.** Corresponds to "special abilities."

- Personal skills (`skills/`) and common skills (`common_skills/`) are available
- Priming automatically displays **names only** of relevant skills based on message content (progressive disclosure)
- Use the `skill` tool to get the full text when details are needed
- **Forgetting-resistant**

### Checking Skills

```
skill(name="newstaff")  # Get full skill text
```

### Creating Skills

```
create_skill(name="deploy-procedure", description="Production deploy procedure", content="...")
```

---

## Automatic Memory Processes

### Priming (Automatic Recall)

Every time you start a conversation, the Priming engine searches relevant memories in parallel across 6 channels and automatically injects them into the system prompt:

| Channel | What It Searches |
|---------|-----------------|
| Sender profile | User information about the other party |
| Recent activity | Recent action timeline |
| Related knowledge | Knowledge via RAG vector search |
| Skill match | Skill names relevant to the message |
| Pending tasks | Task queue summary |
| Episodes | Past experiences via RAG search |

**[IMPORTANT]** Knowledge tagged with `[IMPORTANT]` is always injected as summary pointers in Priming, in addition to RAG search results. Only the summary is shown; use `read_memory_file` for full details. When moving important business rules to knowledge/, add the `[IMPORTANT]` tag.

Priming runs automatically, so no explicit action is needed.

### Consolidation (Memory Integration)

Processes that automatically organize and refine memories:

| Frequency | Process |
|-----------|---------|
| **Daily** | Episodes → Knowledge (extract patterns and lessons) |
| **Daily** | Problem resolution → Procedures (auto-generate procedures from fix feedback) |
| **Weekly** | Knowledge merge + episode compression |

### Forgetting (Active Forgetting)

Accumulating memories indefinitely degrades search accuracy, so active forgetting occurs in 3 stages:

| Stage | Frequency | Process |
|-------|-----------|---------|
| Synaptic downscaling | Daily | Mark chunks with no access for 90 days and fewer than 3 references |
| Neurogenesis reorganization | Weekly | Merge low-activity chunks with similarity above 0.80 |
| Complete forgetting | Monthly | Archive and delete chunks with low activity for 60+ days |

**Protected (never forgotten)**: `procedures/`, `skills/`, `shared/users/`

---

## Choosing the Right Memory Tool

| What You Want to Do | Tool | Example |
|--------------------|------|---------|
| Search by keyword | `search_memory` | `search_memory(query="API config", scope="all")` |
| Read a specific file | `read_memory_file` | `read_memory_file(path="knowledge/api-notes.md")` |
| Write a memory | `write_memory_file` | `write_memory_file(path="knowledge/new-insight.md", content="...")` |
| Archive obsolete memory | `archive_memory_file` | `archive_memory_file(path="knowledge/outdated.md")` |

### Choosing a scope

| scope | Search Target | When to Use |
|-------|--------------|-------------|
| `knowledge` | Knowledge, know-how | "Do I know anything about this?" |
| `episodes` | Past action logs | "Have I done this before?" |
| `procedures` | Procedure guides | "What are the steps for this?" |
| `common_knowledge` | Shared references | "What does the framework spec say?" |
| `all` | All of the above | "Find all related information" |

---

## How RAG (Vector Search) Works

RAG (Retrieval-Augmented Generation) powers memory search:

1. **Indexing**: All memory files are converted to embedding vectors and stored in ChromaDB
2. **Search**: Query text is vectorized and memory chunks with high similarity are retrieved
3. **Graph diffusion**: NetworkX graph-based spreading activation pulls in related peripheral memories
4. **Incremental updates**: Only changed files are re-indexed, so it stays fast even with large memory stores

RAG is used automatically when you call `search_memory`. You don't need to think about the mechanism, but **tips for better search accuracy**:
- Use queries with specific keywords
- When writing memories, use clear titles and content
- Group related information in the same file
