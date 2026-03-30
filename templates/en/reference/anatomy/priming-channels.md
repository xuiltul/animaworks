# Priming Channels Technical Reference

Detailed specification of all channels executed by PrimingEngine.
Includes budget, search sources, filtering, and dynamic adjustment.

Parallel retrieval uses **five channels** (A / B / C / E / F). Channel C0 (important_knowledge) is an auxiliary block inside the same pipeline as Channel C. The former Channel D (Distilled Knowledge) and any “six channel” wording are obsolete.

---

## Channel Overview

| Channel | Budget (tokens) | Source | trust |
|---------|---------------------|--------|-------|
| A: sender_profile | 500 | `shared/users/{sender}/index.md` | medium |
| B: recent_activity | 1300 | `activity_log/` + shared channels | trusted |
| C: related_knowledge | 1200 | RAG vector search (knowledge + common_knowledge) | medium / untrusted |
| C0: important_knowledge | 300 | Chunks tagged with `[IMPORTANT]` | medium |
| E: pending_tasks | 500 | `task_queue.jsonl` + `task_results/` | trusted |
| F: episodes | 500 | RAG vector search (episodes/) | medium |

Additional injection:

| Item | Budget | Source | trust |
|------|-----------|--------|-------|
| Recent outbound | No limit (max 3 items) | activity_log (last 2 hours, `channel_post` / `message_sent`) | trusted |
| Pending human notifications | 500 | `human_notify` events (last 24 hours) | trusted |

Skill and procedure bodies are not injected by Priming. Use paths shown in the system prompt skill catalog (e.g. `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`, `procedures/baz.md`) and load them with `read_memory_file`.

---

## Channel A: sender_profile

Injects the sender's user profile.

- **Source**: Direct read of `shared/users/{sender}/index.md`
- **Budget**: 500 tokens
- **When sender unknown**: Skipped

---

## Channel B: recent_activity

Injects the recent activity timeline.

- **Source**: `activity_log/{date}.jsonl` + latest posts from shared channels
- **Budget**: 1300 tokens

**Priming vs. explicit search:** Channel B injects a **budget-limited**, automatic timeline from `activity_log/`. That is separate from on-demand recall: use `search_memory(query="...", scope="activity_log")` when you need to **query** recent actions (tool results, messages, etc.) beyond what Priming surfaced.

### Trigger-based Filtering

| Trigger | Excluded event types |
|---------|----------------------|
| `heartbeat` / `cron:*` | `tool_use`, `tool_result`, `heartbeat_start`, `heartbeat_end`, `heartbeat_reflection`, `inbox_processing_start`, `inbox_processing_end` |
| `chat` | `cron_executed` |

---

## Channel C: related_knowledge

Injects related knowledge via RAG vector search.

- **Budget**: 1200 tokens
- **Search method**: Dual-query (message context + keywords only)
- **Search target**: Personal `knowledge/` + `shared_common_knowledge` collection
- **Min score**: `config.json` `rag.min_retrieval_score` (default 0.3)

### trust Separation

Search results are separated by trust level based on chunk `origin`:

| trust | Target | Processing |
|-------|------|------|
| `medium` | Personal knowledge, common_knowledge | Consumes budget preferentially |
| `untrusted` | From external platforms (`origin_chain` contains `external_platform`) | Injected with remaining budget. Tagged with `origin=ORIGIN_EXTERNAL_PLATFORM` |

---

## Channel C0: important_knowledge

Always injects summary pointers for chunks tagged with `[IMPORTANT]`.

- **Budget**: 300 tokens
- **Target**: Chunks tagged with `[IMPORTANT]` in `knowledge/`
- **Injection format**: Summary pointers only (not full text). Details fetched via `read_memory_file`
- **Purpose**: Reliable recall of important business rules and decision criteria

---

## Channel E: pending_tasks

Injects task queue summary.

- **Budget**: 500 tokens
- **Source**: `TaskQueueManager.format_for_priming()`
- **Content**:
  - List and summary of `pending` / `in_progress` tasks
  - 🔴 HIGH marker for `source: human` tasks
  - ⚠️ STALE marker for tasks with no update for 30+ minutes
  - 🔴 OVERDUE marker for overdue tasks
  - Progress of active parallel tasks (submit_tasks batches)
  - Completed task results from `task_results/`
  - Failed tasks with `status: failed` + `meta.executor == "taskexec"`

---

## Channel F: episodes

Injects related episodes via RAG vector search.

- **Budget**: 500 tokens
- **Search target**: `episodes/` collection (ChromaDB)
- **Min score**: Same as Channel C (`rag.min_retrieval_score`)

---

## Dynamic Budget Adjustment

Enabled by `config.json` `priming.dynamic_budget: true` (default).

### Budget by Message Type

| Message type | Budget | Config key |
|----------------|-----------|---------|
| greeting | 500 | `priming.budget_greeting` |
| question | 1500 | `priming.budget_question` |
| request | 3000 | `priming.budget_request` |
| heartbeat (fallback) | 200 | `priming.budget_heartbeat` |

### Heartbeat Budget Calculation

```
heartbeat_budget = max(budget_heartbeat, context_window × heartbeat_context_pct)
```

- `heartbeat_context_pct`: Default 0.05 (5% of context window)
- Example: context_window=200000 → `max(200, 200000 × 0.05)` = 10000

---

## Hebbian LTP (Long-Term Potentiation)

Chunks retrieved and displayed by Priming have their activation updated via `record_access()`. This prevents forgetting of frequently recalled memories (integration with Forgetting engine).
