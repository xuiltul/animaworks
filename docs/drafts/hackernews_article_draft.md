# Show HN: AnimaWorks – AI agents that forget, learn, and organize like brains

> **HN投稿タイトル候補:**
> - `Show HN: AnimaWorks – AI agents that forget, learn, and organize like brains`
> - `Show HN: AnimaWorks – Organization-as-Code for LLM agents with neuroscience-based memory`
> - `Show HN: AnimaWorks – Build an AI office where agents remember, forget, and grow`
>
> **投稿先URL:** GitHub リポジトリ直リンク（LPではない）
> **投稿タイミング:** 火-水 8-10AM Pacific、または日曜 6-9PM Pacific

---

## HN ファーストコメント（投稿直後に自分で書く）

Hi HN, I'm the creator of AnimaWorks.

I've been running a small company in Japan for 10+ years. About two years ago, I started building AI assistants for internal operations — customer support, infrastructure monitoring, code review. The existing multi-agent frameworks (CrewAI, AutoGen, LangGraph) all treated agents as stateless functions: spin up, execute, forget everything.

That's not how a real organization works. Employees remember past mistakes. They learn from experience. They forget irrelevant details over time. They have their own identity and communicate through messages, not shared memory.

So I built AnimaWorks — a framework where each AI agent ("Anima") is a persistent, encapsulated individual with:

**1. Neuroscience-based memory lifecycle**

Instead of dumping everything into a vector store, AnimaWorks models memory after the human brain:

- **Priming** (hippocampal pattern completion): Before each LLM call, 5 parallel channels automatically retrieve relevant memories and inject them into the system prompt. No explicit `search_memory` tool call needed. Budget-controlled per message type (greeting: 500 tokens, request: 3000 tokens).

- **Consolidation** (NREM sleep analog): Daily, episodic memories are distilled into semantic knowledge. Weekly, knowledge is merged and episodes are compressed. When an Anima resolves an error, the fix is automatically extracted into a reusable procedure (confidence starts at 0.4, increases with repeated validation).

- **Active forgetting** (synaptic homeostasis): 3 stages — synaptic downscaling (daily: mark chunks with <3 accesses in 90 days), neurogenesis reorganization (weekly: merge >0.80 similarity low-activity chunks), complete forgetting (monthly: archive and delete). Skills, procedures, and user profiles are protected from forgetting.

This means the same question asked on Day 1 and Day 30 may get a different (better) answer — not because someone updated the prompt, but because the agent's memory naturally evolved.

**2. Organization-as-Code**

Define an org structure in Markdown. Each Anima has `identity.md` (immutable personality), `injection.md` (mutable role/guidelines), `permissions.md` (tool/command access), and a `supervisor` field. The hierarchy creates reporting lines, delegation flows, and escalation paths — just like a real company.

Supervisor Animas get tools like `delegate_task`, `org_dashboard`, `ping_subordinate`, and `task_tracker`. Subordinates execute delegated tasks in minimal-context sessions to save tokens.

**3. Heterogeneous multi-model teams**

Not every role needs Opus. AnimaWorks auto-selects execution mode based on model name:

| Role | Model | Mode |
|------|-------|------|
| Manager | claude-opus-4-6 | S (Agent SDK) |
| Engineer | claude-sonnet-4-6 | S (Agent SDK) |
| Ops/Monitor | openai/glm-4.7-flash (vLLM local) | A (LiteLLM) |
| Writer | claude-sonnet-4-6 | A (LiteLLM) |
| Intern | ollama/qwen3:0.6b | B (framework-assisted) |

Three execution modes: **S** (Claude Agent SDK, full autonomy), **A** (LiteLLM + tool_use loop, multi-provider), **B** (framework does memory I/O, LLM just thinks). Mix cloud APIs and local models in the same org.

**4. Process isolation**

Each Anima runs as a separate child process with Unix Domain Socket IPC. No shared memory between agents — they communicate only through text messages via `Messenger`. This is intentional: in a real organization, you can't read your colleague's mind.

**5. 3-path execution separation**

- **Chat/Inbox**: Human conversations and inter-Anima DMs (separate locks)
- **Heartbeat**: Periodic Observe → Plan → Reflect cycle (no execution — that goes to task queue)
- **Cron/TaskExec**: Scheduled tasks and delegated task execution

Each path has independent locks, separate short-term memory files, and appropriate context budgets (tiered system prompt: T1 Full for 128k+ context → T4 Minimal for <16k).

---

**How it compares to OpenClaw** (the 229K-star personal AI assistant):

| | OpenClaw | AnimaWorks |
|---|---|---|
| Focus | Single-user personal assistant | Multi-agent organization |
| Memory | Markdown + SQLite vectors | RAG + Priming + Consolidation + Forgetting |
| Autonomy | Cron + webhooks | Heartbeat (Observe/Plan/Reflect) + Cron + TaskExec |
| Org model | Session routing | Hierarchy, delegation, supervisor tools |
| Process model | Gateway + optional Docker | ProcessSupervisor + Unix socket per agent |
| Multi-model | Yes | Yes, with automatic mode selection (S/A/B) |

OpenClaw is excellent for personal productivity. AnimaWorks is for building persistent teams of specialized agents.

---

**Technical details:**

- Python 3.12+ / FastAPI / ChromaDB + sentence-transformers (multilingual-e5-small, 384-dim)
- Graph-based spreading activation (NetworkX + Personalized PageRank)
- Trust labeling on tool results (`trusted`/`medium`/`untrusted`) for prompt injection defense
- 3-layer outbound rate limiting (per-run dedup, cross-run 30/hr 100/day, behavior-aware)
- Streaming journal (WAL) for crash recovery
- Voice chat (faster-whisper STT + VOICEVOX/ElevenLabs TTS, PTT/VAD, barge-in)

**What's working in production:** I use AnimaWorks daily to run my own company's operations — infrastructure monitoring with local GLM-4.7 models, customer communication via Chatwork/Slack, and development task delegation. It's dogfooded extensively.

GitHub: [link]
Docs: [link]
Paper (memory system evaluation): [link]

Happy to answer questions about the neuroscience-inspired memory design, the multi-model architecture, or anything else.

---

## 技術ブログ記事（HN投稿前に公開する詳細記事）

# Why I Built an AI Framework Around Forgetting

*Most AI frameworks help agents remember. AnimaWorks also helps them forget.*

## The Problem with Infinite Memory

Every AI agent framework today has a memory problem — but not the one you'd expect.

The obvious problem is that agents forget too much. Context windows are finite. Sessions end. Knowledge is lost.

The less obvious problem is that agents remember too much. As memory grows, retrieval degrades. Irrelevant old memories pollute search results. The agent drowns in its own history.

Humans solved this millions of years ago. We have a memory system that automatically:
1. **Retrieves** relevant memories before we consciously think about them (priming)
2. **Consolidates** experiences into abstract knowledge during sleep
3. **Forgets** low-value information to keep the signal-to-noise ratio high

AnimaWorks implements all three.

## How It Works

### Priming: The Hippocampus as a Search Engine

When you see someone's face and instantly recall their name, favorite coffee order, and that awkward thing that happened at last year's party — that's priming. Your hippocampus performs pattern completion *before* your conscious mind asks for it.

In AnimaWorks, the `PrimingEngine` runs 5 parallel retrieval channels before every LLM call:

```
Channel A: Sender Profile     (500 tokens)  — Who is talking to me?
Channel B: Recent Activity     (1300 tokens) — What have I been doing?
Channel C: Related Knowledge   (700 tokens)  — What do I know about this topic?
Channel D: Skill Match         (200 tokens)  — Do I have a procedure for this?
Channel E: Pending Tasks       (300 tokens)  — What am I supposed to be working on?
```

The results are injected into the system prompt with trust-level labels. The agent never calls `search_memory` — relevant context just *appears*, exactly like human priming.

Budget varies by message type: a greeting gets 500 tokens of priming (you don't need much context for "good morning"), while a complex request gets 3000 tokens.

### Consolidation: Learning While Sleeping

Every night (via daily cron), AnimaWorks runs a consolidation pipeline:

1. **Episode → Knowledge**: Today's episodic memories are analyzed by an LLM. Patterns, lessons, and generalizations are extracted and written to `knowledge/` as semantic memory.

2. **Issue → Procedure**: When an Anima resolves an error (detected via `issue_resolved` events in the activity log), the resolution is automatically converted into a reusable procedure in `procedures/`. The procedure starts with confidence 0.4 and increases each time it's validated.

3. **Weekly integration**: Knowledge files are merged to remove redundancy. Old episodes are compressed.

This mirrors the neuroscience of sleep-dependent memory consolidation — specifically, the role of NREM slow-wave sleep in transferring hippocampal memories to neocortical storage.

### Forgetting: The Feature No One Else Built

This is the part that surprises people. Why would you deliberately *delete* an agent's memories?

Because memory is not just storage — it's retrieval. Every irrelevant memory in the store degrades search precision for everything else. The synaptic homeostasis hypothesis in neuroscience suggests that sleep-based synaptic downscaling is essential for maintaining the brain's signal-to-noise ratio.

AnimaWorks implements 3-stage forgetting:

| Stage | Frequency | Action |
|-------|-----------|--------|
| Synaptic downscaling | Daily | Mark chunks with <3 accesses in 90 days |
| Neurogenesis reorganization | Weekly | Merge >0.80 cosine similarity low-activity chunks |
| Complete forgetting | Monthly | Archive and delete chunks with >60 days of low activity |

Protected categories (skills, procedures, user profiles) are exempt — you don't forget how to ride a bike.

Our ablation study shows consolidation improves search precision from 0.333 to 0.667 (100% improvement). Reconsolidation (procedure revision after errors) achieved 100% success rate in Round 2 vs 0% without it.

## Organization-as-Code

The memory system gets the most technical attention, but Organization-as-Code is equally important.

Most multi-agent frameworks model agents as functions in a pipeline:

```python
# CrewAI style
crew = Crew(agents=[researcher, writer, reviewer], tasks=[...])
result = crew.kickoff()  # Agents are born, execute, die
```

AnimaWorks models agents as employees in a company:

```
~/.animaworks/animas/
├── dev-manager/          # Manages the engineering team
│   ├── identity.md       # "I am a detail-oriented engineering manager..."
│   ├── injection.md      # "You manage developer-1 and developer-2..."
│   ├── permissions.md    # Can use GitHub, delegate tasks
│   ├── heartbeat.md      # Check PR queue, review team status
│   ├── status.json       # model: claude-opus-4-6, role: manager
│   ├── episodes/         # What happened today
│   ├── knowledge/        # Lessons learned over weeks
│   └── procedures/       # "How to handle a P1 incident"
├── developer-1/          # Individual contributor
│   ├── identity.md       # Different personality
│   ├── status.json       # model: claude-sonnet-4-6, role: engineer
│   └── ...
└── ops-monitor/          # Infrastructure monitoring
    ├── status.json       # model: openai/glm-4.7-flash (local vLLM)
    └── ...
```

Each Anima lives in its own directory. Has its own memory, personality, permissions, and heartbeat schedule. Communicates with others only through text messages. Cannot read another Anima's memory or internal state.

This encapsulation is the most distinctive design choice. No other framework we've found implements it. It's directly inspired by how real organizations work — you can't read your colleague's mind, and that constraint forces structured communication.

## Multi-Model Cost Optimization

Running everything on Opus is expensive. Running everything on a local model is slow and error-prone. AnimaWorks lets you match the model to the role:

- **Monitoring (GLM-4.7 on local GPU)**: Parses logs, detects anomalies. $0/month.
- **Management (Sonnet)**: Triages issues, delegates tasks. Moderate cost.
- **Complex engineering (Opus)**: Architecture decisions, difficult debugging. High cost, used sparingly.
- **Routine tasks (Ollama/qwen3)**: Simple text processing. $0/month.

Three execution modes handle the differences automatically:
- **Mode S**: Claude Agent SDK (full tool use, subprocess)
- **Mode A**: LiteLLM + tool_use loop (works with any provider)
- **Mode B**: Framework does memory I/O, LLM just generates text (for small models)

The mode is resolved from the model name via wildcard pattern matching. No configuration needed.

## What I Learned Building This

1. **Memory precision matters more than memory size.** A small, well-curated memory store outperforms a large noisy one. This is why forgetting is a feature.

2. **Process isolation prevents cascade failures.** When one agent enters an infinite loop, others keep working. Unix sockets + supervisor pattern = resilience.

3. **Encapsulation forces better communication.** When agents can't peek at each other's internals, they write clearer messages. This mirrors Conway's Law.

4. **Heartbeat without execution is key.** The original design had heartbeats that could execute actions, which led to runaway behaviors. Now heartbeat only observes, plans, and reflects. Execution goes through the task queue.

5. **Trust labels on tool results are essential.** Without them, web search results and Slack messages can inject instructions into the agent's context. Three trust levels (trusted/medium/untrusted) with per-level interpretation rules prevent this.

---

*AnimaWorks is Apache-2.0 licensed. Python 3.12+. Tested on Linux.*

*GitHub: [link] | Docs: [link] | Memory system paper: [link]*

---

> **HN投稿時の注意事項メモ:**
> - タイトルに誇張表現を使わない
> - 投票リングは絶対禁止（永久BAN）
> - 全コメントに10分以内に返信
> - 失敗したら48時間待ってタイトル変更で再投稿可
> - 記事リンクではなくGitHub直リンクを投稿URLに
