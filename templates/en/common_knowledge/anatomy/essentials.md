# AnimaWorks Essential Guide

[IMPORTANT] A single-page guide to understand the full picture of AnimaWorks.
Covers Heartbeat / Cron / team design / memory / cost optimization essentials.
Read this first when onboarding or when you need to organize how concepts relate.
For details on each topic, follow the links at the end of each section.

---

## What is AnimaWorks?

A framework that treats AI agents not as "tools" but as **autonomous personalities**.

- Each Anima has its own personality, memory, and decision-making criteria
- Acts autonomously on a schedule, without waiting for human instructions (Heartbeat / Cron)
- Fills a role within an organization and collaborates with other Animas and humans
- Learns from experience, accumulates memories, and grows

---

## 5 Execution Paths — "When and How Does It Act?"

An Anima runs on five paths. All except Chat start automatically.

| Path | When it runs | What it does | Who uses it |
|------|-------------|-------------|-------------|
| **Chat** | When a human sends a message | Conversational response | Human → Anima |
| **Inbox** | When another Anima sends a DM | Immediate response to internal messages | Anima → Anima |
| **Heartbeat** | Periodic auto-start (default: every 30 min) | Observe → Plan → Reflect. **Does NOT execute tasks** | Automatic |
| **Cron** | Per cron.md schedule (e.g. daily at 9:00) | Execute scheduled tasks at fixed times | Automatic |
| **TaskExec** | When a task appears in `state/pending/` | Execute real work in an LLM session | Automatic (submitted by Heartbeat or submit_tasks) |

Chat and Heartbeat run on **separate locks**, so the Anima can respond to humans instantly even during a patrol.

→ Details: `anatomy/what-is-anima.md`

---

## Heartbeat vs Cron — Two Kinds of Autonomy

Both act "without human instructions," but their purposes are fundamentally different.

| Aspect | Heartbeat (Periodic Patrol) | Cron (Scheduled Task) |
|--------|---------------------------|----------------------|
| **Analogy** | A security guard patrolling the office regularly | A newspaper delivered every morning at 9 AM |
| **Purpose** | Situational awareness, planning, reflection | Execute predetermined work |
| **Does it execute?** | **No.** It only submits tasks via `submit_tasks` or `delegate_task` | **Yes.** LLM type thinks and acts; Command type runs immediately |
| **Interval** | Fixed interval (default 30 min, varies with Activity Level) | Flexible cron expressions (daily 9:00, every Friday 17:00, etc.) |
| **Config file** | `heartbeat.md` (checklist) | `cron.md` (task definitions) |
| **Typical examples** | Check unread messages, detect blockers, review progress | Morning work plan, weekly report, backup execution |

**When in doubt:** "Just check and decide" → Add to Heartbeat checklist. "Do something at a specific time" → Define as a Cron task.

→ Details: `operations/heartbeat-cron-guide.md`

---

## Two Types of Cron — LLM vs Command

Cron tasks come in two types depending on whether thinking is required.

| Aspect | LLM type (`type: llm`) | Command type (`type: command`) |
|--------|----------------------|-------------------------------|
| **Analogy** | "Think about what to prioritize today" | "Press this button every morning" |
| **Judgment** | Yes (output varies by situation) | No (same thing every time) |
| **API cost** | Yes (LLM call) | The command itself: none. However, **a follow-up LLM may start** (see below) |
| **Output** | Variable (differs per task) | Deterministic (command stdout) |
| **Best for** | Planning, reflection, writing, memory organization | Backups, notifications, data fetching, health checks |

### Command Type Follow-up LLM (Important)

Command type executes the command mechanically, but **if stdout has output, an LLM starts by default to analyze the result** (follow-up). So it is not always zero-cost.

```
Command runs → Has stdout?
  → No → Done (no LLM needed)
  → Yes → Matches skip_pattern?
      → Match → Done (LLM skipped)
      → No match → LLM starts to interpret results and decide on action
```

Options to control this follow-up:
- **`trigger_heartbeat: false`** — Always skip the follow-up LLM (when result analysis is unnecessary)
- **`skip_pattern: <regex>`** — Skip only when stdout matches (ignore normal output, let LLM judge anomalies)

### Decision Guide

```
"Do I just do the same thing every time?"
  → Yes, and I don't need to look at the result → Command type + trigger_heartbeat: false
  → Yes, but I need judgment only on anomalies → Command type + skip_pattern (normal pattern)
  → No (judgment varies by situation) → LLM type
  → Command execution + result interpretation every time → LLM type (include command in description)
```

### Examples

```markdown
## Morning Work Plan (LLM type)
schedule: 0 9 * * *
type: llm
Review yesterday's progress from episodes/ and plan today's tasks.

## Backup Execution (Command type — no follow-up needed)
schedule: 0 2 * * *
type: command
trigger_heartbeat: false
command: /usr/local/bin/backup.sh

## Health Check (Command type — skip on normal, LLM judges anomalies)
schedule: */15 * * * *
type: command
skip_pattern: ^OK$
command: /usr/local/bin/health-check.sh
```

→ Details: `operations/heartbeat-cron-guide.md`

---

## How to Route Tasks — submit_tasks vs delegate_task

There are two ways to move a task into execution.

| Aspect | `submit_tasks` | `delegate_task` |
|--------|---------------|----------------|
| **Who executes** | **Your own** TaskExec path | **Direct subordinate** |
| **When to use** | Want to async-execute a task yourself | Want to delegate to a subordinate |
| **DAG/parallel** | `parallel: true` for parallel, `depends_on` for dependencies | One task at a time |
| **Tracking** | task_queue.jsonl + Priming display | `task_tracker` |
| **Typical example** | Execute a task discovered during Heartbeat | Manager delegates work to subordinate |

**Decision flow:**
```
"Should a subordinate do this task?"
  → Yes, I have a direct subordinate → delegate_task
  → No, I'll do it myself → submit_tasks
  → I have no subordinates → submit_tasks
```

→ Details: `operations/task-management.md`, `anatomy/task-architecture.md`

---

## Team Design — Start Solo, Scale Up

### Why Separate Roles?

| Reason | Explanation |
|--------|-------------|
| Prevent context pollution | Giving one agent all phases bloats context and degrades judgment |
| Structural quality assurance | Separate the executor from the verifier (eliminate self-review blind spots) |
| Parallel execution | Independent roles run simultaneously for higher throughput |
| Deepen specialization | Role-specific checklists and memory yield higher quality than a generalist |

### Scaling Stages

| Scale | Composition | When to use |
|-------|------------|-------------|
| **Solo** | 1 Anima handles all roles | Small tasks, prototypes, just starting out |
| **Pair** | PdM + Engineer | Medium-sized routine tasks |
| **Full Team** | PdM + Engineer + Reviewer + Tester | Serious projects |
| **Scaled** | PdM + multiple Engineers + multiple Reviewers + Tester | Large-scale, multi-module |

### Rules of Thumb

- High cost of failure → More role separation
- "The implementer reviews their own code" → Split out a Reviewer
- Many modules can be worked in parallel → Add more Engineers

---

## Memory — 5 Types to Use

| Memory Type | Directory | In a word | Example |
|------------|-----------|-----------|---------|
| **Episodic** | `episodes/` | What you did when | "Investigated Slack API on 3/15" |
| **Semantic** | `knowledge/` | What you learned | "Slack API rate limit is 100/min" |
| **Procedural** | `procedures/` | How to do it | "Gmail auth setup steps" |
| **Skills** | `skills/` | Executable procedures | Image generation skill, research skill |
| **Short-term** | `shortterm/` | Recent context | Current conversation flow |

**Priming (automatic recall)** automatically retrieves relevant memories with each conversation or patrol and injects the needed context into the system prompt. You can also actively search with `search_memory`.

**Consolidation** extracts episodes from activity_log daily and distills them into knowledge. Unused memories are automatically organized by **Forgetting (active forgetting)**.

→ Details: `anatomy/memory-system.md`

---

## Cost Optimization — background_model and Activity Level

### background_model

Heartbeat / Inbox / Cron can run on a lighter model separate from the main model.

| Category | Model used | Target |
|----------|-----------|--------|
| foreground | Main model (e.g. claude-opus-4-6) | Chat (human conversation), TaskExec |
| background | background_model (e.g. claude-sonnet-4-6) | Heartbeat, Inbox, Cron |

Setting: `animaworks anima set-background-model {name} claude-sonnet-4-6`

### Activity Level

Adjust overall activity frequency from 10% to 400%. Directly affects Heartbeat interval.

| Activity Level | Heartbeat interval (base 30 min) | Use case |
|---------------|--------------------------------|----------|
| 200% | 15 min | Busy period, active development |
| 100% (default) | 30 min | Normal operations |
| 50% | 60 min | Low load, cost saving |
| 30% | 100 min | Night, weekends |

**Activity Schedule** can auto-switch by time of day (e.g. 9:00-22:00 at 100%, 22:00-6:00 at 30%).

→ Details: `reference/operations/model-guide.md`, `operations/heartbeat-cron-guide.md`

---

## Organization Basics — Supervisor, Subordinates, Peers

An Anima's hierarchy is determined by the `supervisor` field in `status.json`.

| Relationship | Definition | Communication |
|-------------|-----------|---------------|
| **Supervisor** | The Anima specified in `supervisor` | Progress reports (MUST), problem escalation |
| **Subordinates** | Animas whose `supervisor` is you | Delegate via `delegate_task`, monitor via `org_dashboard` |
| **Peers** | Animas with the same `supervisor` | Direct communication OK |
| **Other departments** | None of the above | Go through your supervisor (no direct contact in principle) |
| **Humans** | When `supervisor: null` (top-level) | Notify via `call_human` |

→ Details: `organization/hierarchy-rules.md`, `organization/roles.md`

---

## First Steps When Stuck

| Situation | What to do |
|-----------|-----------|
| Don't know how to do something | `search_memory(query="keyword", scope="common_knowledge")` |
| Task is blocked | See `troubleshooting/escalation-flowchart.md` |
| Tool isn't working | See `troubleshooting/common-issues.md` |
| Don't know what to do | Run Heartbeat checklist. Check current_state.md and task_queue |
| Unsure about a decision | Ask your supervisor with `send_message(intent="question")` |

→ Full document index: `common_knowledge/00_index.md`
