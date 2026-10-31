# AnimaWorks Essential Guide

[IMPORTANT] An integrated guide to grasp the full picture of AnimaWorks at a glance.
Covers the key points of Heartbeat / Cron / team design / memory / cost optimization.
Refer to this first when reading for the first time or when you want to organize the relationships between concepts.
See the links at the end for details on each topic.

---

## What is AnimaWorks

A framework for operating AI agents not as "tools" but as **autonomous personalities**.

- Each Anima has its own personality, memory, and judgment criteria
- Acts periodically on its own without human instruction (Heartbeat / Cron)
- Takes on roles within the organization and collaborates with other Anima and humans
- Learns from experience, accumulates memory, and grows

---

## The 5 Execution Paths — "When and How It Moves"

Anima operates through the following 5 paths. All except Chat start automatically.

| Path | When it runs | What it does | Who uses it |
|------|---------|---------|---------|
| **Chat** | When a human sends a message | Conversational response | Human → Anima |
| **Inbox** | When a DM arrives from another Anima | Immediate response to organization messages | Anima → Anima |
| **Heartbeat** | Periodic automatic startup (default 30 minutes) | Observation → planning → reflection. **Does not execute** | Automatic |
| **Cron** | Schedule of cron.md (e.g., every morning at 9:00) | Executes fixed tasks at scheduled times | Automatic |
| **TaskExec** | When a regular task is executable and its dependencies are complete | Executes with saved inputs via LLM attempts | Registered via submit_tasks or delegate_task |

Chat and Heartbeat run on **separate locks**, so it can respond immediately to human conversation even during patrols.

→ Details: `anatomy/what-is-anima.md`

---

## Heartbeat vs Cron — Two Autonomous Actions

Both are mechanisms that "run without human instruction," but their purposes are fundamentally different.

| Perspective | Heartbeat (periodic patrol) | Cron (scheduled task) |
|------|---------------------|------------------|
| **Analogy** | A security guard who patrols the office periodically | A newspaper delivery that arrives every morning at 9 |
| **Purpose** | Situation check, planning, reflection | Execution of assigned work |
| **Does it execute?** | **No**. If it finds a task, it only submits it via `submit_tasks` or `delegate_task` | **Yes**. LLM type thinks and executes; Command type executes immediately |
| **Interval** | Fixed interval (default 30 minutes, varies with Activity Level) | Flexibly specified with cron syntax (every day at 9:00, every Friday at 17:00, etc.) |
| **Configuration file** | `HEARTBEAT.md` (checklist) | `cron.md` (task definition) |
| **Typical examples** | Checking unread messages, detecting blockers, progress reflection | Morning work planning, weekly reports, running backups |

**If unsure**: "Just check and decide" → add to the Heartbeat checklist. "Do something at a fixed time" → define as a Cron task.

→ Details: `operations/heartbeat-cron-guide.md`

---

## Two Types of Cron — LLM type vs Command type

There are two types of Cron tasks, depending on whether thinking is required.

| Perspective | LLM type (`type: llm`) | Command type (`type: command`) |
|------|---------------------|---------------------------|
| **Analogy** | "Think about what should be prioritized today" | "Press this button every morning" |
| **Judgment** | Yes (output varies depending on the situation) | No (executes the same thing every time) |
| **API cost** | Yes (LLM call) | The command itself has none. However, **a follow-up LLM may run** (see below) |
| **Output** | Non-deterministic (varies by task) | Deterministic (command stdout) |
| **Suitable tasks** | Planning, reflection, writing, memory organization | Backups, sending notifications, data retrieval, health checks |

### Follow-up LLM for Command type (Important)

The Command type executes the command itself mechanically, but **if stdout has a return value, the LLM runs by default to analyze the result** (follow-up). In other words, it may not be completely zero-cost.

```
コマンド実行 → stdout あり？
  → なし → 終了（LLM 不要）
  → あり → skip_pattern にマッチする？
      → マッチ → 終了（LLM スキップ）
      → マッチしない → LLM が起動して結果を解釈・対処判断
```

Options to control this follow-up:
- **`trigger_heartbeat: false`** — Always skip the follow-up LLM (when result analysis is unnecessary)
- **`skip_pattern: <正規表現>`** — Skip only when stdout matches (ignore only in normal cases; let the LLM judge in abnormal cases)

### Criteria for choosing between them

```
「毎回同じことをするだけ？」
  → はい、結果も見なくてよい → Command型 + trigger_heartbeat: false
  → はい、ただし異常時だけ判断が必要 → Command型 + skip_pattern（正常パターン）
  → いいえ（状況に応じて判断が変わる） → LLM型
  → コマンド実行 + 毎回結果の解釈が必要 → LLM型（description にコマンド実行を指示）
```

### Example descriptions

```markdown
## 毎朝の業務計画（LLM型）
schedule: 0 9 * * *
type: llm
episodes/ から昨日の進捗を確認し、今日のタスクを計画する。

## バックアップ実行（Command型・follow-up不要）
schedule: 0 2 * * *
type: command
trigger_heartbeat: false
command: /usr/local/bin/backup.sh

## 監視チェック（Command型・正常時はスキップ、異常時はLLMが判断）
schedule: */15 * * * *
type: command
skip_pattern: ^OK$
command: /usr/local/bin/health-check.sh
```

→ Details: `operations/heartbeat-cron-guide.md`

---


## How to Flow Tasks — submit_tasks vs delegate_task

There are two ways to put tasks into execution.

| Perspective | `submit_tasks` | `delegate_task` |
|------|---------------|----------------|
| **Who executes** | **Your own** TaskExec path | **Direct subordinates** |
| **When to use** | When you want to asynchronously execute tasks you should do yourself | When you want to delegate to subordinates |
| **DAG/parallel** | Parallel via `parallel: true`, dependencies via `depends_on` | Delegate one at a time |
| **Progress tracking** | `list_tasks` / TaskBoard referencing the regular task store | Track via `task_tracker` |
| **Typical examples** | Executing tasks found in Heartbeat yourself | A supervisor delegating work to subordinates |

**Decision flow**:
```
「このタスクは部下がやるべき？」
  → はい、直属の部下がいる → delegate_task
  → いいえ、自分でやる → submit_tasks
  → 部下がいない → submit_tasks
```

→ Details: `operations/task-management.md`, `anatomy/task-architecture.md`

---

## Team Design — Start Solo and Scale

### Why split roles

| Reason | Description |
|------|------|
| Prevent context pollution | Having one person handle all steps bloats the context and degrades judgment accuracy |
| Structural quality assurance | Separate the executor from the validator (eliminate blind spots in self-review) |
| Parallel execution | Independent roles run concurrently to improve throughput |
| Deeper specialization | Role-specific checklists and memory deliver higher quality than general-purpose agents |

### Scaling stages

| Scale | Structure | When to use |
|------|------|---------|
| **Solo** | 1 Anima covers all roles | Small tasks, prototypes, just starting out |
| **Pair** | PdM + Engineer | Medium-sized routine tasks |
| **Full team** | PdM + Engineer + Reviewer + Tester | Full-scale projects |
| **Scaled** | PdM + multiple Engineers + multiple Reviewers + Tester | Large-scale, multiple modules |

### Guidelines for judgment

- High failure cost → increase role separation
- "The implementer is reviewing their own work" → separate the Reviewer
- Many modules that can be worked on in parallel → increase Engineers

---

## Memory — Using the 5 Types Properly

| Memory type | Directory | In one word | Example |
|-----------|------------|--------|-----|
| **Episodic memory** | `episodes/` | What was done when | "Researched the Slack API on 3/15" |
| **Semantic memory** | `knowledge/` | What was learned | "Slack API rate limit is 100 times/minute" |
| **Procedural memory** | `procedures/` | How to do it | "Gmail authentication setup procedure" |
| **Skills** | `skills/` | Executable procedure documents | Image generation skill, research skill |
| **Short-term memory** | `shortterm/` | Recent context | The flow of the current conversation |

**Priming (automatic recall)** automatically recalls relevant memories during each conversation or patrol and injects the necessary context into the system prompt. Additionally, you can actively search via `search_memory`.

**Consolidation** only episodicizes new activity chunks and does not generate them for re-executions with no changes. Automatic knowledge changes, weekly/monthly organization, and automatic skill learning are disabled by default. Memory storage and on-demand search remain.

→ Details: `anatomy/memory-system.md`

---

## Cost Optimization — background_model and Activity Level

### background_model

Heartbeat / Cron can use an explicitly configured background_model. Inbox uses the main model, and task-specific model specifications take precedence for that task. Do not automatically select based solely on cost.

| Category | Model Used | Target |
|----------|-----------|--------|
| foreground | Main model, or explicitly specified task-specific model | Chat, Inbox, TaskExec |
| background | Explicitly configured background_model; if not set, the main model | Heartbeat, Cron |

Configuration: `animaworks anima set-background-model {名前} claude-sonnet-5-5`

### Activity Level

The overall activity frequency can be adjusted from 10% to 400%. It directly affects the Heartbeat interval.

| Activity Level | Heartbeat interval (with 30-minute base) | Use case |
|---------------|-------------------------------|------|
| 200% | 15 minutes | Busy periods, active development |
| 100% (default) | 30 minutes | Normal operation |
| 50% | 60 minutes | Low load, cost savings |
| 30% | 100 minutes | Nighttime, holidays |

**Activity Schedule** also allows automatic switching by time period (e.g., 100% from 9:00-22:00, 30% from 22:00-6:00).

→ Details: `reference/operations/model-guide.md`, `operations/heartbeat-cron-guide.md`

---

## Organization Basics — Supervisors, Subordinates, Colleagues

Anima's hierarchy is determined by the `supervisor` field in `status.json`.

| Relationship | Definition | Communication |
|------|------|------------------|
| **Supervisor** | Anima specified in `supervisor` | Progress reports (MUST), issue escalation |
| **Subordinate** | Anima whose `supervisor` is you | Delegate via `delegate_task`, monitor via `org_dashboard` |
| **Colleague** | Anima with the same `supervisor` | Direct contact OK |
| **Other department** | None of the above | Via your supervisor (direct contact is generally prohibited) |
| **Human** | When `supervisor: null` (top level) | Notify via `call_human` |

→ Details: `organization/hierarchy-rules.md`, `organization/roles.md`

---

## First Steps When Stuck

| Situation | What to do |
|------|---------|
| Don't know how to operate | `search_memory(query="キーワード", scope="common_knowledge")` |
| Task is blocked | Refer to `troubleshooting/escalation-flowchart.md` |
| Tool is not working | Refer to `troubleshooting/common-issues.md` |
| Don't know what to do | Check current_state.md and `list_tasks`. If needed, refer to the configured Heartbeat checklist |
| Unsure how to decide | Consult your supervisor via `send_message(intent="question")` |

→ Full documentation index: `common_knowledge/00_index.md`
