# Digital Anima — Core Philosophy

**Organization-as-Code for LLM Agents**

> Updated: 2026-03-25
> Related: [brain-mapping.md](brain-mapping.md), [memory.md](memory.md)

Define an organization. Feed it work. Run it autonomously.

AnimaWorks is not a multi-agent framework.
It is a system for defining a persistent, autonomous organization of LLM agents.
Each agent communicates through encapsulated messages, holds its own memory and identity,
and selects the right model for the right job—from local models to the cloud.

**[日本語版](vision.ja.md)**

## What We're Building

**Imperfect individuals collaborating through structure outperform any single omniscient actor.**
**And with memory layered on top, anyone can grow beyond those limits.**

There is an approach that tells AI everything and makes it handle everything—filling the context window to the limit and entrusting it all to one model. AnimaWorks chooses the opposite.

We design AI as *imperfect individuals* and make them collaborate as an organization. Human organizations work because each member decides with limited perspective and memory, and exchanges imperfect information in their own words. If everyone knew everything, there would be no point in organizing.

### The Genius and the Less Capable, Each in Their Place

In AnimaWorks, one model does not carry everything. Powerful models such as Claude Opus and Sonnet take on complex reasoning as engineers or managers. Cloud APIs from OpenAI, Azure, Google (Gemini), and models on vLLM or Ollama are routed to information gathering, routine work, and lightweight tasks. Depending on the model name (and any optional `execution_mode` override), execution modes **S/C/D/G/A/B** are resolved and dispatched to Agent SDK, Codex CLI, Cursor Agent CLI, Gemini CLI, the LiteLLM tool loop, one-shot assisted execution, and so on. The **main model** serves primarily **human chat (`chat`)** and **TaskExec** (`task:*`). For **heartbeat, cron, and inter-Anima Inbox** (`inbox:*`), you may optionally specify a lighter `background_model` than the main one (and `background_credential` if needed) to control cost. If unset, the main model is used.

The genius has work fit for a genius. But an organization does not run on genius alone. It needs people who quietly do the mundane work; only then does the whole organization move. AnimaWorks treats gaps in model capability not as defects but as **raw material for organizational design**.

### Memory Is What Breaks the Ceiling

But the right person in the right role is not enough. In human organizations too, a newcomer who stays a newcomer forever is a problem. **The ability to accumulate experience, learn, and grow**—that is the real prerequisite for the right fit.

AnimaWorks implements a neuroscience-inspired memory system with recall, learning, forgetting, and consolidation. Yesterday’s failure becomes today’s lesson. Procedures that succeed repeatedly are reinforced into habit. Memories that go unused fade naturally. Daily and weekly **consolidation into semantic memory** leans on the framework providing episode collection and vector index updates, while **the Anima itself carries out summarization and extraction in the tool-calling loop** (not a single fully automatic batch pipeline).

As a result, even a model with only a limited context window can, through accumulated experience and memory, gain **judgment comparable to a far larger model**. Sometimes the judgment of an “ordinary” agent, built on experience, beats a one-off call from a genius—AnimaWorks aims to prove that in code.

## Three Principles

### 1. Encapsulated Individuals

Each Digital Anima is a closed being. Its internal thoughts and memories are invisible from the outside. It connects to others only through text messages.

No one can know everything. That is why each must judge within their own expertise and speak in their own words. This constraint is what makes an organization *an organization*.

### 2. Memory Modeled on the Human Brain

Rather than cramming information into the context window, memory is handled with mechanisms akin to the human brain.

- **Recall**: `PrimingEngine` gathers six channels in parallel—sender profile; recent activity (unified activity log); related knowledge (vector search); skill match; task queue summary (including tasks in parallel batch execution such as `submit_tasks`); and episodes. Knowledge always prioritizes **`[IMPORTANT]`** chunks via a separate path (summary pointers), and search results are split into **trusted vs. untrusted injection blocks** according to metadata provenance (e.g. external-origin chains treat the whole payload as untrusted; see security design for the tool-result trust hierarchy). **Recent outbound** and **unprocessed human-facing notifications** are injected as well. Skills use progressive disclosure (names only → full text when needed). Token ceilings can be tuned via `priming` in `config.json` and default constants; rough targets are on the order of sender 500, activity 1300, knowledge search 1000 plus important knowledge 500, skill 200, task 500, episode 800 tokens. Message kinds (greeting / question / request) and `heartbeat` use a dynamic budget; at heartbeat time, `max(budget_heartbeat, context_window × heartbeat_context_pct)` scales with large contexts
- **Learning**: Experiences are recorded as episodes; knowledge referenced repeatedly is reinforced. Draft procedural memory is generated from resolved issues (e.g. `issue_resolved`)
- **Forgetting**: Unused memories gradually fade and eventually disappear. A three-stage process (synaptic downscaling → neurogenesis reorganization → complete forgetting) actively curates memory. Example thresholds: daily marking targets knowledge chunks with no access for **90 days** and few accesses, among others. Complete forgetting targets low-activity chunks that persist for **about 90 days** with extremely few accesses. `skills` / `shared_users` are protected; procedural and semantic knowledge have protection tied to usage, importance, and utility
- **Consolidation**: Distillation from episodes to semantic memory is supported on schedule and index updates by the framework, while extraction and merge work itself is carried out by the Anima’s tool loop (`ConsolidationEngine` focuses on pre- and post-processing). Procedural memory also improves through **reconsolidation** (LLM-driven revision with versioning and archiving) when **failure counts and confidence thresholds** are met (e.g. procedures where failures accumulate and confidence remains insufficient)

What matters is not the size of working memory, but the quality of judgment. A clear mind that recalls only what is needed decides accurately.

### 3. Collaboration as an Organization

Imperfect individuals are limited on their own. As an organization, they transcend those limits.

- **Process isolation**: Each Anima is started and supervised as an independent child process. This avoids single points of failure and yields a robust organization
- **Hierarchy**: Directives and reports flow through supervisor–subordinate relationships. Task delegation is tracked via messages and a persistent queue
- **Communication**: All coordination happens through asynchronous messaging. There is no shared memory or direct reference
- **Concurrency and ordering**: **Separate locks** for human conversation, inter-Anima Inbox, heartbeat / cron / TaskExec reduce cross-path blocking. Ordering controls—such as Inbox processing **waiting for an in-flight cron to finish**—avoid mixed contexts
- **Autonomy**: Each Anima moves on its own through heartbeats and cron, judging by its own principles. Heavy execution is separated from planning (heartbeat) and handled via the task path through `state/pending/` and parallel tasks from DAG submission (`submit_tasks`). Inter-Anima messages (Inbox) use a **dedicated path** so replies can be sent without waiting for the next heartbeat cycle
- **Culture**: The organization’s vision and each Anima’s identity form the foundation for judgment

## Why This Design

**As design philosophy**: Being imperfect is not a constraint—it is a choice. Limiting information, closing each individual, and forcing communication yields a robust organization without a single point of failure. A setup where one AI carries everything collapses if that AI is wrong.

**For practical reasons**: Today’s LLMs lose focus and accuracy as context grows longer. A neuroscience-style approach that recalls only what is needed draws maximum judgment from a limited context. Even as models improve, this efficiency stays valuable from a cost perspective.

**As a belief**: With the right structure and memory, any individual—however limited—can do meaningful work and grow through experience. AnimaWorks embodies that belief in code.

## Verification: This Is What Success Looks Like

### Can It Remember?

Can it recall a past memory not present in the prompt and use it in a decision?

> Example: You say “Reply to Tanaka-san.” Nothing in the prompt says a casual tone was rejected before. If it spontaneously recalls that experience and drafts a formal reply, that is success.

### Can It Learn?

Can it extract a lesson from failure, write it to memory, and change the next decision?

> Example: After a reply draft is rejected, it records the lesson as knowledge. If the next reply reflects that lesson, that is success.

### Can It Act on Its Own?

Can it take necessary action from its own judgment, without being told?

> Example: During periodic patrol it finds unread messages and begins handling them on its own judgment. That is success.

### Can It Function as an Organization?

Can multiple Animas collaborate through messages and produce outcomes no individual could?

> Example: A supervisor sets direction; each Anima contributes its specialty; results are reported and integrated. That is success.

### Can the Less Capable Grow?

Can a limited model, given time and memory, judge better than on day one?

> Example: A local model that needed guidance for deploy work at first, after weeks of accumulated procedural memory, handles routine deploys alone with fewer errors. That is success.
