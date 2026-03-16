# AnimaWorks Features

**[日本語版](features.ja.md)**

> Last updated: 2026-03-06
> See also: [Spec](spec.md) · [Memory System](memory.md) · [Brain Mapping](brain-mapping.md) · [Security](security.md) · [Vision](vision.md)

AnimaWorks treats AI agents not as tools, but as **autonomous individuals**. Each agent (Anima) has its own identity, memory, and judgment. They take on roles within an organization and act on their own — without waiting for human instructions. Powerful models handle engineering; lightweight models handle routine work. Every model is placed where it fits best, and every one of them grows through accumulated experience.

---

## Autonomous Agents

At the heart of AnimaWorks is the idea that agents should act on their own.

Each Anima is defined by a personality (identity.md), a role with behavioral guidelines (injection.md), and domain expertise. These aren't just prompt templates — they form the immutable foundation of who the Anima *is*.

### Heartbeat (Periodic Patrol)

Every 30 minutes, each Anima wakes up and runs through three phases: **Observe → Plan → Reflect**. It surveys its surroundings, makes plans, and reviews its own actions. The heartbeat only plans — actual work is executed on a separate path. You can also set active hours so Animas rest outside business hours.

### Cron (Scheduled Tasks)

Define scheduled tasks in a Markdown file. Both LLM-type (requiring judgment) and Command-type (deterministic script execution) are supported. "Check Slack for unread messages every morning at 9" or "Run backups at 2 AM" — Animas handle these autonomously.

### Bootstrap (First Launch)

When a new Anima is created, it automatically introduces itself and surveys its environment on first boot. This runs asynchronously in the background, so it never blocks other Animas.

---

## Brain-Inspired Memory System

Rather than cramming information into the context window, AnimaWorks handles memory the way the human brain does. For details, see [Memory System](memory.md) and [Brain Mapping](brain-mapping.md).

### Automatic Recall (Priming)

When an Anima receives a message, six channels search for relevant memories in parallel and inject them into the system prompt.

| Channel | What it recalls |
|---------|----------------|
| Sender profile | Past interactions and preferences of the sender |
| Recent activity | What the Anima itself has been doing lately |
| Related knowledge | Learned knowledge relevant to the topic |
| Skill match | Skills that might be useful |
| Pending tasks | What needs to be done right now |
| Episodes | Similar past experiences |

Animas don't need to know what they remember and what they've forgotten. The right memories surface at the right time.

### RAG (Retrieval-Augmented Generation)

Vector search via ChromaDB with a multilingual embedding model, combined with spreading activation over a knowledge graph (Personalized PageRank). Instead of simple keyword matching, related concepts are activated in chains — the way association works in the brain.

### Memory Consolidation

General knowledge is automatically extracted from daily episodic memories. Failed experiences generate procedural memories — "here's how to do it right next time" runbooks. This is an implementation of what happens when humans organize memories during sleep.

### Active Forgetting

Unused memories fade in three stages: synaptic downscaling → neurogenesis reorganization → complete forgetting. Critical memories like procedures and skills are protected. The goal isn't to hoard every memory, but to maintain a clean mind that makes sharp decisions.

### Shared Knowledge (common_knowledge)

A shared knowledge base accessible to every Anima: organizational rules, messaging guides, troubleshooting procedures. The moment a new Anima joins the organization, it has access to institutional knowledge.

---

## Multi-Model, Multi-Provider Execution

AnimaWorks is model-agnostic. Claude, GPT, Gemini, Mistral, local LLMs — use whatever fits best.

### Four Execution Modes

| Mode | Target | Characteristics |
|------|--------|----------------|
| **S** (SDK) | Claude | Richest tool integration via Agent SDK |
| **A** (Autonomous) | GPT, Gemini, Mistral, local models, etc. | Generic tool_use loop via LiteLLM |
| **B** (Basic) | Lightweight / models without tool_use | One-shot execution; the framework handles memory I/O |
| **C** (Codex) | OpenAI Codex | Tool integration via OpenAI SDK |

The mode is auto-detected from the model name using wildcard patterns. Just write a model name in the config and the right mode is selected.

### Background Models

Background tasks like heartbeats and cron can run on a lighter model than the main one. Claude Opus for conversations, Sonnet for patrols, GPT-4.1 mini or a local LLM for log monitoring — optimize the cost-quality balance across the whole organization.

### Local LLM Support

vLLM and Ollama are integrated as OpenAI-compatible APIs. If you have a GPU server, you can run Animas without any cloud API calls.

---

## Organization Structure and Hierarchy

AnimaWorks doesn't just run agents as individuals — it makes them **function as an organization**.

### Hierarchy Definition

A single `supervisor` field defines the org chart. Manager, subordinate, and peer relationships are computed automatically and injected into each Anima's system prompt.

### Task Delegation

Manager Animas assign tasks to subordinates via `delegate_task`. The task is added to the subordinate's queue, a DM notification is sent, and progress is tracked automatically. Managers can view the entire org tree with `org_dashboard` and audit activity with `audit_subordinate`.

### Communication Routing Rules

Progress reports go to your manager. Task delegation goes to direct reports. Cross-department communication goes through your manager. Information flows with the same discipline as a human organization.

### Subordinate Control

Manager Animas can pause/resume subordinates, change their models, and restart their processes — all through tools. Permission checks are automatically verified based on the hierarchy.

---

## Messaging and Communication

All coordination between Animas happens through asynchronous messaging. No shared memory, no direct references.

### Internal Messaging

DMs between Animas (`send_message`) require an explicit intent: report, delegation, or question. Greetings and thank-yous go to shared channels (Board) instead. One topic, one round-trip is the rule — if it gets longer, move it to the Board.

### Shared Channels (Board)

Slack-style shared channels for announcements, FYIs, and acknowledgments across the organization.

### External Service Integration

Messages from Slack (Socket Mode) and Chatwork (Webhook) are automatically received and routed to the target Anima's inbox. @mentions and DMs are processed immediately; unmentioned messages wait for the next heartbeat. Replies are automatically routed back through the same channel.

### Rate Limiting

Three layers of protection against infinite message loops: duplicate send prevention per run, hourly/daily caps (30/hour, 100/day), and behavioral awareness through priming injection of recent outbound history.

### Human Notification

`call_human` sends notifications via Slack, Chatwork, LINE, Telegram, or ntfy. The top-level Anima serves as the point of contact with humans.

---

## Task Management

### Persistent Task Queue

Tasks are recorded in a persistent queue. Tasks from humans are always processed with highest priority.

### Staleness Detection and Deadlines

Tasks with no updates for 30 minutes are flagged as stale; overdue tasks are flagged as overdue. These flags surface through priming, prompting the Anima to take action.

### Parallel Execution

`submit_tasks` submits multiple tasks as a DAG. Dependencies are resolved and independent tasks run concurrently.

---

## Skills and Tools

### Built-in Tools

Memory operations, messaging, task management, skill search — AnimaWorks-native tools available across all execution modes. In Mode S, Claude Code built-in tools (file operations, git, Bash, etc.) and MCP tools are also integrated.

### External Tools

Slack, Chatwork, Gmail, GitHub, AWS, web search, X search, image generation, and more. Per-Anima permissions are controlled through `permissions.json`. Long-running tools (like image generation) execute asynchronously; results are picked up at the next heartbeat.

### Skill System

Skills are managed through progressive disclosure. During priming, only skill names are surfaced. Full skill text is loaded on demand when the Anima decides it's needed. This keeps cognitive load low even with a large skill library. Animas can also create their own skills.

---

## Web UI

### Dashboard

Overview of all Animas and their states. An activity timeline tracks every Anima's actions in real time.

### 3D Workspace

A Three.js-based 3D office space. Characters are arranged according to the org hierarchy. Click on one to start a conversation.

### Chat

Real-time responses via SSE streaming. Scroll back through conversation history with infinite scroll. Supports image send/receive, multi-thread, and multi-tab. Live Tool Activity shows tool execution progress in real time.

### Voice Chat

Speak into the browser and have a voice conversation with an Anima. Speech is transcribed via STT (faster-whisper), processed through the same chat pipeline, and synthesized back via TTS (VOICEVOX / Style-BERT-VITS2 / ElevenLabs). Each Anima can have its own voice settings. Barge-in (interruption) is supported.

### Setup Wizard

Complete the initial setup through a web browser on first launch. Supports 17 languages.

### Responsive Design

Works on desktop, tablet, and smartphone.

---

## Character Asset Generation

Automatically generate each Anima's visual identity.

- **Image generation**: Integrated pipeline with NovelAI and fal.ai (Flux). The LLM auto-composes image prompts from character information. For realistic styles, only a fal.ai API key is needed.
- **Vibe Transfer**: Automatically inherits the supervisor's art style to subordinates, maintaining visual consistency across the organization.
- **Expression variants**: Automatically generates emotion-based variations.
- **3D models**: 3D model generation via Meshy. GLB caching and compression optimize delivery.

---

## Security

Autonomous agents need safety mechanisms designed for autonomy. For details, see [Security Architecture](security.md).

### Prompt Injection Defense

Every piece of data is tagged with a trust level: trusted, medium, or untrusted. Data from external platforms is treated as untrusted even if relayed by a trusted Anima. Imperative text found in external data is treated as information only — never executed as instructions.

### Command Blocking

Hardcoded blocks for destructive commands (`rm -rf /`, etc.) plus per-Anima `permissions.json` for fine-grained control. Each segment of a pipeline is checked individually.

### Message Storm Defense

Multi-layered protection against infinite messaging chains: conversation depth limiters, praise-loop detection, and rate limiting.

---

## Process Management

Each Anima runs as an independent child process with its own Unix socket. If one crashes, the others are unaffected.

- **Automatic crash recovery**: Crashes are detected and the process is restarted automatically. Recovery context is injected on the next boot.
- **Reconciliation**: If an Anima that should be running isn't, it's detected and started automatically.
- **IPC**: Inter-process communication over Unix sockets with keep-alive. Handles large messages and streaming.

---

## CLI

```
animaworks start / stop / restart          # Server control
animaworks init                            # Initial setup

animaworks anima list                      # List all
animaworks anima create --from-md FILE     # Create from character sheet
animaworks anima info NAME                 # Show details
animaworks anima enable / disable NAME     # Enable/disable
animaworks anima set-model NAME MODEL      # Change model
animaworks anima rename NAME NEWNAME       # Rename

animaworks models list                     # List supported models
animaworks models info MODEL               # Model details
```

---

## Configuration

- **Two-layer merge**: Global settings (config.json) and per-Anima settings (status.json) are merged automatically. Per-Anima settings always take priority.
- **models.json**: Define execution modes and context windows per model using wildcard patterns.
- **Role templates**: Six built-in roles — engineer, manager, writer, researcher, ops, general — with preconfigured model, turn limits, and chain limits.
- **Hot reload**: Config changes are detected automatically and applied on the next run. Most settings can be changed without a restart.

---

## Operations

- **Disk management**: A housekeeping job automatically rotates logs, short-term memory, and temporary files.
- **Token usage tracking**: Input/output tokens for every LLM call are measured and recorded.
- **API fault tolerance**: Automatic retries on LLM API failures.
- **Write safety**: File updates use temp-file-plus-rename for atomicity, preventing data corruption on crashes.
