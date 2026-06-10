# AnimaWorks Features

**[日本語版](features.ja.md)**

> Last updated: 2026-05-21
> See also: [Spec](spec.md) · [Memory System](memory.md) · [Brain Mapping](brain-mapping.md) · [Security](security.md) · [Vision](vision.md)

AnimaWorks is a framework that treats AI agents not as one-off tools, but as team members that keep working with memory.
Each agent (Anima) has its own identity, memory, and standards for judgment, and plays a role inside an organization. It does not only respond to human requests; it uses schedules, tasks, and memory to review the situation and choose its next action. Powerful models can handle complex judgment, while lightweight models handle routine work, letting the organization balance cost and quality.

---

## Autonomous Agents

The core of AnimaWorks is agents that move on their own.

Each Anima is defined by personality (`identity.md`), role and behavioral guidelines (`injection.md`), and domain expertise. This is not mere prompt templating — it is the immutable foundation that constitutes who the Anima *is*.

### Heartbeat (Periodic Patrol)

Runs automatically on a 30-minute cycle and executes three phases: **Observe → Plan → Reflect**. It observes the situation, plans, and reflects on its own actions. The heartbeat only plans; actual work runs on a separate path. You can set active hours so Animas rest outside business hours.

### Cron (Scheduled Tasks)

Define scheduled tasks in a Markdown file. Both LLM-type (execution with judgment) and Command-type (deterministic script execution) are supported. Operations such as “check Slack unread every morning at 9” or “run backups at 2 AM every day” are handled autonomously by the Anima itself.

### Bootstrap (First Launch)

When a new Anima is created, self-introduction and environment survey run automatically on first boot. This runs asynchronously in the background so it does not block other Animas.

---

## Brain-Inspired Memory System

Rather than stuffing information into the context window, memory is handled with mechanisms analogous to the human brain. For details, see [Memory System](memory.md) and [Brain Mapping](brain-mapping.md).

### Automatic Recall (Priming)

Depending on the message or trigger, multiple sources are searched **in parallel** and injected into the system prompt as a single “priming” block. The current implementation combines sender, recent activity, important knowledge, related knowledge, pending tasks, episodes, graph context, recent outbound history, and pending human-facing notifications.

| Source | Content |
|--------|---------|
| Sender profile | Past information and preferences about the other party (`shared/users/`) |
| Recent activity | Timeline from the unified activity log and shared channels |
| Important knowledge | Summary pointers for `[IMPORTANT]` knowledge |
| Related knowledge | RAG search over personal `knowledge/` and shared `common_knowledge/`, split by trust level |
| Task / execution state | TaskBoard, persistent task queue, running tasks, overflow inbox, and task result summaries |
| Episodes | RAG search over episodic memory; points to `read_memory_file` when details are needed |
| Graph context | Community context and recent facts from the memory backend |

Retrieved memories pass through the deterministic priming gate, which decides whether each item appears as body text, a pointer, evidence, or is suppressed. The system favors reading full files with `read_memory_file` only when details are actually needed.

Normal chat paths use **variable budgets by trigger** such as greeting, question, request, and heartbeat. `config.json` `priming.budget_*` and `heartbeat_context_pct` tune those caps and heartbeat expansion.

The Anima does not need to track what it remembers and what it forgot. The right memories are recalled when needed.

### RAG (Retrieval-Augmented Generation)

Combines vector search with ChromaDB and a multilingual embedding model with spreading activation over a knowledge graph (Personalized PageRank). Instead of simple keyword matching, related concepts are activated in chains.

ChromaDB and embedding work normally run through the vector worker. If a crash or index inconsistency is detected, RAG repair can quarantine the target Anima’s `vectordb` and rebuild the index from memory files.

### Memory Consolidation

Automatically extracts general knowledge from daily episodic memory. Failed experiences generate procedural memory (“how to do it right” runbooks). Procedures accumulate **failure counts and confidence**; items above a threshold may undergo **reconsolidation** (LLM-driven revision and versioning of runbooks). Daily, weekly, and monthly forgetting jobs and RAG reindex timing can be tuned via `ConsolidationConfig`. This implements the process of organizing memory while “asleep,” as in humans.

### Active Forgetting

Unused memories fade in three stages and are eventually removed: synaptic downscaling → neurogenesis reorganization → complete forgetting. Important memories such as procedures and skills are protected. The design favors a clear mind for sound judgment rather than unlimited accumulation.

### Memory Check Before Action

For side-effecting actions such as external sends, channel posts, human notifications, and memory writes, the action memory gate checks related `[ACTION-RULE]` items and whether required memories have been read. If needed context has not been loaded, execution stops first and prompts the Anima to read the relevant memory.

### Shared Knowledge (`common_knowledge`)

A shared knowledge base all Animas can reference: organizational rules, messaging guides, troubleshooting procedures, and more. From the moment a new Anima joins, it can access the organization’s common knowledge.

---

## Multi-Model, Multi-Provider Execution

AnimaWorks does not lock you to one model. Claude, GPT, Gemini, Mistral, local LLMs — place each where it fits best.

### Six Execution Modes

| Mode | Targets (examples) | Characteristics |
|------|-------------------|-----------------|
| **S** (SDK) | `claude-*` | Richest tool integration via Claude Agent SDK (includes MCP and Claude Code built-in tools) |
| **C** (Codex) | `codex/*` | Via Codex CLI; for OpenAI Codex–family models |
| **D** (Cursor Agent) | `cursor/*` | Cursor Agent CLI child process; MCP-integrated agent loop |
| **G** (Gemini CLI) | `gemini/*` | Gemini CLI child process; stream-json parsing and tool loop |
| **A** (Autonomous) | `openai/*`, `azure/*`, `bedrock/*`, `google/*`, `vertex_ai/*`, `mistral/*`, `xai/*`, `cohere/*`, `zai/*`, `minimax/*`, `moonshot/*`, `deepseek/deepseek-chat`, and **Ollama models with stable tool_use** (explicit patterns such as `ollama/qwen3.5*`, `ollama/qwen3:14b`, `ollama/glm-4.7*`, `ollama/llama4:*`, etc.) | Generic `tool_use` loop via LiteLLM |
| **B** (Basic) | Other `ollama/*` (e.g. `ollama/gemma3*`, smaller qwen3, `ollama/deepseek-r1*`, etc.) | One-shot execution with the framework handling memory I/O. No session chaining |

The mode is inferred automatically from the model name using **fnmatch wildcards** (**more specific patterns win**). Resolution order: `execution_mode` in `status.json` → `~/.animaworks/models.json` → deprecated `model_modes` in `config.json` → code default `DEFAULT_MODEL_MODE_PATTERNS`; if still unset, fall back to **B**.

Mode C (`codex/*`) can authenticate through Codex login, an OpenAI API key, or a dedicated Azure OpenAI credential with `type: "codex_azure"`. Azure-backed Codex still uses `codex/*` as the model family; `azure/*` remains Mode A through LiteLLM.

### Background Model

Background work such as heartbeat and cron can run on a separate, lighter model from the main one — e.g. Claude Opus for dialogue, Sonnet for patrols, GPT-4.1 mini or a local LLM for log monitoring — optimizing cost and quality per organization.

### Local LLM Support

vLLM and Ollama are integrated as OpenAI-compatible APIs. With a GPU server you can operate Animas without cloud APIs.

---

## Organization Structure and Hierarchy

Beyond running each Anima as an individual, **making them function as an organization** is a hallmark of AnimaWorks.

### Hierarchy Definition

A single `supervisor` field defines the org chart. Manager, subordinate, and peer relationships are computed automatically and injected into each Anima’s system prompt.

### Task Delegation

A manager Anima assigns tasks to subordinates via `delegate_task`. Tasks are enqueued for the subordinate, a DM notification is sent, and progress is tracked automatically. Managers can view the whole tree with `org_dashboard` and audit activity with `audit_subordinate`.

### Communication Routing Rules

Progress reports go to the manager; task delegation goes to direct reports; cross-department contact goes through the manager. Information flows with the same order as in a human organization.

### Subordinate Control

Manager Animas can pause/resume subordinates, change models, and restart processes via tools. Permission checks are validated automatically from the hierarchy.

---

## Messaging and Communication

All coordination between Animas uses asynchronous messaging. There is no shared memory or direct cross-reference.

### Internal Messaging

DMs between Animas (`send_message`) require an explicit `intent`. Only `report`, `delegation`, and `question` are allowed; greetings and thanks-only messages go to shared channels (Board). One topic, one round-trip is the rule; if it grows long, move to the Board.

### Shared Channels (Board)

Slack-style shared channels for announcements to all Animas, FYIs, and acknowledgments.

### External Service Integration

Messages from Slack (Socket Mode) and Chatwork (Webhook) are received automatically and delivered to the target Anima’s Inbox. @mentions are handled immediately; messages without mentions wait for the next heartbeat. Replies are routed back on the same channel automatically.

### Rate Limiting

Three layers to prevent infinite message loops: duplicate-send prevention to the same recipient, per-time caps (30/hour, 100/day), and behavioral awareness via priming injection of recent outbound history.

### Human Notification

`call_human` sends notifications to Slack, Chatwork, LINE, Telegram, or ntfy. The top-level Anima is the human-facing contact point.

---

## Task Management

### TaskBoard

TaskBoard is the work board for tasks, processing items, deferred work, suppressed items, background execution, and completed results. Chat and heartbeat priming surface only the tasks that matter in the current context, avoiding repeated display of already processed or suppressed results.

### Persistent Task Queue

The persistent task queue remains for compatibility and lightweight execution paths. When TaskBoard is unavailable, or in older paths, task-queue summaries are used as a fallback. Tasks originating from humans are always processed with highest priority.

### Staleness and Deadlines

If there is no update for 30 minutes, a task is marked stale; if the deadline passes, it is marked overdue. These surface through priming and prompt the Anima to respond.

### Parallel Execution

`submit_tasks` submits multiple tasks as a DAG. Dependencies are resolved and independent tasks run concurrently.

---

## Skills and Tools

### Built-in Tools

AnimaWorks-specific tools: memory (search, read/write, archive), procedure and knowledge metadata, channels (DM and Board), task queue (`backlog_task` / `update_task` / `list_tasks`), `submit_tasks`, notifications, skills, supervisor org tools, and (depending on configuration) credential **vault** and background-task inspection. **Schemas are filtered** by execution mode and trigger (chat, heartbeat, cron, consolidation jobs, etc.).

In **Mode S (Claude Agent SDK)**, besides Claude Code built-in tools, tools in the **`mcp__aw__*` namespace** are exposed via the stdio MCP server `core.mcp.server`. Names on MCP are a **curated subset** (four memory tools, `send_message` / `post_channel`, `call_human`, `delegate_task` / `submit_tasks` / `update_task`, `create_skill`). For example, **`backlog_task` and `list_tasks` are not on MCP** and are used through the full tool path (same handler as other modes). External services depend on `permissions.md` allowances and schema injection.

### External Tools

Besides Slack, Chatwork, Gmail, GitHub, AWS, web search, X search, and image generation, `core/tools/` includes modules for **Discord**, **Notion**, **Google Calendar**, **Google Tasks**, and more. The **`machine`** tool provides a path to launch **external agent CLIs** in an environment decoupled from AnimaWorks memory and messaging—by analogy, industrial machinery such as CNC rather than in-app tools.

Per-Anima allowance is controlled in `permissions.md`. Long-running tools (e.g. image generation) run asynchronously; results are confirmed on later priming or heartbeat via `state/background_notifications/` and **task_results**.

**Extensions**: In addition to bundled modules, placing `~/.animaworks/common_tools/*.py` (org-wide) or each Anima’s `tools/*.py` (personal) enables **dynamic dispatch** like core tools (common tool names that collide with core are skipped on load).

### Skill System

Skills use progressive disclosure. Required skills are surfaced through active skill context, the Skill Router, Skill Hub, or explicit `read_memory_file` calls. Full text loads only when the Anima decides it is needed, keeping cognitive load manageable even with many skills.

The Skill Hub manages installation, inspection, removal, quarantine, and shared availability. Successful procedures can be promoted into probation skills, and the curator reviews usage to suggest archive, merge, or cleanup actions. Animas can also create their own skills.

---

## Web UI

FastAPI under `server/` serves the API and static assets. The main app is a hash-routed single-page app (`#/chat`, etc.) with a left sidebar for navigation. It is protected by login sessions (some paths are public for embedding).

### Dashboard (`#/`)

List, status, and activity summary for Animas. A home widget reads the external tasks API (`/api/external-tasks`) for task lists from multiple sources (currently MVP / mock-heavy).

### Activity (`#/activity`)

Cross-Anima event feed. Filter by type to track heartbeats, DMs, tool use, channel posts, and more.

### Board (`#/board`)

Browse and post to shared channels in the browser.

### Chat (`#/chat`)

Real-time replies via SSE streaming. Past conversations load with infinite scroll. Supports image send/receive, multi-thread and multi-tab, and tool progress (Live Tool Activity).

**Meeting mode**: Pick multiple Animas (up to five) and a chair; messages aggregate into one thread via the meeting-room API (`server/routes/room.py`) and stream over SSE. Switch the UI from normal multi-tab chat to use it.

### Operations and Admin Screens

| Screen | Summary |
|--------|---------|
| **Setup** (`#/setup`) | First-run wizard |
| **Users** (`#/users`) | Human user profiles, aliases, etc. |
| **Anima management** (`#/animas`) | Create, enable/disable, models, org |
| **Process monitor** (`#/processes`) | Child processes and socket state |
| **Server** (`#/server`) | Server info and health |
| **Memory browser** (`#/memory`) | Browse memory files |
| **Logs** (`#/logs`) | Log viewer |
| **Assets** (`#/assets`) | Image / 3D pipelines, step-wise regeneration, uploads, expression variants (`server/routes/assets.py`) |
| **Activity report** (`#/activity-report`) | Cross-org activity by date with LLM-generated narrative (cache + SSE stream) |
| **Tool prompts** (`#/tool-prompts`) | Tool descriptions, dynamic guides, section edits, per-Anima system prompt preview |
| **AI brainstorm** (`#/brainstorm`) | Multi-persona discussion (realist, challenger, etc.) |
| **Team builder** (`#/team-builder` / `#/team-edit`) | Build and edit role layout and initial task ideas from industry × goal presets (`server/routes/team_presets.py`) |
| **Settings** (`#/settings`) | UI, locale, and related settings |

### Interactive Workspace (`/workspace/`)

A Three.js-based 3D office. Characters follow org layout; click to enter chat. Bust-up view supports AI-generated images and expression presets (`live2d.js` module). Includes subviews such as timeline playback and org dashboard. Opens in a separate tab from the main SPA.

### Voice Chat

Browser voice input for voice conversation with an Anima. STT (faster-whisper) transcribes; the same pipeline as text chat runs; TTS (VOICEVOX / Style-BERT-VITS2 / ElevenLabs) synthesizes the reply. WebSocket endpoint `ws://…/ws/voice/{anima_name}`. Per-Anima voice settings. Barge-in supported.

### Setup Wizard and Display Language

First-run language selection supports **17 languages** (`server/static/setup/steps/language.js`). Dashboard copy lives in `server/static/i18n/*.json`; the current build follows **Japanese, English, Korean**, and others, tied to `locale` in `config.json`.

### Responsive Design

Hamburger menu, collapsible sidebar, and layouts for desktop, tablet, and phone.

---

## Character Asset Generation

Automatically builds each Anima’s visual identity.

- **Image generation**: Pipeline integrating NovelAI and fal.ai (Flux). The LLM composes image prompts from character data. For realistic styles, a fal.ai API key alone is enough.
- **Vibe Transfer**: Subordinates automatically inherit the supervisor’s image style for consistent art across the org.
- **Expression variants**: Auto-generates emotion-based variations.
- **3D models**: Meshy-based 3D generation; GLB cache and compression optimize delivery.
- **Management UI**: From the web **Assets** screen, regenerate full-body, bust-up, icon, chibi, 3D, rigging, and other steps individually, or replace assets via manual upload.

---

## Security

Autonomous agents need safeguards that match their autonomy. See [Security Architecture](security.md).

### Prompt Injection Defense

All data carries a trust level (`trusted` / `medium` / `untrusted`). Data from external platforms is treated as `untrusted` even if relayed by a trusted Anima. Imperative text in external data is reference-only information — not executed as instructions.

### Command Blocking

Hardcoded blocks for destructive commands (`rm -rf /`, etc.) plus per-Anima `permissions.md` for allowed commands. Each segment of a pipeline is checked separately.

### Message Storm Defense

Conversation depth limits, praise-loop detection, and layered rate limiting prevent infinite messaging chains between Animas.

---

## Process Management

Each Anima runs as an independent child process with its own Unix socket. If one crashes, others are unaffected.

- **Automatic crash recovery**: Detects crashes and restarts automatically. Recovery context is injected on the next boot.
- **Reconciliation**: If an Anima that should be running is not, it is detected and started.
- **IPC**: Inter-process communication over Unix sockets with keep-alive. Supports large payloads and streaming.

---

## CLI

```
animaworks start / stop / restart          # Server control
animaworks init                            # Initial setup

animaworks anima list                      # List all
animaworks anima create --from-md FILE     # Create
animaworks anima info NAME                 # Details
animaworks anima enable / disable NAME     # Enable/disable
animaworks anima set-model NAME MODEL      # Change model
animaworks anima set-background-model NAME MODEL  # Background model
animaworks anima rename NAME NEWNAME       # Rename

animaworks models list                     # Supported models
animaworks models info MODEL               # Model info
animaworks index --anima NAME              # RAG index
animaworks repair-rag --anima NAME --full  # RAG quarantine / rebuild
animaworks memory status                   # Memory backend status
animaworks skills list                     # Skill Hub
animaworks task ...                        # TaskBoard / task execution
```

---

## Configuration

- **Two-layer merge**: Global settings (`config.json`) and per-Anima settings (`status.json`) merge automatically. Per-Anima settings always win.
- **models.json**: Wildcard patterns on model names define execution mode and context window.
- **Role templates**: Six roles — engineer, manager, writer, researcher, ops, general — with model, turn count, and chain count set per role.
- **Hot reload**: Changes are detected and apply on the next run. Many settings need no restart.

---

## Operations

- **Disk management**: Housekeeping rotates logs, short-term memory, and temp files automatically.
- **TaskBoard cleanup**: Daily housekeeping cleans up stale processing, deferred, suppressed, and background states.
- **Skill Curator**: Produces review proposals such as archive, merge, or cleanup based on skill usage.
- **Token usage tracking**: Measures and records input/output tokens for LLM calls.
- **API fault tolerance**: Automatic retries on LLM API failures.
- **Write safety**: Updates use temp file + rename for atomic writes and to avoid corruption on crash.
