# AnimaWorks — Organization-as-Code

### Talk to your team like a messenger app. Behind the scenes, they ship software 24/7.

AnimaWorks is an open-source framework that brings together persistent AI agents (**Digital Animas**) — each with their own long-term memory, identity, role, and schedule — to operate 24/7 as a **living organization**.

On the surface, you interact with them through a familiar, **LINE-style messenger chat UI**.
Behind the scenes, agents running in isolated OS processes decompose goals, write code in parallel `git worktree` sandboxes, run tests and multi-model reviews on each other's work, open Pull Requests, autonomously repair CI failures, and monitor post-deploy logs without human intervention.

```text
A single message from you / GitHub webhook / Scheduled trigger (Heartbeat / Cron)
   ↓
Lead Anima decomposes & delegates (SQLite TaskStore, mutex keys, delegation contracts)
   ↓
Engineer Animas implement in isolated worktrees → Test → Multi-model PR review → CI self-repair → Deploy verification
   ↓
Only exceptions and approval gates escalate back to you via LINE-style chat, voice call, or notifications
```

AnimaWorks does not hardcode a rigid build pipeline. Instead, it provides **organizational infrastructure**: neuroscience-inspired long-term memory, process isolation, an atomic task board, permission & PII isolation (**Enclave**), and multi-channel communication. Because of that, the exact same organization can ship code, research competitors, handle customer inquiries, take meeting minutes, and coordinate over Slack or phone calls. It is not a workflow engine — **it is an organization**.

**[日本語 README](README_ja.md)** | **[简体中文 README](README_zh.md)** | **[한국어 README](README_ko.md)**

---

## Proven in Production

This is not a toy demo. For over six months, an 8-Anima organization powered by AnimaWorks has continuously developed, operated, and maintained a production SaaS product.

| Metric (Mar–Aug 2026, aggregated from primary execution logs) | Production Result |
|---|---|
| Pull Requests authored autonomously by agents | **302 PRs** (**267** merged) |
| Pull Requests operated by agents (review, CI repair, conflict resolution) | **752 PRs** (**721** merged) |
| Share of tasks self-initiated & executed by the organization | **99.7%** (out of 31,215 total tasks, only 92 were human-initiated) |
| Tasks automatically generated from GitHub events (August alone) | **2,508 tasks** |

<p align="center">
  <img src="docs/images/chat-memory.png" alt="AnimaWorks LINE-style Chat UI and Live Presence" width="760">
  <br><em>The redesigned LINE-style chat UI: monitor team status from the left Anima rail, chat with green/white bubbles, reply with inline quotes, approve release notifications, download attached reports, and watch the live bust-up react in the bottom-right corner.</em>
</p>

<p align="center">
  <img src="docs/images/workspace-dashboard.gif" alt="AnimaWorks Workspace — Real-time Org Tree and Activity Feed" width="760">
  <br><em>3D Workspace: real-time visualization of the org tree, each Anima's live status, and recent actions.</em>
</p>

<p align="center">
  <img src="docs/images/pixel-workspace.gif" alt="AnimaWorks Pixel Office — Live View of the Working Organization" width="760">
  <br><em>The Pixel Office is not a decorative animation — every speech bubble reflects a real task running right now in the background.</em>
</p>

---

## :rocket: Try It in 60 Seconds (Demo Environment)

**No API key required if you already have Claude Code CLI installed or are logged into Codex.**

Clone, set up with a one-liner, and launch a pre-populated 3-Anima organization (Product Manager `kaito`, Lead Engineer `sora`, Assistant `hina` in Japanese presets; `alex`, `kai`, `nova` in English presets) complete with 3 days of activity history. It runs in an isolated directory (`~/.animaworks-demo`) so it never touches your production `~/.animaworks` data.

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks demo
```

Open **http://localhost:18501** to start chatting with a team that is already at work ([Demo Guide →](demo/README.md)).

> **4 Built-in Presets**: `en-business` (English, realistic — default) / `en-anime` (English, anime) / `ja-business` (Japanese, realistic) / `ja-anime` (Japanese, anime). Pass `--reset` when switching presets (e.g., `uv run animaworks demo --reset --preset ja-anime`).

---

## Quick Start (Build Your Own Organization)

### macOS / Linux / WSL

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks start
```

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.ps1 | iex
cd animaworks
uv run animaworks start
```

> **Windows Note**: AnimaWorks uses Unix-specific APIs (`fcntl`, `pty`, `os.killpg`, etc.), so native Windows execution is not supported. Running `setup.ps1` in PowerShell automatically installs WSL and sets up AnimaWorks inside your WSL Linux environment.

Open **http://localhost:18500** in your browser and follow the setup wizard:
1. **Language & Theme** (`en` / `ja` / `zh` / `ko`)
2. **LLM Connection** (Claude Code subscription login, Codex login, API keys, or local Ollama)
3. **Create Your First Anima** (choose from templates such as `dev-lead`, `full-stack`, `assistant`, `marketer`, `translator`, or create a custom role)

You can also initialize and operate everything headlessly from the CLI without a browser:

```bash
uv run animaworks init
uv run animaworks start
uv run animaworks chat alice "Check our sprint goals for today and delegate tasks to the team"
```

<details>
<summary><strong>Docker Compose / Manual pip Installation</strong></summary>

#### Docker (Zero host pollution — recommended)

`docker compose up` launches an isolated Linux container with Node.js, GitHub CLI (`gh`), Playwright, Bubblewrap, and Claude Code / Codex / Gemini CLIs pre-installed.

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
# Configure credentials in .env (ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN / GH_TOKEN, etc.)
docker compose up -d --build
```

Headless initialization without the browser wizard:

```bash
docker exec -it <container> animaworks init --skip-anima
docker exec -it <container> animaworks anima create --name alice --template dev-lead
docker exec -it <container> animaworks config set setup_complete true
docker exec -it <container> animaworks send <your-name> alice "Hello!"
```

#### Manual pip Installation (Python 3.12 / 3.13 recommended)

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[all-tools]"
animaworks start
```

</details>

---

## On the Surface: A Team You Talk to Like a Messenger App

Many autonomous agent frameworks force humans to stare at raw log streams or wire up complex node graphs. AnimaWorks flips that model: **day-to-day communication feels as effortless as LINE or Slack, while deep inspection of memories and background work is one click away**.

### 1. LINE-Style Talk Screen, Left Anima Rail & Live Presence Panel
- **Intuitive Bubble Timeline**: Your messages appear in green bubbles on the right; Anima responses appear in white bubbles on the left, with smooth streaming previews of thoughts and tool executions.
- **Left Anima Rail**: Member avatars line the left edge, showing at a glance who is actively working (blue ring) and who is waiting for your reply (amber badge). Dismiss inactive chats with `×`, re-open via `+`, split the screen into up to 4 panes, or launch a **Meeting Mode** (moderated group discussion with up to 5 Animas).
- **Bottom-Right Presence Panel**: A live bust-up portrait sits in the bottom-right corner of the chat, shifting expressions in real time (`smile`, `thinking`, `troubled`, `surprised`, etc.) and displaying a thought bubble of what the Anima is currently doing.

### 2. Human-Centered Feed & One-Tap Approvals / Reply Quotes
- **Noise-Free Conversation Feed**: In a 24/7 organization, hundreds of Heartbeat patrols, inter-Anima DMs, and Cron jobs happen behind the scenes. The chat UI automatically collapses background activity into **compact single-line bundles** so your conversation stays clean (toggle `Background` in the header anytime to expand everything).
- **Approval Cards & Reply Quotes**: When an Anima needs human sign-off (`call_human`), an interactive card with **Approve / Reject / Comment** buttons appears directly in chat. You can also quote-reply to any message or notification (`Reply Chip`) so the Anima knows the exact context of your instruction.

### 3. Bidirectional File Attachments, Inline Viewer & Slack Mirroring
- **Drag & Drop Any File**: Attach images, PDFs, Markdown, CSVs, source code, Office documents, audio, or video directly into chat.
- **Outbound File Cards from Animas (`attach_file`)**: Animas attach generated reports, spreadsheets, and images directly to their replies. Click any attachment card to open the built-in **Fullscreen File Viewer** (sanitized Markdown preview, syntax-highlighted code, audio/video player).
- **Slack Conversation Mirroring**: Messages exchanged with an Anima on Slack while you are away are automatically mirrored into the Web chat timeline.

### 4. One-Tap Memory Popover & Fullscreen Voice Call
- **D3 Entity-Connection Bubble Graph**: Click an Anima's face icon or the `Memory` button in the chat header to open an anchored popover into their mind. Explore the interactive **D3 entity-connections bubble chart** (`Connections`), current work state (`State`), knowledge & procedures (`Files`), daily episodes (`Journal`), tool history (`Activity`), and scheduled routines (`Routines`).
- **Fullscreen Voice Call (`Call Screen`)**: Click the phone icon to enter a fullscreen voice call powered by a low-latency front-lane model, VOICEVOX / Style-BERT-VITS2 / ElevenLabs / Irodori TTS, and streaming `faster-whisper` STT — complete with lip-synced, blinking bust-up animation and live captions. For critical production incidents, Twilio integration (`core/phone/`) even lets an **Anima place an actual phone call to your mobile phone**.

<p align="center">
  <img src="docs/images/memory-popover.png" alt="AnimaWorks Memory Popover — D3 Entity Connections Bubble Graph" width="760">
  <br>
  <em>The Memory Popover opens with one click from the chat header or avatar, visualizing atomic facts, knowledge, procedures, journals, and an interactive D3 entity-connection graph.</em>
</p>

<p align="center">
  <img src="docs/images/dashboard.png" alt="AnimaWorks Dashboard — Live Org Chart and Task Status" width="760">
  <br>
  <em>Dashboard: oversee the live org chart, items needing attention, model quota consumption, process health, and real-time activity across the entire organization.</em>
</p>

---

## Under the Hood: A 24/7 Autonomous Organization Engine

Behind the friendly chat UI is a battle-tested multi-process runtime designed to run continuously for months without human hand-holding.

```text
FastAPI Server (Thin Control Plane / WebSocket & SSE / Enclave Client)
  ├── MemoryService Process (Dedicated ChromaDB + Embeddings HTTP owner, SQLite WAL)
  ├── Anima Root Process: kaito (Manager)
  │     ├── Front Lane (Low-latency response / Voice / Routing)
  │     ├── Main Chat / Inbox / Heartbeat / Cron Lanes
  │     └── Task Runner Subprocesses (IPC v2 — isolated execution per heavy task)
  ├── Anima Root Process: sora (Lead Engineer)
  └── Anima Root Process: hina (Assistant)
```

### 1. Pick Up Right Where You Left Off — Multi-Layer Memory & Autonomous Maintenance
Unlike stateless chat wrappers that reset every session, each Anima has a filesystem-first, brain-inspired memory architecture:
- **6-Channel Automatic Priming (`core/memory/priming/`)**: Before every turn, the runtime automatically recalls (A) Sender profile, (B) Recent activity, (C) Related knowledge, (E) Pending tasks, (F) Episodes, and (G) Graph/community/recent atomic facts within a dynamic token budget.
- **Hybrid Retrieval Pipeline (`core/memory/retrieval/`)**: Combines Vector search (ChromaDB) + BM25 lexical search + Atomic Facts (`FactRecord`) + Entity Registry + RRF fusion + confidence gating. Select `legacy` (default) or **Neo4j Graph Memory** (`neo4j`) globally or per Anima.
- **Sleep-Cycle Consolidation & Active Forgetting (`core/memory/maintenance/`)**: Every night, Animas consolidate raw daily logs into structured knowledge and prune stale, low-value memories while strictly protecting identity, procedures, and active tasks.

### 2. Act on Their Own Clock — Heartbeat, Cron, SQLite TaskStore & GitHub Gateway
- **Heartbeat & Cron**: Periodic autonomous patrols (`HEARTBEAT.md` with adaptive intervals and quiet hours) and exact-time schedules (`cron.md`).
- **SQLite TaskStore (`shared/taskboard.sqlite3`)**: An atomic, organization-wide task board supporting dependencies, mutex keys (`mutex_key` to prevent two Animas from editing the same resource simultaneously), delegation contracts (`delegated_tasks` with automatic supervisor review on completion), and automatic retry/recovery for stalled workers.
- **GitHub Webhook Gateway**: Automatically turns CI failures, PR review comments, and merge conflicts into actionable tasks routed to the responsible engineer Anima.

### 3. Fault-Tolerant Process Isolation — Phase 3 Process Tree & IPC v2
- **Per-Anima Root Process + Task Runner Subprocesses**: Each Anima runs in its own OS process supervised by `ProcessSupervisor`, and heavy background/delegated tasks spawn isolated `TaskRunner` subprocesses over IPC v2 (`STDIN/STDOUT` JSON-RPC + binary framing). If a tool hangs or crashes, only that subprocess is reaped — the server and other Animas never go down.
- **Dedicated MemoryService Process**: Vector store and embedding models live in a single owner process (`HttpVectorStore` over localhost HTTP), eliminating multi-process ChromaDB corruption and duplicating model weights in RAM.

### 4. Hardware-Grade PII & Secret Isolation — Enclave & Trust Boundaries
- **Enclave (`core/enclave/`)**: Keeps raw API credentials, cookies, and customer PII inside a separate OS user / privilege domain reachable only over a Unix domain socket with a strict operation allowlist (`gmail_list`, `gmail_send`, `git_push`, `sql_query`, etc.). Even if an Anima executes arbitrary Bash commands, it cannot read the raw secrets, and outbound responses pass through **Egress PII scrubbing** (automatic redaction of emails, phone numbers, card numbers, and custom regexes).
- **Multi-Layer Permission Gate (`ToolHandler`)**: Enforces global `permissions.global.json`, per-Anima `permissions.json`, dangerous command pattern blocking, and prompt-injection trust boundaries (`trusted` vs. `external`).

### 5. Multi-Engine Execution & Hybrid Routing
Assign the optimal engine and model to each Anima's role — and use a cheaper `background_model` for Heartbeat/Cron patrols while reserving frontier models for complex coding and user chat:

| Mode | Execution Engine | Primary Models | Capabilities & Tooling |
|---|---|---|---|
| **S** | Claude Agent SDK | `claude-*` (Recommended) | Built-in Claude Code tools (Read/Write/Edit/Bash/Grep/Glob) + **stdio MCP (`mcp__aw__*`)** for AnimaWorks tools. Supports Claude Code subscription login |
| **C** | Codex SDK / CLI | `codex/*`, `openai-codex/*` | Codex sandbox execution + AnimaWorks MCP integration. Supports `codex login` |
| **D** | Cursor Agent CLI | `cursor/*` | Cursor CLI session & MCP-integrated agent loop |
| **G** | Gemini CLI | `gemini/*` | Google Gemini CLI stream-JSON & tool loop integration |
| **X** | Grok Build CLI | `grok/*` | Grok Build agent loop over ACP stdio |
| **A** | LiteLLM Autonomous | GPT, Gemini, DeepSeek, Bedrock, Vertex, Azure, Ollama, vLLM, etc. | CC-compatible tools + memory, tasks, and skill creation across any API or local LLM |

<p align="center">
  <img src="docs/images/asset-management.png" alt="AnimaWorks Asset Management — Character Assets and Expression Variants" width="760">
  <br><em>Asset Management: generate full-body art, 7 expression bust-ups, chibi sprites, pixel-art sheets, and rigged 3D models from personality settings — including Vibe Transfer to inherit a supervisor's visual style.</em>
</p>

---

## How AnimaWorks Compares

| Dimension | AnimaWorks | CrewAI | LangGraph | OpenClaw | OpenAI Agents SDK |
|---|---|---|---|---|---|
| **Core Unit & Philosophy** | **Persistent "Organization" of autonomous agents** | Role-based task crew | State-graph workflow | Personal AI assistant | Lightweight multi-agent SDK |
| **Human UI Experience** | **LINE-style chat + Live bust-up + D3 memory graph + 3D/Pixel office + Voice call** | CLI / Basic logs | LangSmith Studio (dev-focused) | Chat UI | None (code only) |
| **Long-Term Memory** | **Neuroscience-based**: 6-ch priming, Vector+BM25+Facts+Neo4j, nightly consolidation & active forgetting | Cognitive Memory | Checkpointer + Store | SuperMemory | Session-only |
| **Autonomous Triggers** | **Heartbeat + Cron + TaskStore + GitHub webhook gateway (24/7 operation)** | Human-triggered | Human/API-triggered | Cron + Heartbeat | Human/API-triggered |
| **Process & Fault Isolation** | **Per-Anima root process + isolated task runners (IPC v2), auto-restart, RAG self-repair** | Single shared process | Single shared process | Single process | Single shared process |
| **Secret & PII Isolation** | **Enclave (separate OS user + Unix socket + Egress PII scrubbing + trust boundaries)** | None | None | None | Guardrails only |
| **Multi-Model Execution** | **7 engines (Claude SDK / Codex / Cursor / Gemini CLI / Grok / LiteLLM) + cross-provider rate guard** | LiteLLM | LangChain models | OpenAI-compatible | OpenAI-centric |

---

## Why AnimaWorks

**One person alone can do nothing. That's why we built an organization.**

This project was born at the intersection of three careers.

**As a business owner** — I know that "one person alone can do nothing." You need excellent engineers, and you need staff who are good at communication. There are workers who toil quietly, and there are people who occasionally come up with sharp ideas. Genius alone doesn't keep an organization running. When diverse strengths come together, things that could never be achieved alone are achieved.

**As a psychiatrist** — When I observed the internal structure of LLMs, I noticed a structure surprisingly similar to the human brain. Recall, learning, forgetting, consolidation — if I could implement the brain's memory-processing mechanisms directly as an LLM's memory system, I might be able to reproduce the human brain. And if I could treat an LLM as a "pseudo-human," I should be able to build an organization just like with humans.

**As an engineer** — I've been writing code for 30 years. I know the joy of building logic and the thrill of automation. If I can pack all my ideals into code, I can build my ideal organization.

There are already many excellent frameworks for a "standalone AI secretary." But I felt there were still few projects that create human-like units in code and make them function as an organization. AnimaWorks is an AI organization that I've built into my own business and nurture through daily use.

> *The collaboration of imperfect individuals creates a more robust organization than a single omnipotent one.*

Three principles support this:

- **Encapsulation** — Internal thoughts and memories are invisible from the outside. Connection with others happens only through text conversation. Just like a real organization.
- **RAG memory (archive-based)** — We don't cram everything into the window. Priming picks up relevant chunks via RAG, and agents recall on their own using `search_memory` etc.
- **Autonomy** — They don't wait for instructions. They operate on their own clock and judge by their own values.

---

<details>
<summary><strong>API Keys, Authentication & External Integrations Reference</strong></summary>

#### LLM Providers & CLI Subscriptions

| Auth Method / Key | Service | Execution Mode | Notes |
|---|---|---|---|
| `claude` CLI login | Claude Code subscription | S | No API key required. Select in Setup Wizard or Settings |
| `codex login` | OpenAI Codex CLI | C | No API key required. Works with ChatGPT Plus/Pro/Team |
| `ANTHROPIC_API_KEY` | Anthropic API | S / A | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | OpenAI API | A / C | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | Google AI (Gemini) | A / G | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `grok login` | Grok Build CLI | X | Runs `grok/*` models over ACP stdio |
| `cursor-agent` login | Cursor Agent CLI | D | Runs `cursor/*` models |
| `OLLAMA_SERVERS` | Ollama / vLLM local LLMs | A | No API key required (default: `http://localhost:11434`) |

- **Azure OpenAI / AWS Bedrock / Google Vertex AI** are also supported via `credentials` in `config.json`.
- Credentials resolve in priority order: `config.json` → encrypted Vault (`animaworks vault`) → shared credentials → environment variables.

#### Image & 3D Generation (Optional)

| Key / Setting | Service | Purpose |
|---|---|---|
| `NOVELAI_TOKEN` | NovelAI | Anime-style avatars, expression variants, Vibe Transfer |
| `FAL_KEY` | fal.ai (Flux) | Photorealistic / stylized avatar generation |
| Codex Image / Local Diffusers | OpenAI / Local | Full-body, bust-up, and pixel-art sprite sheet (`pixel_sheet.png`) generation |
| `MESHY_API_KEY` | Meshy | Rigged 3D character model generation (`.glb`) |

#### Voice Call & Emergency Phone Alerts (Optional)

| Component | Supported Engines / Services |
|---|---|
| **STT (Speech-to-Text)** | `faster-whisper` (LocalAgreement-2 streaming confirmation, `pip install animaworks[transcribe]`) |
| **TTS (Text-to-Speech)** | VOICEVOX (`:50021`) / Style-BERT-VITS2 & AivisSpeech (`:5000`) / Irodori (`:7861`) / ElevenLabs API |
| **Emergency Phone Call** | Twilio Voice (`core/phone/` — dials a human's phone number and speaks critical incident alerts) |

#### External Messaging & SaaS Integrations (Optional)

| Service | Capabilities |
|---|---|
| **GitHub** | Webhook gateway (auto-tasks for CI failures, PR reviews, conflicts) + per-Anima `gh` identity |
| **Slack** | Socket Mode bidirectional chat, shared Board sync, **Slack-to-Web chat mirroring**, human notifications |
| **Discord / Chatwork** | Gateway / Webhook ingestion, bidirectional messaging, human notifications |
| **Zoom RTMS** | Real-time meeting audio/transcript ingestion, automatic minutes & action item extraction |
| **Google Workspace / Notion** | Read/write Gmail, Google Calendar, Google Sheets, Google Tasks, and Notion pages |
| **Notification Channels** | Slack / Discord / Chatwork / LINE / Telegram / ntfy / Twilio voice call |

</details>

<details>
<summary><strong>Org Hierarchy, Role Templates & Multi-Company Management</strong></summary>

A single `supervisor` field in `status.json` defines reporting lines (supervisors and direct reports). You can even ask a lead Anima in chat to "hire a member with these skills," and the lead Anima will design the appropriate role, personality, and permissions to spawn a new Anima.

| Role Template | Primary Responsibility | Automatically Granted Capabilities |
|---|---|---|
| `manager` | Decision-making, task decomposition & delegation, progress auditing | Supervisor tools (`delegate_task`, `org_dashboard`, inspect/restart/hire subordinates) |
| `engineer` | Code implementation, testing, CI self-repair, PR creation | Isolated worktree/scratch dev conventions, Git/GitHub integration |
| `researcher` | Technical research, competitor analysis, documentation review | Web search, knowledge structuring, report generation |
| `writer` | Content creation, documentation & release note authoring | Editorial quality control, outbound file attachments |
| `ops` | Log monitoring, routine patrols, infrastructure health checks | Low-cost continuous monitoring via lightweight or local LLMs |
| `general` | Assistant, scheduling, frontline coordination | Cross-team communication hub, meeting minutes organization |

In addition, the `animaworks company` command suite lets you define **multiple companies or business units** within a single runtime, complete with channel scope isolation, company splitting (`split`), and export (`export`).

</details>

<details>
<summary><strong>Multi-Layer Security & Trust Boundaries</strong></summary>

Because autonomous agents are given shell and external tool access, AnimaWorks implements defense-in-depth rather than relying on wishful thinking (see [Security](docs/en/security.md) / [Enclave](docs/en/enclave.md)):

1. **Enclave (PII & Production Data Isolation Runtime)**: Locks personal data and production DB/S3 credentials inside a separate OS user domain, returning only safe, Egress-masked facts over a Unix domain socket (`SO_PEERCRED`).
2. **Provenance & Trust Boundary Labeling**: External inputs from web search, Slack, email, etc. are tagged as `untrusted`, propagating the minimum session trust level and tracking RAG metadata provenance to prevent privilege escalation via prompt injection.
3. **5-Stage Command & Path Inspection**: Shell injection detection → Global blocklist (server refuses to start if `permissions.global.json` is missing) → Per-Anima blocked commands → Per-Anima allowlist → Path traversal & symlink escape prevention (including CWD-relative checks).
4. **Filesystem Sandbox & Action Gates**: Each Anima can write only within allowed `file_roots`, with self-modification protection on its own `IDENTITY.md` / `SOUL.md` and `permissions.json`. High-impact external tools are guarded by pre-execution Action Gates.
5. **Authentication, SSRF Prevention & Webhook Verification**: Argon2id password hashing, 48-byte session tokens, HMAC signature verification with replay protection for Slack/Chatwork/Zoom/GitHub, and a media proxy that blocks private IPs and DNS rebinding.

</details>

<details>
<summary><strong>CLI Command Reference</strong></summary>

While everyday chat and monitoring happen in the Web UI, AnimaWorks provides a comprehensive CLI (`animaworks` and the internal `animaworks-tool`) for automation and operations (full details in [CLI Reference](docs/en/reference/cli.md)):

| Category | Key Commands | Description |
|---|---|---|
| **Server / Demo** | `animaworks start [-f] [--port 18500]`<br>`animaworks stop` / `restart`<br>`animaworks demo [--preset en-business] [--reset]` | Start, stop, or restart the server<br>Launch a 3-Anima demo organization with 3 days of history (default port 18501) |
| **Anima Management** | `animaworks anima create / list / info / status`<br>`animaworks anima set-model / set-background-model / set-role`<br>`animaworks anima restart / reload / enable / disable`<br>`animaworks anima audit / permissions / repair-bootstrap` | Create Animas, inspect status, switch models, hot-reload prompts, audit permissions |
| **Task Management** | `animaworks task board / list / show / add`<br>`animaworks task claim / release / done / cancel / note`<br>`animaworks task autonomy [--days N]` | Operate SQLite TaskStore, manage leases, and generate **organization autonomy rate reports** |
| **Chat & Messaging** | `animaworks chat ANIMA "message"`<br>`animaworks send FROM TO "message"`<br>`animaworks board read / post / dm-history`<br>`animaworks heartbeat ANIMA` | Chat from CLI, send inter-Anima DMs, read/post shared Boards, trigger manual Heartbeats |
| **Memory, RAG & Skills** | `animaworks memory status / migrate / backup / cleanup`<br>`animaworks index [--anima NAME] [--full]`<br>`animaworks repair-rag --anima NAME --full`<br>`animaworks skills list / install / inspect / remove` | Manage memory backends (`legacy` / `neo4j`), rebuild RAG indexes, manage Skill Hub |
| **Enclave, Vault & Org** | `animaworks enclave doctor / status / audit`<br>`animaworks vault init / store / get / list`<br>`animaworks company create / list / assign / split / export`<br>`animaworks cost [--today]` / `models list` | Inspect Enclave runtime, manage encrypted credentials, manage companies, track LLM costs |

</details>

<details>
<summary><strong>Tech Stack & Repository Layout</strong></summary>

```text
animaworks/
├── main.py                  # CLI entrypoint
├── core/                    # Digital Anima core framework
│   ├── anima/               # DigitalAnima facade, lifecycle, Heartbeat, Inbox, attachment management
│   ├── agent/               # AgentCore, conversation cycle, Executor factory, pre-context builder
│   ├── runtime/             # Anima root process, task runner supervisor, IPC / IPC v2, streaming
│   ├── execution/           # 7 execution engines (claude, codex, cursor, gemini, grok, litellm) + watchdog & session store
│   ├── memory/              # Long-term memory (priming, retrieval, rag, facts/connections, consolidation, activity)
│   ├── tasks/               # SQLite TaskStore (board, attempts, leases, wakeups, delegation) & external task collectors
│   ├── enclave/             # Secret & PII isolation runtime (gateway, egress masker, raw_store, SSM tunnel, guards)
│   ├── tooling/             # ToolHandler, permission policies, action gates, MCP/external tool dispatch
│   ├── integrations/        # Auto-discovered external tools (GitHub, Slack, Discord, Notion, Google, image gen, etc.)
│   ├── mcp/                 # stdio MCP server for Mode S & external clients (mcp__aw__*)
│   ├── skills/              # Skill Hub, router, curator, procedure-to-skill promotion
│   ├── prompt/              # System prompt, org context & tool guide assembly
│   ├── messaging/           # Inter-Anima DM, shared channels & external outbound routing
│   ├── notification/        # Human notifications (call_human), inline approvals & reply routing
│   ├── voice/               # Voice conversation (faster-whisper STT + multi-engine TTS + low-latency front lane)
│   ├── phone/               # Twilio emergency voice call alerts
│   ├── org/                 # Multi-company, org hierarchy & workspace management
│   ├── config/              # Pydantic config schemas, model resolution & global permission cache
│   ├── auth/                # Argon2id authentication & session management
│   └── i18n/                # Multilingual message catalogs (t())
├── server/                  # FastAPI server + WebSocket/SSE + gateways + Web UI
│   ├── app.py               # ASGI app, lifespan, readiness gates & base-path control
│   ├── supervisor/          # ProcessSupervisor, health checks, exponential backoff restart & auto-update
│   ├── gateways/            # GitHub Webhook, Slack Socket Mode, Discord Gateway, Zoom RTMS
│   ├── routes/              # REST / WebSocket / SSE routes (chat, sessions, memory, assets, attachment_preview, etc.)
│   └── static/              # Web UI (LINE-style Chat, left Rail, Presence, D3 Memory Popover, Call Screen, 3D/Pixel Workspace)
├── cli/                     # animaworks / animaworks-tool CLI implementation, demo runner & TUI
├── templates/               # Locale-specific (ja / en / ko) initial prompts, role conventions & systemd templates
└── demo/                    # 4 presets (ja/en × business/anime) & bundled 3-day activity history
```

</details>

---

## Documentation

Browse the **[Documentation Index (`docs/en/README.md`)](docs/en/README.md)** ([日本語](docs/ja/README.md) / [한국어](docs/ko/README.md)) for design philosophy, deep architectural guides, operations manuals, and auto-generated references:

| Document | Summary |
|---|---|
| **[Vision](docs/en/vision.md)** | Core philosophy of "collaboration of imperfect individuals" and organizational design principles |
| **[Overview](docs/en/overview.md)** | Full feature map of AnimaWorks and entry points to detailed chapters |
| **[Brain Mapping](docs/en/brain-mapping.md)** | How hippocampus, amygdala, prefrontal cortex, and sleep mechanisms map to AnimaWorks modules |
| **[Architecture](docs/en/architecture/index.md)** | Process tree (Phase 3), execution modes, lifecycle, prompt construction, and task management |
| **[Memory System](docs/en/memory/index.md)** | Automatic priming, hybrid retrieval, daily/weekly consolidation, and active forgetting |
| **[Security](docs/en/security.md)** | Trust boundaries, command protection, filesystem sandbox, authentication, and auditing |
| **[Enclave (Isolated Runtime)](docs/en/enclave.md)** | Building & operating the Enclave runtime for PII/production data isolation, SSM tunnels, and Egress masking |
| **[References](docs/en/reference/cli.md)** | [CLI](docs/en/reference/cli.md) / [Tool CLI](docs/en/reference/tool-cli.md) / [HTTP API](docs/en/reference/api.md) / [Configuration](docs/en/reference/config.md) / [Modules](docs/en/reference/modules.md) |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

