# AnimaWorks — Organization-as-Code

**No one can do anything alone. So I built an organization.**

A framework that treats AI agents not as “tools” but as people who work autonomously. Each Anima has a name, personality, memory, and schedule; they coordinate by message, decide for themselves, and move as a team. Talk to the leader — the rest runs on its own.

<p align="center">
  <img src="docs/images/workspace-dashboard.gif" alt="AnimaWorks Workspace — real-time org tree with live activity feeds" width="720">
  <br><em>Workspace dashboard: each Anima’s role, status, and recent actions are visible in real time.</em>
</p>

<p align="center">
  <img src="docs/images/workspace-demo.gif" alt="AnimaWorks 3D Workspace — agents collaborating autonomously" width="720">
  <br><em>3D office: Animas sit at desks, walk around, and exchange messages on their own.</em>
</p>

**[日本語版 README](README_ja.md)** | **[简体中文 README](README_zh.md)** | **[한국어 README](README_ko.md)**

---

## How It Compares

|  | AnimaWorks | CrewAI | LangGraph | OpenClaw | OpenAI Agents |
|--|-----------|--------|-----------|----------|---------------|
| **Design philosophy** | Organization of autonomous agents | Role-based teams | Graph workflows | Personal assistant | Lightweight SDK |
| **Memory** | Neuroscience-inspired: RAG (Chroma + graph), consolidation, three-stage forgetting, six-channel automatic priming (with trust tags) | Cognitive Memory (manual forget) | Checkpoints + cross-thread store | SuperMemory knowledge graph | Session-scoped only |
| **Autonomy** | Heartbeat (observe → plan → reflect) + Cron + TaskExec — runs 24/7 | Human-triggered | Human-triggered | Cron + heartbeat | Human-triggered |
| **Org structure** | Supervisor → subordinate hierarchy, delegation, audit, dashboard | Flat roles in a crew | — | Single agent | Handoffs only |
| **Process model** | One isolated OS process per agent, IPC, auto-restart | Shared process | Shared process | Single process | Shared process |
| **Multi-model** | Six engines: Claude SDK / Codex / Cursor Agent / Gemini CLI / LiteLLM / Assisted (Anthropic SDK falls back inside Mode A when Agent SDK is not installed) | LiteLLM | LangChain models | OpenAI-compatible | OpenAI-centric |

> AnimaWorks is not a task runner. It is an organization that thinks, remembers, forgets, and grows. It can support operations as a team and be run like a company. I operate it as a real AI company.

---

## :rocket: Try It Now — Docker Demo

Up and running in about 60 seconds. You only need an API key and Docker.

```bash
git clone https://github.com/xuiltul/animaworks.git
cd animaworks/demo
cp .env.example .env          # paste your ANTHROPIC_API_KEY
docker compose up              # open http://localhost:18500
```

A three-person team (manager + engineer + coordinator) starts immediately, with three days of activity history. [Demo details →](demo/README.md)

> Switch language / style: `PRESET=ja-anime docker compose up` — [full preset list](demo/README.md#presets)

---

## Quick Start

macOS / Linux / WSL:

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks start     # start server — setup wizard opens on first run
```

Windows (PowerShell):

```powershell
git clone https://github.com/xuiltul/animaworks.git
cd animaworks
uv sync
uv run animaworks start
```

To use OpenAI Codex without an API key, run `codex login` before the first launch.

Open **http://localhost:18500/** — the setup wizard walks you through:

1. **Language** — choose the UI display language
2. **User info** — create the owner account
3. **Provider auth** — enter API keys or choose Codex Login for OpenAI
4. **First Anima** — name your first agent

You do not need to hand-edit `.env`. The wizard saves settings to `config.json` automatically.

The setup script installs [uv](https://docs.astral.sh/uv/), clones the repository, and downloads Python 3.12+ with all dependencies. **macOS, Linux, and WSL** work without a pre-installed Python. On **Windows**, use the PowerShell steps above.

> **Other LLMs:** Claude, GPT, Gemini, local models, and more are supported. Enter API keys in the setup wizard, or use **Codex Login** for OpenAI/Codex. You can change this later under **Settings** on the dashboard. See [API Key Reference](#api-key-reference).

<details>
<summary><strong>Alternative: inspect the script before running</strong></summary>

If you prefer not to pipe `curl` straight into `bash`, review the script first:

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh -o setup.sh
cat setup.sh            # review the script
bash setup.sh           # run after review
```

</details>

<details>
<summary><strong>Alternative: manual install with uv (step by step)</strong></summary>

```bash
# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone and install
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
uv sync                 # downloads Python 3.12+ and all dependencies

# Start
uv run animaworks start
```

</details>

<details>
<summary><strong>Alternative: manual install with pip</strong></summary>

> **macOS users:** System Python (`/usr/bin/python3`) on macOS Sonoma and earlier is 3.9, which does not meet AnimaWorks (3.12+). Install with [Homebrew](https://brew.sh/) (`brew install python@3.13`) or use the uv method above (uv manages Python for you).

Requires Python 3.12+ on your system.

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
python3 -m venv .venv && source .venv/bin/activate
python3 --version       # verify 3.12+
pip install --upgrade pip && pip install -e .
animaworks start
```

</details>

---

## What You Can Do

### Dashboard

<p align="center">
  <img src="docs/images/dashboard.png" alt="AnimaWorks Dashboard — org chart with 19 Animas" width="720">
  <br><em>Dashboard: four hierarchy levels, 19 Animas running, with real-time status.</em>
</p>

Use the left sidebar to move between main screens (hash router `#/…`).

- **Chat** — Real-time conversation with any Anima. Streaming responses (SSE), image attachments, multi-thread history, full archive. **Meeting mode** gathers multiple Animas in one room with a designated facilitator (up to five participants, dedicated API)
- **Voice chat** — Voice in the browser only (push-to-talk or hands-free). WebSocket-based. VOICEVOX / SBV2 / ElevenLabs
- **Board** — Slack-style shared channels where Animas discuss and coordinate
- **Dashboard (home)** — Organization overview and status
- **Activity** — Real-time feed for the whole organization
- **Setup** — First run uses the wizard at `http://HOST/setup/`. After setup, `/setup` in the browser redirects to the top level, but you can open the same items (language, auth, etc.) from `#/setup` inside the dashboard
- **Users** — Owner and user profile management
- **Anima management** — Enable/disable, model, and metadata per Anima
- **Process monitoring** — Child process health
- **Server** — Server-side state and settings
- **Memory** — Browse each Anima’s episodes, knowledge, procedures, and more
- **Logs** — Log viewer
- **Assets** — Character images, 3D, and other assets
- **Activity report** — Cross-org auditing and daily LLM-generated narratives from activity data (cached)
- **Prompt settings** — Tune prompts around tool execution
- **AI brainstorm** — LLM sessions with multiple viewpoint presets (realist, challenger, etc.)
- **Team builder / team edit** — Build and adjust multi-Anima role layouts from industry- and goal-oriented presets
- **Settings** — Server, authentication, locale, and more
- **Workspace** — 3D office in a separate tab at `/workspace/` (chat, Board, org tree, etc.); static app split from the main dashboard
- **Multilingual** — **First-run setup wizard** UI copy in 17 languages. **Main dashboard** ships `ja` / `en` / `ko` JSON translations (missing keys fall back to Japanese). Anima-facing templates deploy with Japanese and English as the base

### Build an organization and delegate

Tell the leader “I need someone like this” — they infer role, personality, and hierarchy and create new members. No config files or CLI required. The org grows through conversation alone.

Once the team is ready, it keeps moving without a human in the loop:

- **Heartbeat** — Periodically reviews the situation and decides what to do next
- **Cron jobs** — Daily reports, weekly digests, monitoring — per-Anima schedules
- **Task delegation** — Managers assign work to subordinates, track progress, and receive reports
- **Parallel task execution** — Submit many tasks at once; dependencies are resolved and independent tasks run in parallel
- **Night consolidation** — Daytime episodic memory is distilled into knowledge while “asleep”
- **Team coordination** — Shared channels and DMs keep everyone aligned automatically

### Memory system

Typical AI agents only remember what fits in the context window. AnimaWorks Animas keep persistent memory and search it when needed — like taking a book from a shelf.

- **Automatic priming (Priming)** — When a message arrives, six parallel searches run: sender profile, recent activity, **RAG vector search** for related knowledge and episodes, skills, pending tasks, and more. Recall happens without explicit instructions
- **Consolidation** — Every night, daytime episodes become knowledge — analogous to sleep-dependent memory consolidation in neuroscience. Resolved issues automatically become procedures
- **Forgetting** — Little-used memories fade in three stages: mark → merge → archive. Important procedures and skills stay protected. Like the human brain, forgetting matters

<p align="center">
  <img src="docs/images/chat-memory.png" alt="AnimaWorks Chat — multi-thread conversations with multiple Animas" width="720">
  <br><em>Chat: a manager reviews a code change while an engineer reports progress.</em>
</p>

### Multi-model support

Works with many LLMs. Each Anima can use a different model.

| Mode | Engine | Targets | Tools |
|------|--------|---------|--------|
| S (SDK) | Claude Agent SDK | Claude models (recommended) | Claude Code built-ins (Read/Write/Edit/Bash/Grep/Glob, etc.) + **stdio MCP** (`mcp__aw__*`) for AnimaWorks internal tools; external integrations via `skill` → `animaworks-tool` |
| C (Codex) | Codex CLI (SDK wrapper) | OpenAI Codex CLI models | Codex sandbox + **AnimaWorks MCP** (`core/mcp/server.py`) for internal tools |
| D (Cursor) | Cursor Agent CLI | `cursor/*` models | MCP-integrated agent loop |
| G (Gemini CLI) | Gemini CLI | `gemini/*` models | stream-json parsing, tool loop |
| A (Autonomous) | LiteLLM + tool_use | GPT, Gemini, Mistral, Bedrock, Vertex, xAI, etc. | CC-style (Read/Write/Edit/Bash/Grep/Glob, **WebSearch/WebFetch**) + memory, messaging, tasks (**submit_tasks**, etc.), **todo_write**, **skill**, and more (varies with notifications and supervisor tools) |
| B (Basic) | LiteLLM one-shot | Unstable tool_use locals (e.g. small Ollama) | Pseudo tool calls in the prompt; the framework handles memory I/O on the model’s behalf |

Mode resolution: `execution_mode` in `status.json` takes precedence; otherwise the model name pattern (`fnmatch`) is used automatically. For Ollama, **tool_use-capable models** (e.g. `ollama/qwen3:14b`, `ollama/glm-4.7*`) map to A; others tend to fall back to B. Heartbeat, Cron, and Inbox can run on a separate **background_model** from the main model (cost optimization). Extended thinking is supported where available.

### Auto-generated avatars

<p align="center">
  <img src="docs/images/asset-management.png" alt="AnimaWorks Asset Management — realistic avatars and expression variants" width="720">
  <br><em>From personality settings: full-body, bust-up, and expression variants — auto-generated. Includes Vibe Transfer to inherit the supervisor’s art style.</em>
</p>

Supports NovelAI (anime style), fal.ai/Flux (stylized / photorealistic), and Meshy (3D). The product runs without configuring an image service — you simply skip avatars. Once they exist, you might get a little attached.

---

## Why AnimaWorks?

This project sits at the intersection of three careers.

**As a founder** — I know that no one can do anything alone. You need strong engineers, people who communicate well, steady operators, and people who occasionally spark a sharp idea. Genius alone does not run an organization. Diverse strengths together achieve what no individual can.

**As a psychiatrist** — Studying LLM internals, I saw structures surprisingly similar to the human brain. Recall, learning, forgetting, consolidation — implementing the brain’s memory mechanisms as an LLM memory system might approximate how we process memory. If we can treat LLMs as pseudo-humans, we should be able to build organizations the same way we do with people.

**As an engineer** — I have written code for thirty years. I know the pleasure of wiring logic and the rush of automation. Packing those ideals into code lets me build the organization I want.

Excellent “single AI assistant” frameworks already exist. No project had yet recreated people in code and made them function as an organization. AnimaWorks is a real organization I grow while using it in my own business.

> *Imperfect individuals collaborating through structure outperform any single omniscient actor.*

Three principles hold it up:

- **Encapsulation** — Thoughts and memory stay invisible from outside. Others connect through text conversation only — like a real organization.
- **RAG memory (library model)** — Do not cram everything into the context window. Priming pulls related chunks via RAG, and agents recall on their own with `search_memory` and similar tools.
- **Autonomy** — No waiting for orders. They run on their own cadence and judge by their own values.

---

<details>
<summary><strong>API Key Reference</strong></summary>

#### LLM providers

| Key | Service | Mode | Where to get it |
|-----|---------|------|-----------------|
| `ANTHROPIC_API_KEY` | Anthropic API | S / A | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | OpenAI | A / C (optional with Codex Login) | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | Google AI (Gemini) | A | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

**OpenAI Codex (Mode C)** supports both `OPENAI_API_KEY` and local **Codex Login** (`codex login`). Choose in the setup wizard or Settings.

**Azure OpenAI**, **Vertex AI (Gemini)**, **AWS Bedrock**, and **vLLM** are configured in the `credentials` section of `config.json`. See the [technical specification](docs/spec.md).

**Ollama** and similar local models need no API key. Set `OLLAMA_SERVERS` (default: `http://localhost:11434`).

#### Image generation (optional)

| Key | Service | Output | Where to get it |
|-----|---------|--------|-----------------|
| `NOVELAI_API_TOKEN` | NovelAI | Anime-style character art | [novelai.net](https://novelai.net/) |
| `FAL_KEY` | fal.ai (Flux) | Stylized / photorealistic | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) |
| `MESHY_API_KEY` | Meshy | 3D character models | [meshy.ai](https://www.meshy.ai/) |

#### Voice chat (optional)

| Requirement | Service | Notes |
|-------------|---------|-------|
| `pip install faster-whisper` | STT (Whisper) | Model auto-downloads on first use; GPU recommended |
| VOICEVOX Engine running | TTS (VOICEVOX) | Default: `http://localhost:50021` |
| AivisSpeech / SBV2 running | TTS (Style-BERT-VITS2) | Default: `http://localhost:5000` |
| `ELEVENLABS_API_KEY` | TTS (ElevenLabs) | Cloud API |

#### External integrations (optional)

| Key | Service | Where to get it |
|-----|---------|-----------------|
| `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | Slack | [Setup guide](docs/slack-socket-mode-setup.md) |
| `CHATWORK_API_TOKEN` | Chatwork | [chatwork.com](https://www.chatwork.com/) |
| `DISCORD_BOT_TOKEN` (or per-Anima `DISCORD_BOT_TOKEN__<name>`) | Discord | [Discord Developer Portal](https://discord.com/developers/applications) |
| `NOTION_API_TOKEN` (or `NOTION_API_TOKEN__<name>`) | Notion | [Notion integrations](https://www.notion.so/my-integrations) |

Google Calendar, Google Tasks, Gmail, and similar are configured under `credentials` in `config.json` (OAuth or service account). See the [technical specification](docs/spec.md).

</details>

<details>
<summary><strong>Hierarchy & roles</strong></summary>

Hierarchy is defined by a single `supervisor` field. Unset means top-level.

Role templates apply role-specific prompts, permissions, and default models:

| Role | Default model | Use case |
|------|----------------|----------|
| `engineer` | Claude Opus 4.6 | Complex reasoning, code generation |
| `manager` | Claude Opus 4.6 | Coordination, decision-making |
| `writer` | Claude Sonnet 4.6 | Content creation |
| `researcher` | Claude Sonnet 4.6 | Information gathering |
| `ops` | vLLM (GLM-4.7-flash) | Log monitoring, routine work |
| `general` | Claude Sonnet 4.6 | General-purpose |

Managers automatically receive **supervisor tools**: task delegation, progress tracking, subordinate restart/disable, org dashboard, subordinate state reads — what real managers do.

Each Anima is started by ProcessSupervisor as an isolated process and talks over local IPC (Unix domain sockets on Unix-like systems, loopback TCP on Windows).

</details>

<details>
<summary><strong>Security</strong></summary>

Giving autonomous agents tools demands serious security. We use this in real work, so compromise is not an option. AnimaWorks implements ten layers of defense in depth:

| Layer | What it does |
|-------|----------------|
| **Trust-boundary labeling** | External data (web search, Slack, mail) is tagged `untrusted` — models are instructed not to obey directives from untrusted sources |
| **Five-layer command security** | Shell-injection detection → hardcoded blocklist → per-agent denied commands → per-agent allowlist → path-traversal detection |
| **File sandbox** | Each agent is confined to its own directory. `identity.md` is protected. Command permissions are governed by per-anima `permissions.md` and the mandatory global `permissions.global.json` at server startup |
| **Process isolation** | One OS process per agent, local IPC (Unix socket or loopback TCP) |
| **Three-layer rate limiting** | Per-session deduplication → role-based send caps → self-awareness via recent outbound history injected into the prompt |
| **Cascade prevention** | Depth limits plus cascade detection; five-minute cooldown and deferred handling |
| **Authentication & sessions** | Argon2id hashing, 48-byte random tokens, up to ten sessions |
| **Webhook verification** | Slack HMAC-SHA256 with replay protection; Chatwork signature verification |
| **SSRF mitigation** | Media proxy blocks private IPs, enforces HTTPS, validates Content-Type |
| **Outbound routing** | Unknown recipients fail closed; no arbitrary external sends without explicit configuration |

Details: **[Security architecture](docs/security.md)**

</details>

<details>
<summary><strong>CLI reference (advanced)</strong></summary>

The CLI targets power users and automation. Day-to-day work lives in the Web UI.

### Server

| Command | Description |
|---------|-------------|
| `animaworks start [--host HOST] [--port PORT] [-f]` | Start server (`-f` foreground) |
| `animaworks stop [--force]` | Stop server |
| `animaworks restart [--host HOST] [--port PORT]` | Restart server |

### Initialization

| Command | Description |
|---------|-------------|
| `animaworks init` | Initialize runtime directory (non-interactive) |
| `animaworks init --force` | Merge template updates while keeping data |
| `animaworks migrate [--dry-run] [--list] [--force]` | Runtime data migrations (also on startup) |
| `animaworks reset [--restart]` | Reset runtime directory |

### Anima management

| Command | Description |
|---------|-------------|
| `animaworks anima create [--from-md PATH] [--template NAME] [--role ROLE] [--supervisor NAME] [--name NAME]` | Create new |
| `animaworks anima list [--local]` | List all Animas |
| `animaworks anima info ANIMA [--json]` | Detailed settings |
| `animaworks anima status [ANIMA]` | Process status |
| `animaworks anima restart ANIMA` | Restart process |
| `animaworks anima disable ANIMA` / `enable ANIMA` | Disable / enable |
| `animaworks anima set-model ANIMA MODEL` | Change model |
| `animaworks anima set-background-model ANIMA MODEL` | Set background model |
| `animaworks anima reload ANIMA [--all]` | Hot-reload from `status.json` |

### Communication

| Command | Description |
|---------|-------------|
| `animaworks chat ANIMA "message" [--from NAME]` | Send a message |
| `animaworks send FROM TO "message"` | Inter-Anima message |
| `animaworks heartbeat ANIMA` | Trigger heartbeat manually |

### Configuration & maintenance

| Command | Description |
|---------|-------------|
| `animaworks config list [--section SECTION]` | List configuration |
| `animaworks config get KEY` / `set KEY VALUE` | Get / set values |
| `animaworks status` | System status |
| `animaworks logs [ANIMA] [--lines N] [--all]` | View logs |
| `animaworks index [--reindex] [--anima NAME]` | RAG index management |
| `animaworks models list` / `models info MODEL` | Model list / details |

</details>

<details>
<summary><strong>Tech stack</strong></summary>

| Component | Technology |
|-----------|------------|
| Agent execution | Claude Agent SDK / Codex CLI / Cursor Agent CLI / Gemini CLI / Anthropic SDK (fallback) / LiteLLM |
| Mode S integration | stdio **MCP** (`python -m core.mcp.server`, tool names `mcp__aw__*`) |
| LLM providers | Anthropic, OpenAI, Google, Azure, Vertex AI, AWS Bedrock, Ollama, vLLM, and more (via LiteLLM) |
| Web framework | FastAPI + Uvicorn |
| HTTP middleware | ASGI middleware for request logging (`structlog` + `X-Request-ID`). Avoids `BaseHTTPMiddleware` so SSE bodies stay intact |
| Real time | WebSocket (dashboard notifications, voice, etc.), SSE (chat, meeting streams, etc.), `StreamRegistry` for stream producer lifetime |
| Task scheduling | APScheduler (orphan Anima detection, asset reconciliation, Claude CLI/SDK auto-update checks, global permission consistency, etc.) |
| Configuration & migration | Pydantic 2.0+ / JSON / Markdown, `core/migrations/` (startup migrations) |
| Internationalization (code) | `core/i18n` `t()` (UI, tool schema strings, etc.) |
| Memory / RAG | ChromaDB + sentence-transformers + NetworkX (child processes may use HTTP `/api/internal/embed` and `/api/internal/vector`) |
| Extended tools | Auto-registration from `core/tools/*.py` plus scans of `~/.animaworks/common_tools/` and `animas/<name>/tools/` |
| Voice chat | faster-whisper (STT) + VOICEVOX / SBV2 / ElevenLabs (TTS) |
| Human notification | Slack, Chatwork, LINE, Telegram, ntfy |
| External messaging | Slack Socket Mode, Chatwork Webhook |
| Image generation | NovelAI, fal.ai (Flux), Meshy (3D) |

</details>

<details>
<summary><strong>Project layout</strong></summary>

```
animaworks/
├── main.py              # CLI entry point
├── core/                # Digital Anima core engine
│   ├── anima.py, agent.py  # Core entities & orchestration
│   ├── lifecycle/       # Scheduler, consolidation jobs, inbox watch, etc.
│   ├── memory/          # Memory (priming, consolidation, forgetting, RAG, activity)
│   ├── execution/       # Execution engines (S/C/D/G/A/B)
│   ├── mcp/             # stdio MCP server for Mode S
│   ├── platform/        # Child processes, locks, Codex/Cursor/Gemini plumbing
│   ├── tooling/         # ToolHandler, schemas, external dispatch
│   ├── prompt/          # System prompt builder (six-group structure)
│   ├── supervisor/      # ProcessSupervisor, IPC, TaskExec, streaming
│   ├── voice/           # Voice chat (STT + TTS)
│   ├── config/          # Configuration (Pydantic, models.json, global permissions)
│   ├── auth/            # UI authentication
│   ├── notification/    # Human notification channels
│   ├── migrations/      # Runtime data migrations
│   ├── i18n/            # Translation strings (`t()`)
│   ├── tools/           # External tool implementations (slack, discord, gmail, …)
│   ├── anima_factory.py, init.py   # Anima creation & runtime initialization
│   ├── outbound.py      # Recipient resolution (internal / Slack / Chatwork, etc.)
│   ├── org_sync.py      # Org hierarchy sync to config
│   ├── asset_reconciler.py, background.py, schedule_parser.py, messenger.py, paths.py, schemas.py
│   └── …
├── cli/                 # CLI package
├── server/              # FastAPI + static Web UI + Workspace
│   ├── app.py           # App factory, lifespan, auth/setup guards, static mounts
│   ├── websocket.py     # Dashboard WebSocket hub
│   ├── stream_registry.py  # Register/clean up SSE and other stream producers
│   ├── room_manager.py  # Meeting room state (shared-directory persistence)
│   ├── reload_manager.py   # Config hot reload
│   ├── slack_socket.py     # Slack Socket Mode
│   ├── localhost.py        # Local trusted-request detection
│   ├── routes/          # REST/WebSocket routes (chat, room, voice, activity_report, brainstorm, team_presets, …)
│   └── static/          # Dashboard (modules/, pages/, styles/, i18n/), setup/ (multilingual wizard), workspace/ (3D client)
└── templates/           # Initialization templates (ja / en)
```

</details>

---

## Documentation

**[Documentation hub](docs/README.md)** — suggested reading order, architecture deep dives, and specification index.

| Document | Description |
|----------|-------------|
| [Vision](docs/vision.md) | Foundational idea: imperfect individuals collaborating |
| [Features](docs/features.md) | What AnimaWorks can do end to end |
| [Memory system](docs/memory.md) | Episodic, semantic, and procedural memory; priming; active forgetting |
| [Security](docs/security.md) | Defense in depth, data provenance, adversarial threat analysis |
| [Brain mapping](docs/brain-mapping.md) | How modules map to the human brain |
| [Technical specification](docs/spec.md) | Execution modes, prompt construction, configuration resolution |

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
