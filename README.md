# AnimaWorks — Organization-as-Code

**No one can do anything alone. So I built an organization.**

As an entrepreneur, I know this: a team of imperfect individuals working together will always outperform a lone genius. As a psychiatrist examining LLMs, I noticed something striking — their internal structure is remarkably similar to the human brain. I poured thirty years of engineering and a deep obsession with automation into building a real organization powered by LLMs. That's AnimaWorks.

Each Anima has its own name, personality, memory, and schedule. They talk to each other through messages, make their own decisions, and collaborate as a team. Just talk to the leader — the rest takes care of itself.

<p align="center">
  <img src="docs/images/workspace-dashboard.gif" alt="AnimaWorks Workspace — real-time org tree with live activity feeds" width="720">
  <br><em>Workspace dashboard: each Anima's role, status, and recent actions displayed in real time.</em>
</p>

<p align="center">
  <img src="docs/images/workspace-demo.gif" alt="AnimaWorks 3D Workspace — agents autonomously collaborating" width="720">
  <br><em>3D office: Animas sitting at desks, walking around, exchanging messages — all on their own.</em>
</p>

**[日本語版 README](README_ja.md)** | **[简体中文 README](README_zh.md)**

---

## :rocket: Try It Now — Docker Demo

60 seconds. Just an API key and Docker.

```bash
git clone https://github.com/xuiltul/animaworks.git
cd animaworks/demo
cp .env.example .env          # paste your ANTHROPIC_API_KEY
docker compose up              # open http://localhost:18500
```

A 3-person team (manager + engineer + coordinator) starts working immediately, with 3 days of activity history pre-loaded. [Read more about the demo →](demo/README.md)

> Switch language/style: `PRESET=ja-anime docker compose up` — [see all presets](demo/README.md#presets)

---

## Quick Start

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks start     # start the server — setup wizard opens on first run
```

Open **http://localhost:18500/** — the setup wizard walks you through it:

1. **Language** — pick your UI language
2. **User info** — create your owner account
3. **API key** — enter your LLM API key (validated in real time)
4. **First Anima** — name your first agent

No `.env` editing needed. The wizard saves everything to `config.json` automatically.

**That's it.** The setup script installs [uv](https://docs.astral.sh/uv/), clones the repo, and downloads Python 3.12+ with all dependencies. Works on **macOS, Linux, and WSL** with no pre-installed Python required.

> **Want to use a different LLM?** AnimaWorks supports Claude, GPT, Gemini, local models, and more. Enter your API key in the setup wizard, or add it later from **Settings** in the dashboard. See [API Key Reference](#api-key-reference) below.

<details>
<summary><strong>Alternative: inspect the script before running</strong></summary>

If you'd rather review the script before executing it:

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

> **macOS users:** The system Python (`/usr/bin/python3`) is 3.9 on macOS Sonoma and earlier — too old for AnimaWorks (requires 3.12+). Install via [Homebrew](https://brew.sh/) (`brew install python@3.13`) or use the uv method above, which handles Python automatically.

Requires Python 3.12+ already on your system.

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
python3 -m venv .venv && source .venv/bin/activate
python3 --version       # verify 3.12+
pip install --upgrade pip && pip install -e .
animaworks start
```

</details>

---

## What You Get

### Dashboard

Your command center. Every Anima's status, activity, and memory stats at a glance.

<p align="center">
  <img src="docs/images/dashboard.png" alt="AnimaWorks Dashboard — org chart with 19 agents" width="720">
  <br><em>Dashboard: 19 Animas across 4 hierarchy levels, all running with real-time status.</em>
</p>

- **Chat** — Talk to any Anima in real time. Streaming responses, image attachments, multi-thread conversations, full history
- **Voice Chat** — Talk with your voice (push-to-talk or hands-free)
- **Board** — Slack-style shared channels where Animas discuss and coordinate on their own
- **Activity** — Real-time feed of everything happening across the organization
- **Memory** — Peek into what each Anima remembers — episodes, knowledge, procedures
- **Settings** — API keys, authentication, system configuration
- **i18n** — 17 languages for UI; templates in Japanese + English with automatic fallback

### 3D Workspace

Watch your Animas work in a 3D office.

- They sit at desks, walk around, and talk to each other on their own
- Idle, working, thinking, sleeping — their state is visible at a glance
- Speech bubbles appear during conversations
- Click to open a live chat; expressions change in real time

---

## Build Your Team

Just like a real company. Tell the leader who you need:

> *"I'd like to hire a researcher who monitors industry trends, and an engineer who manages our infrastructure."*

The leader figures out the right roles, personalities, and reporting structure, then creates new members. No config files to write. No CLI commands to run. The organization grows through conversation.

### The Organization Keeps Running Without You

Just like a real team — once it's in place, it runs on its own:

- **Heartbeats** — Each Anima periodically reviews tasks, reads channels, and decides what to do next
- **Cron jobs** — Scheduled tasks per Anima: daily reports, weekly summaries, monitoring
- **Task delegation** — Managers assign work to subordinates, track progress, and receive reports
- **Parallel task execution** — Submit multiple tasks at once; dependencies are resolved and independent tasks run concurrently
- **Night consolidation** — Daytime episodes are distilled into knowledge while they sleep
- **Team coordination** — Shared channels and DMs keep everyone in sync automatically

### Auto-Generated Avatars

<p align="center">
  <img src="docs/images/asset-management.png" alt="AnimaWorks Asset Management — realistic portraits and expression variants" width="720">
  <br><em>Asset management: realistic full-body, bust-up, and expression variants — all auto-generated from personality settings.</em>
</p>

When a new Anima is created, character images and a 3D model are auto-generated from their personality. If the supervisor already has a portrait, **Vibe Transfer** carries the art style over — so the whole team looks visually consistent.

Supports NovelAI (anime-style), fal.ai/Flux (stylized/photorealistic), and Meshy (3D models). Works fine without any image service configured — agents just won't have avatars.

---

## Why AnimaWorks?

This project was born at the intersection of three careers.

**As an entrepreneur** — I know that no one can do anything alone. I can't, either. You need strong engineers, people who are great at communication, workers who show up and grind every day, and people who occasionally come up with a brilliant idea out of nowhere. No organization runs on genius alone. When you bring diverse strengths together, you achieve things no individual ever could.

**As a psychiatrist** — When I examined the internal structure of LLMs, I noticed something remarkable: they mirror the human brain in surprising ways. Recall, learning, forgetting, consolidation — the mechanisms the brain uses to process memory can be implemented directly as an LLM memory system. If that's the case, we should be able to treat LLMs as pseudo-humans and build organizations with them, just like we do with people.

**As an engineer** — I've been writing code for thirty years. I know the joy of building logic, the thrill of automation. If I pour all my ideals into code, I can build my ideal organization.

Good "single AI assistant" frameworks already exist. But no one had built a project that recreates humans in code and makes them function as an organization. AnimaWorks is a real corporate organization that I'm growing inside my own business, day by day.

> *Imperfect individuals collaborating through structure outperform any single omniscient actor.*

| Traditional Agent Frameworks | AnimaWorks |
|------------------------------|------------|
| Execute and forget | Accumulate through memory |
| One orchestrator calls all the shots | Each agent decides and acts on its own |
| Everyone sees the same context | Each agent recalls what it needs, when it needs it |
| Chain tools in sequence | An organization connected by messages |
| Tune the prompt | Judgment based on personality and values |

Three principles make this work:

- **Encapsulation** — Internal thoughts and memories are invisible from outside. Communication happens only through text. Just like a real organization.
- **Library-style memory** — No cramming everything into a context window. When agents need to remember, they search their own archives — like pulling a book off a shelf.
- **Autonomy** — They don't wait for instructions. They run on their own clocks and make decisions based on their own values.

---

## Memory System

Through a psychiatrist's lens, most AI agents are effectively amnesic — they only remember what fits in the context window. AnimaWorks agents maintain a persistent memory archive and **search it when they need to remember.** Like pulling a book off a shelf. In neuroscience terms, that's recall.

| Memory Type | Neuroscience Analog | What's Stored |
|---|---|---|
| `episodes/` | Episodic memory | Daily activity logs |
| `knowledge/` | Semantic memory | Lessons, rules, learned knowledge |
| `procedures/` | Procedural memory | Step-by-step workflows |
| `skills/` | Skill memory | Reusable task-specific instructions |
| `state/` | Working memory | Current tasks, pending items, task queue |
| `shortterm/` | Short-term memory | Session continuity (chat/heartbeat separated) |
| `activity_log/` | Unified timeline | All interactions (JSONL) |

### Memory Evolves

- **Priming** — When a message arrives, 6 parallel searches fire automatically: sender profile, recent activity, related knowledge, skills, pending tasks, past episodes. Results are injected into the system prompt — agents remember without being told to.
- **Consolidation** — Every night, the day's episodes are distilled into semantic knowledge — the same mechanism as sleep-time memory consolidation in neuroscience. Resolved issues automatically become procedures. Weekly knowledge merge and compression.
- **Forgetting** — Unused memories gradually fade through 3 stages: marking, merging, archival. Important procedures and skills are protected. Just like the human brain, forgetting matters too.

<p align="center">
  <img src="docs/images/chat-memory.png" alt="AnimaWorks Chat — multi-thread conversations with multiple Animas" width="720">
  <br><em>Chat: multi-thread conversations — a manager reviewing code fixes while an engineer reports progress.</em>
</p>

---

## Voice Chat

Talk to your Animas with your voice, right in the browser. No app needed.

- **Push-to-Talk (PTT)** — Hold the mic button to record, release to send
- **VAD Mode** — Hands-free: automatic speech detection starts and stops recording
- **Barge-in** — Start talking to interrupt the Anima mid-sentence
- **Multiple TTS providers** — VOICEVOX, Style-BERT-VITS2/AivisSpeech, ElevenLabs
- **Per-Anima voices** — Each Anima can have a different voice and speaking style

The mechanism is simple — it flows through the same pipeline as text chat: speech → STT (faster-whisper) → Anima reasoning → response text → TTS → audio playback. The Anima doesn't even know it's a voice conversation — it just responds to text.

---

## Multi-Model Support

Runs on any LLM. Each Anima can use a different model — this is "the right person for the right job," in code.

| Mode | Engine | Best For | Tools |
|------|--------|----------|-------|
| S (SDK) | Claude Agent SDK | Claude models (recommended) | Full: Read/Write/Edit/Bash/Grep/Glob via subprocess |
| C (Codex) | Codex SDK | OpenAI Codex CLI models | Full: same as Mode S via Codex subprocess |
| A (Autonomous) | LiteLLM + tool_use | GPT-4o, Gemini, Mistral, vLLM, etc. | search_memory, read/write_file, send_message, etc. |
| A (Fallback) | Anthropic SDK | Claude (when Agent SDK unavailable) | Same as Mode A |
| B (Basic) | LiteLLM 1-shot | Ollama, small local models | Framework handles memory I/O on behalf of the model |

Mode is auto-detected from the model name via wildcard pattern matching. Override per-Anima in `status.json`. Define execution mode and context window per model in `~/.animaworks/models.json`.

**Background model** — Heartbeat, Cron, and Inbox can run on a lighter model than the main one (cost optimization). Set via `animaworks anima set-background-model {name} {model}`.

**Extended thinking** is available for models that support it (Claude, Gemini) — you can watch the Anima's reasoning process in the UI.

### API Key Reference

#### LLM Providers

| Key | Service | Mode | Get it at |
|-----|---------|------|-----------|
| `ANTHROPIC_API_KEY` | Anthropic API | S / A | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | OpenAI | A / C | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | Google AI (Gemini) | A | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

For **Azure OpenAI**, **Vertex AI (Gemini)**, **AWS Bedrock**, and **vLLM** — configure in the `credentials` section of `config.json`. See the [technical spec](docs/spec.md) for details.

For **Ollama** and other local models — no API key needed. Set `OLLAMA_SERVERS` (default: `http://localhost:11434`).

#### Image Generation (Optional)

| Key | Service | Output | Get it at |
|-----|---------|--------|-----------|
| `NOVELAI_API_TOKEN` | NovelAI | Anime-style character images | [novelai.net](https://novelai.net/) |
| `FAL_KEY` | fal.ai (Flux) | Stylized / photorealistic | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) |
| `MESHY_API_KEY` | Meshy | 3D character models | [meshy.ai](https://www.meshy.ai/) |

#### Voice Chat (Optional)

| Requirement | Service | Notes |
|-------------|---------|-------|
| `pip install faster-whisper` | STT (Whisper) | Auto-downloads model on first use. GPU recommended |
| VOICEVOX Engine running | TTS (VOICEVOX) | Default: `http://localhost:50021` |
| AivisSpeech/SBV2 running | TTS (Style-BERT-VITS2) | Default: `http://localhost:5000` |
| `ELEVENLABS_API_KEY` | TTS (ElevenLabs) | Cloud API |

#### External Integrations (Optional)

| Key | Service | Get it at |
|-----|---------|-----------|
| `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | Slack | [Setup guide](docs/slack-socket-mode-setup.md) |
| `CHATWORK_API_TOKEN` | Chatwork | [chatwork.com](https://www.chatwork.com/) |

---

## Hierarchy & Roles

Just like a real organization, hierarchy is defined by a single `supervisor` field. No supervisor means top-level.

Role templates automatically apply role-specific prompts, permissions, and model defaults:

| Role | Default Model | Use Case |
|------|---------------|----------|
| `engineer` | Claude Opus 4.6 | Complex reasoning, code generation |
| `manager` | Claude Opus 4.6 | Coordination, decision-making |
| `writer` | Claude Sonnet 4.6 | Content creation |
| `researcher` | Claude Sonnet 4.6 | Information gathering |
| `ops` | vLLM (GLM-4.7-flash) | Log monitoring, routine tasks |
| `general` | Claude Sonnet 4.6 | General-purpose |

Managers get **supervisor tools** automatically: task delegation, progress tracking, subordinate restart/disable, org dashboard, subordinate state reading — the same things a real manager does.

All communication flows through async messaging via Messenger. Each Anima runs as an isolated subprocess managed by ProcessSupervisor, communicating over Unix Domain Sockets.

---

## Security

When you give autonomous agents real tools, you have to take security seriously. AnimaWorks implements defense-in-depth across 10 layers:

| Layer | What It Does |
|-------|-------------|
| **Trust boundary labeling** | All external data (web search, Slack, email) is tagged `untrusted` — the model is told never to follow directives from untrusted sources |
| **5-layer command security** | Shell injection detection → hardcoded blocklist → per-agent denied commands → per-agent allowlist → path traversal check |
| **File sandboxing** | Each agent is confined to its own directory. Critical files (`permissions.md`, `identity.md`) are immutable to the agent |
| **Process isolation** | One OS process per agent, communicating via Unix Domain Sockets — not TCP |
| **3-layer rate limiting** | Per-session dedup → role-based outbound budgets (manager 60/hr · 300/day down to general 15/hr · 50/day, per-agent override via status.json) → self-awareness via prompt injection of recent sends |
| **Cascade prevention** | Two-layer control: depth limiter (max 6 turns per pair in 10 min) + cascade detection (max 3 round-trips per pair in 30 min). 5-minute cooldown with deferred processing |
| **Auth & sessions** | Argon2id hashing, 48-byte random tokens, max 10 sessions, 0600 file permissions |
| **Webhook verification** | HMAC-SHA256 for Slack (with replay protection) and Chatwork signature verification |
| **SSRF mitigation** | Media proxy blocks private IPs, enforces HTTPS, validates content types, checks DNS resolution |
| **Outbound routing** | Unknown recipients fail-closed. No arbitrary external sends without explicit config |

Details: **[Security Architecture](docs/security.md)**

---

<details>
<summary><strong>CLI Reference (Advanced)</strong></summary>

The CLI is for power users and automation. Day-to-day use is through the Web UI.

### Server

| Command | Description |
|---------|-------------|
| `animaworks start [--host HOST] [--port PORT] [-f]` | Start server (`-f` for foreground) |
| `animaworks stop [--force]` | Stop server |
| `animaworks restart [--host HOST] [--port PORT]` | Restart server |

### Setup

| Command | Description |
|---------|-------------|
| `animaworks init` | Initialize runtime directory (non-interactive) |
| `animaworks init --force` | Merge template updates (preserves data) |
| `animaworks reset [--restart]` | Reset runtime directory |

### Anima Management

| Command | Description |
|---------|-------------|
| `animaworks anima create [--from-md PATH] [--template NAME] [--role ROLE] [--supervisor NAME] [--name NAME]` | Create new (character sheet/template/blank) |
| `animaworks anima list [--local]` | List all Animas (name, enabled/disabled, model, supervisor) |
| `animaworks anima info ANIMA [--json]` | Detailed config (model, role, credential, voice, etc.) |
| `animaworks anima status [ANIMA]` | Show process status |
| `animaworks anima restart ANIMA` | Restart process |
| `animaworks anima disable ANIMA` | Disable (stop) Anima |
| `animaworks anima enable ANIMA` | Enable (start) Anima |
| `animaworks anima set-model ANIMA MODEL [--credential CRED]` | Change model |
| `animaworks anima set-background-model ANIMA MODEL [--credential CRED]` | Set Heartbeat/Cron model |
| `animaworks anima set-background-model ANIMA --clear` | Clear background model |
| `animaworks anima reload ANIMA [--all]` | Hot-reload config from status.json (no restart) |

### Communication

| Command | Description |
|---------|-------------|
| `animaworks chat ANIMA "message" [--from NAME]` | Send message |
| `animaworks send FROM TO "message"` | Inter-Anima message |
| `animaworks heartbeat ANIMA` | Trigger heartbeat manually |

### Configuration & Maintenance

| Command | Description |
|---------|-------------|
| `animaworks config list [--section SECTION]` | List config |
| `animaworks config get KEY` | Get value (dot notation) |
| `animaworks config set KEY VALUE` | Set value |
| `animaworks status` | System status |
| `animaworks logs [ANIMA] [--lines N] [--all]` | View logs |
| `animaworks index [--reindex] [--anima NAME]` | RAG index management |
| `animaworks optimize-assets [--anima NAME]` | Optimize asset images |
| `animaworks remake-assets ANIMA --style-from REF` | Regenerate assets (Vibe Transfer) |
| `animaworks models list` / `animaworks models info MODEL` | Model list / details |

</details>

<details>
<summary><strong>Tech Stack</strong></summary>

| Component | Technology |
|-----------|------------|
| Agent execution | Claude Agent SDK / Codex SDK / Anthropic SDK / LiteLLM |
| LLM providers | Anthropic, OpenAI, Google, Azure, Vertex AI, AWS Bedrock, Ollama, vLLM |
| Web framework | FastAPI + Uvicorn |
| Task scheduling | APScheduler |
| Configuration | Pydantic 2.0+ / JSON / Markdown |
| Memory / RAG | ChromaDB + sentence-transformers + NetworkX |
| Voice chat | faster-whisper (STT) + VOICEVOX / SBV2 / ElevenLabs (TTS) |
| Human notification | Slack, Chatwork, LINE, Telegram, ntfy |
| External messaging | Slack Socket Mode, Chatwork Webhook |
| Image generation | NovelAI, fal.ai (Flux), Meshy (3D) |

</details>

<details>
<summary><strong>Project Structure</strong></summary>

```
animaworks/
├── main.py              # CLI entry point
├── core/                # Digital Anima core engine
│   ├── anima.py, agent.py, lifecycle.py  # Core entities & orchestrator
│   ├── anima_factory.py, init.py         # Initialization & Anima creation
│   ├── schemas.py, paths.py              # Data models & path constants
│   ├── messenger.py, outbound.py         # Messaging & outbound routing
│   ├── background.py, asset_reconciler.py, org_sync.py
│   ├── schedule_parser.py                # cron.md / heartbeat.md parser
│   ├── memory/          # Memory subsystem
│   │   ├── manager.py, conversation.py, shortterm.py
│   │   ├── priming.py   # Auto-recall (6-channel parallel)
│   │   ├── consolidation.py, forgetting.py
│   │   ├── activity.py, streaming_journal.py, task_queue.py
│   │   └── rag/         # RAG engine (ChromaDB + graph spreading activation)
│   ├── execution/       # Execution engines (S/C/A/B)
│   │   ├── agent_sdk.py, codex_sdk.py, litellm_loop.py, assisted.py
│   │   └── anthropic_fallback.py, _session.py
│   ├── tooling/         # Tool dispatch, permission checks, guide generation
│   ├── prompt/          # System prompt builder (6-group structure)
│   ├── supervisor/      # Process supervision (includes pending_executor, scheduler_manager)
│   ├── voice/           # Voice chat (STT + TTS + session management)
│   ├── config/          # Configuration (Pydantic models, vault)
│   ├── notification/    # Human notification (slack, chatwork, line, telegram, ntfy)
│   ├── auth/            # Authentication (Argon2id + sessions)
│   └── tools/           # External tool implementations
├── cli/                 # CLI package (argparse + subcommands)
├── server/              # FastAPI server + Web UI
│   ├── routes/          # API routes (domain-split)
│   └── static/          # Dashboard + Workspace UI
└── templates/           # Initialization templates
    ├── ja/, en/         # Locale-specific (prompts, roles, common_knowledge, common_skills)
    └── _shared/         # Locale-shared (company, etc.)
```

</details>

---

## Template Localization

AnimaWorks templates (system prompts, shared knowledge, skills, role definitions) are organized by locale under `templates/`:

```
templates/
├── _shared/         # Language-independent (defaults.json, etc.)
├── ja/              # Japanese (complete)
└── en/              # English (complete)
```

`load_prompt()` resolves templates with a fallback chain: **requested locale → en → ja**. If a file doesn't exist in your locale, it automatically falls back to English, then Japanese.

Set your locale during setup or in `config.json`:

```json
{ "locale": "en" }
```

### Adding a New Language

1. Copy `templates/en/` to `templates/{your-locale}/` (e.g. `templates/fr/`)
2. Translate the Markdown files. Keep all `{variable}` placeholders intact
3. The fallback chain means you can translate incrementally — untranslated files will fall back to English

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Documentation

**[Full documentation index](docs/README.md)** — reading guides, architecture deep dives, research papers, and design specs.

| Document | Description |
|----------|-------------|
| [Vision](docs/vision.md) | Core philosophy: imperfect individuals collaborating beats a single omniscient model |
| [Features](docs/features.md) | Everything AnimaWorks can do |
| [Memory System](docs/memory.md) | Episodic, semantic, and procedural memory; priming; active forgetting |
| [Security](docs/security.md) | Defense-in-depth model, provenance tracking, adversarial threat analysis |
| [Brain Mapping](docs/brain-mapping.md) | Every module mapped to a region of the human brain |
| [Technical Spec](docs/spec.md) | Execution modes, prompt construction, configuration resolution |

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
