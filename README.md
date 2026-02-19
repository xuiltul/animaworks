# AnimaWorks - Organization-as-Code

**Build an AI office where agents work as autonomous people — not tools.**

Each agent has its own name, personality, memory, and schedule. They communicate through messages, make decisions on their own, and collaborate like a real team. You manage them through a web dashboard — or just talk to the leader and let them handle the rest.

<!-- TODO: hero screenshot / GIF here -->

**[日本語版 README はこちら](README_ja.md)**

---

## Quick Start

**Prerequisites:** Python 3.12+, [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (recommended — no API key needed) or any LLM API key.

```bash
git clone https://github.com/xuiltul/animaworks.git
cd animaworks
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

```bash
animaworks init     # Opens setup wizard in your browser
animaworks start    # Start the server
```

Open **http://localhost:18500/** — your first Anima is ready. Click to start chatting.

**That's it.** Everything from here happens in the browser.

> **Using a different LLM?** Copy `.env.example` to `.env` and add your API key. See [API Key Reference](#api-key-reference) below.

---

## What You Get

### Dashboard

Your command center. See every agent's status, recent activity, and memory stats at a glance.

<!-- TODO: dashboard screenshot -->

- **Chat** — Talk to any Anima with streaming responses, image attachments, and full conversation history
- **Board** — Slack-style shared channels (#general, #ops, etc.) where Animas discuss and coordinate
- **Activity** — Real-time timeline of everything happening across your organization
- **Memory** — Browse each Anima's episodes, knowledge, and procedures
- **Settings** — API keys, authentication, and configuration

### 3D Workspace

An interactive office where your Animas exist as visible characters.

<!-- TODO: workspace screenshot / GIF -->

- Characters sit at desks, walk around, and talk to each other in real time
- Visual states show what each Anima is doing — idle, working, thinking, sleeping
- Message bubbles appear during conversations
- Click any character to open a live chat with expression changes

---

## Build Your Team

Your first Anima is the leader. Tell it who you need:

> *"I'd like to hire a researcher who monitors industry trends, and an engineer who manages our infrastructure."*

The leader creates new team members with the right roles, personalities, and reporting structure — all through conversation. No config files. No CLI commands.

### They Work While You're Away

Once your team exists, they run on their own:

- **Heartbeats** — Each Anima periodically reviews tasks, reads channels, and decides what to do next
- **Cron jobs** — Scheduled tasks per Anima (daily reports, weekly summaries, monitoring)
- **Night consolidation** — Episodes are distilled into knowledge while agents "sleep"
- **Team communication** — Shared channels and direct messages keep everyone in sync

### Auto-Generated Avatars

When a new Anima is created, AnimaWorks can automatically generate a character portrait and 3D model from their personality description. If a supervisor already has a portrait, **Vibe Transfer** applies the same art style to new hires — so your whole team looks visually consistent.

Supports NovelAI (anime-style), fal.ai/Flux (stylized/photorealistic), and Meshy (3D models). Works without any image service configured — agents just won't have visual avatars.

---

## Why AnimaWorks?

Most AI agent frameworks treat agents as stateless functions — they execute, forget, and wait for the next call. AnimaWorks takes a fundamentally different approach:

**Agents are people in an organization, not tools in a pipeline.**

| Traditional Agent Frameworks | AnimaWorks |
|------------------------------|------------|
| Stateless execution | Persistent identity and memory |
| Centralized orchestrator | Self-directed autonomous agents |
| Shared context window | Private memory with selective recall |
| Tool-use chains | Message-passing organization |
| Prompt engineering | Personality and values |

Three principles make this work:

- **Encapsulation** — Each Anima's internal thoughts and memories are invisible to others. Communication happens only through text messages — just like real organizations.
- **Library-style memory** — Instead of cramming everything into a context window, agents search their own memory archives when they need to remember something.
- **Autonomy** — Agents don't wait for instructions. They run on their own clocks and make decisions based on their own values.

> *Imperfect individuals collaborating through structure outperform any single omniscient actor.*

This insight comes from two parallel careers: a psychiatrist who learned that no mind is complete on its own, and an entrepreneur who learned that the right org chart matters more than any individual hire.

---

## Memory System

Most AI agents have something resembling amnesia — they only remember what fits in their context window. AnimaWorks agents maintain a persistent memory archive and **search it when they need to remember**, the way you'd pull a book off a shelf.

| Memory Type | Neuroscience Analog | What's Stored |
|---|---|---|
| `episodes/` | Episodic memory | Daily activity logs |
| `knowledge/` | Semantic memory | Lessons, rules, learned knowledge |
| `procedures/` | Procedural memory | Step-by-step workflows |
| `state/` | Working memory | Current tasks, pending items |
| `shortterm/` | Short-term memory | Session continuity |
| `activity_log/` | Unified timeline | All interactions as JSONL |

### How Memory Evolves

- **Priming** — When a message arrives, 4 parallel searches run automatically: sender profile, recent activity, related knowledge, and skill matching. Results are injected into the system prompt — the agent "remembers" without being told to.
- **Consolidation** — Every night, daily episodes are distilled into semantic knowledge (like sleep-time learning). Weekly, knowledge entries are merged and compressed.
- **Forgetting** — Low-value memories gradually fade through 3 stages: marking, merging, and archival. Important procedures and skills are protected.

---

## Multi-Model Support

AnimaWorks supports any LLM. Each Anima can use a different model.

| Mode | Engine | Best For | Tools |
|------|--------|----------|-------|
| A1 | Claude Agent SDK | Claude models (recommended) | Full: Read/Write/Edit/Bash/Grep/Glob |
| A1 Fallback | Anthropic SDK | Claude (when Agent SDK unavailable) | search_memory, read/write_file, etc. |
| A2 | LiteLLM + tool_use | GPT-4o, Gemini, etc. | search_memory, read/write_file, etc. |
| B | LiteLLM text-based | Ollama, local models | Pseudo tool calls (text-parsed) |

Mode is auto-detected from the model name. Override per-Anima in `config.json` if needed.

### API Key Reference

**Claude Code (Mode A1) requires no API keys.**

#### LLM Providers

| Key | Service | Mode | Get it at |
|-----|---------|------|-----------|
| *(none needed)* | Claude Code | A1 | [docs.anthropic.com](https://docs.anthropic.com/en/docs/claude-code) |
| `ANTHROPIC_API_KEY` | Anthropic API | A1 Fallback / A2 | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | OpenAI | A2 | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | Google AI | A2 | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

#### Image Generation (Optional)

| Key | Service | Output | Get it at |
|-----|---------|--------|-----------|
| `NOVELAI_API_TOKEN` | NovelAI | Anime-style portraits | [novelai.net](https://novelai.net/) |
| `FAL_KEY` | fal.ai (Flux) | Stylized / photorealistic | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) |
| `MESHY_API_KEY` | Meshy | 3D character models | [meshy.ai](https://www.meshy.ai/) |

#### External Integrations (Optional)

| Key | Service | Get it at |
|-----|---------|-----------|
| `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | Slack | [Setup guide](docs/slack-socket-mode-setup.md) |
| `CHATWORK_API_TOKEN` | Chatwork | [chatwork.com](https://www.chatwork.com/) |
| `OLLAMA_SERVERS` | Ollama (local LLM) | Default: `http://localhost:11434` |

---

## Hierarchy & Roles

Hierarchy is defined by a single `supervisor` field. No supervisor = top-level.

Role templates provide specialized prompts, permissions, and model defaults:

| Role | Default Model | Description |
|------|--------------|-------------|
| `engineer` | Opus | Complex reasoning, code generation |
| `manager` | Opus | Coordination, decision-making |
| `writer` | Sonnet | Content creation |
| `researcher` | Haiku | Information gathering |
| `ops` | Local model | Log monitoring, routine tasks |
| `general` | Sonnet | General-purpose |

All communication flows through async messaging. Each Anima runs as an isolated subprocess managed by ProcessSupervisor, communicating via Unix Domain Sockets.

---

<details>
<summary><strong>CLI Reference (Advanced)</strong></summary>

The CLI is for power users and automation. Day-to-day use is through the Web UI.

### Server

| Command | Description |
|---|---|
| `animaworks start [--host HOST] [--port PORT]` | Start server (default: `0.0.0.0:18500`) |
| `animaworks stop` | Stop server |
| `animaworks restart [--host HOST] [--port PORT]` | Restart server |

### Setup

| Command | Description |
|---|---|
| `animaworks init` | Interactive setup wizard |
| `animaworks init --force` | Merge template updates (preserves data) |
| `animaworks reset [--restart]` | Reset runtime directory |

### Anima Management

| Command | Description |
|---|---|
| `animaworks create-anima [--from-md PATH] [--role ROLE] [--name NAME]` | Create from character sheet |
| `animaworks anima status [ANIMA]` | Show process status |
| `animaworks anima restart ANIMA` | Restart process |
| `animaworks list` | List all Animas |

### Communication

| Command | Description |
|---|---|
| `animaworks chat ANIMA "message" [--from NAME]` | Send message |
| `animaworks send FROM TO "message"` | Inter-Anima message |
| `animaworks heartbeat ANIMA` | Trigger heartbeat |

### Configuration

| Command | Description |
|---|---|
| `animaworks config list [--section SECTION]` | Show config |
| `animaworks config get KEY` | Get value (dot notation) |
| `animaworks config set KEY VALUE` | Set value |
| `animaworks status` | System status |
| `animaworks logs [ANIMA] [--lines N]` | View logs |

</details>

<details>
<summary><strong>Tech Stack</strong></summary>

| Component | Technology |
|---|---|
| Agent execution | Claude Agent SDK / Anthropic SDK / LiteLLM |
| LLM providers | Anthropic, OpenAI, Google, Ollama (via LiteLLM) |
| Web framework | FastAPI + Uvicorn |
| Task scheduling | APScheduler |
| Configuration | Pydantic + JSON + Markdown |
| Memory / RAG | ChromaDB + sentence-transformers |
| Graph activation | NetworkX (spreading activation + PageRank) |
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
│   ├── anima.py         #   Encapsulated persona class
│   ├── agent.py         #   Execution mode selection & cycle management
│   ├── anima_factory.py #   Anima creation (template/blank/markdown)
│   ├── memory/          #   Memory subsystem
│   │   ├── manager.py   #     Library-style search & write
│   │   ├── priming.py   #     Auto-recall layer (4-channel parallel)
│   │   ├── consolidation.py #  Memory consolidation (daily/weekly)
│   │   ├── forgetting.py #    Active forgetting (3-stage)
│   │   └── rag/         #     RAG engine (ChromaDB + embeddings)
│   ├── execution/       #   Execution engines (A1/A1F/A2/B)
│   ├── tooling/         #   Tool dispatch & permissions
│   ├── prompt/          #   System prompt builder (24 sections)
│   ├── supervisor/      #   Process isolation (Unix sockets)
│   └── tools/           #   External tool implementations
├── cli/                 # CLI package (argparse + subcommands)
├── server/              # FastAPI server + Web UI
│   ├── routes/          #   API routes (domain-split)
│   └── static/          #   Dashboard + Workspace UI
└── templates/           # Default configs & prompt templates
    ├── roles/           #   Role templates (6 roles)
    └── anima_templates/ #   Anima skeletons
```

</details>

---

## Documentation

| Document | Description |
|----------|-------------|
| [Design Philosophy](docs/vision.md) | Core principles and vision |
| [Memory System](docs/memory.md) | Memory architecture specification |
| [Brain Mapping](docs/brain-mapping.md) | Architecture mapped to neuroscience |
| [Feature Index](docs/features.md) | Comprehensive feature list |
| [Technical Spec](docs/spec.md) | Technical specification |

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
