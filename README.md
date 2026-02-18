# AnimaWorks

**Organization-as-Code for LLM Agents**

AnimaWorks treats AI agents not as tools, but as autonomous members of an organization. Each agent (called a "Digital Anima") has its own persistent identity, private memory, and communication channels. They operate autonomously on their own schedules, collaborating through message-passing — the same way human organizations work.

> *Imperfect individuals collaborating through structure outperform any single omniscient actor.*

**[日本語版 README はこちら](README_ja.md)**

## Core Principles

- **Encapsulation** — Each Anima's internal thoughts and memories are invisible to others. Communication happens only through text messages.
- **Library-style Memory** — Instead of cramming context into prompts, agents search their own memory archives when they need to remember something.
- **Autonomy** — Agents don't wait for instructions. They run on their own clocks (heartbeats and cron schedules) and make decisions based on their own values.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│            Digital Anima: (Alice)                     │
├──────────────────────────────────────────────────────┤
│  Identity ────── Who I am (persistent)               │
│  Agent Core ──── 4 execution modes                   │
│    ├ A1: Claude Agent SDK (Claude models)             │
│    ├ A1 Fallback: Anthropic SDK (when SDK missing)    │
│    ├ A2: LiteLLM + tool_use (GPT-4o, Gemini, etc.)   │
│    └ B:  LiteLLM text-based tool loop (Ollama, etc.)   │
│  Memory ──────── Library-style, self-directed recall  │
│  Boards ──────── Slack-style shared channels          │
│  Permissions ─── Tool/file/command restrictions       │
│  Communication ─ Text + file references               │
│  Lifecycle ───── Message / heartbeat / cron           │
│  Injection ───── Role / values / behavioral rules     │
└──────────────────────────────────────────────────────┘
```

## Neuroscience-Inspired Memory System

Most AI agent frameworks truncate memory to fit the context window — leaving agents with something resembling amnesia. AnimaWorks takes a different approach: agents maintain a persistent memory archive and **search it when they need to remember**, the way humans retrieve information from long-term storage.

| Directory | Neuroscience Model | Contents |
|---|---|---|
| `episodes/` | Episodic memory | Daily activity logs |
| `knowledge/` | Semantic memory | Lessons, rules, learned knowledge |
| `procedures/` | Procedural memory | Step-by-step workflows |
| `state/` | Working memory | Current tasks, pending items |
| `shortterm/` | Short-term memory | Session continuity (context carry-over) |
| `activity_log/` | Unified activity log | All interactions as JSONL timeline |

### Memory Lifecycle

- **Priming** — 4-channel parallel memory retrieval automatically injected into system prompts (sender profile, recent activity, related knowledge, skill matching)
- **Consolidation** — Daily (episodic → semantic, NREM-sleep analog) and weekly (knowledge merge + episode compression)
- **Forgetting** — 3-stage active forgetting based on the synaptic homeostasis hypothesis:
  1. Synaptic downscaling (daily): mark low-access chunks
  2. Neurogenesis reorganization (weekly): merge similar low-activity chunks
  3. Complete forgetting (monthly): archive and remove inactive chunks

## Quick Start

### Requirements

- Python 3.12+
- Anthropic API key

### Installation

```bash
git clone https://github.com/<your-org>/animaworks.git
cd animaworks
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configuration

```bash
# Required
export ANTHROPIC_API_KEY=sk-ant-...

# Optional
export ANIMAWORKS_DATA_DIR=~/.animaworks  # Runtime data directory (default: ~/.animaworks)
```

### Initialize & Run

```bash
# Initialize runtime directory (first time only)
animaworks init

# Start server
animaworks start

# Open Web UI
# Dashboard:  http://localhost:18500/
# Workspace:  http://localhost:18500/workspace/
```

## CLI Reference

### Server Management

| Command | Description |
|---|---|
| `animaworks start [--host HOST] [--port PORT]` | Start server (default: `0.0.0.0:18500`) |
| `animaworks stop` | Stop server (graceful shutdown) |
| `animaworks restart [--host HOST] [--port PORT]` | Restart server |

### Initialization

| Command | Description |
|---|---|
| `animaworks init` | Initialize runtime directory (interactive setup) |
| `animaworks init --force` | Merge template updates (preserves existing data) |
| `animaworks init --from-md PATH [--name NAME]` | Create Anima from Markdown character sheet |
| `animaworks init --blank NAME` | Create empty Anima skeleton |
| `animaworks reset [--restart]` | Reset runtime directory |

### Anima Management

| Command | Description |
|---|---|
| `animaworks create-anima [--from-md PATH] [--role ROLE] [--name NAME]` | Create new Anima |
| `animaworks anima status [ANIMA]` | Show Anima process status |
| `animaworks anima restart ANIMA` | Restart Anima process |
| `animaworks list` | List all Animas |

### Communication

| Command | Description |
|---|---|
| `animaworks chat ANIMA "message" [--from NAME]` | Send message to Anima |
| `animaworks send FROM TO "message"` | Send message between Animas |
| `animaworks heartbeat ANIMA` | Manually trigger heartbeat |

### Configuration & Diagnostics

| Command | Description |
|---|---|
| `animaworks config list [--section SECTION]` | Show configuration |
| `animaworks config get KEY` | Get config value (dot notation: `system.mode`) |
| `animaworks config set KEY VALUE` | Set config value |
| `animaworks status` | Show system status |
| `animaworks logs [ANIMA] [--lines N]` | View logs |

## Execution Modes

Each Anima can use a different LLM model and execution mode. Set via `config.json` per Anima.

| Mode | Engine | Target Models | Tools |
|------|--------|--------------|-------|
| A1 | Claude Agent SDK | Claude models | Read/Write/Edit/Bash/Grep/Glob |
| A1 Fallback | Anthropic SDK | Claude (when Agent SDK unavailable) | search_memory, read/write_file, etc. |
| A2 | LiteLLM + tool_use | GPT-4o, Gemini, etc. | search_memory, read/write_file, execute_command, etc. |
| B | LiteLLM text-based tool loop | Ollama, etc. | Pseudo tool calls (text-parsed JSON) |

## Hierarchy & Roles

- Hierarchy is defined by a single `supervisor` field. No supervisor = top-level.
- Role templates (`--role`) apply specialized prompts, permissions, and defaults:

| Role | Default Model | Description |
|------|--------------|-------------|
| `engineer` | Opus | Complex reasoning, code generation |
| `manager` | Opus | Coordination, decision-making |
| `writer` | Sonnet | Content creation |
| `researcher` | Haiku | Information gathering |
| `ops` | Local model | Log monitoring, routine tasks |
| `general` | Sonnet | General-purpose |

- All communication (directives, reports, coordination) flows through async messaging via Messenger.
- Each Anima runs as an isolated subprocess managed by ProcessSupervisor, communicating via Unix Domain Sockets.

## Web UI

- `http://localhost:18500/` — Dashboard (Anima status, activity timeline, configuration)
- `http://localhost:18500/workspace/` — Interactive Workspace (3D office, chat interface)

## Adding an Anima

One Anima = one directory. Place Markdown files in `~/.animaworks/animas/{name}/`:

```
animas/alice/
├── identity.md          # Personality and expertise (immutable)
├── injection.md         # Role, values, behavioral rules (replaceable)
├── permissions.md       # Tool/file permissions
├── heartbeat.md         # Periodic check schedule
├── cron.md              # Scheduled tasks (YAML)
├── bootstrap.md         # First-run self-setup instructions
├── status.json          # Enabled/disabled, role, model config
├── specialty_prompt.md  # Role-specific expert prompt
├── assets/              # Character images, 3D models
├── activity_log/        # Unified activity log (daily JSONL)
└── skills/              # Skills (YAML frontmatter + Markdown)
```

Or create from a Markdown character sheet:

```bash
animaworks create-anima --from-md character_sheet.md --role engineer --name alice
```

## Tech Stack

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

## Project Structure

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

## About the Author

AnimaWorks is built by a psychiatrist and serial entrepreneur who has been programming since childhood and running real organizations for over a decade.

The core insight — that imperfect individuals collaborating through structure outperform any single omniscient actor — comes from two parallel careers: treating patients who taught him that no mind is complete on its own, and building companies where the right org chart matters more than any individual hire.

## Design Philosophy

For detailed design principles, see [vision.md](docs/vision.md).
For technical specifications, see [spec.md](docs/spec.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
