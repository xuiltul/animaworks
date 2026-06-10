# Digital Anima Requirements Specification v1.4

## 1. Overview

Digital Anima is the minimal unit that encapsulates an AI agent as "a single person."

**Core Design Principles:**

- Internal state is invisible from the outside. The only external interface is text conversation
- Memory is "archive-based." The agent searches for and retrieves only the memories it needs, when it needs them
- Full context is never shared. Information is compressed and interpreted in the agent's own words before communicating
- Heartbeats enable proactive behavior rather than waiting for instructions
- Roles and principles are injected later. Digital Anima itself is an "empty vessel"

**Technical Direction:**

- Agent execution uses **6 modes** (auto-selected from model name patterns via `resolve_execution_mode()`): **S** Claude Agent SDK, **C** Codex CLI, **D** Cursor Agent CLI, **G** Gemini CLI, **A** LiteLLM + tool_use (including Anthropic direct fallback), **B** one-shot assisted (framework handles memory I/O)
- Configuration is unified in **config.json** (Pydantic validation); memories are written in **Markdown**
- User-facing UI strings are resolved via **`core/i18n`** `t()` (hardcoding prohibited)
- Multiple Animas operate collaboratively in a **hierarchical structure** (hierarchy defined by the `supervisor` field, synchronous delegation)

-----

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Digital Anima                      │
│                                                       │
│  Identity ──── Who I am (always resident)             │
│  Agent Core ── 6 execution modes (auto-resolved from model name)│
│    ├ S: Claude Agent SDK                               │
│    ├ C: Codex CLI                                        │
│    ├ D: Cursor Agent CLI                                 │
│    ├ G: Gemini CLI                                       │
│    ├ A: LiteLLM + tool_use (cloud APIs, some Ollama, etc.)│
│    └ B: One-shot assisted (weak models, FW-managed)     │
│  Memory ───── Archive-based long-term memory (recall via autonomous search)│
│    ├ Conversation memory (state/conversation.json, rolling compression)│
│    ├ Short-term memory (shortterm/chat/ and heartbeat/ separated)│
│    └ Unified activity log (activity_log/, JSONL timeline)│
│  Boards ───── Slack-style shared channels             │
│  Permissions ─ Tool/file/command restrictions         │
│  Communication ─ Text + file references               │
│  Lifecycle ── Message receipt / heartbeat / cron      │
│  Injection ── Role/principles/behavior rules (injected later)│
│                                                       │
└──────────────────────────────────────────────────────┘
        ▲                       │
   Text (incoming)         Text (outgoing)
```

-----

## 3. File Structure

```
animaworks/
├── core/
│   ├── anima.py               # DigitalAnima class
│   ├── agent.py               # AgentCore (execution mode selection, cycle management)
│   ├── anima_factory.py       # Anima creation (template/blank/MD)
│   ├── init.py                # Runtime initialization
│   ├── schemas.py             # Data models (Message, CycleResult, etc.)
│   ├── paths.py               # Path resolution
│   ├── messenger.py           # Inter-Anima message send/receive
│   ├── lifecycle/             # Heartbeat, cron, Inbox (package, APScheduler)
│   │   ├── __init__.py        #   LifecycleManager (Scheduler/Inbox/rate limiting mixins)
│   │   ├── scheduler.py       #   Schedule registration
│   │   ├── inbox_watcher.py   #   Inbox monitoring
│   │   ├── rate_limiter.py    #   Message chaining and cooldown
│   │   ├── system_crons.py    #   Server-wide scheduled jobs
│   │   └── system_consolidation.py # Cross-org consolidation triggers
│   ├── outbound.py            # Unified outbound routing (Slack/Chatwork/internal auto-detection)
│   ├── background.py          # Background task management
│   ├── asset_reconciler.py    # Automatic asset generation
│   ├── org_sync.py            # Organization structure sync (status.json → config.json)
│   ├── schedule_parser.py     # cron.md/heartbeat.md parser
│   ├── logging_config.py      # Log configuration
│   ├── memory/                # Memory subsystem (see memory.md for details)
│   │   ├── manager.py         #   Archive-based memory search/write
│   │   ├── conversation.py    #   Conversation memory (entries)
│   │   ├── conversation_*.py  #   Compression, commit, model, prompts, etc. (split modules)
│   │   ├── shortterm.py       #   Short-term memory (chat/heartbeat separated)
│   │   ├── activity.py        #   Unified activity log (JSONL timeline)
│   │   ├── streaming_journal.py #  Streaming journal (WAL)
│   │   ├── priming/           #   Automatic recall layer (multi-source search + deterministic gate)
│   │   ├── action_gate.py     #   Memory check before side-effecting actions
│   │   ├── consolidation.py   #   Memory consolidation (daily/weekly)
│   │   ├── forgetting.py      #   Active forgetting (3 stages)
│   │   ├── reconsolidation.py #   Memory reconsolidation
│   │   ├── task_queue.py      #   Persistent task queue
│   │   ├── taskboard_housekeeping.py # TaskBoard cleanup integration
│   │   ├── resolution_tracker.py # Resolution registry
│   │   ├── rag_search.py      #   Search orchestration
│   │   └── rag/               #   RAG engine (ChromaDB + sentence-transformers)
│   │       ├── indexer.py, retriever.py, graph.py, store.py, http_store.py
│   │       ├── vector_worker_client.py, vector_worker_process.py, vector_worker_server.py
│   │       └── watcher.py, repair.py # File monitoring and RAG repair
│   ├── supervisor/            # Process supervision
│   │   ├── manager.py         #   ProcessSupervisor (child process launch, monitoring)
│   │   ├── ipc.py             #   Unix Domain Socket IPC
│   │   ├── runner.py          #   Anima process runner
│   │   ├── process_handle.py  #   Process handle management
│   │   ├── pending_executor.py #   TaskExec (state/pending/ task execution)
│   │   ├── scheduler_manager.py #  Child-process-side scheduler
│   │   ├── inbox_rate_limiter.py, streaming_handler.py, transport.py, etc.
│   │   └── _mgr_*.py          #   Internal helpers (health, coordination, etc.)
│   ├── notification/          # Human notification
│   │   ├── notifier.py        #   HumanNotifier (call_human integration)
│   │   ├── reply_routing.py   #   Reply routing
│   │   └── channels/          #   Slack, Chatwork, LINE, Telegram, ntfy
│   ├── voice/                 # Voice chat subsystem
│   │   ├── stt.py             #   VoiceSTT (faster-whisper)
│   │   ├── tts_*.py           #   TTS providers (VOICEVOX, ElevenLabs, SBV2)
│   │   └── session.py         #   VoiceSession (STT→Chat IPC→TTS)
│   ├── mcp/                   # stdio MCP (Mode S: tool names `mcp__aw__*`)
│   │   └── server.py
│   ├── config/                # Configuration management
│   │   ├── models.py          #   Public facade (re-exports load_config / load_permissions, etc.)
│   │   ├── schemas.py         #   Pydantic model definitions (AnimaWorksConfig body)
│   │   ├── io.py              #   config.json read/write and cache
│   │   ├── model_mode.py      #   resolve_execution_mode / DEFAULT_MODEL_MODE_PATTERNS
│   │   ├── resolver.py        #   status.json merge resolution
│   │   ├── vault.py           #   VaultManager (~/.animaworks/vault.json + vault.key)
│   │   └── cli.py, etc.       #   migrate, anima_registry, model_config, global_permissions, etc.
│   ├── prompt/                # Prompt and context management
│   │   ├── builder.py         #   System prompt construction (6-group structure)
│   │   ├── assembler.py, sections.py, org_context.py, messaging.py
│   │   └── context.py         #   Context window tracking
│   ├── tooling/               # Tool infrastructure
│   │   ├── handler.py         #   ToolHandler core (dispatch aggregation)
│   │   ├── handler_base.py, handler_memory.py, handler_comms.py, handler_skills.py
│   │   ├── handler_perms.py, handler_org.py, handler_org_dashboard.py
│   │   ├── handler_delegation.py, handler_subordinate_control.py, handler_create_anima.py
│   │   ├── schemas/           #   Tool schemas (domain-specific Python modules)
│   │   ├── guide.py, dispatch.py, permissions.py, skill_tool.py, skill_creator.py
│   │   └── prompt_db.py, org_helpers.py
│   ├── execution/             # Execution engines
│   │   ├── base.py            #   BaseExecutor ABC
│   │   ├── agent_sdk.py       #   Mode S: Claude Agent SDK
│   │   ├── codex_sdk.py       #   Mode C: Codex CLI
│   │   ├── cursor_agent.py    #   Mode D: Cursor Agent CLI
│   │   ├── gemini_cli.py      #   Mode G: Gemini CLI
│   │   ├── anthropic_fallback.py # Inside Mode A: Anthropic SDK direct
│   │   ├── litellm_loop.py    #   Mode A: LiteLLM + tool_use
│   │   ├── assisted.py        #   Mode B: Framework-assisted
│   │   └── _session.py, etc.  #   Session, SDK stream, sanitization, etc.
│   ├── i18n/                  #   User-facing strings (t() / _STRINGS)
│   ├── skills/                # Skill Hub, activation, router, curator, promotion
│   ├── taskboard/             # TaskBoard store, state, cleanup
│   └── tools/                 # External tool implementations
│       ├── web_search.py, x_search.py, slack.py, chatwork.py
│       ├── gmail.py, github.py, google_calendar.py, google_tasks.py
│       ├── discord.py, notion.py, machine.py
│       ├── call_human.py, transcribe.py, aws_collector.py, local_llm.py
│       ├── image_gen.py       #   Images and 3D (image/ subpackage)
│       └── …
├── cli/                       # CLI package
│   ├── parser.py              #   argparse definitions + cli_main()
│   └── commands/              #   Subcommand implementations
├── server/
│   ├── app.py                 # FastAPI (lifespan, middleware, static serving, router registration)
│   ├── slack_socket.py        # Slack Socket Mode client
│   ├── websocket.py           # WebSocketManager (dashboard `/ws`)
│   ├── stream_registry.py     # Chat/SSE stream producer registration and cleanup
│   ├── reload_manager.py      # ConfigReloadManager (config hot reload)
│   ├── room_manager.py        # Meeting rooms (MeetingRoom / RoomManager, `shared/meetings`)
│   ├── localhost.py           # Localhost trust detection (auth bypass)
│   ├── events.py              # Dashboard event emit (WebSocket)
│   ├── dependencies.py        # Compatibility stubs (IPC after process isolation)
│   ├── routes/                # API routes (included under `/api` prefix)
│   │   ├── animas.py, chat.py, sessions.py
│   │   ├── chat_*.py          # Chat handling split (chunk_handler, emotion, images, producer, resume, etc.)
│   │   ├── memory_routes.py, logs_routes.py
│   │   ├── channels.py        # Board / shared channel and DM history
│   │   ├── voice.py           # Voice WebSocket `ws/voice/{name}` (mounted outside `/api` prefix)
│   │   ├── websocket_route.py # Dashboard WebSocket `/ws`
│   │   ├── room.py            # Meeting room REST + SSE
│   │   ├── internal.py        # Internal API (embed/vector, CLI integration notifications, message fetch)
│   │   ├── auth.py            # UI session authentication
│   │   ├── setup.py           # First-run setup API (`/api/setup/*`)
│   │   ├── activity_report.py, brainstorm.py
│   │   ├── external_tasks.py, team_presets.py
│   │   ├── chat_ui_state.py, tool_prompts.py, users.py
│   │   ├── system.py, assets.py, config_routes.py
│   │   ├── webhooks.py        # External messaging webhooks
│   │   └── media_proxy.py     # External image proxy (used from assets routes)
│   └── static/                # Web UI (mounted at `/`, HTML is no-cache)
│       ├── index.html         # SPA shell (`#/chat`, `#/board`, etc.)
│       ├── setup/             # Setup wizard (`/setup`)
│       ├── modules/, pages/, shared/, styles/
│       └── workspace/         # 3D office Workspace (`/workspace`)
├── templates/
│   ├── ja/, en/, ko/          # Locale-specific templates
│   │   ├── prompts/           #   Prompt templates
│   │   ├── anima_templates/   #   Anima scaffolding (_blank)
│   │   ├── roles/             #   Role templates (engineer, researcher, manager, writer, ops, general)
│   │   ├── common_knowledge/  #   Shared knowledge templates
│   │   └── common_skills/     #   Common skill templates
│   └── _shared/               # Locale-independent (organization vision, etc.)
├── main.py                    # CLI entry point
└── tests/                     # Test suite
```

### 3.1 Anima Directory (`~/.animaworks/animas/{name}/`)

Each Anima is composed of the following files and directories:

|File / Directory            |Description                         |
|----------------------------|------------------------------------|
|`identity.md`               |Personality and strengths (immutable baseline)|
|`injection.md`              |Role, principles, behavior rules (replaceable)|
|`permissions.json` / `permissions.md`|Tool/file/command permissions. `permissions.json` takes priority; when not yet migrated, auto-migrate from `permissions.md` (`load_permissions` defined in `core/config/schemas.py`, imported from `core/config/models.py`)|
|`heartbeat.md`              |Periodic check interval and active hours|
|`cron.md`                   |Scheduled tasks (YAML)              |
|`bootstrap.md`              |Self-construction instructions on first launch|
|`status.json`               |Enabled/disabled, role, model settings|
|`specialty_prompt.md`       |Role-specific specialized prompt     |
|`assets/`                   |Character images and 3D models       |
|`transcripts/`              |Conversation transcripts             |
|`skills/`                   |Personal skills (YAML frontmatter + Markdown body)|
|`activity_log/`             |Unified activity log (daily JSONL)   |
|`state/`                    |Working memory (current_state.md, pending.md, pending/, task_queue.jsonl)|
|`episodes/`                 |Episodic memory (daily logs)         |
|`knowledge/`                |Semantic memory (learned knowledge)  |
|`procedures/`               |Procedural memory (runbooks)         |
|`shortterm/`                |Short-term memory (chat/ and heartbeat/ separated, session continuity)|

### 3.2 config.json (Unified Configuration)

All settings are consolidated in `~/.animaworks/config.json`. Validated with the Pydantic `AnimaWorksConfig` model (`core/config/schemas.py`). The encrypted secret store **Vault** is **not** stored in `config.json`; `~/.animaworks/vault.json` and `vault.key` are managed by `VaultManager` in `core/config/vault.py`.

**Top-Level Structure:**

|Section                |Description                          |
|-----------------------|-------------------------------------|
|`version`              |Schema version (integer)             |
|`setup_complete`       |First-run setup completion flag      |
|`locale`               |Default locale (e.g. `ja`)           |
|`system`               |Operation mode, log level            |
|`credentials`           |Per-provider API keys and endpoints (named map)|
|`model_modes`          |※Deprecated. Replaced by `~/.animaworks/models.json`. Referenced as fallback|
|`model_context_windows`|※Backward compatibility. Prefer `context_window` in `models.json`|
|`model_max_tokens`     |Model name pattern (fnmatch) → default `max_tokens` override|
|`anima_defaults`       |Default values applied to all Animas |
|`animas`               |Organization layout (supervisor, speciality) only. Model settings use status.json as SSoT|
|`consolidation`        |Memory consolidation settings (daily/weekly run times and thresholds, daily indexing, etc.)|
|`rag`                  |RAG settings (embedding model, graph spreading activation, retrieval score thresholds, etc.)|
|`prompt`               |System prompt construction (e.g. injection size warning character threshold)|
|`priming`              |Automatic recall settings (per-message-type token budgets)|
|`image_gen`            |Image generation settings (style consistency, Vibe Transfer)|
|`human_notification`   |Human notification settings (channels: Slack/LINE/Telegram/Chatwork/ntfy)|
|`server`               |Server runtime settings (IPC, keep-alive, streaming)|
|`external_messaging`   |External messaging integration (Slack Socket Mode, Chatwork Webhook)|
|`background_task`      |Background tool execution settings (target tools, thresholds, parallel LLM count)|
|`activity_log`         |Log rotation settings (rotation_mode, max_size_mb, max_age_days)|
|`heartbeat`            |Heartbeat interval, timeouts, cascade prevention, idle compaction, etc.|
|`voice`                |Voice chat settings (STT/TTS providers)|
|`housekeeping`         |Periodic disk cleanup settings       |
|`machine`              |Engine priority order for the `machine` external agent tool|
|`workspaces`           |Alias → absolute path (workspace registration)|
|`activity_level`       |Global activity level (10–400%; affects HB interval and max_turns)|
|`activity_schedule`    |Time-of-day `activity_level` (if empty, fixed `activity_level`)|
|`ui`                   |UI theme, demo mode, etc.            |

**Configuration Resolution (2-Layer Merge — status.json SSoT):**

Model settings at Anima startup are resolved in two layers with `status.json` as the Single Source of Truth (SSoT):

1. **Layer 1: status.json** (highest priority) — Model and execution parameters in `animas/{name}/status.json`
2. **Layer 2: config.json anima_defaults** (fallback) — Global defaults in `config.anima_defaults`

The `animas` section in `config.json` holds only the organization layout (`supervisor`, `speciality`).

**AnimaModelConfig Fields (config.json animas):**

|Field         |Type            |Description                        |
|--------------|----------------|-----------------------------------|
|`supervisor`  |`str \| null`   |Name of the supervisory Anima      |
|`speciality`  |`str \| null`   |Free-text area of expertise        |
|`model`       |`str \| null`   |Override (status.json takes priority)|

**status.json Model-Related Fields (SSoT):**

|Field                                |Type            |Default                   |Description                        |
|-------------------------------------|----------------|--------------------------|-----------------------------------|
|`model`                              |`str`           |`claude-sonnet-4-6`       |Model name to use (provider prefix allowed)|
|`fallback_model`                     |`str \| null`   |`null`                    |Fallback model                     |
|`max_tokens`                         |`int`           |`8192`                    |Maximum tokens per response        |
|`max_turns`                          |`int`           |`10000`                   |Maximum turns per cycle            |
|`credential`                         |`str`           |`"anthropic"`             |Credential name to use             |
|`context_threshold`                  |`float`         |`0.50`                    |Threshold for short-term memory externalization (context usage ratio)|
|`max_chains`                         |`int`           |`2`                       |Maximum automatic session continuations|
|`conversation_history_threshold`     |`float`         |`0.30`                    |Compression trigger for conversation memory (context usage ratio)|
|`background_model`                   |`str \| null`   |`null`                    |Lightweight model for heartbeat/inbox/cron. Uses main model when unset|
|`execution_mode`                     |`str \| null`   |`null` (auto-detect)      |`"S"` / `"A"` / `"B"` / `"C"` / `"D"` / `"G"`. Resolved via models.json or DEFAULT_MODEL_MODE_PATTERNS when unset|
|`supervisor`                         |`str \| null`   |`null`                    |Name of the supervisory Anima      |
|`speciality`                         |`str \| null`   |`null`                    |Free-text area of expertise        |

**config.json Example:**

```json
{
  "version": 1,
  "system": { "mode": "server", "log_level": "INFO" },
  "credentials": {
    "anthropic": { "api_key": "", "base_url": null },
    "ollama": { "api_key": "dummy", "base_url": "http://localhost:11434/v1" }
  },
  "anima_defaults": {
    "model": "claude-sonnet-4-6",
    "max_tokens": 4096,
    "max_turns": 10000,
    "credential": "anthropic",
    "context_threshold": 0.50,
    "conversation_history_threshold": 0.30
  },
  "animas": {
    "alice": {},
    "bob": { "model": "gpt-4o", "credential": "openai", "supervisor": "alice" }
  }
}
```

**Security:** config.json is saved with `0o600` permissions (owner read/write only). API key management via environment variables is also supported.

**Context Window Resolution** (`resolve_context_window()` — `core/prompt/context.py`):

1. `~/.animaworks/models.json` `context_window` (highest priority)
2. config.json `model_context_windows` (fnmatch wildcard patterns)
3. Hardcoded default dictionary in code
4. `_DEFAULT_CONTEXT_WINDOW` = 128,000 (final fallback)

The compaction threshold is auto-scaled: for windows ≥ 200K the configured value (default 0.50) is used as-is; below 200K it scales linearly toward 0.98.

### 3.3 Model and Authentication Settings (credentials)

#### Per-Provider credentials Configuration

Define per-provider authentication in the `credentials` section of `config.json` as a named map. Anima's `status.json` references these by key name.

```json
{
  "credentials": {
    "anthropic": {
      "type": "api_key",
      "api_key": "sk-ant-api03-xxxxx",
      "keys": {},
      "base_url": null
    },
    "bedrock": {
      "type": "api_key",
      "api_key": "",
      "keys": {
        "aws_access_key_id": "AKIA...",
        "aws_secret_access_key": "...",
        "aws_region_name": "ap-northeast-1"
      },
      "base_url": null
    },
    "azure": {
      "type": "api_key",
      "api_key": "BKQ5t...",
      "keys": { "api_version": "2025-01-01-preview" },
      "base_url": "https://your-resource.openai.azure.com"
    },
    "vertex": {
      "type": "api_key",
      "api_key": "",
      "keys": {
        "vertex_project": "my-gcp-project",
        "vertex_location": "asia-northeast1",
        "vertex_credentials": "/path/to/service-account.json"
      },
      "base_url": null
    },
    "vllm-gpu": {
      "api_key": "dummy",
      "base_url": "http://localhost:8000/v1"
    }
  }
}
```

vLLM provides an OpenAI-compatible API, so connect with the `openai/` prefix. A dummy `api_key` is required even when authentication is disabled. Anima config: `model: "openai/glm-4.7-flash"`, `credential: "vllm-gpu"`.

#### Model Naming Conventions

Model names include the provider prefix (following LiteLLM naming conventions):

| Provider | Format | Example |
|----------|--------|---------|
| Anthropic direct | `claude-{tier}-{version}` | `claude-opus-4-6`, `claude-sonnet-4-6` |
| AWS Bedrock | `bedrock/{region}.anthropic.claude-{tier}-{version}` | `bedrock/jp.anthropic.claude-sonnet-4-6` |
| Azure OpenAI | `azure/{deployment-name}` | `azure/gpt-4.1-mini` |
| Google Vertex AI | `vertex_ai/{model-name}` | `vertex_ai/gemini-2.5-flash` |
| OpenAI direct | `openai/{model-name}` | `openai/gpt-4.1` |
| Codex | `codex/{model-name}` | `codex/gpt-5.3-codex` |
| Cursor Agent CLI | `cursor/{model-name}` | `cursor/claude-sonnet-4-6` |
| Gemini CLI | `gemini/{model-name}` | (Follows CLI-side model notation) |
| Ollama | `ollama/{model-name}` | `ollama/qwen3:8b` |
| vLLM (local) | `openai/{model-name}` + credential base_url | `openai/glm-4.7-flash` |

#### status.json Model-Related Fields

| Field | Required | Description |
|-------|----------|-------------|
| `model` | Yes | Model name (with prefix above) |
| `credential` | Yes | Key name in config.json `credentials` |
| `execution_mode` | No | Execution mode. Auto-resolved via `DEFAULT_MODEL_MODE_PATTERNS` when unset |
| `mode_s_auth` | No | Authentication method for Mode S (Agent SDK) (`"api"` / `"bedrock"` / `"vertex"`) |

#### execution_mode Auto-Resolution

When `execution_mode` is not set in `status.json`, `resolve_execution_mode()` resolves it in this order:

1. Per-anima explicit override (status.json `execution_mode`)
2. `models.json` (`~/.animaworks/models.json`, user-editable)
3. config.json `model_modes` (deprecated fallback)
4. `DEFAULT_MODEL_MODE_PATTERNS` (code defaults; more specific patterns match first after specificity sort)
5. Default `"B"` (safe fallback)

**Main DEFAULT_MODEL_MODE_PATTERNS mappings** (`core/config/model_mode.py`):

| Pattern | Mode | Description |
|---------|------|-------------|
| `claude-*` | S | Claude direct → Agent SDK |
| `codex/*` | C | Codex → CLI wrapper |
| `cursor/*` | D | Cursor Agent CLI |
| `gemini/*` | G | Gemini CLI |
| `openai/*`, `azure/*`, `bedrock/*`, `vertex_ai/*`, `google/*`, `mistral/*`, `xai/*`, `cohere/*`, `zai/*`, `minimax/*`, `moonshot/*`, `deepseek/deepseek-chat`, etc. | A | Cloud APIs, etc. → LiteLLM + tool_use |
| `ollama/qwen3.5*`, `ollama/qwen3:*` (some sizes), `ollama/qwen3-coder:*`, `ollama/llama4:*`, `ollama/mistral-small3.2:*`, `ollama/devstral*`, `ollama/glm-4.7*`, `ollama/glm-5*`, `ollama/minimax*`, `ollama/kimi-k2*`, `ollama/gpt-oss*`, etc. | A | Ollama models with proven tool_use support |
| `ollama/qwen3:0.6b`–`8b`, `ollama/gemma3*`, `ollama/deepseek-r1*`, `ollama/deepseek-v3*`, `ollama/phi4*`, etc. | B | Weaker or reasoning-specialized models with unstable tools → Basic |
| `ollama/*` | B | Other Ollama → Basic (safe fallback) |

**Note:** `bedrock/*` defaults to Mode A. For Mode S, explicitly set both `"execution_mode": "S"` and `"mode_s_auth": "bedrock"`.

#### Configuration Pattern Examples

**Claude Opus (Anthropic Max Plan):**
```json
{ "model": "claude-opus-4-6", "credential": "anthropic" }
```

**Claude Sonnet (via AWS Bedrock + Mode S):**
```json
{
  "model": "bedrock/jp.anthropic.claude-sonnet-4-6",
  "credential": "bedrock",
  "execution_mode": "S",
  "mode_s_auth": "bedrock"
}
```

**Azure OpenAI:**
```json
{ "model": "azure/gpt-4.1-mini", "credential": "azure" }
```

**RAGConfig Fields:**

|Field                          |Type      |Default                           |Description                        |
|-------------------------------|----------|----------------------------------|-----------------------------------|
|`enabled`                      |`bool`    |`true`                            |Enable/disable RAG functionality   |
|`embedding_model`              |`str`     |`intfloat/multilingual-e5-small`  |Embedding model to use             |
|`use_gpu`                      |`bool`    |`false`                           |Whether to use GPU                 |
|`enable_spreading_activation`  |`bool`    |`true`                            |Enable/disable graph-based spreading activation|
|`max_graph_hops`               |`int`     |`2`                               |Maximum hops for graph traversal   |
|`enable_file_watcher`          |`bool`    |`true`                            |Enable/disable file change monitoring|
|`graph_cache_enabled`          |`bool`    |`true`                            |Enable/disable graph cache         |
|`implicit_link_threshold`      |`float`   |`0.75`                            |Similarity threshold for implicit link generation|
|`spreading_memory_types`       |`list`    |`["knowledge", "episodes"]`        |Memory types targeted by spreading activation|
|`min_retrieval_score`          |`float`   |`0.30`                            |Minimum retrieval score (drop hits below)|
|`skill_match_min_score`        |`float`   |`0.75`                            |Minimum similarity for legacy/auxiliary skill matching|

**PromptConfig Fields (excerpt):**

|Field                          |Type    |Default |Description                        |
|-------------------------------|--------|--------|-----------------------------------|
|`injection_size_warning_chars` |`int`   |`5000`  |Character threshold to warn when injection-related files are too long|

**PrimingConfig Fields:**

|Field                 |Type    |Default |Description                        |
|----------------------|--------|--------|-----------------------------------|
|`dynamic_budget`      |`bool`  |`true`  |Compatibility field; normal chat paths use `enable_dynamic_budget` and the `budget_*` values|
|`budget_greeting`     |`int`   |`500`   |Token budget for greeting messages |
|`budget_question`     |`int`   |`2000`  |Token budget for question messages |
|`budget_request`      |`int`   |`3000`  |Token budget for request messages  |
|`budget_heartbeat`    |`int`   |`200`   |Token budget for heartbeat (fallback)|
|`heartbeat_context_pct`|`float`|`0.05`  |HB context ratio when dynamic budget (5%)|

**ServerConfig Fields:**

|Field                          |Type     |Default |Description                        |
|-------------------------------|---------|--------|-----------------------------------|
|`session_ttl_days`             |`int \| null`|`7` (`null` = unlimited)|UI session cookie TTL |
|`ipc_stream_timeout`           |`int`    |`60`    |IPC streaming per-chunk timeout (seconds)|
|`keepalive_interval`           |`int`    |`30`    |Keep-alive send interval (seconds) |
|`max_streaming_duration`       |`int`    |`1800`  |Maximum streaming duration (seconds)|
|`busy_hang_threshold`          |`int`    |`900`   |Seconds before child process "busy" is treated as unresponsive (reflected in `HealthConfig`)|
|`stream_checkpoint_enabled`    |`bool`   |`true`  |Save tool results during streaming |
|`stream_retry_max`             |`int`    |`3`     |Maximum auto-retry count on stream disconnection|
|`stream_retry_delay_s`         |`float`  |`5.0`   |Wait time between retries (seconds)|
|`llm_num_retries`              |`int`    |`3`     |LLM API call retries (429/5xx/network)|
|`media_proxy`                  |`object` |(defaults)|External image proxy mode, allowed domains, rate limits, etc. (`MediaProxyConfig`)|

**ExternalMessagingConfig Fields:**

|Field                 |Type          |Default    |Description                        |
|----------------------|--------------|-----------|-----------------------------------|
|`preferred_channel`   |`str`         |`"slack"`  |Preferred send channel (`"slack"` or `"chatwork"`)|
|`user_aliases`        |`dict`        |`{}`       |User alias to contact info mapping |
|`slack`               |`object`      |           |Slack settings (enabled, mode, anima_mapping)|
|`chatwork`            |`object`      |           |Chatwork settings (enabled, mode, anima_mapping)|

**BackgroundTaskConfig Fields:**

|Field                     |Type    |Default |Description                        |
|--------------------------|--------|--------|-----------------------------------|
|`enabled`                 |`bool`  |`true`  |Enable/disable background execution|
|`eligible_tools`          |`dict`  |        |Target tool name → `{ "threshold_s": int }` (`BackgroundToolConfig`)|
|`result_retention_hours`  |`int`   |`24`    |Result retention period (hours)    |
|`max_parallel_llm_tasks`  |`int`   |`3`     |Concurrent LLM cap for `submit_tasks`, etc. (1–10)|

**ActivityLogConfig Fields (supplement):**

|Field              |Type   |Default |Description                        |
|-------------------|-------|--------|-----------------------------------|
|`rotation_enabled` |`bool` |`true`  |Enable/disable rotation processing |

**HeartbeatConfig Fields (supplement):**

|Field                      |Type         |Default |Description                        |
|---------------------------|-------------|--------|-----------------------------------|
|`interval_minutes`         |`int`        |`30`    |Heartbeat interval in minutes (config-driven, not heartbeat.md)|
|`soft_timeout_seconds`     |`int`        |`300`   |Seconds before wrap-up reminder in HB session|
|`hard_timeout_seconds`     |`int`        |`600`   |Seconds before forced HB session termination|
|`max_turns`                |`int \| null`|`null`  |HB-specific `max_turns` (falls back to Anima settings when unset)|
|`default_model`            |`str \| null`|`null`  |Global background model for heartbeat/cron (falls back to per-Anima `background_model` / main)|
|`msg_heartbeat_cooldown_s` |`int`      |`300`   |Cooldown for message-triggered HB (seconds)|
|`cascade_window_s`         |`int`        |`1800`  |Sliding window for cascade detection (seconds)|
|`cascade_threshold`        |`int`        |`3`     |Max round-trips per pair within the window|
|`depth_window_s`           |`int`        |`600`   |Window for bidirectional depth limiting (seconds)|
|`max_depth`                |`int`        |`6`     |Max bidirectional exchange depth|
|`actionable_intents`       |`list`       |`["report","question"]`|Intents eligible for message-triggered HB|
|`idle_compaction_minutes`  |`float`      |`10`    |Minutes after last stream end before idle auto-compact|
|`enable_read_ack`          |`bool`       |`false` |Send read ACKs (default off; reduces thank-you loops)|
|`channel_post_cooldown_s`  |`int`        |`300`   |Minimum seconds between Board posts per Anima (`0` = unlimited)|
|`max_messages_per_hour`, `max_messages_per_day`|`int`|`30` / `100`|※Rate limits prefer `ROLE_OUTBOUND_DEFAULTS` + status.json; fields kept for backward compatibility|

### 3.4 HTTP Server (`server/`) Structure and Behavior

`server/app.py` assembles the FastAPI application. Main elements:

**Routing**

- `create_router()` (`server/routes/__init__.py`) groups REST under **`/api`** (animas, chat, channels, memory, sessions, system, config, logs, assets, internal, auth, users, room, webhooks, etc.).
- **`/ws`** — Dashboard WebSocket (`websocket_route.py` → `WebSocketManager`).
- **`/ws/voice/{name}`** — Voice chat (`voice.py`, mounted outside the `/api` prefix).
- **`create_setup_router()`** — First-run setup **`/api/setup/*`**. After completion, middleware returns 403 for non-setup routes and the normal UI redirects `/setup` to `/`.

**Static files**

- **`/setup`** — `static/setup/` (wizard only).
- **`/`** — Entire `static/` tree (`index.html`, etc.). **Cache-Control: no-cache** for `.js` / `.css` / `.html` and for `/` and `/workspace` to ease updates.

**Middleware (approximate application order)**

1. **RequestLoggingMiddleware** — Pure ASGI. Binds `X-Request-ID` (or generated ID) to structlog context. Suppresses logging for `/api/system/health`, etc.
2. **static_cache_control** — Cache headers above.
3. **setup_guard** — While `config.setup_complete` is false, blocks everything except `/api/setup` and `/setup`, redirecting to `/setup/`.
4. **auth_guard** — `core.auth` `auth_mode`. `local_trust` is unauthenticated. Otherwise validates session cookies for `/api/*`, `/ws`, `/ws/*` (exceptions: login, setup, **public icons** `GET /api/animas/{name}/assets/icon*.png`, **verified localhost** when `trust_localhost`).

**Lifecycle (when `setup_complete`)**

- Startup: Load **`permissions.global.json`** (fatal error if missing). `WebSocketManager` heartbeat; `StreamRegistry` cleanup loop.
- **APScheduler (in server process)**: Orphan Anima detection, periodic asset reconciliation, Claude CLI/SDK auto-update checks, global permissions file consistency checks, etc.
- **Background tasks** (after server start, UI responsiveness first): Launch all Anima child processes, frontmatter migration, org sync, Slack Socket Mode, `ConfigReloadManager` registration, asset reconciler.
- Child-process environment variables **`ANIMAWORKS_EMBED_URL` / `ANIMAWORKS_VECTOR_URL`** — **Centralize** RAG embedding and vector operations over **HTTP on the server** (reduces model residency in child processes).

**Internal API (`routes/internal.py`, via `/api` prefix)**

- **`POST /internal/embed`** — Embedding inference (HTTP from child processes).
- **`POST /internal/vector/*`** — Chroma-compatible query/upsert/delete, etc. (per collection).
- **`POST /internal/message-sent`** — Send notifications from CLI, etc. (WebSocket broadcast).
- **`GET /messages/{message_id}`** — Look up message JSON from shared inbox store.

**Meeting rooms**

- `RoomManager` persists `shared/meetings`. `room.py` provides creation, participant management, and meeting chat with **SSE streaming** (max participants enforced by request validation).

-----

## 4. Memory System (Archive-Based)

### 4.1 Design Philosophy

Traditional AI agents mechanically truncate memory and pack it into prompts (truncation-based). This is equivalent to "anterograde amnesia where only the most recent memories exist."

The archive-based approach is different. Just as a person retrieves the documents they need from an archive, **Digital Anima searches for and retrieves only the memories it needs, when it needs them.** There is no upper limit on memory capacity. Only "what is needed right now" enters the context.

### 4.2 Correspondence with Neuroscience Models

```
┌─────────────────────────────────────────────────┐
│  Working Memory (Prefrontal Cortex)             │
│  = Context Window                               │
│  Limited capacity. Temporary holding of          │
│  "what I'm currently thinking about"             │
│  → Delegated to the SDK. No additional           │
│    implementation needed                         │
└──────────────────┬──────────────────────────────┘
                    │ Recall (search) / Encoding (write)
┌──────────────────┴──────────────────────────────┐
│  Long-Term Memory (Cerebral Cortex /             │
│  Hippocampal System)                             │
│                                                    │
│  episodes/   Episodic memory — what happened     │
│              and when                             │
│  knowledge/  Semantic memory — lessons and        │
│              knowledge learned                    │
│  procedures/ Procedural memory — work runbooks    │
└────────────────────────────────────────────────┘
```

### 4.3 Role of Memory Directories

|Directory                     |Brain Analog              |Contents              |Update Method                |
|------------------------------|--------------------------|----------------------|-----------------------------|
|`state/`                      |Persistent part of working memory|Current state, incomplete tasks|Overwritten each cycle       |
|`state/conversation.json`     |Conversation memory       |Rolling conversation history|LLM summarization when threshold exceeded|
|`shortterm/chat/`             |Short-term memory (chat)  |Context carry-over    |Auto-externalized on session switch|
|`shortterm/heartbeat/`        |Short-term memory (heartbeat)|Context carry-over    |Managed separately for chat vs heartbeat|
|`episodes/`                   |Episodic memory (hippocampus)|Daily action logs     |Appended to date-based files |
|`knowledge/`                  |Semantic memory (temporal cortex)|Lessons, rules, counterpart traits|Created/updated by topic     |
|`procedures/`                 |Procedural memory (basal ganglia)|Work runbooks         |Revised as needed            |

### 4.4 Overview of Memory Operations

- **Recall (remember)**: Always search the archive before deciding (`knowledge/` → `episodes/` → `procedures/`)
- **Encoding (write)**: After action, append logs to `episodes/`; record new knowledge in `knowledge/`
- **Consolidation (reflection)**: Transfer from episodic to semantic memory (automatic daily/weekly)
- **Forgetting**: Staged archival of low-activity chunks (3 stages: downscaling → reorganization → forgetting)
- **Priming (automatic recall)**: Multi-source parallel memory search with a deterministic gate, injected into the system prompt

The key to success is a strong system-prompt instruction that **deciding without searching memory is forbidden**.

> For technical details of the memory system (conversation memory, short-term memory, activity log, streaming journal, Priming, RAG, consolidation, and forgetting subsystems), see **[memory.md](memory.md)**.

-----

## 5. Identity (Self-Definition)

The information that allows a Digital Anima to recognize "who it is." Always resident in working memory.

```markdown
# Identity: Tanaka

## Personality Traits
- Cautious, considers risks first
- Detail-oriented, dislikes ambiguity

## Perspective
Prioritizes technical feasibility. "Will it actually work?" is always the starting point for decisions.

## Strengths
- Backend design, performance optimization

## Weaknesses
- UI/UX design decisions, understanding users' emotional needs
```

-----

## 6. Permissions

Restrictions on "what a Digital Anima can do." Permission restrictions create "limited visibility," which produces dependence on others — and that dependence is what makes the organization valuable.

```markdown
# Permissions: Tanaka

## Available Tools
Read, Write, Edit, Bash, Grep, Glob

## Unavailable Tools
WebSearch, WebFetch

## Readable Locations
- Under /project/src/backend/
- Under /project/docs/
- Under /shared/reports/

## Writable Locations
- Under /project/src/backend/
- Under /workspace/Tanaka/

## Invisible Locations
- /project/.env
- Under /project/src/frontend/ (Suzuki's jurisdiction)

## Allowed Commands
npm test, npm run build, git diff, git log

## Disallowed Commands
git push (requires approval), rm -rf, docker
```

Since the frontend code is unreadable, the agent needs to ask a colleague, "What are the frontend constraints?" This "asking because I don't know" is what drives horizontal communication in the organization.

-----

## 7. Communication

### Principles

- Text and file references only. Direct sharing of internal state is prohibited
- Compress and interpret information in your own words before communicating. Never send the full context
- For lengthy content, save it as a file and communicate: "I've placed it here, please take a look"

### Message Structure

```json
{
  "id": "20260213_100000_abc",
  "thread_id": "",
  "reply_to": "",
  "from_person": "Tanaka",
  "to_person": "Suzuki",
  "content": "I've revised the auth API design. I placed it in auth-api-design.md, please review.",
  "intent": "report",
  "source": "internal",
  "timestamp": "2026-02-13T10:00:00Z"
}
```

### Intent (Message Purpose)

Use the `intent` field on `send_message` to state the purpose. Use DMs only for meaningful communication; acknowledgments and thanks belong on the Board (shared channel).

| intent | Description | Use |
|--------|-------------|-----|
| `report` | Report upward | Progress and issues (MUST) |
| `delegation` | Delegate to subordinate | Works with delegate_task |
| `question` | Question or consultation | Coordination with peers |

### Board (Shared Channels)

Slack-style shared channels stored in `shared/channels/{name}.jsonl` as append-only JSONL. Use the Board for acknowledgments, thanks, FYIs, and notifying three or more people.

### Rate Limiting (3 Layers)

Three layers to prevent excessive DM traffic:

| Layer | Limit | Implementation |
|-------|-------|------------------|
| per-run | No duplicate sends to same recipient; max 2 people per run | `_replied_to`, `_posted_channels` |
| cross-run | 30 messages/hour, 100 messages/day | activity_log sliding window |
| behavior-awareness | Recent send history injected via Priming | `PrimingEngine._collect_recent_outbound()` |

`ack`, `error`, `system_alert`, and `call_human` are exempt from rate limits.

### Communication Routing Rules

|Situation              |Recipient           |Notes                        |
|-----------------------|--------------------|-----------------------------|
|Progress/issue reports |Supervisor          |MUST                         |
|Task delegation        |Direct subordinate  |Use delegate_task            |
|Coordination           |Peer (same supervisor)|Direct communication OK      |
|Cross-department contact|Via own supervisor  |Direct contact prohibited in principle|
|Contacting humans      |call_human          |Top-level Anima responsibility|

Suzuki sees only the design document. Tanaka's thought process and discarded alternatives are invisible. This information asymmetry is what enables fresh perspectives from different backgrounds.

-----

## 8. Lifecycle

### 8.1 Activation Triggers and Execution Paths

A Digital Anima has its own internal clock. The four execution paths hold independent locks and can run in parallel.

|Path         |Lock |Trigger |Role |
|-------------|-----|--------|-----|
|**Chat/Inbox** | `_conversation_lock` / `_inbox_lock` | Human chat / Anima DM | Message response. Inbox: immediate, lightweight replies only |
|**Heartbeat** | `_background_lock` | Periodic check (30 min) | Observe → Plan → Reflect. Does not execute |
|**Cron** | `_background_lock` | cron.md schedule | Same context as Heartbeat; scheduled task execution |
|**TaskExec** | `_background_lock` | Task appears in state/pending/ | Delegated task execution (minimal context) |

Heartbeat only observes and plans; tasks to execute are written to `state/pending/` in JSON format. TaskExec detects them via 3-second polling and runs them in a separate LLM session.

### 8.2 Heartbeat

The act of periodically "looking up and scanning the surroundings" at regular intervals. Executes while retaining the main context; does nothing if there is nothing to address.

```markdown
# Heartbeat: Tanaka

## Execution Interval
Every 30 minutes

## Active Hours
9:00 - 22:00 (JST)

## Checklist
- Are there unread messages in the inbox?
- Has a blocker arisen for any in-progress task?
- Have new files been placed in my work area?
- If nothing, do nothing (HEARTBEAT_OK)

## Notification Rules
- Notify relevant parties only when deemed urgent
- Do not repeat the same notification within 24 hours
```

### 8.3 cron

Performs predetermined tasks at predetermined times on its own clock. Unlike heartbeat, it always executes something and produces a result.

cron does not depend on external schedulers or organizational structure. **Each Digital Anima owns its own cron.** Just as a person has their own habit of writing a diary every morning.

Defined in `cron.md` using Markdown + YAML format. Standard 5-field cron expressions (Asia/Tokyo timezone):

```markdown
## Morning Work Planning
schedule: 0 9 * * *
type: llm
Review yesterday's progress from episodes/ and plan today's tasks.

## Weekly Retrospective
schedule: 0 17 * * 5
type: llm
Re-read this week's episodes/ and extract patterns, consolidating them into knowledge/.

## Backup Execution
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

- **LLM type** (`type: llm`): The agent executes with judgment and reasoning (incurs API cost)
- **Command type** (`type: command`): Runs bash or internal tools deterministically (no API cost)
- **Hot-reload**: cron.md changes are automatically picked up on the next execution cycle

**Differences Between Heartbeat and cron:**

|Aspect      |Heartbeat              |cron                   |
|------------|-----------------------|-----------------------|
|Human analog|Occasionally checking email while working|Morning routine, weekly retrospective|
|Context     |Retained               |Not retained (new session)|
|Decision    |"Is there anything I should care about?"|Executed unconditionally|
|When idle   |Does nothing           |Always produces output |
|Ownership   |Internal to the individual|Internal to the individual|

### 8.4 Flow of a Single Cycle

```
Activation (message or heartbeat or cron)
  ↓
Recall: Search the archive for relevant memories
  ↓
Think & Act: Agent Core (S/C/D/G/A/B modes) processes
  ↓
Communicate: Summarize results and send as text or create files
  ↓
Encode: Write action logs, lessons, and knowledge
  ↓
Update state: Update state/
  ↓
Rest
```

-----

## 9. Injectable Slot (Post-Injection)

Digital Anima is an "empty vessel." Roles and principles are injected via Markdown.

```markdown
# Injection: Tanaka

## Role
Tech Lead. Responsible for technical decision-making and code review.
Area of responsibility: backend architecture.

## Principles
Solve user problems through high-quality software.

## Code of Conduct
- Never compromise on quality
- Pursue simplicity
- When in doubt, return to "What is best for the user?"

## Things Not To Do
- Direct access to the production database
- Frontend implementation (delegate to Suzuki)
- Pushing to the main branch without approval
```

-----

## 10. System Prompt Construction

The various Markdown files and templates are combined to build a single system prompt. `build_system_prompt()` in `core/prompt/builder.py` is the hub for the 6-group structure; section assembly works with `assembler.py`, `sections.py`, and related modules. The `trigger` parameter (`chat` / `inbox` / `heartbeat` / `cron` / `task`) selects sections according to the execution path.

```
Group 1: Operating environment and behavior rules
  - environment.md (guardrails, folder structure)
  - Current time (JST)
  - behavior_rules (search before deciding)
  - tool_data_interpretation.md (trust level interpretation for tool results and Priming)

Group 2: Who you are
  - bootstrap.md (first-launch instructions — conditional)
  - company/vision.md (organization vision)
  - identity.md (personality)
  - injection.md (role, behavior guidelines)
  - specialty_prompt.md (role-specific specialized prompt)
  - permissions.json (preferred) / auto-migrated from permissions.md when not migrated (tool, file, command permissions)

Group 3: Current situation
  - state/current_state.md + pending.md (in-progress tasks)
  - TaskBoard / task queue (current, deferred, suppressed, background, fallback queue — conditional)
  - Resolution Registry (resolved issues, last 7 days — conditional)
  - Recent Outbound (send history, last 2 hours, max 3)
  - Priming (RAG automatic recall — conditional)
  - Recent Tool Results (conditional)

Group 4: Memory and capabilities
  - memory_guide (memory directory guide)
  - common_knowledge hint (shared reference hints — conditional)
  - Hiring rules (when newstaff skill present — conditional)
  - Tool guide (per execution mode)
  - External tool guide (when permitted — conditional)

Group 5: Organization and communication
  - hiring context (when solo top-level — conditional)
  - org context (organization structure tree)
  - messaging instructions
  - human notification guidance (top-level & when notifications enabled — conditional)

Group 6: Meta settings
  - emotion metadata (facial expression metadata instructions)
  - A reflection (self-correction prompt for A mode — conditional)
```

**Tiered System Prompt:** Adjusts prompt content in 4 tiers based on context window (T1 FULL 128k+ / T2 STANDARD 32k–128k / T3 LIGHT 16k–32k / T4 MINIMAL <16k).

**Skill injection (progressive disclosure):** Skills are not loaded wholesale by the main priming body. Active skill context, the Skill Router, Skill Hub, and `read_memory_file` provide the body or pointer only when needed. Message-type budgets still apply to the surrounding priming context: greeting=500, question=2000, request=3000, heartbeat=200 (`PrimingConfig` defaults).

Including "Making decisions without searching memory is prohibited" in `behavior_rules` is the key to the success of archive-based memory (validated experimentally).

-----

## 11. Implemented Features

- **Digital Anima class** — Encapsulation and autonomous operation. 1 Anima = 1 directory
- **6 execution modes** — S: Agent SDK / C: Codex CLI / D: Cursor Agent CLI / G: Gemini CLI / A: LiteLLM + tool_use (including Anthropic direct fallback) / B: Assisted (one-shot)
- **Background model** — heartbeat/inbox/cron can run on a separate lightweight model from the main model (cost optimization)
- **ProcessSupervisor** — Launches and monitors each Anima as an independent child process with Unix socket
- **Archive-based memory** — episodes / knowledge / procedures / state. Details in **[memory.md](memory.md)**
- **Priming (multi-source automatic recall)** — sender profile, recent activity, important knowledge, related knowledge, TaskBoard/task state, episodes, graph context, recent outbound, pending human notifications
- **Action memory gate** — side-effecting actions verify related memory and `[ACTION-RULE]` context before execution
- **Memory consolidation and forgetting** — Daily/weekly consolidation; 3-stage forgetting (synaptic homeostasis hypothesis–based)
- **Board/shared channels** — Slack-style shared channels. REST API for channel posts, mentions, DM history
- **Unified outbound routing** — Auto-resolves recipient names to internal Anima or external platforms (Slack/Chatwork) for delivery
- **Heartbeat, cron, TaskExec, TaskBoard** — Schedule management via APScheduler. TaskBoard and TaskExec manage queued, running, deferred, suppressed, and completed work
- **Inter-Anima messaging** — Text communication via Messenger. intent control (report/delegation/question); 3-layer rate limiting
- **Supervisor tools** — Auto-enabled for Animas with subordinates (see tool list below)
- **Unified configuration** — config.json + Pydantic validation. status.json SSoT; models.json for execution mode override
- **Credential Vault** — `vault.json` + `vault.key` (PyNaCl SealedBox, `core/config/vault.py`). Tools: `vault_get` / `vault_store` / `vault_list`
- **Common tools directory** — Scans `~/.animaworks/common_tools/*.py` and loads when names do not collide with core tools (`core/tools/__init__.py`)
- **Skill Hub / Curator** — Skill installation, activation, quarantine, promotion from procedures, and usage-based review
- **FastAPI server** — REST (`/api`) + dashboard WebSocket (`/ws`) + voice (`/ws/voice/{name}`) + first-run setup wizard (`/setup`) + SPA (`#/chat`, etc.) + Workspace (`/workspace`). Internal embed/vector API centralizes child-process RAG; meeting room API + SSE; `StreamRegistry` / `ConfigReloadManager`; Slack Socket Mode integration
- **Voice chat** — WebSocket /ws/voice/{name}. STT (faster-whisper) → Chat IPC → TTS (VOICEVOX/ElevenLabs/SBV2)
- **Anima creation** — From template / blank (_blank) / MD file (create --from-md)
- **Skill progressive disclosure** — Active skill context, Skill Router, and `skill` / `read_memory_file` load full text only when needed
- **External messaging integration** — Slack Socket Mode (real-time bidirectional), Chatwork Webhook (inbound)
- **TaskBoard / persistent task queue** — TaskBoard is primary for current, processing, deferred, suppressed, background, and completed work; task_queue.jsonl remains as compatibility fallback. Includes staleness detection, DAG parallel execution (`submit_tasks`), and delegation prompt context
- **Resolution registry** — Cross-Anima issue resolution tracking via shared/resolutions.jsonl
- **Human notification** — call_human integration. Slack, Chatwork, LINE, Telegram, ntfy channels
- **External tools** — web_search, x_search, slack, chatwork, gmail, github, google_calendar, google_tasks, discord, notion, machine, transcribe, aws_collector, local_llm, image_gen, call_human, etc. (via `permissions` and `ExternalToolDispatcher`)

### 11.1 Internal Tools Catalog

Internal tools provided by the framework. Combines Claude Code–compatible tools with AnimaWorks-specific tools. Mode S uses MCP; Mode C/D/G use each CLI's tool path; Mode A/B use native tool_use, etc. External integrations are mainly via `Bash` + `animaworks-tool` CLI.

**Memory:**

| Tool | Description |
|------|-------------|
| `search_memory` | Archive memory search (scope: knowledge/episodes/procedures/common_knowledge/activity_log/all) |
| `read_memory_file` | Read a memory file |
| `write_memory_file` | Write a memory file |
| `archive_memory_file` | Archive a memory file |
| `report_procedure_outcome` | Feedback on procedural memory execution |
| `report_knowledge_outcome` | Feedback on knowledge usefulness |

**Communication:**

| Tool | Description |
|------|-------------|
| `send_message` | Inter-Anima DM (intent required: report/delegation/question) |
| `post_channel` | Post to Board (shared channel) |
| `read_channel` | Read Board |
| `manage_channel` | Channel management (create, ACL) |
| `read_dm_history` | Read DM history |
| `call_human` | Notify humans (Slack/LINE/Telegram/Chatwork/ntfy) |

**Tasks:**

| Tool | Description |
|------|-------------|
| `backlog_task` | Add to TaskBoard / fallback task queue (human-origin tasks highest priority) |
| `submit_tasks` | DAG batch of tasks (dependency resolution, parallel execution) |
| `update_task` | Update task state |
| `list_tasks` | List tasks (tool or CLI depending on mode) |

**Skills:**

| Tool | Description |
|------|-------------|
| `skill` | Skill lookup (progressive disclosure: names only → full text on demand) |
| `create_skill` | Create a new skill |

**Vault:**

| Tool | Description |
|------|-------------|
| `vault_get` | Retrieve a secret |
| `vault_store` | Store a secret |
| `vault_list` | List secrets |

**Background tasks:**

| Tool | Description |
|------|-------------|
| `check_background_task` | Check background task status |
| `list_background_tasks` | List background tasks |

**Supervisor** (auto-enabled for Animas with subordinates):

| Tool | Scope | Description |
|------|-------|-------------|
| `org_dashboard` | All descendants | Tree of process state, tasks, activity |
| `ping_subordinate` | All descendants | Liveness check (omit name for all at once) |
| `read_subordinate_state` | All descendants | Read current_state.md + pending.md |
| `delegate_task` | Direct subordinates | Delegate task (add to subordinate queue + DM) |
| `task_tracker` | Own delegated tasks | Track progress |
| `audit_subordinate` | All descendants | Activity summary, error frequency, tool usage stats |
| `disable_subordinate` | All descendants (children, grandchildren, …) | Suspend (`_check_descendant`) |
| `enable_subordinate` | All descendants | Resume |
| `set_subordinate_model` | All descendants | Change model |
| `set_subordinate_background_model` | All descendants | Change background model |
| `restart_subordinate` | All descendants | Restart process |

**Other:**

| Tool | Description |
|------|-------------|
| `check_permissions` | Permission check |
| `create_anima` | Create new Anima (when newstaff skill is held) |

-----

## 12. Design Decision Log

|Decision                               |Rationale                                                           |
|---------------------------------------|--------------------------------------------------------------------|
|Memory format: JSON → Markdown files   |Experiments showed that AI reads and writes Markdown more naturally, with better Grep search compatibility|
|Forgetting: score-based → [IMPORTANT] tag + consolidation|Simple tag-based approach is more practical. Consolidation naturally organizes importance|
|config.md → config.json                |From per-anima MD to unified JSON. Pydantic validation + per-anima overrides|
|Do not build the agent loop ourselves  |Delegate to the Claude Agent SDK. No reinventing the wheel         |
|6 execution modes (S/C/D/G/A/B)      |Claude SDK, Codex/Cursor/Gemini CLI, LiteLLM general use, Assisted for weak models. All within the Anima capsule (auto-resolved from model name patterns)|
|agent.py refactoring                   |Split into execution/, tooling/, memory/. ProcessSupervisor for child process launch|
|Package lifecycle as a package         |Split scheduler, Inbox, rate limiting, and system cron/consolidation into `core/lifecycle/` modules|
|Config schema extensions               |`prompt` / `machine` / `workspaces` / `activity_level` / `activity_schedule` / `ui`, etc. Vault lives outside config in `vault.json`|
|Centralize i18n                        |User-facing strings go through `core/i18n` `t()` (no hardcoding in code)|
|Permissions as "limited visibility"   |Not knowing things forces asking others. Omniscience makes organizations meaningless|
|Archive-based memory adopted           |Truncation-based (packing the last N entries into the prompt) does not scale memory. Archive-based has no upper limit on memory capacity|
|cron as the "individual's" internal clock|cron is not an organizational scheduler; each Digital Anima owns its own habits. Just as a person has their own daily routine|
