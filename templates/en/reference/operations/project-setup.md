# Project Setup

Reference for AnimaWorks configuration structure and procedures for adding Anima.
Search and refer to this when you need to change settings.

## Runtime Directory Initialization

Runtime data is stored in `~/.animaworks/` (or `ANIMAWORKS_DATA_DIR`).
On first setup, run `animaworks init` to initialize from templates.

### init Commands

| Command | Description |
|---------|------|
| `animaworks init` | Initialize runtime directory (non-interactive). Does nothing if already exists |
| `animaworks init --force` | Merge template diffs into existing data. Adds new files only; overwrites `prompts/` |
| `animaworks init --skip-anima` | Initialize infrastructure only. Skip Anima creation |
| `animaworks init --template NAME` | Create Anima from template (non-interactive) |
| `animaworks init --from-md PATH` | Create Anima from MD file (non-interactive) |
| `animaworks init --blank NAME` | Create blank Anima (non-interactive) |

**Recommended flow**: Run `animaworks init`, then start the server with `animaworks start`, and add Anima via the web UI setup wizard.

### Directories Created on Initialization

`ensure_runtime_dir` (`core/init.py`) creates the following:

- `animas/` — Anima directories
- `shared/inbox/` — Incoming message queue
- `shared/users/` — User profiles
- `shared/channels/` — Shared channels (initial files for general, ops)
- `shared/dm_logs/` — DM history (fallback)
- `tmp/attachments/` — Temporary attachment storage
- `common_skills/` / `common_knowledge/` — Shared skills and knowledge
- `prompts/` / `company/` — Prompt and organization templates
- `tool_prompts.sqlite3` — Tool prompt DB
- `models.json` — Model name → execution mode mapping (copied from `config_defaults/`)

On each startup, `common_skills` and `common_knowledge` are incrementally synced from templates; only new entries are added (existing files are preserved).

## config.json Structure

AnimaWorks' unified config file is at `~/.animaworks/config.json`.
All settings are defined by the `AnimaWorksConfig` model with these top-level fields:

```json
{
  "version": 1,
  "setup_complete": true,
  "locale": "ja",
  "system": { "mode": "server", "log_level": "INFO" },
  "credentials": {
    "anthropic": { "api_key": "sk-ant-..." },
    "openai": { "api_key": "sk-..." }
  },
  "model_modes": {},
  "anima_defaults": { "model": "claude-sonnet-4-6", "max_tokens": 8192 },
  "animas": {
    "aoi": { "supervisor": null, "speciality": null },
    "taro": { "supervisor": "aoi", "speciality": null }
  },
  "consolidation": { "daily_enabled": true, "daily_time": "02:00" },
  "rag": { "enabled": true },
  "priming": { "dynamic_budget": true },
  "image_gen": {}
}
```

**Note**: The `animas` section holds only org layout (`supervisor`, `speciality`). Model name, credential, max_turns, etc. are stored in each Anima's `status.json` (see "Anima Config Resolution" below).

Role of each section:

| Section | Description |
|---------|------|
| `version` | Config schema version (currently `1`) |
| `setup_complete` | First-time setup complete flag |
| `locale` | UI language (`"ja"` / `"en"`) |
| `system` | Server mode, log level |
| `credentials` | API keys and endpoints (named) |
| `model_modes` | Model name → execution mode override map |
| `anima_defaults` | Default settings for all Anima |
| `animas` | Anima org layout (supervisor, speciality). Model settings are in status.json |
| `consolidation` | Memory consolidation (daily/weekly) settings |
| `rag` | RAG (embedding vector search) settings |
| `priming` | Priming (automatic memory retrieval) token budget |
| `image_gen` | Image generation style settings |

<!-- AUTO-GENERATED:START config_fields -->
### 設定項目リファレンス（自動生成）

#### Anima設定 (per-anima overrides)

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |
| `model` | `str | None` | None |  |

#### デフォルト値 (anima_defaults)

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `model` | `str` | `"claude-sonnet-4-6"` |  |
| `fallback_model` | `str | None` | None |  |
| `background_model` | `str | None` | None |  |
| `background_credential` | `str | None` | None |  |
| `max_tokens` | `int` | `8192` |  |
| `max_turns` | `int` | `20` |  |
| `credential` | `str` | `"anthropic"` |  |
| `context_threshold` | `float` | `0.5` |  |
| `max_chains` | `int` | `2` |  |
| `conversation_history_threshold` | `float` | `0.3` |  |
| `execution_mode` | `str | None` | None |  |
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |
| `thinking` | `bool | None` | None |  |
| `thinking_effort` | `str | None` | None |  |
| `llm_timeout` | `int` | `600` |  |
| `mode_s_auth` | `str | None` | None |  |
| `max_outbound_per_hour` | `int | None` | None |  |
| `max_outbound_per_day` | `int | None` | None |  |
| `max_recipients_per_run` | `int | None` | None |  |

#### AnimaWorksConfig トップレベル

| セクション | 説明 |
|-----------|------|
| `version` | 設定ファイルバージョン |
| `setup_complete` | セットアップ完了フラグ |
| `locale` | ロケール設定 |
| `system` | システム設定（モード、ログレベル） |
| `credentials` | API認証情報 |
| `model_modes` | モデル名→実行モードマッピング |
| `model_context_windows` |  |
| `model_max_tokens` |  |
| `anima_defaults` | Anima設定デフォルト値 |
| `animas` | Anima別設定オーバーライド |
| `consolidation` | 記憶統合設定 |
| `rag` | RAG（検索拡張生成）設定 |
| `priming` | プライミング（自動記憶想起）設定 |
| `image_gen` | 画像生成設定 |
| `human_notification` |  |
| `server` |  |
| `external_messaging` |  |
| `background_task` |  |
| `activity_log` |  |
| `heartbeat` |  |
| `voice` |  |
| `housekeeping` |  |
| `activity_level` |  |
| `activity_schedule` |  |
| `ui` |  |

<!-- AUTO-GENERATED:END -->

## Adding a New Anima

There are three ways to add an Anima. All use `animaworks anima create` or `animaworks init`.
`animaworks anima create` supports `--role` and `--supervisor` options and is recommended.

### Method 1: From Template

Uses predefined templates under `templates/ja/anima_templates/` or `templates/en/anima_templates/`.
Templates include identity.md, injection.md, permissions.json, skills, etc.

```bash
# Create from template (template name is the directory name)
animaworks anima create --template <template_name>

# Create with different name
animaworks anima create --template <template_name> --name <anima_name>
```

Templates are the most recommended method. Character setup is already in place and bootstrap (first-run self-definition) can be skipped.

### Method 2: From Markdown File

Prepare a character sheet (Markdown) and generate an Anima from it.

```bash
animaworks anima create --from-md /path/to/character.md [--name ken] [--role engineer] [--supervisor aoi]
```

- `--name`: Anima name (extracted from sheet if omitted)
- `--role`: Role template (engineer, researcher, manager, writer, ops, general). Default: general
- `--supervisor`: Supervisor Anima name (overrides "supervisor" in character sheet)

The Markdown file is copied as `character_sheet.md` into the Anima directory.
The sheet's "Personality" and "Role and behavior guidelines" sections are reflected in identity.md and injection.md.
permissions.json and specialty_prompt.md are applied from the role template.

The Markdown file SHOULD include:
- Heading in form `# Character: Name`, or an "English name" row in the basic info table (used for name extraction)
- `## Basic Info` — Table with English name, supervisor, model, etc.
- `## Personality` — Reflected in identity.md
- `## Role and behavior guidelines` — Reflected in injection.md

### Method 3: Blank Creation

Creates an Anima with minimal skeleton files.

```bash
animaworks anima create --name aoi
```

`--name` is required. Blank creation produces skeleton files with `{name}` placeholders replaced by the real name.
On first run (bootstrap), the agent self-defines the character through interaction with the user.

### Directory Layout After Creation

All methods produce the following directories and files:

```
~/.animaworks/animas/{name}/
├── identity.md          # Personality definition (invariant baseline)
├── injection.md         # Role and behavior guidelines (variable)
├── bootstrap.md         # First-run instructions (removed when done)
├── permissions.json       # Tool and command permissions
├── heartbeat.md         # Heartbeat config
├── cron.md              # Scheduled task config
├── episodes/            # Episode memory (daily logs)
├── knowledge/            # Semantic memory (learned knowledge)
├── procedures/          # Procedural memory (procedures)
├── skills/              # Personal skills
├── state/               # Working memory
│   ├── current_state.md  # Current task
│   └── task_queue.jsonl # Persistent task queue (pending tasks, etc.)
└── shortterm/           # Short-term memory (session continuity)
    └── archive/
```

### Anima Name Rules

Anima names MUST follow:
- Only lowercase alphanumeric, hyphen (`-`), underscore (`_`)
- Must start with a letter (`a-z`)
- Must not start with underscore (reserved for templates)
- Examples: `aoi`, `taro-dev`, `worker01`

## Execution Modes (S / C / D / G / A / B)

AnimaWorks has **six** execution modes. They are determined from the model name but can be overridden with `execution_mode` in `status.json`.

### Mode S (SDK): Claude Agent SDK

Claude models only. Uses Claude Code subprocess for the richest tool execution.

- **Target models**: `claude-*` (e.g. `claude-sonnet-4-6`, `claude-opus-4-6`)
- **Features**: File ops, Bash execution, and autonomous memory search all via Claude Agent SDK
- **Credential**: MUST use `anthropic`

### Mode C (Codex): Codex CLI

Runs OpenAI Codex-class models via the Codex CLI wrapper.

- **Target models**: `codex/*` (e.g. `codex/o4-mini`, `codex/gpt-4.1`)
- **Features**: MCP + AnimaWorks external-tool path is in the same family as S/D/G
- **Credential**: Per Codex / OpenAI requirements

### Mode D (Cursor Agent): Cursor Agent CLI

Runs the Cursor `cursor-agent` CLI as a child process; MCP exposes AnimaWorks tools.

- **Target models**: `cursor/*`
- **Features**: Requires CLI and auth on the host. Can fall back to Mode A (LiteLLM) when needed
- **Credential**: Depends on Cursor / `agent login` session

### Mode G (Gemini CLI): Gemini CLI

Runs Google's `gemini` CLI as a child process; MCP-integrated.

- **Target models**: `gemini/*`
- **Features**: Needs CLI or `GEMINI_API_KEY`. On fallback to Mode A, `gemini/` may remap to `google/` (or similar) for LiteLLM
- **Credential**: CLI login or API key

### Mode A (Autonomous): LiteLLM + tool_use Loop

For cloud and local models that support tool_use. LiteLLM unifies providers.

- **Target models**: `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash`, `ollama/qwen3:30b`, etc.
- **Features**: Runs tool_use loop via LiteLLM. Framework dispatches tool execution
- **Credential**: Specify credential for each provider

### Mode B (Basic): Assisted (LLM thinks only)

For models without tool_use. LLM only thinks; framework handles memory I/O.

- **Target models**: `ollama/gemma3*`, `ollama/phi4*`, small Ollama models, etc.
- **Features**: Single-shot response. No tool execution
- **Credential**: Usually `ollama` or similar local credential

### How Mode is Determined

You can add explicit mapping in `~/.animaworks/models.json`.
Otherwise, default patterns (fnmatch format) in code are used for matching.

```json
{
  "model_modes": {
    "ollama/my-custom-model": "A",
    "ollama/experimental-*": "B"
  }
}
```

Resolution priority:
1. Anima `execution_mode` field (per-Anima override)
2. `~/.animaworks/models.json` (exact match → wildcard)
3. `config.json` `model_modes` (deprecated fallback)
4. Default patterns in code (exact match → wildcard)
5. `B` if nothing matches (safe fallback)

## Credential Configuration

API keys are managed by name in the `credentials` section.

```json
{
  "credentials": {
    "anthropic": {
      "api_key": "sk-ant-api03-...",
      "base_url": null
    },
    "openai": {
      "api_key": "sk-...",
      "base_url": null
    },
    "ollama": {
      "api_key": "",
      "base_url": "http://localhost:11434"
    }
  }
}
```

Each Anima specifies which credential to use via the `credential` field.

- `api_key` — API key string. Empty string tries env var fallback
- `base_url` — Custom endpoint. Set for Ollama or proxy. `null` for default

**Security**: config.json is saved with file permissions `0600` (MUST). Contains API keys; prevent read access from other users.

## Permissions (permissions.json)

Each Anima's `permissions.json` defines allowed tools, accessible paths, and executable commands.

```markdown
# Permissions: aoi

## Tools
Read, Write, Edit, Bash, Grep, Glob

## Read paths
- Your directory tree
- /shared/

## Write paths
- Your directory tree

## Allowed commands
General commands

## Disallowed commands
rm -rf, system config changes

## External tools
- image_gen: yes
- web_search: yes
- slack: no
```

Permission rules:
- Each Anima reads its own `permissions.json` at startup (MUST)
- ToolHandler performs permission checks and blocks disallowed operations
- External tools (Slack, Gmail, GitHub, etc.) are enabled/disabled individually in the `External tools` section
- `Read paths` / `Write paths` are written in natural language; ToolHandler interprets them

### Blocked Commands

Adding a `## Disallowed commands` section in `permissions.json` blocks execution of the listed commands.
In addition to the system-wide hardcoded block list (dangerous commands like `rm -rf /`), per-Anima block lists are applied.

```markdown
## Disallowed commands
rm -rf, docker rm, git push --force
```

Commands in pipelines are checked per segment (e.g. `cat file | rm -rf` blocks on `rm -rf`).

## Anima Config Resolution (2-Layer Merge)

Anima model settings use **`status.json` as Single Source of Truth (SSoT)**.

### 2-Layer Config Resolution

| Priority | Source | Description |
|----------|--------|------|
| 1 (highest) | `status.json` | In each Anima directory. Holds all model and runtime parameter settings |
| 2 (fallback) | `anima_defaults` | Global defaults in config.json. Applied to fields not set in status.json |

The `animas` section in `config.json` holds only **org layout** (`supervisor`, `speciality`).
Model name, credential, max_turns, etc. are stored in `status.json`.

### status.json Structure

Path: `~/.animaworks/animas/{name}/status.json`

```json
{
  "enabled": true,
  "role": "engineer",
  "model": "claude-opus-4-6",
  "credential": "anthropic",
  "max_tokens": 16384,
  "max_turns": 200,
  "max_chains": 10,
  "context_threshold": 0.80,
  "execution_mode": null
}
```

### Changing Model

Use the CLI to change model:

```bash
animaworks anima set-model <anima_name> <model_name> [--credential <credential_name>]

# Change model for all Anima
animaworks anima set-model --all <model_name>
```

Supervisors change subordinate models via the `set_subordinate_model` tool.

### Applying Config Changes

After editing `status.json`, use `reload` to apply without restarting the process:

```bash
# Single Anima reload
animaworks anima reload <anima_name>

# Reload all Anima
animaworks anima reload --all
```

Reload takes effect via IPC immediately (no downtime). Running sessions finish with old config; new sessions use the new config.

**Typical config change workflow**:

1. `animaworks anima set-model <name> <model>` to change model
2. `animaworks anima reload <name>` to apply immediately

Same applies if you edit `status.json` manually.

### Default Values (anima_defaults)

| Field | Default | Description |
|-------|---------|------|
| `model` | `claude-sonnet-4-6` | LLM model to use |
| `max_tokens` | `8192` | Max tokens per response |
| `max_turns` | `20` | Max turns per session |
| `credential` | `"anthropic"` | Credential name to use |
| `context_threshold` | `0.50` | Short-term memory is externalized when context usage exceeds this threshold |
| `max_chains` | `2` | Max automatic session continuation count |

### Hierarchy

Org hierarchy is defined only by the `supervisor` field (in `config.json` `animas` section).

- `supervisor: null` — Top-level Anima (top of chain of command)
- `supervisor: "aoi"` — Operates as subordinate of aoi

Hierarchy works via messaging for instructions and reports. Supervisors can delegate tasks to subordinates; subordinates report results to supervisors.

## Anima Management Commands

CLI commands for daily Anima operations.
Run while the server is up (`animaworks start`).

| Command | Description | Downtime |
|---------|------|----------|
| `animaworks anima list` | List all Anima and status | None |
| `animaworks anima status [name]` | Show process status for specified Anima (or all if omitted) | None |
| `animaworks anima reload <name>` | Reload status.json and apply model config immediately (no process restart) | None |
| `animaworks anima reload --all` | Reload all Anima config | None |
| `animaworks anima restart <name>` | Fully restart Anima process (use when applying code changes) | 15–30s |
| `animaworks anima set-model <name> <model>` | Change model (updates status.json; requires `reload` to apply) | None |
| `animaworks anima set-model --all <model>` | Change model for all Anima | None |
| `animaworks anima enable <name>` | Enable disabled Anima and start process | — |
| `animaworks anima disable <name>` | Disable Anima (stop process, enabled=false in status.json) | — |
| `animaworks anima create` | Create new Anima (`--from-md`, `--template`, `--blank`) | — |
| `animaworks anima delete <name>` | Delete Anima (archived by default) | — |

### Server Management Commands

| Command | Description |
|---------|------|
| `animaworks start` | Start server |
| `animaworks stop` | Stop server |
| `animaworks restart` | Full restart (all processes respawned) |
| `animaworks status` | Show system-wide status |
| `animaworks reset` | Stop server, remove runtime directory, reinitialize (destructive) |
| `animaworks reset --restart` | Above, then restart server |

### reload vs restart vs system restart

| Command | Action | Downtime | Use case |
|---------|------|----------|----------|
| `anima reload` | IPC ModelConfig swap | None | Model/param changes in status.json |
| `anima restart` | Kill process → respawn | 15–30s | Applying code changes, memory leak mitigation |
| Server restart | All Anima restart + new detection | 15–30s | Applying Anima add/remove |
