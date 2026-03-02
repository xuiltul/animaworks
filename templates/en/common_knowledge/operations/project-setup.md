# Project Setup

Reference for AnimaWorks configuration structure and how to add Anima.
Use when you need to change settings or add new Anima.

## config.json Overview

The main config file is at `~/.animaworks/config.json`.
All settings follow the `AnimaWorksConfig` model with these top-level fields:

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
  "anima_defaults": { "model": "claude-sonnet-4-6", "max_tokens": 4096 },
  "animas": {
    "aoi": { "model": "claude-sonnet-4-6", "supervisor": null },
    "taro": { "model": "openai/gpt-4.1", "credential": "openai", "supervisor": "aoi" }
  },
  "consolidation": { "daily_enabled": true, "daily_time": "02:00" },
  "rag": { "enabled": true },
  "priming": { "dynamic_budget": true },
  "image_gen": {}
}
```

Role of each section:

| Section | Description |
|---------|-------------|
| `version` | Config schema version (currently `1`) |
| `setup_complete` | First-time setup complete flag |
| `locale` | UI language (`"ja"` / `"en"`) |
| `system` | Server mode, log level |
| `credentials` | API keys and endpoints (named) |
| `model_modes` | Model → execution mode overrides |
| `anima_defaults` | Defaults for all Anima |
| `animas` | Per-Anima overrides |
| `consolidation` | Memory consolidation (daily/weekly) |
| `rag` | RAG (embedding search) settings |
| `priming` | Priming token budget |
| `image_gen` | Image generation style |

<!-- AUTO-GENERATED:START config_fields -->
### Config Field Reference (auto-generated)

#### Anima Settings (per-anima overrides)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str | None` | None |  |
| `fallback_model` | `str | None` | None |  |
| `max_tokens` | `int | None` | None |  |
| `max_turns` | `int | None` | None |  |
| `credential` | `str | None` | None |  |
| `context_threshold` | `float | None` | None |  |
| `max_chains` | `int | None` | None |  |
| `conversation_history_threshold` | `float | None` | None |  |
| `execution_mode` | `str | None` | None |  |
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |

#### Default Values (anima_defaults)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"claude-sonnet-4-6"` |  |
| `fallback_model` | `str | None` | None |  |
| `max_tokens` | `int` | `4096` |  |
| `max_turns` | `int` | `20` |  |
| `credential` | `str` | `"anthropic"` |  |
| `context_threshold` | `float` | `0.5` |  |
| `max_chains` | `int` | `2` |  |
| `conversation_history_threshold` | `float` | `0.3` |  |
| `execution_mode` | `str | None` | None |  |
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |

#### AnimaWorksConfig Top Level

| Section | Description |
|---------|-------------|
| `version` | Config version |
| `setup_complete` | Setup complete flag |
| `locale` | Locale |
| `system` | System settings (mode, log level) |
| `credentials` | API credentials |
| `model_modes` | Model → execution mode mapping |
| `anima_defaults` | Default Anima settings |
| `animas` | Per-Anima overrides |
| `consolidation` | Memory consolidation |
| `rag` | RAG settings |
| `priming` | Priming settings |
| `image_gen` | Image generation settings |

<!-- AUTO-GENERATED:END -->

## Adding a New Anima

Three ways to add an Anima, all via CLI:

### Method 1: From Template

Uses predefined templates under `templates/anima_templates/`.
Includes identity.md, injection.md, permissions.md, skills, etc.

```bash
# List templates
animaworks anima create --template <template_name>

# Create with different name
animaworks anima create --template <template_name> --name <anima_name>
```

Templates are recommended; character setup is already in place and bootstrap can be skipped.

### Method 2: From Markdown

Create a character sheet (Markdown) and generate from it.

```bash
animaworks anima create --from-md /path/to/character.md --name taro
```

The Markdown file is copied as `character_sheet.md` in the Anima directory.
On first run (bootstrap), the agent reads it and generates identity.md and injection.md automatically.

Markdown SHOULD include:
- Heading in form `# Character: Name` (used to extract name)
- Character traits, role, appearance, etc.

### Method 3: Blank

Creates Anima with a minimal skeleton.

```bash
animaworks anima create --name yuna
```

Blank creation produces skeleton files with `{name}` replaced by the real name.
During bootstrap, the agent defines the character through interaction.

### Directory Layout After Creation

All methods produce:

```
~/.animaworks/animas/{name}/
├── identity.md          # Personality (invariant baseline)
├── injection.md         # Role and behavior (variable)
├── bootstrap.md         # First-run instructions (removed when done)
├── permissions.md       # Tool and command permissions
├── heartbeat.md         # Heartbeat config
├── cron.md              # Cron task config
├── episodes/            # Episode memory (daily logs)
├── knowledge/           # Semantic memory
├── procedures/          # Procedural memory (procedures)
├── skills/              # Personal skills
├── state/               # Working memory
│   ├── current_task.md  # Current task
│   └── pending.md       # Pending tasks
└── shortterm/           # Short-term (session continuity)
    └── archive/
```

### Anima Name Rules

Names MUST follow:
- Only lowercase letters, digits, hyphen (`-`), underscore (`_`)
- Must start with a letter (`a-z`)
- Must not start with underscore (reserved for templates)
- Examples: `aoi`, `taro-dev`, `worker01`

## Execution Modes (S / A / B)

AnimaWorks has three execution modes. They are determined from model name, but can be overridden.

### Mode S (SDK): Claude Agent SDK

Claude models only. Uses Claude Code subprocess for the richest tool execution.

- **Models**: `claude-*` (e.g. `claude-sonnet-4-6`, `claude-opus-4-6`)
- **Features**: File ops, Bash, memory search via Claude Agent SDK
- **Credential**: Must use `anthropic`

### Mode A (Autonomous): LiteLLM + tool_use loop

For non-Claude models with tool_use. LiteLLM unifies providers.

- **Models**: `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash`, `ollama/qwen3:30b`, etc.
- **Features**: LiteLLM tool_use loop; framework dispatches tool calls
- **Credential**: Provider-specific credential

### Mode B (Basic): Assisted (LLM thinks only)

For models without tool_use. Framework handles memory I/O.

- **Models**: `ollama/gemma3*`, `ollama/phi4*`, small Ollama models
- **Features**: Single-shot response. No tool execution
- **Credential**: Usually `ollama` or similar local credential

### How mode is chosen

Add explicit mapping in `~/.animaworks/models.json`.
Otherwise default fnmatch patterns in code are used.

```json
{
  "model_modes": {
    "ollama/my-custom-model": "A",
    "ollama/experimental-*": "B"
  }
}
```

Priority:
1. Anima `execution_mode` field (per-Anima override)
2. `~/.animaworks/models.json` (exact match → wildcard)
3. `config.json` `model_modes` (deprecated fallback)
4. Default patterns in code (exact → wildcard)
5. `B` if nothing matches (safe fallback)

## Credentials

API keys are stored under named credentials in the `credentials` section.

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

Each Anima specifies a credential via the `credential` field.

- `api_key` — API key string. Empty string tries env vars
- `base_url` — Custom endpoint (Ollama, proxy). `null` for default

**Security**: config.json is saved with mode `0600`. Do not expose API keys.

## Permissions (permissions.md)

Each Anima's `permissions.md` defines allowed tools, paths, and commands.

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

Rules:
- Each Anima reads its own `permissions.md` at startup (MUST)
- ToolHandler blocks disallowed operations
- External tools (Slack, Gmail, GitHub, etc.) are enabled per category

### Blocked Commands

`## 実行できないコマンド` (Disallowed commands) in `permissions.md` blocks those commands.
Together with system-wide blocked patterns (e.g. `rm -rf /`), both layers are checked.

```markdown
## 実行できないコマンド
rm -rf, docker rm, git push --force
```

Pipelines are checked per segment.

## Anima Config Resolution (2-Layer Merge)

**`status.json` is the Single Source of Truth (SSoT)** for model settings.

### Resolution order

| Priority | Source | Description |
|----------|--------|-------------|
| 1 | `status.json` | Per-Anima dir. Holds model and runtime params |
| 2 | `anima_defaults` | Defaults in config.json when not in status.json |

`config.json` `animas` holds only **org layout** (`supervisor`, `speciality`).
Model, credential, max_turns, etc. live in `status.json`.

### status.json structure

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

### Changing model

Use the CLI:

```bash
animaworks anima set-model <anima_name> <model_name> [--credential <credential_name>]

# Change model for all Anima
animaworks anima set-model --all <model_name>
```

Supervisors change subordinate models via the `set_subordinate_model` tool.

### Applying changes

After editing `status.json`, use `reload` to apply without restart:

```bash
# Single Anima
animaworks anima reload <anima_name>

# All Anima
animaworks anima reload --all
```

Reload takes effect via IPC immediately (no downtime). Current sessions finish with old config; new sessions use the new config.

**Typical workflow**:
1. `animaworks anima set-model <name> <model>` to change
2. `animaworks anima reload <name>` to apply

Same applies if you edit `status.json` by hand.

## Anima Management Commands

CLI commands for daily operations.
Run while the server is up (`animaworks start`).

| Command | Description | Downtime |
|---------|-------------|----------|
| `animaworks anima list` | List all Anima and status | None |
| `animaworks anima status [name]` | Show process status (all or one) | None |
| `animaworks anima reload <name>` | Reload status.json (no process restart) | None |
| `animaworks anima reload --all` | Reload all Anima | None |
| `animaworks anima restart <name>` | Fully restart Anima process | 15–30s |
| `animaworks anima set-model <name> <model>` | Change model (needs reload) | None |
| `animaworks anima set-model --all <model>` | Change model for all | None |
| `animaworks anima enable <name>` | Enable and start disabled Anima | — |
| `animaworks anima disable <name>` | Disable (stop process, enabled=false) | — |
| `animaworks anima create` | Create new Anima (`--from-md`, `--template`, `--blank`) | — |
| `animaworks anima delete <name>` | Delete Anima (default: archive) | — |

### Server commands

| Command | Description |
|---------|-------------|
| `animaworks start` | Start server |
| `animaworks stop` | Stop server |
| `animaworks restart` | Full restart (all processes) |
| `animaworks status` | System-wide status |

### reload vs restart vs system restart

| Command | Action | Downtime | When to use |
|---------|--------|----------|-------------|
| `anima reload` | IPC ModelConfig swap | None | Model/param changes in status.json |
| `anima restart` | Kill and respawn process | 15–30s | Code changes, memory issues |
| Server restart | All Anima restart | 15–30s | Adding/removing Anima |
