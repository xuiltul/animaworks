# Project Configuration Guide

Reference for AnimaWorks configuration structure and Anima addition procedures.
Use when: searching or referencing configuration changes are needed.

## Runtime Directory Initialization

Runtime data is placed in `~/.animaworks/` (or `ANIMAWORKS_DATA_DIR`).
During initial setup, initialize from the template using `animaworks init`.

### init Command

| Command | Description |
|---------|------|
| `animaworks init` | Initialize the runtime directory (non-interactive). Does nothing if it already exists |
| `animaworks init --force` | Merge template diff into existing data. Adds only new files; `prompts/` is overwritten |
| `animaworks init --skip-anima` | Initialize infrastructure only. Skips Anima creation |
| `animaworks init --template NAME` | Create Anima non-interactively from template |
| `animaworks init --from-md PATH` | Create Anima non-interactively from MD file |
| `animaworks init --blank NAME` | Create blank Anima non-interactively |

**Recommended flow**: After running `animaworks init`, start the server with `animaworks start` and add Anima via the web UI setup wizard.

### Directories Created During Initialization

The following are created by `ensure_runtime_dir` (`core/infra/runtime_init.py`):

- `animas/` — Anima directory
- `shared/inbox/` — Receive message queue
- `shared/users/` — User profiles
- `shared/channels/` — Shared channels (initial files for general, ops)
- `shared/dm_logs/` — DM history (fallback)
- `tmp/attachments/` — Temporary attachment storage
- `common_skills/` / `common_knowledge/` — Common skills and knowledge
- `prompts/` / `company/` — Prompt and organization templates
- `tool_prompts.sqlite3` — Tool prompt DB
- `models.json` — Model name → execution mode mapping (copied from `config_defaults/`)

At every startup, `common_skills` and `common_knowledge` are incrementally synchronized from templates, adding only new entries (existing files are preserved).

## Overall Structure of config.json

The AnimaWorks integration configuration file is located at `~/.animaworks/config.json`.
All settings are defined using the `AnimaWorksConfig` model and have the following top-level fields.

```json
{
  "version": 1,
  "setup_complete": true,
  "locale": "ja",
  "system": { "mode": "server" },
  "credentials": {
    "anthropic": { "api_key": "sk-ant-..." },
    "openai": { "api_key": "sk-..." }
  },
  "model_modes": {},
  "anima_defaults": { "model": "claude-sonnet-5-5", "max_tokens": 8192 },
  "animas": {
    "aoi": { "supervisor": null, "speciality": null },
    "taro": { "supervisor": "aoi", "speciality": null }
  },
  "consolidation": { "daily_enabled": true, "daily_time": "02:00" },
  "rag": { "enabled": true },
  "priming": { "max_tokens": 2000 },
  "image_gen": {}
}
```

**Note**: The `animas` section only holds the organization layout (`supervisor`, `speciality`). Model names, credentials, and other model settings are recorded in each Anima's `status.json` (see "Anima Configuration Resolution" below).

Role of each section:

| Section | Description |
|-----------|------|
| `version` | Configuration schema version (currently `1`) |
| `setup_complete` | Initial setup completion flag |
| `locale` | UI language (`"ja"` / `"en"`) |
| `system` | Server mode and timezone |
| `credentials` | API keys and endpoints (named) |
| `model_modes` | Model name → execution mode override map |
| `anima_defaults` | Default settings common to all Anima |
| `animas` | Anima organization layout (supervisor, speciality). Model settings are in status.json |
| `consolidation` | Consolidation (daily/weekly) settings |
| `rag` | RAG (embedding vector search) settings |
| `priming` | Token budget for priming (automatic memory retrieval) |
| `image_gen` | Image generation style settings |

<!-- AUTO-GENERATED:START config_fields -->
### Configuration Item Reference (auto-generated)

#### Anima configuration (per-anima overrides)

| Field | Type | Default | Description |
|-----------|-----|----------|------|
| `supervisor` | `str | None` | None |  |
| `company` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |
| `model` | `str | None` | None |  |
| `heartbeat_enabled` | `bool | None` | None |  |
| `background_review_enabled` | `bool | None` | None |  |
| `token_budget_monthly` | `int | None` | None |  |
| `aliases` | `list[str]` | `[]` |  |

#### Default Values (anima_defaults)

| Field | Type | Default | Description |
|-----------|-----|----------|------|
| `model` | `str` | `"claude-sonnet-5-5"` |  |
| `fallback_model` | `str | None` | None |  |
| `fallback_models` | `list[str]` | `[]` |  |
| `background_model` | `str | None` | None |  |
| `background_credential` | `str | None` | None |  |
| `background_thinking_effort` | `str | None` | None |  |
| `voice_thinking_effort` | `str | None` | None |  |
| `max_tokens` | `int` | `8192` |  |
| `credential` | `str` | `"anthropic"` |  |
| `context_threshold` | `float` | `0.5` |  |
| `context_absolute_ceiling` | `float` | `0.75` |  |
| `task_compaction_tokens` | `int` | `0` |  |
| `task_compaction_max` | `int` | `6` |  |
| `max_session_age_hours` | `float` | `24.0` |  |
| `conversation_history_threshold` | `float` | `0.3` |  |
| `execution_mode` | `str | None` | None |  |
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |
| `extra_mcp_servers` | `dict[str, dict]` | `{}` |  |
| `thinking` | `bool | None` | None |  |
| `thinking_effort` | `str | None` | None |  |
| `mode_s_auth` | `str | None` | None |  |
| `default_workspace` | `str` | `""` |  |
| `consolidation_enabled` | `bool` | `True` |  |
| `heartbeat_enabled` | `bool` | `True` |  |
| `token_budget_monthly` | `int | None` | None |  |

#### AnimaWorksConfig Top Level

| Section | Description |
|-----------|------|
| `version` | Configuration file version |
| `setup_complete` | Setup completion flag |
| `locale` | Locale setting |
| `system` | System settings (mode, log level) |
| `credentials` | API authentication credentials |
| `model_modes` | Model name → execution mode mapping |
| `model_context_windows` |  |
| `model_max_tokens` |  |
| `anima_defaults` | Anima configuration defaults |
| `animas` | Per-Anima configuration overrides |
| `consolidation` | Memory consolidation settings |
| `helper_models` |  |
| `background_review` |  |
| `rag` | RAG (retrieval-augmented generation) settings |
| `gpu` |  |
| `memory` |  |
| `skills` |  |
| `chatwork_tool` |  |
| `prompt` |  |
| `priming` | Priming (automatic memory recall) settings |
| `image_gen` | Image generation settings |
| `human_notification` |  |
| `interaction` |  |
| `server` |  |
| `llm_rate_guard` |  |
| `mcp` |  |
| `external_messaging` |  |
| `external_tasks` |  |
| `github_webhook` |  |
| `event_export` |  |
| `background_task` |  |
| `activity_log` |  |
| `logging` |  |
| `heartbeat` |  |
| `voice` |  |
| `phone` |  |
| `housekeeping` |  |
| `inbox` |  |
| `local_llm` |  |
| `workspaces` |  |
| `github_identities` |  |
| `activity_level` |  |
| `activity_schedule` |  |
| `icon_url_template` |  |
| `ui` |  |
| `cli` |  |
| `enclave` |  |
| `enclaves` |  |

<!-- AUTO-GENERATED:END -->

## How to Add a New Anima

There are three ways to add an Anima. All are executed with `animaworks anima create` or `animaworks init`.
`animaworks anima create` has the `--role` and `--supervisor` options and is recommended.

### Method 1: Create from a Template

Use a predefined template (under `templates/ja/anima_templates/` or `templates/en/anima_templates/`).
Templates include IDENTITY.md, SOUL.md, AGENTS.md, permissions.json, skills, etc.

```bash
# テンプレートから作成（テンプレート名はディレクトリ名）
animaworks anima create --template <テンプレート名>

# 名前を変えて作成
animaworks anima create --template <テンプレート名> --name <anima名>
```

Templates are the most recommended option. The character configuration is already set up, so bootstrap (self-definition during initial startup) can be skipped.

### Method 2: Create from a Markdown File

Prepare a character sheet (Markdown) and use it to generate an Anima.

```bash
animaworks anima create --from-md /path/to/character.md [--name ken] [--role engineer] [--supervisor aoi]
```

- `--name`: Anima name (extracted from the sheet if omitted)
- `--role`: Role template (engineer, researcher, manager, writer, ops, general). Default: general
- `--supervisor`: Supervisor Anima name (overrides the “supervisor” specified in the character sheet)

The Markdown file is copied to the Anima directory as `character_sheet.md`.
The sheet’s “Basic Information” is reflected in IDENTITY.md, “Personality” in SOUL.md (profile headings go in IDENTITY.md), and “Role and Action Policy” in AGENTS.md.
permissions.json and specialty_prompt.md are applied from the role template.

The Markdown file SHOULD include:
- A heading in the `# Character: 名前` format, or an “English Name” row in the basic information table (used for automatic name extraction)
- `## 基本情報` — a table with the English name, supervisor, model, etc. Reflected in IDENTITY.md
- `## 人格` — reflected in SOUL.md (profile-related headings are routed to IDENTITY.md)
- `## 役割・行動方針` — reflected in AGENTS.md

### Method 3: Blank Creation

Create an Anima with minimal skeleton files.

```bash
animaworks anima create --name aoi
```

`--name` is required. In blank creation, skeleton files are generated with `{name}` placeholders replaced by the real name.
During first-startup bootstrap, the agent self-defines the character through dialogue with the user.

### Directory Structure After Creation

Either method generates the following directories and files:

```
~/.animaworks/animas/{name}/
├── IDENTITY.md          # プロフィール・外見（不変ベースライン）
├── SOUL.md              # 人格・価値観（不変ベースライン）
├── AGENTS.md            # 役割・行動指針（可変）
├── BOOTSTRAP.md         # 初回起動指示（完了後に削除）
├── permissions.json       # ツール・コマンド権限
├── HEARTBEAT.md         # ハートビート設定
├── cron.md              # 定時タスク設定
├── episodes/            # エピソード記憶（日別ログ）
├── knowledge/           # 意味記憶（学んだ知識）
├── procedures/          # 手続き記憶（手順書）
├── skills/              # 個人スキル
├── state/               # ワーキングメモリ
│   └── current_state.md  # 現在のタスク
└── shortterm/           # 短期記憶（セッション継続用）
    └── archive/
```

### Anima Naming Conventions

Anima names MUST follow these rules:
- Only lowercase alphanumeric characters, hyphens (`-`), and underscores (`_`) are allowed
- Must start with a letter (`a-z`)
- Cannot start with an underscore (reserved for templates)
- Examples: `aoi`, `taro-dev`, `worker01`

## Execution Modes (S / C / D / G / A / B)

AnimaWorks has **6** execution modes. They are automatically determined from the model name, but can be overridden in `status.json` via `execution_mode`.

### Mode S (SDK): Claude Agent SDK

For Claude models only. Uses a Claude Code subprocess, enabling the richest tool execution.

- **Target models**: `claude-*` (e.g., `claude-sonnet-5-5`, `claude-opus-5-5`)
- **Features**: File operations, Bash execution, and autonomous memory search are all performed via the Claude Agent SDK
- **credential**: Uses `anthropic` (MUST)

### Mode C (Codex): Codex CLI

Runs OpenAI Codex-family models via the Codex CLI wrapper.

- **Target models**: `codex/*` (e.g., `codex/o4-mini`, `codex/gpt-4.1`)
- **Features**: The integration path for MCP tools and AnimaWorks external tools is similar to Mode S/D/G
- **credential**: Follows Codex / OpenAI requirements

### Mode D (Cursor Agent): Cursor Agent CLI

Runs Cursor's `cursor-agent` CLI as a child process. Uses AnimaWorks tools via MCP.

- **Target models**: `cursor/*`
- **Features**: Requires CLI and authentication on the host. Can switch to Mode A (LiteLLM) on failure
- **credential**: Depends on Cursor / `agent login` authentication status

### Mode G (Gemini CLI): Gemini CLI

Runs Google's `gemini` CLI as a child process. MCP integration.

- **Target models**: `gemini/*`
- **Features**: Requires CLI or `GEMINI_API_KEY`. On fallback, may be remapped via Mode A from `gemini/` to `google/`, etc.
- **credential**: CLI login or API key

### Mode A (Autonomous): LiteLLM + tool_use loop

For cloud and local models that support tool_use. Unifies providers via LiteLLM.

- **Target models**: `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash`, `ollama/qwen3:30b`, etc.
- **Features**: Runs the tool_use loop via LiteLLM. Tool execution is dispatched by the framework
- **credential**: Specify credentials corresponding to each provider

Mode B is deprecated. Models without tool_use support are not recommended. `execution_mode: "B"` remaining in configuration is treated as Mode A.

### Automatic mode detection mechanism

Explicit mappings can be added via `~/.animaworks/models.json`.
If unspecified, matching is done using default patterns in the code (fnmatch format).

```json
{
  "model_modes": {
    "ollama/my-custom-model": "A",
    "ollama/experimental-*": "A"
  }
}
```

Detection priority:
1. Anima's `execution_mode` field (per-anima override)
2. `~/.animaworks/models.json` (exact match → wildcard)
3. `model_modes` of `config.json` (deprecated fallback)
4. Default patterns in the code (exact match → wildcard)
5. If none match, Mode A

## Credential configuration

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

- `api_key` — API key string. If empty, attempts fallback from environment variables
- `base_url` — Custom endpoint. Set when using Ollama or a proxy. Default is `null`

**Security**: config.json is saved with file permissions `0600` (MUST). Since it contains API keys, prevents reading by other users.

## Permission configuration (permissions.json)

Each Anima's `permissions.json` defines available tools, accessible paths, and executable commands.

```markdown
# Permissions: aoi

## 使えるツール
Read, Write, Edit, Bash, Grep, Glob

## 読める場所
- 自分のディレクトリ配下すべて
- /shared/ 配下

## 書ける場所
- 自分のディレクトリ配下すべて

## 実行できるコマンド
全般的なコマンド

## 実行できないコマンド
rm -rf, システム設定の変更

## 外部ツール
- image_gen: yes
- web_search: yes
- slack: no
```

Permission rules:
- Each Anima must read its own `permissions.json` at startup (MUST)
- ToolHandler performs permission checks and blocks unauthorized operations
- External tools (Slack, Gmail, GitHub, etc.) are individually allowed/denied in the `外部ツール` section
- `読める場所` / `書ける場所` are written in natural language and interpreted by ToolHandler

### Blocked commands

If a `## 実行できないコマンド` section is listed in `permissions.json`, execution of the specified commands is blocked.
In addition to the system-wide hardcoded blocklist (dangerous commands such as `rm -rf /`), per-Anima blocklists are applied.

```markdown
## 実行できないコマンド
rm -rf, docker rm, git push --force
```

Commands in pipelines are also checked individually (e.g., `rm -rf` is blocked in `cat file | rm -rf`).

## Anima configuration resolution (two-layer merge)

**`status.json` is the Single Source of Truth (SSoT)** for Anima model configuration.

### Two-layer structure of configuration resolution

| Priority | Source | Description |
|--------|--------|------|
| 1 (highest) | `status.json` | Placed in each Anima directory. Holds all model and execution parameter settings |
| 2 (fallback) | `anima_defaults` | Global defaults for `config.json`. Applied to fields not set in `status.json` |

The `animas` section of `config.json` holds only the **organization layout** (`supervisor`, `speciality`).
Model settings such as model name and credentials are recorded in `status.json`.

### Structure of status.json

File path: `~/.animaworks/animas/{name}/status.json`

```json
{
  "enabled": true,
  "role": "engineer",
  "model": "claude-opus-5-5",
  "credential": "anthropic",
  "max_tokens": 16384,
  "context_threshold": 0.80,
  "execution_mode": null
}
```

### Model changes

Use CLI commands to change models:

```bash
animaworks anima set-model <anima名> <モデル名> [--credential <credential名>]

# 全 Anima のモデルを一括変更
animaworks anima set-model --all <モデル名>
```

When a supervisor changes a subordinate's model, use the `set_subordinate_model` tool.

### Configuration Reload

After root/CLI has changed the allowed `status.json` configuration, use the `reload` command to apply the model configuration without restarting the process. The `set-model` command requests a reload from the running Anima:

```bash
# 単一 Anima のリロード
animaworks anima reload <anima名>

# 全 Anima のリロード
animaworks anima reload --all
```

The reload is applied immediately via IPC (no downtime). Running sessions complete with the old configuration, and new sessions use the new configuration starting from the next session.

**Typical configuration change workflow**:

1. Change the model with `animaworks anima set-model <name> <model>` (the root API updates `status.json` and requests a reload from the running Anima)
2. After root/CLI updates that are not automatically notified, use `animaworks anima reload <name>`

Do not edit `status.json` directly from the Anima process. Use the root-owned CLI/API operation.

### Default Value List (anima_defaults)

| Field | Default Value | Description |
|-----------|-------------|------|
| `model` | `claude-sonnet-5-5` | LLM model to use |
| `max_tokens` | `8192` | Maximum number of tokens per response |
| `credential` | `"anthropic"` | Credential name to use |
| `context_threshold` | `0.50` | When context usage exceeds this threshold, short-term memory is externalized |

### Hierarchy structure

Defines the organization hierarchy using only the `supervisor` field (listed in the `animas` section of `config.json`).

- `supervisor: null` — Top-level Anima (highest in the chain of command)
- `supervisor: "aoi"` — Operates as a subordinate of aoi

The hierarchy functions through instruction and reporting via messaging. Supervisors can delegate tasks to subordinates, and subordinates report results to supervisors.

## Anima Management Command List

CLI commands for daily Anima operations and management.
Run these while the server is running (`animaworks start` completed).

| Command | Description | Downtime |
|---------|------|-----------|
| `animaworks anima list` | Display a list and status of all Anima | None |
| `animaworks anima status [name]` | Display the process status of a specified Anima (or all if omitted) | None |
| `animaworks anima reload <name>` | Reload status.json and apply the model configuration immediately (no process restart) | None |
| `animaworks anima reload --all` | Reload the configuration of all Anima at once | None |
| `animaworks anima restart <name>` | Fully restart the Anima process (used when reflecting code changes) | 15-30 seconds |
| `animaworks anima set-model <name> <model>` | Update the root-owned status and reload the model configuration of running Anima | None |
| `animaworks anima set-model --all <model>` | Change the model of all Anima at once | None |
| `animaworks anima enable <name>` | Activate a dormant Anima and start its process | — |
| `animaworks anima disable <name>` | Put Anima into dormancy (process shutdown, status.json enabled=false) | — |
| `animaworks anima create` | Create a new Anima (`--from-md`, `--template`, `--blank`) | — |
| `animaworks anima delete <name>` | Delete an Anima (archived by default) | — |

### Server management commands

| Command | Description |
|---------|------|
| `animaworks start` | Starts the server |
| `animaworks stop` | Stops the server |
| `animaworks restart` | Fully restarts the server (regenerates all processes) |
| `animaworks status` | Displays system-wide status |
| `animaworks reset` | After stopping the server, deletes the runtime directory and reinitializes (destructive) |
| `animaworks reset --restart` | Restarts the server after the above |

### Distinction between reload / restart / system reload

| Command | Behavior | Downtime | Use case |
|---------|------|-----------|-------------|
| `anima reload` | Swaps ModelConfig via IPC | None | Model/parameter changes in status.json |
| `anima restart` | Kills and regenerates processes | 15-30 seconds | Applying code changes, memory leak mitigation |
| Server restart | Restarts all Anima + detects new ones | 15-30 seconds | Reflecting Anima additions/removals |
