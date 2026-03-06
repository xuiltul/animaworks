# AnimaWorks CLI Reference

**[日本語版](cli-reference.ja.md)**

> Last updated: 2026-03-06
> See also: [spec.md](spec.md), [api-reference.md](api-reference.md)

---

## Basics

```bash
animaworks <command> [options]
# or
python3 -m main <command> [options]
```

### Global Options

| Option | Type | Description |
|--------|------|-------------|
| `--data-dir` | path | Runtime data directory (default: `~/.animaworks` or `ANIMAWORKS_DATA_DIR`) |
| `--help`, `-h` | — | Show help |

---

## Table of Contents

1. [Initialization & Setup](#1-initialization--setup)
2. [Server Management](#2-server-management)
3. [Chat & Messaging](#3-chat--messaging)
4. [Board (Shared Channels)](#4-board-shared-channels)
5. [Anima Management](#5-anima-management)
6. [Config](#6-config)
7. [RAG Index](#7-rag-index)
8. [Model Information](#8-model-information)
9. [Task Management](#9-task-management)
10. [Logs & Cost](#10-logs--cost)
11. [Asset Management](#11-asset-management)

---

## 1. Initialization & Setup

### `init` — Initialize Runtime Directory

Initialize the runtime directory (`~/.animaworks/`) from templates.

```bash
animaworks init                          # Interactive
animaworks init --from-md character.md   # Create Anima from MD file
animaworks init --template default       # Create from template
animaworks init --blank --name alice     # Create empty Anima
animaworks init --skip-anima             # Infrastructure only
animaworks init --force                  # Merge missing templates into existing runtime
```

| Option | Type | Description |
|--------|------|-------------|
| `--force` | flag | Merge missing templates only |
| `--template` | string | Template name (non-interactive) |
| `--from-md` | path | Create Anima from MD file (non-interactive) |
| `--blank` | string | Create empty Anima (non-interactive) |
| `--skip-anima` | flag | Initialize infrastructure only |
| `--name` | string | Override Anima name (use with `--from-md`) |

**Note:** `--force`, `--template`, `--from-md`, `--blank`, `--skip-anima` are mutually exclusive.

---

## 2. Server Management

### `start` — Start Server

```bash
animaworks start                     # Background
animaworks start -f                  # Foreground (show logs)
animaworks start --host 0.0.0.0 --port 18500
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host` | string | "0.0.0.0" | Bind host |
| `--port` | int | 18500 | Port number |
| `-f`, `--foreground` | flag | — | Foreground execution |

`serve` is an alias for `start`.

---

### `stop` — Stop Server

```bash
animaworks stop           # Normal (SIGTERM)
animaworks stop --force   # Force (SIGKILL + orphan cleanup)
```

| Option | Type | Description |
|--------|------|-------------|
| `--force` | flag | Force kill with SIGKILL, also terminate orphan runners |

---

### `restart` — Restart Server

```bash
animaworks restart
animaworks restart --force
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host` | string | "0.0.0.0" | Bind host |
| `--port` | int | 18500 | Port number |
| `-f`, `--foreground` | flag | — | Foreground execution |
| `--force` | flag | — | Force stop |

---

### `reset` — Full Runtime Reset

Stop server → delete runtime directory → re-initialize.

```bash
animaworks reset              # Reset only
animaworks reset --restart    # Reset then start server
```

---

## 3. Chat & Messaging

### `chat` — Chat with an Anima

```bash
animaworks chat alice "Hello"
animaworks chat alice "How's the task going?" --from admin
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `anima` | positional | Required | Anima name |
| `message` | positional | Required | Message to send |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--from` | string | "human" | Sender name |

---

### `heartbeat` — Manual Heartbeat

```bash
animaworks heartbeat alice
```

---

### `send` — Send Message Between Animas

```bash
animaworks send alice bob "Here's my report"
animaworks send alice bob "Please handle this task" --intent delegation
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `from_person` | positional | Required | Sender |
| `to_person` | positional | Required | Recipient |
| `message` | positional | Required | Body |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--intent` | string | "" | delegation, report, question |
| `--thread-id` | string | null | Thread ID |
| `--reply-to` | string | null | Reply-to message ID |

---

## 4. Board (Shared Channels)

### `board read` — Read Channel

```bash
animaworks board read general
animaworks board read ops --limit 50 --human-only
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `channel` | positional | Required | Channel name |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--limit` | int | 20 | Max messages |
| `--human-only` | flag | — | Human messages only |

---

### `board post` — Post to Channel

```bash
animaworks board post alice general "Today's progress report"
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `from_anima` | positional | Required | Sender Anima name |
| `channel` | positional | Required | Channel name |
| `text` | positional | Required | Message body |

---

### `board dm-history` — DM History

```bash
animaworks board dm-history alice bob --limit 30
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `from_anima` | positional | Required | Your Anima name |
| `peer` | positional | Required | Peer Anima name |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--limit` | int | 20 | Max messages |

---

## 5. Anima Management

### `anima list` — List Animas

```bash
animaworks anima list            # Via server API
animaworks anima list --local    # Direct filesystem scan
```

---

### `anima create` — Create New Anima

```bash
animaworks anima create --from-md character.md --role engineer
animaworks anima create --from-md character.md --supervisor alice --name bob
animaworks anima create --template default --name carol
```

| Option | Type | Description |
|--------|------|-------------|
| `--name` | string | Anima name (required for blank) |
| `--template` | string | Template name |
| `--from-md` | path | Character sheet MD file |
| `--supervisor` | string | Supervisor Anima name |
| `--role` | choice | engineer, researcher, manager, writer, ops, general |

---

### `anima info` — Detailed Information

```bash
animaworks anima info alice
animaworks anima info alice --json
```

Shows model name, execution mode, credential, context window, max_turns, etc.

| Option | Type | Description |
|--------|------|-------------|
| `--json` | flag | JSON output |

---

### `anima status` — Process Status

```bash
animaworks anima status          # All Animas
animaworks anima status alice    # Specific Anima
```

---

### `anima restart` — Restart Process

```bash
animaworks anima restart alice
```

---

### `anima enable` / `anima disable` — Enable/Disable

```bash
animaworks anima enable alice
animaworks anima disable alice
```

---

### `anima delete` — Delete Anima

```bash
animaworks anima delete alice              # Archive to ZIP then delete
animaworks anima delete alice --no-archive  # Delete without archive
animaworks anima delete alice --force       # Skip confirmation
```

| Option | Type | Description |
|--------|------|-------------|
| `--no-archive` | flag | Don't create ZIP archive |
| `--force` | flag | Skip confirmation prompt |

---

### `anima set-model` — Change Model

```bash
animaworks anima set-model alice claude-sonnet-4-6
animaworks anima set-model alice openai/gpt-4.1 --credential azure
animaworks anima set-model --all claude-sonnet-4-6
```

| Argument | Type | Description |
|----------|------|-------------|
| `anima` | positional | Anima name (omit with `--all`) |
| `model` | positional | Model name |

| Option | Type | Description |
|--------|------|-------------|
| `--credential` | string | Credential name |
| `--all` | flag | Apply to all enabled Animas |

---

### `anima set-background-model` — Set Background Model

Set the model used for heartbeat / inbox / cron.

```bash
animaworks anima set-background-model alice claude-sonnet-4-6
animaworks anima set-background-model alice --clear   # Fallback to main model
animaworks anima set-background-model --all claude-sonnet-4-6
```

| Option | Type | Description |
|--------|------|-------------|
| `--credential` | string | Credential name |
| `--all` | flag | Apply to all enabled Animas |
| `--clear` | flag | Remove background model override |

---

### `anima reload` — Hot-Reload Config

Reload status.json without process restart.

```bash
animaworks anima reload alice
animaworks anima reload --all
```

---

### `anima set-role` — Change Role

```bash
animaworks anima set-role alice engineer
animaworks anima set-role alice manager --status-only   # status.json only
animaworks anima set-role alice writer --no-restart      # Skip auto-restart
```

| Argument | Type | Description |
|----------|------|-------------|
| `anima` | positional | Anima name |
| `role` | positional | engineer, researcher, manager, writer, ops, general |

| Option | Type | Description |
|--------|------|-------------|
| `--status-only` | flag | Update status.json only, skip template re-application |
| `--no-restart` | flag | Skip automatic restart |

---

### `anima rename` — Rename Anima

```bash
animaworks anima rename alice alice-v2
animaworks anima rename alice alice-v2 --force
```

Updates config.json, directory, and supervisor references in one operation.

| Option | Type | Description |
|--------|------|-------------|
| `--force` | flag | Skip confirmation prompt |

---

### `anima audit` — Audit Report

Report on subordinate Anima's activity summary, task status, error frequency, and tool usage statistics.

```bash
animaworks anima audit bob
animaworks anima audit bob --days 7
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--days` | int | 1 | Audit period in days (max 30) |

---

## 6. Config

### `config` — Interactive Setup

```bash
animaworks config -i    # Interactive wizard
```

---

### `config get` — Get Value

```bash
animaworks config get system.log_level
animaworks config get credentials.anthropic.api_key --show-secrets
```

| Argument | Type | Description |
|----------|------|-------------|
| `key` | positional | Dot-notation key |

| Option | Type | Description |
|--------|------|-------------|
| `--show-secrets` | flag | Show API key values |

---

### `config set` — Set Value

```bash
animaworks config set system.log_level DEBUG
animaworks config set heartbeat.interval_minutes 15
```

---

### `config list` — List All Settings

```bash
animaworks config list
animaworks config list --section credentials --show-secrets
```

| Option | Type | Description |
|--------|------|-------------|
| `--section` | string | Filter by section |
| `--show-secrets` | flag | Show API key values |

---

### `config export-sections` — Export Sections

Export prompt DB sections to template files.

```bash
animaworks config export-sections
animaworks config export-sections --dry-run
```

---

## 7. RAG Index

### `index` — Build Vector Index

```bash
animaworks index                        # Incremental for all Animas
animaworks index --anima alice          # Specific Anima only
animaworks index --full                 # Full rebuild
animaworks index --shared               # Index shared collections
animaworks index --dry-run              # Preview only
```

| Option | Type | Description |
|--------|------|-------------|
| `--anima` | string | Target Anima name |
| `--full` | flag | Delete existing indexes and rebuild |
| `--shared` | flag | Index common_knowledge/common_skills per Anima |
| `--dry-run` | flag | Show targets without executing |

---

## 8. Model Information

### `models list` — List Supported Models

```bash
animaworks models list
animaworks models list --mode S          # S-mode only
animaworks models list --json
```

| Option | Type | Description |
|--------|------|-------------|
| `--mode` | choice | Filter by S, A, B, C |
| `--json` | flag | JSON output |

---

### `models info` — Model Details

```bash
animaworks models info claude-sonnet-4-6
```

Shows execution mode, context window, threshold, etc.

---

### `models show` — Show models.json

```bash
animaworks models show
animaworks models show --json
```

---

## 9. Task Management

Manage the Anima persistent task queue. Runs via `animaworks-tool`.

### `animaworks-tool task add` — Add Task

```bash
animaworks-tool task add --source human --instruction "Write API documentation" --assignee alice --deadline 1d
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--source` | choice | "anima" | human, anima |
| `--instruction` | string | Required | Task instruction |
| `--assignee` | string | Required | Assignee Anima name |
| `--summary` | string | instruction[:100] | One-line summary |
| `--deadline` | string | null | ISO8601 deadline or `1d`, `2h`, etc. |

---

### `animaworks-tool task update` — Update Task

```bash
animaworks-tool task update --task-id abc123 --status done --summary "Completed"
```

| Option | Type | Description |
|--------|------|-------------|
| `--task-id` | string | Task ID |
| `--status` | choice | pending, in_progress, done, cancelled, blocked |
| `--summary` | string | Updated summary |

---

### `animaworks-tool task list` — List Tasks

```bash
animaworks-tool task list
animaworks-tool task list --status pending
```

| Option | Type | Description |
|--------|------|-------------|
| `--status` | choice | pending, in_progress, done, cancelled, blocked |

---

## 10. Logs & Cost

### `logs` — View Logs

```bash
animaworks logs alice                  # Specific Anima logs
animaworks logs --all                  # Server + all Anima logs
animaworks logs alice --lines 100      # Show 100 lines
animaworks logs alice --date 20260305  # Specific date
```

| Argument | Type | Description |
|----------|------|-------------|
| `anima` | positional | Anima name (omit with `--all`) |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--all` | flag | — | Show all logs |
| `--lines` | int | 50 | Number of lines |
| `--date` | string | null | Date (YYYYMMDD) |

---

### `cost` — Token Usage & Cost

```bash
animaworks cost                     # All Animas, 30 days
animaworks cost alice               # Specific Anima
animaworks cost --today             # Today only
animaworks cost alice --days 7 --json
```

| Argument | Type | Description |
|----------|------|-------------|
| `anima` | positional | Anima name (omit for all) |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--days` | int | 30 | Number of days |
| `--today` | flag | — | Today only |
| `--json` | flag | — | JSON output |

---

## 11. Asset Management

### `optimize-assets` — Optimize 3D Assets

```bash
animaworks optimize-assets --anima alice              # Specific Anima
animaworks optimize-assets --all                       # All optimizations
animaworks optimize-assets --anima alice --simplify 0.3
animaworks optimize-assets --dry-run                   # Preview only
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-a`, `--anima` | string | null | Target Anima |
| `--dry-run` | flag | — | Preview changes only |
| `--simplify` | float | 0.27 | Mesh simplification ratio |
| `--texture-compress` | flag | — | Convert textures to WebP |
| `--texture-resize` | int | 1024 | Texture resolution |
| `--all` | flag | — | Apply strip+simplify+texture+draco |
| `--skip-backup` | flag | — | Don't create backup |

---

### `remake-assets` — Remake Assets (Style Transfer)

```bash
animaworks remake-assets bob --style-from alice
animaworks remake-assets bob --style-from alice --image-style realistic
animaworks remake-assets bob --style-from alice --vibe-strength 0.8 --steps fullbody,bustup
animaworks remake-assets bob --style-from alice --dry-run
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `anima` | positional | Required | Target Anima name |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--style-from` | string | Required | Style reference Anima name |
| `--steps` | string | all steps | Comma-separated (fullbody, bustup, chibi, 3d, rigging, animations) |
| `--prompt` | string | null | Prompt override |
| `--vibe-strength` | float | 0.6 | Vibe Transfer strength (0.0–1.0) |
| `--vibe-info-extracted` | float | 0.8 | Info extraction level (0.0–1.0) |
| `--seed` | int | null | Seed for reproducibility |
| `--image-style` | choice | null | anime, realistic |
| `--no-backup` | flag | — | Skip backup |
| `--dry-run` | flag | — | Show plan without API calls |

---

### `migrate-cron` — Migrate cron.md

Convert Japanese-format cron.md files to standard cron expressions.

```bash
animaworks migrate-cron
```

---

## Common Workflows

### First-Time Setup

```bash
animaworks init                          # Initialize runtime
animaworks start                         # Start server
# Open http://localhost:18500 in browser
```

### Adding an Anima

```bash
animaworks anima create --from-md character.md --role engineer --supervisor alice
animaworks index --anima bob             # Build RAG index
animaworks anima restart bob             # Auto-starts if server running
```

### Changing Models

```bash
animaworks anima set-model alice claude-sonnet-4-6
animaworks anima reload alice            # Hot-reload (no restart needed)
```

### Cost Optimization with Background Model

```bash
animaworks anima set-background-model alice claude-sonnet-4-6
animaworks anima restart alice           # background-model requires restart
```

### Troubleshooting

```bash
animaworks logs alice --lines 100        # Check logs
animaworks anima status alice            # Process status
animaworks anima info alice              # Check configuration
animaworks anima restart alice           # Restart
animaworks stop --force && animaworks start  # Force restart
```
